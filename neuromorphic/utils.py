"""Utility functions for spike recording, packet generation, and connection management."""

from __future__ import annotations

import logging
import random
from typing import Any

import snntorch as snn
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spike recording
# ---------------------------------------------------------------------------


def attach_hooks(
    net: nn.Module,
) -> tuple[nn.Module, list[Any], dict[str, list[torch.Tensor]]]:
    """
    Attach forward hooks to all Leaky and RSynaptic layers.

    Returns
    -------
    net:          The network (unchanged).
    hooks:        List of hook handles (call hook.remove() to detach).
    spike_record: Dict mapping layer name → list of spike tensors per step.
    """
    spike_record: dict[str, list[torch.Tensor]] = {}
    hooks: list[Any] = []

    def _make_hook(layer_name: str):
        def hook(module, input, output):  # noqa: ARG001
            spike_record.setdefault(layer_name, []).append(output[0].detach().cpu())

        return hook

    for name, module in net.named_modules():
        if isinstance(module, (snn.Leaky, snn.RSynaptic)):
            hooks.append(module.register_forward_hook(_make_hook(name)))

    return net, hooks, spike_record


def remove_hooks(hooks: list[Any]) -> None:
    for h in hooks:
        h.remove()
    hooks.clear()


# ---------------------------------------------------------------------------
# Packet bundling helpers
# ---------------------------------------------------------------------------


def bundle_target_cores(
    target_cores: list[tuple[int, int]],
    min_reps: int,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Split one multicast group from target_cores and return the remainder."""
    group = [core for core, _ in target_cores]
    remainder = [(core, reps - min_reps) for core, reps in target_cores if reps - min_reps > 0]
    return group, remainder


def remove_unnecessary_packets(
    layer_name: str,
    source_core: int,
    neuron_idx: int,
    target_cores: list[tuple[int, int]],
    buffer_map: dict[str, int],
) -> list[tuple[int, int]]:
    """
    Remove intra-core packets and subtract already-buffered connections
    from the repetition count of inter-core packets.
    """
    result = []
    for target_core, reps in target_cores:
        if source_core == target_core:
            continue
        key = f"{layer_name}-{neuron_idx}-{target_core}"
        adjusted = reps - int(buffer_map[key]) if key in buffer_map else reps
        result.append((target_core, adjusted))
    return result


def dot_product(
    routing_matrix: torch.Tensor,
    spike_vec: torch.Tensor,
    routing_map: dict[int, list],
) -> list:
    """Multiply routing matrix by spike vector and collect non-zero packet lists."""
    non_zero = torch.mul(routing_matrix, spike_vec)
    packets: list = []
    for h in non_zero[non_zero != 0]:
        packets.extend(routing_map[int(h)])
    return packets


# ---------------------------------------------------------------------------
# Message / address encoding
# ---------------------------------------------------------------------------


class _MessageCounter:
    """Thread-unsafe monotonic counter for generating unique message IDs."""

    def __init__(self, width: int = 20) -> None:
        self._count = 0
        self._width = width
        self._max = 1 << width

    def next(self) -> str:
        msg = self._count
        self._count = (self._count + 1) % self._max
        return f"{msg:0{self._width}b}"

    def reset(self) -> None:
        self._count = 0


_message_counter = _MessageCounter()


def generate_message(message_width: int = 20) -> str:  # noqa: ARG001
    """Return the next unique binary message ID string."""
    return _message_counter.next()


def reset_message_counter() -> None:
    """Reset the message counter (call between experiments)."""
    _message_counter.reset()


def generate_message_neuron_idx(s_idx: int, d_idx: int, message_width: int = 20) -> str:
    half = message_width // 2
    return format(s_idx, f"0{half}b") + format(d_idx, f"0{half}b")


def _count_address_bits(address: str, addr_width: int = 5) -> int:
    return sum(1 for i in range(addr_width) if address[i] == "1")


def repeat_and_convert_packets(
    packets: list[tuple],
    packets_dict: dict,  # noqa: ARG001 – kept for API compat
    address_length: int,
    neuron_idx: bool = True,
) -> tuple[dict[int, list[str]], dict[str, int]]:
    """
    Convert a list of packet tuples into bit-string (message + address) format.

    Each packet is (src_neuron_idx, dest_neuron_start, src_core, dest_cores, reps).

    Returns
    -------
    by_core:   Dict[source_core → list of "message||address" strings].
    expanded:  Dict[message → number of destination bits set].
    """
    by_core: dict[int, list[str]] = {0: [], 1: [], 2: [], 3: [], 4: []}
    expanded: dict[str, int] = {}

    for src_nidx, dest_start, src_core, dest_cores, reps in packets:
        bits = ["0"] * address_length
        for idx in dest_cores:
            bits[idx] = "1"
        address = "".join(bits)

        prev_msg: str | None = None
        for i in range(reps):
            if neuron_idx:
                msg = generate_message_neuron_idx(src_nidx, dest_start + i)
            else:
                msg = generate_message()
                while msg == prev_msg:
                    msg = generate_message()
                prev_msg = msg

            by_core.setdefault(src_core, []).append(msg + address)
            expanded[msg] = _count_address_bits(address, address_length)

    return by_core, expanded


# ---------------------------------------------------------------------------
# Sparsity / connection management
# ---------------------------------------------------------------------------


def count_target_neurons(
    layer_name: str,
    source_core: int,
    neuron_idx: int,
    target_cores: list[tuple[int, int]],
    buffer_map: dict[str, int],
) -> tuple[int, int, int, int]:
    """Return (lr_neurons, sr_neurons, lr_buffered, sr_buffered)."""
    lr_dest = sr_dest = lr_sub = sr_sub = 0
    for target_core, reps in target_cores:
        key = f"{layer_name}-{neuron_idx}-{target_core}"
        buffered = int(buffer_map.get(key, 0))
        if source_core == target_core:
            sr_sub += buffered
            sr_dest += reps
        else:
            lr_sub += buffered
            lr_dest += reps
    return lr_dest, sr_dest, lr_sub, sr_sub


def calculate_lr_sr_conns(mapping: Any, graph: Any) -> tuple[int, int]:
    """Count long-range and short-range connections in the current mapping."""
    lr_total = sr_total = 0
    src_layer = mapping.indices_to_lock["layers"][0]
    dst_layer = mapping.indices_to_lock["layers"][1]

    for layer_name in mapping.mem_potential_sizes:
        if layer_name != src_layer:
            continue
        for source_core, start_idx, end_idx in mapping.core_allocation[layer_name]:
            downstream = list(graph.graph.successors(layer_name))
            for idx in range(start_idx, end_idx + 1):
                for dn in downstream:
                    if dn != dst_layer:
                        continue
                    tc = mapping.NIR_to_cores[dn]
                    lr, sr, lr_s, sr_s = count_target_neurons(
                        layer_name, source_core, idx, tc, mapping.buffer_map
                    )
                    lr_total += lr - lr_s
                    sr_total += sr - sr_s
    return lr_total, sr_total


def choose_conn_remove(mapping: Any, reps: int | None = None) -> Any:
    """
    Randomly select inter-core connections and add them to indices_to_lock.

    Mutates and returns the mapping object.
    """
    src_layer = mapping.indices_to_lock["layers"][0]
    dst_layer = mapping.indices_to_lock["layers"][1]
    src_alloc = mapping.core_allocation[src_layer]
    dst_alloc = mapping.core_allocation[dst_layer]
    locked = mapping.indices_to_lock["indices"]

    count = 0
    while True:
        sc = src_alloc[random.randint(0, len(src_alloc) - 1)]
        dc = dst_alloc[random.randint(0, len(dst_alloc) - 1)]
        if sc[0] == dc[0]:
            continue
        si = random.randint(sc[1], sc[2])
        di = random.randint(dc[1], dc[2])
        if (si, di) in locked:
            continue
        locked.append((si, di))
        count += 1
        if reps is None or count >= reps:
            break

    mapping.map_buffers()
    return mapping


def init_network(net: nn.Module, sample_data: torch.Tensor) -> nn.Module:
    """Run one forward pass to initialise lazy layers (ignores output)."""
    with torch.no_grad():
        try:
            net(sample_data)
        except Exception:
            net(sample_data)
    return net
