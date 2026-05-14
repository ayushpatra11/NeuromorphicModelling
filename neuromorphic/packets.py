"""Packet generation pipeline and dynamic inference for NoC routing simulation."""

from __future__ import annotations

import copy
import hashlib
import logging
import math

import snntorch as snn
import torch
from torch.utils.data import DataLoader

from neuromorphic import utils
from neuromorphic.config import HardwareSpecs, ModelConfig
from neuromorphic.dataset import NavDataset
from neuromorphic.graph import Graph
from neuromorphic.mapping import Mapping
from neuromorphic.model import SpikingNet
from neuromorphic.trainer import Trainer

logger = logging.getLogger(__name__)

_cfg = ModelConfig()
_hw = HardwareSpecs()


# ---------------------------------------------------------------------------
# Pipeline initialisation
# ---------------------------------------------------------------------------


def init_pipeline(dut=None):
    """
    Build the full SNN → graph → mapping → routing-matrix pipeline.

    Returns
    -------
    net, routing_matrices, routing_map, mapping,
    train_set, val_set, max_accuracy, final_accuracy, metrics
    """
    torch.manual_seed(42)
    net = SpikingNet(_cfg)

    sample = torch.randn(_cfg.num_steps, _cfg.num_inputs)
    net = utils.init_network(net, sample)

    indices_to_lock: dict = {"indices": [], "layers": ("lif1", "lif1")}

    gp = Graph(_cfg.num_steps, _cfg.num_inputs)
    gp.export_model(net)
    gp.extract_edges()
    gp.process_graph()
    gp.log(dut)

    mapping = Mapping(net, _cfg.num_steps, _cfg.num_inputs)
    total = sum(mapping.mem_potential_sizes.values())
    cc = max(
        math.ceil((total - _cfg.num_outputs) / (_cfg.num_cores - 1)),
        _cfg.num_outputs,
    )
    mapping.set_core_capacity(cc)
    mapping.map_neurons()
    mapping.map_buffers(indices_to_lock)
    mapping.log(dut)

    seq_len = _cfg.num_steps
    f0 = 40.0 / 100.0

    train_set = NavDataset(
        seq_len,
        _cfg.num_inputs,
        _cfg.recall_duration,
        _cfg.p_group,
        f0,
        _cfg.n_cues,
        _cfg.t_cue,
        _cfg.t_cue_spacing,
        4,
        length=100,
    )
    val_set = NavDataset(
        seq_len,
        _cfg.num_inputs,
        _cfg.recall_duration,
        _cfg.p_group,
        f0,
        _cfg.n_cues,
        _cfg.t_cue,
        _cfg.t_cue_spacing,
        4,
        length=20,
    )
    train_loader = DataLoader(train_set, batch_size=_cfg.bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=0)

    trainer = Trainer(
        net,
        train_loader,
        val_loader,
        _cfg.target_sparsity,
        _cfg.recall_duration,
        graph=gp,
        num_epochs=_cfg.num_epochs,
        learning_rate=_cfg.lr,
        target_frequency=_cfg.target_fr,
        num_steps=_cfg.num_steps,
    )

    if _cfg.train:
        net, mapping, max_acc, final_acc, metrics = trainer.train(_cfg.device, mapping, dut)
    else:
        net.load_state_dict(torch.load("model.pth", map_location=_cfg.device))
        max_acc = None
        final_acc, _ = trainer.eval(_cfg.device, 0, external_model=net, final=True)
        metrics = trainer.mtrcs

    routing_matrices, routing_map = _build_routing_tables(net, gp, mapping)
    return (
        net,
        routing_matrices,
        routing_map,
        mapping,
        train_set,
        val_set,
        max_acc,
        final_acc,
        metrics,
    )


def _build_routing_tables(
    net: SpikingNet,
    gp: Graph,
    mapping: Mapping,
) -> tuple[dict[str, torch.Tensor], dict[int, list]]:
    routing_matrices: dict[str, torch.Tensor] = {}
    routing_map: dict[int, list] = {}
    source_neuron_index = 0

    for layer_name, size in mapping.mem_potential_sizes.items():
        routing_matrix = torch.zeros(size)

        for idx in range(size):
            if layer_name in routing_matrices:
                break

            routing_id = f"{layer_name}-{idx}"
            source_core = mapping.neuron_to_core[routing_id]
            downstream = list(gp.graph.successors(layer_name))

            target_cores: list[tuple[int, int]] = []
            for dn in downstream:
                if dn != "output":
                    target_cores.extend(mapping.NIR_to_cores[dn])

            target_cores = utils.remove_unnecessary_packets(
                layer_name, source_core, idx, target_cores, mapping.buffer_map
            )

            bundled: list[tuple[list[int], int]] = []
            dest_start = 0
            while target_cores:
                minimum = min(reps for _, reps in target_cores)
                group, target_cores = utils.bundle_target_cores(target_cores, minimum)
                bundled.append((group, minimum))

            packet_info: list[tuple] = []
            for group, reps in bundled:
                packet_info.append((source_neuron_index, dest_start, source_core, group, reps))
                h = int(hashlib.shake_256(routing_id.encode()).hexdigest(2), 16)
                routing_map[h] = packet_info
                routing_matrix[idx] = h
                dest_start += reps

            source_neuron_index += 1

        routing_matrices[layer_name] = routing_matrix

    return routing_matrices, routing_map


# ---------------------------------------------------------------------------
# Delay experiment – record packets for one dataset sample
# ---------------------------------------------------------------------------


def record_packets(
    network: SpikingNet,
    routing_matrices: dict[str, torch.Tensor],
    routing_map: dict[int, list],
    mapping: Mapping,
    dataset: NavDataset,
    sample_idx: int = 0,
) -> tuple[dict[int, list[list[str]]], list[dict[str, int]]]:
    """
    Run inference on one sample, convert per-timestep spikes to packet lists.

    Returns
    -------
    packets_by_core:   Dict[core_id → per-timestep lists of "msg||addr" strings]
    expanded_packets:  List (one per timestep) of {message → bit_count}
    """
    net = copy.deepcopy(network)
    utils.reset_message_counter()

    data, _ = dataset[sample_idx]
    net, hooks, spike_record = utils.attach_hooks(net)

    with torch.no_grad():
        net(data.to(_cfg.device))

    utils.remove_hooks(hooks)

    for layer in spike_record:
        spike_record[layer] = torch.squeeze(torch.stack(spike_record[layer]))

    # Collect packets per timestep
    packets_by_ts: list[list[tuple]] = []
    for t in range(_cfg.num_steps):
        ts_packets: list[tuple] = []
        for layer_name in mapping.mem_potential_sizes:
            ts_packets.extend(
                utils.dot_product(
                    routing_matrices[layer_name],
                    spike_record[layer_name][t],
                    routing_map,
                )
            )
        packets_by_ts.append(ts_packets)

    # Encode into bit-string packets
    final_by_core: dict[int, list[list[str]]] = {
        _hw.EAST: [],
        _hw.NORTH: [],
        _hw.WEST: [],
        _hw.SOUTH: [],
        _hw.L1: [],
    }
    expanded_list: list[dict[str, int]] = []

    for ts_pkts in packets_by_ts:
        by_core, expanded = utils.repeat_and_convert_packets(
            ts_pkts, {}, _hw.ADDR_W, neuron_idx=False
        )
        expanded_list.append(expanded)
        for core, pkt_list in by_core.items():
            if core in final_by_core:
                final_by_core[core].append(pkt_list)

    return final_by_core, expanded_list


# Legacy alias
delay_experiment = record_packets


# ---------------------------------------------------------------------------
# Step-wise inference for hardware-in-the-loop simulation
# ---------------------------------------------------------------------------


class DynamicInference:
    """
    Single-timestep inference with injected spike corrections.

    Usage::
        di = DynamicInference(net)
        di.init_membranes()
        di.attach_hooks()
        spk_rec, out = di.advance_inference(data_slice)
    """

    def __init__(self, net: SpikingNet, cfg: ModelConfig | None = None) -> None:
        self.net = net
        self._cfg = cfg or _cfg
        self.spike_record: dict[str, list[torch.Tensor]] = {}
        self.hooks: list = []
        self.spk1 = self.syn1 = self.mem1 = self.mem2 = None

    def init_membranes(self) -> None:
        self.spk1, self.syn1, self.mem1 = self.net.lif1.init_rsynaptic()
        self.mem2 = self.net.lif2.init_leaky()

    def attach_hooks(self) -> None:
        utils.remove_hooks(self.hooks)
        self.spike_record = {}
        for name, module in self.net.named_modules():
            if isinstance(module, (snn.Leaky, snn.RSynaptic)):

                def _hook(mod, inp, out, n=name):  # noqa: ARG001
                    self.spike_record.setdefault(n, []).append(out[0].detach().cpu())

                self.hooks.append(module.register_forward_hook(_hook))

    def advance_inference(
        self,
        data: torch.Tensor,
        skipped_spikes=None,
        add_spikes=None,
    ) -> tuple[dict, torch.Tensor]:
        def _parse(raw) -> list[tuple]:
            if raw is None:
                return []
            out = []
            for s_idx, source in enumerate(raw):
                for d_idx, dest in enumerate(source):
                    for r_idx, elem in enumerate(dest):
                        if elem != 0:
                            out.append((elem, s_idx, d_idx * self._cfg.core_capacity + r_idx))
            return out

        self.spike_record = {}
        out_spk, self.spk1, self.syn1, self.mem1, self.mem2 = self.net.forward_one_ts(
            data.to(self._cfg.device),
            self.spk1,
            self.syn1,
            self.mem1,
            self.mem2,
            cur_sub=_parse(skipped_spikes),
            cur_add=_parse(add_spikes),
            time_first=True,
        )
        return self.spike_record, out_spk


# Keep the old init name as an alias
snn_init = init_pipeline
