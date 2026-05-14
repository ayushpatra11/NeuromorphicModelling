"""Neuron-to-core allocation for binary-tree neuromorphic topologies."""

from __future__ import annotations

import logging
import math
import random
from typing import Any

import snntorch as snn
import torch.nn as nn

logger = logging.getLogger(__name__)

CoreAllocation = list[tuple[int, int, int]]  # (core_id, start_idx, end_idx)
NirToCores = list[tuple[int, int]]  # (core_id, neuron_count)


class Mapping:
    """
    Randomly assigns neurons to cores, ensuring balanced load.

    After construction call:
        set_core_capacity(cc)
        map_neurons()
        map_buffers(indices_to_lock)
    """

    def __init__(
        self,
        net: nn.Module | None = None,
        num_steps: int | None = None,
        num_inputs: int | None = None,
        mem_potential_sizes: dict[str, int] | None = None,
    ) -> None:
        self.num_steps = num_steps
        self.num_inputs = num_inputs
        self.core_capacity: int | None = None
        self.net = net

        self.mem_potential_sizes: dict[str, int] = (
            mem_potential_sizes
            if mem_potential_sizes is not None
            else (self._get_membrane_potential_sizes() if net is not None else {})
        )

        self.core_allocation: dict[str, CoreAllocation] = {}
        self.NIR_to_cores: dict[str, NirToCores] = {}
        self.neuron_to_core: dict[str, int] = {}
        self.buffer_map: dict[str, int] = {}
        self.indices_to_lock: dict[str, Any] | None = None

    def _get_membrane_potential_sizes(self) -> dict[str, int]:
        if self.net is None:
            raise ValueError("Network model not set.")
        sizes: dict[str, int] = {}
        for name, module in self.net.named_modules():
            if isinstance(module, snn.Synaptic):
                _, mem = module.init_leaky()
                sizes[name] = mem.size()[0]
            elif isinstance(module, snn.Leaky):
                mem = module.init_leaky()
                sizes[name] = mem.size()[0]
            elif isinstance(module, snn.RSynaptic):
                sizes[name] = module.linear_features
        return sizes

    def set_core_capacity(self, cc: int) -> None:
        self.core_capacity = cc

    def map_neurons(self) -> None:
        """Perform random neuron-to-core allocation."""
        self.core_allocation, self.NIR_to_cores, self.neuron_to_core = (
            self._allocate_neurons_to_cores()
        )

    def _allocate_neurons_to_cores(
        self,
    ) -> tuple[dict[str, CoreAllocation], dict[str, NirToCores], dict[str, int]]:
        if self.core_capacity is None:
            raise ValueError("Core capacity not set. Call set_core_capacity first.")

        layer_names = list(self.mem_potential_sizes.keys())
        total_neurons = sum(self.mem_potential_sizes.values())
        total_cores = math.ceil(total_neurons / self.core_capacity)

        all_neurons = [
            (layer, nid) for layer in layer_names for nid in range(self.mem_potential_sizes[layer])
        ]
        random.shuffle(all_neurons)

        core_buckets: list[list[tuple[str, int]]] = [[] for _ in range(total_cores)]
        core_counts = [0] * total_cores
        core_id = 0
        neuron_to_core: dict[str, int] = {}

        for layer, nid in all_neurons:
            while core_counts[core_id] >= self.core_capacity:
                core_id = (core_id + 1) % total_cores
            core_buckets[core_id].append((layer, nid))
            neuron_to_core[f"{layer}-{nid}"] = core_id
            core_counts[core_id] += 1

        core_allocation: dict[str, CoreAllocation] = {}
        nir_to_cores: dict[str, NirToCores] = {}

        for layer in layer_names:
            layer_alloc: CoreAllocation = []
            layer_nir: dict[int, int] = {}
            for cid, bucket in enumerate(core_buckets):
                ids = [nid for lname, nid in bucket if lname == layer]
                if ids:
                    layer_alloc.append((cid, min(ids), max(ids)))
                    layer_nir[cid] = len(ids)
            core_allocation[layer] = layer_alloc
            nir_to_cores[layer] = list(layer_nir.items())

        return core_allocation, nir_to_cores, neuron_to_core

    def map_buffers(self, indices_to_lock: dict | None = None) -> None:
        """
        Compute buffer_map: maps "layer-src_idx-dest_core" → intra-core connection count.
        If indices_to_lock is supplied it replaces the stored value.
        """
        if indices_to_lock is not None:
            self.indices_to_lock = indices_to_lock

        if self.indices_to_lock is None:
            raise ValueError("indices_to_lock is not set.")

        dst_layer = self.indices_to_lock["layers"][1]
        src_layer = self.indices_to_lock["layers"][0]
        mapped: dict[str, int] = {}

        for src_idx, dst_idx in self.indices_to_lock["indices"]:
            dst_core = self.neuron_to_core[f"{dst_layer}-{dst_idx}"]
            key = f"{src_layer}-{src_idx}-{dst_core}"
            mapped[key] = mapped.get(key, 0) + 1

        self.buffer_map = mapped

    def get_mappings(
        self,
    ) -> tuple[dict[str, CoreAllocation], dict[str, NirToCores], dict[str, int]]:
        return self.core_allocation, self.NIR_to_cores, self.neuron_to_core

    def log(self, dut=None) -> None:
        lines = ["\n----- MAPPING -----"]
        for layer, size in self.mem_potential_sizes.items():
            lines.append(f"  Layer {layer}: {size} neurons")
        lines += [
            f"Core allocation: {self.core_allocation}",
            f"NIR to cores:    {self.NIR_to_cores}",
            f"Buffer map:      {self.buffer_map}",
            f"Core capacity:   {self.core_capacity}",
        ]
        for line in lines:
            if dut is not None:
                dut._log.info(line)
            else:
                logger.info(line)
