"""Tests for neuromorphic.mapping."""

import math

import pytest

from neuromorphic.mapping import Mapping


class TestMapping:
    def test_init_from_net(self, net, cfg):
        m = Mapping(net, cfg.num_steps, cfg.num_inputs)
        assert len(m.mem_potential_sizes) > 0

    def test_init_from_sizes(self):
        sizes = {"lif1": 32, "lif2": 4}
        m = Mapping(mem_potential_sizes=sizes)
        assert m.mem_potential_sizes == sizes

    def test_allocation_covers_all_neurons(self, net, cfg):
        m = Mapping(net, cfg.num_steps, cfg.num_inputs)
        cc = 10
        m.set_core_capacity(cc)
        m.map_neurons()

        for layer, size in m.mem_potential_sizes.items():
            layer_entries = [k for k in m.neuron_to_core if k.startswith(f"{layer}-")]
            assert len(layer_entries) == size

    def test_no_core_exceeds_capacity(self, net, cfg):
        m = Mapping(net, cfg.num_steps, cfg.num_inputs)
        cc = 8
        m.set_core_capacity(cc)
        m.map_neurons()

        from collections import Counter

        core_usage: Counter = Counter()
        for _, core_id in m.neuron_to_core.items():
            core_usage[core_id] += 1
        for cid, count in core_usage.items():
            assert count <= cc, f"Core {cid} has {count} neurons, capacity {cc}"

    def test_map_buffers_empty_indices(self, net, cfg):
        m = Mapping(net, cfg.num_steps, cfg.num_inputs)
        m.set_core_capacity(10)
        m.map_neurons()
        indices_to_lock = {"indices": [], "layers": ("lif1", "lif1")}
        m.map_buffers(indices_to_lock)
        assert m.buffer_map == {}

    def test_map_buffers_with_indices(self, net, cfg):
        m = Mapping(net, cfg.num_steps, cfg.num_inputs)
        m.set_core_capacity(10)
        m.map_neurons()

        # Find a valid pair of neurons in different cores
        layer = list(m.mem_potential_sizes.keys())[0]
        alloc = m.core_allocation[layer]
        if len(alloc) < 2:
            pytest.skip("Too few cores for this test")

        src_core, s_start, s_end = alloc[0]
        dst_core, d_start, d_end = alloc[1]
        indices_to_lock = {
            "indices": [(s_start, d_start)],
            "layers": (layer, layer),
        }
        m.map_buffers(indices_to_lock)
        # buffer_map should have at most 1 entry for this pair
        assert len(m.buffer_map) <= 1

    def test_num_cores_equals_ceil(self):
        sizes = {"lif1": 20, "lif2": 4}
        m = Mapping(mem_potential_sizes=sizes)
        cc = 8
        m.set_core_capacity(cc)
        m.map_neurons()
        total = sum(sizes.values())
        expected_cores = math.ceil(total / cc)
        actual_cores = len(set(m.neuron_to_core.values()))
        assert actual_cores <= expected_cores

    def test_get_mappings(self, net, cfg):
        m = Mapping(net, cfg.num_steps, cfg.num_inputs)
        m.set_core_capacity(10)
        m.map_neurons()
        ca, nir, n2c = m.get_mappings()
        assert ca == m.core_allocation
        assert nir == m.NIR_to_cores
        assert n2c == m.neuron_to_core
