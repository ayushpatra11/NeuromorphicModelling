"""Tests for neuromorphic.utils."""

import torch

from neuromorphic import utils


class TestMessageEncoding:
    def test_generate_message_neuron_idx_length(self):
        msg = utils.generate_message_neuron_idx(3, 7, message_width=20)
        assert len(msg) == 20

    def test_generate_message_neuron_idx_content(self):
        msg = utils.generate_message_neuron_idx(0, 0, message_width=20)
        assert msg == "0" * 20

    def test_generate_message_neuron_idx_encodes_src(self):
        # src=1, dst=0 → first half is "0000000001", second half is "0000000000"
        msg = utils.generate_message_neuron_idx(1, 0, message_width=20)
        assert msg[:10] == "0000000001"
        assert msg[10:] == "0000000000"

    def test_generate_message_increments(self):
        utils.reset_message_counter()
        m1 = utils.generate_message()
        m2 = utils.generate_message()
        assert m1 != m2

    def test_reset_message_counter(self):
        utils.reset_message_counter()
        m1 = utils.generate_message()
        utils.reset_message_counter()
        m2 = utils.generate_message()
        assert m1 == m2


class TestPacketHelpers:
    def test_bundle_target_cores_remainder(self):
        targets = [(0, 3), (1, 2), (2, 5)]
        group, remainder = utils.bundle_target_cores(targets, 2)
        assert set(group) == {0, 1, 2}
        # reps - 2: (3-2)=1, (2-2)=0, (5-2)=3 → only non-zero
        rem_cores = {c for c, _ in remainder}
        assert 1 not in rem_cores  # reps=0, excluded
        assert 0 in rem_cores and 2 in rem_cores

    def test_bundle_all_equal_reps(self):
        targets = [(0, 2), (1, 2)]
        group, remainder = utils.bundle_target_cores(targets, 2)
        assert len(group) == 2
        assert remainder == []

    def test_remove_unnecessary_packets_intra_core_dropped(self):
        # source_core == target_core should be dropped
        target_cores = [(5, 3), (6, 2)]
        result = utils.remove_unnecessary_packets("lif1", 5, 0, target_cores, {})
        core_ids = [c for c, _ in result]
        assert 5 not in core_ids  # intra-core packet removed
        assert 6 in core_ids

    def test_remove_unnecessary_packets_buffer_subtracted(self):
        target_cores = [(7, 5)]
        buffer_map = {"lif1-0-7": 2}
        result = utils.remove_unnecessary_packets("lif1", 99, 0, target_cores, buffer_map)
        assert result == [(7, 3)]

    def test_dot_product_empty_spike(self):
        routing_matrix = torch.tensor([1.0, 2.0, 3.0])
        spike_vec = torch.zeros(3)
        routing_map = {1: ["a"], 2: ["b"], 3: ["c"]}
        result = utils.dot_product(routing_matrix, spike_vec, routing_map)
        assert result == []

    def test_dot_product_with_spike(self):
        routing_matrix = torch.tensor([42.0, 0.0, 0.0])
        spike_vec = torch.tensor([1.0, 0.0, 0.0])
        routing_map = {42: ["pkt"]}
        result = utils.dot_product(routing_matrix, spike_vec, routing_map)
        assert result == ["pkt"]


class TestRepeatAndConvertPackets:
    def test_basic_conversion(self):
        utils.reset_message_counter()
        packets = [(0, 0, 0, [1, 2], 2)]  # src_nidx, dest_start, src_core, dest_cores, reps
        by_core, expanded = utils.repeat_and_convert_packets(packets, {}, 5, neuron_idx=True)
        # src_core=0 → packets in by_core[0]
        assert 0 in by_core
        assert len(by_core[0]) == 2

    def test_address_bits_set(self):
        utils.reset_message_counter()
        packets = [(0, 0, 2, [0, 3], 1)]
        by_core, expanded = utils.repeat_and_convert_packets(packets, {}, 5, neuron_idx=True)
        pkt = by_core[2][0]
        addr = pkt[20:]  # last 5 chars are the address
        assert addr[0] == "1" and addr[3] == "1"  # bits 0 and 3 set

    def test_expanded_counts_bits(self):
        utils.reset_message_counter()
        packets = [(0, 0, 0, [0, 1, 2], 1)]
        _, expanded = utils.repeat_and_convert_packets(packets, {}, 5, neuron_idx=True)
        assert all(v == 3 for v in expanded.values())


class TestAttachHooks:
    def test_attach_detach(self, net):
        net_out, hooks, spike_record = utils.attach_hooks(net)
        assert len(hooks) > 0
        assert net_out is net

        # Forward pass should populate spike_record
        x = torch.zeros(net.num_steps, 1, net.fc1.in_features)
        net(x)
        assert len(spike_record) > 0

        utils.remove_hooks(hooks)
        assert len(hooks) == 0

    def test_spike_record_length_matches_steps(self, net):
        net, hooks, spike_record = utils.attach_hooks(net)
        T = net.num_steps
        x = torch.zeros(T, 1, net.fc1.in_features)
        net(x)
        utils.remove_hooks(hooks)
        for layer_name, recordings in spike_record.items():
            assert len(recordings) == T, f"{layer_name}: expected {T} steps, got {len(recordings)}"
