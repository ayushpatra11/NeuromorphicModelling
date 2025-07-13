from .model import SpikingNet
from .train import Trainer
from .graph import Graph
from .mapping import Mapping
from .dataset import NavDataset as BinaryNavigationDataset
from . import utils
from .options import Variables
from .options import Specs
import itertools
import copy

import torch
from torch.utils.data import DataLoader
import math
import snntorch as snn
import hashlib

v = Variables()
s = Specs()

def snn_init(dut=None):
    # -------------------------------------------------

    torch.manual_seed(42)
    # Initialize the network
    net = SpikingNet(v)

    # -------------------------------------------------

    sample_data = torch.randn(v.num_steps, v.num_inputs)
    net = utils.init_network(net, sample_data)

    indices_to_lock = {
        "indices": [],
        "layers"  : ("lif1","lif1")}

    # -------------------------------------------------

    gp = Graph(v.num_steps, v.num_inputs)
    gp.export_model(net)
    gp.extract_edges()
    gp.process_graph()
    #gp.plot_graph()
    gp.log(dut)

    # -------------------------------------------------

    mapping = Mapping(net, v.num_steps, v.num_inputs)
    total_neurons = sum(mapping.mem_potential_sizes.values())
    core_capacity = max(math.ceil((total_neurons - v.num_outputs) / (v.num_cores - 1)), v.num_outputs)
    mapping.set_core_capacity(core_capacity)
    mapping.map_neurons()
    mapping.map_buffers(indices_to_lock)
    
    mapping.log(dut)

    # -------------------------------------------------

    # Parameters
    n_in = v.num_inputs
    t_cue_spacing = v.t_cue_spacing
    #silence_duration = v.silence_duration
    recall_duration = v.recall_duration
    seq_len = v.num_steps
    v.num_steps = seq_len
    batch_size = v.bs
    input_f0 = 40. / 100.
    p_group = v.p_group
    n_cues = v.n_cues
    t_cue = v.t_cue
    n_input_symbols = 4

    # Create dataset and dataloader
    train_set = BinaryNavigationDataset(seq_len, n_in, recall_duration, p_group, input_f0, n_cues, t_cue, t_cue_spacing, n_input_symbols, length=100)
    val_set = BinaryNavigationDataset(seq_len, n_in, recall_duration, p_group, input_f0, n_cues, t_cue, t_cue_spacing, n_input_symbols, length=20)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=0)
    # -------------------------------------------------

    trainer = Trainer(net,
                      train_loader,
                      val_loader,
                      v.target_sparcity,
                      v.recall_duration,
                      graph=gp,
                      num_epochs=v.num_epochs, 
                      learning_rate=v.lr, 
                      target_frequency=v.target_fr, 
                      num_steps=v.num_steps)
    if v.train:
        net, mapping, max_accuracy, final_accuracy, metrics = trainer.train(v.device, mapping, dut)
    else:
        net.load_state_dict(torch.load("model.pth"))
        max_accuracy = None
        final_accuracy, _ = trainer.eval(v.device,0,external_model=net, final=True)
        metrics = trainer.mtrcs
    # -------------------------------------------------

    routing_matrices = {}
    routing_map = {}

    source_neuron_index = 0 # index counting over all the neurons, used in verilog of id

    # counstricting routing matrices
    for layer_name, size in mapping.mem_potential_sizes.items(): # a way to get the layer names
        
        routing_matrix = torch.zeros((size))
        for idx in range(size):

            if layer_name in routing_matrices:
                continue

            routing_id = layer_name +"-"+ str(idx)
            source_core = mapping.neuron_to_core[routing_id]

            downstream_nodes = list(gp.graph.successors(layer_name))

            target_cores = []
            for downstream_node in downstream_nodes:
                if downstream_node != "output":
                    target_cores.extend(mapping.NIR_to_cores[downstream_node])

            # Remove skipped packets
            target_cores = utils.remove_unnecessary_packets(layer_name, source_core, idx, target_cores, mapping.buffer_map)

            # bundle packets (bundle several unicast packets into multicast)
            bundled_core_to_cores = []
            dest_neuron_start_index = 0
            while len(target_cores) > 0:
                _, minimum = target_cores[0] # just getting the first repetition value
                for _, reps in target_cores: # find the minimum repetition value
                    if reps < minimum:
                        minimum = reps

                bcc, target_cores = utils.bundle_target_cores(target_cores, minimum)
                bundled_core_to_cores.append((bcc, minimum))

            packet_information = []

            for bcc, reps in bundled_core_to_cores:
                packet_information.append((source_neuron_index, dest_neuron_start_index, source_core, bcc, reps))
                h = int(hashlib.shake_256(routing_id.encode()).hexdigest(2), 16)
                routing_map[h] = packet_information

                routing_matrix[idx] = h
                if reps < 0:
                    print("REP -", reps)
                dest_neuron_start_index += reps

            source_neuron_index += 1

        routing_matrices[layer_name] = routing_matrix

    return net, routing_matrices, routing_map, mapping, train_set, val_set, max_accuracy, final_accuracy, metrics
    
def delay_experiment(network, routing_matrices, routing_map, mapping, dataset, idx=0):
    net = copy.deepcopy(network)
    # Dictionary to store spikes from each layer

    spike_record = {}
    hooks = []

    def reset_spike_record_and_hooks():
        global spike_record, hooks

        # Clear the spike_record dictionary
        spike_record = {}

        # Remove existing hooks if they are already registered
        if 'hooks' in globals():
            for hook in hooks:
                hook.remove()
                hooks = []

    # Function to create a hook that records spikes
    def create_spike_hook(layer_name):
        def hook(module, input, output):
            if layer_name not in spike_record:
                spike_record[layer_name] = []
            spike_record[layer_name].append(output[0].detach().cpu()) # 0 index refers to spk
        return hook

    reset_spike_record_and_hooks()

    # Attach hooks automatically to all Leaky layers
    for name, module in net.named_modules():
        if isinstance(module, snn.Leaky) or isinstance(module, snn.RSynaptic):
            hooks.append(module.register_forward_hook(create_spike_hook(name)))

    

    # if not full_inference: 
    #     return net, routing_matrices, routing_map, dataset, mapping
    # -------------------------------------------------
    # Dictionary to store spikes from each layer

    spike_record = {}
    hooks = []

    # Generate input
    #inputs = trainer.generate_spike_train(inputs, v.num_steps).to(v.device)
    data, _ = dataset[idx]
    
    # Record spikes
    _, _ = net(data.to(v.device))

    # Convert spike records to tensors for easier analysis
    for layer_name in spike_record:
        spike_record[layer_name] = torch.squeeze(torch.stack(spike_record[layer_name]))

    # -------------------------------------------------

    if 'hooks' in globals():
        for hook in hooks:
            hook.remove()
            hooks = []

    # -------------------------------------------------

    packets = []

    for t in range(v.num_steps):
        packets_in_ts = []
        for layer_name, _ in mapping.mem_potential_sizes.items():

            p = utils.dot_product(routing_matrices[layer_name], 
                                        spike_record[layer_name][t], 
                                        routing_map,
                                        )
            
            packets_in_ts.extend(p)

        packets.append(packets_in_ts)

    #final_packets_list = []
    final_packets_dict = {
        s.EAST: [],
        s.NORTH: [],
        s.WEST: [],
        s.SOUTH: [],
        s.L1: []
    }
    expanded_packets_list = []

    for packet in packets:
        # every iteration is one timestep

        temp, expanded_packets = utils.repeat_and_convert_packets(packet, final_packets_dict, s.ADDR_W, neuron_idx=False)
        
        expanded_packets_list.append(expanded_packets)

        for key in final_packets_dict:
            if key in temp:
                final_packets_dict[key].append(temp[key])

    return final_packets_dict, expanded_packets_list


class DynamicInference:
    def __init__(self, net):
        self.net = net
        self.spike_record = {}
        self.hooks = []
        
    def init_membranes(self):
        self.spk1, self.syn1, self.mem1 = self.net.lif1.init_rsynaptic()
        self.mem2 = self.net.lif2.init_leaky()
    # Method to reset spike recording and hooks
    def reset_spike_record_and_hooks(self):
        # Clear the spike_record dictionary
        self.spike_record = {}

        # Remove existing hooks if they are already registered
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    # Method to create a hook that records spikes for a specific layer
    def create_spike_hook(self, layer_name):
        def hook(module, input, output):
            if layer_name not in self.spike_record:
                self.spike_record[layer_name] = []
            self.spike_record[layer_name].append(output[0].detach().cpu())
        return hook

    # Method to attach hooks automatically to all Leaky and RSynaptic layers
    def attach_hooks(self):
        self.reset_spike_record_and_hooks()
        for name, module in self.net.named_modules():
            if isinstance(module, snn.Leaky) or isinstance(module, snn.RSynaptic):
                self.hooks.append(module.register_forward_hook(self.create_spike_hook(name)))

    def advance_inference(self, data, skipped_spikes=None, add_spikes=None):
        skipped_spikes_list = []
        add_spikes_list = []
        if skipped_spikes is not None:
            for s_idx, source_index in enumerate(skipped_spikes):
                for d_idx, dest_core in enumerate(source_index):
                    for r_idx, element in enumerate(dest_core):
                        if element != 0:
                            #print("CORE CAPACITY", v.core_capacity)
                            #print("ELEMENT", element, s_idx, d_idx, r_idx)
                            skipped_spikes_list.append((element, s_idx, (d_idx)*v.core_capacity + r_idx))
        
        if add_spikes is not None:
            for s_idx, source_index in enumerate(add_spikes):
                for d_idx, dest_core in enumerate(source_index):
                    for r_idx, element in enumerate(dest_core):
                        if element != 0:
                            #print("ELEMENT", element, s_idx, d_idx, r_idx)
                            add_spikes_list.append((element, s_idx, (d_idx)*v.core_capacity + r_idx))

        self.spike_record = {}
        
        output_spikes, self.spk1, self.syn1, self.mem1, self.mem2 = \
            self.net.forward_one_ts(data.to(v.device), self.spk1, self.syn1, self.mem1, self.mem2, cur_sub=skipped_spikes_list, cur_add=add_spikes_list, time_first=True)

        return self.spike_record, output_spikes
