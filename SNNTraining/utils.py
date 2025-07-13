import snntorch as snn
import torch
import random

spike_record = {}
counter = 0

# def _reset_spike_record_and_hooks():
#         global spike_record, hooks

#         # Clear the spike_record dictionary
#         spike_record = {}

#         # Remove existing hooks if they are already registered
#         if 'hooks' in globals():
#             for hook in hooks:
#                 hook.remove()
#                 hooks = []

# Function to create a hook that records spikes
def _create_spike_hook(layer_name):
    def hook(module, input, output):
        if layer_name not in spike_record:
            spike_record[layer_name] = []
        spike_record[layer_name].append(output[0].detach().cpu())
    return hook

def attach_hooks(net):
    #_reset_spike_record_and_hooks()
    hooks = []
    # Attach hooks automatically to all Leaky layers
    for name, module in net.named_modules():
        if isinstance(module, snn.Leaky) or isinstance(module, snn.RSynaptic):
            hooks.append(module.register_forward_hook(_create_spike_hook(name)))

    return net, hooks

def bundle_target_cores(target_cores, min_reps):
  res = []
  new_target_cores = []
  for target_core, reps in target_cores:
    res.append(target_core)
    if reps - min_reps > 0:
      new_target_cores.append((target_core, reps - min_reps))

  return res, new_target_cores

def remove_unnecessary_packets(layer_name, source_core, idx, target_cores, buffer_map):
    new_target_cores = []
    #inter_core_packets = []
    for target_core, reps in target_cores:
        
        # format of buffer_map
        connection = layer_name+"-"+str(idx)+"-"+str(target_core)

        # remove inter-core communication
        if source_core == target_core:
            continue

        if connection in buffer_map:
                new_target_cores.append((target_core, reps - int(buffer_map[connection])))
        else:
            new_target_cores.append((target_core, reps))

    return new_target_cores

def dot_product(routing_matrices, spike_record, routing_map):
    packets = []
    exp = torch.mul(routing_matrices, spike_record)
    #print(exp)
    temp = exp
    non_zero_values = temp[temp != 0]

    #print(non_zero_values)

    for hashes in non_zero_values:
        packets.extend(routing_map[int(hashes)])

    return packets

def init_network(net, sample_data):

    try:
        _, _ = net(sample_data)
    except:
        _, _ = net(sample_data)

    return net

def repeat_and_convert_packets(packets, packets_dict, address_length, neuron_idx=True):
    #final_packet_list = []
    #expanded_packets_list = []
    expanded_packets_dict = {}

    dictionary = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: []
    }
    for source_neuron_index, dest_neuron_start_index, source_core, destination_cores, reps in packets:
        address = ""
    
        # Append '0' to the address until it reaches the desired address length
        while len(address) < address_length:
            address += "0"
            # Convert string to list
        
        string_list = list(address)

        for index in destination_cores:
            string_list[index] = '1'

        # Convert list back to string
        updated_address = ''.join(string_list)
        
        prev_mock_message = None
        for i in range(reps):
            if neuron_idx:
                mock_message = generate_message_neuron_idx(source_neuron_index, dest_neuron_start_index+i)
            else:
                mock_message = generate_message()
                if mock_message == prev_mock_message:
                    while mock_message == prev_mock_message:
                        mock_message = generate_message()

                prev_mock_message = mock_message
            temp = [mock_message+updated_address]

            dictionary[source_core].extend(temp)

            #expanded_packets_list.extend(_expand_address_list(updated_address, mock_message))
            expanded_packets_dict[mock_message] = _count_address_list(updated_address, mock_message)

    return dictionary, expanded_packets_dict

# def _expand_address_list(in_a, in_m, addr_width=5):
#     out_list = []

#     # Iterate over each bit in the address width
#     for i in range(addr_width):
#         if in_a[i] == "1":
#             bit_value = 1 << (addr_width - 1)
#             temp = bit_value >> i
#             formatted_value = f"{temp:0{addr_width}b}"
#             # Shift 1 to the position of each set bit, concatenate with message
#             out_list.append(in_m + formatted_value)

#     return out_list

def _count_address_list(in_a, in_m, addr_width=5):
    #out_list = []
    counter = 0

    # Iterate over each bit in the address width
    for i in range(addr_width):
        if in_a[i] == "1":
            counter += 1
    return counter

def generate_message_neuron_idx(s_idx, d_idx, message_width=20):
    
    return format(s_idx, f'0{int(message_width/2)}b') + format(d_idx, f'0{int(message_width/2)}b')

def generate_message(message_width=20):
    global counter

    # Calculate the maximum value based on the message width
    max_value = (1 << message_width) - 1

    # Use the counter as the message and increment it
    message = counter
    counter += 1

    # Reset the counter when it reaches the maximum value
    if counter > max_value:
        counter = 0

    # Convert the message to a binary string, zero-padded to message_width bits
    message_bits = f'{message:0{message_width}b}'
    
    return message_bits

def count_target_neurons(layer_name, source_core, idx, target_cores, buffer_map):

    num_lr_dest_neurons = 0
    num_sr_dest_neurons = 0

    how_many_to_subtract_sr = 0
    how_many_to_subtract_lr = 0

    for target_core, reps in target_cores:
        
        connection = layer_name+"-"+str(idx)+"-"+str(target_core)

        # remove inter-core communication
        if source_core == target_core:
            if connection in buffer_map:
                how_many_to_subtract_sr += int(buffer_map[connection])
            
            num_sr_dest_neurons += reps

        else:
            if connection in buffer_map:
                how_many_to_subtract_lr += int(buffer_map[connection])
            
            num_lr_dest_neurons += reps

    return num_lr_dest_neurons, num_sr_dest_neurons, how_many_to_subtract_lr, how_many_to_subtract_sr

def calculate_lr_sr_conns(mapping, graph):
    
    num_long_range_conns = 0
    num_short_range_conns = 0

    source_layer = mapping.indices_to_lock['layers'][0]
    dest_layer = mapping.indices_to_lock['layers'][1]
    
    for layer_name, size in mapping.mem_potential_sizes.items():
        if layer_name == source_layer:
            for source_core, start_idx, end_idx in mapping.core_allocation[layer_name]:
                downstream_nodes = list(graph.graph.successors(layer_name))
                for idx in range(start_idx, end_idx+1):
                    target_cores = []
                    for downstream_node in downstream_nodes:
                        #if downstream_node != "output":
                        if downstream_node == dest_layer:
                            target_cores = mapping.NIR_to_cores[downstream_node]
                            lr_target_nerons, sr_target_neurons, lr_subtract, sr_subtract = count_target_neurons(layer_name, source_core, idx, target_cores, mapping.buffer_map)

                            
                            num_long_range_conns += lr_target_nerons - lr_subtract
                            num_short_range_conns += sr_target_neurons - sr_subtract

    return num_long_range_conns, num_short_range_conns
    
def choose_conn_remove(mapping, reps=None):
    source_layer = mapping.indices_to_lock['layers'][0]
    dest_layer = mapping.indices_to_lock['layers'][1]

    source_allocation = mapping.core_allocation[source_layer]
    dest_allocation = mapping.core_allocation[dest_layer]

    not_found = True
    counter = 0

    while not_found:
        source_rand_core = random.randint(0, len(source_allocation)-1)
        dest_rand_core = random.randint(0, len(dest_allocation)-1)

        source_core_allocation = source_allocation[source_rand_core]
        dest_core_allocation = dest_allocation[dest_rand_core]


        if source_core_allocation[0] == dest_core_allocation[0]:
            continue

        source_rand_index = random.randint(source_core_allocation[1], source_core_allocation[2])
        dest_rand_index = random.randint(dest_core_allocation[1], dest_core_allocation[2])

        if (source_rand_index, dest_rand_index) in mapping.indices_to_lock['indices']:
            continue

        mapping.indices_to_lock['indices'].append((source_rand_index, dest_rand_index))
        if reps is None:
            not_found = False
        else:
            counter += 1
            if counter >= reps:
                not_found = False

    mapping.map_buffers()

    return mapping

