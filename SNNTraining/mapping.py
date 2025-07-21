import torch
import snntorch as snn  

class Mapping:
    def __init__(self, net=None, num_steps=None, num_inputs=None, mem_potential_sizes=None):
        self.num_steps = num_steps
        self.num_inputs = num_inputs
        self.core_capacity = None
        self.net = net

        if mem_potential_sizes is not None:
            self.mem_potential_sizes = mem_potential_sizes
        else:
            self.mem_potential_sizes = self._get_membrane_potential_sizes() if net is not None else {}

        self.buffer_map = None
        self.indices_to_lock = None
    
    def _get_membrane_potential_sizes(self):
        if self.net is None:
            raise ValueError("Network model has not been set. Please call set_network first.")
        
        sizes = {}
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
    
    def map_neurons(self):
        self.core_allocation, self.NIR_to_cores, self.neuron_to_core = self._allocate_neurons_to_cores()

    def set_core_capacity(self, cc):
        self.core_capacity = cc

    def log(self, dut=None):

        print("\n----- MAPPING -----\n")

        for layer_name, size in self.mem_potential_sizes.items():
            temp = f"Layer: {layer_name}, Number of neurons: {size}"
            if dut is not None:
                    dut._log.info(temp)
            else:
                print(temp)

        print("CORE ALLOCATION:",self.core_allocation)
        print("NIR TO CORES:",self.NIR_to_cores)
        print("BUFFER MAP:",self.buffer_map)

        print("CORE CAPACITY", self.core_capacity)
    
    def _allocate_neurons_to_cores(self):
        import math
        import random

        core_allocation = {}
        NIR_to_cores = {}
        neuron_to_core = {}

        layer_names = list(self.mem_potential_sizes.keys())
        total_neurons = sum(self.mem_potential_sizes.values())

        # Determine number of cores needed
        total_cores = math.ceil(total_neurons / self.core_capacity)

        # Create a list of all neurons with (layer_name, neuron_index)
        all_neurons = []
        for layer_name in layer_names:
            for nid in range(self.mem_potential_sizes[layer_name]):
                all_neurons.append((layer_name, nid))

        # Shuffle for random assignment
        random.shuffle(all_neurons)

        core_id = 0
        core_counts = [0] * total_cores
        core_buckets = [[] for _ in range(total_cores)]

        for layer_name, nid in all_neurons:
            # Assign to the first core with available capacity
            while core_counts[core_id] >= self.core_capacity:
                core_id = (core_id + 1) % total_cores

            core_buckets[core_id].append((layer_name, nid))
            neuron_to_core[f"{layer_name}-{nid}"] = core_id
            core_counts[core_id] += 1

        # Build core_allocation and NIR_to_cores per layer
        for layer_name in layer_names:
            layer_core_allocation = []
            layer_NIR_to_cores = {}

            for cid, bucket in enumerate(core_buckets):
                layer_bucket = [nid for lname, nid in bucket if lname == layer_name]
                if layer_bucket:
                    start_id = min(layer_bucket)
                    end_id = max(layer_bucket)
                    layer_core_allocation.append((cid, start_id, end_id))
                    layer_NIR_to_cores[cid] = len(layer_bucket)

            core_allocation[layer_name] = layer_core_allocation
            NIR_to_cores[layer_name] = list(layer_NIR_to_cores.items())

        return core_allocation, NIR_to_cores, neuron_to_core
    
    def map_buffers(self, indices_to_lock=None):

        if indices_to_lock is not None:
            self.indices_to_lock = indices_to_lock

        mapped_buffer = {}
        for indices in self.indices_to_lock['indices']:
            temp = ""
            #for idx, layer_name in enumerate(indices_to_lock['layers']):
                #if idx == 0:
            temp += str(self.indices_to_lock['layers'][0])+"-"+str(indices[0]) +"-"
            temp += str(self.neuron_to_core[str(self.indices_to_lock['layers'][1]) + "-" + str(indices[1])])

            if temp not in mapped_buffer:
                mapped_buffer[temp] = 1
            else:
                mapped_buffer[temp] += 1

        #return mapped_buffer
        self.buffer_map = mapped_buffer

    def get_mappings(self):
        return self.core_allocation, self.NIR_to_cores, self.neuron_to_core