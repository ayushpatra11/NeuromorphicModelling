import torch
import snntorch as snn  # Ensure this module is correctly imported

class Mapping:
    def __init__(self, net, num_steps, num_inputs):
        self.num_steps = num_steps
        self.num_inputs = num_inputs
        self.core_capacity = None
        self.net = net

        self.mem_potential_sizes = self._get_membrane_potential_sizes()
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
        core_allocation = {}
        NIR_to_cores = {}
        neuron_to_core = {}

        core_id = 0
        core_start_index = 0
        current_core_neurons = 0
        full_capacity_reached = False

        layer_names = list(self.mem_potential_sizes.keys())
        last_layer_name = layer_names[-1]

        for layer_name, num_neurons in self.mem_potential_sizes.items():
            layer_start_index = core_start_index

            if layer_name == last_layer_name:
                if num_neurons > self.core_capacity:
                    raise Exception("Output layer does not fit in one core!")

                # Ensure the last layer is in the same core
                if not full_capacity_reached:
                    core_id += 1
                core_start_index = 0
                current_core_neurons = 0
                layer_start_index = core_start_index
                layer_end_index = layer_start_index + num_neurons - 1
                core_allocation[layer_name] = [(core_id, layer_start_index, layer_end_index)]
                NIR_to_cores[layer_name] = [(core_id, layer_end_index + 1 - layer_start_index)]
                for neuron_id in range(layer_start_index, layer_end_index + 1):
                    neuron_to_core[layer_name + "-" + str(neuron_id)] = core_id
                break

            while num_neurons > 0:
                full_capacity_reached = False
                available_space = self.core_capacity - current_core_neurons
                neurons_to_allocate = min(num_neurons, available_space)

                layer_end_index = layer_start_index + neurons_to_allocate - 1

                if layer_name not in core_allocation:
                    core_allocation[layer_name] = []
                    NIR_to_cores[layer_name] = []

                core_allocation[layer_name].append((core_id, layer_start_index, layer_end_index))
                NIR_to_cores[layer_name].append((core_id, layer_end_index + 1 - layer_start_index))

                for neuron_id in range(layer_start_index, layer_end_index + 1):
                    neuron_to_core[layer_name + "-" + str(neuron_id)] = core_id

                current_core_neurons += neurons_to_allocate
                layer_start_index += neurons_to_allocate
                num_neurons -= neurons_to_allocate

                if current_core_neurons == self.core_capacity:
                    full_capacity_reached = True
                    core_id += 1
                    core_start_index = 0
                    current_core_neurons = 0
                else:
                    core_start_index = layer_start_index

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
