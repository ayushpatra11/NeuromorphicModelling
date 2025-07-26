/************************************************************************************
*
*   File Name: NeuronMapper.cpp
*   Author:  Ayush Patra
*   Description: Implements mapping of neurons to cores based on Neurogrid-like 
*                routing. Each core supports a fixed number of neurons.
*   Version History:        
*       - 2025-07-26: Initial version
*
************************************************************************************/

#include "NeuronMapper.h"
#include <algorithm>
#include <random>

using json = nlohmann::json;

NeuronMapper::NeuronMapper(int total_neurons, int neurons_per_core) 
    : num_neurons(total_neurons), neurons_per_core(neurons_per_core) {
    core_count = (num_neurons + neurons_per_core - 1) / neurons_per_core;
    mapNeurons();
}

void NeuronMapper::mapNeurons() {
    neuron_to_core.clear();
    std::vector<int> core_ids(core_count);
    for (int i = 0; i < core_count; ++i) {
        core_ids[i] = i;
    }

    std::vector<int> neuron_indices(num_neurons);
    for (int i = 0; i < num_neurons; ++i) {
        neuron_indices[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(neuron_indices.begin(), neuron_indices.end(), g);

    for (int i = 0; i < num_neurons; ++i) {
        int core_id = i / neurons_per_core;
        neuron_to_core[neuron_indices[i]] = core_id;
    }

    // Build binary tree among cores and populate core_tree, core_children, core_parent
    core_tree.clear();
    core_children.clear();
    core_parent.clear();
    for (int i = 0; i < core_count; ++i) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        std::vector<int> children;
        if (left < core_count) {
            children.push_back(left);
            core_children[i].first = left;
            core_parent[left] = i;
        }
        if (right < core_count) {
            children.push_back(right);
            core_children[i].second = right;
            core_parent[right] = i;
        }
        core_tree[i] = children;
    }
    exportCoreTreeToJson("../data/core_tree.json");
    exportCoreNeuronMapToJson("../data/neuron_to_core_map.json");
}

int NeuronMapper::getCoreForNeuron(int neuron_id) const {
    auto it = neuron_to_core.find(neuron_id);
    if (it != neuron_to_core.end()) {
        return it->second;
    }
    return -1; // invalid neuron id
}

const std::unordered_map<int, int>& NeuronMapper::getNeuronToCoreMap() const {
    return neuron_to_core;
}

int NeuronMapper::getTotalCores() const {
    return core_count;
}

const std::unordered_map<int, std::vector<int>>& NeuronMapper::getCoreTree() const {
    return core_tree;
}

void NeuronMapper::exportCoreTreeToJson(const std::string& filename) const {
    json j;
    for (const auto& [core, children] : core_tree) {
        j[std::to_string(core)] = children;
    }

    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Failed to open file for writing core tree.");
    out << j.dump(4);
    out.close();
}

void NeuronMapper::exportCoreNeuronMapToJson(const std::string& filename) const {
    json j;
    std::unordered_map<int, std::vector<int>> core_to_neurons;

    for (const auto& [neuron, core] : neuron_to_core) {
        core_to_neurons[core].push_back(neuron);
    }

    for (const auto& [core, neurons] : core_to_neurons) {
        j[std::to_string(core)] = neurons;
    }

    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Failed to open file for writing core-to-neurons map.");
    out << j.dump(4);
    out.close();
}