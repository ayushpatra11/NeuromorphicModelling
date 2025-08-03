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
#include <fstream>
#include <queue>
#include <unordered_set>

using json = nlohmann::json;

void logCoreTreeRecursive(int node, const std::unordered_map<int, std::vector<int>>& core_tree, std::ostream& out, std::string prefix = "", bool isLeft = true, int max_leaf_id = -1) {
    out << prefix;
    out << (isLeft ? "├── " : "└── ");

    bool isLeaf = core_tree.find(node) == core_tree.end();
    if (isLeaf)
        out << "Core " << node << "\n";
    else
        out << "Network Switch " << node << "\n";

    auto it = core_tree.find(node);
    if (it != core_tree.end()) {
        const auto& children = it->second;
        for (size_t i = 0; i < children.size(); ++i) {
            int child = children[i];
            if (child == -1) continue;
            bool childIsLeft = (i != children.size() - 1);
            logCoreTreeRecursive(child, core_tree, out, prefix + (isLeft ? "│   " : "    "), childIsLeft, max_leaf_id);
        }
    }
}

NeuronMapper::NeuronMapper(int total_neurons, int neurons_per_core, const std::vector<std::vector<int>>& conn_matrix)
    : num_neurons(total_neurons), neurons_per_core(neurons_per_core), connectivity_matrix(conn_matrix) {
    core_count = (num_neurons + neurons_per_core - 1) / neurons_per_core;
    mapNeurons();
}

void NeuronMapper::mapNeurons() {
    neuron_to_core.clear();

    int num_cores = num_neurons / neurons_per_core;
    std::vector<std::vector<int>> clusters(num_cores);
    std::vector<bool> visited(num_neurons, false);

    std::vector<int> neuron_ids(num_neurons);
    std::iota(neuron_ids.begin(), neuron_ids.end(), 0);
    std::shuffle(neuron_ids.begin(), neuron_ids.end(), std::mt19937{std::random_device{}()});

    int cluster_index = 0;

    for (int i : neuron_ids) {
        if (visited[i]) continue;
        std::queue<int> q;
        q.push(i);
        visited[i] = true;

        while (!q.empty() && clusters[cluster_index].size() < neurons_per_core) {
            int n = q.front(); q.pop();
            clusters[cluster_index].push_back(n);
            for (int j = 0; j < num_neurons; ++j) {
                if (!visited[j] && n != j && connectivity_matrix[n][j] > 0) {
                    visited[j] = true;
                    q.push(j);
                }
            }
        }

        cluster_index = (cluster_index + 1) % num_cores;
    }

    for (int i = 0; i < num_neurons; ++i) {
        if (!visited[i]) {
            clusters[cluster_index].push_back(i);
            cluster_index = (cluster_index + 1) % num_cores;
        }
    }

    for (int c = 0; c < num_cores; ++c) {
        for (int neuron : clusters[c]) {
            neuron_to_core[neuron] = c;
        }
    }

    core_count = num_cores;

    // Tree construction remains unchanged...
    core_tree.clear();
    core_children.clear();
    core_parent.clear();

    int next_id = core_count;
    std::vector<int> current_level;
    for (int i = 0; i < core_count; ++i) {
        current_level.push_back(i);
    }

    while (current_level.size() > 1) {
        std::vector<int> next_level;
        for (size_t i = 0; i + 1 < current_level.size(); i += 2) {
            int left = current_level[i];
            int right = current_level[i + 1];
            int parent = next_id++;

            core_tree[parent] = {left, right};
            core_children[parent] = {left, right};
            core_parent[left] = parent;
            core_parent[right] = parent;

            next_level.push_back(parent);
        }
        if (current_level.size() % 2 == 1) {
            int lone = current_level.back();
            int dummy = next_id++;
            core_tree[dummy] = {lone, -1};
            core_children[dummy] = {lone, -1};
            core_parent[lone] = dummy;
            next_level.push_back(dummy);
        }
        current_level = next_level;
    }

    if (!current_level.empty()) {
        int root = current_level[0];
        if (core_parent.find(root) == core_parent.end()) {
            core_parent[root] = -1;
        }
    }

    core_count = next_id;

    exportCoreTreeToJson("../data/core_tree.json");
    exportCoreNeuronMapToJson("../data/neuron_to_core_map.json");

    std::ofstream logFile("../data/core_tree_structure.txt");
    if (logFile) {
        logFile << "Binary Core Tree Structure:\n";
        if (!current_level.empty()) {
            logCoreTreeRecursive(current_level[0], core_tree, logFile, "", false, core_count - (next_id - core_count));
        }
        logFile.close();
    }
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

const std::unordered_map<int, int>& NeuronMapper::getCoreParent() const {
    return core_parent;
}

// Recursively serialize the core tree as a nested JSON structure
void serializeCoreTree(int node, const std::unordered_map<int, std::vector<int>>& core_tree, json& j) {
    j["core"] = node;
    if (core_tree.find(node) != core_tree.end()) {
        j["children"] = json::array();
        for (int child : core_tree.at(node)) {
            if (child == -1) continue;
            json childJson;
            serializeCoreTree(child, core_tree, childJson);
            j["children"].push_back(childJson);
        }
    }
}

void NeuronMapper::exportCoreTreeToJson(const std::string& filename) const {
    if (core_parent.empty()) return;

    int root = -1;
    for (const auto& [core, parent] : core_parent) {
        if (parent == -1) {
            root = core;
            break;
        }
    }

    json j;
    serializeCoreTree(root, core_tree, j);

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