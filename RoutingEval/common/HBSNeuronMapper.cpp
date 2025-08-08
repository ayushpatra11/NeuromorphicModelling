/************************************************************************************
*
*   File Name: HBSNeuronMapper.cpp
*   Author:  Ayush Patra
*   Description: Implements mapping of neurons to cores based on HBS-like 
*                routing. Each core supports a fixed number of neurons.
*   Version History:        
*       - 2025-08-06: Initial version
*
************************************************************************************/

#include "HBSNeuronMapper.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>
#include <queue>
#include <unordered_set>

using json = nlohmann::json;

void HBSNeuronMapper::logCoreTreeRecursive(int node, const std::unordered_map<int, std::vector<int>>& core_tree, std::ostream& out, std::string prefix = "", bool isLeft = true, int max_leaf_id = -1) {
    out << prefix;
    out << (isLeft ? "├── " : "└── ");

    bool isLeaf = (core_tree.find(node) == core_tree.end());
    if (isLeaf && node < max_leaf_id)
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

HBSNeuronMapper::HBSNeuronMapper(int total_neurons, int neurons_per_core, const std::vector<std::vector<int>>& conn_matrix)
    : num_neurons(total_neurons), neurons_per_core(neurons_per_core), connectivity_matrix(conn_matrix) {
    core_count = (num_neurons + neurons_per_core - 1) / neurons_per_core;
    mapNeurons();
}

void HBSNeuronMapper::mapNeurons() {
    // Check if neuron-to-core mapping JSON already exists
    std::ifstream infile("../data/neuron_to_core_map.json");
    if (infile.good()) {
        // If file exists, load mapping and populate neuron_to_core, then return early.
        json mapping_json;
        infile >> mapping_json;
        infile.close();
        neuron_to_core.clear();
        // mapping_json: { "core_id": [neuron1, neuron2, ...], ... }
        for (auto it = mapping_json.begin(); it != mapping_json.end(); ++it) {
            int core_id = std::stoi(it.key());
            const auto& neurons = it.value();
            for (int neuron : neurons) {
                neuron_to_core[neuron] = core_id;
            }
        }
        // Optionally, could also reconstruct core_tree/core_parent from file if stored.
        buildHBSTree(core_count, core_tree, core_parent, root_id);

        // Export core tree to JSON and log structure
        exportCoreTreeToJson("../data/core_tree/hbs_core_tree.json");
        std::ofstream tree_log_out("../data/core_tree/hbs_core_tree.txt");
        if (tree_log_out.is_open()) {
            logCoreTreeRecursive(root_id, core_tree, tree_log_out, "", false, core_count);
            tree_log_out.close();
        } else {
            std::cerr << "Failed to open file to log HBS core tree structure.\n";
        }

        return;
    }
    infile.close();

    // If file does not exist, throw error as currently done.
    throw std::runtime_error("Neuron-to-core mapping JSON does not exist. Please generate the mapping first.");
}

// Helper to build non-binary HBS tree: leaf switches with 4 cores, others binary.
void HBSNeuronMapper::buildHBSTree(
    int core_count,
    std::unordered_map<int, std::vector<int>>& core_tree,
    std::unordered_map<int, int>& core_parent,
    int& root_id
) {
    // Bottom-up: group cores into groups of 4 for leaf-level switches.
    std::vector<int> leaf_cores(core_count);
    std::iota(leaf_cores.begin(), leaf_cores.end(), 0);
    std::vector<int> leaf_switches;
    int next_node_id = core_count;
    for (size_t i = 0; i < leaf_cores.size(); i += 4) {
        std::vector<int> children;
        for (size_t j = 0; j < 4; ++j) {
            if (i + j < leaf_cores.size()) {
                children.push_back(leaf_cores[i + j]);
                core_parent[leaf_cores[i + j]] = next_node_id;
            }
        }
        core_tree[next_node_id] = children;
        leaf_switches.push_back(next_node_id);
        ++next_node_id;
    }
    // Now, build higher levels as binary tree over leaf switches (and then internal switches)
    std::vector<int> current_level = leaf_switches;
    while (current_level.size() > 1) {
        std::vector<int> next_level;
        for (size_t i = 0; i < current_level.size(); i += 2) {
            int left = current_level[i];
            int right = (i + 1 < current_level.size()) ? current_level[i + 1] : -1;
            int parent = next_node_id++;
            std::vector<int> children;
            children.push_back(left);
            core_parent[left] = parent;
            if (right != -1) {
                children.push_back(right);
                core_parent[right] = parent;
            }
            core_tree[parent] = children;
            next_level.push_back(parent);
        }
        current_level = next_level;
    }
    // The last node is the root
    if (!current_level.empty()) {
        root_id = current_level[0];
        // root_id is stored in the class and used for traversal/logging
        this->root_id = root_id;
        core_parent[root_id] = -1;
    }
}

int HBSNeuronMapper::getCoreForNeuron(int neuron_id) const {
    auto it = neuron_to_core.find(neuron_id);
    if (it != neuron_to_core.end()) {
        return it->second;
    }
    return -1; // invalid neuron id
}

const std::unordered_map<int, int>& HBSNeuronMapper::getNeuronToCoreMap() const {
    return neuron_to_core;
}

int HBSNeuronMapper::getTotalCores() const {
    return core_count;
}

const std::unordered_map<int, std::vector<int>>& HBSNeuronMapper::getCoreTree() const {
    return core_tree;
}

const std::unordered_map<int, int>& HBSNeuronMapper::getCoreParent() const {
    return core_parent;
}

// Recursively serialize the core tree as a nested JSON structure
void HBSNeuronMapper::serializeCoreTree(int node, const std::unordered_map<int, std::vector<int>>& core_tree, json& j) const {
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

void HBSNeuronMapper::exportCoreTreeToJson(const std::string& filename) const {
    if (core_parent.empty()) return;

    int root = root_id;

    json j;
    serializeCoreTree(root, core_tree, j);

    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Failed to open file for writing core tree.");
    out << j.dump(4);
    out.close();
}

void HBSNeuronMapper::exportCoreNeuronMapToJson(const std::string& filename) const {
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

int HBSNeuronMapper::getRootId() const {
    return root_id;
}