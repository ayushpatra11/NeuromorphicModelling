//####################################################################################
//#
//#   File Name: HBSNeuronMapper.h
//#   Author:  Ayush Patra
//#   Description: Maps neurons to cores based on the HBS (Hierarchical Bit String)
//#                routing strategy. Loads an existing neuron-to-core mapping and
//#                builds a non-binary HBS tree (4-wide leaf switches, binary upper
//#                levels) for simulating multicast routing.
//#   Version History:
//#       - 2025-08-06: Initial version
//#
//####################################################################################

#ifndef HBS_NEURON_MAPPER_H
#define HBS_NEURON_MAPPER_H

#include <fstream>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"  // If you're using JSON (https://github.com/nlohmann/json)

using json = nlohmann::json;

class HBSNeuronMapper {
private:
    int num_neurons;
    int neurons_per_core;
    int core_count;
    int root_id;
    std::unordered_map<int, int> neuron_to_core;
    std::unordered_map<int, std::pair<int, int>> core_children;  // left and right child for each core
    std::unordered_map<int, int> core_parent;                    // parent for each core
    std::unordered_map<int, std::vector<int>> core_tree;
    const std::vector<std::vector<int>>& connectivity_matrix;

    void buildBinaryTree();

public:
    HBSNeuronMapper(int total_neurons, int neurons_per_core, const std::vector<std::vector<int>>& conn_matrix);
    void mapNeurons();
    void buildHBSTree(int core_count, std::unordered_map<int, std::vector<int>>& core_tree,
                      std::unordered_map<int, int>& core_parent, int& root_id);
    int getCoreForNeuron(int neuron_id) const;
    const std::unordered_map<int, int>& getNeuronToCoreMap() const;
    const std::unordered_map<int, std::vector<int>>& getCoreTree() const;
    const std::unordered_map<int, int>& getCoreParent() const;
    int getParentCore(int core_id) const;
    int getTotalCores() const;
    void exportCoreTreeToJson(const std::string& filename) const;
    void exportCoreNeuronMapToJson(const std::string& filename) const;
    int getRootId() const;
    void logCoreTreeRecursive(int node, const std::unordered_map<int, std::vector<int>>& core_tree, std::ostream& out,
                              std::string prefix, bool isLeft, int max_leaf_id);
    void serializeCoreTree(int node, const std::unordered_map<int, std::vector<int>>& core_tree, json& j) const;
    ~HBSNeuronMapper() {
    }
};

#endif  // HBS_NEURON_MAPPER_H