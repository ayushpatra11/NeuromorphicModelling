//####################################################################################
//#
//#   File Name: NeuronMapper.h
//#   Author:  Ayush Patra
//#   Description: Maps neurons to cores based on the Neurogrid routing strategy.
//#                Randomly assigns neurons to cores. Models core-to-core connectivity 
//#                using a binary tree for simulating multicast routing.
//#   Version History:        
//#       - 2025-07-26: Initial version
//#       - 2025-07-26: Updated to binary tree topology for cores
//#
//####################################################################################

#ifndef NEURON_MAPPER_H
#define NEURON_MAPPER_H

#include <unordered_map>
#include <vector>
#include <utility>
#include <random>
#include <stdexcept>
#include <fstream>
#include "nlohmann/json.hpp"  // If you're using JSON (https://github.com/nlohmann/json)

using json = nlohmann::json;

class NeuronMapper {
private:
    int num_neurons;
    int neurons_per_core;
    int core_count;
    std::unordered_map<int, int> neuron_to_core;
    std::unordered_map<int, std::pair<int, int>> core_children; // left and right child for each core
    std::unordered_map<int, int> core_parent; // parent for each core
    std::unordered_map<int, std::vector<int>> core_tree;
    const std::vector<std::vector<int>>& connectivity_matrix; 

    void buildBinaryTree();

public:
    NeuronMapper(int total_neurons, int neurons_per_core, const std::vector<std::vector<int>>& conn_matrix);
    void mapNeurons();
    void assignNeuronsToCores();
    int getCoreForNeuron(int neuron_id) const;
    const std::unordered_map<int, int>& getNeuronToCoreMap() const;
    const std::unordered_map<int, std::vector<int>>& getCoreTree() const;
    const std::unordered_map<int, int>& getCoreParent() const;
    int getParentCore(int core_id) const;
    int getTotalCores() const;
    void exportCoreTreeToJson(const std::string& filename) const;
    void exportCoreNeuronMapToJson(const std::string& filename) const; 
    void logCoreTreeRecursive(int node, const std::unordered_map<int, std::vector<int>>& core_tree, std::ostream& out, std::string prefix, bool isLeft, int max_leaf_id);
    void serializeCoreTree(int node, const std::unordered_map<int, std::vector<int>>& core_tree, json& j) const;
    ~NeuronMapper(){}
};

#endif // NEURON_MAPPER_H