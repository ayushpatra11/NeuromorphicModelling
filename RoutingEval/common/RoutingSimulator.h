/*####################################################################################
#
#   File Name: RoutingSimulator.h
#   Author:  Ayush Patra
#   Description: Simulates Neurogrid-style routing from one neuron to another using
#                a binary tree of cores. Computes and logs routing waste based on
#                deviation from expected connectivity (connectivity matrix).
#   Version History:
#       - 2025-07-26: Initial version
#
####################################################################################*/

#ifndef ROUTING_SIMULATOR_H
#define ROUTING_SIMULATOR_H

#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utils.h"

class RoutingSimulator {
private:
    // Declaration order matches the constructor initializer-list order to avoid -Wreorder
    const std::vector<std::vector<int>>& connectivityMatrix;
    const std::unordered_map<int, int>& neuronToCoreMap;
    const std::unordered_map<int, std::vector<int>>& coreTree;
    const std::unordered_map<int, int>& coreParent;
    Utils routingUtils;
    float weightThreshold;
    string reportDir;

    std::unordered_map<int, std::unordered_set<int>> actualTargetsPerNeuron;
    std::unordered_map<int, std::unordered_set<int>> visitedNeuronsPerNeuron;
    std::map<int, int> wastePerNeuron;
    std::unordered_map<int, int> wastedMessages;

    void traverseTree(int coreId, std::unordered_set<int>& visitedCores);
    void simulateNeuronToNeuron(int sourceNeuron);
    //void routeMessage(int srcCore, int tgtCore, std::unordered_set<int>& visitedCores);

public:
    RoutingSimulator(const std::vector<std::vector<int>>& connectivityMatrix,
                     const std::unordered_map<int, int>& neuronToCoreMap,
                     const std::unordered_map<int, std::vector<int>>& coreTree,
                     const std::unordered_map<int, int>& coreParent, Utils routingUtils, string reportDir);

    void simulate();
    void reportWasteStatistics() const;
    long long getTotalWaste() const;
    std::unordered_map<int, int> getWastedMessagesPerCore() const;
    int findLCA(int sourceCore, int targetCore);
    bool isDescendant(int current, int target);
    std::vector<int> shortestPath(int startCore, int endCore);
    ~RoutingSimulator() {
    }
};

#endif  // ROUTING_SIMULATOR_H