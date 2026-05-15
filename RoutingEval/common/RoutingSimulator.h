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

    // Helper: finds the root node of the core tree (the one with no parent)
    int findRootCore() const;
    // Helper: builds the map of target cores → target neurons for a source neuron
    std::map<int, std::vector<int>> buildTargetCores(int src) const;
    // Helper: resolves the LCA across all target cores; returns -1 if routing should be skipped
    int computeLCA(int srcCore, const std::unordered_set<int>& actualTargetCores, int rootCore);
    // Helper: traces the U/D/L/R/B routing path from srcCore up to the LCA
    std::string traceRoutingPath(int srcCore, int lcaCore, int rootCore);
    // Helper: counts waste as non-target leaf cores under the LCA subtree
    int computeWasteUnderLCA(int src, int lcaCore, const std::unordered_set<int>& actualTargetCores);
    // Helper: logs the final waste summary and writes the JSON report to disk
    void writeReport();

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