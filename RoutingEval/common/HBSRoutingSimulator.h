/*####################################################################################
#
#   File Name: HBSRoutingSimulator.h
#   Author:  Ayush Patra
#   Description: Simulates HBS (Hierarchical Bit String) routing from one neuron to
#                another using a tree of switches with 4-wide leaf groups. Computes
#                and logs routing waste based on deviation from expected connectivity
#                (connectivity matrix).
#   Version History:
#       - 2025-08-06: Initial version
#
####################################################################################*/

#ifndef HBS_ROUTING_SIMULATOR_H
#define HBS_ROUTING_SIMULATOR_H

#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utils.h"

class HBSRoutingSimulator {
private:
    // Declaration order matches the constructor initializer-list order to avoid -Wreorder
    Utils routingUtils;
    float weightThreshold;
    const std::vector<std::vector<int>>& connectivityMatrix;
    const std::unordered_map<int, int>& neuronToCoreMap;
    const std::unordered_map<int, std::vector<int>>& coreTree;
    const std::unordered_map<int, int>& coreParent;
    string reportDir;

    std::unordered_map<int, std::unordered_set<int>> actualTargetsPerNeuron;
    std::unordered_map<int, std::unordered_set<int>> visitedNeuronsPerNeuron;
    std::map<int, int> wastePerNeuron;
    std::unordered_map<int, int> wastedMessages;

    void traverseTree(int coreId, std::unordered_set<int>& visitedCores);
    void simulateNeuronToNeuron(int sourceNeuron);

    // Helper: reconstruct (dstNeuron, coreId) pairs from the connectivity row
    std::vector<std::pair<int, int>> buildTargetNeuronCoreList(int sourceNeuron) const;
    // Helper: log the routing summary header and per-core target lists
    void logRoutingSummary(int sourceNeuron, int sourceCore,
                           const std::unordered_map<int, std::vector<int>>& coreToDstNeurons,
                           const std::vector<std::pair<int, int>>& targetNeuronCoreList,
                           const std::unordered_set<int>& targetCores);
    // Helper: group each target core by its parent switch and child index
    void buildParentSwitchMap(
        const std::unordered_set<int>& targetCores, int sourceCore,
        std::unordered_map<int, std::unordered_map<int, std::unordered_set<int>>>& parentToChildIdxTargets,
        std::unordered_set<int>& parentSwitches);
    // Helper: compute global OR mask of child indices across all parent switches, and log it
    void computeGlobalMask(
        const std::unordered_set<int>& parentSwitches,
        const std::unordered_map<int, std::unordered_map<int, std::unordered_set<int>>>& parentToChildIdxTargets,
        std::unordered_map<int, std::unordered_set<int>>& localMasks, std::unordered_set<int>& globalMaskIndices);
    // Helper: compute and accumulate waste for each parent under the global mask
    void computeWastePerParent(
        int sourceNeuron, const std::unordered_set<int>& parentSwitches,
        const std::unordered_set<int>& globalMaskIndices,
        const std::unordered_map<int, std::unordered_map<int, std::unordered_set<int>>>& parentToChildIdxTargets);

public:
    HBSRoutingSimulator(const std::vector<std::vector<int>>& connectivityMatrix,
                        const std::unordered_map<int, int>& neuronToCoreMap,
                        const std::unordered_map<int, std::vector<int>>& coreTree,
                        const std::unordered_map<int, int>& coreParent, Utils routingUtils, string reportDir);

    void simulate();
    void reportWasteStatistics();
    long long getTotalWaste() const;
    std::unordered_map<int, int> getWastedMessagesPerCore() const;
    int findLCA(int sourceCore, int targetCore);
    bool isDescendant(int current, int target);
    std::vector<int> shortestPath(int startCore, int endCore);
    ~HBSRoutingSimulator() {
    }
};

#endif  // HBS_ROUTING_SIMULATOR_H