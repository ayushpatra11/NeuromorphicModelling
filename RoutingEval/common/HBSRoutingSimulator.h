/*####################################################################################
#
#   File Name: HBSRoutingSimulator.h
#   Author:  Ayush Patra
#   Description: Simulates Neurogrid-style routing from one neuron to another using
#                a binary tree of switches with 4 cores in the leaves. Computes and 
#                logs routing waste based on deviation from expected connectivity 
#                (connectivity matrix).
#   Version History:
#       - 2025-08-06: Initial version
#
####################################################################################*/

#ifndef HBS_ROUTING_SIMULATOR_H
#define HBS_ROUTING_SIMULATOR_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <algorithm>
#include "Utils.h"

class HBSRoutingSimulator {
private:
    Utils routingUtils;
    float weightThreshold;
    string reportDir;
    const std::vector<std::vector<int>>& connectivityMatrix;
    const std::unordered_map<int, int>& neuronToCoreMap;
    const std::unordered_map<int, std::vector<int>>& coreTree;
    const std::unordered_map<int, int>& coreParent;
    
    std::unordered_map<int, std::unordered_set<int>> actualTargetsPerNeuron;
    std::unordered_map<int, std::unordered_set<int>> visitedNeuronsPerNeuron;
    std::map<int, int> wastePerNeuron;
    std::unordered_map<int, int> wastedMessages;
    
    void traverseTree(int coreId, std::unordered_set<int>& visitedCores);
    void simulateNeuronToNeuron(int sourceNeuron);
    //void routeMessage(int srcCore, int tgtCore, std::unordered_set<int>& visitedCores);

public:
        HBSRoutingSimulator(const std::vector<std::vector<int>>& connectivityMatrix,
                         const std::unordered_map<int, int>& neuronToCoreMap,
                         const std::unordered_map<int, std::vector<int>>& coreTree,
                         const std::unordered_map<int, int>& coreParent,
                        Utils routingUtils,
                        string reportDir);
    
        void simulate();
        void reportWasteStatistics() const;
        std::unordered_map<int, int> getWastedMessagesPerCore() const;
        int findLCA(int sourceCore, int targetCore);
        bool isDescendant(int current, int target);
        std::vector<int> shortestPath(int startCore, int endCore);
        ~HBSRoutingSimulator(){}
};

#endif // HBS_ROUTING_SIMULATOR_H