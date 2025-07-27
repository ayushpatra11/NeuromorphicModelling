/**********************************************************************************
 *
 *   File Name: RoutingSimulator.cpp
 *   Author: Ayush Patra
 *   Description: Implements the simulation of spike message routing across a
 *                Neurogrid-inspired binary core tree and calculates routing waste
 *                based on a static neuron connectivity matrix.
 *   Version History:
 *       - 2025-07-26: Initial version
 *       - 2025-07-27: Added support for neuron and core level waste calculation. 
 *
 **********************************************************************************/

#include "RoutingSimulator.h"
#include <iostream>
#include <queue>
#include <unordered_set>

RoutingSimulator::RoutingSimulator(const std::vector<std::vector<int>>& connectivityMatrix,
                                   const std::unordered_map<int, int>& neuronToCoreMap,
                                   const std::unordered_map<int, std::vector<int>>& coreTree,
                                   Utils routingUtils)
    : connectivityMatrix(connectivityMatrix),
      neuronToCoreMap(neuronToCoreMap),
      coreTree(coreTree),
      routingUtils(routingUtils) {
        srand(time(0));
        weightThreshold = 0;
      }

void RoutingSimulator::simulate() {

    //let us first start with checking the mappings. 
    
    int numNeurons = connectivityMatrix.size();
    for (int src = 0; src < numNeurons; ++src) {
    // int src = rand() % numNeurons; // Randomly select a source neuron for debuggin only. 

        //Printing all target neurons and the target cores. 
        std::map<int, vector<int>> targetCores;
        for(int i = 0; i<numNeurons; i++){
            if(connectivityMatrix[src][i] > weightThreshold)
            targetCores[neuronToCoreMap.at(i)].push_back(i);
        }
        routingUtils.logToFile("Number of target cores: "+to_string(targetCores.size()));

        int srcCore = neuronToCoreMap.at(src);
        routingUtils.logToFile("Source neuron " + std::to_string(src) + " belongs to core " + std::to_string(srcCore));

        routingUtils.logToFile("For source neuron "+to_string(src)+" at core"+to_string(neuronToCoreMap.at(src))+" targets are: ");
        for(auto& it: targetCores){
            string st = "";
            for(int num: it.second)
                st += to_string(num)+", ";
            routingUtils.logToFile("Core: "+to_string(it.first)+" \n Neurons: "+st);
            routingUtils.logToFile("Target core " + std::to_string(it.first) + " has " + std::to_string(it.second.size()) + " target neurons.");
        }

        // Build sets for all visited cores and target cores (excluding srcCore)
        std::unordered_set<int> allVisitedCores;
        std::unordered_set<int> actualTargetCores;
        for (const auto& [tgtCore, neuronList] : targetCores) {
            if (tgtCore != srcCore) {
                actualTargetCores.insert(tgtCore);
                std::unordered_set<int> visited;
                routeMessage(srcCore, tgtCore, visited);
                allVisitedCores.insert(visited.begin(), visited.end());
            }
        }

        routingUtils.logToFile("Visited cores during routing:");
        for (int core : allVisitedCores) {
            routingUtils.logToFile("Visited core: " + std::to_string(core));
        }

        for (int core : allVisitedCores) {
            if (core != srcCore && actualTargetCores.find(core) == actualTargetCores.end()) {
                wastedMessages[core]++;
                routingUtils.logToFile("Wasted message at core " + std::to_string(core) + " (visited but not a target core)");
            }
        }

        routingUtils.logToFile("Checking neuron-level waste (i.e., target cores that should not have been visited):");
        int totalNeuronWaste = 0;
        int neuronsPerCore = 64;

        for (int visitedCore : allVisitedCores) {
            // Skip source core and actual targets
            if (visitedCore != srcCore && actualTargetCores.find(visitedCore) == actualTargetCores.end()) {
                totalNeuronWaste += neuronsPerCore;
                routingUtils.logToFile("Neuron-level waste: all " + std::to_string(neuronsPerCore) +
                                    " neurons in core " + std::to_string(visitedCore) +
                                    " received unnecessary spike.");
            }
        }

        // Compute and print total number of valid spikes (excluding same-core) and waste percentage
        int totalValidMessages = 0;
        for (int tgt = 0; tgt < numNeurons; ++tgt) {
            if (connectivityMatrix[src][tgt] > weightThreshold &&
                neuronToCoreMap.at(src) != neuronToCoreMap.at(tgt)) {
                totalValidMessages++;
                routingUtils.logToFile("Valid spike: src neuron " + std::to_string(src) + " (core " + std::to_string(neuronToCoreMap.at(src)) +
                                    ") to tgt neuron " + std::to_string(tgt) + " (core " + std::to_string(neuronToCoreMap.at(tgt)) + ")");
            }
        }

        int totalCoreWaste = 0;
        for (const auto& [core, waste] : wastedMessages) {
            totalCoreWaste += waste;
        }

        routingUtils.logToFile("Total valid messages: " + std::to_string(totalValidMessages));
        routingUtils.logToFile("Total core-level routing waste: " + std::to_string(totalCoreWaste));
        routingUtils.logToFile("Total neuron-level routing waste: " + std::to_string(totalNeuronWaste));
        if (totalValidMessages > 0) {
            float wastePercentage = static_cast<float>(totalNeuronWaste) / (totalNeuronWaste + totalValidMessages) * 100.0f;
            routingUtils.logToFile("Neuron-level waste percentage: " + std::to_string(wastePercentage) + "%");
        }
        routingUtils.logToFile("Finished simulation for source neuron " + std::to_string(src));
    }
}

void RoutingSimulator::routeMessage(int srcCore, int tgtCore, std::unordered_set<int>& visitedCores) {
    // Find the path from srcCore to root
    std::vector<int> pathSrc, pathTgt;
    int current = srcCore;
    while (current >= 0) {
        pathSrc.push_back(current);
        if (current == 0) break;
        current = (current - 1) / 2;
    }

    current = tgtCore;
    while (current >= 0) {
        pathTgt.push_back(current);
        if (current == 0) break;
        current = (current - 1) / 2;
    }

    // Reverse to have path from root to leaf
    std::reverse(pathSrc.begin(), pathSrc.end());
    std::reverse(pathTgt.begin(), pathTgt.end());

    // Find LCA
    int lca = 0;
    size_t i = 0;
    while (i < pathSrc.size() && i < pathTgt.size() && pathSrc[i] == pathTgt[i]) {
        lca = pathSrc[i];
        ++i;
    }

    // Add path from srcCore to LCA (excluding srcCore)
    for (int j = pathSrc.size() - 1; j >= static_cast<int>(i); --j) {
        visitedCores.insert(pathSrc[j]);
    }

    // Add path from LCA to tgtCore (excluding tgtCore)
    for (int j = i; j < pathTgt.size() - 1; ++j) {
        visitedCores.insert(pathTgt[j]);
    }
}

std::unordered_map<int, int> RoutingSimulator::getWastedMessagesPerCore() const {
    return wastedMessages;
}