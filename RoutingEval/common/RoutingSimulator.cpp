/**********************************************************************************
 *
 *   File Name: RoutingSimulator.cpp
 *   Author: Ayush Patra
 *   Description: Implements the simulation of spike message routing across a
 *                Neurogrid-inspired binary core tree and calculates routing waste
 *                based on a static neuron connectivity matrix.
 *   Version History:
 *       - 2025-07-26: Initial version
 *
 **********************************************************************************/

#include "RoutingSimulator.h"
#include <iostream>
#include <queue>
#include <unordered_set>

RoutingSimulator::RoutingSimulator(const std::vector<std::vector<int>>& connectivityMatrix,
                                   const std::unordered_map<int, int>& neuronToCoreMap,
                                   const std::unordered_map<int, std::vector<int>>& coreTree)
    : connectivityMatrix(connectivityMatrix),
      neuronToCoreMap(neuronToCoreMap),
      coreTree(coreTree) {}

void RoutingSimulator::simulate() {
    int numNeurons = connectivityMatrix.size();
    // for (int src = 0; src < numNeurons; ++src) {
    int src = rand() % numNeurons; // Randomly select a source neuron
    int srcCore = neuronToCoreMap.at(src);
    std::unordered_set<int> uniqueTargetCores;
    for (int tgt = 0; tgt < numNeurons; ++tgt) {
        if (connectivityMatrix[src][tgt] > 0.0f) {
            int tgtCore = neuronToCoreMap.at(tgt);
            if (tgtCore != srcCore)
                uniqueTargetCores.insert(tgtCore);
        }
    }

    for (int tgtCore : uniqueTargetCores) {
        std::unordered_set<int> visitedCores;
        routeMessage(srcCore, tgtCore, visitedCores);
        for (int core : visitedCores) {
            if (core != tgtCore)
                wastedMessages[core]++;
        }
    }

    // Compute and print total number of valid spikes (excluding same-core) and waste percentage
    int totalValidMessages = 0;
    for (int tgt = 0; tgt < numNeurons; ++tgt) {
        if (connectivityMatrix[src][tgt] > 0.0f &&
            neuronToCoreMap.at(src) != neuronToCoreMap.at(tgt)) {
            totalValidMessages++;
        }
    }

    int totalWaste = 0;
    for (const auto& [core, waste] : wastedMessages) {
        totalWaste += waste;
    }

    std::cout << "Total valid messages: " << totalValidMessages << std::endl;
    std::cout << "Total routing waste: " << totalWaste << std::endl;
    if (totalValidMessages > 0) {
        float wastePercentage = static_cast<float>(totalWaste) / totalValidMessages * 100.0f;
        std::cout << "Waste percentage: " << wastePercentage << "%" << std::endl;
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