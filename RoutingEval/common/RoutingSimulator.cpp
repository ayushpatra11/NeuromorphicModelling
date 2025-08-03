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
 *       - 2025-07-31: Added support for checking the route, and storing it. 
 *
 **********************************************************************************/

#include "RoutingSimulator.h"
#include <iostream>
#include <queue>
#include <unordered_set>
#include <random>
#include <set>
#include <algorithm>

// Definition for the root core used in LCA checks
const int rootCore = 30;

RoutingSimulator::RoutingSimulator(const std::vector<std::vector<int>>& connectivityMatrix,
                                   const std::unordered_map<int, int>& neuronToCoreMap,
                                   const std::unordered_map<int, std::vector<int>>& coreTree,
                                   const std::unordered_map<int, int>& coreParent,
                                   Utils routingUtils)
    : connectivityMatrix(connectivityMatrix),
      neuronToCoreMap(neuronToCoreMap),
      coreTree(coreTree),
      coreParent(coreParent),
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
            if(connectivityMatrix[src][i] > weightThreshold && neuronToCoreMap.find(i) != neuronToCoreMap.end()) {
                int core = neuronToCoreMap.at(i);
                int sourceCore = neuronToCoreMap.count(src) ? neuronToCoreMap.at(src) : -1;
                // Skip if target core is the same as source core
                if (core != sourceCore) {
                    targetCores[core].push_back(i);
                }
            }
        }
        routingUtils.logToFile("--- Routing Summary for Neuron " + std::to_string(src) + " ---");
        routingUtils.logToFile("Number of target cores: "+to_string(targetCores.size()));

        if (targetCores.empty()) {
            routingUtils.logToFile("No target cores for source neuron " + std::to_string(src) + ". Skipping routing.");
            continue;
        }

        if (neuronToCoreMap.find(src) == neuronToCoreMap.end()) {
            routingUtils.logToFile("Skipping src neuron " + std::to_string(src) + ": no core mapping.");
            continue;
        }
        int srcCore = neuronToCoreMap.at(src);
        routingUtils.logToFile("Source neuron " + std::to_string(src) + " belongs to core " + std::to_string(srcCore));

        routingUtils.logToFile("For source neuron "+to_string(src)+" at core "+to_string(srcCore)+" targets are: ");
        for(auto& it: targetCores){
            string st = "";
            for(int num: it.second)
                st += to_string(num)+", ";
            routingUtils.logToFile("Core: "+to_string(it.first)+" \n Neurons: "+st);
            routingUtils.logToFile("Target core " + std::to_string(it.first) + " has " + std::to_string(it.second.size()) + " target neurons.");
        }

        // Build sets for all visited cores and target cores (excluding srcCore)
        std::unordered_set<int> actualTargetCores;
        for (const auto& [tgtCore, neuronList] : targetCores) {
            if (tgtCore != srcCore) {
                actualTargetCores.insert(tgtCore);
            }
        }

        // Revert to using all actual target cores derived from connectivity matrix
        routingUtils.logToFile("Using all actual target cores based on connectivity.");
        // `actualTargetCores` already populated above and left unchanged

        // --- LCA computation logic using findLCA ---
        int lcaCore;
        if (!actualTargetCores.empty()) {
            if (actualTargetCores.size() == 1) {
                int targetCore = *actualTargetCores.begin();
                if (targetCore != srcCore)
                    lcaCore = findLCA(srcCore, targetCore);
                else
                    lcaCore = srcCore;
            } else {
                auto minmax = std::minmax_element(actualTargetCores.begin(), actualTargetCores.end());
                lcaCore = findLCA(*minmax.first, *minmax.second);
            }
            routingUtils.logToFile("LCA : "+to_string(lcaCore));
            if (coreParent.find(lcaCore) == coreParent.end() && lcaCore != rootCore) {
                routingUtils.logToFile("Error: coreParent not found for core " + std::to_string(lcaCore));
                routingUtils.logToFile("LCA was not correctly reached. Ending path without B.");
                continue;
            }
        } else {
            routingUtils.logToFile("No valid target cores for LCA computation. Skipping routing.");
            continue;
        }

        // Now route once from srcCore to LCA using shortestPath and new routing logic
        std::vector<int> upPath = shortestPath(srcCore, lcaCore);
        std::string routingPath = "";
        std::unordered_set<int> visited;

        int i = 0;
        while (i < upPath.size() && upPath[i] != rootCore && upPath[i] != lcaCore) {
            routingPath += "U";
            visited.insert(upPath[i]);
            ++i;
        }

        if (upPath[i] == lcaCore) {
            routingPath += "B";  // reached LCA directly
        } else {
            routingPath += "D";  // start descent to LCA
            for (++i; i < upPath.size(); ++i) {
                int parent = upPath[i - 1];
                int child = upPath[i];
                routingPath += (child < parent ? "L" : "R");
            }
            routingPath += "B";  // reached LCA and chose to broadcast
        }
        routingUtils.logToFile("Route: " + routingPath);
        for (int tgt : actualTargetCores)
            routingUtils.logToFile(" -> Core " + std::to_string(tgt));

        // Now, for each target core, descend down from LCA to the target core using tree structure
        for (int targetCore : actualTargetCores) {
            int current = lcaCore;
            while (current != targetCore) {
                const auto& children = coreTree.at(current);
                if (std::find(children.begin(), children.end(), targetCore) != children.end()) {
                    current = targetCore;
                } else {
                    bool found = false;
                    for (int child : children) {
                        if (isDescendant(child, targetCore)) {
                            current = child;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        routingUtils.logToFile("Routing error: Could not route down to target core: " + std::to_string(targetCore));
                        break;
                    }
                }
                routingUtils.logToFile("Going down to core: " + std::to_string(current));
            }
        }

        // Add dummy and real visited cores
        std::unordered_set<int> visitedCores = visited;
        visitedCores.insert(lcaCore);
        for (int core : visitedCores) {
            if (coreTree.find(core) == coreTree.end())
                visitedCores.insert(core);
        }

        routingUtils.logToFile("Visited real cores: ");
        std::string visitedStr = "";
        for (int core : visitedCores) {
            if (coreTree.find(core) == coreTree.end())  // only real cores
                visitedStr += std::to_string(core) + " ";
        }
        routingUtils.logToFile(visitedStr);

        for (int core : visitedCores) {
            if (core != srcCore && actualTargetCores.find(core) == actualTargetCores.end()) {
                wastedMessages[core]++;
                routingUtils.logToFile("Wasted message at core " + std::to_string(core) + " (visited but not a target core)");
            }
        }

        routingUtils.logToFile("Checking neuron-level waste (i.e., target cores that should not have been visited):");
        int totalNeuronWaste = 0;
        int neuronsPerCore = 64;
        for (int visitedCore : visitedCores) {
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
                neuronToCoreMap.find(src) != neuronToCoreMap.end() &&
                neuronToCoreMap.find(tgt) != neuronToCoreMap.end() &&
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
    if (srcCore == tgtCore) return;

    std::string pathTrace = "Routing from Core " + std::to_string(srcCore) + " to Core " + std::to_string(tgtCore) + ": ";
    std::vector<int> pathSrc, pathTgt;

    int current = srcCore;
    while (current >= 0) {
        pathSrc.push_back(current);
        if (coreParent.find(current) == coreParent.end() || coreParent.at(current) == -1)
            break;
        current = coreParent.at(current);
    }

    current = tgtCore;
    while (current >= 0) {
        pathTgt.push_back(current);
        if (coreParent.find(current) == coreParent.end() || coreParent.at(current) == -1)
            break;
        current = coreParent.at(current);
    }

    std::reverse(pathSrc.begin(), pathSrc.end());
    std::reverse(pathTgt.begin(), pathTgt.end());

    int lca = -1;
    size_t i = 0;
    while (i < pathSrc.size() && i < pathTgt.size() && pathSrc[i] == pathTgt[i]) {
        lca = pathSrc[i];
        ++i;
    }

    std::unordered_set<int> dummyVisitedInPath;
    for (int j = pathSrc.size() - 1; j >= static_cast<int>(i); --j) {
        if (coreTree.find(pathSrc[j]) != coreTree.end()) {
            pathTrace += "U(" + std::to_string(pathSrc[j]) + ") -> ";
            dummyVisitedInPath.insert(pathSrc[j]);
        } else {
            visitedCores.insert(pathSrc[j]);
        }
    }

    if (coreTree.find(lca) != coreTree.end()) {
        pathTrace += "B(" + std::to_string(lca) + ")";
    }

    routingUtils.logToFile(pathTrace);
}


// Computes the Lowest Common Ancestor (LCA) between two cores in the core tree
int RoutingSimulator::findLCA(int coreA, int coreB) {
    std::vector<int> pathA, pathB;

    // Build path from coreA to root
    while (coreA != -1) {
        pathA.push_back(coreA);
        auto it = coreParent.find(coreA);
        coreA = (it != coreParent.end()) ? it->second : -1;
    }

    // Build path from coreB to root
    while (coreB != -1) {
        pathB.push_back(coreB);
        auto it = coreParent.find(coreB);
        coreB = (it != coreParent.end()) ? it->second : -1;
    }

    // Reverse both paths to start from root
    std::reverse(pathA.begin(), pathA.end());
    std::reverse(pathB.begin(), pathB.end());

    int lca = -1;
    size_t minLength = std::min(pathA.size(), pathB.size());
    for (size_t i = 0; i < minLength; ++i) {
        if (pathA[i] == pathB[i])
            lca = pathA[i];
        else
            break;
    }

    return lca;
}

// Helper function to check if target is a descendant of current in the tree
bool RoutingSimulator::isDescendant(int current, int target) {
    if (current == target) return true;
    if (coreTree.find(current) == coreTree.end()) return false;
    for (int child : coreTree.at(current)) {
        if (isDescendant(child, target)) return true;
    }
    return false;
}

// Computes the shortest path in the core tree between two nodes using BFS
std::vector<int> RoutingSimulator::shortestPath(int startCore, int endCore) {
    std::unordered_map<int, int> parent;
    std::queue<int> q;
    std::unordered_set<int> visited;

    q.push(startCore);
    visited.insert(startCore);
    parent[startCore] = -1;

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        if (current == endCore) break;

        // Add parent if it exists
        if (coreParent.find(current) != coreParent.end()) {
            int p = coreParent.at(current);
            if (visited.find(p) == visited.end()) {
                visited.insert(p);
                q.push(p);
                parent[p] = current;
            }
        }

        // Add children if they exist
        if (coreTree.find(current) != coreTree.end()) {
            for (int child : coreTree.at(current)) {
                if (visited.find(child) == visited.end()) {
                    visited.insert(child);
                    q.push(child);
                    parent[child] = current;
                }
            }
        }
    }

    std::vector<int> path;
    int current = endCore;
    while (current != -1) {
        path.push_back(current);
        current = parent[current];
    }
    std::reverse(path.begin(), path.end());
    return path;
}