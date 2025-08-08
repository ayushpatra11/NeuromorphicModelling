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
#include <fstream>

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
    std::map<int, int> neuronWaste;  // per-source: number of non-target cores that received
    std::unordered_map<int, int> coreWaste; // per-core: times it received when not a target
    
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


        //For each Neuron, producing a Routing Summary. 


        routingUtils.logToFile("--- Routing Summary for Neuron " + std::to_string(src) + " ---");
        routingUtils.logToFile("Number of target cores: "+to_string(targetCores.size()));

        // In case there are NO target CORES (This neuron did not fire at all during evaluation of 155 time steps)
        if (targetCores.empty()) {
            routingUtils.logToFile("No target cores for source neuron " + std::to_string(src) + ". Skipping routing.");
            continue;
        }


        //else continue. 

        // error scenario where core is not found. 
        if (neuronToCoreMap.find(src) == neuronToCoreMap.end()) {
            routingUtils.logToFile("Skipping src neuron " + std::to_string(src) + ": no core mapping.");
            continue;
        }

        //find the source core of the nueorn. 

        int srcCore = neuronToCoreMap.at(src);
        routingUtils.logToFile("Source neuron " + std::to_string(src) + " belongs to core " + std::to_string(srcCore));

        routingUtils.logToFile("For source neuron "+to_string(src)+" at core "+to_string(srcCore)+" targets are: ");

        //logging the target neurons. 
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

        // --- Compute waste for other subtrees under the LCA, but only count *leaf* descendants (real cores) ---
        // Identify all leaf descendants of LCA
        std::unordered_set<int> leafDescendants;
        std::function<void(int)> collectLeafDescendants = [&](int node) {
            if (coreTree.find(node) == coreTree.end()) {
                leafDescendants.insert(node);  // it's a leaf
                return;
            }
            for (int child : coreTree.at(node)) {
                collectLeafDescendants(child);
            }
        };
        collectLeafDescendants(lcaCore);

        routingUtils.logToFile("Visited real cores (leaf descendants under LCA " + std::to_string(lcaCore) + "):");
        std::string visitedStr = "";
        for (int core : leafDescendants) {
            visitedStr += std::to_string(core) + " ";
        }
        routingUtils.logToFile(visitedStr);

        int totalCoreWaste = 0;
        for (int leaf : leafDescendants) {
            if (actualTargetCores.find(leaf) == actualTargetCores.end()) {
                routingUtils.logToFile("Core-level waste: core " + std::to_string(leaf) + " under LCA " + std::to_string(lcaCore) + " was not a target.");
                coreWaste[leaf]++;
                totalCoreWaste++;
            }
        }

        routingUtils.logToFile("Total core-level routing waste: " + std::to_string(totalCoreWaste));
        neuronWaste[src] = totalCoreWaste;

        routingUtils.logToFile("Finished simulation for source neuron " + std::to_string(src));
    }

    // === Final waste summary (core-level only) ===
    long long totalWasteAll = 0;
    for (const auto &kv : neuronWaste) totalWasteAll += kv.second;

    routingUtils.logToFile("\n==== Neurogrid Routing Waste Report ====");
    routingUtils.logToFile("Total illegal deliveries (waste): " + std::to_string(totalWasteAll));

    routingUtils.logToFile("Per-neuron waste (non-zero only):");
    for (const auto &kv : neuronWaste) {
        if (kv.second > 0) routingUtils.logToFile("  Neuron " + std::to_string(kv.first) + ": " + std::to_string(kv.second));
    }

    routingUtils.logToFile("Per-core waste (non-zero only):");
    for (const auto &kv : coreWaste) {
        if (kv.second > 0) routingUtils.logToFile("  Core " + std::to_string(kv.first) + ": " + std::to_string(kv.second));
    }
    routingUtils.logToFile("==================================");

    // Write the same summary to a file
    {
        std::ofstream out("../data/reports/neurogrid_waste_metrics.txt");
        if (out.is_open()) {
            out << "==== Neurogrid Routing Waste Report ====" << '\n';
            out << "Total illegal deliveries (waste): " << totalWasteAll << '\n';
            out << "Per-neuron waste (non-zero only):" << '\n';
            for (const auto &kv : neuronWaste) {
                if (kv.second > 0) out << "  Neuron " << kv.first << ": " << kv.second << '\n';
            }
            out << "Per-core waste (non-zero only):" << '\n';
            for (const auto &kv : coreWaste) {
                if (kv.second > 0) out << "  Core " << kv.first << ": " << kv.second << '\n';
            }
            out << "==================================" << '\n';
            out.close();
        }
    }
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