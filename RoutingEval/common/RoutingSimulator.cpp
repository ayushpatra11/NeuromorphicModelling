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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <queue>
#include <random>
#include <set>
#include <unordered_set>

RoutingSimulator::RoutingSimulator(const std::vector<std::vector<int>>& connectivityMatrix,
                                   const std::unordered_map<int, int>& neuronToCoreMap,
                                   const std::unordered_map<int, std::vector<int>>& coreTree,
                                   const std::unordered_map<int, int>& coreParent, Utils routingUtils, string reportDir)
    : connectivityMatrix(connectivityMatrix),
      neuronToCoreMap(neuronToCoreMap),
      coreTree(coreTree),
      coreParent(coreParent),
      routingUtils(routingUtils),
      reportDir(reportDir) {
    srand(time(0));
    weightThreshold = 0;
}

void RoutingSimulator::simulate() {
    wastePerNeuron.clear();
    wastedMessages.clear();

    int rootCore = findRootCore();
    int numNeurons = connectivityMatrix.size();

    for (int src = 0; src < numNeurons; ++src) {
        // int src = rand() % numNeurons; // Randomly select a source neuron for debugging only.

        // error scenario where core is not found.
        if (neuronToCoreMap.find(src) == neuronToCoreMap.end()) {
            routingUtils.logToFile("Skipping src neuron " + std::to_string(src) + ": no core mapping.");
            continue;
        }

        // Printing all target neurons and the target cores.
        auto targetCores = buildTargetCores(src);

        // For each Neuron, producing a Routing Summary.
        routingUtils.logToFile("--- Routing Summary for Neuron " + std::to_string(src) + " ---");
        routingUtils.logToFile("Number of target cores: " + std::to_string(targetCores.size()));

        // In case there are NO target CORES (This neuron did not fire at all during evaluation of 155 time steps)
        if (targetCores.empty()) {
            routingUtils.logToFile("No target cores for source neuron " + std::to_string(src) + ". Skipping routing.");
            continue;
        }

        // Find the source core of the neuron.
        int srcCore = neuronToCoreMap.at(src);
        routingUtils.logToFile("Source neuron " + std::to_string(src) + " belongs to core " + std::to_string(srcCore));

        // Logging the target neurons.
        routingUtils.logToFile("For source neuron " + std::to_string(src) + " at core " + std::to_string(srcCore) +
                               " targets are: ");
        for (auto& it : targetCores) {
            string st;
            for (int num : it.second) st += std::to_string(num) + ", ";
            routingUtils.logToFile("Core: " + std::to_string(it.first) + " \n Neurons: " + st);
            routingUtils.logToFile("Target core " + std::to_string(it.first) + " has " +
                                   std::to_string(it.second.size()) + " target neurons.");
        }

        // Build set of actual target cores (excluding srcCore).
        // Revert to using all actual target cores derived from connectivity matrix.
        std::unordered_set<int> actualTargetCores;
        for (const auto& [tgtCore, neuronList] : targetCores) {
            if (tgtCore != srcCore) actualTargetCores.insert(tgtCore);
        }
        routingUtils.logToFile("Using all actual target cores based on connectivity.");

        // --- LCA computation logic using findLCA ---
        int lcaCore = computeLCA(srcCore, actualTargetCores, rootCore);
        if (lcaCore == -1) continue;

        // Now route once from srcCore to LCA using shortestPath and new routing logic.
        std::string routingPath = traceRoutingPath(srcCore, lcaCore, rootCore);
        routingUtils.logToFile("Route: " + routingPath);
        for (int tgt : actualTargetCores) routingUtils.logToFile(" -> Core " + std::to_string(tgt));

        // --- Compute waste for other subtrees under the LCA ---
        int waste = computeWasteUnderLCA(src, lcaCore, actualTargetCores);
        routingUtils.logToFile("Total core-level routing waste: " + std::to_string(waste));
        routingUtils.logToFile("Finished simulation for source neuron " + std::to_string(src));
    }

    writeReport();
}

int RoutingSimulator::findRootCore() const {
    for (const auto& kv : coreParent)
        if (kv.second == -1) return kv.first;
    return -1;
}

std::map<int, std::vector<int>> RoutingSimulator::buildTargetCores(int src) const {
    std::map<int, std::vector<int>> targetCores;
    int numNeurons = connectivityMatrix.size();
    int sourceCore = neuronToCoreMap.count(src) ? neuronToCoreMap.at(src) : -1;
    for (int i = 0; i < numNeurons; i++) {
        if (connectivityMatrix[src][i] > weightThreshold && neuronToCoreMap.find(i) != neuronToCoreMap.end()) {
            int core = neuronToCoreMap.at(i);
            // Skip if target core is the same as source core
            if (core != sourceCore) targetCores[core].push_back(i);
        }
    }
    return targetCores;
}

// Returns -1 if routing should be skipped for this neuron.
int RoutingSimulator::computeLCA(int srcCore, const std::unordered_set<int>& actualTargetCores, int rootCore) {
    if (actualTargetCores.empty()) {
        routingUtils.logToFile("No valid target cores for LCA computation. Skipping routing.");
        return -1;
    }

    int lcaCore;
    if (actualTargetCores.size() == 1) {
        int targetCore = *actualTargetCores.begin();
        lcaCore = (targetCore != srcCore) ? findLCA(srcCore, targetCore) : srcCore;
    } else {
        auto minmax = std::minmax_element(actualTargetCores.begin(), actualTargetCores.end());
        lcaCore = findLCA(*minmax.first, *minmax.second);
    }

    routingUtils.logToFile("LCA : " + std::to_string(lcaCore));
    if (coreParent.find(lcaCore) == coreParent.end() && lcaCore != rootCore) {
        routingUtils.logToFile("Error: coreParent not found for core " + std::to_string(lcaCore));
        routingUtils.logToFile("LCA was not correctly reached. Ending path without B.");
        return -1;
    }

    return lcaCore;
}

std::string RoutingSimulator::traceRoutingPath(int srcCore, int lcaCore, int rootCore) {
    std::vector<int> upPath = shortestPath(srcCore, lcaCore);
    std::string routingPath;

    int i = 0;
    while (i < static_cast<int>(upPath.size()) && upPath[i] != rootCore && upPath[i] != lcaCore) {
        routingPath += "U";
        ++i;
    }

    if (upPath[i] == lcaCore) {
        routingPath += "B";  // reached LCA directly
    } else {
        routingPath += "D";  // start descent to LCA
        for (++i; i < static_cast<int>(upPath.size()); ++i) {
            int parent = upPath[i - 1];
            int child = upPath[i];
            routingPath += (child < parent ? "L" : "R");
        }
        routingPath += "B";  // reached LCA and chose to broadcast
    }

    return routingPath;
}

int RoutingSimulator::computeWasteUnderLCA(int src, int lcaCore, const std::unordered_set<int>& actualTargetCores) {
    // Identify all leaf descendants of LCA — only count *leaf* descendants (real cores, not switches)
    std::unordered_set<int> leafDescendants;
    std::function<void(int)> collectLeaves = [&](int node) {
        if (coreTree.find(node) == coreTree.end()) {
            leafDescendants.insert(node);  // it's a leaf core
            return;
        }
        for (int child : coreTree.at(node)) collectLeaves(child);
    };
    collectLeaves(lcaCore);

    routingUtils.logToFile("Visited real cores (leaf descendants under LCA " + std::to_string(lcaCore) + "):");
    std::string visitedStr;
    for (int core : leafDescendants) visitedStr += std::to_string(core) + " ";
    routingUtils.logToFile(visitedStr);

    int totalCoreWaste = 0;
    for (int leaf : leafDescendants) {
        if (actualTargetCores.find(leaf) == actualTargetCores.end()) {
            routingUtils.logToFile("Core-level waste: core " + std::to_string(leaf) + " under LCA " +
                                   std::to_string(lcaCore) + " was not a target.");
            wastedMessages[leaf]++;
            totalCoreWaste++;
        }
    }

    wastePerNeuron[src] = totalCoreWaste;
    return totalCoreWaste;
}

void RoutingSimulator::writeReport() {
    // === Final waste summary (core-level only) ===
    long long totalWasteAll = 0;
    for (const auto& kv : wastePerNeuron) totalWasteAll += kv.second;

    routingUtils.logToFile("\n==== Neurogrid Routing Waste Report ====");
    routingUtils.logToFile("Total illegal deliveries (waste): " + std::to_string(totalWasteAll));

    routingUtils.logToFile("Per-neuron waste (non-zero only):");
    for (const auto& kv : wastePerNeuron)
        if (kv.second > 0)
            routingUtils.logToFile("  Neuron " + std::to_string(kv.first) + ": " + std::to_string(kv.second));

    routingUtils.logToFile("Per-core waste (non-zero only):");
    for (const auto& kv : wastedMessages)
        if (kv.second > 0)
            routingUtils.logToFile("  Core " + std::to_string(kv.first) + ": " + std::to_string(kv.second));

    routingUtils.logToFile("==================================");

    // Save waste metrics to JSON report file
    nlohmann::json j;
    j["total_illegal_deliveries"] = totalWasteAll;
    nlohmann::json per_neuron, per_core;
    for (const auto& kv : wastePerNeuron)
        if (kv.second > 0) per_neuron[std::to_string(kv.first)] = kv.second;
    for (const auto& kv : wastedMessages)
        if (kv.second > 0) per_core[std::to_string(kv.first)] = kv.second;
    j["per_neuron_waste"] = per_neuron;
    j["per_core_waste"] = per_core;

    std::ofstream jout(reportDir);
    if (jout.is_open()) {
        jout << j.dump(4) << std::endl;
        jout.close();
    }
}

long long RoutingSimulator::getTotalWaste() const {
    long long total = 0;
    for (const auto& kv : wastePerNeuron) total += kv.second;
    return total;
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
