/*####################################################################################
#
#   File Name: HBSRoutingSimulator.cpp
#   Author:  Ayush Patra
#   Description: Simulates Neurogrid-style routing from one neuron to another using
#                a binary tree of switches with 4 cores in the leaves. Computes and 
#                logs routing waste based on deviation from expected connectivity 
#                (connectivity matrix).
#   Version History:
#       - 2025-08-06: Initial version
#
####################################################################################*/

#include "HBSRoutingSimulator.h"
#include <queue>
#include <stack>
#include <iostream>
#include <cassert>
#include <sstream>
static constexpr int MAX_GROUP_SIZE = 4;

using namespace std;

namespace {
// Helper: return true if nodeId exists as a key in coreTree (i.e., it is a switch)
bool isSwitch(int nodeId, const unordered_map<int, vector<int>>& coreTree) {
    return coreTree.find(nodeId) != coreTree.end();
}

// Assumption: cores exist only at leaves; internal nodes are switches.
vector<int> collectLeafCores(int node,
                             const unordered_map<int, vector<int>>& coreTree) {
    vector<int> leaves;
    if (!isSwitch(node, coreTree)) {
        leaves.push_back(node);
        return leaves;
    }
    // BFS
    queue<int> q;
    q.push(node);
    while (!q.empty()) {
        int cur = q.front();
        q.pop();
        auto it = coreTree.find(cur);
        if (it == coreTree.end()) {
            // cur is a core
            leaves.push_back(cur);
        } else {
            for (int child : it->second) {
                if (isSwitch(child, coreTree)) q.push(child);
                else leaves.push_back(child);
            }
        }
    }
    return leaves;
}

// Helper: find the index of `child` under parent switch `parent` (0..k-1).
int childIndexUnder(int parent,
                    int child,
                    const unordered_map<int, vector<int>>& coreTree) {
    auto it = coreTree.find(parent);
    if (it == coreTree.end()) return -1;
    const auto& children = it->second;
    for (int i = 0; i < (int)children.size(); ++i) {
        if (children[i] == child) return i;
    }
    return -1;
}

// Helper: render a 4-bit mask string for child indices 0..3 (bit0 is child index 0, leftmost)
static std::string maskToBits(const std::unordered_set<int>& indices) {
    std::string bits = "0000"; // [0][1][2][3], leftmost is child index 0
    for (int i = 0; i < MAX_GROUP_SIZE; ++i) {
        if (indices.count(i)) bits[i] = '1';
    }
    return bits; // e.g., "0111" means children {0,1,2}
}


} // anonymous namespace

HBSRoutingSimulator::HBSRoutingSimulator(
    const vector<vector<int>>& connectivityMatrix,
    const unordered_map<int, int>& neuronToCoreMap,
    const unordered_map<int, vector<int>>& coreTree,
    const unordered_map<int, int>& coreParent,
    Utils routingUtils)
    : routingUtils(routingUtils),
      weightThreshold(1.0f), // default; adjust if your Utils exposes a threshold getter
      connectivityMatrix(connectivityMatrix),
      neuronToCoreMap(neuronToCoreMap),
      coreTree(coreTree),
      coreParent(coreParent) {

    // If Utils exposes a threshold, pick it up (comment out if Utils has no such API)
    // weightThreshold = routingUtils.getWeightThreshold();

    // Precompute exact target core set per source neuron using the threshold rule.
    for (int srcNeuron = 0; srcNeuron < (int)connectivityMatrix.size(); ++srcNeuron) {
        const auto& row = connectivityMatrix[srcNeuron];
        auto n2cIt = neuronToCoreMap.find(srcNeuron);
        if (n2cIt == neuronToCoreMap.end()) continue; // unmapped neuron

        unordered_set<int> tgtCores;
        for (int dstNeuron = 0; dstNeuron < (int)row.size(); ++dstNeuron) {
            if (row[dstNeuron] >= weightThreshold) {
                auto mIt = neuronToCoreMap.find(dstNeuron);
                if (mIt != neuronToCoreMap.end()) {
                    tgtCores.insert(mIt->second);
                }
            }
        }
        if (!tgtCores.empty()) {
            actualTargetsPerNeuron[srcNeuron] = move(tgtCores);
        }
    }
}

void HBSRoutingSimulator::simulate() {
    for (const auto& kv : actualTargetsPerNeuron) {
        int srcNeuron = kv.first;
        simulateNeuronToNeuron(srcNeuron);
    }
}

// Core HBS region-routing without LCA: directly target the *parent switches* of each target core group.
// For all parent switches that contain at least one target, we compute a *global OR mask* of child indices
// (e.g., {0,1,2}). Each such parent then broadcasts to that mask. Waste = sum over selected children of
// (number of leaf cores in that child subtree) - (number of real target cores in that subtree).
void HBSRoutingSimulator::simulateNeuronToNeuron(int sourceNeuron) {
    auto itTargets = actualTargetsPerNeuron.find(sourceNeuron);
    if (itTargets == actualTargetsPerNeuron.end()) return;
    const unordered_set<int>& targetCores = itTargets->second;

    // Reconstruct target neurons (dst) and their cores for detailed logging
    std::vector<std::pair<int,int>> targetNeuronCoreList; // (dstNeuron, coreId)
    const auto& row = connectivityMatrix[sourceNeuron];
    for (int dstNeuron = 0; dstNeuron < (int)row.size(); ++dstNeuron) {
        if (row[dstNeuron] >= weightThreshold) {
            auto mIt = neuronToCoreMap.find(dstNeuron);
            if (mIt != neuronToCoreMap.end()) {
                targetNeuronCoreList.emplace_back(dstNeuron, mIt->second);
            }
        }
    }

    // Build core -> list of target neurons for pretty logging
    std::unordered_map<int, std::vector<int>> coreToDstNeurons;
    for (const auto& pr : targetNeuronCoreList) {
        coreToDstNeurons[pr.second].push_back(pr.first);
    }

    // Source core
    int sourceCore = -1; {
        auto itSrc = neuronToCoreMap.find(sourceNeuron);
        if (itSrc != neuronToCoreMap.end()) sourceCore = itSrc->second;
    }

    // --- Summary header logs ---
    routingUtils.logToFile("--- Routing Summary for Neuron " + std::to_string(sourceNeuron) + " ---");
    routingUtils.logToFile("Number of target cores: " + std::to_string(coreToDstNeurons.size()));
    routingUtils.logToFile("Source neuron " + std::to_string(sourceNeuron) + " belongs to core " + std::to_string(sourceCore));
    routingUtils.logToFile("For source neuron " + std::to_string(sourceNeuron) + " at core " + std::to_string(sourceCore) + " targets are: ");

    // Per-core target lists
    for (auto &kvp : coreToDstNeurons) {
        int tCore = kvp.first;
        auto &dsts = kvp.second;
        std::ostringstream oss1; oss1 << "Core: " << tCore;
        routingUtils.logToFile(oss1.str());
        std::ostringstream oss2; oss2 << " Neurons: ";
        for (size_t i = 0; i < dsts.size(); ++i) { oss2 << dsts[i] << ", "; }
        routingUtils.logToFile(oss2.str());
        routingUtils.logToFile("Target core " + std::to_string(tCore) + " has " + std::to_string(dsts.size()) + " target neurons.");
    }

    routingUtils.logToFile("Using all actual target cores based on connectivity.");

    if (targetNeuronCoreList.empty()) {
        routingUtils.logToFile("No target cores for source neuron " + std::to_string(sourceNeuron) + ". Skipping routing.");
        return;
    }

    // Log: source neuron, list of target neurons and their cores, and the set of target cores
    {
        std::ostringstream oss;
        oss << "[HBS] Source neuron: " << sourceNeuron << "\n";
        oss << "[HBS] Target neurons (dst -> core): ";
        for (size_t i = 0; i < targetNeuronCoreList.size(); ++i) {
            if (i) oss << ", ";
            oss << targetNeuronCoreList[i].first << "->" << targetNeuronCoreList[i].second;
        }
        oss << "\n[HBS] Unique target cores: ";
        size_t i = 0; for (int c : targetCores) { if (i++) oss << ", "; oss << c; }
        routingUtils.logToFile(oss.str());
    }

    // Group targets by their *immediate parent switch* and by child-index under that switch.
    // parentSwitch -> childIndex -> set<coreId>
    unordered_map<int, unordered_map<int, unordered_set<int>>> parentToChildIdxTargets;

    // Also collect the set of parent switches that will receive a packet.
    unordered_set<int> parentSwitches;

    for (int coreId : targetCores) {
        auto pIt = coreParent.find(coreId);
        if (pIt == coreParent.end()) continue; // orphan core? ignore
        int parent = pIt->second; // parent switch of the core
        parentSwitches.insert(parent);

        int idx = childIndexUnder(parent, coreId, coreTree);
        if (idx < 0) {
            // The child might be hanging under an intermediate switch; in that case
            // find the immediate child of `parent` on the path to `coreId` by scanning
            // the parent's children and checking descendant relation.
            auto itChildren = coreTree.find(parent);
            if (itChildren == coreTree.end()) {
                routingUtils.logToFile("[HBS] Warning: parent switch " + std::to_string(parent) + " not found in coreTree. Skipping core " + std::to_string(coreId));
                continue; // skip this target core; malformed tree
            }
            const auto& children = itChildren->second;
            for (int ci = 0; ci < (int)children.size(); ++ci) {
                int child = children[ci];
                if (isDescendant(child, coreId)) { idx = ci; break; }
            }
        }
        if (idx < 0) {
            // As a fallback, place under bucket -1 (should not happen on well-formed trees)
            idx = -1;
        }
        parentToChildIdxTargets[parent][idx].insert(coreId);
    }

    // Compute the *global OR* of child indices across all participating parents.
    std::unordered_map<int, std::unordered_set<int>> localMasks; // parent -> set of child indices with targets
    unordered_set<int> globalMaskIndices; // e.g., {0,1,2}
    for (int parent : parentSwitches) {
        auto it = coreTree.find(parent);
        if (it == coreTree.end()) continue; // defensive
        const auto& children = it->second;
        std::unordered_set<int> localMask;
        for (int ci = 0; ci < (int)children.size() && ci < MAX_GROUP_SIZE; ++ci) {
            auto ptIt = parentToChildIdxTargets[parent].find(ci);
            if (ptIt != parentToChildIdxTargets[parent].end() && !ptIt->second.empty()) {
                localMask.insert(ci);
                globalMaskIndices.insert(ci);
            }
        }
        localMasks[parent] = localMask;
    }

    {
        std::ostringstream oss;
        oss << "Per-parent masks (bits [0..3], left=child0):\n";
        for (int parent : parentSwitches) {
            const auto& lm = localMasks[parent];
            oss << "  parent " << parent << ": " << maskToBits(lm) << "  (indices:";
            int cnt = 0; for (int idx : lm) { oss << (cnt++?",":" ") << idx; } oss << ")\n";
        }
        oss << "Global OR mask: " << maskToBits(globalMaskIndices) << "  (indices:";
        int cnt = 0; for (int idx : globalMaskIndices) { oss << (cnt++?",":" ") << idx; }
        oss << ")";
        routingUtils.logToFile(oss.str());
    }

    // Route-like hints (no LCA used here, but list explicit target cores)
    for (const auto& kvp : coreToDstNeurons) {
        routingUtils.logToFile(std::string("Target core: ") + std::to_string(kvp.first));
    }

    // Now, each parent switch broadcasts to every child index in `globalMaskIndices`.
    // For each selected child index at a given parent, waste = (#leaf cores under that child) - (#targets under that child at this parent).
    int& srcWaste = wastePerNeuron[sourceNeuron];

    for (int parent : parentSwitches) {
        auto it = coreTree.find(parent);
        if (it == coreTree.end()) continue; // not a switch? skip
        const auto& children = it->second;

        for (int ci : globalMaskIndices) {
            if (ci < 0 || ci >= (int)children.size() || ci >= MAX_GROUP_SIZE) continue; // enforce 4-wide leaf groups
            int childNode = children[ci];

            // Leaf cores under this child subtree
            vector<int> leaves = collectLeafCores(childNode, coreTree);
            int leafCount = (int)leaves.size();

            // Number of *actual* target cores under this child for *this parent*
            int targetCountUnderChild = 0;
            auto ptIt = parentToChildIdxTargets[parent].find(ci);
            if (ptIt != parentToChildIdxTargets[parent].end()) {
                targetCountUnderChild = (int)ptIt->second.size();
            }

            int wasteHere = leafCount - targetCountUnderChild;
            if (wasteHere > 0) {
                {
                    std::ostringstream oss;
                    oss << "Core-level waste: under parent " << parent << ", child-index " << ci << ": ";
                    int printed = 0;
                    auto ptIt2 = parentToChildIdxTargets[parent].find(ci);
                    const std::unordered_set<int>* tgtSetPtr = (ptIt2 != parentToChildIdxTargets[parent].end()) ? &ptIt2->second : nullptr;
                    for (int c : leaves) {
                        if (!tgtSetPtr || tgtSetPtr->find(c) == tgtSetPtr->end()) {
                            if (printed++) oss << ", ";
                            oss << "core " << c << " was not a target";
                        }
                    }
                    routingUtils.logToFile(oss.str());
                }
                srcWaste += wasteHere;
                // Attribute waste to the receiving non-target leaf cores
                if (targetCountUnderChild == 0) {
                    // All leaves are waste
                    for (int c : leaves) {
                        wastedMessages[c] += 1; // one extra illegal delivery
                    }
                } else {
                    // Mark only non-target leaves as waste
                    const auto& tgtSet = ptIt->second; // set of target cores under this child
                    for (int c : leaves) {
                        if (tgtSet.find(c) == tgtSet.end()) {
                            wastedMessages[c] += 1;
                        }
                    }
                }
            }
        }
    }

    routingUtils.logToFile("Total core-level routing waste: " + std::to_string(wastePerNeuron[sourceNeuron]));
    routingUtils.logToFile("Finished simulation for source neuron " + std::to_string(sourceNeuron));
}

void HBSRoutingSimulator::traverseTree(int coreId, unordered_set<int>& visitedCores) {
    // Simple DFS that collects all reachable nodes from `coreId` (used for debugging)
    if (visitedCores.count(coreId)) return;
    visitedCores.insert(coreId);
    auto it = coreTree.find(coreId);
    if (it == coreTree.end()) return; // leaf core
    for (int child : it->second) traverseTree(child, visitedCores);
}

void HBSRoutingSimulator::reportWasteStatistics() const {
    long long totalWaste = 0;
    for (const auto& kv : wastePerNeuron) totalWaste += kv.second;

    //cout << "\n==== HBS Routing Waste Report ====" << '\n';
    //cout << "Total illegal deliveries (waste): " << totalWaste << '\n';

    //cout << "Per-neuron waste (non-zero only):" << '\n';
    // for (const auto& kv : wastePerNeuron) {
    //     if (kv.second > 0)
    //         cout << "  Neuron " << kv.first << ": " << kv.second << '\n';
    // }

    // cout << "Per-core waste (non-zero only):" << '\n';
    // for (const auto& kv : wastedMessages) {
    //     if (kv.second > 0)
    //         cout << "  Core " << kv.first << ": " << kv.second << '\n';
    // }
    // cout << "==================================" << "\n";

    // --- Write waste statistics to file ---
    std::ofstream metricsFile("../data/reports/hbs_waste_metrics.txt");
    if (metricsFile.is_open()) {
        metricsFile << "==== HBS Routing Waste Report ====" << '\n';
        metricsFile << "Total illegal deliveries (waste): " << totalWaste << '\n';
        metricsFile << "Per-neuron waste (non-zero only):" << '\n';
        for (const auto& kv : wastePerNeuron) {
            if (kv.second > 0)
                metricsFile << "  Neuron " << kv.first << ": " << kv.second << '\n';
        }
        metricsFile << "Per-core waste (non-zero only):" << '\n';
        for (const auto& kv : wastedMessages) {
            if (kv.second > 0)
                metricsFile << "  Core " << kv.first << ": " << kv.second << '\n';
        }
        metricsFile << "==================================" << "\n";
        metricsFile.close();
    }
}

unordered_map<int, int> HBSRoutingSimulator::getWastedMessagesPerCore() const {
    return wastedMessages;
}

int HBSRoutingSimulator::findLCA(int sourceCore, int targetCore) {
    // Not needed for the current HBS region-routing (we directly target parent switches).
    // Implement a minimal version for completeness / future use.
    unordered_set<int> ancestors;
    int cur = sourceCore;
    while (true) {
        ancestors.insert(cur);
        auto it = coreParent.find(cur);
        if (it == coreParent.end()) break;
        cur = it->second;
    }
    cur = targetCore;
    while (true) {
        if (ancestors.count(cur)) return cur;
        auto it = coreParent.find(cur);
        if (it == coreParent.end()) break;
        cur = it->second;
    }
    return sourceCore; // fallback
}

bool HBSRoutingSimulator::isDescendant(int current, int target) {
    // BFS from current; return true if we reach target
    queue<int> q;
    q.push(current);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        if (u == target) return true;
        auto it = coreTree.find(u);
        if (it == coreTree.end()) continue; // leaf
        for (int v : it->second) q.push(v);
    }
    return false;
}

vector<int> HBSRoutingSimulator::shortestPath(int startCore, int endCore) {
    // Build path via parents; no heavy use in current model, provided for debugging
    // Compute ancestors of start
    unordered_map<int, int> parentMap = coreParent; // copy for local access

    unordered_set<int> anc;
    int cur = startCore;
    while (true) {
        anc.insert(cur);
        auto it = parentMap.find(cur);
        if (it == parentMap.end()) break;
        cur = it->second;
    }
    // Climb from end to the first common ancestor
    vector<int> tail;
    cur = endCore;
    int lca = endCore;
    while (true) {
        tail.push_back(cur);
        if (anc.count(cur)) { lca = cur; break; }
        auto it = parentMap.find(cur);
        if (it == parentMap.end()) break;
        cur = it->second;
    }
    // Path from start to LCA
    vector<int> head;
    cur = startCore;
    while (cur != lca) {
        head.push_back(cur);
        auto it = parentMap.find(cur);
        if (it == parentMap.end()) break;
        cur = it->second;
    }
    head.push_back(lca);
    // Combine (head) + reverse(tail)
    reverse(tail.begin(), tail.end());
    head.insert(head.end(), tail.begin() + 1, tail.end());
    return head;
}