/*####################################################################################
#
#   File Name: main.cpp
#   Author:  Ayush Patra
#   Description: Entry point for running the neurogrid-style routing waste analysis
#                simulation using static neuron-to-neuron connectivity.
#   Version History:        
#       - 2025-07-26: Initial version
#       - 2025-07-26: Refactored to integrate ConnectivityMatrix and NeuronMapper
#
####################################################################################*/

#include "RoutingSimulator.h"
#include "NeuronMapper.h"
#include "HBSNeuronMapper.h"
#include "HBSRoutingSimulator.h"
#include "Utils.h"
#include <iostream>

using namespace std;

int main() {
    // Loading the connectivity matrix from a file
    string matrixFilePath = "../data/dynamic_connectivity_matrix.json";
    Utils routingUtils(matrixFilePath);
    vector<vector<int>> connectivityMatrix = routingUtils.getConnectivityMatrix();
    if (connectivityMatrix.empty()) {
        cerr << "Failed to load connectivity matrix. Exiting." << endl;
        return 1;
    }

    NeuronMapper neuronMapper(512, 16, connectivityMatrix);
    HBSNeuronMapper hbsNeuronMapper(512, 16, connectivityMatrix);
    routingUtils.logToFile("NeuronMapper initialized for Neurogrid and HBS routing approaches. Check \"RoutingEval/data/hbs_core_tree.txt\" and \"RoutingEval/data/core_tree.txt\"...");

    routingUtils.setNeuronCoreMap(neuronMapper.getNeuronToCoreMap());
    //routingUtils.printNeuronMap();

    routingUtils.logToFile("\n\n\n========================STARTING NEUROGRID SIMULATION ===========================================================\n\n\n");
    // Initialize RoutingSimulator and run simulation
    RoutingSimulator simulator(connectivityMatrix,
                               neuronMapper.getNeuronToCoreMap(),
                               neuronMapper.getCoreTree(), neuronMapper.getCoreParent(), routingUtils);
    simulator.simulate();

    routingUtils.logToFile("\n\n\n========================ENDING NEUROGRID SIMULATION ===========================================================\n\n\n");

    routingUtils.logToFile("\n\n\n========================STARTING HBS SIMULATION ===========================================================\n\n\n");
    HBSRoutingSimulator sim(connectivityMatrix, hbsNeuronMapper.getNeuronToCoreMap(),
                            hbsNeuronMapper.getCoreTree(), hbsNeuronMapper.getCoreParent(), routingUtils);
    sim.simulate();                // runs HBS-style parent-targeting + global OR mask routing
    sim.reportWasteStatistics();

    routingUtils.logToFile("\n\n\n========================ENDING HBS SIMULATION ===========================================================\n\n\n");
    // Retrieve and print routing waste
    // auto waste = simulator.getWastedMessagesPerCore();
    // int totalWaste = 0;
    // for (const auto& [core, count] : waste) {
    //     cout << "Core " << core << " wasted messages: " << count << endl;
    //     totalWaste += count;
    // }
    // cout << "Total routing waste: " << totalWaste << endl;

    return 0;
}