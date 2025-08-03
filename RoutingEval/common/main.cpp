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

    NeuronMapper neuronMapper(512, 32, connectivityMatrix);
    routingUtils.logToFile("NeuronMapper initialized...");

    routingUtils.setNeuronCoreMap(neuronMapper.getNeuronToCoreMap());
    routingUtils.printNeuronMap();

    // Initialize RoutingSimulator and run simulation
    RoutingSimulator simulator(connectivityMatrix,
                               neuronMapper.getNeuronToCoreMap(),
                               neuronMapper.getCoreTree(), neuronMapper.getCoreParent(), routingUtils);
    simulator.simulate();

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