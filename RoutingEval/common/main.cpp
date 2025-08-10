/*####################################################################################
#
#   File Name: main.cpp
#   Author:  Ayush Patra
#   Description: Entry point for running the neurogrid-style routing waste analysis
#                simulation using static neuron-to-neuron connectivity.
#   Version History:        
#       - 2025-07-26: Initial version
#       - 2025-07-26: Refactored to integrate ConnectivityMatrix and NeuronMapper
#       - 2025-08-10: Added support for HBS routing and also for multiple samples
#
####################################################################################*/

#include "RoutingSimulator.h"
#include "NeuronMapper.h"
#include "HBSNeuronMapper.h"
#include "HBSRoutingSimulator.h"
#include "Utils.h"
#include <iostream>

using namespace std;

#define NUM_NEURONS 512
#define NUM_SAMPLES 50

int main() {
    Utils routingUtils;
    vector<int> neuronsPerCore = {16, 32, 64};

    for(int neurons_per_core : neuronsPerCore){

        for(int i = 1; i<=NUM_SAMPLES; i++){

            string matrixFilePath = "../data/connectivity_matrix/dynamic_connectivity_matrix_"+to_string(i)+".json";
            routingUtils.setConnectivityMatrix(matrixFilePath);
            vector<vector<int>> connectivityMatrix = routingUtils.getConnectivityMatrix();
            if (connectivityMatrix.empty()) {
                cerr << "Failed to load connectivity matrix. Exiting." << endl;
                return 1;
            }

            string reportDirectoryNeurogrid = "../data/reports/reports_"+to_string(NUM_NEURONS)+"_"+to_string(neurons_per_core)+"/neurogrid/waste_metrics_sample"+to_string(i)+".json";
            string reportDirectoryHBS = "../data/reports/reports_"+to_string(NUM_NEURONS)+"_"+to_string(neurons_per_core)+"/hbs/waste_metrics_sample"+to_string(i)+".json";
            NeuronMapper neuronMapper(NUM_NEURONS, neurons_per_core, connectivityMatrix);
            HBSNeuronMapper hbsNeuronMapper(NUM_NEURONS, neurons_per_core, connectivityMatrix);
            routingUtils.logToFile("NeuronMapper initialized for Neurogrid and HBS routing approaches. Check \"RoutingEval/data/hbs_core_tree.txt\" and \"RoutingEval/data/core_tree.txt\"...");

            routingUtils.setNeuronCoreMap(neuronMapper.getNeuronToCoreMap());
            //routingUtils.printNeuronMap();

            routingUtils.logToFile("\n\n\n========================STARTING NEUROGRID SIMULATION "+to_string(i)+" FOR "+to_string(neurons_per_core)+" NEURONS PER CORE "+"===========================================================\n\n\n");
            // Initialize RoutingSimulator and run simulation
            RoutingSimulator simulator(connectivityMatrix,
                                    neuronMapper.getNeuronToCoreMap(),
                                    neuronMapper.getCoreTree(), neuronMapper.getCoreParent(), routingUtils, reportDirectoryNeurogrid);
            simulator.simulate();

            routingUtils.logToFile("\n\n\n========================ENDING NEUROGRID SIMULATION "+to_string(i)+" FOR "+to_string(neurons_per_core)+" NEURONS PER CORE "+"===========================================================\n\n\n");

            routingUtils.logToFile("\n\n\n========================STARTING HBS SIMULATION "+to_string(i)+" FOR "+to_string(neurons_per_core)+" NEURONS PER CORE "+"===========================================================\n\n\n");
            HBSRoutingSimulator sim(connectivityMatrix, hbsNeuronMapper.getNeuronToCoreMap(),
                                    hbsNeuronMapper.getCoreTree(), hbsNeuronMapper.getCoreParent(), routingUtils, reportDirectoryHBS);
            sim.simulate();                
            sim.reportWasteStatistics();

            routingUtils.logToFile("\n\n\n========================ENDING HBS SIMULATION "+to_string(i)+" FOR "+to_string(neurons_per_core)+" NEURONS PER CORE "+"===========================================================\n\n\n");
        
        }
    }
        
    return 0;
}