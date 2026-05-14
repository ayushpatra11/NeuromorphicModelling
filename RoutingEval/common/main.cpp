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

#ifndef NUM_NEURONS
#define NUM_NEURONS 512
#endif
#ifndef NUM_SAMPLES
#define NUM_SAMPLES 50
#endif
#ifndef NUM_MAPPINGS
#define NUM_MAPPINGS 10
#endif

int main() {
    Utils routingUtils;
    vector<int> neuronsPerCore = {16, 32, 64};

    for(int mapping = 1; mapping <=NUM_MAPPINGS; mapping++){

        for(int neurons_per_core : neuronsPerCore){

            for(int i = 1; i<=NUM_SAMPLES; i++){

                string matrixFilePath = "../data/connectivity_matrix/dynamic_connectivity_matrix_"+to_string(i)+".json";
                routingUtils.setConnectivityMatrix(matrixFilePath);
                vector<vector<int>> connectivityMatrix = routingUtils.getConnectivityMatrix();
                if (connectivityMatrix.empty()) {
                    cerr << "Failed to load connectivity matrix. Exiting." << endl;
                    return 1;
                }

                string reportDirectoryNeurogrid = "../data/reports/mapping"+to_string(mapping)+"/reports_"+to_string(NUM_NEURONS)+"_"+to_string(neurons_per_core)+"/neurogrid/waste_metrics_sample"+to_string(i)+".json";
                string reportDirectoryHBS = "../data/reports/mapping"+to_string(mapping)+"/reports_"+to_string(NUM_NEURONS)+"_"+to_string(neurons_per_core)+"/hbs/waste_metrics_sample"+to_string(i)+".json";
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

                // Side-by-side comparison
                long long neurGridWaste = simulator.getTotalWaste();
                long long hbsWaste      = sim.getTotalWaste();
                routingUtils.logToFile("\n------------------------------------------------------------");
                routingUtils.logToFile("COMPARISON  |  mapping=" + to_string(mapping) +
                                       "  sample=" + to_string(i) +
                                       "  neurons_per_core=" + to_string(neurons_per_core));
                routingUtils.logToFile("  Neurogrid total waste : " + to_string(neurGridWaste));
                routingUtils.logToFile("  HBS       total waste : " + to_string(hbsWaste));
                long long diff = neurGridWaste - hbsWaste;
                if (diff > 0)
                    routingUtils.logToFile("  HBS saves " + to_string(diff) + " illegal deliveries vs Neurogrid");
                else if (diff < 0)
                    routingUtils.logToFile("  Neurogrid saves " + to_string(-diff) + " illegal deliveries vs HBS");
                else
                    routingUtils.logToFile("  Both techniques produce equal waste");
                routingUtils.logToFile("------------------------------------------------------------\n");
            
            }
        }
    }
        
    return 0;
}