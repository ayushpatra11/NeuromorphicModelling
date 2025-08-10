/***********************************************************************************
 *
 *   File Name: SpikeGenerator.h
 *   Author: Ayush Patra
 *   Description: Utility class to generate spike events for neurons during routing
 *                simulation. Spikes can be generated randomly or read from a file.
 *   Version History:
 *       - 2025-07-26: Initial version
 *
 **********************************************************************************/

#ifndef SPIKE_GENERATOR_H
#define SPIKE_GENERATOR_H

#include <vector>
#include <random>
#include <string>

class SpikeGenerator {
public:
    // Constructor
    SpikeGenerator(int num_neurons);

    // Generate random spikes
    std::vector<int> generateRandomSpikes(float spike_prob);

    // Load spike data from a file
    bool loadSpikesFromFile(const std::string& filepath);

    // Get the list of spiking neurons
    std::vector<int> getSpikingNeurons() const;
    ~SpikeGenerator(){}

private:
    int num_neurons_;
    std::vector<int> spikes_; // list of firing neurons
};

#endif // SPIKE_GENERATOR_H