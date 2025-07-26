
/************************************************************************************
*
*   File Name: SpikeGenerator.cpp
*   Author: Ayush Patra
*   Description: Implements the SpikeGenerator class which handles the set of 
*                spiking neurons for a given timestep. Used in simulating spike 
*                activity based on static connectivity.
*
*   Version History:
*       - 2025-07-26: Initial version created.
*
************************************************************************************/


#include "SpikeGenerator.h"
#include <unordered_set>

// Constructor
SpikeGenerator::SpikeGenerator() {
    // initialize empty
}

// Set the neurons that fired
void SpikeGenerator::setSpikingNeurons(const std::unordered_set<int>& spikingNeurons) {
    this->spikingNeurons = spikingNeurons;
}

// Check if a specific neuron is firing
bool SpikeGenerator::isSpiking(int neuronId) const {
    return spikingNeurons.find(neuronId) != spikingNeurons.end();
}

// Get the full set of spiking neurons
const std::unordered_set<int>& SpikeGenerator::getSpikingNeurons() const {
    return spikingNeurons;
}