/***********************************************************************************
 *
 *   File Name: SpikeGenerator.h
 *   Author: Ayush Patra
 *   Description: Utility class to manage spike events for neurons during routing
 *                simulation. Tracks which neurons fired at a given timestep.
 *   Version History:
 *       - 2025-07-26: Initial version
 *
 **********************************************************************************/

#ifndef SPIKE_GENERATOR_H
#define SPIKE_GENERATOR_H

#include <unordered_set>

class SpikeGenerator {
public:
    SpikeGenerator();

    void setSpikingNeurons(const std::unordered_set<int>& spikingNeurons);
    bool isSpiking(int neuronId) const;
    const std::unordered_set<int>& getSpikingNeurons() const;

    ~SpikeGenerator() {
    }

private:
    std::unordered_set<int> spikingNeurons;
};

#endif  // SPIKE_GENERATOR_H
