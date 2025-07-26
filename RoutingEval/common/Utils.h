/*####################################################################################
#
#   File Name: Utils.h
#   Author:  Ayush Patra
#   Description: Utility functions for reading matrices, parsing mappings, logging, 
#                and debugging used throughout the neurogrid routing evaluation project.
#   Version History:        
#       - 2025-07-26: Initial version
#
####################################################################################*/


#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

class Utils {
private:
    string logFileName;
    string connectivityMatrixFilePath;
    vector<vector<int>> connectivityMatrix;
    unordered_map<int, int> neuronCoreMap;
public:
    Utils(string matrixFilePath);
	// Prints a matrix to stdout (for debug)
	void printConnectivityMatrix();

    // Prints a matrix to stdout (for debug)
	void printNeuronMap();

	// Reads a plain text matrix from a file
	static vector<vector<int>> loadConnectivityMatrix(const string& filePath);

	// Initializes a zero matrix of given dimensions
	static vector<vector<int>> initializeZeroMatrix(int rows, int cols);

	// Logs message with timestamp
	void logToFile(const string& message);

	// Creates a log file name based on current date and time
	static string createLogFileName();

    // Gets the log file name
    string getLogFileName() {
        return logFileName;
    }

    // Gets the connectivity matrix file path
    vector<vector<int>> getConnectivityMatrix() {
        return connectivityMatrix;
    }

    // Gets the connectivity map
    unordered_map<int, int> getNeuronCoreMap() {
        return neuronCoreMap;
    }

    // Sets the neuron core map
    void setNeuronCoreMap(const unordered_map<int, int>& map) {
        neuronCoreMap = map;
    }

};

#endif // UTILS_H