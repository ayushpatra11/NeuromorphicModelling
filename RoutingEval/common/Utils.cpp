/************************************************************************************
*
*   File Name: Utils.cpp
*   Author:  Ayush Patra
*   Description: Implementation file for utility functions used in routing analysis
*                and SNN simulation processing. Includes helpers for matrix operations,
*                logging, and data handling.
*   Version History:        
*       - 2025-07-26: Initial version
*
************************************************************************************/

#include "Utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "nlohmann/json.hpp" 
using json = nlohmann::json;

using namespace std;

Utils::Utils(string matrixFilePath): connectivityMatrixFilePath(matrixFilePath) {
        weightThreshold = 0.0435f;
        logFileName = createLogFileName();
        logToFile("Utils initialized");
        // Load the connectivity matrix from the specified file
        try{
            connectivityMatrix = loadConnectivityMatrix(connectivityMatrixFilePath);
        } catch (const exception& e) {
            cerr << "Error loading connectivity matrix: " << e.what() << endl;
            logToFile("Error loading connectivity matrix: " + string(e.what()));
            return;
        }
        logToFile("Connectivity matrix loaded from: " + connectivityMatrixFilePath);
        logToFile("Connectivity matrix dimensions: " + to_string(connectivityMatrix.size()) + "x" + to_string(connectivityMatrix[0].size()));
        //printConnectivityMatrix();
}

void Utils::printConnectivityMatrix() {
    // this will print the matrix for the connectivity matrix
    // can be used for core neuron mapping as well. 
    logToFile("Connecticity matrix:");
    for (const auto& row : connectivityMatrix) {
        string r = "";
        for (const auto& val : row) {
            r += to_string(val) + " ";
        }
        logToFile(r);
    }
}

void Utils::printNeuronMap() {
    // this will print the matrix for the connectivity matrix
    // can be used for core neuron mapping as well. 
    logToFile("Neuron map:");
    for(const auto& pair : neuronCoreMap) {
        logToFile("Neuron " + to_string(pair.first) + " -> Core " + to_string(pair.second));
    }
}

vector<vector<int>> Utils::loadConnectivityMatrix(const string& filePath) {
    //connecticity matrix is made from the python script. 
    ifstream file(filePath);
    vector<vector<int>> matrix;

    if (!file.is_open()) {
        cerr << "Error opening file: " << filePath << endl;
        return matrix;
    }

    json j;
    file >> j;

    for (const auto& row : j) {
        vector<int> binaryRow;
        for (const auto& val : row) {
            binaryRow.push_back(val>weightThreshold? 1: 0);
        }
        matrix.push_back(binaryRow);
    }

    file.close();
    return matrix;
}

vector<vector<int>> Utils::initializeZeroMatrix(int rows, int cols) {
    return vector<vector<int>>(rows, vector<int>(cols, 0));
}

void Utils::logToFile(const string& message) {
    time_t now = time(0);
    tm* localTime = localtime(&now);
    char buffer[80];
    strftime(buffer, sizeof(buffer), "[%Y-%m-%d_%H-%M-%S]: ", localTime);
    ofstream outFile(logFileName, ios_base::app);
    if (outFile.is_open()) {
        outFile <<buffer << message << endl;
        outFile.close();
    } else {
        cerr << "Unable to open log file: " << logFileName << endl;
    }
}

string Utils::createLogFileName(){
    //creating a log file name with the current date and time
    time_t now = time(0);
    tm* localTime = localtime(&now);
    char buffer[80];
    strftime(buffer, sizeof(buffer), "../logs/routing_log_%Y-%m-%d_%H-%M-%S.txt", localTime);
    return string(buffer);
}