#include <iostream>
#include <fstream>
#include <string>
#include "../headers/individual.cuh"
#include "../headers/options.h"

// Intiailize file function <- called at start before an algorihtm to set the header row of the file
// Input: none
// Output: A file titled "performance.csv" is created with a header row to be used in recordGeneration()
void initializeRecording() {
    std::ofstream file;
    file.open("performance.csv", std::ios_base::binary);
    file << "Gen Number, Best Cost (m),\n";
    file.close();
}

// Append file performance function <- called every generation (or so) to record performance 
// Input: pool - array of individuals to be used in recording
//        constants - contains constant values such as pop_size and distance_tol
//        genCount - the number of generations currently having been performed
// Output: "performance.csv" is appended info regarding this generation (such as cost of best individual)
void recordGeneration(individual * pool, options * constants, int genCount) {
    //get the best thread from a generation. get it from the pool
    // Creating a seperate array for the sorted pool to not unshuffle the actual pool
    individual * sortedPool = new individual[constants->pop_size];
    for (int i = 0; i < constants->pop_size; i++) {
        sortedPool[i] = pool[i];
    }
    std::sort(sortedPool, sortedPool + constants->pop_size);

    std::ofstream ExcelFile;
    ExcelFile.open("performance.csv", std::ios_base::app);
    
    ExcelFile << genCount << "," << sortedPool[0].cost << ",\n";

    ExcelFile.close();
    delete [] sortedPool;
}

// Record solution function <- if we got a solution, this is called
// Input: pool - array of individuals to be used in recording what the solution(s) are
//        constants - contains constant values such as pop_size and distance_tol
// Output: constants and individuals with cost < constants->distance_tol are outputting into "results.txt" using << operator
void recordSolution(individual * pool, options * constants)
{
    // sort the pool
    std::sort(pool, pool + constants->pop_size);
    
    // Open a results file .txt
    std::ofstream resultsFile;
    resultsFile.open("results.txt", std::ios_base::binary);

    resultsFile << *constants << std::endl;

    int i = 0;
    while (pool[i].cost < constants->distance_tol && i < constants->pop_size) {
        // Iterate until we see an individual that is not a solution
        resultsFile << pool[i] << std::endl;
        i++;
    }
    resultsFile.close();
}