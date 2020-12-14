#include <fstream>
#include <string>
#include "../headers/individual.cuh"
#include "../headers/options.h"

// Intiailize file function <- called at start before an algorihtm to set the header row of the file
// Input: none
// Output: A file titled "performance.csv" is created with a header row to be used in recordGeneration()
void initializeRecording() {
    /// Create a file called performance.csv
    std::ofstream file;
    file.open("performance.csv", std::ios_base::binary);
    // Output a header row only and then close
    file << "Gen Number, Best Cost (m),\n";
    file.close();
}

// Append file performance function <- called every generation (or so) to record performance 
// Input: pool - array of individuals to be used in recording
//        constants - contains constant values such as pop_size and distance_tol
//        genCount - the number of generations currently having been performed
// Output: "performance.csv" is appended info regarding this generation (such as cost of best individual)
void recordGeneration(individual * pool, options * constants, int genCount) {
    // Creating a seperate array for the sorted pool to not unshuffle the actual pool
    individual * sortedPool = new individual[constants->pop_size];
    for (int i = 0; i < constants->pop_size; i++) {
        sortedPool[i] = pool[i];
    }
    // Sort the copied array
    std::sort(sortedPool, sortedPool + constants->pop_size);
    // Open to append and output best individual to the performance.csv file 
    std::ofstream ExcelFile;
    ExcelFile.open("performance.csv", std::ios_base::app);
    
    ExcelFile << genCount << "," << sortedPool[0].cost << ",\n";
    // Close the excel file and delete the seperate pool array
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
    // output the constants object into the text file first
    resultsFile << *constants << std::endl;

    std::cout << "\n Best Final is " << pool[0].cost << std::endl;
    std::cout << "Worst Final is " << pool[constants->pop_size-1].cost << std::endl;

    // Output all valid solutions to resultsFile
    int i = 0;
    while (pool[i].cost < constants->distance_tol && i < constants->pop_size) {
        // Iterate until we see an individual that is not a solution
        resultsFile << pool[i] << std::endl;
        i++;
    }
    // Got all info out, close file and end
    resultsFile.close();
}

// Function for outputting generational data on the terminal
// Input: pool array and constants for getting individual data
//        genCount for how many generations have been performed
// Output: outputs genCount and best individual's cost (smallest) to the terminal
void terminalDisplay(individual * pool, options * constants, int genCount) {
    // Create a copied array to not impact the pool's ordering
    individual * sortedPool = new individual[constants->pop_size];
    for (int i = 0; i < constants->pop_size; i++) {
        sortedPool[i] = pool[i];
    }
    // Sort the copied array
    std::sort(sortedPool, sortedPool + constants->pop_size);

    // Output onto the terminal the generation and best individual's cost
    std::cout << "\nGeneration: " << genCount << std::endl;
    std::cout << "\t Best Individual Cost: " << sortedPool[0].cost << std::endl;

    delete [] sortedPool;
}