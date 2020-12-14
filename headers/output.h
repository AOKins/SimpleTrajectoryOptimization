#ifndef _OUTPUT_H_
#define _OUTPUT_H_
#include "individual.cuh"
#include "options.h"

// Intiailize file function <- called at start before an algorihtm to set the header row of the file
// Input: none
// Output: A file titled "performance.csv" is created with a header row to be used in recordGeneration()
void initializeRecording();

// Append file performance function <- called every generation (or so) to record performance 
// Input: pool - array of individuals to be used in recording
//        constants - contains constant values such as pop_size and distance_tol
//        genCount - the number of generations currently having been performed
// Output: "performance.csv" is appended info regarding this generation (such as cost of best individual)
void recordGeneration(individual * pool, options * constants, int genCount);

// Record solution function <- if we got a solution, this is called
// Input: pool - array of individuals to be used in recording what the solution(s) are
//        constants - contains constant values such as pop_size and distance_tol
// Output: constants and individuals with cost < constants->distance_tol are outputting into "results.txt" using << operator
void recordSolution(individual * pool, options * constants);

// Function for outputting generational data on the terminal
// Input: pool array and constants for getting individual data
//        genCount for how many generations have been performed
// Output: outputs genCount and best individual's cost (smallest) to the terminal
void terminalDisplay(individual * pool, options * constants, int genCount);

#include "../source/output.cpp"

#endif