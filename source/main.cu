#include <iostream>
#include <string>
#include <random>
#include "../headers/options.h"
#include "../headers/individual.cuh"
#include "../headers/coords.cuh"
#include "physicsCPU.cpp"
#include "device.cu"

// For phase 2 (genetic algorithm introduction)
//      Implement Genetic sorting, crossover, and mutation
//      Implement Genetic algorithm to handle iterative loop

// For phase 3 (polishing initial goals)
//      Reach validation for efficient, consistent, and accurate solutions
//      Documententation and readability at this point is a must
//      Initial objectives reached

// For phase 4 (expanded goals)
//      Implement more in-depth and precise simulation of atmosphere and gravity factors
//      Add more configurable options to allow for more customized objectives
//          Ideas include consideration for terrain

void geneticAlgorithm(options * constants) {

    // Initializing the seed
    std::mt19937_64 randomGen(constants->rng_seed);

    
    // Initializing the pool with random individuals
    individual * pool = (individual*)malloc(sizeof(individual) * (constants->pop_size+1));
    for (int i = 0; i < constants->pop_size; i++) {
        pool[i] = individual(*constants, randomGen);
    }
//    double currentGen = 0;
    callGPU(pool, constants);
    // Sort the pool by cost test
    std::sort(pool, pool + constants->pop_size);
    /*
    do {
        // Perform the kernal
        callGPU(pool, constants);
        // select individuals for parent pool (would require sorting pool)
        // perform genetic crossover and mutation
        newGeneration(pool, constants, randomGen);
        std::sort(pool);
    } while (pool[0].cost > constants.distance_tol || max_generations > currentGen);
    // Output resulting solution

    */
    delete [] pool;
}




int main(int argc, char *argv[]) { // main.exe input.config <- command to run program with file path to config file being "input.config"

    options * config = new options(argv[1]);
    std::cout << *config;
    
    geneticAlgorithm(config);

    std::cout <<"Exiting program...";
    delete config;
}