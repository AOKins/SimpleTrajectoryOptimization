#include <iostream>
#include <string>
#include <random>
#include "../headers/options.h"
#include "../headers/individual.h"
#include "../headers/coords.h"
#include "physics.cpp"

// For phase 2 (genetic algorithm introduction)
//      Implement CUDA kernal to perform population
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

void geneticAlgorithm(options constants) {

    // Initializing the seed
    std::mt19937_64 randomGen(constants.rng_seed);


    // Initializing the pool with random individuals
    individual * pool = (individual*)malloc(sizeof(individual) * constants.pop_size);
    for (int i = 0; i < constants.pop_size; i++) {
        pool[i] = individual(constants, randomGen);
    }
    double currentGen = 0;
    /*
    do {
        // Perform the kernal
        // select individuals for parent pool (would require sorting pool)
        // perform genetic crossover and mutation
    } while (pool[0].cost > constants.distance_tol || max_generations > currentGen);
    
    */

    delete [] pool;
}




int main(int argc, char *argv[]) { // main.exe input.config <- command to run program with file path to config file being "input.config"

    options config(argv[1]);
    std::cout << config;

    geneticAlgorithm(config);
}