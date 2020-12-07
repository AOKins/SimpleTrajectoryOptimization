#include <iostream>
#include <string>
#include <random>
#include "../headers/options.h"
#include "../headers/individual.cuh"
#include "../headers/coords.cuh"
#include "physicsCPU.cpp"
#include "device.cu"


// CPU version (should put in own seperate file later)
/*
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

    
    delete [] pool;
}
*/




int main(int argc, char *argv[]) { // main.exe input.config <- command to run program with file path to config file being "input.config"

    options * config = new options(argv[1]);
    std::cout << *config;
    individual * pool = new individual[config->pop_size];
    callGPU(pool, config);

    std::cout <<"Exiting program...";
    delete config;
    delete [] pool;
}