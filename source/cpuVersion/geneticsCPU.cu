#include "../../headers/coords.cuh"
#include "../../headers/individual.cuh"
#include "../../headers/options.h"
#include "../../headers/output.h"
#include "../physics.cu"


// Method to generate a random mask
// Input: mask - pointer of length 4 to store mask values that range between 1 and 3
//        rng - the random number generator to use
// Output: mask contains 4 values randomly assigned 1,2, or 3
void maskGen(int * mask, std::mt19937_64 & rng)
{
    mask[0] = rng() % 3 + 1;
    mask[1] = rng() % 3 + 1;
    mask[2] = rng() % 3 + 1;
    mask[3] = rng() % 3 + 1;
}

// Method to handle the crossover between two parents to create a new individual
// Input: parent1, parent2 - two individuals to crossover to create new individual out of
//        rng - the random number generator to use
// Output: parent1 holds the new individual, parent2 remains uncahnged
void crossoverCPU(individual& parent1, individual& parent2, std::mt19937_64& rng)
{
    int * mask = new int[4];
    maskGen(mask, rng);

    switch (mask[0]) 
    {
        case (1) :
            parent1.phi = parent1.phi;
            break;
        // 1 - take from parent 1 (which is already in parent1)
        case (2) : // 2 - take from parent 2
            parent1.phi = parent2.phi;
            break;
        case (3) : // 3 - taking average
            parent1.phi = (parent1.phi + parent2.phi) / 2;
            break;
        default :
            printf("invalid mask value");
    }
    // Crossing over theta
    switch (mask[1])
    {
        case (1) :
            parent1.theta = parent1.theta;
            break;
        case (2) : // 2 - take from parent 2
            parent1.theta = parent2.theta;
            break;
        case (3) : // 3 - taking average
            parent1.theta = (parent1.theta + parent2.theta) / 2;
            break;
        default :
            printf("invalid mask value");
    }

    // Crossing over V_nought
    switch (mask[2])
    {
        case (1) :
            parent1.V_nought = parent1.V_nought;
            break;
        case (2) : // 2 - take from parent 2
            parent1.V_nought = parent2.V_nought;
            break;
        case (3) : // 3 - taking average
            parent1.V_nought = (parent1.V_nought + parent2.V_nought) / 2;
            break;
        default :
            printf("invalid mask value");
    }

    switch (mask[3])
    {
        case (1) :
            parent1.time = parent1.time;
            break;
        case (2) : // 2 - take from parent 2
            parent1.time = parent2.time;
            break;
        case (3) : // 3 - taking average
            parent1.time = (parent1.time + parent2.time) / 2;
            break;
        default :
            printf("invalid mask value");
    }
    
    delete [] mask;
}

// Method to handle the overall genetic algorithm behavior
// Input: constants - values from config that contains information such as pop_size, distance_tol, etc. that need to be avaliable
//        pool - pointer to array of individuals
//        copy - pointer to array of copy of pool to draw survivorPool from instead of pool (to prevent contamination from previous calls)
//        id - index value for the individual being performed in the genetic algorithm, this method chosen so it is similar to GPU version in comparing
//        rng - the random number generator to use
// Output: pool[id] contains possibly (if not a local best) new individual that needs to be evaluated for cost
void geneticAlgorithmCPU(options * constants, individual * pool, individual * copy, int id, std::mt19937_64 & rng) {
    int blockID = id / 32; // Get what would be a 32 sized block of the pool that this individual is within 
    individual p1, p2;

    individual * survivorPool = new individual[32];

    for (int i = 0; i < 32; i++) {
        survivorPool[i] = copy[blockID*32+i];
    }

    std::sort(survivorPool, survivorPool+32);

    int output_id = (id + 32) % constants->pop_size;

    if (survivorPool[0].cost < copy[id].cost) {
        p1 = pool[0];
        p2 = pool[1];
        crossoverCPU(p1, p2, rng);
        pool[output_id] = p1;
    }
    else {
        pool[output_id] = copy[id];
    }
    delete [] survivorPool;
}

// Method for performing the search for a solution using a genetic algorithm and physics simulation
// Input: constants - values from config that contains information such as pop_size, distance_tol, etc. that need to be avaliable
//        pool - pointer to array of individuals to use
// Output: pool may contains individuals that are a solution to hitting the target
void callCPU(options * constants, individual * pool )
{

    // Initialize performance file by calling it
    // initializeRecording();
    // Creating a temp array that is to contain a copy of a current generation to draw individuals from in geneticAlgorithm without contamination
    individual * copy = new individual[constants->pop_size]; 

    // Create rng generator
    std::mt19937_64 rng(constants->rng_seed);
    bool foundSolution = false;
    // Use to initialize entire pool
    for (int i = 0; i < constants->pop_size; i++) {
        pool[i] = individual(*constants, rng);
    }

    int genCount = 0;
    do {
        // call simulate for each
        for (int i = 0; i < constants->pop_size; i++) {
            simulate(*constants, &pool[i]);

            if (pool[i].cost < constants->distance_tol) {
                foundSolution = true;
            }
            copy[i] = pool[i]; // Copy individual into temp
        }
        
        // record best here!
        // recordGeneration(pool, constants, genCount);

        // Every display frequency display onto the terminal using terminalDislay() method in output.cpp
        // if (genCount % constants->display_freq == 0) {
        //     terminalDisplay(pool, constants, genCount);
        // }

        // if no solution, perform crossovers
        if (foundSolution == false)
        {
            // Perform crossover method for a given individual
            for (int i = 0; i < constants->pop_size; i++) {
                geneticAlgorithmCPU(constants, pool, copy, i, rng);                
            }
        }
        genCount++;
    // while no solution and genCount < maxGen
    } while (foundSolution == false && genCount < constants->max_generations);

    // Done, can deallocate copy
    delete [] copy;
}
