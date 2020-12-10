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
//        id - index value for the individual being performed in the genetic algorithm, this method chosen so it is similar to GPU version in comparing
//        rng - the random number generator to use
// Output: pool[id] contains possibly (if not a local best) new individual that needs to be evaluated for cost
void geneticAlgorithmCPU(options * constants, individual * pool, int id, std::mt19937_64 & rng) {
    int leftIndex, rightIndex;
    individual self, left, right;
    self = pool[id];
    leftIndex = (constants->pop_size + id-1) % constants->pop_size;
    rightIndex = (constants->pop_size + id+1) % constants->pop_size;
    
    self = pool[id];
    left = pool[leftIndex];
    right = pool[rightIndex];

    if (left.cost < self.cost) {
        crossoverCPU(self, left, rng);
    }
    else if (right.cost < self.cost)
    {
        crossoverCPU(self, right, rng);
    }
}

// Method for performing the search for a solution using a genetic algorithm and physics simulation
// Input: constants - values from config that contains information such as pop_size, distance_tol, etc. that need to be avaliable
//        pool - pointer to array of individuals to use
// Output: pool may contains individuals that are a solution to hitting the target
void callCPU(options * constants, individual * pool )
{
    // Initialize performance file by calling it
    initializeRecording();

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
        }
        // record best here!
        recordGeneration(pool, constants, genCount);

        // if no solution, perform crossovers
        if (foundSolution == false)
        {
            // Perform crossover method for a given individual
            for (int i = 0; i < constants->pop_size; i++) {
                geneticAlgorithmCPU(constants, pool, i, rng);                
            }
        }
        genCount++;
    } while (foundSolution == false && genCount < constants->max_generations);
    // while no solution and genCount < maxGen
    std::cout << genCount << std::endl;

    recordSolution(pool, constants);
}
