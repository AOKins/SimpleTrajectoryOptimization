#include "../headers/coords.cuh"
#include "../headers/individual.cuh"
#include "../headers/options.h"

// NOTE: GPU and CPU may not agree on timing for getting a solution (number of generations for example) because of differences in randomness and/or handling floating point numbers

/*
// input: pool, constants
// constants have copied into local memory (pass by value)
func randomStartKernal
    // get tid 
    // locally, create individual using random constructor
    // assign pool[tid] with newly created individual
    // set foundSolution = false;
*/

/*
// add input - boolean flag found solution (foundSolution)
// assume initially that foundSolution is false 
func simulateGPU
// perform trajectories for an individual parameter set
// derive cost
// if cost <= tolerance then foundSolution = true
*/
enum MaskValues {
    PARENT1 = 1,
    PARENT2,
    AVERAGE
}

/*
void maskGenerator(output, rng) {
    // 1 => p1
    // 2 => p2
    // 3 => average between them
    for (int i = 0; i < 4; i++) {
        output[i] = (rng() % 3) + 1  // random number between 1 and 3 which translates to corresponding enum MaskValues
    }
}
*/


/*
input: individual p1, p2 (pointers but not to global memory)
        options constants (for constants needed)
        randomnumberGenerator (for mask generation)
output: p1 is set new individual (old p1 is gone)
crossover() {
    generateMask(mask, rng) // array that contains mask data (which genes come from which parents or averaging)


    // use mask to determine which genes go to new individual
    // 1 do nothin
    // 2: p1.property = p2.property
    // 3: p1.property = (p1.property + p2.property) / 2
    // Also, have bound checking or something to make sure still value

    // (Optional) - mutations which we can consider later if needed

}


*/





/*
// input: pool, constants, randomNumberGenerator
// output: pool contains new individuals (best survive)
nextGeneration {
    // locally
        copy self (pool[tid])
        copy left-neighbor
        copy right-neighbor

    // _syncthreads() // at this point this thread has all data it needs without needing to worry about cross-contamination

    // if self is worse than left by having greater cost (less desirability)
    if self.cost > left.cost {
        crossover(self, left, constants, rng)
    }
    else if self.cost > right.cost {
        crossover(self, right, constants, rng)
    }
    else {
        do nothing (currently sees itself as best)
    }

    store resulting self into pool[tid]
}
*/




// 3 "Main" Kernals 

// Initializer (performed once)
// 1. randomStartKernal -> Initializes the pool in memory with random individuals

// 2. Simulation Itself (continue until solution found or max_generations reached)
// simulateGPU -> Perform simulate on each individual in a thread to get cost for parameters

// If found solution, no need to call nextGeneration kernal (end of loop)

//3.
// nextGeneration -> If no solution (cost > tolerance for all individuals) then perform crossover to create new individuals




/*
// Inputs:  ind - 
//          
(500 + tid+1) % 500




*/


// Takes in a pool of individuals and creates a new pool out of it
// Input: constants - options struct that holds all configurable variables
//        pool - pointer to array of individuals that have finished their simulations and have updated costs (in CUDA this would be after _syncthreads() for simulate())
//        rngEng - random number generator to be used when needing a random number
//        foundSolution - boolean flag that is true when an individual has a valid solution
//        genCount - how many generations have been performed so far
// Output: pool contains new array of individuals to be tested
void nextGeneration(options constants, individual * pool, std::mt19937_64 & rngEng, bool & foundSolution, int & genCount) {
    /* 
    if pool[tid].cost < constants->distance_tol {
        foundSolution = true;
    }
    if (foundSolution == true || genCount >= constants->max_generations) {
        return;
    } 
    else {
        // Performing crossover
        // Think about how to loop around (for right tid+1 % 500 ) (for left tid-1)
        // Generate random mask that determines *some* paramaters are taken from neighbor
        if (pool[tid].cost > pool[tid-1].cost) {
            // Crossover with neighbor [tid-1] and assign resulting new individual to [tid]
            // This individual can only do this once
        }
        else if (pool[tid].cost > pool[tid+1].cost) {
            // Crossover with neighbor [tid-1] and assign resulting new individual to [tid]
            // This individual can only do this once
        }
        
    }
    */    
}
