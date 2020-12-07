#include "../headers/coords.cuh"
#include "../headers/individual.cuh"
#include "../headers/options.h"
#include <curand.h>
#include <curand_kernel.h>

// Put genetics CUDA code here

// Mask is pointer to array of 4
// Output: mask contains values randomly between 1 and 3 inclusive
__global__ void initializeRandom(curandState_t *state, options *constants)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(constants->rng_seed, index, 0, &state[index]);
}


// Random Start
// Input: pool array containing individuals that need to be asssigned random parameters
//        constants - contains needed values
__device__ void randomStart(individual *pool, options &constants, curandState * state, int tid)
{
    double rand_phi   = curand_uniform(&state[tid]) * (2*PI); //phi randomly set between 0 and 2PI
    double rand_theta = curand_uniform(&state[tid]) * (PI/2); //theta randomly set between 0 and PI/2
    double rand_V     = curand_uniform(&state[tid]) * (constants.max_launch_v - constants.min_launch_v) + constants.min_launch_v; //V_naught randomly set between max and min allowed velocity range
    double rand_time  = curand_uniform(&state[tid]) * (constants.max_time - constants.min_time) + constants.min_time; //time randomly set between max and min allowed time
    // Assign in pool new individual using randomly generated values
    pool[tid] = individual(rand_phi, rand_theta, rand_V, rand_time);
}

// Random Mask generator
__device__ void maskGen (int *mask, curandState *state, int tid)
{
    for (int i = 0; i < 4; i++)
    {
        mask[i] = int(curand_uniform(&state[tid]) * 1000) % 3 + 1;
    }    //come back to put in a check against values that are already thee same as parents
}
// Crossover
// crossover(pool[tid], pool[tid-1], constants)
__device__ void crossover(individual * parent1, individual * parent2, options &constants, curandState * state, int tid)
{
    // Generate a mask to decide which genes get crossed over
    int * mask;
    mask = new int[4];
    maskGen(mask, state, tid);
    // Crossing over phi
    switch (mask[0]) 
    {
        // 1 - take from parent 1 (which is already in parent1)
        case (2) : // 2 - take from parent 2
            parent1->phi = parent2->phi;
            break;
        case (3) : // 3 - taking average
            parent1->phi = (parent1->phi + parent2->phi) / 2;
            break;
    }

    // Crossing over theta
    switch (mask[1])
    {
        case (2) : // 2 - take from parent 2
            parent1->theta = parent2->theta;
            break;
        case (3) : // 3 - taking average
            parent1->theta = (parent1->theta + parent2->theta) / 2;
            break;
    }

    // Crossing over V_nought
    switch (mask[2])
    {
        case (2) : // 2 - take from parent 2
            parent1->V_nought = parent2->V_nought;
            break;
        case (3) : // 3 - taking average
            parent1->V_nought = (parent1->V_nought + parent2->V_nought) / 2;
            break;
    }

    switch (mask[3])
    {
        case (2) : // 2 - take from parent 2
            parent1->time = parent2->time;
            break;
        case (3) : // 3 - taking average
            parent1->time = (parent1->time + parent2->time) / 2;
            break;
    }
    // Crossover complete, if we were doing mutations we would start here
    delete [] mask;
}


// new Generation Kernal

// GA Algorithm 
/*
// initialize curand using constants->rand_seed
// Call device randomStart to populate pool with random
// start for loop
// 


*/

