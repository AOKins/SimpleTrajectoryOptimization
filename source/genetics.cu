#include "../headers/coords.cuh"
#include "../headers/individual.cuh"
#include "../headers/options.h"
#include <curand.h>
#include <curand_kernel.h>

// Put genetics CUDA code here


// Random Start
// Input: pool array containing individuals that need to be asssigned random parameters
//        constants - contains needed values
__device__ void randomStart(individual *pool, options &constants, curandState * state, int tid)
{   // Generate a random number between 0 and 1111 then divide by 10 (to get a double value between 0 and 1)
    double rand_phi   = (double(curand(&state[tid]) % 1111) / 1110.0) * (2*PI); //phi randomly set between 0 and 2PI
    double rand_theta = (double(curand(&state[tid]) % 1111) / 1110.0) * (PI/2); //theta randomly set between 0 and PI/2
    double rand_V     = (double(curand(&state[tid]) % 1111) / 1110.0) * (constants.max_launch_v - constants.min_launch_v) + constants.min_launch_v; //V_naught randomly set between max and min allowed velocity range
    double rand_time  = (double(curand(&state[tid]) % 1111) / 1110.0) * (constants.max_time - constants.min_time) + constants.min_time; //time randomly set between max and min allowed time
    // Assign in pool new individual using randomly generated values
    pool[tid] = individual(rand_phi, rand_theta, rand_V, rand_time);
}

// Mask is pointer to array of 4
// Output: mask contains values randomly between 1 and 3 inclusive
__global__ void initializeRandom(individual * pool, curandState_t *state, options *constants, int * foundSolution)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(constants->rng_seed, index, 0, &state[index]);

    randomStart(pool, *constants, state, index);
    *foundSolution = 0;
}

__device__ void maskGen(int * mask, curandState_t * state, int tid) {
    mask[0] = (curand(&state[tid])) % 3 + 1;
    mask[1] = (curand(&state[tid])) % 3 + 1;
    mask[2] = (curand(&state[tid])) % 3 + 1;
    mask[3] = (curand(&state[tid])) % 3 + 1;
}

// Crossover
// crossover(pool[tid], pool[tid-1], constants)
__device__ void crossover(individual & parent1, individual & parent2, curandState_t * state, int tid)
{
    // Generate a mask to decide which genes get crossed over
    int * mask = new int[4];
    maskGen(mask, state, tid);

    // Crossing over phi
    switch (mask[0]) 
    {
        // 1 - take from parent 1 (which is already in parent1)
        case (2) : // 2 - take from parent 2
            parent1.phi = parent2.phi;
            break;
        case (3) : // 3 - taking average
            parent1.phi = (parent1.phi + parent2.phi) / 2;
            break;
    }

    // Crossing over theta
    switch (mask[1])
    {
        case (2) : // 2 - take from parent 2
            parent1.theta = parent2.theta;
            break;
        case (3) : // 3 - taking average
            parent1.theta = (parent1.theta + parent2.theta) / 2;
            break;
    }

    // Crossing over V_nought
    switch (mask[2])
    {
        case (2) : // 2 - take from parent 2
            parent1.V_nought = parent2.V_nought;
            break;
        case (3) : // 3 - taking average
            parent1.V_nought = (parent1.V_nought + parent2.V_nought) / 2;
            break;
    }

    switch (mask[3])
    {
        case (2) : // 2 - take from parent 2
            parent1.time = parent2.time;
            break;
        case (3) : // 3 - taking average
            parent1.time = (parent1.time + parent2.time) / 2;
            break;
    }
    delete [] mask;
    // Crossover complete, if we were doing mutations we would start here
}
