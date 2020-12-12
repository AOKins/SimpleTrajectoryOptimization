#include "../../headers/coords.cuh"
#include "../../headers/individual.cuh"
#include "../../headers/options.h"
// Includes for cuRAND library to access and call functions and use curandState
#include <curand.h>
#include <curand_kernel.h>

// Put genetics CUDA code here

// Resource for Odd-Even Transposition Sort - https://www.tutorialspoint.com/parallel_algorithm/parallel_algorithm_sorting.htm
// Sort array
template<typename T>
__device__ void sortArray(T * array, int size) {
    __shared__ bool sorted;
    sorted = false;
    int id = threadIdx.x;

    int leftID = (size + id-1) % size;
    int rightID = (size + id+1) % size;
    __syncthreads();
    //old = atomicCAS ( &addr, compare, value );  // old = *addr;  *addr = ((old == compare) ? value : old)
    //atomicCAS(&address, the value that you want to compare, the value use are using to compare to)
    /*
    int atomicCAS (int *id, self.cost, left.cost)

    { //make up keyword
        __lock (id)
        {
            int old = *id;
            *id = (old == self.cost) ? left.cost : old;
            return old;
        }
    }
    int atomicCAS (int *id, self.cost, right.cost)
    */
    int i = 1;
    while (!sorted && i <= 32*32)
    {        
        // Assume sorted until otherwise (a swap was performed)
        sorted = true;
        if (id > 0 && id < 31) {
            if (i % 2 == 0 && id % 2 == 0) {
                //look more into atomicCAS
                if (array[rightID] < array[id]) {
                    T temp = array[id];
                    array[id] = array[rightID];
                    array[rightID] = temp;
                    sorted = false;
                }
            }
            else {
                if (array[id] < array[leftID]) {
                    T temp = array[id];
                    array[id] = array[leftID];
                    array[leftID] = temp;
                    sorted = false;
                }
            }

            if (i % 2 == 1 && id % 2 == 1) {
                if (array[rightID] < array[id]) {
                    T temp = array[id];
                    array[id] = array[rightID];
                    array[rightID] = temp;
                    sorted = false;
                }
            }
            else {
                if (array[id] < array[leftID]) {
                    T temp = array[id];
                    array[id] = array[leftID];
                    array[leftID] = temp;
                    sorted = false;
                }

            }
            i++;    
        }
        __syncthreads();
    }
}



// Random Start
// Input: pool array containing individuals that need to be asssigned random parameters
//        constants - contains needed values
// Output: pool[tid] contains an individual with random launch parameters
__device__ void randomStart(individual *pool, options &constants, curandState * state, int tid)
{   // Generate a random number between 0 and 1111 then divide by 10 (to get a double value between 0 and 1)
    double rand_phi   = (double(curand(&state[tid]) % 1111) / 1110.0) * (2*PI); //phi randomly set between 0 and 2PI
    double rand_theta = (double(curand(&state[tid]) % 1111) / 1110.0) * (PI/2); //theta randomly set between 0 and PI/2
    double rand_V     = (double(curand(&state[tid]) % 1111) / 1110.0) * (constants.max_launch_v - constants.min_launch_v) + constants.min_launch_v; //V_naught randomly set between max and min allowed velocity range
    double rand_time  = (double(curand(&state[tid]) % 1111) / 1110.0) * (constants.max_time - constants.min_time) + constants.min_time; //time randomly set between max and min allowed time
    // Assign in pool new individual using randomly generated values
    pool[tid] = individual(rand_phi, rand_theta, rand_V, rand_time);
}

__global__ void initializeRandom(individual * pool, curandState_t *state, options *constants, int * foundSolution)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(constants->rng_seed, index, 0, &state[index]);
    randomStart(pool, *constants, state, index);
    *foundSolution = 0;
}

// Mask is pointer to array of 4
// Input: mask - pointer to int array of size 4
//        state - pointer to curandState_t array to use for random generation 
//        tid - int for index in the array
// Output: mask contains values randomly between 1 and 3 inclusive
__device__ void maskGenGPU(int * mask, curandState_t * state, int tid) {
    mask[0] = curand(&state[tid]) % 3 + 1;
    mask[1] = curand(&state[tid]) % 3 + 1;
    mask[2] = curand(&state[tid]) % 3 + 1;
    mask[3] = curand(&state[tid]) % 3 + 1;
}

// Crossover
// crossover(pool[tid], pool[tid-1], constants)
__device__ void crossover(individual & parent1, individual & parent2, curandState_t * state, int tid)
{
    // Generate a mask to decide which genes get crossed over
    int * mask = new int[4];
    
    maskGenGPU(mask, state, tid);
    // Crossing over phi
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
    }
    delete [] mask;
    // Crossover complete, if we were doing mutations we would start here
}

// Kernal to perform the genetic algorithm to derive a new generation
// Input: pool - individual array in global memory, assumed to not have a solution and is not ordered
//        constants - contains constant values to use such as pop_size, etc.
//        state - pointer array to be used in crossover for generating random numbers
// Output: pool contains 
__global__ void geneticAlgorithm(individual * pool, options * constants, curandState_t * state) {

    // Tid value for this thread in global memory
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    individual p1, p2;
    // Copy itself into a shared memory pool
    __shared__ individual * survivorPool; 
    survivorPool = new individual[32];
    __syncthreads();

    survivorPool[threadIdx.x] = pool[tid];
    __syncthreads();
    // Sort shared pool in the block by cost
    sortArray(survivorPool, 32);
    
    __syncthreads();
    // use best 2 individuals to crossover, results in p1
/*    if (tid == 0) {
        for (int i = 0; i < 32; i++) {
            printf("%i - %f\n", i, survivorPool[i].cost);
        }
    }*/
    if (survivorPool[0].cost < pool[tid].cost ) {
        p1 = pool[0];
        p2 = pool[1];
        crossover(p1, p2, state, tid);
        // store resulting new individual into global memory
        pool[tid] = p1;
    }
    __syncthreads();

}  
