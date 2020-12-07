#include "../headers/coords.cuh"
#include "../headers/individual.cuh"
#include "../headers/options.h"
#include "physicsCUDA.cu"
#include "genetics.cu"
#include <curand.h>
#include <curand_kernel.h>


// Beating heart
// call randomStart() to fill population with random individuals
// begin loop
// if no solution, 
//      call simulateGPU for this thread
//      perform crossover                          //try to make this pass by reference
__global__ void geneticAlgorithm(individual *pool, options *constants, curandState_t *state)
{ 
    // Initially assume we do not have a solution
    bool foundSolution = false;
    int currentGen = 0;
    // Tid value for this thread
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int leftIndex = (constants->pop_size + tid-1) % constants->pop_size;
    int rightIndex = (constants->pop_size + tid+1) % constants->pop_size;
    // Set pool to contain initially randomly created individuals
    randomStart(pool, *constants, state, tid);

    individual self;
    individual left;
    individual right;
    
    do  {
        simulateGPU(constants, pool, tid);
        self = pool[tid];
        if (self.cost < constants->distance_tol) {
            foundSolution = true;
        }
        __syncthreads();
        if (foundSolution == false) {
            left = pool[leftIndex];
            right = pool[rightIndex];
            __syncthreads();//This is just insureance to make sure that the threads check each other in the right order
            
            // Now checking with neighbors to decide if we should crossover, preference to left (arbitrary)
            if (self.cost > left.cost)
            {
                crossover(&self, &left, *constants, state, tid);
            }
            else if (self.cost > right.cost)
            {
                crossover(&self, &right, *constants, state, tid);            
            }
        }
        //display the best of that generation
        /*
        for (int i = 0; i < currentGen; i++)
        {
            printf("Generation ", i, " best result was: ", pool[tid].cost);
        }
        */
        ++currentGen;
    } while (foundSolution == false && currentGen < constants->max_generations);
    //printf("tid %d : %f", tid, pool[tid].cost);
    //put a statement that states that says you found it
}  


// Kernal caller to manage memory and values needed before calling it
// Input: h_pool - pointer to individual array that holds the individual parameters needing to be computed with
//        h_constants - pointer to options struct that contains the constants needed related to the program
__host__ void callGPU(individual * h_pool, options * h_constants) {
    // Get properties of the gpu to display and also so we could use the maxThreadsPerBlock property
    cudaDeviceProp * properties = new cudaDeviceProp;
    cudaGetDeviceProperties(properties,0);
    std::cout <<"GPU Properties (" << properties->name << " detected)\n";
    int numThreadsUsed = properties->maxThreadsPerBlock;
    std::cout << "\tThreads used: " << numThreadsUsed << "\n"; 
    // Holds how many blocks to use for the kernal to cover the entire pool, assuming that pop_size is divisible by maxThreadsPerBlock
    int numBlocksUsed = h_constants->pop_size / numThreadsUsed;
    std::cout << "\tBlocks being used: " << numBlocksUsed << "\n";

    // Store the number of bytes the pool array is and use when managing memory for CUDA
    size_t poolMemSize = sizeof(individual)*h_constants->pop_size;

    // Allocate and copy over memory into the device
    individual * d_pool;
    cudaMalloc(&d_pool, poolMemSize);
    cudaMemcpy(d_pool, h_pool, poolMemSize, cudaMemcpyHostToDevice);

    options * d_constants;
    cudaMalloc(&d_constants, sizeof(options));
    cudaMemcpy(d_constants, h_constants, sizeof(options), cudaMemcpyHostToDevice);
    
    // Allocate curandState to use for random number generation in CUDA
    curandState *d_state;
    cudaMalloc(&d_state, h_constants->pop_size);

    // Create and use cudaEvents to sync with and record the outcome
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    
    cudaEventRecord(begin);
    // Initialize the random number generator into state
    initializeRandom<<<numThreadsUsed, numBlocksUsed>>>(d_state, d_constants);
    // Perform the algorithm
    geneticAlgorithm<<<numThreadsUsed, numBlocksUsed>>>(d_pool, d_constants, d_state);
    cudaEventRecord(end);

    cudaEventSynchronize(end);

    // Copy results of the pool into host memory
    cudaMemcpy(h_pool, d_pool, poolMemSize, cudaMemcpyDeviceToHost);

    // Free resources from device before ending function
    cudaFree(d_constants);
    cudaFree(d_pool);
    cudaFree(d_state);

    std::sort(h_pool, h_pool + h_constants->pop_size);
    std::cout << "All done!\t" << h_pool[0].cost << "\n";
}