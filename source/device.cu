#include "../headers/coords.cuh"
#include "../headers/individual.cuh"
#include "../headers/options.h"
#include "physicsCUDA.cu"
#include "genetics.cu"
#include <curand.h>
#include <curand_kernel.h>


__global__ void geneticAlgorithm(individual *pool, options *constants, curandState_t *state)
{ 
    // Initially assume we do not have a solution
    // Tid value for this thread
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int leftIndex = (constants->pop_size + tid-1) % constants->pop_size;
    int rightIndex = (constants->pop_size + tid+1) % constants->pop_size;

    // Local holding variables to reduce trips to global memory
    individual self, left, right;
    
    // Copy into local memory
    self = pool[tid];
    left = pool[leftIndex];
    right = pool[rightIndex];
    
    // Now checking with neighbors to decide if we should crossover, preference to left (arbitrary)
    if (self.cost > left.cost)
    {
        crossover(self, left, state, tid);
        pool[tid] = self;
    }
    else if (self.cost > right.cost)
    {
        crossover(self, right, state, tid);
        pool[tid] = self;
    }

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

    int * d_foundSolution;
    int * h_foundSolution = new int(0);
    cudaMalloc(&d_foundSolution, sizeof(int));
    cudaMemcpy(d_constants, h_foundSolution, sizeof(int), cudaMemcpyHostToDevice);

    // Create and use cudaEvents to sync with and record the outcome
    cudaEvent_t initializeStart, startSimulate;
    cudaEvent_t endSimulation, endGenetics;
    cudaEventCreate(&initializeStart);
    cudaEventCreate(&startSimulate);
    cudaEventCreate(&endSimulation);
    cudaEventCreate(&endGenetics);
    
    cudaEventRecord(initializeStart);
    // Initialize the random number generator into state
    initializeRandom<<<numThreadsUsed, numBlocksUsed>>>(d_pool, d_state, d_constants, d_foundSolution);
    cudaEventRecord(startSimulate);
    cudaEventSynchronize(startSimulate);
    // At this point all initialization is finished

    int gen_count = 0;
    do {
        // Perform the algorithm
        simulateGPU<<<numThreadsUsed, numBlocksUsed>>>(d_constants, d_pool,  d_foundSolution);
        cudaEventRecord(endSimulation);
        cudaEventSynchronize(endSimulation);

        // At this point all the simulations are finished including setting costs and found solution determined
        // Copy this boolean to see if a solution was reached
        cudaMemcpy(h_foundSolution, d_foundSolution, sizeof(int), cudaMemcpyDeviceToHost);
        if (*h_foundSolution == 0) {
            // No solution found yet, create new generation
            geneticAlgorithm<<<numThreadsUsed, numBlocksUsed>>>(d_pool, d_constants, d_state);
            cudaEventRecord(endGenetics);
            cudaEventSynchronize(endGenetics);
        }
        gen_count++;
        // continue loop until solution found or max generations reached
    } while (*h_foundSolution == 0 && gen_count < h_constants->max_generations);
    std::cout <<"Final " << *h_foundSolution << "-";

    // Copy results of the pool into host memory
    cudaMemcpy(h_pool, d_pool, poolMemSize, cudaMemcpyDeviceToHost);

    // Free resources from device before ending function
    cudaFree(d_constants);
    cudaFree(d_pool);
    cudaFree(d_state);
    cudaFree(d_foundSolution);
    cudaEventDestroy(endSimulation);
    delete h_foundSolution;

    // Temp debugging output onto terminal to see rough results of the algorithm
    std::sort(h_pool, h_pool + h_constants->pop_size);
    std::cout << "All done!\t" << h_pool[0].cost << "\n";
}