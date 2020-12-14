#include "../physics.cu"
#include "genetics.cu"
// Includes for cuRAND library to access and use curandState to be used in genetic algorithm
#include <curand.h>
#include <curand_kernel.h>
#include "../../headers/output.h" // for calling output methods


// Kernal caller to manage memory and values needed before calling it
// Input: h_pool - pointer to individual array that holds the individual parameters needing to be computed with
//        h_constants - pointer to options struct that contains the constants needed related to the program
// Output: h_pool may contain individuals with valid solutions to hitting the target
__host__ void callGPU(individual * h_pool, options * h_constants) {
    // Get how many threads and blocks to use
    int numThreadsUsed = h_constants->num_threads_per;
    // Holds how many blocks to use for the kernal to cover the entire pool, assuming that pop_size is equal to num_blocks * numThreads
    int numBlocksUsed = h_constants->num_blocks;

    // Store the number of bytes the pool array is and use when managing memory for CUDA
    size_t poolMemSize = sizeof(individual)*h_constants->pop_size;

    // Allocate and copy over memory into the device
    individual * d_pool;
    cudaMalloc(&d_pool, poolMemSize);
    cudaMemcpy(d_pool, h_pool, poolMemSize, cudaMemcpyHostToDevice);

    individual * d_offset_temp;
    cudaMalloc(&d_offset_temp, poolMemSize);

    // Allocate memory for constants object
    options * d_constants;
    cudaMalloc(&d_constants, sizeof(options));
    cudaMemcpy(d_constants, h_constants, sizeof(options), cudaMemcpyHostToDevice);
    
    // Allocate curandState to use for random number generation in CUDA
    curandState_t *d_state;
    cudaMalloc(&d_state, sizeof(curandState_t)*h_constants->pop_size);

    // Allocate memory for integer object for determining if solution is found in a thread
    int * d_foundSolution;
    int * h_foundSolution = new int(0);
    cudaMalloc(&d_foundSolution, sizeof(int));
    cudaMemcpy(d_constants, h_foundSolution, sizeof(int), cudaMemcpyHostToDevice);

    // Initialize the random number generator into state
    initializeRandom<<<numBlocksUsed, numThreadsUsed>>>(d_pool, d_state, d_constants, d_foundSolution);
    cudaDeviceSynchronize();

    // At this point all initialization is finished
    int gen_count = 0;
    do {
        // Perform the algorithm
        simulateGPU<<<numBlocksUsed, numThreadsUsed>>>(d_constants, d_pool,  d_foundSolution);
        cudaDeviceSynchronize();
        // At this point all the simulations are finished including setting costs and found solution determined

        // Copy foundSolution to see if a solution was reached
        cudaMemcpy(h_foundSolution, d_foundSolution, sizeof(int), cudaMemcpyDeviceToHost);

        if (*h_foundSolution == 0) {            // No solution found yet, create new generation
            geneticAlgorithm<<<numBlocksUsed, numThreadsUsed>>>(d_pool, d_constants, d_state);
            cudaDeviceSynchronize();

            // Offset 16 to help diversify the pool, done by calling offsetCopy twice (offset 8 each) to ensure no race condition across all threads
            offsetCopy<<<numBlocksUsed, numThreadsUsed>>>(d_pool, d_offset_temp, d_constants);
            cudaDeviceSynchronize();
            offsetCopy<<<numBlocksUsed, numThreadsUsed>>>(d_offset_temp, d_pool, d_constants);
            cudaDeviceSynchronize();
        }

        gen_count++; // Increment gen_count for next generation
    } while (*h_foundSolution == 0 && gen_count < h_constants->max_generations); // continue loop until solution found or max generations reached
    // End of algorithm

    // Copy results of the pool into host memory
    cudaMemcpy(h_pool, d_pool, poolMemSize, cudaMemcpyDeviceToHost);

    // Free resources from device before ending function
    cudaFree(d_constants);
    cudaFree(d_pool);
    cudaFree(d_offset_temp);
    cudaFree(d_state);
    cudaFree(d_foundSolution);

    // Deallocate host memory
    delete h_foundSolution;
}
