#include "../headers/coords.cuh"
#include "../headers/individual.cuh"
#include "../headers/options.h"


__device__ data3D calculateAtmosphereGPU( options &constants, data3D objectPos, data3D objectVel) {
    data3D result;
    data3D netSpeed;
    // Get the net/relative speed of the object which depends on wind
    netSpeed.x = (objectVel.x - constants.windcomponents.x);
    netSpeed.y = (objectVel.y - constants.windcomponents.y);
    netSpeed.z = (objectVel.z - constants.windcomponents.z);
    // Get resulting forces on the object using netSpeed
    result.x = 0.5 * constants.obj_dragCoeff * constants.atmosphericDensity * constants.obj_area * netSpeed.x*netSpeed.x / constants.obj_mass;
    result.y = 0.5 * constants.obj_dragCoeff * constants.atmosphericDensity * constants.obj_area * netSpeed.y*netSpeed.y / constants.obj_mass;
    result.z = 0.5 * constants.obj_dragCoeff * constants.atmosphericDensity * constants.obj_area * netSpeed.z*netSpeed.z / constants.obj_mass;
    
    return result;
}

__device__ data3D calculateGravityGPU(options &constants, data3D objectPos) {
    data3D result;
    result.z = -constants.gravityAccel;
    return result;
}

__device__ void updateGPU(options &constants, individual & object) {
    data3D atm_accel = calculateAtmosphereGPU(constants, object.position, object.velocity);
    data3D grav_accel = calculateGravityGPU(constants, object.position);
    // Get the net acceleration being acted on the object
    double net_accelX, net_accelY, net_accelZ;
    net_accelX = atm_accel.x + grav_accel.x;
    net_accelY = atm_accel.y + grav_accel.y;
    net_accelZ = atm_accel.z + grav_accel.z;
    
    // New values is set to (current value) plus (rate of change) times (step size)
    object.position.x = object.position.x + object.velocity.x * constants.time_stepSize;
    object.position.y = object.position.y + object.velocity.y * constants.time_stepSize;
    object.position.z = object.position.z + object.velocity.z * constants.time_stepSize;

    object.velocity.x = object.velocity.x + net_accelX*constants.time_stepSize;
    object.velocity.y = object.velocity.y + net_accelY*constants.time_stepSize;
    object.velocity.z = object.velocity.z + net_accelZ*constants.time_stepSize;
}


__global__ void simulateGPU(options * constants, individual * pool) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    individual local_cpy = pool[tid];
    // Iterate for each time step until the total triptime is reached

    for (double c_time = 0; c_time < local_cpy.time; c_time += constants->time_stepSize) {
            updateGPU(*constants, local_cpy);
        
    }
    // Trajectory completed, evaluate cost
    local_cpy.determineCost(constants->target_Loc);
    pool[tid] = local_cpy;    
}

// Kernal caller to manage memory and values needed before calling it
// Input: h_pool - pointer to individual array that holds the individual parameters needing to be computed with
//        h_constants - pointer to options struct that contains the constants needed related to the program
__host__ void callGPU(individual * h_pool, options * h_constants) {
    // Get properties of the gpu to display and also so we could use the maxThreadsPerBlock property
    cudaDeviceProp * properties = new cudaDeviceProp;
    cudaGetDeviceProperties(properties,0);
    std::cout <<"GPU Properties (" << properties->name << " detected)\n";
    std::cout << "\tMaxThreadsPerBlock: " << properties->maxThreadsPerBlock << "\n"; 
    // Holds how many blocks to use for the kernal to cover the entire pool, assuming that pop_size is divisible by maxThreadsPerBlock
    int numBlocksUsed = h_constants->pop_size / properties->maxThreadsPerBlock;
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
    
    // Create and use cudaEvents to sync with and record the outcome
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    
    cudaEventRecord(begin);
    simulateGPU <<<properties->maxThreadsPerBlock, numBlocksUsed>>> (d_constants, d_pool);
    cudaEventRecord(end);

    cudaEventSynchronize(end);

    // Copy results of the pool into host memory
    cudaMemcpy(h_pool, d_pool, poolMemSize, cudaMemcpyDeviceToHost);

    // Free resources from device before ending function
    cudaFree(d_constants);
    cudaFree(d_pool);
}