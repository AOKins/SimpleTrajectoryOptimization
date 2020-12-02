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


    if (tid == 0) {
        printf("%f, %f", pool[0].time, constants->time_stepSize);
    }
    for (double c_time = 0; c_time < local_cpy.time; c_time += constants->time_stepSize) {
            updateGPU(*constants, local_cpy);
        
    }
    // Trajectory completed, evaluate cost
    local_cpy.determineCost(constants->target_Loc);
    pool[tid] = local_cpy;    
}


__host__ void callGPU(individual * h_pool, options * h_constants) {
    cudaDeviceProp * properties = new cudaDeviceProp;
    cudaGetDeviceProperties(properties,0);
    std::cout <<"GPU Properties (" << properties->name << " detected)\n";
    std::cout << "\tMaxThreadsPerBlock: " << properties->maxThreadsPerBlock << "\n"; 
    int numBlocksUsed = h_constants->pop_size / properties->maxThreadsPerBlock;
    std::cout << "\tBlocks being used: " << numBlocksUsed << "\n";

    size_t poolMemSize = sizeof(individual)*h_constants->pop_size;
    individual * d_pool;
    cudaMalloc(&d_pool, poolMemSize);
    cudaMemcpy(d_pool, h_pool, poolMemSize, cudaMemcpyHostToDevice);

    std::cout << "\n\tPool copied\n";

    options * d_constants;
    cudaMalloc(&d_constants, sizeof(options));
    cudaMemcpy(d_constants, h_constants, sizeof(options), cudaMemcpyHostToDevice);
    
    std::cout << "\tConstants copied\n";
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    
    cudaEventRecord(begin);
    simulateGPU <<<properties->maxThreadsPerBlock, numBlocksUsed>>> (d_constants, d_pool);
    cudaEventRecord(end);

    std::cout << "\tWaiting on GPU\n";
    cudaEventSynchronize(end);
    std::cout << "\tsimulateGPU ended\n";
    
    cudaMemcpy(h_pool, d_pool, poolMemSize, cudaMemcpyDeviceToHost);

    cudaFree(d_constants);
    cudaFree(d_pool);
    std::cout << "callGPU finished\n";
}