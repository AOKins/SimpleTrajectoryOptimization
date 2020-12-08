
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
    result.x = 0;
    result.y = 0;
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


__global__ void simulateGPU(options * constants, individual *pool, int *foundSolution) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    individual lcl_ind = pool[tid];
    options lcl_constants = *constants;

    // Iterate for each time step until the total triptime is reached
    for (double c_time = 0; c_time < lcl_ind.time; c_time += lcl_constants.time_stepSize) {
            updateGPU(lcl_constants, lcl_ind);
    }
    // Trajectory completed, evaluate cost
    lcl_ind.determineCost(lcl_constants.target_Loc);

    pool[tid].cost = lcl_ind.cost;
    if (lcl_ind.cost < lcl_constants.distance_tol) {
        (*foundSolution) = 1;
    }
}
