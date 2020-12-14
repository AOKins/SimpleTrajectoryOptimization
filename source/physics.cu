#ifndef _PHYSICS_CPP_
#define _PHYSICS_CPP_
// Functions that handle simulating trajectory on device called from kernal

// Determine the acceleration due the atmospher on the obejct due to drag and wind
// Input: constants for physics contants
//        objectPos for current position of the object (currently unused but would be for possibly more complex atmospher simulations)
//        objectVel for current velocity used in deriving the force due to drag/wind that depends on the objects velocity
// Output: returns resulting acceleration in 3D coordinates system
__host__ __device__ data3D calculateAtmosphere( options &constants, data3D objectPos, data3D objectVel) {
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

// Determine the acceleration due to gravity on the obejct (currently just constant)
// Input: constants for physics contants
//        objectPos for current position of the object (currently unused but would be used for altitude consideration)
// Output: returns resulting acceleration in 3D coordinates system
__host__ __device__ data3D calculateGravity(options &constants, data3D objectPos) {
    data3D result; // default constructor sets all 3 components initially to 0
    result.z = -constants.gravityAccel;
    return result;
}

// Perform a step of the simulation
// Input: constants - access to physics constants needed and step-size
// Output: object is updated in its simulation by one step size
__host__ __device__ void update(options &constants, individual & object) {
    data3D atm_accel = calculateAtmosphere(constants, object.position, object.velocity);
    data3D grav_accel = calculateGravity(constants, object.position);
    // Get the net acceleration being acted on the object
    double net_accelX, net_accelY, net_accelZ;
    net_accelX = atm_accel.x + grav_accel.x;
    net_accelY = atm_accel.y + grav_accel.y;
    net_accelZ = atm_accel.z + grav_accel.z;
    
    // New values is set to (current value) plus (rate of change) times (step size)
    // Update position
    object.position.x = object.position.x + object.velocity.x * constants.time_stepSize;
    object.position.y = object.position.y + object.velocity.y * constants.time_stepSize;
    object.position.z = object.position.z + object.velocity.z * constants.time_stepSize;
    // Update velocity
    object.velocity.x = object.velocity.x + net_accelX*constants.time_stepSize;
    object.velocity.y = object.velocity.y + net_accelY*constants.time_stepSize;
    object.velocity.z = object.velocity.z + net_accelZ*constants.time_stepSize;
}

// Kernal for performaing simulate across all individuals
// Input: constants - contains constant values needed such as pop_size or physics properties
//        pool - array of individuals to be simulated
//        foundSolution - integer that indicates valid solutions, assumed to be 0
// Output: pool[tid] has cost associated with the parameters
__global__ void simulateGPU(options * constants, individual *pool, int *foundSolution) {
    // Derive id to access appriopriate individual and copy into local memory
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    individual lcl_ind = pool[tid];
    // Also copy constants to local memory
    options lcl_constants = *constants;
    // Reset the initial position (or atleast be sure) to 0,0,0
    lcl_ind.position.x = 0;
    lcl_ind.position.y = 0;
    lcl_ind.position.z = 0;

    // Iterate for each time step until the total triptime is reached/exceded
    for (double c_time = 0; c_time < lcl_ind.time; c_time += lcl_constants.time_stepSize) {
        update(lcl_constants, lcl_ind);
    }
    // Trajectory completed, evaluate cost
    lcl_ind.determineCost(lcl_constants.target_Loc);

    // Store resulting cost to global individual and also set foundSolution to 1 if this individual is a valid solution
    pool[tid].cost = lcl_ind.cost;
    if (lcl_ind.cost < lcl_constants.distance_tol) {
        (*foundSolution) = 1;
    }
}


// Simulate a trajectory using a given object to determine how close it is to the target
// Input: constants - contains needed values such as step size and target location
//        object - the individual that contains the parameters to simulate the trajector (angles, V_nought, and total trip time)
// Output: object contains cost for how close it is to the target and final position starting from 0,0,0 
__host__ void simulate(options constants, individual * object) {
    // Reset the initial position (or atleast be sure) to 0,0,0
    object->position.x = 0;
    object->position.y = 0;
    object->position.z = 0;
    
    // Iterate for each time step until the total triptime is reached
    for (double c_time = 0; c_time <= object->time; c_time += constants.time_stepSize) {
        update(constants, *object);
    }
    // Trajectory completed, evaluate cost
    object->determineCost(constants.target_Loc);
}

#endif
