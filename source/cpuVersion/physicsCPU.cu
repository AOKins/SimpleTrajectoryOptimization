#ifndef _PHYSICSCPU_CPP_
#define _PHYSICSCPU_CPP_

#include "../../headers/options.h"
#include "../../headers/coords.cuh"
#include "../../headers/individual.cuh"

// File that contains CPU version of physics simulation calculations to serve as baseline comparison before creating CUDA alternative implementation

// Returns 3D components of the acceleration on the object due to drag/wind imparted by the environment conditions
// Input: constants - get windcomponents, atmospheric density, object's cross-sectional area, and object's drag coefficient
//        objectPos - currently unused, but may have implementation to simulate change in gravity dependent on object's height
//        objectVel - used in getting the net speed that impacts the resulting forces
// Output: data3D object that contains the acceleration due to the atmosphere in all 3 components
data3D calculateAtmosphere(const options &constants, data3D objectPos, data3D objectVel) {
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

// Returns 3D components of the acceleration due to gravity on the object
// Most straightforward, but want to functionalize so that we can easily replace with more computationally intensive simulation
// Input: constants - get gravityAccel value
//        objectPos - currently unused, but may have implementation to simulate change in gravity dependent on object's height
// Output: data3D object that contains the acceleration due to gravity in all 3 components (though only y is expected to have non-zero value)
data3D calculateGravity(const options &constants, data3D objectPos) {
    data3D result;
    result.z = -constants.gravityAccel;
    return result;
}

// Changes an individual's position and velocity through one time step
// Input: constants - passed into calculateAtmosphere and calculateGravity as well as accessing stepSize
//        object - position and velocity members used and updated
// Output: object's position and velocity are changed according to current position and velocity and constants values
void update(const options& constants, individual * object) {
    data3D atm_accel = calculateAtmosphere(constants, object->position, object->velocity);
    data3D grav_accel = calculateGravity(constants, object->position);
    // Get the net acceleration being acted on the object
    data3D net_accel;
    net_accel.x = atm_accel.x + grav_accel.x;
    net_accel.y = atm_accel.y + grav_accel.y;
    net_accel.z = atm_accel.z + grav_accel.z;
    
    // New values is set to (current value) plus (rate of change) times (step size)
    object->position.x = object->position.x + object->velocity.x * constants.time_stepSize;
    object->position.y = object->position.y + object->velocity.y * constants.time_stepSize;
    object->position.z = object->position.z + object->velocity.z * constants.time_stepSize;

    object->velocity.x = object->velocity.x + net_accel.x*constants.time_stepSize;
    object->velocity.y = object->velocity.y + net_accel.y*constants.time_stepSize;
    object->velocity.z = object->velocity.z + net_accel.z*constants.time_stepSize;
}

void simulate(const options constants, individual * object) {
    object->position.x = 0;
    object->position.y = 0;
    object->position.z = 0;
    
    // Iterate for each time step until the total triptime is reached
    for (double c_time = 0; c_time <= object->time; c_time += constants.time_stepSize) {
        update(constants, object);
    }
    // Trajectory completed, evaluate cost
    object->determineCost(constants.target_Loc);
}


#endif