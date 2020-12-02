#ifndef _INDIVIDUAL_H_
#define _INDIVIDUAL_H_

#include "coords.cuh"
#include "options.h"
#include <math.h>
#include <random>

#define PI 3.14159265358979323846

// Struct that holds a single individual set of parameters and relevant data
struct individual {
    /////////////////////////////
    // The variable parameters //
    double phi; // x-y (in) plane angle - Radians
    double theta; // out of plane angle - Radians
    double V_nought; // m/s - initital velocity
    double time; // s - total time for the trajectory
    /////////////////////////////

    data3D position; // Initially 0,0,0 and is updated over time in the simulation
    data3D velocity; // Initially equal to components of V_nought in the 3 dimensions (using inplaneAngle and outplaneAngle to get values), updated over time in the simulation
    double cost; // Holds the cost (desirability) of the individual, smaller is better

    // Constructor that takes constants and rng generator to create a randomized individual (position still 0 0 0 and velocity based on phi/theta)
    individual(options constants, std::mt19937_64 & rngEng ) {
        this->phi = fmod(rngEng(), PI/2);
        this->theta = fmod(rngEng(), 2*PI);
        this->V_nought = fmod(rngEng(), constants.max_launch_v - constants.min_launch_v) + constants.min_launch_v;

        this->time = fmod(rngEng(), constants.max_time - constants.min_time) + constants.min_time;

        // Initial position is origin
        this->position = data3D(0,0,0);

        this->velocity.x = this->V_nought * sin(this->theta) * cos(this->phi);
        this->velocity.y = this->V_nought * sin(this->theta) * sin(this->phi);
        this->velocity.z = this->V_nought * cos(this->theta);
        this->cost = 999999999999; // Initial set to a really bad cost that needs to be updated when a simulated trajectory is complete
    }

    individual(double set_phi, double set_theta, double set_V, double set_time) {
        // For setting the angles, performing a mod on them to keep the values within set bounds
        // Without mod, individuals overtime could have large values that could be described by more readable/usable coterminal angles instead due to mutations and crossover
        // Phi is set within range of 0 (vertical) to 90 degrees (horizontal)
        this->phi = fmod(set_phi, PI/2);
        // Theta is set wtihin 0 to 2PI to allow complete 360 degrees of freedom
        this->theta = fmod(set_phi, 2*PI);
        this->V_nought = set_V;
        this->time = set_time;
        // Initial position is origin
        this->position = data3D(0,0,0);
        // Velocity is components of the set initial velocity
        this->velocity.x = this->V_nought * sin(this->theta) * cos(this->phi);
        this->velocity.y = this->V_nought * sin(this->theta) * sin(this->phi);
        this->velocity.z = this->V_nought * cos(this->theta);

        this->cost = 99999999.99; // Initial set to a really bad cost that needs to be updated when a simulated trajectory is complete
    }

    // Updates and returns this individual's cost
    __host__ __device__ double determineCost(data3D targetLoc) {
        this->cost = sqrt( pow(this->position.x-targetLoc.x , 2) + pow(this->position.y-targetLoc.y ,2) + pow(this->position.z-targetLoc.z ,2));
        return this->cost;
    }
};
// Need overloaded comparison operators

#endif