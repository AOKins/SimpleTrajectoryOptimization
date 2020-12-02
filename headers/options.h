#ifndef _OPTIONS_H_
#define _OPTIONS_H_

#include "coords.cuh"
#include <string>

struct options {
    data3D target_Loc;
    double distance_tol; // How close the projectile must be near the target to succeed
    double time_stepSize; // The change in time in seconds between each approximate change in position/velocity
    double windSpeedmagnitude;
    double windDirection; // Described as an in-plane angle
    data3D windcomponents; // Using windSpeed and windDirection, determines the windSpeed along each axis to use in simulation
    double atmosphericDensity;
    double gravityAccel;
    double obj_dragCoeff;
    double obj_mass;
    double obj_area;
    double rng_seed;

    double min_launch_v;
    double max_launch_v;

    double min_time;
    double max_time;

    int pop_size;

    double max_generations;

    options(std::string filePath);
    void readFile(std::string filePath);
};


std::ostream& operator<<(std::ostream& os, options object);

#include "../source/options.cpp"

#endif
