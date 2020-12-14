#ifndef _OPTIONS_H_
#define _OPTIONS_H_

#include "coords.cuh"
#include <string>

struct options {
    // Target Variables
    data3D target_Loc;
    double distance_tol; // How close the projectile must be near the target to succeed

    // Simulation variables
    double windSpeedmagnitude;
    double windDirection; // Described as an in-plane angle
    data3D windcomponents; // Using windSpeed and windDirection, determines the windSpeed along each axis to use in simulation
    double atmosphericDensity;
    double gravityAccel;
    double time_stepSize; // The change in time in seconds between each approximate change in position/velocity

    // Object variables
    double obj_dragCoeff;
    double obj_mass;
    double obj_area;

    // Ranges for initial velocity and time durations
    double min_launch_v;
    double max_launch_v;

    double min_time;
    double max_time;

    // CUDA settings
    int num_blocks;
    int num_threads_per; // should be 32 because of warp-size!
    bool useCUDA;

    // Genetic Algorithm related variables
    int pop_size;// Derived from num_blocks and num_threads_per
    double rng_seed;
    double max_generations;
    int display_freq;

    // Constructor that takes in strng path to file for reading 
    options(std::string filePath);

    // Reads and sets values to the object from file contents
    void readFile(std::string filePath);
};


std::ostream& operator<<(std::ostream& os, options object);

#include "../source/options.cpp"

#endif
