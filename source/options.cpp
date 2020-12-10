#include "../headers/options.h"
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip> // setprecision
#include <math.h>

// Constructor that takes in a string to where to find the file to read
options::options(std::string filePath) {
    options::readFile(filePath);
}

// Called by constructor to read a file and assign variables to this struct
// Input: filePath - string to where the file is expected to be
// Output: this options object is assigned recognized variables its contents
void options::readFile(std::string filePath) {
    std::string line;
    std::ifstream configFile;
    configFile.open(filePath);

    if (configFile.is_open()) {
        int i = 0;
        while ( std::getline(configFile, line) && ++i) {
            if (line != "" && ( line.find("//") != 0 )) {
                int equals_pivot = line.find("="); // Get index to where the equal sign is
                int end_point = line.find_first_of(" "); // Get index to where the first space is (assuming no spaces from variable name through to end of assignment of value)
                // Get the variable and what it is assigning
                std::string variableName = line.substr(0, equals_pivot   );
                std::string variableValue = line.substr( equals_pivot + 1, end_point - equals_pivot - 1);

                if (variableName == "windSpeed") {
                    this->windSpeedmagnitude = std::stod(variableValue);
                }
                else if (variableName == "windDirection") {
                    this->windDirection = std::stod(variableValue);
                }
                else if (variableName == "atmosphericDensity") {
                    this->atmosphericDensity = std::stod(variableValue);
                }
                else if (variableName == "target_x") {
                    this->target_Loc.x = std::stod(variableValue);
                }
                else if (variableName == "target_y") {
                    this->target_Loc.y = std::stod(variableValue);
                }
                else if (variableName == "target_z") {
                    this->target_Loc.z = std::stod(variableValue);
                }
                else if (variableName == "projectile_mass") {
                    this->obj_mass = std::stod(variableValue);
                }
                else if (variableName == "distance_tol") {
                    this->distance_tol = std::stod(variableValue);
                }
                else if (variableName == "time_stepSize") {
                    this->time_stepSize = std::stod(variableValue);
                }
                else if (variableName == "obj_mass") {
                    this->obj_mass = std::stod(variableValue);
                }
                else if (variableName == "obj_dragCoeff") {
                    this->obj_dragCoeff = std::stod(variableValue);
                }
                else if (variableName == "obj_area") {
                    this->obj_area = std::stod(variableValue);
                }
                else if (variableName == "gravity") {
                    this->gravityAccel = std::stod(variableValue);
                }
                else if (variableName == "max_launch_v") {
                    this->max_launch_v = std::stod(variableValue);
                }
                else if (variableName == "min_launch_v") {
                    this->min_launch_v = std::stod(variableValue);
                }
                else if (variableName == "min_time") {
                    this->min_time = std::stod(variableValue);
                }
                else if (variableName == "max_time") {
                    this->max_time = std::stod(variableValue);
                }
                else if (variableName == "max_generations") {
                    this->max_generations = std::stod(variableValue);
                }
                else if (variableName == "useCUDA") {
                    if (variableValue == "true") {
                        this->useCUDA = true;
                    }
                    else {
                        this->useCUDA = false;
                    }
                }
                else if (variableName == "rng_seed") { // If the conifguration sets time_seed to NONE then time_seed is set to time(0) 
                    if (variableValue != "NONE") {
                        // If variableValue is not NONE or empty, assumption is that it is a valid double value that can be converted and used
                        this->rng_seed = std::stod(variableValue);
                    }
                    else {
                        this->rng_seed = time(0);
                    }
                }
                else if (variableName == "pop_size") {
                    this->pop_size = std::stoi(variableValue);
                }
                else {
                    // If none of the if cases were matches, then this is some unknown variable in the config file and output this to the terminal
                    std::cout << "Unknown variable '" + variableName + "' in line" << i << " in file " + filePath + "!\n";
                }
            }
        }
    }
    else {
        std::cout << "Unable to open " + filePath + " file!\n";
    }

    // Now got values, set derived ones now
    this->windcomponents.x = windSpeedmagnitude * cos(this->windDirection);
    this->windcomponents.y = windSpeedmagnitude * sin(this->windDirection);
    this->windcomponents.z = 0;
}

// Overloaded << operator for options struct
std::ostream& operator<<(std::ostream& os, const options object) {
    os << std::setprecision(6);
    os << "\n========Current Settings========\n";
    os << "Target Location Data\n";
    os << "\tX: " << object.target_Loc.x << " m\tY: " << object.target_Loc.y << " m\tZ: " << object.target_Loc.z << " m\n";
    os << "Environment Conditions\n";
    os << "\twindSpeed: " << object.windDirection << "m/s\twindDirection " << object.windDirection << " radians" << "\tAir Density: " << object.atmosphericDensity << "\n";
    os << "\tGravity: " << object.gravityAccel << "m/s^2\n";
    os << "\tWind Direction (X,Y,Z): " << object.windcomponents.x << " " << object.windcomponents.y << " " << object.windcomponents.z << "\n";
    os << "Projectile Characteristics\n";
    os << "\tMass: " << object.obj_mass << " kg" << "\tDrag: " << object.obj_dragCoeff << "\tCross-Sectional Area: " << object.obj_area << "\n";
    os << "\tTime Range: " << object.min_time << " - " << object.max_time << std::endl; 
    os << "\tV_nought Range: " << object.min_launch_v << " - " << object.max_launch_v << std::endl;
    os << "Simulation Settings\n";
    os << "\tTime Step Size: " << object.time_stepSize << " s\tDistance Tolerance: " << object.distance_tol << " m\n";
    os << "\tRNG Seed: " << object.rng_seed << std::endl;
    os << "\tPopulation Size: " << object.pop_size << "\tMax_generations: " << object.max_generations << std::endl;
    return os;
}