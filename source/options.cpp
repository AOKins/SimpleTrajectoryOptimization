#include "../headers/options.h"
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip> // setprecision
#include <math.h>

// Constructor that takes in a string to where to find the file to read
// Input is string to where the file is to read from
// Output is options struct is initialized
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
        // Read each line
        while ( std::getline(configFile, line) && ++i) {
            // If not an empty line and not a commented out line
            if (line != "" && ( line.find("//") != 0 )) {
                int equals_pivot = line.find("="); // Get index to where the equal sign is
                int end_point = line.find_first_of(" "); // Get index to where the first space is (assuming no spaces from variable name through to end of assignment of value)
                // Get the variable and what it is assigning
                std::string variableName = line.substr(0, equals_pivot   );
                std::string variableValue = line.substr( equals_pivot + 1, end_point - equals_pivot - 1);

                // Determine appriopriate variable based on the variableName and use appriopriate conversion function
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
                else if (variableName == "display_freq") {
                    this->display_freq = std::stoi(variableValue);
                }
                else if (variableName == "useCUDA") {
                    if (variableValue == "true") {
                        this->useCUDA = true;
                    }
                    else { // if variable value is anything other than true, then assume false
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
                else if (variableName == "num_threads_per") {
                    this->num_threads_per = std::stoi(variableValue);
                }
                else if (variableName == "num_blocks") {
                    this->num_blocks = std::stoi(variableValue);
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

    // Determine derived values
    this->pop_size = this->num_threads_per*this->num_blocks;
    // Now got values, set derived ones now for wind in 3D components as opposed to direction/magnitude
    this->windcomponents.x = windSpeedmagnitude * cos(this->windDirection);
    this->windcomponents.y = windSpeedmagnitude * sin(this->windDirection);
    this->windcomponents.z = 0;
}

// Overloaded << operator for options struct
std::ostream& operator<<(std::ostream& os, const options object) {
    os << std::setprecision(6);
    os << "\n========Current Settings========\n";
    os << "Target Location Data\n";
    os << "\tX: " << object.target_Loc.x << "m\n\tY: " << object.target_Loc.y << "m\n\tZ: " << object.target_Loc.z << "m\n";
    os << "Environment Conditions\n";
    os << "\twindSpeed: " << object.windDirection << "m/s\n\twindDirection " << object.windDirection << " rads" << "\n\tAir Density: " << object.atmosphericDensity << "kg/m^3\n";
    os << "\tGravity: " << object.gravityAccel << "m/s^2\n";
    os << "Projectile Characteristics\n";
    os << "\tMass: " << object.obj_mass << " kg" << "\n\tDrag: " << object.obj_dragCoeff << "\n\tCross-Sectional Area: " << object.obj_area << "m^2\n";
    os << "\tTime Range (s): " << object.min_time << " - " << object.max_time << std::endl; 
    os << "\tV_nought Range (m/s): " << object.min_launch_v << " - " << object.max_launch_v << std::endl;
    os << "Simulation Settings\n";
    os << "\tTime Step Size: " << object.time_stepSize << "s\n\tDistance Tolerance: " << object.distance_tol << "m\n";
    os << "\tRNG Seed: " << object.rng_seed << std::endl;
    os << "\tPopulation Size: " << object.pop_size << "\n\tMax_generations: " << object.max_generations << std::endl;
    os << "CUDA Settings\n";
    os << "\tThreads Per Block: " << object.num_threads_per << "\n\tBlocks: " << object.num_blocks << std::endl;

    return os;
}