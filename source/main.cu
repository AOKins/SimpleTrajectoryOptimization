#include <iostream> // For cout
#include "../headers/options.h"
#include "../headers/individual.cuh"
#include "../headers/coords.cuh"
#include "cudaVersion/device.cu" // For callGPU()
#include "cpuVersion/geneticsCPU.cu" // for callCPU()

int main(int argc, char *argv[]) { // main.exe input.config <- command to run program with file path to config file being "input.config"
    // Reading config file with path taken from command line
    options * config = new options(argv[1]);
    // Output onto the terminal
    std::cout << *config;
    // Allocate memory for the pool using the config's pop_size
    individual * pool = new individual[config->pop_size];

    // Determine whether to use cuda version of algorithm or cpu only verison
    if (config->useCUDA == true) {
        // Perform version that utilizes CUDA
        callGPU(pool, config);
    }
    else {
        // Perform CPU only version
        callCPU(config, pool);
    }

    // Output that the program is finished and deallocate memory
    std::cout <<"Exiting program...";
    delete config;
    delete [] pool;
}
