#include <iostream> // For cout
#include "../headers/options.h"
#include "../headers/individual.cuh"
#include "../headers/coords.cuh"
#include "cudaVersion/device.cu" // For callGPU()
#include "cpuVersion/geneticsCPU.cu" // for callCPU()
#include <chrono> // for clock() timing and CLOCKS_PER_SEC

int main(int argc, char *argv[]) { // main.exe input.config <- command to run program with file path to config file being "input.config"
    // Reading config file with path taken from command line
    options * config = new options(argv[1]);
    // Output onto the terminal
    std::cout << *config;
    // Allocate memory for the pool using the config's pop_size
    individual * pool = new individual[config->pop_size];
    float time = 0;
    clock_t start,end;
    // Determine whether to use cuda version of algorithm or cpu only verison
    if (config->useCUDA == true) {
        std::cout << "\n\t\tUSING CUDA\n";
        // Perform version that utilizes CUDA
        start = clock();
        callGPU(pool, config);
        end = clock();
    }
    else {        std::cout << "\n\tNOT\n";

        // Perform CPU only version
        start = clock();
        callCPU(config, pool);
        end = clock();
    }

    time = (float(end) - float(start)) / float(CLOCKS_PER_SEC) * 1000.0;

    std::cout << "Time it took was " << time << " milliseconds\n";
    // Output that the program is finished and deallocate memory
    std::cout <<"Exiting program...";
    delete config;
    delete [] pool;
}
