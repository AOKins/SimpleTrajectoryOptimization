#include <iostream>
#include <string>
#include <random>
#include "../headers/options.h"
#include "../headers/individual.cuh"
#include "../headers/coords.cuh"
#include "physicsCPU.cpp"
#include "device.cu"
#include "geneticsCPU.cu"



int main(int argc, char *argv[]) { // main.exe input.config <- command to run program with file path to config file being "input.config"

    options * config = new options(argv[1]);
    std::cout << *config;
    individual * pool = new individual[config->pop_size];
    if (config->useCUDA == true) {
        // Perform version that utilizes CUDA
        callGPU(pool, config);
    }
    else {
        // Perform CPU only version
        callCPU(config, pool);
    }

    std::cout <<"Exiting program...";
    delete config;
    delete [] pool;
}