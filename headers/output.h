#ifndef _OUTPUT_H_
#define _OUTPUT_H_
#include "individual.cuh"
#include "options.h"

void initializeRecording();

void recordGeneration(individual * pool, options * constants, int genCount);

void recordSolution(individual * pool, options * constants);

#include "../source/output.cpp"

#endif