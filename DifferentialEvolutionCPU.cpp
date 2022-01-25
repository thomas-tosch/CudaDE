/* Copyright 2017 Ian Rankin
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the "Software"), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
 * to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

//
//  DifferentialEvolution.cpp
//
// This class is a wrapper to make calls to the cuda differential evolution code easier to work with.
// It handles all of the internal memory allocation for the differential evolution and holds them
// as device memory for the GPU
//
// Example wrapper usage:
//
// float mins[3] = {0,-1,-3.14};
// float maxs[3] = {10,1,3.13};
//
// DifferentialEvolution minimizer(64,100, 3, 0.9, 0.5, mins, maxs);
//
// minimizer.fmin(NULL);
//
//////////////////////////////////////////////////////////////////////////////////////////////
// However if needed to pass arguements then an example usage is:
//
// // create the min and max bounds for the search space.
// float minBounds[2] = {-50, -50};
// float maxBounds[2] = {100, 200};
//
// // a random array or data that gets passed to the cost function.
// float arr[3] = {2.5, 2.6, 2.7};
//
// // data that is created in host, then copied to a device version for use with the cost function.
// struct data x;
// struct data *d_x;
// gpuErrorCheck(cudaMalloc(&x.arr, sizeof(float) * 3));
// unsigned long size = sizeof(struct data);
// gpuErrorCheck(cudaMalloc((void **)&d_x, size));
// x.v = 3;
// x.dim = 2;
// gpuErrorCheck(cudaMemcpy(x.arr, (void *)&arr, sizeof(float) * 3, cudaMemcpyHostToDevice));
//
// // Create the minimizer with a popsize of 192, 50 generations, Dimensions = 2, CR = 0.9, F = 2
// DifferentialEvolution minimizer(192,50, 2, 0.9, 0.5, minBounds, maxBounds);
//
// gpuErrorCheck(cudaMemcpy(d_x, (void *)&x, sizeof(struct data), cudaMemcpyHostToDevice));
//
// // get the result from the minimizer
// std::vector<float> result = minimizer.fmin(d_x);
//

#include "DifferentialEvolutionCPU.hpp"
#include "DifferentialEvolutionCPU.h"

// Constructor for DifferentialEvolution
//
// @param PopulationSize - the number of agents the DE solver uses.
// @param NumGenerations - the number of generation the differential evolution solver uses.
// @param Dimensions - the number of dimesnions for the solution.
// @param crossoverConstant - the number of mutants allowed pass each generation CR in
//              literature given in the range [0,1]
// @param mutantConstant - the scale on mutant changes (F in literature) given [0,2]
//              default = 0.5
// @param func - the cost function to minimize.
DifferentialEvolutionCPU::DifferentialEvolutionCPU(int PopulationSize, int NumGenerations,
                                             int Dimensions, float crossoverConstant, float mutantConstant,
                                             float *minBounds, float *maxBounds)
{
    popSize = PopulationSize;
    dim = Dimensions;
    numGenerations = NumGenerations;
    CR = crossoverConstant;
    F = mutantConstant;
    cudaError_t ret;

    ret = cudaMalloc(&d_target1, sizeof(float) * popSize * dim);
    gpuErrorCheck(ret);
    ret = cudaMalloc(&d_target2, sizeof(float) * popSize * dim);
    gpuErrorCheck(ret);
    ret = cudaMalloc(&d_mutant, sizeof(float) * popSize * dim);
    gpuErrorCheck(ret);
    ret = cudaMalloc(&d_trial, sizeof(float) * popSize * dim);
    gpuErrorCheck(ret);

    ret = cudaMalloc(&d_cost, sizeof(float) * PopulationSize);
    gpuErrorCheck(ret);

    ret = cudaMalloc(&d_min, sizeof(float) * dim);
    gpuErrorCheck(ret);
    ret = cudaMalloc(&d_max, sizeof(float) * dim);
    gpuErrorCheck(ret);
    ret = cudaMemcpy(d_min, minBounds, sizeof(float) * dim, cudaMemcpyHostToDevice);
    gpuErrorCheck(ret);
    ret = cudaMemcpy(d_max, maxBounds, sizeof(float) * dim, cudaMemcpyHostToDevice);
    gpuErrorCheck(ret);

    h_cost = new float[popSize * dim];
    d_randStates = createRandNumGen(popSize);
}

// fmin
// wrapper to the cuda function C function for differential evolution.
// @param args - this a pointer to arguments for the cost function.
//      This MUST point to device memory or NULL.
//
// @return the best set of parameters
float* DifferentialEvolutionCPU::fmin(void *args)
{
    std::vector<float> result(dim);

    //differentialEvolutionCPU(d_target1, d_trial, d_cost, d_target2, d_min,
      //                    d_max, h_cost, d_randStates, dim, popSize, numGenerations, CR, F, args,
        //                  result.data());

    return h_cost;
}

/* Copyright 2017 Ian Rankin
*
* Permission is hereby granted, free of charge, to any person obtaining a copy of this
* software and associated documentation files (the "Software"), to deal in the Software
* without restriction, including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
* to whom the Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all copies or
* substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

// DifferentialEvolutionGPU.cu
// This file holds the GPU kernel functions required to run differential evolution.
// The software in this files is based on the paper:
// Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continous Spaces,
// Rainer Storn, Kenneth Price (1996)
//
// But is extended upon for use with GPU's for faster computation times.
// This has been done previously in the paper:
// Differential evolution algorithm on the GPU with C-CUDA
// Lucas de P. Veronese, Renato A. Krohling (2010)
// However this implementation is only vaguly based on their implementation.
// Translation: I saw that the paper existed, and figured that they probably
// implemented the code in a similar way to how I was going to implement it.
// Brief read-through seemed to be the same way.
//
// The paralization in this software is done by using multiple cuda threads for each
// agent in the algorithm. If using smaller population sizes, (4 - 31) this will probably
// not give significant if any performance gains. However large population sizes are more
// likly to give performance gains.
//
// HOW TO USE:
// To implement a new cost function write the cost function in DifferentialEvolutionGPU.cu with the header
// __device float fooCost(const float *vec, const void *args)
// @param vec - sample parameters for the cost function to give a score on.
// @param args - any set of arguements that can be passed at the minimization stage
// NOTE: args any memory given to the function must already be in device memory.
//
// Go to the header and add a specifier for your cost functiona and change the COST_SELECTOR
// to that specifier. (please increment from previous number)
//
// Once you have a cost function find the costFunc function, and add into
// preprocessor directives switch statement
//
// ...
// #elif COST_SELECTOR == YOUR_COST_FUNCTION_SPECIFIER
//      return yourCostFunctionName(vec, args);
// ...
//

// for FLT_MAX
#include <cfloat>

#include <iostream>
#include <random>

// for clock()
#include <ctime>
#include <cmath>

// basic function for exiting code on CUDA errors.
// Does no special error handling, just exits the program if it finds any errors and gives an error message.
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

float sphere(const float *vec, const void *args)
{
    const struct dataCPU *a = (struct dataCPU *)args;

    float sum = 0;
    for (int i = 0; i < a->dim; i++) {
        sum += (vec[i] * vec[i]);
    }
    // -450
    //return sum - 450;
    return sum;
}

float rosenbrock(const float *vec, const void *args)
{
    const struct dataCPU *a = (struct dataCPU *)args;

    float sum = 0;
    for (int i = 0; i < a->dim - 1; i++) {
        sum += ((100 * powf(vec[i+1] - powf(vec[i], 2), 2)) + powf(1 - vec[i], 2));
    }
    // +390
    //return sum + 390;
    return sum;
}

float schwefel(const float *vec, const void *args)
{
    const struct dataCPU *a = (struct dataCPU *)args;

    float sum = 0;
    for (int j = 0; j < a->dim; j++) {
        float sum2 = 0;
        for (int i = 0; i < j; i++) {
            sum2 += vec[i];
        }
        sum += pow(sum2, 2);
    }
    return sum;
}

float quatric(const float *vec, const void *args)
{
    curandState_t state;
    curand_init(clock(), /* the seed controls the sequence of random values that are produced */
                0, /* the sequence number is only important with multiple cores */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &state);
    const struct dataCPU *a = (struct dataCPU *)args;
    float sum = 0;
    for (int i = 0; i < a->dim; i++) {
        sum += i * pow(vec[i], 4) + curand(&state) % 1;
    }
    return sum;
}

float ackley(const float *vec, const void *args)
{
    const struct dataCPU *a = (struct dataCPU *)args;
    float sum = 0;
    for (int i = 0; i < a->dim; i++) {
        sum += pow(vec[i], 2);
    }
    float sum2 = 0;
    for (int i = 0; i < a->dim; i++) {
        sum2 += cos(2 * M_PI * vec[i]);
    }
    return 20 + expf(1) - 20 * expf(-0.2 * sqrt((1 / a->dim) * sum)) - expf((1 / a->dim) * sum2);
}

float griewank(const float *vec, const void *args)
{
    const struct dataCPU *a = (struct dataCPU *)args;
    float sum = 0;
    for (int i = 0; i < a->dim; i++) {
        sum += (pow(vec[i], 2) / 4000);
    }
    float mult = 1;
    for (int i = 1; i < a->dim + 1; i++) {
        mult *= (cos(vec[i - 1] / sqrtf(i)));
    }
    // -180
    //return (sum - mult + 1) - 180;
    return (sum - mult + 1);
}

float rastrigin(const float *vec, const void *args)
{
    const struct dataCPU *a = (struct dataCPU *)args;

    float sum = 0;
    for (int i = 0; i < a->dim; i++) {
        sum += (pow(vec[i], 2) - 10 * cos(2 * M_PI * vec[i]));
    }
    //return (10 * a->dim + sum) - 330;
    return (10 * a->dim + sum);
}

float schwefelFunc(const float *vec, const void *args)
{
    const struct dataCPU *a = (struct dataCPU *)args;

    float sum = 0;
    for (int i = 0; i < a->dim; i++) {
        sum += vec[i] * sin(sqrt((float)abs(vec[i])));
    }
    return 418.9829 * a->dim - sum;
}

float salomon(const float *vec, const void *args)
{
    const struct dataCPU *a = (struct dataCPU *)args;

    float sum = 0;
    for (int i = 0; i < a->dim; i++) {
        sum += pow(vec[i], 2);
    }

    float sum2 = 0;
    for (int i = 0; i < a->dim; i++) {
        sum2 += pow(vec[i], 2) + 1;
    }

    return -cos(2*M_PI*sqrt((float)sum)) + 0.1 * sqrt((float) sum2);
}

float whitely(const float *vec, const void *args)
{
    const struct dataCPU *a = (struct dataCPU *)args;
    float sum = 0;
    float total = 0;
    float y = 0;
    for (int j = 0; j < a->dim; j++) {
        sum = 0;
        for (int i = 0; i <= j; i++) {
            y = 100 * pow(vec[j] - pow(vec[i], 2), 2) + pow(1 - vec[i], 2);
            sum += ((y / 4000) - cos(y) + 1);
        }
        total += sum;
    }
    return total;
}

float w(float x, float a, float b, float m)
{
    float sum = 0;
    for (int k = 0; k <= m ; k++) {
        sum += pow(a, k) * cos(2*M_PI*pow(b,k)*(x+0.5));
    }
    return sum;
}

float weierstrass(const float *vec, const void *args)
{
    const struct dataCPU *a = (struct dataCPU *)args;
    float sum = 0;
    for (int i = 0; i < a->dim; i++) {
        sum += w(vec[i], 0.5, 3, 20) - a->dim * w(0, 0.5, 3, 20);
    }
    return sum;
}

float costFunc(const float *vec, const void *args) {
    const struct dataCPU *a = (struct dataCPU *)args;
    if (a->costFun == SPHERE)
    { return sphere(vec, args); }
    else if (a->costFun == ROSENBROCK)
    { return rosenbrock(vec, args); }
    else if (a->costFun == SCHWEFEL)
    { return schwefel(vec, args); }
    else if (a->costFun == QUATRIC)
    { return quatric(vec, args); }
    else if (a->costFun == ACKLEY)
    { return ackley(vec, args); }
    else if (a->costFun == GRIEWANK)
    { return griewank(vec, args); }
    else if (a->costFun == RASTRIGIN)
    { return rastrigin(vec, args); }
    else if (a->costFun == SCHWEFELFUNC)
    { return schwefelFunc(vec, args); }
    else if (a->costFun == SALOMON)
    { return salomon(vec, args); }
    else if (a->costFun == WHITELY)
    { return whitely(vec, args); }
    else if (a->costFun == WEIERSTRASS)
    { return weierstrass(vec, args); }
    return 0;
}

void printCudaVector(float *d_vec, int size)
{
    float *h_vec = new float[size];
    gpuErrorCheck(cudaMemcpy(h_vec, d_vec, sizeof(float) * size, cudaMemcpyDeviceToHost));

    std::cout << "{";
    for (int i = 0; i < size; i++) {
        std::cout << h_vec[i] << ", ";
    }
    std::cout << "}" << std::endl;

    delete[] h_vec;
}

void generateRandomVectorAndInit(float *d_x, float *d_min, float *d_max,
                                            float *d_cost, void *costArgs,
                                            int popSize, int dim, unsigned long seed, int idx)
{
    int idx = idx - 1;
    if (idx >= popSize) return;

    std::mt19937 rng(seed);
    for (int i = 0; i < dim; i++) {
        d_x[(idx*dim) + i] = (rng() * (d_max[0] - d_min[0])) + d_min[0];
    }

    d_cost[idx] = costFunc(&d_x[idx*dim], costArgs);
}

void evolutionKernel(float *d_target,
                                float *d_trial,
                                float *d_cost,
                                float *d_target2,
                                float *d_min,
                                float *d_max,
                                mt19937 rng,
                                int dim,
                                int popSize,
                                int CR, // Must be given as value between [0,999]
                                float F,
                                void *costArgs,
                                int idx)
{
    int idx = idx - 1;
    if (idx >= popSize) return; // stop executing this block if
    // all populations have been used

    // TODO: Better way of generating unique random numbers?
    int a;
    int b;
    int c;
    int d;
    int e;
    int j;
    //////////////////// Random index mutation generation //////////////////
    // select a different random number then index
    do { a = rng() % popSize; } while (a == idx);
    do { b = rng() % popSize; } while (b == idx || b == a);
    do { c = rng() % popSize; } while (c == idx || c == a || c == b);
    do { d = rng() % popSize; } while (d == idx || d == a || d == b || d == c);
    do { e = rng() % popSize; } while (e == idx || e == a || e == b || e == c || e == d);
    j = rng() % dim;

    float F1 = 0.25;
    float F2 = 0.25;
    float F3 = 0.2;
    float F4 = 0.2;

    float best = FLT_MAX;
    int bestIdx = 0;
    for (int i = 0; i < popSize; i++) {
        if (d_cost[i] < best) {
            best = d_cost[i];
            bestIdx = i;
        }
    }

    ///////////////////// MUTATION ////////////////
    for (int k = 1; k <= dim; k++) {
        //printf("%f", curand(state));
        if (rng() < CR || k==dim) {
            // trial vector param comes from vector plus weighted differential
            d_trial[(idx*dim)+j] = d_target[(idx*dim)+j] + (F1 * (d_target[(bestIdx*dim)+j] - d_target[(idx*dim)+j]))
                                   + (F2 * (d_target[(a*dim)+j] - d_target[(idx*dim)+j])) + (F3 * (d_target[(b*dim)+j] - d_target[(c*dim)+j]))
                                   + (F4 * (d_target[(d*dim)+j] - d_target[(e*dim)+j]));
            if(d_trial[(idx*dim)+j] < d_min[0] || d_trial[(idx*dim)+j] > d_max[0] ){
                //printf("out of bounds\n");
                curand_init(clock(), idx,0,state);
                d_trial[(idx*dim)+j] = (curand_uniform(state) * (d_max[0] - d_min[0])) + d_min[0];
            }
        } else {
            d_trial[(idx*dim)+j] = d_target[(idx*dim)+j];
        } // end if else for creating trial vector
        j = (j+1) % dim;
    } // end for loop through parameters
    float score = 0;
    score = costFunc(&d_trial[idx*dim], costArgs);
    if (score < d_cost[idx]) {
        // copy trial into new vector
        for (j = 0; j < dim; j++) {
            d_target2[(idx*dim) + j] = d_trial[(idx*dim) + j];
            //printf("idx = %d, d_target2[%d] = %f, score = %f\n", idx, (idx*dim)+j, d_trial[(idx*dim) + j], score);
        }
        d_cost[idx] = score;
    } else {
        // copy target to the second vector
        for (j = 0; j < dim; j++) {
            d_target2[(idx*dim) + j] = d_target[(idx*dim) + j];
            //printf("idx = %d, d_target2[%d] = %f, score = %f\n", idx, (idx*dim)+j, d_trial[(idx*dim) + j], score);
        }
    }
} // end differentialEvolution function.

void differentialEvolutionCPU(float *d_target,
                           float *d_trial,
                           float *d_cost,
                           float *d_target2,
                           float *d_min,
                           float *d_max,
                           float *h_cost,
                           int dim,
                           int popSize,
                           int maxGenerations,
                           int CR, // Must be given as value between [0,999]
                           float F,
                           void *costArgs,
                           float *h_output)
{
    std::mt19937 rng(clock());
    generateRandomVectorAndInit(d_target, d_min, d_max, d_cost,
            costArgs, popSize, dim, clock());

    for (int i = 1; i <= maxGenerations; i++) {
        evolutionKernel(d_target, d_trial, d_cost, d_target2, d_min, d_max, rng
                dim, popSize, CR, F, costArgs, i);
        float *tmp = d_target;
        d_target = d_target2;
        d_target2 = tmp;
    } // end for (generations)

    int bestIdx = -1;
    float bestCost = FLT_MAX;
    for (int i = 0; i < popSize; i++) {
        float curCost = h_cost[i];
        //std::cout << curCost << ", ";
        if (curCost < bestCost) {
            bestCost = curCost;
            bestIdx = i;
        }
    }
    h_output = d_target+(bestIdx*dim);
}

// allocate the memory needed for random number generators.
void *createRandNumGen(int size)
{
    void *x;
    gpuErrorCheck(cudaMalloc(&x, sizeof(curandState_t)*size));
    return x;
}









