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
//  testMain.cpp
//
// This is a test code to show an example usage of Differential Evolution

#include <stdio.h>
#include <chrono>
#include <cfloat>
#include <math.h>

#include "DifferentialEvolutionCPU.hpp"
#include "DifferentialEvolution.hpp"
#include "DifferentialEvolutionCPU.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

float runTest(int popSize, int dim, int costFun, float minBound, float maxBound, float cr)
{

    float arr[3] = {2.5, 2.6, 2.7};

    // data that is created in host, then copied to a device version for use with the cost function.
    struct data x;
    struct data *d_x;
    gpuErrorCheck(cudaMalloc(&x.arr, sizeof(float) * 3));
    unsigned long size = sizeof(struct data);
    gpuErrorCheck(cudaMalloc((void **)&d_x, size));

    x.v = popSize;
    x.dim = dim;
    x.costFun = costFun;
    float minBounds[x.dim] = {minBound};
    float maxBounds[x.dim] = {maxBound};
    gpuErrorCheck(cudaMemcpy(x.arr, (void *)&arr, sizeof(float) * 3, cudaMemcpyHostToDevice));
    int maxGen = (10000 * x.dim) / x.v;
    // Create the minimizer with a popsize of 192, 50 generations, Dimensions = 2, CR = 0.9, F = 2
    DifferentialEvolution minimizer(x.v,maxGen, x.dim,
                                    cr, 0.5, minBounds, maxBounds);
    gpuErrorCheck(cudaMemcpy(d_x, (void *)&x, sizeof(struct data), cudaMemcpyHostToDevice));
    // get the result from the minimizer
    float* result = minimizer.fmin(d_x);
    //std::cout << x.costFun << std::endl;
    //std::cout << "Result = ";
    float bestCost = FLT_MAX;
    for (int i = 0; i < popSize; i++) {
        float curCost = result[i];
        if (curCost < bestCost) {
            bestCost = curCost;
        }
    }
    //std::cout << bestCost << std::endl;
    //std::cout << "Finished main function." << std::endl;
    return bestCost;
}

float runTestSequential(int popSize, int dim, int costFun, float minBound, float maxBound, float cr)
{

    float arr[3] = {2.5, 2.6, 2.7};
    struct dataCPU x;
    struct dataCPU *d_x;
    unsigned long size = sizeof(struct dataCPU);
    x.v = popSize;
    x.dim = dim;
    x.costFun = costFun;
    float minBounds[x.dim] = {minBound};
    float maxBounds[x.dim] = {maxBound};
    int maxGen = (10000 * x.dim) / x.v;
    std::vector<float> result(dim);
    float d_target1[popSize * dim] = {};
    float d_trial[popSize * dim] = {};
    float d_cost[popSize * dim] = {};
    float d_target2[popSize * dim] = {};
    float h_cost[popSize * dim] = {};
    differentialEvolutionCPU(d_target1, d_trial, d_cost, d_target2, minBounds,
                             maxBounds, h_cost, dim, popSize, maxGen, cr, 0.5, &x,
                             result.data());
    float bestCost = FLT_MAX;
    for (int i = 0; i < popSize; i++) {
        float curCost = h_cost[i];
        if (curCost < bestCost) {
            bestCost = curCost;
        }
    }
    //std::cout << bestCost << std::endl;
    return bestCost;
}



float meanValue(float *values)
{
    float sum = 0;
    for (int i = 0; i < 25; i ++) {
        sum += values[i];
    }
    return sum / 25.0f;
}

float standardDeviation(float *values)
{
    float sum = 0;
    float mean = meanValue(values);
    for (int i = 0; i < 25; i++) {
        sum += pow(std::abs(values[i] - mean), 2);
    }
    return sqrt(sum / 25);
}

float successRate(float *values, float offset)
{
    float success[25] = {0};
    for (int i = 0; i < 25; ++i) {
        if (values[i] < (pow(10, -8) + offset)) {
            success[i] = 1;
        }
    }
    return meanValue(success);
}

int testCase()
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    int dimensions[3] = { 10, 50, 100};
    int popSizes[4] = { 50, 100, 500, 1000 };
    float crossRates[1] = { 0.8 };
    int costFuncs[4] = { SPHERE, ROSENBROCK, GRIEWANK, RASTRIGIN };
    float minBounds[4] = { -5.12, -5, -600, -5.12};
    float maxBounds[4] = { 5.12,   10, 600, 5.12};
    float offsets[4] = { -450,  390 , -180, -330 };
    float costValues[25] = {0};
    float costTimes[25] = {0};

    for (int i = 0; i < sizeof(dimensions)/sizeof(dimensions[0]); i++)
    {
        std::cout << "Dim: " << dimensions[i] << std::endl;
        std::cout << "Func;PopSize;Cost;Time;" << std::endl;
        for (int k = 0; k < sizeof(crossRates)/sizeof(crossRates[0]); k++)
        {
            for (int l = 0; l < sizeof(costFuncs)/sizeof(costFuncs[0]); l++)
            {
                for (int j = 0; j < sizeof(popSizes)/sizeof(popSizes[0]); j++)
                {
                    for (int m = 0; m < 25; m++) {
                        auto t1 = high_resolution_clock::now();
                        costValues[m] = runTest(popSizes[j], dimensions[i],
                                                costFuncs[l], minBounds[l],
                                                maxBounds[l],crossRates[k]
                        );
                        auto t2 = high_resolution_clock::now();
                        duration<double, std::milli> ms_double = t2 - t1;
                        costTimes[m] = ms_double.count();
                    }
                    //std::cout << "Pop: " << popSizes[j] << std::endl;
                    std::cout << "F(" << costFuncs[l] << ")," << popSizes[j] << ",";
                    std::cout << meanValue(costValues) << " (" << standardDeviation(costValues) << "),";
                    std::cout << meanValue(costTimes) << " (" << standardDeviation(costTimes) << "),";
                    std::cout << successRate(costValues, offsets[l]) << std::endl;
                }
            }
        }
        std::cout << std::endl;
    }
}

int main(int argc, char *argv[])
{
    if (argc == 1) {
        return testCase();
    }
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    float F1 = 0.25;
    float F2 = 0.25;
    float F3 = 0.2;
    float F4 = 0.2;
    int dim = 3;
    int popSize = 3;
    int costFun = 3;
    if (argc > 1) {
        popSize = std::stoi(argv[1]);
    }
    if (argc > 2) {
        dim = std::stoi(argv[2]);
    }
    if (argc > 3) {
        costFun = std::stoi(argv[3]);
    }
    float minBound[dim] = {-100};
    float maxBound[dim] = {100};
    if (argc > 4) {
        minBound[0] = std::stoi(argv[4]);
    }
    if (argc > 5) {
        maxBound[0] = std::stoi(argv[5]);
    }
    auto t1 = high_resolution_clock::now();
    runTest(popSize, dim, costFun, minBound[0], maxBound[0], 0.8);
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";
    return 1;
}
