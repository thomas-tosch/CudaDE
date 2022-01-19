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

#include "DifferentialEvolution.hpp"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

int runTest(int popSize, int dim, int costFun, float *minBound, float *maxBound, float cr)
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
    std::vector<float> result = minimizer.fmin(d_x);
    std::cout << x.costFun << std::endl;
    std::cout << "Result = " << result[0] << ", " << result[1] << std::endl;
    std::cout << "Finished main function." << std::endl;
}

int main(int argc, char *argv[])
{
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
    runTest(popSize, dim, costFun, minBound, maxBound, 0.8);
    return 1;
}
