// ==========================================================================
// $Id$
// ==========================================================================
// (C)opyright: 2009
//
//   Ulm University
//
// Creator: Hendrik Lensch, Holger Dammertz
// Email:   hendrik.lensch@uni-ulm.de, holger.dammertz@uni-ulm.de
// ==========================================================================
// $Log$
// ==========================================================================

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>
#include <sys/time.h>

using namespace std;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

#define MAX_BLOCKS 256
#define MAX_THREADS 256

inline __int64_t continuousTimeNs()
{
    timespec now;
    clock_gettime(CLOCK_REALTIME, &now);

    __int64_t result = (__int64_t)now.tv_sec * 1000000000 + (__int64_t)now.tv_nsec;

    return result;
}

__global__ void dotProdKernel(float* dst, const float* a1, const float* a2, int dim)
{
    // Number of the current thread
    unsigned int threadNo = blockDim.x * blockIdx.x + threadIdx.x;
    // Number of all threads
    unsigned int threadSize = gridDim.x * blockDim.x;

    // Sum up every (threadSize)th element starting at the threads index and ending before dim
    float result = 0.0f;
    for (unsigned int t = threadNo; t < dim; t += threadSize)
        result += a1[t] * a2[t];

    // Write the result to dst[threadIdx] if it can contain something
    dst[threadNo] = result;
}

// !!! missing !!!
// Kernel for reducing gridDim.x*blockDim.x elements to gridDim.x elements

// max_blocks * max_threads = dim of vector
// each element = a1[i] * a2[i]

// first, reduce max_blocks * max_threads to max_blocks
// then reduce max_blocks to 1

__global__ void reduceSumKernel(float* dst, const float* src)
{
    extern __shared__ float partialSum[];
    unsigned int threadNo = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int localThreadNo = threadIdx.x;

    // load data into shared memory (shared memory - local to each block)
    partialSum[localThreadNo] = src[threadNo];
    __syncthreads();

    unsigned int blockNo = blockIdx.x;

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) // dimBlock = 16, stride = 8, 4, 2, 1, 0--> out of the loop
    {
    	__syncthreads();
        if (localThreadNo < stride)
            partialSum[localThreadNo] += partialSum[localThreadNo + stride]; // partialSum[0] += partialSum[0 + 8]
    }
    dst[blockNo] = partialSum[0]; // result is stored in the first element
}

/* This program sets up two large arrays of size dim and computes the
 dot product of both arrays.

 Most of the code of previous exercises is reused.
 Mode 0 of the program computes the final dot product as before.

 Mode 1: After computing the dot product and storing the result for all
 MAX_BLOCKS * MAX_THREAD threads, this time, the reduction of the sum
 is to be computed on the GPU.

 Write a reduction sum kernel which is called log(n) times.
 The number of total threads will be divided by nThreads(iter-1)
 in each iteration.

 Inside the kernel, the problem will be reduced by a factor of 2 in
 each step.

 */

int main(int argc, char* argv[])
{

    // parse command line
    int acount = 1;

    if (argc < 3)
    {
        printf("usage: testDotProductStreams <dim> <reduction mode [gold:0, CPU:1, GPU:2]>\n");
        exit(1);
    }

    // number of elements in both vectors
    int dim = atoi(argv[acount++]);

    int mode = atoi(argv[acount++]);

    printf("dim: %d\n", dim);

    // Allocate only pagelocked memory for simplicity
    float* cpuArray1;
    float* cpuArray2;
    float* cpuResult;
    cudaMallocHost((void**)&cpuArray1, dim * sizeof(float));
    cudaMallocHost((void**)&cpuArray2, dim * sizeof(float));
    cudaMallocHost((void**)&cpuResult, MAX_THREADS * MAX_BLOCKS * sizeof(float));

    // initialize the two arrays
    for (int i = 0; i < dim; ++i)
    {
#ifdef RTEST
        cpuArray1[i] = drand48();
        cpuArray2[i] = drand48();
#else
        cpuArray1[i] = 1.0;
        cpuArray2[i] = 1.; // i % 10;
#endif
    }

    // Allocate GPU memory
    float* gpuArray1;
    float* gpuArray2;
    float* gpuResult1; // Two result arrays to be able to move data from one to the other during
                       // reduction
    float* gpuResult2;
    cudaMalloc((void**)&gpuArray1, dim * sizeof(float));
    cudaMalloc((void**)&gpuArray2, dim * sizeof(float));

    cudaMalloc((void**)&gpuResult1, MAX_BLOCKS * MAX_THREADS * sizeof(float));

    cudaMalloc((void**)&gpuResult2,
               MAX_BLOCKS * MAX_THREADS
                   * sizeof(float)); // MAX_BLOCKS elements would be sufficient here...

    // Upload input data

    cudaMemcpy(gpuArray1, cpuArray1, dim * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(gpuArray2, cpuArray2, dim * sizeof(float), cudaMemcpyHostToDevice);

    // Variable for output
    double finalDotProduct = 0.;

    __int64_t startTime = continuousTimeNs();

    // Iterations for benchmarking only the kernel call
    for (int iter = 0; iter < 1000; ++iter)
    {
        // a simplistic way of splitting the problem into threads
        dim3 blockGrid(MAX_BLOCKS);
        dim3 threadBlock(MAX_THREADS);

        unsigned int expectedResultSize;

        switch (mode)
        {
        case 0:
            finalDotProduct = 0.0;

            for (unsigned int i = 0; i < dim; i++)
                finalDotProduct += cpuArray1[i] * cpuArray2[i];

            break;

        case 1:
            // call the dot kernel
            dotProdKernel<<<blockGrid, threadBlock>>>(gpuResult1, gpuArray1, gpuArray2, dim);

            // If dim < launchedThreads, only the first dim elements will contain data
            expectedResultSize = min(dim, MAX_THREADS * MAX_BLOCKS);

            // download and combine the results of multiple threads

            cudaMemcpy(cpuResult, gpuResult1, expectedResultSize * sizeof(float),
                       cudaMemcpyDeviceToHost);

            finalDotProduct = 0.;

            // accumulate the final result on the host
            for (int i = 0; i < expectedResultSize; ++i)
                finalDotProduct += cpuResult[i];

            break;

        case 2:
            // call the dot kernel, store result in gpuResult1
            dotProdKernel<<<blockGrid, threadBlock>>>(gpuResult1, gpuArray1, gpuArray2, dim);
            // dim = number of elements in the input arrays

            // !!! missing !!!
            // Reduce all the dot product summands to one single value,
            // download it to a float and use it to set finalDotProduct.

            // gpuResult1 --> number of blocks * number of threads
            // gpuResult2 --> number of blocks
            // gpuResult --> 1

            reduceSumKernel<<<blockGrid, threadBlock, MAX_THREADS * sizeof(float)>>>(gpuResult2, gpuResult1);
            reduceSumKernel<<<1, blockGrid, MAX_BLOCKS * sizeof(float)>>>(gpuResult1, gpuResult2);
            cudaMemcpy(cpuResult, gpuResult1, sizeof(float),
                                               cudaMemcpyDeviceToHost);
            			finalDotProduct = cpuResult[0];
            break;

        } // end switch
    }

    __int64_t endTime = continuousTimeNs();
    __int64_t runTime = endTime - startTime;

    // Print results and timing
    printf("Result: %f\n", finalDotProduct);
    printf("Time: %f\n", (float)runTime / 1000000000.0f);

    // cleanup GPU memory
    cudaFree(gpuResult1);
    cudaFree(gpuResult2);
    cudaFree(gpuArray2);
    cudaFree(gpuArray1);

    // free page locked memory
    cudaFreeHost(cpuArray1);
    cudaFreeHost(cpuArray2);
    cudaFreeHost(cpuResult);

    checkCUDAError("end of program");

    printf("done\n");
}

void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}
