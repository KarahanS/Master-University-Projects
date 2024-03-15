#include "Tools.h"

#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

using namespace std;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

#define VERBOSE // Prints input matrix and results. Only uncomment for small matrix sizes!
// #define RUN_CPU// Runs CPU code for reference (slow!!!)
#define N 4 // Must be a multiple of THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 16 // per axis -> block has this value squared threads.
void multiplyMatrix(float* result, const float* a, const float* b, const int n)
{
    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            result[i * n + j] = 0.0f;
            for (unsigned int k = 0; k < n; k++)
            {
                result[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
}

void dumpMatrix(const float* m, const int n)
{
    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            cout << setw(3) << setprecision(3) << m[i * n + j] << " ";
        }
        cout << endl;
    }
}

float randF(const float min = 0.0f, const float max = 1.0f)
{
    int randI = rand();
    float randF = (float)randI / (float)RAND_MAX;
    float result = min + randF * (max - min);

    return result;
}


__global__ void multiplyMatrixGpu1(float* C, const float* A,
        const float* B, const int n) {
            int r = blockIdx.x * blockDim.x + threadIdx.x;
            int c = blockIdx.y * blockDim.y + threadIdx.y;
            if(r >= n || c >= n) return;  // out of bounds

            float Pvalue = 0;
            for(int k = 0; k < n; k++) {
                // a[r * n + i] = r-th row, k-th column (row is fixed, column changes)
                // b[k * n + c] = k-th row, c-th column (column is fixed, row changes)
                float M = A[r * n + k];
                float V = B[k * n + c];
                // each thread writes one element
                Pvalue += M * V;
            }
            // n = number of entries per row
            C[r * n + c] = Pvalue;
        }
// N = 100
// will be called with 7x7 grid, where each block has 16x16 threads
// 7*16 = 112 > 100  (for safety, it is 7x7)
// one block computers, 16x16 submatrix
// Matrix Multiplication Kernel with Shared Mem (page 62)
__global__ void multiplyMatrixGpu2(float* C, const float* A,
        const float* B, const int n) {
            int bx = blockIdx.x; int by = blockIdx.y; // block index
            int tx = threadIdx.x; int ty = threadIdx.y; // thread index

            int x = bx * blockDim.x + tx; // x index of C element
            int y = by * blockDim.y + ty; // y index of C element

            if(x >= n || y >= n) return;  // out of bounds

            int blockSizeY = blockDim.y; // number of threads per block
            int blockSizeX = blockDim.x; // number of threads per block
            int tileWidth = blockSizeX;  // in our case 16

            // threads_per_block = 16
            __shared__ float As[THREADS_PER_BLOCK][THREADS_PER_BLOCK]; // submatrix of A
            __shared__ float Bs[THREADS_PER_BLOCK][THREADS_PER_BLOCK]; // submatrix of B

            float sum = 0.0;

            // n = WIDTH
            // loop over the submatrices of A and B
            for(int k = 0; k<n; k+=blockSizeX) {
                __syncthreads();
                As[ty][tx] = A[y * n + k + tx];
                Bs[ty][tx] = B[(k + ty) * n + x];
                __syncthreads();

                for(int i = 0; i < blockSizeX; i++) {
                    sum += As[ty][i] * Bs[i][tx];
                }
            }
            C[y * n + x] = sum;
        }

int main(int argc, char** argv)
{
    __int64_t startTime;
    __int64_t endTime;

    // Allocate all memory
    float* hM1 = new float[N * N];
    float* hM2 = new float[N * N];
    float* hMR = new float[N * N];
    float* gM1;
    cudaMalloc(&gM1, sizeof(float) * N * N);
    float* gM2;
    cudaMalloc(&gM2, sizeof(float) * N * N);
    float* gMR;
    cudaMalloc(&gMR, sizeof(float) * N * N);

    // Initialize matrices and upload to CUDA
    for (unsigned int n = 0; n < N * N; n++)
    {
        hM1[n] = randF(-1.0, 1.0);
        hM2[n] = randF(-1.0, 1.0);
    }
    cudaMemcpy(gM1, hM1, sizeof(int) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(gM2, hM2, sizeof(int) * N * N, cudaMemcpyHostToDevice);
#ifdef VERBOSE
    cout << "Input Matrices:" << endl;
    dumpMatrix(hM1, N);
    cout << endl;
    dumpMatrix(hM2, N);
    cout << endl << endl;
#endif

#ifdef RUN_CPU
    // Calculations on CPU
    startTime = continuousTimeNs();
    multiplyMatrix(hMR, hM1, hM2, N);
    endTime = continuousTimeNs();
#ifdef VERBOSE
    cout << "CPU:" << endl;
    dumpMatrix(hMR, N);
    cout << endl;
#endif
    cout << "CPU time: " << (endTime - startTime) << "ns" << endl;
#endif

    // Calculations on GPU
    int blocksPerGridX =
        N % THREADS_PER_BLOCK == 0 ? N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;
    int blocksPerGridY =
        N % THREADS_PER_BLOCK == 0 ? N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;
    startTime = continuousTimeNs();
    multiplyMatrixGpu1<<<dim3(blocksPerGridX, blocksPerGridY, 1),
                         dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)>>>(gMR, gM1, gM2, N);
    cudaDeviceSynchronize();
    endTime = continuousTimeNs();
    cudaMemcpy(hMR, gMR, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
#ifdef VERBOSE
    cout << "GPU simple:" << endl;
    dumpMatrix(hMR, N);
    cout << endl;
#endif
    cout << "GPU simple time: " << (endTime - startTime) << "ns" << endl;
    startTime = continuousTimeNs();
    multiplyMatrixGpu2<<<dim3(blocksPerGridX, blocksPerGridY, 1),
                         dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)>>>(gMR, gM1, gM2, N);
    cudaDeviceSynchronize();
    endTime = continuousTimeNs();
    cudaMemcpy(hMR, gMR, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
#ifdef VERBOSE
    cout << "GPU advanced:" << endl;
    dumpMatrix(hMR, N);
    cout << endl;
#endif
    cout << "GPU advanced time: " << (endTime - startTime) << "ns" << endl;

    // Free all memory
    cudaFree(gM1);
    cudaFree(gM2);
    cudaFree(gMR);
    delete[] hM1;
    delete[] hM2;
    delete[] hMR;

    checkCUDAError("end of program");
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
