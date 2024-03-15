// ==========================================================================
// $Id$
// ==========================================================================
// (C)opyright: 2009-2010
//
//   Ulm University
//
// Creator: Hendrik Lensch
// Email:   {hendrik.lensch,johannes.hanika}@uni-ulm.de
// ==========================================================================
// $Log$
// ==========================================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "PPM.hh"

using namespace std;
using namespace ppm;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

#define MAX_THREADS 128

//-------------------------------------------------------------------------------

// specify the gamma value to be applied
__device__ __constant__ float gpuGamma[1];


/* compute gamma correction on the float image _src of resolution dim,
 outputs the gamma corrected image should be stored in_dst[blockIdx.x *
 blockDim.x + threadIdx.x]. Each thread computes on pixel element.
 */
__global__ void absKernel(float* _dst, const float* _src1, const float* _src2, int _w)
{
    int x = blockIdx.x * MAX_THREADS + threadIdx.x;
    int y = blockIdx.y;
    int pos = y * _w + x;

    if (x < _w)
    {
        _dst[pos] = _src1[pos] - _src2[pos] > 0 ? _src1[pos] - _src2[pos] : _src2[pos] - _src1[pos];
    }
}

//-------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    int acount = 1; // parse command line

    if (argc < 3)
    {
        printf("usage: %s <inImg> <inImg2>\n", argv[0]);
        exit(1);
    }

    float* img1;
    float* img2;
    float* img3;

    bool success = true;
    int w, h;
    success &= readPPM(argv[acount++], w, h, &img1);
    if (!success) {
        exit(1);
    }
    success &= readPPM(argv[acount++], w, h, &img2);
    if (!success) {
        exit(1);
    }
    int nPix = w * h;

    float* gpuImg1;
    float* gpuImg2;
    float* gpuResImg;

    //-------------------------------------------------------------------------------
    printf("Executing the GPU Version\n");
    // copy the image to the device
    cudaMallocHost((void**)&img3, nPix * 3 * sizeof(float));
    cudaMalloc((void**)&gpuImg1, nPix * 3 * sizeof(float));
    cudaMalloc((void**)&gpuImg2, nPix * 3 * sizeof(float));
    cudaMalloc((void**)&gpuResImg, nPix * 3 * sizeof(float));
    cudaMemcpy(gpuImg1, img1, nPix * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuImg2, img2, nPix * 3 * sizeof(float), cudaMemcpyHostToDevice);


    // calculate the block dimensions
    dim3 threadBlock(MAX_THREADS);
    // select the number of blocks vertically (*3 because of RGB)
    dim3 blockGrid((w * 3) / MAX_THREADS + 1, h, 1);
    printf("bl/thr: %d  %d %d\n", blockGrid.x, blockGrid.y, threadBlock.x);

    absKernel<<<blockGrid, threadBlock>>>(gpuResImg, gpuImg1, gpuImg2, w * 3);

    // download result
    cudaMemcpy(img3, gpuResImg, nPix * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpuResImg);
    cudaFree(gpuImg1);
    cudaFree(gpuImg2);

    writePPM(argv[acount++], w, h, (float*)img3);

    delete[] img1;
    delete[] img2;
    delete[] img3;

    checkCUDAError("end of program");

    printf("  done\n");
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
