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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

#include "PPM.hh"

using namespace std;
using namespace ppm;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

__device__ __constant__ float3 gpuClusterCol[2048];

#define THREADS 256
#define LOG_IMG_SIZE 8
#define IMG_SIZE 256
#define WINDOW 6

/* The function measures for every pixel the distance to all
 clusters, and determines the clusterID of the nearest cluster
 center. It then colors the pixel in the cluster's color.

 The cluster centers are given as an array of linear indices into
 the vector image, i.e.    _clusterInfo[0] = (x_0 + y_0 * _w).

 */
__global__ void voronoiKernel(float3* _dst, int _w, int _h, int _nClusters, const int* _clusterInfo)
{
    // get the shared memory
    extern __shared__ int shm[];

    int nIter = _nClusters / THREADS + 1;
    // load cluster data
    for (int i = 0; i < nIter; ++i)
    {
        int pos = i * THREADS + threadIdx.x;
        if (pos < _nClusters)
        {
            shm[pos] = _clusterInfo[pos];
        }
    }

    __syncthreads();

    // compute the position within the image
    float x = blockIdx.x * blockDim.x + threadIdx.x;
    float y = blockIdx.y;

    int pos = x + y * _w;

    // determine which is the closest cluster
    float minDist = 1000000.;
    int minIdx = 0;
    for (int i = 0; i < _nClusters; ++i)
    {

        float yy = shm[i] >> LOG_IMG_SIZE;
        float xx = shm[i] % IMG_SIZE;

        float dist = (x - xx) * (x - xx) + (y - yy) * (y - yy);
        if (dist < minDist)
        {
            minDist = dist;
            minIdx = i;
        }
    }

    _dst[pos].x = gpuClusterCol[minIdx].x;
    _dst[pos].y = gpuClusterCol[minIdx].y;
    _dst[pos].z = gpuClusterCol[minIdx].z;

    // mark the center of each cluster
    if (minDist <= 2.)
    {
        _dst[pos].x = 255;
        _dst[pos].y = 0.;
        _dst[pos].z = 0.;
    }
}

__device__ float luminance(const float4& _col)
{
    return 0.299 * _col.x + 0.587 * _col.y + 0.114 * _col.z;
}

/** stores a 1 in _dst if the pixel's luminance is a maximum in the
WINDOW x WINDOW neighborhood
 */
__global__ void featureKernel(int* _dst, cudaTextureObject_t texImg, int _w, int _h)
{
    // compute the position within the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;

    float lum = luminance(tex2D<float4>(texImg, x, y));

    bool maximum = false;

    if (lum > 20)
    {
        maximum = true;
        for (int v = y - WINDOW; v < y + WINDOW; ++v)
        {
            for (int u = x - WINDOW; u < x + WINDOW; ++u)
            {

                if (lum < luminance(tex2D<float4>(texImg, u, v)))
                {
                    maximum = false;
                }
            }
        }
    }

    if (maximum)
    {
        _dst[x + y * _w] = 1;
    }
    else
    {
        _dst[x + y * _w] = 0;
    }
}

// Chapter 39. Parallel Prefix Sum (Scan) with CUDA
/*
    Compaction kernel works for each block individually.
    It takes the input array g_idata and writes the result to g_odata.
    temp is the shared memory - always works with local thread index
    g_odata and g_idata always works with the global thread index

    In the first case: each block corresponds to one scanline of the image.
    At the end, each scanline of gpuFeatureImg will be partially prefix summed.
*/
// TODO: Create a subfuction on device to call from both kernels
__global__ void reductionKernel1(int *g_odata, int *g_idata) {
    extern __shared__ int temp[]; // allocated on invocation
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    temp[thid] = g_idata[bid * blockDim.x + thid];

    for(int stride = 1; stride <= blockDim.x / 2; stride <<= 1) {
 		int index = (thid+1) * stride * 2 - 1;
 		if(index < blockDim.x) temp[index] += temp[index-stride];
 		__syncthreads();
    }

    __syncthreads();
    g_odata[bid * blockDim.x + thid] = temp[thid];
}

// only for one block - the last element of each scanline
__global__ void reductionKernel2(int *g_odata, int *g_idata, int w) {
    extern __shared__ int temp[]; // allocated on invocation
    int thid = threadIdx.x;
    // assign each thread to the last element of a scanline
    int scanLine = w * (thid+1);    // scanline
    int lastElement = scanLine - 1; // last element of scanLine

    temp[thid] = g_idata[lastElement]; // last element of scanLine

    for (int stride = 1; stride <= blockDim.x / 2; stride <<= 1) // build sum in place up the tree
    {
        // here, we add temp[index - stride] to temp[index] if index < blockDim.x
        // simulation of first stride:
        /*
            stride = 1
            0 -> 1    temp[1] += temp[1 - 1]
            1 -> 3    temp[3] += temp[3 - 1]
            2 -> 5    temp[5] += temp[5 - 1]
            3 -> 7    temp[7] += temp[7 - 1]
            4 -> 9   x  (exceeds the boundary)

        */
 		int index = (thid+1) * stride * 2 - 1;
 		if(index < blockDim.x) temp[index] += temp[index-stride];

    }

    __syncthreads();
    g_odata[lastElement] = temp[thid];
}


__global__ void spreadingKernel2(int *g_odata, int *g_idata, int w) {
	extern __shared__ int temp[]; // allocated on invocation
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    // slighly different from compactionKernel1
    int scanLine = w * (thid);    // scanline
    int lastElement = scanLine - 1; // last element of scanLine

    temp[thid] = g_idata[lastElement]; // last element of scanLine
    if(thid==0) temp[thid] = 0;
    __syncthreads();

    for(int stride=blockDim.x/2; stride > 0; stride>>=1) {
        __syncthreads();
        /**
         * Simulation for example: 25 43 44 123
         * We have 5 threads:  0 25 43 44 123
         *
         * stride = 2
         * thid=0  --  idx=0 --  temp[2] += temp[0]   (new array= 0 25 43 44 123)
         * thid=1  --  idx=4 --  temp[6] out of boundary - stop
         *
         * stride = 1
         * thid=0  --  idx=0 --  temp[1] += temp[0]   (new array= 0 25 43 44 123)
         * thid=1  --  idx=2 --  temp[3] += temp[2]   (new array= 0 25 44 87 123)
         * thid=2  --  idx=4 --  temp[5] out of boundary - stop
         *
        */
        int idx = 2 * stride * thid;
        if(idx + stride < blockDim.x) temp[idx + stride] += temp[idx];
    }
    __syncthreads();
    g_odata[lastElement] = temp[thid];

}


// each block corresponds to one scanline of the image.
__global__ void spreadingKernel1(int *g_odata, int *g_idata) {
    extern __shared__ int temp[]; // allocated on invocation
    int thid = threadIdx.x;
    int bid = blockIdx.x;

    int scanLineBeginning = bid * (blockDim.x-1);

    if(thid != 0) temp[thid] = g_idata[scanLineBeginning + thid - 1];
    else temp[0] = 0;
    __syncthreads();

    for(int stride=blockDim.x/2; stride > 0; stride>>=1) {
        __syncthreads();
        int idx = 2 * stride * thid;
        if(idx + stride < blockDim.x) temp[idx + stride] += temp[idx];
    }
    __syncthreads();
    if(thid != 0) g_odata[bid * (blockDim.x-1) + thid - 1] = temp[thid];
    // add the previous
    __syncthreads();
    if(thid != blockDim.x - 1 && thid != 0) { // if not last or first (0) element of scanline
    	g_odata[scanLineBeginning + thid - 1] += scanLineBeginning - 1 >= 0 ? g_odata[scanLineBeginning - 1] : 0;
    }
    __syncthreads();
}


__global__ void shiftKernel(int *g_odata, int *g_idata) {
    // copy the input array to the output array with blocks
	int thid = threadIdx.x;
	int x = blockIdx.x * blockDim.x +thid;
    g_odata[x + 1] = g_idata[x];
    if(x == 0) g_odata[x] = 0;
    __syncthreads();
}


// Slides page 37
__global__ void compactKernel(int *g_odata, int *g_idata) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if(g_idata[thid] != g_idata[thid + 1]) g_odata[g_idata[thid]] = thid;
}

// !!! missing !!!
// Kernels for Prefix Sum calculation (compaction, spreading, possibly shifting)
// and for generating the gpuFeatureList from the prefix sum.

/* This program detects the local maxima in an image, writes their
location into a vector and then computes the Voronoi diagram of the
image given the detected local maxima as cluster centers.

A Voronoi diagram simply colors every pixel with the color of the
nearest cluster center. */

int main(int argc, char* argv[])
{

    // parse command line
    int acount = 1;
    if (argc < 4)
    {
        printf("usage: testPrefix <inImg> <outImg> <mode>\n");
        exit(1);
    }
    string inName(argv[acount++]);
    string outName(argv[acount++]);
    int mode = atoi(argv[acount++]);

    // Load the input image
    float* cpuImage;
    int w, h;
    readPPM(inName.c_str(), w, h, &cpuImage);
    int nPix = w * h;  // 65536

    // Allocate GPU memory
    int* gpuFeatureImg; // Contains 1 for a feature, 0 else
    // Can be used to do the reduction step of prefix sum calculation in place
    int* gpuPrefixSumShifted; // Output buffer containing the prefix sum
    // Shifted by 1 since it contains 0 as first element by definition
    int* gpuFeatureList; // List of pixel indices where features can be found.
    float3* gpuVoronoiImg; // Final rgb output image
    cudaMalloc((void**)&gpuFeatureImg, (nPix) * sizeof(int));

    cudaMalloc((void**)&gpuPrefixSumShifted, (nPix + 1) * sizeof(int));
    cudaMalloc((void**)&gpuFeatureList, 10000 * sizeof(int));

    cudaMalloc((void**)&gpuVoronoiImg, nPix * 3 * sizeof(float));

    // color map for the cluster
    float clusterCol[2048 * 3];
    float* ci = clusterCol;
    for (int i = 0; i < 2048; ++i, ci += 3)
    {
        ci[0] = 32 * i % 256;
        ci[1] = (10 * i + 128) % 256;
        ci[2] = (40 * i + 255) % 256;
    }

    cudaMemcpyToSymbol(gpuClusterCol, clusterCol, 2048 * 3 * sizeof(float));

    cudaArray* gpuTex;
    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&gpuTex, &floatTex, w, h);

    // pad to float4 for faster access
    float* img4 = new float[w * h * 4];

    for (int i = 0; i < w * h; ++i)
    {
        img4[4 * i] = cpuImage[3 * i];
        img4[4 * i + 1] = cpuImage[3 * i + 1];
        img4[4 * i + 2] = cpuImage[3 * i + 2];
        img4[4 * i + 3] = 0.;
    }

    // upload to array

    cudaMemcpy2DToArray(gpuTex, 0, 0, img4, w * 4 * sizeof(float), w * 4 * sizeof(float), h,
                        cudaMemcpyHostToDevice);

    // create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = gpuTex;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);

    cout << "setup texture" << endl;
    cout.flush();

    // calculate the block dimensions
    dim3 threadBlock(THREADS);
    dim3 blockGrid(w / THREADS, h, 1);

    printf("blockDim: %d  %d \n", threadBlock.x, threadBlock.y);
    printf("gridDim: %d  %d \n", blockGrid.x, blockGrid.y);

    featureKernel<<<blockGrid, threadBlock>>>(gpuFeatureImg, tex, w, h);

    // variable to store the number of detected features = the number of clusters
    int nFeatures;

    if (mode == 0)
    {
        ////////////////////////////////////////////////////////////
        // CPU compaction:
        ////////////////////////////////////////////////////////////

        // download result

        cudaMemcpy(cpuImage, gpuFeatureImg, nPix * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<int> features;

        // add indices of features where gpuFeatureImg != 0
        float* ii = cpuImage;
        for (int i = 0; i < nPix; ++i, ++ii)
        {
            if (*ii > 0)
            {
                features.push_back(i);
            }
        }

        cout << "nFeatures: " << features.size() << endl;

        nFeatures = features.size();
        // upload feature vector

        cudaMemcpy(gpuFeatureList, &(features[0]), nFeatures * sizeof(int), cudaMemcpyHostToDevice);
    }
    else
    {
        ////////////////////////////////////////////////////////////
        // GPU compaction:
        ////////////////////////////////////////////////////////////



        /*
         gpuFeatureImg -->  nPix many 0s and 1s
         First we want to apply reduction on each scanline:
         there are h many scanlines, so we need h many blocks
        */
        // !!! missing !!!
        // implement the prefixSum algorithm
        // 1. Do the reduction step for all scanlines, one scanline per block.

        reductionKernel1<<<h, w, w * sizeof(int)>>>(gpuFeatureImg, gpuFeatureImg);
        reductionKernel2<<<1, h, h * sizeof(int)>>>(gpuFeatureImg, gpuFeatureImg, w);
        spreadingKernel2<<<1, h+1, (h+1) * sizeof(int)>>>(gpuFeatureImg, gpuFeatureImg, w);
        spreadingKernel1<<<h, (w+1), (w+1) * sizeof(int)>>>(gpuFeatureImg, gpuFeatureImg);
        shiftKernel<<<h, w>>>(gpuPrefixSumShifted, gpuFeatureImg);
        cudaMemset(gpuFeatureList, 0, 10000 * sizeof(int));
        compactKernel<<<h, w>>>(gpuFeatureList, gpuPrefixSumShifted);
        // nFeatures is equal to the last element of gpuPrefixSumShifted
        cudaMemcpy(&nFeatures, gpuPrefixSumShifted + nPix, sizeof(int), cudaMemcpyDeviceToHost);

        // 2. Do the reduction step for the last elements of all scanlines, all in one block

        // 3. Do the spreading step for the last elements of all scanlines, all in one block.
        //    -> The last elements / elements before the scanlines have the right values now.
        // 4. Do the spreading step for all scanlines, one scanline per block.

        // Make sure that gpuFeatureList is filled according to the CPU implementation
        // and that nFeatures has the correct value!


    }

    // now compute the Voronoi Diagram around the detected features.
    voronoiKernel<<<blockGrid, threadBlock, nFeatures * sizeof(int)>>>(gpuVoronoiImg, w, h,
                                                                       nFeatures, gpuFeatureList);

    // download final voronoi image.

    cudaMemcpy(cpuImage, gpuVoronoiImg, nPix * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    // Write to disk
    writePPM(outName.c_str(), w, h, (float*)cpuImage);

    // Cleanup
    cudaDestroyTextureObject(tex);
    cudaFreeArray(gpuTex);
    cudaFree(gpuFeatureList);
    cudaFree(gpuFeatureImg);
    cudaFree(gpuPrefixSumShifted);
    cudaFree(gpuVoronoiImg);

    delete[] cpuImage;
    delete[] img4;

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

