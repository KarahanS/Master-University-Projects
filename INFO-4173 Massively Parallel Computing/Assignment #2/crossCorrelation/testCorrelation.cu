// ==========================================================================
// $Id$
// ==========================================================================
// (C)opyright: 2009-2010
//
//   Ulm University
//
// Creator: Hendrik Lensch, Holger Dammertz
// Email:   hendrik.lensch@uni-ulm.de, holger.dammertz@uni-ulm.de
// ==========================================================================
// $Log$
// ==========================================================================
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

#ifndef _PPM_H
#define _PPM_H

/*! \file  PPM.hh
    \brief provides ppm reader and writer functions
 */

namespace ppm
{

bool readPPM(const char* _fname, int& _w, int& _h, float** _data);

bool writePPM(const char* _fname, int _w, int _h, float* _data);

} // namespace ppm

#endif


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

using namespace std;
using namespace ppm;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

#define THREADS 16

/* Compute the correlation between two images. Both images are loaded as texture.
 Each thread has a specific offset. It then computes the component-wise product of the shifted
 texture image with texImg1. The result image at location  (x,y) has the offset (offsX, offsY)

 dst(x,y) = sum_(i,j)  _img1(i,j) * _img2(i+offsX, j+offsY),

 where sum_(i,j) specifies summing over all pixels in the image.
 */
__global__ void correlationKernel(float3* _dst, cudaTextureObject_t texImg1,
                                  cudaTextureObject_t texImg2, int _w, int _h)
{
    // compute the position within the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // compute the offset of this thread
    int offsX = x - _w / 2;
    int offsY = y - _h / 2;

    float3 res;
    res.x = 0.0;
    res.y = 0.0;
    res.z = 0.0;


    // Loop over all image pixels (per output pixel)
    for (int j = 0; j < _h; ++j)
    {
        for (int i = 0; i < _w; ++i)
        {
            // !!! missing !!!
            // Fetch the pixel from image 1 and the offset pixel from image 2
            // Add the componentwise product of them to res
            // Attention: Texture access is done via tex2D. Use normalized coordinates!

            float4 pixel = tex2D<float4>(texImg1, i / (float)_w, j / (float)_h);
            float4 offsetPixel = tex2D<float4>(texImg2, (i + offsX) / (float)_w, (j + offsY) / (float)_h);
            res.x += pixel.x * offsetPixel.x;
            res.y += pixel.y * offsetPixel.y;
            res.z += pixel.z * offsetPixel.z;
        }
    }
    // return only the positive values and scale to fit the output.
    _dst[x + y * _w].x = max(0., res.x) * 0.05;
    _dst[x + y * _w].y = max(0., res.y) * 0.05;
    _dst[x + y * _w].z = max(0., res.z) * 0.05;

    printf("%d %d %f\n", x, y, _dst[x + y * _w].x);
}

/** this function simply subtracts the average value from an image and scales it to the range of
 * [0,1] from [0,255];
 */
void normalize(float* _img, int _w, int _h)
{

    // first, compute the average
    float avg[3] = {0., 0., 0.};

    float* ii = _img;

    for (int y = 0; y < _h; ++y)
    {
        for (int x = 0; x < _w; ++x)
        {

            avg[0] += *(ii++);
            avg[1] += *(ii++);
            avg[2] += *(ii++);
        }
    }

    avg[0] /= (float)_w * _h;
    avg[1] /= (float)_w * _h;
    avg[2] /= (float)_w * _h;

    // now subtract average

    ii = _img;
    for (int y = 0; y < _h; ++y)
    {
        for (int x = 0; x < _w; ++x)
        {

            *(ii++) -= avg[0];
            *(ii++) -= avg[1];
            *(ii++) -= avg[2];
        }
    }

    ii = _img;
    for (int y = 0; y < _h; ++y)
    {
        for (int x = 0; x < _w * 3; ++x)
        {
            *(ii++) /= 255.;
        }
    }
}

/* This program computes the cross-correlation between two images of
the same size.  The size of the test images is nicely chosen to be
multiples of 16 in each dimension, so don't worry about image
boundaries!

All computations have to be done in the image domain (there exist
faster ways to calculate the cross-correlation in Fourier
domain).
 */

int main(int argc, char* argv[])
{

    // parse command line
    int acount = 1;

    if (argc < 4)
    {
        printf("usage: %s <inImg> <inImg2> <outImg>\n", argv[0]);
        exit(1);
    }

    // Read two images to host memory
    float* img1;
    float* img2;

    bool success = true;
    int w, h;
    success &= readPPM(argv[acount++], w, h, &img1);
    success &= readPPM(argv[acount++], w, h, &img2);
    if (!success) {
        exit(1);
    }

    // let's compute normalized cross products
    // so we normalize both images, i.e. subtract the mean value and scale by 1./255.
    normalize(img1, w, h);
    normalize(img2, w, h);
    int nPix = w * h;

    // pad to float4 for faster access
    float* img3 = new float[w * h * 4];
    for (int i = 0; i < w * h; ++i)
    {
        img3[4 * i] = img1[3 * i];
        img3[4 * i + 1] = img1[3 * i + 1];
        img3[4 * i + 2] = img1[3 * i + 2];
        img3[4 * i + 3] = 0.;
    }
    float* img4 = new float[w * h * 4];
    for (int i = 0; i < w * h; ++i)
    {
        img4[4 * i] = img2[3 * i];
        img4[4 * i + 1] = img2[3 * i + 1];
        img4[4 * i + 2] = img2[3 * i + 2];
        img4[4 * i + 3] = 0.;
    }

    // Prepare arrays to access images as textures (faster due to caching)
    cudaArray_t gpuTex1;
    cudaArray_t gpuTex2;

    // !!! missing !!!
    // Create arrays and upload the padded texture data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaMallocArray(&gpuTex1, &channelDesc, w, h);
    cudaMallocArray(&gpuTex2, &channelDesc, w, h);

    cudaMemcpy2DToArray(gpuTex1, 0, 0, img3, w * 4 * sizeof(float), w * 4 * sizeof(float), h, cudaMemcpyHostToDevice);
    cudaMemcpy2DToArray(gpuTex2, 0, 0, img4, w * 4 * sizeof(float), w * 4 * sizeof(float), h, cudaMemcpyHostToDevice);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    // set the texture mode to normalized
    texDesc.normalizedCoords = 1;
    // and allow texture coordinates to wrap at boundaries
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;

    // create texture objects
    cudaTextureObject_t tex1 = 0;
    cudaTextureObject_t tex2 = 0;

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    resDesc.res.array.array = gpuTex1;
    cudaCreateTextureObject(&tex1, &resDesc, &texDesc, nullptr);
    resDesc.res.array.array = gpuTex2;
    cudaCreateTextureObject(&tex2, &resDesc, &texDesc, nullptr);

    // Allocate gpu memory for the result
    float* gpuResImg;
    cudaMalloc((void**)&gpuResImg, nPix * 3 * sizeof(float));

    // calculate the block dimensions
    dim3 threadBlock(THREADS, THREADS);
    dim3 blockGrid(w / THREADS, h / THREADS, 1);

    printf("blockDim: %d  %d \n", threadBlock.x, threadBlock.y);
    printf("gridDim: %d  %d \n", blockGrid.x, blockGrid.y);

    correlationKernel<<<blockGrid, threadBlock>>>((float3*)gpuResImg, tex1, tex2, w, h);
    // download result
    cudaMemcpy(img1, gpuResImg, nPix * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Store result to disk
    writePPM(argv[acount++], w, h, (float*)img1);

    // Clean up
    cudaDestroyTextureObject(tex1);
    cudaDestroyTextureObject(tex2);
    cudaFree(gpuResImg);

    // !!! missing !!!
    // Free the arrays
    cudaFreeArray(gpuTex1);
    cudaFreeArray(gpuTex2);


    delete[] img1;
    delete[] img2;
    delete[] img3;
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
