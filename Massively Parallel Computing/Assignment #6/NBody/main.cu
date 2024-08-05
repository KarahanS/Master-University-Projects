// by Batuhan Ozcomlekci: https://github.com/Bozcomlekci

#include "Tools.h"
#include "gltools.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>

#include <cuda_runtime.h>

using namespace std;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

#define GUI
#define NUM_FRAMES 100

#define THREADS_PER_BLOCK 128
#define EPS_2 0.00001f
#define GRAVITY 0.00000001f


// __device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai) {   
//     float3 r;   // r_ij  [3 FLOPS]   r.x = bj.x - bi.x;
//     r.y = bj.y - bi.y;   r.z = bj.z - bi.z;   // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
//     float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS_2;  
//     float distSixth = distSqr * distSqr * distSqr;   
//     float invDistCube = 1.0f/sqrtf(distSixth);   // s = m_j * invDistCube [1 FLOP]
//     float s = bj.w * invDistCube;   // a_i =  a_i + s * r_ij [6 FLOPS]
//     ai.x += r.x * s;  
//     ai.y += r.y * s;
//     ai.z += r.z * s;   
//     return ai; 
// } 


// __device__ float3 tile_calculation(float4 myPosition, float3 accel) {   
//     int i;   
//     extern __shared__ float4[] shPosition;   
//     for (i = 0; i < blockDim.x; i++) {     
//         accel = bodyBodyInteraction(myPosition, shPosition[i], accel);   
//     }   
//     return accel; 
// } 

// __global__ void calculate_forces(void *devX, void *devA) {   
//     extern __shared__ float4[] shPosition;
//     float4 *globalX = (float4 *)devX;
//     float4 *globalA = (float4 *)devA;   
//     float4 myPosition;
//     int i, tile;
//     float3 acc = {0.0f, 0.0f, 0.0f};
//     int gtid = blockIdx.x * blockDim.x + threadIdx.x;
//     myPosition = globalX[gtid];
//     for (i = 0, tile = 0; i < N; i += p, tile++) {
//         int idx = tile * blockDim.x + threadIdx.x;
//         shPosition[threadIdx.x] = globalX[idx];
//         __syncthreads();     
//         acc = tile_calculation(myPosition, acc);
//         __syncthreads();   
//     }   // Save the result in global memory for the integration step.    
//     float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};   
//     globalA[gtid] = acc4; 
// } 

float randF(const float min = 0.0f, const float max = 1.0f)
{
    int randI = rand();
    float randF = (float)randI / (float)RAND_MAX;
    float result = min + randF * (max - min);

    return result;
}

inline __device__ float2 operator+(const float2 op1, const float2 op2)
{
    return make_float2(op1.x + op2.x, op1.y + op2.y);
}

inline __device__ float2 operator-(const float2 op1, const float2 op2)
{
    return make_float2(op1.x - op2.x, op1.y - op2.y);
}

inline __device__ float2 operator*(const float2 op1, const float op2)
{
    return make_float2(op1.x * op2, op1.y * op2);
}

inline __device__ float2 operator/(const float2 op1, const float op2)
{
    return make_float2(op1.x / op2, op1.y / op2);
}

inline __device__ void operator+=(float2& a, const float2 b)
{
    a.x += b.x;
    a.y += b.y;
}

// __device__ float2 bodyBodyInteraction2D(float2 bi, float2 bj, float2 ai, float mj) {
//     float2 r = bi - bj;
//     float dist2 = r.x * r.x + r.y * r.y + EPS_2;
//     float sqrt_dist3 = sqrtf(dist2 * dist2 * dist2);
//     return ai + r * (mj / sqrt_dist3) * GRAVITY;
// } 

// __device__ float2 tile_calculation2D(float2 myPosition, float2 accel, float mj) {
//     extern __shared__ float2 shPosition[];
//     for (int i = 0; i < blockDim.x; i++) {
//         accel = bodyBodyInteraction2D(myPosition, shPosition[i], accel, mj);
//     }
//     return accel;
// }

// __global__ void calculate_forces2D(float2* devX, float2* devA, float* devM,  unsigned int numBodies) {
//     extern __shared__ float2 shPosition[];
//     float2* globalX = devX;
//     float2* globalA = devA;
//     float* globalM = devM;
//     float2 myPosition;
//     int gtid = blockIdx.x * blockDim.x + threadIdx.x;
//     myPosition = globalX[gtid];
//     float2 acc = make_float2(0.0f, 0.0f);
//     float mj = globalM[gtid];
//     for (int i = 0, tile = 0; i < numBodies; i += blockDim.x, tile++) {
//         int idx = tile * blockDim.x + threadIdx.x;
//         shPosition[threadIdx.x] = globalX[idx];
//         __syncthreads();
//         acc = tile_calculation2D(myPosition, acc, mj);
//         __syncthreads();
//     }
//     globalA[gtid] = acc;
// }




// TODO 3: Write a kernel that updates the accelerations of all bodies based on the gravitational attraction
// of all other bodies once per frame. This is the kernel which does most of the work.

__global__ void updateAccelerations(float2* positions, float2* accelerations, float* masses, unsigned int numBodies)
{
    extern __shared__ float2 shPosition[];
    float2 myPosition;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    myPosition = positions[gtid];
    float2 acc = make_float2(0.0f, 0.0f);
    for (int i = 0, tile = 0; i < numBodies; i += blockDim.x, tile++) {
        int idx = tile * blockDim.x + threadIdx.x;
        // Cooperatively load the data of many bodies per block into shared memory
        shPosition[threadIdx.x] = positions[idx];
        __syncthreads();
        // Use them for the acceleration calculation in all threads
        for (int j = 0; j< blockDim.x; j++) {
        	float mj =  masses[tile * blockDim.x + j];
            float2 r = shPosition[j] - myPosition;
            float dist2 = r.x * r.x + r.y * r.y + EPS_2;
            float inv_sqrt_dist3 = rsqrtf(dist2 * dist2 * dist2);
            acc += r * (mj * inv_sqrt_dist3);
        }
        __syncthreads();
    }
    accelerations[gtid] = acc * GRAVITY;
}

// Write a kernel that updates the velocities and positions of all bodies based on their accelerations
// once per frame.
// To update a body\u2019s velocity, just add its current acceleration to its velocity.
// To update a body\u2019s position, just add its current velocity to the its position.

__global__ void updateVelocitiesAndPositions(float2* positions, float2* velocities, float2* accelerations, unsigned int numBodies)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies)
    {
        return;
    }

    velocities[i] += accelerations[i];
    positions[i] += velocities[i];
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "Usage: " << argv[0] << " <numBodies>" << endl;
        return 1;
    }
    unsigned int numBodies = atoi(argv[1]);
    unsigned int numBlocks = numBodies / THREADS_PER_BLOCK;
    numBodies = numBlocks * THREADS_PER_BLOCK;

    // allocate memory
    float2* hPositions = new float2[numBodies];
    float2* hVelocities = new float2[numBodies];
    float* hMasses = new float[numBodies];

    // Initialize Positions and speed
    for (unsigned int i = 0; i < numBodies; i++)
    {
        hPositions[i].x = randF(-1.0, 1.0);
        hPositions[i].y = randF(-1.0, 1.0);
        hVelocities[i].x = hPositions[i].y * 0.007f + randF(0.001f, -0.001f);
        hVelocities[i].y = -hPositions[i].x * 0.007f + randF(0.001f, -0.001f);
        hMasses[i] = randF(0.0f, 1.0f) * 10000.0f / (float)numBodies;
    }

    // TODO 1: Allocate GPU memory for
    // - Positions,
    // - Velocities,
    // - Accelerations and
    // - Masses
    // of all bodies and initialize them from the CPU arrays (where available).

    // Allocate GPU memory

    float2* dPositions;
    float2* dVelocities;
    float2* dAccelerations;
    float* dMasses;

    cudaMalloc(&dPositions, sizeof(float2) * numBodies);
    cudaMalloc(&dVelocities, sizeof(float2) * numBodies);
    cudaMalloc(&dAccelerations, sizeof(float2) * numBodies);
    cudaMalloc(&dMasses, sizeof(float) * numBodies);

    // Copy data to GPU

    cudaMemcpy(dPositions, hPositions, sizeof(float2) * numBodies, cudaMemcpyHostToDevice);
    cudaMemcpy(dVelocities, hVelocities, sizeof(float2) * numBodies, cudaMemcpyHostToDevice);
    cudaMemcpy(dMasses, hMasses, sizeof(float) * numBodies, cudaMemcpyHostToDevice);


    // Free host memory not needed again
    delete[] hVelocities; 
    delete[] hMasses;

    // Initialize OpenGL rendering
#ifdef GUI
    initGL();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    GLuint sp = createShaderProgram("white.vs", 0, 0, 0, "white.fs");

    GLuint vb;
    glGenBuffers(1, &vb);
    GL_CHECK_ERROR;
    glBindBuffer(GL_ARRAY_BUFFER, vb);
    GL_CHECK_ERROR;
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * numBodies, hPositions, GL_STATIC_DRAW);
    GL_CHECK_ERROR;
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    GL_CHECK_ERROR;

    GLuint va;
    glGenVertexArrays(1, &va);
    GL_CHECK_ERROR;
    glBindVertexArray(va);
    GL_CHECK_ERROR;
    glBindBuffer(GL_ARRAY_BUFFER, vb);
    GL_CHECK_ERROR;
    glEnableVertexAttribArray(glGetAttribLocation(sp, "inPosition"));
    GL_CHECK_ERROR;
    glVertexAttribPointer(glGetAttribLocation(sp, "inPosition"), 2, GL_FLOAT, GL_FALSE, 0, 0);
    GL_CHECK_ERROR;
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    GL_CHECK_ERROR;
    glBindVertexArray(0);
    GL_CHECK_ERROR;
#endif

    // Calculate
    for (unsigned int t = 0; t < NUM_FRAMES; t++)
    {
        __int64_t computeStart = continuousTimeNs();

        // TODO 3: Update accelerations of all bodies here.
        // Write a kernel that updates the accelerations of all bodies based on the gravitational attraction
        // of all other bodies once per frame.

        updateAccelerations<<<numBlocks, THREADS_PER_BLOCK, sizeof(float2) * THREADS_PER_BLOCK>>>(dPositions, dAccelerations, dMasses, numBodies);
        //calculate_forces2D<<<numBlocks, THREADS_PER_BLOCK, sizeof(float2) * THREADS_PER_BLOCK>>>(dPositions, dAccelerations, dMasses, numBodies);

        // TODO 4: Update velocities and positions of all bodies here.
        updateVelocitiesAndPositions<<<numBlocks, THREADS_PER_BLOCK>>>(dPositions, dVelocities, dAccelerations, numBodies);

        cudaDeviceSynchronize();
        cout << "Frame compute time: " << (continuousTimeNs() - computeStart) << "ns" << endl;

        // TODO 5: Download the updated positions into the hPositions array for rendering.

        cudaMemcpy(hPositions, dPositions, sizeof(float2) * numBodies, cudaMemcpyDeviceToHost);

#ifdef GUI
        // Upload positions to OpenGL
        glBindBuffer(GL_ARRAY_BUFFER, vb);
        GL_CHECK_ERROR;
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * numBodies, hPositions, GL_STATIC_DRAW);
        GL_CHECK_ERROR;
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        GL_CHECK_ERROR;

        // Draw
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        GL_CHECK_ERROR;
        glUseProgram(sp);
        GL_CHECK_ERROR;
        glBindVertexArray(va);
        GL_CHECK_ERROR;
        glDrawArrays(GL_POINTS, 0, numBodies);
        GL_CHECK_ERROR;
        glBindVertexArray(0);
        GL_CHECK_ERROR;
        glUseProgram(0);
        GL_CHECK_ERROR;
        swapBuffers();
#endif
    }

#ifdef GUI
    cout << "Done." << endl;
    sleep(2);
#endif

    // Clean up
#ifdef GUI
    glDeleteProgram(sp);
    GL_CHECK_ERROR;
    glDeleteVertexArrays(1, &va);
    GL_CHECK_ERROR;
    glDeleteBuffers(1, &vb);
    GL_CHECK_ERROR;

    glDeleteProgram(sp);
    exitGL();
#endif

    // TODO 2: Clean up your allocated memory

    delete[] hPositions;
    cudaFree(dPositions);
    cudaFree(dVelocities);
    cudaFree(dAccelerations);
    cudaFree(dMasses);


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
