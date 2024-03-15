#include "SignalStrengthsSortedCuda.h"

#include "CellPhoneCoverage.h"
#include "CudaArray.h"
#include "Helpers.h"

#include <iostream>

#include <cuda_runtime.h>

using namespace std;

// "Smart" CUDA implementation which computes signal strengths
//
// First, all transmitters are sorted into buckets
// Then, all receivers are sorted into buckets
// Then, receivers only compute signal strength against transmitters in nearby buckets
//
// This multi-step algorithm makes the signal strength computation scale much
//  better to high number of transmitters/receivers

struct Bucket
{
    int startIndex; // Start of bucket within array
    int numElements; // Number of elements in bucket
};


///////////////////////////////////////////////////////////////////////////////////////////////
//
// No-operation sorting kernel
//
// This takes in an unordered set, and builds a dummy bucket representation around it
// It does not perform any actual sorting!
//
// This kernel must be launched with a 1,1 configuration (1 grid block, 1 thread).

static __global__ void noSortKernel(const Position* inputPositions, int numInputPositions,
                                    Position* outputPositions, Bucket* outputBuckets)
{
    int numBuckets = BucketsPerAxis * BucketsPerAxis;

    // Copy contents of input positions into output positions

    for (int i = 0; i < numInputPositions; ++i)
        outputPositions[i] = inputPositions[i];

    // Set up the set of buckets to cover the output positions evenly

    for (int i = 0; i < numBuckets; i++)
    {
        Bucket& bucket = outputBuckets[i];

        bucket.startIndex = numInputPositions * i / numBuckets;
        bucket.numElements = (numInputPositions * (i + 1) / numBuckets) - bucket.startIndex;
    }
}

// call this with inputPositions.size() many threads in total (I called it with slightly more for safety)
__global__ void histoKernel(Bucket* outputBuckets, const Position* inputPositions, int numInputPositions)
{
    int thid = threadIdx.x + blockIdx.x * blockDim.x;
    if(thid >= numInputPositions) return; // if the thread is out of bounds, return

    Position pos = inputPositions[thid];
    // space is divided into 16x16 buckets
    // we have to find the bucket index of the position for x and y individually

    int bucketIndexX = (int)(pos.x  * BucketsPerAxis);  // positions are between (0, 0) and (1, 1)
    int bucketIndexY = (int)(pos.y * BucketsPerAxis);

    int bucketIndex = bucketIndexY * BucketsPerAxis + bucketIndexX;  // this is how it is calculated in calculateSignalStrengthsSortedKernel
    atomicAdd(&outputBuckets[bucketIndex].numElements, 1);           // we will calculate the starting index in the next kernel (scan)
}

// we will perform a scan operation to calculate the starting index of each bucket
// call this function with one block and (number of buckets) many threads
// const unsigned int BucketsPerAxis = 16 --> therefore there will be 256 threads in total
__global__ void reductionKernel(Bucket* outputBuckets)
{

    extern __shared__ int temp[]; // allocated on invocation
    int thid = threadIdx.x;  // single block

    temp[thid] = outputBuckets[thid].numElements;
    __syncthreads();

    // from prefix.cu
    for(int stride = 1; stride <= blockDim.x / 2; stride <<= 1) {
    		__syncthreads();
 		int index = (thid+1) * stride * 2 - 1;
 		//printf("%d %d %d\n",index, index-stride, thid);
 		if(index < blockDim.x) temp[index] += temp[index-stride];
    }

    outputBuckets[thid].startIndex = temp[thid];

}

// call this with numBuckets + 1 threads -- 0th thread will be 0
__global__ void spreadKernel(Bucket* outputBuckets) {
    extern __shared__ int temp[]; // allocated on invocation
    int thid = threadIdx.x;  // single block

    if(thid != 0) temp[thid] = outputBuckets[thid - 1].startIndex;
    else temp[0] = 0;

    // from prefix.cu
    for(int stride=blockDim.x/2; stride > 0; stride>>=1) {
        __syncthreads();
        int idx = 2 * stride * thid;
        if(idx + stride < blockDim.x) temp[idx + stride] += temp[idx];
    }
    __syncthreads();
    if(thid != 0) outputBuckets[thid - 1].startIndex = temp[thid];
}

// call this with inputPositions.size() many threads in total (I called it with slightly more for safety)
__global__ void scatterKernel(Bucket* outputBuckets, Position* outputPositions, Position* inputPositions, int numInputPositions) {
    int thid = threadIdx.x + blockIdx.x * blockDim.x;

    if(thid >= numInputPositions) return;       // if the thread is out of bounds, return
    Position pos = inputPositions[thid];        // get the position from inputPositions
    int bucketIndexX = (int)(pos.x  * BucketsPerAxis);  // positions are between (0, 0) and (1, 1)
    int bucketIndexY = (int)(pos.y * BucketsPerAxis);

    int bucketIndex = bucketIndexY * BucketsPerAxis + bucketIndexX;  // this is how it is calculated in calculateSignalStrengthsSortedKernel

    // bucket[startIndex] = number of elements up to this bucket (including this one)
    int startIndex = bucketIndex > 0 ? outputBuckets[bucketIndex - 1].startIndex : 0;
    Bucket& bucket = outputBuckets[bucketIndex];

    // copy the position to the correct position in the outputPositions
    __syncthreads();
    // print the position
    //printf("Position: (%.2f %.2f) -> Bucket: %d -> StartIndex: %d\n", pos.x, pos.y, bucketIndex, startIndex);
    int index = atomicAdd(&bucket.numElements, 1);
    outputPositions[startIndex  + index] = pos;
    __syncthreads();

}

// this is used to set numElements to zero, so that we can create outputPositions by incrementing numElements
__global__ void zeroOutBuckets(Bucket* outputBuckets) {
    int numBuckets = BucketsPerAxis * BucketsPerAxis; // calculate the number of buckets
    int thid = threadIdx.x + blockIdx.x * blockDim.x;

    if(thid >= numBuckets) return;       // if the thread is out of bounds, return
    outputBuckets[thid].numElements = 0;
}



// !!! missing !!!
// Kernels needed for sortPositionsIntoBuckets(...)

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Sort a set of positions into a set of buckets
//
// Given a set of input positions, these will be re-ordered such that
//  each range of elements in the output array belong to the same bucket.
// The list of buckets that is output describes where each such range begins
//  and ends in the re-ordered position array.

// Expectation: cudaOutputPositions must be sorted using buckets (initially it is just an empty array of same size with cudaInputPositions)


__global__ void printBuckets(const Bucket* buckets, int numBuckets) {
    for (int i = 0; i < BucketsPerAxis; i++) {
        for(int j = 0; j < BucketsPerAxis; j++)
            printf("(%d, %d)", buckets[i * BucketsPerAxis + j].numElements, buckets[i * BucketsPerAxis + j].startIndex);
        printf("\n");
    }
}

__global__ void printStartBuckets(const Bucket* buckets, int numBuckets) {
    for (int i = 0; i < BucketsPerAxis; i++) {
        for(int j = 0; j < BucketsPerAxis; j++)
            printf("%d ", buckets[i * BucketsPerAxis + j].startIndex);
        printf("\n");
    }
}

__global__ void printOutputPositions(const Position* positions, int numPositions) {
    for (int i = 0; i < numPositions; i++) {
        printf("(%.2f %.2f) ", positions[i].x, positions[i].y);
        if(i % 10 == 0) printf("\n");
    }
    printf("\n");
}

__global__ void updateStartIndex(Bucket* outputBuckets) {
    // for each bucket we will update the start index
    // start index = start index - numElements

    int thid = threadIdx.x + blockIdx.x * blockDim.x;
    if(thid >= BucketsPerAxis * BucketsPerAxis) return;       // if the thread is out of bounds, return


    if(thid > 0) {
        outputBuckets[thid].startIndex = outputBuckets[thid].startIndex - outputBuckets[thid].numElements;
    }
    else {
        outputBuckets[thid].startIndex = 0;
    }
}

static void sortPositionsIntoBuckets(CudaArray<Position>& cudaInputPositions,
                                     CudaArray<Position>& cudaOutputPositions,
                                     CudaArray<Bucket>& cudaOutputPositionBuckets)
{
    // Bucket sorting with "Counting Sort" is a multi-phase process:
    //
    // 1. Determine how many of the input elements should end up in each bucket (build a histogram)
    //
    // 2. Given the histogram, compute where in the output array that each bucket begins, and how
    // large it is
    //    (perform prefix summation over the histogram)
    //
    // 3. Given the start of each bucket within the output array, scatter elements from the input
    //    array into the output array
    //
    // Your new sort implementation should be able to handle at least 10 million entries, and
    //  run in reasonable time (the reference implementations does the job in 200 milliseconds).

    //=================  Your code here =====================================
    // !!! missing !!!

    // Instead of sorting, we will now run a dummy kernel that just duplicates the
    //  output positions, and constructs a set of dummy buckets. This is just so that
    //  the test program will not crash when you try to run it.
    //
    // This kernel is run single-threaded because it is throw-away code where performance
    //  does not matter; after all, the purpose of the lab is to replace it with a
    //  proper sort algorithm instead!

    //========== Remove this code when you begin to implement your own sorting algorithm ==========

    /*

    cudaInputPositions = Array of input positions (input position = (x, y))
    cudaOutputPositions = Array of output positions (output position = (x, y))
    cudaOutputPositionBuckets = Array of buckets (bucket = (startIndex, numElements))

    */

   /*
    IDEA:
        call histo and take buckets (numElements are correct but startIndexs are still empty)
        call reductionKernel for the first part of prefix sum
        call spreadKernel for the second part of prefix sum
        call scatterKernel to scatter from buckets to outputPositions

   */
    int numPositions = cudaInputPositions.size();
    int numBuckets = BucketsPerAxis * BucketsPerAxis;

    int nBlocks = (numPositions + 255) / 256;
    int nThreads = 256;

    histoKernel<<<nBlocks, nThreads>>>(cudaOutputPositionBuckets.cudaArray(), cudaInputPositions.cudaArray(), numPositions);
    // check cuda error

    // print buckets
    //printf("HISTO\n");
    //printBuckets<<<1, 1>>>(cudaOutputPositionBuckets.cudaArray(), numBuckets);

    reductionKernel<<<1, numBuckets, numBuckets * sizeof(int)>>>(cudaOutputPositionBuckets.cudaArray());            // numBuckets = 256 threads

    spreadKernel<<<1, numBuckets + 1, (numBuckets + 1) * sizeof(int)>>>(cudaOutputPositionBuckets.cudaArray());     // numBuckets + 1 = 257 threads
    // print buckets
    //printf("REDUCED\n");
    //printStartBuckets<<<1, 1>>>(cudaOutputPositionBuckets.cudaArray(), numBuckets);

    zeroOutBuckets<<<1, numBuckets>>>(cudaOutputPositionBuckets.cudaArray());               // numElements is set to zero for the next kernel

    scatterKernel<<<nBlocks, nThreads>>>(cudaOutputPositionBuckets.cudaArray(), cudaOutputPositions.cudaArray(), cudaInputPositions.cudaArray(), numPositions);

    // now we have to update the start index of each bucket
    updateStartIndex<<<1, numBuckets>>>(cudaOutputPositionBuckets.cudaArray());
    cudaDeviceSynchronize();




}


///////////////////////////////////////////////////////////////////////////////////////////////
//
// Go through all transmitters in one bucket, find highest signal strength
// Return highest strength (or the old value, if that was higher)

static __device__ float scanBucket(const Position* transmitters, int numTransmitters,
                                   const Position& receiver, float bestSignalStrength)
{
    for (int transmitterIndex = 0; transmitterIndex < numTransmitters; ++transmitterIndex)
    {
        const Position& transmitter = transmitters[transmitterIndex];

        float strength = signalStrength(transmitter, receiver);
        if (bestSignalStrength < strength)
            bestSignalStrength = strength;
    }

    return bestSignalStrength;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Calculate signal strength for all receivers

static __global__ void calculateSignalStrengthsSortedKernel(const Position* transmitters,
                                                            const Bucket* transmitterBuckets,
                                                            const Position* receivers,
                                                            const Bucket* receiverBuckets,
                                                            float* signalStrengths)
{
    // Determine which bucket the current grid block is processing

    int receiverBucketIndexX = blockIdx.x;
    int receiverBucketIndexY = blockIdx.y;

    int receiverBucketIndex = receiverBucketIndexY * BucketsPerAxis + receiverBucketIndexX;

    const Bucket& receiverBucket = receiverBuckets[receiverBucketIndex];

    int receiverStartIndex = receiverBucket.startIndex;
    int numReceivers = receiverBucket.numElements;

    // Distribute available receivers over the set of available threads

    for (int receiverIndex = threadIdx.x; receiverIndex < numReceivers; receiverIndex += blockDim.x)
    {
        // Locate current receiver within the current bucket

        const Position& receiver = receivers[receiverStartIndex + receiverIndex];
        float& finalStrength = signalStrengths[receiverStartIndex + receiverIndex];

        float bestSignalStrength = 0.f;

        // Scan all buckets in the 3x3 region enclosing the receiver's bucket index

        for (int transmitterBucketIndexY = receiverBucketIndexY - 1;
             transmitterBucketIndexY < receiverBucketIndexY + 2; ++transmitterBucketIndexY)
            for (int transmitterBucketIndexX = receiverBucketIndexX - 1;
                 transmitterBucketIndexX < receiverBucketIndexX + 2; ++transmitterBucketIndexX)
            {
                // Only process bucket if its index is within [0, BucketsPerAxis - 1] along each
                // axis

                if (transmitterBucketIndexX >= 0 && transmitterBucketIndexX < BucketsPerAxis
                    && transmitterBucketIndexY >= 0 && transmitterBucketIndexY < BucketsPerAxis)
                {
                    // Scan bucket for a potential new "highest signal strength"

                    int transmitterBucketIndex =
                        transmitterBucketIndexY * BucketsPerAxis + transmitterBucketIndexX;
                    int transmitterStartIndex =
                        transmitterBuckets[transmitterBucketIndex].startIndex;
                    int numTransmitters = transmitterBuckets[transmitterBucketIndex].numElements;
                    bestSignalStrength = scanBucket(&transmitters[transmitterStartIndex],
                                                    numTransmitters, receiver, bestSignalStrength);
                }
            }

        // Store out the highest signal strength found for the receiver

        finalStrength = bestSignalStrength;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////

void calculateSignalStrengthsSortedCuda(const PositionList& cpuTransmitters,
                                        const PositionList& cpuReceivers,
                                        SignalStrengthList& cpuSignalStrengths)
{
    int numBuckets = BucketsPerAxis * BucketsPerAxis;

    // Copy input positions to device memory

    CudaArray<Position> cudaTempTransmitters(cpuTransmitters.size());
    cudaTempTransmitters.copyToCuda(&(*cpuTransmitters.begin()));

    CudaArray<Position> cudaTempReceivers(cpuReceivers.size());
    cudaTempReceivers.copyToCuda(&(*cpuReceivers.begin()));

    // Allocate device memory for sorted arrays

    CudaArray<Position> cudaTransmitters(cpuTransmitters.size());
    CudaArray<Bucket> cudaTransmitterBuckets(numBuckets);

    CudaArray<Position> cudaReceivers(cpuReceivers.size());
    CudaArray<Bucket> cudaReceiverBuckets(numBuckets);

    // Sort transmitters and receivers into buckets

    sortPositionsIntoBuckets(cudaTempTransmitters, cudaTransmitters, cudaTransmitterBuckets);
    sortPositionsIntoBuckets(cudaTempReceivers, cudaReceivers, cudaReceiverBuckets);

    // Perform signal strength computation
    CudaArray<float> cudaSignalStrengths(cpuReceivers.size());

    int numThreads = 256;
    dim3 grid = dim3(BucketsPerAxis, BucketsPerAxis);

    calculateSignalStrengthsSortedKernel<<<grid, numThreads>>>(
        cudaTransmitters.cudaArray(), cudaTransmitterBuckets.cudaArray(), cudaReceivers.cudaArray(),
        cudaReceiverBuckets.cudaArray(), cudaSignalStrengths.cudaArray());

    // Copy results back to host memory
    cpuSignalStrengths.resize(cudaSignalStrengths.size());
    cudaSignalStrengths.copyFromCuda(&(*cpuSignalStrengths.begin()));
}
