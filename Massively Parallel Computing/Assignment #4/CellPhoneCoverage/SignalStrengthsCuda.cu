#include "SignalStrengthsCuda.h"

#include "CellPhoneCoverage.h"
#include "CudaArray.h"
#include "Helpers.h"

#include <cuda_runtime.h>

// Brute force CUDA implementation which computes signal strengths
//
// It iterates through all transmitter/receiver combinations and, for each receiver, records the
// highest found signal strength

///////////////////////////////////////////////////////////////////////////////////////////////
// Compute maximum signal strength for a set of transmitters/receivers
//
// Each thread evaluates one receiver against all transmitters, and records the highest found
//  strength into the signalStrengths[] array

static __global__ void calculateSignalStrengthsKernel(const Position* transmitters,
                                                      int numTransmitters,
                                                      const Position* receivers, int numReceivers,
                                                      float* signalStrengths)
{
    // Locate the receiver which we should be working on

    int receiverIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (receiverIndex < numReceivers)
    {
        const Position& receiver = receivers[receiverIndex];
        float& finalStrength = signalStrengths[receiverIndex];

        // Iterate through all transmitters

        float bestSignalStrength = 0.f;

        for (int transmitterIndex = 0; transmitterIndex < numTransmitters; ++transmitterIndex)
        {
            const Position& transmitter = transmitters[transmitterIndex];

            // Calculate signal strength between transmitter and receiver

            float strength = signalStrength(transmitter, receiver);

            // If signal strength is higher than for any previously tested transmitter, keep this
            // value

            if (bestSignalStrength < strength)
                bestSignalStrength = strength;
        }

        // Store maximum signal strength for this receiver

        finalStrength = bestSignalStrength;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////

void calculateSignalStrengthsCuda(const PositionList& cpuTransmitters,
                                  const PositionList& cpuReceivers,
                                  SignalStrengthList& cpuSignalStrengths)
{
    // Allocate device memory for input and output arrays

    CudaArray<Position> cudaTransmitters(cpuTransmitters.size());
    CudaArray<Position> cudaReceivers(cpuReceivers.size());
    CudaArray<float> cudaSignalStrengths(cpuReceivers.size());

    // Copy transmitter & receiver arrays to device memory

    cudaTransmitters.copyToCuda(&(*cpuTransmitters.begin()));
    cudaReceivers.copyToCuda(&(*cpuReceivers.begin()));

    // Perform signal strength computation

    int numThreads = std::min(cudaReceivers.size(), 256);
    int numBlocks = (cudaReceivers.size() + numThreads - 1) / numThreads;

    calculateSignalStrengthsKernel<<<numBlocks, numThreads>>>(
        cudaTransmitters.cudaArray(), cudaTransmitters.size(), cudaReceivers.cudaArray(),
        cudaReceivers.size(), cudaSignalStrengths.cudaArray());

    // Copy results back to host memory

    cpuSignalStrengths.resize(cudaSignalStrengths.size());
    cudaSignalStrengths.copyFromCuda(&(*cpuSignalStrengths.begin()));
}
