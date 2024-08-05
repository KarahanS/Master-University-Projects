#include "CellPhoneCoverage.h"

#include "CudaArray.h"
#include "SignalStrengthsCpu.h"
#include "SignalStrengthsCuda.h"
#include "SignalStrengthsSortedCuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

// Simple utility function to check for CUDA runtime errors

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Generate initial positions for transmitters/receivers
//
// By default, their positions are randomly chosen within the [(0,0) .. (1,1)] 2D interval

PositionList generateRandomPositions(int numPositions)
{
    PositionList positions;

    for (int i = 0; i < numPositions; ++i)
    {
        Position position;
        position.x = ((float)rand()) / RAND_MAX;
        position.y = ((float)rand()) / RAND_MAX;
        positions.push_back(position);
    }

    return positions;
}

PositionList createTransmitters(int numTransmitters)
{
    return generateRandomPositions(numTransmitters);
}

PositionList createReceivers(int numReceivers) { return generateRandomPositions(numReceivers); }

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Given a set of signal strengths, compute various statistics for the set

void calcSignalStrengthStatistics(const SignalStrengthList& strengths, float& minStrength,
                                  float& maxStrength, float& averageStrength,
                                  int& numReceiversWithoutCoverage)
{
    if (strengths.empty())
    {
        minStrength = 0.f;
        maxStrength = 0.f;
        averageStrength = 0.f;
        numReceiversWithoutCoverage = 0;
    }
    else
    {
        minStrength = 1e20;
        maxStrength = 0.f;
        averageStrength = 0.f;
        numReceiversWithoutCoverage = 0;

        for (SignalStrengthList::const_iterator strength = strengths.begin();
             strength != strengths.end(); ++strength)
        {
            if (minStrength > *strength)
                minStrength = *strength;
            if (maxStrength < *strength)
                maxStrength = *strength;

            averageStrength += *strength;

            if (*strength < 0.00001f)
                numReceiversWithoutCoverage++;
        }

        averageStrength /= strengths.size();
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        printf("Usage: CellPhoneCoverage <num transmitters> <num receivers> <method [0, 1, 2]>\n");
        return 1;
    }

    int numTransmitters = atoi(argv[1]);
    int numReceivers = atoi(argv[2]);

    int method = atoi(argv[3]);

    float transmissionRadius = sqrt((transmitterPower * 0.001f * 0.001f) / minimumSignalPower);
    float bucketSideLength = 1.f / BucketsPerAxis;

    if (transmissionRadius > bucketSideLength / 2)
    {
        printf("Error: Transmission radius is %f units, but must not be greater than %f units\n",
               transmissionRadius, bucketSideLength / 2);
        return 1;
    }

    printf("\n");

    // Generate transmitters & receivers

    PositionList cpuTransmitters = createTransmitters(numTransmitters);
    PositionList cpuReceivers = createReceivers(numReceivers);

    // Calculate signal strengths

    SignalStrengthList cpuSignalStrengths;

    switch (method)
    {
    case 0:
        printf("Performing brute-force CPU computation\n");
        calculateSignalStrengthsCpu(cpuTransmitters, cpuReceivers, cpuSignalStrengths);
        break;
    case 1:
        printf("Performing brute-force CUDA computation\n");
        calculateSignalStrengthsCuda(cpuTransmitters, cpuReceivers, cpuSignalStrengths);
        break;
    case 2:
        printf("Performing sorted CUDA computation\n");
        calculateSignalStrengthsSortedCuda(cpuTransmitters, cpuReceivers, cpuSignalStrengths);
        break;
    }

    // Evaluate effectiveness of transmitter/receiver positioning

    float minStrength;
    float maxStrength;
    float averageStrength;
    int numReceiversWithoutCoverage;

    calcSignalStrengthStatistics(cpuSignalStrengths, minStrength, maxStrength, averageStrength,
                                 numReceiversWithoutCoverage);

    printf("Number of transmitters: %d\n", numTransmitters);
    printf("Number of receivers: %d\n", numReceivers);
    printf("Transmission radius for each transmitter: %f units\n", transmissionRadius);
    printf("Min strength: %f W\n", minStrength);
    printf("Max strength: %f W\n", maxStrength);
    printf("Average strength: %f W\n", averageStrength);
    printf("Number of receivers without coverage: %d (%.1f%%)\n", numReceiversWithoutCoverage,
           100.f * numReceiversWithoutCoverage / numReceivers);

    checkCUDAError("end of program");

    return 0;
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
