#ifndef _Helpers_h_
#define _Helpers_h_

#include "CellPhoneCoverage.h"

#include <cmath>

#include <cuda_runtime.h>

using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////////
// Given a transmitter and a receiver, calculate signal strength
//
// The signal strength is proportional to 1/distance^2
// When the signal strength goes beneath the receiver's minimum acceptable strength,
//  the strength returned will be zero.
//
// The __host__ __device__ tags allows it to be called both from CPU and from CUDA code
//  and the function must be inlined because __device__ functions always get inlined
//  during compilation

static inline __host__ __device__ float signalStrength(const Position& transmitter,
                                                       const Position& receiver)
{
    // Calculate distance between transmitter and receiver

    Position positionDelta;
    positionDelta.x = receiver.x - transmitter.x;
    positionDelta.y = receiver.y - transmitter.y;

    float distance = sqrt(positionDelta.x * positionDelta.x + positionDelta.y * positionDelta.y);

    // Ensure that distance never is zero; otherwise the 1/distance^2 calculation would
    //  divide-by-zero

    if (distance < 1e-10)
        distance = 1e-10;

    // Calculate signal power at the receiver's location

    float normalizedPower = 1.f / (distance * distance);

    float powerScale = transmitterPower * (0.001f * 0.001f);

    float strength = powerScale * normalizedPower;

    // Clamp too low signal power to zero

    if (strength < minimumSignalPower)
        strength = 0.f;

    return strength;
}

#endif // _Helpers_h_
