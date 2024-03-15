#ifndef _CellPhoneCoverage_h_
#define _CellPhoneCoverage_h_

#include <vector>

// Transmission power, as measured 0.001 units away from the transmitter
const float transmitterPower = 300.f;
// Lowest possible signal power under which the receiver can pick up the signal
const float minimumSignalPower = 1.f;

const unsigned int BucketsPerAxis = 16;
// The bucket sort should subdivide the whole domain into
// BucketsPerAxis x BucketsPerAxis number of buckets

void checkCUDAError(const char* msg);

struct Position
{
    float x, y;
};

typedef std::vector<Position> PositionList; // Set of positions (either transmitters or receivers)

typedef std::vector<float> SignalStrengthList; // Set of signal-strength values

#endif // _CellPhoneCoverage_h_
