#include "Helpers.h"
#include "SignalStrengthsCpu.h"

// Brute force CPU implementation which calculates signal strengths
//
// It iterates through all transmitter/receiver combinations and, for each receiver, records the
// highest found signal strength

void calculateSignalStrengthsCpu(const PositionList& transmitters, const PositionList& receivers,
                                 SignalStrengthList& signalStrengths)
{
    signalStrengths.clear();

    // Iterate through all receivers

    for (PositionList::const_iterator receiver = receivers.begin(); receiver != receivers.end();
         ++receiver)
    {

        float bestSignalStrength = 0.f;

        // Iterate through all transmitters

        for (PositionList::const_iterator transmitter = transmitters.begin();
             transmitter != transmitters.end(); ++transmitter)
        {

            // Calculate signal strength between transmitter and receiver

            float strength = signalStrength(*transmitter, *receiver);

            // If signal strength is higher than for any previously tested transmitter, keep this
            // value

            if (bestSignalStrength < strength)
                bestSignalStrength = strength;
        }

        // Store maximum signal strength for this receiver

        signalStrengths.push_back(bestSignalStrength);
    }
}
