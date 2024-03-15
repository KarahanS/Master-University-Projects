#ifndef _SignalStrengthsSortedCuda_h_
#define _SignalStrengthsSortedCuda_h_

#include "CellPhoneCoverage.h"

void calculateSignalStrengthsSortedCuda(const PositionList& cpuTransmitters,
                                        const PositionList& cpuReceivers,
                                        SignalStrengthList& cpuSignalStrengths);

#endif // _SignalStrengthsSortedCuda_h_
