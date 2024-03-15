#ifndef _SignalStrengthsCuda_h_
#define _SignalStrengthsCuda_h_

#include "CellPhoneCoverage.h"

void calculateSignalStrengthsCuda(const PositionList& cpuTransmitters,
                                  const PositionList& cpuReceivers,
                                  SignalStrengthList& cpuSignalStrengths);

#endif // _SignalStrengthsCuda_h_
