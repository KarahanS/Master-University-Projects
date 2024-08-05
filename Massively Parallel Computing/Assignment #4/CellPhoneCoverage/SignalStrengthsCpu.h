#ifndef _SignalStrengthsCpu_h_
#define _SignalStrengthsCpu_h_

#include "CellPhoneCoverage.h"

void calculateSignalStrengthsCpu(const PositionList& transmitters, const PositionList& receivers,
                                 SignalStrengthList& signalStrengths);

#endif // _SignalStrengthsCpu_h_
