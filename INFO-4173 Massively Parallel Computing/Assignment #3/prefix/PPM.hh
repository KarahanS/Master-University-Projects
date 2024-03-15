// ==========================================================================
// $Id$
// ==========================================================================
// (C)opyright: 2009
//
//   Ulm University
//
// Creator: Hendrik Lensch, Holger Dammertz
// Email:   hendrik.lensch@uni-ulm.de, holger.dammertz@uni-ulm.de
// ==========================================================================
// $Log$
// ==========================================================================

#ifndef _PPM_H
#define _PPM_H

/*! \file  PPM.hh
    \brief provides ppm reader and writer functions
 */
#include <string>

namespace ppm
{

bool readPPM(const char* aFileName, int& oWidth, int& oHeight, float** oData);
bool readPPM(const std::string& aFileName, int& oWidth, int& oHeight, float*& oData);

bool writePPM(const std::string& aFileName, int aWidth, int aHeight, float* aData,
              const std::string& aMagic = std::string("P6"));

} // namespace ppm

#endif
