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

namespace ppm
{

bool readPPM(const char* _fname, int& _w, int& _h, float** _data);

bool writePPM(const char* _fname, int _w, int _h, float* _data);

} // namespace ppm

#endif
