// ==========================================================================
// $Id$
// ==========================================================================
// (C)opyright:
//
//   Ulm University
//
// Creator: Hendrik Lensch
// Email:   hendrik.lensch@uni-ulm.de
// ==========================================================================
// $Log$
// ==========================================================================

#ifndef PPM_H
#define PPM_H

namespace ppm
{

bool readPPM(const char* _fname, int& _w, int& _h, float** _data);

bool writePPM(const char* _fname, int _w, int _h, float* _data);

} // namespace ppm

#endif /* PPM_H */
