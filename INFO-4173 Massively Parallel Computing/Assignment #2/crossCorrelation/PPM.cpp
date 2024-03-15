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

#include "PPM.hh"

#include <cstddef>
#include <fstream>
#include <iostream>

using namespace std;

namespace ppm
{

void munchWhitespace(istream& instream)
{
    int charRead = instream.peek();
    while (charRead >= 0 && charRead <= ' ')
    {
        instream.ignore();
        charRead = instream.peek();
    }
}

void munchComments(istream& instream)
{
    munchWhitespace(instream);
    int charRead = instream.peek();
    while (charRead >= 0 && charRead == '#')
    {
        while ((charRead != '\n') && (charRead != instream.eof()))
        {
            charRead = instream.get();
        }
        charRead = instream.peek();
    }
}

/** default constructor */
bool readPPM(const char* _fname, int& _w, int& _h, float** _data)
{

    std::ifstream in(_fname);

    char P;
    in >> P;
    char five;
    in >> five;
    if (P != 'P' || five != '6')
    {
        cerr << "could not open PPM file " << _fname << endl;
        return false;
    }

    int maxVal;

    munchComments(in);
    in >> _w;
    munchComments(in);
    in >> _h;
    munchComments(in);
    in >> maxVal;

    in.get(); // skip separating whitspace

    int width = _w;
    int height = _h;

    cout << "w h: " << _w << " " << _h << endl;

    unsigned int numChars = 3 * width * height * (maxVal > 255 ? 2 : 1);
    unsigned char* buf = new unsigned char[numChars];
    unsigned int bufidx = 0;

    in.read((char*)buf, numChars);

    size_t d_nPixels = _w * _h;

    *(_data) = new float[3 * d_nPixels];

    /// distribute and convert into floats

    if (maxVal == 0)
        maxVal = 255;

    float* entries = *_data;

    for (unsigned int i = 0; i < d_nPixels * 3; ++i)
    {
        entries[i] = (float)buf[bufidx];
        bufidx += 1;
    }

    delete[] buf;

    cerr << "read fine" << endl;

    return true;
}

bool writePPM(const char* _fname, int _w, int _h, float* _buf)
{

    cerr << "writing " << _fname << endl;

    FILE* f = fopen(_fname, "wb");

    if (!f)
    {
        cerr << "error: could not write file " << _fname << endl;
        return false;
    }

    fprintf(f, "P6\n%d %d\n255\n", _w, _h);

    int nPix = _w * _h;

    unsigned char* buf = new unsigned char[3 * nPix];

    for (int i = 0; i < 3 * nPix; ++i)
    {
        buf[i] = (unsigned char)_buf[i];
    }

    //   // the buffer contents
    int result = fwrite(buf, nPix * 3 * sizeof(unsigned char), 1, f);

    delete[] buf;

    fclose(f);

    if (result != 1)
    {
        cerr << "error: while writing " << _fname << endl;
        return false;
    }

    cerr << "done writing" << endl;

    return true;
}

} // namespace ppm
