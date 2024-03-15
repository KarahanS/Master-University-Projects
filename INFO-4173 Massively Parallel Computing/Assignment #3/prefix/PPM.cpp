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
#include <fstream>
#include <iostream>

namespace ppm
{

void munchWhitespace(std::istream& instream)
{
    int charRead = instream.peek();
    while (charRead >= 0 && charRead <= ' ')
    {
        instream.ignore();
        charRead = instream.peek();
    }
}

void munchComments(std::istream& instream)
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

bool readPPM(const char* aFileName, int& oWidth, int& oHeight, float** oData)
{
    std::string name(aFileName);
    return readPPM(name, oWidth, oHeight, *oData);
}

bool readPPM(const std::string& aFileName, int& oWidth, int& oHeight, float*& oData)
{
    std::ifstream ppmFile;
    ppmFile.open(aFileName.c_str(), std::ios::binary | std::ios::in);

    if (!ppmFile)
    {
        std::cerr << "Could not open file " << aFileName.c_str() << " for reading\n";
        return false;
    }

    std::string magic;
    getline(ppmFile, magic, '\n');

    magic = magic.substr(0, 2); // bug workaround, should not be necessary

    unsigned int maxColor;

    munchComments(ppmFile);
    ppmFile >> oWidth;
    munchComments(ppmFile);
    ppmFile >> oHeight;
    munchComments(ppmFile);
    ppmFile >> maxColor;

    ppmFile.get(); // skip separating whitespace

    unsigned int dataSize = oWidth * oHeight * 3 * (maxColor > 255 ? 2 : 1);

    unsigned char* tmpData = new unsigned char[dataSize];

    if (!tmpData)
    {
        std::cerr << "readPPM: memory allocation failed\n";
        return false;
    }

    if (magic.compare("P3") == 0) // plain format
    {
        for (unsigned int i = 0; i < dataSize; i++)
        {
            ppmFile >> tmpData[i];
        }
    }
    else if (magic.compare("P6") == 0) // raw format
    {
        ppmFile.read((char*)tmpData, dataSize);
    }
    else
    {
        std::cerr << "Image data not initialized! Unsupported format." << std::endl;
        return false;
    }

    oData = new float[oWidth * oHeight * 3];
    if (!oData)
    {
        std::cerr << "readPPM: memory allocation failed\n";
        return false;
    }

    /// distribute and convert into floats
    if (maxColor == 0)
        maxColor = 255;

    unsigned int bufidx = 0;
    float* entries = oData;

    for (int i = 0; i < oWidth * oHeight * 3; ++i, ++bufidx)
    {
        entries[i] = (float)tmpData[bufidx];
    }

    delete[] tmpData;

    ppmFile.close();

    return true;
}

bool writePPM(const std::string& aFileName, int aWidth, int aHeight, float* aData,
              const std::string& aMagic)
{
    std::cerr << "writing " << aFileName.c_str() << std::endl;

    std::ofstream ppmFile(aFileName.c_str(), std::ios::binary | std::ios::out);

    if (!ppmFile)
    {
        std::cerr << "Could not open file " << aFileName.c_str() << " for reading\n";
        return false;
    }

    ppmFile << aMagic.c_str() << std::endl;
    ppmFile << aWidth << " " << aHeight << " " << 255 << std::endl;

    unsigned char* pixels = new unsigned char[3 * aWidth * aHeight];

    if (!pixels)
    {
        std::cerr << "writePPM: memory allocation failed\n";
        return false;
    }

    for (int i = 0; i < 3 * aWidth * aHeight; ++i)
    {
        pixels[i] = (unsigned char)aData[i];
    }

    if (aMagic.compare("P6") == 0) // raw format
    {
        ppmFile.write((char*)pixels, aWidth * aHeight * 3);
    }

    delete[] pixels;

    ppmFile.close();

    std::cerr << "done writing" << std::endl;

    return true;
}

} // namespace ppm
