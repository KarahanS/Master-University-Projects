#ifndef GLTOOLS_H_
#define GLTOOLS_H_

#define GL_GLEXT_PROTOTYPES
#include "glcorearb.h"

typedef unsigned int GLuint;

void initGL();
void exitGL();
GLuint createShaderProgram(const char* vsFilename, const char* tcFilename, const char* teFilename,
                           const char* gsFilename, const char* fsFilename);
void swapBuffers();
void glCheckError(const char* file, unsigned int line);

#ifdef DEBUG
#define GL_CHECK_ERROR glCheckError(__FILE__, __LINE__)
#else
#define GL_CHECK_ERROR
#endif

#endif
