/*
 * Tools.h
 *
 *  Created on: 29.11.2010
 *      Author: Benjamin Resch
 */

#ifndef TOOLS_H_
#define TOOLS_H_

#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <vector>

#define MAX_INT 0x7FFFFFFF
#define MAX_UINT 0xFFFFFFFF
#define MAX_LONGLONG 0x7FFFFFFFFFFFFFFF
#define MAX_ULONGLONG 0xFFFFFFFFFFFFFFFF

#define UNDEFINED MAX_UINT

#define SAFE_DELETE(p)                                                                             \
    if (p)                                                                                         \
    {                                                                                              \
        delete p;                                                                                  \
        p = NULL;                                                                                  \
    }
#define SAFE_DELETE_ARRAY(p)                                                                       \
    if (p)                                                                                         \
    {                                                                                              \
        delete[] p;                                                                                \
        p = NULL;                                                                                  \
    }

#define CLEAR(s) memset(&(s), 0, sizeof(s))

#define ABS(X) ((X) >= 0 ? (X) : -(X))
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define CLAMP(X, LOW, HIGH) (MIN(MAX((LOW), (X)), (HIGH)))
#define DIV_RU(X, Y) (X % Y == 0 ? X / Y : X / Y + 1)
#define SUFFICE(X, Y) (DIV_RU(X, Y) * Y)

#define CONST_UNDEFINED 0xFFFFFFFF
#define CONST_GLOBALLINK 0xFFFFFFFE

inline __int64_t continuousTimeNs()
{
    timespec now;
    clock_gettime(CLOCK_REALTIME, &now);

    __int64_t result = (__int64_t)now.tv_sec * 1000000000 + (__int64_t)now.tv_nsec;

    return result;
}

/**
 * Checks for errors and exits with a notice if errors appeared.
 * @param noProblem True if there was no error.
 * @param name Text which should be shown if an error appeared.
 */
inline void fail(bool noProblem, const char* name)
{
    if (!noProblem)
    {
        std::cerr << "CRITICAL ERROR: " << name << std::endl;
        assert(false); // Make the debugger stop...
        exit(EXIT_FAILURE);
    }
}

template <typename T> class vectorInit
{
    std::vector<T> v;

public:
    vectorInit(const unsigned capacity = 0) { v.reserve(capacity); }

    vectorInit& Add(const T& t)
    {
        v.push_back(t);
        return *this;
    }

    operator std::vector<T>() const { return v; }
};

#endif /* TOOLS_H_ */
