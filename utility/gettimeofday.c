// These code is get from https://raw.githubusercontent.com/Michaelangel007/buddhabrot/master/buddhabrot.cpp
//

#if defined(WIN32) || defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h> // Windows.h -> WinDef.h defines min() max()
#include <stdint.h> // portable: uint64_t   MSVC: __int64 
#include "gettimeofday.h"

// *sigh* Microsoft has this in winsock2.h because they are too lazy to put it in the standard location ... !?!?
// typedef struct timeval {
//     long tv_sec;
//     long tv_usec;
// } timeval;

// *sigh* no gettimeofday on Win32/Win64
int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
    // FILETIME Jan 1 1970 00:00:00
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  nSystemTime;
    FILETIME    nFileTime;
    uint64_t    nTime;

    GetSystemTime( &nSystemTime );
    SystemTimeToFileTime( &nSystemTime, &nFileTime );
    nTime =  ((uint64_t)nFileTime.dwLowDateTime )      ;
    nTime += ((uint64_t)nFileTime.dwHighDateTime) << 32;

    tp->tv_sec  = (long) ((nTime - EPOCH) / 10000000L);
    tp->tv_usec = (long) (nSystemTime.wMilliseconds * 1000);
    return 0;
}

#endif
