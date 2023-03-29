// These code is get from https://raw.githubusercontent.com/Michaelangel007/buddhabrot/master/buddhabrot.cpp
//

#ifndef _WIN_GETTIMEOFDAY
#define _WIN_GETTIMEOFDAY

#if defined(WIN32) || defined(_WIN32)

#ifdef __cplusplus
extern "C" {
#endif

// *sigh* Microsoft has this in winsock2.h because they are too lazy to put it in the standard location ... !?!?
typedef struct timeval {
    long tv_sec;
    long tv_usec;
} timeval;

int gettimeofday(struct timeval * tp, struct timezone * tzp);

#ifdef __cplusplus
}
#endif

#endif

#endif
