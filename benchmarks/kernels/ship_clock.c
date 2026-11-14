// The clock a ship's build runs its compilers under (LD_PRELOAD, from ship.sh). Inside cicc, nvcc's front end, every
// clock reading advances one microsecond, so no optimization decision cicc makes by elapsed time depends on how busy the
// machine is; every other process reads the real clock.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

static int counted = -1;
static unsigned long long readings;

static int in_cicc(void) {
  if (counted < 0) {
    char exe[4096];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    exe[n > 0 ? n : 0] = 0;
    const char *base = strrchr(exe, '/');
    counted = strcmp(base ? base + 1 : exe, "cicc") == 0;
  }
  return counted;
}

static unsigned long long next_ns(void) { return 1000000000ULL + 1000ULL * ++readings; }

int clock_gettime(clockid_t clock, struct timespec *ts) {
  if (!in_cicc()) {
    static int (*real)(clockid_t, struct timespec *);
    if (!real) real = (int (*)(clockid_t, struct timespec *))dlsym(RTLD_NEXT, "clock_gettime");
    return real(clock, ts);
  }
  unsigned long long t = next_ns();
  ts->tv_sec = (time_t)(t / 1000000000ULL);
  ts->tv_nsec = (long)(t % 1000000000ULL);
  return 0;
}

int gettimeofday(struct timeval *tv, void *tz) {
  if (!in_cicc()) {
    static int (*real)(struct timeval *, void *);
    if (!real) real = (int (*)(struct timeval *, void *))dlsym(RTLD_NEXT, "gettimeofday");
    return real(tv, tz);
  }
  unsigned long long t = next_ns();
  if (tv) {
    tv->tv_sec = (time_t)(t / 1000000000ULL);
    tv->tv_usec = (long)(t % 1000000000ULL / 1000);
  }
  return 0;
}

clock_t clock(void) {
  if (!in_cicc()) {
    static clock_t (*real)(void);
    if (!real) real = (clock_t (*)(void))dlsym(RTLD_NEXT, "clock");
    return real();
  }
  return (clock_t)(next_ns() / 1000ULL);
}
