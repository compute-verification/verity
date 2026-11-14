// Pre-included into host code by emp_cuda.sh, so a probe built against glibc 2.38 or later runs on an older node.
// glibc 2.38's headers bind strtol and strtoul to __isoc23_strtol and __isoc23_strtoul, and the libstdc++ linked in
// calls arc4random (2.36); an older libc has none of them. The binary defines all three on what every glibc on x86-64
// since 2.25 has. Device code never sees this, so the SASS is the same.
#if !defined(__CUDA_ARCH__) && defined(__x86_64__) && defined(__linux__)
extern "C" {
long compat_strtol_(const char*, char**, int);
unsigned long compat_strtoul_(const char*, char**, int);
__asm__(".symver compat_strtol_,strtol@GLIBC_2.2.5");
__asm__(".symver compat_strtoul_,strtoul@GLIBC_2.2.5");
long compat_getrandom_(void*, unsigned long, unsigned int) __asm__("getrandom");

long compat_isoc23_strtol_(const char* s, char** end, int base) __asm__("__isoc23_strtol");
long compat_isoc23_strtol_(const char* s, char** end, int base) { return compat_strtol_(s, end, base); }
unsigned long compat_isoc23_strtoul_(const char* s, char** end, int base) __asm__("__isoc23_strtoul");
unsigned long compat_isoc23_strtoul_(const char* s, char** end, int base) { return compat_strtoul_(s, end, base); }
unsigned int compat_arc4random_(void) __asm__("arc4random");
unsigned int compat_arc4random_(void) {
  unsigned int x = 0;
  compat_getrandom_(&x, sizeof x, 0);
  return x;
}
}
#endif
