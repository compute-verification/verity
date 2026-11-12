// Pre-included into host code by emp_cuda.sh, so a probe built against glibc 2.38 or later runs on an older node:
// glibc 2.38's headers bind strtol (atoi's body) to __isoc23_strtol, which an older libc doesn't have, and this binds
// it back to the strtol every glibc on x86-64 has. Device code never sees it, so the SASS is the same.
#if !defined(__CUDA_ARCH__) && defined(__x86_64__) && defined(__linux__)
__asm__(".symver __isoc23_strtol,strtol@GLIBC_2.2.5");
#endif
