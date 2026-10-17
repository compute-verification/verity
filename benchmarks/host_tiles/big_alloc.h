// Large buffers on transparent huge pages (node 2's THP is `madvise`): 2 MiB pages keep the L2 streamer running past what
// would be 4 KiB page boundaries and spare the TLB.  HOST_TILES_THP=0 asks for small pages instead (a diagnostic).
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

static int thp_on(void) {
  const char *e = getenv("HOST_TILES_THP");
  return !(e && !atoi(e));
}

static void *big_alloc(size_t n) {
  size_t align = (size_t)2 << 20, len = (n + align - 1) / align * align;
  uint8_t *p = mmap(NULL, len + align, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) { perror("mmap"); exit(5); }
  uint8_t *a = (uint8_t *)(((uintptr_t)p + align - 1) & ~(uintptr_t)(align - 1));
  madvise(a, len, thp_on() ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
  return a;
}
