// A CUDA block's threads on the host, for the kernels' host builds (pearl_c_sm120/hash_decode_host.cpp, simt_twin.py).
// `run` runs a body for each thread index as a context on the calling thread, in index order, each until it calls `yield`
// (waiting at a barrier) or returns, so a barrier costs a context switch whatever else holds the machine. A barrier counts
// each arrival in `progress`; a round over the block in which no thread arrives or returns is a deadlock, which `run`
// returns (2) instead of hanging. The switch is register-only on x86-64, swapcontext elsewhere (whose signal-mask system
// call costs more than the hashing between barriers).
#pragma once
#include <sys/mman.h>
#include <ucontext.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__x86_64__)
// the callee-saved registers, MXCSR and the x87 control word pushed on the stack being left, whose pointer goes to *from
asm(".text\n.p2align 4\n.hidden host_threads_switch\n.globl host_threads_switch\n.type host_threads_switch,@function\n"
    "host_threads_switch:\n"
    "pushq %rbp\npushq %rbx\npushq %r12\npushq %r13\npushq %r14\npushq %r15\nsubq $16, %rsp\nstmxcsr 8(%rsp)\nfnstcw (%rsp)\n"
    "movq %rsp, (%rdi)\nmovq %rsi, %rsp\n"
    "fldcw (%rsp)\nldmxcsr 8(%rsp)\naddq $16, %rsp\npopq %r15\npopq %r14\npopq %r13\npopq %r12\npopq %rbx\npopq %rbp\nret\n"
    ".size host_threads_switch, .-host_threads_switch\n");
extern "C" void host_threads_switch(void** from, void* to);
#endif

namespace host_threads {

inline unsigned long progress;
inline void (*g_body)(void*);
inline void* g_arg;
void entry();

#if defined(__x86_64__)
struct Thread {
  void* sp;
  bool done = false;
};
inline void* g_sched;
inline Thread* self;
inline void yield() { host_threads_switch(&self->sp, g_sched); }
inline void resume(Thread& t) { host_threads_switch(&g_sched, t.sp); }
inline void start(Thread& t, char* stack, size_t size) {   // a frame the switch's return enters `entry` from
  auto* sp = reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(stack + size) & ~uintptr_t{15});
  *--sp = 0;
  *--sp = reinterpret_cast<uint64_t>(&entry);
  for (int r = 0; r < 6; r++) *--sp = 0;
  sp -= 2;
  sp[0] = 0x037F;   // the x87 control word and MXCSR a thread starts with: their defaults
  sp[1] = 0x1F80;
  t.sp = sp;
}
inline void finish() { yield(); }
#else
struct Thread {
  ucontext_t ctx;
  bool done = false;
};
inline ucontext_t g_sched;
inline Thread* self;
inline void yield() { swapcontext(&self->ctx, &g_sched); }
inline void resume(Thread& t) { swapcontext(&g_sched, &t.ctx); }
inline void start(Thread& t, char* stack, size_t size) {
  getcontext(&t.ctx);
  t.ctx.uc_stack.ss_sp = stack;
  t.ctx.uc_stack.ss_size = size;
  t.ctx.uc_link = &g_sched;
  makecontext(&t.ctx, entry, 0);
}
inline void finish() {}
#endif

inline void entry() {
  g_body(g_arg);
  self->done = true;
  progress++;
  finish();
}

// body(arg) for each of n threads, *index the running one's; 0 once every thread returned, 2 on a deadlock, 3 without stacks
inline int run(int n, unsigned* index, void (*body)(void*), void* arg) {
  constexpr size_t STACK = size_t{1} << 20, PAGE = 4096;   // each thread's stack, above a guard page
  static char* pool = nullptr;
  static int pooled = 0;
  if (n > pooled) {
    if (pool) munmap(pool, (STACK + PAGE) * pooled);
    void* m = mmap(nullptr, (STACK + PAGE) * n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (m == MAP_FAILED) {
      pool = nullptr;
      pooled = 0;
      return 3;
    }
    pool = static_cast<char*>(m);
    pooled = n;
    for (int t = 0; t < n; t++) mprotect(pool + (STACK + PAGE) * t, PAGE, PROT_NONE);
  }
  g_body = body;
  g_arg = arg;
  std::vector<Thread> ts(n);
  for (int t = 0; t < n; t++) start(ts[t], pool + (STACK + PAGE) * t + PAGE, STACK);
  for (int left = n; left;) {
    const unsigned long before = progress;
    for (int t = 0; t < n; t++) {
      if (ts[t].done) continue;
      self = &ts[t];
      *index = (unsigned)t;
      resume(ts[t]);
      left -= ts[t].done;
    }
    if (left && progress == before) return 2;
  }
  return 0;
}

}  // namespace host_threads
