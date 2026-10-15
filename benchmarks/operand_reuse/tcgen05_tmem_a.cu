// Compile-only (sm_100a): a tcgen05.mma whose A operand is read from Tensor Memory at the address the previous
// tcgen05.mma wrote its D to.  kind::tf32 keeps both D (F32) and A (TF32) at one element per 32-bit TMEM column.
// Between the two MMAs there is a commit, an mbarrier wait and a fence, and no arithmetic instruction; whether the
// datapath layouts make D's entries A's entries element for element is the ISA's Data Path Layout figures, not checked
// here.  The descriptors are kernel parameters, so this kernel is for reading SASS, not for running.
//
// Build: nvcc -O2 -std=c++17 -arch=sm_100a -cubin -o tcgen05_tmem_a.cubin tcgen05_tmem_a.cu && cuobjdump -sass ...

#include <cstdint>

__device__ __forceinline__ void wait_phase(uint32_t bar, uint32_t phase) {
    asm volatile("{ .reg .pred p;\n"
                 "W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "   @!p bra W; }\n" ::"r"(bar),
                 "r"(phase));
}

extern "C" __global__ void __launch_bounds__(128) tmem_a_reuse(uint64_t adesc, uint64_t bdesc, uint32_t idesc,
                                                               uint32_t accumulate, uint32_t* out) {
    __shared__ uint32_t tmem_base;
    __shared__ __align__(8) uint64_t bar;
    const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_base));
    const uint32_t sbar = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    const int warp = threadIdx.x >> 5;
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;\n" ::"r"(sbase));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
    }
    if (threadIdx.x == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(sbar));
    asm volatile("tcgen05.fence::before_thread_sync;\n");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;\n");
    const uint32_t t = tmem_base;

    if (threadIdx.x == 0) {
        asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0;\n"
                     "  tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }\n" ::"r"(t),
                     "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accumulate));
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n" ::"r"(sbar));
    }
    wait_phase(sbar, 0);
    asm volatile("tcgen05.fence::after_thread_sync;\n");

    if (threadIdx.x == 0) {
        asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0;\n"
                     "  tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p; }\n" ::"r"(t + 32),
                     "r"(t), "l"(bdesc), "r"(idesc), "r"(accumulate));
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n" ::"r"(sbar));
    }
    wait_phase(sbar, 1);
    asm volatile("tcgen05.fence::after_thread_sync;\n");

    uint32_t r;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n" : "=r"(r) : "r"(t + 32 + ((warp * 32) << 16)));
    asm volatile("tcgen05.wait::ld.sync.aligned;\n");
    out[threadIdx.x] = r;
    __syncthreads();
    if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;\n" ::"r"(t));
}
