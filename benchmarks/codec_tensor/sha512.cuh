// wt.cu's and reader.cu's SHA-512: the GPU's one (verity/ml/kernels/sha512.cuh), in the form their kernels run, five
// passes of one 16-round body with the schedule in place.
#pragma once
#include "../../verity/ml/kernels/sha512.cuh"

// One SHA-512 compression of the big-endian words w (clobbered) into s.
__host__ __device__ __forceinline__ void sha512_compress(uint64_t s[8], uint64_t w[16]) { sx::compress<0, 2>(s, w); }
