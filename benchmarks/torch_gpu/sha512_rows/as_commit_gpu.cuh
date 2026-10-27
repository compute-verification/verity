// Force-included into commit_gpu.cu (nvcc -include, -DSX_V/-DSX_ROLL from engine.variant_flags; ruler.sh): commit_gpu's
// sha512.cuh as is, but with sha::compress sha512x.cuh's compress<SX_V, SX_ROLL>, so reference.py's check gates the variant
// on every committer path through sha::compress and `commit_gpu sha` times it on commit_gpu's own ruler. compress_from,
// compress_kw and the host's digest stay commit_gpu's.
#pragma once
#include <cuda_runtime.h>
#define compress compress_commit_gpu
#include "../../pouw/commit_gpu/sha512.cuh"
#undef compress
#include "sha512x.cuh"

#ifndef SX_V
#define SX_V 0
#endif
#ifndef SX_ROLL
#define SX_ROLL 1
#endif

namespace sha {
HD void compress(uint64_t* st, uint64_t* w) { sx::compress<SX_V, SX_ROLL>(st, w); }
}  // namespace sha
