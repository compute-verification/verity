// pd.cubin: pd.cuh's launches beside the hash_h3.cuh and hash_fast.cuh kernels they share tables with (s512_heads fills
// this module's header tables).  run.sh builds it: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -cubin
#include "pd.cuh"
