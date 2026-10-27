// hash_prefill.cuh's kernels (PL_INSTANCE(851), as pearl_c_sm90.cu builds them) for the host, beside hash_decode_host.cpp's
// hash_h3.cuh and hash_decode.cuh kernels, under its `pd_run` (tests/test_pearl_c_sm90_prefill.py)
//   g++ -O2 -std=c++20 -pthread -shared -fPIC -o hash_prefill_host.so hash_prefill_host.cpp
#define PD_SM90
#define PD_HOST_EXTRA "../pearl_c_sm90/hash_prefill.cuh"
#define PD_HOST_EXTRA_INSTANCES PL_INSTANCE(851)
#define PD_HOST_EXTRA_KERNELS PL_KERNELS(851)
#include "../pearl_c_sm120/hash_decode_host.cpp"
