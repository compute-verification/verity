// The SHA-512 kernels alone (hash_h3.cuh's and hash_fast.cuh's), for eta.py's `sha2` mode: built per launch-bound variant
// (-DRS_FAST_NT=… -DRS_FAST_MINB=…) with build.sh's flags, so each variant's registers and speed are measured without
// rebuilding the ship.
#include "../pouw/pearl_c/hash_h3.cuh"
#include "../pouw/pearl_c_sm120/hash_fast.cuh"
