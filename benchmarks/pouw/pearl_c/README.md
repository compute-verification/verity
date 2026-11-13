# pearl_c: the hashing Pearl-C's kernels share, and the stand-ins its tests run on

This directory measures nothing and has no runner. It holds the device hashing that Pearl-C's kernel directories
include, and what lets their tests run on a CPU: host builds of that hashing and a stand-in for the CUDA driver.

| File | What it is |
| --- | --- |
| `sha512.cuh` | Pearl-C's names for the GPU's one SHA-512 (`verity/definitions/kernels/sha512.cuh`): the round constants, one compression and the byte-order helpers |
| `hash_h3.cuh` | hashing format -h3, SHA-512 throughout: the labels, A's row digests, the tile leaves and the commitment trees, as `verity.protocols.pouw.prover.schemes.pearl_c` defines them |
| `hash_host.cpp` | `hash_h3.cuh` and `../pearl_c_sm120/hash_fast.cuh` built for the host: a grid's blocks run in order, a block's threads in order or under one barrier |
| `simt_twin.py` | a host build of the same kernels that runs a block's CUDA threads thread for thread under one barrier (`build()`), for byte tests without a GPU |
| `host_threads.h` | a CUDA block's threads as contexts on one host thread, for `simt_twin.py` and `../pearl_c_sm120/hash_decode_host.cpp` |
| `fake_cuda.c` | a CPU stand-in for libcuda: device memory is host memory, copies are bounds-checked, kernels do nothing, and every allocation, stream and launch is logged as a JSON line to `$FAKE_LOG` |

## Who reads it

- `../pearl_c_sm120/pearl_c_sm120.cu` and `../pearl_c_sm90/pearl_c_sm90.cu` include `hash_h3.cuh`; `../pearl_c4/`,
  `../masked_gemm/` and `../filler/` include `sha512.cuh`. A change to either header changes those kernels' builds, so
  they change only with their kernels' gates.
- The tests in `../tests/` build `hash_host.cpp` and `fake_cuda.c` with the host's compiler, and the Pearl-C runners'
  dry runs load the stand-in in libcuda's place (`PEARLC_LIBCUDA`). `../pearl_c_sm120/fake_tma.c` extends it.

Nothing here touches a GPU or computes a number: the host builds exist to hold the device hashing to the scheme's
bytes, which the tests compare.
