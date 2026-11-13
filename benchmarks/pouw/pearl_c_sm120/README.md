# pearl_c_sm120: Pearl-C's kernels on the RTX PRO 6000, checked bit for bit

Pearl-C's device kernels for sm_120a (`pearl_c_sm120.cu`: the served variant `sm120`, G = 4, and `sm120-unpromoted`),
with hashing format -h3, SHA-512 throughout. `run.py` checks every buffer the kernels write against the CPU reference
(`verity.protocols.pouw.prover.schemes.pearl_c`) before it times anything. Its timing is diagnostic, phase by phase; a
number of record is the GPU harness's (`../harness/`), and the served one is `../pearl_c_vllm/`'s.

## The runner, `run.py`

`run.py` runs on one RTX PRO 6000 from a tree built off the card, through the CUDA driver API with the standard library
alone. `card.sh` is the command of its `research run`: it takes the lease, binds to the GPU's NUMA node and starts
`run.py` twice, the check alone on both variants and then the check and the timing. Its header has the full command.

~~~text
bash benchmarks/pouw/pearl_c_sm120/build.sh [OUT]   # off the card, nvcc 12.8 or later; default /tmp/pearl-c-sm120-ship
bash benchmarks/pouw/pearl_c_sm120/card.sh          # under research run, with OUT sent as ship.tar
python3 run.py [check.json]                         # in the built tree, on the card
~~~

- `build.sh` compiles the cubin and refuses it unless `sass_gate.py` passes: no ptxas warning, spill or stack frame,
  every FP8 atom and every peel step the intended tensor-core instruction in the stated counts, and no other MMA form.
  It writes `fixture.py`'s inputs and expected buffers, the toolchain's command line and the tree's `MANIFEST.sha256`.
- `run.py` first refuses any GPU but the expected one (one visible device, compute capability 12.0, its UUID). It then
  runs every kernel on the fixture and compares each buffer byte for byte, the GEMMs again on all, one and three CTAs.
  A variant is timed only when all its buffers match. It exits 3 on a refused identity and 4 on a failed check.
- The timing, at each of `SHAPES`, is each kernel's median over `REPS`, the call's phases and the call end to end
  with and without its hashes, with consistency checks between the kernels that write the same words.

`run.py` prints one JSON object a line as each part finishes and a summary on stderr. A `card.sh` run writes, under
its `$OUT`: `ship.sha256`, `check.jsonl` and `check.txt`, `run.jsonl` and `run.txt`, `load.txt` and, when bound,
`numa.json`.

## The other files

| File | What it is |
| --- | --- |
| `mainloop_sm120.cuh` | the TMA-fed mainloop of the 128 x 128 GEMMs |
| `hash_fast.cuh` | the row digests and tile leaves of `../pearl_c/hash_h3.cuh` in two launches: a thread a 1 KiB segment, then a thread a row or tile for the fold |
| `hash_decode.cuh`, `hash_decode_host.cpp` | decode's hashing in the served pipeline (A's whole commitment and E_A's lines in one launch, and the tile leaves), with its host build for byte tests |
| `fake_tma.c` | the CPU stand-in for libcuda (`../pearl_c/fake_cuda.c`) with the tensor-map entry point the mainloop's host side calls |

## Who reads it

The vLLM adapter loads `run.py` and `fixture.py` by path as the served kernel and names the variant by the sha256 of
those two files, so a change to either is a new variant, which serves only once it has passed its own gate.
`../pearl_c_sm90/` includes `hash_fast.cuh` and `hash_decode.cuh`; `../pearl_c_vllm/` and `../served_zk/` build and load
this tree; the tests in `../tests/` run `run.py` on the stand-in driver and the hashing on its host builds. `run.py`
reads the environment itself, because it runs outside the tree; `../settings.py` declares each variable. Nothing here
is a verifier, and nothing but `run.py` touches a GPU.
