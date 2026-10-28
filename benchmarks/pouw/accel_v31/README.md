# accel_v31: Dither's no-fold call on the GPU

The GPU prover of one Dither call, the binding that runs it a call at a time, and the CPU reference and verifier checks
that judge its output bit for bit. The call is `verity/core/protocols/pouw/circuit/dither.py`'s:

- m rows of A against n rows of B, at contraction K;
- HYB noise in noise tiles of L′ inner indices;
- the prefix sums T_s after each segment of L inner indices (S = ⌈K / L⌉ segments);
- and, for a served call, its output O mod 2^32.

The call's values are committed as hm96 rows (hiding leaves) in one frame-v3-sha512 tree, before the check coin. The
verifier then opens the proof blocks that the check coin draws. A served call commits T_1 .. T_{S−1} and O, since O
implies T_S. A filler call commits T_1 .. T_S over its pad. `benchmarks/pouw/highutil` runs its periods through this
binding.

## Layout

- `v31.cu`: the prover, one call on the GPU, in four steps:
  - the call's coins, from its six seeds: d_A, d_B, U, V, the noise tiles' permutations, and a filler call's pad;
  - X and W′;
  - the protocol GEMM's prefix sums on the tensor cores, and a served call's O;
  - the commit, on `commit_gpu`'s persistent path, which runs beside the next call's compute.

  Its header lists the kernels, the modes and the builds: the `v31` bench binary, and `libv31.so` (`-DV31_LIB`).
- `wg_proto.cuh`: the protocol GEMM on the H200's warpgroup MMAs (sm_90a, `-DV31_WG=1`).
- `v31lib.py`: the ctypes binding over `libv31.so`, with one `Prover` per shape, and the GPU gate (`gate`).
- `v31.py`: the reference and the verifier's checks, plus the tools that read `v31`'s output. It is a client of `dither.py`
  and adds numpy versions of dither's noise and values.
- `fill_dither.py`: checks the GPU's calls against `dither.py` itself, rather than against `v31.py`'s numpy versions.
- `run.sh`: the builds, gates and benches under a GPU lease, through `research run`. Its header has the command and
  every knob.
- `h200_step.py`: the H200's step, from `v31 fill-bench`'s JSON lines.
- `h200_pipes.py`: each pipe's duty over an overlapped call, from Nsight Compute.
- `duty.metrics`: an Nsight Compute metric list, read by `benchmarks/torch_gpu/sha512_rows/beside.sh`.

## API

The binding and the verifier, on one served call (run from the source tree's root with `PYTHONPATH=.`). The `Prover`'s
docstring describes the call order, the salts and the knobs; `fill_check`'s describes each check.

~~~python
from v31 import CallNoise, Fill, fill_buffer, fill_commit, fill_picks, fill_values, fill_verdict, served_seeds
from v31lib import Prover, build_lib

p = Prover(m, n, k, lp=256, L=2048, served=True, lib=build_lib("libv31.so"))
p.set_weights(B)                               # n x k int8 (or p.point_weights(device_ptr))
seeds = served_seeds(coin, receipt, call)      # dither.seeds; a filler's are call_seeds(salt, coin)
c = p.prove(seeds, p.load_a(A))                # returns at once; a filler call passes no A and proves its pad
root = p.root(c)                               # the value root, once the call's commit is done

picks = fill_picks(p.f, check_coin, q)         # q proof blocks (i, j, s)
ok, fails, line = fill_verdict(p.f, seeds, root, picks, lambda i, j, s: {**p.answer(c, i, j, s), "A": A[i]}, "call")

f = Fill(m, n, k, o=True)                      # dither.Shape at the prover's defaults: L′ 256, L 2,048, rows of 16 KiB
T, O = fill_values(f, A, B, CallNoise.expand(f, seeds))
assert fill_commit(f, fill_buffer(f, T, O), p.ks, p.calls[c]["off"]).root == root
~~~

## Commands

The `v31` binary, and the Python tools run from the source tree's root with `PYTHONPATH=.`:

~~~text
v31 fill-bench REPS OUT PDIR LP L ROW NAME:M:N:K ...  per shape, a checked call and then the timed arms, one JSON line each
v31 fill-once PDIR LP L ROW NAME:M:N:K                one shape's calls, serial and then overlapped, for a profiler
v31 attrs                                             the kernels' registers and shared memory, as JSON
v31.py fill-params DIR NAME:M:N:K ... [--served]      the CGP1 params that fill-bench and fill-once read
v31.py fill-sample DIR                                a fill-bench shape's two sampled tiles against the reference
v31.py fill-table BENCH.jsonl ... [--bar B]           γ over cuBLASLt and CUTLASS, and where the call's time goes
v31.py timeline TRACE.csv                             a kernel timeline of fill-once's calls
v31lib.py gate --lib LIB [--q 64]                     the GPU's values and tree against the reference, and the verdicts
fill_dither.py --lib LIB [--cells 64]                 the GPU's calls against dither.py
h200_step.py BENCH.jsonl ... [--wgmma S]              the H200 step's seconds and γ
h200_pipes.py OUT_DIR                                 each pipe's duty over an overlapped call
~~~

The CPU tests are `benchmarks/pouw/tests/test_accel_v31.py` and `test_accel_v31_h200.py`, in the `benchmarks/pouw` suite
(`uv run tools/verity/check/suites.py benchmarks/pouw`). The GPU gates run under `run.sh`.
