# benchmarks/hash/dither

PoUW's served linear in compiled vLLM, on int8-exact Llama-3.1-8B. Each arm replaces the int8 linear with kernels that
compute the served y and, from the same operands, the window's tables, seeds and receipts (H1 + fx1's, or J10-R's
per-call noise). An arm is gated in its own run before it is timed: its roots, receipts and z against its CPU twin and the
Definition (`h1tile.reference` and `h1tile.j10r_reference`, `verity/core/protocols/pouw/circuit/h1tile.py`), a no-write
control refused and its faults caught. It is then timed against stock vLLM int8, rep by rep. Nothing a verifier runs reads
this directory.

## The arm of record

`sbench.sh` builds the libraries the arms load for the card (`SBENCH_ARCH`: 120a, else 100a or 90a) and runs the served
bench, `sbench.py`, on one leased GPU from the source tree's root. Extra arguments go to `sbench.py`:

~~~sh
RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=WHO bash benchmarks/hash/dither/sbench.sh --arms stock,exact,dither-1s --gate --ops decode --t 64
~~~

Each arm is a resident vLLM engine in its own worker process on that GPU, and the parent alternates reps between them.
`--gate` first holds each arm's greedy tokens to `stock`'s and runs its window's checks; then each rep times decode at
batch 32 (`tpot_b32_ms`) and, with `--ops sat`, tokens a second at saturation. The run writes `summary.json` and
`stop.json` under `$RESEARCH_RUN_DIR/out`, the layout `benchmarks/harness`'s `served_table` reads. `sbench.py`'s
docstring lists every arm and flag, and `settings.py` every variable they read (`DITHER_`, `SBENCH_`, `HB_`).

What the rest of the tree reads from here:

- `h1u.cu`: the proof-block kernels on the GPU. H1 + fx1's seeds and proof blocks (`h1_seeds`, `h1u`, the one-stream
  `kh_one`) and J10-R's (`kh_seeds_j10r`, `kh_units_multi_j10r` and its `_sm` form). `benchmarks/pouw/served_h1` and
  `benchmarks/torch_gpu/j10r_sm90` build it, `verity/core/protocols/pouw`'s tests read it, and
  `verity/core/protocols/pouw/circuit/h1device.py` states the domain it serves on sm_90.
- `j10r_ws.py`: a J10-R call's workspace, sized from the loaded build's own flags (`workspace_bytes`).
- `j10r_units.py`: J10-R's proof blocks on the host, each receipt bit for bit `h1tile.j10r_reference`'s.
- `forms.py`: every arm's form label, from one function (`form_of`), which each arm's stats, checks and gate verdicts carry.
- The arms' window modules, below, and `h1c.*` and `h1r.*`, which the `benchmarks/pouw` suite imports on the CPU.

## The arms

Each arm's module patches vLLM before the engine is built (`patch`), and its window hooks the engine's steps after
(`dither_fused.hook`, or `dither_one.hook` for the one-stream and dither-a arms), so its launches are captured in vLLM's
CUDA graphs:

~~~python
import dither_one as DO
DO.patch(t=T, h1c="libh1c.so", h1o="libh1o.so")   # before LLM(): the int8 linear becomes one op opaque to Inductor
llm = LLM(model=int8_exact_checkpoint, ...)
DO.hook(llm)                                      # the window's tables, trees and round around each step
~~~

- `dither_fused.py` (arms `dither`, `dither-side`): y from `h1f.cu`'s one GEMM, which also computes the previous step's
  receipts.
- `dither_one.py` (`dither-1s*`): y = dequant(z) from `h1o.cu`'s one stream, whose noised MMAs are the ones the call's
  receipts fold.
- `dither_a.py` (`dither-a*`): option A of the credit form. A plain GEMM serves y and commits z, and the receipts run on a
  side stream.
- `dither_so.py` (`dither-a-*-so<D>`): dither-a in the statement's order. A step's tables go under its round's root before
  its coin, and its proof blocks run after it. `j10r_so.py` is its J10-R form's references.
- `hb_side.py` (`dither-hb`), `hb_one.py` (`dither-1s-hb`) and `hb_peel.py` (`dither-1s-hbp`'s gate): the same windows
  with `h1u.cu`'s proof blocks in place of `h1r.cu`'s or `h1o.cu`'s.
- `hb_sbench.py` and `hb_served.sh`: `sbench.py` with the hb arms on an H200 or a B200, building on a CPU node first.
- `j10served.py` and `j10rtx.py`: the one-stream arms against their twins on J10 banded noise, served and at kernel level.
  They are measured only: nothing in them is served or credited.

## Kernels

- `h1c.cu` (`h1c.py`): the window's tables on the device, bit for bit `served_h1/commit.py`'s: hm96-sha512 hiding leaves in
  one frame-v3-sha512 tree a table.
- `h1r.cu` (`h1r.py`): a step's H1 + fx1 proof blocks in two launches, each receipt bit for bit `h1tile.reference`'s.
- `h1o.cu`, `h1f.cu`: the one-stream and fused served linears.
- `h1noise.cu`: H1's per-proof-block noise alone, the floor any H1 + fx1 kernel pays; `h1sha.cuh`: the SHA-512 of the
  noise chunks that it and `h1u.cu` include.
- `kb.cu`, `kb_pg.cu` and `kcapi.cu`: C APIs over compute accounting's honest GEMMs (`benchmarks/pouw/dc`,
  `benchmarks/pouw/edited_gemm`), for the kernel-level benches.

## Kernel-level gates

Each holds one kernel to its reference on a full served step, then times it, on one leased RTX PRO 6000:

- `h1c.sh` runs `h1cgate.py` (`h1c.cu`'s tables against `commit.py`) or `h1agate.py` (the served calls' A tables).
- `h1r.sh` runs `h1rgate.py` (`h1r.cu`'s receipts, every kernel version).
- `sbench.sh` with `SBENCH_SCRIPT=h1ogate.py` runs `h1o.cu`'s proof-block forms.
- `wrapcheck.py`: the wrap correction against H1V3 on the CPU, before `sbench.sh` asks for a GPU.

## References

The CPU code a gate holds a kernel to, beside the Definition:

- `h1v3_ref.py`: H1V3 itself (`h1tile.reference(..., stream=True)`) from the tree on `PYTHONPATH`.
- `h1v3_ca.py`: the unpeeled one-stream proof block with the exact wrap correction, on h1tile's word functions.
- `j10r_ca.py`, `j10b_ref.py` and `j10ca_ref.py`: J10-R's call-scale noise, J10's banded noise, and credit's banded
  reference run under its own tree.

## Other benches

- `job.sh`: libraries built on a CPU node, then kernel benches on a leased H200 or B200, one after another: `kbench`
  (`hb_kbench.py`, checked after by `verify.py`), `h1` (`h1bench.py`), `h1u` (`h1ubench.py`), `prof` (`hb_prof.py` under
  ncu), `units` and `unitsprof` (`unitsbench.py`) and `partials` (`benchmarks/pouw`'s partials tests on each build).
- `kbench.sh` (`kbench.py`): compute accounting's honest linear against vLLM's int8 GEMM and cuBLASLt on an RTX PRO 6000.
- `dmmsweep.py`: dither-a's GEMM configs at the decode shapes, each gated before it is timed.
- `native.sh` (`native.py`): vLLM serving the int8 pin natively, one arm a process.

## Tools

- `pin.sh`, `pin_glm53.sh` (`pin_glm53.toml`): the int8 Llama-3.1-8B and GLM-5.3 pins in a node's HF cache.
- `numabind.py`: runs a timed command on the leased GPU's NUMA node.
- `kernels.py`, `stepprof.py`, `overlap.py`: an arm's kernels, a decode step by part, and what runs beside the proof
  blocks, from an nsys capture's SQLite export.
- `picks.py`: each Inductor kernel's autotune choice in an engine's process, or a fixed one (`first`, `last`), so two
  processes compiling the same kernel run the same config.
- `dbw.py`, `dbw_side.py`: the remote bandwidth audit beside sbench's arms on the GPU they share (`--bw-passes`).
- `dither_manifest.py`, `live_audit.py`: a dither-a-so window's manifest, its check against the challenger's own run, and
  a live window's audit.
- `settings.py`: the registry of every variable this directory reads.

## Tests

`tests/` runs on the CPU as the directory's suite (`pyproject.toml`): `numabind.py`'s plan on a fake sysfs, `dbw.py`'s report, the J10-R
workspace sizes and `j10r_so.py`'s references, `live_audit.py` and `dither_manifest.py`, and the settings registry. The GPU
gates run in each bench's own run, before it times anything.
