# benchmarks/hash/dither

PoUW's served H1 + fx1 and J10-R receipts in compiled vLLM, on int8-exact Llama-3.1-8B, and the benches that time them
against stock vLLM int8. Each arm replaces the int8-exact linear with kernels that serve y and, from the same operands,
compute the execution phase's tables, seeds and receipts. An arm is gated in its own run before it is timed: its roots, receipts
and z against its CPU twin and the Definition (`h1tile.reference` and `h1tile.j10r_reference`,
`verity/protocols/pouw/prover/circuit/h1tile.py`). Nothing a verifier runs reads this directory.

The arms' names (`dither`, `dither-1s`, `dither-a`, ...) are identifiers the runs' records carry. What they serve are
H1's and J10-R's folds, not Dither, PoUW's no-fold design (`verity/protocols/pouw/verifier/dither/`).

## The arm of record

`forms.form_of` labels every arm, and each arm's stats, check documents and gate verdicts carry its label. One family is
labelled `form=A (provisional)`, the provisional credit form, and it is the arm of record: `dither-a` in the statement's
order with per-proof-block z/y records, `dither-a-{hbk,hbr,hbc}-zy-{aom,aomn}[-pt]-so<D>` (`dither_so.py`), its proof
blocks under H1's noise or, on credit's layout, J10-R's (`DITHER_SO_UNITS=j10r`). Every other arm is labelled `beside`:
measured beside it, never credited.

`sbench.sh` builds the libraries the named arms load for the card (`SBENCH_ARCH`: 120a, else 100a or 90a) and runs the
served bench, `sbench.py`, on one leased GPU from the source tree's root. Extra arguments go to `sbench.py`:

~~~sh
RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=WHO bash benchmarks/hash/dither/sbench.sh \
  --arms stock,exact,dither-a-hbr-zy-aomn-so4 --gate --ops decode --t 64
~~~

Each arm is a resident vLLM engine in its own worker process on that GPU, and the parent alternates reps between them.
`--gate` first holds each arm's greedy tokens to its reference arm's and runs its execution phase's checks; then each rep times
decode at batch 32 (`tpot_b32_ms`) and, with `--ops sat`, tokens a second at saturation. The run writes `summary.json`
and `stop.json` under `$RESEARCH_RUN_DIR/out`, the layout `benchmarks/harness`'s `served_table` reads. `sbench.py`'s
docstring lists every arm and flag, and `settings.py` every variable they read (`DITHER_`, `SBENCH_`, `HB_`).

## What the rest of the tree reads

- `h1u.cu`: the proof-block kernels: H1 + fx1's seeds and proof blocks, the one-stream `kh_one`, and J10-R's (`kh_seeds_j10r`, `kh_units_multi_j10r`). `benchmarks/pouw/served_h1` and `benchmarks/torch_gpu/j10r_sm90`
  build it, and `verity/protocols/pouw/prover/circuit/h1device.py` states the domain it serves on sm_90.
- `j10r_ws.py`: a J10-R call's workspace, sized from the loaded build's own flags (`workspace_bytes`).
- `j10r_units.py`: J10-R's proof blocks on the host, each receipt bit for bit `h1tile.j10r_reference`'s.
- `forms.py`: every arm's form label, from one table (`form_of`).
- `sbench.sh`'s receipts-only build of `h1u.cu`, which `benchmarks/pouw/served_h1`'s producer loads; and the execution phase
  modules (`dither_a.py`, `dither_fused.py`, `dither_one.py`), `h1c.py`, `h1r.py` and `settings.py`, which
  the `benchmarks/pouw` suite imports on the CPU.

## The arms

Each arm's module patches vLLM before `sbench.py` builds the arm's engine.

- `dither_a.py` (`dither-a*`): option A of the credit form. A plain GEMM serves y and commits z, and the receipts run on
  a side stream.
- `dither_so.py` (`dither-a-*-so<D>`): dither-a in the statement's order, the arm of record. A step's tables go under its
  round's root before its coin, and its proof blocks run after it. `j10r_so.py` holds its J10-R form's references;
  `dither_manifest.py` an ended execution phase's manifest and its check against the challenger's own run; `live_audit.py` a live
  execution phase's audit.
- `dither_one.py` (`dither-1s*`): y = dequant(z) from `h1o.cu`'s one stream, whose noised MMAs are the ones the call's
  receipts fold.
- `dither_fused.py` (`dither`, `dither-side`): y from `h1f.cu`'s one GEMM, which also computes the previous step's
  receipts.

## Kernels

- `h1c.cu` (`h1c.py`): the execution phase's tables on the device, bit for bit `benchmarks/pouw/served_h1/commit.py`'s.
- `h1r.cu` (`h1r.py`): a step's H1 + fx1 proof blocks in two launches.
- `h1o.cu`, `h1f.cu`: the one-stream and fused served linears.
- `h1sha.cuh`: the SHA-512 of the noise chunks, which `h1u.cu` includes.

## Gates and kernel-level benches

- `h1c.sh` runs `h1cgate.py` (`h1c.cu`'s tables against `commit.py`) or `h1agate.py` (the served calls' A tables);
  `h1r.sh` runs `h1rgate.py`; `sbench.sh` with `SBENCH_SCRIPT=h1ogate.py` runs `h1o.cu`'s proof-block forms. Each holds
  its kernel to its reference on a full served step, then times it, on one leased RTX PRO 6000. `wrapcheck.py` holds the
  wrap correction to H1V3 on the CPU before `sbench.sh` asks for a GPU.
- The references the gates hold kernels to, beside the Definition: `h1v3_ref.py` (H1V3 itself, from the tree on
  `PYTHONPATH`).
- `job.sh`: `h1u.cu`'s libraries built on a CPU node, then `benchmarks/pouw`'s partials tests on each build on a leased
  H200 or B200.
- `dmmsweep.py`: dither-a's GEMM configs at the decode shapes, each gated before it is timed. `native.sh` (`native.py`):
  vLLM serving the int8 pin natively, one arm a process.

## Tools

- `pin.sh`, `pin_glm53.sh` (`pin_glm53.toml`): the int8 Llama-3.1-8B and GLM-5.3 pins in a node's HF cache.
- `numabind.py`: runs a timed command on the leased GPU's NUMA node.
- `kernels.py`, `stepprof.py`: from an nsys capture's SQLite export, an arm's kernels and a decode step by part.
- `picks.py`: each Inductor kernel's autotune choice in an engine's process, or a fixed one, so two processes compiling the
  same kernel run the same config.
- `dbw.py`, `dbw_side.py`: PoUB beside sbench's arms on the GPU they share (`--bw-passes`).
- `settings.py`: the registry of every variable this directory reads.

## Tests

`tests/` is the directory's suite, on the CPU (`pyproject.toml`): `numabind.py`'s plan on a fake sysfs, `dbw.py`'s report,
J10-R's workspace sizes, `j10r_so.py`'s references, `live_audit.py`, and the settings registry. The GPU gates run in each
bench's own run, before it times anything.
