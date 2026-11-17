# benchmarks/hash/dither

PoUW's served H1 + fx1 and J10-R receipts in compiled vLLM, on int8-exact Llama-3.1-8B, and the benches that time them
against stock vLLM int8. Each arm replaces the int8-exact linear with kernels that serve y and, from the same operands,
compute the execution phase's tables, seeds and receipts. An arm is gated in its own run before it is timed: its roots, receipts
and z against its CPU twin and the Definition (`h1tile.reference` and `h1tile.j10r_reference`,
`verity/protocols/resource_accounting/compute/pouw/prover/circuit/h1tile.py`). Nothing a verifier runs reads this directory.

The arms' names (`dither`, `dither-1s`, `dither-a`, ...) are identifiers the runs' records carry. What they serve are
H1's and J10-R's folds, not Dither, PoUW's no-fold design (`verity/protocols/experimental/dither/`).

## The arm of record

`forms.form_of` labels every arm, and each arm's stats, check documents and gate verdicts carry its label. One family is
labelled `form=A (provisional)`, the provisional credit form, and it is the arm of record: `dither-a` in the statement's
order with per-proof-block z/y records, `dither-a-{hbk,hbr,hbc}-zy-{aom,aomn}[-pt]-so<D>`, its proof
blocks under H1's noise or, on credit's layout, J10-R's (`DITHER_SO_UNITS=j10r`). Every other arm is labelled `beside`:
measured beside it, never credited.

## What the rest of the tree reads

- `h1u.cu` (with the kernels, below): the proof-block kernels: H1 + fx1's seeds and proof blocks, the one-stream `kh_one`, and J10-R's (`kh_seeds_j10r`, `kh_units_multi_j10r`).
  `verity/protocols/resource_accounting/compute/pouw/prover/circuit/h1device.py` states the domain it serves on sm_90.
- `j10r_units.py`: J10-R's proof blocks on the host, each receipt bit for bit `h1tile.j10r_reference`'s.
- `forms.py`: every arm's form label, from one table (`form_of`).

## Kernels

The kernels are in `verity/protocols/experimental/dither/prover/kernels/`, beside Dither's circuit. Two front ends are in
`prover/`: `h1r.py`, whose tensor runtime is `h1r_torch.py` here, and `j10r_ws.py`, a J10-R call's workspace sized from the
loaded build's own flags (`workspace_bytes`). The other front ends and the scripts that build the kernels stay here.

- `h1c.cu`: the execution phase's tables on the device.
- `h1r.cu` (`prover/h1r.py`, on torch through `h1r_torch.py`): a step's H1 + fx1 proof blocks in two launches.
- `h1o.cu`, `h1f.cu`: the one-stream and fused served linears.
- `h1sha.cuh`: the SHA-512 of the noise chunks, which `h1u.cu` includes.

## Gates and kernel-level benches

- The references the gates hold kernels to, beside the Definition: `h1v3_ref.py` (H1V3 itself, from the tree on
  `PYTHONPATH`).
- `job.sh`: `h1u.cu`'s libraries built on a CPU node, then `benchmarks/pouw`'s partials tests on each build on a leased
  H200 or B200.

## Tools

- `pin.sh`: the int8 Llama-3.1-8B pin in a node's HF cache.
- `benchmarks/kernels/numabind/numabind.py` runs each timed command here on the leased GPU's NUMA node.
- `kernels.py`, `stepprof.py`: from an nsys capture's SQLite export, an arm's kernels and a decode step by part.
- `picks.py`: each Inductor kernel's autotune choice in an engine's process, or a fixed one, so two processes compiling the
  same kernel run the same config.
- `dbw.py`, `dbw_side.py`: PoUB beside sbench's arms on the GPU they share (`--bw-passes`).
- `settings.py`: the registry of every variable this directory reads.

## Tests

`tests/` is the directory's suite, on the CPU (`pyproject.toml`): `dbw.py`'s report,
J10-R's workspace sizes and the settings registry. The GPU gates run in each
bench's own run, before it times anything.
