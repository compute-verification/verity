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
order with per-proof-block z/y records, `dither-a-{hbk,hbr,hbc}-zy-{aom,aomn}[-pt]-so<D>` (`dither_so.py`), its proof
blocks under H1's noise or, on credit's layout, J10-R's (`DITHER_SO_UNITS=j10r`). Every other arm is labelled `beside`:
measured beside it, never credited.

## What the rest of the tree reads

- `h1u.cu` (with the kernels, below): the proof-block kernels: H1 + fx1's seeds and proof blocks, the one-stream `kh_one`, and J10-R's (`kh_seeds_j10r`, `kh_units_multi_j10r`). `benchmarks/pouw/served_h1` builds
  it, and `verity/protocols/resource_accounting/compute/pouw/prover/circuit/h1device.py` states the domain it serves on sm_90.
- `j10r_units.py`: J10-R's proof blocks on the host, each receipt bit for bit `h1tile.j10r_reference`'s.
- `forms.py`: every arm's form label, from one table (`form_of`).
- The execution phase modules (`dither_a.py`, `dither_fused.py`, `dither_one.py`), `h1c.py`, `h1r_torch.py` and `settings.py`, which
  the `benchmarks/pouw` suite imports on the CPU.

## The arms

- `dither_a.py` (`dither-a*`): option A of the credit form. A plain GEMM serves y and commits z, and the receipts run on
  a side stream.
- `dither_so.py` (`dither-a-*-so<D>`): dither-a in the statement's order, the arm of record. A step's tables go under its
  round's root before its coin, and its proof blocks run after it. `j10r_so.py` holds its J10-R form's references;
  `dither_manifest.py` an ended execution phase's manifest and its check against the challenger's own run.
- `dither_one.py` (`dither-1s*`): y = dequant(z) from `h1o.cu`'s one stream, whose noised MMAs are the ones the call's
  receipts fold.
- `dither_fused.py` (`dither`, `dither-side`): y from `h1f.cu`'s one GEMM, which also computes the previous step's
  receipts.

## Kernels

The kernels are in `verity/protocols/experimental/dither/prover/kernels/`, beside Dither's circuit. Two front ends are in
`prover/`: `h1r.py`, whose tensor runtime is `h1r_torch.py` here, and `j10r_ws.py`, a J10-R call's workspace sized from the
loaded build's own flags (`workspace_bytes`). The other front ends and the scripts that build the kernels stay here.

- `h1c.cu` (`h1c.py`): the execution phase's tables on the device, bit for bit `benchmarks/pouw/served_h1/commit.py`'s.
- `h1r.cu` (`prover/h1r.py`, on torch through `h1r_torch.py`): a step's H1 + fx1 proof blocks in two launches.
- `h1o.cu`, `h1f.cu`: the one-stream and fused served linears.
- `h1sha.cuh`: the SHA-512 of the noise chunks, which `h1u.cu` includes.

## Gates and kernel-level benches

- The references the gates hold kernels to, beside the Definition: `h1v3_ref.py` (H1V3 itself, from the tree on
  `PYTHONPATH`).
- `job.sh`: `h1u.cu`'s libraries built on a CPU node, then `benchmarks/pouw`'s partials tests on each build on a leased
  H200 or B200.
- `dmmsweep.py`: dither-a's GEMM configs at the decode shapes, each gated before it is timed.

## Tools

- `pin.sh`: the int8 Llama-3.1-8B pin in a node's HF cache.
- `benchmarks/kernels/numabind/numabind.py` runs each timed command here on the leased GPU's NUMA node.
- `dbw.py`, `dbw_side.py`: PoUB beside sbench's arms on the GPU they share (`--bw-passes`).
- `settings.py`: the registry of every variable this directory reads.

## Tests

`tests/` is the directory's suite, on the CPU (`pyproject.toml`): `dbw.py`'s report,
J10-R's workspace sizes, `j10r_so.py`'s references, `dither_manifest.py`'s check, and the settings registry.
