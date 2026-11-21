# PoUW's kernels

The GPU code of proof of useful work: kernels that compute a credited matrix product with a scheme's noise, commit its
values with Verity's commitment code, and are held bit for bit to the scheme's Python reference.

The guarantee: none of its own; PoUW's is in `../../README.md`.

Not yet: nothing on the auditor's side holds a binary to its scheme, and the checks below are the prover's; each build
keeps its own `run.py` beside the shared driver, and the torch adapter its own device run.

## Pieces

| piece | what it is | where |
| --- | --- | --- |
| a scheme | the mathematics as a Python reference: tiles, noise from the salt, the committed roots and indices, the exact product and the peel, the exact model of each architecture's tensor atom; it owns its circuit template | `../schemes/`, `../circuit/`, `verity/circuits/silicon/models.py` |
| a build | one scheme on one architecture: its own main loop and atoms over shared headers (formats, noise lines, peel factors, hashing); `build.sh` yields one binary | `pearl_c_sm90/`, `pearl_c_sm120/`, `pearl_c/build.sh`, `cubin_flags.sh` |
| the driver | loads a build's binary and runs its steps, with its streams, graphs and passes; takes the build and the scheme's device record as parameters | `pearl_c/driver.py`, each build's `run.py`, `adapters/torch/verity_torch/pouw/pearl_c/run_device.py` |
| the hardware seam | a card is a data file, an architecture's compile target and the toolkits that build for it another | `calibration/hardware/cards/`, `calibration/hardware/architectures/` |
| the commitment seam | Verity's SHA-512, row commitment and salt stream: a build includes them and decides only when to call them; a scheme's receipts (keyed folds) are its own | `verity/commitments/cuda/` |
| the instruction gate | at build time, the binary's tensor instructions against its table; on sm_90a also the numeric gate, enforced, with the toolkit sm_90a's pins name | `../build_gate.py`, each `sass_gate.json`; `../sass_gate.py`, `../sass_pins.json` |
| the bit-exact check | every buffer a build writes against the scheme's reference: `run.py` on a card, the host twins and the stand-in driver library off it | `run.py`, `pearl_c/fake_cuda.c`, `pearl_c/simt_twin.py`, `*_host.cpp` |
| the source name | `source_digest` over `run.py`, `grouped.py` and `driver.py`, which the adapter records with each pass | `../serving/pearl_c/kernel.py` |
| other device code | the filler, the int8 GEMMs, NCP's committed values, the served ZK statement's rows, the timing kit | `filler/`, `floored_gemm/`, `masked_gemm/`, `edited_gemm/`, `pouw_cuda/`, `served_zk/`, `gpu_kit/` |

## What varies

| scheme | sm_90a (H100, H200) | sm_100a | sm_120a |
| --- | --- | --- | --- |
| Pearl-C, FP8 (`pearl-c-h100-v1-h3`, `pearl-c-sm120-v1-h3`) | served | never written | exists |
| Pearl-C4, FP4 (`pearl-c-nvfp4-v0`) | never written | never written | removed |
| INT8: the floored, masked and edited GEMMs | never written | never written | exists |
| the filler | never written | never written | exists |
| Dither (`verity/protocols/experimental/dither/prover/kernels/`) | exists | exists | exists |
| NCP (`ncp-v2`): its committed values, one source for every card's target | exists | exists | exists |

Removed: Pearl-C4's sm_120a kernel; the parent of `5b9ef0b3a9a` last holds it.

A new implementation supplies: a scheme (its reference, exact model and circuit template) and a build (its main loop,
atoms, `build.sh` and `sass_gate.json`); a new card of a known architecture is only its card file.
Fixed for every implementation: the commitment code, the driver, the argument layout and the gates.

## Interfaces

- From the workload: `PearlCDeviceRun.gemm(w, x, weight, bias)`, one call of linear `w`, from the torch adapter.
- From a scheme: its reference functions and its exact model, which a build's bit-exact check holds it to.
- Between a build and the driver: one binary, one argument layout (`struct Args` in `pearl_c/hash_h3.cuh`), and its
  steps (`Pipeline` in `run.py`).
- From `run`: the execution phase's salt, as data.
- It gives: committed rows and roots, to the proof layer; no proof, coin or sample reaches a kernel.

## Try it

~~~sh
uv run pytest tests/benchmarks/pouw/test_pearl_c_sm90.py tests/benchmarks/pouw/test_pearl_c_sm120.py -q -k dry_run
~~~

Builds the stand-in driver library with the host's C compiler and runs each build's `run.py` on it, with no GPU: every
launch's pointers and sizes against its kernel's footprint. It prints `3 passed, 33 deselected`.
