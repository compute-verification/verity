# PoUW's prover side

`verity.protocols.accounting.pouw.prover` is what a PoUW prover builds. No module of PoUW's verifier
(`verity/protocols/accounting/pouw/verifier/`) imports it; that directory's tests do, and so does the script that writes
`circuit.words`' vectors as Lean data (`verity/protocols/accounting/pouw/verifier/lean/scripts/row_prims_vectors.py`). `circuit/`
holds each scheme's Definitions with their word and Boolean forms and reference evaluators, an execution phase's
Program and sampling design, and the leaves, seeds and anchors a call's rows are committed under.
`schemes/` holds the named schemes (NCP, Pearl's FP8 scheme and Pearl-C), each a
`verity.protocols.accounting.pouw.verifier.protocol.PoUWScheme`, and Pearl-C's device record. `serving/` is what runs
beside a served model, one directory a scheme (its README): a workload's adapter reaches it through `run_matmul` and
`observe_matmul` and the other names `serving` exports, and through no deeper module. `lowering/` is H1's lowering, staged
by Flock's `class_statement.UnitStager`: the H1 tile proof block's statement and the execution phase's commitments it is opened
against (`h1unit.py`, its tile and statement form passed in), the stagers the sampled prover is handed (`stagers.py`), and
the live proof blocks' statements with the verifier's own files of them (`live.py`); the vLLM adapter's `pouw_h1_circuits`
commits its circuit files, and `benchmarks/pouw/h1_prove/` drives it. A caller imports a module by its own
name and takes the names its `__all__` lists: `schemes.get(name)` returns a configured scheme, and
`circuit.traced_as(scheme, m, k, n)` its traced circuit. Target, not yet: every name another file reads is in its
module's `__all__`. Four modules of `schemes/` state none, and five lists are short of what is read (`circuit.h1tile`,
`circuit.live`, `circuit.ncp2`, `circuit.pc8` and `circuit.plan`): each by names the module only
imports, which a test, a sidecar or a benchmark reads through it, and `circuit.live` by one name that is a retired term.
Pearl-C4 (`pearl-c-nvfp4-v0`) is experimental: `SCHEMES` names it and raises until
`verity.protocols.experimental.pearl_c4` is imported, which registers it (`schemes.register`). The package is stated over the
verifier's parameters: it imports `verity.circuits`, `verity_vllm.definitions`, `verity.commitments`,
`verity.protocols.accounting.pouw.verifier` and SPOT's verifier (`verity.protocols.proofs.spot.verifier`, for a window's Program and
sampling design), with `verity.errors`, and nothing else of the workspace, except that `serving/` also imports SPOT's
device code and `verity_flock` (inside functions), and no tensor library: the workload hands `serving/` its tensors and H1's
device (`serving/tensors.py`; the device adapter does, at import), and `lowering/` imports `verity_flock`, its own
distribution's row in `tests/import_order.toml` until `verity_flock` joins `verity`. It reads no environment variable
outside `kernels/`, and nothing under `benchmarks/` or `adapters/`; outside `kernels/`, `serving/` and `lowering/` it reads
no file. Its tests are in `tests/` and in
`verity/protocols/accounting/pouw/verifier/tests/`; the device adapter's own tests cover the device code
it hands in, and the vLLM adapter's own tests drive `serving/` on its engine.

`kernels/` holds the device code, one directory each:

- `pouw_cuda/`: the sources of the vLLM adapter's PoUW device libraries, NCP's and H1's committed values as functions
  that compile for the host and the device.
- `pearl_c/`: the hashing format -h3 (SHA-512) that Pearl-C's kernels include, with its host builds and a stand-in for
  the CUDA driver, so their tests run on a CPU.
- `pearl_c_sm120/`: Pearl-C's kernels for sm_120a, with the `build.sh` that builds its cubin under its `sass_gate.py`
  and the `run.py` that checks every buffer bit for bit against the CPU reference and runs it through the CUDA driver
  API with the standard library alone, reading its own environment as it runs outside the tree; the vLLM adapter loads
  `run.py` by path as the served kernel.
- `pearl_c_sm90/`: the same for sm_90a, with the fused-MoE grouped call.
- `floored_gemm/`: the useful int8 GEMM on the exact grid, its operands noised and its product peeled, with receipts of
  its running sums.
- `masked_gemm/`: `floored_gemm/`'s GEMM with its operands hidden, one kernel for each way of hiding them.
- `edited_gemm/`: the served linear and attention products on dithered or edited int8 operands, with receipts, on
  `masked_gemm/`'s mainloop.
- `gpu_kit/`: the kernel-engineering skill's timing protocol as code (poisoning, clock reads, SASS checks), for a
  standalone `.cu` bench to include.
- `served_zk/`: a served Pearl-C window's rows committed as hm96-sha512 leaves in the ZK statement's format, and that
  window registered, sampled, proved and judged by PoUW's verifier.
- `filler/`: the DRAM-free filler, int8 MMAs on operands derived from a round's coins whose receipt a host recomputes
  bit for bit: `dcap.cu` (the decode step's proxy beside it), `pfill.cuh` (the proved filler) with its host build,
  `lfill.cuh` (the lane filler) and `flin.cuh` (the fused filler). Its scorer, arms, drivers and Python twins are in
  `benchmarks/pouw/filler/`.

The SASS gate is `sass_gate.py`: no shipped kernel computes a value the verifier's scalar model can't replay. Its pins
are `sass_pins.json`, whose numeric pin is built from `ieee_pin.cu` and changes only with ptxas; the listings and probes
its tests read are `tests/fixtures/sass_gate/`. The kernels' checks, ships and runs on a card, and the gate's tests, are
in `benchmarks/pouw/`.
