# PoUW's prover side

`verity.protocols.accounting.pouw.prover` is what a PoUW prover builds. No module of PoUW's verifier
(`verity/protocols/accounting/pouw/verifier/`) imports it; that directory's tests do, and so does the script that writes
`circuit.words`' vectors as Lean data (`verity/protocols/accounting/pouw/verifier/lean/scripts/row_prims_vectors.py`). `circuit/`
holds each scheme's Definitions with their word and Boolean forms and reference evaluators, an execution phase's
Program and sampling design, and the leaves, seeds and anchors a call's rows are committed under.
`schemes/` holds the named schemes (NCP, Pearl's FP8 scheme, Pearl-C and Pearl-C4), each a
`verity.protocols.accounting.pouw.verifier.protocol.PoUWScheme`, and Pearl-C's device record. A caller imports a module by its own
name and takes the names its `__all__` lists: `schemes.get(name)` returns a configured scheme, and
`circuit.traced_as(scheme, m, k, n)` its traced circuit. Target, not yet: every name another file reads is in its
module's `__all__`. Four modules of `schemes/` state none, and six lists are short of what is read (`circuit.h1tile`,
`circuit.live`, `circuit.ncp2`, `circuit.pc8`, `circuit.plan` and `schemes.pearl_c4`): each by names the module only
imports, which a test, a sidecar or a benchmark reads through it, and `circuit.live` by one name and
`schemes.pearl_c4` by two that are retired terms. The package is stated over the
verifier's parameters: it imports `verity.circuits`, `verity.definitions`, `verity.commitments`,
`verity.protocols.accounting.pouw.verifier` and SPOT's verifier (`verity.protocols.proofs.spot.verifier`, for a window's Program and
sampling design), with `verity.errors`, and nothing else of the workspace. Outside `kernels/` it reads no file and no environment
variable, and nothing under `benchmarks/` or `adapters/`. Its tests are in `tests/` and in
`verity/protocols/accounting/pouw/verifier/tests/`.

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
- `nvfp4_sm120/`: the plain NVFP4 and FP8 GEMMs on sm_120a, with no proof of work, the baseline on the same mainloop
  that a Pearl-C4 or Pearl-C kernel is timed against.
- `harness/native/`: the sources of the GPU harness's device library (`libpouw_harness.so`: fills, flushes, events,
  graphs, cuBLASLt and the CUTLASS baselines), which its `build.sh` builds with `nvfp4_sm120/`'s plain GEMMs.
- `gpu_kit/`: the kernel-engineering skill's timing protocol as code (poisoning, clock reads, SASS checks), for a
  standalone `.cu` bench to include.
- `served_zk/`: a served Pearl-C window's rows committed as hm96-sha512 leaves in the ZK statement's format, and that
  window registered, sampled, proved and judged by PoUW's verifier.

The SASS gate is `sass_gate.py`: no shipped kernel computes a value the verifier's scalar model can't replay. Its pins
are `sass_pins.json`, whose numeric pin is built from `ieee_pin.cu` and changes only with ptxas; the listings and probes
its tests read are `tests/fixtures/sass_gate/`. The kernels' checks, ships and runs on a card, and the gate's tests, are in
`benchmarks/pouw/`.
