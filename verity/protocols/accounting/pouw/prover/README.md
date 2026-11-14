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
variable, and nothing under `benchmarks/` or `adapters/`. Its tests are in `verity/protocols/accounting/pouw/verifier/tests/`.

`kernels/` holds the device code: `pouw_cuda/`, the sources of the vLLM adapter's PoUW device libraries, and Pearl-C's
kernels, `pearl_c/` (the hashing format -h3 both cubins include), `pearl_c_sm120/` and `pearl_c_sm90/`, each with the
`build.sh` that builds its cubin under `sass_gate.py`'s gate and the `run.py` that checks and runs it through the CUDA
driver API with the standard library alone, reading its own environment as it runs outside the tree. The gate's pins are
`sass_pins.json`, whose numeric pin is built from `ieee_pin.cu`. Their checks, ships and runs on a card are in
`benchmarks/pouw/`, and the vLLM adapter loads `run.py` by path as the served kernel.
