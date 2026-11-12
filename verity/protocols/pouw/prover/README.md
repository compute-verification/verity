# PoUW's prover side

`verity.protocols.pouw.prover` is what a PoUW prover builds. No module of PoUW's verifier
(`verity/protocols/pouw/verifier/`) imports it; that directory's tests do, and so does the script that writes
`circuit.words`' vectors as Lean data (`verity/protocols/pouw/verifier/lean/scripts/row_prims_vectors.py`). `circuit/`
holds each scheme's Definitions with their word and Boolean forms and reference evaluators, an execution phase's
Program and sampling design, and the leaves, seeds and anchors a call's rows are committed under.
`schemes/` holds the named schemes (NCP, Pearl's FP8 scheme, Pearl-C and Pearl-C4), each a
`verity.protocols.pouw.verifier.protocol.PoUWScheme`, and Pearl-C's device record. A caller imports a module by its own
name and takes the names its `__all__` lists: `schemes.get(name)` returns a configured scheme, and
`circuit.traced_as(scheme, m, k, n)` its traced circuit. Target, not yet: every name another file reads is in its
module's `__all__`. Four modules of `schemes/` state none, and seven lists are short of what is read: `circuit.words`
by `sq300` and `up48`; `circuit.live` by one name and `schemes.pearl_c4` by two that are retired terms; and six by
names the module only imports, which a test, a sidecar or a benchmark reads through it (`circuit.h1tile`,
`circuit.live`, `circuit.ncp2`, `circuit.pc8`, `circuit.plan` and `schemes.pearl_c4`). The package is stated over the
verifier's parameters: it imports `verity.circuits`, `verity.definitions`, `verity.commitments`,
`verity.protocols.pouw.verifier` and SPOT's verifier (`verity.protocols.spot.verifier`, for a window's Program and
sampling design), with `verity.errors`, and nothing else of the workspace. It reads no file and no environment
variable, and nothing under `benchmarks/` or `adapters/`. Its tests are in `verity/protocols/pouw/verifier/tests/`.
