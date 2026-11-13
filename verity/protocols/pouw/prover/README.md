# PoUW's prover side

`verity.protocols.pouw.prover` is what a PoUW prover builds and PoUW's verifier (`verity/protocols/pouw/verifier/`)
never imports. `circuit/` holds each scheme's Definitions with their word and Boolean forms and reference evaluators, an
execution phase's Program and sampling design, and the leaves, seeds and anchors a call's rows are committed under.
`schemes/` holds the named schemes (NCP, Pearl's FP8 scheme, Pearl-C and Pearl-C4), each a
`verity.protocols.pouw.verifier.protocol.PoUWScheme`, and Pearl-C's device record. A caller imports a module by its own
name and takes the names its `__all__` lists: `schemes.get(name)` returns a configured scheme, and
`circuit.traced_as(scheme, m, k, n)` its traced circuit. Target, not yet: every name another file reads is in its
module's `__all__`. Six modules state none (`circuit.h1window`, `circuit.live` and four of `schemes/`), and eleven lists
are short of what is read: eight of the modules a `circuit-check` sidecar binds, `circuit.plan`, `circuit.words` and
`schemes.pearl_c4`. The package is stated over the verifier's parameters: it imports `verity.circuits`,
`verity.definitions`, `verity.commitments`, `verity.protocols.pouw.verifier` and SPOT's verifier
(`verity.protocols.spot.verifier`, for a window's Program and sampling design), with `verity.errors`, and nothing else
of the workspace. It reads no file and no environment variable, and nothing under `benchmarks/` or `adapters/`. Its
tests are in `verity/protocols/pouw/verifier/tests/`.
