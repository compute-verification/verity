# Flock

Flock's executable verifier, `flock-verify` and the other executables `lakefile.toml` builds, in Lean 4 at the
toolchain `lean-toolchain` pins. What it checks, its commands
and its agreement with upstream's Rust verifier are in `verity/protocols/flock/agreement/README.md`; its guarantees are in
`security/Definitions/Flock/Guarantees/` and `security/Proofs/Flock/`.

It fetches nothing. It requires, by path, the three base packages every Lean package of the tree loads:
`verity/circuits/base` (`Base`: bytes, SHA-512, canonical JSON and the JSON parser), `circuits` (`Circuits`: the
program format, its decoder, the partition and annotations) and `commitments` (`Commitments`: frame-v3 and the sampling
of units). Their declarations keep the namespace `Flock`. A module that moved there keeps its old name here as a
re-export, one `public import` and nothing else (`Flock/Bytes.lean` re-exports `Base.Bytes`), until every importer names
the base module. A job that builds this package outside the checkout copies it with those three
(`verity/protocols/flock/pod/lake_closure.py`).

The libraries are layers, each loading only the ones before it: the verifier (`Flock`'s modules outside the next two),
the ZK server (`FlockZkServer`, which imports `Flock.Firewall` and the modules under it) and the recursion (`FlockRec`,
which imports the `Flock.Rec*` modules, with `VStar`). `verity/protocols/flock/prover/tests/test_lean_layers.py` gives each module its layer and fails on an import
of a layer above; the imports that still point up are its `UP`, each with what removes it.

The proofs are about the Lean definitions, so the running binary also trusts the pinned toolchain's compiler and the Lean
runtime it links, the C compiler that builds both, and the C library under them, and no declaration here swaps a
definition for code no proof reads (`@[extern]`, `@[implemented_by]`, `unsafe`, `native_decide`), which
`verity/protocols/flock/prover/tests/test_lean_verifier.py` refuses.
