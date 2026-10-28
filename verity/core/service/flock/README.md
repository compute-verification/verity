# Flock

C-Flock's executable verifier of record, `flock-verify` and the other executables `lakefile.toml` builds, in Lean 4 with
no dependencies, at the toolchain `lean-toolchain` pins. What it checks, its commands
and its agreement with upstream's Rust verifier are in `verity/ml/flock/verifier/README.md`; its guarantees are in
`Security/Definitions/Flock/Guarantees/` and `Security/Proofs/Flock/`.

The proofs are about the Lean definitions, so the running binary also trusts the pinned toolchain's compiler and the Lean
runtime it links, the C compiler that builds both, and the C library under them, and no declaration here swaps a
definition for code no proof reads (`@[extern]`, `@[implemented_by]`, `unsafe`, `native_decide`), which
`verity/ml/flock/tests/test_lean_verifier.py` refuses.
