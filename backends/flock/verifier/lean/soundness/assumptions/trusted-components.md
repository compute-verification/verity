# Trusted components

These are not assumptions of the theorem, but the verdict still depends on them.

- **Lean's kernel.**
  - Re-check the proofs with an independent checker (nanoda, `lean4checker`).
  - Numeric checks use `decide +kernel`, which is kernel evaluation. The development never uses `native_decide`.
- **Lean's compiler, runtime and the C compiler, for the running binary.** Only the Lean verifier runs in production;
  upstream's Rust verifier is a CI cross-validation, not a second verifier. Proofs are about Lean definitions, not
  machine code.
  - **Mitigation:** the agreement tests of spec §17.1. The Rust implementation must reach the same verdict on the corpus
    and on fuzzed mutations, so a miscompilation that changes a tested verdict fails CI. Untested inputs stay exposed.
  - **Also trusted:** any `@[extern]` or `@[implemented_by]` in the verification logic. There is none today.
- **The model against the executable verifier.** Until the refinement theorem (if `Flock.Verify` accepts, the model
  accepts on the decoded proof and record), the theorem is about `Model`. Alignment rests on shared arithmetic
  (`Model.Arith`) and step order; the refinement proofs belong to the level-3 plan.
- **`Arith.Correct`.** These are facts about the concrete GHASH arithmetic: Lagrange weights on the skip nodes and the
  output coset, and the `Pcomb` weights on both (one `nodes` fact, so that all three use the same 128 points), the
  packing basis, the Reed–Solomon novel basis, the pinned constants, and the balance of query coins. They are proved for
  the instance the executable verifier defines (`execArith_correct`, from lane `flock-verifier`'s `FlockLevel3`). Two
  hold only on a range. The
  domain points `ω_q` are distinct only for `2^d ≤ 2^64` (`ω_q` is the low word of `q`). The novel basis is a basis
  only for `L ≤ 40` columns (the range the executable's normalisers are checked on). `Level.Fits` carries those bounds,
  and every `fast100` level fits.
- **The theorem statements.** A proof of the wrong statement is worthless. The reviewer's job is `../README.md` §1,
  `../ASSUMPTIONS.md`, `Soundness.lean`, `Model/*` and `Accounting/Bound.lean`.
- **Pinned dependencies.** ArkLib `b2e456fc`; through its manifest, VCVio `d7089e46`, CompPoly `v4.34.0-patch2` and
  Mathlib `v4.34.0`. Lean 4.34.0. The definitions used from them (Reed–Solomon codes, `mcaError`, Johnson bounds) are
  part of what the statement means.
- **Everything around the verifier:** the session record's custody, the coin server, and the registered roots.
