# Trusted components

These are not assumptions of the theorem, but the verdict still depends on them.

- **Lean's kernel.**
  - **Mitigation:** the Lean audit sends every declaration to the kernel again (`tools/verity/lean/Replay.lean`), and
    `audit.py --fresh` re-checks each root's whole import closure, Mathlib and ArkLib included (`leanchecker --fresh`).
    Both use Lean's own kernel; an independent one (nanoda, lean4lean) would remove this trust.
  - Numeric checks use `decide +kernel`, which is kernel evaluation. The development never uses `native_decide`.
- **Lean's compiler, runtime and the C compiler, for the running binary.** Only the Lean verifier runs in production;
  upstream's Rust verifier is a cross-check (the agreement job), not a second verifier. Proofs are about Lean
  definitions, not machine code.
  - **Mitigation:** the agreement tests of spec §17.1. The Rust implementation must reach the same verdict on the corpus
    and on fuzzed mutations, so a miscompilation that changes a tested verdict fails the agreement job. Untested inputs
    stay exposed.
  - **Also trusted:** compiled code that no proof sees. The executable's `../../../../../verity/core/service/flock/lean-audit.json` lists it under
    `escapes`, each entry with the test that covers it: the `partial def`s `Flock.canon` (canonical JSON of the
    statement's objects) and `Flock.TemplateQuery.Desc.gates` (a descriptor's gate count). The audit fails on any
    `partial def`, `opaque`, `unsafe`, `@[extern]` or `@[implemented_by]` not listed there.
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
- **The theorem statements.** A proof of the wrong statement is worthless. The reviewer's job is `../../../../lean-audit.json`'s
  `pins` and `reads`, and `../ASSUMPTIONS.md` for the named assumptions.
  - `pins`: each pinned theorem's signature as Lean prints it, with its named assumptions.
  - `reads`: per module, the definitions those statements depend on (followed through the `meaning` modules,
    `FlockSoundness` and `FlockLevel3`), and which pins read them. Some are in proof files (`LinkSound.lean`,
    `CompiledSound.lean`, `CompiledLock.lean`).
  - The audit fails when a signature or a definition it reads changes, until `audit.py --update` rewrites the record.
    When a rewritten record lands on `main`, Daniel gets a DM showing it, and that is the check: a statement reviewer
    is optional, and nothing waits on one. A theorem cited but not pinned (`../ASSUMPTIONS.md` names them) is read in
    its source.
- **Pinned dependencies.** ArkLib `b2e456fc`; through its manifest, VCVio `d7089e46`, CompPoly `df591bb8`
  (`v4.34.0-patch2`) and Mathlib `5ed29652` (`v4.34.0`). Lean 4.34.0. The definitions used from them (Reed–Solomon
  codes, `mcaError`, Johnson bounds) are part of what the statement means. `reads` does not follow them, so a bump's
  reviewer compares them by hand.
- **Everything around the verifier's checks:** the session record's custody, the verifier's source of coins (spec §7),
  and the registered roots.
