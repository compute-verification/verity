module

public import Definitions.Poub.Assumptions.RateAuditDeployed
public import Definitions.Poub.Assumptions.RemoteHeld
public import Definitions.Poub.Assumptions.RemoteSketch

/-!
# The receipts' sketch bound, derived (trusted)

`ResourceAccounting.Remote.Held.Assumptions.SketchBound T δs`, the memory line's premise that a correct receipt was computed
while its block was in node memory or on a chip during its pass, on a trace over a remote session's outcomes
(`ResourceAccounting.Remote.Sketch.Outcome`): the prover's own coins, a uniform registered store, and SHA-512's outputs, whose
keys at the candidate seeds are within `ε_ro` of uniform by the named random-oracle premise
`ResourceAccounting.Assumptions.SketchHashRandom`. Its error is

  `δs = (Σ_j G_j)·(ε + N/2^h) + (Σ_j qtot_j)/2^(8r) + ε_ro`,

over pass `j`'s `G_j` candidate seeds and their `qtot_j` receipt-hash queries: per candidate, `ε` that some sector's
signs block has rational rank below `r` (`ResourceAccounting.Assumptions.SketchRank`) and `N/2^h` that a tag was right without
its query; per query, `2^-(8r)` that it named a block's sketch with a sector of the block never obtained (a sector the
device did not obtain is uniform, and at rank `r` at most `256^(32−r)` of its values give one sketch,
`ResourceAccounting.Guarantees.SketchRankFiber`); and `ε_ro`, the keys' distance from uniform.

* `δSketch g b Q P N εrank`: the error at a fixed receipt layout, `Q/2^g + (Q + P)·(εrank + N/2^512) + (Q + P)²/2^b`,
  `g = 8r` the guess term's bits at rank `r` and `b` the sign keys' birthday bits; real, as `SketchBound` reads it.
* `SketchBoundRun`: at every parameter.
* `SketchBoundDeployed`: at the deployed receipts (`poub.receipts`: 4096-byte blocks of `m = 128` sectors,
  64 sign rows under 512-bit sign keys, SHA-512 tags, `h = 512`) and rank `r = 27`, with `Q` the SHA-512 compressions
  the prover can make over the session, its budget `2^80` (`audit.QUERIES_LOG2`) on any hardware it reaches (grinding a
  seed reads no block, so it can run off the node): at most `Q + P` candidate seeds and `Q` queries in all give, with
  the keys within the 512-bit sign keys' birthday `(Q + P)²/2^513` of uniform,

    `δs = Q/2^216 + (Q + P)·(ε + N/2^512) + (Q + P)²/2^513` (`δSketch 216 513 Q P N ε`),

  `timing.sketch_failure(P, 64, blocks=N, queries=Q)`, the error `ledger.sketch` puts in the memory line's δ. A
  16-row session's receipts (256-bit sign keys) are `SketchBoundRun`'s at `ks = 16`, `r = 12` and
  `ε_ro = (Q + P)²/2^257`, `timing.sketch_failure(P, 16, blocks=N, queries=Q)`.

What stays assumed: SHA-512's keys at the candidate seeds are within `ε_ro` of uniform (`SketchHashRandom`, claim
`random-oracle`), the
store is uniform (the outcome's coins), the devices obtain sectors whole and verbatim (`RateSketch.Prover`, claim
`rate-sketch-verbatim`), the signs' rank tail (`SketchRank`, evaluated outside Lean), the hash budget (`KeyBudget`), and
the model's receipts and obtained blocks are the trace's (`Receipted`, `Present`).
-/

@[expose] public section

namespace ResourceAccounting.Guarantees.Remote

open ResourceAccounting.Remote.Held ResourceAccounting.Remote.Held.Assumptions ResourceAccounting.Remote.Sketch
  ResourceAccounting.Remote.Sketch.Assumptions

/-- **The sketch bound at every parameter.** On a trace over a session's outcomes whose receipts and obtained blocks
are the session devices' (`Receipted`, `Present`), with SHA-512's keys at the candidate seeds within `ε_ro` of
uniform (`SketchHashRandom`), within the per-pass hash budget `qtot` (`KeyBudget`) and with signs whose rank tail below
`r ≤ 32` is at most `ε` (`SketchRank`), `SketchBound` holds at
`(Σ_j G_j)·(ε + N/2^h) + (Σ_j qtot_j)/2^(8r) + ε_ro`. -/
def SketchBoundRun : Prop :=
  ∀ (β Hs : Type) [Fintype β] [Fintype Hs] (P N m h S L q ks r : ℕ) (G qtot : Fin P → ℕ) (ε εro : ℝ)
    (A : Remote.Sketch.Run β Hs P N m h S L q ks G) (T : Trace (Outcome β Hs N m)),
    r ≤ 32 → 0 ≤ ε → 0 ≤ εro → T.passes ≤ P →
    ResourceAccounting.Assumptions.SketchRank ks m r (ENNReal.ofReal ε) →
    ResourceAccounting.Assumptions.SketchHashRandom A εro → KeyBudget A qtot → Receipted A T → Present A T →
    SketchBound T (((∑ j, G j : ℕ) : ℝ) * (ε + N / 2 ^ h) + ((∑ j, qtot j : ℕ) : ℝ) / 2 ^ (8 * r) + εro)

/-- **The sketch bound's error at a fixed receipt layout**, `δs = Q/2^g + (Q + P)·(εrank + N/2^512) + (Q + P)²/2^b`:
`Q` the prover's SHA-512 compressions over the session, `P` the passes, `N` the blocks, `εrank` the signs' rank tail
(`SketchRank`'s bound, a parameter of this error and neither the audit's error `ε` that `Verity.Scheme.GetsBack` adds
nor a scheme's own term `Verity.Scheme.δscheme`), `g = 8r` the guess term's bits at rank `r`, and `b` the bits of the
sign keys' birthday term. One function at two parameter sets: the deployed receipts' `δSketch 216 513` (`r = 27`,
512-bit sign keys) and P2's `δSketch 96 257` (`r = 12`, 256-bit sign keys). It is real, as `SketchBound` reads it; a
scheme whose term it is takes `ENNReal.ofReal` of it. -/
noncomputable def δSketch (g b Q P N : ℕ) (εrank : ℝ) : ℝ :=
  (Q : ℝ) / 2 ^ g + ((Q : ℝ) + P) * (εrank + N / 2 ^ 512) + ((Q : ℝ) + P) ^ 2 / 2 ^ b

/-- **The sketch bound at the deployed receipts.** `SketchBoundRun` at 4096-byte blocks (`m = 128`), 64 sign rows,
SHA-512 tags and rank `r = 27`, for a prover of `Q` SHA-512 compressions over the session (its budget, `2^80` in
`audit.QUERIES_LOG2`, on any hardware it reaches): at most `Q + P` candidate seeds and `Q` receipt-hash queries in all,
with SHA-512's keys at them within the 512-bit sign keys' birthday `(Q + P)²/2^513` of uniform, at
`δSketch 216 513`. -/
def SketchBoundDeployed : Prop :=
  ∀ (β Hs : Type) [Fintype β] [Fintype Hs] (P N S L q Q : ℕ) (G qtot : Fin P → ℕ) (ε : ℝ)
    (A : Remote.Sketch.Run β Hs P N 128 512 S L q 64 G) (T : Trace (Outcome β Hs N 128)),
    0 ≤ ε → T.passes ≤ P →
    ResourceAccounting.Assumptions.SketchRank 64 128 27 (ENNReal.ofReal ε) →
    ∑ j, G j ≤ Q + P → ∑ j, qtot j ≤ Q →
    ResourceAccounting.Assumptions.SketchHashRandom A (((Q : ℝ) + P) ^ 2 / 2 ^ 513) → KeyBudget A qtot → Receipted A T →
    Present A T →
    SketchBound T (δSketch 216 513 Q P N ε)

end ResourceAccounting.Guarantees.Remote
