import Definitions.Accounting.Assumptions.RateAuditDeployed
import Definitions.Accounting.Assumptions.PoubHeld
import Definitions.Accounting.Assumptions.PoubSketch

/-!
# The receipts' sketch bound, derived (trusted)

`Accounting.Poub.Held.Assumptions.SketchBound T δs`, the memory line's premise that a correct receipt was computed
while its block was in node memory or on a chip during its pass, on a trace over a remote session's outcomes
(`Accounting.Poub.Sketch.Outcome`): the prover's own coins, a uniform registered store, and SHA-512's outputs, whose
keys at the candidate seeds are within `ε_ro` of uniform by the named random-oracle premise
`Accounting.Assumptions.SketchHashRandom`. Its error is

  `δs = (Σ_j G_j)·(ε + N/2^h) + (Σ_j qtot_j)/2^(8r) + ε_ro`,

over pass `j`'s `G_j` candidate seeds and their `qtot_j` receipt-hash queries: per candidate, `ε` that some sector's
signs block has rational rank below `r` (`Accounting.Assumptions.SketchRank`) and `N/2^h` that a tag was right without
its query; per query, `2^-(8r)` that it named a block's sketch with a sector of the block never obtained (a sector the
device did not obtain is uniform, and at rank `r` at most `256^(32−r)` of its values give one sketch,
`Accounting.Guarantees.SketchRankFiber`); and `ε_ro`, the keys' distance from uniform.

* `SketchBoundSession`: at every parameter.
* `SketchBoundDeployed`: at the deployed receipts (`poub.receipts`: 4096-byte blocks of `m = 128` sectors,
  16 sign rows, SHA-512 tags, `h = 512`) and rank `r = 12`, with `Q` the SHA-512 compressions the prover can make over
  the session, the node's `C` times the outside-compute factor `F` (grinding a seed reads no block, so it can run off
  the node): at most `Q + P` candidate seeds and `Q` queries in all give, with the keys within the 256-bit sign keys'
  birthday `(Q + P)²/2^257` of uniform,

    `δs = Q/2^96 + (Q + P)·(ε + N/2^512) + (Q + P)²/2^257`,

  `timing.sketch_failure(C, P, 16, blocks=N, outside=F)`, the error `ledger.sketch` puts in the memory line's δ.

What stays assumed: SHA-512's keys at the candidate seeds are within `ε_ro` of uniform (`SketchHashRandom`, claim
`random-oracle`), the
store is uniform (the outcome's coins), the devices obtain sectors whole and verbatim (`RateSketch.Device`, claim
`rate-sketch-verbatim`), the signs' rank tail (`SketchRank`, evaluated outside Lean), the hash budget (`KeyBudget`), and
the model's receipts and obtained blocks are the trace's (`Receipted`, `Present`).
-/

namespace Accounting.Guarantees.Poub

open Accounting.Poub.Held Accounting.Poub.Held.Assumptions Accounting.Poub.Sketch
  Accounting.Poub.Sketch.Assumptions

/-- **The sketch bound at every parameter.** On a trace over a session's outcomes whose receipts and obtained blocks
are the session devices' (`Receipted`, `Present`), with SHA-512's keys at the candidate seeds within `ε_ro` of
uniform (`SketchHashRandom`), within the per-pass hash budget `qtot` (`KeyBudget`) and with signs whose rank tail below
`r ≤ 32` is at most `ε` (`SketchRank`), `SketchBound` holds at
`(Σ_j G_j)·(ε + N/2^h) + (Σ_j qtot_j)/2^(8r) + ε_ro`. -/
def SketchBoundSession : Prop :=
  ∀ (β Hs : Type) [Fintype β] [Fintype Hs] (P N m h S L q ks r : ℕ) (G qtot : Fin P → ℕ) (ε εro : ℝ)
    (A : Session β Hs P N m h S L q ks G) (T : Trace (Outcome β Hs N m)),
    r ≤ 32 → 0 ≤ ε → 0 ≤ εro → T.passes ≤ P →
    Accounting.Assumptions.SketchRank ks m r (ENNReal.ofReal ε) →
    Accounting.Assumptions.SketchHashRandom A εro → KeyBudget A qtot → Receipted A T → Present A T →
    SketchBound T (((∑ j, G j : ℕ) : ℝ) * (ε + N / 2 ^ h) + ((∑ j, qtot j : ℕ) : ℝ) / 2 ^ (8 * r) + εro)

/-- **The sketch bound at the deployed receipts.** `SketchBoundSession` at 4096-byte blocks (`m = 128`), 16 sign rows,
SHA-512 tags and rank `r = 12`, for a prover of `Q` SHA-512 compressions over the session (the node's times the
outside-compute factor `F`): at most `Q + P` candidate seeds and `Q` receipt-hash queries in all, with SHA-512's keys
at them within `(Q + P)²/2^257` of uniform. -/
def SketchBoundDeployed : Prop :=
  ∀ (β Hs : Type) [Fintype β] [Fintype Hs] (P N S L q Q : ℕ) (G qtot : Fin P → ℕ) (ε : ℝ)
    (A : Session β Hs P N 128 512 S L q 16 G) (T : Trace (Outcome β Hs N 128)),
    0 ≤ ε → T.passes ≤ P →
    Accounting.Assumptions.SketchRank 16 128 12 (ENNReal.ofReal ε) →
    ∑ j, G j ≤ Q + P → ∑ j, qtot j ≤ Q →
    Accounting.Assumptions.SketchHashRandom A (((Q : ℝ) + P) ^ 2 / 2 ^ 257) → KeyBudget A qtot → Receipted A T →
    Present A T →
    SketchBound T ((Q : ℝ) / 2 ^ 96 + ((Q : ℝ) + P) * (ε + N / 2 ^ 512) + ((Q : ℝ) + P) ^ 2 / 2 ^ 257)

end Accounting.Guarantees.Poub
