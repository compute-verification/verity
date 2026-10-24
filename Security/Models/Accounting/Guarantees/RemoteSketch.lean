import Models.Accounting.Assumptions.RateAuditDeployed
import Models.Accounting.Assumptions.RemoteHeld
import Models.Accounting.Assumptions.RemoteSketch

/-!
# The receipts' sketch bound, derived (trusted)

`Accounting.Remote.Held.Assumptions.SketchBound T δs`, the memory line's premise that a correct receipt was computed
while its block was in node memory or on a chip during its pass, on a trace over a remote session's coins
(`Accounting.Remote.Sketch.Coins`): the prover's own, a uniform registered store, and each candidate seed's signs and
receipt hash, uniform signs and a random function. Its error is

  `δs = (Σ_j G_j)·(ε + N/2^h) + (Σ_j qtot_j)/2^(8r)`,

over pass `j`'s `G_j` candidate seeds and their `qtot_j` receipt-hash queries: per candidate, `ε` that some sector's
signs block has rational rank below `r` (`Accounting.Assumptions.SketchRank`) and `N/2^h` that a tag was right without
its query; per query, `2^-(8r)` that it named a block's sketch with a sector of the block never obtained (a sector the
device did not obtain is uniform, and at rank `r` at most `256^(32−r)` of its values give one sketch,
`Accounting.Guarantees.SketchRankFiber`).

* `SketchBoundSession`: at every parameter.
* `SketchBoundDeployed`: at the deployed receipts (`remote_bandwidth.receipts`: 4096-byte blocks of `m = 128` sectors,
  16 sign rows, SHA-512 tags, `h = 512`) and rank `r = 12`, with `C` the node's SHA-512 compressions over the session:
  at most `C + P` candidate seeds and `C` queries in all give

    `δs = C/2^96 + (C + P)·(ε + N/2^512)`,

  `timing.sketch_failure(C, P, 16, blocks=N)`, the error `ledger.sketch` puts in the memory line's δ.

What stays assumed: the hash is a random function at each seed and the store is uniform (the model's coins, claims
`random-oracle` and `rate-sketch-verbatim`), the devices obtain sectors whole and verbatim (`RateSketch.Device`), the
signs' rank tail (`SketchRank`, evaluated outside Lean), the hash budget (`KeyBudget`), and the model's receipts and
obtained blocks are the trace's (`Receipted`, `Present`).
-/

namespace Accounting.Guarantees.Remote

open Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Sketch
  Accounting.Remote.Sketch.Assumptions

/-- **The sketch bound at every parameter.** On a trace over a session's coins whose receipts and obtained blocks are
the session devices' (`Receipted`, `Present`), within the per-pass hash budget `qtot` (`KeyBudget`) and with signs
whose rank tail below `r ≤ 32` is at most `ε` (`SketchRank`), `SketchBound` holds at
`(Σ_j G_j)·(ε + N/2^h) + (Σ_j qtot_j)/2^(8r)`. -/
def SketchBoundSession : Prop :=
  ∀ (β : Type) [Fintype β] (P N m h S L q ks r : ℕ) (G qtot : Fin P → ℕ) (ε : ℝ)
    (A : Session β P N m h S L q ks G) (T : Trace (Coins β P N m h ks G)),
    r ≤ 32 → 0 ≤ ε → T.passes ≤ P →
    Accounting.Assumptions.SketchRank ks m r (ENNReal.ofReal ε) →
    KeyBudget A qtot → Receipted A T → Present A T →
    SketchBound T (((∑ j, G j : ℕ) : ℝ) * (ε + N / 2 ^ h) + ((∑ j, qtot j : ℕ) : ℝ) / 2 ^ (8 * r))

/-- **The sketch bound at the deployed receipts.** `SketchBoundSession` at 4096-byte blocks (`m = 128`), 16 sign rows,
SHA-512 tags and rank `r = 12`, for a node of `C` SHA-512 compressions over the session: at most `C + P` candidate seeds
and `C` receipt-hash queries in all. -/
def SketchBoundDeployed : Prop :=
  ∀ (β : Type) [Fintype β] (P N S L q C : ℕ) (G qtot : Fin P → ℕ) (ε : ℝ)
    (A : Session β P N 128 512 S L q 16 G) (T : Trace (Coins β P N 128 512 16 G)),
    0 ≤ ε → T.passes ≤ P →
    Accounting.Assumptions.SketchRank 16 128 12 (ENNReal.ofReal ε) →
    ∑ j, G j ≤ C + P → ∑ j, qtot j ≤ C →
    KeyBudget A qtot → Receipted A T → Present A T →
    SketchBound T ((C : ℝ) / 2 ^ 96 + ((C : ℝ) + P) * (ε + N / 2 ^ 512))

end Accounting.Guarantees.Remote
