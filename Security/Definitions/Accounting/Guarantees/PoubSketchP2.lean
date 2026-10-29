import Definitions.Accounting.Guarantees.PoubSketch

/-!
# The receipts' sketch bound over PoUS's P2 store (trusted; drafted)

PoUB's P2 mode (`poub.p2`, its `PROTOCOL.md` §11) receipts a unit of four of `C*`'s 2,056-byte
blocks, 8,224 bytes, `m = 257` sectors of 32 bytes: the trace's block (`Accounting.Poub.Held.Trace`) is a unit, and
`N` counts units (`⌊blocks/4⌋`; a store's last `blocks mod 4` blocks are not audited). A block's own receipt would not
do: 2,056 bytes is not a whole number of 32-byte sectors, and the sketch's guess term is per sector, `2^-(8r)` at rank
`r ≤` a sector's bytes, so 8-byte sectors would bound it at `Q/2^64`; four blocks are 257 whole sectors and keep
`Q/2^96` at `r = 12`. The decode computes each block's partial sketch against the unit's signs of its 2,056 bytes
(those of unit bytes `2,056·(j mod 4)` on, bytes `257·(j mod 4)` on of each 1,028-byte sign row), and the unit's sketch
is the four partials' sum, so the bound is on the unit.

* `SketchBoundP2`: `SketchBoundSession` at P2's receipts (`m = 257`, 16 sign rows, SHA-512 tags, `h = 512`) and rank
  `r = 12`, with `Q` the prover's SHA-512 compressions over the session (the node's `C` times the outside-compute factor
  `F`), as `SketchBoundDeployed` is at 4096-byte blocks:

    `δs = Q/2^96 + (Q + P)·(ε + N/2^512) + (Q + P)²/2^257`,

  with `ε = 257 · timing.sign_rank_at_most(16, 11) = 2^-95.2491`, the bound on a 16 × 32 sign block's rank falling
  below 12 over the unit's sectors (evaluated outside Lean, as at `m = 128`), above `SketchBoundDeployed`'s
  `2^-96.2548` by the sectors' count.

`SketchBoundLine` is at every `m`, so the memory line's `SketchBoundAt T O δs` at P2's units is `SketchBoundLine` at
`X = Outcome β Hs N 257` under `SketchBoundP2`: no new assumption. What stays assumed is `SketchBoundDeployed`'s, at
`m = 257`: `SketchHashRandom` (claim `random-oracle`), the devices obtaining sectors whole and verbatim
(`RateSketch.Device`, claim `rate-sketch-verbatim`, here a sector of a decoded block's bytes), the signs' rank tail at 257
sectors (`SketchRank 16 257 12`), the hash budget (`KeyBudget`), and `Receipted`, `Present`, and the store uniform (the
outcome's coins). P2's `C*` is the encode of the public `W`, so a uniform store is the ideal encoding model: a sector the
device never held, read, took in or rebuilt is uniform to it. A device that rebuilds a unit obtains it (`Present`
counts it), and what rebuilding costs, in part or whole, is the codec's, the memory line's debit
(`Accounting.Assumptions.WorkHardnessP2`, `SquaringCeilingP2`).

**Drafted, not decided**: the claims await a ruling; `audit.ledger` refuses a P2 record
until a line takes this.
-/

namespace Accounting.Guarantees.Poub

open Accounting.Poub.Held Accounting.Poub.Held.Assumptions Accounting.Poub.Sketch
  Accounting.Poub.Sketch.Assumptions

/-- **The sketch bound at P2's receipts** (drafted). `SketchBoundSession` at a unit of four 2,056-byte blocks
(`m = 257`), 16 sign rows, SHA-512 tags and rank `r = 12`, for a prover of `Q` SHA-512 compressions over the session:
at most `Q + P` candidate seeds and `Q` receipt-hash queries in all, with SHA-512's keys at them within `(Q + P)²/2^257`
of uniform. -/
def SketchBoundP2 : Prop :=
  ∀ (β Hs : Type) [Fintype β] [Fintype Hs] (P N S L q Q : ℕ) (G qtot : Fin P → ℕ) (ε : ℝ)
    (A : Session β Hs P N 257 512 S L q 16 G) (T : Trace (Outcome β Hs N 257)),
    0 ≤ ε → T.passes ≤ P →
    Accounting.Assumptions.SketchRank 16 257 12 (ENNReal.ofReal ε) →
    ∑ j, G j ≤ Q + P → ∑ j, qtot j ≤ Q →
    Accounting.Assumptions.SketchHashRandom A (((Q : ℝ) + P) ^ 2 / 2 ^ 257) → KeyBudget A qtot → Receipted A T →
    Present A T →
    SketchBound T ((Q : ℝ) / 2 ^ 96 + ((Q : ℝ) + P) * (ε + N / 2 ^ 512) + ((Q : ℝ) + P) ^ 2 / 2 ^ 257)

end Accounting.Guarantees.Poub
