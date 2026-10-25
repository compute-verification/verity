import Models.Pouw.Assumptions.PearlC.TTOutRowSeed

/-!
# The named assumption `FragDraw`, the fragment corollary's per-row draw (trusted; a `Prop` hypothesis, not an `axiom`)

Named by Daniel's ruling (30 Sep, 5:52 PM PDT). Only the fragment corollary takes it, `ttOutRowSeed_skipClass`
(`RowSeedSkip.lean`), the skip class's supporting line. P2's main line, `ttOutRowSeed_of_ttOut` and
`ttOutTileRowSeed_of_ttOutTile`, does not read it and takes no named assumption.

The corollary's other conditions stay definitions in `TTOutRowSeed.lean`, taken as hypotheses: (C4)
`CostModel.FragmentPriced` and (C6) `PearlCSem.RowDrawn`. C6's injective key idealizes the row leaf's collision
resistance.
-/

namespace Pouw.PearlC.Assumptions

open Pouw.PearlC Pouw.Fp8Atom

variable {Q R S : Type}

/-- **`FragDraw` per row** at `Λ` (the ruling's item 1), assumed: a row's own draw leaves at most `Λ` fragments except
with probability `2^−128`. For every oracle `H` (so every per-job `B̃` draw and every other answer), every activation
input and every row `i` of unit `u` that passes the domain rules, all fixed before the row's draw, the probability over
the row's oracle answer `r` at `key i` that row `i`'s fragment count on `A′` from that answer and on `B̃` exceeds `Λ`
is at most `2^−128`. It is `a-commit-latency.md`'s `FragDraw d Λ` at `d = devAt p G pr`, whose flags `fragCount`
reads. Nothing here proves it: it is a claim about the noise distribution at the record's pipeline. -/
def FragDraw [Fintype R] [DecidableEq Q] (sem : PearlCSem Q R S) (p : Pipeline) (G : ℕ) (s : S) (U : Workload) (u : ℕ)
    (key : ℕ → (Fin (U.layout.shape u).k → ℕ) → Q) (Λ : ℕ) : Prop :=
  ∀ (H : Q → R) (act : Codes) (i : Fin (U.layout.shape u).m),
    sem.OperandOK 1 (U.layout.shape u).k (rowCodes act i) →
      pr (fun r : R => Λ < fragCount p G
        (sem.unitChain (Function.update H (key i (rowWords (U.layout.shape u).k act i)) r) s U u act).1
        (sem.unitChain H s U u act).2 i) ≤ 1 / 2 ^ 128

end Pouw.PearlC.Assumptions
