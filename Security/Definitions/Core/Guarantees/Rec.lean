module

public import Definitions.Core.Rec

/-!
# Zero knowledge carries through a recursion

`RecZK`: over any recursion `Rec.system O G go` (`Definitions/Core/Rec.lean`), zero knowledge of the outer proof system
`O` and hm96's hiding at `δ₁` give zero knowledge of the recursion, additively: `2 (kI + nC) δ₁` for the prefix's
commitment strings, swapped for zero commitments, and each session's `O` bound, its runs replaced one at a time. The
leak at public stop positions, the multiplicative form, is its corollary (`Verity.recZK_leak`, through
`Claims.zkmalRLeak_of_cuts`).
-/

@[expose] public section

namespace Verity.Guarantees

open Claims Verity
open scoped NNReal

/-- **`RecZK`: zero knowledge carries through a recursion.** For every outer proof system `O`, compiled recursion `G`,
`δ₁` and continuation test `go` that is `goesOn` at every real run (`Rec.Compiled.GoesOnAt`: `goesOn` itself, or that
the last answer is a message), `Rec.system O G go` is zero knowledge at `δ₁` (its simulator `Rec.sim`, going on past
each session where `go` holds, within `2 (kI x + nC x) δ₁ + Σ_{j < nS x} εS δ₁ x j` and `nS x · γS x` times `Fork`'s
collision probability) given:

* (Z1) `O` is zero knowledge at `δ₁` against a malicious verifier with any auxiliary tape (`ZkSystem.ZK`), its runs
  one at a time;
* (Z2) hm96 leaves hide at `δ₁` (`Hm96Hiding`), and the prefix's strings are hm96 leaves of its messages
  (`Rec.Compiled.Commits`);
* (Z3) each session's witness is in `O`'s relation at every history the honest prefix and any verifier give, unless
  the public stop fired (`Rec.Compiled.WitnessIn`);
* (Z4) the number of commitment strings in each prefix round is public (`kI x`, `nC x`), and so is each session's
  shape: its rounds, its bound's terms and its coin types (`Rec.Compiled.ByShape`).

The inner prover needs no zero knowledge of its own: its messages never leave their commitments. -/
def RecZK : Prop :=
  ∀ {W WS Sa Gp : Type} {NO ΩO : Msg → Type} [Fintype Sa] [Nonempty Sa] [AddCommGroup Gp] [Fintype Gp]
    [∀ x, Fintype (NO x)] [∀ x, Nonempty (NO x)] [∀ x, Fintype (ΩO x)] [∀ x, Nonempty (ΩO x)]
    (H : Msg → Msg) (O : ZkSystem WS NO ΩO) (G : Rec.Compiled W WS Sa Gp) (δ₁ : ℝ≥0)
    (go : ℕ → Msg → List (Msg × Out) → Bool), G.GoesOnAt go →
    O.ZK H δ₁ → Hm96Hiding G.mask (fun y => H (G.saltMsg y)) δ₁ → G.Commits H → G.ByShape O δ₁ →
      G.WitnessIn H O → (Rec.system O G go).ZK H δ₁

end Verity.Guarantees
