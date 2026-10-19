import Models.Accounting.Guarantees.TS
import Definitions.Accounting.TS.Selected
import Definitions.Accounting.TS.Registered

/-!
# A padded window's records and its counting step (trusted)

The registration's and the epoch's guarantees, which a padded window's credit reads as `RegisteredRight` and
`EpochFormedRight` (`ProofsSoundOfSelected`'s `δ_reg` and `δ_epoch`), and the counting step from (I8) at the published
credit (note:20261018T2200Z-draft-padded-credit-rule, round 5; note:20261018T2346Z-draft-spec-work-page §5). Each is
`DrawnRightOff`'s shape, with its own failure term.

* `RegisteredRightOff`: outside `δ_reg = S_reg·(δ_bind + δ_ks)`, on an accepted registration every registered
  `ok_B,j` is the forming verdict of its raw row `B_j`, and every weight's `S_B` is `Σ_j ok_B,j`.
* `EpochFormedRightOff`: outside `δ_epoch = S_ep·(δ_bind + δ_ks)`, on an accepted epoch and its windows every formed
  row in the salt-bound epoch tree is its registered raw row's forming under that salt, and every window of the salt
  that opens a formed row opens that forming.
* `CountingSel`: from (I8) at the published credit at `δ₁` and the window's budget at `Λ`,
  `S·fixed ≥ Pub − N_w·Λ·ln(1/δ_s)` on accepted paths outside `δ_s + δ₁`; with `ProofsSoundOfSelected`,
  `δ₁ = S·(δ_bind + δ_ks) + δ_reg + δ_epoch`.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Pouw.RelativeSecurity

/-- **The registration's certificates are right, outside its binding and soundness failures.** Over the weights'
registration (`Registration`), a sampled run of proofs whose record units are always proved (`AlwaysProved`), accepted
only if every drawn unit's proof is (`Of`), whose verdict's and sum's Definitions make a record right when its units are
(`Computes`): if in every round binding fails with probability at most `δ_bind`, the proofs are sound jointly at `δ_ks`
and each unit's relation pins its rightness, as in `DrawnRightOff`, then outside an event of probability at most
`S·(δ_bind + δ_ks)` (`δ_reg`, `S` the registration's rounds) every record of an accepted registration is right: each
registered `ok_B,j` is the forming verdict of its raw row `B_j`, and each weight's `S_B` is `Σ_j ok_B,j`. -/
def RegisteredRightOff : Prop :=
  ∀ {J Q Pos V : Type} [Nonempty V] (C B : Type) [Fintype C] [Fintype B] (S m s : ℕ)
    (rg : Registration C B S m s J Q Pos V) (δbind δks : ℝ),
    rg.AlwaysProved → rg.Of → rg.Computes →
    (∀ w, prob (fun x => rg.toDrawnLayer.MovedIn x w) ≤ δbind) → rg.toDrawnLayer.JointSound δks →
    rg.toDrawnLayer.Pins →
    ∃ E : Paths (Paths C m × B) S → Prop, prob E ≤ S * (δbind + δks) ∧
      ∀ x, rg.accepts x → ¬ E x → ∀ r, rg.Right x r

/-- **The salt's formed rows are right and bound across its windows, outside the run's binding and soundness
failures.** Over a salt's epoch and its windows (`Epoch`: the epoch's round, then the windows'), whose record units are
always proved (`AlwaysProved`), accepted only if every drawn unit's proof is (`Of`), whose windows' slots reading a
record are that record's row of this salt's tree (`DomainBound`: the tree's domain `(salt, root_W)` is a public input
of every tile instance, so a stale salt's row is another position), whose forming's Definitions make a record right
when its units are (`Computes`): if binding, joint soundness and pinning hold as in `DrawnRightOff`, then outside an
event of probability at most `S·(δ_bind + δ_ks)` (`δ_epoch`) on every accepted path each formed row is its raw row's
forming under the salt's noise, and every accepted proof of a drawn unit, in any window of the salt, that opens a
record's formed row opens that forming. The binding across the salt's windows is the tree's domain (`DomainBound`)
with the positions' binding: no window opens a record other than as committed. -/
def EpochFormedRightOff : Prop :=
  ∀ {J Q Pos V : Type} [Nonempty V] (C B : Type) [Fintype C] [Fintype B] (S m s : ℕ)
    (ep : Epoch C B S m s J Q Pos V) (δbind δks : ℝ),
    ep.AlwaysProved → ep.Of → ep.DomainBound → ep.Computes →
    (∀ w, prob (fun x => ep.toDrawnLayer.MovedIn x w) ≤ δbind) → ep.toDrawnLayer.JointSound δks →
    ep.toDrawnLayer.Pins →
    ∃ E : Paths (Paths C m × B) S → Prop, prob E ≤ S * (δbind + δks) ∧
      ∀ x, ep.accepts x → ¬ E x → (∀ e, ep.Right x e) ∧
        ∀ w i k e, ep.accepted x w i → ep.reads (w, ep.toDrawnLayer.drawnAt x w i) k = some e →
          ep.opened x w i k = ep.form x (ep.toDrawnLayer.committed x (ep.rawPos e))

/-- **The counting step.** For a play with (I8) at a credit `Pub` at `δ₁` (`CreditSound`; `ProofsSoundOfSelected` gives
it at the published credit with `δ₁ = S·(δ_bind + δ_ks) + δ_reg + δ_epoch`), whose ideal game's saving per unit of
detection is at most `Λ > 0` (`RoundGame.BudgetAt`; `F_sel·Λ₀` on a padded window, `CompilerSel`), and every
`δ_s > 0`: an accepted window's fixed words, `S` rounds of them, fall below `Pub − N_w·Λ·ln(1/δ_s)` with probability at
most `δ_s + δ₁`. -/
def CountingSel : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S : ℕ} {K T L U : Type} (P : Play Ω S K T L U) (pub : Paths Ω S → ℝ) (Nw Λ δs δ₁ : ℝ),
    0 < S → 0 < Nw → 0 < Λ → 0 < δs → RoundGame.BudgetAt Λ P.game → CreditSound P pub Nw δ₁ →
    prob (fun x => P.accepts x ∧ S * (P.run x).fixed < pub x - Nw * Λ * Real.log (1 / δs)) ≤ δs + δ₁

end Accounting.Guarantees.TS
