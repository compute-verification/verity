import Models.Accounting.Assumptions.TSSelected

/-!
# A padded window's guarantee statement: (I8) at the published credit (trusted)

`ProofsSoundOfDrawn`'s counterpart for a window that credits only its selected units (`Accounting.TS.SelectLayer`,
note:20261018T2200Z-draft-padded-credit-rule). Statement draft, no proof: it goes to a red team, then to a decision,
before it is listed in `lean-audit.json` and proved as `Accounting.SecurityProofs.TS.ProofsSoundOfSelected`.

* `ProofsSoundOfSelected`: outside binding, joint soundness and grinding failures
  (`S·(δ_bind + δ_ks) + μ·Σ_{u exposed} k_u`), an accepted padded window is accepted by the ideal game and its fixed
  words per round are at least its published credit less its deficiency's words, over `S` (`CreditSound`).
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

open Classical in
/-- **(I8) for a padded window, at its published credit.** Over a sampled window's proofs with selections
(`SelectLayer`) whose positions' writers read them (`WriterReads`), whose selections are units' outputs
(`SelectionsComputed`), whose exposed units' candidates are committed before their coins (`CandidatesEarly`), whose
tally is in every draw (`AlwaysProved`) and publishes at most the selected weight when right (`TallyComputes`), with a
play that is its drawn units' proofs' (`Of`), the run's modes `mode` and folded words `folded`, a share covering every
selected unit that isn't good with its selections (`SelCovers`: a property of the share's definition, proved for each
game; a share charging each unpaid unit its `selConeWt`, so each selection unit the cone it gates, has it) and E1(c)
per selected unit (`SelectedFixed`, at H-μ's point mass `μ`): if binding, joint soundness and pinning hold as in
`DrawnRightOff`, then outside an event of probability at most `S·(δ_bind + δ_ks) + μ·Σ_{u exposed} k_u` every accepted
window is accepted by the ideal game and its fixed words per round are at least its published credit less
`d·N_w`, over `S` (`CreditSound`).

What it rules out: a wrong tally unit is a drawn unit whose proof is unsound (`JointSound`), so outside the event the
published credit is at most the selected weight; a padding unit counts only when selected, and a wrong selection
that selects it leaves every unit it gates not good, in the share; a selection computed after a unit's coin from
candidates committed before it buys `k·μ`. It reads no weight sum: the padded words `S·N_w` bound the credit, not
the reverse. -/
def ProofsSoundOfSelected : Prop :=
  ∀ {K T L U J Q Pos V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype J] [Nonempty V]
    (C B : Type) [Fintype C] [Fintype B] (S m s : ℕ) (P : Play (Paths C m × B) S K T L U)
    (sl : SelectLayer C B S m s J Q Pos V) (mode : Paths (Paths C m × B) S → Fin S × J → Mode)
    (wt : Fin S → J → ℝ) (folded : Paths (Paths C m × B) S → ℝ) (Nw δbind δks μ : ℝ),
    0 < S → sl.toDrawnLayer.WriterReads → sl.SelectionsComputed → sl.CandidatesEarly → sl.AlwaysProved →
    sl.TallyComputes wt → sl.toDrawnLayer.Of P → (∀ w, prob (fun x => sl.toDrawnLayer.MovedIn x w) ≤ δbind) →
    sl.toDrawnLayer.JointSound δks → sl.toDrawnLayer.Pins → sl.SelCovers P mode wt Nw →
    SelectedFixed P sl mode wt folded μ →
      CreditSound P sl.published Nw
        (S * (δbind + δks) + μ * ∑ w, ∑ j, (if sl.Exposed w j then (sl.cands w j : ℝ) else 0))

end Accounting.Guarantees.TS
