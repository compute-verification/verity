import Models.Accounting.Assumptions.TSSelected

/-!
# A padded window's guarantee statements (trusted)

`ProofsSoundOfDrawn`'s and `CompilerC`'s counterparts for a window that credits only its selected units
(`Accounting.TS.SelectLayer`, note:20261018T2200Z-draft-padded-credit-rule, round 3.1). Statement drafts, no proof: they
go to a red team, then to a decision, before they are listed in `lean-audit.json` and proved as
`Accounting.SecurityProofs.TS.G`.

* `ProofsSoundOfSelected`: outside binding and joint soundness failures (`S·(δ_bind + δ_ks)`), an accepted padded
  window is accepted by the ideal game and its fixed words per round are at least its published credit less its
  deficiency's words, over `S` (`CreditSound`).
* `CompilerSel`: Lemma C at the published credit: the run's time against the credited fraction of the padded honest
  time, the luck and sampling terms absolute (so, against the credited work, scaled by the padding ratio), and the
  sampling term's `Λ` carrying the selection units' cone factor.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **(I8) for a padded window, at its published credit.** Over a sampled window's proofs with selections
(`SelectLayer`) with nonnegative unit weights, whose positions' writers read them (`WriterReads`), whose selections have
sources (`SelectionsSourced`), whose floor operands are written by their certificates' writers (`CertifiedOperands`),
whose selections are committed before their units' coins (`NoLateSelection`), whose noise steps reaching more than
one credited unit are always proved and whose noise slots are units' outputs or anchors (`NoiseProved`), whose
always-proved units are in every draw (`AlwaysProved`), with the tally among them (`TallyIn`: every limb unit), reading every gate position
(`GatesTallied`) and publishing at most the selected weight when right (`TallyComputes`, provable from the tally's
Program when it reads only the gates and its own limbs and carries, `TallyReads`), whose registration's certificates are right
(`RegisteredRight`, the registration's own guarantee at its `δ_reg`, which a composition adds), with a play that is its
drawn units' proofs' (`Of`), the run's modes `mode` and folded words `folded`, a share covering every selected unit
that isn't good with its selections (`SelCovers`: a property of the share's definition, proved for each game; a share
charging each unpaid unit its `selConeWt`, so each selection unit outside the always-proved set the cone it gates, has
it) and E1(c) per selected unit (`SelectedFixed`): if binding, joint soundness and pinning hold as in `DrawnRightOff`,
then outside an event of probability at most `S·(δ_bind + δ_ks)` every accepted window is accepted by the ideal game
and its fixed words per round are at least its published credit less `d·N_w`, over `S` (`CreditSound`).

What it rules out: a wrong always-proved unit (the tally, an A row's forming, a done flag) is a drawn unit whose proof
is unsound (`JointSound`), so outside the event the published credit is at most the selected weight and every gate
from the set is honest; a wrong forming of a certified row (A or B) is likewise unsound; a padding unit counts only
when selected, and a wrong selection unit outside the set leaves every unit it gates not good, in the share; an
operand rewritten after its certificate, and a selection committed after its unit's noise, are refused by structure;
a wrong noise step (a seed, a line, a factor chosen to cancel) reaching more than one credited unit is an always-proved
unit's unsound proof, and a noise value the prover fills is refused by structure. -/
def ProofsSoundOfSelected : Prop :=
  ∀ {K T L U J Q Pos V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype J] [Nonempty V]
    (C B : Type) [Fintype C] [Fintype B] (S m s : ℕ) (P : Play (Paths C m × B) S K T L U)
    (sl : SelectLayer C B S m s J Q Pos V) (mode : Paths (Paths C m × B) S → Fin S × J → Mode)
    (wt : Fin S → J → ℝ) (folded : Paths (Paths C m × B) S → ℝ) (Nw δbind δks : ℝ),
    0 < S → (∀ w j, 0 ≤ wt w j) → sl.toDrawnLayer.WriterReads → sl.SelectionsSourced → sl.CertifiedOperands →
    sl.NoLateSelection → sl.NoiseProved wt → sl.AlwaysProved → sl.TallyIn → sl.GatesTallied → sl.TallyComputes wt →
    sl.RegisteredRight → sl.toDrawnLayer.Of P → (∀ w, prob (fun x => sl.toDrawnLayer.MovedIn x w) ≤ δbind) →
    sl.toDrawnLayer.JointSound δks → sl.toDrawnLayer.Pins → sl.SelCovers P mode wt Nw →
    SelectedFixed P sl mode wt folded →
      CreditSound P sl.published Nw (S * (δbind + δks))

/-- **Lemma C at the published credit.** In 𝒫_samp over a padded window (`SelectLayer`), under (I1)–(I6) as in
`CompilerC`, with (I8) as `CreditSound` at the published credit (`ProofsSoundOfSelected` gives it), the selection
units' cone factor at most `F_sel ≥ 1` (`SelFan`: `1` when every selection is proved or registered,
`SelectionsProved`; about `3` at cone-proportional draws; `1 + n_c` for an `ok_A` at uniform draws) and `Λ₀ > 0`
bounding the round's saving per unit of detection on the tiles, with `F_sel·Λ₀` bounding it on the window
(`RoundGame.BudgetAt`): for every certificate `π`, outside an event of probability at most `δ`, every accepted run
satisfies

`T(M)/T(H) ≥ ((Pub/(S·N_w))·(1 − ε_skip − ε_mand − ε_machine) − ε_jl − ε_sample(F_sel·Λ₀))·(1 − ε_sched)(1 − ε_clock)`,

with `H` the honest run of the padded program with every slot selected and `δ′ = δ − S(δ_bind + δ_ks)`. The run is
compared against the credited work, `(Pub/(S·N_w))·T(H)`, not the padded honest time, which includes the padding's
honest work; the per-word terms are relative and scale with the credit, and the luck and sampling terms are words of
the padded window (`d` is a share of `N_w`), so against the credited work they are multiplied by the padding ratio
`S·N_w/Pub`, as the window's `k_prime` sizes `K′` by `N/credited`. At `Pub = S·N_w` and `F_sel = 1` it is
`CompilerC`. Lemma C(a) (that `BudgetAt (F_sel·Λ₀)` holds at the window's draw law) is not stated. -/
def CompilerSel : Prop :=
  ∀ {K R T L U F X J Q Pos V : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
    [Fintype X] [Nonempty R] [Nonempty F] [Nonempty X] [Fintype J] [Nonempty V]
    (Z : Model K R T L U F X) (H : Honest K) (C B : Type) [Fintype C] [Fintype B] (S m s : ℕ)
    (P : Play (Paths C m × B) S K T L U) (sl : SelectLayer C B S m s J Q Pos V) (wt : Fin S → J → ℝ)
    (lam : R → ℝ) (Λ₀ Fsel p δ δbind δks : ℝ),
    IsCert lam → Z.Wf H.prog → 0 < H.CH → 0 < H.fH → 0 < S → 0 < p → p < 1 → 0 < Λ₀ → 1 ≤ Fsel →
    0 < δ - S * (δbind + δks) → (∀ x, (P.run x).Split) →
    (∀ x, FacetCeilings Z.G (P.run x).run) → (∀ x, WordsWritten Z (P.run x)) → (∀ x, ScalarRoute Z (P.run x)) →
    (∀ x, SideUnits Z (P.run x)) → (∀ x, NoSum (P.run x)) → (∀ x, FoldOpt Z (P.run x)) →
    (∀ x, FillOpt Z (P.run x)) → sl.SelFan wt Fsel → RoundGame.BudgetAt (Fsel * Λ₀) P.game →
    CreditSound P sl.published Z.Nw (S * (δbind + δks)) →
    prob (fun x => P.accepts x ∧ (P.run x).run.time / H.time <
      (sl.published x / (S * Z.Nw) * (1 - Z.epsSkip H.prog lam - Z.epsMand H.prog lam - Z.epsMachine H.prog lam) -
        Z.epsJl H.prog lam p (δ - S * (δbind + δks)) S -
        Z.epsSample H.prog lam (Fsel * Λ₀) p (δ - S * (δbind + δks)) S) *
        (1 - Z.epsSched H) * (1 - Z.epsClock H)) ≤ δ

end Accounting.Guarantees.TS
