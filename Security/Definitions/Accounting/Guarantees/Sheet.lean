import Definitions.Accounting.Assumptions.Sheet

/-!
# What the data-sheet axioms give (trusted)

Statements over A1–A7 (`Definitions.Accounting.Assumptions.Sheet`), each `def G : Prop`,
proved as `Accounting.SecurityProofs.TS.G` in `Proofs.Accounting.TS`:

* `SheetFacetCeilings`: (I1) on the data-sheet machine, from A1–A5.
* `SheetScalarRouteHalf`: (I3) at the ½ floor (`q, q_T ≤ ½`, `b_T ≤ 0`), from A6 and the labelling's split.
* `SheetDIdealFloor`: Theorem D-ideal (2.7♭) on the data-sheet machine, its (I1)–(I5) replaced by A1–A7 at the ½
  floor.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **(I1) from the data sheet.** On a run with counts `≥ 0`, A1–A5 make every facet of the data-sheet machine a
ceiling. -/
def SheetFacetCeilings : Prop :=
  ∀ {K P : Type} [Fintype K] [DecidableEq P] (S : Sheet K P) (Q : Run K),
    (∀ k, 0 ≤ Q.n k) → SheetClock S Q → SheetDispatch Q → SheetTensor S Q → SheetPipe S Q → SheetSmem S Q →
    FacetCeilings S.machine Q

/-- **(I3) at the ½ floor.** On a labelled run the labelling splits, A6 gives SCALAR-ROUTE at any floors
`q, q_T ≤ ½` and `b_T ≤ 0`. -/
def SheetScalarRouteHalf : Prop :=
  ∀ {K R T L U F X : Type} [Fintype T] [Fintype L] [Fintype U] (Z : Model K R T L U F X) (A : Anatomy K T L U),
    A.Split → SheetWords Z A → Z.q ≤ 1 / 2 → Z.qT ≤ 1 / 2 → Z.bT ≤ 0 → ScalarRoute Z A

/-- **Theorem D-ideal (2.7♭) on the data-sheet machine.** As `DIdealFloor` for a model whose machine is sheet `Sh`'s,
at SCALAR-ROUTE's floors `q, q_T ≤ ½` and `b_T ≤ 0`, with (I1)–(I5) replaced by A1–A7 and (I6)–(I8) kept: outside an
event of probability at most `δ`, every accepted run satisfies
`T(M)/T_cred ≥ 1 − ε^♭_skip − ε^♭_cert − ε^♭_route − ε^♭_mand − ε^♭_jl`. -/
def SheetDIdealFloor : Prop :=
  ∀ {K P T L U F X : Type} [Fintype K] [Fintype P] [DecidableEq P] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X]
    (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X) (Fl : Labelled K) (Ω : Type) [Fintype Ω] (S : ℕ)
    (Pl : Play Ω S K T L U) (lam : SheetRow P → ℝ) (p δ δbind δzk : ℝ) (nπ : ℕ),
    Z.G = Sh.machine → IsCert lam → Z.Wf Fl → 0 < S → 0 < p → p < 1 →
    0 < δ - S * (δbind + nπ * δzk) → (∀ x, (Pl.run x).Split) → Z.q ≤ 1 / 2 → Z.qT ≤ 1 / 2 → Z.bT ≤ 0 →
    (∀ x, SheetClock Sh (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sh (Pl.run x).run) → (∀ x, SheetPipe Sh (Pl.run x).run) →
    (∀ x, SheetSmem Sh (Pl.run x).run) → (∀ x, SheetWords Z (Pl.run x)) → (∀ x, SheetSemantics Z (Pl.run x)) →
    (∀ x, FoldOpt Z (Pl.run x)) → (∀ x, FillOpt Z (Pl.run x)) → LuckAt p Pl.game →
    ProofsSound Pl Z.Nw δbind δzk nπ →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Z.credit Fl <
      1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
        Z.epsJl Fl lam p (δ - S * (δbind + nπ * δzk)) S) ≤ δ

end Accounting.Guarantees.TS
