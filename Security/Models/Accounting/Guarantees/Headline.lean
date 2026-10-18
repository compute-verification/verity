import Definitions.Accounting.Headline
import Models.Accounting.Assumptions.Sheet

/-!
# The families' headline guarantees (trusted)

One statement per family and reading: the family's credited work is at least `1 − ε` of its credit, `ε` the reading's
headline, on one device's machine, one design's TS model on it and its floor program ♭, all exact data
(`Definitions.Accounting.Headline`). Each is `Models.Accounting.Guarantees.Sheet`'s `SheetDIdealFloor` at that data,
its parameters fixed (`π*`, `p`, `S`, `δ`), with every hypothesis that the data discharge gone and the rest left as
they are: the named assumptions of `Models.Accounting.Assumptions` and the proof layer's pieces. Each is proved as
`Accounting.SecurityProofs.TS.G` in `Proofs.Accounting.Headline`.

* `Sm120SheetHeadline`: sm_120 (RTX PRO 6000) at its data sheet, H1 + fx1 at SCALAR-ROUTE's ½ floors: outside an
  event of probability at most `2^-40`, every accepted window runs at least `1 − 0.585%` of its credit.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **sm_120's headline at the data sheet.** On DS-120's machine (`Sm120.sheet`), H1 + fx1's TS model (`Sm120.model`,
`q = q_T = ½`, `N_w = 128`) and its floor program (`Sm120.flat`), at the issue facet's certificate (`Sm120.piStar`),
luck `p = ¾` a round over `Sm120.checks = 48,000` rounds: for any window's play `Pl` and proof layer `pf` whose
binding and soundness cost at most `2^-41` over the window, if the labelling splits, A1–A7 and FOLD-OPT hold on every
run, the rounds' luck is at most `¾`, the play is the proofs' (`Of`), binding fails with probability at most `δ_bind` a
round, every proof is sound up to `δ_ZK` (`SoundAt`), the relations pin every committed value (`Pins`) and E1(c)
holds (`RightFixed`), then outside an event of probability at most `2^-40` every accepted window's time is at least
`1 − 117/20000` (`1 − 0.585%`) of the credit `T_cred = B_♭/f_max`. FILL-OPT is not a hypothesis: H1 + fx1 has no
filler, so it holds at any run. -/
def Sm120SheetHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 17) Unit Empty) {V : Type} {nπ : ℕ}
    (pf : ProofLayer Ω Sm120.checks nπ V) (δbind δzk : ℝ),
    (Sm120.checks : ℝ) * (δbind + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm120.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm120.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm120.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm120.sheet (Pl.run x).run) → (∀ x, SheetWords Sm120.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm120.model (Pl.run x)) → (∀ x, FoldOpt Sm120.model (Pl.run x)) →
    LuckAt (3 / 4) Pl.game → pf.Of Pl → (∀ w, prob (fun x => pf.Moved x w) ≤ δbind) → pf.SoundAt δzk →
    pf.Pins → RightFixed Pl pf Sm120.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm120.model.credit Sm120.flat < 1 - 117 / 20000) ≤
      (1 / 2) ^ 40

end Accounting.Guarantees.TS
