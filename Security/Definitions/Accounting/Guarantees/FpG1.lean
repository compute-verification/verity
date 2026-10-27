import Definitions.Accounting.Headline
import Definitions.Pouw.FpG1
import Definitions.Accounting.Assumptions.Sheet

/-!
# FP8-g1's and W4A8-g1's headlines on sm_120 (trusted)

`Definitions.Accounting.Guarantees.Headline`'s `Sm120SheetHeadline` (H1 + fx1 on DS-120's sheet, `1 − 5.860%` outside
`2^-40`) at H1's windows whose committed leaves are FP8-g1's or W4A8-g1's: its conclusion and every premise as they
are but binding's, over proof layers whose values carry the round's operand leaves (`Pouw.FpG1.Leaves`), the round's
other values in a free `W` as the record's `V`, with at least one proof a round, and one premise more, that every leaf
the layer commits is g1's on H1's grids (`Leaves.Fp8 Grids.h1`, `Leaves.W4a8 Grids.h1`).

Binding is `ProofLayer.BindingAt` at `δ_bind`, the form (I8)'s pieces take (`ProofsSoundOfPinned`), where the
record takes its fork form at `δ_fork` (`BindingForkAt`) and reads it at `√δ_fork` (`BindingOfFork`). So the budget
reads `δ_bind` where the record's reads `√δ_fork`, a fork-form bound `δ_fork` meets the premise at `δ_bind = √δ_fork`
(`Proofs.Accounting.Headline.Whole`'s `moved_le_sqrt`), and no number moves.

The credit is the record's, `T_cred = B_♭/f_max` of H1's floor program: it counts the int8 GEMM's MACs, which on a
g1 window carry the FP8 (or FP4) model's products as integers `R + b`. Whether those MACs are the model's useful work
is open and is not settled here.

The g1 premise reads what the layer commits (`ProofLayer.committed`) at a value type the statements fix, so neither an
empty reading nor an empty proof layer meets it: a layer that commits an H1 weight off E4M3's grid fails it
(`Proofs.Pouw.FpG1.H1`'s `offGrid_not_fp8`), while it meets every other premise
(`Proofs.Accounting.Headline.FpG1`'s `offGrid_premises`). It narrows the windows the statement covers, and no
more: the g1 premise does no work in the bound. The conclusion reads no operand value, so the bound is the record's,
and a layer may carry on-grid leaves that no relation reads, which is why this statement is the record's bound at
g1's windows.

What is g1's is that the premises hold at an honest g1 window: `Proofs.Accounting.Headline.FpG1`'s
`sm120FpG1_honest_applies` and `sm120W4a8G1_honest_applies` discharge every premise at an honest window whose layer
checks H1's committed ranges, which g1's leaves meet (`Leaves.fp8_inH1`, `Leaves.w4a8_inH1`). The witness isn't a
served window: its relation is H1's ranges, not a served Program's, its receipt game is the honest one, and its time
is exactly the credit, so it shows only that every premise can hold at once.

The measured twins are the same at `Sm120MeasuredHeadline` (H1 + fx1 on sm_120's measured vector VF3s-16,
`1 − 0.591%` outside `2^-40`): its conclusion and every premise as they are but binding's, read the same way. Their
0.591% rests on the measured record's premises, (I3) at `9/8` (`ScalarRoute`) among them, which is assumed and not
proved, as it is for the record itself: at A6's proved `½` floor the same bound on the same vector is 7.791%
(`Sm120MeasuredHeadline`'s docstring). Beside the g1 premise, which narrows the windows covered, they take
no premise the measured record doesn't: binding at `δ_bind` is what the record's fork form gives at `√δ_fork`.
`Proofs.Accounting.Headline.FpG1Measured` has their honest witnesses on the measured data and the off-grid twin.

The statements:

* `Sm120FpG1SheetHeadline`: FP8-g1, E4M3 activations to 120 and weights to 104.
* `Sm120W4a8G1SheetHeadline`: W4A8-g1, the record's int8 activations beside doubled E2M1 weights under block scales
  to 9.
* `Sm120FpG1MeasuredHeadline`, `Sm120W4a8G1MeasuredHeadline`: the same two on the measured vector, at `1 − 0.591%`.
-/

namespace Accounting.Guarantees.FpG1

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity Pouw.FpG1

/-- **FP8-g1's headline on sm_120 at the data sheet**, under full checking with live coins. `Sm120SheetHeadline`, with
binding at `δ_bind` (`ProofLayer.BindingAt`) in place of its fork form, at a window whose proofs, at least one a round
(`n_π + 1`), open the round's operand leaves beside its other values (`Leaves × W`), every committed leaf an FP8-g1
level on H1's grids (`Leaves.Fp8 Grids.h1`: E4M3's integers to 120 for activations and to 104 for weights): under the
record's other premises, outside an event of probability at most `2^-40`
every accepted window's time is at least `1 − 5860/100000` (`1 − 5.860%`) of the credit `T_cred = B_♭/f_max`. The
credit counts the int8 GEMM's MACs that carry the FP8 model's products; whether they are its useful work is open.
The g1 premise does no work in the bound: it narrows the windows covered, and the bound is the record's.
`ε_sample` and the Fiat–Shamir premium are separate lines, as in the record. -/
def Sm120FpG1SheetHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 17) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {W : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks (nπ + 1) (Leaves × W))
    (δbind δzk : ℝ),
    (∀ x w i, (pf.committed x w i).1.Fp8 Grids.h1) →
    (Sm120.checks : ℝ) * (δbind + ((nπ + 1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm120.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm120.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm120.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm120.sheet (Pl.run x).run) → (∀ x, SheetWords Sm120.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm120.model (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm120.model (Pl.run x) ((1 - Uf.share x / Sm120.checks) * Sm120.model.Nw)
      Sm120.piStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → pf.BindingAt δbind → ProofSoundAt pf δzk → ProofsPin pf → RightFixed Pl pf Sm120.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm120.model.credit Sm120.flat < 1 - 5860 / 100000) ≤
      (1 / 2) ^ 40

/-- **W4A8-g1's headline on sm_120 at the data sheet**, under full checking with live coins. `Sm120SheetHeadline`, with
binding at `δ_bind` (`ProofLayer.BindingAt`) in place of its fork form, at a window whose proofs, at least one a round,
open the round's operand leaves beside its other values, every committed leaf W4A8-g1's on H1's grids
(`Leaves.W4a8 Grids.h1`: the record's int8 activations in `[−126, 126]`, weights a doubled E2M1 level under an integer
block scale in `[1, 9]`): under the record's other premises, outside an event of probability at
most `2^-40` every accepted window's time is at least `1 − 5860/100000` of the credit `T_cred = B_♭/f_max`. The credit
counts the int8 GEMM's MACs that carry the FP4 weights' products; whether they are the model's useful work is open.
The g1 premise does no work in the bound: it narrows the windows covered, and the bound is the record's. -/
def Sm120W4a8G1SheetHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 17) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {W : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks (nπ + 1) (Leaves × W))
    (δbind δzk : ℝ),
    (∀ x w i, (pf.committed x w i).1.W4a8 Grids.h1) →
    (Sm120.checks : ℝ) * (δbind + ((nπ + 1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm120.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm120.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm120.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm120.sheet (Pl.run x).run) → (∀ x, SheetWords Sm120.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm120.model (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm120.model (Pl.run x) ((1 - Uf.share x / Sm120.checks) * Sm120.model.Nw)
      Sm120.piStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → pf.BindingAt δbind → ProofSoundAt pf δzk → ProofsPin pf → RightFixed Pl pf Sm120.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm120.model.credit Sm120.flat < 1 - 5860 / 100000) ≤
      (1 / 2) ^ 40

/-- **FP8-g1's headline on sm_120's measured vector**, under full checking with live coins. `Sm120MeasuredHeadline`,
with binding at `δ_bind` (`ProofLayer.BindingAt`) in place of its fork form, at a window whose proofs, at least one a
round (`n_π + 1`), open the round's operand leaves beside its other values (`Leaves × W`), every committed leaf an
FP8-g1 level on H1's grids (`Leaves.Fp8 Grids.h1`): under the measured record's other premises, outside an event of
probability at most `2^-40` every accepted window's time is at least `1 − 591/100000` (`1 − 0.591%`) of the credit
`T_cred = B_♭/f_max` on VF3s-16. The 0.591% rests on those premises, (I3) at `9/8` (`ScalarRoute`) among them, which is
assumed, as it is for the measured record: at A6's proved `½` floor the same bound is 7.791%. Beside the g1 premise,
which does no work in the bound and narrows the windows covered, the statement takes no premise the measured record
doesn't: binding at `δ_bind` is what the record's fork form gives at `√δ_fork`. The credit counts the int8 GEMM's MACs
that carry the FP8 model's products; whether they are its useful work is open. -/
def Sm120FpG1MeasuredHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 15) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {W : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks (nπ + 1) (Leaves × W))
    (δbind δzk : ℝ),
    (∀ x w i, (pf.committed x w i).1.Fp8 Grids.h1) →
    (Sm120.checks : ℝ) * (δbind + ((nπ + 1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm120Measured.machine (Pl.run x).run) →
    (∀ x, WordsWritten Sm120Measured.model (Pl.run x)) → (∀ x, ScalarRoute Sm120Measured.model (Pl.run x)) →
    (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm120Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm120.checks) * Sm120Measured.model.Nw) (onFacet Sm120Measured.piStar)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → pf.BindingAt δbind → ProofSoundAt pf δzk → ProofsPin pf →
    RightFixed Pl pf Sm120Measured.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm120Measured.model.credit Sm120Measured.flat < 1 - 591 / 100000) ≤ (1 / 2) ^ 40

/-- **W4A8-g1's headline on sm_120's measured vector**, under full checking with live coins.
`Sm120MeasuredHeadline`, with binding at `δ_bind` (`ProofLayer.BindingAt`) in place of its fork form, at a window whose
proofs, at least one a round, open the round's operand leaves beside its other values, every committed leaf W4A8-g1's
on H1's grids (`Leaves.W4a8 Grids.h1`): under the measured record's other premises, outside an event of probability at
most `2^-40` every accepted window's time is at least `1 − 591/100000` of the credit `T_cred = B_♭/f_max` on VF3s-16.
The 0.591% rests on those premises, (I3) at `9/8` (`ScalarRoute`) among them, which is assumed, as it is for the
measured record: at A6's proved `½` floor the same bound is 7.791%. Beside the g1 premise, which does no work in the
bound and narrows the windows covered, the statement takes no premise the measured record doesn't: binding at `δ_bind`
is what the record's fork form gives at `√δ_fork`. The credit counts the int8 GEMM's MACs that carry the FP4 weights'
products; whether they are the model's useful work is open. -/
def Sm120W4a8G1MeasuredHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 15) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {W : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks (nπ + 1) (Leaves × W))
    (δbind δzk : ℝ),
    (∀ x w i, (pf.committed x w i).1.W4a8 Grids.h1) →
    (Sm120.checks : ℝ) * (δbind + ((nπ + 1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm120Measured.machine (Pl.run x).run) →
    (∀ x, WordsWritten Sm120Measured.model (Pl.run x)) → (∀ x, ScalarRoute Sm120Measured.model (Pl.run x)) →
    (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm120Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm120.checks) * Sm120Measured.model.Nw) (onFacet Sm120Measured.piStar)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → pf.BindingAt δbind → ProofSoundAt pf δzk → ProofsPin pf →
    RightFixed Pl pf Sm120Measured.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm120Measured.model.credit Sm120Measured.flat < 1 - 591 / 100000) ≤ (1 / 2) ^ 40

end Accounting.Guarantees.FpG1
