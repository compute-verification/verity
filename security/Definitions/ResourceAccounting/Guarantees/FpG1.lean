module

public import Definitions.Registration.Assumptions.Sheet
public import Definitions.Pouw.Headline.Sm120Measured
public import Definitions.Pouw.FpG1
public import Definitions.Pouw.FpG1.H1
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-! FP8-g1's and W4A8-g1's headlines on sm_120, at H1's windows whose committed leaves are g1's; whether the credited MACs are the model's useful work is open. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.FpG1

open ResourceAccounting.TS ResourceAccounting.TS.Headline ResourceAccounting.Assumptions.TS Pouw.RelativeSecurity Pouw.FpG1

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

end ResourceAccounting.Guarantees.FpG1
