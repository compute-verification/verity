module

public import Definitions.Registration.Assumptions.Sheet
public import Definitions.Pouw.Headline
public import Definitions.Pouw.Headline.Sm120Measured
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

/-! The headline guarantees, one per family and reading, per node and under full checking with live coins; FOLD-OPT at `π*` is a conjecture, the measured reading's `9/8` floors are assumed, not proved, and the sheet reading rests on A2u and A5b, which no vendor document states. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.TS

open ResourceAccounting.TS ResourceAccounting.TS.Headline ResourceAccounting.Assumptions.TS Pouw.RelativeSecurity

/-- **sm_120's headline at the data sheet**, under full checking with live coins. On DS-120's machine (`Sm120.sheet`),
H1 + fx1's TS model (`Sm120.model`, `q = q_T = ½`, `N_w = 128`) and its floor program (`Sm120.flat`), at its certificate
(`Sm120.piStar`: 3564/3761 on issue, 197/3761 on shared memory), over a window of `Sm120.checks = 48,000` equal live
rounds on any finite coins `Ω`: for any window's play `Pl` and proof layer `pf` whose binding and soundness cost at most
`2^-41` over the window, if the labelling splits, A1–A7 hold on every run and FOLD-OPT per word folded at `π*` prices
every run's fold at the words it folds (its unfolded share `Uf`), the play's ideal game is the receipt game `G`'s, whose
openings are pinned (`Pinned`), whose deficient words have point mass at most `¾` given the cheater's view
(`OperandHMu`, the first luck premise), whose receipt luck is at most `½` (`ReceiptLuckAt`, the second, HMU's B′) and
whose whole-unit luck is at most `½` (`WholeLuckAt`, B′ at a unit whose fold is skipped), the play is the proofs'
(`Of`), two runs off one commitment open it differently with probability at most `δ_fork` a round (`BindingForkAt`,
binding at `√δ_fork`), every proof is sound up to `δ_ZK` (`ProofSoundAt`), the relations pin every committed value
(`ProofsPin`) and E1(c) holds (`RightFixed`), then outside an event of probability at most `2^-40` every accepted
window's time is at least `1 − 5860/100000` (`1 − 5.860%`) of the credit `T_cred = B_♭/f_max`. FILL-OPT is not a
hypothesis: H1 + fx1 has no filler, so it holds at any run. `ε_sample` and the Fiat–Shamir premium are separate lines.
The bound is per node: an accepted window's node, its GPUs, host CPU and peer GPUs together, performed at
least `1 − 5.860%` of the credited work. Which device did it is the attribution layer's: read per GPU, it needs LOC's
host deduction and every billable GPU on the node credited each window.

The window's shape is the statement's: 48,000 equal live rounds, one check a round. With `S` rounds in place of
48,000 its two branches never give ε < 5.779%, and under 6% only from 17,460 rounds (Compose's live line is under 1%
only from 10,552); the windows run so far have at most 18.

The bound is tight to its own floor: `N_w(w* + φ*)` is `17·3564/3761 = 16.1095` a unit at `π*` against
`B_♭ = 17.0976` (`B_♭` and the credit don't move with `π*`), so the ε holds only while a cheater's accepted program
costs at least `(1 − 5.860%)·B_♭/(1 − 41/S) = 16.1094` a unit at `π*`, and its margin is FOLD-OPT's.
`π*`'s DS_smem weight is A5b's (above): fbw4096_K4_BT256 costs 16.79 a unit at
`π*` against `N_w φ* = 15.162`. A6's `SheetWords` sums over the design's `vec`, CUDA's lane classes and the 64-bit
ones (ALU64, CVT64, DMUL, IMADW, MOV64), so a run that heads vector words with IMAD.WIDE meets it.

This ε rests on A2u and A5b. A2 (`SheetDispatch`) counts each uniform-datapath instruction in an issue slot, which no
vendor document states, and DS-120 has no uniform-pipe row. Without A2u the machine model's LP gives 2.79% with a
measured uniform-pipe row and 100% with none. Measured, it holds at saturation: beside a saturating vector stream no
kernel's total rate exceeds 1 over 279 measurements. A2's catch-all counts an instruction the 122 classes don't name in
`NOROW`, at its issue slot alone, so no instruction is free at `π*`: a count of any class costs at least 3564/3761
there, and a bulk instruction is `S/99 KB` of a UTMABIG count. A5b (`SheetSmem`) holds a bulk or TMA instruction's size
`S` in the banks at A5's 32 B an SMSP-cycle (above). -/
def Sm120SheetHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 17) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks nπ V) (δfork δzk : ℝ),
    (Sm120.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm120.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm120.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm120.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm120.sheet (Pl.run x).run) → (∀ x, SheetWords Sm120.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm120.model (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm120.model (Pl.run x) ((1 - Uf.share x / Sm120.checks) * Sm120.model.Nw)
      Sm120.piStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork → ProofSoundAt pf δzk → ProofsPin pf → RightFixed Pl pf Sm120.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm120.model.credit Sm120.flat < 1 - 5860 / 100000) ≤
      (1 / 2) ^ 40

/-- **sm_120's headline on its measured vector**, under full checking with live coins. On VF3s-16's machine
(`Sm120Measured.machine`), H1 + fx1's TS model on it (`Sm120Measured.model`: the MMA forms at their committed words an
instruction, `q = q_T = 9/8`, `b_T = 0`, `N_w = 128`) and its floor program (`Sm120Measured.flat`), at JI3t_0's
certificate (`Sm120Measured.piStar`), over a window of `Sm120.checks = 48,000` equal live rounds on any finite coins
`Ω`: for any window's play `Pl` and proof layer `pf` whose binding and soundness cost at most `2^-41` over the window,
if the labelling splits, (I1) `FacetCeilings` holds on VF3s-16's rows, (I2)/(I4) `WordsWritten` at the model's `ω`,
(I3) `ScalarRoute` at its floors and (I5) `NoSum` hold on every run and (I6) FOLD-OPT per word folded at `π*` prices
every run's fold at the words it folds (its unfolded share `Uf`), the play's ideal game is the receipt game `G`'s, whose
openings are pinned
(`Pinned`), whose deficient words have point mass at most `¾` given the cheater's view (`OperandHMu`, the first luck
premise), whose receipt luck is at most `½` (`ReceiptLuckAt`, the second, HMU's B′) and whose whole-unit luck is at
most `½` (`WholeLuckAt`, B′ at a unit whose fold is skipped), the play is the proofs' (`Of`), two runs off one
commitment open it differently with probability at most `δ_fork` a round (`BindingForkAt`, binding at `√δ_fork`), every
proof is sound up to `δ_ZK` (`ProofSoundAt`), the relations pin every committed value (`ProofsPin`) and E1(c) holds
(`RightFixed`), then outside an event of probability at most `2^-40` every accepted
window's time is at least `1 − 591/100000` (`1 − 0.591%`) of the credit `T_cred = B_♭/f_max`. (I5) `SideUnits` and
FILL-OPT are not hypotheses: the model has no side unit and no filler, so they hold at any run. On the lane route (I3)
at `9/8` is what `ClassPSlots` gives (`1 + (32 − w)/130` slots a word) once
writers create at most `w = 15` incidences, a cap nothing proves yet; on the table route it is SCALAR-ROUTE's, outside
`ClassPSlots`. `ε_sample` and the Fiat–Shamir premium are separate lines.
The bound is per node: an accepted window's node, its GPUs, host CPU and peer GPUs together, performed at
least `1 − 0.591%` of the credited work. Which device did it is the attribution layer's: read per GPU, it needs LOC's
host deduction and every billable GPU on the node credited each window.

So the 0.591% rests on (I3) at `9/8`, an assumed premise, not on proved ones. At A6's `½` floor, which Lean proves from
A6 (`SheetScalarRouteHalf`, the floor the sheet headline reads), the same bound on the same vector is 7.791%: π* moves
to issue 0.645 + JI3t_0 0.355 and the vector route binds (10.24% on VF3r-16; the exporter's instance at `q = ½`). The
0.591% reads `9/8` until `ClassPSlots`' cap is proved.

The window's shape is the statement's: 48,000 equal live rounds, one check a round. With `S` rounds in place of
48,000 its two branches give ε < 1% only from 8,320 rounds (Compose's live line only from 10,552), and the
windows run so far have at most 18.

The vector is VF3s-16, a refit of VF3r-16's joint issue rows, not model.toml's `current` (VF3r-16):
bit-exact census programs load VF3r-16's `π*`, JI3s_0, at up to 1.0383, and honest's IMMA16832 form at 1.0107, so
(I1) on its rows is false of runs that ran. No census program
loads a VF3s-16 row above 1 (`benchmarks/pouw/tests/test_lean_headline.py`). The credit is 2.9% lower: ♭'s bound
`B_♭` is 19.259 a unit at JI3t_0 against VF3r-16's 19.829 at JI3s_0. The ε reads `ln(8/7) ≥ 2/15`, its series'
first term, where the sheet's reads `1/8`: 0.588% at the data, under 0.591%.

(I1) has no catch-all: VF3s-16's rows price the 122 census classes it measured, and an instruction none of them names
is on no row. A2's catch-all is the sheet's: VF3s-16 itself prices some listed classes (UNI64, the TMA forms) at no
issue slot, so "at its issue slot alone" is no measured fact here. -/
def Sm120MeasuredHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 15) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm120.checks nπ V) (δfork δzk : ℝ),
    (Sm120.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm120Measured.machine (Pl.run x).run) →
    (∀ x, WordsWritten Sm120Measured.model (Pl.run x)) → (∀ x, ScalarRoute Sm120Measured.model (Pl.run x)) →
    (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm120Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm120.checks) * Sm120Measured.model.Nw) (onFacet Sm120Measured.piStar)) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork → ProofSoundAt pf δzk → ProofsPin pf →
    RightFixed Pl pf Sm120Measured.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm120Measured.model.credit Sm120Measured.flat < 1 - 591 / 100000) ≤ (1 / 2) ^ 40

end ResourceAccounting.Guarantees.TS
