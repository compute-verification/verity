module

public import Definitions.Registration.Assumptions.Sheet
public import Definitions.Pouw.HeadlineDc
public import Definitions.Pouw.HeadlineDc.Sm100Measured
public import Definitions.Pouw.HeadlineDc.Sm90MeasuredI3
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

/-! The H200's and B200's headline guarantees in `Headline`'s form; FOLD-OPT at each `π*` is a conjecture, A2u is measured but no vendor document states it, and `S8MmaExact` is outside every statement. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.TS

open ResourceAccounting.TS ResourceAccounting.TS.Headline ResourceAccounting.Assumptions.TS Pouw.RelativeSecurity

/-- **The H200's headline at the data sheet**, under full checking with live coins. On DS-90's machine
(`Sm90.sheet`), `dither-2s-licpw-redux-p1-l32`'s TS model (`Sm90.model`: every census MMA class a writer kind at its
committed words an instruction and `UNI` at 2, `q = q_T = ½`, `N_w = 4096`) and its floor program (`Sm90.flat`, ffill
out of it at p = 1), over `Sm90.checks = 48,000` rounds on any finite coins `Ω`: for any window's play `Pl` and proof
layer `pf` whose binding and soundness cost at most `2^-41` over the window, if the labelling splits, A1–A7 hold on
every run, FOLD-OPT per word folded at `π*` (`Sm90.lamStar`, DS_issue alone: at one issue slot an instruction it
is an instruction count, no pipe premise) prices every run's fold at the
words it folds (its unfolded share `Uf`, the words whose fold it skips), the play's ideal game is the receipt game
`G`'s, whose openings are pinned (`Pinned`), whose deficient words have point mass at most `¾` given the cheater's view
(`OperandHMu`), whose receipt luck is at most `½` (`ReceiptLuckAt`, HMU's B′) and whose whole-unit luck is at most `½`
(`WholeLuckAt`, B′ at a unit whose fold is skipped), the play is the proofs' (`Of`), two runs off one commitment open
it differently with probability at most `δ_fork` a round (`BindingForkAt`, binding at `√δ_fork`), every proof is sound
up to `δ_ZK` (`ProofSoundAt`), the relations pin every committed value (`ProofsPin`) and E1(c) holds (`RightFixed`),
then outside an event of probability at most `2^-40` every accepted window's time is at least `1 − 944/100000`
(`1 − 0.944%`) of the credit `T_cred = B_♭/f_max`. The unfolded branch binds, at exactly
`150882547268931/16000000000000000` (0.9430159%); the folded gives 0.8618368%. FOLD-OPT's premise has no slack
(`φ*` is CP_RED2's price, a fold the tree runs bit-exact); the reading's against 1% is 0.058% (every fold route that
much cheaper). FILL-OPT is not a hypothesis: ♭ credits no filler. A3 (`SheetTensor`) is per SM clock at the clock
each tensor figure is stated at, the 1830 MHz tensor boost clock, not f_max's 1980 MHz: read per second at f_max, the
machine model put this reading at 2.253% on a menu whose only routes were CP_SEG4 and CP_SEG4U, since its ε first
moves at tensor lines 1.060× the sheet's. -/
def Sm90SheetHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm90.checks (Fin 50) (Fin 34) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm90.checks nπ V) (δfork δzk : ℝ),
    (Sm90.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm90.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm90.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm90.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm90.sheet (Pl.run x).run) → (∀ x, SheetWords Sm90.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm90.model (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm90.model (Pl.run x) ((1 - Uf.share x / Sm90.checks) * Sm90.model.Nw) Sm90.lamStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork → ProofSoundAt pf δzk → ProofsPin pf → RightFixed Pl pf Sm90.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm90.model.credit Sm90.flat < 1 - 944 / 100000) ≤
      (1 / 2) ^ 40

/-- **The H200's headline on its measured vector**, under full checking with live coins. On DC-90-ri's machine
(`Sm90Measured.machine`), `dither-2s-licpw-redux-p1-l32`'s TS model on it (`Sm90Measured.model`: honest's IGMMA s8
m64n256k32 and `UNI` at 2 the writer kinds, `q = q_T = ½`, `b_T = 0`, `N_w = 4096`) and its floor program
(`Sm90Measured.flat`: `setup`'s floor, every word on the IGMMA, and its fold by the generic CP_RED2 and CP_RED8, both
built), over `Sm90.checks = 48,000` rounds on any finite coins `Ω`: for any window's play `Pl` and proof layer `pf`
whose binding and soundness cost at most `2^-41` over the window, if the labelling splits, (I1) `FacetCeilings` holds on
DC-90-ri's rows, A6 (`SheetWords`) and (I5) `NoSum` hold on every run, FOLD-OPT per word folded at `π*`
(`Sm90Measured.lamStar`: issue 0.003598 + JI90_0 0.966699 + JI90_1 0.029703, rounded) prices every run's fold at the
words it folds (its unfolded share `Uf`), and the luck and proof premises hold as in `Sm90SheetHeadline`
(`BindingForkAt` at `δ_fork`, `ProofSoundAt`, `ProofsPin`), then outside an event of probability at most `2^-40` every
accepted window's time is at least `1 − 1412/100000` (`1 − 1.412%`, over 1%: no check reads ffill's chains, and ♭ folds
by built routes alone) of the credit `T_cred = B_♭/f_max`. The folded branch binds, at exactly
`2495272303972424451357093036809987368800019/176832869229494800775000000000000000000000000` (1.4110908%), reading
`ln(8/7) ≥ 2/15` as `Sm120MeasuredHeadline` does; the unfolded gives 1.3529949%. The cheater's menu holds every counted
fold known to come near the bit-exact loops, none built, and ♭'s holds none of them: the moves of LI on li and each loop
(its XORs as an ATOM.E.XOR and two LD.E back, its multiplies as IMAD.WIDE, whose low word does li's multiply, or both),
whose classes carry the JI90 lines the H200 measured beside the IGMMA (ATOME32, and `UNLISTED`, where IMAD.WIDE and LD.E
count, each 0.9788 at `π*`, as an IMAD and a LOP3 are), and li-bulk at its measured cost, built, with its counted BT1024
and trimmed W = ∞ members. So FOLD-OPT at `π*` is a premise only about folds off the cheater's menu; the two-slot round
trip is off it, since li's XOR has three inputs a half step and one ATOM.E.XOR and one LD.E carry two (at this `π*` it
prices at 1.50 `φ*`). FOLD-OPT at `π*` is open: `φ*` is CP_RED's and CP_RED2's price (their generic and IMAD.WIDE
members tie), and every class a fold route issues prices above 0 there, the stores and ISETP at their measured JI90
lines. FOLD-OPT's premise has no slack; the reading reaches its 3% target with every fold route 2.38% cheaper. Its twin
at (I3)'s `q = q_T = 3/5` is `Sm90MeasuredI3Headline`. `NoSum` carries the tensor forms off the menu: DC-90-ri prices
them at the scalar coefficient on its joint rows. Three kinds would move the reading if they were on it: the
register-sourced IGMMA form, whose coefficient fitted from the H200's probes gives about 1.64% in the machine model at
an earlier design; DMMA, which at the scalar coefficient heads a word for about 0.0077 against honest's 0.0153 (at its
tensor-pipe coefficient, about 0.031, it moves nothing); and the 2:4-sparse forms, whose classes carry the issue row
alone (no run timed them beside the IGMMA), so that chained into a dense product they head a word for under `2·10⁻⁶` at
this `π*` and for 0 at the twin's. -/
def Sm90MeasuredHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm90.checks (Fin 50) (Fin 2) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm90.checks nπ V) (δfork δzk : ℝ),
    (Sm90.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm90Measured.machine (Pl.run x).run) →
    (∀ x, SheetWords Sm90Measured.model (Pl.run x)) → (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm90Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm90.checks) * Sm90Measured.model.Nw) Sm90Measured.lamStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork → ProofSoundAt pf δzk → ProofsPin pf →
    RightFixed Pl pf Sm90Measured.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm90Measured.model.credit Sm90Measured.flat < 1 - 1412 / 100000) ≤ (1 / 2) ^ 40

/-- **The H200's headline on its measured vector under (I3) at 3/5**, the twin of `Sm90MeasuredHeadline`, under full
checking with live coins. On DC-90-ri's machine (`Sm90Measured.machine`), the same TS model at (I3)'s floors
(`Sm90MeasuredI3.model`: `Sm90Measured.model`'s writer kinds, vector classes and fold routes, `q = q_T = 3/5`,
`b_T = 0`, `N_w = 4096`) and the same floor program (`Sm90Measured.flat`), over `Sm90.checks = 48,000` rounds on any
finite coins `Ω`: for any window's play `Pl` and proof layer `pf` whose binding and soundness cost at most `2^-41` over
the window, if the labelling splits, (I1) `FacetCeilings` holds on DC-90-ri's rows, (I2)/(I4) `WordsWritten`, (I3)
`ScalarRoute` and (I5) `NoSum` hold on every run, FOLD-OPT per word folded at `π*` (`Sm90MeasuredI3.lamStar`: JI90_0
0.819832 + JI90_1 0.033778 + JPA90_0 0.146390, rounded) prices every run's fold at the words it folds (its unfolded
share `Uf`), and the luck and proof premises hold as in `Sm90SheetHeadline` (`BindingForkAt` at `δ_fork`,
`ProofSoundAt`, `ProofsPin`), then outside an event of probability at most `2^-40` every accepted window's time is at
least `1 − 1086/100000` (`1 − 1.086%`, over 1%) of the credit `T_cred = B_♭/f_max`. The folded branch binds, at exactly
`5625257050663081207487178157051972415029440436967/518065046570785549145507812500000000000000000000000`
(1.0858206%), reading `ln(8/7) ≥ 2/15`; the unfolded gives 1.0274482%.

So the 1.086% rests on (I3) at 3/5 (`accounting-ts-scalar-route`), an assumed premise, where `Sm90MeasuredHeadline`'s
1.412% rests on A6 alone, and both are over 1%. Nothing proved gives 3/5 here: A6 gives ½, since it lets a 64-bit head
(IMAD.WIDE, DFMA) write two words a lane, and `ClassPSlots`, which counts at least one slot a word, takes each output
writer's slot to write nothing else, the case those heads break. 3/5 is the least floor of denominator at most 5 at
which the bound is at its least: the same 1.086% at every `q` from about 0.5884 up, and under 1% at no `q` (sm_120's
measured headline reads `9/8`, the machine model `1.22`). At 3/5 the lane route no longer writes words at the IGMMA's
rate on JI90_0, so `π*` puts 0.15 on JPA90_0; `φ*` there is the price of li-bulk's trimmed W = ∞ member, a counted fold,
not built, with the built CP_RED2 `4.6·10⁻¹⁰` of it above, and ♭'s fold, by built routes alone, is 1.0042 `φ*`; `w*` is
`UNI`'s word, under honest's IGMMA's by `6.6·10⁻⁹` of it. `B_♭`, ♭ and the cheater's menu are `Sm90MeasuredHeadline`'s.
FOLD-OPT at this `π*` is open and doesn't follow from FOLD-OPT at DC-90-ri's: its JPA90_0 weight prices ALU and the
IGMMA alone, so it prices the classes apart from DC-90-ri's `π*`. Every class a fold route issues prices above 0 here (a
free class would make FOLD-OPT harder to meet, not easier). The two-slot round trip on CP_RED8, at `φ*·(1 − 3.32·10⁻¹⁰)`
at this statement's earlier `π*`, is off the menu, since li's XOR has three inputs a half step and one ATOM.E.XOR and
one LD.E carry two; at this `π*` it prices at 1.28 `φ*`, and the three-slot one, on the menu, at 1.69 `φ*`. FOLD-OPT's
premise has no slack; the reading, over 1% already, reaches its 3% target with every fold route 2.86% cheaper. Cite it
with its premise: DC-90-ri 1.086% under (I3) at 3/5; 1.412% at the proved ½ floor; credit against the built ♭, the
cheater priced at every known fold, counted ones included. -/
def Sm90MeasuredI3Headline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm90.checks (Fin 50) (Fin 2) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm90.checks nπ V) (δfork δzk : ℝ),
    (Sm90.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm90Measured.machine (Pl.run x).run) →
    (∀ x, WordsWritten Sm90MeasuredI3.model (Pl.run x)) → (∀ x, ScalarRoute Sm90MeasuredI3.model (Pl.run x)) →
    (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm90MeasuredI3.model (Pl.run x)
      ((1 - Uf.share x / Sm90.checks) * Sm90MeasuredI3.model.Nw) Sm90MeasuredI3.lamStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork → ProofSoundAt pf δzk → ProofsPin pf →
    RightFixed Pl pf Sm90MeasuredI3.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm90MeasuredI3.model.credit Sm90Measured.flat < 1 - 1086 / 100000) ≤ (1 / 2) ^ 40

/-- **The B200's headline at the data sheet**, under full checking with live coins. On DS-100's machine (`Sm100.sheet`),
`dither-2s-licpw-redux-ldtm0-l32`'s TS model (`Sm100.model`: every census MMA class a writer kind at its committed words
an instruction and `UNI` at 2, `q = q_T = ½`, `N_w = 32768`) and its floor program (`Sm100.flat`, crediting no
`tcgen05.ld`), over `Sm100.checks = 48,000` rounds on any finite coins `Ω`: for any window's play `Pl` and proof layer
`pf` whose binding and soundness cost at most `2^-41` over the window, if the labelling splits, A1–A7 hold on every run
and FOLD-OPT per word folded at `π*` (`Sm100.lamStar`, DS_issue ⅔ + DS_alu ⅓) prices every run's fold at the words it
folds (its unfolded share `Uf`), and the luck and proof premises hold as in `Sm90SheetHeadline` (`BindingForkAt` at
`δ_fork`, `ProofSoundAt`, `ProofsPin`), then outside an event of probability at most `2^-40` every accepted window's
time is at least `1 − 783/100000` (`1 − 0.783%`) of the credit `T_cred = B_♭/f_max`. The unfolded branch binds, at
exactly `40108546277242306023361573/5123200000000000000000000000` (0.7828807%); the folded gives 0.6983541%. At this
`π*` a LOP3 costs 4/3 and an IMAD or FFMA ⅔, so FOLD-OPT says, among other things, that li's XOR of two words has no
IMAD- or FP-pipe program of 1 instruction. Within a modelled menu (IMAD, IMAD.HI, IDP.4A, IDP.2A and FFMA2; IMAD.WIDE,
HFMA2 and the ALU class outside it) that is a checked fact, and so is the next count: no program of 1 or 2 instructions
computes li's `x ⊕ y ⊕ K` for any `K` (an exhaustive search: k = 1, and k = 2, all 1,456 two-instruction sequences
unsat), so such a program costs at least 2 here. It is not proved over the B200's whole instruction set. FOLD-OPT's
premise has no slack (`φ*` is CP_RED2's and CP_RED8's price); the reading's against 1% is 0.219%. FILL-OPT is not a
hypothesis: the design has no filler, so it holds at any run. -/
def Sm100SheetHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm100.checks (Fin 52) (Fin 27) Unit Empty) (G : ReceiptGame Ω Sm100.checks)
    (Uf : Unfolded Ω Sm100.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm100.checks nπ V) (δfork δzk : ℝ),
    (Sm100.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, SheetClock Sm100.sheet (Pl.run x).run) → (∀ x, SheetDispatch (Pl.run x).run) →
    (∀ x, SheetTensor Sm100.sheet (Pl.run x).run) → (∀ x, SheetPipe Sm100.sheet (Pl.run x).run) →
    (∀ x, SheetSmem Sm100.sheet (Pl.run x).run) → (∀ x, SheetWords Sm100.model (Pl.run x)) →
    (∀ x, SheetSemantics Sm100.model (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm100.model (Pl.run x) ((1 - Uf.share x / Sm100.checks) * Sm100.model.Nw)
      Sm100.lamStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork → ProofSoundAt pf δzk → ProofsPin pf → RightFixed Pl pf Sm100.model.Nw →
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm100.model.credit Sm100.flat < 1 - 783 / 100000) ≤
      (1 / 2) ^ 40

/-- **The B200's headline on its measured vector**, under full checking with live coins. On DC-100-ri's machine
(`Sm100Measured.machine`), `dither-2s-licpw-redux-ldtm0-l32`'s TS model on it (`Sm100Measured.model`: every MMA class
DC-100-ri times and `UNI` at 2 the writer kinds, the forms it doesn't time off the menu, `q = q_T = ½`, `b_T = 0`,
`N_w = 32768`) and its floor program (`Sm100Measured.flat`), over `Sm100.checks = 48,000` rounds on any finite coins
`Ω`: for any window's play `Pl` and proof layer `pf` whose binding and soundness cost at most `2^-41` over the window,
if the labelling splits, (I1) `FacetCeilings` holds on DC-100-ri's rows, A6 (`SheetWords`) and (I5) `NoSum` hold on
every run and FOLD-OPT per word folded at `π*` (`Sm100Measured.lamStar`, issue alone) prices every run's fold at the
words it folds (its unfolded share `Uf`), and the luck and proof premises hold as in `Sm90SheetHeadline`
(`BindingForkAt` at `δ_fork`, `ProofSoundAt`, `ProofsPin`), then outside an event of probability at most `2^-40` every
accepted window's time is at least `1 − 900/100000` (`1 − 0.900%`) of the credit `T_cred = B_♭/f_max`, under 1%. The
unfolded branch binds, at exactly `71934604043397/8000000000000000` (0.8991826%); the folded gives 0.8149027%. With each
new class at one issue slot and no pipe row (the sound price: no row is asserted that nobody timed), ♭ folds by the
global CP_RED2, at `π*` the price of CP_RED2, the cheater's cheapest, so `1 − Y` is ε_skip alone (0.8145%); priced with
each new class as SHFL on every row, it reads 1.2401% here, over 1%. FOLD-OPT's premise has no slack; the reading's
against 1% is 0.102%. FILL-OPT is not a hypothesis: the design has no filler. -/
def Sm100MeasuredHeadline : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (Pl : Play Ω Sm100.checks (Fin 52) (Fin 13) Unit Empty) (G : ReceiptGame Ω Sm100.checks)
    (Uf : Unfolded Ω Sm100.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm100.checks nπ V) (δfork δzk : ℝ),
    (Sm100.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) ≤ (1 / 2) ^ 41 → (∀ x, (Pl.run x).Split) →
    (∀ x, FacetCeilings Sm100Measured.machine (Pl.run x).run) →
    (∀ x, SheetWords Sm100Measured.model (Pl.run x)) → (∀ x, NoSum (Pl.run x)) →
    (∀ x, FoldOptFoldedAt Sm100Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm100.checks) * Sm100Measured.model.Nw) Sm100Measured.lamStar) →
    Pl.game = G.game → G.Pinned → OperandHMu (3 / 4) G → ReceiptLuckAt (1 / 2) G → WholeLuckAt (1 / 2) G Uf →
    pf.Of Pl → BindingForkAt pf δfork → ProofSoundAt pf δzk → ProofsPin pf →
    RightFixed Pl pf Sm100Measured.model.Nw →
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm100Measured.model.credit Sm100Measured.flat < 1 - 900 / 100000) ≤ (1 / 2) ^ 40

end ResourceAccounting.Guarantees.TS
