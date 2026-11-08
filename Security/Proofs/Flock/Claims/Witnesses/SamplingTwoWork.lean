module

public import Proofs.Flock.Claims.Witnesses.SamplingGeneric
public import Proofs.Flock.SamplingGeneric.TwoCount
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

/-!
# Compute credit is not certified with deferred interiors

Two-stage sampling commits only the replay blocks' boundaries at serve time and each sampled replay block's interior after
the sample, and `RTwo` reads a trace only off the interiors: two boundary traces that agree there are in it together
(`Claims.Sampling.rTwo_congr`). So what `CompilerKSTwo` certifies holds whether or not the interiors were ever
computed: work skipped at serve time and done late, for the sampled replay blocks only, passes as the work done in full.
The toy here is an explicit instance, at a binding commitment.

* `notNotStmt`: the program `x ↦ ¬¬x` (gate `0` the input, `1 = ¬0`, `2 = ¬1` returned), sampled `all`, with the proof
  blocks `{0}`, `{1}`, `{2}` (`notNotPt`) and the replay blocks `{0}`, `{1, 2}` (`notNotPρ`, first stage `all`, `k = 1`),
  gate `1` its replay block's interior. It is in both scopes (`inScope_notNot`, `inScopeTwo_notNot`).
* `notNotCm`: the commitment whose openings of `x ↦ ¬¬x`'s values at `true` pass, which binds (`notNotCm_binds`).
* `fullW` opens all three values; `lateW` only the input and the returned gate, what serving commits, with the work of
  `¬x` never done. Both have footprint zero (`notNot_footprint`); `fullW`'s one-stage incorrect set is empty and `lateW`'s
  is `{1, 2}` (`fullW_faulty`, `lateW_faulty`).
* `deferred_not_credited`: at `δ = 0`, `RTwo` takes both, and the one-stage relation `R` takes `fullW` and refuses
  `lateW`.

`R` reads every proof block's committed values. With the interiors committed at serve time, as a serve-time commitment
of every proof block's values does, the proof blocks are the replay blocks, the design is one-stage, and `CompilerKS`
at `R` applies: skipped work is an incorrect proof block.
-/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

open Claims.Sampling

/-- `x ↦ ¬¬x`: the input, its negation and the negation of that, returned. -/
def notNotProg : Prog := ⟨[.input, .not 0, .not 1], [2]⟩

/-- The program, sampled `all`. -/
def notNotStmt : AuditStatement := ⟨[], notNotProg, ⟨"", 0, []⟩, ⟨[.all]⟩, []⟩

/-- The proof blocks `{0}`, `{1}`, `{2}`: one class of one stratum. -/
def notNotPt : Partition := ⟨[[[[0], [1], [2]]]]⟩

/-- The replay blocks `{0}` and `{1, 2}`: one class of one stratum. Gate `1` is read only within its replay block, so
serving doesn't commit it. -/
def notNotPρ : Partition := ⟨[[[[0], [1, 2]]]]⟩

/-- The two-stage design: the replay blocks `notNotPρ`, its first stage `all`, `k = 1`. -/
def notNotTwo : Design.Two := ⟨notNotPρ, ⟨[.all]⟩, [1]⟩

/-- The commitment whose openings of `x ↦ ¬¬x`'s values at `true` pass (`false` at gate `1`, `true` elsewhere), with no
call. -/
def notNotCm : CommitmentScheme Bool := ⟨fun _ o => .ret (o.val == (o.pos != 1)), fun _ _ _ => []⟩

/-- The trace that opens every value. -/
def fullW : Wit := ⟨[⟨0, true, []⟩, ⟨1, false, []⟩, ⟨2, true, []⟩]⟩

/-- **The late trace**: what serving commits, the input and the returned gate, with gate `1` opened nowhere. -/
def lateW : Wit := ⟨[⟨0, true, []⟩, ⟨2, true, []⟩]⟩

theorem notNotCm_binds (H : Msg → Msg) : notNotCm.Binds H := by
  intro roots o o' hpos hne ho ho'
  have h1 : o.val = (o.pos != 1) := beq_iff_eq.1 ho
  have h2 : o'.val = (o'.pos != 1) := beq_iff_eq.1 ho'
  exact absurd (by rw [h1, h2, hpos]) hne

theorem fullW_val (H : Msg → Msg) : fullW.val H notNotCm notNotStmt 0 = some true ∧
    fullW.val H notNotCm notNotStmt 1 = some false ∧ fullW.val H notNotCm notNotStmt 2 = some true := by
  refine ⟨?_, ?_, ?_⟩ <;> simp [Wit.val, fullW, CommitmentScheme.Opens, notNotCm, OC.eval]

theorem lateW_val (H : Msg → Msg) : lateW.val H notNotCm notNotStmt 0 = some true ∧
    lateW.val H notNotCm notNotStmt 1 = none ∧ lateW.val H notNotCm notNotStmt 2 = some true := by
  refine ⟨?_, ?_, ?_⟩ <;> simp [Wit.val, lateW, CommitmentScheme.Opens, notNotCm, OC.eval]

/-! ## The scopes -/

theorem inScope_notNot : InScope notNotPt notNotStmt := by
  refine ⟨⟨fun g x hx i hi => ?_, by decide⟩, ⟨by decide, by decide, by decide, by decide, by decide⟩, ?_⟩
  · rcases g with _ | _ | _ | g
    · simp [notNotStmt, notNotProg] at hx
      subst hx
      simp [BGate.reads] at hi
    · simp [notNotStmt, notNotProg] at hx
      subst hx
      simp [BGate.reads] at hi
      omega
    · simp [notNotStmt, notNotProg] at hx
      subst hx
      simp [BGate.reads] at hi
      omega
    · simp [notNotStmt, notNotProg] at hx
  · refine ⟨rfl, ?_⟩
    simp [notNotStmt, notNotPt, OneStage.WellFormed]

theorem inScopeTwo_notNot : InScopeTwo notNotPt notNotTwo notNotStmt :=
  ⟨inScope_notNot.1, inScope_notNot.2.1, by decide, by decide,
    ⟨⟨rfl, by simp [notNotTwo, notNotPρ, OneStage.WellFormed]⟩, by decide, by decide⟩⟩

/-! ## Each proof block on `x ↦ ¬¬x`'s values -/

theorem notNot_correct0 {ref : ℕ → Option Bool} (h0 : ref 0 = some true) : notNotStmt.prog.Correct ref [0] := by
  have hg : notNotStmt.prog.gates[0]? = some .input := rfl
  intro g hg'
  obtain rfl := List.mem_singleton.1 hg'
  refine ⟨fun x hx i hi _ => ?_, fun _ => ⟨true, h0, ?_⟩⟩
  · rw [hg, Option.some.injEq] at hx
    subst hx
    simp [BGate.reads] at hi
  · rw [Prog.runIn]
    exact h0

theorem notNot_correct1 {ref : ℕ → Option Bool} (h0 : ref 0 = some true) (h1 : ref 1 = some false) :
    notNotStmt.prog.Correct ref [1] := by
  have hg : notNotStmt.prog.gates[1]? = some (.not 0) := rfl
  have hread : notNotStmt.prog.read ref 0 = ref 0 := rfl
  intro g hg'
  obtain rfl := List.mem_singleton.1 hg'
  refine ⟨fun x hx i hi _ => ?_, fun _ => ⟨false, h1, ?_⟩⟩
  · rw [hg, Option.some.injEq] at hx
    subst hx
    obtain rfl : i = 0 := by simpa [BGate.reads] using hi
    rw [hread, h0]
    simp
  · rw [Prog.runIn, hg]
    simp [BGate.val, hread, h0]

theorem notNot_correct2 {ref : ℕ → Option Bool} (h1 : ref 1 = some false) (h2 : ref 2 = some true) :
    notNotStmt.prog.Correct ref [2] := by
  have hg : notNotStmt.prog.gates[2]? = some (.not 1) := rfl
  have hread : notNotStmt.prog.read ref 1 = ref 1 := rfl
  intro g hg'
  obtain rfl := List.mem_singleton.1 hg'
  refine ⟨fun x hx i hi _ => ?_, fun _ => ⟨true, h2, ?_⟩⟩
  · rw [hg, Option.some.injEq] at hx
    subst hx
    obtain rfl : i = 1 := by simpa [BGate.reads] using hi
    rw [hread, h1]
    simp
  · rw [Prog.runIn, hg]
    simp [BGate.val, hread, h1]

/-! ## The two-stage footprint -/

/-- **The footprint is zero** at the interior `1 ↦ false` for every trace that opens the input and the returned gate
at `true`: whatever it opens at gate `1`, every proof block is correct on its replay block's values. -/
theorem notNot_footprint (H : Msg → Msg) {w : Wit} (h0 : w.val H notNotCm notNotStmt 0 = some true)
    (h2 : w.val H notNotCm notNotStmt 2 = some true) :
    footprint H notNotCm notNotStmt notNotPt notNotPρ w (fun _ _ => false) = fun _ _ => 0 := by
  classical
  have hn0 : (0 : ℕ) ∉ [1, 2] := by decide
  have hb1 : notNotStmt.prog.Boundary [1, 2] 1 = false := by decide
  have hb2 : notNotStmt.prog.Boundary [1, 2] 2 = true := by decide
  have hb0 : notNotStmt.prog.Boundary [0] 0 = true := by decide
  have r0 : notNotStmt.prog.refIn [0] (w.val H notNotCm notNotStmt) (fun _ => false) 0 = some true := by
    rw [Prog.refIn_of_boundary hb0, h0]
  have s0 : notNotStmt.prog.refIn [1, 2] (w.val H notNotCm notNotStmt) (fun _ => false) 0 = some true := by
    rw [Prog.refIn_of_not_mem hn0, h0]
  have s1 : notNotStmt.prog.refIn [1, 2] (w.val H notNotCm notNotStmt) (fun _ => false) 1 = some false := by
    simp [Prog.refIn, hb1]
  have s2 : notNotStmt.prog.refIn [1, 2] (w.val H notNotCm notNotStmt) (fun _ => false) 2 = some true := by
    rw [Prog.refIn_of_boundary hb2, h2]
  funext ρ t
  unfold footprint wrongIn
  refine Finset.card_eq_zero.2 (Finset.filter_eq_empty_iff.2 fun v hv => not_not.2 ?_)
  have hv3 : v < 3 := by simpa [notNotPt, Partition.blocks] using Finset.mem_range.1 (cell_subset _ _ _ _ hv)
  have hρ : replayOf notNotPt notNotPρ v = some ρ := (Finset.mem_filter.1 hv).2.1
  interval_cases v
  · obtain rfl : (0 : ℕ) = ρ :=
      Option.some.inj ((show replayOf notNotPt notNotPρ 0 = some 0 by decide).symm.trans hρ)
    exact ⟨[0], [0], rfl, rfl, notNot_correct0 r0⟩
  · obtain rfl : (1 : ℕ) = ρ :=
      Option.some.inj ((show replayOf notNotPt notNotPρ 1 = some 1 by decide).symm.trans hρ)
    exact ⟨[1], [1, 2], rfl, rfl, notNot_correct1 s0 s1⟩
  · obtain rfl : (1 : ℕ) = ρ :=
      Option.some.inj ((show replayOf notNotPt notNotPρ 2 = some 1 by decide).symm.trans hρ)
    exact ⟨[2], [1, 2], rfl, rfl, notNot_correct2 s1 s2⟩

/-! ## The one-stage incorrect sets -/

theorem notNot_notFaulty0 (H : Msg → Msg) {w : Wit} (h0 : w.val H notNotCm notNotStmt 0 = some true) :
    ¬ Faulty H notNotCm notNotStmt notNotPt w 0 := by
  have hb : notNotStmt.prog.Boundary [0] 0 = true := by decide
  exact fun hf => hf ⟨[0], rfl, fun _ => false, notNot_correct0 (by rw [Prog.refIn_of_boundary hb, h0])⟩

theorem notNot_notFaulty1 (H : Msg → Msg) {w : Wit} (h0 : w.val H notNotCm notNotStmt 0 = some true)
    (h1 : w.val H notNotCm notNotStmt 1 = some false) : ¬ Faulty H notNotCm notNotStmt notNotPt w 1 := by
  have hn : (0 : ℕ) ∉ [1] := by decide
  have hb : notNotStmt.prog.Boundary [1] 1 = true := by decide
  exact fun hf => hf ⟨[1], rfl, fun _ => false,
    notNot_correct1 (by rw [Prog.refIn_of_not_mem hn, h0]) (by rw [Prog.refIn_of_boundary hb, h1])⟩

theorem notNot_notFaulty2 (H : Msg → Msg) {w : Wit} (h1 : w.val H notNotCm notNotStmt 1 = some false)
    (h2 : w.val H notNotCm notNotStmt 2 = some true) : ¬ Faulty H notNotCm notNotStmt notNotPt w 2 := by
  have hn : (1 : ℕ) ∉ [2] := by decide
  have hb : notNotStmt.prog.Boundary [2] 2 = true := by decide
  exact fun hf => hf ⟨[2], rfl, fun _ => false,
    notNot_correct2 (by rw [Prog.refIn_of_not_mem hn, h1]) (by rw [Prog.refIn_of_boundary hb, h2])⟩

/-- Proof block `{1}` commits gate `1`, so a trace that opens no value there leaves it faulty. -/
theorem notNot_faulty1 (H : Msg → Msg) {w : Wit} (h : w.val H notNotCm notNotStmt 1 = none) :
    Faulty H notNotCm notNotStmt notNotPt w 1 := by
  rintro ⟨u, hu, ι, hc⟩
  rw [show notNotPt.blocks[1]? = some [1] from rfl, Option.some.injEq] at hu
  subst hu
  have hb : notNotStmt.prog.Boundary [1] 1 = true := by decide
  obtain ⟨b, hb', _⟩ := (hc 1 (List.mem_singleton_self 1)).2 hb
  rw [Prog.refIn_of_boundary hb, h] at hb'
  cases hb'

/-- Proof block `{2}` reads gate `1`, so a trace that opens no value there leaves it faulty. -/
theorem notNot_faulty2 (H : Msg → Msg) {w : Wit} (h : w.val H notNotCm notNotStmt 1 = none) :
    Faulty H notNotCm notNotStmt notNotPt w 2 := by
  rintro ⟨u, hu, ι, hc⟩
  rw [show notNotPt.blocks[2]? = some [2] from rfl, Option.some.injEq] at hu
  subst hu
  have hg : notNotStmt.prog.gates[2]? = some (.not 1) := rfl
  have hr := (hc 2 (List.mem_singleton_self 2)).1 _ hg 1 (by simp [BGate.reads]) (by decide)
  have hread : notNotStmt.prog.read (notNotStmt.prog.refIn [2] (w.val H notNotCm notNotStmt) ι) 1 =
      w.val H notNotCm notNotStmt 1 := by
    show notNotStmt.prog.refIn [2] (w.val H notNotCm notNotStmt) ι 1 = _
    exact Prog.refIn_of_not_mem (by decide)
  exact hr (hread.trans h)

theorem fullW_faulty (H : Msg → Msg) : faulty H notNotCm notNotStmt notNotPt fullW = ∅ := by
  obtain ⟨h0, h1, h2⟩ := fullW_val H
  ext v
  simp only [faulty, Finset.mem_filter, Finset.mem_range, Finset.notMem_empty, iff_false, not_and]
  intro hv
  have hv3 : v < 3 := by simpa [notNotPt, Partition.blocks] using hv
  interval_cases v
  · exact notNot_notFaulty0 H h0
  · exact notNot_notFaulty1 H h0 h1
  · exact notNot_notFaulty2 H h1 h2

theorem lateW_faulty (H : Msg → Msg) : faulty H notNotCm notNotStmt notNotPt lateW = {1, 2} := by
  obtain ⟨h0, h1, -⟩ := lateW_val H
  ext v
  simp only [faulty, Finset.mem_filter, Finset.mem_range, Finset.mem_insert, Finset.mem_singleton]
  constructor
  · rintro ⟨hv, hf⟩
    have hv3 : v < 3 := by simpa [notNotPt, Partition.blocks] using hv
    interval_cases v
    · exact absurd hf (notNot_notFaulty0 H h0)
    · exact Or.inl rfl
    · exact Or.inr rfl
  · rintro (rfl | rfl)
    · exact ⟨by decide, notNot_faulty1 H h1⟩
    · exact ⟨by decide, notNot_faulty2 H h1⟩

/-! ## The relations -/

theorem notNot_Pa (B : Finset ℕ) : notNotStmt.design.Pa notNotPt B = if window 0 3 B = ∅ then 1 else 0 := by
  simp [Design.Pa, notNotStmt, notNotPt, classesPa, OneStage.Pa]

theorem notNotTwo_replay_Pa (B : Finset ℕ) :
    notNotTwo.replay.Pa notNotTwo.part B = if window 0 2 B = ∅ then 1 else 0 := by
  simp [Design.Pa, notNotTwo, notNotPρ, classesPa, OneStage.Pa]

theorem notNotTwo_Pa_zero : notNotTwo.Pa notNotPt (fun _ _ => 0) = 1 := by
  have hw : window 0 2 (∅ : Finset ℕ) = ∅ := by decide
  rw [Design.Two.Pa_zero, notNotTwo_replay_Pa, if_pos hw]

/-- **Compute credit is not certified with deferred interiors**: at every parser and partitions that read the toy, at
`δ = 0`, `RTwo` takes the trace that never computes gate `1` as it takes the one that does, while the one-stage relation
`R` takes only the latter. -/
theorem deferred_not_credited (H : Msg → Msg) {parse : Msg → Option AuditStatement}
    {partition : AuditStatement → Option Partition} {two : AuditStatement → Option Design.Two} {x : Msg}
    (hs : parse x = some notNotStmt) (hPt : partition notNotStmt = some notNotPt)
    (hd : two notNotStmt = some notNotTwo) :
    lateW.val H notNotCm notNotStmt 1 = none ∧
      Claims.Sampling.RTwo H notNotCm parse partition two 0 x lateW ∧
      Claims.Sampling.RTwo H notNotCm parse partition two 0 x fullW ∧
      ¬ Claims.Sampling.R H notNotCm parse partition 0 x lateW ∧ Claims.Sampling.R H notNotCm parse partition 0 x fullW := by
  obtain ⟨l0, l1, l2⟩ := lateW_val H
  obtain ⟨f0, -, f2⟩ := fullW_val H
  have hM : ∀ w : Wit, w.val H notNotCm notNotStmt 0 = some true → w.val H notNotCm notNotStmt 2 = some true →
      footprint H notNotCm notNotStmt notNotPt notNotTwo.part w (fun _ _ => false) ∈ notNotTwo.M notNotPt 0 := by
    intro w h0 h2
    show (0 : ℝ) < notNotTwo.Pa notNotPt (footprint H notNotCm notNotStmt notNotPt notNotPρ w fun _ _ => false)
    rw [notNot_footprint H h0 h2, notNotTwo_Pa_zero]
    norm_num
  refine ⟨l1, ⟨_, _, _, hs, hPt, hd, _, hM lateW l0 l2⟩, ⟨_, _, _, hs, hPt, hd, _, hM fullW f0 f2⟩, ?_,
    ⟨_, _, hs, hPt, ?_⟩⟩
  · rintro ⟨s, P, hs', hP', hR⟩
    rw [hs, Option.some.injEq] at hs'
    subst hs'
    rw [hPt, Option.some.injEq] at hP'
    subst hP'
    have h : (0 : ℝ) < notNotStmt.design.Pa notNotPt (faulty H notNotCm notNotStmt notNotPt lateW) := hR
    have hw : window 0 3 ({1, 2} : Finset ℕ) ≠ ∅ := by decide
    rw [lateW_faulty, notNot_Pa, if_neg hw] at h
    exact lt_irrefl _ h
  · show (0 : ℝ) < notNotStmt.design.Pa notNotPt (faulty H notNotCm notNotStmt notNotPt fullW)
    have hw : window 0 3 (∅ : Finset ℕ) = ∅ := by decide
    rw [fullW_faulty, notNot_Pa, if_pos hw]
    norm_num

end Claims.Witness.SamplingGeneric
