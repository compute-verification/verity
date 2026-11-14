module

public import Definitions.NetworkCertifier.Guarantees.MonitoringPeriodFails
public import Proofs.NetworkCertifier.MonitoringPeriodChecked
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
import Mathlib.Order.CompletePartialOrder
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
# The period's leak with each failure charged as it happens

`monitoringPeriodLeakFails`, `monitoringPeriodLeakFailsAtMost` and `monitoringPeriodLeakCharged` are `Guarantees.MonitoringPeriodLeakFails`,
`Guarantees.MonitoringPeriodLeakFailsAtMost` and `Guarantees.MonitoringPeriodLeakCharged`. Each is `monitoringPeriodLeakChecked` applied once a fail
pattern and summed over the patterns, so each rests on what `Guarantees.MonitoringPeriodLeakChecked` rests on and, for the
charged form, on the weights' sum.

**At one pattern.** `checked_at_pattern` is `monitoringPeriodLeakChecked` at the secrets of `Zs` whose run has the fail pattern
`φ`, at the pattern's operator `op φ` and registered values `ρ φ`: both are fixed before the secrets, so the theorem
reads them as its own operator and registered values. It gives the held clause at that pattern and the count
`#Θ · windowsCount` of the pattern's secrets.

**Over the patterns.** The fail patterns of exactly `j` connections, each below its bound, are among
`failPatterns bound j` patterns (`exists_failPatterns`), so a count that holds at each pattern holds, times that
number, over them (`encard_failsExactly_le`, by `encard_biUnion_le`): `monitoringPeriodLeakFails`. Summed over `0 … j` it is
`monitoringPeriodLeakFailsAtMost`, whose held clause is `checked_at_pattern`'s at each run's own pattern. Under weights that sum
to at most one, sets of at most `c j` secrets with `c j ≤ w j · T` wherever one is inhabited hold at most `T` secrets
together (`encard_weighted_le`): `monitoringPeriodLeakCharged`.

**The factor.** `failPatterns bound 0 = 1` (`failPatterns_zero`), at one bound `c` it is `C(N, j) · c^j`
(`failPatterns_const`), and over every number of failures it sums to `∏ (bound + 1)` (`failPatterns_sum`). A fail cell
is one of its lane's cells, so every pattern is below a bound on the schedule's cells (`failsBelow_of_cells_le`).
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling Finset
open scoped NNRat

theorem failPatterns_zero {κ : Type} [Fintype κ] (bound : κ → ℕ) : failPatterns bound 0 = 1 := by
  rw [failPatterns, Finset.powersetCard_zero, Finset.sum_singleton, Finset.prod_empty]

/-- At one bound `c`: `C(N, j) · c^j`. -/
theorem failPatterns_const {κ : Type} [Fintype κ] (c j : ℕ) :
    failPatterns (fun _ : κ => c) j = (Fintype.card κ).choose j * c ^ j := by
  have h : ∀ J ∈ (Finset.univ : Finset κ).powersetCard j, ∏ _c ∈ J, c = c ^ j := fun J hJ => by
    rw [Finset.prod_const, (Finset.mem_powersetCard.1 hJ).2]
  unfold failPatterns
  rw [Finset.sum_congr rfl h, Finset.sum_const, Finset.card_powersetCard, Finset.card_univ, smul_eq_mul]

/-- Every number of failures: `MonitoringPeriodLeakAccepted`'s factor for the cells. -/
theorem failPatterns_sum {κ : Type} [Fintype κ] (bound : κ → ℕ) :
    ∑ j ∈ Finset.range (Fintype.card κ + 1), failPatterns bound j = ∏ c, (bound c + 1) := by
  unfold failPatterns
  rw [Finset.prod_add_one, Finset.sum_powerset, Finset.card_univ]

/-- The fail cells a pattern whose failed connections are `J` may hold at connection `c`. -/
def failSlots {κ : Type} [DecidableEq κ] (bound : κ → ℕ) (J : Finset κ) (c : κ) : Finset (Option ℕ) :=
  if c ∈ J then (Finset.range (bound c)).image some else {none}

/-- **The fail patterns of exactly `j` connections, each below its bound, are among `failPatterns bound j`
patterns.** -/
theorem exists_failPatterns {κ : Type} [Fintype κ] (bound : κ → ℕ) (j : ℕ) :
    ∃ U : Finset (κ → Option ℕ), U.card ≤ failPatterns bound j ∧
      ∀ φ, failCount φ = j → FailsBelow bound φ → φ ∈ U := by
  classical
  refine ⟨(Finset.univ.powersetCard j).biUnion fun J => Fintype.piFinset (failSlots bound J), ?_,
    fun φ hj hb => ?_⟩
  · refine Finset.card_biUnion_le.trans (le_of_eq ?_)
    unfold failPatterns
    refine Finset.sum_congr rfl fun J _ => ?_
    have hcard : ∀ c, (failSlots bound J c).card = if c ∈ J then bound c else 1 := by
      intro c
      by_cases hc : c ∈ J
      · simp only [failSlots, hc, ↓reduceIte]
        rw [Finset.card_image_of_injective _ (Option.some_injective ℕ), Finset.card_range]
      · simp only [failSlots, hc, ↓reduceIte, Finset.card_singleton]
    rw [Fintype.card_piFinset]
    exact (Finset.prod_congr rfl fun c _ => hcard c).trans (Fintype.prod_ite_mem J bound)
  · refine Finset.mem_biUnion.2 ⟨Finset.univ.filter fun c => (φ c).isSome, ?_, ?_⟩
    · exact Finset.mem_powersetCard.2 ⟨Finset.subset_univ _, hj⟩
    · refine Fintype.mem_piFinset.2 fun c => ?_
      by_cases hc : (φ c).isSome
      · obtain ⟨i, hi⟩ := Option.isSome_iff_exists.1 hc
        have hmem : c ∈ Finset.univ.filter fun c => (φ c).isSome := Finset.mem_filter.2 ⟨Finset.mem_univ c, hc⟩
        simp only [failSlots, hmem, ↓reduceIte]
        exact Finset.mem_image.2 ⟨i, Finset.mem_range.2 (hb c i hi), hi.symm⟩
      · have hmem : c ∉ Finset.univ.filter fun c => (φ c).isSome := fun h => hc (Finset.mem_filter.1 h).2
        simp only [failSlots, hmem, ↓reduceIte]
        exact Finset.mem_singleton.2 (Option.not_isSome_iff_eq_none.1 hc)

/-- **A count that holds at each fail pattern, summed over the patterns of exactly `j` failures.** `pat z` is `z`'s
pattern and `G φ` the secrets counted at the pattern `φ`. -/
theorem encard_failsExactly_le {Z κ : Type} [Fintype κ] (bound : κ → ℕ) (j : ℕ) (pat : Z → κ → Option ℕ)
    (G : (κ → Option ℕ) → Set Z) (B : ℕ∞) (h : ∀ φ, {z | pat z = φ ∧ z ∈ G φ}.encard ≤ B) :
    {z | failCount (pat z) = j ∧ FailsBelow bound (pat z) ∧ z ∈ G (pat z)}.encard ≤
      ((failPatterns bound j : ℕ) : ℕ∞) * B := by
  classical
  obtain ⟨U, hU, hmem⟩ := exists_failPatterns bound j
  have hsub : {z | failCount (pat z) = j ∧ FailsBelow bound (pat z) ∧ z ∈ G (pat z)} ⊆
      ⋃ φ ∈ U, {z | pat z = φ ∧ z ∈ G φ} := by
    rintro z ⟨hj, hb, hg⟩
    exact Set.mem_iUnion₂.2 ⟨pat z, hmem _ hj hb, rfl, hg⟩
  calc _ ≤ (⋃ φ ∈ U, {z | pat z = φ ∧ z ∈ G φ}).encard := Set.encard_le_encard hsub
    _ ≤ ∑ φ ∈ U, ({z | pat z = φ ∧ z ∈ G φ} : Set Z).encard := encard_biUnion_le _ _
    _ ≤ ∑ _φ ∈ U, B := Finset.sum_le_sum fun φ _ => h φ
    _ = (U.card : ℕ∞) * B := by rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ ((failPatterns bound j : ℕ) : ℕ∞) * B := mul_le_mul' (by exact_mod_cast hU) le_rfl

/-- **Every fail cell is below a bound on the schedule's cells**: the fail cell is one of the lane's cells. -/
theorem failsBelow_of_cells_le {κ Cell : Type} (s : CertifierSchedule κ) (arr : κ → Arrivals Cell) (bound : κ → ℕ)
    (h : ∀ c, s.cells c ≤ bound c) : FailsBelow bound (failCell s arr) := fun c _ hi =>
  (List.mem_range.1 (List.mem_of_find?_eq_some hi)).trans_le (h c)

/-- **Weights that sum to at most one**: sets `S j`, each of at most `c j` elements, and `c j ≤ w j · T` wherever
`S j` has an element, hold at most `T` elements together. -/
theorem encard_weighted_le {Z : Type} (N T : ℕ) (w : ℕ → ℚ≥0) (hw : KraftWeights w N) (S : ℕ → Set Z) (c : ℕ → ℕ)
    (hS : ∀ j, (S j).encard ≤ (c j : ℕ∞)) (hc : ∀ j, (S j).Nonempty → (c j : ℚ≥0) ≤ w j * (T : ℚ≥0)) :
    (⋃ j ∈ Finset.range (N + 1), S j).encard ≤ (T : ℕ∞) := by
  classical
  have hw' : ∑ j ∈ Finset.range (N + 1), w j ≤ 1 := hw
  have key : ∀ j, ∃ mj : ℕ, (S j).encard ≤ (mj : ℕ∞) ∧ (mj : ℚ≥0) ≤ w j * (T : ℚ≥0) := by
    intro j
    rcases (S j).eq_empty_or_nonempty with he | hne
    · refine ⟨0, ?_, ?_⟩
      · rw [he, Set.encard_empty]
        exact zero_le
      · rw [Nat.cast_zero]
        exact zero_le
    · exact ⟨c j, hS j, hc j hne⟩
  choose m hm₁ hm₂ using key
  have hsum : ∑ j ∈ Finset.range (N + 1), m j ≤ T := by
    have h : ((∑ j ∈ Finset.range (N + 1), m j : ℕ) : ℚ≥0) ≤ (T : ℚ≥0) := by
      rw [Nat.cast_sum]
      calc ∑ j ∈ Finset.range (N + 1), (m j : ℚ≥0) ≤ ∑ j ∈ Finset.range (N + 1), w j * (T : ℚ≥0) :=
            Finset.sum_le_sum fun j _ => hm₂ j
        _ = (∑ j ∈ Finset.range (N + 1), w j) * (T : ℚ≥0) := (Finset.sum_mul _ _ _).symm
        _ ≤ 1 * (T : ℚ≥0) := mul_le_mul' hw' le_rfl
        _ = (T : ℚ≥0) := one_mul _
    exact_mod_cast h
  calc (⋃ j ∈ Finset.range (N + 1), S j).encard ≤ ∑ j ∈ Finset.range (N + 1), (S j).encard :=
        encard_biUnion_le _ _
    _ ≤ ∑ j ∈ Finset.range (N + 1), (m j : ℕ∞) := Finset.sum_le_sum fun j _ => hm₁ j
    _ = ((∑ j ∈ Finset.range (N + 1), m j : ℕ) : ℕ∞) := (Nat.cast_sum _ _).symm
    _ ≤ (T : ℕ∞) := by exact_mod_cast hsum

section Compiled

open Claims.SamplingGeneric (Arg Compiled)

/-- `monitoringPeriodLeakChecked` at one fail pattern `φ`: over the secrets of `Zs` whose run has the pattern, at the pattern's
operator and registered values. -/
theorem checked_at_pattern
    {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (H : Msg → Msg) (D : Compiled Coin A) (hks : D.KS H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ) (reg chk : AuditStatement → Finset ℕ)
    (valid₁ valid₂ : AuditStatement → (ℕ → Bool) → Prop) (ε₁ ε₂ : Msg → ℝ) (Dj : ℕ)
    {n : ℕ} (sw : Fin n → AuditStatement) (Pw : Fin n → Partition) (β : Fin n → ℕ)
    (hplaced : WindowsPlaced D.partition δ sw Pw β)
    (hchk : ∀ i, CheckedUnique (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i))
      (fun r => valid₁ (sw i) r ∧ valid₂ (sw i) r))
    {Conn : Type} (ρ : (Conn → Option ℕ) → Fin n → ℕ → Bool)
    (op : (Conn → Option ℕ) → Fin n → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool)
    {Cell Srv Dev Z Par : Type} (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin n → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
    (rd : Par → (Fin n → List (Obs L K S)) → Conn → List (ℕ × CellView Cell))
    (allows : Par → CertifierSchedule Conn → Prop) (hread : f.ReadsBy rd allows)
    (Θ : Set Par) (Zs : Set Z) (schedOf : Z → CertifierSchedule Conn) (srvOf : Z → Srv)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z)
    (hch : MonitoringPeriodChannelReg f allows Θ Zs schedOf srvOf dec fun p => (run tap p).1)
    (x : Z → Fin n → Msg) (wit : Z → Fin n → Wit) (tops : AuditStatement → List (Opening Bool))
    (hb : D.commit.Binds H)
    (htap : ∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → CertifierRegisters H D.commit (lay s) (k s)
      (op (failCell (schedOf z) (arr (srvOf z))) i (runsBefore (run tap (schedOf z, srvOf z)).2 i)) s (tops s)
      ((run tap (schedOf z, srvOf z)).2 i)) (φ : Conn → Option ℕ) :
    (∀ z ∈ Zs, failCell (schedOf z) (arr (srvOf z)) = φ →
        (∀ i, ExecutionPhaseGoodChecked H D L K S lay k δ reg (fun s r => valid₁ s r ∧ valid₂ s r) Dj (sw i) (ρ φ i) tops
          (x z i) (wit z i) ((run tap (schedOf z, srvOf z)).2 i)) →
        ∃ θ ∈ Θ, (run tap (schedOf z, srvOf z)).1 = rd θ (run tap (schedOf z, srvOf z)).2 ∧
          ∀ i, (run tap (schedOf z, srvOf z)).2 i ∈
            AccSeqsChecked (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i))
              (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ φ i))
              (fun r => valid₁ (sw i) r ∧ valid₂ (sw i) r) δ
              (op φ i (runsBefore (run tap (schedOf z, srvOf z)).2 i)) Dj) ∧
      {z | failCell (schedOf z) (arr (srvOf z)) = φ ∧ z ∈ Zs ∧
          ∀ i, ExecutionPhaseGoodChecked H D L K S lay k δ reg (fun s r => valid₁ s r ∧ valid₂ s r) Dj (sw i) (ρ φ i) tops
            (x z i) (wit z i) ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
        Θ.encard * ((windowsCount lay k δ (fun s => reg s ∪ chk s) Dj sw Pw β : ℕ) : ℕ∞) := by
  have hφ := (monitoringPeriodLeakChecked H D hks L K S lay k δ hδ reg chk valid₁ valid₂ ε₁ ε₂ Dj sw Pw β hplaced hchk
    (ρ φ) (op φ) arr f tap run htrust rd allows hread Θ
    {z | z ∈ Zs ∧ failCell (schedOf z) (arr (srvOf z)) = φ} schedOf srvOf dec
    (And.intro (fun z hz => hch.1 z hz.1) fun z hz => hch.2 z hz.1) x wit tops hb).2 fun z hz i s hs => by
      have h' := htap z hz.1 i s hs
      rw [hz.2] at h'
      exact h'
  refine ⟨fun z hz hp hg => hφ.1 z ⟨hz, hp⟩ hg, (Set.encard_le_encard ?_).trans hφ.2⟩
  rintro z ⟨hp, hz, hg⟩
  exact ⟨⟨hz, hp⟩, hg⟩

/-- **`Guarantees.MonitoringPeriodLeakFails`**: `monitoringPeriodLeakChecked` at each fail pattern (`checked_at_pattern`), summed over
the patterns of exactly `j` failures (`encard_failsExactly_le`). Its first clause is `monitoringPeriodLeakChecked`'s. -/
theorem monitoringPeriodLeakFails : Guarantees.MonitoringPeriodLeakFails := by
  intro Coin _ _ _ A H D hks L K S lay k δ hδ reg chk valid₁ valid₂ ε₁ ε₂ Dj n sw Pw β hplaced hchk Conn _ ρ op
    Cell Srv Dev Z Par arr f tap run htrust rd allows hread Θ Zs schedOf srvOf dec hch x wit tops hb bound j
  refine ⟨(monitoringPeriodLeakChecked H D hks L K S lay k δ hδ reg chk valid₁ valid₂ ε₁ ε₂ Dj sw Pw β hplaced hchk
    (ρ fun _ => none) (op fun _ => none) arr f tap run htrust rd allows hread Θ Zs
    schedOf srvOf dec hch x wit tops hb).1, fun htap => ?_⟩
  refine (Set.encard_le_encard ?_).trans
    (encard_failsExactly_le bound j (fun z => failCell (schedOf z) (arr (srvOf z)))
      (fun φ => {z | z ∈ Zs ∧ ∀ i, ExecutionPhaseGoodChecked H D L K S lay k δ reg (fun s r => valid₁ s r ∧ valid₂ s r) Dj
        (sw i) (ρ φ i) tops (x z i) (wit z i) ((run tap (schedOf z, srvOf z)).2 i)}) _ fun φ =>
      (checked_at_pattern H D hks L K S lay k δ hδ reg chk valid₁ valid₂ ε₁ ε₂ Dj sw Pw β hplaced hchk
        ρ op arr f tap run htrust rd allows hread Θ Zs
        schedOf srvOf dec hch x wit tops hb htap φ).2)
  rintro z ⟨hz, hj, hb', hg⟩
  exact ⟨hj, hb', hz, hg⟩

/-- **`Guarantees.MonitoringPeriodLeakFailsAtMost`**: held at each run's own pattern (`checked_at_pattern`), and
`monitoringPeriodLeakFails` summed over `0 … j`. -/
theorem monitoringPeriodLeakFailsAtMost : Guarantees.MonitoringPeriodLeakFailsAtMost := by
  intro Coin _ _ _ A H D hks L K S lay k δ hδ reg chk valid₁ valid₂ ε₁ ε₂ Dj n sw Pw β hplaced hchk Conn _ ρ op
    Cell Srv Dev Z Par arr f tap run htrust rd allows hread Θ Zs schedOf srvOf dec hch x wit tops hb bound j
  have hi := fun i => monitoringPeriodLeakFails
    H D hks L K S lay k δ hδ reg chk valid₁ valid₂ ε₁ ε₂ Dj sw Pw β hplaced hchk
    ρ op arr f tap run htrust rd allows hread Θ Zs
    schedOf srvOf dec hch x wit tops hb bound i
  refine ⟨(hi 0).1, fun htap => ⟨fun z hz hg =>
    (checked_at_pattern H D hks L K S lay k δ hδ reg chk valid₁ valid₂ ε₁ ε₂ Dj sw Pw β hplaced hchk
      ρ op arr f tap run htrust rd allows hread Θ Zs
      schedOf srvOf dec hch x wit tops hb htap (failCell (schedOf z) (arr (srvOf z)))).1 z hz rfl hg, ?_⟩⟩
  classical
  refine (Set.encard_le_encard (t := ⋃ i ∈ Finset.range (j + 1),
    {z | z ∈ Zs ∧ failCount (failCell (schedOf z) (arr (srvOf z))) = i ∧
      FailsBelow bound (failCell (schedOf z) (arr (srvOf z))) ∧
      ∀ w, ExecutionPhaseGoodChecked H D L K S lay k δ reg (fun s r => valid₁ s r ∧ valid₂ s r) Dj (sw w)
        (ρ (failCell (schedOf z) (arr (srvOf z))) w)
        tops (x z w) (wit z w) ((run tap (schedOf z, srvOf z)).2 w)}) ?_).trans ?_
  · rintro z ⟨hz, hj, hb', hg⟩
    exact Set.mem_iUnion₂.2 ⟨_, Finset.mem_range.2 (Nat.lt_succ_of_le hj), hz, rfl, hb', hg⟩
  · refine (encard_biUnion_le _ _).trans ?_
    rw [Nat.cast_sum, Finset.sum_mul]
    exact Finset.sum_le_sum fun i _ => (hi i).2 htap

/-- **`Guarantees.MonitoringPeriodLeakCharged`**: `monitoringPeriodLeakFails` at each number of failures, under the weights
(`encard_weighted_le`). -/
theorem monitoringPeriodLeakCharged : Guarantees.MonitoringPeriodLeakCharged := by
  intro Coin _ _ _ A H D hks L K S lay k δ hδ reg chk valid₁ valid₂ ε₁ ε₂ Dj n sw Pw β hplaced hchk Conn _ ρ op
    Cell Srv Dev Z Par arr f tap run htrust rd allows hread Θ Zs schedOf srvOf dec hch x wit tops hb bound w hw T htap
  refine (Set.encard_le_encard ?_).trans
    (encard_weighted_le (Fintype.card Conn) T w hw
      (fun j => {z | z ∈ Zs ∧ failCount (failCell (schedOf z) (arr (srvOf z))) = j ∧
        FailsBelow bound (failCell (schedOf z) (arr (srvOf z))) ∧
        ((failPatterns bound j * (Θ.card * windowsCount lay k δ (fun s => reg s ∪ chk s) Dj sw Pw β) : ℕ) : ℚ≥0) ≤
          w j * (T : ℚ≥0) ∧
        ∀ i, ExecutionPhaseGoodChecked H D L K S lay k δ reg (fun s r => valid₁ s r ∧ valid₂ s r) Dj (sw i)
          (ρ (failCell (schedOf z) (arr (srvOf z))) i) tops (x z i) (wit z i) ((run tap (schedOf z, srvOf z)).2 i)})
      (fun j => failPatterns bound j * (Θ.card * windowsCount lay k δ (fun s => reg s ∪ chk s) Dj sw Pw β))
      (fun j => ?_) fun j hne => ?_)
  · rintro z ⟨hz, hb', hle, hg⟩
    exact Set.mem_iUnion₂.2 ⟨failCount (failCell (schedOf z) (arr (srvOf z))),
      Finset.mem_range.2 (Nat.lt_succ_of_le (Finset.card_le_univ _)), hz, rfl, hb', hle, hg⟩
  · refine (Set.encard_le_encard ?_).trans (((monitoringPeriodLeakFails
      H D hks L K S lay k δ hδ reg chk valid₁ valid₂ ε₁ ε₂ Dj sw Pw β hplaced hchk
      ρ op arr f tap run htrust rd allows hread (Θ : Set Par) Zs
      schedOf srvOf dec hch x wit tops hb bound j).2 htap).trans (le_of_eq ?_))
    · rintro z ⟨hz, hj, hb', -, hg⟩
      exact ⟨hz, hj, hb', hg⟩
    · rw [Set.encard_coe_eq_coe_finsetCard, Nat.cast_mul, Nat.cast_mul]
  · obtain ⟨z, hz⟩ := hne
    exact hz.2.2.2.1

end Compiled

end NetTiming
