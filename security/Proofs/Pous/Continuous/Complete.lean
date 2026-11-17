module

public import Proofs.Pous.Statements
public import Proofs.Pous.Graded.Tail
public import Proofs.Pous.Continuous.PerAnswer

/-! The proof of `Pous.Guarantees.ContinuousComplete`, the graded audit's tail rule read with a miss in place of a
hit. -/

@[expose] public section

namespace PousContinuous

open Pous PousGraded PousThm1
open scoped ENNReal

variable {M : IdealModel}

section

variable {sch : Scheme M} (mach : ContMachine sch) (D Q D_late Q_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t)
  (C : sch.Code) (ω : M.Ω) (arr : ℕ → ℕ)

theorem after_contNext : ∀ (j : ℕ) (h : List (Fin sch.B)) (s : ContState sch.B),
    after (fun _ => contNext mach D_late W pp C ω arr) j h s = h.foldl (contNext mach D_late W pp C ω arr) s
  | _, [], _ => rfl
  | j, _ :: is, _ => after_contNext (j + 1) is _

open Classical in
theorem misses_hits : ∀ (j : ℕ) (l : List (Fin sch.B)) (s : ContState sch.B),
    misses (fun _ => contNext mach D_late W pp C ω arr)
        (fun _ s i => (contRecord mach D Q D_late Q_late W pp C ω arr s i).verdict ≠ .hit) j l s =
      ((contRecords mach D Q D_late Q_late W pp C ω arr s l).filter fun r => r.verdict = .hit).length
  | _, [], _ => rfl
  | j, i :: is, s => by
    rw [misses, misses_hits (j + 1) is, contRecords, List.filter_cons]
    by_cases hh : (contRecord mach D Q D_late Q_late W pp C ω arr s i).verdict = .hit
    · simp only [hh, ne_eq, not_true_eq_false, ↓reduceIte, decide_true, List.length_cons]
      omega
    · simp only [hh, ne_eq, not_false_eq_true, ↓reduceIte, decide_false, Bool.false_eq_true, zero_add]

theorem contRecords_length : ∀ (s : ContState sch.B) (l : List (Fin sch.B)),
    (contRecords mach D Q D_late Q_late W pp C ω arr s l).length = l.length
  | _, [] => rfl
  | s, i :: is => by
    rw [contRecords, List.length_cons, List.length_cons, contRecords_length _ is]

end

theorem length_filter_add_not {α : Type} (P : α → Prop) [DecidablePred P] :
    ∀ l : List α, (l.filter fun r => decide (¬ P r)).length + (l.filter fun r => decide (P r)).length = l.length
  | [] => rfl
  | r :: l => by
    have := length_filter_add_not P l
    by_cases hr : P r
    · simp only [List.filter_cons, hr, not_true_eq_false, decide_false, decide_true, Bool.false_eq_true,
        ↓reduceIte, List.length_cons]
      omega
    · simp only [List.filter_cons, hr, not_false_eq_true, decide_false, decide_true, Bool.false_eq_true,
        ↓reduceIte, List.length_cons]
      omega

theorem binTail_eq_upper (n t : ℕ) (c : ℝ≥0∞) (ht : t ≤ n) :
    binTail n t c = ∑ m ∈ Finset.Icc (n - t) n, (n.choose m : ℝ≥0∞) * c ^ m * (1 - c) ^ (n - m) := by
  unfold binTail
  refine Finset.sum_nbij' (fun j => n - j) (fun m => n - m) ?_ ?_ ?_ ?_ ?_
  · intro j hj
    rw [Finset.mem_range] at hj
    rw [Finset.mem_Icc]
    omega
  · intro m hm
    rw [Finset.mem_Icc] at hm
    rw [Finset.mem_range]
    omega
  · intro j hj
    rw [Finset.mem_range] at hj
    omega
  · intro m hm
    rw [Finset.mem_Icc] at hm
    omega
  · intro j hj
    rw [Finset.mem_range] at hj
    rw [Nat.choose_symm (by omega), Nat.sub_sub_self (by omega)]
    ring

section

variable {sch : Scheme M} (mach : ContMachine sch) (D Q D_late Q_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t)
  (C : sch.Code) (ω : M.Ω) (arr : ℕ → ℕ)

open Classical in
theorem pr_reject_le (n : ℕ) (q : ℝ≥0∞)
    (hq : ∀ h : List (Fin sch.B), h.length < n →
      pr (fun i => (contRecord mach D Q D_late Q_late W pp C ω arr (contAfter mach D_late W pp C ω arr h) i).verdict
        ≠ .hit) ≤ q)
    (f₀ : ℝ) :
    pr (fun I : Fin n → Fin sch.B =>
        ¬ contAccepts f₀ (contRecords mach D Q D_late Q_late W pp C ω arr .start (List.ofFn I))) ≤
      ∑ m ∈ (Finset.range (n + 1)).filter (fun m : ℕ => f₀ < m),
        (n.choose m : ℝ≥0∞) * q ^ m * (1 - q) ^ (n - m) := by
  set A := (Finset.range (n + 1)).filter fun m : ℕ => f₀ < m with hA
  let rs : (Fin n → Fin sch.B) → List ContRecord := fun I =>
    contRecords mach D Q D_late Q_late W pp C ω arr .start (List.ofFn I)
  have hcount : ∀ I, ((rs I).filter fun r => decide (r.verdict ≠ .hit)).length +
      ((rs I).filter fun r => decide (r.verdict = .hit)).length = n := fun I => by
    rw [length_filter_add_not, contRecords_length, List.length_ofFn]
  have hrej : ∀ I, ¬ contAccepts f₀ (rs I) →
      ((rs I).filter fun r => decide (r.verdict ≠ .hit)).length ∈ A := fun I hI => by
    rw [hA, Finset.mem_filter, Finset.mem_range]
    exact ⟨by have := hcount I; omega, lt_of_not_ge hI⟩
  by_cases hne : A.Nonempty
  · set m₀ := A.min' hne with hm₀
    have hm₀A : m₀ ∈ A := Finset.min'_mem A hne
    rw [hA, Finset.mem_filter, Finset.mem_range] at hm₀A
    have hmem : ∀ m, m ∈ A ↔ m₀ ≤ m ∧ m ≤ n := fun m => by
      constructor
      · intro hm
        refine ⟨Finset.min'_le A m hm, ?_⟩
        rw [hA, Finset.mem_filter, Finset.mem_range] at hm
        omega
      · rintro ⟨h1, h2⟩
        rw [hA, Finset.mem_filter, Finset.mem_range]
        exact ⟨by omega, hm₀A.2.trans_le (by exact_mod_cast h1)⟩
    have hAeq : A = Finset.Icc m₀ n := by
      ext m
      rw [hmem, Finset.mem_Icc]
    rw [hAeq]
    by_cases hq1 : q ≤ 1
    ·
      have htail := pr_hits_ge_le_binomial (fun _ => contNext mach D_late W pp C ω arr)
        (fun _ s i => (contRecord mach D Q D_late Q_late W pp C ω arr s i).verdict ≠ .hit) q hq1 n (n - m₀) 0
        .start fun h hh => by
          rw [after_contNext]
          exact hq h hh
      rw [binTail_eq_upper n (n - m₀) q (Nat.sub_le n m₀), Nat.sub_sub_self (by omega)] at htail
      refine le_trans (pr_mono fun I hI => ?_) htail
      rw [misses_hits]
      have h1 := (hmem _).1 (hrej I hI)
      have h2 := hcount I
      change ((rs I).filter fun r => decide (r.verdict = .hit)).length ≤ n - m₀
      omega
    ·
      refine (pr_le_one _).trans (le_trans ?_ (Finset.single_le_sum (fun _ _ => zero_le)
        (Finset.mem_Icc.2 ⟨by omega, le_rfl⟩)))
      rw [Nat.choose_self, Nat.cast_one, one_mul, Nat.sub_self, pow_zero, mul_one]
      exact one_le_pow₀ (le_of_lt (lt_of_not_ge hq1))
  ·
    rw [Finset.not_nonempty_iff_eq_empty] at hne
    refine le_trans (pr_mono fun I hI => ?_) (pr_false'.trans_le (zero_le))
    have := hrej I hI
    rw [hne] at this
    exact Finset.notMem_empty _ this

end

open Classical in
theorem continuous_complete : Pous.Guarantees.ContinuousComplete := by
  intro M sch D Q D_late Q_late n q W mach arr hq f₀
  have : Nonempty (Fin sch.B) := ⟨⟨0, sch.B_pos⟩⟩
  let rej : M.Ω × sch.Coins → (Fin n → Fin sch.B) → Prop := fun a I =>
    ¬ contAccepts f₀ (contRecords mach D Q D_late Q_late W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1 arr .start
      (List.ofFn I))
  have hre : pr (fun x : Outcome sch n => rej (x.1, x.2.1) x.2.2) =
      pr (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => rej y.1 y.2) :=
    pr_equiv (Equiv.prodAssoc _ _ _).symm (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => rej y.1 y.2)
  refine le_of_eq_of_le hre (pr_prod_le_of_forall rej _ fun a => ?_)
  exact pr_reject_le mach D Q D_late Q_late W _ _ a.1 arr n q (hq a) f₀

def honest (sch : Scheme M) : ContMachine sch where
  rest _ _ C _ _ _ _ := List.ofFn fun k : Fin (sch.B * sch.ℓ) => C (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2
  step _ _ _ m msg := .inl fun b => m.getD (finProdFinEquiv (msg.getD ⟨0, sch.B_pos⟩, b)).val false

section

variable {sch : Scheme M} (D Q D_late Q_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t) (C : sch.Code) (ω : M.Ω)
  (arr : ℕ → ℕ)

theorem honest_held (s : ContState sch.B) : (contMem (honest sch) W pp C ω arr s).length = sch.B * sch.ℓ :=
  List.length_ofFn

theorem honest_step (h : List (Fin sch.B)) (m : List Bool) (i : Fin sch.B) :
    (honest sch).step W pp h m (some i) = .inl fun b => m.getD (finProdFinEquiv (i, b)).val false :=
  rfl

theorem honest_hit (s : ContState sch.B) (i : Fin sch.B) :
    (contRecord (honest sch) D Q D_late Q_late W pp C ω arr s i).verdict = .hit := by
  have hround : ∀ T, (honest sch).round W pp s.sent T (contMem (honest sch) W pp C ω arr s) (some i) =
      .ret (some (C i), contMem (honest sch) W pp C ω arr s) := by
    intro T
    have hans : (fun b => (contMem (honest sch) W pp C ω arr s).getD (finProdFinEquiv (i, b)).val false) = C i := by
      funext b
      have hlt : (finProdFinEquiv (i, b)).val < (contMem (honest sch) W pp C ω arr s).length := by
        rw [honest_held]
        exact (finProdFinEquiv (i, b)).isLt
      rw [← List.getElem_eq_getD (h := hlt) false]
      simp only [contMem, honest, List.getElem_ofFn, Fin.eta, Equiv.symm_apply_apply]
    cases T <;> simp only [ContMachine.round, honest_step, hans]
  simp only [contRecord, contProg, hround, contVerdict, OracleAlgorithm.run, OracleAlgorithm.work, OracleAlgorithm.depth, Nat.not_lt_zero,
    or_self, ↓reduceIte]

end

theorem honest_complete (sch : Scheme M) (D Q D_late Q_late n : ℕ) (W : Bits sch.n) (arr : ℕ → ℕ) (f₀ : ℝ)
    (hf : 0 ≤ f₀) :
    pr (fun x : Outcome sch n =>
        ¬ contAccepts f₀ (contRecords (honest sch) D Q D_late Q_late W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2
          x.1 arr .start (List.ofFn x.2.2))) = 0 := by
  refine le_antisymm (le_trans (continuous_complete sch D Q D_late Q_late n 0 W (honest sch) arr
    (fun a h _ => le_of_eq ?_) f₀) (le_of_eq ?_)) (zero_le)
  · refine le_antisymm (le_trans (pr_mono fun i hi => hi (honest_hit D Q D_late Q_late W _ _ a.1 arr _ i))
      (le_of_eq pr_false')) (zero_le)
  · refine Finset.sum_eq_zero fun m hm => ?_
    rw [Finset.mem_filter] at hm
    have : m ≠ 0 := fun h0 => by
      rw [h0, Nat.cast_zero] at hm
      exact absurd hm.2 (not_lt.2 hf)
    rw [zero_pow this, mul_zero, zero_mul]

end PousContinuous
