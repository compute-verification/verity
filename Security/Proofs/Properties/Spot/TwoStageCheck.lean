module

public import Properties.Spot.TwoStageCheck
public import Proofs.Properties.Spot.FactorUpper
public import Proofs.Properties.Spot.Interval
public import Proofs.Flock.Claims.Witnesses.SamplingTwoRankin

/-!
# A two-stage point's bits bound its count

`Properties.Spot.TwoStageCheckSound` (the sets' form), `TwoStageCheckPatternsSound` (the patterns') and
`TwoStageCheckServedSound` (the capped sets'). Each composes Rankin's bound (`Claims.Sampling.admissibleTwo_rankin`,
`admissibleTwo_rankin_patterns`, `admissibleTwo_rankin_served`) at the uniform rate `D.q`, which the registered first
stage meets (`firstStage_le`, from `bernoulli_Pa_le` and `Design.Two.subsetInputs_Pa_le`), with each replay block's
factor bound at its class (`factorUpperSound`, `factorUpperPatternsSound`, `factorUpperCappedSound` at the class's cap
`N_c · S`): the product over replay blocks is at most `e^{Σ_c count_c max(v_c, 0)}`
(`prod_le_exp_classes`), `δ^{−θ} ≤ e^{θ lnUpper(1/δ)}` (`rpow_neg_le`), and `e^L ≤ 2^b` at `checkTwoStage`'s `b`
(`exp_le_two_rpow`).

Each honest witness concludes its guarantee's premises at a replay block of two one-bit proof blocks sampled
`bernoulli:1/2` (`FactorUpperWitness.pairDesign`), where the point returns, and the kernel's evaluation returns a point
of a million deployed-shaped replay blocks too. Each conclusion fails at a first stage that samples nothing
(`TwoStageCheckWitness.nothingDesign`, which is `nothingTwo`), at the `1` bit a point claiming the rate `1/2` there
could return: the count is `3` (`_nontrivial`).
-/

@[expose] public section

namespace Properties.Spot.SecurityProofs

open Finset
open Claims.Sampling
open _root_.Spot.TwoStageCheck
open _root_.Spot.Interval (lnLower lnUpper)

namespace TwoStageCheckProof

/-- **`classSum` is a sum over the classes**: each class's factor bound returns `v i`, and the sum is
`Σ_i count_i max(v_i, 0)`. -/
theorem classSum_spec (f : ReplayClass → Option ℚ) :
    ∀ (l : List ReplayClass) (x : ℚ), classSum f l = some x →
      ∃ v : ℕ → ℚ, (∀ i c, l[i]? = some c → f c = some (v i)) ∧
        x = ∑ i ∈ range l.length, ((l.getD i default).count : ℚ) * max (v i) 0
  | [], x, h => by
    simp only [classSum, Option.some.injEq] at h
    exact ⟨fun _ => 0, fun i c hc => by simp at hc, by simp [← h]⟩
  | c :: cs, x, h => by
    obtain ⟨v0, hc⟩ : ∃ v0, f c = some v0 := by
      cases hc : f c with
      | none => simp [classSum, hc] at h
      | some v0 => exact ⟨v0, rfl⟩
    obtain ⟨s, hs⟩ : ∃ s, classSum f cs = some s := by
      cases hs : classSum f cs with
      | none => simp [classSum, hc, hs] at h
      | some s => exact ⟨s, rfl⟩
    simp only [classSum, hc, hs, Option.some.injEq] at h
    obtain ⟨v, hv, rfl⟩ := classSum_spec f cs s hs
    refine ⟨fun i => if i = 0 then v0 else v (i - 1), fun i c' hi => ?_, ?_⟩
    · cases i with
      | zero =>
        simp only [List.getElem?_cons_zero, Option.some.injEq] at hi
        subst hi
        simpa using hc
      | succ i =>
        simp only [List.getElem?_cons_succ] at hi
        simpa using hv i c' hi
    · rw [← h, List.length_cons, sum_range_succ']
      simp only [List.getD_cons_succ, List.getD_cons_zero, ite_true, Nat.add_one_ne_zero, ite_false,
        Nat.add_sub_cancel]
      ring

theorem getElem?_getD {l : List ReplayClass} {i : ℕ} (hi : i < l.length) :
    l[i]? = some (l.getD i default) := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi]

/-- The product of the replay blocks' factors, each at most `e^{m_c}` at its class `c`, is at most
`e^{Σ_c count_c m_c}` when class `c` holds at most `count_c` replay blocks and each `m_c ≥ 0`. -/
theorem prod_le_exp_classes {n K : ℕ} {G m : ℕ → ℝ} {cnt : ℕ → ℕ} (cls : ℕ → ℕ)
    (hG0 : ∀ ρ < n, 0 ≤ G ρ) (hG : ∀ ρ < n, G ρ ≤ Real.exp (m (cls ρ))) (hm : ∀ i, 0 ≤ m i)
    (hcls : ∀ ρ < n, cls ρ < K) (hcount : ∀ i < K, ((range n).filter (cls · = i)).card ≤ cnt i) :
    ∏ ρ ∈ range n, G ρ ≤ Real.exp (∑ i ∈ range K, (cnt i : ℝ) * m i) := by
  calc ∏ ρ ∈ range n, G ρ ≤ ∏ ρ ∈ range n, Real.exp (m (cls ρ)) :=
        prod_le_prod₀ (fun ρ hρ => hG0 ρ (mem_range.1 hρ)) fun ρ hρ => hG ρ (mem_range.1 hρ)
    _ = Real.exp (∑ ρ ∈ range n, m (cls ρ)) := (Real.exp_sum _ _).symm
    _ ≤ Real.exp (∑ i ∈ range K, (cnt i : ℝ) * m i) := by
        refine Real.exp_le_exp.2 ?_
        rw [← sum_fiberwise_of_maps_to (g := cls) (t := range K)
          (fun ρ hρ => mem_range.2 (hcls ρ (mem_range.1 hρ))) (fun ρ => m (cls ρ))]
        refine sum_le_sum fun i hi => ?_
        refine (sum_congr rfl fun ρ hρ => congrArg m (mem_filter.1 hρ).2).trans_le ?_
        rw [sum_const, nsmul_eq_mul]
        exact mul_le_mul_of_nonneg_right (by exact_mod_cast hcount i (mem_range.1 hi)) (hm i)

/-- `δ^{−θ} ≤ e^{θ lnUpper(1/δ)}` at `δ > 0`, `θ ≥ 0`. -/
theorem rpow_neg_le {δ θ : ℚ} (p : ℕ) (hδ : 0 < δ) (hθ : 0 ≤ θ) :
    (δ : ℝ) ^ (-(θ : ℝ)) ≤ Real.exp ((θ * lnUpper (1 / δ) p : ℚ) : ℝ) := by
  rw [Real.rpow_def_of_pos (by exact_mod_cast hδ)]
  refine Real.exp_le_exp.2 ?_
  have h : -Real.log δ ≤ ((lnUpper (1 / δ) p : ℚ) : ℝ) := by
    have h1 : Real.log ((1 / δ : ℚ) : ℝ) = -Real.log δ := by
      rw [Rat.cast_div, Rat.cast_one, one_div, Real.log_inv]
    rw [← h1]
    exact lnUpperSound (1 / δ) p
  have hθ' : (0 : ℝ) ≤ θ := by exact_mod_cast hθ
  push_cast
  nlinarith [mul_le_mul_of_nonneg_left h hθ']

/-- `e^L ≤ 2^b` at `checkTwoStage`'s `b = L/lnLower 2 p` (`L ≥ 0`) or `L/lnUpper 2 p` (`L < 0`). -/
theorem exp_le_two_rpow (L : ℚ) (p : ℕ) (hlo : 0 < lnLower 2 p) :
    Real.exp L ≤ (2 : ℝ) ^ (((if 0 ≤ L then L / lnLower 2 p else L / lnUpper 2 p : ℚ)) : ℝ) := by
  rw [Real.rpow_def_of_pos two_pos]
  refine Real.exp_le_exp.2 ?_
  have h2 : (0 : ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  have hl : ((lnLower 2 p : ℚ) : ℝ) ≤ Real.log 2 := by simpa using lnLowerSound 2 p
  have hu : Real.log 2 ≤ ((lnUpper 2 p : ℚ) : ℝ) := by simpa using lnUpperSound 2 p
  have hlo' : (0 : ℝ) < (lnLower 2 p : ℚ) := by exact_mod_cast hlo
  split_ifs with hL
  · have hL' : (0 : ℝ) ≤ L := by exact_mod_cast hL
    push_cast
    rw [mul_div_assoc', le_div_iff₀ hlo']
    nlinarith [mul_le_mul_of_nonneg_left hl hL']
  · have hL' : (L : ℝ) < 0 := by exact_mod_cast not_le.1 hL
    have hup : (0 : ℝ) < (lnUpper 2 p : ℚ) := h2.trans_le hu
    push_cast
    rw [mul_div_assoc', le_div_iff₀ hup]
    nlinarith [mul_le_mul_of_nonpos_left hu hL'.le]

/-- A point `checkTwoStage` returns is in range. -/
theorem valid_of_check {D : Point} {b : ℚ} (hb : checkTwoStage D = some b) :
    0 ≤ D.q ∧ D.q ≤ 1 ∧ 0 ≤ D.θ ∧ 0 < D.δ ∧ 0 < lnLower 2 D.p := by
  by_cases hv : D.valid = true
  · simpa [Point.valid, and_assoc] using hv
  · simp [checkTwoStage, lnCount, hv] at hb

/-- **The count from Rankin's bound and the classes**: a count at most `δ^{−θ} ∏_ρ G_ρ`, each `G_ρ ≥ 0` at most
`e^v` for its class's factor bound `v`, is at most `2^b` at `checkTwoStage`'s `b`. -/
theorem bits_of_rankin {D : Point} {b : ℚ} (hb : checkTwoStage D = some b) {n : ℕ} {G : ℕ → ℝ} {C : ℝ}
    (cls : ℕ → ℕ) (hC : C ≤ (D.δ : ℝ) ^ (-(D.θ : ℝ)) * ∏ ρ ∈ range n, G ρ) (hG0 : ∀ ρ < n, 0 ≤ G ρ)
    (hG : ∀ ρ < n, ∀ c v, D.classes[cls ρ]? = some c → classFactor D c = some v → G ρ ≤ Real.exp v)
    (hcls : ∀ ρ < n, ∃ c, D.classes[cls ρ]? = some c)
    (hcount : ∀ i c, D.classes[i]? = some c → ((range n).filter (cls · = i)).card ≤ c.count) :
    C ≤ (2 : ℝ) ^ (b : ℝ) := by
  obtain ⟨-, -, hθ, hδ, hlo⟩ := valid_of_check hb
  obtain ⟨L, hL, rfl⟩ : ∃ L, lnCount D = some L ∧
      b = if 0 ≤ L then L / lnLower 2 D.p else L / lnUpper 2 D.p := by
    unfold checkTwoStage at hb
    cases h : lnCount D with
    | none => rw [h] at hb; exact absurd hb (by simp)
    | some L => rw [h] at hb; exact ⟨L, rfl, (Option.some.inj hb).symm⟩
  have hv : D.valid = true := by
    by_contra hv
    simp [lnCount, hv] at hL
  unfold lnCount at hL
  simp only [hv, ↓reduceIte] at hL
  obtain ⟨x, hx, hLx⟩ : ∃ x, classSum (classFactor D) D.classes = some x ∧
      D.θ * lnUpper (1 / D.δ) D.p + x = L := by
    cases h : classSum (classFactor D) D.classes with
    | none => rw [h] at hL; exact absurd hL (by simp)
    | some x => rw [h] at hL; exact ⟨x, rfl, Option.some.inj hL⟩
  obtain ⟨v, hvf, rfl⟩ := classSum_spec _ _ _ hx
  refine le_trans ?_ (exp_le_two_rpow L D.p hlo)
  have hlt : ∀ ρ < n, cls ρ < D.classes.length := fun ρ hρ => by
    obtain ⟨c, hc⟩ := hcls ρ hρ
    exact (List.getElem?_eq_some_iff.1 hc).1
  have hP := prod_le_exp_classes (m := fun i => max ((v i : ℚ) : ℝ) 0)
    (cnt := fun i => (D.classes.getD i default).count) cls hG0
    (fun ρ hρ => by
      obtain ⟨c, hc⟩ := hcls ρ hρ
      exact (hG ρ hρ c (v (cls ρ)) hc (hvf _ _ hc)).trans (Real.exp_le_exp.2 (le_max_left _ _)))
    (fun i => le_max_right _ _) hlt (fun i hi => hcount i _ (getElem?_getD hi))
  calc C ≤ (D.δ : ℝ) ^ (-(D.θ : ℝ)) * ∏ ρ ∈ range n, G ρ := hC
    _ ≤ Real.exp ((D.θ * lnUpper (1 / D.δ) D.p : ℚ) : ℝ) *
          Real.exp (∑ i ∈ range D.classes.length,
            ((D.classes.getD i default).count : ℝ) * max ((v i : ℚ) : ℝ) 0) :=
        mul_le_mul (rpow_neg_le D.p hδ hθ) hP (prod_nonneg fun ρ hρ => hG0 ρ (mem_range.1 hρ))
          (Real.exp_pos _).le
    _ = Real.exp L := by
        rw [← Real.exp_add, ← hLx]
        push_cast
        ring

/-- **The registered first stage meets Rankin's rates' hypothesis** at the uniform rate `q`. -/
theorem firstStage_le {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hs : InScopeTwo Pt d s) {q : ℝ}
    (hq1 : q ≤ 1) (h : FirstStageAt d q) :
    ∀ T ⊆ range d.part.blocks.length, d.replay.Pa d.part T ≤ ∏ _ρ ∈ T, (1 - q) := by
  intro T hT
  rcases h with ⟨num, den, hr, hle⟩ | ⟨Q, c, ic, hr, hP, hQ, hqQ⟩
  · obtain ⟨hpa, hle1⟩ := bernoulli_Pa_le hs hr
    exact (hpa T hT).trans (prod_le_prod₀ (fun _ _ => by linarith) fun _ _ => by linarith)
  · refine (Design.Two.subsetInputs_Pa_le d (Q := Q) hr hP hQ T hT).trans
      (prod_le_prod₀ (fun ρ _ => ?_) fun ρ _ => ?_)
    · split_ifs with hρ
      · have hN : (0 : ℝ) < c.flatten.length := by exact_mod_cast (Nat.zero_le ρ).trans_lt hρ
        rw [sub_nonneg, div_le_one hN]
        exact_mod_cast hQ
      · norm_num
    · split_ifs with hρ
      · have hN : (0 : ℝ) < c.flatten.length := by exact_mod_cast (Nat.zero_le ρ).trans_lt hρ
        rw [sub_le_sub_iff_left, le_div_iff₀ hN]
        exact hqQ
      · linarith

theorem rankinFactor_nonneg (s : AuditStatement) (Pt : Partition) (d : Design.Two) {q : ℝ} (hq : q ≤ 1) (θ : ℝ)
    (ρ : ℕ) : 0 ≤ rankinFactor s Pt d q θ ρ :=
  sum_nonneg fun _ _ => mul_nonneg (pow_nonneg (by norm_num) _) (Real.rpow_nonneg (d.passes_nonneg Pt hq _ ρ) θ)

theorem rankinFactorPatterns_nonneg (s : AuditStatement) (Pt : Partition) (d : Design.Two) {R : Type} [Fintype R]
    [DecidableEq R] (req : Fin s.prog.outs.length → R) {q : ℝ} (hq : q ≤ 1) (θ : ℝ) (ρ : ℕ) :
    0 ≤ rankinFactorPatterns s Pt d req q θ ρ :=
  sum_nonneg fun _ _ => mul_nonneg (pow_nonneg (by norm_num) _)
    (Real.iSup_nonneg fun _ => Real.rpow_nonneg (d.passes_nonneg Pt hq _ ρ) θ)

theorem rankinFactorCapped_nonneg (s : AuditStatement) (Pt : Partition) (d : Design.Two) (S : ℕ) {q : ℝ} (hq : q ≤ 1)
    (θ : ℝ) (ρ : ℕ) : 0 ≤ rankinFactorCapped s Pt d S q θ ρ :=
  sum_nonneg fun _ _ => mul_nonneg (pow_nonneg (by norm_num) _) (Real.rpow_nonneg (d.passes_nonneg Pt hq _ ρ) θ)

theorem blocksIn_subset (Pt Pρ : Partition) (ρ : ℕ) : blocksIn Pt Pρ ρ ⊆ range Pt.blocks.length :=
  biUnion_subset.2 fun _ _ => filter_subset _ _

end TwoStageCheckProof

open TwoStageCheckProof

/-- **A sets' point's bits bound the count**, at a point whose form is the sets'. -/
theorem twoStageCheckSound_form (D : Point) (b : ℚ) (hb : checkTwoStage D = some b) (hform : D.form = .sets)
    (s : AuditStatement) (Pt : Partition) (d : Design.Two) (cls : ℕ → ℕ) (hs : InScopeTwo Pt d s)
    (hfirst : FirstStageAt d D.q) (hcov : ClassesCover s Pt d D cls) :
    ((∑ T ∈ admissibleTwo Pt d D.δ, 2 ^ bitsOf s Pt T : ℕ) : ℝ) ≤ (2 : ℝ) ^ (b : ℝ) := by
  obtain ⟨-, hq1, hθ, hδ, -⟩ := valid_of_check hb
  have hq1' : ((D.q : ℚ) : ℝ) ≤ 1 := by exact_mod_cast hq1
  have hR := admissibleTwo_rankin s Pt d (fun _ => (D.q : ℝ)) D.δ D.θ hs (firstStage_le hs hq1' hfirst)
    (fun _ => hq1') (by exact_mod_cast hδ) (by exact_mod_cast hθ)
  refine bits_of_rankin hb cls hR (fun ρ _ => rankinFactor_nonneg s Pt d hq1' _ ρ) (fun ρ hρ c v hc hv => ?_)
    (fun ρ hρ => (hcov.1 ρ hρ).imp fun _ h => h.1) hcov.2
  obtain ⟨c', hc', hcv⟩ := hcov.1 ρ hρ
  rw [hc, Option.some.injEq] at hc'
  subst hc'
  have hv' : _root_.Spot.FactorUpper.lnFactorUpper c.cells D.params = some v := by
    simpa [classFactor, hform] using hv
  exact factorUpperSound c.cells D.params v hv' s Pt d ρ hcv

/-- **A patterns' point's bits bound the patterns' count**, at a point whose form is the patterns'. -/
theorem twoStageCheckPatternsSound_form (D : Point) (b : ℚ) (hb : checkTwoStage D = some b)
    (hform : D.form = .patterns) (s : AuditStatement) (Pt : Partition) (d : Design.Two) (R : Type) [Fintype R]
    [DecidableEq R] (req : Fin s.prog.outs.length → R) (cls : ℕ → ℕ) (hs : InScopeTwo Pt d s)
    (hfirst : FirstStageAt d D.q) (hcov : ClassesCover s Pt d D cls) (hreq : RequestsCover s Pt d req D cls) :
    ((∑ p ∈ (admissibleTwo Pt d D.δ).image (fun T r => lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req T r)),
        2 ^ ∑ r, lightBits s Pt (servedBits req r) (p r) : ℕ) : ℝ) ≤ (2 : ℝ) ^ (b : ℝ) := by
  obtain ⟨-, hq1, hθ, hδ, -⟩ := valid_of_check hb
  have hq1' : ((D.q : ℚ) : ℝ) ≤ 1 := by exact_mod_cast hq1
  have hR := admissibleTwo_rankin_patterns s Pt d (fun _ => (D.q : ℝ)) D.δ D.θ R req hs
    (firstStage_le hs hq1' hfirst) (fun _ => hq1') (by exact_mod_cast hδ) (by exact_mod_cast hθ) hreq.1
  refine bits_of_rankin hb cls hR (fun ρ _ => rankinFactorPatterns_nonneg s Pt d req hq1' _ ρ)
    (fun ρ hρ c v hc hv => ?_) (fun ρ hρ => (hcov.1 ρ hρ).imp fun _ h => h.1) hcov.2
  obtain ⟨c', hc', hcv⟩ := hcov.1 ρ hρ
  rw [hc, Option.some.injEq] at hc'
  subst hc'
  have hv' : _root_.Spot.FactorUpper.lnFactorUpperPatterns c.cells c.reads c.N D.S D.params = some v := by
    simpa [classFactor, hform] using hv
  obtain ⟨hN, hserve⟩ := hreq.2.2 ρ hρ c hc
  exact factorUpperPatternsSound c.cells c.reads c.N D.S D.params v hv' s Pt d R req ρ hcv hN hserve
    fun r r' hne => (hreq.2.1 r r' hne).mono (blocksTo_mono req (blocksIn_subset Pt d.part ρ) r)
      (blocksTo_mono req (blocksIn_subset Pt d.part ρ) r')

/-- **A capped point's bits bound the capped count**, at a point whose form is the capped one, at replay block `ρ`'s
cap `N_c · S` at its class `c`. -/
theorem twoStageCheckServedSound_form (D : Point) (b : ℚ) (hb : checkTwoStage D = some b) (hform : D.form = .capped)
    (s : AuditStatement) (Pt : Partition) (d : Design.Two) (R : Type) [Fintype R] [DecidableEq R]
    (req : Fin s.prog.outs.length → R) (cls : ℕ → ℕ) (hs : InScopeTwo Pt d s) (hfirst : FirstStageAt d D.q)
    (hcov : ClassesCover s Pt d D cls) (hserve : ServedCover s Pt d req D cls) :
    ((∑ T ∈ admissibleTwo Pt d D.δ, 2 ^ min (bitsOf s Pt T)
        (∑ r ∈ univ.filter (fun r => (blocksTo s.prog Pt req T r).Nonempty),
          min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r)) : ℕ) : ℝ) ≤ (2 : ℝ) ^ (b : ℝ) := by
  obtain ⟨-, hq1, hθ, hδ, -⟩ := valid_of_check hb
  have hq1' : ((D.q : ℚ) : ℝ) ≤ 1 := by exact_mod_cast hq1
  obtain ⟨S, hS⟩ : ∃ S : ℕ → ℕ, ∀ ρ, S ρ = if ρ < d.part.blocks.length then
      (D.classes.getD (cls ρ) default).N * D.S
    else ∑ r ∈ univ.filter (fun r => (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r).Nonempty), servedBits req r :=
    ⟨_, fun _ => rfl⟩
  have hcap : ∀ ρ < d.part.blocks.length, ∀ c, D.classes[cls ρ]? = some c → S ρ = c.N * D.S := by
    intro ρ hρ c hc
    simp only [hS, hρ, ↓reduceIte, List.getD_eq_getElem?_getD, hc, Option.getD_some]
  have hle : ∀ ρ, ∑ r ∈ univ.filter (fun r => (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r).Nonempty),
      servedBits req r ≤ S ρ := by
    intro ρ
    by_cases hρ : ρ < d.part.blocks.length
    · obtain ⟨c, hc, -⟩ := hcov.1 ρ hρ
      rw [hcap ρ hρ c hc]
      exact hserve.2 ρ hρ c hc
    · simp only [hS, hρ, ↓reduceIte, le_refl]
  have hR := admissibleTwo_rankin_served s Pt d (fun _ => (D.q : ℝ)) D.δ D.θ R req S hs
    (firstStage_le hs hq1' hfirst) (fun _ => hq1') (by exact_mod_cast hδ) (by exact_mod_cast hθ) hserve.1 hle
  refine bits_of_rankin hb cls hR (fun ρ _ => rankinFactorCapped_nonneg s Pt d _ hq1' _ ρ)
    (fun ρ hρ c v hc hv => ?_) (fun ρ hρ => (hcov.1 ρ hρ).imp fun _ h => h.1) hcov.2
  obtain ⟨c', hc', hcv⟩ := hcov.1 ρ hρ
  rw [hc, Option.some.injEq] at hc'
  subst hc'
  have hv' : _root_.Spot.FactorUpper.lnFactorUpperCapped c.cells (c.N * D.S) D.params = some v := by
    simpa [classFactor, hform] using hv
  rw [hcap ρ hρ c hc]
  exact factorUpperCappedSound c.cells (c.N * D.S) D.params v hv' s Pt d ρ hcv

/-- **A sets' point's bits bound the count.** -/
theorem twoStageCheckSound : TwoStageCheckSound :=
  fun D b hb => twoStageCheckSound_form { D with form := .sets } b hb rfl

/-- **A patterns' point's bits bound the patterns' count.** -/
theorem twoStageCheckPatternsSound : TwoStageCheckPatternsSound :=
  fun D b hb => twoStageCheckPatternsSound_form { D with form := .patterns } b hb rfl

/-- **A capped point's bits bound the capped count.** -/
theorem twoStageCheckServedSound : TwoStageCheckServedSound :=
  fun D b hb => twoStageCheckServedSound_form { D with form := .capped } b hb rfl

section Witnesses

open Claims.Witness.SamplingGeneric
open Claims.Witness.Sampling (toyStmt toyPart)
open FactorUpperWitness (pairStmt pairPart pairDesign emptyStmt)
open FactorUpperProof (pair_covers pair_readsCover)
open TwoStageCheckWitness

namespace TwoStageCheckProof

theorem pairStmt_inScope : InScopeTwo pairPart pairDesign pairStmt := inScopeTwo_rankin

theorem pair_firstStage (form : Form) : FirstStageAt pairDesign ((pairPoint form).q : ℝ) :=
  Or.inl ⟨1, 2, rfl, by norm_num [pairPoint]⟩

theorem pair_classesCover (form : Form) : ClassesCover pairStmt pairPart pairDesign (pairPoint form) fun _ => 0 := by
  refine ⟨fun ρ hρ => ⟨_, rfl, ?_⟩, fun i c hc => ?_⟩
  · obtain rfl : ρ = 0 := by
      have : pairDesign.part.blocks.length = 1 := rfl
      omega
    exact pair_covers
  · cases i with
    | zero =>
      simp only [pairPoint, List.getElem?_cons_zero, Option.some.injEq] at hc
      subst hc
      decide
    | succ i => simp [pairPoint] at hc

theorem pair_requestsCover (form : Form) :
    RequestsCover pairStmt pairPart pairDesign (fun _ => ()) (pairPoint form) fun _ => 0 := by
  refine ⟨fun r => ⟨0, fun v hv => ?_⟩, fun r r' h => absurd (Subsingleton.elim r r') h, fun ρ hρ c hc => ?_⟩
  · have hv2 : v < 2 := mem_range.1 (blocksTo_subset _ _ _ _ _ hv)
    interval_cases v <;> decide
  · obtain rfl : ρ = 0 := by
      have : pairDesign.part.blocks.length = 1 := rfl
      omega
    simp only [pairPoint, List.getElem?_cons_zero, Option.some.injEq] at hc
    subst hc
    exact ⟨(card_le_univ _).trans (by simp),
      fun ⟨⟩ _ => ⟨(by decide : servedBits (fun _ : Fin pairStmt.prog.outs.length => ()) () = 2), pair_readsCover⟩⟩

theorem pair_servedCover (form : Form) :
    ServedCover pairStmt pairPart pairDesign (fun _ => ()) (pairPoint form) fun _ => 0 := by
  refine ⟨fun r r' h => absurd (Subsingleton.elim r r') h, fun ρ hρ c hc => ?_⟩
  obtain rfl : ρ = 0 := by
    have : pairDesign.part.blocks.length = 1 := rfl
    omega
  simp only [pairPoint, List.getElem?_cons_zero, Option.some.injEq] at hc
  subst hc
  refine (sum_le_sum_of_subset (filter_subset _ _)).trans ?_
  rw [Fintype.sum_unique]
  exact (by decide : servedBits (fun _ : Fin pairStmt.prog.outs.length => ()) () ≤ 1 * 2)

/-- `nothingDesign` over `emptyStmt` and `soloPart` is `nothingTwo` over `toyStmt` and `toyPart`. -/
theorem nothing_count :
    (∑ T ∈ admissibleTwo soloPart nothingDesign (9 / 10), 2 ^ bitsOf emptyStmt soloPart T : ℕ) = 3 :=
  (rfl : (∑ T ∈ admissibleTwo soloPart nothingDesign (9 / 10), 2 ^ bitsOf emptyStmt soloPart T : ℕ) =
    ∑ T ∈ admissibleTwo toyPart nothingTwo (9 / 10), 2 ^ bitsOf toyStmt toyPart T).trans nothingTwo_count

theorem nothing_patterns_count :
    (∑ p ∈ (admissibleTwo soloPart nothingDesign (9 / 10)).image
        (fun T r => lightOf emptyStmt soloPart (servedBits oneReq r) (blocksTo emptyStmt.prog soloPart oneReq T r)),
        2 ^ ∑ r, lightBits emptyStmt soloPart (servedBits oneReq r) (p r) : ℕ) = 3 := by
  have himg : ∀ S : Finset (Finset ℕ), S = {∅, {0}} →
      S.image (fun T r => lightOf toyStmt toyPart (servedBits toyReq r) (blocksTo toyStmt.prog toyPart toyReq T r)) =
        {fun _ => none, fun _ => some none} := by
    rintro S rfl
    rw [image_insert, image_singleton, toy_pat_empty, toy_pat_zero]
  change (∑ p ∈ (admissibleTwo toyPart nothingTwo (9 / 10)).image
      (fun T r => lightOf toyStmt toyPart (servedBits toyReq r) (blocksTo toyStmt.prog toyPart toyReq T r)),
      2 ^ ∑ r, lightBits toyStmt toyPart (servedBits toyReq r) (p r) : ℕ) = 3
  rw [himg _ (by rw [nothingTwo_admissible]; decide), sum_pair toy_pat_ne]
  simp [lightBits, toy_servedBits]

theorem nothing_served_count :
    (∑ T ∈ admissibleTwo soloPart nothingDesign (9 / 10), 2 ^ min (bitsOf emptyStmt soloPart T)
      (∑ r ∈ univ.filter (fun r => (blocksTo emptyStmt.prog soloPart oneReq T r).Nonempty),
        min (bitsOf emptyStmt soloPart (blocksTo emptyStmt.prog soloPart oneReq T r)) (servedBits oneReq r)) : ℕ) =
      3 :=
  (rfl : (∑ T ∈ admissibleTwo soloPart nothingDesign (9 / 10), 2 ^ min (bitsOf emptyStmt soloPart T)
      (∑ r ∈ univ.filter (fun r => (blocksTo emptyStmt.prog soloPart oneReq T r).Nonempty),
        min (bitsOf emptyStmt soloPart (blocksTo emptyStmt.prog soloPart oneReq T r)) (servedBits oneReq r)) : ℕ) =
    ∑ T ∈ admissibleTwo toyPart nothingTwo (9 / 10), 2 ^ min (bitsOf toyStmt toyPart T)
      (∑ r ∈ univ.filter (fun r => (blocksTo toyStmt.prog toyPart toyReq T r).Nonempty),
        min (bitsOf toyStmt toyPart (blocksTo toyStmt.prog toyPart toyReq T r)) (servedBits toyReq r))).trans
    nothingTwo_countServed

end TwoStageCheckProof

/-- **`TwoStageCheckSound`'s honest witness**: `pairDesign` is in scope, its first stage samples at `1/2`, one class
covers its replay block, and the point returns there and at a million deployed-shaped replay blocks. -/
theorem twoStageCheckSound_honest :
    InScopeTwo pairPart pairDesign pairStmt ∧ FirstStageAt pairDesign ((pairPoint .sets).q : ℝ) ∧
      ClassesCover pairStmt pairPart pairDesign (pairPoint .sets) (fun _ => 0) ∧
      (checkTwoStage (pairPoint .sets)).isSome = true ∧ (checkTwoStage (deployedPoint .sets)).isSome = true :=
  ⟨pairStmt_inScope, pair_firstStage _, pair_classesCover _, by decide +kernel, by decide +kernel⟩

/-- **`TwoStageCheckSound`'s conclusion fails** at a first stage that samples nothing (`nothingDesign`), at `1` bit:
the count is `3`. -/
theorem twoStageCheckSound_nontrivial :
    ¬ ((∑ T ∈ admissibleTwo soloPart nothingDesign (9 / 10), 2 ^ bitsOf emptyStmt soloPart T : ℕ) : ℝ) ≤
      (2 : ℝ) ^ (((1 : ℚ) : ℝ)) := by
  rw [nothing_count]
  norm_num

/-- **`TwoStageCheckPatternsSound`'s honest witness**: `twoStageCheckSound_honest`'s, and the one request reading both
of `pairStmt`'s outputs reads in its one replay block, serves `2` bits and reads at most the class's `reads`. -/
theorem twoStageCheckPatternsSound_honest :
    InScopeTwo pairPart pairDesign pairStmt ∧ FirstStageAt pairDesign ((pairPoint .patterns).q : ℝ) ∧
      ClassesCover pairStmt pairPart pairDesign (pairPoint .patterns) (fun _ => 0) ∧
      RequestsCover pairStmt pairPart pairDesign (fun _ => ()) (pairPoint .patterns) (fun _ => 0) ∧
      (checkTwoStage (pairPoint .patterns)).isSome = true ∧
      (checkTwoStage (deployedPoint .patterns)).isSome = true :=
  ⟨pairStmt_inScope, pair_firstStage _, pair_classesCover _, pair_requestsCover _, by decide +kernel,
    by decide +kernel⟩

/-- **`TwoStageCheckPatternsSound`'s conclusion fails** at `nothingDesign`, with one request reading the one output,
at `1` bit: the patterns' count is `3`. -/
theorem twoStageCheckPatternsSound_nontrivial :
    ¬ ((∑ p ∈ (admissibleTwo soloPart nothingDesign (9 / 10)).image
          (fun T r => lightOf emptyStmt soloPart (servedBits oneReq r) (blocksTo emptyStmt.prog soloPart oneReq T r)),
          2 ^ ∑ r, lightBits emptyStmt soloPart (servedBits oneReq r) (p r) : ℕ) : ℝ) ≤
      (2 : ℝ) ^ (((1 : ℚ) : ℝ)) := by
  rw [nothing_patterns_count]
  norm_num

/-- **`TwoStageCheckServedSound`'s honest witness**: `twoStageCheckSound_honest`'s, and the one request reading both
of `pairStmt`'s outputs serves `2` bits, the class's one request times the point's `S`. -/
theorem twoStageCheckServedSound_honest :
    InScopeTwo pairPart pairDesign pairStmt ∧ FirstStageAt pairDesign ((pairPoint .capped).q : ℝ) ∧
      ClassesCover pairStmt pairPart pairDesign (pairPoint .capped) (fun _ => 0) ∧
      ServedCover pairStmt pairPart pairDesign (fun _ => ()) (pairPoint .capped) (fun _ => 0) ∧
      (checkTwoStage (pairPoint .capped)).isSome = true ∧ (checkTwoStage (deployedPoint .capped)).isSome = true :=
  ⟨pairStmt_inScope, pair_firstStage _, pair_classesCover _, pair_servedCover _, by decide +kernel,
    by decide +kernel⟩

/-- **`TwoStageCheckServedSound`'s conclusion fails** at `nothingDesign`, with one request reading the one output,
at `1` bit: the capped count is `3`. -/
theorem twoStageCheckServedSound_nontrivial :
    ¬ ((∑ T ∈ admissibleTwo soloPart nothingDesign (9 / 10), 2 ^ min (bitsOf emptyStmt soloPart T)
        (∑ r ∈ univ.filter (fun r => (blocksTo emptyStmt.prog soloPart oneReq T r).Nonempty),
          min (bitsOf emptyStmt soloPart (blocksTo emptyStmt.prog soloPart oneReq T r)) (servedBits oneReq r)) : ℕ) :
        ℝ) ≤ (2 : ℝ) ^ (((1 : ℚ) : ℝ)) := by
  rw [nothing_served_count]
  norm_num

end Witnesses

end Properties.Spot.SecurityProofs
