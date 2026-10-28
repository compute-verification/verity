import Definitions.NetworkCertifier.Guarantees.Horizon
import Proofs.NetworkCertifier.PeriodSum
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# The wrong units over a horizon: the escapes multiply

A uniform `k`-subset of `n` units misses a set of `f` units with probability `C(n - f, k) / C(n, k)`, at most
`(1 - k/n)^f` (`choose_sub_le`, `pr_disjoint_le`). Over windows on independent coins, each with its wrong
set fixed before its own draw, every draw missing its window's set while the sets hold `F` units in total has
probability at most `r^F` when each window's miss has probability at most `r` to the power of its set's size
(`pr_adaptive_le`, by induction over the windows: the first window's coins, then the rest's at each of them). With
each window's soundness error added by a union bound, that is `horizonWrongUnits`.
-/

namespace NetTiming

open Claims Finset

/-! ### `Pr` over products and equivalences -/

theorem Pr_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (p : β → Prop) :
    Pr (fun a => p (e a)) = Pr p := by
  classical
  unfold Pr
  rw [Fintype.card_congr e]
  congr 2
  rw [← Fintype.card_subtype, ← Fintype.card_subtype]
  exact_mod_cast Fintype.card_congr (e.subtypeEquiv fun _ => Iff.rfl)

theorem Pr_prod_eq_Ex {α β : Type} [Fintype α] [Fintype β] (p : α × β → Prop) :
    Pr p = Ex fun a => Pr fun b => p (a, b) := by
  classical
  unfold Ex
  rw [Pr_eq_sum p, Fintype.sum_prod_type, Fintype.card_prod, Finset.sum_div]
  simp only [Pr_eq_sum (fun b : β => p (_, b))]
  push_cast
  rw [Finset.sum_div]
  refine Finset.sum_congr rfl fun a _ => ?_
  ring

theorem Ex_ite_const {α : Type} [Fintype α] (p : α → Prop) [DecidablePred p] (c : ℝ) :
    (Ex fun a => if p a then c else 0) = Pr p * c := by
  unfold Ex
  rw [Pr_eq_sum p, div_mul_eq_mul_div, Finset.sum_mul]
  congr 1
  refine Finset.sum_congr rfl fun a _ => ?_
  by_cases h : p a <;> simp [h]

theorem Pr_or_le {α : Type} [Fintype α] (p q : α → Prop) : Pr (fun a => p a ∨ q a) ≤ Pr p + Pr q := by
  classical
  unfold Pr
  rw [← add_div]
  refine div_le_div_of_nonneg_right ?_ (by positivity)
  exact_mod_cast (Finset.card_le_card fun a ha => by
    simp only [Finset.mem_filter, Finset.mem_union, Finset.mem_univ, true_and] at ha ⊢; exact ha).trans
    (Finset.card_union_le _ _)

/-! ### A uniform draw's miss -/

theorem choose_pred_le (m k n : ℕ) (hm : m ≤ n) (hk : k ≤ n) :
    ((m - 1).choose k : ℝ) ≤ m.choose k * (1 - k / n) := by
  rcases m with _ | m
  · rcases k with _ | k
    · simp
    · simp
  have hn : (0 : ℝ) < n := by exact_mod_cast (show 0 < n by omega)
  have h := Nat.choose_mul_succ_eq m k
  have hpos : (0 : ℝ) < m + 1 := by positivity
  have hkn : (k : ℝ) / n ≤ 1 := (div_le_one hn).2 (by exact_mod_cast hk)
  have hsub : (((m + 1 - k : ℕ) : ℝ)) ≤ (m + 1) * (1 - k / n) := by
    rcases le_or_gt k (m + 1) with hkm | hkm
    · rw [Nat.cast_sub hkm]
      push_cast
      have h2 : (m + 1 : ℝ) * (1 - k / n) = (m + 1 - k) + k * (1 - (m + 1) / n) := by ring
      have h3 : ((m : ℝ) + 1) / n ≤ 1 := (div_le_one hn).2 (by exact_mod_cast hm)
      have h4 : 0 ≤ (k : ℝ) * (1 - (m + 1) / n) := mul_nonneg (by positivity) (by linarith)
      linarith
    · rw [Nat.sub_eq_zero_of_le hkm.le]
      push_cast
      exact mul_nonneg hpos.le (by linarith)
  have h' : ((m.choose k : ℕ) : ℝ) * (m + 1) = ((m + 1).choose k : ℕ) * ((m + 1 - k : ℕ) : ℝ) := by
    exact_mod_cast h
  simp only [Nat.add_sub_cancel]
  have hc : (0 : ℝ) ≤ ((m + 1).choose k : ℕ) := by positivity
  have := mul_le_mul_of_nonneg_left hsub hc
  nlinarith [h', this, hpos]

theorem choose_sub_le (n k : ℕ) (hk : k ≤ n) (f : ℕ) :
    ((n - f).choose k : ℝ) ≤ n.choose k * (1 - k / n) ^ f := by
  induction f with
  | zero => simp
  | succ f ih =>
    have h0 : 0 ≤ 1 - (k : ℝ) / n := by
      rcases Nat.eq_zero_or_pos n with hn | hn
      · subst hn; simp
      · have : (k : ℝ) / n ≤ 1 := (div_le_one (by exact_mod_cast hn)).2 (by exact_mod_cast hk)
        linarith
    have e : n - (f + 1) = n - f - 1 := by omega
    rw [e, pow_succ, ← mul_assoc]
    exact (choose_pred_le (n - f) k n (Nat.sub_le n f) hk).trans (mul_le_mul_of_nonneg_right ih h0)

theorem pr_disjoint_le {Ω : Type} [Fintype Ω] [Nonempty Ω] {n : ℕ} (draw : Ω → Finset (Fin n)) (k : ℕ)
    (hcard : ∀ x, (draw x).card = k)
    (hfib : ∀ S : Finset (Fin n), S.card = k → (univ.filter fun x => draw x = S).card * n.choose k = Fintype.card Ω)
    (W : Finset (Fin n)) {q : ℝ} (hq : q * n ≤ k) :
    Pr (fun x => Disjoint (draw x) W) ≤ (1 - q) ^ W.card := by
  classical
  have hΩ : (0 : ℝ) < Fintype.card Ω := by exact_mod_cast Fintype.card_pos
  have hk : k ≤ n := by
    have := Finset.card_le_univ (draw (Classical.arbitrary Ω))
    rwa [hcard, Fintype.card_fin] at this
  have hC : (0 : ℝ) < n.choose k := by exact_mod_cast Nat.choose_pos hk
  have hsum : (univ.filter fun x => Disjoint (draw x) W).card =
      ∑ S ∈ powersetCard k (univ \ W), (univ.filter fun x => draw x = S).card := by
    rw [Finset.card_eq_sum_card_fiberwise (f := draw) (t := powersetCard k (univ \ W))]
    · refine Finset.sum_congr rfl fun S hS => ?_
      rw [Finset.filter_filter]
      congr 1
      refine Finset.filter_congr fun x _ => ⟨fun h => h.2, fun h => ⟨?_, h⟩⟩
      rw [h]
      obtain ⟨hsub, -⟩ := Finset.mem_powersetCard.1 hS
      exact Finset.disjoint_left.2 fun u hu huW => (Finset.mem_sdiff.1 (hsub hu)).2 huW
    · intro x hx
      rw [Finset.mem_coe, Finset.mem_filter] at hx
      rw [Finset.mem_coe, Finset.mem_powersetCard]
      exact ⟨fun u hu => Finset.mem_sdiff.2 ⟨Finset.mem_univ u, Finset.disjoint_left.1 hx.2 hu⟩, hcard x⟩
  have hnat : (univ.filter fun x => Disjoint (draw x) W).card * n.choose k =
      (n - W.card).choose k * Fintype.card Ω := by
    rw [hsum, Finset.sum_mul, Finset.sum_congr rfl fun S hS => hfib S (Finset.mem_powersetCard.1 hS).2,
      Finset.sum_const, Finset.card_powersetCard, Finset.card_sdiff_of_subset (Finset.subset_univ W), Finset.card_univ,
      Fintype.card_fin, smul_eq_mul]
  have hpr : Pr (fun x => Disjoint (draw x) W) = ((n - W.card).choose k : ℝ) / n.choose k := by
    rw [Pr_eq_sum, Finset.sum_boole, div_eq_div_iff hΩ.ne' hC.ne']
    exact_mod_cast hnat
  rw [hpr, div_le_iff₀ hC]
  refine (choose_sub_le n k hk W.card).trans ?_
  rw [mul_comm]
  refine mul_le_mul_of_nonneg_right ?_ hC.le
  rcases Nat.eq_zero_or_pos n with hn | hn
  · subst hn
    have : W.card = 0 := by simpa using Finset.card_le_univ W
    rw [this, pow_zero, pow_zero]
  · have hn' : (0 : ℝ) < n := by exact_mod_cast hn
    have hkn : (k : ℝ) / n ≤ 1 := (div_le_one hn').2 (by exact_mod_cast hk)
    have hqk : q ≤ k / n := (le_div_iff₀ hn').2 hq
    exact pow_le_pow_left₀ (by linarith) (by linarith) _

/-- The draw's hypotheses are satisfiable: the coins that are the `k`-subsets themselves. -/
theorem sampleUniform_val (n k : ℕ) :
    (∀ x : {S : Finset (Fin n) // S.card = k}, x.val.card = k) ∧
      ∀ S : Finset (Fin n), S.card = k →
        (univ.filter fun x : {S : Finset (Fin n) // S.card = k} => x.val = S).card * n.choose k =
          Fintype.card {S : Finset (Fin n) // S.card = k} := by
  classical
  refine ⟨fun S => S.2, fun S hS => ?_⟩
  have h1 : (univ.filter fun x : {S : Finset (Fin n) // S.card = k} => x.val = S).card = 1 := by
    rw [Finset.card_eq_one]
    exact ⟨⟨S, hS⟩, by ext x; simp [Subtype.ext_iff]⟩
  rw [h1, one_mul, Fintype.card_finset_len, Fintype.card_fin]

/-! ### The escapes multiply over windows on independent coins -/

theorem pr_adaptive_le {Ω : Type} [Fintype Ω] [Nonempty Ω] {r : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1) :
    ∀ (m : ℕ) (A : Fin m → (Fin m → Ω) → Ω → Prop) (f : Fin m → (Fin m → Ω) → ℕ),
      (∀ i ω ω', (∀ j, j < i → ω j = ω' j) → A i ω = A i ω') →
      (∀ i ω ω', (∀ j, j < i → ω j = ω' j) → f i ω = f i ω') →
      (∀ i ω, Pr (A i ω) ≤ r ^ f i ω) → ∀ F : ℕ,
      Pr (fun ω => (∀ i, A i ω (ω i)) ∧ F ≤ ∑ i, f i ω) ≤ r ^ F := by
  intro m
  induction m with
  | zero =>
    intro A f _ _ _ F
    rcases Nat.eq_zero_or_pos F with rfl | hF
    · simpa using Pr_le_one (fun ω : Fin 0 → Ω => (∀ i, A i ω (ω i)) ∧ 0 ≤ ∑ i, f i ω)
    · calc _ ≤ Pr (fun _ : Fin 0 → Ω => False) := Pr_mono fun ω h => by simp at h; omega
        _ = 0 := Pr_false
        _ ≤ r ^ F := pow_nonneg hr0 F
  | succ m ih =>
    intro A f hA hf hesc F
    classical
    let e : Ω × (Fin m → Ω) ≃ (Fin (m + 1) → Ω) :=
      { toFun := fun p => (Fin.cons p.1 p.2 : Fin (m + 1) → Ω), invFun := fun ω => (ω 0, Fin.tail ω),
        left_inv := fun p => by simp, right_inv := fun ω => by simp }
    let ω₀ : Fin (m + 1) → Ω := fun _ => Classical.arbitrary Ω
    have hA0 : ∀ ω, A 0 ω = A 0 ω₀ := fun ω => hA 0 ω ω₀ fun j hj => absurd hj (Fin.not_lt_zero j)
    have hf0 : ∀ ω, f 0 ω = f 0 ω₀ := fun ω => hf 0 ω ω₀ fun j hj => absurd hj (Fin.not_lt_zero j)
    have hcons : ∀ (x : Ω) (ω₁ ω₂ : Fin m → Ω) (i : Fin m), (∀ j, j < i → ω₁ j = ω₂ j) →
        ∀ j, j < i.succ → (Fin.cons x ω₁ : Fin (m + 1) → Ω) j = (Fin.cons x ω₂ : Fin (m + 1) → Ω) j := by
      intro x ω₁ ω₂ i h j hj
      cases j using Fin.cases with
      | zero => simp
      | succ j => simpa only [Fin.cons_succ] using h j (Fin.succ_lt_succ_iff.1 hj)
    have hx : ∀ x, (Pr fun ω' => (fun ω => (∀ i, A i ω (ω i)) ∧ F ≤ ∑ i, f i ω) (e (x, ω'))) ≤
        if A 0 ω₀ x then r ^ (F - f 0 ω₀) else 0 := by
      intro x
      split_ifs with h0
      · refine (Pr_mono ?_).trans (ih (fun i ω' => A i.succ (Fin.cons x ω')) (fun i ω' => f i.succ (Fin.cons x ω'))
          (fun i ω₁ ω₂ h => hA i.succ _ _ (hcons x ω₁ ω₂ i h)) (fun i ω₁ ω₂ h => hf i.succ _ _ (hcons x ω₁ ω₂ i h))
          (fun i ω' => hesc i.succ _) (F - f 0 ω₀))
        rintro ω' ⟨hall, hF⟩
        refine ⟨fun i => by simpa only [e, Equiv.coe_fn_mk, Fin.cons_succ] using hall i.succ, ?_⟩
        have := hF
        simp only [e, Equiv.coe_fn_mk, Fin.sum_univ_succ, hf0] at this
        omega
      · calc _ ≤ Pr (fun _ : Fin m → Ω => False) := Pr_mono fun ω' h => by
              obtain ⟨hall, -⟩ := h
              have := hall 0
              simp only [e, Equiv.coe_fn_mk, Fin.cons_zero] at this
              rw [hA0] at this
              exact h0 this
          _ = 0 := Pr_false
    rw [← Pr_equiv e, Pr_prod_eq_Ex]
    calc _ ≤ Ex fun x => if A 0 ω₀ x then r ^ (F - f 0 ω₀) else 0 := Ex_mono hx
      _ = Pr (A 0 ω₀) * r ^ (F - f 0 ω₀) := Ex_ite_const _ _
      _ ≤ r ^ f 0 ω₀ * r ^ (F - f 0 ω₀) := mul_le_mul_of_nonneg_right (hesc 0 ω₀) (pow_nonneg hr0 _)
      _ = r ^ (f 0 ω₀ + (F - f 0 ω₀)) := (pow_add _ _ _).symm
      _ ≤ r ^ F := pow_le_pow_of_le_one hr0 hr1 (by omega)

/-! ### The horizon -/

/-- **The wrong units over a horizon**: `Guarantees.HorizonWrongUnitsLe`. -/
theorem horizonWrongUnits : Guarantees.HorizonWrongUnitsLe := by
  intro m Ω _ _ n k draw hcard hunif W hW acc ε hsound q hq0 hq1 hq F
  refine ⟨?_, fun δ hδ hlog => ?_⟩
  · classical
    have hmiss := pr_adaptive_le (Ω := Ω) (r := 1 - q) (by linarith) (by linarith) m
      (fun i ω x => Disjoint (draw i x) (W i ω)) (fun i ω => (W i ω).card)
      (fun i ω ω' h => by simp only [hW i ω ω' h]) (fun i ω ω' h => by simp only [hW i ω ω' h])
      (fun i ω => pr_disjoint_le (draw i) (k i) (hcard i) (hunif i) (W i ω) (hq i)) F
    calc Pr (fun ω => (∀ i, acc i ω) ∧ F ≤ ∑ i, (W i ω).card)
        ≤ Pr (fun ω => ((∀ i, Disjoint (draw i (ω i)) (W i ω)) ∧ F ≤ ∑ i, (W i ω).card) ∨
            ∃ i ∈ (univ : Finset (Fin m)), acc i ω ∧ ¬Disjoint (draw i (ω i)) (W i ω)) := by
          refine Pr_mono fun ω h => ?_
          obtain ⟨hacc, hF⟩ := h
          by_cases hd : ∀ i, Disjoint (draw i (ω i)) (W i ω)
          · exact Or.inl ⟨hd, hF⟩
          · push Not at hd
            obtain ⟨i, hi⟩ := hd
            exact Or.inr ⟨i, Finset.mem_univ i, hacc i, hi⟩
      _ ≤ _ := Pr_or_le _ _
      _ ≤ (1 - q) ^ F + ∑ i, ε i :=
          add_le_add hmiss ((Pr_exists_mem_le_sum univ _).trans (Finset.sum_le_sum fun i _ => hsound i))
  · have h1 : 1 - q ≤ Real.exp (-q) := by linarith [Real.add_one_le_exp (-q)]
    calc (1 - q) ^ F ≤ Real.exp (-q) ^ F := pow_le_pow_left₀ (by linarith) h1 F
      _ = Real.exp (F * (-q)) := (Real.exp_nat_mul _ _).symm
      _ ≤ Real.exp (-Real.log (1 / δ)) := Real.exp_le_exp.2 (by nlinarith)
      _ = δ := by rw [one_div, Real.log_inv, neg_neg, Real.exp_log hδ]

end NetTiming
