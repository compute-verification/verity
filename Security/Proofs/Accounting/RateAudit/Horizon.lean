import Mathlib
import Models.Accounting.Guarantees.RateAuditHorizon
import Proofs.Accounting.RateAudit.Chain

/-!
# The causal horizon bound: the proof

Rounds `0, …, n − 1` each draw a sample of `s_i` items, uniform over `M`, after a set of good items `G_i` is fixed.
`G_i` may be any function of the earlier rounds' coins, their samples among them, and of round `i`'s own other coins,
but not of round `i`'s sample (`Rate.Causal`). Then every sample lands in its round's good set while the good sets,
weighted by the sample counts, hold at most `K` items in all, with probability at most `(K / (M·∑ s_i))^(∑ s_i)`
(`Horizon.pr_causal_le`). The proof peels round 0: its good set is fixed by its own coins, so its sample lands in it
with probability `(g₀/M)^s₀` (`pr_sample`); given round 0's coins, the later rounds are a causal family with `K − s₀·g₀`
left (`pr_and_le_mul`), and weighted AM–GM joins the two (`step_le`).

A chain's passes (`Rate.Chain`) are such rounds, with `M = N`: pass `j` is accepted exactly when its sample lands in its
correct receipts, which the store, the earlier passes' coins and its own stream and oracle family fix
(`Chain.pr_horizon_le`). With `RateChainPaidRO`'s conservation at every count, the window's reads pay for the correct
receipts the sample certifies (`RateChainSampledRO`).
-/

namespace Accounting.Rate

open Pous (Bits pr)
open scoped ENNReal NNReal

namespace Horizon

/-! ## Weighted AM–GM -/

/-- Weighted AM–GM at natural weights: `u^a·v^c ≤ m^(a + c)` when `a·u + c·v ≤ (a + c)·m`. -/
theorem pow_mul_pow_le (a c : ℕ) (u v m : ℝ≥0) (h : (a : ℝ≥0) * u + c * v ≤ (a + c) * m) :
    u ^ a * v ^ c ≤ m ^ (a + c) := by
  rcases Nat.eq_zero_or_pos (a + c) with h0 | hpos
  · obtain ⟨rfl, rfl⟩ : a = 0 ∧ c = 0 := by omega
    simp
  have hn : ((a : ℝ≥0) + c) ≠ 0 := by
    have : (0 : ℝ≥0) < (a : ℝ≥0) + c := by exact_mod_cast hpos
    exact this.ne'
  set w₁ : ℝ≥0 := a / ((a : ℝ≥0) + c) with hw₁
  set w₂ : ℝ≥0 := c / ((a : ℝ≥0) + c) with hw₂
  have hw : w₁ + w₂ = 1 := by
    rw [hw₁, hw₂, ← add_div, div_self hn]
  have hg := NNReal.geom_mean_le_arith_mean2_weighted w₁ w₂ u v hw
  have hm : w₁ * u + w₂ * v ≤ m := by
    rw [hw₁, hw₂, div_mul_eq_mul_div, div_mul_eq_mul_div, ← add_div, div_le_iff₀ (pos_iff_ne_zero.2 hn), mul_comm m]
    exact h
  have key : (u ^ (w₁ : ℝ) * v ^ (w₂ : ℝ)) ^ (a + c) ≤ m ^ (a + c) := pow_le_pow_left₀ (by positivity) (hg.trans hm) _
  have hr : ∀ (x w : ℝ≥0) (k : ℕ), w * ((a : ℝ≥0) + c) = k → (x ^ (w : ℝ)) ^ (a + c) = x ^ k := by
    intro x w k hk
    rw [← NNReal.rpow_natCast, ← NNReal.rpow_mul, ← NNReal.rpow_natCast]
    congr 1
    have := congrArg (fun y : ℝ≥0 => (y : ℝ)) hk
    push_cast at this ⊢
    exact this
  have e₁ : (u ^ (w₁ : ℝ)) ^ (a + c) = u ^ a := hr u w₁ a (by rw [hw₁, div_mul_cancel₀ _ hn])
  have e₂ : (v ^ (w₂ : ℝ)) ^ (a + c) = v ^ c := hr v w₂ c (by rw [hw₂, div_mul_cancel₀ _ hn])
  rwa [mul_pow, e₁, e₂] at key

/-- The induction step's arithmetic: a round's `(g/M)^a` and the rest's `(K'/(M·c))^c` multiply to at most the whole's
`((a·g + K')/(M·(a + c)))^(a + c)`. -/
theorem step_le (M g K' a c : ℕ) (hM : 0 < M) :
    ((g : ℝ≥0∞) / M) ^ a * ((K' : ℝ≥0∞) / (M * c)) ^ c ≤
      (((a * g + K' : ℕ) : ℝ≥0∞) / (M * (a + c : ℕ))) ^ (a + c) := by
  rcases Nat.eq_zero_or_pos c with rfl | hc
  · rcases Nat.eq_zero_or_pos a with rfl | ha
    · simp
    simp only [pow_zero, mul_one, add_zero]
    refine pow_le_pow_left₀ (by positivity) ?_ _
    have hMa : ((a : ℝ≥0∞)) ≠ 0 := by exact_mod_cast ha.ne'
    calc (g : ℝ≥0∞) / M = (a * g : ℝ≥0∞) / (a * M) :=
          (ENNReal.mul_div_mul_left _ _ hMa (ENNReal.natCast_ne_top a)).symm
      _ ≤ ((a * g + K' : ℕ) : ℝ≥0∞) / (M * a) := by
          rw [mul_comm (a : ℝ≥0∞) M]
          refine ENNReal.div_le_div_right ?_ _
          push_cast
          exact le_self_add
  have hM' : (M : ℝ≥0) ≠ 0 := by exact_mod_cast hM.ne'
  have hMc : (M : ℝ≥0) * c ≠ 0 := mul_ne_zero hM' (by exact_mod_cast hc.ne')
  have hMac : (M : ℝ≥0) * (a + c : ℕ) ≠ 0 := mul_ne_zero hM' (by exact_mod_cast (by omega : a + c ≠ 0))
  have cast_div : ∀ (x : ℕ) (y : ℝ≥0) (hy : y ≠ 0), (x : ℝ≥0∞) / (y : ℝ≥0∞) = (((x : ℝ≥0) / y : ℝ≥0) : ℝ≥0∞) :=
    fun x y hy => by rw [ENNReal.coe_div hy]; simp
  have e₁ : ((g : ℝ≥0∞) / M) = (((g : ℝ≥0) / M : ℝ≥0) : ℝ≥0∞) := by
    rw [← cast_div g M hM']; simp
  have e₂ : ((K' : ℝ≥0∞) / (M * c)) = (((K' : ℝ≥0) / (M * c) : ℝ≥0) : ℝ≥0∞) := by
    rw [← cast_div K' _ hMc]; simp
  have e₃ : (((a * g + K' : ℕ) : ℝ≥0∞) / (M * (a + c : ℕ))) =
      ((((a * g + K' : ℕ) : ℝ≥0) / (M * (a + c : ℕ)) : ℝ≥0) : ℝ≥0∞) := by
    rw [← cast_div _ _ hMac]; simp
  rw [e₁, e₂, e₃, ← ENNReal.coe_pow, ← ENNReal.coe_pow, ← ENNReal.coe_mul, ← ENNReal.coe_pow, ENNReal.coe_le_coe]
  refine pow_mul_pow_le a c _ _ _ (le_of_eq ?_)
  rw [← NNReal.coe_inj]
  have hM'' : (M : ℝ) ≠ 0 := by exact_mod_cast hM.ne'
  have hc' : (c : ℝ) ≠ 0 := by exact_mod_cast hc.ne'
  have hac : (a : ℝ) + c ≠ 0 := by positivity
  push_cast
  field_simp

/-! ## Probability -/

theorem pr_le_one {Ω : Type} [Fintype Ω] (P : Ω → Prop) : pr P ≤ 1 := by
  classical
  unfold pr
  exact ENNReal.div_le_of_le_mul (by rw [one_mul]; exact_mod_cast Finset.card_le_univ _)

theorem pr_false {Ω : Type} [Fintype Ω] : pr (fun _ : Ω => False) = 0 := by
  classical
  simp [pr]

/-- `s` uniform items all land in a set of `g` of the `M`: `(g/M)^s`. -/
theorem pr_sample {M : ℕ} (G : Finset (Fin M)) (s : ℕ) :
    pr (fun I : Fin s → Fin M => ∀ t, I t ∈ G) = ((G.card : ℝ≥0∞) / M) ^ s := by
  classical
  unfold pr
  rw [Finset.filter_congr_decidable, Bandwidth.AuditRO.card_sample G s]
  simp [div_eq_mul_inv, mul_pow, ENNReal.inv_pow]

/-- A first event of the first coordinate, then a second one bounded by `ε` at every first coordinate where the first
holds: at most the first's probability times `ε`. -/
theorem pr_and_le_mul {X Y : Type} [Fintype X] [Fintype Y] (A : X → Prop) (B : X × Y → Prop) (ε : ℝ≥0∞)
    (h : ∀ x, A x → pr (fun y => B (x, y)) ≤ ε) : pr (fun p : X × Y => A p.1 ∧ B p) ≤ pr A * ε := by
  classical
  unfold pr at h ⊢
  rw [Finset.filter_congr_decidable, Finset.filter_congr_decidable]
  rcases Nat.eq_zero_or_pos (Fintype.card Y) with hY | hY
  · have h0 : (Finset.univ.filter (fun p : X × Y => A p.1 ∧ B p)).card = 0 :=
      Nat.eq_zero_of_le_zero ((Finset.card_le_univ _).trans (by simp [Fintype.card_prod, hY]))
    rw [h0]
    simp
  have hy : ∀ x, A x → ((Finset.univ.filter (fun y => B (x, y))).card : ℝ≥0∞) ≤ ε * Fintype.card Y :=
    fun x hx => (ENNReal.div_le_iff_le_mul (Or.inl (by exact_mod_cast hY.ne')) (Or.inl (ENNReal.natCast_ne_top _))).1
      (by convert h x hx)
  have e : (Finset.univ.filter (fun p : X × Y => A p.1 ∧ B p)).card =
      ∑ x ∈ Finset.univ.filter A, (Finset.univ.filter (fun y => B (x, y))).card := by
    rw [Finset.card_filter, Fintype.sum_prod_type, Finset.sum_filter]
    refine Finset.sum_congr rfl (fun x _ => ?_)
    by_cases hx : A x
    · rw [Finset.card_filter]
      simp only [hx, true_and, ↓reduceIte]
    · simp [hx]
  have hYne : (Fintype.card Y : ℝ≥0∞) ≠ 0 := by exact_mod_cast hY.ne'
  rw [e, Nat.cast_sum, Fintype.card_prod, Nat.cast_mul]
  calc (∑ x ∈ Finset.univ.filter A, ((Finset.univ.filter (fun y => B (x, y))).card : ℝ≥0∞)) /
        (Fintype.card X * Fintype.card Y)
      ≤ (∑ _x ∈ Finset.univ.filter A, ε * Fintype.card Y) / (Fintype.card X * Fintype.card Y) :=
        ENNReal.div_le_div_right (Finset.sum_le_sum (fun x hx => hy x (Finset.mem_filter.1 hx).2)) _
    _ = ((Finset.univ.filter A).card * ε) * Fintype.card Y / (Fintype.card X * Fintype.card Y) := by
        rw [Finset.sum_const, nsmul_eq_mul, mul_assoc]
    _ = (Finset.univ.filter A).card * ε / Fintype.card X :=
        ENNReal.mul_div_mul_right _ _ hYne (ENNReal.natCast_ne_top _)
    _ = ((Finset.univ.filter A).card : ℝ≥0∞) / Fintype.card X * ε := by
        rw [div_eq_mul_inv, div_eq_mul_inv, mul_right_comm]

/-! ## The bound -/

variable {M : ℕ}

/-- Every round's sample lands in its good set while the good sets, weighted by the sample counts, hold at most `K`
items: at most `(K/(M·∑ s_i))^(∑ s_i)`, for `M > 0`. -/
theorem pr_causal_le (hM : 0 < M) : ∀ (n : ℕ) (X : Fin n → Type) [∀ i, Fintype (X i)] (s : Fin n → ℕ)
    (G : (i : Fin n) → ((j : Fin n) → X j × (Fin (s j) → Fin M)) → Finset (Fin M)), Causal G → ∀ K : ℕ,
    pr (fun ω : (j : Fin n) → X j × (Fin (s j) → Fin M) =>
        (∀ i t, (ω i).2 t ∈ G i ω) ∧ ∑ i, s i * (G i ω).card ≤ K)
      ≤ ((K : ℝ≥0∞) / (M * (∑ i, s i : ℕ))) ^ (∑ i, s i) := by
  intro n
  induction n with
  | zero =>
    intro X _ s G _ K
    simpa using pr_le_one _
  | succ n ih =>
    intro X _ s G hG K
    classical
    let T : Type := (i : Fin n) → X i.succ × (Fin (s i.succ) → Fin M)
    let e : ((j : Fin (n + 1)) → X j × (Fin (s j) → Fin M)) ≃ ((Fin (s 0) → Fin M) × T) × X 0 :=
      { toFun := fun ω => (((ω 0).2, Fin.tail ω), (ω 0).1)
        invFun := fun q => Fin.cons (q.2, q.1.1) q.1.2
        left_inv := fun ω => by simp
        right_inv := fun q => by simp }
    rw [pr_equiv e]
    refine pr_le_of_fiber _ _ (fun x₀ => ?_)
    simp only [e, Equiv.coe_fn_symm_mk]
    rcases isEmpty_or_nonempty T with hT | ⟨⟨t₀⟩⟩
    · exact (le_of_eq (Bandwidth.pr_eq_zero_of_card _ (by simp))).trans zero_le
    set G₀ := G 0 (Fin.cons (x₀, fun _ => ⟨0, hM⟩) t₀) with hG₀def
    have hG₀ : ∀ I t, G 0 (Fin.cons (x₀, I) t : (j : Fin (n + 1)) → X j × (Fin (s j) → Fin M)) = G₀ :=
      fun I t => hG 0 _ _ (fun j hj => absurd hj (Fin.not_lt_zero j)) rfl
    have hsum : ∑ i, s i = s 0 + ∑ i : Fin n, s i.succ := Fin.sum_univ_succ s
    by_cases hK : s 0 * G₀.card ≤ K
    swap
    · refine (Bandwidth.pr_mono (fun p hp => ?_)).trans (le_of_eq pr_false) |>.trans zero_le
      have h2 := hp.2
      rw [Fin.sum_univ_succ, hG₀] at h2
      omega
    obtain ⟨K', rfl⟩ : ∃ K', K = s 0 * G₀.card + K' := ⟨K - s 0 * G₀.card, by omega⟩
    let G' : (Fin (s 0) → Fin M) → (i : Fin n) → T → Finset (Fin M) :=
      fun I i t => G i.succ (Fin.cons (x₀, I) t)
    have hG' : ∀ I, Causal (G' I) := by
      intro I i ω ω' hlt hi
      refine hG i.succ _ _ (fun j hj => ?_) (by simpa using hi)
      cases j using Fin.cases with
      | zero => rfl
      | succ j => simpa using hlt j (Fin.succ_lt_succ_iff.1 hj)
    refine (Bandwidth.pr_mono (fun p hp => (?_ : (∀ t, p.1 t ∈ G₀) ∧
        ((∀ i t, (p.2 i).2 t ∈ G' p.1 i p.2) ∧ ∑ i, s i.succ * (G' p.1 i p.2).card ≤ K')))).trans ?_
    · obtain ⟨hacc, hs⟩ := hp
      refine ⟨fun t => ?_, fun i t => ?_, ?_⟩
      · simpa [hG₀] using hacc 0 t
      · simpa [G'] using hacc i.succ t
      · rw [Fin.sum_univ_succ, hG₀] at hs
        simp only [G']
        omega
    have h₁ := pr_and_le_mul (fun I : Fin (s 0) → Fin M => ∀ t, I t ∈ G₀)
      (fun p : (Fin (s 0) → Fin M) × T =>
        (∀ i t, (p.2 i).2 t ∈ G' p.1 i p.2) ∧ ∑ i, s i.succ * (G' p.1 i p.2).card ≤ K')
      (((K' : ℝ≥0∞) / (M * (∑ i : Fin n, s i.succ : ℕ))) ^ (∑ i : Fin n, s i.succ))
      (fun I _ => ih (fun i => X i.succ) (fun i => s i.succ) (G' I) (hG' I) K')
    refine h₁.trans ?_
    rw [pr_sample, hsum]
    exact step_le M G₀.card K' (s 0) _ hM

/-- `pr_causal_le` at any `M`: with no items, a positive sample count leaves no coins at all. -/
theorem pr_causal_le' (n M K : ℕ) (X : Fin n → Type) [∀ i, Fintype (X i)] (s : Fin n → ℕ)
    (G : (i : Fin n) → ((j : Fin n) → X j × (Fin (s j) → Fin M)) → Finset (Fin M)) (hG : Causal G) :
    pr (fun ω : (j : Fin n) → X j × (Fin (s j) → Fin M) =>
        (∀ i t, (ω i).2 t ∈ G i ω) ∧ ∑ i, s i * (G i ω).card ≤ K)
      ≤ ((K : ℝ≥0∞) / (M * (∑ i, s i : ℕ))) ^ (∑ i, s i) := by
  classical
  rcases Nat.eq_zero_or_pos M with rfl | hM
  · by_cases hS : ∑ i, s i = 0
    · rw [hS, pow_zero]
      exact pr_le_one _
    · obtain ⟨i, -, hi⟩ := Finset.exists_ne_zero_of_sum_ne_zero hS
      refine (le_of_eq (Bandwidth.pr_eq_zero_of_card _ ?_)).trans zero_le
      rw [Fintype.card_pi]
      exact Finset.prod_eq_zero (Finset.mem_univ i) (by simp [hi])
  · exact pr_causal_le hM n X s G hG K

end Horizon

namespace Chain

variable {P N b h ρ S L q r s : ℕ}

/-- A chain's passes are sampled as a whole: every pass accepted with at most `K` correct receipts in all is at most
`(K/(P·N))^(s·P)`. -/
theorem pr_horizon_le (A : Chain P N b h ρ S L q r s) (K : ℕ) :
    pr (fun x : Store N b × (Fin P → PassCoins N b h ρ r s) =>
        (∀ j, (A.dev j x.2).Accepts x.1 (x.2 j).1 (A.oracle j x.1 x.2) (x.2 j).2.2) ∧
          ∑ j, ((A.dev j x.2).correct x.1 (x.2 j).1 (A.oracle j x.1 x.2)).card ≤ K)
      ≤ ((K : ℝ≥0∞) / (P * N)) ^ (s * P) := by
  classical
  rw [pr_equiv (Equiv.prodComm _ _)]
  refine pr_le_of_fiber _ _ (fun E => ?_)
  let : Fintype (Bits r → Oracle N b h ρ) := inferInstance
  let : Fintype (Bits ρ × (Bits r → Oracle N b h ρ)) := inferInstance
  let e : (Fin P → PassCoins N b h ρ r s) ≃ (Fin P → (Bits ρ × (Bits r → Oracle N b h ρ)) × (Fin s → Fin N)) :=
    Equiv.piCongrRight (fun _ => (Equiv.prodAssoc _ _ _).symm)
  rw [pr_equiv e]
  let G : (j : Fin P) → (Fin P → (Bits ρ × (Bits r → Oracle N b h ρ)) × (Fin s → Fin N)) → Finset (Fin N) :=
    fun j ω => (A.dev j (e.symm ω)).correct E (e.symm ω j).1 (A.oracle j E (e.symm ω))
  have hG : Causal G := by
    intro i ω ω' hlt hi
    have hη : (fun k : {k : Fin P // k < i} => e.symm ω k.1) = (fun k => e.symm ω' k.1) :=
      funext fun k => by simp [e, hlt k.1 k.2]
    have h₁ : (e.symm ω i).1 = (e.symm ω' i).1 := by simp [e, hi]
    have h₂ : (e.symm ω i).2.1 = (e.symm ω' i).2.1 := by simp [e, hi]
    simp only [G, Chain.dev, Chain.oracle]
    rw [hη, h₁, h₂]
  have key := Horizon.pr_causal_le' P N (s * K) (fun _ => Bits ρ × (Bits r → Oracle N b h ρ)) (fun _ => s) G hG
  refine (Bandwidth.pr_mono (fun ω hω => ?_)).trans (key.trans (le_of_eq ?_))
  · refine ⟨fun i t => Finset.mem_filter.2 ⟨Finset.mem_univ _, hω.1 i t⟩, ?_⟩
    rw [← Finset.mul_sum]
    exact Nat.mul_le_mul_left s hω.2
  · simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
    rcases Nat.eq_zero_or_pos s with rfl | hs
    · simp
    rw [mul_comm P s]
    congr 1
    have hs' : (s : ℝ≥0∞) ≠ 0 := by exact_mod_cast hs.ne'
    push_cast
    rw [show (N : ℝ≥0∞) * (s * P) = s * (P * N) by ring]
    exact ENNReal.mul_div_mul_left _ _ hs' (ENNReal.natCast_ne_top s)

end Chain

end Accounting.Rate

namespace Accounting.SecurityProofs

open Accounting.Rate
open Pous (pr)
open scoped ENNReal

/-- **The causal horizon bound** (`Accounting.Guarantees.HorizonSampling`). -/
theorem HorizonSampling : Accounting.Guarantees.HorizonSampling := by
  intro n M K X _ s G hG
  exact Horizon.pr_causal_le' n M K X s G hG

/-- **A chain's passes are sampled as a whole** (`Accounting.Guarantees.RateHorizon`). -/
theorem RateHorizon : Accounting.Guarantees.RateHorizon := by
  intro P N b h ρ S L q r s K A
  exact Chain.pr_horizon_le A K

/-- **The route `Sampled`, with side sources that ignore the oracle** (`Accounting.Guarantees.RateChainSampledRO`). -/
theorem RateChainSampledRO : Accounting.Guarantees.RateChainSampledRO := by
  intro P N b h ρ S L q r s K lam τ tier A hA
  have h₁ := Chain.pr_horizon_le A K
  have h₂ := RateChainPaidRO P N b h ρ S L q r s lam τ tier A hA
  refine (Bandwidth.pr_mono (fun x hx => ?_)).trans ((Bandwidth.pr_or_le _ _).trans (add_le_add h₁ h₂))
  by_cases hK : ∑ j, ((A.dev j x.2).correct x.1 (x.2 j).1 (A.oracle j x.1 x.2)).card ≤ K
  · exact Or.inl ⟨hx.1, hK⟩
  · refine Or.inr (hx.2.trans_le ?_)
    rw [← Finset.sum_mul]
    exact Nat.mul_le_mul_right _ (by omega)

/-- **The route `Sampled`, with any side sources** (`Accounting.Guarantees.RateChainSampled`). -/
theorem RateChainSampled : Accounting.Guarantees.RateChainSampled := by
  intro P N b h ρ S L q r s K m lam τ tier A
  have h₁ := Chain.pr_horizon_le A K
  have h₂ := RateChainPaid P N b h ρ S L q r s m lam τ tier A
  refine (Bandwidth.pr_mono (fun x hx => ?_)).trans ((Bandwidth.pr_or_le _ _).trans (add_le_add h₁ h₂))
  by_cases hK : ∑ j, ((A.dev j x.2).correct x.1 (x.2 j).1 (A.oracle j x.1 x.2)).card ≤ K
  · exact Or.inl ⟨hx.1, hK⟩
  · refine Or.inr (hx.2.trans_le ?_)
    rw [← Finset.sum_mul]
    exact Nat.mul_le_mul_right _ (by omega)

end Accounting.SecurityProofs
