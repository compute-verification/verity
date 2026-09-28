import Proofs.Flock.Soundness.Audit.Partitioning.Guarantees

/-!
# Two-stage detection risk: the proofs

The proofs of the two-stage statements of `Partitioning.Guarantees`.
-/

namespace FlockSoundness.Audit.Partitioning.SecurityProofs

/-- With `G = C(x, j)`, `B = C(x, j+1)` and `A = C(x, j+2)`: `B² ≤ 2AG + 2BG + G²`. Through `B (j+1) = G (x − j)` and
`A (j+2) = B (x − j − 1)`, it is `G² (j s² + 2(j+1)² s + (j+1)²(j+2)) ≥ 0` with `s = x − j`. -/
theorem choose_sq_le (x j : ℕ) :
    ((x.choose (j + 1) : ℝ)) ^ 2 ≤
      2 * x.choose (j + 2) * x.choose j + 2 * x.choose (j + 1) * x.choose j + (x.choose j : ℝ) ^ 2 := by
  rcases lt_or_ge x j with hx | hx
  · simp [Nat.choose_eq_zero_of_lt hx, Nat.choose_eq_zero_of_lt (hx.trans (Nat.lt_succ_self j)),
      Nat.choose_eq_zero_of_lt (hx.trans (by omega : j < j + 2))]
  set G : ℝ := (x.choose j : ℝ)
  set B : ℝ := (x.choose (j + 1) : ℝ)
  set A : ℝ := (x.choose (j + 2) : ℝ)
  set s : ℝ := (x : ℝ) - j
  have hs : 0 ≤ s := by simp only [s]; have : (j : ℝ) ≤ x := by exact_mod_cast hx
                        linarith
  have hB : B * ((j : ℝ) + 1) = G * s := by
    have := Nat.choose_succ_right_eq x j
    simp only [B, G, s]
    rw [← Nat.cast_sub hx]
    exact_mod_cast this
  have hA : A * ((j : ℝ) + 2) = B * (s - 1) := by
    have := Nat.choose_succ_right_eq x (j + 1)
    rcases eq_or_lt_of_le hx with h | h
    · subst h
      simp [A, B, Nat.choose_eq_zero_of_lt (by omega : j < j + 2)]
    · have h' : j + 1 ≤ x := h
      have e : (x.choose (j + 2) : ℝ) * ((j + 2 : ℕ) : ℝ) = (x.choose (j + 1) : ℝ) * ((x - (j + 1) : ℕ) : ℝ) := by
        exact_mod_cast this
      rw [Nat.cast_sub h'] at e
      push_cast at e
      simp only [A, B, s]
      linear_combination e
  have hG : 0 ≤ G := by positivity
  have key : ((j : ℝ) + 1) ^ 2 * ((j : ℝ) + 2) *
      (2 * A * G + 2 * B * G + G ^ 2 - B ^ 2) =
      G ^ 2 * (j * s ^ 2 + 2 * ((j : ℝ) + 1) ^ 2 * s + ((j : ℝ) + 1) ^ 2 * ((j : ℝ) + 2)) := by
    linear_combination (-((j : ℝ) + 2) * (B * ((j : ℝ) + 1) + G * s) + 2 * ((j : ℝ) + 1) * G * (s - 1) +
      2 * ((j : ℝ) + 1) * ((j : ℝ) + 2) * G) * hB + 2 * ((j : ℝ) + 1) ^ 2 * G * hA
  have hpos : 0 < ((j : ℝ) + 1) ^ 2 * ((j : ℝ) + 2) := by positivity
  have hrhs : 0 ≤ G ^ 2 * (j * s ^ 2 + 2 * ((j : ℝ) + 1) ^ 2 * s + ((j : ℝ) + 1) ^ 2 * ((j : ℝ) + 2)) := by
    positivity
  have : 0 ≤ 2 * A * G + 2 * B * G + G ^ 2 - B ^ 2 := by
    by_contra hneg
    push Not at hneg
    have := mul_neg_of_pos_of_neg hpos hneg
    linarith
  linarith

theorem RiskConcave : Guarantees.RiskConcave := by
  intro N k m p hk hp0 hp
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 2 := ⟨k - 2, by omega⟩
  set T : ℝ := (N.choose (j + 2) : ℝ) with hT
  set q : ℝ := 1 - p with hq
  have hpq : p ≤ q := by linarith
  have hq0 : 0 < q := by linarith
  set f : ℕ → ℝ := fun m => q + p * ((N - m).choose (j + 2) : ℝ) / T with hf
  have hψ : ∀ m, detectionRisk N (j + 2) p m = -Real.log (f m) := fun m => rfl
  have hfpos : ∀ m, 0 < f m := fun m => by
    have : 0 ≤ p * ((N - m).choose (j + 2) : ℝ) / T := by positivity
    simp only [f]
    linarith
  have key : f (m + 1) ^ 2 ≤ f m * f (m + 2) := by
    rcases lt_or_ge N (m + 2) with hN | hN
    · have h0 : ∀ i, m ≤ i → (N - i).choose (j + 2) = 0 := fun i hi => Nat.choose_eq_zero_of_lt (by omega)
      simp only [f, h0 m le_rfl, h0 (m + 1) (by omega), h0 (m + 2) (by omega)]
      exact le_of_eq (by ring)
    · obtain ⟨x, hx⟩ : ∃ x, N - m = x + 2 := ⟨N - m - 2, by omega⟩
      have e1 : N - (m + 1) = x + 1 := by omega
      have e2 : N - (m + 2) = x := by omega
      have hcore := choose_sq_le x j
      set G : ℝ := (x.choose j : ℝ)
      set B : ℝ := (x.choose (j + 1) : ℝ)
      set A : ℝ := (x.choose (j + 2) : ℝ)
      have c2 : ((x + 2).choose (j + 2) : ℝ) = A + 2 * B + G := by
        rw [Nat.choose_succ_succ', Nat.choose_succ_succ', Nat.choose_succ_succ' x (j + 1)]
        push_cast
        ring
      have c1 : ((x + 1).choose (j + 2) : ℝ) = A + B := by
        rw [Nat.choose_succ_succ']
        push_cast
        ring
      have hdT : A + 2 * B + G ≤ T := by
        rw [← c2]
        show ((x + 2).choose (j + 2) : ℝ) ≤ (N.choose (j + 2) : ℝ)
        exact_mod_cast Nat.choose_le_choose (j + 2) (by omega : x + 2 ≤ N)
      have hG : 0 ≤ G := by positivity
      have h1 : B ^ 2 - A * G ≤ G * T := by
        have h2 : G * (A + 2 * B + G) ≤ G * T := mul_le_mul_of_nonneg_left hdT hG
        have h3 : G * (A + 2 * B + G) = A * G + 2 * B * G + G ^ 2 := by ring
        linarith
      have hpoly : p ^ 2 * (B ^ 2 - A * G) ≤ p * q * (G * T) := by
        rcases le_total (B ^ 2 - A * G) 0 with h | h
        · have : 0 ≤ p * q * (G * T) := by positivity
          have := mul_nonpos_of_nonneg_of_nonpos (sq_nonneg p) h
          linarith
        · calc p ^ 2 * (B ^ 2 - A * G) ≤ p * q * (B ^ 2 - A * G) := by
                rw [sq]
                exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hpq hp0) h
            _ ≤ p * q * (G * T) := mul_le_mul_of_nonneg_left h1 (by positivity)
      rcases eq_or_lt_of_le (show (0 : ℝ) ≤ T by positivity) with hT0 | hTpos
      · simp only [f, ← hT0, div_zero]
        exact le_of_eq (by ring)
      · have hdiff : f m * f (m + 2) - f (m + 1) ^ 2 = (p * q * (G * T) - p ^ 2 * (B ^ 2 - A * G)) / T ^ 2 := by
          simp only [f, hx, e1, e2, c1, c2]
          field_simp
          ring
        have : 0 ≤ f m * f (m + 2) - f (m + 1) ^ 2 := by
          rw [hdiff]
          exact div_nonneg (by linarith) (by positivity)
        linarith
  rw [hψ, hψ, hψ]
  have hlog := Real.log_le_log (by have := hfpos (m + 1); positivity) key
  rw [Real.log_pow, Real.log_mul (hfpos m).ne' (hfpos (m + 2)).ne'] at hlog
  push_cast at hlog
  linarith

theorem Concentration : Guarantees.Concentration := by
  intro N k R p hk hkN hp0 hp
  have hN : 0 < N := by omega
  set ψ : ℕ → ℝ := detectionRisk N k p with hψ
  have h0 : ψ 0 = 0 := by
    have hT : (N.choose k : ℝ) ≠ 0 := by exact_mod_cast (Nat.choose_pos hkN).ne'
    simp [ψ, detectionRisk, hT]
  have hanti : Antitone fun t => ψ (t + 1) - ψ t :=
    antitone_nat_of_succ_le fun t => RiskConcave N k t p hk hp0 hp
  have shift : ∀ c b t, c ≤ b → ψ (b + t) - ψ b ≤ ψ (c + t) - ψ c := by
    intro c b t hcb
    induction t with
    | zero => simp
    | succ t ih =>
        have := hanti (show c + t ≤ b + t by omega)
        simp only at this
        rw [← add_assoc, ← add_assoc]
        linarith
  have step : ∀ M a, a ≤ N →
      (((M + a) / N : ℕ) : ℝ) * ψ N + ψ ((M + a) % N) ≤ ((M / N : ℕ) : ℝ) * ψ N + ψ (M % N) + ψ a := by
    intro M a ha
    have hr : M % N < N := Nat.mod_lt _ hN
    have hM := Nat.mod_add_div M N
    by_cases h : M % N + a < N
    · obtain ⟨h1, h2⟩ := (Nat.div_mod_unique hN).2 ⟨show M % N + a + N * (M / N) = M + a by omega, h⟩
      rw [h1, h2]
      have := shift 0 a (M % N) (Nat.zero_le _)
      rw [zero_add, h0, add_comm a] at this
      linarith
    · obtain ⟨h1, h2⟩ := (Nat.div_mod_unique hN).2
        ⟨show M % N + a - N + N * (M / N + 1) = M + a by rw [mul_add, mul_one]; omega, by omega⟩
      rw [h1, h2]
      have := shift (M % N + a - N) a (N - a) (by omega)
      rw [show a + (N - a) = N by omega, show M % N + a - N + (N - a) = M % N by omega] at this
      push_cast
      linarith
  intro m hm
  induction R with
  | zero => simp [h0]
  | succ R ih =>
      have ih' := ih (fun i => m i.castSucc) (fun i => hm _)
      have hs := step (∑ i : Fin R, m i.castSucc) (m (Fin.last R)) (hm _)
      simp only [Fin.sum_univ_castSucc]
      linarith

section Nested

open Finset Game
open scoped ENNReal

/-- Averaging over a pair of coins is averaging over the first of the average over the second. -/
theorem avg_prod_eq {C D : Type} [Fintype C] [Fintype D] (f : C × D → ℝ≥0∞) :
    avg f = avg fun c => avg fun d => f (c, d) := by
  unfold avg
  rw [Fintype.sum_prod_type, Fintype.card_prod, Nat.cast_mul,
    ENNReal.mul_inv (Or.inr (ENNReal.natCast_ne_top _)) (Or.inl (ENNReal.natCast_ne_top _))]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun d _ => ?_
  ring

theorem prCoin_prod {C D : Type} [Fintype C] [Fintype D] (p : C × D → Prop) :
    prCoin p = avg fun c => prCoin fun d => p (c, d) := by
  unfold prCoin
  exact avg_prod_eq _

/-- **The nested law's escape, in `ℝ≥0∞`.** Each replay unit `i` contributes its Bernoulli factor (`Law.bernFactor`)
at the escape, under the uniform `k`-subset, of its own wrong proof units: its two coins are independent of the
other replay units'. -/
theorem nested_escape (R N k a b : ℕ) [NeZero b] (hk : k ≤ N) (hab : a ≤ b) (B : Finset (Fin (R * N))) :
    (nested R N k a b hk).escape B =
      ∏ i, Law.bernFactor a b
        ((Law.subset N k hk).escape (univ.filter fun j : Fin N => finProdFinEquiv (i, j) ∈ B)) := by
  classical
  set Bi : Fin R → Finset (Fin N) := fun i => univ.filter fun j : Fin N => finProdFinEquiv (i, j) ∈ B
  set E : Fin R → ℝ≥0∞ := fun i => (Law.subset N k hk).escape (Bi i)
  have hdisj : ∀ ω : (Fin R → Fin b) × (Fin R → (Law.subset N k hk).Ω),
      Disjoint ((nested R N k a b hk).draw ω) B ↔
        ∀ i ∈ (univ : Finset (Fin R)), (ω.1 i).val < a → Disjoint (ω.2 i).1 (Bi i) := by
    intro ω
    simp only [Finset.disjoint_left, nested, Law.subset, Law.bernoulli, mem_filter, mem_univ, true_and, Bi]
    constructor
    · intro h i _ hi j hj hjB
      exact h (a := finProdFinEquiv (i, j)) (by simpa using And.intro hi hj) hjB
    · rintro h v ⟨hv1, hv2⟩ hvB
      exact h _ trivial hv1 hv2 (by rw [Prod.mk.eta, Equiv.apply_symm_apply]; exact hvB)
  have hstage : ∀ (c : Fin R → Fin b) (i : Fin R),
      prCoin (fun S : (Law.subset N k hk).Ω => (c i).val < a → Disjoint S.1 (Bi i)) =
        if (c i).val < a then E i else 1 := by
    intro c i
    by_cases h : (c i).val < a
    · simp only [h, true_implies, ite_true]
      rfl
    · simp only [h, false_implies, ite_false]
      unfold prCoin
      simp only [ite_true]
      exact avg_const 1
  have hin : ∀ c : Fin R → Fin b,
      (prCoin fun d : Fin R → (Law.subset N k hk).Ω =>
        ∀ i ∈ (univ : Finset (Fin R)), (c i).val < a → Disjoint (d i).1 (Bi i)) =
      ∏ i, if (c i).val < a then E i else 1 := fun c =>
    (prCoin_pi_forall (Ω := fun _ : Fin R => (Law.subset N k hk).Ω) univ
      (fun i (S : (Law.subset N k hk).Ω) => (c i).val < a → Disjoint S.1 (Bi i))).trans
      (Finset.prod_congr rfl fun i _ => hstage c i)
  change prCoin (fun ω : (Fin R → Fin b) × (Fin R → (Law.subset N k hk).Ω) =>
    Disjoint ((nested R N k a b hk).draw ω) B) = _
  simp_rw [hdisj]
  rw [prCoin_prod]
  simp only [hin]
  rw [avg_pi_prod (Ω := fun _ : Fin R => Fin b) (fun i (x : Fin b) => if x.val < a then E i else 1)]
  exact Finset.prod_congr rfl fun i _ => Law.avg_bern hab _

theorem NestedEscape : Guarantees.NestedEscape := by
  intro R N k a b _ hk hab B
  classical
  have hb : (0 : ℝ) < b := by exact_mod_cast NeZero.pos b
  have hT : (0 : ℝ) < N.choose k := by exact_mod_cast Nat.choose_pos hk
  have hfac : ∀ i, (Law.bernFactor a b
      ((Law.subset N k hk).escape (univ.filter fun j : Fin N => finProdFinEquiv (i, j) ∈ B))).toReal =
      1 - (a / b : ℝ) + (a / b : ℝ) * ((N - wrongIn B i).choose k : ℝ) / (N.choose k : ℝ) := by
    intro i
    rw [Law.subset_escape]
    have hE : (((N - (univ.filter fun j : Fin N => finProdFinEquiv (i, j) ∈ B).card).choose k : ℕ) : ℝ≥0∞) /
        ((N.choose k : ℕ) : ℝ≥0∞) ≠ ⊤ :=
      ENNReal.div_ne_top (ENNReal.natCast_ne_top _) (by exact_mod_cast (Nat.choose_pos hk).ne')
    unfold Law.bernFactor
    rw [ENNReal.toReal_div, ENNReal.toReal_add (ENNReal.mul_ne_top (ENNReal.natCast_ne_top _) hE)
      (ENNReal.natCast_ne_top _), ENNReal.toReal_mul, ENNReal.toReal_div]
    simp only [ENNReal.toReal_natCast]
    rw [Nat.cast_sub hab]
    unfold wrongIn
    field_simp
    ring
  have hprod : ((nested R N k a b hk).escape B).toReal =
      ∏ i, (1 - (a / b : ℝ) + (a / b : ℝ) * ((N - wrongIn B i).choose k : ℝ) / (N.choose k : ℝ)) := by
    rw [nested_escape R N k a b hk hab B, ENNReal.toReal_prod]
    exact Finset.prod_congr rfl fun i _ => hfac i
  refine ⟨hprod, fun hlt => ?_⟩
  rw [hprod]
  simp only [detectionRisk, Finset.sum_neg_distrib, neg_neg]
  rw [Real.exp_sum]
  refine Finset.prod_congr rfl fun i _ => (Real.exp_log ?_).symm
  have hp : (a : ℝ) / b < 1 := (div_lt_one hb).2 (by exact_mod_cast hlt)
  have : 0 ≤ (a / b : ℝ) * ((N - wrongIn B i).choose k : ℝ) / (N.choose k : ℝ) := by positivity
  linarith

end Nested

end FlockSoundness.Audit.Partitioning.SecurityProofs
