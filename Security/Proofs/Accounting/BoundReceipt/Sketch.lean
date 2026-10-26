import Mathlib
import Definitions.Accounting.Guarantees.BoundSketch

/-!
# The deployed bound receipt's sketch: the proofs

Lindsey's lemma over `𝔽_p^n`, without characters. Write `D_c(x, y) = [c = ⟨x, y⟩] − 1/p`. For a fixed `x`,
`Σ_c D_c(x, y)·D_c(x, y') = [⟨x, y⟩ = ⟨x, y'⟩] − 1/p` (`sum_out`), and summed over `x` that is `p^n(1 − 1/p)` at
`y = y'` and `0` otherwise, since a nonzero linear form's kernel has `p^{n−1}` points (`card_kernel`). So at a leaf
where the block side outputs `o(x)`, Cauchy–Schwarz over `x` bounds the advantage over `1/p` by
`√(‖f‖²‖g‖²p^n)` (`leafX`; the tile side's leaf by symmetry, `leafY`). Each bit sent splits one side's weight in two,
and the advantages add (`adv_le`): `2^bits` leaves.
-/

namespace Accounting.BoundSketch

open Finset

variable {X Y O O' : Type}

theorem Protocol.eval_map (g : O → O') (π : Protocol X Y O) (x : X) (y : Y) :
    (π.map g).eval x y = g (π.eval x y) := by
  induction π with
  | outX o => rfl
  | outY o => rfl
  | sendX s t f iht ihf => simp only [Protocol.map, Protocol.eval]; split_ifs <;> assumption
  | sendY s t f iht ihf => simp only [Protocol.map, Protocol.eval]; split_ifs <;> assumption

theorem Protocol.bits_map (g : O → O') (π : Protocol X Y O) : (π.map g).bits = π.bits := by
  induction π with
  | outX o => rfl
  | outY o => rfl
  | sendX s t f iht ihf => simp only [Protocol.map, Protocol.bits, iht, ihf]
  | sendY s t f iht ihf => simp only [Protocol.map, Protocol.bits, iht, ihf]

variable {p : ℕ} [hp : Fact p.Prime] {ι : Type} [Fintype ι] [DecidableEq ι]

omit hp [DecidableEq ι] in
theorem ip_comm (x y : ι → ZMod p) : ip x y = ip y x := by
  simp only [ip, mul_comm]

omit [DecidableEq ι] in
theorem ip_sub (x y y' : ι → ZMod p) : ip x y - ip x y' = ip x (y - y') := by
  simp only [ip, Pi.sub_apply, mul_sub, Finset.sum_sub_distrib]

theorem ip_sub_single (x w : ι → ZMod p) (i : ι) (t : ZMod p) :
    ip (x - Pi.single i t) w = ip x w - t * w i := by
  simp only [ip, Pi.sub_apply, sub_mul, Finset.sum_sub_distrib, Pi.single_apply, ite_mul, zero_mul,
    Finset.sum_ite_eq', Finset.mem_univ, ite_true]

theorem ip_add_single (x w : ι → ZMod p) (i : ι) (t : ZMod p) :
    ip (x + Pi.single i t) w = ip x w + t * w i := by
  simp only [ip, Pi.add_apply, add_mul, Finset.sum_add_distrib, Pi.single_apply, ite_mul, zero_mul,
    Finset.sum_ite_eq', Finset.mem_univ, ite_true]

/-- Every fiber of a linear form `x ↦ ⟨x, w⟩` with `w_i ≠ 0` has as many points as its kernel. -/
theorem card_fiber_eq (w : ι → ZMod p) {i : ι} (hi : w i ≠ 0) (c : ZMod p) :
    (univ.filter fun x : ι → ZMod p => ip x w = c).card = (univ.filter fun x : ι → ZMod p => ip x w = 0).card := by
  refine Finset.card_nbij' (fun x => x - Pi.single i (c / w i)) (fun x => x + Pi.single i (c / w i)) ?_ ?_ ?_ ?_
  · intro x hx
    simp only [coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at hx ⊢
    rw [ip_sub_single, hx, div_mul_cancel₀ c hi, sub_self]
  · intro x hx
    simp only [coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at hx ⊢
    rw [ip_add_single, hx, div_mul_cancel₀ c hi, zero_add]
  · intro x _
    exact sub_add_cancel x _
  · intro x _
    exact add_sub_cancel_right x _

/-- A nonzero linear form on `𝔽_p^n` has `p^{n−1}` zeros. -/
theorem card_kernel (w : ι → ZMod p) (hw : w ≠ 0) :
    (p : ℝ) * ((univ.filter fun x : ι → ZMod p => ip x w = 0).card : ℝ) = (p : ℝ) ^ Fintype.card ι := by
  obtain ⟨i, hi⟩ := Function.ne_iff.1 hw
  have h := Finset.card_eq_sum_card_fiberwise (f := fun x : ι → ZMod p => ip x w) (s := univ) (t := univ)
    (fun _ _ => Finset.mem_coe.2 (mem_univ _))
  rw [Finset.sum_congr rfl (fun c _ => card_fiber_eq w hi c), sum_const, card_univ, card_univ, ZMod.card,
    smul_eq_mul, Fintype.card_fun, ZMod.card] at h
  exact_mod_cast h.symm

theorem sum_balanced (y y' : ι → ZMod p) :
    ∑ x : ι → ZMod p, ((if ip x y = ip x y' then (1 : ℝ) else 0) - 1 / p) =
      if y = y' then (p : ℝ) ^ Fintype.card ι * (1 - 1 / p) else 0 := by
  have hp0 : (p : ℝ) ≠ 0 := by exact_mod_cast hp.out.ne_zero
  have hcard : ((Fintype.card (ι → ZMod p) : ℕ) : ℝ) = (p : ℝ) ^ Fintype.card ι := by
    rw [Fintype.card_fun, ZMod.card]; push_cast; rfl
  rw [Finset.sum_sub_distrib, Finset.sum_boole, sum_const, card_univ, nsmul_eq_mul, hcard]
  split_ifs with h
  · subst h
    rw [filter_true_of_mem (fun x _ => rfl), card_univ, hcard]
    ring
  · have hk := card_kernel (y - y') (sub_ne_zero.2 h)
    have e : (univ.filter fun x : ι → ZMod p => ip x y = ip x y') = univ.filter fun x => ip x (y - y') = 0 := by
      ext x
      simp only [mem_filter, mem_univ, true_and, ← ip_sub, sub_eq_zero]
    rw [e]
    field_simp
    linarith [hk]

theorem sum_out (a b : ZMod p) :
    ∑ c : ZMod p, ((if c = a then (1 : ℝ) else 0) - 1 / p) * ((if c = b then (1 : ℝ) else 0) - 1 / p) =
      (if a = b then (1 : ℝ) else 0) - 1 / p := by
  have hp0 : (p : ℝ) ≠ 0 := by exact_mod_cast hp.out.ne_zero
  have e : ∀ c : ZMod p, ((if c = a then (1 : ℝ) else 0) - 1 / p) * ((if c = b then (1 : ℝ) else 0) - 1 / p) =
      ((if c = a then (if a = b then (1 : ℝ) else 0) else 0) - (if c = a then (1 : ℝ) / p else 0)) -
        (if c = b then (1 : ℝ) / p else 0) + 1 / (p : ℝ) ^ 2 := by
    intro c
    by_cases hca : c = a <;> by_cases hcb : c = b <;> by_cases hab : a = b <;> simp_all <;> ring
  simp only [e, sum_add_distrib, sum_sub_distrib, sum_ite_eq', mem_univ, ite_true, sum_const, card_univ, ZMod.card,
    nsmul_eq_mul]
  field_simp
  ring

theorem leafX (o : (ι → ZMod p) → ZMod p) (f g : (ι → ZMod p) → ℝ) :
    |∑ x, ∑ y, f x * g y * ((if o x = ip x y then (1 : ℝ) else 0) - 1 / p)| ≤
      Real.sqrt ((∑ x, f x ^ 2) * (∑ y, g y ^ 2) * (p : ℝ) ^ Fintype.card ι) := by
  apply Real.abs_le_sqrt
  set D : ZMod p → (ι → ZMod p) → (ι → ZMod p) → ℝ := fun c x y => (if c = ip x y then (1 : ℝ) else 0) - 1 / p
    with hD
  have h1 : ∑ x, ∑ y, f x * g y * ((if o x = ip x y then (1 : ℝ) else 0) - 1 / p) =
      ∑ x, f x * ∑ y, g y * D (o x) x y := by
    refine sum_congr rfl fun x _ => ?_
    rw [mul_sum]
    exact sum_congr rfl fun y _ => by simp only [hD]; ring
  have h2 : ∀ x, (∑ y, g y * D (o x) x y) ^ 2 ≤ ∑ c, (∑ y, g y * D c x y) ^ 2 := fun x =>
    single_le_sum (f := fun c => (∑ y, g y * D c x y) ^ 2) (fun c _ => sq_nonneg _) (mem_univ (o x))
  have hx : ∀ x, ∑ c, (∑ y, g y * D c x y) ^ 2 =
      ∑ y, ∑ y', g y * g y' * ((if ip x y = ip x y' then (1 : ℝ) else 0) - 1 / p) := by
    intro x
    calc ∑ c, (∑ y, g y * D c x y) ^ 2 = ∑ c, ∑ y, ∑ y', g y * g y' * (D c x y * D c x y') := by
          refine sum_congr rfl fun c _ => ?_
          rw [sq, sum_mul_sum]
          exact sum_congr rfl fun y _ => sum_congr rfl fun y' _ => by ring
      _ = ∑ y, ∑ y', ∑ c, g y * g y' * (D c x y * D c x y') := by
          rw [sum_comm]
          exact sum_congr rfl fun y _ => sum_comm
      _ = _ := by
          refine sum_congr rfl fun y _ => sum_congr rfl fun y' _ => ?_
          rw [← mul_sum]
          congr 1
          exact sum_out _ _
  have h3 : ∑ x, ∑ c, (∑ y, g y * D c x y) ^ 2 = (p : ℝ) ^ Fintype.card ι * (1 - 1 / p) * ∑ y, g y ^ 2 := by
    rw [sum_congr rfl fun x _ => hx x, sum_comm, mul_sum]
    refine sum_congr rfl fun y _ => ?_
    rw [sum_comm]
    simp_rw [← mul_sum, sum_balanced, mul_ite, mul_zero, sum_ite_eq, mem_univ, ite_true]
    ring
  rw [h1]
  have hf : 0 ≤ ∑ x, f x ^ 2 := sum_nonneg fun _ _ => sq_nonneg _
  have hg : 0 ≤ ∑ y, g y ^ 2 := sum_nonneg fun _ _ => sq_nonneg _
  have hpn : 0 ≤ (p : ℝ) ^ Fintype.card ι := by positivity
  have h1p : 0 ≤ 1 / (p : ℝ) := by positivity
  calc (∑ x, f x * ∑ y, g y * D (o x) x y) ^ 2 ≤ (∑ x, f x ^ 2) * ∑ x, (∑ y, g y * D (o x) x y) ^ 2 :=
        sum_mul_sq_le_sq_mul_sq _ _ _
    _ ≤ (∑ x, f x ^ 2) * ∑ x, ∑ c, (∑ y, g y * D c x y) ^ 2 :=
        mul_le_mul_of_nonneg_left (sum_le_sum fun x _ => h2 x) hf
    _ = (∑ x, f x ^ 2) * ((p : ℝ) ^ Fintype.card ι * (1 - 1 / p) * ∑ y, g y ^ 2) := by rw [h3]
    _ ≤ (∑ x, f x ^ 2) * (∑ y, g y ^ 2) * (p : ℝ) ^ Fintype.card ι := by
        nlinarith [mul_nonneg (mul_nonneg hf hg) (mul_nonneg hpn h1p)]

theorem leafY (o : (ι → ZMod p) → ZMod p) (f g : (ι → ZMod p) → ℝ) :
    |∑ x, ∑ y, f x * g y * ((if o y = ip x y then (1 : ℝ) else 0) - 1 / p)| ≤
      Real.sqrt ((∑ x, f x ^ 2) * (∑ y, g y ^ 2) * (p : ℝ) ^ Fintype.card ι) := by
  have e : ∑ x, ∑ y, f x * g y * ((if o y = ip x y then (1 : ℝ) else 0) - 1 / p) =
      ∑ y, ∑ x, g y * f x * ((if o y = ip y x then (1 : ℝ) else 0) - 1 / p) := by
    rw [sum_comm]
    exact sum_congr rfl fun y _ => sum_congr rfl fun x _ => by rw [ip_comm x y]; ring
  rw [e, mul_comm (∑ x, f x ^ 2)]
  exact leafX o g f

theorem sqrt_le_left {a b G P : ℝ} (h : a ≤ b) (hG : 0 ≤ G) (hP : 0 ≤ P) :
    Real.sqrt (a * G * P) ≤ Real.sqrt (b * G * P) :=
  Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right h hG) hP)

theorem sqrt_le_mid {F a b P : ℝ} (h : a ≤ b) (hF : 0 ≤ F) (hP : 0 ≤ P) :
    Real.sqrt (F * a * P) ≤ Real.sqrt (F * b * P) :=
  Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left h hF) hP)

theorem sq_ite_le (c : Prop) [Decidable c] (u : ℝ) : (if c then u else 0) ^ 2 ≤ u ^ 2 := by
  split_ifs <;> simp [sq_nonneg]

theorem sq_ite_le' (c : Prop) [Decidable c] (u : ℝ) : (if c then 0 else u) ^ 2 ≤ u ^ 2 := by
  split_ifs <;> simp [sq_nonneg]

/-- The advantage over `1/p` of a protocol's output at weights `f`, `g`: at most `2^bits·√(‖f‖²‖g‖²p^n)`. -/
theorem adv_le (π : Protocol (ι → ZMod p) (ι → ZMod p) (ZMod p)) (f g : (ι → ZMod p) → ℝ) :
    |∑ x, ∑ y, f x * g y * ((if π.eval x y = ip x y then (1 : ℝ) else 0) - 1 / p)| ≤
      2 ^ π.bits * Real.sqrt ((∑ x, f x ^ 2) * (∑ y, g y ^ 2) * (p : ℝ) ^ Fintype.card ι) := by
  have hpn : 0 ≤ (p : ℝ) ^ Fintype.card ι := by positivity
  induction π generalizing f g with
  | outX o =>
    simp only [Protocol.bits, pow_zero, one_mul, Protocol.eval]
    exact leafX o f g
  | outY o =>
    simp only [Protocol.bits, pow_zero, one_mul, Protocol.eval]
    exact leafY o f g
  | sendX s t e iht ihe =>
    have hsplit : ∑ x, ∑ y, f x * g y * ((if (Protocol.sendX s t e).eval x y = ip x y then (1 : ℝ) else 0) - 1 / p) =
        ∑ x, ∑ y, (if s x then f x else 0) * g y * ((if t.eval x y = ip x y then (1 : ℝ) else 0) - 1 / p) +
          ∑ x, ∑ y, (if s x then 0 else f x) * g y * ((if e.eval x y = ip x y then (1 : ℝ) else 0) - 1 / p) := by
      rw [← sum_add_distrib]
      refine sum_congr rfl fun x _ => ?_
      rw [← sum_add_distrib]
      refine sum_congr rfl fun y _ => ?_
      cases h : s x <;> simp [Protocol.eval, h]
    have hg : 0 ≤ ∑ y, g y ^ 2 := sum_nonneg fun _ _ => sq_nonneg _
    have h₁ := iht (fun x => if s x then f x else 0) g
    have h₂ := ihe (fun x => if s x then 0 else f x) g
    have s₁ := sqrt_le_left (sum_le_sum (s := univ) fun x _ => sq_ite_le (s x = true) (f x)) hg hpn
    have s₂ := sqrt_le_left (sum_le_sum (s := univ) fun x _ => sq_ite_le' (s x = true) (f x)) hg hpn
    have p₁ : (2 : ℝ) ^ t.bits ≤ 2 ^ max t.bits e.bits := pow_le_pow_right₀ (by norm_num) (le_max_left _ _)
    have p₂ : (2 : ℝ) ^ e.bits ≤ 2 ^ max t.bits e.bits := pow_le_pow_right₀ (by norm_num) (le_max_right _ _)
    have q₀ := Real.sqrt_nonneg ((∑ x, f x ^ 2) * (∑ y, g y ^ 2) * (p : ℝ) ^ Fintype.card ι)
    rw [hsplit, Protocol.bits, pow_succ]
    calc _ ≤ |∑ x, ∑ y, (if s x then f x else 0) * g y * ((if t.eval x y = ip x y then (1 : ℝ) else 0) - 1 / p)| +
          |∑ x, ∑ y, (if s x then 0 else f x) * g y * ((if e.eval x y = ip x y then (1 : ℝ) else 0) - 1 / p)| :=
          abs_add_le _ _
      _ ≤ 2 ^ max t.bits e.bits * Real.sqrt ((∑ x, f x ^ 2) * (∑ y, g y ^ 2) * (p : ℝ) ^ Fintype.card ι) +
          2 ^ max t.bits e.bits * Real.sqrt ((∑ x, f x ^ 2) * (∑ y, g y ^ 2) * (p : ℝ) ^ Fintype.card ι) := by
          gcongr
          · exact h₁.trans (mul_le_mul p₁ s₁ (Real.sqrt_nonneg _) (by positivity))
          · exact h₂.trans (mul_le_mul p₂ s₂ (Real.sqrt_nonneg _) (by positivity))
      _ = _ := by ring
  | sendY s t e iht ihe =>
    have hsplit : ∑ x, ∑ y, f x * g y * ((if (Protocol.sendY s t e).eval x y = ip x y then (1 : ℝ) else 0) - 1 / p) =
        ∑ x, ∑ y, f x * (if s y then g y else 0) * ((if t.eval x y = ip x y then (1 : ℝ) else 0) - 1 / p) +
          ∑ x, ∑ y, f x * (if s y then 0 else g y) * ((if e.eval x y = ip x y then (1 : ℝ) else 0) - 1 / p) := by
      rw [← sum_add_distrib]
      refine sum_congr rfl fun x _ => ?_
      rw [← sum_add_distrib]
      refine sum_congr rfl fun y _ => ?_
      cases h : s y <;> simp [Protocol.eval, h]
    have hf : 0 ≤ ∑ x, f x ^ 2 := sum_nonneg fun _ _ => sq_nonneg _
    have h₁ := iht f (fun y => if s y then g y else 0)
    have h₂ := ihe f (fun y => if s y then 0 else g y)
    have s₁ := sqrt_le_mid (sum_le_sum (s := univ) fun y _ => sq_ite_le (s y = true) (g y)) hf hpn
    have s₂ := sqrt_le_mid (sum_le_sum (s := univ) fun y _ => sq_ite_le' (s y = true) (g y)) hf hpn
    have p₁ : (2 : ℝ) ^ t.bits ≤ 2 ^ max t.bits e.bits := pow_le_pow_right₀ (by norm_num) (le_max_left _ _)
    have p₂ : (2 : ℝ) ^ e.bits ≤ 2 ^ max t.bits e.bits := pow_le_pow_right₀ (by norm_num) (le_max_right _ _)
    rw [hsplit, Protocol.bits, pow_succ]
    calc _ ≤ |∑ x, ∑ y, f x * (if s y then g y else 0) * ((if t.eval x y = ip x y then (1 : ℝ) else 0) - 1 / p)| +
          |∑ x, ∑ y, f x * (if s y then 0 else g y) * ((if e.eval x y = ip x y then (1 : ℝ) else 0) - 1 / p)| :=
          abs_add_le _ _
      _ ≤ 2 ^ max t.bits e.bits * Real.sqrt ((∑ x, f x ^ 2) * (∑ y, g y ^ 2) * (p : ℝ) ^ Fintype.card ι) +
          2 ^ max t.bits e.bits * Real.sqrt ((∑ x, f x ^ 2) * (∑ y, g y ^ 2) * (p : ℝ) ^ Fintype.card ι) := by
          gcongr
          · exact h₁.trans (mul_le_mul p₁ s₁ (Real.sqrt_nonneg _) (by positivity))
          · exact h₂.trans (mul_le_mul p₂ s₂ (Real.sqrt_nonneg _) (by positivity))
      _ = _ := by ring

theorem trace_sketch {r m : ℕ} (E B : Fin r × Fin m → ZMod p) : ∑ i, sketch E B i i = ip E B := by
  simp only [sketch, ip, Fintype.sum_prod_type]

end Accounting.BoundSketch

namespace Accounting.SecurityProofs

open Finset Accounting.BoundSketch

theorem InnerProductRelay : Accounting.Guarantees.InnerProductRelay := by
  intro p _ ι _ _ μ ν κE κR π hμ hν hμ1 hν1 hκE hκR
  have e : ∑ x, ∑ y, μ x * ν y * (1 / p : ℝ) = 1 / p :=
    calc ∑ x, ∑ y, μ x * ν y * (1 / p : ℝ) = ∑ x, μ x * ((∑ y, ν y) * (1 / p)) :=
          sum_congr rfl fun x _ => by
            rw [sum_mul, mul_sum]
            exact sum_congr rfl fun y _ => by ring
      _ = (∑ x, μ x) * ((∑ y, ν y) * (1 / p)) := (sum_mul univ μ _).symm
      _ = 1 / p := by rw [hμ1, hν1]; ring
  have hsplit : ∑ x, ∑ y, μ x * ν y * (if π.eval x y = ip x y then (1 : ℝ) else 0) =
      ∑ x, ∑ y, μ x * ν y * ((if π.eval x y = ip x y then (1 : ℝ) else 0) - 1 / p) +
        ∑ x, ∑ y, μ x * ν y * (1 / p : ℝ) := by
    rw [← sum_add_distrib]
    refine sum_congr rfl fun x _ => ?_
    rw [← sum_add_distrib]
    exact sum_congr rfl fun y _ => by ring
  have hμ2 : ∑ x, μ x ^ 2 ≤ κE :=
    calc ∑ x, μ x ^ 2 ≤ ∑ x, κE * μ x :=
          sum_le_sum fun x _ => by rw [sq]; exact mul_le_mul_of_nonneg_right (hκE x) (hμ x)
      _ = κE := by rw [← mul_sum, hμ1, mul_one]
  have hν2 : ∑ y, ν y ^ 2 ≤ κR :=
    calc ∑ y, ν y ^ 2 ≤ ∑ y, κR * ν y :=
          sum_le_sum fun y _ => by rw [sq]; exact mul_le_mul_of_nonneg_right (hκR y) (hν y)
      _ = κR := by rw [← mul_sum, hν1, mul_one]
  have hκE0 : 0 ≤ κE := (hμ 0).trans (hκE 0)
  have hpn : 0 ≤ (p : ℝ) ^ Fintype.card ι := by positivity
  rw [hsplit, e, add_comm]
  refine add_le_add le_rfl ?_
  refine (le_abs_self _).trans ((adv_le π μ ν).trans ?_)
  refine mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt ?_) (by positivity)
  exact mul_le_mul_of_nonneg_right (mul_le_mul hμ2 hν2 (sum_nonneg fun _ _ => sq_nonneg _) hκE0) hpn

theorem BoundSketchRelay : Accounting.Guarantees.BoundSketchRelay := by
  intro p _ r m μ ν κE κR π hμ hν hμ1 hν1 hκE hκR
  have h := InnerProductRelay p (Fin r × Fin m) μ ν κE κR (π.map fun v => ∑ i, v i i) hμ hν hμ1 hν1 hκE hκR
  rw [Protocol.bits_map, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin] at h
  refine le_trans (sum_le_sum fun E _ => sum_le_sum fun B _ => ?_) h
  refine mul_le_mul_of_nonneg_left ?_ (mul_nonneg (hμ E) (hν B))
  rw [Protocol.eval_map, ← trace_sketch]
  by_cases h₁ : π.eval E B = sketch E B
  · simp [h₁]
  · have h₀ : (if π.eval E B = sketch E B then (1 : ℝ) else 0) = 0 := by simp [h₁]
    rw [h₀]
    split_ifs <;> norm_num

end Accounting.SecurityProofs
