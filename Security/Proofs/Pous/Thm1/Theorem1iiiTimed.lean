import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Fintype.Powerset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Definitions.Pous
import Definitions.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers
import Proofs.Pous.Thm1.Theorem1iTimed
import Proofs.Pous.Thm1.Theorem1iiSeq

/-!
# Report Theorem 1(iii), timed: `PousProofs.theorem1_iii_timed`, as drafted

The sampling compressor (report §3, proof of (iii)), with the seed fixed inside each deterministic pair. At a
setup, `f_j(i) = Pr_I[the answer at position j is c_i when i_j := i]`, and `L_j = {i : f_j(i) ≥ η}`.
Position `j` is *bad* if `|L_j| > p_η·B`, where `p_η = (1 − (β − S − B)/(B·ℓ))·ℓ/(ℓ − log₂ R)`.

1. With no bad position, `Pr_I[pass] ≤ Pr[∀ j, i_j ∈ L_j] + Σ_j Pr[pass ∧ i_j ∉ L_j] ≤ p_η^k + k·η`
   (`pr_joint_le`, which uses `pr_coord_and_le`: coordinate `j` is independent of the others).
2. For position `j` and a seed `s` of `R` challenge vectors, the compressor sends the state and, for each
   block, either a sample `r` whose answer at `j` (with `i_j := i`) is `c_i`, or the raw block. Messages with
   `h` sampled blocks cost `hitCost = S + B + h·log₂ R + (B − h)·ℓ`, and at most `2^β` of them fit
   (`card_hitFits_le`). The expander reruns the responder once, a `(D, Q)`-bounded program, so `TimedINC`
   gives `Pr_setup[fits] ≤ π` for every seed (`pr_hitFits_le`). If `j` is bad and the seed misses no block of
   `L_j`, the message fits (`hitCost_le`). A uniform seed misses some block of `L_j` with probability at most
   `B·(1 − η)^R ≤ 2^-λ` (`pr_miss_le`, `miss_bound`). Averaging over the seed gives
   `Pr[j bad] ≤ π + 2^-λ` (`pr_prod_snd_le`, `pr_prod_fst_le`).
3. A union bound over the `k` positions and conditioning on the setup give the drafted bound.

The seed is never sent. The bound `π` holds for each fixed seed's deterministic pair, and the seed is averaged
only in the analysis, so the report's `λ` seed bits drop out of `p_η`.
-/

namespace PousThm1

open Pous
open scoped ENNReal

/-! ### Uniform probability -/

theorem pr_false' {α : Type} [Fintype α] : pr (fun _ : α => False) = 0 := by
  classical
  rw [pr_def]
  simp

theorem pr_or_le {α : Type} [Fintype α] (P Q : α → Prop) : pr (fun a => P a ∨ Q a) ≤ pr P + pr Q := by
  classical
  simp only [pr_def]
  rw [← ENNReal.add_div]
  gcongr
  rw [Finset.filter_or]
  exact_mod_cast Finset.card_union_le _ _

theorem pr_add_pr_not {α : Type} [Fintype α] [Nonempty α] (P : α → Prop) :
    pr P + pr (fun a => ¬ P a) = 1 := by
  classical
  rw [pr_def, pr_def, ← ENNReal.add_div, ← Nat.cast_add, Finset.card_filter_add_card_filter_not,
    Finset.card_univ]
  exact ENNReal.div_self (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)

/-- `R` independent uniform samples all satisfy `P`: `Pr[P]^R`. -/
theorem pr_pi_all {α : Type} [Fintype α] (R : ℕ) (P : α → Prop) :
    pr (fun s : Fin R → α => ∀ r, P (s r)) = pr P ^ R := by
  classical
  rw [pr_def, pr_def]
  have : (Finset.univ.filter fun s : Fin R → α => ∀ r, P (s r)) =
      Fintype.piFinset fun _ => Finset.univ.filter P := by
    ext s; simp [Fintype.mem_piFinset]
  rw [this, Fintype.card_piFinset, Finset.prod_const, Finset.card_univ, Fintype.card_fin, Fintype.card_fun,
    Fintype.card_fin, div_eq_mul_inv, div_eq_mul_inv, mul_pow, Nat.cast_pow, Nat.cast_pow, ENNReal.inv_pow]

/-- If every fibre over the first coordinate has probability at most `c`, so does the product event. -/
theorem pr_prod_fst_le {α β : Type} [Fintype α] [Fintype β] [Nonempty α] [Nonempty β]
    (P : α → β → Prop) (c : ℝ≥0∞) (h : ∀ a, pr (P a) ≤ c) : pr (fun x : α × β => P x.1 x.2) ≤ c := by
  have := pr_prod_le P (fun _ => False) c fun a _ => h a
  rwa [pr_false', zero_add] at this

/-- If every fibre over the second coordinate has probability at most `c`, so does the product event. -/
theorem pr_prod_snd_le {α β : Type} [Fintype α] [Fintype β] [Nonempty α] [Nonempty β]
    (P : α → β → Prop) (c : ℝ≥0∞) (h : ∀ b, pr (fun a => P a b) ≤ c) :
    pr (fun x : α × β => P x.1 x.2) ≤ c := by
  rw [← pr_equiv (Equiv.prodComm β α) (fun x : α × β => P x.1 x.2)]
  exact pr_prod_fst_le (fun b a => P a b) c h

/-- **Independence of one coordinate.** A uniform vector has `I j = i`, and satisfies an event of its other
coordinates, with probability at most `1/B` times that event's. -/
theorem pr_coord_and_le {ι : Type} [Fintype ι] [DecidableEq ι] {B : ℕ} [NeZero B] (j : ι) (i : Fin B)
    (Ψ : (ι → Fin B) → Prop) :
    pr (fun I : ι → Fin B => I j = i ∧ Ψ (Function.update I j i)) ≤
      (B : ℝ≥0∞)⁻¹ * pr (fun I : ι → Fin B => Ψ (Function.update I j i)) := by
  classical
  let e := Equiv.funSplitAt j (Fin B)
  have key : ∀ I : ι → Fin B, Function.update I j i = e.symm (i, (e I).2) := by
    intro I; funext x
    by_cases hx : x = j
    · subst hx; simp [e]
    · simp [e, hx]
  have h1 : pr (fun I : ι → Fin B => I j = i ∧ Ψ (Function.update I j i)) =
      pr (fun y : Fin B × ({x // x ≠ j} → Fin B) => y.1 = i ∧ Ψ (e.symm (i, y.2))) := by
    rw [← pr_equiv e]
    refine congrArg pr (funext fun I => ?_)
    rw [key]
    simp [e]
  have h2 : pr (fun I : ι → Fin B => Ψ (Function.update I j i)) =
      pr (fun z : {x // x ≠ j} → Fin B => Ψ (e.symm (i, z))) := by
    calc pr (fun I : ι → Fin B => Ψ (Function.update I j i))
        = pr (fun I : ι → Fin B => (fun y : Fin B × ({x // x ≠ j} → Fin B) => Ψ (e.symm (i, y.2))) (e I)) :=
          congrArg pr (funext fun I => by rw [key])
      _ = pr (fun y : Fin B × ({x // x ≠ j} → Fin B) => Ψ (e.symm (i, y.2))) :=
          pr_equiv e (fun y : Fin B × ({x // x ≠ j} → Fin B) => Ψ (e.symm (i, y.2)))
      _ = pr (fun z : {x // x ≠ j} → Fin B => Ψ (e.symm (i, z))) :=
          pr_snd (α := Fin B) (fun z : {x // x ≠ j} → Fin B => Ψ (e.symm (i, z)))
  rw [h1, h2]
  refine pr_and_le (α := Fin B) (β := {x // x ≠ j} → Fin B) (fun b => b = i)
    (fun _ z => Ψ (e.symm (i, z))) (B : ℝ≥0∞)⁻¹ (pr fun z : {x // x ≠ j} → Fin B => Ψ (e.symm (i, z))) ?_
    fun _ => le_rfl
  rw [pr_def, Fintype.card_fin]
  simp [Finset.filter_eq']

/-! ### Programs -/

section progMap

variable {M : IdealModel} {α β : Type} (g : α → β) (f : M.Qry → M.Ans)

end progMap

/-! ### The joint responder at a fixed setup -/

/-- **Joint responder, fixed setup.** `ans I` answers the challenge vector `I`; `L j` holds the blocks `i` at
position `j` answered correctly with probability above `η` when `i_j = i`. If every `|L j| ≤ p·B`, the audit
passes with probability at most `p^k + k·η`. -/
theorem pr_joint_le {k B ℓ : ℕ} [NeZero B] (ans : (Fin k → Fin B) → Fin k → Bits ℓ) (C : Fin B → Bits ℓ)
    (η p : ℝ≥0∞) (L : Fin k → Finset (Fin B))
    (hL : ∀ j, ∀ i ∉ L j, pr (fun I => ans (Function.update I j i) j = C i) ≤ η)
    (hp : ∀ j, ((L j).card : ℝ≥0∞) ≤ p * B) :
    pr (fun I => ans I = fun j => C (I j)) ≤ p ^ k + k * η := by
  classical
  have hB0 : (B : ℝ≥0∞) ≠ 0 := by exact_mod_cast NeZero.ne B
  have hBt : (B : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  calc pr (fun I => ans I = fun j => C (I j))
      ≤ pr (fun I : Fin k → Fin B => (∀ j, I j ∈ L j) ∨ ∃ j ∈ Finset.univ, ∃ i ∈ (L j)ᶜ,
          I j = i ∧ ans (Function.update I j i) j = C i) := by
        refine pr_mono fun I hI => ?_
        by_cases hall : ∀ j, I j ∈ L j
        · exact Or.inl hall
        · push Not at hall
          obtain ⟨j, hj⟩ := hall
          refine Or.inr ⟨j, Finset.mem_univ _, I j, Finset.mem_compl.2 hj, rfl, ?_⟩
          rw [Function.update_eq_self]
          exact congrFun hI j
    _ ≤ pr (fun I : Fin k → Fin B => ∀ j, I j ∈ L j) + ∑ j, ∑ i ∈ (L j)ᶜ,
          pr (fun I : Fin k → Fin B => I j = i ∧ ans (Function.update I j i) j = C i) := by
        refine (pr_or_le _ _).trans (add_le_add le_rfl ?_)
        exact (pr_exists_le _ _).trans (Finset.sum_le_sum fun j _ => pr_exists_le _ _)
    _ ≤ p ^ k + ∑ _j : Fin k, η := by
        refine add_le_add ?_ (Finset.sum_le_sum fun j _ => ?_)
        · rw [pr_def]
          have : (Finset.univ.filter fun I : Fin k → Fin B => ∀ j, I j ∈ L j) = Fintype.piFinset L := by
            ext I; simp [Fintype.mem_piFinset]
          rw [this, Fintype.card_piFinset, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin, Nat.cast_prod,
            Nat.cast_pow, ENNReal.div_le_iff (pow_ne_zero _ hB0) (ENNReal.pow_ne_top hBt), ← mul_pow]
          calc ∏ j, ((L j).card : ℝ≥0∞) ≤ ∏ _j : Fin k, p * B := Finset.prod_le_prod fun j _ => hp j
            _ = (p * B) ^ k := by rw [Finset.prod_const, Finset.card_univ, Fintype.card_fin]
        · calc ∑ i ∈ (L j)ᶜ, pr (fun I : Fin k → Fin B => I j = i ∧ ans (Function.update I j i) j = C i)
              ≤ ∑ i ∈ (L j)ᶜ, (B : ℝ≥0∞)⁻¹ * η := by
                refine Finset.sum_le_sum fun i hi => (pr_coord_and_le j i fun I' => ans I' j = C i).trans ?_
                exact mul_le_mul' le_rfl (hL j i (Finset.mem_compl.1 hi))
            _ ≤ ∑ _i : Fin B, (B : ℝ≥0∞)⁻¹ * η := Finset.sum_le_sum_of_subset (Finset.subset_univ _)
            _ = η := by
                rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, ← mul_assoc,
                  ENNReal.mul_inv_cancel hB0 hBt, one_mul]
    _ = p ^ k + k * η := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-! ### Missing a likely block -/

/-- `R` uniform samples miss some block of `L`, where each block of `L` is hit by a sample with probability at
least `η`: at most `B·(1 − η)^R`. -/
theorem pr_miss_le {X : Type} [Fintype X] [Nonempty X] {B : ℕ} (R : ℕ) (hit : X → Fin B → Prop) (η : ℝ≥0∞)
    (L : Finset (Fin B)) (hL : ∀ i ∈ L, η ≤ pr (fun x => hit x i)) :
    pr (fun s : Fin R → X => ∃ i ∈ L, ∀ r, ¬ hit (s r) i) ≤ B * (1 - η) ^ R := by
  refine (pr_exists_le _ _).trans ?_
  calc ∑ i ∈ L, pr (fun s : Fin R → X => ∀ r, ¬ hit (s r) i)
      ≤ ∑ _i ∈ L, (1 - η) ^ R := by
        refine Finset.sum_le_sum fun i hi => ?_
        rw [pr_pi_all R (fun x => ¬ hit x i)]
        gcongr
        have h1 := pr_add_pr_not (fun x => hit x i)
        have hne : pr (fun x => hit x i) ≠ ⊤ := ne_top_of_le_ne_top ENNReal.one_ne_top (pr_le_one _)
        calc pr (fun x => ¬ hit x i) = 1 - pr (fun x => hit x i) := by
              rw [← h1, ENNReal.add_sub_cancel_left hne]
          _ ≤ 1 - η := tsub_le_tsub_left (hL i hi) _
    _ ≤ ∑ _i : Fin B, (1 - η) ^ R := Finset.sum_le_sum_of_subset (Finset.subset_univ _)
    _ = B * (1 - η) ^ R := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- The sample count `R = ⌈(λ + log₂ B)·ln 2 / η⌉` makes `B·(1 − η)^R ≤ 2^-λ`. -/
theorem miss_bound (B lam R : ℕ) (hB : 0 < B) (η : ℝ) (hη0 : 0 < η) (hη1 : η < 1)
    (hR : R = ⌈((lam : ℝ) + Real.logb 2 B) * Real.log 2 / η⌉₊) :
    (B : ℝ≥0∞) * (1 - ENNReal.ofReal η) ^ R ≤ (2⁻¹ : ℝ≥0∞) ^ lam := by
  have hB' : (0 : ℝ) < B := by exact_mod_cast hB
  have hreal : (B : ℝ) * (1 - η) ^ R ≤ (2⁻¹ : ℝ) ^ lam := by
    have hRge : ((lam : ℝ) + Real.logb 2 B) * Real.log 2 / η ≤ R := hR ▸ Nat.le_ceil _
    have hηR : ((lam : ℝ) + Real.logb 2 B) * Real.log 2 ≤ η * R := by
      rw [div_le_iff₀ hη0] at hRge
      linarith
    have hlog : Real.logb 2 B * Real.log 2 = Real.log B := by
      rw [Real.logb, div_mul_cancel₀ _ (Real.log_pos one_lt_two).ne']
    have h1 : (1 - η) ^ R ≤ Real.exp (-(η * R)) := by
      rw [← Real.rpow_natCast, ← Real.exp_log (by linarith : (0 : ℝ) < 1 - η), ← Real.exp_mul]
      refine Real.exp_le_exp.2 ?_
      have := Real.log_le_sub_one_of_pos (by linarith : (0 : ℝ) < 1 - η)
      nlinarith [(Nat.cast_nonneg R : (0 : ℝ) ≤ R)]
    have h2 : Real.exp (-(η * R)) ≤ Real.exp (-(lam * Real.log 2 + Real.log B)) := by
      refine Real.exp_le_exp.2 ?_
      nlinarith [hηR, hlog]
    have h3 : Real.exp (-(lam * Real.log 2 + Real.log B)) = (2⁻¹ : ℝ) ^ lam / B := by
      rw [Real.exp_neg, Real.exp_add, Real.exp_log hB', Real.exp_nat_mul, Real.exp_log two_pos, mul_inv,
        inv_pow, div_eq_mul_inv]
    calc (B : ℝ) * (1 - η) ^ R ≤ B * ((2⁻¹ : ℝ) ^ lam / B) := by
          gcongr; exact h1.trans (h2.trans h3.le)
      _ = (2⁻¹ : ℝ) ^ lam := by field_simp
  have hsub : (1 : ℝ≥0∞) - ENNReal.ofReal η = ENNReal.ofReal (1 - η) := by
    rw [ENNReal.ofReal_sub _ hη0.le, ENNReal.ofReal_one]
  calc (B : ℝ≥0∞) * (1 - ENNReal.ofReal η) ^ R = ENNReal.ofReal ((B : ℝ) * (1 - η) ^ R) := by
        rw [hsub, ENNReal.ofReal_mul (Nat.cast_nonneg _), ENNReal.ofReal_natCast,
          ENNReal.ofReal_pow (by linarith)]
    _ ≤ ENNReal.ofReal ((2⁻¹ : ℝ) ^ lam) := ENNReal.ofReal_le_ofReal hreal
    _ = (2⁻¹ : ℝ≥0∞) ^ lam := by
        rw [ENNReal.ofReal_pow (by norm_num), ENNReal.ofReal_inv_of_pos two_pos, ENNReal.ofReal_ofNat]

/-! ### The sampling compressor -/

/-- The sampling compressor's message: the state, and for each block either the index of a sample that rebuilds it
or the raw block. -/
abbrev HitDesc (S B ℓ R : ℕ) : Type := Bits S × (Fin B → Fin R ⊕ Bits ℓ)

/-- The report's length for `h` rebuilt blocks: `S + B + h·log₂ R + (B − h)·ℓ` bits. -/
noncomputable def hitCost (S B ℓ R h : ℕ) : ℝ := S + B + h * Real.logb 2 R + ((B : ℝ) - h) * ℓ

/-- The messages whose length fits in `β` bits. -/
noncomputable def hitFits (S B ℓ R β : ℕ) : Finset (HitDesc S B ℓ R) :=
  Finset.univ.filter fun x => hitCost S B ℓ R (Finset.univ.filter fun i => (x.2 i).isLeft).card ≤ β

/-- `R^h · 2^(ℓ(B − h)) ≤ 2^(β − S − B)` when `h` rebuilt blocks fit. -/
theorem hit_count_le (S B ℓ R β h : ℕ) (hh : h ≤ B) (hc : hitCost S B ℓ R h ≤ β) :
    ((R ^ h * (2 ^ ℓ) ^ (B - h) : ℕ) : ℝ) ≤ (2 : ℝ) ^ ((β : ℝ) - S - B) := by
  have hRh : ((R : ℝ) ^ h) ≤ (2 : ℝ) ^ ((h : ℝ) * Real.logb 2 R) := by
    rcases Nat.eq_zero_or_pos R with rfl | hR
    · rcases Nat.eq_zero_or_pos h with rfl | hh0
      · simp
      · rw [Nat.cast_zero, zero_pow hh0.ne']; positivity
    · rw [mul_comm, Real.rpow_mul (by norm_num), Real.rpow_logb (by norm_num) (by norm_num)
        (by exact_mod_cast hR), Real.rpow_natCast]
  have h2 : ((2 : ℝ) ^ ℓ) ^ (B - h) = (2 : ℝ) ^ (((B : ℝ) - h) * ℓ) := by
    rw [← pow_mul, ← Real.rpow_natCast, Nat.cast_mul, Nat.cast_sub hh, mul_comm]
  push_cast
  rw [h2]
  unfold hitCost at hc
  calc (R : ℝ) ^ h * 2 ^ (((B : ℝ) - h) * ℓ)
      ≤ 2 ^ ((h : ℝ) * Real.logb 2 R) * 2 ^ (((B : ℝ) - h) * ℓ) := by gcongr
    _ = 2 ^ ((h : ℝ) * Real.logb 2 R + ((B : ℝ) - h) * ℓ) := (Real.rpow_add two_pos _ _).symm
    _ ≤ 2 ^ ((β : ℝ) - S - B) := Real.rpow_le_rpow_of_exponent_le one_le_two (by linarith)

/-- **Counting sampling messages.** At most `2^β` messages fit. -/
theorem card_hitFits_le (S B ℓ R β : ℕ) : (hitFits S B ℓ R β).card ≤ 2 ^ β := by
  classical
  let G : Finset (Fin B → Fin R ⊕ Bits ℓ) := Finset.univ.filter fun g =>
    hitCost S B ℓ R (Finset.univ.filter fun i => (g i).isLeft).card ≤ β
  let t : Finset (Fin B) → (i : Fin B) → Finset (Fin R ⊕ Bits ℓ) := fun H i =>
    if i ∈ H then Finset.univ.map Function.Embedding.inl else Finset.univ.map Function.Embedding.inr
  have hprod : hitFits S B ℓ R β = Finset.univ ×ˢ G := by
    ext x; simp [hitFits, G]
  have hsub : G ⊆ (Finset.univ.filter fun H : Finset (Fin B) => hitCost S B ℓ R H.card ≤ β).biUnion
      fun H => Fintype.piFinset (t H) := by
    intro g hg
    rw [Finset.mem_filter] at hg
    rw [Finset.mem_biUnion]
    refine ⟨Finset.univ.filter fun i => (g i).isLeft, by simpa using hg.2, ?_⟩
    rw [Fintype.mem_piFinset]
    intro i
    cases hgi : g i <;> simp [t, hgi]
  have hcardH : ∀ H : Finset (Fin B), (Fintype.piFinset (t H)).card = R ^ H.card * (2 ^ ℓ) ^ (B - H.card) := by
    intro H
    rw [Fintype.card_piFinset]
    have : ∀ i, (t H i).card = if i ∈ H then R else 2 ^ ℓ := by
      intro i
      by_cases hi : i ∈ H <;> simp [t, hi]
    simp_rw [this]
    rw [Finset.prod_ite, Finset.prod_const, Finset.prod_const]
    have h1 : (Finset.univ.filter fun i => i ∈ H).card = H.card := by simp
    have h2 := Finset.card_filter_add_card_filter_not (s := Finset.univ) (fun i => i ∈ H)
    rw [Finset.card_univ, Fintype.card_fin, h1] at h2
    rw [h1, show (Finset.univ.filter fun i => i ∉ H).card = B - H.card by omega]
  have hG : (G.card : ℝ) ≤ 2 ^ B * (2 : ℝ) ^ ((β : ℝ) - S - B) := by
    calc (G.card : ℝ)
        ≤ (((Finset.univ.filter fun H : Finset (Fin B) => hitCost S B ℓ R H.card ≤ β).biUnion
            fun H => Fintype.piFinset (t H)).card : ℝ) := by exact_mod_cast Finset.card_le_card hsub
      _ ≤ ∑ H ∈ Finset.univ.filter fun H : Finset (Fin B) => hitCost S B ℓ R H.card ≤ β,
            ((Fintype.piFinset (t H)).card : ℝ) := by exact_mod_cast Finset.card_biUnion_le
      _ ≤ ∑ _H ∈ Finset.univ.filter fun H : Finset (Fin B) => hitCost S B ℓ R H.card ≤ β,
            (2 : ℝ) ^ ((β : ℝ) - S - B) := by
          refine Finset.sum_le_sum fun H hH => ?_
          rw [hcardH]
          exact hit_count_le S B ℓ R β H.card (by simpa using H.card_le_univ) (Finset.mem_filter.1 hH).2
      _ ≤ 2 ^ B * (2 : ℝ) ^ ((β : ℝ) - S - B) := by
          rw [Finset.sum_const, nsmul_eq_mul]
          gcongr
          calc (((Finset.univ.filter fun H : Finset (Fin B) => hitCost S B ℓ R H.card ≤ β).card : ℕ) : ℝ)
              ≤ ((Finset.univ : Finset (Finset (Fin B))).card : ℝ) := by
                exact_mod_cast Finset.card_filter_le _ _
            _ = 2 ^ B := by rw [Finset.card_univ, Fintype.card_finset, Fintype.card_fin]; push_cast; rfl
  have : ((hitFits S B ℓ R β).card : ℝ) ≤ 2 ^ β := by
    rw [hprod, Finset.card_product, Finset.card_univ, Fintype.card_fun, Fintype.card_bool, Fintype.card_fin]
    push_cast
    calc (2 : ℝ) ^ S * G.card ≤ 2 ^ S * (2 ^ B * (2 : ℝ) ^ ((β : ℝ) - S - B)) := by gcongr
      _ = 2 ^ β := by
          rw [← Real.rpow_natCast 2 S, ← Real.rpow_natCast 2 B, ← Real.rpow_add two_pos,
            ← Real.rpow_add two_pos, ← Real.rpow_natCast]
          congr 1; ring
  exact_mod_cast this

open Classical in
/-- **The timed sampling compressor.** Let `st` pick an `S`-bit state from the setup and `pred σ r i` rebuild block
`i` from sample `r` within `(D, Q)`. Then the rebuilt blocks fit, `hitCost ≤ β`, with probability at most `π`
under `TimedINC D Q β π`. -/
theorem pr_hitFits_le {M : IdealModel} (sch : Scheme M) (D Q β S R : ℕ) (π : ℝ≥0∞)
    (hinc : TimedINC sch D Q β π) (W : Bits sch.n) (st : Bits sch.n → Bits sch.t → sch.Code → M.Ω → Bits S)
    (pred : Bits sch.n → Bits sch.t → Bits S → Fin R → Fin sch.B → Prog M (Bits sch.ℓ))
    (hpred : ∀ W pp σ r i, (pred W pp σ r i).Bounded D Q) :
    pr (fun a : M.Ω × sch.Coins => hitCost S sch.B sch.ℓ R (Finset.univ.filter fun i => ∃ r,
      (pred W (sch.enc W a.1 a.2).1 (st W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) r i).run
        (M.answer a.1) = (sch.enc W a.1 a.2).2 i).card ≤ β) ≤ π := by
  obtain ⟨c, d, hcd⟩ := exists_code (hitFits S sch.B sch.ℓ R β) β (card_hitFits_le S sch.B sch.ℓ R β)
  let msg : Bits sch.n → Bits sch.t → sch.Code → M.Ω → Fin sch.B → Fin R ⊕ Bits sch.ℓ := fun W pp C ω i =>
    if h : ∃ r, (pred W pp (st W pp C ω) r i).run (M.answer ω) = C i then Sum.inl h.choose else Sum.inr (C i)
  let cmp : Bits sch.n → Bits sch.t → sch.Code → M.Ω → Bits β :=
    fun W pp C ω => c (st W pp C ω, msg W pp C ω)
  let exp : Bits sch.n → Bits sch.t → Bits β → Fin sch.B → Prog M (Bits sch.ℓ) :=
    fun W pp w i => ((d w).2 i).elim (fun r => pred W pp (d w).1 r i) Prog.ret
  have hexp : ∀ W pp w i, (exp W pp w i).Bounded D Q := by
    intro W pp w i
    cases h : (d w).2 i with
    | inl r => simp only [exp, h, Sum.elim_inl]; exact hpred _ _ _ _ _
    | inr b => simp only [exp, h, Sum.elim_inr]; exact ret_bounded _ _ _
  refine le_trans (pr_mono fun a ha => ?_) (hinc W cmp exp hexp)
  have hfit : (st W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1,
      msg W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) ∈ hitFits S sch.B sch.ℓ R β := by
    rw [hitFits, Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    convert ha using 4
    rename_i i
    simp only [msg]
    split_ifs with h <;> simp [h]
  show ∀ i, (exp W _ (cmp W _ _ a.1) i).run (M.answer a.1) = _
  intro i
  simp only [exp, cmp, hcd _ hfit, msg]
  split_ifs with h
  · exact h.choose_spec
  · rfl

/-- Many likely blocks, all rebuilt, fit: if `m > p_η·B` and `m ≤ h`, then `hitCost h ≤ β`. -/
theorem hitCost_le (S B ℓ R β h m : ℕ) (hB : 0 < B) (hRℓ : Real.logb 2 R < ℓ) (hm : m ≤ h)
    (hbig : ¬ (m : ℝ) ≤ (1 - ((β : ℝ) - S - B) / (B * ℓ)) * ℓ / (ℓ - Real.logb 2 R) * B) :
    hitCost S B ℓ R h ≤ β := by
  have hL0 : 0 ≤ Real.logb 2 R := by
    rcases Nat.eq_zero_or_pos R with rfl | hR
    · simp
    · exact Real.logb_nonneg one_lt_two (by exact_mod_cast hR)
  have hℓ : (0 : ℝ) < ℓ := hL0.trans_lt hRℓ
  have hd : (0 : ℝ) < ℓ - Real.logb 2 R := by linarith
  have hB' : (0 : ℝ) < B := by exact_mod_cast hB
  have key : (1 - ((β : ℝ) - S - B) / (B * ℓ)) * ℓ / (ℓ - Real.logb 2 R) * B =
      (B * ℓ - β + S + B) / (ℓ - Real.logb 2 R) := by
    field_simp
    ring
  rw [key, not_le, div_lt_iff₀ hd] at hbig
  have hmh : (m : ℝ) ≤ h := by exact_mod_cast hm
  have : (B * ℓ - β + S + B : ℝ) < h * (ℓ - Real.logb 2 R) :=
    hbig.trans_le (mul_le_mul_of_nonneg_right hmh hd.le)
  unfold hitCost
  nlinarith

end PousThm1

namespace PousProofs

open Pous PousThm1
open scoped ENNReal

/-- Report Theorem 1(iii), timed, simultaneous reveal, by the sampling compressor. The compressor's seed is fixed
inside each deterministic `(cmp, exp)` pair, so it costs no bits (the report charges `λ` bits for it). -/
theorem theorem1_iii_timed {M : IdealModel} (sch : Scheme M) (D Q β S k lam R : ℕ) (π : ℝ≥0∞) (η : ℝ)
    (hη0 : 0 < η) (hη1 : η < 1) (hR : R = ⌈((lam : ℝ) + Real.logb 2 sch.B) * Real.log 2 / η⌉₊)
    (hRℓ : Real.logb 2 R < sch.ℓ) (hinc : TimedINC sch D Q β π)
    (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : SimResponder sch S D Q k) :
    pr (simAccepts sch W A₁ A₂) ≤
      k * π + ENNReal.ofReal ((1 - ((β : ℝ) - S - sch.B) / (sch.B * sch.ℓ)) * sch.ℓ / (sch.ℓ - Real.logb 2 R)) ^ k
        + k * ENNReal.ofReal η + k * (2⁻¹ : ℝ≥0∞) ^ lam := by
  classical
  set pη : ℝ := (1 - ((β : ℝ) - S - sch.B) / (sch.B * sch.ℓ)) * sch.ℓ / (sch.ℓ - Real.logb 2 R) with hpη
  have hB := sch.B_pos
  have : Nonempty (Fin sch.B) := ⟨⟨0, hB⟩⟩
  -- the responder's answers at a setup, and whether position `j` rebuilds block `i` when `i_j := i`
  let ans : M.Ω × sch.Coins → (Fin k → Fin sch.B) → Fin k → Bits sch.ℓ := fun a I =>
    (A₂.prog W (sch.enc W a.1 a.2).1 (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) I).run (M.answer a.1)
  let hit : M.Ω × sch.Coins → Fin k → (Fin k → Fin sch.B) → Fin sch.B → Prop := fun a j I i =>
    ans a (Function.update I j i) j = (sch.enc W a.1 a.2).2 i
  -- the likely blocks `L_j(η)` and the event that some position has too many of them
  let Lset : M.Ω × sch.Coins → Fin k → Finset (Fin sch.B) := fun a j =>
    Finset.univ.filter fun i => ENNReal.ofReal η ≤ pr (fun I => hit a j I i)
  let Bad : Fin k → M.Ω × sch.Coins → Prop := fun j a => ¬ ((Lset a j).card : ℝ) ≤ pη * sch.B
  have hre : pr (simAccepts sch W A₁ A₂) = pr (fun x : (M.Ω × sch.Coins) × (Fin k → Fin sch.B) =>
      ans x.1 x.2 = fun j => (sch.enc W x.1.1 x.1.2).2 (x.2 j)) :=
    pr_equiv (Equiv.prodAssoc _ _ _).symm (fun x : (M.Ω × sch.Coins) × (Fin k → Fin sch.B) =>
      ans x.1 x.2 = fun j => (sch.enc W x.1.1 x.1.2).2 (x.2 j))
  rw [hre]
  -- a setup with no bad position passes with probability at most `p_η^k + k·η`
  have hA : ∀ a, ¬ (∃ j ∈ Finset.univ, Bad j a) →
      pr (fun I => ans a I = fun j => (sch.enc W a.1 a.2).2 (I j)) ≤
        ENNReal.ofReal pη ^ k + k * ENNReal.ofReal η := by
    intro a ha
    refine pr_joint_le (ans a) _ (ENNReal.ofReal η) (ENNReal.ofReal pη) (Lset a) (fun j i hi => ?_) fun j => ?_
    · simp only [Lset, Finset.mem_filter, Finset.mem_univ, true_and, not_le] at hi
      exact hi.le
    · have hj : ((Lset a j).card : ℝ) ≤ pη * sch.B := by
        by_contra hc; exact ha ⟨j, Finset.mem_univ _, hc⟩
      calc ((Lset a j).card : ℝ≥0∞) = ENNReal.ofReal ((Lset a j).card : ℝ) := (ENNReal.ofReal_natCast _).symm
        _ ≤ ENNReal.ofReal (pη * sch.B) := ENNReal.ofReal_le_ofReal hj
        _ = ENNReal.ofReal pη * sch.B := by
            rw [ENNReal.ofReal_mul' (Nat.cast_nonneg _), ENNReal.ofReal_natCast]
  -- a position is bad with probability at most `π + 2^-λ`: average the seeded compressors over the seed
  have hBad : ∀ j, pr (Bad j) ≤ π + (2⁻¹ : ℝ≥0∞) ^ lam := by
    intro j
    let pred : (Fin R → Fin k → Fin sch.B) → Bits sch.n → Bits sch.t → Bits S → Fin R → Fin sch.B →
        Prog M (Bits sch.ℓ) :=
      fun s W pp σ r i => progMap (fun out => out j) (A₂.prog W pp σ (Function.update (s r) j i))
    calc pr (Bad j) = pr (fun x : (M.Ω × sch.Coins) × (Fin R → Fin k → Fin sch.B) => Bad j x.1) :=
          (pr_fst _).symm
      _ ≤ pr (fun x : (M.Ω × sch.Coins) × (Fin R → Fin k → Fin sch.B) =>
            hitCost S sch.B sch.ℓ R (Finset.univ.filter fun i => ∃ r, hit x.1 j (x.2 r) i).card ≤ β ∨
            ∃ i ∈ Lset x.1 j, ∀ r, ¬ hit x.1 j (x.2 r) i) := by
          refine pr_mono fun x hx => ?_
          by_cases hm : ∃ i ∈ Lset x.1 j, ∀ r, ¬ hit x.1 j (x.2 r) i
          · exact Or.inr hm
          · refine Or.inl (hitCost_le S sch.B sch.ℓ R β _ (Lset x.1 j).card hB hRℓ ?_ hx)
            refine Finset.card_le_card fun i hi => ?_
            rw [Finset.mem_filter]
            refine ⟨Finset.mem_univ _, ?_⟩
            by_contra hno
            push Not at hno
            exact hm ⟨i, hi, hno⟩
      _ ≤ pr (fun x : (M.Ω × sch.Coins) × (Fin R → Fin k → Fin sch.B) =>
            hitCost S sch.B sch.ℓ R (Finset.univ.filter fun i => ∃ r, hit x.1 j (x.2 r) i).card ≤ β) +
          pr (fun x : (M.Ω × sch.Coins) × (Fin R → Fin k → Fin sch.B) =>
            ∃ i ∈ Lset x.1 j, ∀ r, ¬ hit x.1 j (x.2 r) i) := pr_or_le _ _
      _ ≤ π + (2⁻¹ : ℝ≥0∞) ^ lam := by
          refine add_le_add (pr_prod_snd_le (α := M.Ω × sch.Coins) (β := Fin R → Fin k → Fin sch.B)
            (fun a s => hitCost S sch.B sch.ℓ R (Finset.univ.filter fun i => ∃ r, hit a j (s r) i).card ≤ β)
            π fun s => ?_) (pr_prod_fst_le (α := M.Ω × sch.Coins) (β := Fin R → Fin k → Fin sch.B)
            (fun a s => ∃ i ∈ Lset a j, ∀ r, ¬ hit a j (s r) i) _ fun a => ?_)
          · have := pr_hitFits_le sch D Q β S R π hinc W A₁ (pred s)
              fun W pp σ r i => bounded_progMap _ (A₂.bounded _ _ _ _)
            simpa only [pred, run_progMap] using this
          · refine (pr_miss_le R (fun I i => hit a j I i) (ENNReal.ofReal η) (Lset a j)
              fun i hi => (Finset.mem_filter.1 hi).2).trans ?_
            exact miss_bound sch.B lam R hB η hη0 hη1 hR
  calc pr (fun x : (M.Ω × sch.Coins) × (Fin k → Fin sch.B) =>
        ans x.1 x.2 = fun j => (sch.enc W x.1.1 x.1.2).2 (x.2 j))
      ≤ pr (fun a => ∃ j ∈ Finset.univ, Bad j a) + (ENNReal.ofReal pη ^ k + k * ENNReal.ofReal η) :=
        pr_prod_le (α := M.Ω × sch.Coins) (β := Fin k → Fin sch.B)
          (fun a I => ans a I = fun j => (sch.enc W a.1 a.2).2 (I j)) (fun a => ∃ j ∈ Finset.univ, Bad j a)
          (ENNReal.ofReal pη ^ k + k * ENNReal.ofReal η) hA
    _ ≤ ∑ _j : Fin k, (π + (2⁻¹ : ℝ≥0∞) ^ lam) + (ENNReal.ofReal pη ^ k + k * ENNReal.ofReal η) :=
        add_le_add ((pr_exists_le Finset.univ Bad).trans (Finset.sum_le_sum fun j _ => hBad j)) le_rfl
    _ = k * π + ENNReal.ofReal pη ^ k + k * ENNReal.ofReal η + k * (2⁻¹ : ℝ≥0∞) ^ lam := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        ring

end PousProofs
