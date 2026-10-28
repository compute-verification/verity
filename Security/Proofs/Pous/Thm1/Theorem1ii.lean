import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.Powerset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Definitions.Pous
import Definitions.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers

/-!
# Report Theorem 1(ii), untimed: a proof of `Pous.Guarantees.Theorem1ii`

The basic compression argument (report §3, proof of Theorem 1(ii)). Fix `W`, `A₁` and the product-form responder
`A₂`. For a setup `a = (ω, r)` with `σ = A₁(W, pp, C, ω)`, block `i` is *good* if `A₂(W, pp, σ, ω, i) = C i`, and
`E a` says the wrong blocks fit in `β` bits: `S + B + #wrong·ℓ ≤ β`.

1. Averaging (`pr_all_good_le`). Off `E`, `#good ≤ p₀·B`, so `k` uniform challenges all land on good blocks with
   probability `(#good/B)^k ≤ p₀^k`. Hence `Pr[accept] ≤ Pr[E] + p₀^k`.
2. Compression. The message `(σ, patch)`, where `patch i = some (C i)` on wrong blocks and `none` on good ones, takes
   at most `2^S·2^B·2^(β−S−B) = 2^β` values on `E` (`card_fits_le`), so it has a `β`-bit code (`exists_code`). The
   expander decodes the message and returns the patched block where there is one, else `A₂`'s answer. On `E` it
   rebuilds `C`, so `Pr[E] ≤ π` by `INC(β, π)`.
-/

namespace PousThm1ii

open Pous
open scoped ENNReal

/-! ### Uniform probability -/

/-- `pr` does not depend on how the sample space is labelled. -/
theorem pr_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (P : β → Prop) :
    pr (fun a => P (e a)) = pr P := by
  classical
  rw [pr_def, pr_def, Fintype.card_congr e]
  congr 2
  rw [← Fintype.card_subtype, ← Fintype.card_subtype]
  exact Fintype.card_congr (e.subtypeEquiv fun _ => Iff.rfl)

/-- **Conditioning on the first coordinate.** If every fibre off `E` has probability at most `c`, the product event
has probability at most `Pr[E] + c`. -/
theorem pr_prod_le {α β : Type} [Fintype α] [Fintype β] [Nonempty α] [Nonempty β]
    (P : α → β → Prop) (E : α → Prop) (c : ℝ≥0∞) (h : ∀ a, ¬ E a → pr (P a) ≤ c) :
    pr (fun x : α × β => P x.1 x.2) ≤ pr E + c := by
  classical
  have hα0 : (Fintype.card α : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hβ0 : (Fintype.card β : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hαt : (Fintype.card α : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have hβt : (Fintype.card β : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have hfib : ∀ a, ((Finset.univ.filter (P a)).card : ℝ≥0∞) ≤
      (if E a then 1 else 0) * Fintype.card β + c * Fintype.card β := by
    intro a
    by_cases ha : E a
    · simp only [ha, ite_true, one_mul]
      refine le_add_right ?_
      exact_mod_cast (Finset.card_filter_le _ _).trans (Finset.card_univ).le
    · simp only [ha, ite_false, zero_mul, zero_add]
      have := h a ha
      rwa [pr_def, ENNReal.div_le_iff hβ0 hβt] at this
  have hcount : ((Finset.univ.filter fun x : α × β => P x.1 x.2).card : ℝ≥0∞) =
      ∑ a, ((Finset.univ.filter (P a)).card : ℝ≥0∞) := by
    rw [Finset.natCast_card_filter, Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun a _ => ?_
    rw [Finset.natCast_card_filter]
  have hE : ((Finset.univ.filter E).card : ℝ≥0∞) = ∑ a, (if E a then 1 else 0 : ℝ≥0∞) :=
    Finset.natCast_card_filter _ _
  rw [pr_def, pr_def, hcount, Fintype.card_prod, Nat.cast_mul, ENNReal.div_le_iff (mul_ne_zero hα0 hβ0)
    (ENNReal.mul_ne_top hαt hβt)]
  calc ∑ a, ((Finset.univ.filter (P a)).card : ℝ≥0∞)
      ≤ ∑ a, ((if E a then 1 else 0) * Fintype.card β + c * Fintype.card β) :=
        Finset.sum_le_sum fun a _ => hfib a
    _ = ((Finset.univ.filter E).card : ℝ≥0∞) * Fintype.card β +
          Fintype.card α * (c * Fintype.card β) := by
        rw [Finset.sum_add_distrib, ← Finset.sum_mul, ← hE, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    _ = (((Finset.univ.filter E).card : ℝ≥0∞) / Fintype.card α + c) *
          (Fintype.card α * Fintype.card β) := by
        rw [add_mul, ← mul_assoc (_ / (Fintype.card α : ℝ≥0∞)), ENNReal.div_mul_cancel hα0 hαt]
        ring

/-- `k` indices, all in `G`: `|G|^k` of them. -/
theorem card_pi_mem (k B : ℕ) (G : Finset (Fin B)) :
    (Finset.univ.filter fun I : Fin k → Fin B => ∀ j, I j ∈ G).card = G.card ^ k := by
  have : (Finset.univ.filter fun I : Fin k → Fin B => ∀ j, I j ∈ G) = Fintype.piFinset fun _ => G := by
    ext I; simp [Fintype.mem_piFinset]
  rw [this, Fintype.card_piFinset, Finset.prod_const, Finset.card_univ, Fintype.card_fin]

/-! ### The threshold `p₀` -/

/-- If `g` of `B` blocks are right and the other `b` do not fit, `S + B + b·ℓ > β`, then `g ≤ p₀·B`. -/
theorem good_le_p₀ (B ℓ β S g b : ℕ) (hB : 0 < B) (hgb : g + b = B) (h : β < S + B + b * ℓ) :
    (g : ℝ) ≤ p₀ β S B ℓ * B := by
  unfold p₀
  rcases Nat.eq_zero_or_pos ℓ with rfl | hℓ
  · simp only [Nat.cast_zero, mul_zero, div_zero, sub_zero, one_mul]
    exact_mod_cast (by omega : g ≤ B)
  · have hB' : (0 : ℝ) < B := by exact_mod_cast hB
    have hℓ' : (0 : ℝ) < ℓ := by exact_mod_cast hℓ
    have hgb' : (g : ℝ) + b = B := by exact_mod_cast hgb
    have h' : (β : ℝ) < S + B + b * ℓ := by exact_mod_cast h
    have key : (1 - ((β : ℝ) - S - B) / (B * ℓ)) * B = B - ((β : ℝ) - S - B) / ℓ := by
      field_simp
    rw [key]
    have : ((β : ℝ) - S - B) / ℓ < b := by
      rw [div_lt_iff₀ hℓ']; linarith
    linarith

open Classical in
/-- **Averaging.** `k` uniform challenges all land on blocks the responder gets right with probability at most
`Pr[the wrong blocks fit in β bits] + p₀^k`. -/
theorem pr_all_good_le {Ω : Type} [Fintype Ω] [Nonempty Ω] (B ℓ β S k : ℕ) (hB : 0 < B)
    (good : Ω → Fin B → Prop) :
    pr (fun x : Ω × (Fin k → Fin B) => ∀ j, good x.1 (x.2 j)) ≤
      pr (fun a => S + B + (Finset.univ.filter fun i => ¬ good a i).card * ℓ ≤ β) +
        ENNReal.ofReal (p₀ β S B ℓ) ^ k := by
  have : Nonempty (Fin B) := ⟨⟨0, hB⟩⟩
  refine pr_prod_le (α := Ω) (β := Fin k → Fin B) (fun a I => ∀ j, good a (I j))
    (fun a => S + B + (Finset.univ.filter fun i => ¬ good a i).card * ℓ ≤ β) (ENNReal.ofReal (p₀ β S B ℓ) ^ k)
    fun a ha => ?_
  have hgb := Finset.card_filter_add_card_filter_not (s := Finset.univ) (good a)
  rw [Finset.card_univ, Fintype.card_fin] at hgb
  have hr := good_le_p₀ B ℓ β S _ _ hB hgb (by omega)
  have hB0 : ((B : ℝ≥0∞) ^ k) ≠ 0 := pow_ne_zero _ (by exact_mod_cast hB.ne')
  have hBt : ((B : ℝ≥0∞) ^ k) ≠ ⊤ := ENNReal.pow_ne_top (ENNReal.natCast_ne_top _)
  have hcard : (Finset.univ.filter fun I : Fin k → Fin B => ∀ j, good a (I j)).card =
      (Finset.univ.filter (good a)).card ^ k := by
    rw [← card_pi_mem]; congr 1; ext I; simp
  rw [pr_def, hcard, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin, Nat.cast_pow, Nat.cast_pow,
    ENNReal.div_le_iff hB0 hBt, ← mul_pow]
  gcongr
  calc ((Finset.univ.filter (good a)).card : ℝ≥0∞)
      = ENNReal.ofReal ((Finset.univ.filter (good a)).card : ℝ) := (ENNReal.ofReal_natCast _).symm
    _ ≤ ENNReal.ofReal (p₀ β S B ℓ * B) := ENNReal.ofReal_le_ofReal hr
    _ = ENNReal.ofReal (p₀ β S B ℓ) * B := by
        rw [ENNReal.ofReal_mul' (Nat.cast_nonneg _), ENNReal.ofReal_natCast]

/-! ### Compression -/

/-- **Codes from counting.** A set of at most `2^β` points has a `β`-bit code that decodes every point of it. -/
theorem exists_code {X : Type} [Fintype X] [Nonempty X] (T : Finset X) (β : ℕ) (h : T.card ≤ 2 ^ β) :
    ∃ (c : X → Bits β) (d : Bits β → X), ∀ x ∈ T, d (c x) = x := by
  classical
  obtain ⟨e⟩ := Function.Embedding.nonempty_of_card_le (α := T) (β := Bits β) (by simpa using h)
  refine ⟨fun x => if hx : x ∈ T then e ⟨x, hx⟩ else fun _ => false,
    fun w => if hw : ∃ y, e y = w then hw.choose.1 else Classical.arbitrary X, fun x hx => ?_⟩
  have hex : ∃ y, e y = e ⟨x, hx⟩ := ⟨_, rfl⟩
  simp only [hx, dite_true, hex]
  exact congrArg Subtype.val (e.injective hex.choose_spec)

/-- **Counting patches.** Assignments of an optional `ℓ`-bit block to each of `B` positions, with at most `L` bits
of blocks in total: at most `2^B · 2^L`. -/
theorem card_patches_le (B ℓ L : ℕ) :
    (Finset.univ.filter fun g : Fin B → Option (Bits ℓ) =>
      (Finset.univ.filter fun i => g i ≠ none).card * ℓ ≤ L).card ≤ 2 ^ B * 2 ^ L := by
  classical
  let t : Finset (Fin B) → (i : Fin B) → Finset (Option (Bits ℓ)) := fun D i =>
    if i ∈ D then Finset.univ.erase none else {none}
  have hsub : (Finset.univ.filter fun g : Fin B → Option (Bits ℓ) =>
      (Finset.univ.filter fun i => g i ≠ none).card * ℓ ≤ L) ⊆
      (Finset.univ.filter fun D : Finset (Fin B) => D.card * ℓ ≤ L).biUnion
        fun D => Fintype.piFinset (t D) := by
    intro g hg
    rw [Finset.mem_filter] at hg
    rw [Finset.mem_biUnion]
    refine ⟨Finset.univ.filter fun i => g i ≠ none, by simpa using hg.2, ?_⟩
    rw [Fintype.mem_piFinset]
    intro i
    by_cases hi : g i = none <;> simp [t, hi]
  refine (Finset.card_le_card hsub).trans ((Finset.card_biUnion_le).trans ?_)
  have hD : ∀ D ∈ (Finset.univ.filter fun D : Finset (Fin B) => D.card * ℓ ≤ L),
      (Fintype.piFinset (t D)).card ≤ 2 ^ L := by
    intro D hD
    rw [Finset.mem_filter] at hD
    rw [Fintype.card_piFinset]
    have : ∀ i, (t D i).card = if i ∈ D then 2 ^ ℓ else 1 := by
      intro i
      by_cases hi : i ∈ D
      · simp [t, hi, Finset.card_erase_of_mem, Fintype.card_option]
      · simp [t, hi]
    simp_rw [this]
    rw [Finset.prod_ite_mem, Finset.univ_inter, Finset.prod_const, ← pow_mul]
    exact Nat.pow_le_pow_right (by norm_num) (by rw [mul_comm]; exact hD.2)
  refine (Finset.sum_le_card_nsmul _ _ _ hD).trans ?_
  rw [smul_eq_mul]
  refine Nat.mul_le_mul_right _ ?_
  refine (Finset.card_filter_le _ _).trans ?_
  rw [Finset.card_univ, Fintype.card_finset, Fintype.card_fin]

/-- The compressor's message: the responder's state, and a replacement for each block it gets wrong. -/
abbrev Desc (S B ℓ : ℕ) : Type := Bits S × (Fin B → Option (Bits ℓ))

/-- The messages that fit in `β` bits: `S` for the state, `B` for the set of replaced blocks, `ℓ` per replaced
block. -/
def fits (S B ℓ β : ℕ) : Finset (Desc S B ℓ) :=
  Finset.univ.filter fun x => S + B + (Finset.univ.filter fun i => x.2 i ≠ none).card * ℓ ≤ β

/-- At most `2^β` messages fit: `2^S` states times `2^B·2^(β−S−B)` patches (none if `β < S + B`). -/
theorem card_fits_le (S B ℓ β : ℕ) : (fits S B ℓ β).card ≤ 2 ^ β := by
  by_cases hβ : S + B ≤ β
  · have hsub : fits S B ℓ β ⊆ Finset.univ ×ˢ (Finset.univ.filter fun g : Fin B → Option (Bits ℓ) =>
        (Finset.univ.filter fun i => g i ≠ none).card * ℓ ≤ β - S - B) := by
      intro x hx
      rw [fits, Finset.mem_filter] at hx
      rw [Finset.mem_product, Finset.mem_filter]
      exact ⟨Finset.mem_univ _, Finset.mem_univ _, by omega⟩
    refine (Finset.card_le_card hsub).trans ?_
    rw [Finset.card_product, Finset.card_univ, Fintype.card_fun, Fintype.card_bool, Fintype.card_fin]
    calc 2 ^ S * _ ≤ 2 ^ S * (2 ^ B * 2 ^ (β - S - B)) := Nat.mul_le_mul_left _ (card_patches_le B ℓ _)
      _ = 2 ^ β := by rw [← pow_add, ← pow_add]; congr 1; omega
  · have : fits S B ℓ β = ∅ := by
      ext x
      rw [fits, Finset.mem_filter]
      simp only [Finset.mem_univ, true_and, Finset.notMem_empty, iff_false]
      omega
    rw [this, Finset.card_empty]
    exact Nat.zero_le _

/-- Replace every block where the prediction `P` differs from `C`. -/
def patch {B ℓ : ℕ} (P C : Fin B → Bits ℓ) : Fin B → Option (Bits ℓ) :=
  fun i => if P i = C i then none else some (C i)

/-- The patched block where there is one, else the prediction: this is `C`. -/
theorem patch_getD {B ℓ : ℕ} (P C : Fin B → Bits ℓ) (i : Fin B) : (patch P C i).getD (P i) = C i := by
  by_cases h : P i = C i <;> simp [patch, h]

/-- If the blocks where `P` differs from `C` fit (`S + B + #wrong·ℓ ≤ β`), so does the message `(σ, patch P C)`. -/
theorem mem_fits_patch {S B ℓ β : ℕ} (σ : Bits S) (P C : Fin B → Bits ℓ) {_ : DecidablePred fun i => ¬ P i = C i}
    (h : S + B + (Finset.univ.filter fun i => ¬ P i = C i).card * ℓ ≤ β) : (σ, patch P C) ∈ fits S B ℓ β := by
  rw [fits, Finset.mem_filter]
  refine ⟨Finset.mem_univ _, ?_⟩
  convert h using 4
  ext i
  by_cases hi : P i = C i <;> simp [patch, hi]

end PousThm1ii

open Pous PousThm1ii in
theorem PousProofs.theorem1_ii : Pous.Guarantees.Theorem1ii := by
  intro M sch β S k π hinc W A₁ A₂
  obtain ⟨c, d, hcd⟩ := exists_code (fits S sch.B sch.ℓ β) β (card_fits_le S sch.B sch.ℓ β)
  let good : M.Ω × sch.Coins → Fin sch.B → Prop := fun a i =>
    A₂ W (sch.enc W a.1 a.2).1 (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) a.1 i =
      (sch.enc W a.1 a.2).2 i
  have hre : pr (prodAccepts sch k W A₁ A₂) =
      pr (fun x : (M.Ω × sch.Coins) × (Fin k → Fin sch.B) => ∀ j, good x.1 (x.2 j)) :=
    pr_equiv (Equiv.prodAssoc _ _ _).symm
      (fun x : (M.Ω × sch.Coins) × (Fin k → Fin sch.B) => ∀ j, good x.1 (x.2 j))
  rw [hre]
  refine (pr_all_good_le sch.B sch.ℓ β S k sch.B_pos good).trans (add_le_add ?_ le_rfl)
  let cmp : Bits sch.n → Bits sch.t → sch.Code → M.Ω → Bits β :=
    fun W pp C ω => c (A₁ W pp C ω, patch (A₂ W pp (A₁ W pp C ω) ω) C)
  let exp : Bits sch.n → Bits sch.t → Bits β → M.Ω → sch.Code :=
    fun W pp w ω i => ((d w).2 i).getD (A₂ W pp (d w).1 ω i)
  refine le_trans (pr_mono fun a ha => ?_) (hinc W cmp exp)
  have hfit := mem_fits_patch (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1)
    (A₂ W (sch.enc W a.1 a.2).1 (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1) a.1)
    (sch.enc W a.1 a.2).2 ha
  show exp W _ (cmp W _ _ a.1) a.1 = _
  simp only [exp, cmp, hcd _ hfit]
  funext i
  exact patch_getD _ _ i
