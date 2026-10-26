import Definitions.Accounting.Guarantees.FillerLuck
import Proofs.Accounting.TS.Rounds

/-!
# The filler's per-check luck

Proofs of `Definitions.Accounting.Guarantees.FillerLuck`. Lemma P2 is PoUW's parity lemma on `F₂^K` (`parity_prob_sq`,
Cauchy–Schwarz and Parseval: `(2p − 1)² ≤ 2^R/2^K`), read at every deficit (`luck_of_sq`) and over a column of uniform
parity by splitting the column's noise into its parity and a uniform rest (`exists_parity_equiv`); a word ring's
equality is read mod 2 through `φ`. A tile's cell is the cell lemma at each setting of the other rows and columns
(`Equiv.piSplitAt`). The windows are inductions over the rounds; `FillerBlocksDeficit` is `budget_tail` at `Λ = 1`.
A pinned receipt's draw passes with probability at most one less its wrong share at every setting of the live coins
(`prob_le_one_sub_not`, `sum_pinned_le`), so its windows are the same inductions at `p = 0`.
-/

namespace Accounting.SecurityProofs.FillerLuck

open Finset Accounting.Filler Accounting.TS Pouw.RelativeSecurity

/-! The helpers sit in `Cell` so that they can't clash with a name in a file that imports this one. -/

namespace Cell

theorem luck_nonneg (d : ℕ) : 0 ≤ luck d := by
  unfold luck
  positivity

theorem luck_anti {d d' : ℕ} (h : d ≤ d') : luck d' ≤ luck d := by
  unfold luck
  have hd : (d : ℝ) ≤ d' := by exact_mod_cast h
  have : (2 : ℝ) ^ (-(d' : ℝ) / 2 - 1) ≤ (2 : ℝ) ^ (-(d : ℝ) / 2 - 1) :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)
  linarith

theorem luck_eq (d : ℕ) : luck d = 1 / 2 + (2 : ℝ) ^ (-(d : ℝ) / 2) / 2 := by
  rw [luck, Real.rpow_sub (by norm_num), Real.rpow_one]

theorem half_pow_sq (d : ℕ) : ((2 : ℝ) ^ (-(d : ℝ) / 2)) ^ 2 = 1 / 2 ^ d := by
  rw [← Real.rpow_natCast, ← Real.rpow_mul (by norm_num),
    show -(d : ℝ) / 2 * ((2 : ℕ) : ℝ) = -(d : ℝ) by push_cast; ring, Real.rpow_neg (by norm_num), Real.rpow_natCast,
    one_div]

theorem luck_zero : luck 0 = 1 := by
  rw [luck_eq]
  norm_num

theorem luck_one_le : luck 1 ≤ 427 / 500 := by
  rw [luck_eq]
  have hs0 : 0 ≤ (2 : ℝ) ^ (-((1 : ℕ) : ℝ) / 2) := by positivity
  have hsq : ((2 : ℝ) ^ (-((1 : ℕ) : ℝ) / 2)) ^ 2 ≤ (177 / 250) ^ 2 := by
    rw [half_pow_sq]
    norm_num
  have := (abs_le_of_sq_le_sq' hsq (by norm_num)).2
  linarith

/-- **Lemma P2's square bound at every deficit**: `(2p − 1)² ≤ 2^R/2^K` gives `p ≤ ½ + 2^(−(K−R)/2−1)`. -/
theorem luck_of_sq {p : ℝ} {K R : ℕ} (hp : p ≤ 1) (h : (2 * p - 1) ^ 2 ≤ 2 ^ R / 2 ^ K) : p ≤ luck (K - R) := by
  rcases le_or_gt K R with hKR | hRK
  · rw [Nat.sub_eq_zero_of_le hKR, luck_zero]
    exact hp
  · obtain ⟨d, rfl⟩ : ∃ d, K = R + d := ⟨K - R, by omega⟩
    rw [Nat.add_sub_cancel_left, luck_eq]
    have hs0 : 0 ≤ (2 : ℝ) ^ (-(d : ℝ) / 2) := by positivity
    have h2 : (2 : ℝ) ^ R / 2 ^ (R + d) = 1 / 2 ^ d := by
      rw [pow_add]
      field_simp
    have hsq : (2 * p - 1) ^ 2 ≤ ((2 : ℝ) ^ (-(d : ℝ) / 2)) ^ 2 := by
      rw [half_pow_sq, ← h2]
      exact h
    have := (abs_le_of_sq_le_sq' hsq hs0).2
    linarith

theorem prob_none {Ω : Type} [Fintype Ω] : prob (fun _ : Ω => False) = 0 := by
  unfold prob
  simp

theorem prob_all {Ω : Type} [Fintype Ω] [Nonempty Ω] : prob (fun _ : Ω => True) = 1 := by
  classical
  rw [prob_eq_expect]
  unfold Pouw.RelativeSecurity.expect
  simp only [ite_true, sum_const, card_univ, nsmul_eq_mul, mul_one]
  exact div_self (Nat.cast_ne_zero.mpr Fintype.card_ne_zero)

theorem exists_of_prob_pos {Ω : Type} [Fintype Ω] {p : Ω → Prop} (h : 0 < prob p) : ∃ ω, p ω := by
  by_contra hn
  push Not at hn
  have : prob p = 0 := by
    rw [show p = fun _ => False from funext fun ω => propext ⟨hn ω, False.elim⟩]
    exact prob_none
  linarith

theorem prob_vec {K : ℕ} (z : Fin K → ZMod 2) : prob (fun u : Fin K → ZMod 2 => u = z) = 1 / 2 ^ K := by
  classical
  rw [prob_eq_expect]
  unfold Pouw.RelativeSecurity.expect
  simp

theorem card_fiber_eq {Y : Type} [Fintype Y] {K : ℕ} (π : Y → Fin K → ZMod 2)
    (hπ : ∀ z, prob (fun y => π y = z) = 1 / 2 ^ K) (z z' : Fin K → ZMod 2) :
    Fintype.card {y // π y = z} = Fintype.card {y // π y = z'} := by
  classical
  have hY : (Fintype.card Y : ℝ) ≠ 0 := by
    intro h0
    have := hπ z
    unfold prob at this
    rw [h0, div_zero] at this
    exact (by positivity : (0 : ℝ) < 1 / 2 ^ K).ne this
  have h := (hπ z).trans (hπ z').symm
  unfold prob at h
  have h' := (div_left_inj' hY).mp h
  rw [Fintype.card_subtype, Fintype.card_subtype]
  convert (Nat.cast_inj (R := ℝ)).mp h' using 2 <;> ext <;> simp

/-- **A column of uniform parity splits into its parity and a uniform rest.** -/
theorem exists_parity_equiv {Y : Type} [Fintype Y] {K : ℕ} (π : Y → Fin K → ZMod 2)
    (hπ : ∀ z, prob (fun y => π y = z) = 1 / 2 ^ K) :
    ∃ (m : ℕ) (e : (Fin K → ZMod 2) × Fin m ≃ Y), ∀ b, π (e b) = b.1 := by
  classical
  have hc : ∀ z, Fintype.card {y // π y = z} = Fintype.card {y // π y = 0} := fun z => card_fiber_eq π hπ z 0
  let F : ∀ z, {y // π y = z} ≃ Fin (Fintype.card {y // π y = 0}) := fun z => Fintype.equivFinOfCardEq (hc z)
  exact ⟨_, (Equiv.sigmaEquivProd _ _).symm.trans
    ((Equiv.sigmaCongrRight fun z => (F z).symm).trans (Equiv.sigmaFiberEquiv π)), fun b => ((F b.1).symm b.2).2⟩

/-- **Lemma P2 on `F₂`** over a column of uniform parity, at every deficit. -/
theorem parity_le_balanced {Y : Type} [Fintype Y] {K R : ℕ} (π : Y → Fin K → ZMod 2)
    (hπ : ∀ z, prob (fun y => π y = z) = 1 / 2 ^ K) (F : (Fin K → ZMod 2) → Fin R → ZMod 2)
    (G : Y → Fin R → ZMod 2) (a : (Fin K → ZMod 2) → ZMod 2) (b : Y → ZMod 2) :
    prob (fun xy : (Fin K → ZMod 2) × Y => F xy.1 ⬝ᵥ G xy.2 + a xy.1 + b xy.2 = xy.1 ⬝ᵥ π xy.2) ≤
      luck (K - R) := by
  obtain ⟨m, e, he⟩ := exists_parity_equiv π hπ
  rw [prob_equiv ((Equiv.prodCongr (Equiv.refl _) e).symm.trans (Equiv.prodAssoc _ _ _).symm)]
  refine prob_le_of_slices _ (luck_nonneg _) fun i => ?_
  refine le_of_eq_of_le ?_ (luck_of_sq (prob_le_one _)
    (parity_prob_sq F (fun z => G (e (z, i))) a fun z => b (e (z, i))))
  congr 1
  funext xz
  simp [he]

theorem bit_map {A : Type} [CommRing A] (φ : A →+* ZMod 2) (z : ZMod 2) : φ (bit z) = z := by
  rw [bit, map_natCast, ZMod.natCast_zmod_val]

/-- The honest word takes `K` products, so `K − R` counts the products a run's word is short. -/
theorem cellWord_products {K : ℕ} {Y A : Type} [CommRing A] (c : Y → Fin K → A) :
    Products K (cellWord c) :=
  ⟨fun k x => bit (x k), fun k y => c y k, 0, 0, fun x y => by simp [cellWord]⟩

theorem parityLuck : Accounting.Guarantees.FillerLuck.ParityLuck := by
  intro Y A _ _ φ K c R P hc hP
  obtain ⟨f, g, a, b, hP⟩ := hP
  have key : ∀ xy : (Fin K → ZMod 2) × Y, P xy.1 xy.2 = cellWord c xy.1 xy.2 →
      (fun j => φ (f j xy.1)) ⬝ᵥ (fun j => φ (g j xy.2)) + φ (a xy.1) + φ (b xy.2) =
        xy.1 ⬝ᵥ parity φ c xy.2 := by
    intro xy h
    have h' := congrArg φ h
    rw [hP, cellWord, map_add, map_add, map_sum, map_sum] at h'
    simp only [map_mul, bit_map] at h'
    exact h'
  exact (prob_mono key).trans (parity_le_balanced (parity φ c) hc (fun x j => φ (f j x)) (fun y j => φ (g j y))
    (fun x => φ (a x)) fun y => φ (b y))

theorem nonempty_of_balanced {Y A : Type} [Fintype Y] [CommRing A] {K : ℕ} {φ : A →+* ZMod 2}
    {c : Y → Fin K → A} (hc : ParityBalanced φ c) : Nonempty Y := by
  by_contra hY
  rw [not_nonempty_iff] at hY
  have := hc 0
  unfold prob at this
  rw [Fintype.card_eq_zero, Nat.cast_zero, div_zero] at this
  exact (by positivity : (0 : ℝ) < 1 / 2 ^ K).ne this

theorem piSplitAt_update {α β : Type} [DecidableEq α] (i : α) (x x₀ : β) (f : {j // j ≠ i} → β) :
    Function.update ((Equiv.piSplitAt i fun _ => β).symm (x₀, f)) i x =
      (Equiv.piSplitAt i fun _ => β).symm (x, f) := by
  funext j
  by_cases hj : j = i
  · subst hj
    simp [Equiv.piSplitAt_symm_apply]
  · simp [Equiv.piSplitAt_symm_apply, hj]

/-- **A tile's cell**: the cell lemma at each setting of the other rows' and columns' noise. -/
theorem tile_cell_le {K : ℕ} {ρ ν Y A : Type} [DecidableEq ρ] [DecidableEq ν] [Fintype ρ] [Fintype ν]
    [Fintype Y] [CommRing A] (φ : A →+* ZMod 2) (c : Y → Fin K → A) (hc : ParityBalanced φ c)
    (W : Noise K ρ ν Y → A) (r : ρ) (s : ν) (R : ℕ) (hW : CellProducts W r s R) :
    prob (fun ω : Noise K ρ ν Y => W ω = cellWord c (ω.1 r) (ω.2 s)) ≤ luck (K - R) := by
  obtain ⟨y₀⟩ := nonempty_of_balanced hc
  rw [prob_equiv ((Equiv.prodCongr (Equiv.piSplitAt r fun _ => Fin K → ZMod 2) (Equiv.piSplitAt s fun _ => Y)).trans
    (Equiv.prodProdProdComm _ _ _ _))]
  refine prob_le_of_slices _ (luck_nonneg _) fun o => ?_
  have hP : Products R fun x y =>
      W ((Equiv.piSplitAt r fun _ => Fin K → ZMod 2).symm (x, o.1), (Equiv.piSplitAt s fun _ => Y).symm (y, o.2)) := by
    obtain ⟨f, g, a, b, h⟩ :=
      hW ((Equiv.piSplitAt r fun _ => Fin K → ZMod 2).symm (0, o.1), (Equiv.piSplitAt s fun _ => Y).symm (y₀, o.2))
    refine ⟨f, g, a, b, fun x y => ?_⟩
    have hxy := h x y
    simp only [piSplitAt_update] at hxy
    exact hxy
  refine le_of_eq_of_le ?_ (parityLuck φ c R _ hc hP)
  congr 1
  funext xy
  simp [Equiv.piSplitAt_symm_apply]

theorem parity_bits {A : Type} [CommRing A] (φ : A →+* ZMod 2) {K : ℕ} (y : Fin K → ZMod 2) :
    parity φ bitsCol y = y := by
  funext k
  simp [parity, bitsCol, bit_map]

theorem balanced_of_equiv {Y A M : Type} [Fintype Y] [Fintype M] [Nonempty M] [CommRing A] {K : ℕ}
    (φ : A →+* ZMod 2) (c : Y → Fin K → A) (e : Y ≃ (Fin K → ZMod 2) × M) (he : ∀ y, parity φ c y = (e y).1) :
    ParityBalanced φ c := by
  intro z
  rw [prob_equiv e, ← prob_vec z, ← prob_fst (β := M) (fun u => u = z)]
  congr 1
  funext b
  simp [he]

/-- bU's entry mod 2 (`uOf t ≡ t₀ ⊕ t₆`) and `t / 2` determine the byte. -/
theorem bu_split_injective :
    Function.Injective fun t : Fin 256 => (((uOf t : ℤ) : ZMod 2), (⟨(t : ℕ) / 2, by omega⟩ : Fin 128)) := by
  intro t t' h
  simp only [Prod.mk.injEq, Fin.mk.injEq] at h
  obtain ⟨h1, h2⟩ := h
  rw [ZMod.intCast_eq_intCast_iff'] at h1
  unfold uOf at h1
  push_cast at h1
  have := t.isLt
  have := t'.isLt
  ext
  omega

theorem balanced_bu {A : Type} [CommRing A] (φ : A →+* ZMod 2) (K : ℕ) : ParityBalanced φ (buCol (K := K)) := by
  let e₁ : Fin 256 ≃ ZMod 2 × Fin 128 := Equiv.ofBijective _
    ((Fintype.bijective_iff_injective_and_card _).mpr ⟨bu_split_injective, by simp⟩)
  refine balanced_of_equiv φ buCol ((Equiv.arrowCongr (Equiv.refl _) e₁).trans
    (Equiv.arrowProdEquivProdArrow (Fin K) (fun _ => ZMod 2) fun _ => Fin 128)) fun t => ?_
  funext k
  simp [parity, buCol, e₁, map_intCast]

theorem balanced_bits {A : Type} [CommRing A] (φ : A →+* ZMod 2) (K : ℕ) : ParityBalanced φ (bitsCol (K := K)) :=
  balanced_of_equiv φ bitsCol (Equiv.prodPUnit _).symm fun y => by rw [parity_bits]; rfl

/-- **The spot check's pass bound**: a uniform cell finds its word right with probability at most
`1 − (1 − 0.854)·shortShare W`. -/
theorem spot_pass_le {K : ℕ} {ρ ν Y A : Type} [DecidableEq ρ] [DecidableEq ν] [Fintype ρ] [Fintype ν]
    [Fintype Y] [CommRing A] (φ : A →+* ZMod 2) (c : Y → Fin K → A) (hc : ParityBalanced φ c)
    (W : ρ → ν → Noise K ρ ν Y → A) :
    prob (fun a : Noise K ρ ν Y × (ρ × ν) => CellRight c W a.1 a.2.1 a.2.2) ≤
      1 - (1 - 427 / 500) * shortShare W := by
  classical
  rw [prob_eq_expect, expect_prod_swap, shortShare, prob_eq_expect]
  have hpt : ∀ rs : ρ × ν, expect (fun ω : Noise K ρ ν Y =>
      if CellRight c W (ω, rs).1 (ω, rs).2.1 (ω, rs).2.2 then (1 : ℝ) else 0) ≤
      1 + -(1 - 427 / 500) * (if Short (W rs.1 rs.2) rs.1 rs.2 then 1 else 0) := by
    intro rs
    split_ifs with hb
    · obtain ⟨R, hR, hW⟩ := hb
      rw [← prob_eq_expect (fun ω : Noise K ρ ν Y => CellRight c W (ω, rs).1 (ω, rs).2.1 (ω, rs).2.2)]
      have := (tile_cell_le φ c hc (W rs.1 rs.2) rs.1 rs.2 R hW).trans ((luck_anti (by omega)).trans luck_one_le)
      exact (le_of_eq_of_le rfl this).trans (by linarith)
    · rw [mul_zero, add_zero]
      exact expect_le_of_le zero_le_one fun ω => by split_ifs <;> norm_num
  have := (expect_mono hpt).trans (expect_affine_le zero_le_one _ _)
  linarith

theorem mul_exp_le (v : ℝ) : (1 - v) * Real.exp v ≤ 1 := by
  have h := Real.add_one_le_exp (-v)
  have he : Real.exp (-v) * Real.exp v = 1 := by rw [← Real.exp_add, neg_add_cancel, Real.exp_zero]
  nlinarith [Real.exp_pos v]

/-- **One draw's budget**: an event of probability at most `1 − v` weighted by `e^v` sums to at most `|Ω|`. -/
theorem sum_exp_le {Ω : Type} [Fintype Ω] (q : Ω → Prop) (v : ℝ) (h : prob q ≤ 1 - v) :
    (∑ a, (by classical exact if q a then Real.exp v else 0)) ≤ (Fintype.card Ω : ℝ) := by
  classical
  have hcard : ((univ.filter q).card : ℝ) ≤ (1 - v) * Fintype.card Ω := by
    rcases Nat.eq_zero_or_pos (Fintype.card Ω) with h0 | h0
    · have : (univ.filter q).card = 0 := Nat.eq_zero_of_le_zero ((card_filter_le _ _).trans_eq
        (by rw [card_univ, h0]))
      rw [this, h0]
      simp
    · have hC : (0 : ℝ) < Fintype.card Ω := by exact_mod_cast h0
      unfold prob at h
      rwa [div_le_iff₀ hC] at h
  calc (∑ a, if q a then Real.exp v else 0) = ((univ.filter q).card : ℝ) * Real.exp v := by
        rw [← Finset.sum_filter, sum_const, nsmul_eq_mul]
    _ ≤ (1 - v) * Fintype.card Ω * Real.exp v := mul_le_mul_of_nonneg_right hcard (Real.exp_pos v).le
    _ = (1 - v) * Real.exp v * Fintype.card Ω := by ring
    _ ≤ 1 * Fintype.card Ω := mul_le_mul_of_nonneg_right (mul_exp_le v) (Nat.cast_nonneg _)
    _ = Fintype.card Ω := one_mul _

theorem sum_ite_mono {Ω : Type} [Fintype Ω] (p q : Ω → Prop) (h : ∀ a, p a → q a) (f : Ω → ℝ)
    (hf : ∀ a, 0 ≤ f a) :
    (∑ a, (by classical exact if p a then f a else 0)) ≤ ∑ a, (by classical exact if q a then f a else 0) := by
  classical
  refine sum_le_sum fun a _ => ?_
  by_cases hp : p a
  · simp [hp, h a hp]
  · rw [ite_eq_right hp]
    split_ifs
    · exact hf a
    · rfl

theorem luckAt_full {K : ℕ} {ρ ν Y A : Type} [DecidableEq ρ] [DecidableEq ν] [Fintype ρ] [Fintype ν]
    [Fintype Y] [CommRing A] (φ : A →+* ZMod 2) (c : Y → Fin K → A) (hc : ParityBalanced φ c) :
    ∀ {n : ℕ} (g : RoundGame (Noise K ρ ν Y) n), FullFill c g → RoundGame.LuckAt (luck 1) g
  | _, .done, _ => trivial
  | _, .round d pass next, h => by
    obtain ⟨⟨W, hd, hpass⟩, hnext⟩ := h
    have := nonempty_of_balanced hc
    refine ⟨fun ω => ?_, ?_, fun ω => luckAt_full φ c hc (next ω) (hnext ω)⟩
    · rw [hd ω]
      exact ⟨prob_nonneg _, prob_le_one _⟩
    · by_cases h0 : 0 < shortShare W
      · obtain ⟨⟨r, s⟩, R, hR, hW⟩ := exists_of_prob_pos h0
        have hall : prob (fun ω : Noise K ρ ν Y => 0 < d ω) = 1 := by
          rw [show (fun ω => 0 < d ω) = fun _ => True from funext fun ω => by simp [hd ω, h0]]
          exact prob_all
        rw [hall, mul_one]
        calc prob (fun ω => pass ω ∧ 0 < d ω) ≤ prob (fun ω => W r s ω = cellWord c (ω.1 r) (ω.2 s)) :=
              prob_mono fun ω h => hpass ω h.1 r s
          _ ≤ luck (K - R) := tile_cell_le φ c hc (W r s) r s R hW
          _ ≤ luck 1 := luck_anti (by omega)
      · have : prob (fun ω : Noise K ρ ν Y => pass ω ∧ 0 < d ω) = 0 := by
          rw [show (fun ω => pass ω ∧ 0 < d ω) = fun _ => False from funext fun ω => by simp [hd ω, h0]]
          exact prob_none
        rw [this]
        exact mul_nonneg (luck_nonneg 1) (prob_nonneg _)

theorem budget_spot {K : ℕ} {ρ ν Y A : Type} [DecidableEq ρ] [DecidableEq ν] [Fintype ρ] [Fintype ν]
    [Fintype Y] [CommRing A] (φ : A →+* ZMod 2) (c : Y → Fin K → A) (hc : ParityBalanced φ c) :
    ∀ {n : ℕ} (g : RoundGame (Noise K ρ ν Y × (ρ × ν)) n), SpotFill c g →
      RoundGame.BudgetAt (1 / (1 - 427 / 500)) g
  | _, .done, _ => trivial
  | _, .round d pass next, h => by
    obtain ⟨⟨W, hd, hpass⟩, hnext⟩ := h
    refine ⟨fun a => by rw [hd a]; exact prob_nonneg _, ?_, fun a => budget_spot φ c hc (next a) (hnext a)⟩
    have hv : ∀ a, d a / (1 / (1 - 427 / 500)) = (1 - 427 / 500) * shortShare W := fun a => by
      rw [hd a]
      ring
    simp only [hv]
    exact (sum_ite_mono pass _ hpass _ fun _ => (Real.exp_pos _).le).trans
      (sum_exp_le _ _ (spot_pass_le φ c hc W))

theorem budget_blocks {K : ℕ} {ρ ν Y A Ω₀ : Type} [DecidableEq ρ] [DecidableEq ν] [Fintype ρ] [Fintype ν]
    [Fintype Y] [CommRing A] [Fintype Ω₀] (φ : A →+* ZMod 2) (c : Y → Fin K → A) (hc : ParityBalanced φ c)
    {p₀ : ℝ} (hp₀ : p₀ ≤ 1) :
    ∀ {n : ℕ} (g : RoundGame (Ω₀ × (Noise K ρ ν Y × (ρ × ν))) n), FillBlocks p₀ c g → RoundGame.BudgetAt 1 g
  | _, .done, _ => trivial
  | _, .round d pass next, h => by
    classical
    obtain ⟨⟨d₀, pass₀, W, hd₀, hpass₀, hd, hpass⟩, hnext⟩ := h
    have hu := prob_nonneg fun rs : ρ × ν => Short (W rs.1 rs.2) rs.1 rs.2
    refine ⟨fun a => ?_, ?_, fun a => budget_blocks φ c hc hp₀ (next a) (hnext a)⟩
    · rw [hd a]
      have : 0 ≤ (1 - p₀) * d₀ := mul_nonneg (by linarith) hd₀
      unfold shortShare
      positivity
    set v₀ := (1 - p₀) * d₀
    set v₁ := (1 - 427 / 500) * shortShare W
    have h₀ := sum_exp_le pass₀ v₀ hpass₀
    have h₁ := sum_exp_le _ v₁ (spot_pass_le φ c hc W)
    calc (∑ a, if pass a then Real.exp (d a / 1) else 0)
        ≤ ∑ a : Ω₀ × (Noise K ρ ν Y × (ρ × ν)), (if pass₀ a.1 then Real.exp v₀ else 0) *
            (if CellRight c W a.2.1 a.2.2.1 a.2.2.2 then Real.exp v₁ else 0) := by
          refine sum_le_sum fun a _ => ?_
          by_cases hp : pass a
          · obtain ⟨h0, h1⟩ := hpass a hp
            simp only [hp, h0, h1, ↓reduceIte, hd a, div_one, Real.exp_add]
            exact le_rfl
          · rw [ite_eq_right hp]
            split_ifs <;> positivity
      _ = (∑ a₀, if pass₀ a₀ then Real.exp v₀ else 0) *
            ∑ a₁ : Noise K ρ ν Y × (ρ × ν), if CellRight c W a₁.1 a₁.2.1 a₁.2.2 then Real.exp v₁ else 0 := by
          rw [Fintype.sum_prod_type, Finset.sum_mul_sum]
      _ ≤ (Fintype.card Ω₀ : ℝ) * Fintype.card (Noise K ρ ν Y × (ρ × ν)) :=
          mul_le_mul h₀ h₁ (sum_nonneg fun _ _ => by split_ifs <;> positivity) (Nat.cast_nonneg _)
      _ = Fintype.card (Ω₀ × (Noise K ρ ν Y × (ρ × ν))) := by
          rw [Fintype.card_prod Ω₀ (Noise K ρ ν Y × (ρ × ν)), Nat.cast_mul]

/-- **An event and its complement**: `Pr[q] ≤ 1 − Pr[¬q]` (equality on a nonempty type). -/
theorem prob_le_one_sub_not {Ω : Type} [Fintype Ω] (q : Ω → Prop) : prob q ≤ 1 - prob fun a => ¬q a := by
  classical
  unfold prob
  rcases Nat.eq_zero_or_pos (Fintype.card Ω) with h0 | h0
  · simp [h0]
  · have hC : (0 : ℝ) < Fintype.card Ω := by exact_mod_cast h0
    have hsum : ((univ.filter q).card : ℝ) + ((univ.filter fun a => ¬q a).card : ℝ) = Fintype.card Ω := by
      exact_mod_cast (card_filter_add_card_filter_not (s := univ) q).trans (card_univ (α := Ω))
    have key : ((univ.filter q).card : ℝ) ≤ Fintype.card Ω - ((univ.filter fun a => ¬q a).card : ℝ) := by
      linarith
    rw [div_le_iff₀ hC, sub_mul, one_mul, div_mul_cancel₀ _ hC.ne']
    convert key

/-- **One pinned draw's budget**: a uniform tile's receipt right, weighted by `e^v` with `v` the wrong share at the live
coins, sums to at most `|C × τ|`. -/
theorem sum_pinned_le {C τ V : Type} [Fintype C] [Fintype τ] (G R : C → τ → V) (v : C → ℝ)
    (hv : ∀ c, v c = prob fun t => R c t ≠ G c t) :
    (∑ a : C × τ, (by classical exact if R a.1 a.2 = G a.1 a.2 then Real.exp (v a.1) else 0)) ≤
      (Fintype.card (C × τ) : ℝ) := by
  classical
  rw [Fintype.sum_prod_type]
  refine (sum_le_sum fun c _ => sum_exp_le (fun t => R c t = G c t) (v c) ?_).trans ?_
  · rw [hv c]
    exact prob_le_one_sub_not _
  · rw [sum_const, card_univ, Fintype.card_prod, nsmul_eq_mul, Nat.cast_mul]

theorem luckAt_pinned {D C τ V : Type} [Fintype C] [Fintype τ] (F : D → C → τ → V) :
    ∀ {n : ℕ} (g : RoundGame C n), FullPinnedFill F g → RoundGame.LuckAt 0 g
  | _, .done, _ => trivial
  | _, .round d pass next, h => by
    obtain ⟨⟨w, R, hd, hpass⟩, hnext⟩ := h
    refine ⟨fun c => ?_, ?_, fun c => luckAt_pinned F (next c) (hnext c)⟩
    · rw [hd c]
      exact ⟨prob_nonneg _, prob_le_one _⟩
    · have : (fun c => pass c ∧ 0 < d c) = fun _ => False := funext fun c => by
        refine propext ⟨fun ⟨hp, hpos⟩ => ?_, False.elim⟩
        have : (fun t => R c t ≠ F (w c) c t) = fun _ => False :=
          funext fun t => propext ⟨fun hne => hne (hpass c hp t), False.elim⟩
        rw [hd c, wrongShare, this, prob_none] at hpos
        exact lt_irrefl _ hpos
      rw [this, prob_none, zero_mul]

theorem budget_spot_pinned {D C τ V : Type} [Fintype C] [Fintype τ] (F : D → C → τ → V) :
    ∀ {n : ℕ} (g : RoundGame (C × τ) n), SpotPinnedFill F g → RoundGame.BudgetAt 1 g
  | _, .done, _ => trivial
  | _, .round d pass next, h => by
    obtain ⟨⟨w, R, hd, hpass⟩, hnext⟩ := h
    refine ⟨fun a => by rw [hd a]; exact prob_nonneg _, ?_,
      fun a => budget_spot_pinned F (next a) (hnext a)⟩
    simp only [hd, div_one]
    exact (sum_ite_mono pass _ hpass _ fun _ => (Real.exp_pos _).le).trans
      (sum_pinned_le (fun c t => F (w c) c t) R (fun c => wrongShare F (w c) c (R c)) fun _ => rfl)

theorem budget_blocks_pinned {D C τ V Ω₀ : Type} [Fintype C] [Fintype τ] [Fintype Ω₀] (F : D → C → τ → V)
    {p₀ : ℝ} (hp₀ : p₀ ≤ 1) :
    ∀ {n : ℕ} (g : RoundGame (Ω₀ × (C × τ)) n), PinnedFillBlocks p₀ F g → RoundGame.BudgetAt 1 g
  | _, .done, _ => trivial
  | _, .round d pass next, h => by
    classical
    obtain ⟨⟨d₀, pass₀, w, R, hd₀, hpass₀, hd, hpass⟩, hnext⟩ := h
    refine ⟨fun a => ?_, ?_, fun a => budget_blocks_pinned F hp₀ (next a) (hnext a)⟩
    · rw [hd a]
      have : 0 ≤ (1 - p₀) * d₀ := mul_nonneg (by linarith) hd₀
      have := prob_nonneg fun t => R a.1 a.2.1 t ≠ F (w a.1 a.2.1) a.2.1 t
      unfold wrongShare
      linarith
    set v₀ := (1 - p₀) * d₀
    have h₀ := sum_exp_le pass₀ v₀ hpass₀
    calc (∑ a, if pass a then Real.exp (d a / 1) else 0)
        ≤ ∑ a : Ω₀ × (C × τ), (if pass₀ a.1 then Real.exp v₀ else 0) *
            (if R a.1 a.2.1 a.2.2 = F (w a.1 a.2.1) a.2.1 a.2.2 then
              Real.exp (wrongShare F (w a.1 a.2.1) a.2.1 (R a.1 a.2.1)) else 0) := by
          refine sum_le_sum fun a _ => ?_
          by_cases hp : pass a
          · obtain ⟨h0, h1⟩ := hpass a hp
            simp only [hp, h0, h1, ↓reduceIte, hd a, div_one, Real.exp_add]
            exact le_rfl
          · rw [ite_eq_right hp]
            split_ifs <;> positivity
      _ = ∑ a₀, (if pass₀ a₀ then Real.exp v₀ else 0) *
            ∑ a₁ : C × τ, if R a₀ a₁.1 a₁.2 = F (w a₀ a₁.1) a₁.1 a₁.2 then
              Real.exp (wrongShare F (w a₀ a₁.1) a₁.1 (R a₀ a₁.1)) else 0 := by
          rw [Fintype.sum_prod_type]
          refine sum_congr rfl fun a₀ _ => ?_
          rw [Finset.mul_sum]
      _ ≤ ∑ a₀ : Ω₀, (if pass₀ a₀ then Real.exp v₀ else 0) * (Fintype.card (C × τ) : ℝ) :=
          sum_le_sum fun a₀ _ => mul_le_mul_of_nonneg_left
            (sum_pinned_le (fun c t => F (w a₀ c) c t) (R a₀) (fun c => wrongShare F (w a₀ c) c (R a₀ c))
              fun _ => rfl)
            (by split_ifs <;> positivity)
      _ ≤ (Fintype.card Ω₀ : ℝ) * Fintype.card (C × τ) := by
          rw [← Finset.sum_mul]
          exact mul_le_mul_of_nonneg_right h₀ (Nat.cast_nonneg _)
      _ = Fintype.card (Ω₀ × (C × τ)) := by
          rw [Fintype.card_prod Ω₀ (C × τ), Nat.cast_mul]

end Cell

open Cell

theorem ParityLuck : Accounting.Guarantees.FillerLuck.ParityLuck := parityLuck

theorem FillerParityBalanced : Accounting.Guarantees.FillerLuck.FillerParityBalanced := fun K =>
  ⟨balanced_bits toF2 K, balanced_bits _ K, balanced_bu toF2 K, balanced_bu _ K⟩

theorem FillerCellLuck : Accounting.Guarantees.FillerLuck.FillerCellLuck := by
  intro K ρ ν _ _ _ _ r s R
  obtain ⟨h1, h2, h3, h4⟩ := FillerParityBalanced K
  exact ⟨fun W hW => tile_cell_le _ _ h1 W r s R hW, fun W hW => tile_cell_le _ _ h2 W r s R hW,
    fun W hW => tile_cell_le _ _ h3 W r s R hW, fun W hW => tile_cell_le _ _ h4 W r s R hW⟩

theorem FillerLuckAt : Accounting.Guarantees.FillerLuck.FillerLuckAt := by
  intro K ρ ν Y A _ _ _ _ _ _ φ c S g hc hg
  exact luckAt_full φ c hc g hg

theorem FillerBudget : Accounting.Guarantees.FillerLuck.FillerBudget := by
  intro K ρ ν Y A _ _ _ _ _ _ φ c S g hc hg
  exact budget_spot φ c hc g hg

theorem FillerBlocksDeficit : Accounting.Guarantees.FillerLuck.FillerBlocksDeficit := by
  intro K ρ ν Y A Ω₀ _ _ _ _ _ _ _ φ c p₀ δ S g hc hp₀ hδ hg
  have := Accounting.SecurityProofs.TS.budget_tail g 1 one_pos (budget_blocks φ c hc hp₀ g hg) (Real.log (1 / δ))
  have hexp : Real.exp (-Real.log (1 / δ)) = δ := by rw [one_div, Real.log_inv, neg_neg, Real.exp_log hδ]
  rwa [one_mul, hexp] at this

theorem PinnedFillLuckAt : Accounting.Guarantees.FillerLuck.PinnedFillLuckAt := by
  intro D C τ V _ _ F S g hg
  exact luckAt_pinned F g hg

theorem PinnedFillBudget : Accounting.Guarantees.FillerLuck.PinnedFillBudget := by
  intro D C τ V _ _ F S g hg
  exact budget_spot_pinned F g hg

theorem PinnedFillBlocksDeficit : Accounting.Guarantees.FillerLuck.PinnedFillBlocksDeficit := by
  intro D C τ V Ω₀ _ _ _ F p₀ δ S g hp₀ hδ hg
  have := Accounting.SecurityProofs.TS.budget_tail g 1 one_pos (budget_blocks_pinned F hp₀ g hg)
    (Real.log (1 / δ))
  have hexp : Real.exp (-Real.log (1 / δ)) = δ := by rw [one_div, Real.log_inv, neg_neg, Real.exp_log hδ]
  rwa [one_mul, hexp] at this

end Accounting.SecurityProofs.FillerLuck
