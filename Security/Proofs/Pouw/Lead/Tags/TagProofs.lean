import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Row tags against drawn weights: how often two rows' words agree (proofs)

Row tags put `Σ_e d_e·(g_e − h)` on a column whose tag weights are digits `g ∈ [0, q)^ι`, read as `g − h`, where `d` is
the difference of two rows' tags and the drawn weights `G : Fin n → ι → Fin q` hold one column's digits each. Two rows
with the same data and coins have equal words in column `j` exactly when that term is `0`; with other coins, when it
equals their coin difference there.

* **At most `1/q`** (`col_card_le`): for `d ≠ 0` and any target, at most `q^(|ι|−1)` of the `q^|ι|` digit vectors hit it.
  Over a tile of columns the draws are independent (`tile_card_le`), so two rows' tiles agree on all of `J` for at most
  a `q^−|J|` share of the draws.
* **At least `1/q` for a one-digit difference** (`col_card_single`): for `d = δ·e_k` and a target `δ·(s − h)`, exactly
  `q^(|ι|−1)`. Summed over the draws (`tile_sum_agree`), the expected number of agreeing columns in `J` is `|J|/q`.
* **The success set** (`tag_success_set`): over any family of row pairs, tiles and tries fixed before the draw, the
  draws on which some pair's tiles are equal are at most the sum of the members' `q^−|J|` shares.

Exact tile equality is what these bound. A run that patches a tile from another word by word (the chunk model's
`pathRun_attained`) saves the agreeing columns instead, `|J|/q` of them on average for a one-digit difference.
-/

namespace Pouw.Dimension.Tag

open Finset

variable {ι : Type} [Fintype ι] [DecidableEq ι] {q : ℕ}

/-- A column's tag term for the tag difference `d` and the column's digits `g`, read as `g − h`. -/
def tagTerm (h : ℤ) (d : ι → ℤ) (g : ι → Fin q) : ℤ := ∑ e, d e * ((g e : ℤ) - h)

theorem card_ne (k : ι) : Fintype.card {e // e ≠ k} = Fintype.card ι - 1 := by
  rw [Fintype.card_subtype_compl, Fintype.card_subtype_eq]

theorem tagTerm_split (h : ℤ) (d : ι → ℤ) (k : ι) (g : ι → Fin q) :
    tagTerm h d g = d k * ((g k : ℤ) - h) + ∑ e ∈ univ.erase k, d e * ((g e : ℤ) - h) := by
  unfold tagTerm
  rw [← Finset.add_sum_erase _ _ (mem_univ k)]

/-- **At most `1/q` of a column's draws hit a target** when the tag difference is nonzero. -/
theorem col_card_le (h : ℤ) (d : ι → ℤ) {k : ι} (hk : d k ≠ 0) (c : ℤ) :
    (univ.filter fun g : ι → Fin q => tagTerm h d g = c).card ≤ q ^ (Fintype.card ι - 1) := by
  let f : (ι → Fin q) → ({e // e ≠ k} → Fin q) := fun g e => g e
  have hinj : Set.InjOn f (univ.filter fun g : ι → Fin q => tagTerm h d g = c) := by
    intro g hg g' hg' hfg
    simp only [coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at hg hg'
    have hoff : ∀ e, e ≠ k → g e = g' e := fun e he => congrFun hfg ⟨e, he⟩
    have hrest : ∑ e ∈ univ.erase k, d e * ((g e : ℤ) - h) = ∑ e ∈ univ.erase k, d e * ((g' e : ℤ) - h) :=
      Finset.sum_congr rfl fun e he => by rw [hoff e (ne_of_mem_erase he)]
    have hkk : d k * ((g k : ℤ) - h) = d k * ((g' k : ℤ) - h) := by
      have := hg.trans hg'.symm
      rw [tagTerm_split h d k g, tagTerm_split h d k g', hrest] at this
      linarith
    have hk' : (g k : ℤ) = g' k := by have := mul_left_cancel₀ hk hkk; linarith
    funext e
    by_cases he : e = k
    · subst he; exact Fin.ext (by exact_mod_cast hk')
    · exact hoff e he
  calc _ ≤ (univ : Finset ({e // e ≠ k} → Fin q)).card :=
        card_le_card_of_injOn f (fun _ _ => mem_coe.2 (mem_univ _)) hinj
    _ = q ^ (Fintype.card ι - 1) := by rw [card_univ, Fintype.card_fun, Fintype.card_fin, card_ne]

/-- **Exactly `1/q` for a one-digit difference**: with `d = δ·e_k` and the target `δ·(s − h)`, the draws that hit it are
those with digit `s` at `k`, `q^(|ι|−1)` of them. -/
theorem col_card_single (h : ℤ) (k : ι) {δ : ℤ} (hδ : δ ≠ 0) (s : Fin q) :
    (univ.filter fun g : ι → Fin q => tagTerm h (Pi.single k δ) g = δ * ((s : ℤ) - h)).card =
      q ^ (Fintype.card ι - 1) := by
  refine le_antisymm (col_card_le h _ (k := k) (by simpa using hδ) _) ?_
  let f : ({e // e ≠ k} → Fin q) → (ι → Fin q) := fun φ e => if he : e = k then s else φ ⟨e, he⟩
  have hmaps : ∀ φ ∈ (univ : Finset ({e // e ≠ k} → Fin q)),
      f φ ∈ univ.filter fun g : ι → Fin q => tagTerm h (Pi.single k δ) g = δ * ((s : ℤ) - h) := by
    intro φ _
    refine mem_filter.2 ⟨mem_univ _, ?_⟩
    unfold tagTerm
    rw [Finset.sum_eq_single k (fun e _ he => by simp [he]) (by simp)]
    simp [f]
  have hinj : Set.InjOn f (univ : Finset ({e // e ≠ k} → Fin q)) := by
    intro φ _ φ' _ hff
    funext ⟨e, he⟩
    have := congrFun hff e
    simpa [f, he] using this
  calc q ^ (Fintype.card ι - 1) = (univ : Finset ({e // e ≠ k} → Fin q)).card := by
        rw [card_univ, Fintype.card_fun, Fintype.card_fin, card_ne]
    _ ≤ _ := card_le_card_of_injOn f hmaps hinj

section Tiles

variable {β : Type} [Fintype β] [DecidableEq β] {n : ℕ}

theorem card_filter_forall (t : Fin n → Finset β) :
    (univ.filter fun G : Fin n → β => ∀ j, G j ∈ t j).card = ∏ j, (t j).card := by
  rw [← Fintype.card_piFinset]
  congr 1
  ext G
  simp [Fintype.mem_piFinset]

/-- **A tile agrees for at most a `1/q` share of the draws a column**: the draws meeting a condition on each column of
`J` number at most the product of each condition's count, times the free columns' `|β|`. -/
theorem tile_card_le (P : Fin n → β → Prop) [∀ j b, Decidable (P j b)] (J : Finset (Fin n)) :
    (univ.filter fun G : Fin n → β => ∀ j ∈ J, P j (G j)).card =
      ∏ j, (if j ∈ J then (univ.filter (P j)).card else Fintype.card β) := by
  have := card_filter_forall (fun j => if j ∈ J then univ.filter (P j) else (univ : Finset β))
  calc _ = (univ.filter fun G : Fin n → β => ∀ j, G j ∈ (if j ∈ J then univ.filter (P j) else univ)).card := by
        congr 1
        ext G
        simp only [mem_filter, mem_univ, true_and]
        constructor
        · intro hG j
          by_cases hj : j ∈ J <;> simp [hj, hG]
        · intro hG j hj
          simpa [hj] using hG j
    _ = _ := by
        rw [this]
        exact Finset.prod_congr rfl fun j _ => by by_cases hj : j ∈ J <;> simp [hj]

theorem card_filter_at (P : β → Prop) [DecidablePred P] (j : Fin n) :
    (univ.filter fun G : Fin n → β => P (G j)).card = (univ.filter P).card * Fintype.card β ^ (n - 1) := by
  have := tile_card_le (fun _ => P) {j}
  simp only [mem_singleton, forall_eq] at this
  rw [this, ← Finset.mul_prod_erase univ _ (mem_univ j), if_pos rfl]
  congr 1
  rw [Finset.prod_congr rfl (g := fun _ => Fintype.card β) fun i hi => by simp [(mem_erase.1 hi).1], prod_const,
    card_erase_of_mem (mem_univ j), card_univ, Fintype.card_fin]

/-- **The expected number of agreeing columns**: summed over every draw, the columns of `J` meeting their condition
total `Σ_{j∈J}` of each condition's count, times `|β|^(n−1)`. -/
theorem tile_sum_agree (P : Fin n → β → Prop) [∀ j b, Decidable (P j b)] (J : Finset (Fin n)) :
    ∑ G : Fin n → β, (J.filter fun j => P j (G j)).card =
      ∑ j ∈ J, (univ.filter (P j)).card * Fintype.card β ^ (n - 1) := by
  simp only [card_filter]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [← card_filter, ← card_filter]
  exact card_filter_at (P j) j

end Tiles

variable {n : ℕ}

/-- **Row tags: two rows' words agree in `|J|/q` columns on average** when their tags differ in one digit by `δ ≠ 0`
and their coin differences in the columns of `J` are `δ·(s j − h)`: over every draw of the tag weights, the agreeing
columns of `J` total `|J|·q^(|ι|−1)·(q^|ι|)^(n−1)`, a `1/q` share of `|J|` times the draws. -/
theorem tag_sum_agree (h : ℤ) (k : ι) {δ : ℤ} (hδ : δ ≠ 0) (J : Finset (Fin n)) (s : Fin n → Fin q) :
    ∑ G : Fin n → ι → Fin q,
        (J.filter fun j => tagTerm h (Pi.single k δ) (G j) = δ * ((s j : ℤ) - h)).card =
      J.card * q ^ (Fintype.card ι - 1) * (q ^ Fintype.card ι) ^ (n - 1) := by
  rw [tile_sum_agree (fun j g => tagTerm h (Pi.single k δ) g = δ * ((s j : ℤ) - h)) J]
  rw [Finset.sum_congr rfl fun j _ => by rw [col_card_single h k hδ (s j)]]
  rw [sum_const, Fintype.card_fun, Fintype.card_fin, nsmul_eq_mul, Nat.cast_id]
  ring

/-- **Two rows' tiles are equal for at most a `q^−|J|` share of the draws** when their tags differ (`d k ≠ 0`), for any
coin differences `c`. -/
theorem tag_tile_le (h : ℤ) (d : ι → ℤ) {k : ι} (hk : d k ≠ 0) (J : Finset (Fin n)) (c : Fin n → ℤ) :
    (univ.filter fun G : Fin n → ι → Fin q => ∀ j ∈ J, tagTerm h d (G j) = c j).card ≤
      (q ^ (Fintype.card ι - 1)) ^ J.card * (q ^ Fintype.card ι) ^ (n - J.card) := by
  rw [tile_card_le (fun j g => tagTerm h d g = c j) J]
  calc ∏ j, (if j ∈ J then (univ.filter fun g : ι → Fin q => tagTerm h d g = c j).card
          else Fintype.card (ι → Fin q))
      ≤ ∏ j, (if j ∈ J then q ^ (Fintype.card ι - 1) else q ^ Fintype.card ι) := by
        refine Finset.prod_le_prod fun j _ => ?_
        by_cases hj : j ∈ J
        · simpa [hj] using col_card_le h d hk (c j)
        · simp [hj]
    _ = _ := by
        rw [Finset.prod_ite, prod_const, prod_const, filter_mem_eq_inter, univ_inter]
        congr 2
        rw [filter_not, card_sdiff_of_subset (by simp), card_univ, Fintype.card_fin]
        simp

/-- **The success set**: for any family `R` of (row pair, tile, try) fixed before the draw, each with tags differing
(`d r (k r) ≠ 0`) and coin differences `c r`, the draws on which some member's tiles are equal number at most the sum
of `tag_tile_le`'s bounds, a `Σ_r q^−|J r|` share. A prover that tries `T` inputs chosen before the draw is the family
with `T` times the members. -/
theorem tag_success_set {R : Type} (F : Finset R) (h : ℤ) (d : R → ι → ℤ) (k : R → ι)
    (hk : ∀ r ∈ F, d r (k r) ≠ 0) (J : R → Finset (Fin n)) (c : R → Fin n → ℤ) :
    (univ.filter fun G : Fin n → ι → Fin q => ∃ r ∈ F, ∀ j ∈ J r, tagTerm h (d r) (G j) = c r j).card ≤
      ∑ r ∈ F, (q ^ (Fintype.card ι - 1)) ^ (J r).card * (q ^ Fintype.card ι) ^ (n - (J r).card) := by
  calc _ ≤ (F.biUnion fun r => univ.filter fun G : Fin n → ι → Fin q =>
          ∀ j ∈ J r, tagTerm h (d r) (G j) = c r j).card := by
        refine card_le_card fun G hG => ?_
        obtain ⟨r, hr, hG⟩ := (mem_filter.1 hG).2
        exact mem_biUnion.2 ⟨r, hr, mem_filter.2 ⟨mem_univ _, hG⟩⟩
    _ ≤ ∑ r ∈ F, (univ.filter fun G : Fin n → ι → Fin q => ∀ j ∈ J r, tagTerm h (d r) (G j) = c r j).card :=
        card_biUnion_le
    _ ≤ _ := sum_le_sum fun r hr => tag_tile_le h (d r) (hk r hr) (J r) (c r)

end Pouw.Dimension.Tag
