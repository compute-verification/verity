import Proofs.Pouw.Lead.CoinBranch.GrindProofs

/-!
# Coin branch: a program chosen from the coins, in `MH100w`, cell by cell (proofs; draft semantics, pending review)

`MH100w`'s programs are straight-line and fixed before the coins. A program that reads coin bits and branches is, at
each grid point `x`, the straight-line program `prog x` it chose there, run on every grid point where it makes the same
choice: its cell `{y | prog y = prog x}`. `DistinctWritesMH100Floor` holds for every set of grid points, so it applies
cell by cell (`adaptive_floor`): the chosen program pays `c` for each target that stays pairwise distinct and non-free
on its cell.

With `lost_le`'s window conditions (`adaptive_lost`), a cell of measure `q^−g` loses at most `g·M + |Ec|` targets, `M`
the most words whose window holds one block of digits: the adaptive bound is `c·(|Ω| − g·M − |Ec|)` at every point.

What reading costs is not here: a branch or predicate per coin bit read is priced separately (`adaptive_total`). The
targets that merge on a cell are the design's: a word that equals another on every point of a cell, or a free value
there, earns nothing at the points of that cell. For served sparse noise (`ssnWordB_eq_of_zero`), a row's words after
two steps with equal data constants merge on the cell where the bits between them are 0 (`ssnWordB_step_merge` for
consecutive steps). In the served 4-lane layout (`l4_step_merge`) that is one bit: reading it merges a row's `n` words
of a step into the step before on half the points, so `lost_le`'s `M` is at least `n` there. Counted
(`l4_cell_loss`): on that cell every set of words distinct and non-free is at least `n` short of the layout's
`m·n·S`, so the floor there credits at most `c·(m·n·S − n)`.
-/

namespace Pouw.SecurityProofs.Lead.CoinBranch

open Pouw.Dimension Pouw.Dimension.H100

section Cells

variable {G : Type}

/-- **The floor, cell by cell**: a well-formed `MH100w` program chosen at each grid point from the coins, whose choice at
`x` holds every target of `T` on the points where it is chosen, pays at `x` at least `c` for each target of a subset
`D` pairwise distinct and non-free on that cell. -/
theorem adaptive_floor (hDW : DistinctWritesMH100Floor) {c : ℕ} {pr : Prices100} (hc : Floor c pr)
    {F : Set (G → ℚ)} (hF : ∀ q : ℚ, (fun _ : G => q) ∈ F) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ))
    (hL : ∀ f ∈ L, f ∈ F) (prog : G → List (Op100 G)) (hwf : ∀ x, ∀ o ∈ prog x, o.WFw pr F)
    (T : Finset (G → ℚ)) (hout : ∀ x, OutputsOn fpSem {y | prog y = prog x} (prog x) L T)
    (x : G) (D : Finset (G → ℚ)) (hD : D ⊆ T) (hDd : DistinctOn {y | prog y = prog x} F D) :
    c * D.card ≤ cost100 pr (prog x) :=
  hDW c pr hc G F hF {y | prog y = prog x} fpSem L (prog x) D hL (hwf x) hDd fun t ht => hout x t (hD ht)

/-- **With the reads priced**: if choosing at `x` costs `read x` (its branches and predicates), the total at `x` is at
least `read x` plus `c` for each target distinct and non-free on the cell. -/
theorem adaptive_total (hDW : DistinctWritesMH100Floor) {c : ℕ} {pr : Prices100} (hc : Floor c pr)
    {F : Set (G → ℚ)} (hF : ∀ q : ℚ, (fun _ : G => q) ∈ F) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ))
    (hL : ∀ f ∈ L, f ∈ F) (prog : G → List (Op100 G)) (hwf : ∀ x, ∀ o ∈ prog x, o.WFw pr F)
    (T : Finset (G → ℚ)) (hout : ∀ x, OutputsOn fpSem {y | prog y = prog x} (prog x) L T) (read : G → ℕ)
    (x : G) (D : Finset (G → ℚ)) (hD : D ⊆ T) (hDd : DistinctOn {y | prog y = prog x} F D) :
    read x + c * D.card ≤ read x + cost100 pr (prog x) :=
  Nat.add_le_add_left (adaptive_floor hDW hc hF fpSem L hL prog hwf T hout x D hD hDd) _

open Classical in
/-- **The adaptive bound for affine targets**: on the coin digits `ι → Fin q`, a well-formed `MH100w` program chosen from
the coins, whose choice at `x` holds every target word on its cell, pays at `x` at least `c·(|Ω| − g·M − |Ec|)` when
that cell has measure at least `q^−g`, under `lost_le`'s window conditions. Each digit's worth of reading can merge
up to `M` words, so the bound is empty once a program has read `|Ω|/M` digits' worth. -/
theorem adaptive_lost {ι : Type} [Fintype ι] [DecidableEq ι] {q : ℕ} (hq : 1 < q)
    (hDW : DistinctWritesMH100Floor) {c : ℕ} {pr : Prices100} (hc : Floor c pr)
    {K U E : Type} [DecidableEq K] [DecidableEq U] [DecidableEq E]
    (blk : ι → K) (ord : K → ℕ) (hord : Function.Injective ord)
    (Ω : Finset U) (ν : U → ℕ) (hν : Set.InjOn ν Ω) (wc : U → ℚ) (wa : U → ι → ℚ) (win : U → Finset K)
    (hwin : ∀ u ∈ Ω, ∀ u' ∈ Ω, ν u' < ν u → ∃ k, blk k ∈ win u ∧ wa u k ≠ wa u' k ∧
      ∀ k', ord (blk k) < ord (blk k') → wa u k' = wa u' k')
    (hwin0 : ∀ u ∈ Ω, ∃ k, blk k ∈ win u ∧ wa u k ≠ 0 ∧ ∀ k', ord (blk k) < ord (blk k') → wa u k' = 0)
    (Ec : Finset E) (ec : E → ℚ) (ea : E → ι → ℚ)
    (F : Set ((ι → Fin q) → ℚ)) (hF : ∀ f ∈ F, (∃ c : ℚ, f = fun _ => c) ∨ ∃ e ∈ Ec, f = Ssn.aff (ec e) (ea e))
    (hFc : ∀ r : ℚ, (fun _ : ι → Fin q => r) ∈ F)
    (M : ℕ) (hM : ∀ b, (Ω.filter fun u => b ∈ win u).card ≤ M)
    (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List ((ι → Fin q) → ℚ)) (hL : ∀ f ∈ L, f ∈ F)
    (prog : (ι → Fin q) → List (Op100 (ι → Fin q))) (hwf : ∀ x, ∀ o ∈ prog x, o.WFw pr F)
    (hout : ∀ x, OutputsOn fpSem {y | prog y = prog x} (prog x) L (Ω.image fun u => Ssn.aff (wc u) (wa u)))
    (x : ι → Fin q) (g : ℕ)
    (hcell : q ^ Fintype.card ι ≤ q ^ g * (Finset.univ.filter fun y => prog y = prog x).card) :
    c * (Ω.card - (g * M + Ec.card)) ≤ cost100 pr (prog x) := by
  obtain ⟨D, hDΩ, hDd, hDc, hΩ⟩ := Grind.lost_le hq _ g hcell blk ord hord Ω ν hν wc wa win hwin hwin0 Ec ec ea F
    hF M hM
  have hcoe : (↑(Finset.univ.filter fun y => prog y = prog x) : Set (ι → Fin q)) = {y | prog y = prog x} := by
    ext y; simp
  rw [hcoe] at hDd
  have h := adaptive_floor hDW hc hFc fpSem L hL prog hwf _ hout x _ (Finset.image_subset_image hDΩ) hDd
  rw [hDc] at h
  exact le_trans (Nat.mul_le_mul_left c (by omega)) h

end Cells

section Ssn

open Finset Pouw.Dimension.Ssn

variable {m S C L : ℕ}

/-- **Two of a row's words merge where the bits they weigh differently are 0**: if row `i`'s words of column `j` after
steps `t` and `t'` have equal data constants, they are equal at every coin point where each bit with different weights
in the two is 0. For the 4-lane layout and `t' = t − 1` that is the one bit of step `t`'s lane, so a program that reads
it merges the row's `n` words of step `t` into step `t − 1`'s on half the points. -/
theorem ssnWordB_eq_of_zero (src : Fin S → Fin C) (used : Fin S → Fin L → Prop) [∀ s q, Decidable (used s q)]
    (pos : Fin S → Fin L → ℕ) (κ : Fin m → ℕ → Fin S → ℤ) (W : Fin S → Fin L → ℕ → ℤ) (i : Fin m) (j : ℕ)
    (t t' : Fin S) (hκ : κ i j t = κ i j t') (x : Bits m C)
    (hx : ∀ k, bitCoef src used pos W i j t k ≠ bitCoef src used pos W i j t' k → x k = 0) :
    ssnWordB src used pos κ W i j t x = ssnWordB src used pos κ W i j t' x := by
  unfold ssnWordB aff
  rw [hκ]
  congr 1
  refine Finset.sum_congr rfl fun k _ => ?_
  by_cases h : bitCoef src used pos W i j t k = bitCoef src used pos W i j t' k
  · rw [h]
  · simp [hx k h]

/-- Consecutive steps weigh a bit of row `i` the same unless the later step reads it. -/
theorem bitCoef_step (src : Fin S → Fin C) (used : Fin S → Fin L → Prop) [∀ s q, Decidable (used s q)]
    (pos : Fin S → Fin L → ℕ) (W : Fin S → Fin L → ℕ → ℤ) (i : Fin m) (j : ℕ) {t t' : Fin S}
    (htt : (t' : ℕ) + 1 = t) (k : Fin m × Fin C × Fin 32)
    (hk : ¬ (k.1 = i ∧ ∃ q, used t q ∧ src t = k.2.1 ∧ pos t q = k.2.2)) :
    bitCoef src used pos W i j t k = bitCoef src used pos W i j t' k := by
  unfold bitCoef
  split_ifs with hki
  · refine Finset.sum_congr rfl fun s _ => Finset.sum_congr rfl fun q _ => ?_
    by_cases hst : s = t
    · subst hst
      have hn : ¬ (used s q ∧ src s = k.2.1 ∧ pos s q = k.2.2) := fun h => hk ⟨hki, q, h⟩
      have hst' : ¬ s ≤ t' := by rw [Fin.le_iff_val_le_val]; omega
      simp only [hn, hst', and_false, ite_false]
    · have hiff : s ≤ t ↔ s ≤ t' := by
        rw [Fin.le_iff_val_le_val, Fin.le_iff_val_le_val]
        have := Fin.val_ne_of_ne hst
        omega
      simp only [hiff]
  · rfl

/-- **One step's bits decide a merge**: a row's words after steps `t − 1` and `t` with equal data constants are equal
wherever every bit step `t` reads in that row is 0. -/
theorem ssnWordB_step_merge (src : Fin S → Fin C) (used : Fin S → Fin L → Prop) [∀ s q, Decidable (used s q)]
    (pos : Fin S → Fin L → ℕ) (κ : Fin m → ℕ → Fin S → ℤ) (W : Fin S → Fin L → ℕ → ℤ) (i : Fin m) (j : ℕ)
    {t t' : Fin S} (htt : (t' : ℕ) + 1 = t) (hκ : κ i j t = κ i j t') (x : Bits m C)
    (hx : ∀ k : Fin m × Fin C × Fin 32, k.1 = i → (∃ q, used t q ∧ src t = k.2.1 ∧ pos t q = k.2.2) → x k = 0) :
    ssnWordB src used pos κ W i j t x = ssnWordB src used pos κ W i j t' x :=
  ssnWordB_eq_of_zero src used pos κ W i j t t' hκ x fun k hne => by
    by_contra hxk
    exact hne (bitCoef_step src used pos W i j htt k fun ⟨h1, h2⟩ => hxk (hx k h1 h2))

/-- **The served 4-lane layout: one bit merges a row's step into the one before.** For `t ≥ 1`, in every column whose
data constants are equal across the step, row `i`'s words after steps `t − 1` and `t` are equal wherever its bit
`l4Pos t 0` of coin word `l4Src t` is 0: half the coins, for all `n` columns at once. -/
theorem l4_step_merge (hC : S + 3 ≤ 32 * C) (κ : Fin m → ℕ → Fin S → ℤ) (i : Fin m) (j : ℕ) {t t' : Fin S}
    (htt : (t' : ℕ) + 1 = t) (hκ : κ i j t = κ i j t') (x : Bits m C)
    (hx : ∀ k : Fin m × Fin C × Fin 32, k.1 = i → k.2.1 = l4Src hC t → (k.2.2 : ℕ) = l4Pos t 0 → x k = 0) :
    ssnWordB (l4Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l4Pos s q) κ w1l i j t x =
      ssnWordB (l4Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l4Pos s q) κ w1l i j t' x := by
  refine ssnWordB_step_merge _ _ _ κ w1l i j htt hκ x fun k hk ⟨q, hq, hs, hp⟩ => hx k hk hs.symm ?_
  have hq0 : q = 0 := hq.resolve_left (by omega)
  subst hq0
  simpa using hp.symm

/-- `n` disjoint pairs inside `T`, each meeting `D ⊆ T` at most once, leave `D` at least `n` short of `T`. -/
theorem card_add_le_of_pairs {α : Type} [DecidableEq α] {n : ℕ} (T D : Finset α) (a b : Fin n → α) (hD : D ⊆ T)
    (ha : ∀ j, a j ∈ T) (hb : ∀ j, b j ∈ T)
    (hdisj : ∀ j j', j ≠ j' → a j ≠ a j' ∧ a j ≠ b j' ∧ b j ≠ a j' ∧ b j ≠ b j')
    (hone : ∀ j, ¬ (a j ∈ D ∧ b j ∈ D)) : D.card + n ≤ T.card := by
  let e : Fin n → α := fun j => if a j ∈ D then b j else a j
  have he : Function.Injective e := by
    intro j j' h
    by_contra hne
    obtain ⟨h1, h2, h3, h4⟩ := hdisj j j' hne
    simp only [e] at h
    split_ifs at h <;> contradiction
  have hdis : Disjoint D (univ.image e) := by
    rw [disjoint_left]
    intro x hxD hxE
    obtain ⟨j, -, rfl⟩ := mem_image.1 hxE
    simp only [e] at hxD
    split_ifs at hxD with h
    · exact hone j ⟨h, hxD⟩
    · exact h hxD
  have hsub : D ∪ univ.image e ⊆ T := union_subset hD fun x hx => by
    obtain ⟨j, -, rfl⟩ := mem_image.1 hx
    simp only [e]
    split_ifs
    exacts [hb j, ha j]
  calc D.card + n = (D ∪ univ.image e).card := by
        rw [card_union_of_disjoint hdis, card_image_of_injective _ he, card_univ, Fintype.card_fin]
    _ ≤ T.card := card_le_card hsub

/-- **On the cell of one bit, the 4-lane layout loses a row's `n` words.** For `t ≥ 1` and a row `i` whose data
constants are equal across step `t` in every column, any words `D` of the layout pairwise distinct and non-free on a
set `X` of coin points where row `i`'s bit `l4Pos t 0` of word `l4Src t` is 0 number at least `n` fewer than the
layout's: `adaptive_floor` there credits at most `c·(m·n·S − n)`. -/
theorem l4_cell_loss (hC : S + 3 ≤ 32 * C) (κ : Fin m → ℕ → Fin S → ℤ) (hS : 0 < S) (n : ℕ) (hn : n ≤ 28559)
    (i : Fin m) {t t' : Fin S} (htt : (t' : ℕ) + 1 = t) (hκ : ∀ j < n, κ i j t = κ i j t') (X : Set (Bits m C))
    (hX : ∀ x ∈ X, ∀ k : Fin m × Fin C × Fin 32, k.1 = i → k.2.1 = l4Src hC t → (k.2.2 : ℕ) = l4Pos t 0 → x k = 0)
    (D : Finset (Bits m C → ℚ)) (hD : D ⊆ l4Words hC κ n) (hDd : DistinctOn X (FreeB m C) D) :
    D.card + n ≤ (l4Words hC κ n).card := by
  classical
  set φ : Fin m × ℕ × Fin S → Bits m C → ℚ := fun x =>
    ssnWordB (l4Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l4Pos s q) κ w1l x.1 x.2.1 x.2.2 with hφ
  have hdom : (univ ×ˢ range n ×ˢ univ : Finset (Fin m × ℕ × Fin S)).card = m * (n * S) := by
    simp [card_product]
  have hinj : Set.InjOn φ (univ ×ˢ range n ×ˢ univ : Finset (Fin m × ℕ × Fin S)) :=
    card_image_iff.1 ((ssn1lServed_full hC κ hS n hn).2.trans hdom.symm)
  have hmem : ∀ (j : Fin n) (s : Fin S), ((i, (j : ℕ), s) : Fin m × ℕ × Fin S) ∈ (univ ×ˢ range n ×ˢ univ) :=
    fun j s => by simp
  have hne : t ≠ t' := fun h => by rw [h] at htt; omega
  refine card_add_le_of_pairs _ D (fun j => φ (i, j, t)) (fun j => φ (i, j, t')) hD
    (fun j => mem_image.2 ⟨_, hmem j t, rfl⟩) (fun j => mem_image.2 ⟨_, hmem j t', rfl⟩) ?_ ?_
  · intro j j' hjj
    have hv : (j : ℕ) ≠ j' := fun h => hjj (Fin.ext h)
    refine ⟨fun h => ?_, fun h => ?_, fun h => ?_, fun h => ?_⟩ <;>
      exact hv (congrArg (fun x : Fin m × ℕ × Fin S => x.2.1) (hinj (hmem _ _) (hmem _ _) h))
  · rintro j ⟨h1, h2⟩
    have heq : φ (i, j, t) = φ (i, j, t') :=
      hDd.1 _ h1 _ h2 fun x hx => l4_step_merge hC κ i j htt (hκ j j.isLt) x (hX x hx)
    exact hne (congrArg (fun x : Fin m × ℕ × Fin S => x.2.2) (hinj (hmem _ _) (hmem _ _) heq))

end Ssn

end Pouw.SecurityProofs.Lead.CoinBranch
