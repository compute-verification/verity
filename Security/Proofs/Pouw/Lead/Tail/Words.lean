import Proofs.Pouw.Lead.Tail.Blocks
import Proofs.Pouw.Lead.Tags.Repeats

/-!
# What the three fiber counts share

The lead commits whole MMA tiles (`lead_m`, `lead_n`) in 64-row coin tiles, so a tile's position `8 r + c` is the word
of its row `r` and column `c` (`tileOf_at`), and a coin tile's row blocks are `4 ct + a` for `a < 4`. A block that
repeats a tile of its own has two row blocks `a < b` of its coin tile whose tiles agree (`withinAt_pair`), and a coin
tile has at most `pairs` such pairs (`pairsAt_card`). A set of functions on `Fin 64` that agree off one 16-index block
only when equal has at most `|X|^48` members (`card_le_of_block`).
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Finset Pouw.Lead

theorem lead_m (T m n K : ℕ) : (lead T m n K).m = 16 * (lead T m n K).rowBlocks := by
  simp only [Layout.rowBlocks, lead, padRows]; omega

theorem lead_n (T m n K : ℕ) : (lead T m n K).n = 8 * (lead T m n K).colBlocks := by
  simp only [Layout.colBlocks, lead, padCols]; omega

theorem lead_tiles (T m n K : ℕ) : (lead T m n K).tiles = ((lead T m n K).rowBlocks + 3) / 4 := by
  simp only [Layout.tiles, Layout.rowBlocks, lead, padRows]; omega

/-- A tile's position `8 r + c` is the word of its row `r` and column `c`. -/
theorem tileOf_at {L : Pouw.Lead.Layout} (hLm : L.m = 16 * L.rowBlocks) (hLn : L.n = 8 * L.colBlocks)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) {rb cb : ℕ} (hrb : rb < L.rowBlocks) (hcb : cb < L.colBlocks)
    (t : Fin L.S) {r c : ℕ} (hr : r < 16) (hc : c < 8) :
    tileOf L κ ω rb cb t ⟨8 * r + c, by unfold tileWords; omega⟩ =
      word L κ ω ⟨16 * rb + r, by omega⟩ t ⟨8 * cb + c, by omega⟩ := by
  have h1 : (8 * r + c) / 8 = r := by omega
  have h2 : (8 * r + c) % 8 = c := by omega
  unfold tileOf
  rw [dite_eq_left ⟨by rw [h1]; omega, by rw [h2]; omega⟩]
  congr 1 <;> exact Fin.ext (by simp only [h1, h2])

/-- The pairs `a < b` of row blocks of coin tile `ct`, both in the call. -/
def pairsAt (L : Pouw.Lead.Layout) (ct : ℕ) : Finset (ℕ × ℕ) :=
  (range 4 ×ˢ range 4).filter fun p => p.1 < p.2 ∧ 4 * ct + p.2 < L.rowBlocks

theorem pairsAt_card (L : Pouw.Lead.Layout) (ct : ℕ) : #(pairsAt L ct) ≤ pairs L := by
  have key : ∀ k ≤ 4, #((range k ×ˢ range k).filter fun p : ℕ × ℕ => p.1 < p.2) = k * (k - 1) / 2 := by decide
  calc #(pairsAt L ct) ≤ #((range (rho L) ×ˢ range (rho L)).filter fun p : ℕ × ℕ => p.1 < p.2) := by
        refine card_le_card fun p hp => ?_
        simp only [pairsAt, mem_filter, mem_product, mem_range] at hp ⊢
        unfold rho
        omega
    _ = pairs L := key _ (min_le_left _ _)

/-- **A block's own repeat is a pair's**: two of its coin tile's row blocks `a < b` whose tiles agree. -/
theorem withinAt_pair {L : Pouw.Lead.Layout} {κ : Fin L.m → Fin L.S → Fin L.n → ℤ} {ω : Coins L} {β : Blk L}
    (h : withinAt L κ ω β) : ∃ p ∈ pairsAt L β.2.1,
      tileOf L κ ω (4 * β.2.1 + p.1) β.2.2 β.1 = tileOf L κ ω (4 * β.2.1 + p.2) β.2.2 β.1 := by
  obtain ⟨rb, hrb, rb', hrb', h1, h2, hne, heq⟩ := h
  have e1 : 4 * (β.2.1 : ℕ) + rb % 4 = rb := by omega
  have e2 : 4 * (β.2.1 : ℕ) + rb' % 4 = rb' := by omega
  rcases lt_or_gt_of_ne hne with hlt | hlt
  · refine ⟨(rb % 4, rb' % 4), ?_, ?_⟩
    · simp only [pairsAt, mem_filter, mem_product, mem_range]; omega
    · simp only [e1, e2]; exact heq
  · refine ⟨(rb' % 4, rb % 4), ?_, ?_⟩
    · simp only [pairsAt, mem_filter, mem_product, mem_range]; omega
    · simp only [e1, e2]; exact heq.symm

/-- **Functions on `Fin 64` pinned by 48 of their values**: if two members of `A` that agree off block `b`'s 16
indices are equal, `A` has at most `|X|^48` members. -/
theorem card_le_of_block {X : Type} [Fintype X] (A : Finset (Fin 64 → X)) {b : ℕ} (hb : b < 4)
    (h : ∀ v ∈ A, ∀ v' ∈ A, (∀ k : Fin 64, (k : ℕ) / 16 ≠ b → v k = v' k) → v = v') :
    #A ≤ Fintype.card X ^ 48 := by
  classical
  let idx : Fin 48 → Fin 64 := fun k => ⟨if (k : ℕ) < 16 * b then k else k + 16, by split_ifs <;> omega⟩
  have hc := card_le_card_of_injOn (s := A) (t := univ) (fun v k => v (idx k)) (fun _ _ => mem_coe.2 (mem_univ _))
    fun v hv v' hv' hvv => h v hv v' hv' fun k hk => by
      have hk' : (if (k : ℕ) < 16 * b then (k : ℕ) else k - 16) < 48 := by split_ifs <;> omega
      have e : idx ⟨_, hk'⟩ = k := by
        refine Fin.ext ?_
        simp only [idx]
        split_ifs <;> omega
      rw [← e]
      exact congrFun hvv _
  simpa only [card_univ, Fintype.card_fun, Fintype.card_fin] using hc

end Pouw.SecurityProofs.Lead.Tail
