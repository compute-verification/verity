import Definitions.Pouw.Lead.Gap
import Proofs.Pouw.Lead.Tags.Exact

/-!
# Tag distinctness at tile grain, and the words' exactness

`repeats_le`: committed tiles repeat rarely, so copying whole tiles saves at most the coin-branch term. `word_exact`:
the served 32-bit register holds each committed word exactly (the served layout's completeness; `G` doesn't read it).

`not_repeats_le`: as stated, `repeats_le` is false at `K = 0`, where stock and so `coinBranchGap` are zero while a
32 × 8 call's two step-0 tiles agree at some coins.
-/

namespace Pouw.SecurityProofs.Lead.Tags

open Pouw.Lead

/-- **Tiles repeat rarely**: for any data, the committed tiles less their distinct values, in words
and averaged over the coins, are at most `coinBranchGap`. Reads `word`, `rowTag`, `colCode`, `rotWeight` and
`tileOf`. -/
theorem repeats_le (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ) :
    ∑ ω : Coins (lead T m n K), (tileWords * repeats (lead T m n K) κ ω : ℚ) ≤
      Fintype.card (Coins (lead T m n K)) * coinBranchGap (lead T m n K) := by
  sorry

/-- The coins with every tag weight and step-1 bit zero. -/
def zeroTagCoins (L : Pouw.Lead.Layout) : Coins L := (fun _ _ => 0, fun _ _ => 0, fun _ _ => 3, fun _ _ _ _ => 0)

/-- At zero data and `zeroTagCoins`, every step-0 word is `−β_j`. -/
theorem word_zeroTagCoins (L : Pouw.Lead.Layout) (hS : L.S = 1) (i : Fin L.m) (t : Fin L.S) (j : Fin L.n) :
    word L (fun _ _ _ => 0) (zeroTagCoins L) i t j = -bias L.S j := by
  have ht : (t : ℕ) = 0 := by omega
  unfold word
  rw [add_eq_left]
  refine Finset.sum_eq_zero fun τ _ => ?_
  have hτ : (τ : ℕ) = 0 := by omega
  rw [ite_eq_right_iff]
  intro _
  simp [coinTerm, hτ, zeroTagCoins, Coins.G]

/-- **`repeats_le` is false at `K = 0`**: stock and so `coinBranchGap` are zero, but at zero data and
`zeroTagCoins` the call's two step-0 tiles of a 32 × 8 call are both `−β`. -/
theorem not_repeats_le : ¬ ∀ (T m n K : ℕ)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ),
    ∑ ω : Coins (lead T m n K), (tileWords * repeats (lead T m n K) κ ω : ℚ) ≤
      Fintype.card (Coins (lead T m n K)) * coinBranchGap (lead T m n K) := by
  intro h
  have h0 := h 0 32 8 0 fun _ _ _ => 0
  set L := lead 0 32 8 0 with hL
  have hS : L.S = 1 := rfl
  have hgap : coinBranchGap L = 0 := by simp [coinBranchGap, stockRegs, hL, lead]
  rw [hgap, mul_zero] at h0
  have htile : ∀ rb < 2, tileOf L (fun _ _ _ => 0) (zeroTagCoins L) rb 0 ⟨0, by omega⟩ =
      fun p : Fin tileWords => -bias L.S ((p : ℕ) % 8) := by
    intro rb hrb
    funext p
    have hp : (p : ℕ) < 128 := p.2
    unfold tileOf
    rw [dite_eq_left ⟨by simp only [hL, lead, padRows]; omega, by simp only [hL, lead, padCols]; omega⟩,
      word_zeroTagCoins L hS]
    simp
  have hidx : (tileIdx L).card = 2 := by simp [tileIdx, hL, lead, Layout.rowBlocks, Layout.colBlocks, padRows,
    padCols, stepsFor]
  have himg : ((tileIdx L).image fun x => tileOf L (fun _ _ _ => 0) (zeroTagCoins L) x.1 x.2.1 x.2.2).card ≤ 1 := by
    refine Finset.card_le_one.2 fun a ha b hb => ?_
    have key : ∀ y ∈ (tileIdx L).image fun x => tileOf L (fun _ _ _ => 0) (zeroTagCoins L) x.1 x.2.1 x.2.2,
        y = fun p : Fin tileWords => -bias L.S ((p : ℕ) % 8) := by
      intro y hy
      obtain ⟨⟨rb, cb, t⟩, hx, rfl⟩ := Finset.mem_image.1 hy
      simp only [tileIdx, Finset.mem_product, Finset.mem_range, Finset.mem_univ, and_true, hL, lead,
        Layout.rowBlocks, Layout.colBlocks, padRows, padCols] at hx
      have hcb : cb = 0 := by omega
      have ht : t = ⟨0, by omega⟩ := Fin.ext (by omega)
      subst hcb ht
      exact htile rb (by omega)
    rw [key a ha, key b hb]
  have hrep : 1 ≤ repeats L (fun _ _ _ => 0) (zeroTagCoins L) := by
    unfold repeats
    omega
  have hle := Finset.single_le_sum (f := fun ω => (tileWords * repeats L (fun _ _ _ => 0) ω : ℚ))
    (fun ω _ => by positivity) (Finset.mem_univ (zeroTagCoins L))
  have : (1 : ℚ) ≤ repeats L (fun _ _ _ => 0) (zeroTagCoins L) := by exact_mod_cast hrep
  have htw : (tileWords : ℚ) = 128 := by norm_num [tileWords]
  simp only [htw] at hle h0
  linarith

/-- **The words fit a 32-bit register** for `K ≤ 65,536` and data in range. -/
theorem word_exact (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (hK : K ≤ 65536) (hκ : DataInRange (lead T m n K) κ) :
    ∀ ω i t j, -(2 ^ 31) ≤ word (lead T m n K) κ ω i t j ∧ word (lead T m n K) κ ω i t j < 2 ^ 31 :=
  fun ω i t j => lead_word_int32 (by omega) hκ ω i t j

end Pouw.SecurityProofs.Lead.Tags
