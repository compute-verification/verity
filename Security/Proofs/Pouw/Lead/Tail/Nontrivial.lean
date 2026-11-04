import Proofs.Pouw.Lead.Tail.G
import Proofs.Pouw.Lead.Tail.Gap

/-!
# The tail form's nontrivial lemma

`nontrivialTail`: at the three points the fixed gap is at most 0.127% of served stock (`gapFixedAt`), below the honest
work, which is at least the words, `m n S ≥ m n K / 28`; and at most `2^30` tiles `etaLead ≤ 2^−34` (`tailAt`).
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead Pouw.Guarantees.Lead

/-- The fixed gap is below the honest work where its share of served stock is at most 1%: the honest work is at
least the words, `m n S ≥ m n K / 28`, more than served stock `rows · cols · K / 32 ≤ m n K / 32`. -/
theorem gapFixed_lt_honest {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (hshare : gapFixedServed L C ≤ 1 / 100)
    (hpos : 0 < servedRegs L) (hrows : L.rows ≤ L.m) (hcols : L.cols ≤ L.n) (hS : L.K ≤ 28 * L.S) :
    gapFixed L C < honestRegs L C := by
  rw [gapFixedServed, div_le_iff₀ hpos] at hshare
  have hsv : (L.rows * L.cols * L.K : ℚ) ≤ L.m * L.n * L.K := by
    have h1 : (L.rows : ℚ) ≤ L.m := by exact_mod_cast hrows
    have h2 : (L.cols : ℚ) ≤ L.n := by exact_mod_cast hcols
    have h3 : (L.rows * L.cols : ℚ) ≤ L.m * L.n := mul_le_mul h1 h2 (Nat.cast_nonneg _) (Nat.cast_nonneg _)
    exact mul_le_mul_of_nonneg_right h3 (Nat.cast_nonneg _)
  have hKS : (L.m * L.n * L.K : ℚ) ≤ 28 * (L.m * L.n * L.S) := by
    have : (L.K : ℚ) ≤ 28 * L.S := by exact_mod_cast hS
    have : (0 : ℚ) ≤ L.m * L.n := by positivity
    nlinarith
  have hw : wordsRegs L ≤ honestRegs L C := by
    have : (0 : ℚ) ≤ shaWordsRegs * wordsBlocks L C := by
      unfold shaWordsRegs wordsBlocks wordsRegs; positivity
    have : (0 : ℚ) ≤ shaARegs * aBlocks L := by unfold shaARegs aBlocks; positivity
    unfold honestRegs; linarith
  unfold servedRegs at hpos hshare
  unfold wordsRegs at hw
  linarith

/-- **The tail form's nontrivial lemma** (`Pouw.Guarantees.Lead.NontrivialTail`). -/
theorem nontrivialTail : NontrivialTail := by
  intro n K hn hK L hL
  have hS : K ≤ 28 * stepsFor K := by simp only [stepsFor]; omega
  have pos : ∀ T m : ℕ, 0 < m → 0 < servedRegs (lead T m n K) := fun T m hm => by
    show (0 : ℚ) < (m * n * K : ℚ) / 32
    have : (0 : ℚ) < m := by exact_mod_cast hm
    have : (0 : ℚ) < n := by exact_mod_cast (by omega : 0 < n)
    have : (0 : ℚ) < K := by exact_mod_cast (by omega : 0 < K)
    positivity
  have rows : ∀ T m : ℕ, (lead T m n K).rows ≤ (lead T m n K).m := fun T m => by
    show m ≤ padRows m
    unfold padRows; omega
  have cols : ∀ T m : ℕ, (lead T m n K).cols ≤ (lead T m n K).n := fun T m => by
    show n ≤ padCols n
    unfold padCols; omega
  have heta : L.rowBlocks * L.colBlocks * L.S ≤ 2 ^ 30 → etaLead L < 1 := fun ht => by
    have hL' : L ∈ [pt512x64 n K, pt512x512 n K, pt8192 n K, lead 1 1 n K] := by
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hL ⊢
      tauto
    exact (tailAt 1 n K hn hK L hL' ht).trans_lt (by norm_num)
  obtain ⟨h64, h512, h8192, -⟩ := gapFixedAt 1 n K hn hK
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hL
  rcases hL with rfl | rfl | rfl
  · exact ⟨fun C hC => gapFixed_lt_honest C ((h64 C hC).trans (by norm_num)) (pos 512 64 (by norm_num))
      (rows 512 64) (cols 512 64) hS, heta⟩
  · exact ⟨fun C hC => gapFixed_lt_honest C ((h512 C hC).trans (by norm_num)) (pos 512 512 (by norm_num))
      (rows 512 512) (cols 512 512) hS, heta⟩
  · exact ⟨fun C hC => gapFixed_lt_honest C ((h8192 C hC).trans (by norm_num)) (pos 8192 8192 (by norm_num))
      (rows 8192 8192) (cols 8192 8192) hS, heta⟩

end Pouw.SecurityProofs.Lead.Tail
