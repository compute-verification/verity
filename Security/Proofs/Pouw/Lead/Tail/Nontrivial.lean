import Proofs.Pouw.Lead.Tail.G
import Proofs.Pouw.Lead.Tail.Gap

/-!
# The tail form's nontrivial lemma

`nontrivialTail`: at the three points the fixed gap is at most 0.162% of stock (`gapFixedAt`), below the honest work,
which is at least the words, `m n S ≥ m n K / 28`; and at most `2^30` tiles `etaLead ≤ 2^−40` (`tailAt`).
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead Pouw.Guarantees.Lead

/-- The fixed gap is below the honest work where its share of stock is at most 1%: the honest work is at least the
words, `m n S ≥ m n K / 28`, more than the stock `m n K / 32`. -/
theorem gapFixed_lt_honest {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (hshare : gapFixedShare L C ≤ 1 / 100)
    (hpos : 0 < stockRegs L) (hS : L.K ≤ 28 * L.S) : gapFixed L C < honestRegs L C := by
  rw [gapFixedShare, div_le_iff₀ hpos] at hshare
  have hKS : (L.m * L.n * L.K : ℚ) ≤ 28 * (L.m * L.n * L.S) := by
    have : (L.K : ℚ) ≤ 28 * L.S := by exact_mod_cast hS
    have : (0 : ℚ) ≤ L.m * L.n := by positivity
    nlinarith
  have hw : wordsRegs L ≤ honestRegs L C := by
    have : (0 : ℚ) ≤ shaWordsRegs * wordsBlocks L C := by
      unfold shaWordsRegs wordsBlocks wordsRegs; positivity
    have : (0 : ℚ) ≤ shaARegs * aBlocks L := by unfold shaARegs aBlocks; positivity
    unfold honestRegs; linarith
  unfold stockRegs at hpos hshare
  unfold wordsRegs at hw
  linarith

/-- **The tail form's nontrivial lemma** (`Pouw.Guarantees.Lead.NontrivialTail`). -/
theorem nontrivialTail : NontrivialTail := by
  intro n K hn hK L hL
  have hS : K ≤ 28 * stepsFor K := by simp only [stepsFor]; omega
  have pos : ∀ {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (q R c : ℕ), L.m = 16 * q →
      L.n = 8 * ((n + 7) / 8) → L.rows = R → L.cap = c → L.S = stepsFor L.K → L.fwdMacs = 973078528 →
      L.chunkWords = 65536 → 0 < q → L.K = K → ChunksOf L C → 0 < stockRegs L :=
    fun C q R c hm hn' hr hc hS' hF hcw hq hKL hC => by
      obtain ⟨-, -, -, -, -, -, -, -, -, hpos⟩ := fixedVars C n q R c hm hn' hr hc hS' hF hcw hq hn (hKL ▸ hK) hC
      exact hpos
  have heta : L.rowBlocks * L.colBlocks * L.S ≤ 2 ^ 30 → etaLead L < 1 := fun ht => by
    have hL' : L ∈ [pt512x64 n K, pt512x512 n K, pt8192 n K, lead 1 1 n K] := by
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hL ⊢
      rcases hL with h | h | h <;> simp only [h, true_or, or_true]
    exact (tailAt 1 n K hn hK L hL' ht).trans_lt (by norm_num)
  obtain ⟨h64, h512, h8192, -⟩ := gapFixedAt 1 n K hn hK
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hL
  rcases hL with rfl | rfl | rfl
  · exact ⟨fun C hC => gapFixed_lt_honest C ((h64 C hC).trans (by norm_num))
      (pos C 4 64 32 rfl rfl rfl rfl rfl rfl rfl (by norm_num) rfl hC) hS, heta⟩
  · exact ⟨fun C hC => gapFixed_lt_honest C ((h512 C hC).trans (by norm_num))
      (pos C 32 512 32 rfl rfl rfl rfl rfl rfl rfl (by norm_num) rfl hC) hS, heta⟩
  · exact ⟨fun C hC => gapFixed_lt_honest C ((h8192 C hC).trans (by norm_num))
      (pos C 512 8192 32 rfl rfl rfl rfl rfl rfl rfl (by norm_num) rfl hC) hS, heta⟩

end Pouw.SecurityProofs.Lead.Tail
