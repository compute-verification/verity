import Proofs.Pouw.Lead.Statement.Gap

/-!
# The tail's fixed gap at the three points and decode

`gapFixedAt` evaluates `Pouw.Lead.gapFixed` as `Statement.gapAt` evaluates the gap. A call of the lead in its
variables (`fixedVars`, `Statement.leadVars` with `gapFixed` in place of `gapRegs`) has the fixed gap
`gapFixedExpr`, linear in `p K`, `p S`, `p`, `K` and the chunks once the point fixes `q`, the unpadded rows and the
cap, and `linarith` bounds it by the stock from `Statement.products`.
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead Pouw.Guarantees.Lead Pouw.SecurityProofs.Lead.Statement

/-- The fixed gap of a call of `16 q` rows, `8 p` columns, `R` unpadded rows, cap `c`, inner dimension `K`, `S` steps
and `ch` chunks, at the lead's constants: the head, the network, the coin-branch mean and slack, and `p*`. -/
def gapFixedExpr (q p R c K S ch : ℚ) : ℚ :=
  23 * ch +
    256 / 3 * (16 * q * (8 * p) * S + 351538 / 100 * (4 * (16 * q * (8 * p) * S) / 128 + ch) +
      344905 / 100 * (R * K / 128)) * c * R / (16 * q * 973078528) +
    128 * (q * p * S) * (4 / 7 ^ 8 + 1 / 2 ^ 16) + 8 / 10000 * (16 * q * (8 * p) * K / 32) +
    35 / 1000000 * (16 * q * (8 * p) * K / 32)

theorem gapFixed_eq {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (q p : ℕ) (hm : L.m = 16 * q)
    (hn : L.n = 8 * p) (hF : L.fwdMacs = 973078528) (hq : 0 < q) (hp : 0 < p) (hK : 0 < L.K) :
    gapFixed L C = gapFixedExpr q p L.rows L.cap L.K L.S C.chunks := by
  have hrb : L.rowBlocks = q := by simp only [Layout.rowBlocks, hm]; omega
  have hcb : L.colBlocks = p := by simp only [Layout.colBlocks, hn]; omega
  rw [gapFixed, networkGap_eq L C (by rw [hm]; positivity) (by rw [hn]; positivity) (by positivity)
    (by rw [hF]; norm_num)]
  simp only [gapFixedExpr, headGap, coinBranchMean, coinSlack, pStarGap, honestRegs, wordsRegs, wordsBlocks,
    aBlocks, shaWordsRegs, shaARegs, nMax, tileWords, stockRegs, hrb, hcb, hm, hn, hF]
  push_cast
  ring

/-- **A call of the lead in its variables**, for the fixed gap: `Statement.leadVars`'s variables with
`gapFixedExpr`. -/
theorem fixedVars {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (n q R c : ℕ) (hm : L.m = 16 * q)
    (hn : L.n = 8 * ((n + 7) / 8)) (hr : L.rows = R) (hc : L.cap = c) (hS : L.S = stepsFor L.K)
    (hF : L.fwdMacs = 973078528) (hcw : L.chunkWords = 65536) (hq : 0 < q) (hn4 : 4096 ≤ n) (hK : 4096 ≤ L.K)
    (hC : ChunksOf L C) :
    ∃ p S ch : ℚ, 512 ≤ p ∧ 8 * p ≤ n + 7 ∧ 28 * S ≤ L.K + 34 ∧
      65536 * ch ≤ 16 * q * (8 * p) * S + 65535 ∧ gapFixed L C = gapFixedExpr q p R c L.K S ch ∧
      stockRegs L = 16 * q * (8 * p) * L.K / 32 ∧ 0 < stockRegs L := by
  have hp : 512 ≤ (n + 7) / 8 := by omega
  have hch : 65536 * C.chunks ≤ 16 * q * (8 * ((n + 7) / 8)) * L.S + 65535 := by
    rw [hC, wordChunks, hcw, hm, hn]
    have := Nat.div_mul_le_self (16 * q * (8 * ((n + 7) / 8)) * L.S + 65536 - 1) 65536
    omega
  have hst : stockRegs L = 16 * q * (8 * (((n + 7) / 8 : ℕ) : ℚ)) * L.K / 32 := by
    simp only [stockRegs, hm, hn]; push_cast; ring
  refine ⟨((n + 7) / 8 : ℕ), L.S, C.chunks, by exact_mod_cast hp, ?_, ?_, by exact_mod_cast hch, ?_, hst, ?_⟩
  · have : 8 * ((n + 7) / 8) ≤ n + 7 := Nat.mul_div_le (n + 7) 8
    exact_mod_cast this
  · exact_mod_cast hS ▸ stepsFor_le L.K (by omega)
  · rw [gapFixed_eq C q _ hm hn hF hq (by omega) (by omega), hr, hc]
  · rw [hst]
    have : (0 : ℚ) < q := by exact_mod_cast hq
    have : (0 : ℚ) < ((n + 7) / 8 : ℕ) := by exact_mod_cast (by omega : 0 < (n + 7) / 8)
    have : (0 : ℚ) < L.K := by exact_mod_cast (by omega : 0 < L.K)
    positivity

/-- **A share bound at a point** for the fixed gap: `Statement.share_le` with `gapFixedExpr`. -/
theorem fixedShare_le {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (n q R c : ℕ) (B : ℚ) (hm : L.m = 16 * q)
    (hn : L.n = 8 * ((n + 7) / 8)) (hr : L.rows = R) (hc : L.cap = c) (hS : L.S = stepsFor L.K)
    (hF : L.fwdMacs = 973078528) (hcw : L.chunkWords = 65536) (hq : 0 < q) (hn4 : 4096 ≤ n) (hK : 4096 ≤ L.K)
    (hC : ChunksOf L C)
    (h : ∀ p K S ch : ℚ, 512 ≤ p → 4096 ≤ K → 28 * S ≤ K + 34 → 65536 * ch ≤ 16 * q * (8 * p) * S + 65535 →
      gapFixedExpr q p R c K S ch ≤ B * (16 * q * (8 * p) * K / 32)) :
    gapFixedShare L C ≤ B := by
  obtain ⟨p, S, ch, hp, -, hS', hch, hg, hst, hpos⟩ := fixedVars C n q R c hm hn hr hc hS hF hcw hq hn4 hK hC
  rw [gapFixedShare, div_le_iff₀ hpos, hg, hst]
  exact h p L.K S ch hp (by exact_mod_cast hK) hS' hch

/-- **A served bound at a point** for the fixed gap: `Statement.served_le` with `gapFixedExpr`. -/
theorem fixedServed_le {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (n q R c : ℕ) (X : ℚ) (hm : L.m = 16 * q)
    (hn : L.n = 8 * ((n + 7) / 8)) (hr : L.rows = R) (hcols : L.cols = n) (hc : L.cap = c)
    (hS : L.S = stepsFor L.K) (hF : L.fwdMacs = 973078528) (hcw : L.chunkWords = 65536) (hq : 0 < q) (hR : 0 < R)
    (hn4 : 4096 ≤ n) (hK : 4096 ≤ L.K) (hC : ChunksOf L C)
    (h : ∀ p K S ch u : ℚ, 512 ≤ p → 4096 ≤ K → 28 * S ≤ K + 34 → 65536 * ch ≤ 16 * q * (8 * p) * S + 65535 →
      8 * p ≤ u + 7 → gapFixedExpr q p R c K S ch ≤ X * (R * u * K / 32)) :
    gapFixedServed L C ≤ X := by
  obtain ⟨p, S, ch, hp, hpn, hS', hch, hg, -, -⟩ := fixedVars C n q R c hm hn hr hc hS hF hcw hq hn4 hK hC
  have hsv : servedRegs L = R * n * L.K / 32 := by simp only [servedRegs, hr, hcols]
  have hpos : 0 < servedRegs L := by
    rw [hsv]
    have : (0 : ℚ) < R := by exact_mod_cast hR
    have : (0 : ℚ) < n := by exact_mod_cast (by omega : 0 < n)
    have : (0 : ℚ) < L.K := by exact_mod_cast (by omega : 0 < L.K)
    positivity
  rw [gapFixedServed, div_le_iff₀ hpos, hg, hsv]
  exact h p L.K S ch n hp (by exact_mod_cast hK) hS' hch hpn

/-- **The tail's fixed gap at the three points and decode** (`Pouw.Guarantees.Lead.GapFixedAt`): rational
arithmetic over `n, K ≥ 4,096`, as `Statement.gapAt`: at each point, over `p ≥ 512` and `K ≥ 4,096`,
`Statement.products` (and at decode `(8 p − 7) K ≤ n K`), then `linarith`. -/
theorem gapFixedAt : GapFixedAt := fun T n K hn hK => by
  refine ⟨fun C hC => fixedShare_le C n 4 64 32 _ rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_,
    fun C hC => fixedShare_le C n 32 512 32 _ rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_,
    fun C hC => fixedShare_le C n 512 8192 32 _ rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_,
    fun C hC => fixedShare_le C n 4 64 320 _ rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_,
    fun C hC => fixedShare_le C n 32 512 320 _ rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_,
    fun C hC => fixedShare_le C n 512 8192 320 _ rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_,
    fun C hC => ⟨fixedServed_le C n 1 1 32 _ rfl rfl rfl rfl rfl rfl rfl rfl (by norm_num) (by norm_num) hn hK hC ?_,
      fixedShare_le C n 1 1 32 _ rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_⟩,
    fun C hC => ⟨fixedServed_le C n 1 1 320 _ rfl rfl rfl rfl rfl rfl rfl rfl (by norm_num) (by norm_num) hn hK hC ?_,
      fixedShare_le C n 1 1 320 _ rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_⟩⟩
  all_goals first
    | intro p K S ch u hp hK hS hch hu
      obtain ⟨h₁, h₂, h₃, h₄⟩ := products hp hK hS
      have h₅ : (8 * p - 7) * K ≤ u * K := by nlinarith
      push_cast at hch ⊢
      simp only [gapFixedExpr]
      linarith
    | intro p K S ch hp hK hS hch
      obtain ⟨h₁, h₂, h₃, h₄⟩ := products hp hK hS
      push_cast at hch ⊢
      simp only [gapFixedExpr]
      linarith

end Pouw.SecurityProofs.Lead.Tail
