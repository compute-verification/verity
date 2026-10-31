import Proofs.Pouw.Lead.Statement.Gap

/-!
# The tail's fixed gap at the three points and decode

`gapFixedAt` evaluates `Pouw.Lead.gapFixed` as `Statement.gapAt` evaluates the gap. A call of the lead in its
variables (`fixedVars`, `Statement.leadVars` with `gapFixed` in place of `gapRegs`) has the fixed gap
`gapFixedExpr`, linear in `p K`, `p S`, `p`, `K` and the chunks once the point fixes `q`, and `linarith` bounds it by
served stock from `fixedProducts`.
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead Pouw.Guarantees.Lead Pouw.SecurityProofs.Lead.Statement

/-- The fixed gap of a call of `16 q` rows, `8 p` columns, inner dimension `K`, `S` steps and `ch` chunks, at the
lead's constants: the head, the coin-branch mean and slack, and `p*`. -/
def gapFixedExpr (q p K S ch : ℚ) : ℚ :=
  23 * ch + 128 * (q * p * S) * (4 / 7 ^ 8 + 1 / 2 ^ 16) + 8 / 10000 * (16 * q * (8 * p) * K / 32) +
    35 / 1000000 * (16 * q * (8 * p) * K / 32)

theorem gapFixed_eq {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (q p : ℕ) (hm : L.m = 16 * q)
    (hn : L.n = 8 * p) : gapFixed L C = gapFixedExpr q p L.K L.S C.chunks := by
  have hrb : L.rowBlocks = q := by simp only [Layout.rowBlocks, hm]; omega
  have hcb : L.colBlocks = p := by simp only [Layout.colBlocks, hn]; omega
  simp only [gapFixed, gapFixedExpr, headGap, coinBranchMean, coinSlack, pStarGap, tileWords, stockRegs, hrb, hcb,
    hm, hn]
  push_cast
  ring

/-- **A call of the lead in its variables**, for the fixed gap: `Statement.leadVars`'s variables with
`gapFixedExpr`, over `n, K ≥ 4,096`. -/
theorem fixedVars {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (n q : ℕ) (hm : L.m = 16 * q)
    (hn : L.n = 8 * ((n + 7) / 8)) (hS : L.S = stepsFor L.K) (hcw : L.chunkWords = 65536) (hn4 : 4096 ≤ n)
    (hK : 4096 ≤ L.K) (hC : ChunksOf L C) :
    ∃ p S ch : ℚ, 512 ≤ p ∧ 8 * p ≤ n + 7 ∧ 28 * S ≤ L.K + 34 ∧
      65536 * ch ≤ 16 * q * (8 * p) * S + 65535 ∧ gapFixed L C = gapFixedExpr q p L.K S ch := by
  have hp : 512 ≤ (n + 7) / 8 := by omega
  have hch : 65536 * C.chunks ≤ 16 * q * (8 * ((n + 7) / 8)) * L.S + 65535 := by
    rw [hC, wordChunks, hcw, hm, hn]
    have := Nat.div_mul_le_self (16 * q * (8 * ((n + 7) / 8)) * L.S + 65536 - 1) 65536
    omega
  refine ⟨((n + 7) / 8 : ℕ), L.S, C.chunks, by exact_mod_cast hp, ?_, ?_, by exact_mod_cast hch, ?_⟩
  · have : 8 * ((n + 7) / 8) ≤ n + 7 := Nat.mul_div_le (n + 7) 8
    exact_mod_cast this
  · exact_mod_cast hS ▸ stepsFor_le L.K
  · rw [gapFixed_eq C q _ hm hn]

/-- The products `linarith` reads at `n, K ≥ 4,096`: `28 p S ≤ p K + 34 p`, `4,096 p ≤ p K`, `512 K ≤ p K` and
`512 · 4,096 ≤ p K`. -/
theorem fixedProducts {p K S : ℚ} (hp : 512 ≤ p) (hK : 4096 ≤ K) (hS : 28 * S ≤ K + 34) :
    28 * (p * S) ≤ p * K + 34 * p ∧ 4096 * p ≤ p * K ∧ 512 * K ≤ p * K ∧ 512 * 4096 ≤ p * K :=
  ⟨by nlinarith, by nlinarith, by nlinarith, by nlinarith⟩

/-- **A bound at a point** for the fixed gap: its share of served stock `R n K / 32` is at most `X` wherever
`gapFixedExpr` is at most `X` times it, over the variables' range with `8 p ≤ n + 7`. -/
theorem fixedServed_le {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (n q R : ℕ) (X : ℚ) (hm : L.m = 16 * q)
    (hn : L.n = 8 * ((n + 7) / 8)) (hr : L.rows = R) (hcols : L.cols = n) (hS : L.S = stepsFor L.K)
    (hcw : L.chunkWords = 65536) (hR : 0 < R) (hn4 : 4096 ≤ n) (hK : 4096 ≤ L.K) (hC : ChunksOf L C)
    (h : ∀ p K S ch u : ℚ, 512 ≤ p → 4096 ≤ K → 28 * S ≤ K + 34 → 65536 * ch ≤ 16 * q * (8 * p) * S + 65535 →
      8 * p ≤ u + 7 → gapFixedExpr q p K S ch ≤ X * (R * u * K / 32)) :
    gapFixedServed L C ≤ X := by
  obtain ⟨p, S, ch, hp, hpn, hS', hch, hg⟩ := fixedVars C n q hm hn hS hcw hn4 hK hC
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
arithmetic over `n, K ≥ 4,096`, as `Statement.gapAt`: at each point `fixedProducts` and `(8 p − 7) K ≤ n K`, then
`linarith`. -/
theorem gapFixedAt : GapFixedAt := fun T n K hn hK => by
  refine ⟨fun C hC => fixedServed_le C n 4 64 _ rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_,
    fun C hC => fixedServed_le C n 32 512 _ rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_,
    fun C hC => fixedServed_le C n 512 8192 _ rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_,
    fun C hC => fixedServed_le C n 1 1 _ rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_⟩ <;>
  · intro p K S ch u hp hK hS hch hu
    obtain ⟨h₁, h₂, h₃, h₄⟩ := fixedProducts hp hK hS
    have h₅ : (8 * p - 7) * K ≤ u * K := by nlinarith
    push_cast at hch ⊢
    simp only [gapFixedExpr]
    linarith

end Pouw.SecurityProofs.Lead.Tail
