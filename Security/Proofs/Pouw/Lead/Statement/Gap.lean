import Definitions.Pouw.Guarantees.Lead

/-!
# The gap at the three points, and the nontrivial lemma

`gapAt` and `gapAt320` evaluate `Pouw.Lead.gapShare` at T 512 × 64, T 512 × 512 and T 8,192 for every call of
`n, K ≥ 4,096`, at caps 32 and 320; the worst call is at `K = 4,110`, the first `K` past 4,096 that adds a step.
`gapAtDecode` evaluates the gap at decode at both caps, as a share of served compute (`gapServed`) and then of stock.
`nontrivial` is the guarantee's nontrivial lemma. The network term is `Network.networkGap_le` and
`Network.networkGap_le320`. `admits_of_caps`: the calls the verifier admits (`Layout.Admits`) include every call of at
most `2^15` rows, `2^20` columns and an inner dimension from 1 to `2^16`. The cap doesn't enter the gap.

The proof writes a call of the lead in its variables (`leadVars`): `16 q` rows, `8 p` columns with `p ≥ 512`, `K`
with `28 S ≤ K + 34` (`stepsFor`), and `65,536` chunks at most `16 q · 8 p · S + 65,535` (`wordChunks`). The gap is
then `gapExpr`, linear in `p K`, `p S`, `p`, `K` and the chunks once the point fixes `q`, the unpadded rows and the
cap, and `linarith` bounds it by the stock `16 q · 8 p · K / 32` from `28 p S ≤ p K + 34 p`, `4,096 p ≤ p K`,
`512 K ≤ p K` and `2^21 ≤ p K` (`products`).
-/

namespace Pouw.SecurityProofs.Lead.Statement

open Pouw.Lead Pouw.Guarantees.Lead Pouw.Dimension.H100

/-- The network term without the call's `n` and `K`: `nMax · honestRegs · cap · rows / (m · fwdMacs)`. -/
theorem networkGap_eq (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) (hm : (L.m : ℚ) ≠ 0) (hn : (L.n : ℚ) ≠ 0)
    (hK : (L.K : ℚ) ≠ 0) (hF : (L.fwdMacs : ℚ) ≠ 0) :
    networkGap L C = nMax * honestRegs L C * L.cap * L.rows / (L.m * L.fwdMacs) := by
  unfold networkGap betaLead wMac Layout.callBits
  field_simp

/-- The gap of a call of `16 q` rows, `8 p` columns, `R` unpadded rows, cap `c`, inner dimension `K`, `S` steps and
`ch` chunks, at the lead's constants. -/
def gapExpr (q p R c K S ch : ℚ) : ℚ :=
  23 * ch + 16064 / 1000000 * (8 * p * K / 32) + 84 / 100000 * (16 * q * (8 * p) * K / 32) +
    35 / 1000000 * (16 * q * (8 * p) * K / 32) +
    256 / 3 * (16 * q * (8 * p) * S + 351538 / 100 * (4 * (16 * q * (8 * p) * S) / 128 + ch) +
      344905 / 100 * (R * K / 128)) * c * R / (16 * q * 973078528) +
    q * p * 128 * (S * (2 * 2 ^ 34 + 1) / 2 ^ 100)

theorem gapRegs_eq {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (q p : ℕ) (hm : L.m = 16 * q)
    (hn : L.n = 8 * p) (hF : L.fwdMacs = 973078528) (hmem : L.mem = 2 ^ 34) (hq : 0 < q) (hp : 0 < p)
    (hK : 0 < L.K) :
    gapRegs L C = gapExpr q p L.rows L.cap L.K L.S C.chunks := by
  have hrb : L.rowBlocks = q := by simp only [Layout.rowBlocks, hm]; omega
  have hcb : L.colBlocks = p := by simp only [Layout.colBlocks, hn]; omega
  rw [gapRegs, networkGap_eq L C (by rw [hm]; positivity) (by rw [hn]; positivity) (by positivity)
    (by rw [hF]; norm_num)]
  simp only [gapExpr, headGap, hidingGap, coinBranchGap, pStarGap, tableGap, honestRegs, wordsRegs, wordsBlocks,
    aBlocks, shaWordsRegs, shaARegs, nMax, tileWords, stockRegs, hrb, hcb, hm, hn, hF, hmem]
  push_cast
  ring

theorem stepsFor_le (K : ℕ) (hK : 21 ≤ K) : 28 * stepsFor K ≤ K + 34 := by
  simp only [stepsFor]; omega

/-- **A call of the lead in its variables**: `16 q` rows, `8 p` columns with `512 ≤ p` and `8 p ≤ n + 7`,
`28 S ≤ K + 34`, its chunks at most `(16 q · 8 p · S + 65,535) / 65,536`, its gap `gapExpr` and its stock
`16 q · 8 p · K / 32 > 0`. -/
theorem leadVars {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (n q R c : ℕ) (hm : L.m = 16 * q)
    (hn : L.n = 8 * ((n + 7) / 8)) (hr : L.rows = R) (hc : L.cap = c) (hS : L.S = stepsFor L.K)
    (hF : L.fwdMacs = 973078528) (hmem : L.mem = 2 ^ 34) (hcw : L.chunkWords = 65536) (hq : 0 < q)
    (hn4 : 4096 ≤ n) (hK : 4096 ≤ L.K) (hC : ChunksOf L C) :
    ∃ p S ch : ℚ, 512 ≤ p ∧ 8 * p ≤ n + 7 ∧ 28 * S ≤ L.K + 34 ∧
      65536 * ch ≤ 16 * q * (8 * p) * S + 65535 ∧ gapRegs L C = gapExpr q p R c L.K S ch ∧
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
  · rw [gapRegs_eq C q _ hm hn hF hmem hq (by omega) (by omega), hr, hc]
  · rw [hst]
    have : (0 : ℚ) < q := by exact_mod_cast hq
    have : (0 : ℚ) < ((n + 7) / 8 : ℕ) := by exact_mod_cast (by omega : 0 < (n + 7) / 8)
    have : (0 : ℚ) < L.K := by exact_mod_cast (by omega : 0 < L.K)
    positivity

/-- The products `linarith` reads: `28 p S ≤ p K + 34 p`, `4,096 p ≤ p K`, `512 K ≤ p K` and `2^21 ≤ p K`. -/
theorem products {p K S : ℚ} (hp : 512 ≤ p) (hK : 4096 ≤ K) (hS : 28 * S ≤ K + 34) :
    28 * (p * S) ≤ p * K + 34 * p ∧ 4096 * p ≤ p * K ∧ 512 * K ≤ p * K ∧ 512 * 4096 ≤ p * K :=
  ⟨by nlinarith, by nlinarith, by nlinarith, by nlinarith⟩

/-- **A share bound at a point**: the gap's share of stock is at most `B` wherever `gapExpr` is at most `B` times
the stock over the variables' range. -/
theorem share_le {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (n q R c : ℕ) (B : ℚ) (hm : L.m = 16 * q)
    (hn : L.n = 8 * ((n + 7) / 8)) (hr : L.rows = R) (hc : L.cap = c) (hS : L.S = stepsFor L.K)
    (hF : L.fwdMacs = 973078528) (hmem : L.mem = 2 ^ 34) (hcw : L.chunkWords = 65536) (hq : 0 < q)
    (hn4 : 4096 ≤ n) (hK : 4096 ≤ L.K) (hC : ChunksOf L C)
    (h : ∀ p K S ch : ℚ, 512 ≤ p → 4096 ≤ K → 28 * S ≤ K + 34 → 65536 * ch ≤ 16 * q * (8 * p) * S + 65535 →
      gapExpr q p R c K S ch ≤ B * (16 * q * (8 * p) * K / 32)) :
    gapShare L C ≤ B := by
  obtain ⟨p, S, ch, hp, -, hS', hch, hg, hst, hpos⟩ :=
    leadVars C n q R c hm hn hr hc hS hF hmem hcw hq hn4 hK hC
  rw [gapShare, div_le_iff₀ hpos, hg, hst]
  exact h p L.K S ch hp (by exact_mod_cast hK) hS' hch

/-- **A served bound at a point**: the gap's share of served compute `R n K / 32` is at most `X` wherever `gapExpr`
is at most `X` times it, over the variables' range with `8 p ≤ n + 7`. -/
theorem served_le {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (n q R c : ℕ) (X : ℚ) (hm : L.m = 16 * q)
    (hn : L.n = 8 * ((n + 7) / 8)) (hr : L.rows = R) (hcols : L.cols = n) (hc : L.cap = c)
    (hS : L.S = stepsFor L.K) (hF : L.fwdMacs = 973078528) (hmem : L.mem = 2 ^ 34) (hcw : L.chunkWords = 65536)
    (hq : 0 < q) (hR : 0 < R) (hn4 : 4096 ≤ n) (hK : 4096 ≤ L.K) (hC : ChunksOf L C)
    (h : ∀ p K S ch u : ℚ, 512 ≤ p → 4096 ≤ K → 28 * S ≤ K + 34 → 65536 * ch ≤ 16 * q * (8 * p) * S + 65535 →
      8 * p ≤ u + 7 → gapExpr q p R c K S ch ≤ X * (R * u * K / 32)) :
    gapServed L C ≤ X := by
  obtain ⟨p, S, ch, hp, hpn, hS', hch, hg, -, -⟩ :=
    leadVars C n q R c hm hn hr hc hS hF hmem hcw hq hn4 hK hC
  have hsv : servedRegs L = R * n * L.K / 32 := by simp only [servedRegs, hr, hcols]
  have hpos : 0 < servedRegs L := by
    rw [hsv]
    have : (0 : ℚ) < R := by exact_mod_cast hR
    have : (0 : ℚ) < n := by exact_mod_cast (by omega : 0 < n)
    have : (0 : ℚ) < L.K := by exact_mod_cast (by omega : 0 < L.K)
    positivity
  rw [gapServed, div_le_iff₀ hpos, hg, hsv]
  exact h p L.K S ch n hp (by exact_mod_cast hK) hS' hch hpn

/-- The gap's share of stock at the three points, cap 32: rational arithmetic over `n, K ≥ 4,096`, with
`stepsFor K ≤ (K + 34) / 28` and `wordChunks ≤ m n S / 65,536 + 1`; at each point, over `p ≥ 512` and `K ≥ 4,096`,
`products`, then `linarith`. -/
theorem gapAt : GapAt := fun n K hn hK => by
  refine ⟨fun C hC => share_le C n 4 64 32 _ rfl rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_,
    fun C hC => share_le C n 32 512 32 _ rfl rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_,
    fun C hC => share_le C n 512 8192 32 _ rfl rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_⟩ <;>
  · intro p K S ch hp hK hS hch
    obtain ⟨h₁, h₂, h₃, h₄⟩ := products hp hK hS
    push_cast at hch ⊢
    simp only [gapExpr]
    linarith

/-- The gap's share of stock at the three points, cap 320. -/
theorem gapAt320 : GapAt320 := fun n K hn hK => by
  refine ⟨fun C hC => share_le C n 4 64 320 _ rfl rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_,
    fun C hC => share_le C n 32 512 320 _ rfl rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_,
    fun C hC => share_le C n 512 8192 320 _ rfl rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_⟩ <;>
  · intro p K S ch hp hK hS hch
    obtain ⟨h₁, h₂, h₃, h₄⟩ := products hp hK hS
    push_cast at hch ⊢
    simp only [gapExpr]
    linarith

/-- The gap at decode, padded to 16 rows, at caps 32 and 320: its share of served compute, at each point the stock's
bound with `(8 p − 7) K ≤ n K`, then of the padded stock. -/
theorem gapAtDecode : GapAtDecode := fun T n K hn hK => by
  refine ⟨fun C hC => ⟨served_le C n 1 1 32 _ rfl rfl rfl rfl rfl rfl rfl rfl rfl (by norm_num) (by norm_num) hn hK
        hC ?_, share_le C n 1 1 32 _ rfl rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_⟩,
    fun C hC => ⟨served_le C n 1 1 320 _ rfl rfl rfl rfl rfl rfl rfl rfl rfl (by norm_num) (by norm_num) hn hK
        hC ?_, share_le C n 1 1 320 _ rfl rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_⟩⟩
  all_goals first
    | intro p K S ch u hp hK hS hch hu
      obtain ⟨h₁, h₂, h₃, h₄⟩ := products hp hK hS
      have h₅ : (8 * p - 7) * K ≤ u * K := by nlinarith
      push_cast at hch ⊢
      simp only [gapExpr]
      linarith
    | intro p K S ch hp hK hS hch
      obtain ⟨h₁, h₂, h₃, h₄⟩ := products hp hK hS
      push_cast at hch ⊢
      simp only [gapExpr]
      linarith

/-- The gap is below the honest work where its share of stock is at most 1%: the honest work is at least the words,
`m n S ≥ m n K / 28`, more than the stock `m n K / 32`. -/
theorem gap_lt_honest {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (hshare : gapShare L C ≤ 1 / 100)
    (hpos : 0 < stockRegs L) (hS : L.K ≤ 28 * L.S) : gapRegs L C < honestRegs L C := by
  rw [gapShare, div_le_iff₀ hpos] at hshare
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

/-- The nontrivial lemma: the honest work exceeds the gap at the three points, and the empty strategy
does no work and hashes nothing. -/
theorem nontrivial : Nontrivial := by
  intro n K hn hK
  have hS : K ≤ 28 * stepsFor K := by simp only [stepsFor]; omega
  have pos : ∀ {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (q R c : ℕ), L.m = 16 * q →
      L.n = 8 * ((n + 7) / 8) → L.rows = R → L.cap = c → L.S = stepsFor L.K → L.fwdMacs = 973078528 →
      L.mem = 2 ^ 34 → L.chunkWords = 65536 → 0 < q → L.K = K → ChunksOf L C → 0 < stockRegs L :=
    fun C q R c hm hn' hr hc hS' hF hmem hcw hq hKL hC => by
      obtain ⟨-, -, -, -, -, -, -, -, -, hpos⟩ :=
        leadVars C n q R c hm hn' hr hc hS' hF hmem hcw hq hn (hKL ▸ hK) hC
      exact hpos
  obtain ⟨h64, h512, h8192⟩ := gapAt n K hn hK
  refine ⟨fun C hC => gap_lt_honest C ((h64 C hC).trans (by norm_num))
      (pos C 4 64 32 rfl rfl rfl rfl rfl rfl rfl rfl (by norm_num) rfl hC) hS,
    fun C hC => gap_lt_honest C ((h512 C hC).trans (by norm_num))
      (pos C 32 512 32 rfl rfl rfl rfl rfl rfl rfl rfl (by norm_num) rfl hC) hS,
    fun C hC => gap_lt_honest C ((h8192 C hC).trans (by norm_num))
      (pos C 512 8192 32 rfl rfl rfl rfl rfl rfl rfl rfl (by norm_num) rfl hC) hS, ?_⟩
  intro L pr
  refine ⟨⟨fun _ _ => 0, fun _ _ => rfl, fun _ => ⟨[], [], 0, [], 0, fun _ => 0⟩⟩, fun _ => ?_⟩
  exact ⟨rfl, rfl⟩

/-- **The calls the verifier admits include every call of at most `2^15` rows, `2^20` columns and an inner dimension
from 1 to `2^16`**, at either cap: at most `2,048 · 131,072 · 2,342 ≤ 2^40` tiles. Llama-3.2-1B's calls, its LM head of
128,256 columns included, are among them at up to `2^15` tokens a step. -/
theorem admits_of_caps (T m n K : ℕ) (hm : m ≤ 2 ^ 15) (hn : n ≤ 2 ^ 20) (hK : 1 ≤ K) (hK' : K ≤ 2 ^ 16) :
    (lead T m n K).Admits ∧ (lead320 T m n K).Admits := by
  have h : (lead T m n K).Admits := by
    refine ⟨hK, ?_⟩
    simp only [Layout.rowBlocks, Layout.colBlocks, lead, padRows, padCols, stepsFor]
    have hR : (16 * ((m + 15) / 16) + 15) / 16 ≤ 2048 := by omega
    have hC : (8 * ((n + 7) / 8) + 7) / 8 ≤ 131072 := by omega
    have hS : 1 + (K - 21 + 27) / 28 ≤ 2342 := by omega
    calc _ ≤ 2048 * 131072 * 2342 := Nat.mul_le_mul (Nat.mul_le_mul hR hC) hS
      _ ≤ 2 ^ 40 := by norm_num
  exact ⟨h, h⟩

end Pouw.SecurityProofs.Lead.Statement
