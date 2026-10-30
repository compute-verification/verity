import Definitions.Pouw.Guarantees.Lead

/-!
# The gap at every admitted call and at the three points, and the nontrivial lemma

Every share is of served stock (`Pouw.Lead.servedRegs`, the plain matmul of the call's served shape). `gapAll` bounds
the gap at every call the verifier admits by the closed form `Pouw.Lead.gapBound`. `gapAt` evaluates it at
T 512 × 64, T 512 × 512 and T 8,192 for every call of `n ≥ 512` and `K ≥ 2,048`, the served shapes', and
`gapAtDecode` at decode; the worst call is near `n = 513` and `K = 2,066`, the first `K` past 2,048 that adds a step.
`nontrivial` is the guarantee's nontrivial lemma. `admits_of_caps`: the calls the verifier admits (`Layout.Admits`)
include every call of at most `2^15` rows, `2^20` columns and an inner dimension from 1 to `2^16`. The cap doesn't
enter the gap: the network term is a forward's (`Network.networkFwd`).

The proof writes a call of the lead in its variables (`leadVars`): `16 q` rows, `8 p` columns, `K` with
`28 S ≤ K + 34` (`stepsFor`), and `65,536` chunks at most `16 q · 8 p · S + 65,535` (`wordChunks`). The gap is then
`gapExpr`, which `gapExpr_le` bounds at every admitted call by `gapBound` times served stock. At a point, which fixes
`q` and the served rows `R`, it is linear in `p K`, `p S`, `p`, `K` and the chunks, and `linarith` bounds it by served
stock `R · u · K / 32` from `28 p S ≤ p K + 34 p`, `2,048 p ≤ p K`, `64 K ≤ p K` and `2^17 ≤ p K` (`products`, at
`p ≥ 64`) and `(8 p − 7) K ≤ u K`.
-/

namespace Pouw.SecurityProofs.Lead.Statement

open Pouw.Lead Pouw.Guarantees.Lead Pouw.Dimension.H100

/-- The gap of a call of `16 q` rows, `8 p` columns, inner dimension `K`, `S` steps and `ch` chunks, at the lead's
constants. -/
def gapExpr (q p K S ch : ℚ) : ℚ :=
  23 * ch + 16064 / 1000000 * (8 * p * K / 32) + 84 / 100000 * (16 * q * (8 * p) * K / 32) +
    35 / 1000000 * (16 * q * (8 * p) * K / 32) + q * p * 128 * (S * (2 * 2 ^ 40 + 1) / 2 ^ 100)

theorem gapRegs_eq {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (q p : ℕ) (hm : L.m = 16 * q)
    (hn : L.n = 8 * p) (hmem : L.mem = 2 ^ 40) : gapRegs L C = gapExpr q p L.K L.S C.chunks := by
  have hrb : L.rowBlocks = q := by simp only [Layout.rowBlocks, hm]; omega
  have hcb : L.colBlocks = p := by simp only [Layout.colBlocks, hn]; omega
  simp only [gapExpr, gapRegs, headGap, hidingGap, coinBranchGap, pStarGap, tableGap, stockRegs, tileWords, hrb,
    hcb, hm, hn, hmem]
  push_cast
  ring

theorem stepsFor_le (K : ℕ) : 28 * stepsFor K ≤ K + 34 := by
  simp only [stepsFor]; omega

/-- **The gap's bound in its variables**: from `28 S ≤ K + 34` and `65,536 ch ≤ 16 q · 8 p · S + 65,535`, copy-and-patch
and `p*` over padded stock, the chunks' heads and the tables at `(K + 34) / 28` steps, hiding's row, and one partial
chunk's head. -/
theorem gapExpr_le {q p K S ch : ℚ} (hq : 0 ≤ q) (hp : 0 ≤ p) (hS : 28 * S ≤ K + 34)
    (hch : 65536 * ch ≤ 16 * q * (8 * p) * S + 65535) :
    gapExpr q p K S ch ≤ 16 * q * (8 * p) * K / 32 * (875 / 1000000) +
      16 * q * (8 * p) * (K + 34) / 32 * (23 / 57344 + 8 * (2 * 2 ^ 40 + 1) / (7 * 2 ^ 100)) +
      16064 / 1000000 * (8 * p * K / 32) + 23 := by
  have h := mul_le_mul_of_nonneg_left hS (mul_nonneg hq hp)
  simp only [gapExpr]
  linarith

/-- **The gap at every admitted call** (`Pouw.Guarantees.Lead.GapAll`): `gapExpr_le` at the call's `16 ⌈m / 16⌉` rows
and `8 ⌈n / 8⌉` columns, and `gapBound` times served stock `m n K / 32` is its right side. -/
theorem gapAll : GapAll := by
  intro T m n K hm hn hadm C hC
  obtain ⟨a, ha⟩ : ∃ a, (m + 15) / 16 = a := ⟨_, rfl⟩
  obtain ⟨b, hb⟩ : ∃ b, (n + 7) / 8 = b := ⟨_, rfl⟩
  have hpr : padRows m = 16 * a := by rw [padRows, ha]
  have hpc : padCols n = 8 * b := by rw [padCols, hb]
  have hch : 65536 * C.chunks ≤ 16 * a * (8 * b) * stepsFor K + 65535 := by
    rw [hC]
    show 65536 * ((padRows m * padCols n * stepsFor K + 65536 - 1) / 65536) ≤ _
    rw [hpr, hpc]
    have := Nat.div_mul_le_self (16 * a * (8 * b) * stepsFor K + 65536 - 1) 65536
    omega
  have hm' : (0 : ℚ) < m := Nat.cast_pos.2 (by omega)
  have hn' : (0 : ℚ) < n := Nat.cast_pos.2 (by omega)
  have hK' : (0 : ℚ) < K := Nat.cast_pos.2 (by have := hadm.1; omega)
  have hm0 := hm'.ne'
  have hn0 := hn'.ne'
  have hK0 := hK'.ne'
  have hsv : servedRegs (lead T m n K) = (m * n * K : ℚ) / 32 := rfl
  have hgb : gapBound m n K * ((m * n * K : ℚ) / 32) =
      16 * a * (8 * b) * K / 32 * (875 / 1000000) +
        16 * a * (8 * b) * (K + 34) / 32 * (23 / 57344 + 8 * (2 * 2 ^ 40 + 1) / (7 * 2 ^ 100)) +
        16064 / 1000000 * (8 * b * K / 32) + 23 := by
    rw [gapBound, hpr, hpc]
    push_cast
    field_simp
    ring
  rw [gapServed, div_le_iff₀ (by rw [hsv]; positivity), hsv, hgb,
    gapRegs_eq C a b (show padRows m = 16 * a from hpr) (show padCols n = 8 * b from hpc) rfl]
  exact gapExpr_le (Nat.cast_nonneg _) (Nat.cast_nonneg _) (by exact_mod_cast stepsFor_le K)
    (by exact_mod_cast hch)

/-- **A call of the lead in its variables**: `16 q` rows, `8 p` columns with `64 ≤ p` and `8 p ≤ n + 7`,
`28 S ≤ K + 34`, its chunks at most `(16 q · 8 p · S + 65,535) / 65,536`, and its gap `gapExpr`. -/
theorem leadVars {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (n q : ℕ) (hm : L.m = 16 * q)
    (hn : L.n = 8 * ((n + 7) / 8)) (hS : L.S = stepsFor L.K) (hmem : L.mem = 2 ^ 40) (hcw : L.chunkWords = 65536)
    (hn5 : 512 ≤ n) (hC : ChunksOf L C) :
    ∃ p S ch : ℚ, 64 ≤ p ∧ 8 * p ≤ n + 7 ∧ 28 * S ≤ L.K + 34 ∧
      65536 * ch ≤ 16 * q * (8 * p) * S + 65535 ∧ gapRegs L C = gapExpr q p L.K S ch := by
  have hp : 64 ≤ (n + 7) / 8 := by omega
  have hch : 65536 * C.chunks ≤ 16 * q * (8 * ((n + 7) / 8)) * L.S + 65535 := by
    rw [hC, wordChunks, hcw, hm, hn]
    have := Nat.div_mul_le_self (16 * q * (8 * ((n + 7) / 8)) * L.S + 65536 - 1) 65536
    omega
  refine ⟨((n + 7) / 8 : ℕ), L.S, C.chunks, by exact_mod_cast hp, ?_, ?_, by exact_mod_cast hch,
    gapRegs_eq C q _ hm hn hmem⟩
  · have : 8 * ((n + 7) / 8) ≤ n + 7 := Nat.mul_div_le (n + 7) 8
    exact_mod_cast this
  · exact_mod_cast hS ▸ stepsFor_le L.K

/-- The products `linarith` reads: `28 p S ≤ p K + 34 p`, `2,048 p ≤ p K`, `64 K ≤ p K` and `64 · 2,048 ≤ p K`. -/
theorem products {p K S : ℚ} (hp : 64 ≤ p) (hK : 2048 ≤ K) (hS : 28 * S ≤ K + 34) :
    28 * (p * S) ≤ p * K + 34 * p ∧ 2048 * p ≤ p * K ∧ 64 * K ≤ p * K ∧ 64 * 2048 ≤ p * K :=
  ⟨by nlinarith, by nlinarith, by nlinarith, by nlinarith⟩

/-- **A bound at a point**: the gap's share of served stock `R n K / 32` is at most `X` wherever `gapExpr` is at most
`X` times it, over the variables' range with `8 p ≤ n + 7`. -/
theorem served_le {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (n q R : ℕ) (X : ℚ) (hm : L.m = 16 * q)
    (hn : L.n = 8 * ((n + 7) / 8)) (hr : L.rows = R) (hcols : L.cols = n) (hS : L.S = stepsFor L.K)
    (hmem : L.mem = 2 ^ 40) (hcw : L.chunkWords = 65536) (hR : 0 < R) (hn5 : 512 ≤ n) (hK : 2048 ≤ L.K)
    (hC : ChunksOf L C)
    (h : ∀ p K S ch u : ℚ, 64 ≤ p → 2048 ≤ K → 28 * S ≤ K + 34 → 65536 * ch ≤ 16 * q * (8 * p) * S + 65535 →
      8 * p ≤ u + 7 → gapExpr q p K S ch ≤ X * (R * u * K / 32)) :
    gapServed L C ≤ X := by
  obtain ⟨p, S, ch, hp, hpn, hS', hch, hg⟩ := leadVars C n q hm hn hS hmem hcw hn5 hC
  have hsv : servedRegs L = R * n * L.K / 32 := by simp only [servedRegs, hr, hcols]
  have hpos : 0 < servedRegs L := by
    rw [hsv]
    have : (0 : ℚ) < R := by exact_mod_cast hR
    have : (0 : ℚ) < n := by exact_mod_cast (by omega : 0 < n)
    have : (0 : ℚ) < L.K := by exact_mod_cast (by omega : 0 < L.K)
    positivity
  rw [gapServed, div_le_iff₀ hpos, hg, hsv]
  exact h p L.K S ch n hp (by exact_mod_cast hK) hS' hch hpn

/-- The gap's share of served stock at the three points: rational arithmetic over `n ≥ 512` and `K ≥ 2,048`, with
`stepsFor K ≤ (K + 34) / 28`, `wordChunks ≤ m n S / 65,536 + 1` and `8 p ≤ n + 7`; at each point `products`, then
`linarith`. -/
theorem gapAt : GapAt := fun n K hn hK => by
  refine ⟨fun C hC => served_le C n 4 64 _ rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_,
    fun C hC => served_le C n 32 512 _ rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_,
    fun C hC => served_le C n 512 8192 _ rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC ?_⟩ <;>
  · intro p K S ch u hp hK hS hch hu
    obtain ⟨h₁, h₂, h₃, h₄⟩ := products hp hK hS
    have h₅ : (8 * p - 7) * K ≤ u * K := by nlinarith
    push_cast at hch ⊢
    simp only [gapExpr]
    linarith

/-- The gap at decode, padded to 16 rows, as a share of the one served row's stock, with `(8 p − 7) K ≤ n K`. -/
theorem gapAtDecode : GapAtDecode := fun T n K hn hK C hC =>
  served_le C n 1 1 _ rfl rfl rfl rfl rfl rfl rfl (by norm_num) hn hK hC fun p K S ch u hp hK hS hch hu => by
    obtain ⟨h₁, h₂, h₃, h₄⟩ := products hp hK hS
    have h₅ : (8 * p - 7) * K ≤ u * K := by nlinarith
    push_cast at hch ⊢
    simp only [gapExpr]
    linarith

/-- The gap is below the honest work where its share of served stock is at most 1%: the honest work is at least the
words, `m n S ≥ m n K / 28`, more than served stock `rows · cols · K / 32 ≤ m n K / 32`. -/
theorem gap_lt_honest {L : Pouw.Lead.Layout} (C : Pouw.Lead.Commit L) (hshare : gapServed L C ≤ 1 / 100)
    (hpos : 0 < servedRegs L) (hrows : L.rows ≤ L.m) (hcols : L.cols ≤ L.n) (hS : L.K ≤ 28 * L.S) :
    gapRegs L C < honestRegs L C := by
  rw [gapServed, div_le_iff₀ hpos] at hshare
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

/-- The nontrivial lemma: the honest work exceeds the gap at the three points, and the empty strategy does no work,
exchanges nothing and hashes nothing. -/
theorem nontrivial : Nontrivial := by
  intro n K hn hK
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
  obtain ⟨h64, h512, h8192⟩ := gapAt n K hn hK
  refine ⟨fun C hC => gap_lt_honest C ((h64 C hC).trans (by norm_num)) (pos 512 64 (by norm_num))
      (rows 512 64) (cols 512 64) hS,
    fun C hC => gap_lt_honest C ((h512 C hC).trans (by norm_num)) (pos 512 512 (by norm_num))
      (rows 512 512) (cols 512 512) hS,
    fun C hC => gap_lt_honest C ((h8192 C hC).trans (by norm_num)) (pos 8192 8192 (by norm_num))
      (rows 8192 8192) (cols 8192 8192) hS, ?_⟩
  intro L pr
  exact ⟨⟨fun _ _ => 0, fun _ _ => rfl, fun _ => 0, fun _ => ⟨[], [], 0, [], 0, fun _ => 0⟩⟩,
    fun _ => ⟨rfl, rfl, rfl⟩⟩

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
