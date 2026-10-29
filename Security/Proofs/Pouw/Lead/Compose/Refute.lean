import Proofs.Pouw.Lead.Compose.Guarantee
import Proofs.Pouw.Dimension.H100Proofs

/-!
# The lead's guarantee as stated has a strategy that writes no word (proofs)

`not_G : ¬ Pouw.Guarantees.Lead.G`. `G` takes every call shape, and at `lead 512 1 1 4096` (one row, one column, 147
steps) one strategy meets every premise of `G`, the stream all zeros (`RawStream` with no coin), is accepted at every
coin value and hashes one message, yet writes no word:

* each committed MMA tile has one live word (`tileOf` is zero past position 0), and every word lies in
  `[−2^33, 2^33)` (`word_mem`), so a pre-coin table of `2^34` tiles, one a value, holds every committed tile
  (`TileGrain` with no step);
* its trace is the `other` instructions its compressions' registers need (`ShaWrites`), and nothing else.

Its priced W1 work is the commitment's hash part, rounded up to whole compressions (130,070 registers at 32 each),
while `G` asks for the 147 words too, less a gap of about 25 registers, so `G`'s conclusion fails at every coin value,
with no collision. The words floor `G` needs (`WordsFloor`) fails with it (`not_wordsFloor`).
-/

namespace Pouw.SecurityProofs.Lead.Compose

open Finset Pouw.Lead Pouw.Assumptions.Lead Pouw.Dimension.H100 Pouw.Dimension.Pointwise

theorem bal13_abs (z : ℤ) (q : Fin 4) : |bal13 z q| ≤ 6 := by
  unfold bal13
  have h0 := Int.emod_nonneg ((z + 14280) / 13 ^ (q : ℕ)) (by norm_num : (13 : ℤ) ≠ 0)
  have h1 := Int.emod_lt_of_pos ((z + 14280) / 13 ^ (q : ℕ)) (by norm_num : (0 : ℤ) < 13)
  rw [abs_le]
  constructor <;> linarith

theorem coinsG_abs {L : Pouw.Lead.Layout} (ω : Coins L) (q : Fin 4) (j : Fin L.n) : |ω.G q j| ≤ 3 := by
  unfold Coins.G
  have := (ω.2.2.1 q j).isLt
  rw [abs_le]
  constructor <;> omega

theorem coinsS_abs {T m n K : ℕ} (ω : Coins (lead T m n K)) (t : Fin (lead T m n K).S) (k : Fin (lead T m n K).tiles)
    (ℓ : Fin 3) (j : Fin (lead T m n K).n) : |ω.s t k ℓ j| ≤ 127 := by
  unfold Coins.s
  have h : (ω.2.2.2 t k ℓ j : ℕ) < 128 := (ω.2.2.2 t k ℓ j).isLt
  have he : (2 : ℤ) ^ (lead T m n K).e - 1 = 127 := by norm_num [lead]
  rw [he, abs_le]
  constructor <;> omega

theorem abs_sum_fin_le {k : ℕ} (f : Fin k → ℤ) (c : ℤ) (h : ∀ x, |f x| ≤ c) : |∑ x, f x| ≤ k * c :=
  (abs_sum_le_sum_abs _ _).trans ((sum_le_card_nsmul _ _ c fun x _ => h x).trans (by simp))

/-- A step's coin term is at most `2^16` in absolute value at the lead. -/
theorem coinTerm_abs {T m n K : ℕ} (ω : Coins (lead T m n K)) (i : Fin (lead T m n K).m) (t : Fin (lead T m n K).S)
    (j : Fin (lead T m n K).n) : |coinTerm _ ω i t j| ≤ 2 ^ 16 := by
  unfold coinTerm
  by_cases ht : (t : ℕ) = 0
  · rw [ite_eq_left ht]
    calc _ ≤ _ := abs_add_le _ _
      _ ≤ ((4 : ℕ) : ℤ) * 18 + ((4 : ℕ) : ℤ) * 6 := add_le_add
          (abs_sum_fin_le _ _ fun q => by
            rw [abs_mul]
            exact mul_le_mul (bal13_abs _ q) (coinsG_abs ω q j) (abs_nonneg _) (by norm_num))
          (abs_sum_fin_le _ _ fun q => by
            have hb : |((ω.2.1 i q : ℕ) : ℤ)| ≤ 1 := by
              have := (ω.2.1 i q).isLt
              rw [abs_le]
              constructor <;> omega
            rw [abs_mul]
            exact (mul_le_mul hb (bal13_abs _ q) (abs_nonneg _) (by norm_num)).trans (by norm_num))
      _ ≤ 2 ^ 16 := by norm_num
  · rw [ite_eq_right ht]
    have h1 : |1 + ((ω.1 i t : ℕ) : ℤ)| ≤ 2 := by
      have := (ω.1 i t).isLt
      rw [abs_le]
      constructor <;> omega
    calc _ ≤ _ := abs_add_le _ _
      _ ≤ 2 + ((3 : ℕ) : ℤ) * (127 * 127) := add_le_add h1
          (abs_sum_fin_le _ _ fun ℓ => by
            rw [abs_mul]
            refine mul_le_mul ?_ ?_ (abs_nonneg _) (by norm_num)
            · unfold rotWeight
              split_ifs <;> norm_num
            · split_ifs
              · exact coinsS_abs ω _ _ _ _
              · norm_num)
      _ ≤ 2 ^ 16 := by norm_num

/-- The counterexample's call: one row, one column, 147 steps. -/
abbrev Lx : Pouw.Lead.Layout := lead 512 1 1 4096

theorem Lx_S : Lx.S = 147 := by norm_num [lead, stepsFor]

/-- The data: zero. -/
def κx : Fin Lx.m → Fin Lx.S → Fin Lx.n → ℤ := fun _ _ _ => 0

/-- **Every word of the call lies in `[−2^33, 2^33)`.** -/
theorem word_mem (ω : Coins Lx) (i : Fin Lx.m) (t : Fin Lx.S) (j : Fin Lx.n) :
    -2 ^ 33 ≤ word Lx κx ω i t j ∧ word Lx κx ω i t j < 2 ^ 33 := by
  have hc : |∑ q : Fin 4, colCode j q| ≤ ((4 : ℕ) : ℤ) * 6 := abs_sum_fin_le _ _ fun q => bal13_abs _ q
  have hb : 0 ≤ bias Lx.S j ∧ bias Lx.S j ≤ 2 ^ 16 := by
    unfold bias
    rw [Lx_S]
    rw [abs_le] at hc
    push_cast at hc
    constructor
    · exact Int.ediv_nonneg (by push_cast; linarith) (by norm_num)
    · exact (Int.ediv_le_self _ (by push_cast; linarith)).trans (by push_cast; linarith)
  have hs : |∑ τ : Fin Lx.S, (if τ ≤ t then κx i τ j + coinTerm Lx ω i τ j else 0)| ≤ 147 * 2 ^ 16 := by
    refine (abs_sum_le_sum_abs _ _).trans ((sum_le_card_nsmul _ _ (2 ^ 16) fun τ _ => ?_).trans ?_)
    · split_ifs
      · simpa [κx] using coinTerm_abs ω i τ j
      · norm_num
    · simp [Lx_S]
  unfold word
  rw [abs_le] at hs
  constructor <;> linarith [hb.1, hb.2, hs.1, hs.2]

/-- The tile holding one value at position 0. -/
def oneWord (v : ℤ) : Fin tileWords → ℤ := fun p => if (p : ℕ) = 0 then v else 0

/-- Each committed tile is its word at position 0. -/
theorem tileOf_Lx (ω : Coins Lx) (t : Fin Lx.S) :
    tileOf Lx κx ω 0 0 t = oneWord (word Lx κx ω ⟨0, by norm_num [lead]⟩ t ⟨0, by norm_num [lead]⟩) := by
  funext p
  unfold tileOf oneWord
  by_cases hp : (p : ℕ) = 0
  · have h : 16 * 0 + (p : ℕ) / 8 < Lx.m ∧ 8 * 0 + (p : ℕ) % 8 < Lx.n := by simp [hp, lead]
    rw [dite_eq_left h, ite_eq_left hp]
    congr 1 <;> exact Fin.ext (by simp [hp])
  · have h : ¬ (16 * 0 + (p : ℕ) / 8 < Lx.m ∧ 8 * 0 + (p : ℕ) % 8 < Lx.n) := by
      simp only [lead]
      omega
    rw [dite_eq_right h, ite_eq_right hp]

/-- Its one message's length: `128 · 36` bytes. -/
@[irreducible] def xLen : ℕ := 128 * 36

/-- Its one message: `xLen` zero bytes, 37 compressions. -/
def xMsg : Hash.Msg := List.replicate xLen 0

/-- Its compression registers: `⌈3,515.38 · 37⌉`. -/
@[irreducible] def xRegs : ℕ := 130070

/-- The commitment format: one words message, no A message, one chunk. -/
def Cx : Pouw.Lead.Commit Lx where
  msgs _ _ := [xMsg]
  root _ _ := 0
  aMsgs := []
  chunks := 1

/-- The pre-coin table: slot `a < 2^34` holds the one-word tile of value `a − 2^33`. -/
def prex (a : ℕ) : Fin tileWords → ℤ := if a < 2 ^ 34 then oneWord ((a : ℤ) - 2 ^ 33) else 0

/-- The strategy: the table `prex`, no step, the table's tiles held, and `xRegs` `other` instructions. -/
def σx : Strategy Lx where
  pre := prex
  pre_supp _ ha := ite_eq_right (not_lt.2 ha)
  exec _ :=
    { trace := List.replicate xRegs (.other 0)
      steps := []
      shaRegs := xRegs
      hashed := [xMsg]
      egress := 0
      root := 0
      holds := fun y => ∃ a, prex a = y }

theorem blocks_xMsg : blocks xMsg = 37 := by
  rw [blocks, xMsg, List.length_replicate, xLen]

theorem work_σx (ω : Coins Lx) : σx.work h100Nominal ω = 32 * xRegs := by
  simp [Strategy.work, σx, cost100, List.map_replicate, List.sum_replicate, Op100.price, h100Nominal, mul_comm]

theorem chunks_Cx : Cx.chunks = wordChunks Lx := by
  simp [Cx, wordChunks, lead, stepsFor]

theorem commitL1w_Cx (H : Hash.Msg → Hash.Digest) (κ : Fin Lx.m → Fin Lx.S → Fin Lx.n → ℤ) :
    CommitL1w H Cx (fun _ => false) κ := by
  refine ⟨chunks_Cx, fun ω => ?_⟩
  simp only [Cx, List.append_nil, List.dedup_cons_of_notMem List.not_mem_nil, List.dedup_nil, List.map_cons,
    List.map_nil, List.sum_cons, List.sum_nil, add_zero, shaCredit, blocks_xMsg]
  norm_num [wordsBlocks, wordsRegs, aBlocks, shaWordsRegs, shaARegs, lead, stepsFor]

/-- The table holds every committed tile. -/
theorem holds_tileOf (ω : Coins Lx) (rb : ℕ) (hrb : rb < Lx.rowBlocks) (cb : ℕ) (hcb : cb < Lx.colBlocks)
    (t : Fin Lx.S) : (σx.exec ω).holds (tileOf Lx κx ω rb cb t) := by
  have hrb0 : rb = 0 := by simpa [Layout.rowBlocks, lead] using hrb
  have hcb0 : cb = 0 := by simpa [Layout.colBlocks, lead] using hcb
  subst hrb0 hcb0
  obtain ⟨h1, h2⟩ := word_mem ω ⟨0, by norm_num [lead]⟩ t ⟨0, by norm_num [lead]⟩
  have h0 : 0 ≤ word Lx κx ω ⟨0, by norm_num [lead]⟩ t ⟨0, by norm_num [lead]⟩ + 2 ^ 33 := by linarith
  have hlt : (word Lx κx ω ⟨0, by norm_num [lead]⟩ t ⟨0, by norm_num [lead]⟩ + 2 ^ 33).toNat < 2 ^ 34 := by
    have : (((word Lx κx ω ⟨0, by norm_num [lead]⟩ t ⟨0, by norm_num [lead]⟩ + 2 ^ 33).toNat : ℕ) : ℤ) < 2 ^ 34 := by
      rw [Int.toNat_of_nonneg h0]
      linarith
    exact_mod_cast this
  refine ⟨(word Lx κx ω ⟨0, by norm_num [lead]⟩ t ⟨0, by norm_num [lead]⟩ + 2 ^ 33).toNat, ?_⟩
  show prex _ = _
  rw [tileOf_Lx, prex, ite_eq_left hlt, Int.toNat_of_nonneg h0]
  congr 1
  ring

/-- **The lead's guarantee as stated is false**: at one row and one column, a pre-coin table holds every committed
tile, and a strategy that writes no word meets every premise. -/
theorem not_G : ¬ Pouw.Guarantees.Lead.G := by
  intro hG
  let Real : Strategy Lx → Prop := fun σ => σ = σx
  have hκ : DataInRange Lx κx := fun _ _ _ => by norm_num [κx]
  have hraw : RawStream Lx (fun _ _ _ => 0) := ⟨fun _ _ => none, fun _ _ _ => rfl, fun _ _ _ => rfl⟩
  have hw1 : W1 h100Nominal Real := by
    rintro σ rfl ω o ho
    simp only [σx, List.mem_replicate] at ho
    rw [ho.2]
    trivial
  have htile : TileGrain (fun _ _ _ => 0) Real := by
    rintro σ rfl ω - y ⟨a, rfl⟩
    refine ⟨0, .inl a, ?_⟩
    simp [σx, run, Strategy.init]
  have hlane : LaneSemantics (fun _ _ _ _ => 0) Real := by
    rintro σ rfl ω a p v h
    simp [σx] at h
  have hone : OneInstructionMinimal Real := by
    rintro σ rfl ω
    simp [σx, writes, regCount, List.map_replicate, List.sum_replicate, Pouw.Lead.Op100.regs]
  have hsha : ShaWrites (fun _ => false) Real := by
    rintro σ rfl ω
    simp only [σx, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, shaCredit, blocks_xMsg]
    norm_num [shaWordsRegs, xRegs]
  have hhon : HonestHashRuled (fun _ => 0) Cx κx Real := by
    rintro σ rfl ω - y hy
    simp only [Cx, List.append_nil, List.mem_singleton] at hy
    exact ⟨xMsg, by simp [σx], by rw [hy]⟩
  have hout : OutsourceWithinEgressLead h100Nominal Real := by
    rintro σ rfl
    refine ⟨σx, rfl, rfl, fun ω => ⟨rfl, fun _ h => h, rfl, rfl, ?_⟩⟩
    have : (0 : ℚ) ≤ networkGap Lx := by unfold networkGap stockRegs; positivity
    linarith
  have hacc : ∀ ω ∈ (univ : Finset (Coins Lx)), Accepted Lx (fun _ => 0) Cx κx σx ω := fun ω _ =>
    ⟨fun rb hrb cb hcb t => holds_tileOf ω rb hrb cb hcb t, rfl⟩
  rcases hG h100Nominal (fun _ _ _ _ => 0) (fun _ => 0) 512 1 1 4096 κx (fun _ _ _ => 0) Cx
      (fun _ => false) (cert := ()) (fun _ => certCode) Real Real hκ hraw (commitL1w_Cx _ _) h32Sat hw1 htile hlane
      hone (fun _ h => h) hsha hhon (fun _ => rfl) hout σx rfl univ hacc with h | ⟨ω, -, a, ha, b, hb, hab, -⟩
  · simp only [work_σx, sum_const, card_univ, nsmul_eq_mul] at h
    have hN : (0 : ℚ) < Fintype.card (Coins Lx) := by
      have ω₀ : Coins Lx := (fun _ _ => 0, fun _ _ => 0, fun _ _ => 0, fun _ _ _ _ => ⟨0, by norm_num [lead]⟩)
      exact_mod_cast Fintype.card_pos_iff.2 ⟨ω₀⟩
    have h' : 32 * (honestRegs Lx Cx - gapRegs Lx Cx) ≤ 32 * xRegs := by
      rw [← mul_le_mul_iff_of_pos_left hN]
      push_cast at h
      linarith
    revert h'
    norm_num [honestRegs, gapRegs, wordsRegs, wordsBlocks, aBlocks, shaWordsRegs, shaARegs, headGap, hidingGap,
      coinBranchGap, pStarGap, networkGap, tableGap, Layout.rowBlocks, Layout.colBlocks, stockRegs, Cx, xRegs, lead,
      stepsFor, tileWords]
  · simp only [σx, Cx, List.cons_append, List.nil_append, List.append_nil, List.mem_cons, List.not_mem_nil,
      or_false, or_self] at ha hb
    exact hab (ha.trans hb.symm)

/-- **The words floor fails with it**: `G` follows from the words floor (`G_of_wordsFloor`). -/
theorem not_wordsFloor : ¬ WordsFloor := fun h => not_G (G_of_wordsFloor h)

end Pouw.SecurityProofs.Lead.Compose
