import Proofs.Pouw.Lead.Compose.Guarantee
import Proofs.Pouw.Dimension.H100Proofs
import Mathlib.Logic.Equiv.Nat

/-!
# The lead's guarantee as stated has a strategy that writes no word (proofs)

`not_G : ¬ Pouw.Guarantees.Lead.G`. At `lead 512 64 4096 4096` (the point T 512 × 64 at `n = K = 4,096`), one strategy
meets every premise of `G`, is accepted at every coin value and hashes one message, yet writes no word:

* its pre-coin table holds every integer at every tile position (`pre k = Equiv.intEquivNat.symm k`), so `TileGrain`
  holds with no step, and `Accepted`'s `holds` reads values at any address;
* its trace is the `other` instructions its compressions' registers need (`ShaWrites`), and nothing else.

Its priced W1 work is the commitment's hash part, about 4.24 · 10⁹ registers at 32 each, while `G` asks for the words
too (38,535,168 registers, less a gap of about 6.3 · 10⁴), so `G`'s conclusion fails at every coin value, with no
collision. The words floor `G` needs (`WordsFloor`) fails with it (`not_wordsFloor`).
-/

namespace Pouw.SecurityProofs.Lead.Compose

open Finset Pouw.Lead Pouw.Assumptions.Lead Pouw.Dimension.H100 Pouw.Dimension.Pointwise

/-- The counterexample's call. -/
abbrev Lx : Pouw.Lead.Layout := lead 512 64 4096 4096

/-- Its one message's length: `128 · 1,206,860` bytes. -/
@[irreducible] def xLen : ℕ := 128 * 1206860

/-- Its one message: `xLen` zero bytes, `1,206,861` compressions. -/
def xMsg : Hash.Msg := List.replicate xLen 0

/-- Its compression registers: `⌈3,515.38 · 1,206,861⌉`. -/
@[irreducible] def xRegs : ℕ := 4242575023

/-- The commitment format: one words message, no A message, 588 chunks. -/
def Cx : Pouw.Lead.Commit Lx where
  msgs _ _ := [xMsg]
  root _ _ := 0
  aMsgs := []
  chunks := 588

/-- The strategy: every integer in its pre-coin table, no step, and `xRegs` `other` instructions. -/
def σx : Strategy Lx where
  pre k _ := Equiv.intEquivNat.symm k
  exec _ :=
    { trace := List.replicate xRegs (.other 0)
      steps := []
      shaRegs := xRegs
      hashed := [xMsg]
      egress := 0
      root := 0
      holds := fun _ _ => True }

theorem blocks_xMsg : blocks xMsg = 1206861 := by
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

/-- **The lead's guarantee as stated is false**: a strategy that writes no word meets every premise. -/
theorem not_G : ¬ Pouw.Guarantees.Lead.G := by
  intro hG
  let Real : Strategy Lx → Prop := fun σ => σ = σx
  have hκ : DataInRange Lx fun _ _ _ => 0 := fun _ _ _ => by norm_num
  have hw1 : W1 h100Nominal Real := by
    rintro σ rfl ω o ho
    simp only [σx, List.mem_replicate] at ho
    rw [ho.2]
    trivial
  have htile : TileGrain (fun _ _ _ => 0) Real := by
    rintro σ rfl ω - p v -
    exact ⟨0, .inl (Equiv.intEquivNat v), by simp [σx, run, Strategy.init]⟩
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
  have hhon : HonestHashRuled (fun _ => 0) Cx (fun _ _ _ => 0) Real := by
    rintro σ rfl ω - y hy
    simp only [Cx, List.append_nil, List.mem_singleton] at hy
    exact ⟨xMsg, by simp [σx], by rw [hy]⟩
  have hout : OutsourceWithinEgressLead h100Nominal Real := by
    rintro σ rfl
    refine ⟨σx, rfl, rfl, fun ω => ⟨rfl, fun _ _ h => h, rfl, rfl, ?_⟩⟩
    have : (0 : ℚ) ≤ networkGap Lx := by unfold networkGap stockRegs; positivity
    linarith
  have hacc : ∀ ω ∈ (univ : Finset (Coins Lx)), Accepted Lx (fun _ => 0) Cx (fun _ _ _ => 0) σx ω :=
    fun ω _ => ⟨fun _ _ _ => trivial, rfl⟩
  rcases hG h100Nominal (fun _ _ _ _ => 0) (fun _ => 0) 512 64 4096 4096 (fun _ _ _ => 0) (fun _ _ _ => 0) Cx
      (fun _ => false) (cert := ()) (fun _ => certCode) Real Real hκ (commitL1w_Cx _ _) h32Sat hw1 htile hlane
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
      coinBranchGap, pStarGap, networkGap, stockRegs, Cx, xRegs, lead, stepsFor]
  · simp only [σx, Cx, List.cons_append, List.nil_append, List.append_nil, List.mem_cons, List.not_mem_nil,
      or_false, or_self] at ha hb
    exact hab (ha.trans hb.symm)

/-- **The words floor fails with it**: `G` follows from the words floor (`G_of_wordsFloor`). -/
theorem not_wordsFloor : ¬ WordsFloor := fun h => not_G (G_of_wordsFloor h)

end Pouw.SecurityProofs.Lead.Compose
