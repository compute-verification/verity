import Definitions.Pouw.Guarantees.Lead
import Proofs.Pouw.Lead.MH100W.Floor
import Proofs.Pouw.Lead.CoinBranch.Writes
import Proofs.Pouw.Lead.Tags.Distinct
import Proofs.Pouw.Lead.PStar.Table
import Proofs.Pouw.Lead.Commit.Credit

/-!
# The lead's guarantee from its pieces

`G`'s proof, at each accepted coin value with no break, on the egress-free twin (`OutsourceWithinEgressLead`):
1. the twin's run holds every committed tile (`TileGrain`), so its whole-tile writes are at least the distinct
   committed tiles less the table hits (`CoinBranch.distinct_le_writes`), and the distinct tiles are the committed
   tiles less the repeats; in a call of whole tiles the committed tiles' words are `m n S` (`perCoin`);
2. its hashing credit is at least the honest commitment's (`HonestHashRuled`, `Commit.hashed_credit_ge`), at least
   the honest compressions' (`CommitL1w`), and its compressions write that many registers (`ShaWrites`);
3. `OneInstructionMinimal` and `MH100W.cost_ge_regs` (`W1`, `H32`) price those registers at 32 each (`perCoin`);
4. summed over `O`, the repeats and table hits are at most `|Ω|` times `coinBranchGap` and `tableGap`
   (`Tags.repeats_le`, `PStar.table_le`), the twin's saving at most `|O|` times `32 · networkGap`, and the head,
   hiding and `p*` terms are nonnegative.
-/

namespace Pouw.SecurityProofs.Lead.Compose

open Pouw.Lead Pouw.Assumptions.Lead Pouw.Dimension.H100 Pouw.Dimension.Pointwise

/-- **One coin value's floor**: at a coin value where a realizable strategy in a call of whole tiles
exchanges nothing, is accepted and hashed no collision, its priced W1 work is at least 32 times the honest registers
less the repeated and table-held tiles' words. -/
theorem perCoin (pr : Prices100) (H : Hash.Msg → Hash.Digest) (T m n K : ℕ)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (stream : Coins (lead T m n K) → ℕ → Fin tileWords → ℤ) (C : Pouw.Lead.Commit (lead T m n K))
    (isA : Hash.Msg → Bool) (Real : Strategy (lead T m n K) → Prop) (hm : 16 ∣ m) (hn : 8 ∣ n)
    (hcommit : CommitL1w H C isA κ) (hpr : H32 pr) (hw1 : W1 pr Real) (hgrain : TileGrain stream Real)
    (hmin : OneInstructionMinimal Real) (hsha : ShaWrites isA Real) (hhash : HonestHashRuled H C κ Real)
    (σ : Strategy (lead T m n K)) (hσ : Real σ) (ω : Coins (lead T m n K)) (he : (σ.exec ω).egress = 0)
    (hacc : Accepted (lead T m n K) H C κ σ ω) (hnb : ¬ Break (lead T m n K) H C κ σ ω) :
    32 * (honestRegs (lead T m n K) C -
        tileWords * ((repeats (lead T m n K) κ ω : ℚ) + tableHits (lead T m n K) κ σ stream ω)) ≤
      (σ.work pr ω : ℚ) := by
  have hreg : (32 * regCount (σ.exec ω).trace : ℚ) ≤ σ.work pr ω := by
    exact_mod_cast MH100W.cost_ge_regs pr hpr _ (hw1 σ hσ ω)
  have hmin' : ((tileWords * writes (σ.exec ω).steps + (σ.exec ω).shaRegs : ℕ) : ℚ) ≤ regCount (σ.exec ω).trace := by
    exact_mod_cast hmin σ hσ ω
  have hdist : (((tileIdx (lead T m n K)).image fun x => tileOf (lead T m n K) κ ω x.1 x.2.1 x.2.2).card : ℚ) ≤
      writes (σ.exec ω).steps + tableHits (lead T m n K) κ σ stream ω := by
    exact_mod_cast CoinBranch.distinct_le_writes κ σ stream ω (fun y hy => hgrain σ hσ ω he y hy) hacc.1
  have hrep : (repeats (lead T m n K) κ ω : ℚ) = (tileIdx (lead T m n K)).card -
      ((tileIdx (lead T m n K)).image fun x => tileOf (lead T m n K) κ ω x.1 x.2.1 x.2.2).card := by
    rw [repeats, Nat.cast_sub Finset.card_image_le]
  have hwords : (tileWords * (tileIdx (lead T m n K)).card : ℚ) = wordsRegs (lead T m n K) := by
    rw [tileIdx, Finset.card_product, Finset.card_product, Finset.card_range, Finset.card_range, Finset.card_univ,
      Fintype.card_fin]
    obtain ⟨a, rfl⟩ := hm
    obtain ⟨b, rfl⟩ := hn
    have ha : (16 * a + 15) / 16 = a := by omega
    have hb : (8 * b + 7) / 8 = b := by omega
    simp only [wordsRegs, Layout.rowBlocks, Layout.colBlocks, lead, ha, hb, tileWords]
    push_cast
    ring
  have hhash : shaWordsRegs * wordsBlocks (lead T m n K) C + shaARegs * aBlocks (lead T m n K) ≤
      (σ.exec ω).shaRegs :=
    (hcommit.2 ω).trans ((Commit.hashed_credit_ge H C κ isA σ ω (hhash σ hσ ω hacc) hnb).trans (hsha σ hσ ω))
  have htw : (tileWords : ℚ) = 128 := by norm_num [tileWords]
  unfold honestRegs
  push_cast at hmin' hwords ⊢
  rw [htw] at hmin' hwords ⊢
  linarith

/-- **The lead's guarantee** (`Pouw.Guarantees.Lead.G`), from `perCoin` on the twin and the
averaged bounds. -/
theorem G : Pouw.Guarantees.Lead.G := by
  intro pr fpSem H T m n K κ stream C isA Device cert runC can Real hm hn hκ hraw hC h32 hw1 htile hlane hone hcard
    hsha hhon _hhost hout σ hcan O hacc
  by_cases hB : ∃ ω ∈ O, Break (lead T m n K) H C κ σ ω
  · exact Or.inr hB
  refine Or.inl ?_
  push Not at hB
  obtain ⟨σ', hσ', -, htwin⟩ := hout σ (hcard σ hcan)
  have hacc' : ∀ ω ∈ O, Accepted (lead T m n K) H C κ σ' ω := fun ω hω => by
    obtain ⟨-, hh, hr, -, -⟩ := htwin ω
    obtain ⟨hhold, hroot⟩ := hacc ω hω
    exact ⟨fun rb hrb cb hcb t => hh _ (hhold rb hrb cb hcb t), hr.trans hroot⟩
  have hnb' : ∀ ω ∈ O, ¬ Break (lead T m n K) H C κ σ' ω := fun ω hω => by
    unfold Break
    rw [(htwin ω).2.2.2.1]
    exact hB ω hω
  have htw : (tileWords : ℚ) = 128 := by norm_num [tileWords]
  have hstep : ∀ ω ∈ O,
      32 * honestRegs (lead T m n K) C - 32 * (tileWords * repeats (lead T m n K) κ ω : ℚ) -
          32 * (tileWords * tableHits (lead T m n K) κ σ' stream ω : ℚ) - 32 * networkGap (lead T m n K) ≤
        (σ.work pr ω : ℚ) := by
    intro ω hω
    have h1 := perCoin pr H T m n K κ stream C isA Real hm hn hC h32 hw1 htile hone hsha hhon σ' hσ' ω (htwin ω).1
      (hacc' ω hω) (hnb' ω hω)
    have h2 := (htwin ω).2.2.2.2
    rw [htw] at h1 ⊢
    linarith
  have hsum := Finset.sum_le_sum hstep
  simp only [Finset.sum_sub_distrib, ← Finset.mul_sum, Finset.sum_const, nsmul_eq_mul] at hsum
  have hrep : ∑ ω ∈ O, (tileWords * repeats (lead T m n K) κ ω : ℚ) ≤
      Fintype.card (Coins (lead T m n K)) * coinBranchGap (lead T m n K) :=
    (Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ O) fun _ _ _ => by positivity).trans
      (Tags.repeats_le T m n K κ hm hn hκ)
  have htab : ∑ ω ∈ O, (tileWords * tableHits (lead T m n K) κ σ' stream ω : ℚ) ≤
      Fintype.card (Coins (lead T m n K)) * tableGap (lead T m n K) :=
    (Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ O) fun _ _ _ => by positivity).trans
      (PStar.table_le T m n K κ σ' stream hm hn hraw)
  rw [← Finset.mul_sum] at hrep htab
  have hO : (O.card : ℚ) ≤ Fintype.card (Coins (lead T m n K)) := by exact_mod_cast Finset.card_le_univ O
  have hN : (0 : ℚ) ≤ Fintype.card (Coins (lead T m n K)) := Nat.cast_nonneg _
  have hng : 0 ≤ networkGap (lead T m n K) := by unfold networkGap stockRegs; positivity
  have hng' := mul_le_mul_of_nonneg_right hO hng
  have hhead := mul_nonneg hN (show 0 ≤ headGap (lead T m n K) C by unfold headGap; positivity)
  have hhid := mul_nonneg hN (show 0 ≤ hidingGap (lead T m n K) by unfold hidingGap; positivity)
  have hps := mul_nonneg hN (show 0 ≤ pStarGap (lead T m n K) by unfold pStarGap stockRegs; positivity)
  unfold gapRegs
  linarith

/-- **The honest lemma**: the nominal W1 table meets `H32` (`H32Sat`), and the honest strategy
(write each committed tile once, hash the honest commitment) with the protocol's coin stream and commitment meets
every premise and is accepted at every coin value. -/
theorem honest : Pouw.Guarantees.Lead.Honest := by
  sorry

end Pouw.SecurityProofs.Lead.Compose
