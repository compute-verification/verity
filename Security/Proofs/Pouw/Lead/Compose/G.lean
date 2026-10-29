import Definitions.Pouw.Guarantees.Lead
import Proofs.Pouw.Lead.MH100W.Floor
import Proofs.Pouw.Lead.CoinBranch.Writes
import Proofs.Pouw.Lead.Tags.Distinct
import Proofs.Pouw.Lead.PStar.Table
import Proofs.Pouw.Lead.Commit.Credit
import Proofs.Pouw.Lead.Network.Term

/-!
# The lead's guarantee from its pieces

`G`'s proof, at each accepted coin value with no break, on the egress-free twin (`OutsourceWithinEgressLead`), which
is accepted where the original is (it registers its root and exchanges nothing) and hashed no collision where the
original hashed none (beyond the original's, it hashes only the honest commitment's messages):
1. the twin hashes every honest words message (`HonestHashRuled` with no collision), so each committed tile is one of
   its run's whole-tile writes or a tile it starts with (`TileGrain`), and its whole-tile writes are at least the
   distinct committed tiles less the table hits (`CoinBranch.distinct_le_writes`), and the distinct tiles are the
   committed tiles less the repeats; the lead commits whole tiles, so their words are `m n S` (`perCoin`);
2. its hashing credit is at least the honest commitment's (`HonestHashRuled` at egress 0, `Commit.hashed_credit_ge`),
   at least the honest compressions' (`CommitL1w`), and its compressions write that many registers (`ShaWrites`);
3. `OneInstructionMinimal` and `MH100W.cost_ge_regs` (`W1`, `H32`) price those registers at 32 each (`perCoin`);
4. summed over `O`, the repeats and table hits are at most `|Ω|` times `coinBranchGap` and `tableGap`
   (`Tags.repeats_le` at the call `Accepted` admits, when `O` is nonempty; `PStar.table_le`); the twin's saving is at
   most `betaLead` a bit the original exchanged, at most `callBits` at an accepted value (`Accepted`), so `|O|` times
   `32 · networkGap`; and the head, hiding and `p*` terms are nonnegative.
-/

namespace Pouw.SecurityProofs.Lead.Compose

open Pouw.Lead Pouw.Assumptions.Lead Pouw.Dimension.H100 Pouw.Dimension.Pointwise

/-- **One coin value's floor**: at a coin value where a realizable strategy exchanges nothing, is
accepted and hashed no collision, its priced W1 work is at least 32 times the honest registers less the repeated
and table-held tiles' words. -/
theorem perCoin (pr : Prices100) (H : Hash.Msg → Hash.Digest) (T m n K : ℕ)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (stream : CoinStream (lead T m n K)) (C : Pouw.Lead.Commit (lead T m n K))
    (isA : Hash.Msg → Bool) (Real : Strategy (lead T m n K) → Prop)
    (hcommit : CommitL1w H C isA κ) (hpr : H32 pr) (hw1 : W1 pr Real) (hgrain : TileGrain H C κ stream Real)
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
  have hall : ∀ y ∈ C.msgs H (word (lead T m n K) κ ω), y ∈ (σ.exec ω).hashed := fun y hy => by
    obtain ⟨x, hx, hxy⟩ := hhash σ hσ ω he hacc y (List.mem_append_left _ hy)
    by_cases hne : x = y
    · exact hne ▸ hx
    · exact absurd ⟨x, List.mem_append_left _ hx, y, List.mem_append_right _ (List.mem_append_left _ hy), hne,
        congrArg Verity.Digest.bytes hxy⟩ hnb
  have hdist : (((tileIdx (lead T m n K)).image fun x => tileOf (lead T m n K) κ ω x.1 x.2.1 x.2.2).card : ℚ) ≤
      writes (σ.exec ω).steps + tableHits (lead T m n K) κ σ stream ω := by
    exact_mod_cast CoinBranch.distinct_le_writes κ σ stream ω (hgrain σ hσ ω he hall)
  have hrep : (repeats (lead T m n K) κ ω : ℚ) = (tileIdx (lead T m n K)).card -
      ((tileIdx (lead T m n K)).image fun x => tileOf (lead T m n K) κ ω x.1 x.2.1 x.2.2).card := by
    rw [repeats, Nat.cast_sub Finset.card_image_le]
  have hwords : (tileWords * (tileIdx (lead T m n K)).card : ℚ) = wordsRegs (lead T m n K) := by
    rw [tileIdx, Finset.card_product, Finset.card_product, Finset.card_range, Finset.card_range, Finset.card_univ,
      Fintype.card_fin]
    have ha : (16 * ((m + 15) / 16) + 15) / 16 = (m + 15) / 16 := by omega
    have hb : (8 * ((n + 7) / 8) + 7) / 8 = (n + 7) / 8 := by omega
    simp only [wordsRegs, Layout.rowBlocks, Layout.colBlocks, lead, padRows, padCols, ha, hb, tileWords]
    push_cast
    ring
  have hhash : shaWordsRegs * wordsBlocks (lead T m n K) C + shaARegs * aBlocks (lead T m n K) ≤
      (σ.exec ω).shaRegs :=
    (hcommit.2 ω).trans ((Commit.hashed_credit_ge H C κ isA σ ω (hhash σ hσ ω he hacc) hnb).trans (hsha σ hσ ω))
  have htw : (tileWords : ℚ) = 128 := by norm_num [tileWords]
  unfold honestRegs
  push_cast at hmin' hwords ⊢
  rw [htw] at hmin' hwords ⊢
  linarith

/-- **The lead's guarantee** (`Pouw.Guarantees.Lead.G`), from `perCoin` on the twin and the
averaged bounds. -/
theorem G : Pouw.Guarantees.Lead.G := by
  intro pr H T m n K κ stream C isA can Real hC h32 hw1 htile hone hcard hsha hhon hout σ hcan O hacc
  by_cases hB : ∃ ω ∈ O, Break (lead T m n K) H C κ σ ω
  · exact Or.inr hB
  refine Or.inl ?_
  push Not at hB
  obtain ⟨σ', hσ', -, htwin⟩ := hout σ (hcard σ hcan)
  have hacc' : ∀ ω ∈ O, Accepted (lead T m n K) H C κ σ' ω := fun ω hω => by
    obtain ⟨he, hr, -, -⟩ := htwin ω
    obtain ⟨hadm, hroot, hbits⟩ := hacc ω hω
    refine ⟨hadm, hr.trans hroot, ?_⟩
    rw [he]
    exact (Nat.cast_nonneg _).trans hbits
  have hnb' : ∀ ω ∈ O, ¬ Break (lead T m n K) H C κ σ' ω := fun ω hω hb => by
    unfold Break Hash.CollisionIn at hb
    obtain ⟨x, hx, y, hy, hxy⟩ := hb
    have hsub : ∀ z ∈ (σ'.exec ω).hashed ++ (C.msgs H (word (lead T m n K) κ ω) ++ C.aMsgs),
        z ∈ (σ.exec ω).hashed ++ (C.msgs H (word (lead T m n K) κ ω) ++ C.aMsgs) := fun z hz => by
      rcases List.mem_append.1 hz with hz | hz
      · rcases (htwin ω).2.2.1 z hz with h | h
        · exact List.mem_append_left _ h
        · exact List.mem_append_right _ h
      · exact List.mem_append_right _ hz
    apply hB ω hω
    unfold Break Hash.CollisionIn
    exact ⟨x, hsub x hx, y, hsub y hy, hxy⟩
  have htw : (tileWords : ℚ) = 128 := by norm_num [tileWords]
  have hβ : 0 ≤ betaLead (lead T m n K) C := by
    unfold betaLead nMax wMac honestRegs wordsBlocks wordsRegs aBlocks shaWordsRegs shaARegs
    positivity
  have hbits0 : 0 ≤ (lead T m n K).callBits := by unfold Layout.callBits; positivity
  have hnet : 32 * networkGap (lead T m n K) C = betaLead (lead T m n K) C * (lead T m n K).callBits := by
    unfold networkGap
    ring
  have hstep : ∀ ω ∈ O,
      32 * honestRegs (lead T m n K) C - 32 * (tileWords * repeats (lead T m n K) κ ω : ℚ) -
          32 * (tileWords * tableHits (lead T m n K) κ σ' stream ω : ℚ) - 32 * networkGap (lead T m n K) C ≤
        (σ.work pr ω : ℚ) := by
    intro ω hω
    have h1 := perCoin pr H T m n K κ stream C isA Real hC h32 hw1 htile hone hsha hhon σ' hσ' ω (htwin ω).1
      (hacc' ω hω) (hnb' ω hω)
    have h2 := (htwin ω).2.2.2
    have h3 := mul_le_mul_of_nonneg_left (hacc ω hω).2.2 hβ
    rw [htw] at h1 ⊢
    linarith
  have hsum := Finset.sum_le_sum hstep
  simp only [Finset.sum_sub_distrib, ← Finset.mul_sum, Finset.sum_const, nsmul_eq_mul] at hsum
  have hrep : ∑ ω ∈ O, (tileWords * repeats (lead T m n K) κ ω : ℚ) ≤
      Fintype.card (Coins (lead T m n K)) * coinBranchGap (lead T m n K) := by
    rcases O.eq_empty_or_nonempty with rfl | ⟨ω₀, hω₀⟩
    · rw [Finset.sum_empty]
      exact mul_nonneg (Nat.cast_nonneg _) (by unfold coinBranchGap stockRegs; positivity)
    · exact (Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ O) fun _ _ _ => by positivity).trans
        (Tags.repeats_le T m n K κ (hacc ω₀ hω₀).1)
  have htab : ∑ ω ∈ O, (tileWords * tableHits (lead T m n K) κ σ' stream ω : ℚ) ≤
      Fintype.card (Coins (lead T m n K)) * tableGap (lead T m n K) :=
    (Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ O) fun _ _ _ => by positivity).trans
      (PStar.table_le T m n K κ σ' stream)
  rw [← Finset.mul_sum] at hrep htab
  have hO : (O.card : ℚ) ≤ Fintype.card (Coins (lead T m n K)) := by exact_mod_cast Finset.card_le_univ O
  have hN : (0 : ℚ) ≤ Fintype.card (Coins (lead T m n K)) := Nat.cast_nonneg _
  have hng : 0 ≤ networkGap (lead T m n K) C := by
    unfold networkGap
    exact div_nonneg (mul_nonneg hβ hbits0) (by norm_num)
  have hng' := mul_le_mul_of_nonneg_right hO hng
  have hhead := mul_nonneg hN (show 0 ≤ headGap (lead T m n K) C by unfold headGap; positivity)
  have hhid := mul_nonneg hN (show 0 ≤ hidingGap (lead T m n K) by unfold hidingGap; positivity)
  have hps := mul_nonneg hN (show 0 ≤ pStarGap (lead T m n K) by unfold pStarGap stockRegs; positivity)
  unfold gapRegs
  linarith

end Pouw.SecurityProofs.Lead.Compose
