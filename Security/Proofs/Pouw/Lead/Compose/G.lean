import Definitions.Pouw.Guarantees.Lead
import Proofs.Pouw.Lead.MH100W.Floor
import Proofs.Pouw.Lead.CoinBranch.Writes
import Proofs.Pouw.Lead.Tags.Distinct
import Proofs.Pouw.Lead.PStar.Table
import Proofs.Pouw.Lead.Commit.Credit
import Proofs.Pouw.Dimension.H100Proofs

/-!
# The lead's guarantee from its pieces

`G`'s proof, at each accepted coin value with no break, on the egress-free twin (`OutsourceWithinEgressLead`):
1. the twin's run holds every committed tile (`TileGrain`), so its whole-tile writes are at least the distinct
   committed tiles less the table hits (`CoinBranch.distinct_le_writes`), and the distinct tiles are the committed
   tiles less the repeats; the lead commits whole tiles, so their words are `m n S` (`perCoin`);
2. its hashing credit is at least the honest commitment's (`HonestHashRuled`, `Commit.hashed_credit_ge`), at least
   the honest compressions' (`CommitL1w`), and its compressions write that many registers (`ShaWrites`);
3. `OneInstructionMinimal` and `MH100W.cost_ge_regs` (`W1`, `H32`) price those registers at 32 each (`perCoin`);
4. summed over `O`, the repeats and table hits are at most `|Ω|` times `coinBranchGap` and `tableGap`
   (`Tags.repeats_le`, `PStar.table_le`), the twin's saving at most `|O|` times `32 · networkGap`, and the head,
   hiding and `p*` terms are nonnegative.
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
    have ha : (16 * ((m + 15) / 16) + 15) / 16 = (m + 15) / 16 := by omega
    have hb : (8 * ((n + 7) / 8) + 7) / 8 = (n + 7) / 8 := by omega
    simp only [wordsRegs, Layout.rowBlocks, Layout.colBlocks, lead, padRows, padCols, ha, hb, tileWords]
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
  intro pr H T m n K κ stream C isA can Real hC h32 hw1 htile hone hcard hsha hhon hout σ hcan O hacc
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
    have h1 := perCoin pr H T m n K κ stream C isA Real hC h32 hw1 htile hone hsha hhon σ' hσ' ω (htwin ω).1
      (hacc' ω hω) (hnb' ω hω)
    have h2 := (htwin ω).2.2.2.2
    rw [htw] at h1 ⊢
    linarith
  have hsum := Finset.sum_le_sum hstep
  simp only [Finset.sum_sub_distrib, ← Finset.mul_sum, Finset.sum_const, nsmul_eq_mul] at hsum
  have hrep : ∑ ω ∈ O, (tileWords * repeats (lead T m n K) κ ω : ℚ) ≤
      Fintype.card (Coins (lead T m n K)) * coinBranchGap (lead T m n K) :=
    (Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ O) fun _ _ _ => by positivity).trans
      (Tags.repeats_le T m n K κ)
  have htab : ∑ ω ∈ O, (tileWords * tableHits (lead T m n K) κ σ' stream ω : ℚ) ≤
      Fintype.card (Coins (lead T m n K)) * tableGap (lead T m n K) :=
    (Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ O) fun _ _ _ => by positivity).trans
      (PStar.table_le T m n K κ σ' stream)
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

/-- A trace of `other` instructions writes one register each. -/
theorem regCount_other {G : Type} (P : List (Op100 G)) (h : ∀ o ∈ P, ∃ f, o = .other f) : regCount P = P.length := by
  unfold regCount
  rw [List.map_congr_left fun o ho => show Pouw.Lead.Op100.regs o = 1 by obtain ⟨f, rfl⟩ := h o ho; rfl]
  simp

/-- A run's last write is what its slot holds. -/
theorem run_write_last {A V : Type} [DecidableEq A] {w : ℕ} (m₀ : A → Fin w → V) (P : List (Step A V w)) (a : A)
    (p : Fin w) (v : V) : run m₀ (P ++ [.write a p v]) a p = v := by
  simp [run, List.foldl_append, Step.apply]

/-- The committed tiles at `ω`, in `tileIdx`'s order. -/
noncomputable def tilesAt (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) :
    List (Fin tileWords → ℤ) :=
  (tileIdx L).toList.map fun x => tileOf L κ ω x.1 x.2.1 x.2.2

/-- The honest commitment's compressions, the words' and A's, each rounded up. -/
noncomputable def honestBlocks (L : Pouw.Lead.Layout) : ℕ :=
  ⌈4 * wordsRegs L / 128 + (wordChunks L : ℚ)⌉₊ + ⌈aBlocks L⌉₊

/-- One message of `honestBlocks L + 1` compressions. -/
noncomputable def honestMsg (L : Pouw.Lead.Layout) : Hash.Msg := List.replicate (128 * honestBlocks L) 0

/-- A commitment format meeting `CommitL1w`: one words message of `honestBlocks L + 1` compressions. -/
noncomputable def honestCommit (L : Pouw.Lead.Layout) : Pouw.Lead.Commit L where
  msgs _ _ := [honestMsg L]
  root _ _ := 0
  aMsgs := []
  chunks := wordChunks L

/-- The registers the honest message's compressions write, at ShaWrites' credit rounded up. -/
noncomputable def honestShaRegs (L : Pouw.Lead.Layout) : ℕ := ⌈shaWordsRegs * blocks (honestMsg L)⌉₊

/-- **The honest strategy**: no pre-coin table; at each coin value one `other` instruction a committed word, one write
a committed tile, and the honest message hashed. -/
noncomputable def honestStrategy (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) : Strategy L where
  pre _ _ := 0
  pre_supp _ _ := rfl
  exec ω :=
    { trace := ((tilesAt L κ ω).flatMap fun v => (List.finRange tileWords).map fun q => .other fun _ => (v q : ℚ)) ++
        List.replicate (honestShaRegs L) (.other 0)
      steps := (tilesAt L κ ω).map fun v => .write (.inl 0) 0 v
      shaRegs := honestShaRegs L
      hashed := [honestMsg L]
      egress := 0
      root := 0
      holds := fun y => y ∈ tilesAt L κ ω }

theorem honest_trace_other (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) :
    ∀ o ∈ ((honestStrategy L κ).exec ω).trace, ∃ f, o = .other f := by
  intro o ho
  simp only [honestStrategy, List.mem_append, List.mem_flatMap, List.mem_map, List.mem_replicate] at ho
  rcases ho with ⟨_, _, _, _, rfl⟩ | ⟨_, rfl⟩
  · exact ⟨_, rfl⟩
  · exact ⟨_, rfl⟩

theorem blocks_honestMsg (L : Pouw.Lead.Layout) : blocks (honestMsg L) = honestBlocks L + 1 := by
  rw [blocks, honestMsg, List.length_replicate]
  omega

/-- **The honest lemma**: the nominal W1 table meets `H32` (`H32Sat`), and at every call the honest strategy (write each
committed tile once, hash one message of the honest commitment's compressions rounded up), with an empty coin stream,
meets every premise and is accepted at every coin value. -/
theorem honest : Pouw.Guarantees.Lead.Honest := by
  intro T m n K H κ
  refine ⟨⟨fun _ _ => none, fun _ _ _ => rfl, fun _ _ _ _ _ h => by cases h⟩, honestCommit _, fun _ => false,
    fun σ => σ = honestStrategy _ κ, honestStrategy _ κ, ?_, h32Sat, ?_, ?_, ?_, fun _ h => h, ?_, ?_, ?_, rfl, ?_⟩
  · refine ⟨rfl, fun ω => ?_⟩
    simp only [honestCommit, List.append_nil, List.dedup_cons_of_notMem List.not_mem_nil, List.dedup_nil,
      List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, shaCredit, blocks_honestMsg, wordsBlocks,
      Bool.false_eq_true, ↓reduceIte]
    have hX := Nat.le_ceil (4 * wordsRegs (lead T m n K) / 128 + (wordChunks (lead T m n K) : ℚ))
    have hY := Nat.le_ceil (aBlocks (lead T m n K))
    have hY0 : 0 ≤ aBlocks (lead T m n K) := by unfold aBlocks; positivity
    unfold honestBlocks
    push_cast
    norm_num only [shaWordsRegs, shaARegs]
    linarith
  · rintro σ rfl ω o ho
    obtain ⟨f, rfl⟩ := honest_trace_other _ κ ω o ho
    trivial
  · rintro σ rfl ω - y hy
    obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hy
    refine ⟨i + 1, .inl 0, ?_⟩
    simp only [honestStrategy]
    rw [← List.map_take, List.take_add_one, List.getElem?_eq_getElem hi, Option.toList_some, List.map_append,
      List.map_singleton, run_write_last]
  · rintro σ rfl ω
    rw [regCount_other _ (honest_trace_other _ κ ω)]
    simp only [honestStrategy, writes, List.map_map, Function.comp_def, Step.cost, List.length_append,
      List.length_replicate, List.length_flatMap, List.length_map, List.length_finRange, List.map_const',
      List.sum_replicate, smul_eq_mul, mul_one, Nat.mul_comm tileWords]
    rfl
  · rintro σ rfl ω
    simp only [honestStrategy, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, shaCredit,
      Bool.false_eq_true, ↓reduceIte, honestShaRegs]
    exact Nat.le_ceil _
  · rintro σ rfl ω - y hy
    simp only [honestCommit, List.append_nil, List.mem_singleton] at hy
    exact ⟨honestMsg _, List.mem_singleton_self _, by rw [hy]⟩
  · rintro σ rfl
    refine ⟨_, rfl, rfl, fun ω => ⟨rfl, fun _ h => h, rfl, rfl, ?_⟩⟩
    have : 0 ≤ networkGap (lead T m n K) := by unfold networkGap stockRegs; positivity
    linarith
  · intro ω
    refine ⟨fun rb hrb cb hcb t => List.mem_map.2 ⟨(rb, cb, t), ?_, rfl⟩, rfl⟩
    simp [tileIdx, hrb, hcb]

end Pouw.SecurityProofs.Lead.Compose
