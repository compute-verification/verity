import Definitions.Pouw.Guarantees.Lead

/-!
# The lead's guarantee composed from the words floor (proofs)

`G_of_wordsFloor : WordsFloor → Pouw.Guarantees.Lead.G`: the lead's work guarantee from one piece taken as a hypothesis,
the words floor (`WordsFloor`), with every other step proved from `G`'s named premises.

At a coin value `ω` of `O` where nothing hashed collides with the honest commitment, for the egress-free twin `σ′`
`OutsourceWithinEgressLead` gives:

* `32 · regCount ≤ work σ′` (`regs_le_cost`): `H32` floors every register writer at 32 a register, and `W1` makes each
  instruction well formed;
* `tileWords · writes + shaRegs ≤ regCount`: `OneInstructionMinimal`, a tile write `tileWords` registers;
* the hash part of the honest work is at most `shaRegs` (`hash_le_shaRegs`): `HonestHashRuled` and no collision put
  every distinct honest message among the hashed ones, `CommitL1w` bounds their ShaWrites credit below by the hash part,
  and `ShaWrites` bounds the hashed credit by `shaRegs`;
* `work σ′ ≤ work σ + 32 · networkGap`: `OutsourceWithinEgressLead`.

Summed over `O`, the words floor bounds `Σ tileWords · writes`, and `|O| ≤ |Ω|` charges the network term within the
gap.
-/

namespace Pouw.SecurityProofs.Lead.Compose

open Finset Pouw.Lead Pouw.Assumptions.Lead Pouw.Dimension.H100 Pouw.Dimension.Pointwise

/-- The gap less the network term: the words floor's share. -/
def wordsGap (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) : ℚ :=
  headGap L C + hidingGap L + coinBranchGap L + pStarGap L + tableGap L

theorem gapRegs_eq (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) : gapRegs L C = wordsGap L C + networkGap L := by
  unfold gapRegs wordsGap; ring

/-- The hash part of the honest work: the commitment's compressions at ShaWrites' registers. -/
def hashRegs (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) : ℚ := shaWordsRegs * wordsBlocks L C + shaARegs * aBlocks L

theorem honestRegs_eq (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) : honestRegs L C = wordsRegs L + hashRegs L C := by
  unfold honestRegs hashRegs; ring

/-- **The words floor** at the lead (the piece taken as a hypothesis): summed over coin values `O` on which an
egress-free realizable strategy is accepted, its run's tile writes, `tileWords` registers each, are at least `|O|` times
the words less `|Ω|` times the words gap. -/
def WordsFloor : Prop :=
  ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (H : Hash.Msg → Hash.Digest) (T m n K : ℕ)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (stream : Coins (lead T m n K) → ℕ → Fin tileWords → ℤ) (C : Pouw.Lead.Commit (lead T m n K))
    (Real : Strategy (lead T m n K) → Prop),
    DataInRange (lead T m n K) κ → RawStream (lead T m n K) stream → C.chunks = wordChunks (lead T m n K) →
    TileGrain stream Real → LaneSemantics fpSem Real →
    ∀ σ, Real σ → ∀ O : Finset (Coins (lead T m n K)),
      (∀ ω ∈ O, (σ.exec ω).egress = 0 ∧ Accepted (lead T m n K) H C κ σ ω) →
      (O.card : ℚ) * wordsRegs (lead T m n K) - (Fintype.card (Coins (lead T m n K)) : ℚ) * wordsGap _ C ≤
        ∑ ω ∈ O, ((tileWords * writes (σ.exec ω).steps : ℕ) : ℚ)

/-- An instruction's price is at least 32 a register it writes, under `H32`, when it is well formed. -/
theorem op_regs_le_price {G : Type} {pr : Prices100} (h : H32 pr) {F : Set (G → ℚ)} (o : Op100 G)
    (ho : o.WFw pr F) : 32 * Op100.regs o ≤ o.price pr := by
  obtain ⟨hs, htc, hffma, hint, hfp64, hother⟩ := h
  cases o with
  | wgmma s outs => exact Nat.mul_le_mul_right _ (hs s)
  | mma outs p => exact (Nat.mul_le_mul_right _ htc).trans ho
  | fp32 => simpa [Op100.regs, Op100.price] using hffma
  | int => simpa [Op100.regs, Op100.price] using hint
  | fp64 => simpa [Op100.regs, Op100.price] using hfp64
  | other => simpa [Op100.regs, Op100.price] using hother
  | move => simp [Op100.regs, Op100.price]
  | operand => simp [Op100.regs, Op100.price]

/-- **A trace costs at least 32 a register it writes**, under `H32`, when each instruction is well formed. -/
theorem regs_le_cost {G : Type} {pr : Prices100} (h : H32 pr) {F : Set (G → ℚ)} :
    ∀ P : List (Op100 G), (∀ o ∈ P, o.WFw pr F) → 32 * regCount P ≤ cost100 pr P
  | [], _ => by simp [regCount, cost100]
  | o :: P, hP => by
    have h1 := op_regs_le_price h o (hP o List.mem_cons_self)
    have h2 := regs_le_cost h P fun o' ho' => hP o' (List.mem_cons_of_mem _ ho')
    simp only [regCount, cost100, List.map_cons, List.sum_cons] at h2 ⊢
    omega

theorem shaCredit_nonneg (isA : Hash.Msg → Bool) (x : Hash.Msg) : 0 ≤ shaCredit isA x := by
  unfold shaCredit shaARegs shaWordsRegs
  split_ifs <;> positivity

/-- A nonnegative function sums less over a duplicate-free list than over any list holding its elements. -/
theorem sum_le_of_nodup_subset {α : Type} (f : α → ℚ) (hf : ∀ a, 0 ≤ f a) {l₁ l₂ : List α} (hd : l₁.Nodup)
    (hs : l₁ ⊆ l₂) : (l₁.map f).sum ≤ (l₂.map f).sum := by
  obtain ⟨l, hp, hsub⟩ := hd.subperm hs
  rw [← (hp.map f).sum_eq]
  exact (hsub.map f).sum_le_sum fun a ha => by
    obtain ⟨b, -, rfl⟩ := List.mem_map.1 ha
    exact hf b

/-- **The hash part is paid in compression registers**: at an accepted coin value where nothing hashed collides with
the honest commitment, a realizable strategy's compressions write at least the honest work's hash part. -/
theorem hash_le_shaRegs {L : Pouw.Lead.Layout} {H : Hash.Msg → Hash.Digest} {C : Pouw.Lead.Commit L} {isA : Hash.Msg → Bool}
    {κ : Fin L.m → Fin L.S → Fin L.n → ℤ} {Real : Strategy L → Prop} (hC : CommitL1w H C isA κ)
    (hsha : ShaWrites isA Real) (hhon : HonestHashRuled H C κ Real) {σ : Strategy L} (hσ : Real σ) {ω : Coins L}
    (hacc : Accepted L H C κ σ ω) (hnb : ¬ Break L H C κ σ ω) : hashRegs L C ≤ (σ.exec ω).shaRegs := by
  have hsub : (C.msgs H (word L κ ω) ++ C.aMsgs).dedup ⊆ (σ.exec ω).hashed := fun y hy => by
    rw [List.mem_dedup] at hy
    obtain ⟨x, hx, hxy⟩ := hhon σ hσ ω hacc y hy
    by_cases hne : x = y
    · exact hne ▸ hx
    · exact absurd ⟨x, List.mem_append_left _ hx, y, List.mem_append_right _ hy, hne, congrArg Verity.Digest.bytes hxy⟩ hnb
  calc hashRegs L C ≤ (((C.msgs H (word L κ ω) ++ C.aMsgs).dedup).map (shaCredit isA)).sum := hC.2 ω
    _ ≤ ((σ.exec ω).hashed.map (shaCredit isA)).sum :=
        sum_le_of_nodup_subset _ (shaCredit_nonneg isA) (List.nodup_dedup _) hsub
    _ ≤ _ := hsha σ hσ ω

theorem networkGap_nonneg (L : Pouw.Lead.Layout) : 0 ≤ networkGap L := by
  unfold networkGap stockRegs; positivity

/-- **The lead's work guarantee, from the words floor.** -/
theorem G_of_wordsFloor (hW : WordsFloor) : Pouw.Guarantees.Lead.G := by
  intro pr fpSem H T m n K κ stream C isA Device cert runC can Real hκ hraw hC h32 hw1 htile hlane hone hcard hsha hhon
    _hhost hout σ hcan O hacc
  by_cases hB : ∃ ω ∈ O, Break (lead T m n K) H C κ σ ω
  · exact Or.inr hB
  refine Or.inl ?_
  push Not at hB
  obtain ⟨σ', hσ', -, htwin⟩ := hout σ (hcard σ hcan)
  have hacc' : ∀ ω ∈ O, (σ'.exec ω).egress = 0 ∧ Accepted (lead T m n K) H C κ σ' ω := fun ω hω => by
    obtain ⟨he, hh, hr, -, -⟩ := htwin ω
    obtain ⟨hhold, hroot⟩ := hacc ω hω
    exact ⟨he, fun rb hrb cb hcb t => hh _ (hhold rb hrb cb hcb t), hr.trans hroot⟩
  have hnb' : ∀ ω ∈ O, ¬ Break (lead T m n K) H C κ σ' ω := fun ω hω => by
    unfold Break
    rw [(htwin ω).2.2.2.1]
    exact hB ω hω
  have hfloor := hW fpSem H T m n K κ stream C Real hκ hraw hC.1 htile hlane σ' hσ' O hacc'
  have hstep : ∀ ω ∈ O,
      32 * (((tileWords * writes (σ'.exec ω).steps : ℕ) : ℚ) + hashRegs (lead T m n K) C) -
          32 * networkGap (lead T m n K) ≤
        (σ.work pr ω : ℚ) := by
    intro ω hω
    have hreg : (32 * regCount (σ'.exec ω).trace : ℚ) ≤ σ'.work pr ω := by
      exact_mod_cast regs_le_cost h32 _ (hw1 σ' hσ' ω)
    have hmin : ((tileWords * writes (σ'.exec ω).steps + (σ'.exec ω).shaRegs : ℕ) : ℚ) ≤
        regCount (σ'.exec ω).trace := by
      exact_mod_cast hone σ' hσ' ω
    have hh := hash_le_shaRegs hC hsha hhon hσ' (hacc' ω hω).2 (hnb' ω hω)
    have hn := (htwin ω).2.2.2.2
    push_cast at hmin ⊢
    linarith
  have hsum := sum_le_sum hstep
  rw [sum_sub_distrib, ← mul_sum, sum_add_distrib, sum_const, nsmul_eq_mul, sum_const, nsmul_eq_mul] at hsum
  have hO : (O.card : ℚ) ≤ Fintype.card (Coins (lead T m n K)) := by exact_mod_cast card_le_univ O
  have hng := mul_le_mul_of_nonneg_right hO (networkGap_nonneg (lead T m n K))
  rw [honestRegs_eq, gapRegs_eq]
  linarith

end Pouw.SecurityProofs.Lead.Compose
