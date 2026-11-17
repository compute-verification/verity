module

public import Proofs.Poub.BoundReceipt.Interactive
public import Proofs.Poub.RateAudit.Witnesses

/-! The honest and nontrivial lemmas of `BoundReceiptLink` and `BoundReceiptLinkInteractive`, at one block of one
bit. -/

@[expose] public section

namespace ResourceAccounting.BoundReceipt

open ResourceAccounting.Rate
open Pous (Bits)

def idleRest (N a h Lz qz : ℕ) : Rest N a h Lz qz where
  mem _ := false
  side _ _ := false
  chip := .done fun _ _ => false
  bounded _ := ⟨Nat.zero_le _, Nat.zero_le _⟩

def idleNode : Node 1 1 1 1 0 0 1 0 0 where
  gpu _ := Completeness.honest 1 1 1 0
  rest _ := idleRest 1 1 1 0 0

def blankNode : Node 1 1 1 2 0 0 0 0 0 where
  gpu _ := blank 1 1 2 0 0 0 0
  rest _ := idleRest 1 1 2 0 0

theorem idleNode_event (E : Store 1 1) (z : Store 1 1) (G : Mix 1 1 1 1) (I : Fin 1 → Fin 1) :
    idleNode.Accepts E z G I ∧ idleNode.compute 1 1 E z G ≤ 1 ∧ idleNode.tileWork 1 E z G ≤ 0 ∧
      ¬ idleNode.footprint E z G * 1 < floor 1 0 0 0 1 1 1 (1 - 0) 0 := by
  obtain ⟨ha, hf, hc⟩ := honest_one E noStream (atZ G z) I
  refine ⟨fun i => Or.inl (ha i), ?_, ?_, ?_⟩
  · show (Completeness.honest 1 1 1 0).compute 1 1 E noStream (atZ G z) ≤ 1
    omega
  · simp [Node.tileWork, idleNode, idleRest, Chip.rebuilds]
  · show ¬ (Completeness.honest 1 1 1 0).footprint E noStream (atZ G z) * 1 < floor 1 0 0 0 1 1 1 (1 - 0) 0
    rw [hf]
    unfold floor
    omega

theorem blankNode_event (E : Store 1 1) (z : Store 1 1) (G : Mix 1 1 1 2) :
    blankNode.compute 0 1 E z G ≤ 0 ∧ blankNode.tileWork 1 E z G ≤ 0 ∧
      blankNode.footprint E z G * 1 < floor 1 0 0 0 0 0 1 (1 - 0) 0 := by
  refine ⟨?_, ?_, ?_⟩
  · show (blank 1 1 2 0 0 0 0).compute 0 1 E noStream (atZ G z) ≤ 0
    exact (blank_compute _ _ _ _ _).le
  · simp [Node.tileWork, blankNode, idleRest, Chip.rebuilds]
  · show (blank 1 1 2 0 0 0 0).footprint E noStream (atZ G z) * 1 < floor 1 0 0 0 0 0 1 (1 - 0) 0
    rw [blank_footprint]
    unfold floor
    omega

end ResourceAccounting.BoundReceipt

namespace ResourceAccounting.SecurityProofs

open ResourceAccounting.Rate ResourceAccounting.BoundReceipt
open Pous (Bits pr)
open scoped ENNReal

theorem link_error_lt :
    (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + 2 * ((1 : ℕ) : ℝ≥0∞) / 2 ^ 2 +
        ((Nat.choose 0 (1 - 0) : ℕ) : ℝ≥0∞) / 2 ^ 0 + ((Nat.choose 0 (0 + 1) : ℕ) : ℝ≥0∞) / 2 ^ 0 < 1 := by
  have t1 : (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 = 0 := by simp
  have t2 : 2 * ((1 : ℕ) : ℝ≥0∞) / 2 ^ 2 = 2⁻¹ := by
    rw [Nat.cast_one, mul_one, pow_two, ENNReal.div_eq_inv_mul,
      ENNReal.mul_inv (Or.inl two_ne_zero) (Or.inl ENNReal.ofNat_ne_top), mul_assoc,
      ENNReal.inv_mul_cancel two_ne_zero ENNReal.ofNat_ne_top, mul_one]
  have t3 : ((Nat.choose 0 (1 - 0) : ℕ) : ℝ≥0∞) / 2 ^ 0 = 0 := by
    rw [show Nat.choose 0 (1 - 0) = 0 from rfl]
    simp
  simp only [t1, t2, t3, zero_add, add_zero]
  exact ENNReal.inv_lt_one.2 ENNReal.one_lt_two

/-- **`BoundReceiptLink`'s honest lemma**: some node, the one whose GPU tags every block and whose rest does nothing,
at one block of one bit, is accepted within one mix query's compute and no tile work, with its footprint at the floor,
with probability 1. -/
theorem BoundReceiptLink_honest :
    ∃ n : Node 1 1 1 1 0 0 1 0 0,
      pr (fun x : Store 1 1 × BoundReceipt.Coins 1 1 1 1 1 =>
          n.Accepts x.1 x.2.1.1 x.2.1.2 x.2.2 ∧ n.compute 1 1 x.1 x.2.1.1 x.2.1.2 ≤ 1 ∧
            n.tileWork 1 x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧
            ¬ n.footprint x.1 x.2.1.1 x.2.1.2 * 1 < floor 1 0 0 0 1 1 1 (1 - 0) 0) = 1 :=
  ⟨idleNode, pr_one fun x => idleNode_event x.1 x.2.1.1 x.2.1.2 x.2.2⟩

/-- **`BoundReceiptLink`'s nontrivial lemma**: with the check accepting everything, some node, one that reads nothing,
is within its compute and tile work and short of the floor on every coin, above the error `2^-1`. -/
theorem BoundReceiptLink_nontrivial :
    ∃ n : Node 1 1 1 2 0 0 0 0 0,
      ¬ pr (fun x : Store 1 1 × BoundReceipt.Coins 1 1 1 2 1 =>
          True ∧ n.compute 0 1 x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧ n.tileWork 1 x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧
            n.footprint x.1 x.2.1.1 x.2.1.2 * 1 < floor 1 0 0 0 0 0 1 (1 - 0) 0)
        ≤ (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + 2 * ((1 : ℕ) : ℝ≥0∞) / 2 ^ 2 +
          ((Nat.choose 0 (1 - 0) : ℕ) : ℝ≥0∞) / 2 ^ 0 + ((Nat.choose 0 (0 + 1) : ℕ) : ℝ≥0∞) / 2 ^ 0 := by
  refine ⟨blankNode, ?_⟩
  have hev : ∀ x : Store 1 1 × BoundReceipt.Coins 1 1 1 2 1,
      True ∧ blankNode.compute 0 1 x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧ blankNode.tileWork 1 x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧
        blankNode.footprint x.1 x.2.1.1 x.2.1.2 * 1 < floor 1 0 0 0 0 0 1 (1 - 0) 0 := fun x =>
    ⟨trivial, blankNode_event x.1 x.2.1.1 x.2.1.2⟩
  rw [pr_one hev, not_le]
  exact link_error_lt

/-- **`BoundReceiptLinkInteractive`'s honest lemma**: `BoundReceiptLink_honest`'s node, as a node whose answers
ignore the mix, has `QueryDetermined` and passes as it does there, with probability 1. -/
theorem BoundReceiptLinkInteractive_honest :
    ∃ n : Node 1 1 1 1 0 0 1 0 0,
      (InteractiveNode.ofNode n).QueryDetermined ∧
        pr (fun x : Store 1 1 × BoundReceipt.Coins 1 1 1 1 1 =>
            ((InteractiveNode.ofNode n).under x.2.1.2).Accepts x.1 x.2.1.1 x.2.1.2 x.2.2 ∧
              ((InteractiveNode.ofNode n).under x.2.1.2).compute 1 1 x.1 x.2.1.1 x.2.1.2 ≤ 1 ∧
              ((InteractiveNode.ofNode n).under x.2.1.2).tileWork 1 x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧
              ¬ ((InteractiveNode.ofNode n).under x.2.1.2).footprint x.1 x.2.1.1 x.2.1.2 * 1 <
                floor 1 0 0 0 1 1 1 (1 - 0) 0) = 1 :=
  ⟨idleNode, InteractiveNode.ofNode_queryDetermined idleNode, pr_one fun x => by
    rw [InteractiveNode.under_ofNode]
    exact idleNode_event x.1 x.2.1.1 x.2.1.2 x.2.2⟩

/-- **`BoundReceiptLinkInteractive`'s nontrivial lemma**: `BoundReceiptLink_nontrivial`'s node, as a node whose
answers ignore the mix, with the check accepting everything, above the error `2^-1`. -/
theorem BoundReceiptLinkInteractive_nontrivial :
    ∃ n : Node 1 1 1 2 0 0 0 0 0,
      ¬ pr (fun x : Store 1 1 × BoundReceipt.Coins 1 1 1 2 1 =>
          True ∧ ((InteractiveNode.ofNode n).under x.2.1.2).compute 0 1 x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧
            ((InteractiveNode.ofNode n).under x.2.1.2).tileWork 1 x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧
            ((InteractiveNode.ofNode n).under x.2.1.2).footprint x.1 x.2.1.1 x.2.1.2 * 1 <
              floor 1 0 0 0 0 0 1 (1 - 0) 0)
        ≤ (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + 2 * ((1 : ℕ) : ℝ≥0∞) / 2 ^ 2 +
          ((Nat.choose 0 (1 - 0) : ℕ) : ℝ≥0∞) / 2 ^ 0 + ((Nat.choose 0 (0 + 1) : ℕ) : ℝ≥0∞) / 2 ^ 0 := by
  refine ⟨blankNode, ?_⟩
  have hev : ∀ x : Store 1 1 × BoundReceipt.Coins 1 1 1 2 1,
      True ∧ ((InteractiveNode.ofNode blankNode).under x.2.1.2).compute 0 1 x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧
        ((InteractiveNode.ofNode blankNode).under x.2.1.2).tileWork 1 x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧
        ((InteractiveNode.ofNode blankNode).under x.2.1.2).footprint x.1 x.2.1.1 x.2.1.2 * 1 <
          floor 1 0 0 0 0 0 1 (1 - 0) 0 := fun x => by
    rw [InteractiveNode.under_ofNode]
    exact ⟨trivial, blankNode_event x.1 x.2.1.1 x.2.1.2⟩
  rw [pr_one hev, not_le]
  exact link_error_lt

end ResourceAccounting.SecurityProofs
