module

public import Definitions.Poub.Assumptions.RemoteHeld
public import Definitions.Poub.Remote.DrawInstances
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.Prob

/-! The premises of `Held` meet on a node that brings absent blocks back by rebuilds, its disks and its links at
once. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.Remote.Meets

open Finset ResourceAccounting.Remote.Held ResourceAccounting.Remote.Held.Assumptions ResourceAccounting.Remote.Instances Pouw

theorem over_pulse {Ω : Type} (t₁ B : ℕ) (ω : Ω) (u v : ℕ) : «over» (pulse t₁ B) ω u v ≤ B := by
  simp only [«over», pulse, sum_ite_eq']
  split_ifs <;> omega

theorem bucket_pulse {Ω : Type} {r : ℝ} (hr : 0 ≤ r) (t₁ B : ℕ) (ω : Ω) {u v : ℕ} :
    («over» (pulse t₁ B) ω u v : ℝ) ≤ r * (v - u : ℕ) + B := by
  have h : («over» (pulse t₁ B) ω u v : ℝ) ≤ B := by exact_mod_cast over_pulse t₁ B ω u v
  have h0 : 0 ≤ r * ((v - u : ℕ) : ℝ) := mul_nonneg hr (Nat.cast_nonneg _)
  linarith

theorem not_bucket_pulse {r : ℝ} {t₁ B : ℕ} (hr : r < B) :
    ¬ ∀ (ω : Unit) u v, u ≤ v → («over» (pulse t₁ B) ω u v : ℝ) ≤ r * (v - u : ℕ) + 0 := fun h => by
  have h1 := h () t₁ (t₁ + 1) (Nat.le_succ t₁)
  have h2 : «over» (pulse t₁ B) () t₁ (t₁ + 1) = B := by simp [«over», pulse]
  have h3 : t₁ + 1 - t₁ = 1 := by omega
  rw [h2, h3, Nat.cast_one, mul_one, add_zero] at h1
  linarith

variable {N D t₁ Br Bd Bl : ℕ} {Ω : Type}

theorem holds_of_all [Fintype Ω] {T : Trace Ω} {P : Ω → Prop} (h : ∀ ω, P ω) : T.Holds 0 P :=
  (pr_eq_zero (P := fun ω => T.accepts ω ∧ ¬ P ω) fun ω hω => hω.2 (h ω)).le

theorem burst_back (j : Fin N) (hj : (j : ℕ) < Br + Bd + Bl) (ω : Ω) :
    j ∉ (burst (Ω := Ω) N D t₁ Br Bd Bl).present ω t₁ ∧ j ∈ (burst (Ω := Ω) N D t₁ Br Bd Bl).present ω (t₁ + 1) := by
  refine ⟨fun h => ?_, ?_⟩
  · have h' : j ∈ burstMem N t₁ (Br + Bd + Bl) t₁ ∪ ∅ := h
    rw [union_empty, burstMem, if_pos (Nat.le_refl t₁), mem_filter] at h'
    exact absurd h'.2 (not_le.2 hj)
  · show j ∈ burstMem N t₁ (Br + Bd + Bl) (t₁ + 1) ∪ ∅
    rw [union_empty, burstMem, if_neg (by omega : ¬ t₁ + 1 ≤ t₁)]
    exact mem_univ j

theorem burst_routed : Routed (burst (Ω := Ω) N D t₁ Br Bd Bl) := by
  intro ω t
  show ((burstMem N t₁ (Br + Bd + Bl) (t + 1) ∪ ∅) \ (burstMem N t₁ (Br + Bd + Bl) t ∪ ∅)).card ≤
    pulse t₁ Br ω t + pulse t₁ Bd ω t + pulse t₁ Bl ω t
  simp only [union_empty, burstMem, pulse]
  rcases lt_trichotomy t t₁ with h | h | h
  · rw [if_pos (by omega : t + 1 ≤ t₁), if_pos (Nat.le_of_lt h), Finset.sdiff_self, card_empty]
    exact Nat.zero_le _
  · rw [if_neg (by omega : ¬ t + 1 ≤ t₁), if_pos (Nat.le_of_eq h), if_pos h, if_pos h, if_pos h]
    refine (card_le_card_of_injOn (t := range (Br + Bd + Bl)) (fun j : Fin N => (j : ℕ)) (fun j hj => ?_)
      (fun a _ b _ hab => Fin.ext hab)).trans_eq (card_range _)
    have h1 := (mem_sdiff.1 (mem_coe.1 hj)).2
    have h2 : ¬ Br + Bd + Bl ≤ (j : ℕ) := fun h => h1 (mem_filter.2 ⟨mem_univ _, h⟩)
    exact mem_coe.2 (mem_range.2 (show (j : ℕ) < Br + Bd + Bl by omega))
  · rw [if_neg (by omega : ¬ t + 1 ≤ t₁), if_neg (by omega : ¬ t ≤ t₁), Finset.sdiff_self, card_empty]
    exact Nat.zero_le _

theorem burst_chip (S : ℕ) : ChipCeiling (burst (Ω := Ω) N D t₁ Br Bd Bl) S := fun _ _ => by
  show (∅ : Finset (Fin N)).card ≤ S
  rw [card_empty]
  exact Nat.zero_le S

theorem burst_causal : Causal (burst (Ω := Ω) N D t₁ Br Bd Bl) 0 (2 * D) := fun _ _ =>
  ⟨fun _ _ h => Nat.mul_le_mul_right D h, Nat.zero_le _, Nat.le_refl _⟩

theorem burst_paced : Paced (burst (Ω := Ω) N D t₁ Br Bd Bl) (burstPace D) := fun _ _ =>
  ⟨fun _ _ h => Nat.mul_le_mul_right D h, rfl, Nat.le_refl _, fun g _ => Nat.lt_succ_self g,
    fun _ _ h => Nat.mul_le_mul_right D h, fun _ _ => Nat.le_refl _,
    fun _ _ _ h₁ h₂ => ⟨Nat.mul_le_mul_right D h₁, Nat.mul_le_mul_right D h₂⟩⟩

theorem burst_covers : Covers (burst (Ω := Ω) N D t₁ Br Bd Bl) := fun _ _ _ _ => rfl

theorem burst_sketch [Fintype Ω] (ht : t₁ + 1 < D) : SketchBound (burst (Ω := Ω) N D t₁ Br Bd Bl) 0 := by
  apply holds_of_all
  intro ω p _ j _
  refine ⟨p * D + (t₁ + 1), Nat.le_add_right _ _, ?_, ?_⟩
  · show p * D + (t₁ + 1) < (p + 1) * D
    rw [Nat.add_mul, Nat.one_mul]
    omega
  · show j ∈ burstMem N t₁ (Br + Bd + Bl) (p * D + (t₁ + 1)) ∪ ∅
    rw [union_empty, burstMem, if_neg (by omega : ¬ p * D + (t₁ + 1) ≤ t₁)]
    exact mem_univ j

theorem burst_coins [Fintype Ω] : LiveCoins (burst (Ω := Ω) N D t₁ Br Bd Bl) 0 0 := by
  apply holds_of_all
  intro _
  show ((∑ p ∈ range 2, ((univ : Finset (Fin N)) \ univ).card : ℕ) : ℝ) ≤ 0
  simp

theorem burst_identity : Identity (burst (Ω := Ω) N D t₁ Br Bd 0) := fun _ t =>
  show (if t = t₁ then 0 else 0) = 0 from ite_self 0

theorem burst_meets [Fintype Ω] {s E β βl : ℝ} (hs : 0 ≤ s) (hE : 0 ≤ E) (hβ : 0 ≤ β) (hβl : 0 ≤ βl) (S : ℕ)
    (ht : t₁ + 1 < D) :
    (∀ ω, (burst (Ω := Ω) N D t₁ Br Bd Bl).accepts ω) ∧ Links (burst (Ω := Ω) N D t₁ Br Bd Bl) βl Bl ∧
      SquaringCeiling (burst (Ω := Ω) N D t₁ Br Bd Bl) s E Br ∧ Disks (burst (Ω := Ω) N D t₁ Br Bd Bl) β Bd ∧
      ChipCeiling (burst (Ω := Ω) N D t₁ Br Bd Bl) S ∧ Routed (burst (Ω := Ω) N D t₁ Br Bd Bl) ∧
      Causal (burst (Ω := Ω) N D t₁ Br Bd Bl) 0 (2 * D) ∧ Paced (burst (Ω := Ω) N D t₁ Br Bd Bl) (burstPace D) ∧
      Covers (burst (Ω := Ω) N D t₁ Br Bd Bl) ∧ SketchBound (burst (Ω := Ω) N D t₁ Br Bd Bl) 0 ∧
      LiveCoins (burst (Ω := Ω) N D t₁ Br Bd Bl) 0 0 :=
  ⟨fun _ => trivial, fun ω _ _ _ => bucket_pulse hβl t₁ Bl ω, fun ω _ _ _ => bucket_pulse (div_nonneg hs hE) t₁ Br ω,
    fun ω _ _ _ => bucket_pulse hβ t₁ Bd ω, burst_chip S, burst_routed, burst_causal, burst_paced, burst_covers,
    burst_sketch ht, burst_coins⟩

theorem burst_meets_held [Fintype Ω] {s E β : ℝ} (hs : 0 ≤ s) (hE : 0 ≤ E) (hβ : 0 ≤ β) (S : ℕ) (ht : t₁ + 1 < D) :
    Identity (burst (Ω := Ω) N D t₁ Br Bd 0) ∧ SquaringCeiling (burst (Ω := Ω) N D t₁ Br Bd 0) s E Br ∧
      Disks (burst (Ω := Ω) N D t₁ Br Bd 0) β Bd ∧ ChipCeiling (burst (Ω := Ω) N D t₁ Br Bd 0) S ∧
      Routed (burst (Ω := Ω) N D t₁ Br Bd 0) ∧
      Causal (burst (Ω := Ω) N D t₁ Br Bd 0) 0 (2 * D) ∧ SketchBound (burst (Ω := Ω) N D t₁ Br Bd 0) 0 ∧
      LiveCoins (burst (Ω := Ω) N D t₁ Br Bd 0) 0 0 := by
  obtain ⟨-, -, hsq, hd, hS, hr, hc, -, -, hk, hw⟩ :=
    burst_meets (Ω := Ω) (N := N) (D := D) (t₁ := t₁) (Br := Br) (Bd := Bd) (Bl := 0) (βl := 0) hs hE hβ le_rfl S ht
  exact ⟨burst_identity, hsq, hd, hS, hr, hc, hk, hw⟩

def w2 : Trace Unit := burst 130023424 85000000 0 743590 1054688 1536

def w2Held : Trace Unit := burst 130023424 85000000 0 743590 1054688 0

theorem w2_meets :
    w2.accepts () ∧ Links w2 (50e9 / 4096 / 1e9) (1536 : ℕ) ∧ SquaringCeiling w2 191.5 98016 (743590 : ℕ) ∧
      Disks w2 (4.32e9 / 4096 / 1e9) (1054688 : ℕ) ∧ ChipCeiling w2 743590 ∧ Routed w2 ∧
      Causal w2 0 (2 * 85000000) ∧ Paced w2 (burstPace 85000000) ∧ Covers w2 ∧ SketchBound w2 0 ∧
      LiveCoins w2 0 0 ∧
      (∀ j : Fin 130023424, (j : ℕ) < 743590 + 1054688 + 1536 → j ∉ w2.present () 0 ∧ j ∈ w2.present () 1) ∧
      ¬ Links w2 (50e9 / 4096 / 1e9) 0 ∧ ¬ SquaringCeiling w2 191.5 98016 0 ∧
      ¬ Disks w2 (4.32e9 / 4096 / 1e9) 0 := by
  obtain ⟨ha, hl, hsq, hd, hS, hr, hc, hp, hcov, hk, hw⟩ :=
    burst_meets (Ω := Unit) (N := 130023424) (D := 85000000) (t₁ := 0) (Br := 743590) (Bd := 1054688) (Bl := 1536)
      (s := 191.5) (E := 98016) (β := 4.32e9 / 4096 / 1e9) (βl := 50e9 / 4096 / 1e9)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) 743590 (by norm_num)
  exact ⟨ha (), hl, hsq, hd, hS, hr, hc, hp, hcov, hk, hw, fun j hj => burst_back j hj (),
    not_bucket_pulse (by norm_num), not_bucket_pulse (by norm_num), not_bucket_pulse (by norm_num)⟩

theorem w2_meets_held :
    Identity w2Held ∧ SquaringCeiling w2Held 191.5 98016 (743590 : ℕ) ∧
      Disks w2Held (4.32e9 / 4096 / 1e9) (1054688 : ℕ) ∧ ChipCeiling w2Held 743590 ∧ Routed w2Held ∧
      Causal w2Held 0 (2 * 85000000) ∧ SketchBound w2Held 0 ∧ LiveCoins w2Held 0 0 :=
  burst_meets_held (Ω := Unit) (N := 130023424) (D := 85000000) (t₁ := 0) (Br := 743590) (Bd := 1054688)
    (s := 191.5) (E := 98016) (β := 4.32e9 / 4096 / 1e9) (by norm_num) (by norm_num) (by norm_num) 743590
    (by norm_num)

end ResourceAccounting.SecurityProofs.Remote.Meets
