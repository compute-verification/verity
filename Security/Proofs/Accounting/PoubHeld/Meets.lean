import Proofs.Accounting.PoubHeld
import Mathlib.Tactic.NormNum.OfScientific

/-!
# The premises of `Held` meet on a node that uses every route

The token-bucket ceilings (`SquaringCeiling`, `Disks`, `Links`) hold together with every other premise of `Held`,
`HeldOverTime` and `HeldOverTimeLinks` on a session whose node brings absent blocks back by all three routes at once,
so the guarantees cover nodes that rebuild, read their disks and take blocks over their links.

The session (`burst`) is two passes of `D` ticks, every receipt correct. Its node has its first `Br + Bd + Bl`
registered blocks absent until tick `t₁` and brings them all back at `t₁`: `Br` rebuilt, `Bd` read from its disks, `Bl`
over its links (`burst_back`). It meets every premise at any rates (`burst_meets`, and `burst_meets_held` for `Held`'s
Identity), while no rate below the burst meets the same ceiling with burst 0 (`not_bucket_pulse`): that was the
bare-rate form, which over one-tick intervals allowed no rebuild, disk read or link block at all.

`w2_meets` and `w2_meets_held` are W2's instance: 8 × Qwen2.5-32B on one node, at the ledger's nanosecond tick.
-/

namespace Accounting.SecurityProofs.Poub.Meets

open Finset Accounting.Poub.Held Accounting.Poub.Held.Assumptions Pouw

/-- `B` blocks at tick `t₁` and none at any other. -/
def pulse (t₁ B : ℕ) : Unit → ℕ → ℕ := fun _ t => if t = t₁ then B else 0

theorem over_pulse (t₁ B : ℕ) (ω : Unit) (u v : ℕ) : over (pulse t₁ B) ω u v ≤ B := by
  simp only [over, pulse, sum_ite_eq']
  split_ifs <;> omega

/-- A pulse meets the token bucket at any rate with itself for the burst. -/
theorem bucket_pulse {r : ℝ} (hr : 0 ≤ r) (t₁ B : ℕ) (ω : Unit) {u v : ℕ} :
    (over (pulse t₁ B) ω u v : ℝ) ≤ r * (v - u : ℕ) + B := by
  have h : (over (pulse t₁ B) ω u v : ℝ) ≤ B := by exact_mod_cast over_pulse t₁ B ω u v
  have h0 : 0 ≤ r * ((v - u : ℕ) : ℝ) := mul_nonneg hr (Nat.cast_nonneg _)
  linarith

/-- A pulse of `B` meets no ceiling of rate below `B` with burst 0: over its one tick, it brings in `B`. -/
theorem not_bucket_pulse {r : ℝ} {t₁ B : ℕ} (hr : r < B) :
    ¬ ∀ ω u v, u ≤ v → (over (pulse t₁ B) ω u v : ℝ) ≤ r * (v - u : ℕ) + 0 := fun h => by
  have h1 := h () t₁ (t₁ + 1) (Nat.le_succ t₁)
  have h2 : over (pulse t₁ B) () t₁ (t₁ + 1) = B := by simp [over, pulse]
  have h3 : t₁ + 1 - t₁ = 1 := by omega
  rw [h2, h3, Nat.cast_one, mul_one, add_zero] at h1
  linarith

/-- The registered blocks in node memory at tick `t`: all of them after `t₁`, and until then all but the first `k`. -/
def burstMem (N t₁ k t : ℕ) : Finset (Fin N) :=
  if t ≤ t₁ then (univ.filter fun j : Fin N => k ≤ (j : ℕ)) else univ

/-- Two passes of `D` ticks, each reading every registered block correctly, on a node whose first `Br + Bd + Bl`
registered blocks are absent until tick `t₁` and all come back at `t₁`: `Br` rebuilt, `Bd` read from its disks and `Bl`
carried over its links. -/
def burst (N D t₁ Br Bd Bl : ℕ) : Trace Unit where
  blocks := N
  passes := 2
  accepts := fun _ => True
  tick := fun _ p => p * D
  plan := fun _ _ => univ
  correct := fun _ _ => univ
  mem := fun _ t => burstMem N t₁ (Br + Bd + Bl) t
  chip := fun _ _ => ∅
  rebuilt := pulse t₁ Br
  disk := pulse t₁ Bd
  link := pulse t₁ Bl

/-- Its groups: each pass its own, the first credited and the second closing, each coin released as the group before
ends. -/
def burstPace (D : ℕ) : Pace Unit where
  credited := 1
  first := fun _ g => g
  released := fun _ g => g * D
  ended := fun _ g => (g + 1) * D

variable {N D t₁ Br Bd Bl : ℕ}

theorem holds_of_all {T : Trace Unit} {P : Unit → Prop} (h : ∀ ω, P ω) : T.Holds 0 P :=
  (pr_eq_zero (P := fun ω => T.accepts ω ∧ ¬ P ω) fun ω hω => hω.2 (h ω)).le

/-- Its first `Br + Bd + Bl` registered blocks are absent at `t₁` and back at `t₁ + 1`. -/
theorem burst_back (j : Fin N) (hj : (j : ℕ) < Br + Bd + Bl) (ω : Unit) :
    j ∉ (burst N D t₁ Br Bd Bl).present ω t₁ ∧ j ∈ (burst N D t₁ Br Bd Bl).present ω (t₁ + 1) := by
  refine ⟨fun h => ?_, ?_⟩
  · have h' : j ∈ burstMem N t₁ (Br + Bd + Bl) t₁ ∪ ∅ := h
    rw [union_empty, burstMem, if_pos (Nat.le_refl t₁), mem_filter] at h'
    exact absurd h'.2 (not_le.2 hj)
  · show j ∈ burstMem N t₁ (Br + Bd + Bl) (t₁ + 1) ∪ ∅
    rw [union_empty, burstMem, if_neg (by omega : ¬ t₁ + 1 ≤ t₁)]
    exact mem_univ j

theorem burst_routed : Routed (burst N D t₁ Br Bd Bl) := by
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

theorem burst_chip (S : ℕ) : ChipCeiling (burst N D t₁ Br Bd Bl) S := fun _ _ => by
  show (∅ : Finset (Fin N)).card ≤ S
  rw [card_empty]
  exact Nat.zero_le S

theorem burst_causal : Causal (burst N D t₁ Br Bd Bl) 0 (2 * D) := fun _ _ =>
  ⟨fun _ _ h => Nat.mul_le_mul_right D h, Nat.zero_le _, Nat.le_refl _⟩

theorem burst_paced : Paced (burst N D t₁ Br Bd Bl) (burstPace D) := fun _ _ =>
  ⟨fun _ _ h => Nat.mul_le_mul_right D h, rfl, Nat.le_refl _, fun g _ => Nat.lt_succ_self g,
    fun _ _ h => Nat.mul_le_mul_right D h, fun _ _ => Nat.le_refl _,
    fun _ _ _ h₁ h₂ => ⟨Nat.mul_le_mul_right D h₁, Nat.mul_le_mul_right D h₂⟩⟩

theorem burst_covers : Covers (burst N D t₁ Br Bd Bl) := fun _ _ _ _ => rfl

theorem burst_sketch (ht : t₁ + 1 < D) : SketchBound (burst N D t₁ Br Bd Bl) 0 := by
  apply holds_of_all
  intro ω p _ j _
  refine ⟨p * D + (t₁ + 1), Nat.le_add_right _ _, ?_, ?_⟩
  · show p * D + (t₁ + 1) < (p + 1) * D
    rw [Nat.add_mul, Nat.one_mul]
    omega
  · show j ∈ burstMem N t₁ (Br + Bd + Bl) (p * D + (t₁ + 1)) ∪ ∅
    rw [union_empty, burstMem, if_neg (by omega : ¬ p * D + (t₁ + 1) ≤ t₁)]
    exact mem_univ j

theorem burst_coins : LiveCoins (burst N D t₁ Br Bd Bl) 0 0 := by
  apply holds_of_all
  intro _
  show ((∑ p ∈ range 2, ((univ : Finset (Fin N)) \ univ).card : ℕ) : ℝ) ≤ 0
  simp

theorem burst_identity : Identity (burst N D t₁ Br Bd 0) := fun _ t =>
  show (if t = t₁ then 0 else 0) = 0 from ite_self 0

/-- **The premises meet with every route used.** At any rates, the session is accepted and meets every premise of
`HeldOverTimeLinks` (`HeldOverTime`'s and `Held`'s but Identity, with `Causal` over `[0, 2·D]`), its node bringing
`Br + Bd + Bl` absent blocks back at once by rebuilds, disk reads and links. -/
theorem burst_meets {s E β βl : ℝ} (hs : 0 ≤ s) (hE : 0 ≤ E) (hβ : 0 ≤ β) (hβl : 0 ≤ βl) (S : ℕ)
    (ht : t₁ + 1 < D) :
    (burst N D t₁ Br Bd Bl).accepts () ∧ Links (burst N D t₁ Br Bd Bl) βl Bl ∧
      SquaringCeiling (burst N D t₁ Br Bd Bl) s E Br ∧ Disks (burst N D t₁ Br Bd Bl) β Bd ∧
      ChipCeiling (burst N D t₁ Br Bd Bl) S ∧ Routed (burst N D t₁ Br Bd Bl) ∧
      Causal (burst N D t₁ Br Bd Bl) 0 (2 * D) ∧ Paced (burst N D t₁ Br Bd Bl) (burstPace D) ∧
      Covers (burst N D t₁ Br Bd Bl) ∧ SketchBound (burst N D t₁ Br Bd Bl) 0 ∧
      LiveCoins (burst N D t₁ Br Bd Bl) 0 0 :=
  ⟨trivial, fun ω _ _ _ => bucket_pulse hβl t₁ Bl ω, fun ω _ _ _ => bucket_pulse (div_nonneg hs hE) t₁ Br ω,
    fun ω _ _ _ => bucket_pulse hβ t₁ Bd ω, burst_chip S, burst_routed, burst_causal, burst_paced, burst_covers,
    burst_sketch ht, burst_coins⟩

/-- **`Held`'s premises meet with rebuilds and disk reads used**: the same session with no link block meets Identity
too. -/
theorem burst_meets_held {s E β : ℝ} (hs : 0 ≤ s) (hE : 0 ≤ E) (hβ : 0 ≤ β) (S : ℕ) (ht : t₁ + 1 < D) :
    Identity (burst N D t₁ Br Bd 0) ∧ SquaringCeiling (burst N D t₁ Br Bd 0) s E Br ∧
      Disks (burst N D t₁ Br Bd 0) β Bd ∧ ChipCeiling (burst N D t₁ Br Bd 0) S ∧ Routed (burst N D t₁ Br Bd 0) ∧
      Causal (burst N D t₁ Br Bd 0) 0 (2 * D) ∧ SketchBound (burst N D t₁ Br Bd 0) 0 ∧
      LiveCoins (burst N D t₁ Br Bd 0) 0 0 := by
  obtain ⟨-, -, hsq, hd, hS, hr, hc, -, -, hk, hw⟩ :=
    burst_meets (N := N) (D := D) (t₁ := t₁) (Br := Br) (Bd := Bd) (Bl := 0) (βl := 0) hs hE hβ le_rfl S ht
  exact ⟨burst_identity, hsq, hd, hS, hr, hc, hk, hw⟩

/-- W2's session in this form, at the nanosecond tick: `N = 130,023,424` registered blocks, passes of 85 ms, and one
rebuild batch (`B_r = S = 743,590` blocks, what the chips hold in flight), one disk read (`B_d = 1,054,688` blocks,
4.32·10⁹ B, the disks' ceiling over the one-second window it is measured in) and one link window (`B_l = 1,536`
blocks, 6 MiB, the node's largest TCP receive window) coming back at tick 0. -/
def w2 : Trace Unit := burst 130023424 85000000 0 743590 1054688 1536

/-- W2's session with no link block, for `Held`'s Identity. -/
def w2Held : Trace Unit := burst 130023424 85000000 0 743590 1054688 0

/-- **The premises meet at W2's instance** (8 × Qwen2.5-32B on one node, its disks and NIC as
measured), at the nanosecond tick: the squaring ceiling `s = 191.5` a tick
(1.915·10¹¹ a second) at `E = 98,016` a block (p1024/k3/masked), the disks' `β = 4.32·10⁹ / 4096 / 10⁹` blocks a tick
(the three disks' highest seconds, vdb by rate), the one 400 Gb/s NIC's `β_l = 50·10⁹ / 4096 / 10⁹`, and the chips'
`S = 743,590` blocks. W2's session meets every premise of `HeldOverTimeLinks` with its first `B_r + B_d + B_l` blocks
brought back at once, and meets none of the three ceilings with burst 0. -/
theorem w2_meets :
    w2.accepts () ∧ Links w2 (50e9 / 4096 / 1e9) (1536 : ℕ) ∧ SquaringCeiling w2 191.5 98016 (743590 : ℕ) ∧
      Disks w2 (4.32e9 / 4096 / 1e9) (1054688 : ℕ) ∧ ChipCeiling w2 743590 ∧ Routed w2 ∧
      Causal w2 0 (2 * 85000000) ∧ Paced w2 (burstPace 85000000) ∧ Covers w2 ∧ SketchBound w2 0 ∧
      LiveCoins w2 0 0 ∧
      (∀ j : Fin 130023424, (j : ℕ) < 743590 + 1054688 + 1536 → j ∉ w2.present () 0 ∧ j ∈ w2.present () 1) ∧
      ¬ Links w2 (50e9 / 4096 / 1e9) 0 ∧ ¬ SquaringCeiling w2 191.5 98016 0 ∧
      ¬ Disks w2 (4.32e9 / 4096 / 1e9) 0 := by
  obtain ⟨ha, hl, hsq, hd, hS, hr, hc, hp, hcov, hk, hw⟩ :=
    burst_meets (N := 130023424) (D := 85000000) (t₁ := 0) (Br := 743590) (Bd := 1054688) (Bl := 1536)
      (s := 191.5) (E := 98016) (β := 4.32e9 / 4096 / 1e9) (βl := 50e9 / 4096 / 1e9)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) 743590 (by norm_num)
  exact ⟨ha, hl, hsq, hd, hS, hr, hc, hp, hcov, hk, hw, fun j hj => burst_back j hj (),
    not_bucket_pulse (by norm_num), not_bucket_pulse (by norm_num), not_bucket_pulse (by norm_num)⟩

/-- **`Held`'s premises meet at W2's instance**, with its rebuild batch and disk read brought back at once and no link
block. -/
theorem w2_meets_held :
    Identity w2Held ∧ SquaringCeiling w2Held 191.5 98016 (743590 : ℕ) ∧
      Disks w2Held (4.32e9 / 4096 / 1e9) (1054688 : ℕ) ∧ ChipCeiling w2Held 743590 ∧ Routed w2Held ∧
      Causal w2Held 0 (2 * 85000000) ∧ SketchBound w2Held 0 ∧ LiveCoins w2Held 0 0 :=
  burst_meets_held (N := 130023424) (D := 85000000) (t₁ := 0) (Br := 743590) (Bd := 1054688)
    (s := 191.5) (E := 98016) (β := 4.32e9 / 4096 / 1e9) (by norm_num) (by norm_num) (by norm_num) 743590
    (by norm_num)

end Accounting.SecurityProofs.Poub.Meets
