import Definitions.Accounting.Guarantees.PoubHeld
import Proofs.Pouw.Prob
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

/-!
# The snapshot a PoUB shows held

`Accounting.Guarantees.Poub.Held` from its premises: at each outcome where the sketch bound and the live coins' bound
hold (a union bound, `holdsAnd`), each credited pass's registered blocks are its wrong receipts, the blocks in node
memory at its start, those on a chip then (at most `S`) and those that came in during the pass (`pass_le`). The passes
are consecutive (`sum_passes`), so what came in during them came in over `[t_0, t_P) ⊆ [s₀, e]`, one interval: at most
`s/E + β` a tick and one burst of each, `Br + Bd`, by rebuilds and disk reads, none over the links.

`heldOverTime_of_links` reduces `HeldOverTime` to `HeldOverTimeLinks`, at links of rate 0, which is all Identity says.
Both time-held forms are proved in `Proofs.Accounting.PoubHeld.OverTime`.
-/

namespace Accounting.SecurityProofs.Poub

open Finset Accounting.Poub.Held Accounting.Poub.Held.Assumptions Pouw

variable {Ω : Type} {T : Trace Ω}

section Holds

variable [Fintype Ω]

theorem holdsMono {δ : ℝ} {P Q : Ω → Prop} (h : ∀ ω, T.accepts ω → P ω → Q ω) (hp : T.Holds δ P) :
    T.Holds δ Q :=
  (pr_mono fun ω ⟨ha, hq⟩ => ⟨ha, fun hp => hq (h ω ha hp)⟩).trans hp

/-- The union bound: two bounds hold together except with their errors summed. -/
theorem holdsAnd {δ₁ δ₂ : ℝ} {P Q : Ω → Prop} (h₁ : T.Holds δ₁ P) (h₂ : T.Holds δ₂ Q) :
    T.Holds (δ₁ + δ₂) fun ω => P ω ∧ Q ω := by
  refine (pr_mono (P' := fun ω => (T.accepts ω ∧ ¬ P ω) ∨ (T.accepts ω ∧ ¬ Q ω)) ?_).trans
    ((pr_or_le _ _).trans (add_le_add h₁ h₂))
  intro ω ⟨ha, hn⟩
  by_cases hp : P ω
  · exact Or.inr ⟨ha, fun hq => hn ⟨hp, hq⟩⟩
  · exact Or.inl ⟨ha, hp⟩

end Holds

/-- A block present at `t ≥ u` was present at `u` or came in at some tick of `[u, t)`. -/
theorem present_subset (ω : Ω) {u t : ℕ} (h : u ≤ t) :
    T.present ω t ⊆ T.present ω u ∪ (Ico u t).biUnion fun τ => T.present ω (τ + 1) \ T.present ω τ := by
  induction t, h using Nat.le_induction with
  | base => exact subset_union_left
  | succ t hut ih =>
    intro j hj
    by_cases hp : j ∈ T.present ω t
    · rcases mem_union.1 (ih hp) with h | h
      · exact mem_union_left _ h
      · refine mem_union_right _ ?_
        obtain ⟨τ, hτ, hj'⟩ := mem_biUnion.1 h
        exact mem_biUnion.2 ⟨τ, by simp only [mem_Ico] at hτ ⊢; omega, hj'⟩
    · exact mem_union_right _ (mem_biUnion.2 ⟨t, by simp only [mem_Ico]; omega, mem_sdiff.2 ⟨hj, hp⟩⟩)

/-- One pass: its registered blocks are its wrong receipts, the blocks in node memory at its start, those on a chip
then, and those that came in during it. -/
theorem pass_le (ω : Ω) {S : ℕ} (hS : ∀ t, (T.chip ω t).card ≤ S) (p : ℕ)
    (hk : ∀ j ∈ T.correct ω p, ∃ t, T.tick ω p ≤ t ∧ t < T.tick ω (p + 1) ∧ j ∈ T.present ω t) :
    (T.plan ω p).card ≤ (T.plan ω p \ T.correct ω p).card + (T.mem ω (T.tick ω p)).card + S +
      ∑ τ ∈ Ico (T.tick ω p) (T.tick ω (p + 1)), T.entered ω τ := by
  have h1 := card_le_card_sdiff_add_card (s := T.plan ω p) (t := T.correct ω p)
  have h2 := card_le_card_sdiff_add_card (s := T.correct ω p) (t := T.present ω (T.tick ω p))
  have h3 : (T.correct ω p \ T.present ω (T.tick ω p)).card ≤
      ∑ τ ∈ Ico (T.tick ω p) (T.tick ω (p + 1)), T.entered ω τ := by
    refine (card_le_card ?_).trans card_biUnion_le
    intro j hj
    obtain ⟨hc, hn⟩ := mem_sdiff.1 hj
    obtain ⟨t, hut, htv, hjt⟩ := hk j hc
    rcases mem_union.1 (present_subset ω hut hjt) with h | h
    · exact absurd h hn
    · obtain ⟨τ, hτ, hj'⟩ := mem_biUnion.1 h
      exact mem_biUnion.2 ⟨τ, by simp only [mem_Ico] at hτ ⊢; omega, hj'⟩
  have h4 : (T.present ω (T.tick ω p)).card ≤ (T.mem ω (T.tick ω p)).card + S :=
    (card_union_le (T.mem ω (T.tick ω p)) (T.chip ω (T.tick ω p))).trans
      (Nat.add_le_add_left (hS _) _)
  omega

/-- Consecutive passes: the ticks of each, summed over the passes, are the ticks from the first's start to the last's
end. -/
theorem sum_passes (f : ℕ → ℕ) (t : ℕ → ℕ) (ht : Monotone t) (P : ℕ) :
    ∑ p ∈ range P, ∑ τ ∈ Ico (t p) (t (p + 1)), f τ = ∑ τ ∈ Ico (t 0) (t P), f τ := by
  induction P with
  | zero => simp
  | succ n ih => rw [sum_range_succ, ih, sum_Ico_consecutive _ (ht (Nat.zero_le n)) (ht (Nat.le_succ n))]

/-- `Held` at one outcome. -/
theorem held_of (ω : Ω) {S s₀ e : ℕ} {s E β Br Bd W : ℝ} (hE : 0 < E) (hs : 0 ≤ s) (hβ : 0 ≤ β)
    (hI : Identity T) (hsq : SquaringCeiling T s E Br) (hd : Disks T β Bd) (hS : ChipCeiling T S) (hr : Routed T)
    (hm : Monotone (T.tick ω)) (h0 : s₀ ≤ T.tick ω 0) (hP : T.tick ω T.passes ≤ e)
    (hk : ∀ p < T.passes, ∀ j ∈ T.correct ω p,
      ∃ t, T.tick ω p ≤ t ∧ t < T.tick ω (p + 1) ∧ j ∈ T.present ω t)
    (hw : (T.wrong ω : ℝ) ≤ W) :
    (T.planned ω : ℝ) - W - ((s / E + β) * ((e : ℝ) - s₀) + (Br + Bd)) - T.passes * S ≤ T.held ω := by
  have huv : T.tick ω 0 ≤ T.tick ω T.passes := hm (Nat.zero_le _)
  have hsum : T.planned ω ≤ T.wrong ω + T.held ω + T.passes * S +
      ∑ τ ∈ Ico (T.tick ω 0) (T.tick ω T.passes), T.entered ω τ := by
    have := sum_le_sum fun p (hp : p ∈ range T.passes) => pass_le ω (hS ω) p (hk p (mem_range.1 hp))
    rw [sum_add_distrib, sum_add_distrib, sum_add_distrib, sum_const, card_range, smul_eq_mul,
      sum_passes _ _ hm] at this
    exact this
  have hin : ∑ τ ∈ Ico (T.tick ω 0) (T.tick ω T.passes), T.entered ω τ ≤
      over T.rebuilt ω (T.tick ω 0) (T.tick ω T.passes) + over T.disk ω (T.tick ω 0) (T.tick ω T.passes) := by
    have h := sum_le_sum fun τ (_ : τ ∈ Ico (T.tick ω 0) (T.tick ω T.passes)) => hr ω τ
    simp only [sum_add_distrib, hI ω, sum_const_zero, add_zero] at h
    exact h
  have hreb : (over T.rebuilt ω (T.tick ω 0) (T.tick ω T.passes) : ℝ) ≤
      s / E * ((T.tick ω T.passes - T.tick ω 0 : ℕ) : ℝ) + Br := hsq ω _ _ huv
  have hdisk : (over T.disk ω (T.tick ω 0) (T.tick ω T.passes) : ℝ) ≤
      β * ((T.tick ω T.passes - T.tick ω 0 : ℕ) : ℝ) + Bd := hd ω _ _ huv
  have hspan : ((T.tick ω T.passes - T.tick ω 0 : ℕ) : ℝ) ≤ (e : ℝ) - s₀ := by
    have h0' : (s₀ : ℝ) ≤ T.tick ω 0 := by exact_mod_cast h0
    have hP' : (T.tick ω T.passes : ℝ) ≤ e := by exact_mod_cast hP
    rw [Nat.cast_sub huv]
    linarith
  have hX : 0 ≤ s / E + β := add_nonneg (div_nonneg hs hE.le) hβ
  have hprod : (s / E + β) * ((T.tick ω T.passes - T.tick ω 0 : ℕ) : ℝ) ≤ (s / E + β) * ((e : ℝ) - s₀) :=
    mul_le_mul_of_nonneg_left hspan hX
  have hexp : (s / E + β) * ((T.tick ω T.passes - T.tick ω 0 : ℕ) : ℝ) =
      s / E * ((T.tick ω T.passes - T.tick ω 0 : ℕ) : ℝ) + β * ((T.tick ω T.passes - T.tick ω 0 : ℕ) : ℝ) :=
    add_mul _ _ _
  have hsumR : (T.planned ω : ℝ) ≤ T.wrong ω + T.held ω + T.passes * S +
      ((∑ τ ∈ Ico (T.tick ω 0) (T.tick ω T.passes), T.entered ω τ : ℕ) : ℝ) := by
    exact_mod_cast hsum
  have hinR : ((∑ τ ∈ Ico (T.tick ω 0) (T.tick ω T.passes), T.entered ω τ : ℕ) : ℝ) ≤
      over T.rebuilt ω (T.tick ω 0) (T.tick ω T.passes) + over T.disk ω (T.tick ω 0) (T.tick ω T.passes) := by
    exact_mod_cast hin
  linarith

/-- **The snapshot held, under Identity** (`Accounting.Guarantees.Poub.Held`). -/
theorem Held : Accounting.Guarantees.Poub.Held := by
  intro Ω _ T S s₀ e s E β Br Bd W δs δc hE hs hβ hI hsq hd hS hr hc hk hw
  refine holdsMono (fun ω ha h => ?_) (holdsAnd hk hw)
  obtain ⟨hm, h0, hP⟩ := hc ω ha
  exact held_of ω hE hs hβ hI hsq hd hS hr hm h0 hP h.1 h.2

/-- Identity is the links' ceiling at rate 0 and burst 0. -/
theorem links_of_identity (hI : Identity T) : Links T 0 0 := fun ω u v _ => by
  have h0 : over T.link ω u v = 0 := Finset.sum_eq_zero fun t _ => hI ω t
  rw [h0, Nat.cast_zero, zero_mul, add_zero]

/-- **The time-held line under Identity is the link-charged one at links of rate 0 and burst 0**: `HeldOverTime`
follows from `HeldOverTimeLinks`. -/
theorem heldOverTime_of_links (h : Accounting.Guarantees.Poub.HeldOverTimeLinks) :
    Accounting.Guarantees.Poub.HeldOverTime := by
  intro Ω _ T K S s E β Br Bd W δs δc hE hs hβ hW hI hsq hd hS hr hp hcov hk hw
  simpa only [add_zero] using h Ω T K S s E β 0 Br Bd 0 W δs δc hE hs hβ le_rfl hW (links_of_identity hI) hsq hd hS
    hr hp hcov hk hw

end Accounting.SecurityProofs.Poub
