module

public import Definitions.Slowdown.Guarantees.Coverage
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Inv
import Proofs.Core.Game.Prob

/-! `CapabilityFloor`, proved, with its witnesses at the toy task `ResourceAccounting.Slowdown.Toy`. -/

@[expose] public section

/-- **Covered success below the floor is rare** (`Slowdown.Guarantees.CapabilityFloor`). -/
theorem Slowdown.SecurityProofs.CapabilityFloor : Slowdown.Guarantees.CapabilityFloor := by
  intro In L m G 𝒜 Core cost Tmax Lc η hcov hfloor C pos
  refine le_trans (Verity.Game.prob_mono (fun p hp => ?_) _ _) (hcov C pos)
  obtain ⟨hA, hS, hc⟩ := hp
  exact ⟨hA, hS, lt_of_lt_of_le hc (min_le_right _ _),
    fun hCore => absurd (hfloor C p.1 hA hCore) (not_le.2 (lt_of_lt_of_le hc (min_le_left _ _)))⟩

namespace ResourceAccounting.Slowdown.Toy

open scoped ENNReal

open Classical in
theorem prob_eq (E : Unit → Set (Fin 1 → List Bool) → Prop) :
    task.prob E = (if E () {y | y 0 = [true]} then 2⁻¹ else 0) + if E () {y | y 0 = [false]} then 2⁻¹ else 0 := by
  simp [Task.prob, Task.game, Verity.Game.prob, task]

theorem not_both (R : Actual.Run Unit) (pos : Fin 1 → Unit × ℕ × ℕ) :
    ¬ (Succeeds R pos {y | y 0 = [true]} ∧ Succeeds R pos {y | y 0 = [false]}) := by
  rintro ⟨⟨y, hy, hr⟩, ⟨y', hy', hr'⟩⟩
  obtain ⟨g, hg, hb⟩ := hr 0
  obtain ⟨g', hg', hb'⟩ := hr' 0
  obtain rfl : g = g' := R.A.cell_inj (hg.trans hg'.symm)
  have h : y 0 = y' 0 := hb.symm.trans hb'
  rw [show y 0 = [true] from hy, show y' 0 = [false] from hy'] at h
  simp at h

theorem guess_succeeds : Succeeds (guess true) pos {y | y 0 = [true]} :=
  ⟨fun _ => [true], rfl,
    fun _ => ⟨⟨0, Nat.one_pos⟩, rfl, (by decide : Actual.bits 1 (BitVec.ofBool true).toNat = [true])⟩⟩

theorem prob_le_half (E : Unit → Set (Fin 1 → List Bool) → Prop)
    (h : ¬ (E () {y | y 0 = [true]} ∧ E () {y | y 0 = [false]})) : task.prob E ≤ 2⁻¹ := by
  rw [prob_eq]
  by_cases ht : E () {y | y 0 = [true]}
  · have hf : ¬ E () {y | y 0 = [false]} := fun hf => h ⟨ht, hf⟩
    simp [ht, hf]
  · by_cases hf : E () {y | y 0 = [false]} <;> simp [ht, hf]

theorem prob_pos (E : Unit → Set (Fin 1 → List Bool) → Prop) (h : E () {y | y 0 = [true]}) :
    ¬ task.prob E ≤ 0 := by
  rw [prob_eq]
  simp [h]

end ResourceAccounting.Slowdown.Toy

/-- **`CapabilityFloor`'s premises hold together** (`Slowdown.Guarantees.CapabilityFloorHonest`). -/
theorem Slowdown.SecurityProofs.CapabilityFloorHonest : Slowdown.Guarantees.CapabilityFloorHonest := by
  refine ⟨fun C pos => ResourceAccounting.Slowdown.Toy.prob_le_half _ fun h =>
      ResourceAccounting.Slowdown.Toy.not_both (C ()) pos ⟨h.1.2.1, h.2.2.1⟩,
    fun _ _ _ h => h, fun h => ResourceAccounting.Slowdown.Toy.prob_pos _ ?_ (h ResourceAccounting.Slowdown.Toy.comp
      ResourceAccounting.Slowdown.Toy.pos)⟩
  exact ⟨Set.mem_univ _, ResourceAccounting.Slowdown.Toy.guess_succeeds, (by decide : 1 < 10), (by decide : ¬ 2 ≤ 1)⟩

/-- **`CapabilityFloor`'s conclusion can fail** (`Slowdown.Guarantees.CapabilityFloorNontrivial`). -/
theorem Slowdown.SecurityProofs.CapabilityFloorNontrivial : Slowdown.Guarantees.CapabilityFloorNontrivial :=
  ResourceAccounting.Slowdown.Toy.prob_pos _ ⟨Set.mem_univ _, ResourceAccounting.Slowdown.Toy.guess_succeeds, by decide⟩
