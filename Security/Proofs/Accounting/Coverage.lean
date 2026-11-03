import Definitions.Accounting.Guarantees.Coverage
import Proofs.Core.Game.Prob

/-! `Nci.SecurityProofs.CapabilityFloor : Nci.Guarantees.CapabilityFloor`: below `min Lc Tmax`, a computation in `𝒜`
has no covered structure (each costs at least `Lc`), so its success there is among coverage's uncovered successes below
`Tmax`. Its witnesses (`CapabilityFloorHonest`, `CapabilityFloorNontrivial`) are proved at the toy task
`Accounting.Slowdown.Toy`: a position holds one gate, so a run reaches at most one of the coin's two targets, and the
guess reaches one. -/

/-- **Covered success below the floor is rare** (`Nci.Guarantees.CapabilityFloor`). -/
theorem Nci.SecurityProofs.CapabilityFloor : Nci.Guarantees.CapabilityFloor := by
  intro In L m G 𝒜 Core cost Tmax Lc η hcov hfloor C pos
  refine le_trans (Verity.Game.prob_mono (fun p hp => ?_) _ _) (hcov C pos)
  obtain ⟨hA, hS, hc⟩ := hp
  exact ⟨hA, hS, lt_of_lt_of_le hc (min_le_right _ _),
    fun hCore => absurd (hfloor C p.1 hA hCore) (not_le.2 (lt_of_lt_of_le hc (min_le_left _ _)))⟩

namespace Accounting.Slowdown.Toy

open scoped ENNReal

open Classical in
/-- The toy task's probability: half the event at each coin. -/
theorem prob_eq (E : Unit → Set (Fin 1 → List Bool) → Prop) :
    task.prob E = (if E () {y | y 0 = [true]} then 2⁻¹ else 0) + if E () {y | y 0 = [false]} then 2⁻¹ else 0 := by
  simp [Task.prob, Task.game, Verity.Game.prob, task]

/-- A position holds one gate, so a run reaches at most one of the two targets. -/
theorem not_both (R : Actual.Run Unit) (pos : Fin 1 → Unit × ℕ × ℕ) :
    ¬ (Succeeds R pos {y | y 0 = [true]} ∧ Succeeds R pos {y | y 0 = [false]}) := by
  rintro ⟨⟨y, hy, hr⟩, ⟨y', hy', hr'⟩⟩
  obtain ⟨g, hg, hb⟩ := hr 0
  obtain ⟨g', hg', hb'⟩ := hr' 0
  obtain rfl : g = g' := R.A.cell_inj (hg.trans hg'.symm)
  have h : y 0 = y' 0 := hb.symm.trans hb'
  rw [show y 0 = [true] from hy, show y' 0 = [false] from hy'] at h
  simp at h

/-- The guess `1` reaches the target `[1]`. -/
theorem guess_succeeds : Succeeds (guess true) pos {y | y 0 = [true]} :=
  ⟨fun _ => [true], rfl,
    fun _ => ⟨⟨0, Nat.one_pos⟩, rfl, (by decide : Actual.bits 1 (BitVec.ofBool true).toNat = [true])⟩⟩

/-- An event that the coin's two targets never both satisfy has probability at most `1/2`. -/
theorem prob_le_half (E : Unit → Set (Fin 1 → List Bool) → Prop)
    (h : ¬ (E () {y | y 0 = [true]} ∧ E () {y | y 0 = [false]})) : task.prob E ≤ 2⁻¹ := by
  rw [prob_eq]
  by_cases ht : E () {y | y 0 = [true]}
  · have hf : ¬ E () {y | y 0 = [false]} := fun hf => h ⟨ht, hf⟩
    simp [ht, hf]
  · by_cases hf : E () {y | y 0 = [false]} <;> simp [ht, hf]

/-- An event satisfied at the target `[1]` has positive probability. -/
theorem prob_pos (E : Unit → Set (Fin 1 → List Bool) → Prop) (h : E () {y | y 0 = [true]}) :
    ¬ task.prob E ≤ 0 := by
  rw [prob_eq]
  simp [h]

end Accounting.Slowdown.Toy

/-- **`CapabilityFloor`'s premises hold together** (`Nci.Guarantees.CapabilityFloorHonest`). -/
theorem Nci.SecurityProofs.CapabilityFloorHonest : Nci.Guarantees.CapabilityFloorHonest := by
  refine ⟨fun C pos => Accounting.Slowdown.Toy.prob_le_half _ fun h =>
      Accounting.Slowdown.Toy.not_both (C ()) pos ⟨h.1.2.1, h.2.2.1⟩,
    fun _ _ _ h => h, fun h => Accounting.Slowdown.Toy.prob_pos _ ?_ (h Accounting.Slowdown.Toy.comp
      Accounting.Slowdown.Toy.pos)⟩
  exact ⟨Set.mem_univ _, Accounting.Slowdown.Toy.guess_succeeds, (by decide : 1 < 10), (by decide : ¬ 2 ≤ 1)⟩

/-- **`CapabilityFloor`'s conclusion can fail** (`Nci.Guarantees.CapabilityFloorNontrivial`). -/
theorem Nci.SecurityProofs.CapabilityFloorNontrivial : Nci.Guarantees.CapabilityFloorNontrivial :=
  Accounting.Slowdown.Toy.prob_pos _ ⟨Set.mem_univ _, Accounting.Slowdown.Toy.guess_succeeds, by decide⟩
