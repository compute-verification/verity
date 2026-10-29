import Definitions.Accounting.Guarantees.Coverage
import Proofs.Core.Game.Prob

/-! `Nci.SecurityProofs.CapabilityFloor : Nci.Guarantees.CapabilityFloor`: below `min Lc Tmax`, a computation in `𝒜`
has no covered structure (each costs at least `Lc`), so its success there is among coverage's uncovered successes below
`Tmax`. Its witnesses (`CapabilityFloorHonest`, `CapabilityFloorNontrivial`) are proved at the toy task
`Accounting.Slowdown.Toy`. -/

/-- **Covered success below the floor is rare** (`Nci.Guarantees.CapabilityFloor`). -/
theorem Nci.SecurityProofs.CapabilityFloor : Nci.Guarantees.CapabilityFloor := by
  intro In L m G 𝒜 Core cost Tmax Lc η hcov hfloor C pos
  refine le_trans (Verity.Game.prob_mono (fun p hp => ?_) _ _) (hcov C pos)
  obtain ⟨hA, hS, hc⟩ := hp
  exact ⟨hA, hS, lt_of_lt_of_le hc (min_le_right _ _),
    fun hCore => absurd (hfloor C p.1 hA hCore) (not_le.2 (lt_of_lt_of_le hc (min_le_left _ _)))⟩

/-- **`CapabilityFloor`'s premises hold together** (`Nci.Guarantees.CapabilityFloorHonest`). -/
theorem Nci.SecurityProofs.CapabilityFloorHonest : Nci.Guarantees.CapabilityFloorHonest := by
  refine ⟨fun C pos => ?_, fun _ _ _ _ => le_rfl⟩
  simp [Accounting.Slowdown.Task.prob, Accounting.Slowdown.Task.game, Verity.Game.prob,
    Accounting.Slowdown.Toy.covered]

/-- **`CapabilityFloor`'s conclusion can fail** (`Nci.Guarantees.CapabilityFloorNontrivial`). -/
theorem Nci.SecurityProofs.CapabilityFloorNontrivial : Nci.Guarantees.CapabilityFloorNontrivial := by
  unfold Nci.Guarantees.CapabilityFloorNontrivial
  have hs : Accounting.Slowdown.Succeeds Accounting.Slowdown.Toy.run Accounting.Slowdown.Toy.pos Set.univ :=
    ⟨fun i => i.elim0, Set.mem_univ _, fun i => i.elim0⟩
  simp [Accounting.Slowdown.Task.prob, Accounting.Slowdown.Task.game, Verity.Game.prob,
    Accounting.Slowdown.Toy.task, Accounting.Slowdown.Toy.comp, Accounting.Slowdown.Toy.free, hs]
