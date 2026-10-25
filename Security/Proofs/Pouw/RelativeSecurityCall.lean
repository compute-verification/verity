import Proofs.Pouw.RelativeSecurityGuarantees
import Models.Pouw.Guarantees.RelativeSecurityCall

/-!
# Relative security's floors at J10-R's call scope

RT-ALGO-17. At a call's shape Theorem K's design is J10-R's call coins, and unit `(I, J)`'s target read at its slice
of them is the call's target at its cell-block (`dPrefix_unitCell`): its columns' `U` planes and its rows' dither bits
are the call's, and its constants the call's at those columns and rows. Every call cell-block is exactly one unit's
(`unitCell_surjective`, `unitCell_injective`), so the call-scope premises are the floors of record's at the call's
shape, neither stronger nor weaker (`kScopeOneSidedCall_iff`, `kScopeOverlapOneSidedCall_iff`; the charges are theirs
verbatim), and each call floor is the floor of record at the call's shape, its count
`L · (nu uc)(mu ur) B = L · mu nu · ur uc B`.

The per-prover forms follow from the unit ones the same way (`stagedFloorKOneSidedCallPerProver_of_unit`,
`overlapFloorKOneSidedCallPerProver_of_unit`); neither is proved yet.
-/

namespace Pouw.SecurityProofs.RelativeSecurity

open Pouw.RelativeSecurity Pouw.Assumptions

section Slices

variable {Ar : Type} [CommRing Ar] {mu nu ur uc L np B : ℕ}

/-- Every cell-block of a call is one unit's. -/
theorem unitCell_surjective : Function.Surjective (unitCell : UCells mu nu ur uc B → CallCells mu nu ur uc B) :=
  fun c => ⟨(((finProdFinEquiv.symm c.2.1).1, (finProdFinEquiv.symm c.1).1), (finProdFinEquiv.symm c.1).2,
    (finProdFinEquiv.symm c.2.1).2, c.2.2), by simp only [unitCell, Prod.mk.eta, Equiv.apply_symm_apply]⟩

/-- No cell-block of a call is two units'. -/
theorem unitCell_injective : Function.Injective (unitCell : UCells mu nu ur uc B → CallCells mu nu ur uc B) := by
  rintro ⟨⟨I, J⟩, i, l, t⟩ ⟨⟨I', J'⟩, i', l', t'⟩ h
  simp only [unitCell, Prod.mk.injEq, EmbeddingLike.apply_eq_iff_eq] at h
  obtain ⟨⟨rfl, rfl⟩, ⟨rfl, rfl⟩, rfl⟩ := h
  rfl

/-- **A unit's target at its J10-R slice is the call's at its cell-block**: the call's prefix word at call column
`uc J + i` and call row `ur I + l` reads that column's `U` planes and that row's dither bits, which are unit `(I, J)`'s
column `i`'s and row `l`'s, on the call's constants there. -/
theorem dPrefix_unitCell (pc : Fin (nu * uc) → Fin B → Fin L → Ar) (vq : Fin B → Fin L → Fin (mu * ur) → Ar)
    (w : Fin np → Ar) (u : UCells mu nu ur uc B) (t : ℕ) (z : CallBits mu nu ur uc L np B → Ar) :
    dPrefix pc vq w (unitCell u).1 (unitCell u).2.1 t z =
      dPrefix (unitPc pc u.1.2) (unitVq vq u.1.1) w u.2.1 u.2.2.1 t fun v => z (unitBit u.1.1 u.1.2 v) := by
  simp only [dPrefix, dIncr, dP_eval, dX_eval]
  rfl

end Slices

section Staged

variable {Mc : Machine} {S : Service} {R St mu nu ur uc L np B : ℕ} {Ar : Type} [CommRing Ar]

/-- **Theorem K's one-sided scope at a call is the floor of record's at the call's shape.** -/
theorem kScopeOneSided_of_call {run : CallRunMap Mc S R St mu nu ur uc L np B Ar} {wt : Fin np → Ar}
    {φ : Ar →+* ZMod 2} {p₀ : Fin np} (h : KScopeOneSidedCall Mc S run wt φ p₀) : KScopeOneSided Mc S run wt φ p₀ := by
  intro Ω' _ M hM r x
  obtain ⟨h1, h2, h3⟩ := h Ω' M hM r x
  refine ⟨h1, fun w s c => ?_, fun w s c c' i hne => ?_⟩
  · obtain ⟨u, rfl⟩ := unitCell_surjective c
    refine (h2 w s u).trans (funext fun z => ?_)
    rw [dPrefix_unitCell]
    rfl
  · obtain ⟨u, rfl⟩ := unitCell_surjective c
    obtain ⟨u', rfl⟩ := unitCell_surjective c'
    exact h3 w s u u' i fun e => hne (congrArg unitCell e)

/-- `KScopeOneSidedCall` is `KScopeOneSided` at the call's shape, read by unit: neither stronger nor weaker. -/
theorem kScopeOneSidedCall_iff {run : CallRunMap Mc S R St mu nu ur uc L np B Ar} {wt : Fin np → Ar}
    {φ : Ar →+* ZMod 2} {p₀ : Fin np} : KScopeOneSidedCall Mc S run wt φ p₀ ↔ KScopeOneSided Mc S run wt φ p₀ := by
  refine ⟨kScopeOneSided_of_call, fun h Ω' _ M hM r x => ?_⟩
  obtain ⟨h1, h2, h3⟩ := h Ω' M hM r x
  refine ⟨h1, fun w s u => (h2 w s (unitCell u)).trans (funext fun z => ?_),
    fun w s u u' i hne => h3 w s (unitCell u) (unitCell u') i (unitCell_injective.ne hne)⟩
  rw [dPrefix_unitCell]
  rfl

end Staged

section Overlap

variable {Mc : Machine} {S : Service} {R St ur uc L np : ℕ} {mu nu B : Fin St → ℕ} {Ar : Type} [CommRing Ar]

/-- **Theorem K∥'s one-sided scope at the calls is the floor of record's at the calls' shapes.** -/
theorem kScopeOverlapOneSided_of_call {run : OCallRunMap Mc S R St mu nu ur uc L np B Ar} {wt : Fin np → Ar}
    {φ : Ar →+* ZMod 2} {p₀ : Fin np} (h : KScopeOverlapOneSidedCall Mc S run wt φ p₀) :
    KScopeOverlapOneSided Mc S run wt φ p₀ := by
  intro Ω' _ M hM r x
  obtain ⟨h1, h2, h3⟩ := h Ω' M hM r x
  refine ⟨h1, fun w c => ?_, fun w W c c' i hne => ?_⟩
  · obtain ⟨t, c⟩ := c
    obtain ⟨u, rfl⟩ := unitCell_surjective c
    refine (h2 w ⟨t, u⟩).trans (funext fun z => ?_)
    beta_reduce
    rw [dPrefix_unitCell (z := fun v => z ⟨t, v⟩)]
    rfl
  · obtain ⟨t, c⟩ := c
    obtain ⟨t', c'⟩ := c'
    obtain ⟨u, rfl⟩ := unitCell_surjective c
    obtain ⟨u', rfl⟩ := unitCell_surjective c'
    exact h3 w W ⟨t, u⟩ ⟨t', u'⟩ i fun e => hne (congrArg oUnitCell e)

/-- No cell-block of a round's calls is two units'. -/
theorem oUnitCell_injective :
    Function.Injective (oUnitCell : OUCells St mu nu ur uc B → OCells St (fun t => nu t * uc) (fun t => mu t * ur) B) := by
  rintro ⟨t, u⟩ ⟨t', u'⟩ h
  simp only [oUnitCell, Sigma.mk.injEq] at h
  obtain ⟨rfl, h⟩ := h
  rw [unitCell_injective (eq_of_heq h)]

/-- `KScopeOverlapOneSidedCall` is `KScopeOverlapOneSided` at the calls' shapes, read by unit: neither stronger nor
weaker. -/
theorem kScopeOverlapOneSidedCall_iff {run : OCallRunMap Mc S R St mu nu ur uc L np B Ar} {wt : Fin np → Ar}
    {φ : Ar →+* ZMod 2} {p₀ : Fin np} :
    KScopeOverlapOneSidedCall Mc S run wt φ p₀ ↔ KScopeOverlapOneSided Mc S run wt φ p₀ := by
  refine ⟨kScopeOverlapOneSided_of_call, fun h Ω' _ M hM r x => ?_⟩
  obtain ⟨h1, h2, h3⟩ := h Ω' M hM r x
  refine ⟨h1, fun w u => (h2 w (oUnitCell u)).trans (funext fun z => ?_),
    fun w W u u' i hne => h3 w W (oUnitCell u) (oUnitCell u') i (oUnitCell_injective.ne hne)⟩
  simp only [oUnitCell]
  rw [dPrefix_unitCell (z := fun v => z ⟨u.1, v⟩)]
  rfl

end Overlap

/-- **The staged floor at J10-R's call scope**: the floor of record at the call's shape. -/
theorem StagedFloorKOneSidedCall : Pouw.Guarantees.RelativeSecurity.StagedFloorKOneSidedCall := by
  intro Mc S R St mu nu ur uc L B e run δb δz κ Vmax hSt hκ hC hδb hδz hS hK hP
  have hN : mu * nu * (ur * uc * B) = nu * uc * (mu * ur) * B := by ring
  rw [hN] at hC ⊢
  exact StagedFloorKOneSided Mc S R St (nu * uc) L B (mu * ur) e run δb δz κ Vmax hSt hκ hC hδb hδz hS
    (kScopeOneSided_of_call hK) hP

/-- **The overlapped floor at J10-R's call scope**: the floor of record at the calls' shapes. -/
theorem OverlapFloorKOneSidedCall : Pouw.Guarantees.RelativeSecurity.OverlapFloorKOneSidedCall := by
  intro Mc S R St ur uc L mu nu B e run δb δz κ Vmax hκ hC hδb hδz hS hK hP
  have hN : ∑ t, mu t * nu t * (ur * uc * B t) = ∑ t, nu t * uc * (mu t * ur) * B t :=
    Finset.sum_congr rfl fun t _ => by ring
  rw [hN] at hC ⊢
  exact OverlapFloorKOneSided Mc S R St L (fun t => nu t * uc) B (fun t => mu t * ur) e run δb δz κ Vmax hκ hC hδb
    hδz hS (kScopeOverlapOneSided_of_call hK) hP

/-- `StagedFloorKOneSidedCallPerProver` follows from `StagedFloorKOneSidedPerProver`, at the call's shape. -/
theorem stagedFloorKOneSidedCallPerProver_of_unit (h : Pouw.Guarantees.RelativeSecurity.StagedFloorKOneSidedPerProver) :
    Pouw.Guarantees.RelativeSecurity.StagedFloorKOneSidedCallPerProver := by
  intro Mc S R St mu nu ur uc L B e run sha fb fz cb db cz dz δb βb δz βz κ Vmax hSt hκ hC hδb hβb hδz hβz hS hK hP
  have hN : mu * nu * (ur * uc * B) = nu * uc * (mu * ur) * B := by ring
  rw [hN] at hC ⊢
  exact h Mc S R St (nu * uc) L B (mu * ur) e run sha fb fz cb db cz dz δb βb δz βz κ Vmax hSt hκ hC hδb hβb hδz hβz
    hS (kScopeOneSided_of_call hK) hP

/-- `OverlapFloorKOneSidedCallPerProver` follows from `OverlapFloorKOneSidedPerProver`, at the calls' shapes. -/
theorem overlapFloorKOneSidedCallPerProver_of_unit
    (h : Pouw.Guarantees.RelativeSecurity.OverlapFloorKOneSidedPerProver) :
    Pouw.Guarantees.RelativeSecurity.OverlapFloorKOneSidedCallPerProver := by
  intro Mc S R St ur uc L mu nu B e run sha fb fz cb db cz dz δb βb δz βz κ Vmax hκ hC hδb hβb hδz hβz hS hK hP
  have hN : ∑ t, mu t * nu t * (ur * uc * B t) = ∑ t, nu t * uc * (mu t * ur) * B t :=
    Finset.sum_congr rfl fun t _ => by ring
  rw [hN] at hC ⊢
  exact h Mc S R St L (fun t => nu t * uc) B (fun t => mu t * ur) e run sha fb fz cb db cz dz δb βb δz βz κ Vmax hκ hC
    hδb hβb hδz hβz hS (kScopeOverlapOneSided_of_call hK) hP

end Pouw.SecurityProofs.RelativeSecurity
