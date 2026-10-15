import Proofs.Flock.VBridge.Family.AccRows
import Proofs.Flock.Recursive.Registrants

/-!
# `NoRegistrantCollision` where no two sessions clash

`NoRegistrantCollision`'s pairs, and `NoRegistrantCollisionS`'s (`AccRows.lean`, keyed by registration), are
`RegColl`'s (`Recursive/Registrants.lean`): two sessions' registered reads whose openings or hm96 leaves collide.
Where every accepted outer proof's history and registration has no `RegColl` (`RegClear`), both hold; the recursive
audit prices `RegColl` once, by `regFinder` (`prob_regColl_le`, `zk_sessions_recursive_read_off`), or counts its
finding a collision as a fork (`zk_sessions_recursive_fork_read_off`).

* `RegClear`: no `RegColl` at an accepted outer proof; `regClear`: it holds, since `Accepted` carries `¬ RegColl`;
* `noRegistrantCollision_of_clear`, `rowsShared_of_clear`: `NoRegistrantCollision`, and so `RowsShared` where no read
  is segmented (`ReadsUnsegmented`);
* `noRegistrantCollisionS_of_clear`, `siteRowsShared_of_clear`: `NoRegistrantCollisionS`, and so `SiteRowsShared`
  where no read is segmented.
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Refine FlockSoundness.Binding FlockSoundness.Registered Audit
open FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Recursion FlockSoundness.Model
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)

/-- **No two sessions clash at an accepted outer proof** (`RegColl`). -/
def RegClear : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ¬ RegColl (I := ℐ) F.zs R ω t R₂

/-- **No two sessions clash at an accepted outer proof**: `Accepted`'s last conjunct. -/
theorem regClear : F.RegClear dg N k Rw := fun _ _ _ _ _ _ ha => ha.2

/-- **`NoRegistrantCollision` where no two sessions clash**: its two collisions are `RegColl`'s two. -/
theorem noRegistrantCollision_of_clear (h : F.RegClear dg N k Rw) : F.NoRegistrantCollision dg N k Rw := by
  intro R ω t R₂ τ o ha s s' u u' q q' hr hr' _ _
  exact ⟨fun hc => h R ω t R₂ τ o ha ⟨s, s', .inl (u, q), .inl (u', q'), hr, hr', Or.inl hc⟩,
    fun hc => h R ω t R₂ τ o ha ⟨s, s', .inl (u, q), .inl (u', q'), hr, hr', Or.inr hc⟩⟩

/-- **`RowsShared` where no two sessions clash and no read is segmented.** -/
theorem rowsShared_of_clear (h : F.RegClear dg N k Rw) (hu : F.ReadsUnsegmented dg N k Rw) : F.RowsShared dg N k Rw :=
  F.rowsShared_of dg N k Rw (F.noRegistrantCollision_of_clear dg N k Rw h) hu

/-- **`NoRegistrantCollisionS` where no two sessions clash**: its two collisions are `RegColl`'s two. -/
theorem noRegistrantCollisionS_of_clear (h : F.RegClear dg N k Rw) : F.NoRegistrantCollisionS dg N k Rw := by
  intro R ω t R₂ τ o ha s s' u u' q q' hr hr' _ _
  exact ⟨fun hc => h R ω t R₂ τ o ha ⟨s, s', .inl (u, q), .inl (u', q'), hr, hr', Or.inl hc⟩,
    fun hc => h R ω t R₂ τ o ha ⟨s, s', .inl (u, q), .inl (u', q'), hr, hr', Or.inr hc⟩⟩

/-- **`SiteRowsShared` where no two sessions clash and no read is segmented.** -/
theorem siteRowsShared_of_clear (h : F.RegClear dg N k Rw) (hu : F.ReadsUnsegmented dg N k Rw) :
    F.SiteRowsShared dg N k Rw :=
  F.siteRowsShared_of_sharedRowsS dg N k Rw (F.sharedRowsS_of dg N k Rw (F.noRegistrantCollisionS_of_clear dg N k Rw h))
    hu

end VStarFamily

end FlockVBridge.Family
