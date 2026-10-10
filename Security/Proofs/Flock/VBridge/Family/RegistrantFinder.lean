import Proofs.Flock.VBridge.Family.Registrants
import Proofs.Flock.Recursive.Registrants

/-!
# `NoRegistrantCollision` where no two sessions clash

`NoRegistrantCollision`'s pairs are `RegColl`'s (`Recursive/Registrants.lean`): two sessions' registered reads at one
root and position whose openings or hm96 leaves collide. Where every accepted outer proof's history and registration
has no `RegColl`, it holds; the recursive audit prices `RegColl` once, by `regFinder` (`prob_regColl_le`), in
`zk_sessions_recursive_read_off`.

* `noRegistrantCollision_of_clear`, `rowsShared_of_clear`: `NoRegistrantCollision`, and so `RowsShared`, at accepted
  outer proofs without `RegColl`.
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

/-- **`NoRegistrantCollision` where no two sessions clash**: its two collisions are `RegColl`'s two. -/
theorem noRegistrantCollision_of_clear
    (h : ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ¬ RegColl (I := ℐ) F.zs R ω t R₂) :
    F.NoRegistrantCollision dg N k Rw := by
  intro R ω t R₂ τ o ha s s' u u' q q' hr hr' hport hat
  exact ⟨fun hc => h R ω t R₂ τ o ha ⟨s, s', .inl (u, q), .inl (u', q'), hr, hr', hport, hat, Or.inl hc⟩,
    fun hc => h R ω t R₂ τ o ha ⟨s, s', .inl (u, q), .inl (u', q'), hr, hr', hport, hat, Or.inr hc⟩⟩

/-- **`RowsShared` where no two sessions clash.** -/
theorem rowsShared_of_clear
    (h : ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ¬ RegColl (I := ℐ) F.zs R ω t R₂) :
    F.RowsShared dg N k Rw :=
  F.rowsShared_of dg N k Rw (F.noRegistrantCollision_of_clear dg N k Rw h)

end VStarFamily

end FlockVBridge.Family
