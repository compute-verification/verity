module

public import Proofs.Flock.Claims.Witnesses.Encodings
public import Proofs.Flock.Claims.Deploy

/-!
# The registered encodings' statements say something

Each closed binding statement of the registered encodings (`ByPosBinds`, `CodeBinds`, `LinkBinds`, `RelinkR`) is a
guarantee that the lock's `claims.generic` names, held to two witnesses (the lock's `witnesses`): an honest one, whose
conclusion is its premises holding together at one instance, and a nontrivial one, at which its conclusion fails. Each
is at the hash SHA-512 (`FlockDeploy.sha512`) and the toys of `Witnesses.Encodings`.

* Honest: the repetition code `repCode` meets the code's premises (`codeBinds_honest`); its scheme and the record
  `trueCm` bind (`byPosBinds_honest`), one pair links them at position `0` (`linkBinds_honest`), and the toy trace is
  in the sampling relation under their sum (`relinkR_honest`).
* Nontrivial: the sum of two commitments whose openings all pass (`byPosBinds_nontrivial`), a code with no columns
  (`codeBinds_nontrivial`), openings that make no call (`linkBinds_nontrivial`), and a record that rejects every
  opening (`relinkR_nontrivial`).
-/

@[expose] public section

namespace Claims.Encodings

open Claims.Sampling (RelS LogColl)
open Claims.Witness.Sampling (accCm rejCm toyStmt toyPart toyWit)
open Claims.Witness.SamplingGeneric (trueCm trueCm_binds)
open Claims.Witness.Encodings

/-- **`ByPosBinds`' premises hold together**: the repetition code's scheme and the record `trueCm` bind. -/
theorem byPosBinds_honest : repCode.scheme.Binds FlockDeploy.sha512 ∧ trueCm.Binds FlockDeploy.sha512 :=
  ⟨repCode_binds _, trueCm_binds (Coin := Bool) _⟩

/-- **`ByPosBinds`' conclusion fails** at the sum of two commitments whose openings all pass. -/
theorem byPosBinds_nontrivial : ¬ (byPos (fun _ _ => true) accCm accCm).Binds FlockDeploy.sha512 :=
  byPos_fails _

/-- **`CodeBinds`' premises hold together** at the repetition code: distance one, radius zero, a binding column
check. -/
theorem codeBinds_honest :
    repCode.Distance (fun _ => 1) ∧ (∀ roots, 2 * (repCode.len roots - repCode.agree roots) < 1) ∧
      repCode.ColBinds FlockDeploy.sha512 :=
  repCode_meets _

/-- **`CodeBinds`' conclusion fails** at a code with no columns. -/
theorem codeBinds_nontrivial : ¬ badCode.scheme.Binds FlockDeploy.sha512 :=
  badCode_fails _

/-- **`LinkBinds`' premises hold together**: the record `trueCm` and the repetition code's scheme bind, and one pair
links them at position `0`. -/
theorem linkBinds_honest :
    trueCm.Binds FlockDeploy.sha512 ∧ repCode.scheme.Binds FlockDeploy.sha512 ∧
      linkedAt FlockDeploy.sha512 trueCm repCode.scheme [] [pair₀] 0 ≠ none :=
  ⟨trueCm_binds (Coin := Bool) _, repCode_binds _, by rw [linkedAt₀]; exact Option.some_ne_none _⟩

/-- **`LinkBinds`' conclusion fails** at openings that make no call. -/
theorem linkBinds_nontrivial :
    ¬ LogColl (((accCm.opens [] ⟨0, true, []⟩).eval FlockDeploy.sha512).2 ++
      ((accCm.opens [] ⟨0, false, []⟩).eval FlockDeploy.sha512).2 ++ linkLog FlockDeploy.sha512 accCm accCm [] []) :=
  linkColl_fails _

/-- **`RelinkR`'s premises hold together** at the repetition code, the record `trueCm`, the pair `pair₀` and the toy
trace, in the sampling relation under their sum at `δ = 0`. -/
theorem relinkR_honest :
    repCode.scheme.Binds FlockDeploy.sha512 ∧ trueCm.Binds FlockDeploy.sha512 ∧
      Linked FlockDeploy.sha512 trueCm repCode.scheme toyStmt.roots (fun g => enc₀ toyStmt.roots g = true) [pair₀] ∧
      RelS FlockDeploy.sha512 (byPos enc₀ repCode.scheme trueCm) 0 toyStmt toyPart toyWit :=
  relink_meets _

/-- **`RelinkR`'s conclusion fails** where the record rejects every opening: the relinked toy trace is outside the
sampling relation at `δ = 0`, and no call is made. -/
theorem relinkR_nontrivial :
    ¬ (RelS FlockDeploy.sha512 rejCm 0 toyStmt toyPart (relink (fun _ _ => false) toyStmt.roots toyWit []) ∨
      LogColl ((toyWit.vals.flatMap fun o =>
          (((byPos (fun _ _ => false) rejCm rejCm).opens toyStmt.roots o).eval FlockDeploy.sha512).2) ++
        linkLog FlockDeploy.sha512 rejCm rejCm toyStmt.roots [])) :=
  relink_fails _

end Claims.Encodings
