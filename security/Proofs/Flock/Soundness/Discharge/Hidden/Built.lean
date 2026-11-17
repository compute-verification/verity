module

public import Flock.Desc
public import Proofs.Flock.Soundness.Discharge.Hidden.Facts
public import Proofs.Flock.Soundness.ExecCheck
public import Proofs.Flock.Soundness.ExecSetup
public import Proofs.Flock.Soundness.ParseStmt

/-! A parsed statement's unit is `Built` (`built_of_parseStmt`), so a statement reading `D.unroll`'s unit takes it from the parse rather than as a hypothesis. -/

@[expose] public section

namespace FlockSoundness.Discharge.Hidden

open Flock

theorem built_of_parseStmt {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {D : Desc}
    (h : Desc.parseStmt tags cf tables = .ok D) (ht : tags.typed = false) : Built D.unroll.unit := by
  unfold Desc.parseStmt at h
  split at h
  · split at h
    · next c hc =>
      cases h
      exact unrolled_built hc
    · cases h
  · unfold Desc.parse at h
    split at h
    · next c hc =>
      cases h
      have P := parse_facts (parse_untyped ht hc).1
      obtain ⟨ls, hls⟩ := P.check
      exact unit_built P (check_facts P.typed hls)
    · cases h

end FlockSoundness.Discharge.Hidden
