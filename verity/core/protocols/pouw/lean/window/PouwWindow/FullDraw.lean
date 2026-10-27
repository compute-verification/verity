import Flock.Draw
import Flock.Norm

/-!
# A drawn H1 leaf's sessions draw every unit

A drawn leaf's 33 statements are each proved and checked in full: the verifier's draw policy for a leaf statement's
session is `subset` of every unit of its partition, a non-empty population. flock-verify takes the record's draw as
it is (any law it decodes, `Flock.Draw.UnitDraw.ofJson`), so the leaf verifier holds each session record to the
policy itself (`checkFull`, which `pouw-leaf-own` runs on every session of a statement):

* `recordDraw`: the record's `unit_draw`, rebuilt (`Flock.norm`), as flock-verify reads it (`Main.recordDraw`);
* `checkFull`: that draw decodes, its law is `subset k` with `k` its population, `0 < k`, and its units are every unit
  `[0, k)`; `checkFull_ok` states what its `.ok` gives.
-/

namespace PouwWindow.Lanes

open Lean (Json)

/-- A record's `unit_draw`, when it is a draw (not absent, not null), rebuilt (`Flock.norm`): `Main.recordDraw`. -/
def recordDraw (record : String) : Option Json :=
  match Json.parse record with
  | .ok j => match j.getObjVal? "unit_draw" with
    | .ok Json.null => none
    | .ok d => some (Flock.norm d)
    | .error _ => none
  | .error _ => none

/-- **The session's record draws every unit**: its draw decodes (`UnitDraw.ofJson`), is `subset k` of a population of
`k` units, `0 < k`, and its units are `[0, k)`; else the reason, named. -/
def checkFull (record : String) : Except String Unit :=
  match recordDraw record with
  | none => .error "the session's record has no draw"
  | some d => match Flock.Draw.UnitDraw.ofJson d with
    | .error e => .error s!"the session's draw: {e}"
    | .ok D => match D.law with
      | .subset k =>
        if k = D.population ∧ 0 < k ∧ D.units = Array.range k then .ok ()
        else .error s!"the session's draw is subset:{k} of {D.units.size} units over {D.population}, not every unit"
      | _ => .error "the session's draw's law is not subset"

theorem checkFull_ok {record : String} (h : checkFull record = .ok ()) :
    ∃ d D k, recordDraw record = some d ∧ Flock.Draw.UnitDraw.ofJson d = .ok D ∧ D.law = .subset k ∧
      k = D.population ∧ 0 < k ∧ D.units = Array.range k := by
  unfold checkFull at h
  split at h
  · cases h
  · next d hd =>
    split at h
    · cases h
    · next D hD =>
      split at h
      · next k hk =>
        split at h
        · next hc => exact ⟨d, D, k, hd, hD, hk, hc.1, hc.2.1, hc.2.2⟩
        · cases h
      · cases h

end PouwWindow.Lanes
