import FlockSoundness.Discharge.ZkReg.Defs

/-!
# The registered reads of a hidden-output `verify --zk --registered F` session

`AcceptsZKR` reads the statements `Stmt.setupTables` sets up, whichever setup the tags choose, so it is the event at
hidden outputs too (`Stmt.setupHidden`, the default `--statement`), as `AcceptsZK` is `zk_session_soundH`'s. What
differs is the drawn statement: setupHidden's is `HmOut.drawn c pub d`, `HmRow.drawn` at the circuit without its output
words. `regPathsH` is `regPaths` at that drawn statement. `OwnReadsH` is `OwnReads`: the check reads each instance's
unit from the population's header (`Registered.checkSession`, `Registered.tableUnits`), not from the drawn statement's.
-/

namespace FlockSoundness.Discharge.ZkReg

open Flock
open FlockSoundness.Discharge.Exec (recordDraw)

variable {n : ℕ} {Gate : Type}

/-- **The registered reads are the verifier's own at the record's draw, under hidden outputs**: each registered read
`p` is the row `(u, q)` of a drawn instance `u`, its port is one of `own` and is the circuit's port `q` by name, and its
position is the one `own` gives for `u`'s unit, as the population `pub`'s header names the units
(`Registered.unitIndices`). -/
def OwnReadsH (own : Array Registered.Port) (c : Circuit) (pub : Public) (t : Exec.Transcript)
    (isRead : (Fin n × Fin c.ports.size) ⊕ Gate → Prop) (port : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port)
    (readAt : (Fin n × Fin c.ports.size) ⊕ Gate → ℕ) : Prop :=
  ∀ p, isRead p → ∃ u q d D units, p = .inl (u, q) ∧ port p ∈ own ∧
    (List.range c.ports.size).find? (fun j => c.ports[j]!.name == (port p).name) = some q.val ∧
    recordDraw t.record = some d ∧ Draw.UnitDraw.ofJson d = .ok D ∧ u.val ∈ D.proved ∧
    Registered.unitIndices pub.header = .ok units ∧ (port p).reads[units.getD u.val 0]? = some (some (readAt p))

/-- **The paths of the registered reads, under hidden outputs**: `regPaths` at the drawn statement
`HmOut.drawn c pub d`. -/
def regPathsH (c : Circuit) (pub : Public) (t : Exec.Transcript)
    (port : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port) : (Fin n × Fin c.ports.size) ⊕ Gate → ByteArray
  | .inl (u, q) => Option.getD (do
      let d ← recordDraw t.record
      let D ← (Draw.UnitDraw.ofJson d).toOption
      let dp ← (HmOut.drawn c pub d).toOption
      let tb ← dp.tables
      let (_, paths) ← (Registered.entry dp (port (.inl (u, q)))).toOption
      return paths.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) ByteArray.empty)
      ByteArray.empty
  | .inr _ => ByteArray.empty

end FlockSoundness.Discharge.ZkReg
