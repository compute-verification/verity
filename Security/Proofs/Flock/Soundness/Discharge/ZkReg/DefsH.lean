import Proofs.Flock.Soundness.Discharge.ZkReg.Defs

/-!
# The registered reads of a hidden-output `verify --zk --registered F` session

`AcceptsZKR` reads the statements `Stmt.setupTables` sets up, whichever setup the tags choose, so it is the event at
hidden outputs too (`Stmt.setupHidden`, the default `--statement`), as `AcceptsZK` is `zk_session_soundH`'s. What
differs is the drawn statement: setupHidden's is `HmOut.drawn c pub d`, `HmRow.drawn` at the circuit without its output
words. `OwnReadsH` and `regPathsH` are `OwnReads` and `regPaths` at that drawn statement.
-/

namespace FlockSoundness.Discharge.ZkReg

open Flock
open FlockSoundness.Discharge.Exec (recordDraw)

variable {n : ℕ} {Gate : Type}

/-- **The registered reads are the verifier's own at the record's draw, under hidden outputs**: `OwnReads` at the drawn
statement `HmOut.drawn c pub d`. -/
def OwnReadsH (own : Array Registered.Port) (c : Circuit) (pub : Public) (t : Exec.Transcript)
    (isRead : (Fin n × Fin c.ports.size) ⊕ Gate → Prop) (port : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port)
    (readAt : (Fin n × Fin c.ports.size) ⊕ Gate → ℕ) : Prop :=
  ∀ p, isRead p → ∃ u q d D dp idx, p = .inl (u, q) ∧ port p ∈ own ∧
    (List.range c.ports.size).find? (fun j => c.ports[j]!.name == (port p).name) = some q.val ∧
    recordDraw t.record = some d ∧ Draw.UnitDraw.ofJson d = .ok D ∧ HmOut.drawn c pub d = .ok dp ∧
    Registered.unitIndices dp = .ok idx ∧ u.val ∈ D.proved ∧
    (port p).reads[idx.getD (D.proved.idxOf u.val) 0]? = some (some (readAt p))

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
