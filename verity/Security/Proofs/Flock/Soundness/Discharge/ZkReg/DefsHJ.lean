import Proofs.Flock.Soundness.Discharge.ZkReg.DefsH
import Proofs.Flock.Soundness.Discharge.Composed.DefsJ

/-!
# The registered reads of a hidden-output `verify --zk --registered F` session of `J` tables

`AcceptsZKR` reads every statement `Stmt.setupTables` sets up, so it is the event at `J` tables too: `checkRegistered`
runs on each table's statement. At `count > 1` table `j`'s statement is setupHidden's at its part of the record's draw,
`drawJ I d j` (`ZkHidden.setupOf_tableJH`), over the drawn statement `HmOut.drawn c pub (drawJ I d j)`, and a unit `u`
is drawn into table `partOf I d u` (`ZkHidden.tabZHJ`). `OwnReadsHJ` and `regPathsHJ` are `OwnReadsH` and `regPathsH`
with each read `(u, q)` at the table `u` is drawn into: the draw is that table's, `drawJ I d (partOf I d u)`, and `u`'s
rank is its rank in that table's draw. At one table they are `OwnReadsH` and `regPathsH` (`ownReadsHJ_one`,
`regPathsHJ_one`).
-/

namespace FlockSoundness.Discharge.ZkReg

open Flock
open FlockSoundness.Discharge.Exec (Inputs recordDraw)
open FlockSoundness.Discharge.Composed (nTab drawJ partOf)

variable {n : ℕ} {Gate : Type}

/-- **The registered reads are the verifier's own at their tables' draws, under hidden outputs, at `J` tables**:
`OwnReadsH` with each read `(u, q)` at the table `u` is drawn into (`partOf I d u`), whose draw is
`drawJ I d (partOf I d u)` and whose drawn statement is `HmOut.drawn c pub` at it. -/
def OwnReadsHJ (I : Inputs) (own : Array Registered.Port) (c : Circuit) (pub : Public) (t : Exec.Transcript)
    (isRead : (Fin n × Fin c.ports.size) ⊕ Gate → Prop) (port : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port)
    (readAt : (Fin n × Fin c.ports.size) ⊕ Gate → ℕ) : Prop :=
  ∀ p, isRead p → ∃ u q d D dp idx, p = .inl (u, q) ∧ port p ∈ own ∧
    (List.range c.ports.size).find? (fun j => c.ports[j]!.name == (port p).name) = some q.val ∧
    recordDraw t.record = some d ∧ Draw.UnitDraw.ofJson (drawJ I d (partOf I d u.val)) = .ok D ∧
    HmOut.drawn c pub (drawJ I d (partOf I d u.val)) = .ok dp ∧
    Registered.unitIndices dp = .ok idx ∧ u.val ∈ D.proved ∧
    (port p).reads[idx.getD (D.proved.idxOf u.val) 0]? = some (some (readAt p))

/-- **The paths of the registered reads, under hidden outputs, at `J` tables**: `regPathsH` with each row `(u, q)` at the
table `u` is drawn into, through the path that table's drawn statement gives. -/
def regPathsHJ (I : Inputs) (c : Circuit) (pub : Public) (t : Exec.Transcript)
    (port : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port) : (Fin n × Fin c.ports.size) ⊕ Gate → ByteArray
  | .inl (u, q) => Option.getD (do
      let d ← recordDraw t.record
      let D ← (Draw.UnitDraw.ofJson (drawJ I d (partOf I d u.val))).toOption
      let dp ← (HmOut.drawn c pub (drawJ I d (partOf I d u.val))).toOption
      let tb ← dp.tables
      let (_, paths) ← (Registered.entry dp (port (.inl (u, q)))).toOption
      return paths.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) ByteArray.empty)
      ByteArray.empty
  | .inr _ => ByteArray.empty

theorem drawJ_one {I : Inputs} (h1 : I.count ≤ 1) (d : Lean.Json) (j : ℕ) : drawJ I d j = d := by
  simp [drawJ, h1]

/-- At one table, `OwnReadsHJ` is `OwnReadsH`. -/
theorem ownReadsHJ_one {I : Inputs} (h1 : I.count ≤ 1) (own : Array Registered.Port) (c : Circuit) (pub : Public)
    (t : Exec.Transcript) (isRead : (Fin n × Fin c.ports.size) ⊕ Gate → Prop)
    (port : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port) (readAt : (Fin n × Fin c.ports.size) ⊕ Gate → ℕ) :
    OwnReadsHJ I own c pub t isRead port readAt ↔ OwnReadsH own c pub t isRead port readAt := by
  simp only [OwnReadsHJ, OwnReadsH, drawJ_one h1]

/-- At one table, `regPathsHJ` is `regPathsH`. -/
theorem regPathsHJ_one {I : Inputs} (h1 : I.count ≤ 1) (c : Circuit) (pub : Public) (t : Exec.Transcript)
    (port : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port) :
    regPathsHJ I c pub t port = regPathsH c pub t port := by
  funext p
  rcases p with ⟨u, q⟩ | g
  · simp only [regPathsHJ, regPathsH, drawJ_one h1]
  · rfl

end FlockSoundness.Discharge.ZkReg
