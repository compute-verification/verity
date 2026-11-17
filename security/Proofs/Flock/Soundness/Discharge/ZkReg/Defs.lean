module

public import Flock.Registered
public import Proofs.Flock.Soundness.Discharge.ZkExec.Event
public import Proofs.Flock.Soundness.Registered.Defs
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.CategoryTheory.Category.Init
import Mathlib.Combinatorics.Matroid.Init
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic.Positivity.Finset
import VCVio.Prelude.Core

/-! `verify --zk --registered F` as an event of the audit (`AcceptsZKR`), and what its registered check reads at a
sampled statement. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkReg

open Lean (Json)
open Flock
open FlockSoundness.Discharge.Exec (Inputs recordDraw setupOf)
open FlockSoundness.Discharge.ZkExec (zkInputs)

export Flock (checkRegistered)

/-- **verity-verify-proof `--zk --registered F` accepts the transcript**, `own` F's ports: the session's statements set up from
`zkInputs I` and the record's sample, each passes the registered check against `own`, every port of each one's circuit is
a port of `own` (`Registered.checkComplete`, which verity-verify-proof runs whenever `--registered` is given), each passes
`Zk.shapeOk`, and `Flock.Zk.verify` accepts the record and the proof files. -/
def AcceptsZKR (I : Inputs) (own : Array Registered.Port) (t : Exec.Transcript) : Prop :=
  ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧ sts.forM (checkRegistered · (some own)) = .ok () ∧
    sts.forM (fun (st : Stmt) => Registered.checkComplete st.c own) = .ok () ∧
    sts.forM Flock.Zk.shapeOk = .ok () ∧ Flock.Zk.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok ()

variable {n : ℕ} {Reg : Type}

/-- verity-verify-proof `--zk --registered F` accepts the live audit's transcript. -/
def LiveAcceptsZKR (I : Inputs) (own : Array Registered.Port) (o : Reg × Finset (Fin n) × Exec.Transcript) : Prop :=
  AcceptsZKR I own o.2.2

variable {Gate : Type}

def OwnReads (own : Array Registered.Port) (c : Circuit) (pub : Public) (t : Exec.Transcript)
    (isRead : (Fin n × Fin c.ports.size) ⊕ Gate → Prop) (port : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port)
    (readAt : (Fin n × Fin c.ports.size) ⊕ Gate → ℕ) : Prop :=
  ∀ p, isRead p → ∃ u q d D dp idx, p = .inl (u, q) ∧ port p ∈ own ∧
    (List.range c.ports.size).find? (fun j => c.ports[j]!.name == (port p).name) = some q.val ∧
    recordDraw t.record = some d ∧ Draw.UnitSample.ofJson d = .ok D ∧ HmRow.drawn c pub d = .ok dp ∧
    Registered.unitIndices dp = .ok idx ∧ u.val ∈ D.proved ∧
    (port p).reads[idx.getD (D.proved.idxOf u.val) 0]? = some (some (readAt p))

def regPaths (c : Circuit) (pub : Public) (t : Exec.Transcript)
    (port : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port) : (Fin n × Fin c.ports.size) ⊕ Gate → ByteArray
  | .inl (u, q) => Option.getD (do
      let d ← recordDraw t.record
      let D ← (Draw.UnitSample.ofJson d).toOption
      let dp ← (HmRow.drawn c pub d).toOption
      let tb ← dp.tables
      let (_, paths) ← (Registered.entry dp (port (.inl (u, q)))).toOption
      return paths.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) ByteArray.empty)
      ByteArray.empty
  | .inr _ => ByteArray.empty

end FlockSoundness.Discharge.ZkReg
