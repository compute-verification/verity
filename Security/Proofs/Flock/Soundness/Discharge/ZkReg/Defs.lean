import Proofs.Flock.Soundness.Discharge.ZkExec.Event
import Proofs.Flock.Soundness.Registered.Defs

/-!
# `verify --zk --registered F`, as an event of the audit

`flock-verify verify --zk --registered F` (`Main.verifyCmd`) checks a `--zk` session as `ZkExec.AcceptsZK` does, and
holds every table's statement to the verifier's own registered reads `own` (F's ports, `Registered.ofJson`): after
`Stmt.setupTables` and before `Zk.shapeOk`, `Main.buildSession` runs `Flock.checkRegistered` on each statement, the
library's definition, which `AcceptsZKR` reads as it is. `AcceptsZKR` is that verdict; it implies `AcceptsZK`.

What the registered check reads at a drawn statement over a shared-row population `pub` (`Registered.check`): the
statement's units, as its header names them (`Registered.unitIndices`, which `HmRow.drawn` sets to the population's
units at the draw's proved instances), and the header's `registered` entry of each of `own`'s ports, a position and a
path per table row (`Registered.entry`). Instance `r` of the drawn statement is the population's instance
`D.proved[r]`, so a read at the population's instance `u` is the drawn statement's at `u`'s rank `D.proved.idxOf u`:

* `OwnReads own c pub t`: the layout's registered reads are the verifier's own at the record's draw: each at a row
  `(u, q)` of a drawn instance `u`, its port among `own` and the circuit's port `q` by name (`checkPort`'s `find?`),
  at the position the verifier's program reads in the unit the drawn statement names for `u`;
* `regPaths c pub t port`: the path the drawn statement's header gives for that row, at the table row `u`'s ref names.
-/

namespace FlockSoundness.Discharge.ZkReg

open Lean (Json)
open Flock
open FlockSoundness.Discharge.Exec (Inputs recordDraw setupOf)
open FlockSoundness.Discharge.ZkExec (zkInputs)

export Flock (checkRegistered)

/-- **flock-verify `--zk --registered F` accepts the transcript**, `own` F's ports: the session's statements set up from
`zkInputs I` and the record's draw, each passes the registered check against `own`, every port of each one's circuit is
a port of `own` (`Registered.checkComplete`, which flock-verify runs whenever `--registered` is given), each passes
`Zk.shapeOk`, and `Flock.Zk.verify` accepts the record and the proof files. -/
def AcceptsZKR (I : Inputs) (own : Array Registered.Port) (t : Exec.Transcript) : Prop :=
  ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧ sts.forM (checkRegistered · (some own)) = .ok () ∧
    sts.forM (fun (st : Stmt) => Registered.checkComplete st.c own) = .ok () ∧
    sts.forM Flock.Zk.shapeOk = .ok () ∧ Flock.Zk.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok ()

variable {n : ℕ} {Reg : Type}

/-- flock-verify `--zk --registered F` accepts the live audit's transcript. -/
def LiveAcceptsZKR (I : Inputs) (own : Array Registered.Port) (o : Reg × Finset (Fin n) × Exec.Transcript) : Prop :=
  AcceptsZKR I own o.2.2

variable {Gate : Type}

/-- **The registered reads are the verifier's own at the record's draw.** Each registered read `p` is the row `(u, q)`
of a drawn instance `u` (the record's draw `d` decodes to `D`, and `u ∈ D.proved`), its port is one of `own` and is
the circuit's port `q` by name, and its position is the one `own` gives for the unit the drawn statement over `pub`
(`HmRow.drawn c pub d`) names for `u`. -/
def OwnReads (own : Array Registered.Port) (c : Circuit) (pub : Public) (t : Exec.Transcript)
    (isRead : (Fin n × Fin c.ports.size) ⊕ Gate → Prop) (port : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port)
    (readAt : (Fin n × Fin c.ports.size) ⊕ Gate → ℕ) : Prop :=
  ∀ p, isRead p → ∃ u q d D dp idx, p = .inl (u, q) ∧ port p ∈ own ∧
    (List.range c.ports.size).find? (fun j => c.ports[j]!.name == (port p).name) = some q.val ∧
    recordDraw t.record = some d ∧ Draw.UnitDraw.ofJson d = .ok D ∧ HmRow.drawn c pub d = .ok dp ∧
    Registered.unitIndices dp = .ok idx ∧ u.val ∈ D.proved ∧
    (port p).reads[idx.getD (D.proved.idxOf u.val) 0]? = some (some (readAt p))

/-- **The paths of the registered reads**: at the row `(u, q)`, the path the drawn statement's header gives port
`port (u, q)` (`Registered.entry`) at the table row that `u`'s instance's ref names; empty elsewhere. -/
def regPaths (c : Circuit) (pub : Public) (t : Exec.Transcript)
    (port : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port) : (Fin n × Fin c.ports.size) ⊕ Gate → ByteArray
  | .inl (u, q) => Option.getD (do
      let d ← recordDraw t.record
      let D ← (Draw.UnitDraw.ofJson d).toOption
      let dp ← (HmRow.drawn c pub d).toOption
      let tb ← dp.tables
      let (_, paths) ← (Registered.entry dp (port (.inl (u, q)))).toOption
      return paths.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) ByteArray.empty)
      ByteArray.empty
  | .inr _ => ByteArray.empty

end FlockSoundness.Discharge.ZkReg
