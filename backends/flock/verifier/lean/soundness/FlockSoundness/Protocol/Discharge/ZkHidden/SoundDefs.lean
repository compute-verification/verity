import FlockSoundness.Protocol.Discharge.ZkHidden.Defs
import FlockSoundness.Protocol.Discharge.HiddenExec.Placed
import FlockSoundness.Protocol.Discharge.ZkExec.LiveStatements
import FlockSoundness.Protocol.Discharge.HiddenDec.Statements

/-!
# `RunsZC` at the hidden-output `--zk` tables (statement)

ZkExec's `RunsZCLive` (`ZkExec/LiveStatements.lean`) for the sessions production `--zk` writes, whose tables
`Stmt.setupTables` sets up with `Stmt.setupHidden` (hidden outputs, the default `--statement`). Its `opnAtZ` reads the
statement `Stmt.setupH` sets up, so the hidden case is a different `RunsZC` instance:

* `opnAtZH`: `opnAtZ` at the statement `Stmt.setupHidden` sets up at the record's draw under the `--zk` tags and coin
  tree.
* `RunsZCLiveH`: `RunsZCLive` with `tablesAtZKH`, `opnAtZH`, `TablesAtZKHLive`, `TagsOkH` and `DrawOkZKH` for
  `tablesAtZK`, `opnAtZ`, `TablesAtZKLive`, `TagsOk` and `DrawOkZK`, and `TablesAtZKHLive`'s hypotheses on the circuit
  (`HmOut.parse` accepts it at the `--zk` tags; its unit returns the outputs). The decoder `decAtZ`, the empty tail and
  `capOfRoot` are the public case's.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock Refine Exec Game Model DecodeZL ZkSession ZkExec

variable {n mPts : ℕ} {Reg : Type}

/-- **The `--zk` openings at draw `S` under hidden outputs** (`opnAtZ` with `setupHidden` for `setupH`): `opnZ` at the
statement setupHidden sets up at the record's draw `dj S` under the `--zk` tags and coin tree, and its `--zk` schedule,
on the first two proof files. -/
noncomputable def opnAtZH (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) (I : Inputs) (dj : Finset (Fin n) → Lean.Json)
    (S : Finset (Fin n)) (R : Reg) (proofs : Array ByteArray) : ZkLink.OpensZ 𝔽 Dig (List UInt8) :=
  match Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
      I.program with
  | .ok st => match Zk.schedule st.m with
    | .ok (sch, _) => opnZ (Setup.ofCircuit st) sch (nXZ T S R) #[proofs[0]!, proofs[1]!]
    | .error _ => default
  | .error _ => default

end FlockSoundness.Discharge.ZkHidden
