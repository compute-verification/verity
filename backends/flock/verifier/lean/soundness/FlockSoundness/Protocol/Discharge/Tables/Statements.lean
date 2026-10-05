import FlockSoundness.Protocol.Discharge.Tables.Sim
import FlockSoundness.Protocol.Discharge.Exec.LiveOn
import FlockSoundness.Protocol.Discharge.Exec.Setup

/-!
# Sessions of `J` tables: the statements

The open guarantee is sessions of more than one table (`--session-tables J`, `J > 1`). Each statement is a `Prop`
here, and its theorem is below it.

* `SessionTransfer`: the protocol-level transfer at `J` tables. Against any live prover of `liveSession` (one `Commit`
  of `J` roots, one link-point draw every table reads, the `2J` rep streams in any order the prover picks with each
  round's coins drawn after its bytes, then every proof file), the live verdict holds with probability at most the
  model session's acceptance of every table (`modelSession`, the tables concurrently through `batch`) against the
  simulated prover `simSession`, whenever every table's decoder is right (`DecodesTable`, one per table).
* `SessionBLModel`: the salted session on the tables `ts` is `modelSession`'s.
* `ExecRefinesTables`: R11 for any plan, `ExecRefines … (LiveAccepts I) ⊥ (audit … (batchedSessionL … plan)) (simAuditJ …)`
  given `DecodesOnJ`.
* `HeadlineTables`: `Prog.flock_headline_live_on` with `plan S R` of any length `J`, at the live stratified draw.
* `HeadlineTablesSubset`: the same at the live subset draw, with count curve `C(n − K, kd) / C(n, kd)`; flock-verify
  takes no other draw at `J > 1`.
* `AcceptsTables`: flock-verify's acceptance at `count > 1`: the record's draw, when there is one, is a subset draw;
  the session's units (the draw's, without one the population's) split into `count` equal parts; and table `j` is
  `setupH`'s statement under `Tags.tableIdentity count j` at the subset draw `partDraw` of part `j`.

The decoder (`DecodesTable`, inside `DecodesOnJ`) is a named hypothesis per draw, registration and table: at one table
it is `DecodesOn`'s, which [wave2-dec-zl] discharges for #828's tables, and its coin half is [wave2-plays-coin]'s.
-/

namespace FlockSoundness.Discharge.Tables

open Game Model Accounting Refine Audit Audit.Partition Exec
open scoped ENNReal

open Lean (Json) in
/-- Part `j` of `count` equal consecutive parts of the session's units, as the subset draw of the population that
`Stmt.setupTables` sets table `j` up at. -/
def partDraw (population : ℕ) (units : Array ℕ) (count j : ℕ) : Json :=
  Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson population),
    ("k", Lean.toJson (units.size / count)),
    ("units", Json.arr ((units.extract (j * (units.size / count)) (j * (units.size / count) + units.size / count)).map
      Lean.toJson))]

end FlockSoundness.Discharge.Tables
