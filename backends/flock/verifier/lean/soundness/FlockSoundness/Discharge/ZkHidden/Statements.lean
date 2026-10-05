import FlockSoundness.Discharge.ZkHidden.Defs
import FlockSoundness.Discharge.HiddenExec.Placed
import FlockSoundness.Protocol.Discharge.ZkHidden.Statements
import FlockSoundness.Discharge.Hidden.Theorems
import FlockSoundness.Discharge.HiddenExec.Statements

/-!
# The statements: flock-verify `--zk`'s acceptance and its tables under hidden outputs

ZkExec's acceptance statements and ZkLink's `TablesAtZKLive` take `hh : I.tags.hiddenOutputs = false`: they are about
`Stmt.setupH`'s statements, and under hidden outputs (the default `--statement`, PROTOCOL §16.13) `Stmt.setupTables`
calls `Stmt.setupHidden`. Each statement here is one of them for the sessions production `--zk` writes, as HiddenExec's
are for the public path (#869). Where the original names `setupH`, this one names `setupOne` (without `hh`) or
`setupHidden` (under `hh = true`); otherwise the conclusion is the original's.

* `AcceptsZKSetupOne`, `AcceptsZKOneAny`, `AcceptsZKRetainedAny`: ZkExec's `AcceptsZKSetupH`, `AcceptsZKOne`,
  `AcceptsZKRetained` without `hh`.
* `TablesAtZKHLive`: ZkLink's `TablesAtZKLive` under `hh = true`, at `tablesAtZKH`, with #869's `TableAtHOfAccepts`
  hypotheses on the circuit (`HmOut.parse` accepts it, at the `--zk` tags; its unit returns the outputs).
* `AcceptsZKAcceptedH`: lp-zkbind's `acceptsZK_accepted` (the placement every `--zk` table's binding is read at) for
  one hidden-output table, with #869's `AcceptedH` for `Placed.Accepted`.
* `ZkSchedOK`: the half of `TablesAtZKLive` that reads no setup: at any statement `Zk.shapeOk` passes, `Zk.schedule`'s
  schedule and `t_pad` have the facts `DrawSetupZK` and `DrawSetupZKH` record.

`zk_flock_countC_execOS_crH` (`Count.lean`) is #874's `zk_flock_countC_execOS_cr` at the hidden-output plan.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock FlockLevel3 Game Model Refine Exec DecodeZL ZkSession ZkExec ZkLink HiddenExec
open scoped ENNReal NNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## Acceptance -/

/-- **Each table of a `--zk` acceptance is `setupOne`'s statement** under the `--zk` tags and the coin tree, at some
draw (the record's, for one table), and passes the `--zk` shape check (`AcceptsZKSetupH` without `hh`). -/
def AcceptsZKSetupOne : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), AcceptsZK I t →
    ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧
      Flock.Zk.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok () ∧ 0 < sts.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), Flock.Zk.shapeOk sts[j] = .ok () ∧ ∃ d,
        setupOne (tagsAt (zkTags I) I.count j) I.circuitFile I.publicFile (zkCoins I) I.tables d I.partition
          I.program = .ok sts[j] ∧ (I.count ≤ 1 → d = recordDraw t.record)

/-- **A `--zk` acceptance under retaining tags has every round's bytes** (`AcceptsZKRetained` without `hh`). -/
def AcceptsZKRetainedAny : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), I.tags.retainRounds = true → AcceptsZK I t →
    ∃ spec sts sess, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧ Record.decode spec t.record = .ok sess ∧
      ∀ s ∈ sess.streams.toList, AllRetained s.rounds

/-- **A one-table hidden-output `--zk` acceptance is placed** (lp-zkbind's `acceptsZK_accepted` at one table, with
`AcceptedH` for `Accepted`): the table is setupHidden's statement `st` at the record's draw, it passes the shape check,
and at any regions there is an `AcceptedH` of `st` under the `--zk` tags on the session's files, coins and tables. -/
def AcceptsZKAcceptedH : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), I.tags.hiddenOutputs = true → I.tags.typed = false → ∀ {c : Flock.Circuit},
    HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c → (∀ p ∈ c.ports.toList, p.v2 = false) →
    I.count ≤ 1 → AcceptsZK I t →
    ∃ st, Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (recordDraw t.record)
        I.partition I.program = .ok st ∧ Flock.Zk.shapeOk st = .ok () ∧
      ∀ regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)),
        ∃ (T : Model.Statement) (a : AcceptedH T), a.st = st ∧ HEq a.regs regs ∧ a.tags = zkTags I ∧
          a.cf = I.circuitFile ∧ a.pf = I.publicFile ∧ a.coins = zkCoins I ∧ a.tables = I.tables ∧
          a.draw = recordDraw t.record

/-! ## The schedule -/

/-! ## The tables -/

variable {n : ℕ} {mPts : ℕ} {Reg : Type}

end FlockSoundness.Discharge.ZkHidden
