module

public import Flock.Zk
public import Proofs.Flock.Soundness.Discharge.Exec.Event
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-!
# flock-verify's `--zk` acceptance, as an event of the audit

`flock-verify verify --zk` (`Main.verifyCmd` with `--zk`, live/PROTOCOL.md, `Flock.Zk`) checks a
`flock-circuit --zk` session directory. It reads the same flags as a plain `verify` (`Exec.Inputs`) and differs in four
places, all of which this file restates:

* the coins are the keyed coin tree of the session's tables (`zkCoins`: `Coins.tree (Zk.coinSpec (tableNames J))`),
  never `--coins` (`verifyCmd` refuses `--zk` with `--coins`);
* the tags are `--statement`'s with the `--zk` identity's `hooks` (`Zk.tags`, at the circuit's `mask_log` and the session's table count `J`): `zkTags`;
* every table's statement passes `Zk.shapeOk` (level 0's query count and the mask slot's words);
* the verdict is `Flock.Zk.verify` (the masked replay, level 0's hiding, the inner proof), not `Flock.verify`.

`AcceptsZK I t` is that verdict on the transcript `t`: the session's statement set up from `zkInputs I` and the
record's sample (`Exec.setupOf`, `Stmt.setupTables`), every table's shape check, and `Zk.verify` accepting the record and
proofs.
`verifyCmd` refuses more: a statement's registered reads (`checkRegistered`, under `--registered`), its header's
identifier against the verifier's own (`checkIdentifier`, `--identifier`), and a coin-tree key an earlier session of the
run used (`CoinTree.checkFresh`), which only shrink the event.

`AcceptsZK` covers every tag `--statement` names, hidden outputs (`setupTables` calls `setupHidden` per table) and any
table count and sample: nothing here fixes them. Lemmas that read `setupH`'s statement take `hh : hiddenOutputs = false`.
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkExec

open Lean (Json)
open Game Audit Exec

/-- The coins `verify --zk` sets a session up on: the keyed coin tree of the session's tables (`Zk.coinSpec`: the
session's stream, then each table's two reps). -/
def zkCoins (I : Inputs) : Flock.Coins := .tree (Flock.Zk.coinSpec (I.tags.tableNames I.count))

/-- The tags `verify --zk` sets a session up under: `--statement`'s (`I.tags`) with the `--zk` identity's `hooks` at the
circuit's `mask_log` and the session's table count (`Zk.tags`). -/
def zkTags (I : Inputs) : Flock.Tags :=
  Flock.Zk.tags I.tags (Flock.Zk.maskLogOf I.circuitFile) I.count

/-- `I` as `verify --zk` sets the session up: `--zk`'s tags and coins in place of `--statement`'s and `--coins`'. -/
def zkInputs (I : Inputs) : Inputs := { I with tags := zkTags I, coins := zkCoins I }

/-- **flock-verify accepts the `--zk` transcript**: the session's statement sets up from `zkInputs I` and the record's
sample, every table passes `Zk.shapeOk`, and `Flock.Zk.verify` accepts the record and the proof files. -/
def AcceptsZK (I : Inputs) (t : Transcript) : Prop :=
  ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧ sts.forM Flock.Zk.shapeOk = .ok () ∧
    Flock.Zk.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok ()

variable {n : ℕ} {Reg : Type}

/-- flock-verify `--zk` accepts the live audit's transcript. -/
def LiveAcceptsZK (I : Inputs) (o : Reg × Finset (Fin n) × Transcript) : Prop := AcceptsZK I o.2.2

/-! ## The `--zk` tags change only the identity -/

theorem zkTags_eq (I : Inputs) : zkTags I = Flock.Zk.tags I.tags (Flock.Zk.maskLogOf I.circuitFile) I.count := rfl

theorem zkTags_hm96Rows (I : Inputs) : (zkTags I).hm96Rows = I.tags.hm96Rows := by rw [zkTags_eq]; rfl
theorem zkTags_typed (I : Inputs) : (zkTags I).typed = I.tags.typed := by rw [zkTags_eq]; rfl
theorem zkTags_hiddenOutputs (I : Inputs) : (zkTags I).hiddenOutputs = I.tags.hiddenOutputs := by rw [zkTags_eq]; rfl
theorem zkTags_retainRounds (I : Inputs) : (zkTags I).retainRounds = I.tags.retainRounds := by rw [zkTags_eq]; rfl
theorem zkTags_merkleLeaf (I : Inputs) : (zkTags I).merkleLeaf = I.tags.merkleLeaf := by rw [zkTags_eq]; rfl
theorem zkTags_manyTables (I : Inputs) : (zkTags I).manyTables = I.tags.manyTables := by rw [zkTags_eq]; rfl

end FlockSoundness.Discharge.ZkExec
