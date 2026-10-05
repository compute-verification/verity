import Flock.Zk
import FlockSoundness.Discharge.Exec.Event
import FlockSoundness.Protocol.Discharge.ZkExec.Event

/-!
# flock-verify's `--zk` acceptance, as an event of the audit

`flock-verify verify --zk` (`Main.verifyCmd` with `--zk`, live/PROTOCOL.md, PROTOCOL.md §16.14) checks a
`flock-circuit --zk` session directory. It reads the same flags as a plain `verify` (`Exec.Inputs`) and differs in four
places, all of which this file restates:

* the coins are the keyed coin tree of the session's tables (`zkCoins`: `Coins.tree (Zk.coinSpec (tableNames J))`),
  never `--coins` (`verifyCmd` refuses `--zk` with `--coins`);
* the tags are `--statement`'s with the coin tree (`withCoins`, the identity on a tree) and the `--zk` identity's
  `hooks` (`Zk.tags`, at the circuit's `mask_log` and the session's table count `J`): `zkTags`;
* every table's statement passes `Zk.shapeOk` (level 0's query count and the mask slot's words);
* the verdict is `Flock.Zk.verify` (the masked replay, level 0's hiding, the inner proof), not `Flock.verify`.

`AcceptsZK I t` is that verdict on the transcript `t`: the session's statement set up from `zkInputs I` and the
record's draw (`Exec.setupOf`, `Stmt.setupTables`), every table's shape check, and `Zk.verify` accepting the record and
proofs.
`verifyCmd` refuses more: a statement's registered reads (`checkRegistered`, under `--registered`), its header's
identifier against the verifier's own (`checkIdentifier`, `--identifier`), and a coin-tree key an earlier session of the
run used (`CoinTree.checkFresh`), which only shrink the event.

`AcceptsZK` covers every tag `--statement` names, hidden outputs (`setupTables` calls `setupHidden` per table) and any
table count and draw: nothing here fixes them. Lemmas that read `setupH`'s statement take `hh : hiddenOutputs = false`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Lean (Json)
open Game Audit Exec

variable {n : ℕ} {Reg : Type}

/-- flock-verify `--zk` accepts the live audit's transcript. -/
def LiveAcceptsZK (I : Inputs) (o : Reg × Finset (Fin n) × Transcript) : Prop := AcceptsZK I o.2.2

/-! ## The `--zk` tags change only the identity -/

theorem zkTags_hm96Rows (I : Inputs) : (zkTags I).hm96Rows = I.tags.hm96Rows := by rw [zkTags_eq]; rfl
theorem zkTags_retainRounds (I : Inputs) : (zkTags I).retainRounds = I.tags.retainRounds := by rw [zkTags_eq]; rfl
theorem zkTags_merkleLeaf (I : Inputs) : (zkTags I).merkleLeaf = I.tags.merkleLeaf := by rw [zkTags_eq]; rfl
theorem zkTags_manyTables (I : Inputs) : (zkTags I).manyTables = I.tags.manyTables := by rw [zkTags_eq]; rfl

end FlockSoundness.Discharge.ZkExec
