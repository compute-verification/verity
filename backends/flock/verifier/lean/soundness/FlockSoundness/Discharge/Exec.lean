import FlockSoundness.Discharge.Exec.Event
import FlockSoundness.Discharge.Exec.SessionL
import FlockSoundness.Discharge.Exec.Setup
import FlockSoundness.Discharge.Exec.Tables
import FlockSoundness.Discharge.Exec.Headline
import FlockSoundness.Discharge.Exec.CompiledL
import FlockSoundness.Discharge.Exec.CompiledTableL
import FlockSoundness.Discharge.Exec.SessionSoundL
import FlockSoundness.Discharge.Exec.SessionAccL
import FlockSoundness.Discharge.Exec.SessionBatchL
import FlockSoundness.Discharge.Exec.SessionBatchAccL

/-!
# R11: flock-verify's acceptance and the audit as Lean defines it

* `Event`: flock-verify's verdict on a session directory (`Accepts`), the live audit whose outcome carries it
  (`auditLive`), and R11 at one strategy pair (`ExecRefines`);
* `SessionL`: the batched session under a leaf scheme (`tableAfterL`, `sessionBL`, `batchedSessionL`), and over any
  table run (`sessionG`, `batchedSessionG`);
* `Setup`: flock-verify's acceptance sets up each table with `setupH` (`accepts_setupH`, `setupH_public`);
* `Tables`: each accepted table is one accepting run of `tableAfterL` at `leaf512` (`accepts_tables_hm96`);
* `CompiledL`: compiled soundness of one table under any leaf scheme (`table_sound_compiledL`), the salted form of
  `table_sound_compiled`;
* `CompiledTableL`: the same for any level-0 table (`table_sound_compiled_of_tableL`), the reps' terms independent of
  it (`advRTL_eq`);
* `SessionSoundL`, `SessionAccL`, `SessionBatchL`, `SessionBatchAccL`: `Session.lean`, `SessionAcc.lean`,
  `SessionBatch.lean` and `SessionBatchAcc.lean` over a leaf scheme: each table of the salted batched session is
  knowledge sound (`session_knowledge_soundL`, `session_knowledge_sound_accL`, `joint_le_BL`, `joint_le_accBL`);
* `Headline`: the headline carried to the live audit through `ExecRefines` (`Prog.flock_headline_exec`,
  `flock_headline_exec_L`).
-/
