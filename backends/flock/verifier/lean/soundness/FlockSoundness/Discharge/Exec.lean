import FlockSoundness.Discharge.Exec.Event
import FlockSoundness.Discharge.Exec.SessionL
import FlockSoundness.Discharge.Exec.Setup
import FlockSoundness.Discharge.Exec.Tables
import FlockSoundness.Discharge.Exec.Headline

/-!
# R11: flock-verify's acceptance and the audit as Lean defines it

* `Event`: flock-verify's verdict on a session directory (`Accepts`), the live audit whose outcome carries it
  (`auditLive`), and R11 at one strategy pair (`ExecRefines`);
* `SessionL`: the batched session under a leaf scheme (`tableAfterL`, `sessionBL`, `batchedSessionL`), and over any
  table run (`sessionG`, `batchedSessionG`);
* `Setup`: flock-verify's acceptance sets up each table with `setupH` (`accepts_setupH`, `setupH_public`);
* `Tables`: each accepted table is one accepting run of `tableAfterL` at `leaf512` (`accepts_tables_hm96`);
* `Headline`: the headline carried to the live audit through `ExecRefines` (`Prog.flock_headline_exec`,
  `flock_headline_exec_L`).
-/
