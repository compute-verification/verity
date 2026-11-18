module

public import Spot.ChallengerSession
public import Definitions.Core.Challenger
public import Definitions.Core.Label

/-!
# The challenger's release log, read as the core's log

`Spot.Challenger.releasedLogOf` reads a session's release log into entries in the log's one order, and
`Spot.Challenger.cut` checks a cut on them. These state what that code checks in the core's terms (`Verity.Challenger.Log`):

* `coreLog`: the code's log as the core's, a recorded entry the commitment `(label, index, root)`, a released one the
  coin `(label, index)` (`Verity.Label.coin`), in the code's order;
* `CutMeans`: on a log `releasedLogOf` accepts, `cut log payloads keys` accepting gives `Log.Cut` at the recorded
  commitments of the families `payloads` and the coins of the families `keys`;
* `KeyedBySlotMeans`: on a log `releasedLogOf` accepts, every family's coins are keyed by slot (`Log.KeyedBySlot`),
  member `s` released after the family's root `s` is recorded, once.

Where the code's log and the core's differ: the code carries times and refuses a release before its root's receipt; it
takes a root and its release as a pair, so a recorded root with no release after it is refused, where the core's log
interleaves records and releases freely; it drops a coin tree's lines; families interleave in both.
-/

@[expose] public section

namespace Spot.Challenger

/-- The code's log as the core's (the module docstring). -/
def coreLog (log : Log) : Verity.Challenger.Log (String × Nat × ByteArray) (String × Nat) :=
  log.map fun e => match e.kind with
    | .recorded => .recorded (e.label, e.index, e.root)
    | .released => .released (e.label, e.index)

/-- The root the code's log records as member `s` of family `fam`, the empty root where it records none. -/
def slotRoot (log : Log) (fam : String) (s : Nat) : String × Nat × ByteArray :=
  (fam, s, ((log.find? fun e => e.kind == .recorded && e.label == fam && e.index == s).map (·.root)).getD ByteArray.empty)

/-- **What `cut` checks is the core's cut** on the log `releasedLogOf` reads. -/
def CutMeans : Prop :=
  ∀ (c : LiveCommitment) (lines : List String) (log : Log) (payloads keys : List String),
    releasedLogOf c lines = .ok log → cut log payloads keys = true →
      (coreLog log).Cut (fun r => r.1 ∈ payloads)
        (fun i => ∃ k, (coreLog log).coins[i]? = some k ∧ k.1 ∈ keys)

/-- **The log `releasedLogOf` reads is keyed by slot** in every family: its coin `i` is member `(label, index)`
(`(0, ·)` past its last coin). -/
def KeyedBySlotMeans : Prop :=
  ∀ (c : LiveCommitment) (lines : List String) (log : Log) (fam : String),
    releasedLogOf c lines = .ok log →
      (coreLog log).KeyedBySlot (fun i => ((coreLog log).coins[i]?).getD ("", 0)) fam (slotRoot log fam)

end Spot.Challenger
