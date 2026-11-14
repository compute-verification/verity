module

public import Proofs.Properties.Verity.Sampled.HonestSession.PublicFile
public import Proofs.Properties.Verity.Sampled.HonestSession.Program

/-!
# `verify --zk` sets up the table at the sample file

`Stmt.setupHidden` at table 0's inputs and `draw₀` returns `st₀` (`setupHidden_ok`) and its statement passes the ZK shape
check (`shapeOk_ok`). The walk reads `CircuitFile.parseCommit_ok` and `hmIn_parse`, the partition check at
`Program.partitionDigest_ok`, `PublicFile`'s loads, draw and check, the template query's unit derivation
(`HmRow.deriveTemplateUnits`, `HmLinks.check`) over `Program`'s descriptor, then the schedule, the pins, Δ and the
regions, as `Proofs.Flock.SessionLoop.Concrete.Setup` and `Trace` walk `setupH`.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.Setup

open Flock
open FlockSoundness.Discharge

theorem draw_ofJson : Flock.Draw.UnitDraw.ofJson draw₀ = .ok { law := .subset 1, population := nUnits, units := #[0] } := by
  sorry -- piece P3 (setup walk)

theorem setupHidden_ok :
    Stmt.setupHidden (Composed.inputsJ inputs 0).tags (Composed.inputsJ inputs 0).circuitFile
      (Composed.inputsJ inputs 0).publicFile (Composed.inputsJ inputs 0).coins (Composed.inputsJ inputs 0).tables
      (some draw₀) (Composed.inputsJ inputs 0).partition (Composed.inputsJ inputs 0).program none = .ok st₀ := by
  sorry -- piece P3 (setup walk)

theorem shapeOk_ok : Zk.shapeOk st₀ = .ok () := by
  sorry -- piece P3 (setup walk)

end Verity.Sampled.Instance.HonestSession.Setup
