module

public import Proofs.Properties.Verity.Sampled.HonestSession.PublicFile
public import Proofs.Properties.Verity.Sampled.HonestSession.Program

/-!
# `verify --zk` sets up the table at the sample file

`Stmt.setupHidden` at table 0's inputs and `draw₀` returns `st₀` and its statement passes the ZK shape check. The
walk reads `CircuitFileFacts.parseCommit_ok` and `hmIn_parse`, the partition check at
`ProgramFacts.partitionDigest_ok`, `PublicFileFacts`' loads, draw and check, the template query's unit derivation
(`HmRow.deriveTemplateUnits`, `HmLinks.check`) over `Program`'s descriptor, then the schedule, the pins, Δ and the
regions, as `Proofs.Flock.SessionLoop.Concrete.Setup` and `Trace` walk `setupH`. These facts are the named proposition
`SetupFacts`, proved by `ProgramSetup.setupFacts_prog` given `PublicFileFacts` and four reads of the public file: the
pieces that use it take it as a hypothesis, and `Witness` supplies it.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open Flock
open FlockSoundness.Discharge

/-- **Facts about `verify --zk`'s setup of the concrete session** (`ProgramSetup.setupFacts_prog`, given four reads of
the public file): the sample file `draw₀` reads to one unit
of a one-unit subset law, the setup at table 0 returns the statement `st₀`, and `st₀` passes the zero-knowledge shape
check. -/
structure SetupFacts : Prop where
  /-- The sample file reads to the subset-of-one law over `nUnits` with unit `0`. -/
  draw_ofJson : Flock.Draw.UnitDraw.ofJson draw₀ = .ok { law := .subset 1, population := nUnits, units := #[0] }
  /-- `Stmt.setupHidden` at table 0's inputs and `draw₀` returns `st₀`. -/
  setupHidden_ok :
    Stmt.setupHidden (Composed.inputsJ inputs 0).tags (Composed.inputsJ inputs 0).circuitFile
      (Composed.inputsJ inputs 0).publicFile (Composed.inputsJ inputs 0).coins (Composed.inputsJ inputs 0).tables
      (some draw₀) (Composed.inputsJ inputs 0).partition (Composed.inputsJ inputs 0).program none = .ok st₀
  /-- `st₀` passes the zero-knowledge shape check. -/
  shapeOk_ok : Zk.shapeOk st₀ = .ok ()

end Verity.Sampled.Instance.HonestSession
