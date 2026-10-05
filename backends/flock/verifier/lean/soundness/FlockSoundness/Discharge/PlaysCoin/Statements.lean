import FlockSoundness.Discharge.Exec.LiveOn
import FlockSoundness.Discharge.Exec.Shape
import FlockSoundness.Protocol.Discharge.PlaysCoin.Statements

/-!
# The coin half of `DecodesAll` and `DecodesOn` (R11b): statements

`Decodes` (`Refine/LiveSim.lean`) asks for each rep's run on the tape `repTape`: each round's messages, decoded from its
bytes, then the round's coin step `coinMsg (one j)`; after the last round, the messages `fin` decodes from the proof
files. The refinement (`accepts_verdict_hm96`) gives each rep's run on two lists, its messages and its coin arrays
(`Plays`). With `Plays.coinVec` at `n ≠ 1`, a run's coin step is determined by its array (`coinArr`). So with
`one := fun _ => true` the coin steps are `coinMsg true` of the rounds, and what remains is the messages: the run's
messages, cut at the coins of the rep's spine, must be the decoder's groups.

* `ToTExact`: a run on two lists is a run on one tape whose coin steps are its arrays, as `coinArr` reads them.
* `CoinsOfRounds`: on well-formed rounds, those coin steps are `coinMsg true` of the rounds.
* `TapeOfGroups`: a tape is its coin steps and its messages cut at them (`cut`). When those are the rounds'
  `coinMsg true` and the decoder's groups (`roundGroups`) then `tail`, it is `tapeFrom dec (fun _ => true) j rs` then
  `tail`.
* `CutOfPat`: messages that are groups then a tail, along a spine whose coin pattern (`coinPat`) is theirs
  (`groupPat`), cut back into those groups and tail.
* `DecodesOfRuns`, `DecodesAllOfRuns`, `DecodesOnOfRuns`: `Decodes`, `DecodesAll` and `DecodesOn` at
  `one := fun _ => true`, from each rep's `RepRun`: a run on its rounds' coins whose messages, cut at its spine's coins,
  are the decoder's groups of the rounds and then `fin`'s messages.
* `DecodesOfRunsAt`: `DecodesOfRuns` for a game with no spine, such as one that goes on past a reject: each rep's spine
  is chosen at the outcome and holds along the runs that end where its `RepRun` ends (`SpineAt`).
-/

namespace FlockSoundness.Discharge.PlaysCoin

open Flock FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge.Exec Model Accounting Audit
  Audit.Partition

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## Definitions -/

/-! ## Statements -/

end FlockSoundness.Discharge.PlaysCoin
