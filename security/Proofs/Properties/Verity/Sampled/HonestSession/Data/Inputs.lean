module

public import Proofs.Properties.Verity.Sampled.HonestSession.Data.Params
public import Proofs.Properties.Verity.Sampled

/-!
# The honest session's inputs and the values the verifier derives from them

`inputs` are the ones v2 hands `flock-verify` on `θ₀` and the registered files: the statement's tags, the circuit and
public files, one table, the verifier's partition object and program, no work, stratified or closure input. The sample
file `draw₀` samples the one unit. The verifier's derived values (the circuit, the public file as each side reads it,
its drawn statement and the table's statement) are named by `getD` on the function that computes them, so a piece states
that the function returns `.ok` of the name and no statement evaluates it.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open Lean (Json)
open Flock
open FlockSoundness.Discharge

/-- The inputs v2 hands `flock-verify`. -/
def inputs : Exec.Inputs where
  tags := Flock.Tags.circuit
  circuitFile := circuitFile
  publicFile := publicFile
  coins := default
  count := 1
  tables := []
  partition := some partitionObj
  program := some programFile
  work := none
  stratified := none
  closure := none

/-- `verify --zk`'s tags at the inputs. -/
def tagsZ : Flock.Tags := ZkExec.zkTags inputs

/-- The sample file: the one unit of the population of one, under the subset law of one. -/
def draw₀ : Json := Flock.Draw.UnitDraw.toJson { law := .subset 1, population := nUnits, units := #[0] }

/-- The circuit `HmOut.parse` reads from the circuit file. -/
def cH : Flock.Circuit := (HmOut.parse tagsZ circuitFile []).toOption.getD default

/-- The public file as the circuit's hidden outputs read it. -/
def pubO : Flock.Public := (HmOut.loadPublic tagsZ cH publicFile).toOption.getD default

/-- The public file over the circuit's empty public input ports. -/
def pubI : Flock.Public := (HmIn.loadPublic tagsZ cH {} publicFile).toOption.getD default

/-- The sampled statement's public file, at the sample file. -/
def pubD : Flock.Public := (HmIn.drawn cH {} pubI draw₀).toOption.getD default

/-- The table's statement `verify --zk` sets up at the sample file. -/
def st₀ : Flock.Stmt :=
  (Stmt.setupHidden (Composed.inputsJ inputs 0).tags (Composed.inputsJ inputs 0).circuitFile
    (Composed.inputsJ inputs 0).publicFile (Composed.inputsJ inputs 0).coins (Composed.inputsJ inputs 0).tables
    (some draw₀) (Composed.inputsJ inputs 0).partition (Composed.inputsJ inputs 0).program none).toOption.getD default

theorem tags_find : Flock.Tags.find? Spot.OneStageZk.STATEMENT = some inputs.tags := rfl

theorem typed_false : inputs.tags.typed = false := rfl

theorem tagsJ_eq : (Composed.inputsJ inputs 0).tags = tagsZ := by
  show Exec.tagsAt _ inputs.count 0 = _
  unfold Exec.tagsAt
  exact ite_eq_left (by decide)

end Verity.Sampled.Instance.HonestSession
