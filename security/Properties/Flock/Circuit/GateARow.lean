module

public import Properties.Flock.Circuit.CircuitOf
public import Spot.FilePart

/-!
# `Γ`'s unit: the committed circuit has one gate a row

`GateARow`, a count over the game's circuit of committed files that reads no play: the circuit of a file
(`Flock.Bridge.circuitOfFile`, `Properties.Flock.Circuit.CircuitOf`) against the rows the verifier counts from the
parsed file (`Spot.FilePart`). It is proved (`Flock.Bridge.gateARow`); `Properties.Verity.Committed` says what it ties.
-/

@[expose] public section

namespace Flock.Bridge

/-- **`Γ`'s unit: the committed circuit has one gate a row.** Every unit of the game's circuit of the committed files
(`Flock.Bridge.circuitOfFile`, its units `partsOfFile`) has as many gates as its circuit file has rows a unit: the `m`
computed rows the verifier counts from what the files parse to (`Flock.Bridge.parsedOf`, `Spot.FilePart.ofCircuit`),
the unit net's rows stacked `units_per_vu` times. A unit's gates are its computed gates: the input gates are in no
unit. Files that don't parse have a circuit with no unit, so of them it says nothing.

It is a count over the game's circuit and reads no play. What it does not say: that a row is one gate of the sampling
claims' program, which lowers a row to three gates (`Spot.FilePart.block`). -/
def GateARow : Prop :=
  ∀ (tags : Flock.Tags) (cf : ByteArray) (tables : List (String × ByteArray)) (pf : ByteArray)
    (u : Fin (Flock.Bridge.unitsOfFile tags cf tables pf)),
      (Flock.Bridge.circuitOfFile tags cf tables pf).size ((Flock.Bridge.partsOfFile tags cf tables pf).gates u) =
        (Flock.Bridge.parsedOf tags cf tables pf).elim 0 fun r => (Spot.FilePart.ofCircuit r.D.unroll r.n).m

end Flock.Bridge
