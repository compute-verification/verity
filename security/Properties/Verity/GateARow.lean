module

public import Properties.Flock.Circuit.CircuitOf
public import Spot.FilePart

/-! `GateARow`, `Γ`'s unit: a count over the game's circuit of committed files that reads no play. -/

@[expose] public section

namespace Verity

/-- **`Γ`'s unit: the committed circuit has one gate a row.** At all committed files that parse
(`Flock.Bridge.parsedOf`), every unit of the game's circuit of them (`Flock.Bridge.circuitOfFile`, its units
`partsOfFile`) has as many gates as its circuit file has rows a unit: the `m` computed rows the verifier counts from
the parsed file (`Spot.FilePart.ofCircuit`), the unit net's rows stacked `units_per_vu` times. A unit's gates are its
computed gates: the input gates are in no unit.

It is a count over the game's circuit and reads no play. What it does not say: that a row is one gate of the sampling
claims' program, which lowers a row to three gates (`Spot.FilePart.block`); or anything of files that don't parse,
whose circuit has no unit. -/
def GateARow : Prop :=
  ∀ (tags : Flock.Tags) (cf : ByteArray) (tables : List (String × ByteArray)) (pf : ByteArray)
    (r : Flock.Bridge.Parsed), Flock.Bridge.parsedOf tags cf tables pf = some r →
    ∀ u : Fin (Flock.Bridge.unitsOfFile tags cf tables pf),
      (Flock.Bridge.circuitOfFile tags cf tables pf).size ((Flock.Bridge.partsOfFile tags cf tables pf).gates u) =
        (Spot.FilePart.ofCircuit r.D.unroll r.n).m

end Verity
