import FlockSoundness.Discharge.PrivateCircuit.Interp
import FlockSoundness.Discharge.PrivateCircuit.Statements

/-!
# Protocol 2, the private circuit: padding computes the circuit

* `pad_correct`: `PadCorrect` at the universal unit (`Interp.lean`). A unit of `C` padded to the class's `(G, nIn)`, run
  by `UniversalUnit_v1{G, nIn, nOut}`, computes what `interpret` computes on the unit; every unit past `C`'s is a dummy,
  which outputs `nOut` zeros on every input.
  - `wires_sim`: gate by gate, the padded unit's wires hold the netlist's, each at its shifted index, while the
    indices fit `K` bits (`Nat.lt_log2_self`).
  - `gateFn_op`: `build`'s gate function at op's two bits is `interpret`'s rule.
* `pad_wrong`: `PadWrong`, the wrong units' count through `PadCorrect`, at any scheme.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

/-- **Padding computes the circuit, at the universal unit**: `PadCorrect` for `UniversalUnit_v1`'s interpreter, the
padding of `universal.pad`, and the dummy unit. -/
theorem pad_correct :
    PadCorrect Universal.fits Universal.pad Universal.units Universal.size Universal.run Universal.eval Universal.dummy := by
  sorry

/-- **Padding adds no wrong unit, and hides none.** -/
theorem pad_wrong : PadWrong := by
  sorry

end FlockSoundness.Discharge.PrivateCircuit
