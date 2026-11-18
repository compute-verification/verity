module

/-!
# The label of a value

**A value's label** is where its commitment places it, named in the code's own coordinates, and nothing about when:

* `Label.wire registration writer field unit`: a wire's value, the leaf `unit` of writer template `writer`'s output
  tree of `field`, in registration `registration`. It is the leaf a reader row reads (`Circuits.Wire.leafAt`), the
  coordinates SPOT's wiring (`Spot.Edges.edgesOf`) and Flock's registered reads (`Flock.Registered`) read;
* `Label.input program value shape`: a registered input, value `value` of program `program`, its rows of `shape`
  (words, leaves and word width, as the registered value's domain binds them: `Spot.Registered.Entry`);
* `Label.coin family index`: a challenger's coin, member `index` of family `family` (`Challenger.Family`; a code log
  entry's label and index, `Spot.Challenger.Entry`).

A label holds no time, window, position in a log or coin index of a release: when a value is recorded or a coin
released is the challenger's log's (`Challenger.Log`). `Properties.Spot.Labels` reads the code's coordinates as labels.
-/

@[expose] public section

namespace Verity

/-- **The label of a value** (the module docstring), its names of type `Name`. -/
inductive Label (Name : Type) where
  | wire (registration writer field : Name) (unit : Nat)
  | input (program value : Name) (shape : List Nat)
  | coin (family : Name) (index : Nat)
  deriving DecidableEq

end Verity
