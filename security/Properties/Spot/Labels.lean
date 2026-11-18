module

public import Spot.ChallengerSession
public import Spot.Registered
public import Definitions.Core.Label

/-!
# The code's coordinates of a value, read as its label

The core's label of a value (`Verity.Label`) in the coordinates the code already places values by, with names
`String`:

* `Circuits.Wire.toLabel`: the leaf a wire's reader row reads (`Circuits.Wire.leafAt`), the writer template's output tree
  of its field and the writer unit, in a registration;
* `Spot.Registered.Entry.toLabel`: a registered value of a program with its row shape (words, leaves, word width), the
  coordinates its domain binds;
* `Spot.Challenger.Entry.toLabel`: a challenger log entry's family and index, a coin's.

None reads a time, a window or a position in the log.
-/

@[expose] public section

/-- The label of the leaf reader unit `u`'s row `row` reads by wire `w`, in registration `registration`; none where the
wire doesn't cover that row. -/
def Circuits.Wire.toLabel (registration : String) (w : Circuits.Wire) (u row : Nat) : Option (Verity.Label String) :=
  (w.leafAt u row).map (.wire registration w.writer w.out)

/-- The label of registered value `e` of program `program`. -/
def Spot.Registered.Entry.toLabel (program : String) (e : Spot.Registered.Entry) : Verity.Label String :=
  .input program e.value [e.words, e.leaves, e.wordBits]

/-- The label of the coin a challenger log entry names. -/
def Spot.Challenger.Entry.toLabel (e : Spot.Challenger.Entry) : Verity.Label String :=
  .coin e.label e.index
