module

public import Definitions.Spot.Dataflow
public import Definitions.Spot.Partition
public import Definitions.Spot.LongValue
public import Definitions.Spot.Oblivious
public import Definitions.Spot.AddressMap
public import Definitions.Spot.Necessity
public import Definitions.Spot.Exfiltration
public import Definitions.Spot.SamplePressure
public import Definitions.Spot.ChiPool
import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Algebra.Order.Field.Power
import Mathlib.Analysis.Complex.Order
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Data.EReal.Inv

/-! Populations' vocabulary: units over committed value slots with public read and write sets, which SPOT's guarantee is about. -/

@[expose] public section
