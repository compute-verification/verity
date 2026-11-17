module

public import Definitions.ResourceAccounting.Gates.Model
public import Definitions.ResourceAccounting.Gates.Registry
public import Definitions.ResourceAccounting.Gates.Codec
public import Definitions.ResourceAccounting.Gates.Ranges
public import Definitions.ResourceAccounting.Gates.Declared
public import Definitions.ResourceAccounting.Gates.Circuit

/-! The circuit model (`ResourceAccounting.Gates`): a registered circuit's gates over bounded integers, its window and the interval analysis that proves it, its census, the stacking property, the registry's primitives and the word codec. -/

@[expose] public section
