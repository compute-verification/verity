module

public import Definitions.ResourceAccounting.Gates.Model
public import Definitions.ResourceAccounting.Gates.Registry
public import Definitions.ResourceAccounting.Gates.Codec
public import Definitions.ResourceAccounting.Gates.Ranges
public import Definitions.ResourceAccounting.Gates.Declared
public import Definitions.ResourceAccounting.Gates.Circuit

/-! The circuit model (`ResourceAccounting.Gates`): a registered circuit's gates (`IntDag`) over bounded integers with their
classes, provenance and committed outputs, its bounded window and the interval analysis that proves it, its census and
the stacking property on it, the registry's primitives each in one class, the word codec with what computing its
words in the model means, the evaluators whose output intervals property/v1 reads, with its rules for them, what a
declared bound may rest on, with the int8 circuits whose window needs nothing else, and the model as a core circuit
(`IntDag.toCircuit`). -/

@[expose] public section
