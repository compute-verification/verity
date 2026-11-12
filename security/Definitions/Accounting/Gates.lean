module

public import Definitions.Accounting.Gates.Model
public import Definitions.Accounting.Gates.Registry
public import Definitions.Accounting.Gates.Codec
public import Definitions.Accounting.Gates.Ranges
public import Definitions.Accounting.Gates.Declared
public import Definitions.Accounting.Gates.Circuit

/-! The circuit model (`Accounting.Gates`): a declared circuit's gates (`IntDag`) over bounded integers with their
classes, provenance and committed outputs, its bounded window and the interval analysis that proves it, its census and
the stacking property on it, the registry's primitives each in one class, the word codec with what computing its
words in the model means, the evaluators whose output intervals property/v1 reads, with its rules for them, what a
declared bound may rest on, with the int8 circuits whose window needs nothing else, and the model as a core circuit
(`IntDag.toCircuit`). -/

@[expose] public section
