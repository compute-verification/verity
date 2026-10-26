import Definitions.Accounting.Gates.Model
import Definitions.Accounting.Gates.Registry
import Definitions.Accounting.Gates.Codec
import Definitions.Accounting.Gates.Ranges
import Definitions.Accounting.Gates.Declared
import Definitions.Accounting.Gates.Cut
import Definitions.Accounting.Gates.Table

/-! The circuit model (`Accounting.Gates`): a declared circuit's gates (`Dag`) over bounded integers with their
classes, provenance and committed outputs, its bounded window and the interval analysis that proves it, its census and
the stacking property on it, the registry's primitives each in one class, the word codec with what computing its
words in the model means, the evaluators whose output intervals property/v1 reads, with its rules for them, and what a
declared bound may rest on, with the int8 circuits whose window needs nothing else, a circuit's cut at a set of its
gates, the min-cut bound's circuit, and the table circuit, a row of registered words picked by a token. -/
