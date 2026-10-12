"""`verity.ml.circuits.silicon.fp4`, under its C-Flock name: the same module object, so `boolean_export`'s in-place tracing patches reach
every caller."""
import sys

from verity.ml.circuits.silicon import fp4

sys.modules[__name__] = fp4
