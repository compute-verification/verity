"""`verity.definitions.silicon.fp4`, under its Flock name: the same module object, so `boolean_export`'s in-place tracing patches reach
every caller."""
import sys

from verity.definitions.silicon import fp4

sys.modules[__name__] = fp4
