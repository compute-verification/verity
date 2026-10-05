"""`catalog.silicon.fp`, under its C-Flock name: the same module object, so `boolean_export`'s in-place tracing patches reach
every caller."""
import sys

from catalog.silicon import fp

sys.modules[__name__] = fp
