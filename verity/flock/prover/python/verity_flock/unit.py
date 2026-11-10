"""`verity.definitions.silicon.tc_step`, under its Flock name: the same module object, so `boolean_export`'s in-place tracing patches reach
every caller."""
import sys

from verity.definitions.silicon import tc_step

sys.modules[__name__] = tc_step
