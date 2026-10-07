"""`verity.primitives.circuits.boolean.forms`, under its C-Flock name: the same module object, so `boolean_export`'s in-place tracing patches reach
every caller."""
import sys

from verity.primitives.circuits.boolean import forms

sys.modules[__name__] = forms
