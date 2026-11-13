"""`verity.circuits.boolean.forms`, under its Flock name: the same module object, so `boolean_export`'s in-place tracing patches reach
every caller."""
import sys

from verity.circuits.boolean import forms

sys.modules[__name__] = forms
