"""Boolean Definitions of the majority of three bits whose builders read state besides their operands, each of which
`circuit_check.ambient` must refuse (`test_ambient.py`), and `Plain`, which reads none. None is registered."""
from __future__ import annotations

import os
import random
import time

from verity.ml.circuits.boolean.forms import maj
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.ops.boolean import scalar as SB

P = (("a", SB.BIT), ("b", SB.BIT), ("c", SB.BIT))
_LOWERED = [0]


def _plain(C, a, b, c):
    return [maj(C, a[0], b[0], c[0])]


def _counted(C, a, b, c):
    """maj as `forms.maj` writes it on every other call, else as ab + c (a + b): the same bit, other gates."""
    _LOWERED[0] += 1
    x, y, z = a[0], b[0], c[0]
    return [maj(C, x, y, z) if _LOWERED[0] % 2 else C.AND(x, y) ^ C.AND(z, x ^ y)]


def _set_order(C, a, b, c):
    """The operands in the order a set of their names iterates, which the strings' hashes decide."""
    bit = {"a": a[0], "b": b[0], "c": c[0]}
    return [maj(C, *[bit[k] for k in {"a", "b", "c"}])]  # the set's order is what this builder reads


def _environ(C, a, b, c):
    os.environ.get("VERITY_AMBIENT_TEST")
    return _plain(C, a, b, c)


def _clocked(C, a, b, c):
    time.perf_counter()
    return _plain(C, a, b, c)


def _drawn(C, a, b, c):
    random.random()
    return _plain(C, a, b, c)


def _def(name, build):
    return definition(f"AmbientTest{name}", 1, build, P, SB.BIT, register=False)


Plain = _def("Plain", _plain)
Counted = _def("Counted", _counted)
SetOrder = _def("SetOrder", _set_order)
Environ = _def("Environ", _environ)
Clocked = _def("Clocked", _clocked)
Drawn = _def("Drawn", _drawn)
