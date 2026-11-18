"""`restrict.ports`, in(S) and out(S) by a walk of C's gates, is `boundary.in_gates` and `boundary.out_gates` read from the
description, on every block of `test_restrict`'s fixtures: its units as one root-scope part."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from verity.circuits import boundary as B
from verity.circuits import partition_object as PO
from verity.circuits import restrict as R
from verity.circuits.parts import Part

_SPEC = importlib.util.spec_from_file_location("restrict_fixtures", Path(__file__).parent / "test_restrict.py")
_T = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_T)


@pytest.mark.parametrize("name,units", _T._cases())
def test_the_walked_ports_are_the_compact_boundary(name, units):
    program, _c, bits, partition, b = _T._setup(name, units)
    tp = PO.TemplatePartition(program, partition["query"])
    part = Part(program.circuit.root_scope, frozenset(tp.locate(u) for lo, hi in units for u in range(lo, hi)))
    ins, outs = B.in_gates(part), B.out_gates(part)
    assert isinstance(ins, frozenset) and isinstance(outs, frozenset), (ins, outs)
    assert R.ports(bits, b.words) == (sorted(ins), sorted(outs))
