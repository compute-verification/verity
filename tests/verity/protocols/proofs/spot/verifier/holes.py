"""Tests that wait on a hole of the one circuit: a derivation from C and the partition the served verifier does not make
yet, which refuses by name (`one_stage.one_circuit`, Lean's `Flock.OneCircuit.notDerived`). Such a test is an expected
failure only where it stops at its own hole's refusal: it passes the day the hole is filled, and fails where it stops
anywhere else."""
from __future__ import annotations

import functools
from collections.abc import Callable

import pytest

from verity.protocols.proofs.spot.verifier.one_stage.one_circuit import PREFIX

#: the holes by the name each refuses with: Core's `Circuit.restrict` (C_S; Lean's door), the base argument's lowering
#: of C_S (`lower`), the boundary's `RegisteredCircuit.pos`, `Partition.io` (in(S)'s rows of the block circuit) and
#: in(S)'s row cuts
HOLES = ("restrict", "lower", "pos", "io", "ioCuts")


def refusal(hole: str) -> str:
    """The text every refusal at `hole` holds."""
    if hole not in HOLES:
        raise ValueError(f"{hole!r} is no hole of the one circuit: one of {HOLES}")
    return f"{PREFIX}{hole} :"


def waits_on(*holes: str) -> Callable[[Callable], Callable]:
    """The test, an expected failure where what it raises names one of `holes`' refusals; any other failure is its own."""
    texts = {hole: refusal(hole) for hole in holes}

    def mark(test: Callable) -> Callable:
        @functools.wraps(test)
        def run(*args, **kwargs):
            try:
                return test(*args, **kwargs)
            except Exception as e:
                for hole, text in texts.items():
                    if text in str(e):
                        pytest.xfail(f"waits on the hole {hole}")
                raise
        run.holes = holes
        return run
    return mark
