"""The challenger's salt check of record (`core_check.salt_of_record`): the Lean's, equal to its Python twin's on
the salt difftest's cases, the Lean's alone where they differ, and a refusal where the Lean gives no answer."""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.protocols.accounting.compute.pouw.verifier import core_check as CC
from verity.protocols.accounting.compute.pouw.verifier import verifier as V

PKG = Path(__file__).resolve().parent / "lean" / "check"
for _name in ("pouw_rounds_cases", "pouw_credit_cases"):
    if _name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(_name, PKG / f"{_name}.py")
        sys.modules[_name] = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(sys.modules[_name])
rc = sys.modules["pouw_rounds_cases"]


@pytest.fixture
def asked(monkeypatch):
    """`verifier.lean` as credit and the salt check ask it: ``fns``, each function asked in order; ``say[fn]``, an answer
    to `fn` in place of the Lean's."""
    lean, out = V.lean, SimpleNamespace(fns=[], say={})

    def spy(fn, value):
        out.fns.append(fn)
        return out.say[fn](value) if fn in out.say else lean(fn, value)

    monkeypatch.setattr(V, "lean", spy)
    return out


@pytest.mark.parametrize("name", ["honest", *rc.SALT_NAMES])
def test_the_salt_check_of_record_is_the_leans_alone(name, asked):
    case = rc.salt_instance(random.Random(f"{name}/0"), name)
    python = CC.salt_twin(*rc.salt_args(case))
    assert CC.salt_of_record(*rc.salt_args(case)) == python and asked.fns == ["salt"]
    assert python is None or name != "honest"
    said = None if python is not None else {"refused": ["salt", "stub"]}
    asked.say["salt"] = lambda _v: said
    assert CC.salt_of_record(*rc.salt_args(case)) == (None if said is None else ("salt", "stub"))

    def no_answer(_v):
        raise V.LeanError("lake is not on the PATH")

    asked.say["salt"] = no_answer
    refused = CC.salt_of_record(*rc.salt_args(case))
    assert refused == (V.NO_ANSWER, "the Lean verifier gave no answer to salt, so it stands refused: lake is not on "
                                    "the PATH"), refused

