"""One-stage v1's case for a Program alone, with a stand-in record: what a check that reads only the Program and its
division needs of a window's records audit, before any window is served.

`v1_case` writes the case as a5 writes a window's (`replay.Case.case`'s params, a v2 record, its evidence), over one
stand-in port and one stand-in registered value: its record names the Program and nothing of a served window. The
suite's cases start from it (`tests/test_served_cases.py`), and `rehearsal_limits.py` asks the inference-only policy's
Lean check of a window class's own Program through it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from verity.circuits import codec
from verity.protocols.proofs.spot.verifier import service as S
from verity.protocols.proofs.spot.verifier.one_stage import partition as P
from verity.protocols.proofs.spot.verifier.one_stage import registration as R

HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("one_stage_replay", HERE / "replay.py")
RP = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = RP
_spec.loader.exec_module(RP)


def v1_case(program, query, previous: str | None = None) -> dict:
    """v1's case as a5 writes it (`replay.Case.case`'s params, a v2 record, its evidence) for ``program``."""
    part = P.partition(program, query)
    ports = {"x": {"leaves": 8, "domain": "02" * 64}}
    law = {"law": "stratified", "k": 1}
    rec = R.record(program, query, law, S.SCHEME, [{"port": "x", "root": "01" * 64, **ports["x"]}],
                   {"kind": "served", "run": "r-served-cases", "row": 0}, previous=previous,
                   registered=[{"value": "weights/w", "root": "ab" * 64, "leaves": 8, "words": 4, "domain": "cd" * 64}])
    params = RP.params(program, query, law, ports, None, "1/100", partition=part)
    evidence = {"draw": {"units": [0]}, "sessions": ["5e" * 64], "verdicts": {"m0": {"accepted": True}}}
    return {"commitment": codec.canonical_json(rec).hex(), "evidence": codec.canonical_json(evidence).hex(),
            "params": params}
