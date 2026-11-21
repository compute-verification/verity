"""`reference.pearl_plus_answer`'s public is Pearl+'s protocol's (`PouwVerifier.PearlPlus.Protocol.publicJson`) byte for
byte: the credit, ``operands`` from the verdict's tie and ``compared_with_registered_weights`` false, held to the JSON the
Lean's own test pins (its ``#guard`` on `publicJson`) without building the Lean."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import reference as R
from verity_difftest import canonical

ROOT = Path(__file__).resolve().parents[6]
PROTOCOL = (ROOT / "verity" / "protocols" / "accounting" / "compute" / "pouw" / "verifier" / "lean" / "pearlplus"
            / "PouwPearlPlus" / "Protocol.lean")
#: `Protocol.lean`'s test credit (`credit`): 3 MACs credited on 1 tile of 4 checked, at ρ 1/400 and ε 1/100
WORK = {"credited": 3, "credited_tiles": 1, "checked_macs": 4, "rho": "1/400", "epsilon": "1/100", "over_cap_draws": 0}
STRING = r'"(?:[^"\\]|\\.)*"'


def _pinned() -> str:
    """The JSON `Protocol.lean`'s ``#guard`` pins for `publicJson` of its test credit, not committed."""
    lean = PROTOCOL.read_text()
    m = re.search(rf"#guard \(publicJson ⟨credit, \.notCommitted⟩\)\.compress ==((?:\s*{STRING}\s*(?:\+\+)?)+)", lean)
    assert m, f"{PROTOCOL.name} pins no publicJson of its test credit"
    return "".join(json.loads(s) for s in re.findall(STRING, m.group(1)))


def _verified(tie: dict | None) -> dict:
    public = {"work": WORK, "passes": [{}], **({"tie": tie} if tie is not None else {})}
    return {"draws": 3, "answer": {"verdict": "accept", "reason": None, "public": public,
                                   "profile": {"delta": "1/100", "tiles": [[0, 0]], "exclusion_tiles": 0}}}


def test_the_references_public_is_the_leans_byte_for_byte():
    untied = R.pearl_plus_answer(b"{}", _verified({"record": "00", "tables": 4, "weights": "weights untied (W v1)"}))
    assert canonical(untied["public"]) == _pinned()
    tied = R.pearl_plus_answer(b"{}", _verified({"record": "00", "tables": 6}))
    assert canonical(tied["public"]) == _pinned().replace('"not-committed"', '"committed-before-coin"')
    lean = PROTOCOL.read_text()
    names = [re.search(rf'def {n} : String := "([^"]*)"', lean).group(1)
             for n in ("OPERANDS_COMMITTED", "OPERANDS_NOT_COMMITTED")]
    assert tuple(names) == R.PEARL_PLUS_OPERANDS


def test_a_verdict_tied_to_no_epoch_record_has_no_answer_of_pearl_pluss_protocol():
    with pytest.raises(ValueError, match="tied to no epoch record"):
        R.pearl_plus_answer(b"{}", _verified(None))
