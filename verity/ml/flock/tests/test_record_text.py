"""A session record's canonical text (`verity_flock.record_text.session_text`): Lean's `Json.compress` on the shared vectors
(`live/src/record_text_vectors.json`, whose expected texts Lean printed; `live/src/record_text.rs` runs the same cases), a
fixed point of itself, and the agreement harness's stdlib copy (`verifier/record_text.py`) the same file."""
import json
from pathlib import Path

import pytest

from verity_flock import record_text as RT
from verity_flock import rec_live as RL

FLOCK = Path(__file__).resolve().parents[1]
VECTORS = json.loads((FLOCK / "live" / "src" / "record_text_vectors.json").read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", VECTORS, ids=[c["name"] for c in VECTORS])
def test_session_text_is_leans_compress(case):
    assert RT.session_text(json.loads(case["input"])) == case["canonical"]
    assert RT.session_text(json.loads(case["canonical"])) == case["canonical"]


def test_numbers_as_lean_prints_them():
    assert [RT.session_text(x) for x in (-191.0, 1e-05, 1.5e20, 0.53, -0.0, 3)] == \
        ["-191", "0.00001", "150000000000000000000", "0.53", "0", "3"]


def test_refuses_what_is_not_json():
    for bad in (float("nan"), float("inf"), {1: 2}, object()):
        with pytest.raises(ValueError):
            RT.session_text(bad)


def test_the_harness_copy_is_the_same_file():
    assert (FLOCK / "verifier" / "record_text.py").read_bytes() == (FLOCK / "python" / "verity_flock" / "record_text.py").read_bytes()


def test_a_record_in_it():
    rec = {"kind": RL.SESSION_KIND, "verdict": {"secs": 1.0}, "streams": [{"b": 1, "a": "x\ty"}]}
    assert RL.session_text is RT.session_text
    assert RT.session_text(rec) == '{"kind":"flock-live-session/v1","streams":[{"a":"x\\u0009y","b":1}],"verdict":{"secs":1}}'
