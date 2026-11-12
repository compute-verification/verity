"""What a negative may count as Lean refusing a proof (`a0.lean_refused`): a verdict of Lean's own on a proof it read. A
refusal at setup, or for want of a file of the session (`a0.NO_SESSION_FILES`: the prover sent no proof, or the ZK server
held it back), is a verdict on no proof; `a0.lean_read_no_proof` names the second, for a negative that passes on it by
name. The verdicts are the shapes `a0.lean_verify` returns."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a0

DIGEST = "5e" * 64
MISSING = "session files: no such file or directory (error code: 2)\n  file: /s/circuit-00.rep0.bin"

#: name: (the verdict, `lean_refused`, `lean_read_no_proof`)
VERDICTS = {
    "the proof fails a check": ({"accepted": False, "why": "lincheck: ConsistencyFailed (sumcheck-final)", "digest": DIGEST},
                                True, False),
    "an input is missing at setup": ({"accepted": False, "why": "setup: the public file is not in the archive",
                                      "digest": DIGEST}, False, False),
    "no proof file of the session": ({"accepted": False, "why": MISSING, "digest": DIGEST}, False, True),
    "the session names no rep file": ({"accepted": False, "why": "session files: the session names no rep file circuit/rep0",
                                       "digest": DIGEST}, False, True),
    "a reason that only mentions them": ({"accepted": False, "why": "transcript: a digest of none of the session files: rep0",
                                          "digest": DIGEST}, True, False),
    "a0's stand-in for no verdict": ({"accepted": False, "why": "no verdict (rc 1): session files: gone", "digest": None},
                                     False, False),
    "no session recorded": ({"accepted": False, "of_record": True, "why": "no session recorded"}, False, False),
    "refused with no reason": ({"accepted": False, "why": None, "digest": DIGEST}, False, False),
    "accepted": ({"accepted": True, "why": None, "digest": DIGEST}, False, False),
}


@pytest.mark.parametrize("name", VERDICTS)
def test_a_refusal_counts_only_where_lean_read_a_proof(name):
    v, refused, no_proof = VERDICTS[name]
    assert (a0.lean_refused(v), a0.lean_read_no_proof(v)) == (refused, no_proof), name


@pytest.mark.parametrize("name", VERDICTS)
def test_a_negative_that_takes_either_refusal_keeps_the_verdict_it_had(name):
    """A11's tampered member passes on either (its prover's self-check sends no proof): the two together are every
    refusal of Lean's own outside setup, the rule before a refusal for want of the session's files stopped counting."""
    v = VERDICTS[name][0]
    why = v.get("why") or ""
    before = v.get("accepted") is False and v.get("digest") is not None and bool(why) and not why.startswith("setup:")
    assert (a0.lean_refused(v) or a0.lean_read_no_proof(v)) == before, name


def test_a_named_check_is_held_to_its_name_wherever_lean_makes_it():
    """A negative that names its check (`names`) is refused by that check alone, at setup too, and by no other refusal."""
    at_setup = {"accepted": False, "why": "setup: the tables do not hash to the header's", "digest": DIGEST}
    assert a0.lean_refused(at_setup, names="do not hash to the header's") and not a0.lean_refused(at_setup)
    missing = {"accepted": False, "why": MISSING, "digest": DIGEST}
    assert not a0.lean_refused(missing, names="partition") and not a0.lean_read_no_proof(at_setup)
