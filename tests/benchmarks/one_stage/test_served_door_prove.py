"""`served_door_prove.sh`'s verdicts: the Lean verify of the registered session, of the wrong-out one, and the door's
step each pass only on their own answer (`answered`), and the script exits non-zero on any other."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "benchmarks" / "one_stage" / "served_door_prove.sh"
ACCEPTED, REJECTED = '^VERDICT .*"accepted":true[,}]', '^VERDICT .*"accepted":false[,}]'
#: each step's log, its status, the status and line its answer has, and whether `answered` takes it
CASES = {
    "registered accepts": ('VERDICT {"session":"s","accepted":true,"why":"","zk":true}', 0, 0, ACCEPTED, True),
    "wrong-out rejects": ('VERDICT {"session":"s","accepted":false,"why":"registered","zk":true}', 1, 1, REJECTED, True),
    "wrong-out ends on a traceback": ("Traceback (most recent call last):\nRuntimeError: died", 1, 1, REJECTED, False),
    "wrong-out rejects, then a traceback": (
        'VERDICT {"session":"s","accepted":false,"why":"x","zk":true}\nTraceback (most recent call last):', 1, 1,
        REJECTED, False),
    "wrong-out's inputs refused": ("usage: verity-verify-proof verify", 2, 1, REJECTED, False),
    "wrong-out accepts": ('VERDICT {"session":"s","accepted":true,"why":"","zk":true}', 0, 1, REJECTED, False),
    "registered panics": ('PANIC at Flock.x Flock.X:1:2: oops\nVERDICT {"session":"s","accepted":true}', 0, 0, ACCEPTED,
                          False),
    "registered aborts": ("INTERNAL PANIC: out of memory", 134, 0, ACCEPTED, False),
    "door accepts": ('STEP {"step": "verify", "refusal": null, "findings": []}', 0, 0, '^STEP .*"refusal": null', True),
    "door refuses": ('STEP {"step": "verify", "refusal": "proofs: x", "findings": []}', 0, 0, '^STEP .*"refusal": null',
                     False),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_verify_passes_only_on_its_own_answer_and_a_crash_is_no_reject(tmp_path, case):
    """A negative (the wrong-out session) passes only on verify's documented reject, status 1 with its VERDICT line
    ``"accepted":false``; the registered session only on status 0 with ``"accepted":true``; the door's step only on
    status 0 with its STEP line's ``"refusal": null``. A traceback, a panic line, an abort or another status is none of
    them, and the script says ERROR and exits 1 in place of ending in ``true``."""
    text = SCRIPT.read_text()
    fn = re.search(r"^answered\(\) \{.*?^\}$", text, re.M | re.S)
    assert fn and text.rstrip().endswith("exit $bad") and text.count("answered $O/") == 2, (
        f"{SCRIPT.name} no longer judges both Lean verifies and the door's step by `answered` and exits on them")
    log, status, want, line, ok = CASES[case]
    (tmp_path / "log").write_text(log + "\n")
    r = subprocess.run(["bash", "-c", f'{fn.group(0)}\nanswered "$1" {status} {want} \'{line}\'', "-", str(tmp_path / "log")],
                       check=False, env={**os.environ, "PYTHONPATH": str(REPO)})
    assert (r.returncode == 0) == ok


def test_the_audit_road_places_a_templated_members_unit():
    """The door lowers each block to a templated circuit, whose unit a5 places at staging with flock-tmpl-unit and
    refuses without it (`a0.templated`): the audit's records_audit command passes the binary the script built."""
    text = SCRIPT.read_text().replace("\\\n", " ")
    audit = [line for line in text.splitlines() if "python benchmarks/prove_path/records_audit.py audit" in line]
    assert len(audit) == 1 and "--tmpl-unit-bin $UNIT_BIN " in audit[0], (
        f"{SCRIPT.name}'s audit road gives records_audit.py no --tmpl-unit-bin $UNIT_BIN: every templated member stops at staging")
    assert "[ -x $UNIT_BIN ]" in text
    assert '"--tmpl-unit-bin"' in (REPO / "benchmarks" / "prove_path" / "records_audit.py").read_text()
