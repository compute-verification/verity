"""The v2 executable's ``refusal`` (`Spot/MembersRefusal.lean`), as `one_stage.members.refusal` calls it: the audit's
check run on a case read in either form, which answers why the case is refused and never a verdict. Here on cases its
parsers refuse, which need no file; a case every check passes, refused for its form alone (`members.ALL_PASSED`), is a
whole audit's, which `benchmarks/one_stage` runs."""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from verity.protocols.spot.verifier.one_stage import members as M, proofs as PF

#: slow: the first of them builds the executable
pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]


def test_the_refusals_run_answers_a_reason_under_its_own_name_and_exits_1():
    """Parameters naming no proofs, and per-member parameters naming nothing else, are refused under ``params:`` as
    the audit refuses them; the line is exactly ``{decided, refusal, run}`` with the words `REFUSAL`, names no verdict
    and comes with exit 1."""
    why, decided = M.refusal({}, b"", b"")
    assert decided and why == ("params: the parameters name no proofs (the auditor's circuit and public files' digests "
                               "and the verifier's registered reads)")
    why, decided = M.refusal({"proofs": {"members": {}}}, b"", b"")
    assert decided and why.startswith("params: ") and why != M.ALL_PASSED
    p = subprocess.run([str(PF.v2_executable()), "refusal", "/dev/null", "/dev/null", "/dev/null"], capture_output=True,
                       text=True)
    line = json.loads(p.stdout)
    assert p.returncode == 1 and sorted(line) == ["decided", "refusal", "run"] and line["run"] == M.REFUSAL
    assert line["refusal"].startswith("params: ")
    assert not {"verdict", "accept", "accepted", "public", "profile"} & set(line)


def test_the_refusals_usage_says_what_it_is_and_the_other_commands_stand():
    """``refusal`` with fewer than its three files prints its usage, which says it is no verdict, and exits 64; the
    executable's other commands are `membersMain`'s as before."""
    exe = str(PF.v2_executable())
    p = subprocess.run([exe, "refusal", "/dev/null"], capture_output=True, text=True)
    assert p.returncode == 64 and p.stdout == "" and M.REFUSAL in p.stderr and "usage: refusal PARAMS" in p.stderr
    assert subprocess.run([exe, "id"], capture_output=True, text=True).stdout.strip() == PF.V2
    got, refused = PF.members_parsed({}, {})
    assert got is None and refused.startswith("params: ")
