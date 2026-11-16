"""The v2 executable's ``check`` on a case in the per-member form, as `one_stage.members.check` calls it: the audit's
check, which reads a case in either form and answers its verdict. Here on cases its parsers refuse, which need no
file; a case every session accepts, which it accepts, is a whole audit's, which `benchmarks/one_stage` runs."""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from verity.protocols.proofs.spot.verifier.one_stage import members as M, proofs as PF

#: slow: the first of them builds the executable
pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]


def test_the_check_refuses_what_its_parsers_refuse_in_either_form_and_exits_1():
    """Parameters naming no proofs, and per-member parameters naming nothing else, are rejected under ``params:``;
    the line is the interface's outcome, its verdict ``reject``, with exit 1 and nothing published."""
    assert M.check({}, b"", b"") == (
        "reject", "params: the parameters name no proofs (the auditor's circuit and public files' digests and the "
                  "verifier's registered reads)")
    verdict, why = M.check({"proofs": {"members": {}}}, b"", b"")
    assert verdict == "reject" and why.startswith("params: "), why
    p = subprocess.run([str(PF.v2_executable()), "check", "/dev/null", "/dev/null", "/dev/null"], capture_output=True,
                       text=True)
    line = json.loads(p.stdout)
    assert p.returncode == M.EXITS["reject"] and line["verdict"] == "reject" and line["protocol"] == PF.V2
    assert line["reason"].startswith("params: ") and line["public"] is None and line["profile"] is None


def test_the_executable_has_no_command_beside_the_checks_and_the_parsers():
    """``refusal``, the command the per-member form had while the check refused it, is no command: the executable
    prints its usage and exits 64; ``id`` and ``members`` stand."""
    exe = str(PF.v2_executable())
    p = subprocess.run([exe, "refusal", "/dev/null", "/dev/null", "/dev/null"], capture_output=True, text=True)
    assert p.returncode == 64 and p.stdout == "" and "usage: " in p.stderr
    assert subprocess.run([exe, "id"], capture_output=True, text=True).stdout.strip() == PF.V2
    got, refused = PF.members_parsed({}, {})
    assert got is None and refused.startswith("params: ")
