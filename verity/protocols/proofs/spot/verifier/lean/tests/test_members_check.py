"""The v2 executable's ``check`` on a case in the per-member form, as `one_stage.members.check` calls it: the audit's
check, which reads a case in either form and answers its verdict; and the streamed check's two commands, ``plan`` and
``check-stream`` (`Spot.ZkMain`), which answer as ``check`` does. Here on cases its parsers refuse, which need no
file; a case every session accepts, which it accepts, is a whole audit's, which `benchmarks/one_stage` runs and
streams."""
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


def test_the_executable_has_no_command_beside_the_checks_the_plan_and_the_parsers():
    """``refusal``, the command the per-member form had while the check refused it, is no command: the executable
    prints its usage and exits 64; ``id`` and ``members`` stand. ``plan`` and ``check-stream`` are commands only with
    their own count of files: with another, the usage and exit 64."""
    exe = str(PF.v2_executable())
    null = ["/dev/null"] * 3
    for argv in (["refusal", *null], ["plan", *null, "/dev/null"], ["check-stream", *null]):
        p = subprocess.run([exe, *argv], capture_output=True, text=True)
        assert p.returncode == 64 and p.stdout == "" and "usage: " in p.stderr, argv
    assert subprocess.run([exe, "id"], capture_output=True, text=True).stdout.strip() == PF.V2
    got, refused = PF.members_parsed({}, {})
    assert got is None and refused.startswith("params: ")


def test_plan_and_check_stream_answer_as_check_where_the_parsers_refuse(tmp_path):
    """On a case the parsers refuse, ``plan`` names no order: nothing on stdout, ``check``'s reason on stderr, exit 1.
    ``check-stream`` on a list of no blobs prints ``check``'s own line and exits as it does, 1."""
    exe = str(PF.v2_executable())
    null = ["/dev/null"] * 3
    check = subprocess.run([exe, "check", *null], capture_output=True, text=True)
    reason = json.loads(check.stdout)["reason"]
    assert check.returncode == M.EXITS["reject"] and reason.startswith("params: ")
    plan = subprocess.run([exe, "plan", *null], capture_output=True, text=True)
    assert (plan.returncode, plan.stdout, plan.stderr.strip()) == (1, "", f"{PF.V2}: {reason}")
    (tmp_path / "list").write_text("")
    got = subprocess.run([exe, "check-stream", *null, str(tmp_path / "list")], capture_output=True, text=True)
    assert (got.returncode, got.stdout) == (check.returncode, check.stdout)
