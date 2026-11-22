"""The relay check's `lean` (`relay_check.lean_rows`): each of its outcomes on a network attester run at the scan's
w63k4 schedule, ``network-rows`` a stand-in that answers as told (its real verdicts are the agreement test's, in the
suite of the network checks' Lean): accepted, refused with its reason, not covered (a schedule with a
`max_tokens_cap`, the bytes egress form, a deployment not of the key and session the check was handed) and absent,
none of the last two run or read as accepted, and a verdict that isn't the Python check's the problem
``lean-disagrees``."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from tests.verity.protocols.accounting.network.attester.relay_runs import _run
from verity.protocols.proof.spot.verifier.one_stage import lean_child as LC
from verity.protocols.accounting.network.attester import network_attester_serve as ws
from verity.protocols.accounting.network.verifier import connection_rows_check as CRC, lean_audit as LA
from verity.protocols.accounting.network.verifier import record_files as RF
from verity.protocols.accounting.network.verifier import relay_check as RC
from verity.protocols.accounting.network.verifier.network_attester import W63K4, NetworkAttesterParams

UNCERTIFIED = "requests window 0 isn't certified"


def served(out: Path, params: NetworkAttesterParams = W63K4) -> Path:
    out.mkdir(parents=True)
    requests = [ws.id_request(model="m", user=f"user-{j}", prompt=[j + 1], max_tokens=2,
                              frame_bytes=params.cell_bytes, offset=0) for j in range(params.lanes)]
    _run(out, out.parent / f"{out.name}-key", requests, [ws.id_response([1, 2], params.payload)] * params.lanes,
         params=params, id_payload=params.payload)
    return out


def forged(honest: Path, out: Path) -> Path:
    """The honest run with its first certificate's signature changed."""
    shutil.copytree(honest, out)
    lines = [json.loads(x) for x in (out / "certified.jsonl").read_text().splitlines()]
    sig = lines[0]["certificate"]["signature"]
    lines[0]["certificate"]["signature"] = sig[:-1] + ("0" if sig[-1] != "0" else "1")
    (out / "certified.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return out


def stand_in(d: Path, verdict: str | None, reason: str | None = None, before: str = "") -> Path:
    """A ``network-rows`` that runs the shell lines ``before``, then answers ``verdict`` (with ``reason``) where it was
    started with ``LEAN_ABORT_ON_PANIC=1``; when ``verdict`` is None it fails if run."""
    d.mkdir(parents=True, exist_ok=True)
    exe, answer = d / "network-rows", d / "answer.json"
    answer.write_text(json.dumps({"protocol": RC.ROWS, "verdict": verdict, "reason": reason}) + "\n")
    exe.write_text("#!/bin/sh\nexit 9\n" if verdict is None else
                   f"#!/bin/sh\n[ \"$LEAN_ABORT_ON_PANIC\" = 1 ] || exit 9\n{before}cat '{answer}'\n")
    exe.chmod(0o755)
    return exe


def server_stand_in(d: Path, before: str = "") -> Path:
    """A ``network-records`` whose ``verdicts`` server, started with ``LEAN_ABORT_ON_PANIC=1``, runs ``before`` on each
    request and then answers every execution phase accepted."""
    d.mkdir(parents=True, exist_ok=True)
    exe = d / "network-records"
    accept = json.dumps({"output": {"link_windows": [{"direction": "egress", "link": "a", "verdict": "accept",
                                                      "window": 0}]}})
    exe.write_text(f"#!/bin/sh\n[ \"$LEAN_ABORT_ON_PANIC\" = 1 ] || exit 9\nwhile read -r line; do\n{before}"
                   f"echo '{accept}'\ndone\n")
    exe.chmod(0o755)
    return exe


#: a stand-in's lines that write a panic line Lean's runtime writes with no ``PANIC``, one that carries it, and a kill
DEATHS = {"index": "echo 'Error: index out of bounds' >&2\n",
          "panic": "echo 'PANIC at NetworkCheck.Rows.audit NetworkCheck.Rows:1:2: unreachable' >&2\n",
          "signal": "kill -9 $$\n"}
REFUSED = {"index": (LC.LeanPanic, "lean-panic: .*: Error: index out of bounds"),
           "panic": (LC.LeanPanic, "lean-panic: .*: PANIC at NetworkCheck.Rows.audit"),
           "signal": (LC.LeanExit, "lean-exit: .*killed by SIGKILL")}


@pytest.mark.parametrize("death", sorted(DEATHS))
def test_a_network_rows_that_panics_or_is_killed_is_refused_by_name_whatever_it_answered(tmp_path, death):
    exe = stand_in(tmp_path / death, "accept", before=DEATHS[death])
    kind, why = REFUSED[death]
    with pytest.raises(kind, match=why):
        RC.check(served(tmp_path / "run"), rows_exe=exe)


@pytest.mark.parametrize("death", sorted(DEATHS))
def test_a_verdicts_server_that_panics_or_is_killed_is_refused_by_name_whatever_it_answered(tmp_path, death):
    assert LA.Server(server_stand_in(tmp_path / "honest")).verdicts({})[0]["verdict"] == "accept"
    server = LA.Server(server_stand_in(tmp_path / death, DEATHS[death]))
    kind, why = REFUSED[death]
    with pytest.raises(kind, match=why):
        server.verdicts({})


def test_network_rows_accepting_a_run_the_python_check_accepts_is_accepted(tmp_path):
    result = RC.check(served(tmp_path / "run"), rows_exe=stand_in(tmp_path / "accept", "accept"))
    assert (result["accepted"], result["lean"]) == (True, "accepted"), result["problems"]


def test_network_rows_refusing_a_run_the_python_check_refuses_is_refused_with_its_reason(tmp_path):
    run = forged(served(tmp_path / "honest"), tmp_path / "forged")
    why = "requests window 0: the certificate isn't the key's in the session"
    result = RC.check(run, rows_exe=stand_in(tmp_path / "reject", "reject", why))
    assert (result["accepted"], result["lean"]) == (False, f"refused: {why}")
    assert UNCERTIFIED in result["problems"] and not any(p.startswith("lean-disagrees") for p in result["problems"])


def test_a_capped_or_bytes_form_schedule_is_not_covered_and_network_rows_isnt_run(tmp_path):
    never = stand_in(tmp_path / "never", None)
    for params, why in ((replace(W63K4, max_tokens_cap=4), "not-covered: the schedule has a max_tokens_cap, and "
                         "network-rows holds no response to its request's max_tokens"),
                        (replace(W63K4, egress="bytes"), "not-covered: the rows audit's schedule is the id egress "
                         "form's, not 'bytes''s")):
        result = RC.check(served(tmp_path / params.egress / str(params.max_tokens_cap), params), rows_exe=never)
        assert (result["accepted"], result["lean"]) == (True, why), result["problems"]


def test_network_rows_absent_decides_nothing_and_says_so(tmp_path):
    exe = tmp_path / "no-such" / "network-rows"
    result = RC.check(served(tmp_path / "run"), rows_exe=exe)
    assert (result["accepted"], result["lean"]) == (True, f"absent: network-rows is not built on this device (no {exe})")
    assert json.loads(json.dumps(result))["lean"] != "accepted"


def test_a_verdict_that_isnt_the_python_checks_is_lean_disagrees_with_both(tmp_path):
    honest = served(tmp_path / "honest")
    result = RC.check(honest, rows_exe=stand_in(tmp_path / "reject", "reject", "a reason"))
    assert (result["accepted"], result["lean"]) == (False, "refused: a reason")
    assert result["problems"] == ["lean-disagrees: network-rows refused: a reason; the Python check accepts"]
    result = RC.check(forged(honest, tmp_path / "forged"), rows_exe=stand_in(tmp_path / "accept", "accept"))
    assert (result["accepted"], result["lean"]) == (False, "accepted")
    assert f"lean-disagrees: network-rows accepted; the Python check refuses ({UNCERTIFIED})" in result["problems"]


def test_a_run_not_of_the_key_the_check_was_handed_is_refused_with_network_rows_not_run(tmp_path):
    run = served(tmp_path / "run")
    dep = json.loads((run / "deployment.json").read_text())
    own = CRC.key_sha512(RF.read_public(dep["key"])), bytes.fromhex(dep["session"])
    result = RC.check(run, under=own, rows_exe=stand_in(tmp_path / "accept", "accept"))
    assert (result["accepted"], result["lean"]) == (True, "accepted"), result["problems"]
    for under in ((bytes(64), own[1]), (own[0], bytes(32))):
        result = RC.check(run, under=under, rows_exe=stand_in(tmp_path / "never", None))
        assert (result["accepted"], result["lean"]) == (False, RC.UNHELD)
        assert result["problems"][0].startswith("the deployment record's ")
