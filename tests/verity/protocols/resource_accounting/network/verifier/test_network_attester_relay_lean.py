"""The relay check's `lean` (`relay_check.lean_rows`): each of its outcomes on a network attester run at the scan's
w63k4 schedule, ``network-rows`` a stand-in that answers as told (its real verdicts are the agreement test's, in the
suite of the network checks' Lean): accepted, refused with its reason, not covered (a schedule with a
`max_tokens_cap`, the bytes egress form) and absent, none of the last two run or read as accepted, and a verdict that
isn't the Python check's the problem ``lean-disagrees``."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

from tests.verity.protocols.resource_accounting.network.attester.relay_runs import _run
from verity.protocols.resource_accounting.network.attester import network_attester_serve as ws
from verity.protocols.resource_accounting.network.verifier import relay_check as RC
from verity.protocols.resource_accounting.network.verifier.network_attester import W63K4, NetworkAttesterParams

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


def stand_in(d: Path, verdict: str | None, reason: str | None = None) -> Path:
    """A ``network-rows`` that answers ``verdict`` (with ``reason``), and when ``verdict`` is None fails if run."""
    d.mkdir(parents=True, exist_ok=True)
    exe, answer = d / "network-rows", d / "answer.json"
    answer.write_text(json.dumps({"protocol": RC.ROWS, "verdict": verdict, "reason": reason}) + "\n")
    exe.write_text("#!/bin/sh\nexit 9\n" if verdict is None else f"#!/bin/sh\ncat '{answer}'\n")
    exe.chmod(0o755)
    return exe


def test_network_rows_accepting_a_run_the_python_check_accepts_is_accepted(tmp_path):
    result = RC.check(served(tmp_path / "run"), rows_exe=stand_in(tmp_path / "accept", "accept"))
    assert (result["accepted"], result["lean"]) == (True, "accepted"), result["problems"]


def test_network_rows_refusing_a_run_the_python_check_refuses_is_refused_with_its_reason(tmp_path):
    run = forged(served(tmp_path / "honest"), tmp_path / "forged")
    why = "requests window 0: the certificate isn't the key's in the session"
    result = RC.check(run, rows_exe=stand_in(tmp_path / "reject", "reject", why))
    assert (result["accepted"], result["lean"]) == (False, f"refused: {why}")
    assert UNCERTIFIED in result["problems"] and f"network-rows: {why}" in result["problems"]
    assert not any(p.startswith("lean-disagrees") for p in result["problems"])


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
