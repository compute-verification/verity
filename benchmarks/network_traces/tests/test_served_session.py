"""served_session: the plan is the gate's scan rehearsal inside the scan's class (PROMPT_IDS prompt ids in, at most KO
out), which this tree's gate accepts, with the endpoint's sealed flags and its hand start, and a request the gate or
the class refuses is refused with its reason; a session through a stub gate keeps
the session's fetch (with witness/) 0700 under KEEP and publishes only the bundle less witness/ and its outcome, removes
a published file that holds a prompt's ids, and refuses a KEEP inside --out; --at waits on the injected clock and sends
nothing when late; chain runs chain_record.sh's warden path on the kept fetch, and is refused without the leak guard;
on the endpoint's records artifact it reads requests.jsonl rebuilt in KEEP from the artifact's users map and the
certified I/O, and refuses an endpoint with neither or an I/O that doesn't read whole; its coins are the challenger's
(CR_CHALLENGER, CR_MAIN, and the gate's CR_PEER, CR_CHALLENGER_KEY and CR_GATE_KEY), the gate's by default and
``local`` for a dry run, and a challenger it can't name, or a CR_K=V naming the challenger or the scan's coins, is
refused before chain_record.sh runs."""
from __future__ import annotations

import io
import json
import os
import shlex
import sys
import tarfile
from datetime import datetime, UTC
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import served_session as SS  # noqa: E402
import scan_zk as sz  # noqa: E402

RUN = "c20261028T120000Z-0000"

STUB = r'''#!PYTHON
import io, json, os, sys, tarfile
from pathlib import Path
d, cmd = Path(os.environ["STUB_DIR"]), sys.argv[-1]
if cmd.startswith("warden-rehearse "):
    (d / "prompts.json").write_bytes(sys.stdin.buffer.read())
    (d / "request").write_text(cmd)
    print(json.dumps({"run": "RUN", "device": "warden"}))
elif cmd == "status RUN":
    print(json.dumps({"run": "RUN", "end": {"rc": 0}}))
elif cmd == "fetch RUN":
    files = {"end.json": json.dumps({"run": "RUN", "rc": 0, "verdict": {"accepted": True, "problems": 0}}).encode(),
             "out/deployment.json": json.dumps({"windows": 4, "origin_unix": 1.5,
                                                "params": {"window": 32, "bucket_ms": 100}}).encode(),
             "out/certificates.jsonl": b"", "witness/prompts.json": (d / "prompts.json").read_bytes()}
    if (d / "leak").exists():
        files["out/cells.jsonl"] = (d / "prompts.json").read_bytes()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo("RUN/" + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    sys.stdout.buffer.write(buf.getvalue())
else:
    sys.exit(f"stub gate: no verb {cmd!r}")
'''.replace("PYTHON", sys.executable).replace("RUN", RUN)


def test_the_plan_is_the_scan_rehearsal_at_the_scans_class_and_the_gate_takes_it():
    assert (SS.KI, SS.KO) == (sz.CLASS.KI, sz.CLASS.KO) and SS.PROMPT_IDS == 256 < SS.KI
    a = SS.parser().parse_args(["plan"])
    a.minutes = SS.A.warden_minutes(a.requests, a.height) + SS.MARGIN
    ps = SS.prompts(a.prompts, a.prompt_ids, a.vocab, b"seed")
    assert len(ps) == 64 and {len(p) for p in ps} == {SS.PROMPT_IDS} and all(0 <= t < SS.VOCAB for p in ps for t in p)
    assert SS.prompts(2, 4, 256, b"seed") != SS.prompts(2, 4, 256, b"other")
    data = json.dumps(ps, separators=(",", ":")).encode()
    doc = SS.plan(a, data)
    assert doc["request"] == ("warden-rehearse --peer vy-dc-h200-2 --port 18782 --minutes 10 --requests 113 --height 12 "
                              "--scan --prompts --tokens 64")
    assert doc["prompts"] == {"count": 64, "ids": 64 * SS.PROMPT_IDS} and doc["prompts_bytes"] <= SS.A.WARDEN_PROMPTS_BYTES
    assert doc["endpoint"].endswith("--port 18782 --protocol ids --id-payload 503 --admit release-order "
                                    "--scheme rows-sha512 --seal --retain-exclude hidden_m1")
    run = shlex.split(doc["runs"]["chain"])
    assert "--store-read" in run[:run.index("--")] and run[run.index("--run-max") + 1] == "14400"
    inner = SS.parser().parse_args(shlex.split(run[-1])[2:])
    assert (inner.cmd, inner.endpoint, inner.eos_ids, inner.draw, inner.challenger, str(inner.challenger_key)) == (
        "chain", "ART", "128001", 8, f"gate:{SS.GATE}", "$RESEARCH_RUN_DIR/inputs/challenger-key.json")
    assert all(w in doc["hand_start"] for w in ("--required-manifest ROW/manifest.json", "not examples/vllm/serve.sh",
                                                 "--disk-floor-gb 40"))
    long = SS.parser().parse_args(["plan", "--prompt-ids", str(SS.KI)])
    long.minutes = a.minutes
    assert SS.plan(long, b"[[1]]")["endpoint"].endswith(f"--max-model-len {SS.KI + SS.KO}")


@pytest.mark.parametrize("flags, why", [
    (["--tokens", "65"], "1 to KO = 64"),
    (["--tokens", "0"], "1 to KO = 64"),
    (["--prompt-ids", "1025"], "1 to KI = 1024"),
    (["--prompts", "200", "--prompt-ids", "1024"], "the gate refuses it: --prompts: over 1048576 bytes"),
    (["--requests", "100000", "--minutes", "60"], "the gate refuses it: --requests '100000'"),
    (["--minutes", "5"], "the gate refuses it: --minutes 5"),
])
def test_a_request_the_gate_or_the_class_refuses_is_refused_with_its_reason(flags, why, capsys):
    assert SS.main(["plan", *flags]) == 2
    assert why in capsys.readouterr().err


def _session(tmp_path, monkeypatch, *flags):
    (tmp_path / "bin").mkdir(exist_ok=True)
    (tmp_path / "bin" / "ssh").write_text(STUB)
    (tmp_path / "bin" / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.defpath}")
    monkeypatch.setenv("STUB_DIR", str(tmp_path))
    (tmp_path / "out").mkdir(exist_ok=True)
    return SS.main(["session", "--out", str(tmp_path / "out"), "--keep", str(tmp_path / "keep"),
                    "--key", str(tmp_path / "key"), "--prompts", "3", *flags])


def test_a_session_keeps_the_lanes_fetch_apart_and_publishes_the_bundle_less_witness(tmp_path, monkeypatch):
    assert _session(tmp_path, monkeypatch) == 0
    assert (tmp_path / "request").read_text().endswith("--scan --prompts --tokens 64")
    prompts = (tmp_path / "prompts.json").read_bytes()
    assert [len(p) for p in json.loads(prompts)] == [SS.PROMPT_IDS] * 3
    keep, kept = tmp_path / "keep", tmp_path / "keep" / RUN / "fetch.tar.gz"
    assert [p.stat().st_mode & 0o777 for p in (keep, keep / RUN, kept)] == [0o700, 0o700, 0o600]
    assert f"{RUN}/witness/prompts.json" in SS.members(kept.read_bytes())
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == ["bundle.tgz", "session.json", "start.json", "status.json"]
    shown = SS.members((out / "bundle.tgz").read_bytes())
    assert sorted(shown) == [f"{RUN}/end.json", f"{RUN}/out/certificates.jsonl", f"{RUN}/out/deployment.json"]
    doc = json.loads((out / "session.json").read_text())
    assert (doc["run"], doc["rc"], doc["verdict"]["accepted"], doc["windows"], doc["session_s"]) == (RUN, 0, True, 4,
                                                                                                     12.8)
    assert doc["prompts"] == {"count": 3, "ids": 3 * SS.PROMPT_IDS} and doc["keep"] == str(kept)
    assert not any(n in p.read_bytes() for p in out.iterdir() for n in SS.needles(json.loads(prompts)))


def test_a_published_file_holding_a_prompt_is_removed_and_the_session_exits_4(tmp_path, monkeypatch):
    (tmp_path / "leak").touch()
    assert _session(tmp_path, monkeypatch) == 4
    assert not (tmp_path / "out" / "bundle.tgz").exists() and (tmp_path / "out" / "session.json").exists()


def test_a_keep_inside_out_is_refused_before_anything_is_sent(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", os.defpath)
    (tmp_path / "out").mkdir()
    assert SS.main(["session", "--out", str(tmp_path / "out"), "--keep", str(tmp_path / "out" / "keep")]) == 2
    assert list((tmp_path / "out").iterdir()) == []


def test_at_waits_on_the_clock_and_a_late_launch_sends_nothing():
    at = "2026-10-28T12:00:00Z"
    t = datetime(2026, 10, 28, 12, tzinfo=UTC).timestamp()
    slept = []
    SS.wait_until(at, now=lambda: t - 30, sleep=slept.append)
    SS.wait_until(at, now=lambda: t + SS.LATE_S, sleep=slept.append)
    assert slept == [30]
    with pytest.raises(SS.Late, match="nothing sent"):
        SS.wait_until(at, now=lambda: t + SS.LATE_S + 1, sleep=slept.append)


def test_public_is_the_bundle_less_witness():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in ("r/end.json", "r/out/cells.jsonl", "r/witness/check.json", "r/witness/x/salts.bin"):
            info = tarfile.TarInfo(name)
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    assert sorted(SS.members(SS.public(buf.getvalue()))) == ["r/end.json", "r/out/cells.jsonl"]


def test_chain_runs_chain_records_warden_path_on_the_kept_fetch_and_needs_the_leak_guard(tmp_path, monkeypatch):
    here = tmp_path / "here"
    here.mkdir()
    (here / "chain_record.sh").write_text('env | grep "^CR_" | sort > "$SEEN"\n')
    monkeypatch.setattr(SS, "HERE", here)
    monkeypatch.setenv("SEEN", str(tmp_path / "seen"))
    keep = tmp_path / "keep"
    (keep / RUN / "endpoint").mkdir(parents=True)
    (keep / RUN / "endpoint" / "requests.jsonl").write_text("")
    (keep / RUN / "fetch.tar.gz").write_bytes(b"")
    (tmp_path / "main.txt").write_text("abc\n")
    argv = ["chain", "--run", RUN, "--keep", str(keep), "--eos-ids", "256", "--challenger", "local", "--main",
            str(tmp_path / "main.txt"), "CR_N2_DRAW=4"]
    assert SS.main(argv) == 2
    (here / "leak_guard.py").write_text("")
    assert SS.main(argv) == 0
    seen = dict(line.split("=", 1) for line in (tmp_path / "seen").read_text().splitlines())
    assert seen == {"CR_PATH": "warden", "CR_GATES": str(keep / RUN / "fetch.tar.gz"), "CR_MEM_GB": "96",
                    "CR_N2_DRAW": "4", "CR_TIE_WINDOWS": "8", "CR_ENDPOINT": str(keep / RUN / "endpoint"),
                    "CR_EOS_IDS": "256", "CR_CHALLENGER": "local", "CR_MAIN": str(tmp_path / "main.txt")}
    assert SS.main(["chain", "--run", RUN, "--keep", str(keep)]) == 2
    assert SS.main([*argv, "N2_DRAW=4"]) == 2
    assert SS.main([*argv, "--draw", "8"]) == 0
    assert "CR_SCAN_ARGS=--draw 8" in (tmp_path / "seen").read_text().splitlines()


def test_chain_on_vllms_art_reads_requests_jsonl_rebuilt_in_keep_from_its_users_map(tmp_path, monkeypatch, capsys):
    here, art, keep = tmp_path / "here", tmp_path / "art" / "serve", tmp_path / "keep"
    for d in (here, art, keep / RUN, tmp_path / "bin"):
        d.mkdir(parents=True)
    (here / "chain_record.sh").write_text('env | grep "^CR_ENDPOINT=" > "$SEEN"\n')
    (here / "leak_guard.py").write_text("")
    (keep / RUN / "fetch.tar.gz").write_bytes(b"")
    for f in ("advice_trace.jsonl", "windows.jsonl", "serve.json"):
        (art / f).write_text("{}\n")
    users = [json.dumps({"engine_id": f"e{j}", "user": f"user-{j}"}) + "\n" for j in range(3)]
    (art / SS.USERS).write_text("".join(users))
    (tmp_path / "bin" / "research").write_text(f'#!/bin/sh\n[ "$*" = "data fetch art:ep --to $5" ] && cp -a {art.parent}/. "$5/"\n')
    (tmp_path / "bin" / "research").chmod(0o755)
    monkeypatch.setattr(SS, "HERE", here)
    monkeypatch.setenv("SEEN", str(tmp_path / "seen"))
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.defpath}")
    io = {"accepted": True, "violations": [], "records": [
        {"user": "user-0", "prompt": [5, 6], "max_tokens": 4, "output": [9, 128001]},
        {"user": "user-1", "prompt": [7], "max_tokens": 4, "output": [1, 2]},
        {"user": "user-2", "prompt": [8], "max_tokens": 4, "output": [1, 2, 3, 4]}]}
    monkeypatch.setattr(SS, "certified_io", lambda fetch: io)
    (tmp_path / "main.txt").write_text("abc\n")
    argv = ["chain", "--run", RUN, "--keep", str(keep), "--challenger", "local", "--main", str(tmp_path / "main.txt"),
            "--endpoint", "art:ep", "--eos-ids", "128001"]
    assert SS.main(argv) == 0
    dest = keep / RUN / "endpoint-rebuilt"
    assert (tmp_path / "seen").read_text().strip() == f"CR_ENDPOINT={dest}" and dest.stat().st_mode & 0o777 == 0o700
    got = [json.loads(x) for x in (dest / "serve" / "requests.jsonl").read_text().splitlines()]
    assert [(r["engine_id"], r["user"], r["prompt_ids"], r["streamed_ids"], r["finish"]) for r in got] == [
        ("e0", "user-0", [5, 6], [9, 128001], {"reason": "stop", "stop_reason": None}),
        ("e1", "user-1", [7], [1, 2], {"reason": "stop", "stop_reason": None}),
        ("e2", "user-2", [8], [1, 2, 3, 4], {"reason": "length", "stop_reason": None})]
    assert not (art / "requests.jsonl").exists()
    for bad in ({"engine_id": "e3", "user": None}, {"engine_id": "e3", "user": "user-1"}):
        (art / SS.USERS).write_text("".join(users) + json.dumps(bad) + "\n")
        assert SS.main(argv) == 2
        assert f"engine request e3's user {bad['user']!r} is no certified request left" in capsys.readouterr().err
    io["accepted"] = False
    assert SS.main(argv) == 2 and "doesn't read whole off its cells" in capsys.readouterr().err
    (art / SS.USERS).unlink()
    assert SS.main([*argv[:-4], "--endpoint", str(art), "--eos-ids", "128001"]) == 2
    assert f"no {SS.USERS} beside an advice_trace.jsonl" in capsys.readouterr().err
    assert SS.main([*argv[:-4], "--endpoint", "art:other", "--eos-ids", "128001"]) == 2
    assert "only when launched with --store-read" in capsys.readouterr().err


def _gate_chain(tmp_path, monkeypatch):
    here = tmp_path / "here"
    here.mkdir(exist_ok=True)
    (here / "chain_record.sh").write_text('for k in CR_SCAN_ARGS CR_CHALLENGER CR_MAIN CR_PEER CR_CHALLENGER_KEY '
                                          'CR_GATE_KEY; do if [ -n "${!k+x}" ]; then printf "%s=%s\\n" "$k" "${!k}"; fi; '
                                          'done > "$SEEN"\n')
    (here / "leak_guard.py").write_text("")
    monkeypatch.setattr(SS, "HERE", here)
    monkeypatch.setenv("SEEN", str(tmp_path / "seen"))
    (tmp_path / "challenger-key.json").write_text("{}")
    (tmp_path / "gate-key").write_text("")
    (tmp_path / "main.txt").write_text("abc\n")
    (tmp_path / "keep" / RUN).mkdir(parents=True, exist_ok=True)
    (tmp_path / "keep" / RUN / "fetch.tar.gz").write_bytes(b"")
    return ["chain", "--run", RUN, "--keep", str(tmp_path / "keep"), "--draw", "8", "--key", str(tmp_path / "gate-key"),
            "--main", str(tmp_path / "main.txt"), "--challenger-key", str(tmp_path / "challenger-key.json")]


def test_the_chains_coins_are_the_gates_by_default_and_nothing_falls_back_to_the_verifiers_own(tmp_path, monkeypatch,
                                                                                               capsys):
    argv = _gate_chain(tmp_path, monkeypatch)

    def seen() -> dict:
        got = dict(line.split("=", 1) for line in (tmp_path / "seen").read_text().splitlines())
        (tmp_path / "seen").unlink()
        return got

    assert SS.main(argv) == 0
    assert seen() == {"CR_SCAN_ARGS": "--draw 8", "CR_CHALLENGER": f"gate:{SS.GATE}",
                      "CR_MAIN": str(tmp_path / "main.txt"), "CR_PEER": SS.AUDITOR,
                      "CR_CHALLENGER_KEY": str(tmp_path / "challenger-key.json"),
                      "CR_GATE_KEY": str(tmp_path / "gate-key")}
    assert SS.main([*argv, "--coin-peer", SS.PEER]) == 0 and seen()["CR_PEER"] == SS.PEER
    assert SS.main([*argv, "--challenger", "local"]) == 0
    assert seen() == {"CR_SCAN_ARGS": "--draw 8", "CR_CHALLENGER": "local", "CR_MAIN": str(tmp_path / "main.txt")}
    for flags, why in [
        (argv[:-2], "needs --challenger-key"),
        ([*argv, "--challenger", "verifier"], "gate:USER@HOST (the auditor's, on the network certifier's machine) or "
                                              "local (a dry run's)"),
        ([*argv, "--key", str(tmp_path / "no-key")], "no gate key"),
        ([*argv, "--main", str(tmp_path / "no-main")], "no --main"),
        ([*argv, "CR_CHALLENGER=local"], "CR_CHALLENGER: the chain's challenger is --challenger's"),
        ([*argv, "CR_PEER=vy-dc-h200-2"], "CR_PEER: the chain's challenger is --challenger's"),
        ([*argv, "CR_SCAN_ARGS=--peer vy-dc-h200-2"], "CR_SCAN_ARGS: the scan's draw and its coins"),
        ([*argv, "CR_SCAN_CMD=python3 scan_zk.py --challenger local"], "CR_SCAN_CMD: the scan's draw and its coins"),
    ]:
        assert SS.main(flags) == 2, flags
        assert why in capsys.readouterr().err, flags
    assert not (tmp_path / "seen").exists()
