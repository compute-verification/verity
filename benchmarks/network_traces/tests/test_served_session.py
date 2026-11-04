"""served_session: the plan is the gate's scan rehearsal inside the scan's class (PROMPT_IDS prompt ids in, or each
prompt's length drawn in a LO:HI span, at most KO out), which this tree's gate accepts, with the endpoint's sealed flags
and its hand start, and a request the gate or the class refuses is refused with its reason; a session through a stub
gate keeps
the session's fetch (with witness/) 0700 under KEEP and publishes only the bundle less witness/ and its outcome, removes
a published file that holds a prompt's ids, and refuses a KEEP inside --out; --at waits on the injected clock and sends
nothing when late; chain runs chain_record.sh's network certifier path on the kept fetch, opening no endpoint records, and is
refused without the leak guard; its draw on the gate's coins names the auditor's node as the coin link's peer, and the
served node is refused."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shlex
import shutil
import signal
import subprocess
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
#: a witness value (the stub's witness/certified.jsonl) its status plants with STUB_NOTE: two of the guard's text pieces
PLANTED = hashlib.sha256(b"a certified line").hexdigest()
#: the warden's stall probe's record of its cgroup, as the gate's unit runs it: the unit's name begins one of the guard's
#: aligned text pieces (`leak_guard.TEXT` bytes), as it did on a served run
_CGROUP = '{"cgroups": {"warden": {"path": "/sys/fs/cgroup/system.slice/system-vy\\\\x2dcertifier.slice/'
STALLS = (" " * (-len(_CGROUP) % 32) + _CGROUP + f'vy-certifier@{RUN}.service"}}}}}}\n').encode()

STUB = r'''#!PYTHON
import io, json, os, signal, sys, tarfile
from pathlib import Path
d, cmd = Path(os.environ["STUB_DIR"]), sys.argv[-1]
if cmd.startswith("warden-rehearse "):
    (d / "prompts.json").write_bytes(sys.stdin.buffer.read())
    (d / "request").write_text(cmd)
    print(json.dumps({"device": "warden", "run": "RUN", "unit": "vy-certifier@RUN.service"}, sort_keys=True))
elif cmd == "status RUN":
    if os.environ.get("STUB_FIFO"):
        open(os.environ["STUB_FIFO"], "w")
        signal.pause()
    print(json.dumps({"run": "RUN", "end": {"rc": 0}, "note": os.environ.get("STUB_NOTE")}))
elif cmd == "fetch RUN":
    files = {"end.json": json.dumps({"run": "RUN", "rc": 0, "verdict": {"accepted": True, "problems": 0}}).encode(),
             "out/deployment.json": json.dumps({"windows": 4, "origin_unix": 1.5,
                                                "params": {"window": 32, "bucket_ms": 100}}).encode(),
             "out/certificates.jsonl": b"", "witness/prompts.json": (d / "prompts.json").read_bytes(),
             "witness/certified.jsonl": b"PLANTED\n", "witness/stalls.json": STALLS}
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
'''.replace("PYTHON", sys.executable).replace("STALLS", repr(STALLS)).replace("RUN", RUN).replace("PLANTED", PLANTED)


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
    assert sorted(doc["runs"]) == ["chain", "session"]
    run = shlex.split(doc["runs"]["chain"])
    assert "--store-read" not in run and run[run.index("--run-max") + 1] == "14400"
    inner = SS.parser().parse_args(shlex.split(run[-1])[2:])
    assert (inner.cmd, inner.draw, inner.coins, str(inner.challenger_key)) == (
        "chain", 8, "gate", "$RESEARCH_RUN_DIR/inputs/challenger-key.json")
    assert all(w in doc["hand_start"] for w in ("--required-manifest ROW/manifest.json", "not examples/vllm/serve.sh",
                                                 "--disk-floor-gb 40", "VERITY_STAGING_BOUNDED stays unset"))
    long = SS.parser().parse_args(["plan", "--prompt-ids", str(SS.KI)])
    long.minutes = a.minutes
    assert SS.plan(long, b"[[1]]")["endpoint"].endswith(f"--max-model-len {SS.KI + SS.KO}")


def test_a_plan_at_another_scan_class_asks_the_gate_for_it_and_its_session_runs_the_plans_request():
    """``--scan-class w63k4`` is passed to the gate, which takes prompts of its KI = 32 ids and 4 tokens out, and the
    plan's session run is the plan's request: its class, prompt ids, tokens and vocabulary."""
    a = SS.parser().parse_args(["plan", "--scan-class", "w63k4", "--prompt-ids", "32", "--tokens", "4", "--prompts", "4",
                                "--requests", "4", "--height", "10"])
    a.minutes = SS.A.warden_minutes(a.requests, a.height) + SS.MARGIN
    ps = SS.prompts(a.prompts, a.prompt_ids, a.vocab, b"seed")
    doc = SS.plan(a, json.dumps(ps).encode())
    assert doc["request"] == (f"warden-rehearse --peer vy-dc-h200-2 --port 18782 --minutes {a.minutes} --requests 4 "
                              "--height 10 --scan --prompts --tokens 4 --scan-class w63k4")
    session = SS.parser().parse_args(shlex.split(shlex.split(doc["runs"]["session"])[-1])[2:])
    assert {k: v for k, v in vars(session).items() if k in vars(a)} == {**vars(a), "cmd": "session"}
    assert "--scan-class" not in SS.plan(SS.parser().parse_args(["plan", "--minutes", "10"]), b"[[1]]")["request"]


@pytest.mark.parametrize("flags, why", [
    (["--tokens", "65"], "1 to KO = 64"),
    (["--tokens", "0"], "1 to KO = 64"),
    (["--prompt-ids", "1025"], "1 to KI = 1024"),
    (["--scan-class", "w63k4", "--prompt-ids", "33", "--tokens", "4"], "1 to KI = 32"),
    (["--scan-class", "w63k4", "--prompt-ids", "32"], "1 to KO = 4"),
    (["--scan-class", "w63k4", "--prompt-ids", "1:33", "--tokens", "4"], "--prompt-ids 1:33: the scan's class takes 1 "
                                                                         "to KI = 32"),
    (["--prompts", "200", "--prompt-ids", "1024"], "the gate refuses it: --prompts: over 1048576 bytes"),
    (["--requests", "100000", "--minutes", "60"], "the gate refuses it: --requests '100000'"),
    (["--minutes", "5"], "the gate refuses it: --minutes 5"),
])
def test_a_request_the_gate_or_the_class_refuses_is_refused_with_its_reason(flags, why, capsys):
    assert SS.main(["plan", *flags]) == 2
    assert why in capsys.readouterr().err


def test_prompt_ids_lo_hi_draws_each_prompts_length_in_the_span_from_the_seed():
    """``--prompt-ids LO:HI``: prompt j's length is drawn in LO … HI from the seed and its ids are the fixed-length
    draw's first ones, so N (or N:N) keeps every prompt N ids, as before; a malformed span is the parser's error."""
    assert [SS.span(t) for t in ("24", "1:32", "5:5")] == [24, (1, 32), 5]
    for bad in ("x", "1:", ":4", "0:4", "5:4", "1:2:3"):
        with pytest.raises(argparse.ArgumentTypeError):
            SS.span(bad)
    ps = SS.prompts(64, (1, 32), SS.VOCAB, b"seed")
    assert {len(p) for p in ps} <= set(range(1, 33)) and len({len(p) for p in ps}) > 8
    assert ps == SS.prompts(64, (1, 32), SS.VOCAB, b"seed") != SS.prompts(64, (1, 32), SS.VOCAB, b"other")
    assert all(p == q[:len(p)] for p, q in zip(ps, SS.prompts(64, 32, SS.VOCAB, b"seed")))
    assert SS.prompts(2, 4, 256, b"seed")[1][3] == int.from_bytes(
        hashlib.sha512(b"seed" + (1).to_bytes(4, "big") + (3).to_bytes(4, "big")).digest()[:8], "big") % 256


def test_a_span_plan_at_w63k4_is_the_gates_request_and_its_session_runs_the_span(tmp_path, capsys):
    """The w63k4 plan at ``--prompt-ids 1:32`` is the gate's request (its ids the group's total), its session runs the
    span, and its endpoint's context is HI + KO; the local stand-in, which serves one prompt length, refuses a span."""
    a = SS.parser().parse_args(["plan", "--scan-class", "w63k4", "--prompt-ids", "1:32", "--tokens", "4", "--prompts",
                                "4", "--requests", "4", "--height", "10"])
    a.minutes = SS.A.warden_minutes(a.requests, a.height) + SS.MARGIN
    ps = SS.prompts(a.prompts, a.prompt_ids, a.vocab, b"seed")
    doc = SS.plan(a, json.dumps(ps).encode())
    assert doc["prompts"] == {"count": 4, "ids": sum(map(len, ps))} and len({len(p) for p in ps}) > 1
    session = SS.parser().parse_args(shlex.split(shlex.split(doc["runs"]["session"])[-1])[2:])
    assert (session.prompt_ids, session.scan_class) == ((1, 32), "w63k4")
    wide = SS.parser().parse_args(["plan", "--prompt-ids", f"1:{SS.KI}", "--minutes", str(a.minutes)])
    assert SS.plan(wide, b"[[1]]")["endpoint"].endswith(f"--max-model-len {SS.KI + SS.KO}")
    assert SS.main(["session", "--out", str(tmp_path), "--gate", "local", "--vocab", "256", "--prompt-ids", "1:8"]) == 2
    assert "the stand-in serves prompts of one length, not --prompt-ids 1:8" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


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
    assert sorted(p.name for p in out.iterdir()) == ["bundle.tgz", "leak-guard.json", "session.json", "status.json"]
    report = json.loads((out / "leak-guard.json").read_text())
    assert (report["verdict"], report["gate_runs"], report["held"], report["released"]) == ("clean", [RUN], 3, 3)
    assert list((keep / "held").iterdir()) == []
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


def test_a_published_file_holding_a_witness_value_is_moved_out_and_the_session_exits_4(tmp_path, monkeypatch):
    """The gate's status planting a value of the witness (no prompt's ids): the leak guard moves status.json to KEEP's
    quarantine and releases the rest."""
    monkeypatch.setenv("STUB_NOTE", PLANTED)
    assert _session(tmp_path, monkeypatch) == 4
    out, q = tmp_path / "out", tmp_path / "keep" / "quarantine"
    assert sorted(p.name for p in out.iterdir()) == ["bundle.tgz", "leak-guard.json", "session.json"]
    assert [p.name for p in q.glob("*/files/status.json")] == ["status.json"]
    assert [d["path"] for d in json.loads((out / "leak-guard.json").read_text())["found"]] == ["status.json"]


def test_sigterm_mid_session_releases_nothing(tmp_path):
    """A timeout's SIGTERM to the job's process group while it polls the gate (its status held on a FIFO the test opens)
    ends the session (exit 143) with nothing released: --out stays empty and its files stay in KEEP's hold."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ssh").write_text(STUB)
    (tmp_path / "bin" / "ssh").chmod(0o755)
    (tmp_path / "out").mkdir()
    fifo = tmp_path / "status.fifo"
    os.mkfifo(fifo)
    env = dict(os.environ, PATH=f"{tmp_path / 'bin'}{os.pathsep}{os.defpath}", STUB_DIR=str(tmp_path),
               STUB_FIFO=str(fifo))
    p = subprocess.Popen([sys.executable, str(HERE / "served_session.py"), "session", "--out", str(tmp_path / "out"),
                          "--keep", str(tmp_path / "keep"), "--key", str(tmp_path / "key"), "--prompts", "3"],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    with open(fifo, "rb"):
        os.killpg(p.pid, signal.SIGTERM)
        said, _ = p.communicate()
    assert p.returncode == 143, said
    assert "SIGTERM ended the session: nothing released" in said
    assert list((tmp_path / "out").iterdir()) == []
    (hold,) = (tmp_path / "keep" / "held").iterdir()
    assert list(hold.iterdir()) == [] and (tmp_path / "keep" / RUN / "start.json").is_file()


def test_the_gates_start_reply_stays_in_keep_since_it_names_the_unit_the_warden_records(tmp_path, monkeypatch):
    """The gate's start reply names the run's systemd unit, vy-certifier@RUN.service, as the warden's stall probe
    records its cgroup in the witness (stalls.json), and no public file of the run holds it: published, the leak guard
    holds the reply as a piece of the witness, and an accepted session exited 4. The reply goes to KEEP/RUN alone
    (0600), so the session exits on the gate's verdict."""
    assert _session(tmp_path, monkeypatch) == 0
    out, kept = tmp_path / "out", tmp_path / "keep" / RUN
    reply = kept / "start.json"
    assert json.loads(reply.read_bytes())["unit"] == f"vy-certifier@{RUN}.service"
    assert reply.stat().st_mode & 0o777 == 0o600 and not (out / "start.json").exists()
    assert json.loads((out / "leak-guard.json").read_text())["found"] == []
    published, q = tmp_path / "published", tmp_path / "q"
    published.mkdir()
    shutil.copyfile(reply, published / "start.json")
    got = subprocess.run([sys.executable, str(HERE / "leak_guard.py"), str(published), str(q), "--gates",
                          str(kept / "fetch.tar.gz"), "--dry", "--report", str(tmp_path / "report.json")],
                         capture_output=True, text=True)
    assert got.returncode == 3, got.stdout + got.stderr
    assert [d["path"] for d in json.loads((tmp_path / "report.json").read_text())["found"]] == ["start.json"]
    doc = json.loads((q / "leak-guard.json").read_text())
    (piece,) = [p for f in doc["found"] for p in f["pieces_found"]]
    assert (piece["witness"], piece["keypath"]) == (f"{RUN}/witness/stalls.json", ["unit"])


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
    argv = ["chain", "--run", RUN, "--keep", str(keep), "CR_N2_DRAW=4"]
    (keep / RUN).mkdir(parents=True)
    (keep / RUN / "fetch.tar.gz").write_bytes(b"")
    assert SS.main(argv) == 2
    (here / "leak_guard.py").write_text("")
    assert SS.main(argv) == 0
    seen = dict(line.split("=", 1) for line in (tmp_path / "seen").read_text().splitlines())
    assert seen == {"CR_PATH": "warden", "CR_GATES": str(keep / RUN / "fetch.tar.gz"), "CR_MEM_GB": "96",
                    "CR_N2_DRAW": "4"}
    assert SS.main(["chain", "--run", "c-none", "--keep", str(keep)]) == 2
    with pytest.raises(SystemExit):
        SS.main([*argv, "--endpoint", "art:ep"])
    assert SS.main([*argv, "N2_DRAW=4"]) == 2
    assert SS.main([*argv, "--draw", "8"]) == 0
    assert "CR_SCAN_ARGS=--draw 8" in (tmp_path / "seen").read_text().splitlines()


def _gate_chain(tmp_path, monkeypatch, peer):
    here = tmp_path / "here"
    here.mkdir(exist_ok=True)
    (here / "chain_record.sh").write_text('env | grep "^CR_SCAN_ARGS=" > "$SEEN"\n')
    (here / "leak_guard.py").write_text("")
    monkeypatch.setattr(SS, "HERE", here)
    monkeypatch.setenv("SEEN", str(tmp_path / "seen"))
    (tmp_path / "challenger-key.json").write_text("{}")
    (tmp_path / "gate-key").write_text("")
    (tmp_path / "main.txt").write_text("abc\n")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = json.dumps({"device": "warden", "peer": peer}).encode()
        info = tarfile.TarInfo(f"{RUN}/request.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    (tmp_path / "keep" / RUN).mkdir(parents=True, exist_ok=True)
    (tmp_path / "keep" / RUN / "fetch.tar.gz").write_bytes(buf.getvalue())
    return ["chain", "--run", RUN, "--keep", str(tmp_path / "keep"), "--coins", "gate", "--draw", "8",
            "--challenger-key", str(tmp_path / "challenger-key.json"), "--key", str(tmp_path / "gate-key"),
            "--main", str(tmp_path / "main.txt")]


def test_a_gate_coin_draw_runs_on_the_auditors_side_and_never_on_the_served_node(tmp_path, monkeypatch, capsys):
    argv = _gate_chain(tmp_path, monkeypatch, "vy-dc-h200-2")
    assert SS.main(argv) == 0
    scan = (tmp_path / "seen").read_text().strip().removeprefix("CR_SCAN_ARGS=").split(" ")
    assert scan == ["--draw", "8", "--challenger", f"gate:{SS.GATE}", "--main", str(tmp_path / "main.txt"), "--peer",
                    SS.AUDITOR, "--challenger-key", str(tmp_path / "challenger-key.json"),
                    "--gate-key", str(tmp_path / "gate-key")]
    for peer in ("vy-dc-h200-2", "vy-nebius-1"):
        argv = _gate_chain(tmp_path, monkeypatch, "vy-nebius-1")
        assert SS.main([*argv, "--coin-peer", peer]) == 2
        assert "is the served node, the prover's" in capsys.readouterr().err
    assert SS.main([*argv, "CR_SCAN_ARGS=--peer vy-dc-h200-2"]) == 2
    assert SS.main([*argv[:-2], "--key", str(tmp_path / "no-key")]) == 2
    assert "no gate key" in capsys.readouterr().err
    assert SS.main([a for a in argv if a not in ("--draw", "8")]) == 2
    assert "give K" in capsys.readouterr().err
