"""served_session: the plan is the gate's scan rehearsal at the scan's class (KI prompt ids in, KO out), which this tree's
gate accepts, and a request the gate or the class refuses is refused with its reason; a session through a stub gate keeps
the lane's fetch (with witness/) 0700 under KEEP and publishes only the bundle less witness/ and its outcome, removes
a published file that holds a prompt's ids, and refuses a KEEP inside --out; --at waits on the injected clock and sends
nothing when late; chain runs chain_record.sh's warden path on the kept fetch, and is refused without the leak guard."""
from __future__ import annotations

import io
import json
import os
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
    assert (SS.KI, SS.KO) == (sz.CLASS.KI, sz.CLASS.KO)
    a = SS.parser().parse_args(["plan"])
    a.minutes = SS.A.warden_minutes(a.requests, a.height) + SS.MARGIN
    ps = SS.prompts(a.prompts, a.prompt_ids, a.vocab, b"seed")
    assert len(ps) == 64 and {len(p) for p in ps} == {SS.KI} and all(0 <= t < SS.VOCAB for p in ps for t in p)
    assert SS.prompts(2, 4, 256, b"seed") != SS.prompts(2, 4, 256, b"other")
    data = json.dumps(ps, separators=(",", ":")).encode()
    doc = SS.plan(a, data)
    assert doc["request"] == ("warden-rehearse --peer vy-dc-h200-2 --port 18782 --minutes 20 --requests 400 --height 12 "
                              "--scan --prompts --tokens 64")
    assert doc["prompts"] == {"count": 64, "ids": 64 * SS.KI} and doc["prompts_bytes"] <= SS.A.WARDEN_PROMPTS_BYTES
    assert "--port 18782 --protocol ids --id-payload 503 --admit release-order" in doc["endpoint"]


@pytest.mark.parametrize("flags, why", [
    (["--tokens", "32"], "exactly KO = 64"),
    (["--prompt-ids", "1025"], "1 to KI = 1024"),
    (["--prompts", "200"], "the gate refuses it: --prompts: over 1048576 bytes"),
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
    assert [len(p) for p in json.loads(prompts)] == [SS.KI] * 3
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
    assert doc["prompts"] == {"count": 3, "ids": 3 * SS.KI} and doc["keep"] == str(kept)
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
    (keep / RUN / "fetch.tar.gz").write_bytes(b"")
    argv = ["chain", "--run", RUN, "--keep", str(keep), "--eos-ids", "256", "CR_N2_DRAW=4"]
    assert SS.main(argv) == 2
    (here / "leak_guard.py").write_text("")
    assert SS.main(argv) == 0
    seen = dict(line.split("=", 1) for line in (tmp_path / "seen").read_text().splitlines())
    assert seen == {"CR_PATH": "warden", "CR_GATES": str(keep / RUN / "fetch.tar.gz"), "CR_MEM_GB": "96",
                    "CR_N2_DRAW": "4", "CR_TIE_WINDOWS": "8", "CR_ENDPOINT": str(keep / RUN / "endpoint"),
                    "CR_EOS_IDS": "256"}
    assert SS.main(["chain", "--run", RUN, "--keep", str(keep)]) == 2
    assert SS.main([*argv, "N2_DRAW=4"]) == 2
