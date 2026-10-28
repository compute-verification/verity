"""served_session: the plan is the gate's scan rehearsal inside the scan's class (PROMPT_IDS prompt ids in, at most KO
out), which this tree's gate accepts, with the endpoint's sealed flags and its hand start, and a request the gate or
the class refuses is refused with its reason; a session through a stub gate keeps
the session's fetch (with witness/) 0700 under KEEP and publishes only the bundle less witness/ and its outcome, removes
a published file that holds a prompt's ids, and refuses a KEEP inside --out; --at waits on the injected clock and sends
nothing when late; io writes the kept run's certified I/O 0600 in KEEP; endpoint runs the endpoint step on the served
node's records in a private copy whose requests.jsonl is rebuilt from the users map and the certified I/O (moved out of
the run dir first), publishes only its report, windows.jsonl and serve.json, and writes nothing from an endpoint that
didn't serve the session's flags or when a published file would carry a request's values (a planted prompt length,
seed or trace record); chain runs chain_record.sh's warden path on the kept fetch with the endpoint's art or a dry
rehearsal's records, and is refused without the leak guard; its coins are the challenger's (CR_CHALLENGER, CR_MAIN,
and the gate's CR_PEER, CR_CHALLENGER_KEY and CR_GATE_KEY), the gate's by default with the auditor's node as the coin
link's peer and ``local`` for a dry run, and a challenger it can't name, or a CR_K=V naming the challenger or the
scan's coins, is refused before chain_record.sh runs."""
from __future__ import annotations

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
    run = shlex.split(doc["runs"]["chain"])
    assert "--store-read" in run[:run.index("--")] and run[run.index("--run-max") + 1] == "14400"
    inner = SS.parser().parse_args(shlex.split(run[-1])[2:])
    assert (inner.cmd, inner.endpoint, inner.eos_ids, inner.draw, inner.challenger, str(inner.challenger_key)) == (
        "chain", "ART", None, 8, f"gate:{SS.GATE}", "$RESEARCH_RUN_DIR/inputs/challenger-key.json")
    assert all(w in doc["hand_start"] for w in ("--required-manifest ROW/manifest.json", "not examples/vllm/serve.sh",
                                                 "--disk-floor-gb 40", "VERITY_STAGING_BOUNDED stays unset"))
    ep = shlex.split(doc["runs"]["endpoint"])
    inner = SS.parser().parse_args(shlex.split(ep[-1])[2:])
    assert ep[ep.index("--") + 1:-1] == ["bash", "-lc"] and ep[ep.index("--on") + 1] == "vy-dc-h200-2"
    assert ep[ep.index("--send") + 1] == SS.CERTIFIED
    assert (inner.cmd, inner.out, inner.port, inner.io) == ("endpoint", Path("$OUT"), 18782,
                                                            f"$RESEARCH_RUN_DIR/inputs/{SS.CERTIFIED}")
    io_run = shlex.split(doc["runs"]["io"])
    assert io_run[io_run.index("--on") + 1] == SS.AUDITOR and "served_session.py io --run RUN" in io_run[-1]
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


def _serve_dir(d: Path, ps: list[list[int]]) -> Path:
    """A serve dir as serve-openai writes it: serve.json (vLLM's own seed nested in it, as served), requests.jsonl and
    advice_trace.jsonl with request values, the users map and windows.jsonl."""
    d.mkdir()
    (d / "serve.json").write_text(json.dumps({"address": ["127.0.0.1", SS.PORT], "protocol": "ids", "id_payload": 503,
                                              "admit": "release-order", "scheme": "rows-sha512", "sealed": "windows/w<k>/",
                                              "retain_exclude": "hidden_m1", "default_seed": 7,
                                              "engine": {"engine_kwargs": {"seed": 0, "max_model_len": 512}}}))
    (d / "requests.jsonl").write_text("".join(json.dumps({"engine_id": f"e{j}", "user": f"user-{j}", "prompt_ids": p,
                                                          "streamed_ids": [1, 2]}) + "\n" for j, p in enumerate(ps)))
    (d / SS.USERS).write_text("".join(json.dumps({"engine_id": f"e{j}", "user": f"user-{j}"}) + "\n"
                                      for j in range(len(ps))))
    (d / "advice_trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        {"kind": "header", "config": {"max_num_seqs": 4}},
        *({"kind": "arrive", "request": f"e{j}", "before_step": 0, "prompt_len": len(p), "sampling": {"seed": 900 + j}}
          for j, p in enumerate(ps)),
        {"kind": "step", "step": 0, "T": 8, "rows": [["e0", 0, 8]], "emitted": {"e0": [1]}}]))
    (d / "windows.jsonl").write_text(json.dumps({"window": 0, "requests": ["e0", "e1"], "served_rows": {"count": 1}}) + "\n")
    (d / "seals.jsonl").write_text("{}\n")
    return d


def _certified(d: Path, ps: list[list[int]]) -> Path:
    """io's file: the warden run's deployment.json and its certified I/O, each prompt answered with [1, 2]."""
    d.mkdir(parents=True, exist_ok=True)
    (d / SS.CERTIFIED).write_text(json.dumps({
        "deployment": {"run_id": RUN, "origin_unix": 0.0, "windows": 4, "params": {"window": 10, "bucket_ms": 10000}},
        "io": {"accepted": True, "requests": len(ps), "tokens": 2 * len(ps), "violations": [],
               "records": [{"user": f"user-{j}", "prompt": p, "max_tokens": 4, "output": [1, 2]}
                           for j, p in enumerate(ps)]}}))
    return d / SS.CERTIFIED


def _stub_step(monkeypatch, report: dict | None = None) -> dict:
    """chain_record.served_endpoint, stubbed: what it was given in ``seen``, and an accepted report (or ``report``)."""
    import chain_record as CR
    seen = {}

    def step(dep, io, ep, eos):
        seen.update(dep=dep, ep=ep, eos=eos, requests=[json.loads(x) for x in (ep / "requests.jsonl").read_text()
                                                       .splitlines()])
        return report or {"warden_run": dep["run_id"], "endpoint": str(ep), "io": {k: v for k, v in io.items()
                                                                                   if k != "records"},
                          "chain": {"bits": {"bound": True, "rebuilt": True,
                                             "commit_windows": [{"bound": True, "rebuilt": True}]}}}
    monkeypatch.setattr(CR, "served_endpoint", step)
    monkeypatch.setenv("PYTHONPATH", "")  # endpoint sets it; this restores it after the test
    return seen


def test_io_writes_the_kept_runs_certified_io_private_in_keep(tmp_path, monkeypatch, capsys):
    ps = SS.prompts(2, 4, 256, b"seed")
    doc = json.loads(_certified(tmp_path / "c", ps).read_text())
    monkeypatch.setattr(SS, "certified_io", lambda fetch: (doc["deployment"], doc["io"]))
    keep = tmp_path / "keep"
    assert SS.main(["io", "--run", RUN, "--keep", str(keep)]) == 2 and "no kept fetch" in capsys.readouterr().err
    (keep / RUN).mkdir(parents=True)
    (keep / RUN / "fetch.tar.gz").write_bytes(b"")
    assert SS.main(["io", "--run", RUN, "--keep", str(keep)]) == 0
    dest = keep / RUN / SS.CERTIFIED
    assert json.loads(dest.read_text()) == doc and dest.stat().st_mode & 0o777 == 0o600
    assert not any(n in capsys.readouterr().out.encode() for n in SS.needles(ps))
    doc["io"]["accepted"] = False
    assert SS.main(["io", "--run", RUN, "--keep", str(keep)]) == 2 and "doesn't read whole" in capsys.readouterr().err


def test_endpoint_runs_the_step_on_the_served_nodes_records_and_publishes_only_its_report_and_public_records(
        tmp_path, monkeypatch, capsys):
    import chain_record as CR
    seen = _stub_step(monkeypatch)
    ps = SS.prompts(2, SS.PROMPT_IDS, SS.VOCAB, b"seed")
    src, run = _serve_dir(tmp_path / "serve", ps), tmp_path / "run"
    out = run / "out"
    listing = sorted(p.name for p in src.iterdir())
    sent = _certified(run / "inputs", ps)
    assert SS.main(["endpoint", "--from", str(src), "--io", str(sent), "--out", str(out)]) == 0
    keep = tmp_path / "serve.endpoint"
    assert not sent.exists() and (keep / SS.CERTIFIED).stat().st_mode & 0o777 == 0o600
    assert keep.stat().st_mode & 0o777 == 0o700 and (keep / "records" / "requests.jsonl").stat().st_mode & 0o777 == 0o600
    assert sorted(p.name for p in out.iterdir()) == ["report.json", "serve.json", "windows.jsonl"]
    rep = json.loads((out / "report.json").read_text())
    assert (rep["verdict"], rep["at"], rep["warden_run"]) == ("accepted", CR.SERVED_NODE, RUN)
    assert seen["ep"] == keep / "records" and seen["eos"] == [128001] and seen["dep"]["run_id"] == RUN
    assert [(r["engine_id"], r["user"], r["prompt_ids"], r["streamed_ids"]) for r in seen["requests"]] == [
        ("e0", "user-0", ps[0], [1, 2]), ("e1", "user-1", ps[1], [1, 2])]
    assert sorted(p.name for p in src.iterdir()) == listing
    published = b"".join(p.read_bytes() for p in out.iterdir()) + capsys.readouterr().out.encode()
    assert not any(n in published for n in [*SS.needles(ps), b'"prompt_len"', b'"sampling"', b'"emitted"'])
    stdin = tmp_path / "stdin.json"
    stdin.write_bytes(_certified(tmp_path / "kept", ps).read_bytes())
    shutil.rmtree(out)
    with stdin.open() as f:
        monkeypatch.setattr(sys, "stdin", f)
        assert SS.main(["endpoint", "--from", str(src), "--io", "-", "--out", str(out)]) == 0
    assert json.loads((out / "report.json").read_text())["verdict"] == "accepted"


def test_endpoint_publishes_nothing_that_carries_a_planted_prompt_length_or_seed(tmp_path, monkeypatch, capsys):
    """A request's values never reach the endpoint's art: a prompt length, a seed or a trace's record planted in a file
    it would publish, or in the step's report, and nothing is written."""
    seen = _stub_step(monkeypatch)
    ps = SS.prompts(2, SS.PROMPT_IDS, SS.VOCAB, b"seed")
    src, run = _serve_dir(tmp_path / "serve", ps), tmp_path / "run"
    out = run / "out"
    io = _certified(tmp_path / "kept", ps)
    argv = ["endpoint", "--from", str(src), "--io", str(io), "--out", str(out)]
    windows = (src / "windows.jsonl").read_text()
    for plant, why in (
            ({"kind": "arrive", "request": "e0", "before_step": 3, "prompt_len": 256, "sampling": {"seed": 1234}},
             "windows.jsonl: a request's prompt ids or a request-valued field"),
            ({"window": 1, "prompt_len": 256}, "windows.jsonl: a request's prompt ids or a request-valued field"),
            ({"window": 1, "seed": 1234}, "windows.jsonl: a record with seed"),
            ({"kind": "step", "step": 0, "T": 8}, "windows.jsonl: a scheduler trace's step record"),
            ({"window": 1, "rows": [["e0", 0, 8]]}, "windows.jsonl: a record with rows"),
            ({"window": 1, "ids": ps[1]}, "windows.jsonl: a request's prompt ids")):
        (src / "windows.jsonl").write_text(windows + json.dumps(plant) + "\n")
        assert SS.main(argv) == 2, plant
        err = capsys.readouterr().err
        assert why in err and "nothing written" in err and not out.exists(), (plant, err)
    (src / "windows.jsonl").write_text(windows)
    for plant, why in (({"prompt_len": 256}, "a request's prompt ids or a request-valued field"),
                       ({"seed": 1234}, "a record with seed")):
        rep = {"warden_run": RUN, "io": {"accepted": True}, "chain": {"bits": {"bound": True, "rebuilt": True}}}
        _stub_step(monkeypatch, rep | plant if "seed" in plant else rep | {"chain": {"bits": rep["chain"]["bits"] | plant}})
        assert SS.main(argv) == 2 and f"report.json: {why}" in capsys.readouterr().err and not out.exists()
    _stub_step(monkeypatch)
    assert SS.main(argv) == 0 and seen is not None
    shutil.rmtree(out)

    serve = json.loads((src / "serve.json").read_text())
    for k, v, why in (("retain_exclude", None, "--retain-exclude hidden_m1 (serve.json: None)"),
                      ("sealed", None, "--seal (serve.json: None)"),
                      ("address", ["127.0.0.1", 18783], "--port 18782 (serve.json: 18783)"),
                      ("admit", "arrival", "--admit release-order (serve.json: 'arrival')")):
        (src / "serve.json").write_text(json.dumps(serve | {k: v}))
        assert SS.main(argv) == 2 and why in capsys.readouterr().err and not out.exists()
    (src / "serve.json").write_text(json.dumps(serve))
    assert SS.main([*argv[:-1], str(SS.TREE / "endpoint-out")]) == 2 and "is in the tree" in capsys.readouterr().err
    assert not (SS.TREE / "endpoint-out").exists()
    for keep, what in ((out / "keep", "is inside --out,"), (run / "keep", "is inside --out's directory, the run dir")):
        sent = _certified(run / "inputs", ps)
        assert SS.main([*argv[:3], "--io", str(sent), "--out", str(out), "--keep", str(keep)]) == 2
        assert what in capsys.readouterr().err and not sent.exists() and not out.exists() and not keep.exists()
    for f in ("advice_trace.jsonl", "requests.jsonl"):
        (src / f).rename(tmp_path / f)
        assert SS.main(argv) == 2 and f"no {f}" in capsys.readouterr().err
        (tmp_path / f).rename(src / f)


def test_chain_runs_chain_records_warden_path_on_the_kept_fetch_and_needs_the_leak_guard(tmp_path, monkeypatch):
    here = tmp_path / "here"
    here.mkdir()
    (here / "chain_record.sh").write_text('env | grep "^CR_" | sort > "$SEEN"\n')
    monkeypatch.setattr(SS, "HERE", here)
    monkeypatch.setenv("SEEN", str(tmp_path / "seen"))
    keep = tmp_path / "keep"
    (keep / RUN / "endpoint").mkdir(parents=True)
    for f in ("requests.jsonl", "advice_trace.jsonl"):
        (keep / RUN / "endpoint" / f).write_text("")
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


def test_chain_reads_the_endpoints_art_as_given_and_needs_no_eos_ids_for_it(tmp_path, monkeypatch):
    here, keep = tmp_path / "here", tmp_path / "keep"
    for d in (here, keep / RUN):
        d.mkdir(parents=True)
    (here / "chain_record.sh").write_text('env | grep "^CR_E" | sort > "$SEEN"\n')
    (here / "leak_guard.py").write_text("")
    (keep / RUN / "fetch.tar.gz").write_bytes(b"")
    monkeypatch.setattr(SS, "HERE", here)
    monkeypatch.setenv("SEEN", str(tmp_path / "seen"))
    (tmp_path / "main.txt").write_text("abc\n")
    assert SS.main(["chain", "--run", RUN, "--keep", str(keep), "--endpoint", "art:ep", "--challenger", "local",
                    "--main", str(tmp_path / "main.txt")]) == 0
    assert (tmp_path / "seen").read_text().splitlines() == ["CR_ENDPOINT=art:ep"]


def test_the_endpoints_requests_jsonl_is_rebuilt_from_its_users_map_and_the_certified_io(tmp_path, capsys):
    ps = [[5, 6], [7], [8]]
    src = _serve_dir(tmp_path / "serve", ps)
    io = [{"user": "user-0", "prompt": [5, 6], "max_tokens": 4, "output": [9, 128001]},
          {"user": "user-1", "prompt": [7], "max_tokens": 4, "output": [1, 2]},
          {"user": "user-2", "prompt": [8], "max_tokens": 4, "output": [1, 2, 3, 4]}]
    dest = SS.rebuild(src, tmp_path / "keep" / "records", io, [128001])
    assert dest.stat().st_mode & 0o777 == 0o700 and sorted(p.name for p in dest.iterdir()) == sorted(SS.SERVED)
    got = [json.loads(x) for x in (dest / "requests.jsonl").read_text().splitlines()]
    assert [(r["engine_id"], r["user"], r["prompt_ids"], r["streamed_ids"], r["finish"]) for r in got] == [
        ("e0", "user-0", [5, 6], [9, 128001], {"reason": "stop", "stop_reason": None}),
        ("e1", "user-1", [7], [1, 2], {"reason": "stop", "stop_reason": None}),
        ("e2", "user-2", [8], [1, 2, 3, 4], {"reason": "length", "stop_reason": None})]
    users = (src / SS.USERS).read_text()
    for bad in ({"engine_id": "e3", "user": None}, {"engine_id": "e3", "user": "user-1"}):
        (src / SS.USERS).write_text(users + json.dumps(bad) + "\n")
        with pytest.raises(SS.Refused, match=f"engine request e3's user {bad['user']!r} is no certified request left"):
            SS.rebuild(src, tmp_path / "keep" / "records", io, [128001])


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
