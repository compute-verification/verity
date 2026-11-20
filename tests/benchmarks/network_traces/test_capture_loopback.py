"""`capture_loopback.sh` with stand-ins for what it starts: a `tcpdump` that says it listens and waits for the job's
SIGINT, or copies a capture read back; an interpreter that records the rehearsal's, the publisher's and the reader's
command lines, leaves the rehearsal's deployment, record heads and cells' lines, and runs the publisher; and a `sleep` that returns at once. What the job gives the rehearsal for the egress
form the launch names, and for none; that a form that isn't one, or a flag that names one, ends the job before any
capture; and that a run in another form than the launch's isn't read."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = (Path(__file__).resolve().parents[3] / "benchmarks" / "network_traces")
ROOT = HERE.parents[1]
FRONT, RELAY = 18001, 18002
TCPDUMP = """\
#!{python}
import json, os, shutil, signal, sys

args = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps(["tcpdump", *args]) + "\\n")
kept = args[args.index("-w") + 1]
if "-r" in args:
    shutil.copyfile(args[args.index("-r") + 1], kept)
    sys.exit(0)
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
open(kept, "wb").write(b"packets")
print("listening on lo", file=sys.stderr, flush=True)
signal.pause()
"""
PYTHON = """\
#!{python}
import json, os, sys

args = sys.argv[1:]
if args[0] == "-c":
    os.execv({python!r}, [{python!r}, *args])
if args[0] == "-m":
    args = args[1:]
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps([os.path.basename(args[0]), *args[1:]]) + "\\n")
if args[0].endswith(".auditor"):
    os.execv({python!r}, [{python!r}, "-m", *args])
if args[1:2] == ["rehearse"]:
    run = args[args.index("--out") + 1]
    os.makedirs(run)
    form = os.environ.get("STUB_FORM") or ("id" if "--ids" in args else "bytes")
    json.dump({{"upstream": ["127.0.0.1", {front}], "listen": ["127.0.0.1", {relay}]}},
              open(os.path.join(run, "endpoints.json"), "w"))
    json.dump({{"params": {{"egress": form}}}}, open(os.path.join(run, "deployment.json"), "w"))
    for private in ("cells.salts", "ingress.records"):
        open(os.path.join(run, private), "wb").write(b"the prover's")
    json.dump({{"head": {{"root": "ab" * 64, "counts": [1]}}, "salt_c": "cd" * 64, "certificate": {{}}}},
              open(os.path.join(run, "certified.jsonl"), "w"))
    json.dump({{"registration": {{"root": "ef" * 64, "wiring": "01" * 64}}, "cost": {{"rows": 1}}}},
              open(os.path.join(run, "cells.jsonl"), "w"))
else:
    flow = {{"to_server": {{"stream_bytes": 4242}}, "from_server": {{"stream_bytes": 4343}}}}
    json.dump({{"between": [flow], "others": [flow], "record": {{"connections": [flow]}},
               "compared": {{"to_server": {{"says": "agrees"}}, "from_server": {{"says": "agrees"}},
                            "pairing": {{"says": "agrees", "flows": 1}}}}}},
              open(args[args.index("--out") + 1], "w"))
    print("the reader's table: 4242 4343")
"""
#: the stand-ins, by the name the job calls each. With a `sleep` that returns at once the job's two waits end on the
#: stand-in `tcpdump`'s line and on its file's size, and on no clock
STAND_INS = {"tcpdump": TCPDUMP, "python": PYTHON, "sleep": "#!/bin/sh\n"}


def job(tmp_path: Path, *flags: str, **env: str) -> tuple[subprocess.CompletedProcess, list[list[str]], Path]:
    """The job under `tmp_path`'s stand-ins, with `flags` and `env`: its process, the command lines the stand-ins
    recorded, in order, and its `$OUT`."""
    bin_, run = tmp_path / "bin", tmp_path / f"run{len(list(tmp_path.glob('run*')))}"
    bin_.mkdir(exist_ok=True)
    for name, text in STAND_INS.items():
        (bin_ / name).write_text(text.format(python=sys.executable, front=FRONT, relay=RELAY))
        (bin_ / name).chmod(0o755)
    log = run / "calls.jsonl"
    run.mkdir()
    p = subprocess.run(["bash", str(HERE / "capture_loopback.sh"), *flags], cwd=ROOT, capture_output=True, text=True,
                       env={"PATH": f"{bin_}:/usr/bin:/bin", "HOME": str(tmp_path), "PY": str(bin_ / "python"),
                            "STUB_LOG": str(log), "RESEARCH_RUN_DIR": str(run), **env})
    return p, [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else [], run / "out"


def test_the_rehearsal_is_in_the_launchs_egress_form_and_in_bytes_when_it_gives_none(tmp_path):
    """With no `EGRESS`, and with `bytes`, the rehearsal gets the job's three flags and the launch's, nothing for the
    form; with `id` it gets `--ids` before the launch's. Either way the capture is the loopback's TCP kept at 54
    bytes a packet, which holds no content, what is kept of it is its packets on the deployment's two ports, and the
    reader gets the kept file, the front's address with its port, the relay's and the run. The run, the kept packets
    (the front's port is the wire before the attester) and the reader's result and table, which count each flow, are
    in the job's temporary directory, deleted with it; $OUT/run is what the gate's publisher gives of the run, with no
    head, salt, wiring or cost, and $OUT/capture_flows.json the reader's three verdicts alone. The exit is the
    reader's."""
    for env, form in (({}, []), ({"EGRESS": "bytes"}, []), ({"EGRESS": "id"}, ["--ids"])):
        p, calls, out = job(tmp_path, "--requests", "3", "--lanes", "64", **env)
        assert p.returncode == 0, p.stdout + p.stderr
        capture, rehearsal, publish, kept, reader = calls
        assert capture[:7] == ["tcpdump", "-i", "lo", "-U", "-s", "54", "-w"] and capture[8:] == ["tcp"]
        run = rehearsal[rehearsal.index("--out") + 1]
        assert rehearsal[1:] == ["rehearse", "--fake", "--watermark", "--out", run, *form,
                                 "--requests", "3", "--lanes", "64"]
        assert Path(run).parent == Path(capture[7]).parent and not Path(run).exists()
        w = Path(capture[7]).parent
        assert kept == ["tcpdump", "-r", capture[7], "-w", f"{w}/run.pcap", f"tcp port {FRONT} or tcp port {RELAY}"]
        assert reader == ["capture_flows.py", "--capture", f"{w}/run.pcap", "--server", f"127.0.0.1:{FRONT}",
                          "--certifier", "127.0.0.1", "--run", run, "--out", f"{w}/capture_flows.json"]
        assert publish == ["verity.protocols.compose.verifier.live_challenger.auditor", "publish", run, f"{out}/run"]
        assert sorted(p.name for p in (out / "run").iterdir()) == ["deployment.json", "outcome.json"]
        published = " ".join(p.read_text() for p in (out / "run").iterdir())
        assert not any(s in published for s in ("head", "salt_c", "wiring", "cost", "ab" * 64, "cd" * 64, "01" * 64))
        assert not w.exists() and sorted(p.name for p in out.iterdir()) == ["capture_flows.json", "run"]
        says = {"says": {"to_server": "agrees", "from_server": "agrees", "pairing": "agrees"}}
        assert json.loads((out / "capture_flows.json").read_text()) == says
        assert not any(n in p.stdout for n in ("4242", "4343", "table")), p.stdout


def test_a_form_that_isnt_one_or_a_flag_that_names_one_ends_the_job_before_any_capture(tmp_path):
    for flags, env, said in (((), {"EGRESS": "text"}, "EGRESS is bytes or id, not 'text'"),
                             (("--requests", "3", "--ids"), {}, "the egress form is EGRESS's, not --ids's"),
                             (("--egress", "id"), {"EGRESS": "id"}, "the egress form is EGRESS's, not --egress's"),
                             (("--egress=bytes",), {}, "the egress form is EGRESS's, not --egress=bytes's")):
        p, calls, out = job(tmp_path, *flags, **env)
        assert p.returncode == 4 and said in p.stdout, p.stdout + p.stderr
        assert calls == [] and list(out.iterdir()) == []


def test_a_run_in_another_form_than_the_launchs_isnt_read(tmp_path):
    """A flag the job's own check doesn't know as the form's can still set it; the run's deployment says the form it
    ran in, and the job reads a run only in the launch's."""
    p, calls, out = job(tmp_path, "--requests", "3", STUB_FORM="id")
    assert p.returncode == 1 and "the run's egress form is id, not EGRESS's (bytes)" in p.stdout, p.stdout + p.stderr
    assert len(calls) == 3 and calls[0][0] == "tcpdump" and calls[1][1] == "rehearse" and calls[2][1] == "publish"
    assert sorted(p.name for p in out.iterdir()) == ["run"]
