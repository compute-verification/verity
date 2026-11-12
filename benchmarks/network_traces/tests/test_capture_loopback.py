"""`capture_loopback.sh` with stand-ins for what it starts: a `tcpdump` that says it listens and waits for the job's
SIGINT, or copies a capture read back; an interpreter that records the rehearsal's and the reader's command lines and
leaves the rehearsal's deployment; and a `sleep` that returns at once. What the job gives the rehearsal for the egress
form the launch names, and for none; that a form that isn't one, or a flag that names one, ends the job before any
capture; and that a run in another form than the launch's isn't read."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
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
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps([os.path.basename(args[0]), *args[1:]]) + "\\n")
if args[1:2] == ["rehearse"]:
    run = args[args.index("--out") + 1]
    os.makedirs(run)
    form = os.environ.get("STUB_FORM") or ("id" if "--ids" in args else "bytes")
    json.dump({{"upstream": ["127.0.0.1", {front}], "listen": ["127.0.0.1", {relay}], "params": {{"egress": form}}}},
              open(os.path.join(run, "deployment.json"), "w"))
else:
    print("the reader's table")
"""


def job(tmp_path: Path, *flags: str, **env: str) -> tuple[subprocess.CompletedProcess, list[list[str]], Path]:
    """The job under `tmp_path`'s stand-ins, with `flags` and `env`: its process, the command lines the stand-ins
    recorded, in order, and its `$OUT`."""
    bin_, run = tmp_path / "bin", tmp_path / f"run{len(list(tmp_path.glob('run*')))}"
    bin_.mkdir(exist_ok=True)
    for name, text in (("tcpdump", TCPDUMP), ("python", PYTHON), ("sleep", "#!/bin/sh\n")):
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
    form; with `id` it gets `--ids` before the launch's. Either way the capture is the loopback's TCP at the reader's
    snapshot length, what is kept of it is its packets on the deployment's two ports, and the reader gets the kept
    file, the front's address with its port, the relay's and the run. The exit is the reader's."""
    for env, form in (({}, []), ({"EGRESS": "bytes"}, []), ({"EGRESS": "id"}, ["--ids"])):
        p, calls, out = job(tmp_path, "--requests", "3", "--lanes", "64", **env)
        assert p.returncode == 0, p.stdout + p.stderr
        capture, rehearsal, kept, reader = calls
        assert capture[:7] == ["tcpdump", "-i", "lo", "-U", "-s", "130", "-w"] and capture[8:] == ["tcp"]
        assert rehearsal[1:] == ["rehearse", "--fake", "--watermark", "--out", f"{out}/run", *form,
                                 "--requests", "3", "--lanes", "64"]
        assert kept == ["tcpdump", "-r", capture[7], "-w", f"{out}/run.pcap", f"tcp port {FRONT} or tcp port {RELAY}"]
        assert reader == ["capture_flows.py", "--capture", f"{out}/run.pcap", "--server", f"127.0.0.1:{FRONT}",
                          "--certifier", "127.0.0.1", "--run", f"{out}/run", "--out", f"{out}/capture_flows.json"]
        assert (out / "run.pcap").read_bytes() == b"packets" and not Path(capture[7]).exists()
        assert (out / "capture_flows.txt").read_text() == p.stdout == "the reader's table\n"


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
    assert len(calls) == 2 and calls[0][0] == "tcpdump" and calls[1][1] == "rehearse"
    assert not (out / "run.pcap").exists() and not (out / "capture_flows.json").exists()
