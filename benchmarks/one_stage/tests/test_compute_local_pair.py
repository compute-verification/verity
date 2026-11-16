"""`benchmarks/compute/serve.sh`'s local pair on a toy pass: its challenger (``COINS=local``), the scratch auditor gate
`pearlplus_local_e2e.py init` starts (serve.sh's ``gate_up``), each opened session's challenger its own ``serve.py`` on
127.0.0.1, and its verifier, serve.sh's ``retain``, ``python-verify`` and ``lean`` steps (`served_passes.py`, ``lean --exe
build``) run as processes of their own under the gate's ``PEARLC_CHALLENGER_*``; then `pearlplus_local_e2e.py stop`
(serve.sh's ``gate_down``).

The serve is the pouw suite's toy pass (`proved_pass`: k = 128 in 16 x 16 tiles, so ``verify --test-instance`` and no
``--class``), salted by a live salt-only session on the gate, as `pearlplus_local_e2e.py run` salts it. It checks that the
verifier's sample is a withheld sample session's on that gate (``draw.from`` ``challenger``), that the Lean takes the
run of record (``--of-record``, which `served_passes.py lean` passes only on a live session's epoch coins) and agrees
with the Python, that both accept and credit the toy nothing (a test instance's verdict), and that once ``stop`` has run
no process the gate started is left: a third session opened with no client, its challenger still listening, among them.

serve.sh itself is not run: its first step builds the sm_90a cubin with nvcc and exits without it, and its verify holds
the passes to the benchmark's class, which no toy pass is. So a change to its own wiring (the gate's JSON it reads, the
environment it exports, its step lines) is not caught here; a change to the verbs and commands it calls is.

What it catches that a cheaper test can't: the gate's verbs, serve.py and the Lean coins and verifier as processes, which
the stand-in tests stub. It builds ``spot-verifier`` and ``pouw-pearlplus`` with lake first (minutes cold), so it is slow."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import select
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E2E = ROOT / "benchmarks" / "one_stage" / "pearlplus_local_e2e.py"
SP = ROOT / "benchmarks" / "pouw" / "pearl_c_vllm" / "served_passes.py"
SCHEME = "pearl-c-h100-v1-h3"
pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]


def _module(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


T = _module("test_pearl_c_vllm", ROOT / "benchmarks" / "pouw" / "tests" / "test_pearl_c_vllm.py")


def _py(argv: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *argv], cwd=ROOT, capture_output=True, text=True, env=env)


def _held(root: Path) -> tuple[dict[int, int], dict[int, str]]:
    """Each live process whose command line names `root` (a session's ``serve`` verb and its serve.py), with a pidfd on
    it, so its exit can be waited on; and each process that went away while it was read, with what reading it raised."""
    got, gone = {}, {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            named = str(root).encode() in (d / "cmdline").read_bytes()
            state = (d / "stat").read_text().rsplit(")", 1)[1].split()[0]
            if named and state != "Z":
                got[int(d.name)] = os.pidfd_open(int(d.name))
        except (OSError, IndexError) as e:
            gone[int(d.name)] = type(e).__name__
    return got, gone


def _ledger(root: Path) -> list[dict]:
    path = root / "var" / "windows.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_the_local_challenger_samples_the_verifier_and_stop_leaves_none_of_its_processes(tmp_path, monkeypatch):
    from verity.protocols.resource_accounting.compute.pouw.prover.serving.h1 import window as PHW
    from verity_vllm.pipeline.commit_coins import ChallengerGate

    root, out, scratch = tmp_path / "auditor", tmp_path / "out", tmp_path / "scratch"
    init = _py([str(E2E), "init", "--root", str(root)])
    assert init.returncode == 0, init.stderr[-2000:]
    gate = json.loads(init.stdout.strip().splitlines()[-1])
    assert (gate["peer"], gate["lane"], gate["params"]) == ("local", "vllm", [4, 6])
    env = {**os.environ, "PEARLC_CHALLENGER_GATE": gate["gate"], "PEARLC_CHALLENGER_PEER": gate["peer"],
           "PEARLC_CHALLENGER_LANE": gate["lane"]}
    piece = out / "pieces" / "pouw"
    try:
        g = ChallengerGate(gate["gate"])
        w = T.weights_root(T.proved_pass(scratch / "probe", SCHEME)[3].weight_roots)
        salt, _ = PHW.reserve_salt(g.call, g.ledger, gate["lane"], gate["peer"], 0)
        client, commitment, opening, _, carried = PHW.open_session("compute local pair", g.call, salt["session"], w, 0)
        monkeypatch.setattr(T, "live_coin", lambda after, entropy=7: (commitment, opening))
        path = T.proved_pass(scratch / "pass", SCHEME, live=0)[0]
        path.write_text(json.dumps({**json.loads(path.read_text()), "session": carried}))
        client.close()
        served, wr = T._serve_dir(scratch, [path]), scratch / "window-record.json"

        retain = _py([str(SP), "retain", "--passes", str(served), "--window-record", str(wr)], env)
        verify = _py([str(SP), "verify", "--passes", str(served), "--tie", str(wr), "--test-instance", "--out",
                      str(piece / "verify.json"), "--answer", str(piece / "pearlplus-answer.json"), "--jobs", "2"], env)
        lean = _py([str(SP), "lean", "--passes", str(served), "--verify", str(piece / "verify.json"), "--exe", "build",
                    "--out", str(piece / "pearlplus-lean.json")], env)

        pending, _ = PHW.reserve_salt(g.call, g.ledger, gate["lane"], gate["peer"], 0)
        g.call("open", "--session", pending["session"], "--root", hashlib.sha512(b"no client").hexdigest())
    finally:
        held, _ = _held(root)
        stop = _py([str(E2E), "stop", "--root", str(root), "--out", str(out)])
        for fd in held.values():
            select.select([fd], [], [])
            os.close(fd)
    left, gone = _held(root)
    for fd in left.values():
        os.close(fd)

    assert retain.returncode == 0, retain.stdout[-2000:]
    assert verify.returncode == 0, verify.stdout[-2000:] + verify.stderr[-2000:]
    record = json.loads((served / "coin.json").read_text())
    draw = record["draw"]
    assert draw["from"] == "challenger" and draw["reserved"]["session"] not in (salt["session"], pending["session"])
    answer = json.loads((piece / "pearlplus-answer.json").read_text())
    work = answer["public"]["work"]
    assert answer["verdict"] == "accept" and work["credited"] == 0 and work["credited_tiles"] > 0

    assert lean.returncode == 0, lean.stdout[-2000:] + lean.stderr[-2000:]
    check = json.loads((piece / "pearlplus-lean-check.json").read_text())
    assert check["differ"] == [] and check["reason"] is None and "--of-record" in check["argv"]
    got = json.loads((piece / "pearlplus-lean.json").read_text())
    assert (got["verdict"], got["commitment"], got["source"]) == ("accept", answer["commitment"], record["source"])
    assert got["work"] == work

    sessions = {e.get("session"): e for e in _ledger(root) if e["kind"] == "reserve"}
    assert {salt["session"], draw["reserved"]["session"], pending["session"]} <= set(sessions)
    assert (sessions[draw["reserved"]["session"]]["body"].get("withheld") is True
            and not sessions[salt["session"]]["body"].get("withheld"))

    assert stop.returncode == 0, stop.stderr[-2000:]
    assert (out / "auditor" / "windows.jsonl").is_file() and (out / "auditor" / "public").is_file()
    assert int((root / f"{pending['session']}.pid").read_text()) in held
    assert sorted(left) == [], f"left running: {sorted(left)}; went away while read: {gone}"
