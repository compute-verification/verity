"""The remote bandwidth audit beside sbench's served arms on the GPU they share (`sbench.py --bw-passes GIB`, `dbw.py`).

One process stands in for the auditor: the remote audit's own verifier (`benchmarks/poub_session/verifier.py`'s
`Verifier`, its coins from an in-process challenger, the Lean verifier of record deciding) on a loopback socket. Its
prover is `prover.py`, unmodified, on the GPU sbench leased (CUDA_VISIBLE_DEVICES inherited) and on --prover-cpus,
over a store of GIB GiB of seeded payload under kv.cu's p1024 masked codec (`device.py`, the store `run.sh` registers on
one GPU). That store stands in for the node's registered one: the engines' weights arenas that `node.py` registers can't
share a GPU with sbench's arms. sbench sends one JSON line a command on stdin:

    {"op": "session", "k": K, "paced": "1:1000:2773:W"}   session K, paced (`verifier.pace_of`), its first coin now
    {"op": "stop"}                                       the registration closed, the prover's bye, and exit

and reads one JSON line an event on stdout, each with its `t_ns` on the host's monotonic clock (sbench's workers'
too): `ready` (the registration, the prover's calibration, the round trips and the allowance A), `started` (session
K's first coin sent: its passes begin), `passes` (the prover's `closed`: its last pass is done), `ended` (K's verdict
and failure, its pass timeline and memory lines; `dbw.session_of`), `stopped`, or `error`. The prover's plan (honest,
skip:F, one session each) is fixed at its start. OUT/verifier.json is the verifier's record as `verifier.py` writes
it, and OUT/prover/ the prover's.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "poub_session"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import verifier as V
import wire

from verity.core.protocols.poub import audit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RS = ROOT / "benchmarks" / "poub_session"

EVENTS = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr
_lock = threading.Lock()


def emit(event: str, **kw) -> None:
    with _lock:
        EVENTS.write(json.dumps({"event": event, "t_ns": time.monotonic_ns(), **kw}) + "\n")


def lines(r: dict) -> dict:
    """A session's verdict, pass timeline and memory lines, without its draws: what sbench's report reads."""
    seen = r.get("windows_seen", [])
    passes = []
    for w in seen:
        at = w["released_ns"]
        for got in w["arrivals_ns"]:
            passes.append([at, got])
            at = got
    lg = r.get("ledger", {})
    mem = lg.get("memory", {})
    paced = lg.get("paced") or {}
    return {"k": r["k"], "accepted": r["accepted"], "failure": r["failure"], "pace": r.get("pace"),
            "allowance_ns": r["allowance_ns"], "passes_ns": passes,
            "slack_ns": [w["slack_ns"] for w in seen if w.get("slack_ns") is not None],
            "memory": {k: mem.get(k) for k in ("share", "residual", "epsilon", "hbm_bytes", "block_bytes")}
            | {"average_share": mem.get("average", {}).get("share")},
            "paced": {k: paced[k] for k in ("snapshot", "time", "refused") if k in paced},
            "gamma": lg.get("spec", {}).get("gamma"), "end_ns": r.get("end_ns")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan", required=True, help="the prover's sessions in order (honest, skip:F), comma-separated")
    ap.add_argument("--gib", type=float, default=24.0, help="the store, whole 1 GiB chunks")
    ap.add_argument("--lib", help="session.so (without it the prover's options name its backend: --cpu BLOCKS)")
    ap.add_argument("--kv", help="kv.so (p1024, masked)")
    ap.add_argument("--lean-exe", type=Path, help="a built poub-check-verifier (default: the verifier builds it)")
    ap.add_argument("--jitter-ms", type=float, default=50.0)
    ap.add_argument("--quantile", default="max", choices=("p50", "p99", "p999", "max"))
    ap.add_argument("--pings", type=int, default=1000)
    ap.add_argument("--pace-pass-ms", type=float, default=0.0,
                    help="a paced stretch's groups are counted at this pass (0: the calibration's median)")
    ap.add_argument("--hbm-bytes", type=int, default=V.GPU_HBM_BYTES, help="the memory line's capacity: one GPU's")
    ap.add_argument("--prover-py", default=sys.executable)
    ap.add_argument("--prover-cpus", help="the prover's cores (taskset -c; default this process's)")
    ap.add_argument("prover_args", nargs="*",
                    help="after --: prover.py's and device.py's own options, leading NAME=VALUE words its environment "
                         "(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=44 under sbench.sh's SBENCH_MPS=1)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.setswitchinterval(2e-4)
    server = wire.listen("127.0.0.1", 0)
    port = server.getsockname()[1]
    own = {}
    while a.prover_args and "=" in a.prover_args[0] and not a.prover_args[0].startswith("-"):
        name, _, value = a.prover_args.pop(0).partition("=")
        own[name] = value
    store = ["--lib", a.lib, "--kv", a.kv, "--gib", str(a.gib)] if a.lib else []
    cmd = [a.prover_py, str(RS / "prover.py"), "--connect", f"127.0.0.1:{port}", "--out", str(a.out / "prover"),
           "--plan", a.plan, *store, *a.prover_args]
    if a.prover_cpus:
        cmd = ["taskset", "-c", a.prover_cpus, *cmd]
    env = os.environ | {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"} | own
    with open(a.out / "prover.log", "w") as log:
        prover = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    stopping = threading.Event()
    session = {"k": None, "started": True, "ended": 0}
    plan = len(a.plan.split(","))

    def watch() -> None:
        """The prover exits once its plan's last session has ended (`prover.py`), else only on an error."""
        rc = prover.wait()
        if not stopping.is_set() and not (rc == 0 and session["ended"] == plan):
            emit("error", why=f"the prover exited {rc} with its plan unfinished (OUT/prover.log)")
            os._exit(3)

    threading.Thread(target=watch, daemon=True).start()
    sock, peer = server.accept()
    conn = wire.Conn(sock)
    send, recv = conn.send, conn.recv

    def sent(head: dict, body: bytes = b"") -> None:
        if head.get("type") == "end":
            session["ended"] += 1
        send(head, body)
        if head.get("type") == "session":
            session.update(k=head["k"], started=False)
        elif head.get("type") == "coin" and not session["started"]:
            session["started"] = True
            emit("started", k=session["k"])

    def got():
        head, body, stamp = recv()
        if head.get("type") == "closed":
            emit("passes", k=head.get("k"), at_ns=stamp)
        return head, body, stamp

    conn.send, conn.recv = sent, got
    v = V.Verifier(conn, source=os.urandom(32), windows=1, window_ns=10 ** 9, samples=2773, delta=2.0 ** -40,
                   jitter_ns=round(a.jitter_ms * 1e6), log=lambda m: print(m, flush=True), hbm_bytes=a.hbm_bytes,
                   lean_exe=a.lean_exe)
    try:
        reg, cal = v.register()
    except (audit.RegistryError, ValueError, wire.Closed) as e:
        stopping.set()
        emit("error", why=f"registration: {type(e).__name__}: {e}")
        v.stop()
        prover.kill()
        return 3
    rtt = v.ping(a.pings)
    allowance = v.allowance(rtt, a.quantile)
    if not a.pace_pass_ms and "pass_ns" not in cal:
        stopping.set()
        emit("error", why="the prover's calibration has no pass_ns (the CPU backend's): give --pace-pass-ms")
        v.stop()
        prover.kill()
        return 3
    pass_ms = a.pace_pass_ms or cal["pass_ns"]["p50"] / 1e6
    results: list[dict] = []

    def record() -> None:
        (a.out / "verifier.json").write_text(json.dumps(
            {"schema": "verity/remote-session/verifier/v0", "peer": list(peer), "registration": wire.registration_json(reg),
             "verdict": "lean", "calibration": cal, "rtt": rtt, "allowance_ns": allowance, "sessions": results,
             "registry": v.registry_json()}, indent=1))

    record()
    emit("ready", blocks=reg.blocks, gib=a.gib, calibration=cal, rtt=rtt, allowance_ns=allowance, pass_ms=pass_ms,
         prover=cmd, prover_env=own)
    rc = 0
    for raw in sys.stdin:
        msg = json.loads(raw)
        if msg["op"] == "stop":
            break
        try:
            r = v.session(msg["k"], allowance, v.clock.now(), V.pace_of(msg["paced"], pass_ms))
        except (wire.Closed, audit.SessionError) as e:
            emit("error", why=f"session {msg['k']}: {type(e).__name__}: {e}")
            rc = 3
            break
        results.append(r)
        record()
        emit("ended", **lines(r))
    stopping.set()
    try:
        closed = v.close_registration(reg.node)
        record()
    except audit.RegistryError as e:
        closed, rc = {"error": f"{type(e).__name__}: {e}"}, 3
    with contextlib.suppress(OSError):
        conn.send({"type": "bye"})
    v.stop()
    emit("stopped", closed=closed, prover_rc=prover.wait())
    return rc


if __name__ == "__main__":
    sys.exit(main())
