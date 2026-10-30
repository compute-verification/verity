"""Infra's challenger gate (`benchmarks/live_challenger/auditor.py`) run on one node
under a test key, for a rehearsal of a served window (`pouw_h1_rehearsal.py`); run by hand or by that job, never collected.

    python -I local_gate.py DIR init --tree TGZ --commit SHA [--params W,H] [--peer NAME] [--ports LO-HI] [--coins lean|stub]
    python -I local_gate.py DIR VERB ...   # the gate's verbs as the lane's key: reserve, open, status, ledger, pins,
                                           # pubkey, fetch, period-open, period-close, model-key, models
                                           # (`auditor.gate(["--ssh", WORDS, "--key", LANE])`)
    python -I local_gate.py DIR admin VERB ...   # vy-admin's (`auditor.gate([VERB, ...])`): model-register
    python -I local_gate.py DIR wait SESSION   # blocks until the session's unit has exited; its `run` answer

``init`` unpacks infra's tree (``git archive SHA`` of `ARCHIVE`'s paths, refused without one) as ``DIR/tree/SHA``, so
`auditor.pinned_commit` names SHA, makes the host's state in ``DIR/host`` (`auditor.Host`, its pins log recording SHA as
its own main, since nothing here fetches main) and its LMS key at ``--params``
(default 4,5: 32 leaves), and builds the Lean verifier whose `one_stage.verifier.coins` serve.py flips its coins from
(``--coins lean``; ``stub`` serves `stub_serve.py`'s seeded bytes instead, and ``init`` falls back to it, saying why, when
the build fails). Every verb then runs infra's own code from that tree, which goes first on ``sys.path`` before any
``verity`` import: `auditor.gate` with its hooks as root and `auditor` would run them on the auditor host (`reserve`,
`admit`, `period`, `await_commitment`, `model_register`), each in this process, and a session's unit (`auditor.run`,
serve.py with ``--root`` and the session's secret) a detached ``DIR run SESSION`` that holds ``DIR/units/SESSION.lock``
while it runs, which is the gate's ``active``. The model registry is infra's (`models.py` of the tree): ``admin
model-register`` reads the checkpoint role's manifest and core's pin from the tree, so the archive carries both
checkpoint manifests (vLLM's and the torch loop's) beside ``verity``. What differs from the auditor host: one account
runs everything (no ``auditor`` user, ``runuser``, sudoers or systemd unit, so ``own`` does nothing and the registry isn't
made append-only); the key is a test key of ``--params``, not ``lms.DEPLOYED``; serve.py
listens on the node itself (no ``--peer``, so no ``ssh -R`` tunnel), on ``--ports`` (default 27811-27899, off the live
tunnels' 7811-7899); and with ``--coins stub`` its coins are seeded bytes, not the Lean verifier's.
"""
from __future__ import annotations

import fcntl
import json
import os
import shlex
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

CONFIG = "gate.json"
#: what the tree's archive holds: infra's code and core, and every checkpoint manifest `models.MANIFESTS` reads, so a role
#: of any registers
ARCHIVE = ("verity", "benchmarks/live_challenger", "integrations/vllm/manifests/checkpoints-int8.json")


def _config(d: Path) -> dict:
    return json.loads((d / CONFIG).read_text())


def _auditor(d: Path, cfg: dict):
    """Infra's auditor.py from the pinned tree, which puts that tree first on sys.path (no `verity` imported before it)."""
    assert "verity" not in sys.modules, "verity was imported before infra's tree went on sys.path"
    sys.path.insert(0, str(Path(cfg["tree"]) / "benchmarks" / "live_challenger"))
    import auditor
    lo, hi = cfg["ports"]
    auditor.PORTS = range(lo, hi + 1)
    return auditor


def _host(A, d: Path, cfg: dict):
    return A.Host(root=d / "host", peers=d / "host" / "peers", name=cfg["name"], pins=d / "host" / "pins.jsonl")


def _params(cfg: dict):
    from verity.ml.crypto import lms
    return lms.Params(64, *cfg["params"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _lock(d: Path, session: str) -> Path:
    return d / "units" / f"{session}.lock"


def _active(d: Path):
    def active(session: str) -> bool:
        path = _lock(d, session)
        if not path.exists():
            return False
        with open(path, "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(f, fcntl.LOCK_UN)
            return False
    return active


def _start(d: Path, A):
    """`auditor.start_unit`: returns once the unit holds its lock (systemd's Type=exec), so the gate sees it active."""
    def start(session: str) -> None:
        (d / "units").mkdir(exist_ok=True)
        log = open(d / "units" / f"{session}.log", "ab")
        p = subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), str(d), "run", session],
                             stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, stderr=log, start_new_session=True)
        line = p.stdout.readline()
        p.stdout.close()
        if line != b"running\n":
            raise A.Refused(f"session {session} was admitted but its unit didn't start (see units/{session}.log): its "
                            "window is spent, an open entry with no end")
    return start


def init(d: Path, argv: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="local_gate.py DIR init")
    p.add_argument("--tree", required=True, help=f"git archive SHA {' '.join(ARCHIVE)}, gzipped")
    p.add_argument("--commit", required=True)
    p.add_argument("--params", default="4,5")
    p.add_argument("--peer", default="local")
    p.add_argument("--ports", default="27811-27899")
    p.add_argument("--coins", choices=("lean", "stub"), default="lean")
    p.add_argument("--name", default="local-gate")
    a = p.parse_args(argv)
    tree = d / "tree" / a.commit
    tree.mkdir(parents=True)
    with tarfile.open(a.tree) as t:
        t.extractall(tree, filter="data")
    if missing := [p for p in ARCHIVE if not (tree / p).exists()]:
        print(f"local_gate: the tree {a.tree} has no {', '.join(missing)}: make it with "
              f"`git archive --format=tar.gz -o TGZ {a.commit} {' '.join(ARCHIVE)}`", file=sys.stderr)
        return 2
    (d / "host").mkdir(mode=0o700)
    (d / "host" / "sessions").mkdir()
    (d / "units").mkdir()
    (d / "host" / "peers").write_text(f"{a.peer} localhost\n")
    (d / "host" / "coin-ports").write_text(f"{a.peer} {a.ports}\n")
    cfg = {"tree": str(tree), "commit": a.commit, "params": [int(x) for x in a.params.split(",")], "peer": a.peer,
           "ports": [int(x) for x in a.ports.split("-")], "coins": a.coins, "name": a.name}
    if a.coins == "lean":
        pkg = tree / "verity" / "core" / "service" / "sampled_proofs"
        b = subprocess.run(["lake", "build", "sampled-proofs-verifier"], cwd=pkg, capture_output=True, text=True)
        cfg["lean_build"] = {"rc": b.returncode, "tail": (b.stdout + b.stderr)[-3000:]}
        if b.returncode:
            cfg["coins"], cfg["coins_why"] = "stub", f"lake build sampled-proofs-verifier exited {b.returncode}"
    (d / CONFIG).write_text(json.dumps(cfg, indent=1) + "\n")
    A = _auditor(d, cfg)
    from verity.ml.crypto import lms
    host = _host(A, d, cfg)
    A.record_pin(host.pins, a.commit, a.commit, _now(), seal=lambda p: None)
    key, state = A.load_key(host, _params(cfg), 4)
    state.close()
    public = lms.public_key(key)
    doc = {"W": key.params.W, "H": key.params.H, "public": public.hex(), "public_sha512": __import__("hashlib").sha512(public).hexdigest()}
    (host.root / "pubkey.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({**{k: v for k, v in cfg.items() if k != "lean_build"}, "pubkey": doc}))
    return 0


def unit(d: Path, session: str) -> int:
    """The session's unit (`vy-challenger@SESSION.service`): `auditor.run` with serve.py's argv as the unit's, except no
    ``--peer`` (serve.py listens on this node) and the test key's ``--params``; ``stub`` coins through stub_serve.py."""
    cfg = _config(d)
    A = _auditor(d, cfg)
    host = _host(A, d, cfg)
    lock = open(_lock(d, session), "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    sys.stdout.write("running\n")
    sys.stdout.flush()
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    body = A.sessions_of(A.read_ledger(host.ledger))[session]["open"]["body"]
    argv = A.serve_argv(host, session, body)
    if "--peer" in argv:
        i = argv.index("--peer")
        del argv[i:i + 4]
    argv += ["--params", ",".join(map(str, cfg["params"])), "--keygen-workers", "4"]
    if cfg["coins"] == "stub":
        os.environ["COINS_LOG"] = str(d / "units" / f"{session}.coins")
        argv[2:2] = [str(Path(cfg["tree"]) / "benchmarks" / "live_challenger" / "tests" / "stub_serve.py")]
    signer = A.key_signer(host, _params(cfg), 4)
    try:
        e = A.run(host, session, signer=signer, pinned=A.pinned_commit(), now=_now, argv=argv)
        doc = {"seq": e["seq"], "session": e["session"], "rc": e["body"]["rc"], "argv": argv}
    except A.Refused as e:
        doc = {"refused": str(e), "argv": argv}
    (d / "units" / f"{session}.json").write_text(json.dumps(doc) + "\n")
    return 0


def wait(d: Path, session: str) -> int:
    path = _lock(d, session)
    if not path.exists():
        print(json.dumps({"refused": f"session {session} has no unit"}))
        return 3
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
    print((d / "units" / f"{session}.json").read_text().strip())
    return 0


def gate(d: Path, argv: list[str], admin: bool = False) -> int:
    cfg = _config(d)
    A = _auditor(d, cfg)
    host = _host(A, d, cfg)
    active = _active(d)
    signer = []

    def sign():
        if not signer:
            signer.append(A.key_signer(host, _params(cfg), 4))
        return signer[0]
    pinned = A.pinned_commit()
    hooks = {
        "active": active,
        "own": lambda path: None,
        "reserve": lambda s: A.reserve(host, s, signer=sign(), pinned=pinned, now=_now()),
        "admit": lambda s: A.admit(host, s, signer=sign(), pinned=pinned, now=_now(), active=active),
        "start": _start(d, A),
        "ready": lambda s: A.await_commitment(host, s, active),
        "period": lambda args: A.period(host, args, signer=sign(), pinned=pinned, now=_now(), active=active),
        "register": lambda args: A.model_register(host, args, pinned=pinned, now=_now()),
        "seal": lambda path: None,
    }
    if admin:
        rc, out = A.gate(argv, host, by="vy-admin", **hooks)
    else:
        rc, out = A.gate(["--ssh", shlex.join(argv), "--key", os.environ.get("LOCAL_GATE_LANE", "vllm")], host, **hooks)
    sys.stdout.buffer.write(out)
    return rc


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    d, verb, rest = Path(argv[0]).resolve(), argv[1], argv[2:]
    if verb == "init":
        return init(d, rest)
    if verb == "run" and len(rest) == 1:
        return unit(d, rest[0])
    if verb == "wait" and len(rest) == 1:
        return wait(d, rest[0])
    if verb == "admin":
        return gate(d, rest, admin=True)
    return gate(d, argv[1:])


if __name__ == "__main__":
    sys.exit(main())
