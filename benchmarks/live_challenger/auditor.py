"""The auditor host's challenger service (vy-challenger-1; RT4-INT-17, -18): `serve.py` runs only as the `auditor` user, from a
tree infra pinned, one session a window, each recorded in a signed append-only ledger before its first coin.

    vy-challenger-gate open --period P --commit HEX --peer NAME --scheme S --families JSON [--epoch E] [--minutes M] [--lane L]
    vy-challenger-gate status SESSION | ledger | pubkey | fetch SESSION

- **Who reaches it.** `research`'s ssh key on the host is forced to `vy-challenger-gate` (`command=` in its authorized_keys),
  which is this file's `gate` as root through one sudoers line, so a lane names a window and gets a session; it never names
  code. Infra's `vy-admin` runs the same gate. Nothing else runs as `research` there, and nothing runs as `auditor` but the
  session unit and `admit`.
- **What runs.** `/opt/vy-challenger/current` -> `/opt/vy-challenger/<commit>`, a root-owned `git archive` of that commit
  (`pin.sh`); the commit is the directory's name, and every entry names it (`pinned`). The key (`/var/lib/vy-challenger/key`,
  serve.py's layout) and every session's `out/` are `auditor`'s, in a directory only `auditor` and root can enter.
- **A window** is `(period, commit)`: the period the lane names and the served Commit's root (128 hex digits). `admit`
  appends the window's `open` entry, under the ledger's lock, or refuses it when the ledger already has one: a window gets
  one session, whatever became of it. The `open` entry is signed and fsynced before `serve.py` starts, so before its first
  coin. Its `host` is the string the challenger's own record signs (`<host>.<session>`), which ties `record.json` to it.
- **An `end` entry** follows when `serve.py` exits: its exit code, whether its client closed it, and the SHA-512 of each file
  of `out/`, which seals the release log. A session with an `open` and no `end` never ended (its host stopped): a refused
  window to the verifier, like one whose `closed` is false.
- **The ledger** (`/var/lib/vy-challenger/windows.jsonl`, `chattr +a`) is one JSON object a line: `schema`, `seq` (from 0),
  `prev` (SHA-512 of the previous line's bytes, 128 zeros first), `kind`, `session`, `body` and `sig`, an LMS signature under
  the challenger key over `entry_message(I, entry)`. `verify_ledger` is the reference check. Infra publishes the ledger and
  each ended session to the store (`publish.sh`).

Every signature of the key (`admit`'s, `serve.py`'s record, `end`'s) is taken under `sign.lock`, so `lms.LeafState`, which
refuses a second signer, never sees two.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
sys.path.insert(0, str(TREE))

SCHEMA = "verity/challenger-auditor/v1"
ROLE = "verity/challenger-auditor-record/v1"
TAG = "verity/challenger-auditor/entry/v1"
ZERO = "0" * 128
PORTS = range(7811, 7900)
MINUTES = (1, 720)
COMMIT = re.compile(r"^[0-9a-f]{128}$")
PINNED = re.compile(r"^[0-9a-f]{40}$")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
SCHEME = re.compile(r"^[a-z0-9][a-z0-9/._-]{0,127}$")
SESSION = re.compile(r"^s\d{8}T\d{6}Z-[0-9a-f]{4}$")
OUT_FILES = ("record.json", "key.json", "releases.jsonl", "challenger.json", "address.json", "tunnel.log", "serve.log")


class Refused(Exception):
    """A request the gate or the session refuses, with why."""


@dataclass(frozen=True)
class Host:
    """Where the service keeps its state, and the name its records give the host."""

    root: Path = Path("/var/lib/vy-challenger")
    peers: Path = Path("/etc/vy/peers")
    name: str = "vy-challenger-1"

    @property
    def ledger(self) -> Path:
        return self.root / "windows.jsonl"

    @property
    def sessions(self) -> Path:
        return self.root / "sessions"

    @property
    def key(self) -> Path:
        return self.root / "key"


def utc(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pinned_commit(tree: Path = TREE) -> str:
    """The commit this tree was pinned at: its directory's name under /opt/vy-challenger."""
    if not PINNED.match(tree.name):
        raise Refused(f"{tree} is not a pinned tree (/opt/vy-challenger/<commit>, pin.sh): no session runs from it")
    return tree.name


# --- the ledger ---------------------------------------------------------------------------------------------------------------

def _field(b: bytes) -> bytes:
    return len(b).to_bytes(8, "big") + b


def entry_message(I: bytes, entry: dict) -> bytes:
    """What an entry's `sig` signs: SHA-512([ROLE] || I || [identity_digest_sha512(TAG, entry without sig)]), each [x] x
    prefixed with its length (u64), as `challenger.record_message` frames the challenger's record under its own role."""
    from verity.ml.commitments.identity import identity_digest_sha512
    if len(I) != 16:
        raise ValueError(f"an LMS identifier is 16 bytes, not {len(I)}")
    body = {k: v for k, v in entry.items() if k != "sig"}
    return hashlib.sha512(_field(ROLE.encode()) + I + _field(identity_digest_sha512(TAG, body))).digest()


def read_ledger(path: Path) -> list[bytes]:
    try:
        return [ln for ln in path.read_bytes().split(b"\n") if ln]
    except FileNotFoundError:
        return []


def entries(lines: list[bytes]) -> list[dict]:
    return [json.loads(ln) for ln in lines]


def sessions_of(lines: list[bytes]) -> dict[str, dict[str, dict]]:
    """session -> {"open": entry, "end": entry} as far as the ledger has them."""
    out: dict[str, dict[str, dict]] = {}
    for e in entries(lines):
        out.setdefault(e["session"], {})[e["kind"]] = e
    return out


def window_of(e: dict) -> tuple[str, str]:
    return e["body"]["window"]["period"], e["body"]["window"]["commit"]


@contextmanager
def locked(path: Path) -> Iterator[None]:
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def append(host: Host, kind: str, session: str, body_of: Callable[[list[bytes]], dict], I: bytes,
           sign: Callable[[bytes], bytes]) -> dict:
    """Append one signed entry, its body `body_of` the ledger as it stands (which may refuse), under the ledger's lock;
    fsynced."""
    with locked(host.root / "ledger.lock"):
        lines = read_ledger(host.ledger)
        entry = {"schema": SCHEMA, "seq": len(lines), "prev": hashlib.sha512(lines[-1]).hexdigest() if lines else ZERO,
                 "kind": kind, "session": session, "body": body_of(lines)}
        with locked(host.root / "sign.lock"):
            entry["sig"] = sign(entry_message(I, entry)).hex()
        fd = os.open(host.ledger, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, json.dumps(entry, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)
    return entry


def verify_ledger(lines: list[bytes], I: bytes, verify: Callable[[bytes, bytes], bool]) -> dict[tuple[str, str], dict]:
    """The reference check of a published ledger under the challenger key: every entry signed, `seq` from 0, `prev` chained,
    one `open` a window and one `end` a session, after its `open`. Returns window -> {"open", "end" or None}; raises
    ValueError at the first entry that fails."""
    windows: dict[tuple[str, str], dict] = {}
    by_session: dict[str, tuple[str, str]] = {}
    prev = ZERO
    for i, ln in enumerate(lines):
        e = json.loads(ln)
        if e.get("schema") != SCHEMA or e.get("seq") != i or e.get("prev") != prev:
            raise ValueError(f"entry {i}: not entry {i} of an {SCHEMA} ledger chained to its predecessor")
        if not verify(entry_message(I, e), bytes.fromhex(e["sig"])):
            raise ValueError(f"entry {i}: its signature doesn't verify under the challenger key")
        if e["kind"] == "open":
            w = window_of(e)
            if w in windows or e["session"] in by_session:
                raise ValueError(f"entry {i}: a second open for window {w} or session {e['session']}")
            windows[w], by_session[e["session"]] = {"open": e, "end": None}, w
        elif e["kind"] == "end":
            w = by_session.get(e["session"])
            if w is None or windows[w]["end"] is not None:
                raise ValueError(f"entry {i}: an end for session {e['session']}, which has no open or has ended")
            windows[w]["end"] = e
        else:
            raise ValueError(f"entry {i}: kind {e['kind']!r}")
        prev = hashlib.sha512(ln).hexdigest()
    return windows


# --- the key (serve.py's) -----------------------------------------------------------------------------------------------------

def load_key(host: Host, params=None, workers: int | None = None):
    """The challenger's key and leaf state in `host.key`, made there when absent (serve.py's `load_key`)."""
    sys.path.insert(0, str(HERE))
    import serve
    from verity.ml.crypto import lms
    return serve.load_key(host.key, params or lms.DEPLOYED, workers or len(os.sched_getaffinity(0)))


def key_signer(host: Host, params=None, workers: int | None = None):
    """(I, sign, key_sha512) for `append`; each signature opens the key's leaf state and closes it after."""
    from verity.ml.crypto import lms
    key, state = load_key(host, params, workers)
    state.close()
    public = lms.public_key(key)

    def sign(message: bytes) -> bytes:
        with lms.LeafState(host.key / "state.json", public, key.params.H) as st:
            return lms.sign(key, st, message).encode()
    return public[:16], sign, hashlib.sha512(public).hexdigest()


# --- the gate (root) ----------------------------------------------------------------------------------------------------------

def peers(host: Host) -> list[str]:
    names = [ln.split()[0] for ln in host.peers.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return [n for n in names if n != host.name]


def parse_open(argv: list[str], host: Host) -> dict:
    p = argparse.ArgumentParser(prog="vy-challenger-gate open", exit_on_error=False, add_help=False)
    for flag in ("--period", "--commit", "--peer", "--scheme", "--families"):
        p.add_argument(flag, required=True)
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--minutes", type=int, default=120)
    p.add_argument("--lane", default=None)
    try:
        a = p.parse_args(argv)
    except (argparse.ArgumentError, SystemExit) as e:
        raise Refused(f"open: {e}; usage: {p.format_usage().strip()}") from None
    if not NAME.match(a.period):
        raise Refused(f"--period {a.period!r}: 1 to 64 of [A-Za-z0-9._:-], starting with a letter or digit")
    if not COMMIT.match(a.commit):
        raise Refused("--commit: the served Commit's root, 128 lowercase hex digits (a SHA-512)")
    if a.peer not in peers(host):
        raise Refused(f"--peer {a.peer!r}: one of {peers(host)} (/etc/vy/peers)")
    if not SCHEME.match(a.scheme):
        raise Refused(f"--scheme {a.scheme!r}: [a-z0-9/._-], up to 128")
    try:
        families = json.loads(a.families)
    except json.JSONDecodeError:
        families = None
    if (not isinstance(families, list) or not families or len(a.families) > 4096
            or not all(isinstance(f, list) and len(f) == 2 and isinstance(f[0], str) and SCHEME.match(f[0])
                       and type(f[1]) is int and f[1] > 0 for f in families)):
        raise Refused("--families: a JSON list of [label, count] with count >= 1, up to 4096 bytes")
    if not 0 <= a.epoch < 2 ** 63:
        raise Refused("--epoch: 0 <= E < 2^63")
    if not MINUTES[0] <= a.minutes <= MINUTES[1]:
        raise Refused(f"--minutes: {MINUTES[0]} to {MINUTES[1]}")
    if a.lane is not None and not NAME.match(a.lane):
        raise Refused(f"--lane {a.lane!r}: 1 to 64 of [A-Za-z0-9._:-]")
    return {"window": {"period": a.period, "commit": a.commit}, "peer": a.peer, "scheme": a.scheme, "families": families,
            "epoch": a.epoch, "minutes": a.minutes, "lane": a.lane}


def free_port(lines: list[bytes], active: Callable[[str], bool]) -> int:
    busy = {s["open"]["body"]["port"] for sid, s in sessions_of(lines).items() if "end" not in s and active(sid)}
    for port in PORTS:
        if port not in busy:
            return port
    raise Refused(f"every port of {PORTS.start}..{PORTS.stop - 1} has a live session")


def gate_open(argv: list[str], host: Host, *, by: str, now: datetime, token: str, admit: Callable[[str], None],
              start: Callable[[str], None], own: Callable[[Path], None]) -> dict:
    """Validate a request, hand it to `admit` (as `auditor`) and `start` its unit: the session id, its port on the peer."""
    req = parse_open(argv, host)
    w = (req["window"]["period"], req["window"]["commit"])
    if any(window_of(s["open"]) == w for s in sessions_of(read_ledger(host.ledger)).values() if "open" in s):
        raise Refused(f"window {req['window']} already has its session in the ledger: one session a window")
    session = f"s{now.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{token}"
    d = host.sessions / session
    d.mkdir(mode=0o755)
    req.update(session=session, by=by, requested_utc=utc(now))
    (d / "request.json").write_text(json.dumps(req, indent=1, sort_keys=True) + "\n")
    own(d)
    with locked(host.root / "gate.lock"):
        admit(session)
        start(session)
    entry = sessions_of(read_ledger(host.ledger))[session]["open"]
    return {"session": session, "port": entry["body"]["port"], "peer": req["peer"], "host": entry["body"]["host"],
            "seq": entry["seq"]}


def gate_status(session: str, host: Host, active: Callable[[str], bool]) -> dict:
    if not SESSION.match(session):
        raise Refused(f"no session {session!r}")
    s = sessions_of(read_ledger(host.ledger)).get(session)
    if s is None:
        raise Refused(f"no session {session} in the ledger")
    return {"session": session, "open": s.get("open"), "end": s.get("end"), "active": "end" not in s and active(session)}


def gate_fetch(session: str, host: Host) -> bytes:
    """The session's directory (its request and out/) as a gzipped tar."""
    gate_status(session, host, lambda _: False)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(host.sessions / session, arcname=session)
    return buf.getvalue()


def gate(argv: list[str], host: Host = Host(), **hooks: Any) -> tuple[int, bytes]:
    """The gate's verbs; (exit code, stdout). A refusal is exit 3 with {"refused": why}."""
    if len(argv) == 2 and argv[0] == "--ssh":
        if len(argv[1]) > 8192:
            return 3, json.dumps({"refused": "the request is over 8192 bytes"}).encode() + b"\n"
        try:
            argv = shlex.split(argv[1])
        except ValueError as e:
            return 3, json.dumps({"refused": f"the request doesn't parse as shell words: {e}"}).encode() + b"\n"
    verb, rest = (argv[0], argv[1:]) if argv else ("", [])
    active = hooks.get("active") or unit_active
    try:
        if verb == "open":
            doc = gate_open(rest, host, by=hooks.get("by") or os.environ.get("SUDO_USER") or "root",
                            now=hooks.get("now") or datetime.now(timezone.utc), token=hooks.get("token") or secrets.token_hex(2),
                            admit=hooks.get("admit") or as_auditor_admit, start=hooks.get("start") or start_unit,
                            own=hooks.get("own") or own_by_auditor)
        elif verb == "status" and len(rest) == 1:
            doc = gate_status(rest[0], host, active)
        elif verb == "ledger" and not rest:
            return 0, host.ledger.read_bytes() if host.ledger.exists() else b""
        elif verb == "pubkey" and not rest:
            doc = json.loads((host.root / "pubkey.json").read_text())
        elif verb == "fetch" and len(rest) == 1:
            return 0, gate_fetch(rest[0], host)
        else:
            raise Refused("verbs: open --period P --commit HEX --peer NAME --scheme S --families JSON [--epoch E] "
                          "[--minutes M] [--lane L] | status SESSION | ledger | pubkey | fetch SESSION")
    except Refused as e:
        return 3, json.dumps({"refused": str(e)}).encode() + b"\n"
    return 0, json.dumps(doc, sort_keys=True).encode() + b"\n"


def _auditor(args: list[str]) -> list[str]:
    return ["runuser", "-u", "auditor", "--", "/usr/bin/env", "-i", "PYTHONDONTWRITEBYTECODE=1", "HOME=/var/lib/vy-challenger",
            "PATH=/usr/bin:/bin", "/usr/bin/python3", "-I", str(Path(__file__).resolve()), *args]


def as_auditor_admit(session: str) -> None:
    p = subprocess.run(_auditor(["admit", session]), capture_output=True, text=True)
    if p.returncode:
        raise Refused(p.stderr.strip().splitlines()[-1] if p.stderr.strip() else f"admit exited {p.returncode}")


def own_by_auditor(d: Path) -> None:
    import pwd
    pw = pwd.getpwnam("auditor")
    os.chown(d, pw.pw_uid, pw.pw_gid)


def start_unit(session: str) -> None:
    """Blocks until the unit runs (Type=exec), so the next `admit`, under the gate's lock, sees it active."""
    p = subprocess.run(["systemctl", "start", f"vy-challenger@{session}.service"], capture_output=True, text=True)
    if p.returncode:
        raise Refused(f"session {session} was admitted but its unit didn't start ({p.stderr.strip()}): its window is spent, "
                      f"an open entry with no end")


def unit_active(session: str) -> bool:
    return subprocess.run(["systemctl", "is-active", "--quiet", f"vy-challenger@{session}.service"]).returncode == 0


# --- the session (auditor) ----------------------------------------------------------------------------------------------------

def request_of(host: Host, session: str) -> dict:
    if not SESSION.match(session):
        raise Refused(f"no session {session!r}")
    return json.loads((host.sessions / session / "request.json").read_text())


def admit(host: Host, session: str, *, signer, pinned: str, now: datetime, active: Callable[[str], bool]) -> dict:
    """The window's `open` entry, with a port no live session holds, or Refused when the ledger already has one for it."""
    req = request_of(host, session)
    I, sign, key_sha512 = signer
    w = (req["window"]["period"], req["window"]["commit"])

    def body_of(lines: list[bytes]) -> dict:
        for sid, s in sessions_of(lines).items():
            if "open" in s and (window_of(s["open"]) == w or sid == session):
                raise Refused(f"window {req['window']} already has session {sid} in the ledger: one session a window")
        return {"window": req["window"], "pinned": pinned, "key_sha512": key_sha512, "host": f"{host.name}.{session}",
                "peer": req["peer"], "port": free_port(lines, active), "scheme": req["scheme"], "epoch": req["epoch"],
                "families": req["families"], "minutes": req["minutes"], "lane": req["lane"], "by": req["by"],
                "started_utc": utc(now)}
    return append(host, "open", session, body_of, I, sign)


def serve_argv(host: Host, session: str, open_body: dict, tree: Path = TREE) -> list[str]:
    return [sys.executable, "-I", str(tree / "benchmarks" / "live_challenger" / "serve.py"),
            "--out", str(host.sessions / session / "out"), "--port", str(open_body["port"]), "--scheme", open_body["scheme"],
            "--families", json.dumps(open_body["families"]), "--epoch", str(open_body["epoch"]), "--peer", open_body["peer"],
            "--peers", str(host.peers), "--key", str(host.key), "--host", open_body["host"]]


def run(host: Host, session: str, *, signer, pinned: str, now: Callable[[], datetime], argv: list[str] | None = None,
        wait: Callable[[subprocess.Popen, float], int] | None = None) -> dict:
    """Serve the admitted session (serve.py, holding `sign.lock` until it listens or exits), then append its `end` entry."""
    I, sign, _ = signer
    s = sessions_of(read_ledger(host.ledger)).get(session, {})
    if "open" not in s or "end" in s:
        raise Refused(f"session {session} has no open entry, or has ended: nothing to serve")
    body = s["open"]["body"]
    out = host.sessions / session / "out"
    rc, why = 2, None
    if body["pinned"] != pinned:
        why = f"admitted at {body['pinned']}, but this tree is {pinned}: the pin moved, so the session doesn't run"
    else:
        out.mkdir(mode=0o755, exist_ok=True)
        with open(out / "serve.log", "wb") as log, locked(host.root / "sign.lock"):
            proc = subprocess.Popen(argv or serve_argv(host, session, body), stdout=subprocess.PIPE, stderr=log,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            first = proc.stdout.readline()
        with open(out / "stdout.log", "wb") as f:
            f.write(first)
        rc = (wait or _wait)(proc, body["minutes"] * 60)
        proc.stdout.close()
    summary = {}
    try:
        summary = json.loads((out / "challenger.json").read_text())
    except (OSError, ValueError):
        pass
    files = {n: hashlib.sha512((out / n).read_bytes()).hexdigest() for n in OUT_FILES if (out / n).is_file()}
    end = {"rc": rc, "closed": rc == 0 and summary.get("ended") == "closed by the client",
           "ended": why or summary.get("ended") or f"serve.py exited {rc}", "files": files, "ended_utc": utc(now())}
    return append(host, "end", session, lambda lines: end, I, sign)


def _wait(proc: subprocess.Popen, seconds: float) -> int:
    try:
        return proc.wait(seconds)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        return proc.wait()


# --- main ---------------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    host = Host()
    if argv[:1] == ["gate"]:
        rc, out = gate(argv[1:], host)
        sys.stdout.buffer.write(out)
        return rc
    try:
        if argv[:1] == ["admit"] and len(argv) == 2:
            e = admit(host, argv[1], signer=key_signer(host), pinned=pinned_commit(), now=datetime.now(timezone.utc),
                      active=unit_active)
            print(json.dumps({"seq": e["seq"], "session": e["session"]}))
        elif argv[:1] == ["run"] and len(argv) == 2:
            e = run(host, argv[1], signer=key_signer(host), pinned=pinned_commit(), now=lambda: datetime.now(timezone.utc))
            print(json.dumps({"seq": e["seq"], "session": e["session"], "rc": e["body"]["rc"]}))
        elif argv[:1] == ["keygen"] and len(argv) == 1:
            from verity.ml.crypto import lms
            key, state = load_key(host)
            state.close()
            public = lms.public_key(key)
            doc = {"W": key.params.W, "H": key.params.H, "public": public.hex(),
                   "public_sha512": hashlib.sha512(public).hexdigest()}
            (host.root / "pubkey.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
            print(json.dumps(doc))
        else:
            print("usage: auditor.py gate VERB ... | admit SESSION | run SESSION | keygen", file=sys.stderr)
            return 2
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
