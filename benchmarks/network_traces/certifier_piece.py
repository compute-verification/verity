"""certifier-piece: the network certifier's piece of `served_verdicts.sh` (its ``CERTIFIER_CMD``), and the front that puts
the active warden in front of serve-openai there (its ``CERTIFIER_FRONT``).

    python certifier_piece.py front PORT SERVE_DIR OUT [--gate USER@HOST | --gate local] [--key KEY] [--peer NAME]
        [--prompts 4] [--prompt-ids 24] [--tokens 4] [--vocab 128000] [--height 10] [--minutes M] [--bucket-ms MS]
        [--warmup N]
    python certifier_piece.py piece SERVE_DIR WINDOW DIR [--eos-ids IDS] [--lean EXE] [--draw K] [--chain-cmd CMD]
        [CR_K=V ...]
    python certifier_piece.py relay --listen PORT --to PORT

* **front** runs before served_verdicts.sh's requests when ``CERTIFIER_FRONT`` names it: the gate's scan rehearsal
  (`served_session.session`, ``warden-rehearse --scan --prompts --window-per-group``) against serve-openai on PORT, one
  group of ``--prompts`` requests (at most the scan's 4 lanes) of ``--prompt-ids`` ids below ``--vocab``, each answered
  with exactly ``--tokens`` ids, drawn from a fresh secret seed. A group is sent half a bucket into link-window 0 and
  its cells are link-window 0's alone (``--tokens`` at most 4 keeps the group in one link-window), so the engine admits
  the group together and closes it into one Commit window, the run's earliest: the window the run audits (with
  ``REQUESTS=1`` the requests after it make a smaller window, which `served_verdicts.py window` passes over). The
  defaults are the window circuit's class 4,32,4 (4 slots, prompts of at most 32 ids, 4 outputs). The scan's schedule
  gives a request D·τ from its admission to its first egress cell, by which a short answer must be whole, and an
  endpoint slower than that fails each request late: 2.7 s at the scan's tick τ of 100 ms. ``--bucket-ms MS`` runs
  the scan at the tick MS instead (the gate's `auditor.SCAN_BUCKET_MS`, 100 to 4000: D·τ = 108 s at 4000), each
  sampling block T·τ long, and the minutes are counted there. ``--warmup N`` (none by default: an endpoint that
  readies its first window before it listens needs none) first sends N plain completions straight to PORT (`warmup`:
  no warden, no release header), one at a time, each of the group's shape and awaited, so the endpoint's first window
  (H1's first reservation, the kernels' first compile) is behind it; it then waits until serve-openai records each in
  `served_session.USERS`, which it does only once the request's window has closed. So each warm-up is a window of one
  request, closed before the gate is asked, which the pick passes over for the group's whenever the group's window
  holds two or more; one that fails or isn't recorded within `WARMUP_CLOSE_S` refuses the front. A warm-up doesn't
  make a slow endpoint pass; a tick whose D·τ covers its answer does. The gate's fetch,
  ``witness/`` with it, is kept under SERVE_DIR/`PRIVATE`/RUN/fetch.tar.gz (0700, the serve dir stays on the node);
  OUT gets ``session.json`` and ``bundle.tgz``, the public outcome, and the run's kept directory a copy of
  ``session.json`` (`SESSION`), whose gate the piece's placement names. ``--gate local`` is the dry run's (this
  tree's gate in this process, a `relay` to PORT its tunnel): a warden on the prover's node is no auditor's device,
  so it never stands for the gate.
  Exit as `served_session.session`'s.
* **piece** writes window WINDOW's core answer to DIR/answer.json (`verity/certifier/v0`, `NetworkCheck.Period`'s rows
  form): accept only when the chain accepted and ``network-period check`` accepted the period, whose answer it then is.
  1. The kept warden run: exactly one fetch under SERVE_DIR/`PRIVATE`, whose rehearsal the gate's check passed (its
     ``end.json``: exit 0 and accepted, as the front's exit reads it). With none the run's load didn't go through the
     front (it wasn't set, or the gate, its tunnel or ssh failed); a rehearsal that failed (each request late past the
     warden's D·τ, or cut) certified no served answer, so there is none for the protocol to reject (`rehearsed`).
  2. The window's requests are certified ones: each window request's user names one of the warden's certified
     requests (`warden_chain.io`), with the same prompt and streamed ids; else reject.
  3. The chain (`served_session.py chain`: chain_record.sh's warden path, certs, endpoint, N2, tie and scan) on the
     kept run, with the serve dir's endpoint records (`ENDPOINT`, copied to SERVE_DIR/`PRIVATE`/RUN/endpoint), its run
     dir and quarantine under SERVE_DIR/`PRIVATE`/RUN, so nothing of the witness reaches a record. N2 and the tie
     take every link-window of a run of at most 8 (`sized`). Its report's verdict ``accepted`` goes on; any other is a
     reject with its why.
  4. The period's link-window is the scan's (``answers.json``'s one window): every window request's release and
     streamed ids lie in it, else reject.
  5. The period check: `scan_period.rows_case` of the warden run and the chain's scan, checked by ``network-period``
     (``--lean``, else built in verity/core/protocols/network/lean); its answer is the piece's, an accept's profile
     with its ``placement`` (`placement`): the certifier on the gate's host, and the chain, the period check and
     their proofs' coins in this piece on the prover's node, so `trusted-host` is assumed with the auditor's devices
     not placed.
  DIR also gets the chain's public files (`PUBLISHED`) and the check's three input files in DIR/period.
  All of them and the answer are written to a fresh hold under SERVE_DIR/`PRIVATE`/held (0700) and reach DIR only
  through the record's leak guard (`leak_guard.py`), which reads DIR and the hold against the warden run's witness,
  moves what holds a piece of it to SERVE_DIR/`PRIVATE`/quarantine and only then releases the rest. The period files
  it releases become DIR/`CASE` (`write_case`), the case as `composite.py` reads a piece's: ``{"commitment",
  "evidence", "params"}``, the commitment and evidence hex and the params ``params.json``'s canonical JSON, all inline,
  since certifier/v0 reads no blob (about 83 MB on a served window, nearly all of it the params). DIR/period then
  goes, so the case has one copy; the guard reads the files before they are hex, which it searches only for a binary
  witness file's pieces. Only the protocol rejects: a step that can't run (no kept run, a rehearsal the gate's check
  didn't pass, a build, a tool or the chain failing without a report, an input missing, the leak guard failing, or
  SIGTERM from the piece's time limit) is `NotRun`, and the piece then refuses, releasing nothing from the hold: DIR
  keeps only its log and gets DIR/`REFUSED` naming why, with no answer, which `served_verdicts.py` reads as the
  certifier missing. Each step it runs (the build, the chain, the check and the leak guard) is a process group of its
  own, which the piece ends however the step ends (`run_group`: SIGTERM, `GRACE_S`, SIGKILL), so nothing a step
  started outlives it; a SIGTERM while a step runs is held until the step's group has ended, so a step never outlives
  the refusal either. ``--draw K`` is the scan's K drawn units (every unit by default); a CR_K=V goes to the chain;
  ``--chain-cmd`` replaces the chain (a test's, which writes the report and the scan's answers under
  RESEARCH_RUN_DIR/out).
* **relay**: a TCP relay from 127.0.0.1:``--listen`` to 127.0.0.1:``--to``, the local gate's tunnel.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import select
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import urllib.request
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
sys.path[:0] = [str(HERE)]
import served_session as SS  # noqa: E402

#: where the front keeps the gate's fetch and the piece its private files, under the serve dir
PRIVATE = Path("private") / "certifier"
#: the endpoint's records the chain reads (network_e2e `chain.ENDPOINT`)
ENDPOINT = ("advice_trace.jsonl", "requests.jsonl", "windows.jsonl", "serve.json")
PROTOCOL = "verity/certifier/v0"
#: the named assumption the certifier's guarantee reads (`NetworkCheck.Period`): the certifier is the auditor's device
PREMISE = "trusted-host"
#: the front's public outcome, copied beside the kept fetch
SESSION = "session.json"
#: the chain's files the piece publishes; never endpoint/report.json, whose Commit-window rows give each window's
#: request, carried and output counts and its opening time, of which only a padded count may be public
PUBLISHED = ("report/report.json", "report/report.md", "job.json", "leak-guard.json", "certs.json", "scan/summary.json",
             "scan/answers.json", "n2/summary.json", "tie/summary.json")
#: a piece's refusal to run, which `served_verdicts.py` reads as the subprotocol missing
REFUSED = "refused.txt"
#: the piece's case, as `composite.py` reads it
CASE = "case.json"
LEAN_DIR = TREE / "verity" / "core" / "protocols" / "network" / "lean"
#: the window circuit's class (4 slots, prompts of at most 32 ids, 4 outputs) and the scan's lanes
PROMPTS, PROMPT_IDS, TOKENS = 4, 24, 4
HEIGHT = 10
#: how long a warm-up request may take, and how long the front waits for serve-openai to record the warm-ups'
#: windows closed
WARMUP_REQUEST_S, WARMUP_CLOSE_S = 600, 600
#: how long a step's process group has after SIGTERM before SIGKILL: inside `timeout -k`'s minute and the research
#: runner's grace, so the piece has ended its steps before anything ends the piece; and how often it looks
GRACE_S, POLL_S = 5, 0.05
#: the SIGTERMs that came while a step ran (`run_group`), which `ended` holds until the step's group has ended
_held: list[int] | None = None


def answer(verdict: str, reason: str) -> dict:
    """A refusal in the form `Answer.toJson` writes."""
    return {"commitment": None, "profile": None, "protocol": PROTOCOL, "public": None, "reason": reason[:2000],
            "verdict": verdict}


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.is_file() else []


# ---------------------------------------------------------------------------------------------------------------- front

def front(a: argparse.Namespace) -> int:
    serve_dir, out = a.serve_dir.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    keep = serve_dir / PRIVATE
    keep.mkdir(parents=True, exist_ok=True)
    for p in (serve_dir / "private", keep):
        p.chmod(0o700)
    a.requests = a.prompts
    a.keep, a.out, a.at, a.window_per_group = keep, out, None, True
    if a.minutes is None:
        tick = a.bucket_ms or SS.A.SCAN["--bucket-ms"]
        a.minutes = SS.A.warden_group_minutes(1, a.tokens, a.height, tick) + SS.MARGIN
    if a.gate == "local":
        a.tunnel = lambda ssh, port: [sys.executable, str(Path(__file__).resolve()), "relay", "--listen", str(port),
                                      "--to", str(a.port)]
    try:
        if a.prompts > SS.A.SCAN["--lanes"]:
            raise SS.Refused(f"--prompts {a.prompts}: one group, at most the scan's {SS.A.SCAN['--lanes']} lanes")
        if SS.A.scan_group_windows(a.tokens) != 1:
            raise SS.Refused(f"--tokens {a.tokens}: a group past one link-window, which the period's rows form refuses")
        if a.warmup < 0:
            raise SS.Refused(f"--warmup {a.warmup}: a count of requests, 0 for none")
        ps = SS.prompts(a.prompts, a.prompt_ids, a.vocab, secrets.token_bytes(32))
        data = json.dumps(ps, separators=(",", ":")).encode()
        SS.check(a, data)
        if a.warmup:
            warm = warmup(a, serve_dir)
            (out / "warmup.json").write_text(json.dumps({"requests": warm}, indent=1) + "\n")
        rc = SS.session(a, data, ps)
        if (doc := out / "session.json").is_file():
            shutil.copyfile(doc, keep / json.loads(doc.read_text())["run"] / SESSION)
        return rc
    except SS.Refused as e:
        (out / "session.json").write_text(json.dumps({"refused": str(e)}, indent=1) + "\n")
        print(f"refused: {e}", file=sys.stderr)
        return e.code


def warmup(a: argparse.Namespace, serve_dir: Path, *, clock: Callable[[], float] = time.monotonic,
           sleep: Callable[[float], object] = time.sleep) -> list[dict]:
    """``--warmup``'s requests (the module docstring), each its user, finish and seconds; `served_session.Refused`
    when one fails or serve-openai hasn't recorded every one's window closed within `WARMUP_CLOSE_S`."""
    tag, done = secrets.token_hex(4), []
    for i, p in enumerate(SS.prompts(a.warmup, a.prompt_ids, a.vocab, secrets.token_bytes(32))):
        body = {"user": f"warmup-{tag}-{i}", "prompt": p, "max_tokens": a.tokens, "temperature": 0.0, "top_p": 1.0}
        req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/completions", json.dumps(body).encode(),
                                     {"content-type": "application/json"})
        t = clock()
        try:
            with urllib.request.urlopen(req, timeout=WARMUP_REQUEST_S) as r:  # noqa: S310
                got = json.loads(r.read())
        except (OSError, ValueError) as e:
            raise SS.Refused(f"--warmup: request {i}: {type(e).__name__}: {e}") from None
        done.append({"user": body["user"], "finish_reason": ((got.get("choices") or [{}])[0]).get("finish_reason"),
                     "seconds": round(clock() - t, 3)})
        print(f"warm-up {i + 1}/{a.warmup}: {done[-1]}", flush=True)
    users, until = {d["user"] for d in done}, clock() + WARMUP_CLOSE_S
    while left := users - {r.get("user") for r in _lines(serve_dir / SS.USERS)}:
        if clock() > until:
            raise SS.Refused(f"--warmup: serve-openai recorded no closed window of {sorted(left)} in "
                             f"{serve_dir / SS.USERS} within {WARMUP_CLOSE_S} s")
        sleep(1)
    return done


def relay(listen: int, to: int, *, ready: Callable[[], None] = lambda: None) -> None:
    srv = socket.create_server(("127.0.0.1", listen))
    ready()

    def pipe(src: socket.socket, dst: socket.socket) -> None:
        try:
            while chunk := src.recv(1 << 16):
                dst.sendall(chunk)
        except OSError as e:
            print(f"relay {listen} to {to}: {e!r}, closing the connection", file=sys.stderr, flush=True)
        finally:
            for s in (src, dst):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError as e:
                    print(f"relay {listen} to {to}: shutdown: {e!r}", file=sys.stderr, flush=True)

    while True:
        conn, _ = srv.accept()
        try:
            up = socket.create_connection(("127.0.0.1", to))
        except OSError:
            conn.close()
            continue
        for x, y in ((conn, up), (up, conn)):
            threading.Thread(target=pipe, args=(x, y), daemon=True).start()


# ---------------------------------------------------------------------------------------------------------------- piece

class NotRun(Exception):
    """The piece couldn't run to a verdict for a reason outside the protocol: a build, a tool or an input failed."""


class Rejected(Exception):
    """The window isn't what the certifier certified."""


def kept_run(serve_dir: Path) -> str:
    runs = sorted(p.parent.name for p in (serve_dir / PRIVATE).glob("*/fetch.tar.gz"))
    if len(runs) != 1:
        raise NotRun(f"{len(runs)} warden runs kept under {serve_dir / PRIVATE} ({runs[:4]}), not one: the run's "
                        "load goes through the warden only with CERTIFIER_FRONT")
    return runs[0]


def placement(serve_dir: Path, run: str) -> dict:
    """Which host ran each party, and what `trusted-host`'s premise rests on: the network certifier ran on the gate's
    host (the kept `SESSION`'s ``gate``; ``local`` is the prover's node), and the chain's checks and draws, the period
    check and their proofs' coins ran in this piece on the prover's node, the kept run's peer. No device of the
    auditor's is on the node's links, so the premise is assumed of both hosts; ``links`` are the node's to the gate,
    and K counts the certified one alone."""
    kept = serve_dir / PRIVATE / run
    if not (kept / SESSION).is_file():
        raise NotRun(f"{kept} has no {SESSION}, which names the gate the certifier ran on: keep the run with this "
                     "tree's front")
    gate = json.loads((kept / SESSION).read_text())["gate"]
    node = SS.served_peer(kept / "fetch.tar.gz") or socket.gethostname()
    if gate == "local":
        return {"kind": "prover-node", "parties": {"network_certifier": node, "prover": node, "verifier": node},
                PREMISE: f"unmet: the network certifier ran on the prover's node {node} (the front's --gate local)",
                "links": [], "uncounted": f"every link of {node}'s"}
    host = gate.rpartition("@")[2]
    return {"kind": "not-placed", "parties": {"network_certifier": host, "prover": node, "verifier": node},
            PREMISE: (f"assumed, the auditor's devices not placed: the network certifier ran on {host}, not the "
                      f"prover's {node}, but the chain's checks and draws, the period check and their proofs' coins "
                      f"ran on {node}, and no device of the auditor's is on {node}'s links"),
            "links": [{"name": "certified", "from": host, "to": node, "counted": True,
                       "via": "the warden's tunnel to the endpoint's port"},
                      {"name": "gate", "from": node, "to": host, "counted": False,
                       "via": "ssh, both ways: the session's request and prompts, its status and its fetch"}],
            "uncounted": f"every link of {node}'s but the certified one"}


def window_requests(serve_dir: Path, window: int) -> list[dict]:
    reqs = [r for r in _lines(serve_dir / "requests.jsonl") if window in (r.get("windows") or ())]
    if not reqs:
        raise NotRun(f"window {window} has no request in {serve_dir / 'requests.jsonl'}")
    return reqs


def unpack(fetch: Path) -> tuple[Path, Path]:
    """The kept fetch unpacked in a 0700 scratch directory beside it, and its warden run."""
    d = Path(tempfile.mkdtemp(dir=fetch.parent, prefix="unpacked-"))
    try:
        with tarfile.open(fetch, "r:gz") as tar:
            tar.extractall(d, filter="data")
    except (OSError, tarfile.TarError) as e:
        rm(d)
        raise NotRun(f"{fetch}: {type(e).__name__}: {e}") from None
    run = next((p.parent.parent if p.parent.name == SS.A.WITNESS else p.parent for p in d.rglob("certified.jsonl")),
               None)
    if run is None:
        rm(d)
        raise NotRun(f"{fetch}: no warden run (a certified.jsonl) in the fetch")
    return d, run


def rehearsed(run: Path) -> None:
    """`NotRun` unless the gate's check passed the kept rehearsal (its ``end.json``): a failed one, late or otherwise,
    is an infrastructure failure, like a busy gate."""
    try:
        end = json.loads((run / "end.json").read_text())
    except (OSError, ValueError) as e:
        raise NotRun(f"the kept run's end.json: {type(e).__name__}: {e}") from None
    v = end.get("verdict") or {}
    if end.get("rc") != 0 or v.get("accepted") is not True:
        raise NotRun(f"the gate's rehearsal didn't pass, so it certified no served answer: exit {end.get('rc')}, "
                     f"{end.get('ended')}; its check accepted {v.get('accepted')} with {v.get('problems')} problems, "
                     f"failed {v.get('failed')}")


def bind(reqs: list[dict], io: dict) -> dict[str, dict]:
    """Each window request's certified record by its user; Rejected when one isn't the warden's."""
    if not io["accepted"]:
        raise Rejected(f"the warden's certified I/O doesn't read whole off its cells: {io['violations'][:3]}")
    by, got = {r["user"]: r for r in io["records"]}, {}
    for r in reqs:
        c = by.get(r.get("user"))
        if c is None:
            raise Rejected(f"window request {r.get('engine_id')}'s user {r.get('user')!r} is no certified request: "
                           "the window served traffic the warden didn't see")
        for key, theirs in (("prompt_ids", "prompt"), ("streamed_ids", "output")):
            if key in r and r[key] != c[theirs]:
                raise Rejected(f"window request {r.get('engine_id')}'s {key} aren't the certified {theirs}")
        got[r["user"]] = c
    return got


def in_link_window(certified: dict[str, dict], T: int, w: int) -> None:
    for user, c in certified.items():
        buckets = [c["release"], *(b for _, b in c["frames"])]
        if None in buckets or {b // T for b in buckets} != {w}:
            raise Rejected(f"certified request {user!r} spans link-windows {sorted({b // T for b in buckets if b is not None})}"
                           f", not the period's {w} alone")


def endpoint_copy(serve_dir: Path, dest: Path) -> Path:
    dest.mkdir(mode=0o700, parents=True, exist_ok=True)
    for f in ENDPOINT:
        if not (serve_dir / f).is_file():
            raise NotRun(f"the serve dir has no {f}, which the chain's endpoint step reads")
        shutil.copyfile(serve_dir / f, dest / f)
        (dest / f).chmod(0o600)
    return dest


def sized(windows: int, sets: list[str]) -> list[str]:
    """N2's draw and the tie's link-windows for a run of ``windows`` link-windows: every one when they are at most
    `served_session.chain`'s 8 (whose draw of 8 a shorter run refuses), unless a CR_K=V of ``sets`` names them."""
    given = {kv.partition("=")[0] for kv in sets}
    if windows > 8:
        return sets
    return [*(["CR_N2_DRAW="] if "CR_N2_DRAW" not in given else []),
            *([f"CR_TIE_WINDOWS={windows}"] if "CR_TIE_WINDOWS" not in given else []), *sets]


def live(pid: int, pgid: int | None = None) -> bool:
    """Whether ``pid`` is a process (of process group ``pgid``, when given) that hasn't exited."""
    try:
        stat = Path("/proc", str(pid), "stat").read_bytes()
    except OSError:
        return False
    state, _ppid, pgrp = stat[stat.rindex(b")") + 2:].split()[:3]
    return state not in (b"Z", b"X") and pgid in (None, int(pgrp))


def members(pgid: int) -> list[int]:
    """The processes of process group ``pgid`` that haven't exited."""
    return [int(e.name) for e in os.scandir("/proc") if e.name.isdigit() and live(int(e.name), pgid)]


def end_group(p: subprocess.Popen) -> None:
    """SIGTERM to the step's process group, `GRACE_S` for its processes to exit, SIGKILL to the rest and their exit
    awaited, and only then its leader reaped: until then the group has a member, its leader, so its id names no other
    group while this signals it."""
    until = time.monotonic() + GRACE_S
    os.killpg(p.pid, signal.SIGTERM)
    while members(p.pid) and time.monotonic() < until:
        time.sleep(POLL_S)
    os.killpg(p.pid, signal.SIGKILL)
    while members(p.pid):
        time.sleep(POLL_S)
    p.wait()


def run_group(argv: list[str], **kw) -> int:
    """``argv``'s exit, run as one of the piece's steps (``kw`` as `subprocess.Popen` reads them): in a process group of
    its own, which `end_group` ends however the step ends, so nothing the step started outlives it. A SIGTERM while it
    runs is held (`ended`) until the group has ended, then raised as `NotRun`. The group keeps the piece's session, by
    which the research runner's timeout still reaches it should the piece itself be killed."""
    global _held
    p, _held = None, []
    try:
        p = subprocess.Popen(argv, process_group=0, **kw)
        fd = os.pidfd_open(p.pid)
        try:
            while not _held and not select.select([fd], [], [], 1)[0]:
                pass
        finally:
            os.close(fd)
    finally:
        try:
            if p is not None:
                end_group(p)
        finally:
            came, _held = _held, None
    if came:
        ended(came[0], None)
    return p.returncode


def captured(argv: list[str], **kw) -> subprocess.CompletedProcess:
    """`run_group` with the step's stdout and stderr read back as text, as `subprocess.run`'s ``capture_output=True,
    text=True`` gives them."""
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        rc = run_group(argv, stdout=out, stderr=err, **kw)
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(argv, rc, out.read(), err.read())


def run_chain(keep: Path, run: str, endpoint: Path, eos: str, draw: int | None, sets: list[str],
              cmd: str | None = None) -> int:
    """`served_session.py chain` on the kept run (or ``cmd``, run as ``bash -c CMD _ KEEP RUN ENDPOINT``), its run dir
    (RESEARCH_RUN_DIR, whose out/ holds the report) and quarantine private under KEEP/RUN."""
    d = keep / run
    env = dict(os.environ, RESEARCH_RUN_DIR=str(d / "chain"), CR_QUARANTINE=str(d / "quarantine"))
    (d / "chain").mkdir(mode=0o700, exist_ok=True)
    argv = ["bash", "-c", cmd, "_", str(keep), run, str(endpoint)] if cmd else [
        sys.executable, str(HERE / "served_session.py"), "chain", "--run", run, "--keep", str(keep),
        "--endpoint", str(endpoint), "--eos-ids", eos, *(["--draw", str(draw)] if draw else []), *sets]
    with open(d / "chain" / "stdout.log", "w") as log:
        return run_group(argv, cwd=TREE, env=env, stdout=log, stderr=subprocess.STDOUT)


def period_exe(lean: str | None) -> str:
    if lean:
        return lean
    got = captured(["lake", "build", "network-period"], cwd=LEAN_DIR)
    exe = LEAN_DIR / ".lake" / "build" / "bin" / "network-period"
    if got.returncode or not exe.is_file():
        raise NotRun(f"lake build network-period (exit {got.returncode}): {(got.stdout + got.stderr)[-400:]}")
    return str(exe)


def period_check(exe: str, case: dict, d: Path) -> dict:
    import scan_period as SP
    try:
        return SP.lean_check(exe, case, d, run=captured)
    except (SystemExit, OSError, ValueError) as e:
        raise NotRun(f"network-period check: {type(e).__name__}: {e}") from None


def case_of(warden: Path, scan: Path) -> dict:
    import scan_period as SP
    try:
        return SP.rows_case(warden, scan)
    except (SystemExit, OSError, KeyError, ValueError) as e:
        raise NotRun(f"the period's rows case: {type(e).__name__}: {e}") from None


def publish(chain_out: Path, d: Path) -> None:
    for rel in PUBLISHED:
        if (chain_out / rel).is_file():
            (d / "chain" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(chain_out / rel, d / "chain" / rel)


def decide(serve_dir: Path, window: int, d: Path, *, chain: Callable[[Path, str, Path, int], int],
           check: Callable[[dict, Path], dict], case: Callable[[Path, Path], dict] = case_of) -> tuple[dict, Path | None]:
    """Window ``window``'s answer (module docstring) and the unpacked fetch (for the leak guard), its files in ``d``;
    ``chain`` is given KEEP, the run, the endpoint's records and the run's link-windows. `NotRun` when a step can't run."""
    import warden_chain as WC
    keep = serve_dir / PRIVATE
    run = kept_run(serve_dir)
    reqs = window_requests(serve_dir, window)
    unpacked = None
    try:
        unpacked, warden = unpack(keep / run / "fetch.tar.gz")
        rehearsed(warden)
        try:
            io = WC.io(warden)
            dep, params = WC.deployment(warden)
        except SystemExit as e:
            raise Rejected(f"the kept warden run: {e}") from None
        T = params.window
        certified = bind(reqs, io)
        rc = chain(keep, run, endpoint_copy(serve_dir, keep / run / "endpoint"), int(dep["windows"]))
        out = keep / run / "chain" / "out"
        publish(out, d)
        report = out / "report" / "report.json"
        if not report.is_file():
            raise NotRun(f"the chain wrote no report (exit {rc}): see chain/ and the kept run's stdout.log")
        doc = json.loads(report.read_text())
        if doc.get("verdict") != "accepted":
            raise Rejected(f"the chain {doc.get('verdict')}: {json.dumps(doc.get('why'), default=str)}"[:1800])
        try:
            (w,) = [x["window"] for x in json.loads((out / "scan" / "answers.json").read_text())["windows"]]
        except (OSError, ValueError, KeyError) as e:
            raise NotRun(f"the scan's answers.json names no one link-window: {type(e).__name__}: {e}") from None
        in_link_window(certified, T, w)
        got = check(case(warden, out / "scan"), d / "period")
        if got.get("protocol") != PROTOCOL or got.get("verdict") not in ("accept", "reject", "undecided"):
            raise NotRun(f"network-period answered no {PROTOCOL} verdict: {json.dumps(got)[:400]}")
        return got, unpacked
    except Rejected as e:
        return answer("reject", str(e)), unpacked
    except BaseException:
        if unpacked is not None:
            rm(unpacked)
        raise


def guard(d: Path, hold: Path, unpacked: Path | None, quarantine: Path) -> int:
    """The record's leak guard on DIR and the hold against the warden run's witness, which releases the hold to DIR once
    it passes (0 when nothing of the witness was found, 3 when a find was moved to the quarantine)."""
    quarantine.mkdir(mode=0o700, parents=True, exist_ok=True)
    got = captured([sys.executable, str(HERE / "leak_guard.py"), str(d), str(quarantine),
                    *(["--gates", str(unpacked)] if unpacked is not None else []), "--hold", str(hold),
                    "--release-to", str(d), "--report", str(quarantine / "piece-guard.json")])
    print(f"leak guard (exit {got.returncode}): {got.stdout.strip()[-400:]}", flush=True)
    return got.returncode


def write_case(d: Path) -> None:
    """DIR/`CASE` from DIR/period's three files, when the leak guard left all three, and DIR/period gone."""
    p = d / "period"
    commitment, evidence, params = p / "commitment.bin", p / "evidence.json", p / "params.json"
    if not all(f.is_file() for f in (commitment, evidence, params)):
        return
    with open(d / CASE, "wb") as out, open(params, "rb") as src:
        out.write(b'{"commitment":"%s","evidence":"%s","params":' % (commitment.read_bytes().hex().encode(),
                                                                     evidence.read_bytes().hex().encode()))
        shutil.copyfileobj(src, out, 1 << 24)
        out.write(b"}\n")
    rm(p)
    print(f"case: {CASE}, {(d / CASE).stat().st_size} bytes", flush=True)


def rm(*paths: Path) -> None:
    """``rm -rf`` as a child process: the suites' read guard misreads `shutil.rmtree`'s fd-relative walk."""
    if paths:
        subprocess.run(["rm", "-rf", "--", *map(str, paths)], check=False)


def refuse(d: Path, why: str) -> int:
    """DIR emptied but for its log, and DIR/`REFUSED` naming why: no answer, so the certifier reads missing."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    rm(*(p for p in d.iterdir() if p.name != "piece.log"))
    (d / REFUSED).write_text(why + "\n")
    print(f"CERTIFIER refused: {why}", flush=True)
    return 3


def ended(signum: int, _frame) -> None:
    if _held is not None:
        _held.append(signum)
        return
    raise NotRun(f"{signal.Signals(signum).name}: the piece's time limit (CERTIFIER_TIMEOUT) ended it before a verdict")


def piece(a: argparse.Namespace) -> int:
    serve_dir, d = a.serve_dir.resolve(), a.dir.resolve()
    d.mkdir(parents=True, exist_ok=True)
    for kv in a.set:
        if not kv.startswith("CR_") or "=" not in kv:
            print(f"refused: {kv!r}: CR_K=V, a chain_record.sh variable", file=sys.stderr)
            return 2
    was, unpacked, hold = signal.signal(signal.SIGTERM, ended), None, None
    try:
        if not any((serve_dir / PRIVATE).glob("*/fetch.tar.gz")):
            raise NotRun(f"no warden run kept under {serve_dir / PRIVATE}: the run's load didn't go through the "
                         "certifier's front (CERTIFIER_FRONT unset, or the gate, its tunnel or ssh failed)")
        (serve_dir / PRIVATE / "held").mkdir(mode=0o700, exist_ok=True)
        hold = Path(tempfile.mkdtemp(dir=serve_dir / PRIVATE / "held", prefix=f"window-{a.window}-"))
        exe = period_exe(a.lean)
        chain = lambda keep, run, ep, n: run_chain(keep, run, ep, a.eos_ids, a.draw, sized(n, a.set),  # noqa: E731
                                                   a.chain_cmd)
        got, unpacked = decide(serve_dir, a.window, hold, chain=chain, check=lambda c, p: period_check(exe, c, p))
        if isinstance(got.get("profile"), dict):
            got = got | {"profile": got["profile"] | {"placement": placement(serve_dir, kept_run(serve_dir))}}
        (hold / "answer.json").write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
        g = guard(d, hold, unpacked, serve_dir / PRIVATE / "quarantine")
        if g not in (0, 3) or not (d / "answer.json").is_file():
            raise NotRun(f"the leak guard failed (exit {g}): DIR withheld, the piece's files left in {hold}")
        write_case(d)
    except NotRun as e:
        return refuse(d, str(e))
    except (OSError, ValueError, tarfile.TarError) as e:
        traceback.print_exc()
        return refuse(d, f"the piece failed on a file: {type(e).__name__}: {e}")
    finally:
        if unpacked is not None:
            rm(unpacked)
        if hold is not None and hold.is_dir() and not any(p.is_file() for p in hold.rglob("*")):
            rm(hold)
        signal.signal(signal.SIGTERM, was)
    print(f"CERTIFIER {json.dumps({k: got.get(k) for k in ('verdict', 'reason', 'public')}, sort_keys=True)}",
          flush=True)
    return 0 if got.get("verdict") == "accept" else 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="certifier_piece.py", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("front")
    f.add_argument("port", type=int)
    f.add_argument("serve_dir", type=Path)
    f.add_argument("out", type=Path)
    f.add_argument("--gate", default=SS.GATE, help="USER@HOST, the gate's; or local, the dry run's")
    f.add_argument("--key", type=Path, default=SS.KEY)
    f.add_argument("--peer", default="vy-dc-h200")
    f.add_argument("--prompts", type=int, default=PROMPTS)
    f.add_argument("--prompt-ids", type=int, default=PROMPT_IDS)
    f.add_argument("--tokens", type=int, default=TOKENS)
    f.add_argument("--vocab", type=int, default=SS.VOCAB)
    f.add_argument("--height", type=int, default=HEIGHT)
    f.add_argument("--minutes", type=int)
    f.add_argument("--bucket-ms", type=int, metavar="MS",
                   help="the scan's tick, the gate's 100 to 4000 (the scan's 100 when not given)")
    f.add_argument("--warmup", type=int, default=0, metavar="N",
                   help="plain requests of the group's shape straight to PORT first, each in a window of its own "
                        "(none by default)")
    q = sub.add_parser("piece")
    q.add_argument("serve_dir", type=Path)
    q.add_argument("window", type=int)
    q.add_argument("dir", type=Path)
    q.add_argument("--eos-ids", default="128001")
    q.add_argument("--lean", help="network-period, built (else built here)")
    q.add_argument("--draw", type=int, help="the scan's K drawn units (every unit by default)")
    q.add_argument("--chain-cmd", help="replaces served_session.py chain: bash -c CMD _ KEEP RUN ENDPOINT")
    q.add_argument("set", nargs="*", help="CR_K=V")
    r = sub.add_parser("relay")
    r.add_argument("--listen", type=int, required=True)
    r.add_argument("--to", type=int, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    if a.cmd == "front":
        return front(a)
    if a.cmd == "piece":
        return piece(a)
    relay(a.listen, a.to)
    return 0


if __name__ == "__main__":
    sys.exit(main())
