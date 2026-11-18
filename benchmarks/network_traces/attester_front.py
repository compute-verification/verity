"""attester-front: the front that puts the active network attester in front of serve-openai in `served_verdicts.sh`
(its ``CERTIFIER_FRONT``), the gate's client: it keeps the gate's fetch under the serve dir, where the piece
(`attester_piece.py`, its ``CERTIFIER_CMD``) reads the kept run.

    python attester_front.py front PORT SERVE_DIR OUT --peer NAME --gate USER@HOST|local [--key KEY]
        [--prompts 4] [--prompt-ids 24] [--tokens 4] [--vocab 128000] [--scan-class w63k4] [--height 10]
        [--minutes M] [--bucket-ms MS] [--warmup N]
    python attester_front.py relay --listen PORT --to PORT
    python attester_front.py front PORT SERVE_DIR OUT --peer NAME --session --client-port PORT --windows N
        --gate ... [--key KEY] [--scan-class w63k4] [--height H] [--minutes M] [--bucket-ms MS]
    python attester_front.py close SERVE_DIR OUT

* **front** runs before served_verdicts.sh's requests when ``CERTIFIER_FRONT`` names it: the gate's scan rehearsal
  (`served_session.session`, ``warden-rehearse --scan --prompts --window-per-group``) against serve-openai on PORT, one
  group of ``--prompts`` requests (at most the scan's 4 lanes) of ``--prompt-ids`` ids below ``--vocab``, each answered
  with exactly ``--tokens`` ids, sampled from a fresh secret seed. A group is sent half a bucket into execution phase 0
  and its cells are execution phase 0's alone (``--tokens`` at most 4 keeps the group in one execution phase), so the
  engine admits the group together and closes it into one Commit window, the run's earliest: the window the run audits (with
  ``REQUESTS=1`` the requests after it make a smaller window, which `served_verdicts.py window` passes over). The
  defaults are the window circuit's class 4,32,4 (4 slots, prompts of at most 32 ids, 4 outputs), and the gate runs
  the network attester at its scan class's schedule (``--scan-class``, `V3C_SCAN_CLASS`: w63k4, at 5 egress cells),
  the one the chain's scan at that class takes. The scan's schedule
  gives a request D·τ from its admission to its first egress cell, by which a short answer must be whole, and an
  endpoint slower than that fails each request late: 2.7 s at the scan's tick τ of 100 ms. ``--bucket-ms MS`` runs
  the scan at the tick MS instead (the gate's `auditor.SCAN_BUCKET_MS`, 100 to 4000: D·τ = 108 s at 4000), each
  sampling block T·τ long, and the minutes are counted there. ``--warmup N`` (none by default: an endpoint that
  readies its first window before it listens needs none) first sends N plain completions straight to PORT (`warmup`:
  no network attester, no release header), one at a time, each of the group's shape and awaited, so the endpoint's first window
  (H1's first reservation, the kernels' first compile) is behind it; it then waits until serve-openai records each in
  `served_session.USERS`, which it does only once the request's window has closed. So each warm-up is a window of one
  request, closed before the gate is asked, which the pick passes over for the group's whenever the group's window
  holds two or more; one that fails or isn't recorded within `WARMUP_CLOSE_S` refuses the front. A warm-up doesn't
  make a slow endpoint pass; a tick whose D·τ covers its answer does. The gate's fetch,
  ``witness/`` with it, is kept under SERVE_DIR/`PRIVATE`/RUN/fetch.tar.gz (0700, the serve dir stays on the node);
  OUT gets ``session.json`` and ``bundle.tgz``, the public outcome, and the run's kept directory a copy of
  ``session.json`` (`SESSION`), whose gate the piece's placement names. ``--gate local`` is the dry run's (this
  tree's gate in this process, a `relay` to PORT its tunnel): a network attester on the developer's node is no auditor's device,
  so it never stands for the gate.
  Exit as `served_session.session`'s.
* **relay**: a TCP relay from 127.0.0.1:``--listen`` to 127.0.0.1:``--to``, the local gate's tunnel.
* **front --session** and **close** put the gate's relay session (`auditor.py`'s "A relay session") around the served
  run's own load, in the rehearsal front's place: the load's clients are the run's, and the front sends no group.
  - ``front --session`` asks the gate for ``relay-session``: PORT is the endpoint's, ``--client-port`` the port of
    the peer's loopback the load's clients connect to and ``--windows`` the session's length in execution phases.
    ``--height`` at the front's default, 10, which signs 204 of them, is raised to the least whose key signs the
    session's (`auditor.network_attester_height`), and ``--minutes`` is by default `auditor.session_minutes` at that height
    plus `served_session.MARGIN`, at most the gate's: for the longest session, 1119 execution phases, height 13,
    with ``--minutes`` 66 at the scan's 100 ms tick (59 min 40 s of them) and 126 at 200 ms (119 min 21 s). The
    gate refuses any other ``--height`` too low for the session. This tree's gate checks the request first. It
    returns once the gate's ``status`` shows the relay listening: exit 0, with
    SERVE_DIR/`PRIVATE`/RUN/`RELAY` (0600) holding ``client``, 127.0.0.1:CLIENT_PORT on the peer, and
    ``deployment``, the relay's (its schedule, ``origin_unix`` and ``windows``), and one line on stdout naming the
    run, the client's address and that file. Exit 2 when the gate refused the request or a session is open already,
    1 when the session ended before its relay listened, the reason in OUT/session.json. The session then runs to
    its execution phases' end whatever its caller does: nothing ends it early, and its last connection is admitted
    `NetworkAttesterParams.span` + 1 buckets before that end.
  - ``close`` runs after the load: it waits for the session's end, keeps the lane's fetch where the piece reads it
    (SERVE_DIR/`PRIVATE`/RUN/fetch.tar.gz) and publishes to OUT what the rehearsal front does (``session.json``,
    ``bundle.tgz`` and ``status.json``, through the leak guard), with the front's exits: 0 when the gate's check
    accepted the run.
  - The piece's steps close a session that no ``close`` ended before they read the kept run (`attester_piece.close_open`): the same
    wait and the same kept files, nothing published.
  - The piece answers for a session's run as for a rehearsal's, only when the gate's check accepted the whole run
    (`rehearsed`). One connection the relay failed closed, one cut from s + D on at the session's end, one refused at
    a full queue or one execution phase the relay faulted, whichever load it carried, leaves every window of the
    served run without the attester's answer. The run's records cover the connections that crossed the relay; that
    nothing else reaches the endpoint is the placement's, as a rehearsal's.
  - ``--gate local`` is the dry run's: this tree's gate on a scratch host under SERVE_DIR/`PRIVATE`, its unit a
    process of the front's (`local_unit`) with a `relay` each way in the place of its ssh.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import attester_piece as P  # noqa: E402
import served_session as SS  # noqa: E402

#: the front's refusal without ``--gate``
NO_GATE = ("--gate is unset: USER@HOST, the gate's, which `verity run` passes from the binding "
           "network-attester.network-attester.gate, or local, the network attester on this machine under a key made for the run")
#: the front's refusal without ``--peer``
NO_PEER = ("--peer is unset: the served node's name in the gate's peers file, which `verity run` passes from the "
           "binding network-attester.network-attester.peer")
#: the window circuit's class (4 slots, prompts of at most 32 ids, 4 outputs) and the scan's lanes
PROMPTS, PROMPT_IDS, TOKENS = 4, 24, 4
HEIGHT = 10
#: how long a warm-up request may take, and how long the front waits for serve-openai to record the warm-ups'
#: windows closed
WARMUP_REQUEST_S, WARMUP_CLOSE_S = 600, 600



def front(a: argparse.Namespace, *, sleep: Callable[[float], object] = time.sleep) -> int:
    serve_dir, out = a.serve_dir.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    keep = serve_dir / P.PRIVATE
    keep.mkdir(parents=True, exist_ok=True)
    for p in (serve_dir / "private", keep):
        p.chmod(0o700)
    if a.session:
        return open_session(a, keep, out, sleep=sleep)
    a.requests = a.prompts
    a.keep, a.out, a.at, a.window_per_group = keep, out, None, True
    if a.minutes is None:
        tick = a.bucket_ms or SS.A.SCAN["--bucket-ms"]
        a.minutes = SS.A.network_attester_group_minutes(1, a.tokens, a.height, tick) + SS.MARGIN
    if a.gate == "local":
        a.tunnel = lambda ssh, port: [sys.executable, str(Path(__file__).resolve()), "relay", "--listen", str(port),
                                      "--to", str(a.port)]
    try:
        if a.gate is None:
            raise SS.Refused(NO_GATE)
        if a.peer is None:
            raise SS.Refused(NO_PEER)
        a.gate = SS.gate_at(a.gate)
        if a.prompts > SS.A.SCAN["--lanes"]:
            raise SS.Refused(f"--prompts {a.prompts}: one group, at most the scan's {SS.A.SCAN['--lanes']} lanes")
        if SS.A.scan_group_windows(a.tokens) != 1:
            raise SS.Refused(f"--tokens {a.tokens}: a group past one execution phase, which the period's rows form refuses")
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
            shutil.copyfile(doc, keep / json.loads(doc.read_text())["run"] / P.SESSION)
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
    while left := users - {r.get("user") for r in P._lines(serve_dir / SS.USERS)}:
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


def relays(pairs: list[tuple[int, int]]) -> None:
    """`relay` for each (listen, to) of `pairs`, the last in this thread once every other listens."""
    *more, last = pairs
    for pair in more:
        listening = threading.Event()
        threading.Thread(target=relay, args=pair, kwargs={"ready": listening.set}, daemon=True).start()
        listening.wait()
    relay(*last)


# -------------------------------------------------------------------------------------------------------------- session

#: a relay session the front opened, under SERVE_DIR/`P.PRIVATE`/RUN (0600): the run, its gate and key, the front's
#: arguments its request is made of (`SESSION_ARGS`) and, once the relay listens, where a client connects (``client``)
#: and the relay's deployment; the session is open until ``fetch.tar.gz`` is beside it
RELAY = "relay.json"
SESSION_ARGS = ("peer", "port", "client_port", "windows", "minutes", "height", "bucket_ms", "scan_class")
#: how often the front asks the gate whether the relay listens yet: the run's first bucket starts a few seconds after
READY_POLL_S = 1


class Unready(SS.Refused):
    """A relay session that ended, or whose unit did, before the front or the close had what it waited for."""
    code = 1


def open_sessions(keep: Path) -> list[dict]:
    """Each relay session under `keep` that no close has ended: its `RELAY`."""
    return [json.loads(p.read_text()) for p in sorted(keep.glob(f"*/{RELAY}"))
            if not (p.parent / "fetch.tar.gz").is_file()]


def session_args(doc: dict, keep: Path, out: Path) -> argparse.Namespace:
    """The front's arguments a kept `RELAY` holds, as `served_session` reads a session's."""
    return argparse.Namespace(**doc["args"], gate=doc["gate"], key=Path(doc["key"]), unit=doc.get("unit"),
                              run=doc["run"], keep=keep, out=out, at=None)


def write_relay(keep: Path, a: argparse.Namespace, run: str, relay_: dict | None = None) -> Path:
    doc = {"run": run, "gate": a.gate, "key": str(a.key), "args": {k: getattr(a, k) for k in SESSION_ARGS},
           **({} if getattr(a, "unit", None) is None else {"unit": a.unit}), **(relay_ or {})}
    path = keep / run / RELAY
    path.parent.mkdir(mode=0o700, exist_ok=True)
    with open(path, "w", opener=lambda p, flags: os.open(p, flags, 0o600)) as f:
        f.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return path


def gate_status(a: argparse.Namespace, keep: Path, run: str) -> dict | None:
    """The gate's ``status RUN``, None where it didn't answer: over ssh, or this tree's gate on ``--gate local``'s
    scratch host, whose unit is the process ``a.unit``."""
    if a.gate == "local":
        host = SS.scratch_host(keep / "local-gate", a.peer)
        rc, got = SS.A.gate(["--ssh", f"status {run}", "--key", SS.LANE], host,
                            attester_active=lambda r: a.unit is not None and P.live(a.unit))
        got = got if rc == 0 else None
    else:
        got = SS.status(a, run)
    return None if got is None else json.loads(got)


def waited(a: argparse.Namespace, keep: Path, run: str, done: Callable[[dict], bool], every: float,
           sleep: Callable[[float], object]) -> dict:
    """The gate's status of `run` once `done` takes it, asked every `every` seconds. `Unready` when the unit is twice
    seen ended with no end, which a unit that ran to its end always writes, or when the session's own minutes of
    asking pass, by which the gate has stopped it."""
    gone = 0
    for _ in range(int(a.minutes * 60 / every) + 1):
        st = gate_status(a, keep, run)
        if st is not None:
            if done(st):
                return st
            gone = gone + 1 if st.get("active") is False and st.get("end") is None else 0
            if gone > 1:
                raise Unready(f"relay session {run}: its unit ended and wrote no end.json (the gate's status)")
        sleep(every)
    raise Unready(f"relay session {run}: {a.minutes} minutes of asking, the session's own, and the gate's status "
                  "didn't show what was awaited")


def open_session(a: argparse.Namespace, keep: Path, out: Path, *, sleep: Callable[[float], object] = time.sleep) -> int:
    """``front --session`` (the module docstring): the gate's relay session started, and its `RELAY` kept once the
    relay listens. Exit 0 then, 2 when the session isn't started (the reason in OUT/session.json), 1 when it ended
    before its relay listened."""
    try:
        if a.gate is None:
            raise SS.Refused(NO_GATE)
        if a.peer is None:
            raise SS.Refused(NO_PEER)
        a.gate = SS.gate_at(a.gate)
        if a.client_port is None or a.windows is None:
            raise SS.Refused("--session: with --client-port PORT, the peer's loopback port the load's clients "
                             "connect to, and --windows N, the session's length in execution phases")
        if a.warmup:
            raise SS.Refused("--session sends no request of its own: --warmup is the rehearsal front's")
        if a.height == HEIGHT:
            a.height = SS.A.network_attester_height(a.windows)
        if a.minutes is None:
            tick = a.bucket_ms or SS.A.SCAN["--bucket-ms"]
            a.minutes = min(SS.A.session_minutes(a.windows, a.height, SS.A.SCAN["--window"] * tick) + SS.MARGIN,
                            SS.A.ATTESTER_MINUTES[1])
        with tempfile.TemporaryDirectory() as d:
            try:
                SS.A.REHEARSALS[SS.words(a)[0]](SS.words(a)[1:], SS.scratch_host(Path(d), a.peer))
            except SS.A.Refused as e:
                raise SS.Refused(f"the gate refuses it: {e}") from None
        if opened := open_sessions(keep):
            raise SS.Refused(f"relay session {opened[0]['run']} is open under {keep}: close it first "
                             "(attester_front.py close)")
        if a.gate == "local":
            _, run = SS.local_start(a, b"", keep)
            (keep / run).mkdir(mode=0o700, exist_ok=True)
            with open(keep / run / "unit.log", "wb") as log:
                a.unit = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "unit", str(keep), run,
                                           a.peer, str(a.port)], stdin=subprocess.DEVNULL, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True).pid
        else:
            run = SS.start(a, b"", keep)
        write_relay(keep, a, run)
        st = waited(a, keep, run, lambda st: st.get("end") is not None or st.get("relay") is not None,
                    READY_POLL_S, sleep)
        if st.get("relay") is None:
            raise Unready(f"relay session {run} ended before its relay listened: exit {st['end'].get('rc')}, "
                          f"{st['end'].get('ended')}")
        kept = write_relay(keep, a, run, st["relay"])
        print(json.dumps({"run": run, "client": st["relay"]["client"], "relay": str(kept)}, sort_keys=True), flush=True)
        return 0
    except SS.Refused as e:
        (out / "session.json").write_text(json.dumps({"refused": str(e)}, indent=1) + "\n")
        print(f"refused: {e}", file=sys.stderr)
        return e.code


def local_unit(a: argparse.Namespace) -> int:
    """``--gate local``'s unit of a relay session: this tree's `auditor.certifier_run` on the scratch host under KEEP,
    a `relay` both ways in the place of its ssh, to the endpoint's port and from the clients' to the relay."""
    host = SS.scratch_host(a.keep / "local-gate", a.peer)
    me = [sys.executable, str(Path(__file__).resolve()), "relay"]
    end = SS.A.certifier_run(host, a.run, pinned=SS.pinned(),
                             tunnel_argv=lambda ssh, port, back: [*me, "--listen", str(port), "--to", str(a.port),
                                                                  "--listen", str(back[0]), "--to", str(back[1])])
    return 0 if end["rc"] == 0 else 1


def ended_fetch(a: argparse.Namespace, keep: Path, hold: Path, sleep: Callable[[float], object]) -> bytes:
    """The open session ``a.run``'s end awaited (the gate's status every `served_session.POLL_S` seconds; the last,
    less its ``relay``, which the bundle's deployment holds, is `hold`'s ``status.json``), then the lane's fetch."""
    st = waited(a, keep, a.run, lambda st: st.get("end") is not None, SS.POLL_S, sleep)
    (hold / "status.json").write_text(json.dumps({k: v for k, v in st.items() if k != "relay"}, sort_keys=True) + "\n")
    try:
        if a.gate == "local":
            return SS.A.attester_fetch(a.run, SS.scratch_host(keep / "local-gate", a.peer), witness=True)
        return SS.fetch(a, a.run)
    except subprocess.CalledProcessError as e:
        raise Unready(f"relay session {a.run}: the gate's fetch failed (exit {e.returncode}): "
                      f"{e.stderr[-400:]!r}") from None
    except SS.A.Refused as e:
        raise Unready(f"relay session {a.run}: the gate's fetch: {e}") from None


def close(a: argparse.Namespace, *, sleep: Callable[[float], object] = time.sleep) -> int:
    """``close`` (the module docstring): the one open relay session's end awaited and its fetch kept and published
    as a rehearsal front's. Exit as `served_session.session`'s; 2 with no session open, or more than one."""
    serve_dir, out = a.serve_dir.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    keep = serve_dir / P.PRIVATE
    try:
        opened = open_sessions(keep)
        if len(opened) != 1:
            raise SS.Refused(f"{len(opened)} relay sessions open under {keep}, not one: the front (front --session) "
                             "opens it before the load")
        b = session_args(opened[0], keep, out)
        b.close = lambda hold: ended_fetch(b, keep, hold, sleep)
        rc = SS.session(b, b"", [])
        if (doc := out / "session.json").is_file():
            shutil.copyfile(doc, keep / b.run / P.SESSION)
        return rc
    except SS.Refused as e:
        (out / "session.json").write_text(json.dumps({"refused": str(e)}, indent=1) + "\n")
        print(f"refused: {e}", file=sys.stderr)
        return e.code


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="attester_front.py", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("front")
    f.add_argument("port", type=int)
    f.add_argument("serve_dir", type=Path)
    f.add_argument("out", type=Path)
    f.add_argument("--gate", help="USER@HOST, the gate's; or local, the network attester on this machine")
    f.add_argument("--key", type=Path, default=SS.KEY)
    f.add_argument("--peer", help="the served node's name in the gate's peers file")
    f.add_argument("--prompts", type=int, default=PROMPTS)
    f.add_argument("--prompt-ids", type=int, default=PROMPT_IDS)
    f.add_argument("--tokens", type=int, default=TOKENS)
    f.add_argument("--vocab", type=int, default=SS.VOCAB)
    f.add_argument("--scan-class", choices=sorted(SS.A.SCAN_CLASSES), default=P.V3C_SCAN_CLASS,
                   help=f"the gate's scan class, the schedule its network attester runs at (default {P.V3C_SCAN_CLASS})")
    f.add_argument("--height", type=int, default=HEIGHT)
    f.add_argument("--minutes", type=int)
    f.add_argument("--bucket-ms", type=int, metavar="MS",
                   help="the scan's tick, the gate's 100 to 4000 (the scan's 100 when not given)")
    f.add_argument("--warmup", type=int, default=0, metavar="N",
                   help="plain requests of the group's shape straight to PORT first, each in a window of its own "
                        "(none by default)")
    f.add_argument("--session", action="store_true",
                   help="the gate's relay session, kept up for the served run's own load: no group is sent")
    f.add_argument("--client-port", type=int, metavar="PORT",
                   help="with --session: the port of the peer's loopback the load's clients connect to")
    f.add_argument("--windows", type=int, metavar="N", help="with --session: its length, in execution phases")
    r = sub.add_parser("relay")
    r.add_argument("--listen", type=int, required=True, action="append")
    r.add_argument("--to", type=int, required=True, action="append")
    c = sub.add_parser("close")
    c.add_argument("serve_dir", type=Path)
    c.add_argument("out", type=Path)
    u = sub.add_parser("unit")
    u.add_argument("keep", type=Path)
    u.add_argument("run")
    u.add_argument("peer")
    u.add_argument("port", type=int)
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    if a.cmd == "front":
        return front(a)
    if a.cmd == "close":
        return close(a)
    if a.cmd == "unit":
        return local_unit(a)
    if len(a.listen) != len(a.to):
        raise SystemExit("relay: one --to for each --listen")
    relays(list(zip(a.listen, a.to, strict=True)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
