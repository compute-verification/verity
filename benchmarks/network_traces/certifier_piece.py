"""certifier-piece: the network certifier's piece of `served_verdicts.sh` (its ``CERTIFIER_CMD``), and the front that puts
the active network certifier in front of serve-openai there (its ``CERTIFIER_FRONT``).

    python certifier_piece.py front PORT SERVE_DIR OUT [--gate USER@HOST | --gate local] [--key KEY] [--peer NAME]
        [--prompts 4] [--prompt-ids 24] [--tokens 4] [--vocab 128000] [--scan-class w63k4] [--height 10]
        [--minutes M] [--bucket-ms MS] [--warmup N]
    python certifier_piece.py piece SERVE_DIR WINDOW DIR [--eos-ids IDS] [--lean EXE] [--rows EXE] [--draw K]
        [--chain-cmd CMD] [CR_K=V ...]
    python certifier_piece.py chain SERVE_DIR WINDOW DIR V3C [--eos-ids IDS] [--rows EXE] [--draw K] [--class w63k4]
        [--chain-cmd CMD] [CR_K=V ...]
    python certifier_piece.py period SERVE_DIR WINDOW DIR V3C [--lean EXE] [--rows EXE]
    python certifier_piece.py relay --listen PORT --to PORT
    python certifier_piece.py front PORT SERVE_DIR OUT --session --client-port PORT --windows N [--gate ...]
        [--key KEY] [--peer NAME] [--scan-class w63k4] [--height H] [--minutes M] [--bucket-ms MS]
    python certifier_piece.py close SERVE_DIR OUT

* **front** runs before served_verdicts.sh's requests when ``CERTIFIER_FRONT`` names it: the gate's scan rehearsal
  (`served_session.session`, ``warden-rehearse --scan --prompts --window-per-group``) against serve-openai on PORT, one
  group of ``--prompts`` requests (at most the scan's 4 lanes) of ``--prompt-ids`` ids below ``--vocab``, each answered
  with exactly ``--tokens`` ids, sampled from a fresh secret seed. A group is sent half a bucket into execution phase 0
  and its cells are execution phase 0's alone (``--tokens`` at most 4 keeps the group in one execution phase), so the
  engine admits the group together and closes it into one Commit window, the run's earliest: the window the run audits (with
  ``REQUESTS=1`` the requests after it make a smaller window, which `served_verdicts.py window` passes over). The
  defaults are the window circuit's class 4,32,4 (4 slots, prompts of at most 32 ids, 4 outputs), and the gate runs
  the network certifier at its scan class's schedule (``--scan-class``, `V3C_SCAN_CLASS`: w63k4, at 5 egress cells),
  the one the chain's scan at that class takes. The scan's schedule
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
  tree's gate in this process, a `relay` to PORT its tunnel): a warden on the developer's node is no auditor's device,
  so it never stands for the gate.
  Exit as `served_session.session`'s.
* **piece** writes window WINDOW's core answer to DIR/answer.json (`verity/certifier/v0`, `NetworkCheck.Period`'s rows
  form): accept only when ``network-rows check`` accepted the network certifier's records, the chain accepted and
  ``network-period check`` accepted the period, whose answer it then is.
  1. The kept network certifier run: exactly one fetch under SERVE_DIR/`PRIVATE`, whose rehearsal the gate's check
     passed (its ``end.json``: exit 0 and accepted, as the front's exit reads it). With none the run's load didn't go
     through the front (it wasn't set, or the gate, its tunnel or ssh failed); a rehearsal that failed (each request
     late past the network certifier's D·τ, or cut) certified no served answer, so there is none for the protocol to
     reject (`rehearsed`). Its records are then audited as rows (`rows_check`): `warden_chain.rows_audit_case` of the
     run, checked by ``network-rows`` (``--rows``, else built in verity/protocols/accounting/certifier/verifier/lean;
     `NetworkCheck.RowsAudit`, `ROWS`), which holds each record to its certified head and each certificate to the
     deployment's key in its session, then runs the certifier program's rows audit (`NetworkCertifier.Rows.audit`):
     every egress row must be the one the schedule the records show renders, so cells cut, clocked or ordered
     otherwise are refused. It holds no egress cell to the id egress form; step 2 does (`warden_chain.io`, which
     refuses an egress cell that isn't an id-form cell), and step 3's chain does, in its rerun of the network
     certifier's own check on the run (`warden_chain.certificates`). Its reject is the piece's, with its reason's
     numbers as N. The deployment is the kept
     fetch's own (`warden_chain.own_record`), as every check below reads it. The case's files go to a 0700 scratch
     directory beside the unpacked run, removed once the check has answered, so the records reach no hold.
  2. The window's requests are certified ones: each window request's user names one of the network certifier's certified
     requests (`warden_chain.io`), with the same prompt and streamed ids; else reject.
  3. The chain (`served_session.py chain`: chain_record.sh's network certifier path, certs, endpoint, N2, tie and scan)
     on the kept run, with the serve dir's endpoint records (`ENDPOINT`, copied to SERVE_DIR/`PRIVATE`/RUN/endpoint), its run
     dir and quarantine under SERVE_DIR/`PRIVATE`/RUN, so nothing of the witness reaches a record. N2 and the tie
     take every link-window of a run of at most 8 (`sized`). Its report's verdict ``accepted`` goes on, and so does
     ``accepted on the period check`` (`chain_record.RESTS`: the scan's acceptance rests on step 5's check, which
     then decides); any other is a reject with its why.
  4. The period's link-window is the scan's (``answers.json``'s one window): every window request's release and
     streamed ids lie in it, else reject.
  5. The period check: `scan_period.rows_case` of the warden run and the chain's scan, checked by ``network-period``
     (``--lean``, else built in verity/protocols/accounting/certifier/verifier/lean); its answer is the piece's, an accept's profile
     with its ``placement`` (`placement`): the machine the network certifier ran on, and the machine that ran each of
     the piece's own steps, the chain and the period check, each read from the machine as it runs the step (`ran_on`).
     The names are labels: the piece establishes no `trusted-host` and gives no bound on that ground (`UNMET`).
  DIR also gets the chain's public part (`PUBLIC`: its report's verdicts and ε at δ; never the scan's answers, the
  endpoint's report, the report's counts or a step's summary, which are private) and the link-window's five
  certificates (`CERTIFICATES`, `certificates`: the three records' certificates and the two row registrations' of the
  case's evidence, each with its own fields alone). They and the answer are written to a fresh hold under
  SERVE_DIR/`PRIVATE`/held (0700) and reach DIR only through the record's leak guard (`leak_guard.py`), which reads DIR
  and the hold against the warden run's witness, DIR declared to hold those files, the split's `CHAINED` and the logs
  alone (`PIECE_FILES`), moves what holds a piece of it or lies at another path to SERVE_DIR/`PRIVATE`/quarantine and
  only then releases the rest. The check's three input files are the piece's case, and a case's evidence and params
  are witness (`composite.py`), as the scan's answers are, that evidence less its certificates: neither is ever in
  DIR. Before the guard reads the hold, its period files become SERVE_DIR/`CASES`/`CASE` (`write_case`; the directory
  0700, the file 0600), the case as `composite.py` reads a piece's: ``{"commitment", "evidence", "params"}``, the
  commitment and evidence hex and the params ``params.json``'s canonical JSON, all inline, since certifier/v0 reads no
  blob (about 83 MB on a served window, nearly all of it the params); the certificates are cut from that evidence, and
  the hold's period files then go, so the case has one copy. It is there only where the step ended with its files
  released (`settle`), and no answer is beside it: the answer is DIR's. Only the protocol rejects: a step that can't
  run (no kept run, a rehearsal the gate's check didn't pass, a build, a tool or the chain failing without a report,
  an input missing, the leak guard failing, or SIGTERM from the piece's time limit) is `NotRun`, and the piece then
  refuses, releasing nothing from the hold and keeping no case: DIR keeps only its logs (`LOGS`) and gets
  DIR/`REFUSED` naming why, with no answer, which `served_verdicts.py` reads as the certifier missing.
  Each step it runs (the build, the chain, the check and the leak
  guard) is a process group of its own, which the piece ends however the step ends (`run_group`: SIGTERM, `GRACE_S`,
  SIGKILL), so nothing a step started outlives it; a SIGTERM while a step runs is held until the step's group has
  ended, so a step never outlives the refusal either. ``--draw K`` is the scan's sample, K proof blocks (every proof
  block by default); a CR_K=V goes to the chain; ``--chain-cmd`` replaces the chain (a test's, which writes the report
  and the scan's answers under RESEARCH_RUN_DIR/out).
* **chain** and **period** are the piece split in two around V3c (`served_verdicts.sh`'s ``CERTIFIER_CHAIN_CMD`` and
  ``CERTIFIER_PERIOD_CMD``, each run as ``bash -c CMD _ D K DIR V3C``, e.g. ``CERTIFIER_CHAIN_CMD='python
  benchmarks/network_traces/certifier_piece.py chain "$@"'``), so the period's compute audit is V3c's audit of the
  served window itself. **chain** runs steps 1 to 4 once V3c's window is committed, its scan on V3c's trees
  (``scan_zk.py --class CLASS --v3c V3C/window`` in CR_SCAN_ARGS: the window's committed scan state and slot entries):
  a reject is DIR/answer.json as the piece's, an accept DIR/`CHAINED`. **period** runs once V3c's audit wrote
  V3C/period.json: steps 1, 2 and 4 again on the chain's kept report (no chain runs), then step 5 on the rows case whose
  one compute audit is V3C/period.json (`scan_period.rows_case`), held to the class circuit of the verifier's own
  window-circuit record beside V3C (`class_circuit`); its answer is DIR/answer.json. The check reads an audit's state
  entry among its ``roots`` and its ``registered`` alone, and V3c's registration holds its `SCAN_STATE` in its window's
  ``activations.values`` alone, so a served window's answer is a reject at the state tie. It answers nothing
  where the chain step refused (DIR/`REFUSED`) or rejected (its answer stands), and refuses where it left no
  DIR/`CHAINED`, the chain's scan read no V3c window or another, or V3C has no window-circuit record beside it. Each
  releases through its own hold and refuses as
  the piece does, keeping ``served_verdicts.sh``'s logs (`LOGS`).
* **relay**: a TCP relay from 127.0.0.1:``--listen`` to 127.0.0.1:``--to``, the local gate's tunnel.
* **front --session** and **close** put the gate's relay session (`auditor.py`'s "A relay session") around the served
  run's own load, in the rehearsal front's place: the load's clients are the run's, and the front sends no group.
  - ``front --session`` asks the gate for ``relay-session``: PORT is the endpoint's, ``--client-port`` the port of
    the peer's loopback the load's clients connect to and ``--windows`` the session's length in execution phases.
    ``--height`` at the piece's default, 10, which signs 204 of them, is raised to the least whose key signs the
    session's (`auditor.warden_height`), and ``--minutes`` is by default `auditor.session_minutes` at that height
    plus `served_session.MARGIN`, at most the gate's: for the longest session, 1119 execution phases, height 13,
    with ``--minutes`` 66 at the scan's 100 ms tick (59 min 40 s of them) and 126 at 200 ms (119 min 21 s). The
    gate refuses any other ``--height`` too low for the session. This tree's gate checks the request first. It
    returns once the gate's ``status`` shows the relay listening: exit 0, with
    SERVE_DIR/`PRIVATE`/RUN/`RELAY` (0600) holding ``client``, 127.0.0.1:CLIENT_PORT on the peer, and
    ``deployment``, the relay's (its schedule, ``origin_unix`` and ``windows``), and one line on stdout naming the
    run, the client's address and that file. Exit 2 when the gate refused the request or a session is open already,
    1 when the session ended before its relay listened, the reason in OUT/session.json. The session then runs to
    its execution phases' end whatever its caller does: nothing ends it early, and its last connection is admitted
    `WardenParams.span` + 1 buckets before that end.
  - ``close`` runs after the load: it waits for the session's end, keeps the lane's fetch where the piece reads it
    (SERVE_DIR/`PRIVATE`/RUN/fetch.tar.gz) and publishes to OUT what the rehearsal front does (``session.json``,
    ``bundle.tgz`` and ``status.json``, through the leak guard), with the front's exits: 0 when the gate's check
    accepted the run.
  - The piece's steps close a session that no ``close`` ended before they read the kept run (`close_open`): the same
    wait and the same kept files, nothing published.
  - The piece answers for a session's run as for a rehearsal's, only when the gate's check accepted the whole run
    (`rehearsed`). One connection the relay failed closed, one cut from s + D on at the session's end, one refused at
    a full queue or one execution phase the relay faulted, whichever load it carried, leaves every window of the
    served run without the certifier's answer. The run's records cover the connections that crossed the relay; that
    nothing else reaches the endpoint is the placement's, as a rehearsal's.
  - ``--gate local`` is the dry run's: this tree's gate on a scratch host under SERVE_DIR/`PRIVATE`, its unit a
    process of the front's (`local_unit`) with a `relay` each way in the place of its ssh.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
#: the protocol ``network-rows`` answers, step 1's rows audit of the network certifier's records
ROWS = "verity/network-rows/v1"
#: the named assumption the certifier's guarantee reads (`NetworkCheck.Period`): the certifier is the auditor's device
PREMISE = "trusted-host"
#: What a placement says of it, word for word, followed by ``; `` and the names
UNMET = "unmet: this piece establishes no trusted-host and gives no bound on that ground"
#: the machine of a chain step whose `CHAINED` names none
UNNAMED = "a machine the chain step did not name"
#: the front's public outcome, copied beside the kept fetch
SESSION = "session.json"
#: What DIR/chain gets of the chain's out/, by file: its JSON with only the key paths named (``a.b`` b under a, ``*``
#: every key at its level). Each is public as the protocol's output, its sample or its structure: the chain's and each
#: step's verdict, ε at δ (the sample's law, population and size) and where it runs in the clear, what the path stubs
#: and how its negatives came out. Everything else stays in the kept run: the scan's answers (its registration and
#: each link-window's answer and row registrations, the period check's evidence less its certificates: witness, as a
#: case's evidence is), the endpoint's report (each Commit window's request, carried, step and output counts, its
#: opening time and its advice's bits), of which only a padded count may be public, report.md and the report's other
#: keys (the certified I/O's counts, the advice's bits as K's term), and the steps' summaries (the scan's admissions
#: and response lengths, the tie's prompt lengths and releases, the warden's check of its witness).
PUBLIC = {"report/report.json": ("schema", "verdict", "path", "scope", "stubbed", "negatives_refused",
                                 "steps.*.verdict", "steps.*.eps_at_delta", "steps.*.in_the_clear")}
#: a piece's refusal to run, which `served_verdicts.py` reads as the subprotocol missing
REFUSED = "refused.txt"
#: the piece's case, as `composite.py` reads it, and where it is kept under the serve dir: a case's evidence and
#: params are witness, so it is never in DIR
CASE = "case.json"
CASES = Path("private") / "pieces" / "certifier"
#: the link-window's five certificates in DIR (`certificates`): the case's evidence's, and nothing else of it
CERTIFICATES = "certificates.json"
#: the scan class of V3c's default window class (`served_verdicts.sh`'s V3C_CLASS)
V3C_SCAN_CLASS = "w63k4"
#: the split piece's chain step's accept, which its period step reads: V3c's directory, and the machine the chain
#: step ran on as that machine names itself (``ran_on``)
CHAINED = "chain.json"
#: the logs `served_verdicts.sh` and `served_verdicts.py` write in DIR, which a refusal keeps
LOGS = ("piece.log", "chain.log", "period.log")
#: every file DIR may hold when the leak guard reads it and the hold (its ``--public``): the answer, the logs, the
#: split's `CHAINED`, the certificates and `PUBLIC`'s
PIECE_FILES = ("answer.json", *LOGS, CHAINED, CERTIFICATES, *(f"chain/{rel}" for rel in PUBLIC))
LEAN_DIR = TREE / "verity" / "protocols" / "accounting" / "certifier" / "verifier" / "lean"
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

def front(a: argparse.Namespace, *, sleep: Callable[[float], object] = time.sleep) -> int:
    serve_dir, out = a.serve_dir.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    keep = serve_dir / PRIVATE
    keep.mkdir(parents=True, exist_ok=True)
    for p in (serve_dir / "private", keep):
        p.chmod(0o700)
    if a.session:
        return open_session(a, keep, out, sleep=sleep)
    a.requests = a.prompts
    a.keep, a.out, a.at, a.window_per_group = keep, out, None, True
    if a.minutes is None:
        tick = a.bucket_ms or SS.A.SCAN["--bucket-ms"]
        a.minutes = SS.A.warden_group_minutes(1, a.tokens, a.height, tick) + SS.MARGIN
    if a.gate == "local":
        a.tunnel = lambda ssh, port: [sys.executable, str(Path(__file__).resolve()), "relay", "--listen", str(port),
                                      "--to", str(a.port)]
    try:
        a.gate = SS.gate_at(a.gate)
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


def relays(pairs: list[tuple[int, int]]) -> None:
    """`relay` for each (listen, to) of `pairs`, the last in this thread once every other listens."""
    *more, last = pairs
    for pair in more:
        listening = threading.Event()
        threading.Thread(target=relay, args=pair, kwargs={"ready": listening.set}, daemon=True).start()
        listening.wait()
    relay(*last)


# -------------------------------------------------------------------------------------------------------------- session

#: a relay session the front opened, under SERVE_DIR/`PRIVATE`/RUN (0600): the run, its gate and key, the front's
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
                            certifier_active=lambda r: a.unit is not None and live(a.unit))
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
        a.gate = SS.gate_at(a.gate)
        if a.client_port is None or a.windows is None:
            raise SS.Refused("--session: with --client-port PORT, the peer's loopback port the load's clients "
                             "connect to, and --windows N, the session's length in execution phases")
        if a.warmup:
            raise SS.Refused("--session sends no request of its own: --warmup is the rehearsal front's")
        if a.height == HEIGHT:
            a.height = SS.A.warden_height(a.windows)
        if a.minutes is None:
            tick = a.bucket_ms or SS.A.SCAN["--bucket-ms"]
            a.minutes = min(SS.A.session_minutes(a.windows, a.height, SS.A.SCAN["--window"] * tick) + SS.MARGIN,
                            SS.A.CERTIFIER_MINUTES[1])
        with tempfile.TemporaryDirectory() as d:
            try:
                SS.A.REHEARSALS[SS.words(a)[0]](SS.words(a)[1:], SS.scratch_host(Path(d), a.peer))
            except SS.A.Refused as e:
                raise SS.Refused(f"the gate refuses it: {e}") from None
        if opened := open_sessions(keep):
            raise SS.Refused(f"relay session {opened[0]['run']} is open under {keep}: close it first "
                             "(certifier_piece.py close)")
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
    end = SS.A.certifier_run(host, a.run, pinned=SS.pinned(), now=lambda: SS.datetime.now(SS.UTC),
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
            return SS.A.certifier_fetch(a.run, SS.scratch_host(keep / "local-gate", a.peer), witness=True)
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
    keep = serve_dir / PRIVATE
    try:
        opened = open_sessions(keep)
        if len(opened) != 1:
            raise SS.Refused(f"{len(opened)} relay sessions open under {keep}, not one: the front (front --session) "
                             "opens it before the load")
        b = session_args(opened[0], keep, out)
        b.close = lambda hold: ended_fetch(b, keep, hold, sleep)
        rc = SS.session(b, b"", [])
        if (doc := out / "session.json").is_file():
            shutil.copyfile(doc, keep / b.run / SESSION)
        return rc
    except SS.Refused as e:
        (out / "session.json").write_text(json.dumps({"refused": str(e)}, indent=1) + "\n")
        print(f"refused: {e}", file=sys.stderr)
        return e.code


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


def ran_on(chained: Path | None = None) -> dict[str, str]:
    """Which machine ran each of the piece's own steps, by the name each machine gives itself as it runs the step:
    this one for the period check and its proofs' challenges, and for the chain's checks and samples too unless
    ``chained``, the split's `CHAINED`, was written by a chain step, which names its own machine there."""
    here = socket.gethostname()
    if chained is None:
        return {"chain": here, "period": here}
    return {"chain": json.loads(chained.read_text()).get("ran_on") or UNNAMED, "period": here}


def placement(serve_dir: Path, run: str, steps: dict[str, str] | None = None) -> dict:
    """Which machine ran each party and each of the piece's own steps, as labels read from the run: the network
    certifier ran on the gate's machine (the kept `SESSION`'s ``gate``; ``local`` is the developer's node), and
    ``steps`` (`ran_on`, this machine by default) names the machine that ran the chain's checks and samples and the one
    that ran the period check and its proofs' challenges. This piece establishes no `trusted-host` and gives no bound
    on that ground (`UNMET`), whatever the names say. ``kind`` is ``prover-node`` where the certifier ran on the
    developer's node, else ``not-placed``. ``links`` are the node's to the gate, and K counts the certified one
    alone."""
    kept = serve_dir / PRIVATE / run
    if not (kept / SESSION).is_file():
        raise NotRun(f"{kept} has no {SESSION}, which names the gate the certifier ran on: keep the run with this "
                     "tree's front")
    gate = json.loads((kept / SESSION).read_text())["gate"]
    node = SS.served_peer(kept / "fetch.tar.gz") or socket.gethostname()
    steps = ran_on() if steps is None else steps
    host = node if gate == "local" else gate.rpartition("@")[2]
    names = ((f"the network certifier ran on the developer's node {node} (the front's --gate local)" if gate == "local"
              else f"the network certifier ran on {host}, not the developer's {node}")
             + f"; the chain's checks and samples ran on {steps['chain']}, and the period check and its proofs' "
               f"challenges on {steps['period']}")
    where = {"parties": {"network_certifier": host, "prover": node, "verifier": steps["period"]}, "steps": dict(steps),
             PREMISE: f"{UNMET}; {names}"}
    if gate == "local":
        return {"kind": "prover-node", **where, "links": [], "uncounted": f"every link of {node}'s"}
    return {"kind": "not-placed", **where,
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


def certifier_run(d: Path) -> Path | None:
    """The network certifier run in the unpacked fetch ``d``, None where it holds none."""
    return next((p.parent.parent if p.parent.name == SS.A.WITNESS else p.parent for p in d.rglob("certified.jsonl")),
                None)


def unpack(fetch: Path) -> tuple[Path, Path]:
    """The kept fetch unpacked in a 0700 scratch directory beside it, and its network certifier run."""
    d = Path(tempfile.mkdtemp(dir=fetch.parent, prefix="unpacked-"))
    try:
        with tarfile.open(fetch, "r:gz") as tar:
            tar.extractall(d, filter="data")
    except (OSError, tarfile.TarError) as e:
        rm(d)
        raise NotRun(f"{fetch}: {type(e).__name__}: {e}") from None
    run = certifier_run(d)
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
    """Each window request's certified record by its user; Rejected when one isn't the network certifier's."""
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
    """N2's sample and the tie's execution phases for a run of ``windows`` execution phases: every one when they are at
    most `served_session.chain`'s 8 (whose sample of 8 a shorter run refuses), unless a CR_K=V of ``sets`` names
    them."""
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
        signal.getsignal(signal.SIGTERM)(came[0], None)
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
    """`served_session.py chain` on the kept run (or ``cmd``, run as ``bash -c CMD _ KEEP RUN ENDPOINT CR_K=V ...``),
    its run dir (RESEARCH_RUN_DIR, whose out/ holds the report) and quarantine private under KEEP/RUN."""
    d = keep / run
    env = dict(os.environ, RESEARCH_RUN_DIR=str(d / "chain"), CR_QUARANTINE=str(d / "quarantine"))
    (d / "chain").mkdir(mode=0o700, exist_ok=True)
    argv = ["bash", "-c", cmd, "_", str(keep), run, str(endpoint), *sets] if cmd else [
        sys.executable, str(HERE / "served_session.py"), "chain", "--run", run, "--keep", str(keep),
        "--endpoint", str(endpoint), "--eos-ids", eos, *(["--draw", str(draw)] if draw else []), *sets]
    with open(d / "chain" / "stdout.log", "w") as log:
        return run_group(argv, cwd=TREE, env=env, stdout=log, stderr=subprocess.STDOUT)


def lean_exe(name: str, given: str | None) -> str:
    """``given``, else the executable ``name`` of `LEAN_DIR`, built here."""
    if given:
        return given
    got = captured(["lake", "build", name], cwd=LEAN_DIR)
    exe = LEAN_DIR / ".lake" / "build" / "bin" / name
    if got.returncode or not exe.is_file():
        raise NotRun(f"lake build {name} (exit {got.returncode}): {(got.stdout + got.stderr)[-400:]}")
    return str(exe)


def period_check(exe: str, case: dict, d: Path) -> dict:
    import scan_period as SP
    try:
        return SP.lean_check(exe, case, d, run=captured)
    except (SystemExit, OSError, ValueError) as e:
        raise NotRun(f"network-period check: {type(e).__name__}: {e}") from None


def rows_check(exe: str, run: Path, record: dict) -> None:
    """Step 1's rows audit of the network certifier run ``run`` (module docstring) by ``network-rows`` ``exe``, under
    ``record``, the record of its deployment (`warden_chain.RECORD`): `Rejected` on its reject, `NotRun` where the case
    can't be read or the check gives no `ROWS` accept or reject."""
    import scan_period as SP
    import warden_chain as WC
    try:
        case = WC.rows_audit_case(run, record)
    except (SystemExit, OSError, KeyError, ValueError) as e:
        raise NotRun(f"the network certifier's rows case: {type(e).__name__}: {e}") from None
    scratch = Path(tempfile.mkdtemp(dir=run.parent, prefix="rows-"))
    try:
        got = SP.lean_check(exe, case, scratch, run=captured)
    except (SystemExit, OSError, ValueError) as e:
        raise NotRun(f"network-rows check: {type(e).__name__}: {e}") from None
    finally:
        rm(scratch)
    if got.get("protocol") != ROWS or got.get("verdict") not in ("accept", "reject"):
        raise NotRun(f"network-rows answered no {ROWS} accept or reject: {json.dumps(got)[:400]}")
    if got["verdict"] == "reject":
        why = re.sub(r"[0-9]+", "N", str(got["reason"]))
        raise Rejected(f"the network certifier's records, audited as rows: {why}")


def own_rows(exe: str) -> Callable[[Path], None]:
    """`decide`'s ``rows``: `rows_check` by ``network-rows`` ``exe`` under the kept run's own deployment
    (`warden_chain.own_record`), the one the piece's other checks read."""
    import warden_chain as WC

    def rows(run: Path) -> None:
        try:
            record = WC.own_record(run)
        except (SystemExit, OSError, KeyError, ValueError) as e:
            raise NotRun(f"the network certifier's rows case: {type(e).__name__}: {e}") from None
        rows_check(exe, run, record)
    return rows


def case_of(warden: Path, scan: Path, v3c: Path | None = None, circuit: str | None = None) -> dict:
    import scan_period as SP
    try:
        return SP.rows_case(warden, scan, v3c=v3c, circuit=circuit)
    except (SystemExit, OSError, KeyError, ValueError) as e:
        raise NotRun(f"the period's rows case: {type(e).__name__}: {e}") from None


def class_circuit(v3c: Path) -> str:
    """The class circuit V3c's audit is held to: the Program of the verifier's own window-circuit record beside V3C
    (`served_verdicts.sh`'s ``window-circuit-w<K>-<class>.json`` beside ``v3c-w<K>-<class>``,
    `scan_period.circuit_of_record`); `NotRun` without one."""
    import scan_period as SP
    try:
        return SP.circuit_of_record(v3c.parent / f"window-circuit-{v3c.name.removeprefix('v3c-')}.json")
    except (SystemExit, OSError, ValueError) as e:
        raise NotRun(f"the verifier's window-circuit record: {type(e).__name__}: {e}") from None


def v3c_case(v3c: Path) -> Callable[[Path, Path], dict]:
    """The period step's case: `case_of` with V3C/period.json as its one compute audit, held to the verifier's class
    circuit (`class_circuit`), once the chain's scan read V3C/window's trees (its ``summary.json``'s ``v3c``); `NotRun`
    for a scan of no V3c window or another, or no window-circuit record."""
    def case(warden: Path, scan: Path) -> dict:
        try:
            read = (json.loads((scan / "summary.json").read_text()).get("v3c") or {}).get("window")
        except (OSError, ValueError) as e:
            raise NotRun(f"the chain's scan summary: {type(e).__name__}: {e}") from None
        if read != str(v3c / "window"):
            raise NotRun(f"the chain's scan read {read or 'no'} V3c window, not {v3c / 'window'}: run the chain step "
                         "on this V3c")
        return case_of(warden, scan, v3c / "period.json", class_circuit(v3c))
    return case


def public(doc: dict, keys: tuple[str, ...]) -> dict:
    """``doc`` with only the key paths ``keys`` name (`PUBLIC`'s form)."""
    out = {}
    for k, v in doc.items():
        rest = [r for head, _, r in (p.partition(".") for p in keys) if head in ("*", k)]
        if "" in rest:
            out[k] = v
        elif rest and isinstance(v, dict):
            out[k] = public(v, tuple(rest))
    return out


def publish(chain_out: Path, d: Path) -> None:
    """``d``/chain: each `PUBLIC` file the chain wrote, less every key `PUBLIC` doesn't name."""
    for rel, keys in PUBLIC.items():
        if (chain_out / rel).is_file():
            (d / "chain" / rel).parent.mkdir(parents=True, exist_ok=True)
            doc = public(json.loads((chain_out / rel).read_text()), keys)
            (d / "chain" / rel).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def decide(serve_dir: Path, window: int, d: Path, *, chain: Callable[[Path, str, Path, int], int] | None,
           check: Callable[[dict, Path], dict] | None, rows: Callable[[Path], None],
           case: Callable[[Path, Path], dict] = case_of) -> tuple[dict | None, Path | None]:
    """Window ``window``'s answer (module docstring) and the unpacked fetch (for the leak guard), its files in ``d``;
    ``rows`` is step 1's rows audit of the network certifier run (`own_rows`); ``chain`` is given KEEP, the run, the
    endpoint's records and the number of its link's execution phases the run holds, or is None to read the report of the
    chain the split's chain step ran; ``check`` None stops once the chain accepted, with no answer (the split's chain
    step). `NotRun` when a step can't run."""
    import chain_record as CR
    import warden_chain as WC
    keep = serve_dir / PRIVATE
    run = kept_run(serve_dir)
    reqs = window_requests(serve_dir, window)
    unpacked = None
    try:
        unpacked, warden = unpack(keep / run / "fetch.tar.gz")
        rehearsed(warden)
        rows(warden)
        try:
            io = WC.io(warden)
            dep, params = WC.deployment(warden)
        except SystemExit as e:
            raise Rejected(f"the kept warden run: {e}") from None
        T = params.window
        certified = bind(reqs, io)
        out = keep / run / "chain" / "out"
        report = out / "report" / "report.json"
        if chain is None:
            if not report.is_file():
                raise NotRun(f"no chain report under {keep / run}: the chain step didn't run here")
        else:
            rc = chain(keep, run, endpoint_copy(serve_dir, keep / run / "endpoint"), int(dep["windows"]))
            publish(out, d)
            if not report.is_file():
                raise NotRun(f"the chain wrote no report (exit {rc}): see chain/ and the kept run's stdout.log")
        doc = json.loads(report.read_text())
        if doc.get("verdict") not in ("accepted", CR.RESTS):
            raise Rejected(f"the chain {doc.get('verdict')}: {json.dumps(doc.get('why'), default=str)}"[:1800])
        try:
            (w,) = [x["window"] for x in json.loads((out / "scan" / "answers.json").read_text())["windows"]]
        except (OSError, ValueError, KeyError) as e:
            raise NotRun(f"the scan's answers.json names no one link-window: {type(e).__name__}: {e}") from None
        in_link_window(certified, T, w)
        if check is None:
            return None, unpacked
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
    """The record's leak guard on DIR and the hold against the warden run's witness, DIR declared to hold only
    `PIECE_FILES`, which releases the hold to DIR once it passes (0 when nothing was found, 3 when a find, a file
    holding a piece of the witness or of another path, was moved to the quarantine)."""
    quarantine.mkdir(mode=0o700, parents=True, exist_ok=True)
    got = captured([sys.executable, str(HERE / "leak_guard.py"), str(d), str(quarantine),
                    *(["--gates", str(unpacked)] if unpacked is not None else []), "--hold", str(hold),
                    "--release-to", str(d), *(x for f in PIECE_FILES for x in ("--public", f)),
                    "--report", str(quarantine / "piece-guard.json")])
    print(f"leak guard (exit {got.returncode}): {got.stdout.strip()[-400:]}", flush=True)
    return got.returncode


def certificates(evidence: bytes) -> dict:
    """The certificates of a case's ``evidence``, what the network certifier signed where the period check accepts
    them, each with its own fields alone (`scan_period.CERTIFICATE_KEYS`): ``records``, each link-window's three
    records' certificates (requests, responses, fails), and ``rows``, its two row registrations' (ingress, egress),
    with the link, direction and window its registration names, which the certificate's message signs. Nothing else of
    the evidence is here: no registration's head, no answer and no audit."""
    from scan_period import CERTIFICATE_KEYS
    windows = json.loads(evidence)["windows"]
    rows = [{"link": r["link"], "direction": r["direction"], "window": r["window"], "key_id": r["certificate"]["key"],
             "digest": r["certificate"]["digest"], "signature": r["certificate"]["signature"]}
            for x in windows for r in x["registrations"]]
    return {"records": [{k: c[k] for k in CERTIFICATE_KEYS} for x in windows for c in x["certificates"]],
            "rows": [{k: r[k] for k in CERTIFICATE_KEYS} for r in rows]}


def write_case(hold: Path, case: Path) -> None:
    """The hold's period files, the three `network-period` read, made the case ``case`` (its directory 0700, the file
    0600) and the hold's `CERTIFICATES`, and gone from the hold: the case is under no path the leak guard releases.
    Nothing where the period check wrote none. The certificates are written compact, as the evidence holds them."""
    p = hold / "period"
    commitment, evidence, params = p / "commitment.bin", p / "evidence.json", p / "params.json"
    if not all(f.is_file() for f in (commitment, evidence, params)):
        return
    raw = evidence.read_bytes()
    try:
        signed = certificates(raw)
    except (KeyError, TypeError) as e:
        raise NotRun(f"the period's evidence names no certificates: {type(e).__name__}: {e}") from None
    case.parent.parent.mkdir(mode=0o700, exist_ok=True)
    case.parent.mkdir(mode=0o700, exist_ok=True)
    case.parent.chmod(0o700)
    with open(case, "wb", opener=lambda path, flags: os.open(path, flags, 0o600)) as out, open(params, "rb") as src:
        os.fchmod(out.fileno(), 0o600)
        out.write(b'{"commitment":"%s","evidence":"%s","params":' % (commitment.read_bytes().hex().encode(),
                                                                     raw.hex().encode()))
        shutil.copyfileobj(src, out, 1 << 24)
        out.write(b"}\n")
    (hold / CERTIFICATES).write_text(json.dumps(signed, sort_keys=True, separators=(",", ":")) + "\n")
    rm(p)
    print(f"case: the serve dir's {CASES / CASE}, {case.stat().st_size} bytes; {CERTIFICATES}: "
          f"{len(signed['records'])} records' and {len(signed['rows'])} rows' certificates", flush=True)


def rm(*paths: Path) -> None:
    """``rm -rf`` as a child process: the suites' read guard misreads `shutil.rmtree`'s fd-relative walk."""
    if paths:
        subprocess.run(["rm", "-rf", "--", *map(str, paths)], check=False)


def refuse(d: Path, why: str) -> int:
    """DIR emptied but for its logs, and DIR/`REFUSED` naming why: no answer, so the certifier reads missing. SIGTERM is
    ignored while it writes and then handled as before, so the children of a process that refused don't inherit it
    ignored."""
    was = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        rm(*(p for p in d.iterdir() if p.name not in LOGS))
        (d / REFUSED).write_text(why + "\n")
        print(f"CERTIFIER refused: {why}", flush=True)
    finally:
        signal.signal(signal.SIGTERM, was)
    return 3


def ended(limit: str) -> Callable[[int, object], None]:
    """The piece's SIGTERM handler under its time limit ``limit``: held while a step's group runs (`run_group`), else
    `NotRun` naming the limit."""
    def on(signum: int, _frame) -> None:
        if _held is not None:
            _held.append(signum)
            return
        raise NotRun(f"{signal.Signals(signum).name}: the piece's time limit ({limit}) ended it before a verdict")
    return on


def said(got: dict) -> int:
    print(f"CERTIFIER {json.dumps({k: got.get(k) for k in ('verdict', 'reason', 'public')}, sort_keys=True)}",
          flush=True)
    return 0 if got.get("verdict") == "accept" else 1


def close_open(serve_dir: Path, *, sleep: Callable[[float], object] = time.sleep) -> None:
    """The piece's guard for a relay session nobody closed (no `close` ran after the load): its end awaited and its
    fetch and `SESSION` kept as `close` keeps them, with nothing published. Nothing where none is open; `NotRun` when
    it can't be closed."""
    keep = serve_dir / PRIVATE
    opened = open_sessions(keep)
    if not opened:
        return
    if len(opened) > 1:
        raise NotRun(f"{len(opened)} relay sessions open under {keep}, not one: the run's load goes through one")
    (keep / "held").mkdir(mode=0o700, exist_ok=True)
    hold = Path(tempfile.mkdtemp(dir=keep / "held", prefix="close-"))
    try:
        b = session_args(opened[0], keep, hold)
        b.close = lambda h: ended_fetch(b, keep, h, sleep)
        doc, _ = SS._session(b, b"", [], keep, hold)
        shutil.copyfile(hold / "session.json", keep / doc["run"] / SESSION)
    except SS.Refused as e:
        raise NotRun(f"relay session {opened[0]['run']}, which no close ended, couldn't be closed here: {e}") from None
    finally:
        rm(hold)
    print(f"relay session {doc['run']} closed here: no close ran after the load (certifier_piece.py close)", flush=True)


def settle(serve_dir: Path, window: int, d: Path, limit: str,
           step: Callable[[Path], tuple[dict | None, Path | None]], chained: Path | None = None) -> dict | None:
    """``step``'s answer: its files written to a fresh hold under SERVE_DIR/`PRIVATE`/held (``step`` is given it), an
    accept's profile with its `placement` (with the machines that ran the piece's steps, `ran_on` of ``chained``), the
    hold's period files made the case under SERVE_DIR/`CASES` and the hold's certificates (`write_case`), and the
    hold then released to DIR by the leak guard; None where the split's
    chain step accepted. The case is this step's: one an earlier step left goes first, and one written here goes
    unless the step ends with its files released. `NotRun` (SIGTERM included, naming ``limit``) for a step that can't
    run."""
    was, unpacked, hold = signal.signal(signal.SIGTERM, ended(limit)), None, None
    case, released = serve_dir / CASES / CASE, False
    try:
        case.unlink(missing_ok=True)
        close_open(serve_dir)
        if not any((serve_dir / PRIVATE).glob("*/fetch.tar.gz")):
            raise NotRun(f"no warden run kept under {serve_dir / PRIVATE}: the run's load didn't go through the "
                         "certifier's front (CERTIFIER_FRONT unset, or the gate, its tunnel or ssh failed)")
        (serve_dir / PRIVATE / "held").mkdir(mode=0o700, exist_ok=True)
        hold = Path(tempfile.mkdtemp(dir=serve_dir / PRIVATE / "held", prefix=f"window-{window}-"))
        got, unpacked = step(hold)
        if got is not None:
            if isinstance(got.get("profile"), dict):
                where = placement(serve_dir, kept_run(serve_dir), ran_on(chained))
                got = got | {"profile": got["profile"] | {"placement": where}}
            (hold / "answer.json").write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
        write_case(hold, case)
        g = guard(d, hold, unpacked, serve_dir / PRIVATE / "quarantine")
        if g not in (0, 3) or got is not None and not (d / "answer.json").is_file():
            raise NotRun(f"the leak guard failed (exit {g}): DIR withheld, the piece's files left in {hold}")
        released = True
        return got
    except (OSError, ValueError, tarfile.TarError) as e:
        traceback.print_exc()
        raise NotRun(f"the piece failed on a file: {type(e).__name__}: {e}") from None
    finally:
        if not released:
            case.unlink(missing_ok=True)
        if unpacked is not None:
            rm(unpacked)
        if hold is not None and hold.is_dir() and not any(p.is_file() for p in hold.rglob("*")):
            rm(hold)
        signal.signal(signal.SIGTERM, was)


def sets_ok(sets: list[str]) -> bool:
    bad = [kv for kv in sets if not kv.startswith("CR_") or "=" not in kv]
    for kv in bad:
        print(f"refused: {kv!r}: CR_K=V, a chain_record.sh variable", file=sys.stderr)
    return not bad


def piece(a: argparse.Namespace) -> int:
    serve_dir, d = a.serve_dir.resolve(), a.dir.resolve()
    d.mkdir(parents=True, exist_ok=True)
    if not sets_ok(a.set):
        return 2

    def step(hold: Path) -> tuple[dict | None, Path | None]:
        exe, rows = lean_exe("network-period", a.lean), lean_exe("network-rows", a.rows)
        chain = lambda keep, run, ep, n: run_chain(keep, run, ep, a.eos_ids, a.draw, sized(n, a.set),  # noqa: E731
                                                   a.chain_cmd)
        return decide(serve_dir, a.window, hold, chain=chain, check=lambda c, p: period_check(exe, c, p),
                      rows=own_rows(rows))
    try:
        got = settle(serve_dir, a.window, d, "CERTIFIER_TIMEOUT", step)
    except NotRun as e:
        return refuse(d, str(e))
    return said(got)


def v3c_scan(v3c: Path, cls: str, sets: list[str]) -> list[str]:
    """``sets`` with the scan's class and V3c's window first in CR_SCAN_ARGS (chain_record.sh splits it on whitespace)."""
    window = v3c / "window"
    if not (window / "registration.json").is_file():
        raise NotRun(f"V3c committed no window at {window}: the chain's scan reads its scan state and slot entries")
    if any(c.isspace() for c in f"{window}{cls}"):
        raise NotRun(f"{window}: chain_record.sh splits CR_SCAN_ARGS on whitespace, so give V3c a path without it")
    given = next((kv.partition("=")[2] for kv in sets if kv.startswith("CR_SCAN_ARGS=")), "")
    return [*(kv for kv in sets if not kv.startswith("CR_SCAN_ARGS=")),
            f"CR_SCAN_ARGS={f'--class {cls} --v3c {window} {given}'.strip()}"]


def chain_step(a: argparse.Namespace) -> int:
    serve_dir, d, v3c = a.serve_dir.resolve(), a.dir.resolve(), a.v3c.resolve()
    d.mkdir(parents=True, exist_ok=True)
    for f in (d / "answer.json", d / CHAINED, d / REFUSED, d / CERTIFICATES, serve_dir / CASES / CASE):
        f.unlink(missing_ok=True)
    if not sets_ok(a.set):
        return 2
    try:
        sets = v3c_scan(v3c, a.scan_class, a.set)
        chain = lambda keep, run, ep, n: run_chain(keep, run, ep, a.eos_ids, a.draw, sized(n, sets),  # noqa: E731
                                                   a.chain_cmd)
        got = settle(serve_dir, a.window, d, "CERTIFIER_TIMEOUT",
                     lambda hold: decide(serve_dir, a.window, hold, chain=chain, check=None,
                                         rows=own_rows(lean_exe("network-rows", a.rows))))
    except NotRun as e:
        return refuse(d, str(e))
    if got is not None:
        return said(got)
    (d / CHAINED).write_text(json.dumps({"chain": "accepted", "ran_on": socket.gethostname(), "v3c": str(v3c)},
                                        indent=1, sort_keys=True) + "\n")
    print(f"CERTIFIER-CHAIN accepted on V3c's window {v3c / 'window'}: the period step checks its period.json",
          flush=True)
    return 0


def period_step(a: argparse.Namespace) -> int:
    serve_dir, d, v3c = a.serve_dir.resolve(), a.dir.resolve(), a.v3c.resolve()
    d.mkdir(parents=True, exist_ok=True)
    if (d / REFUSED).is_file():
        print(f"CERTIFIER refused at the chain: {(d / REFUSED).read_text().strip()}", flush=True)
        return 3
    if (d / "answer.json").is_file():
        return said(json.loads((d / "answer.json").read_text()))

    def step(hold: Path) -> tuple[dict | None, Path | None]:
        if not (d / CHAINED).is_file():
            raise NotRun(f"no {CHAINED} in DIR: the chain step (CERTIFIER_CHAIN_CMD) didn't run, or didn't finish")
        exe, rows = lean_exe("network-period", a.lean), lean_exe("network-rows", a.rows)
        return decide(serve_dir, a.window, hold, chain=None, check=lambda c, p: period_check(exe, c, p),
                      rows=own_rows(rows), case=v3c_case(v3c))
    try:
        got = settle(serve_dir, a.window, d, "CERTIFIER_PERIOD_TIMEOUT", step, d / CHAINED)
    except NotRun as e:
        return refuse(d, str(e))
    return said(got)


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
    f.add_argument("--scan-class", choices=sorted(SS.A.SCAN_CLASSES), default=V3C_SCAN_CLASS,
                   help=f"the gate's scan class, the schedule its network certifier runs at (default {V3C_SCAN_CLASS})")
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
    q = sub.add_parser("piece")
    q.add_argument("serve_dir", type=Path)
    q.add_argument("window", type=int)
    q.add_argument("dir", type=Path)
    q.add_argument("--eos-ids", default="128001")
    q.add_argument("--lean", help="network-period, built (else built here)")
    q.add_argument("--rows", help="network-rows, built (else built here)")
    q.add_argument("--draw", type=int, help="the scan's sample, K proof blocks (every proof block by default)")
    q.add_argument("--chain-cmd", help="replaces served_session.py chain: bash -c CMD _ KEEP RUN ENDPOINT CR_K=V ...")
    q.add_argument("set", nargs="*", help="CR_K=V")
    for name in ("chain", "period"):
        q = sub.add_parser(name)
        q.add_argument("serve_dir", type=Path)
        q.add_argument("window", type=int)
        q.add_argument("dir", type=Path)
        q.add_argument("v3c", type=Path, help="V3c's directory: window/ (window-commit's) and period.json")
        q.add_argument("--rows", help="network-rows, built (else built here)")
    q = sub.choices["chain"]
    q.add_argument("--eos-ids", default="128001")
    q.add_argument("--draw", type=int, help="the scan's sample, K proof blocks (every proof block by default)")
    q.add_argument("--class", dest="scan_class", default=V3C_SCAN_CLASS,
                   help=f"the scan's class, V3c's window class's (default {V3C_SCAN_CLASS})")
    q.add_argument("--chain-cmd", help="replaces served_session.py chain: bash -c CMD _ KEEP RUN ENDPOINT CR_K=V ...")
    q.add_argument("set", nargs="*", help="CR_K=V")
    sub.choices["period"].add_argument("--lean", help="network-period, built (else built here)")
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
    if a.cmd == "piece":
        return piece(a)
    if a.cmd == "chain":
        return chain_step(a)
    if a.cmd == "period":
        return period_step(a)
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
