"""certifier-piece: the network certifier's piece of `served_verdicts.sh` (its ``CERTIFIER_CMD``), and the front that puts
the active network certifier in front of serve-openai there (its ``CERTIFIER_FRONT``).

    python certifier_piece.py front PORT SERVE_DIR OUT [--gate USER@HOST | --gate local] [--key KEY] [--peer NAME]
        [--prompts 4] [--prompt-ids 24 | --prompt-ids LO:HI] [--tokens 4] [--vocab 128000] [--scan-class w63k4]
        [--height 10] [--minutes M] [--bucket-ms MS] [--warmup N] [--stop-prompt N] [--attack NAME [--attack-nth K]]
    python certifier_piece.py piece SERVE_DIR WINDOW DIR [--certifier-key RECORD] [--lean EXE] [--rows EXE] [--draw K]
        [--chain-cmd CMD] [CR_K=V ...]
    python certifier_piece.py chain SERVE_DIR WINDOW DIR V3C [--certifier-key RECORD] [--rows EXE] [--draw K]
        [--class w63k4] [--chain-cmd CMD] [CR_K=V ...]
    python certifier_piece.py period SERVE_DIR WINDOW DIR V3C [--certifier-key RECORD] [--lean EXE] [--rows EXE]
    python certifier_piece.py relay --listen PORT --to PORT

* **front** runs before served_verdicts.sh's requests when ``CERTIFIER_FRONT`` names it: the gate's scan rehearsal
  (`served_session.session`, ``warden-rehearse --scan --prompts --window-per-group``) against serve-openai on PORT, one
  group of ``--prompts`` requests (at most the scan's 4 lanes) of ``--prompt-ids`` ids below ``--vocab`` (``LO:HI``:
  each prompt's length drawn in LO … HI), sampled from a fresh secret seed, each with ``--tokens`` its max_tokens, which
  the model's EOS ends sooner. A group is sent half a bucket into execution phase 0
  and its cells are execution phase 0's alone (``--tokens`` at most 4 keeps the group in one execution phase), so the
  engine admits the group together and closes it into one Commit window, the run's earliest: the window the run audits (with
  ``REQUESTS=1`` the requests after it make a smaller window, which `served_verdicts.py window` passes over). The
  defaults fill the scan's 4 lanes with prompts of at most 32 ids and 4 outputs each, and the gate runs the network
  certifier at its scan class's schedule (``--scan-class``, `V3C_SCAN_CLASS`: w63k4, at 5 egress cells; `gate_params`),
  the one the chain's scan at that class takes, whose link-window of 32 buckets is a window of the window circuit's
  class 8,32,4,4,16 (8 slots, 4 lanes at each of its 2 boundaries; `prove_path.py window-class`). The scan's schedule
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
  make a slow endpoint pass; a tick whose D·τ covers its answer does. ``--stop-prompt N`` then puts last in the group
  a prompt whose answer the model's own EOS ends (`stop_prompt`): up to N fresh prompts of HI ids, each sent as a
  warm-up is, until one's answer finishes on a stop within ``--tokens``. The gate's requests are greedy, as a warm-up
  is: the rehearsal's request records the gate's default temperature 0 and top_p 1, which the network certifier's
  client puts on every request (`auditor.sampling_argv`). Serving is batch-invariant (served_verdicts.sh's
  ``VLLM_BATCH_INVARIANT=1``), so the group's answer to that prompt stops alike; none of N stopping refuses the front,
  before the gate is asked. The gate's fetch, ``witness/`` with it, is kept under SERVE_DIR/`PRIVATE`/RUN/fetch.tar.gz
  (0700, the serve dir stays on the node);
  OUT gets ``session.json`` and ``bundle.tgz``, the public outcome, and the run's kept directory a copy of
  ``session.json`` (`SESSION`), whose gate the piece's placement names. ``--gate local`` is the dry run's (this
  tree's gate in this process, a `relay` to PORT its tunnel): a warden on the prover's node is no auditor's device,
  so it never stands for the gate. ``--attack NAME`` (after the warm-ups) puts the prover's relay playing attack NAME
  (`attacks.py relay`; ``--attack-nth K``, the completion request it plays on) between the gate's tunnel and PORT, at
  the scan's cells and delay and the front's tick (`attack_relay`), its log the prover's under
  SERVE_DIR/`PRIVATE`/attacks; `attacks.py judge` reads the outcome. Exit as `served_session.session`'s.
* **piece** writes window WINDOW's core answer to DIR/answer.json (`verity/certifier/v0`, `NetworkCheck.Period`'s rows
  form): accept only when ``network-rows check`` accepted the network certifier's records, the chain accepted and
  ``network-period check`` accepted the period, whose answer it then is.
  1. The kept network certifier run: exactly one fetch under SERVE_DIR/`PRIVATE` (with none the run's load didn't go
     through the front: it wasn't set, or the gate, its tunnel or ssh failed), held to the auditor's record of the
     run's deployment (`held`: ``--certifier-key RECORD``, a `warden_chain.RECORD` file, the gate's
     own ``out/deployment.json`` of the run as the auditor read it off the gate, with its SHA-512, key and session).
     Every ``deployment.json`` the fetch holds must be the record's, and every check below reads the network
     certifier's key, session and schedule off the record, never off the fetch, which the prover's node holds and
     `trusted-host` doesn't cover; with no record, or a fetch that isn't the record's (rewritten, or re-signed under
     another key), the piece refuses. The fetch's rehearsal must be one the gate's check passed (its ``end.json``: exit
     0 and accepted, as the front's exit reads it): one that failed (each request late past the network certifier's
     D·τ, or cut) certified no served answer, so there is none for the protocol to reject, and the refusal names the
     network certifier's own check's problems (`rehearsed`). One whose certified
     egress holds a cell that is neither an id-form cell nor a `FAIL`, bytes the server sent that are no output's, is a
     reject. Its records are then audited as rows (`rows_check`): `warden_chain.rows_audit_case` of the run, checked
     by ``network-rows`` (``--rows``, else built in verity/core/protocols/network/lean; `NetworkCheck.RowsAudit`,
     `ROWS`), which holds each record to its certified head and each certificate to the record's key in its session,
     then runs the certifier program's rows audit (`NetworkCertifier.Rows.audit`): every egress row must be
     the one the schedule the records show renders, so cells cut, clocked or ordered otherwise are refused. Its reject
     is the piece's, with its reason's numbers as N. The case's files go to a 0700 scratch directory beside the
     unpacked run, removed once the check has answered, so the records reach no hold.
  2. The audited Commit period's requests are certified ones (`bind`): each one's user names one of the network
     certifier's certified requests (`warden_chain.io`), it carries the network certifier's run in its release, with
     the same prompt and streamed ids; and of the requests the endpoint recorded carrying that run, no two name one
     certified request and every certified request is one; else reject. These read the endpoint's records, the
     prover's, in the clear: they refuse what a relay between the network certifier and an honest endpoint does
     (`attacks.py`); the verdict of record waits on the egress tie, which binds the committed values to the certified
     cells.
  3. The chain (`served_session.py chain`: chain_record.sh's network certifier path, certs, N2 and scan, which opens
     no endpoint records or advice) on the kept run, its run dir and quarantine under SERVE_DIR/`PRIVATE`/RUN, so
     nothing of the witness reaches a record. N2 takes every execution phase of a run of at most 8 (`sized`). Its
     report's verdict ``accepted`` goes on; any other is a reject with its why.
  4. The period's execution phase is the scan's (the one entry of ``answers.json``'s ``windows``): every audited
     request's release and streamed ids lie in it, else reject.
  5. The period check: `scan_period.rows_case` of the warden run under the auditor's record and the chain's scan,
     checked by ``network-period``
     (``--lean``, else built in verity/core/protocols/network/lean); its answer is the piece's, an accept's profile
     with its ``placement`` (`placement`): the certifier on the gate's host, and the chain, the period check and
     their proofs' coins in this piece on the prover's node, so `trusted-host` is assumed with the auditor's devices
     not placed.
  DIR also gets the chain's public part (`PUBLIC`: its report's verdicts and ε at δ, and the scan's answers; never the
  report's counts or a step's summary, which are private) and the check's three input files in
  DIR/period. All of them and the answer are written to a fresh hold under SERVE_DIR/`PRIVATE`/held (0700), where the
  kind scan (`kind_scan.py`) must find every field of them one of the four public kinds or the network certifier's
  certificate, else the piece refuses (its findings in the hold's `KIND_SCAN`), and reach
  DIR only through the record's leak guard (`leak_guard.py`), which reads DIR and the hold against the warden run's
  witness, DIR declared to hold those files, the split's `CHAINED` and the logs alone (`PIECE_FILES`), moves what
  holds a piece of it or lies at another path to SERVE_DIR/`PRIVATE`/quarantine and only then releases the rest. The
  period files it releases become DIR/`CASE` (`write_case`), the case as `composite.py` reads a piece's:
  ``{"commitment", "evidence", "params"}``, the commitment and evidence hex and the params ``params.json``'s canonical
  JSON, all inline, since certifier/v0 reads no blob (about 83 MB on a served window, nearly all of it the params).
  DIR/period then goes, so the case has one copy; the guard reads the files before they are hex, which it searches
  only for a binary witness file's pieces. Only the protocol rejects: a step that can't run (no kept run, no auditor's
  record or a fetch that isn't the record's, a rehearsal
  the gate's check didn't pass, a build, a tool or the chain failing without a report, an input missing, a field the
  kind scan finds, the leak guard failing, or SIGTERM from the piece's time limit) is `NotRun`, and the piece then
  refuses, releasing nothing
  from the hold: DIR keeps only its logs (`LOGS`) and gets DIR/`REFUSED` naming why, with no answer, which
  `served_verdicts.py` reads as the certifier missing. Each step it runs (the builds, the checks, the chain and the leak
  guard) is a process group of its own, which the piece ends however the step ends (`run_group`: SIGTERM, `GRACE_S`,
  SIGKILL), so nothing a step started outlives it; a SIGTERM while a step runs is held until the step's group has
  ended, so a step never outlives the refusal either. ``--draw K`` is the scan's sample, K proof blocks (every proof
  block by default); a CR_K=V goes to the chain; ``--chain-cmd`` replaces the chain (a test's, which writes the report
  and the scan's answers under RESEARCH_RUN_DIR/out).
* **chain** and **period** are the piece split in two around V3c (`served_verdicts.sh`'s ``CERTIFIER_CHAIN_CMD`` and
  ``CERTIFIER_PERIOD_CMD``, each run as ``bash -c CMD _ D K DIR V3C``, e.g. ``CERTIFIER_CHAIN_CMD='python
  benchmarks/network_traces/certifier_piece.py chain "$@" --certifier-key "${CERTIFIER_KEY:-}"'``, the follow-on's
  CERTIFIER_KEY naming the auditor's record, and the period's alike), so the period's compute audit is V3c's audit of
  the served window itself, on the run's registration record, which remote with the network certifier's egress tie
  requires of every compute audit (`Core.RemoteNetwork.egress`); the whole piece's are its scan's own registrations, so
  ``served_verdicts.sh`` takes only the split at ``PRODUCT=remote-network``. **chain** runs steps 1 to 4 once V3c's window is committed, its scan on V3c's trees
  (``scan_zk.py --class CLASS --v3c V3C/window`` in CR_SCAN_ARGS: the window's committed scan state and slot entries):
  a reject is DIR/answer.json as the piece's, an accept DIR/`CHAINED`. **period** runs once V3c's audit wrote
  V3C/period.json: steps 1, 2 and 4 again on the chain's kept report (no chain runs), then step 5 on the rows case whose
  one compute audit is V3C/period.json (`scan_period.rows_case`), held to the class circuit of the verifier's own
  window-circuit record beside V3C (`class_circuit`); its answer is DIR/answer.json. It answers nothing
  where the chain step refused (DIR/`REFUSED`) or rejected (its answer stands), and refuses where it left no
  DIR/`CHAINED`, the chain's scan read no V3c window or another, or V3C has no window-circuit record beside it. Each
  releases through its own hold and refuses as
  the piece does, keeping ``served_verdicts.sh``'s logs (`LOGS`).
* **relay**: a TCP relay from 127.0.0.1:``--listen`` to 127.0.0.1:``--to``, the local gate's tunnel.
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
from collections import Counter
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
sys.path[:0] = [str(HERE)]
import served_session as SS  # noqa: E402

#: where the front keeps the gate's fetch and the piece its private files, under the serve dir
PRIVATE = Path("private") / "certifier"
PROTOCOL = "verity/certifier/v0"
#: the protocol ``network-rows`` answers, step 1's rows audit of the network certifier's records
ROWS = "verity/network-rows/v1"
#: the named assumption the certifier's guarantee reads (`NetworkCheck.Period`): the certifier is the auditor's device
PREMISE = "trusted-host"
#: the follow-on's variable naming the auditor's record of the kept run's deployment (`warden_chain.RECORD`), which its
#: hooks pass to piece, chain and period as ``--certifier-key``: they read no environment for it
CERTIFIER_KEY = "CERTIFIER_KEY"
#: the front's public outcome, copied beside the kept fetch
SESSION = "session.json"
#: What DIR/chain gets of the chain's out/, by file: its JSON with only the key paths named (``a.b`` b under a, ``*``
#: every key at its level). Each is public as the protocol's output, its draw or its structure: the chain's and each
#: step's verdict, ε at δ (the draw's law, population and size) and where it runs in the clear, what the path stubs
#: and how its negatives came out; and the scan's answers, each link-window's answer and the registrations it is
#: checked against (hiding commitments), the period check's evidence. Everything else stays in the kept run: report.md
#: and the report's other keys (the certified I/O's counts), and the steps' summaries (the scan's admissions and
#: response lengths, the warden's check of its witness).
PUBLIC = {"report/report.json": ("schema", "verdict", "path", "scope", "stubbed", "negatives_refused",
                                 "steps.*.verdict", "steps.*.eps_at_delta", "steps.*.in_the_clear"),
          "scan/answers.json": ("format", "delta", "registration", "windows")}
#: a piece's refusal to run, which `served_verdicts.py` reads as the subprotocol missing
REFUSED = "refused.txt"
#: the piece's case, as `composite.py` reads it
CASE = "case.json"
#: the scan class of V3c's window class behind the front (`served_verdicts.sh`'s V3C_CLASS, 8,32,4,4,16, whose window
#: is the scan's link-window)
V3C_SCAN_CLASS = "w63k4"
#: what `bind`'s rejects say: a served request the network certifier didn't carry, a certified one the endpoint didn't
#: serve, and one certified request served twice
NOT_SEEN = "the Commit period served traffic the network certifier didn't see"
UNSERVED = "the network certifier carried traffic the endpoint didn't serve"
SHARED = "one certified request served twice"
#: the split piece's chain step's accept, which its period step reads
CHAINED = "chain.json"
#: the kind scan's findings, which a refusing piece leaves in its hold
KIND_SCAN = "kind-scan.txt"
#: the logs `served_verdicts.sh` and `served_verdicts.py` write in DIR, which a refusal keeps
LOGS = ("piece.log", "chain.log", "period.log")
#: every file DIR may hold when the leak guard reads it and the hold (its ``--public``): the answer, the logs, the
#: split's `CHAINED`, the period check's three input files and `PUBLIC`'s
PIECE_FILES = ("answer.json", *LOGS, CHAINED, "period/commitment.bin", "period/evidence.json", "period/params.json",
               *(f"chain/{rel}" for rel in PUBLIC))
LEAN_DIR = TREE / "verity" / "core" / "protocols" / "network" / "lean"
#: the front's group: the scan's 4 lanes, each a prompt of 24 ids (V3c's window class takes at most 32) and 4 outputs
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
    attack = None
    try:
        if a.prompts > SS.A.SCAN["--lanes"]:
            raise SS.Refused(f"--prompts {a.prompts}: one group, at most the scan's {SS.A.SCAN['--lanes']} lanes")
        if SS.A.scan_group_windows(a.tokens) != 1:
            raise SS.Refused(f"--tokens {a.tokens}: a group past one link-window, which the period's rows form refuses")
        for flag, n in (("--warmup", a.warmup), ("--stop-prompt", a.stop_prompt)):
            if n < 0:
                raise SS.Refused(f"{flag} {n}: a count of requests, 0 for none")
        ps = SS.prompts(a.prompts, a.prompt_ids, a.vocab, secrets.token_bytes(32))
        data = json.dumps(ps, separators=(",", ":")).encode()
        SS.check(a, data)
        warm = {}
        if a.warmup:
            warm["requests"] = warmup(a, serve_dir)
            (out / "warmup.json").write_text(json.dumps(warm, indent=1) + "\n")
        if a.stop_prompt:
            ps[-1], warm["stop_prompt"] = stop_prompt(a, serve_dir)
            (out / "warmup.json").write_text(json.dumps(warm, indent=1) + "\n")
            data = json.dumps(ps, separators=(",", ":")).encode()
            SS.check(a, data)
        if a.attack:
            attack, a.port = attack_relay(a, keep)
        rc = SS.session(a, data, ps)
        if (doc := out / "session.json").is_file():
            shutil.copyfile(doc, keep / json.loads(doc.read_text())["run"] / SESSION)
        return rc
    except SS.Refused as e:
        (out / "session.json").write_text(json.dumps({"refused": str(e)}, indent=1) + "\n")
        print(f"refused: {e}", file=sys.stderr)
        return e.code
    finally:
        if attack is not None:
            end_group(attack)


def gate_params(scan_class: str = V3C_SCAN_CLASS, bucket_ms: int | None = None):
    """The schedule the gate's network certifier runs a front's scan rehearsal at, at the scan class ``scan_class`` and
    the tick ``bucket_ms`` (the scan's own when None): `auditor.warden_command`'s flags as warden_serve.py parses them,
    its `WardenParams`."""
    import warden_serve as WSV
    argv = SS.A.warden_command({"requests": 1, "scan": True, "scan_class": scan_class, "bucket_ms": bucket_ms},
                               Path("out"), "http://127.0.0.1:1")
    return WSV._params(WSV.parser().parse_args(argv[2:]))


def attack_relay(a: argparse.Namespace, keep: Path) -> tuple[subprocess.Popen, int]:
    """``--attack``'s relay (`attacks.py relay`, in a process group of its own) from a free loopback port to PORT, at
    the scan's payload and delay and the front's tick, its log at KEEP/attacks/NAME.jsonl (the prover's, 0700): the
    process and its port, once it listens."""
    (keep / "attacks").mkdir(mode=0o700, exist_ok=True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    S = SS.A.SCAN
    p = subprocess.Popen([sys.executable, str(HERE / "attacks.py"), "relay", "--attack", a.attack, "--listen", str(port),
                          "--to", str(a.port), "--cell-bytes", str(S["--cell-bytes"]),
                          "--bucket-ms", str(a.bucket_ms or S["--bucket-ms"]), "--delay", str(S["--delay"]),
                          "--nth", str(a.attack_nth), "--log", str(keep / "attacks" / f"{a.attack}.jsonl")],
                         stdout=subprocess.PIPE, text=True, process_group=0)
    line = p.stdout.readline()
    if not line.startswith("relay listening"):
        end_group(p)
        raise SS.Refused(f"--attack {a.attack}: the relay didn't listen: {line.strip()!r}")
    threading.Thread(target=lambda: [print(x, end="", flush=True) for x in p.stdout], daemon=True).start()
    print(f"attack {a.attack}: the relay on {port} to {a.port}", flush=True)
    return p, port


def stop_prompt(a: argparse.Namespace, serve_dir: Path, **kw) -> tuple[list[int], list[dict]]:
    """``--stop-prompt N``'s prompt (the module docstring), the first of N fresh prompts of HI ids whose warm-up answer
    finishes on a stop, and the warm-ups it took; `served_session.Refused` when none of N does."""
    hi = SS.bounds(a.prompt_ids)[1]
    ps = SS.prompts(a.stop_prompt, hi, a.vocab, secrets.token_bytes(32))
    tried = warmup(a, serve_dir, ps, **kw)
    if tried[-1]["finish_reason"] != "stop":
        raise SS.Refused(f"--stop-prompt {a.stop_prompt}: no answer to {a.stop_prompt} prompts of {hi} ids finished "
                         f"on a stop within --tokens {a.tokens}")
    return ps[len(tried) - 1], tried


def warmup(a: argparse.Namespace, serve_dir: Path, ps: list[list[int]] | None = None, *,
           clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], object] = time.sleep) -> list[dict]:
    """``--warmup``'s requests (the module docstring), each its user, finish and seconds; or `stop_prompt`'s, `ps`
    until one's answer finishes on a stop. `served_session.Refused` when one fails or serve-openai hasn't recorded
    every one's execution phase closed within `WARMUP_CLOSE_S`."""
    flag = "--warmup" if ps is None else "--stop-prompt"
    ps = SS.prompts(a.warmup, a.prompt_ids, a.vocab, secrets.token_bytes(32)) if ps is None else ps
    tag, done = secrets.token_hex(4), []
    for i, p in enumerate(ps):
        body = {"user": f"warmup-{tag}-{i}", "prompt": p, "max_tokens": a.tokens, "temperature": 0.0, "top_p": 1.0}
        req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/completions", json.dumps(body).encode(),
                                     {"content-type": "application/json"})
        t = clock()
        try:
            with urllib.request.urlopen(req, timeout=WARMUP_REQUEST_S) as r:  # noqa: S310
                got = json.loads(r.read())
        except (OSError, ValueError) as e:
            raise SS.Refused(f"{flag}: request {i}: {type(e).__name__}: {e}") from None
        done.append({"user": body["user"], "finish_reason": ((got.get("choices") or [{}])[0]).get("finish_reason"),
                     "seconds": round(clock() - t, 3)})
        print(f"{flag} {i + 1}/{len(ps)}: {done[-1]}", flush=True)
        if flag == "--stop-prompt" and done[-1]["finish_reason"] == "stop":
            break
    users, until = {d["user"] for d in done}, clock() + WARMUP_CLOSE_S
    while left := users - {r.get("user") for r in _lines(serve_dir / SS.USERS)}:
        if clock() > until:
            raise SS.Refused(f"{flag}: serve-openai recorded no closed execution phase of {sorted(left)} in "
                             f"{serve_dir / SS.USERS} within {WARMUP_CLOSE_S} s")
        sleep(1)
    return done


def relay(listen: int, to: int, *, host: str = "127.0.0.1", ready: Callable[[], None] = lambda: None) -> None:
    """A TCP relay from 127.0.0.1:``listen`` to ``host``:``to``, one upstream connection a connection."""
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
            up = socket.create_connection((host, to))
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
    """The kept fetch unpacked in a 0700 scratch directory beside it, and its network certifier run."""
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


def auditor_record(path: str | None) -> dict:
    """The auditor's record of the kept run's deployment at ``path`` (`warden_chain.RECORD`); `NotRun` without one,
    whose key and session would be the kept fetch's."""
    if not path:
        raise NotRun(f"no auditor's record of the network certifier's deployment (--certifier-key, {CERTIFIER_KEY}): "
                     f"its key and session would be the kept fetch's, a file on the prover's node, which {PREMISE} "
                     "doesn't cover")
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise NotRun(f"the auditor's record {path}: {type(e).__name__}: {e}") from None


def held(run: Path, record: dict) -> None:
    """`NotRun` unless the kept run is ``record``'s, the auditor's record of its deployment
    (`warden_chain.deployment_of`): naming the copy of ``deployment.json`` that isn't, and the field it differs in."""
    import warden_chain as WC
    try:
        WC.deployment_of(run, record)
    except (SystemExit, OSError, KeyError, TypeError, ValueError) as e:
        raise NotRun(f"the kept fetch isn't the auditor's record of the network certifier's deployment: {e}") from None


def device_problems(run: Path) -> list[str]:
    """The network certifier's own check of the kept run (`warden_serve.check` on its witness), each problem once with
    its numbers (connections, cells, phases) as N: what refused the rehearsal, none of its counts."""
    import warden_chain as WC
    try:
        got = WC.WS.check(WC.witness(run))
    except (OSError, ValueError, KeyError, SystemExit) as e:
        return [f"the check didn't run: {type(e).__name__}"]
    return list(dict.fromkeys(re.sub(r"\d+", "N", p) for p in got["problems"]))


def rehearsed(run: Path) -> None:
    """`NotRun` unless the gate's check passed the kept rehearsal (its ``end.json``): a failed one, late or otherwise,
    is an infrastructure failure, like a busy gate, and names the network certifier's own check's problems
    (`device_problems`): a connection the network certifier failed closed (late, past the egress cells, upstream) is
    inside k_fail and certifies no served answer. But one whose certified egress holds a cell that is neither an
    id-form cell nor a `FAIL`, bytes the server sent that are no output's, is Rejected."""
    import warden_chain as WC
    try:
        end = json.loads((run / "end.json").read_text())
    except (OSError, ValueError) as e:
        raise NotRun(f"the kept run's end.json: {type(e).__name__}: {e}") from None
    v = end.get("verdict") or {}
    if end.get("rc") != 0 or v.get("accepted") is not True:
        problems = device_problems(run)
        try:
            violations, unread = WC.io(run)["violations"], ""
        except (OSError, ValueError, KeyError, SystemExit) as e:
            violations, unread = [], f"; its certified I/O unread ({type(e).__name__}: {e})"
        if any(why == WC.NOT_ID_CELL for _, why in violations):
            raise Rejected(f"the network certifier's certified egress holds {WC.NOT_ID_CELL.removeprefix('an ')}: "
                           f"bytes the server sent that are no output's; the network certifier's own check: "
                           f"{problems[:4]}")
        raise NotRun(f"the gate's rehearsal didn't pass, so it certified no served answer: exit {end.get('rc')}, "
                     f"{end.get('ended')}; its check accepted {v.get('accepted')} with {v.get('problems')} problems, "
                     f"failed {v.get('failed')}; the network certifier's own check: {problems[:4]}{unread}")


def release_run(r: dict) -> str | None:
    """The network certifier's run a served request's release names (requests.jsonl's ``release``, [run, N] under
    ``--admit release-order``, run "" without the run header), None without one."""
    rel = r.get("release")
    return (rel[0] or None) if isinstance(rel, list) and len(rel) == 2 else None


def bind(reqs: list[dict], io: dict, run_id: str | None = None, served: list[dict] | None = None) -> dict[str, dict]:
    """Each audited request's certified record by its user; Rejected when one isn't the network certifier's: its user
    names no certified request, it carries no release of the network certifier's run ``run_id`` (when given), or its
    prompt or streamed ids aren't the certified ones. Of ``served`` (when given: every request the endpoint recorded,
    of which those carrying the run's release and ``reqs`` are read), Rejected when two name one certified request or
    a certified request is none of them."""
    if not io["accepted"]:
        raise Rejected(f"the warden's certified I/O doesn't read whole off its cells: {io['violations'][:3]}")
    by, got = {r["user"]: r for r in io["records"]}, {}
    for r in reqs:
        eid, user = r.get("engine_id"), r.get("user")
        c = by.get(user)
        if c is None:
            raise Rejected(f"audited request {eid}'s user {user!r} is no certified request: {NOT_SEEN}")
        if run_id is not None and release_run(r) != run_id:
            raise Rejected(f"audited request {eid} (user {user!r}) carries no release of the network certifier's run "
                           f"({r.get('release')!r}): {NOT_SEEN}")
        for key, theirs in (("prompt_ids", "prompt"), ("streamed_ids", "output")):
            if key in r and r[key] != c[theirs]:
                raise Rejected(f"audited request {eid}'s {key} aren't the certified {theirs}")
        got[user] = c
    if served is not None:
        mine = {json.dumps(r, sort_keys=True): r
                for r in [*reqs, *(r for r in served if run_id is not None and release_run(r) == run_id)]}
        named = Counter(r.get("user") for r in mine.values() if r.get("user") in by)
        if twice := sorted(u for u, n in named.items() if n > 1):
            raise Rejected(f"certified request {twice[0]!r} is {named[twice[0]]} served requests: {SHARED}")
        if unserved := sorted(set(by) - set(named)):
            raise Rejected(f"certified request {unserved[0]!r} is no served request: {UNSERVED}")
    return got


def in_link_window(certified: dict[str, dict], T: int, w: int) -> None:
    for user, c in certified.items():
        buckets = [c["release"], *(b for _, b in c["frames"])]
        if None in buckets or {b // T for b in buckets} != {w}:
            raise Rejected(f"certified request {user!r} spans link-windows {sorted({b // T for b in buckets if b is not None})}"
                           f", not the period's {w} alone")


def sized(windows: int, sets: list[str]) -> list[str]:
    """N2's sample for a run of ``windows`` execution phases: every one when they are at most `served_session.chain`'s
    8 (whose sample of 8 a shorter run refuses), unless a CR_N2_DRAW=V of ``sets`` names it."""
    given = {kv.partition("=")[0] for kv in sets}
    if windows > 8:
        return sets
    return [*(["CR_N2_DRAW="] if "CR_N2_DRAW" not in given else []), *sets]


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


def run_chain(keep: Path, run: str, draw: int | None, sets: list[str], cmd: str | None = None) -> int:
    """`served_session.py chain` on the kept run (or ``cmd``, run as ``bash -c CMD _ KEEP RUN CR_K=V ...``),
    its run dir (RESEARCH_RUN_DIR, whose out/ holds the report) and quarantine private under KEEP/RUN."""
    d = keep / run
    env = dict(os.environ, RESEARCH_RUN_DIR=str(d / "chain"), CR_QUARANTINE=str(d / "quarantine"))
    (d / "chain").mkdir(mode=0o700, exist_ok=True)
    argv = ["bash", "-c", cmd, "_", str(keep), run, *sets] if cmd else [
        sys.executable, str(HERE / "served_session.py"), "chain", "--run", run, "--keep", str(keep),
        *(["--draw", str(draw)] if draw else []), *sets]
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
    ``record``, the auditor's record of its deployment: `Rejected` on its reject, `NotRun` where the case can't be read
    or the check gives no `ROWS` accept or reject."""
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


def case_of(warden: Path, scan: Path, v3c: Path | None = None, circuit: str | None = None, *, record: dict) -> dict:
    import scan_period as SP
    try:
        return SP.rows_case(warden, scan, v3c=v3c, circuit=circuit, record=record)
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


def v3c_case(v3c: Path, record: dict) -> Callable[[Path, Path], dict]:
    """The period step's case: `case_of` under ``record`` with V3C/period.json as its one compute audit, held to the
    verifier's class circuit (`class_circuit`), once the chain's scan read V3C/window's trees (its ``summary.json``'s
    ``v3c``); `NotRun` for a scan of no V3c window or another, or no window-circuit record."""
    def case(warden: Path, scan: Path) -> dict:
        try:
            read = (json.loads((scan / "summary.json").read_text()).get("v3c") or {}).get("window")
        except (OSError, ValueError) as e:
            raise NotRun(f"the chain's scan summary: {type(e).__name__}: {e}") from None
        if read != str(v3c / "window"):
            raise NotRun(f"the chain's scan read {read or 'no'} V3c window, not {v3c / 'window'}: run the chain step "
                         "on this V3c")
        return case_of(warden, scan, v3c / "period.json", class_circuit(v3c), record=record)
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


def decide(serve_dir: Path, window: int, d: Path, *, record: dict, chain: Callable[[Path, str, int], int] | None,
           check: Callable[[dict, Path], dict] | None, rows: Callable[[Path], None],
           case: Callable[[Path, Path], dict] | None = None) -> tuple[dict | None, Path | None]:
    """Window ``window``'s answer (module docstring) and the unpacked fetch (for the leak guard), its files in ``d``;
    ``record`` is the auditor's record of the kept run's deployment, which the fetch must be (`held`) and whose key,
    session and schedule every check reads; ``rows`` is step 1's rows audit of the network certifier run
    (`rows_check`); ``chain`` is given KEEP, the run and the run's link-windows, or is None to read the report of the
    chain the split's chain step ran; ``check`` None stops once the chain accepted, with no answer (the split's chain
    step); ``case`` is the period's case (`case_of` under ``record`` by default). `NotRun` when a step can't run."""
    import warden_chain as WC
    keep = serve_dir / PRIVATE
    run = kept_run(serve_dir)
    reqs = window_requests(serve_dir, window)
    case = case or (lambda warden, scan: case_of(warden, scan, record=record))
    unpacked = None
    try:
        unpacked, warden = unpack(keep / run / "fetch.tar.gz")
        held(warden, record)
        rehearsed(warden)
        rows(warden)
        try:
            io = WC.io(warden)
            dep, params = WC.deployment_of(warden, record)
        except SystemExit as e:
            raise Rejected(f"the kept warden run: {e}") from None
        T = params.window
        certified = bind(reqs, io, dep["run_id"], _lines(serve_dir / "requests.jsonl"))
        out = keep / run / "chain" / "out"
        report = out / "report" / "report.json"
        if chain is None:
            if not report.is_file():
                raise NotRun(f"no chain report under {keep / run}: the chain step didn't run here")
        else:
            rc = chain(keep, run, int(dep["windows"]))
            publish(out, d)
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


def kind_scan(hold: Path) -> None:
    """`kind_scan.scan` of the hold: `NotRun` on a field of no public kind, its findings (file and key path) in the
    hold's `KIND_SCAN`, never in the refusal, which DIR keeps."""
    import kind_scan as KS
    found, kinds = KS.scan(hold)
    print(f"kind scan: {len(found)} findings, {json.dumps({k: kinds[k] for k in KS.KINDS})}", flush=True)
    if found:
        (hold / KIND_SCAN).write_text("".join(f"{f}\n" for f in found))
        raise NotRun(f"the kind scan found fields of no public kind ({len(found)}) in "
                     f"{sorted({f.file for f in found})}: DIR withheld, the piece's files and the findings "
                     f"({KIND_SCAN}) left in {hold}")


def settle(serve_dir: Path, d: Path, limit: str,
           step: Callable[[Path], tuple[dict | None, Path | None]]) -> dict | None:
    """``step``'s answer: its files written to a fresh hold under SERVE_DIR/`PRIVATE`/held (``step`` is given it), an
    accept's profile with its `placement`, each field of every file of one public kind or a certificate (`kind_scan`),
    all released to DIR by the leak guard and the case written; None where the split's chain step accepted. `NotRun`
    (SIGTERM included, naming ``limit``) for a step that can't run or a field the kind scan finds."""
    was, unpacked, hold = signal.signal(signal.SIGTERM, ended(limit)), None, None
    try:
        if not any((serve_dir / PRIVATE).glob("*/fetch.tar.gz")):
            raise NotRun(f"no warden run kept under {serve_dir / PRIVATE}: the run's load didn't go through the "
                         "certifier's front (CERTIFIER_FRONT unset, or the gate, its tunnel or ssh failed)")
        (serve_dir / PRIVATE / "held").mkdir(mode=0o700, exist_ok=True)
        hold = Path(tempfile.mkdtemp(dir=serve_dir / PRIVATE / "held", prefix="piece-"))
        got, unpacked = step(hold)
        if got is not None:
            if isinstance(got.get("profile"), dict):
                got = got | {"profile": got["profile"] | {"placement": placement(serve_dir, kept_run(serve_dir))}}
            (hold / "answer.json").write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
        kind_scan(hold)
        g = guard(d, hold, unpacked, serve_dir / PRIVATE / "quarantine")
        if g not in (0, 3) or got is not None and not (d / "answer.json").is_file():
            raise NotRun(f"the leak guard failed (exit {g}): DIR withheld, the piece's files left in {hold}")
        write_case(d)
        return got
    except (OSError, ValueError, tarfile.TarError) as e:
        traceback.print_exc()
        raise NotRun(f"the piece failed on a file: {type(e).__name__}: {e}") from None
    finally:
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
        record = auditor_record(a.certifier_key)
        exe, rows = lean_exe("network-period", a.lean), lean_exe("network-rows", a.rows)
        chain = lambda keep, run, n: run_chain(keep, run, a.draw, sized(n, a.set), a.chain_cmd)  # noqa: E731
        return decide(serve_dir, a.window, hold, record=record, chain=chain,
                      check=lambda c, p: period_check(exe, c, p), rows=lambda run: rows_check(rows, run, record))
    try:
        got = settle(serve_dir, d, "CERTIFIER_TIMEOUT", step)
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
    for f in ("answer.json", CHAINED, REFUSED, CASE):
        (d / f).unlink(missing_ok=True)
    if not sets_ok(a.set):
        return 2
    try:
        sets = v3c_scan(v3c, a.scan_class, a.set)
        chain = lambda keep, run, n: run_chain(keep, run, a.draw, sized(n, sets), a.chain_cmd)  # noqa: E731

        def step(hold: Path) -> tuple[dict | None, Path | None]:
            record = auditor_record(a.certifier_key)
            rows = lean_exe("network-rows", a.rows)
            return decide(serve_dir, a.window, hold, record=record, chain=chain, check=None,
                          rows=lambda run: rows_check(rows, run, record))
        got = settle(serve_dir, d, "CERTIFIER_TIMEOUT", step)
    except NotRun as e:
        return refuse(d, str(e))
    if got is not None:
        return said(got)
    (d / CHAINED).write_text(json.dumps({"chain": "accepted"}, indent=1, sort_keys=True) + "\n")
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
        record = auditor_record(a.certifier_key)
        exe, rows = lean_exe("network-period", a.lean), lean_exe("network-rows", a.rows)
        return decide(serve_dir, a.window, hold, record=record, chain=None,
                      check=lambda c, p: period_check(exe, c, p), rows=lambda run: rows_check(rows, run, record),
                      case=v3c_case(v3c, record))
    try:
        got = settle(serve_dir, d, "CERTIFIER_PERIOD_TIMEOUT", step)
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
    f.add_argument("--prompt-ids", type=SS.span, default=PROMPT_IDS, help="N, or LO:HI each prompt's length drawn in")
    f.add_argument("--stop-prompt", type=int, default=0, metavar="N",
                   help="the group's last prompt the first of N, tried as warm-ups, whose answer ends on a stop")
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
    import attacks as AT
    f.add_argument("--attack", choices=AT.RELAYED,
                   help="the prover's relay playing this attack between the gate's tunnel and PORT (attacks.py)")
    f.add_argument("--attack-nth", type=int, default=1, metavar="K",
                   help="the completion request, from 1, an attack on one connection plays on")
    for name in ("piece", "chain", "period"):
        q = sub.add_parser(name)
        q.add_argument("serve_dir", type=Path)
        q.add_argument("window", type=int)
        q.add_argument("dir", type=Path)
        if name != "piece":
            q.add_argument("v3c", type=Path, help="V3c's directory: window/ (window-commit's) and period.json")
        q.add_argument("--certifier-key", metavar="RECORD",
                       help="the auditor's record of the kept run's deployment, the gate's own (certifier-key/v1; the "
                            f"hooks pass ${CERTIFIER_KEY}): without it the piece refuses")
        q.add_argument("--rows", help="network-rows, built (else built here)")
    q = sub.choices["piece"]
    q.add_argument("--lean", help="network-period, built (else built here)")
    q.add_argument("--draw", type=int, help="the scan's sample, K proof blocks (every proof block by default)")
    q.add_argument("--chain-cmd", help="replaces served_session.py chain: bash -c CMD _ KEEP RUN CR_K=V ...")
    q.add_argument("set", nargs="*", help="CR_K=V")
    q = sub.choices["chain"]
    q.add_argument("--draw", type=int, help="the scan's sample, K proof blocks (every proof block by default)")
    q.add_argument("--class", dest="scan_class", default=V3C_SCAN_CLASS,
                   help=f"the scan's class, V3c's window class's (default {V3C_SCAN_CLASS})")
    q.add_argument("--chain-cmd", help="replaces served_session.py chain: bash -c CMD _ KEEP RUN CR_K=V ...")
    q.add_argument("set", nargs="*", help="CR_K=V")
    sub.choices["period"].add_argument("--lean", help="network-period, built (else built here)")
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
    if a.cmd == "chain":
        return chain_step(a)
    if a.cmd == "period":
        return period_step(a)
    relay(a.listen, a.to)
    return 0


if __name__ == "__main__":
    sys.exit(main())
