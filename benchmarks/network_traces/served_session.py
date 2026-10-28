"""served-session: the session of record, the network certifier in front of a served model, as jobs on the nodes
(nothing reaches a node otherwise). The gate's scan rehearsal (`vy-challenger-gate warden-rehearse --scan --prompts`,
`benchmarks/live_challenger/auditor.py`) runs against the endpoint on the peer's loopback, by default `PEER`'s
`verity-vllm serve-openai` on 127.0.0.1:18782, inside the scan's class (`scan_zk.W63`: at most KI = 1,024 prompt ids and
KO = 64 ids out a request). The session's prompts are of `PROMPT_IDS` = 256 ids, since prompts of 1,024 reach the warden
late against its D = 2.7 s (about 1.2 s of ingress and 1.2 to 4 s of committed prefill), and each is answered
greedily with at most 64 ids: the base Llama-3.2-1B ends at its one EOS id, 128001, after 1 to 5 ids in 15 to 27% of
requests. The network chain's run of record (`chain_record.sh`'s warden path) then proves the session.

    python served_session.py plan [FLAGS]
    python served_session.py session --out DIR [FLAGS] [--gate USER@HOST | --gate local] [--key KEY] [--keep DIR]
        [--at UTC]
    python served_session.py io --run RUN [--keep DIR]
    python served_session.py endpoint --from SERVE_DIR --io FILE|- --out DIR [--eos-ids 128001] [--port 18782]
        [--keep DIR]
    python served_session.py chain --run RUN [--keep DIR] [--endpoint DIR|ART] [--eos-ids IDS] [--draw K]
        [--coins verifier | --coins gate --challenger-key RECORD [--coin-peer NAME] [--main FILE]] [CR_K=V ...]

FLAGS: [--peer PEER] [--port 18782] [--prompts 64] [--requests 113] [--height 12] [--minutes M]
       [--prompt-ids 256] [--tokens 64] [--vocab 128000]

``--requests`` is by default `REQUESTS`, what h200-2's free disk seals by the endpoint's sizing; ``--tokens`` is each
request's max_tokens, at most KO.

* **plan** prints the request's words, the minutes it takes (``--minutes`` by default `auditor.warden_minutes` of its
  requests and height plus `MARGIN`), the prompts' size, the endpoint's flags and each run's command. The request and
  its prompts are checked by this tree's gate first (`auditor.parse_warden` and `warden_prompts` on a scratch host), so
  a request the gate would refuse is refused here, with the gate's reason. A gate pinned at an older tree takes
  400 requests at most.
* **session** sends the request, waiting until ``--at`` (a UTC time) when given; launched more than `LATE_S` after it,
  it sends nothing (exit 3). The prompts are ``--prompts`` prompts of ``--prompt-ids`` ids below ``--vocab`` (Llama 3's
  128,000 regular ids by default), drawn from a fresh secret seed this process alone holds: they are the prover's,
  which the gate writes to its run's ``witness/`` alone, so they are written nowhere here.
  - ``--gate USER@HOST`` (the challenger's gate, `GATE`, under the lane's gate key ``--key``): the request with the
    prompts on its stdin (``start.json``), ``status`` every `POLL_S` seconds until the run ends (``status.json``), then
    ``fetch``.
  - ``--gate local``, the dry rehearsal: this tree's gate on a scratch host under KEEP, its rehearsal run in this
    process (`auditor.certifier_run`), where the tunnel is the stand-in endpoint itself (`demo_stub.py serve --out`,
    the endpoint's records to KEEP/RUN/endpoint) listening on the port the run picks. The stand-in serves byte ids:
    give ``--vocab`` at most 256.

  The lane's own fetch, with ``witness/``, goes to KEEP/RUN/fetch.tar.gz (KEEP and KEEP/RUN 0700, refused inside the
  run dir: custody publishes all of ``--out``). ``--out`` gets ``bundle.tgz``, what ``fetch`` gives any other key (the
  bundle less ``witness/``), and ``session.json``, the public outcome: the gate run, its rc, the check's verdict and
  counts, the link-windows, the origin and the session's span. Before it exits, every file of ``--out`` and member of
  ``bundle.tgz`` is read for a prompt's ids, and one that holds them is removed (exit 4). Exit 0 when the gate's check
  accepted the run, its rc (or 1) when it didn't, 2 when the gate refused the request.
* **io** writes the kept warden run's certified I/O (`warden_chain.io`: each request's prompt and streamed ids as the
  warden's cells carry them) and its deployment.json to KEEP/RUN/certified-io.json (0600), on `AUDITOR`, and prints only
  its path and counts.
* **endpoint** is the chain's endpoint step on the served node, where the endpoint's records stay: serve-openai's serve
  dir (``--from``, its ``--out``) holds requests.jsonl and advice_trace.jsonl, whose arrive, step and finish records
  carry each request's prompt length, sampling (its seed), rows and emitted ids. It copies the trace, windows.jsonl,
  serve.json and `USERS` to ``--keep`` (0700, by default SERVE_DIR.endpoint beside the serve dir) with requests.jsonl
  rebuilt there from `USERS` and the certified I/O (``--io``: ``io``'s file, or ``-`` for stdin; a file under the run
  dir, ``--out``'s directory as research run's ``$OUT`` is, is moved to ``--keep`` first, since custody publishes the
  run dir), runs `chain_record.served_endpoint` on that
  copy (`verity-vllm advice`, then the advice bound to the certified I/O and rebuilt from it), and writes to ``--out``
  its report.json and `RECORDS`. It writes nothing unless serve.json shows the endpoint served with `ENDPOINT`'s flags
  and no published file holds a request's prompt ids, a trace's arrive, step or finish record, or a request-valued
  field (`valued`).
* **chain** runs chain_record.sh's warden path on the kept fetch (CR_PATH=warden, CR_GATES=KEEP/RUN/fetch.tar.gz), with
  the endpoint's report (``--endpoint``, ``endpoint``'s art), or a dry rehearsal's endpoint records (KEEP/RUN/endpoint
  by default) and the model's EOS ids (``--eos-ids``, the stand-in's ``--vocab``), N2's period drawn
  on the verifier's coins (CR_N2_DRAW 8) and the tie on 8 link-windows (CR_TIE_WINDOWS)
  at CR_MEM_GB 96, unless a CR_K=V says otherwise. The scan proves every unit, or ``--draw K`` of them, on the
  verifier's own coins (``--coins verifier``, the default and the fallback) or on the gate's (``--coins gate``). It is
  refused unless the tree has the record's leak guard (``leak_guard.py``), which chain_record.sh runs on its
  way out, whatever the exit.

**The coin link.** A draw on the gate's coins (``--coins gate``: ``scan_zk.py prove --draw K --challenger gate:GATE
--main FILE --peer P --challenger-key RECORD --gate-key KEY``, in CR_SCAN_ARGS) runs on the auditor's side: P
(``--coin-peer``, `AUDITOR` by default) is the node the chain runs on, which holds the lane's gate key and the kept
fetch. The challenger reserves one of P's ports in its /etc/vy/coin-ports and its coin link carries the
session's coins over ssh to that port of P's loopback, where the draw's client connects. The gate key and the warden
run's witness are the auditor's, so the draw never runs on the served node, which is the prover's: a ``--coin-peer``
that is the kept run's peer (its request.json's) or `PEER` is refused. Only P's own line serves P, and a reserve for a
node with no line is refused: `AUDITOR`'s line is ports 7821-7830 (with its /etc/vy/peers entry), and on
a node without one the chain draws on the verifier's coins. ``--main`` is the verifier's copy of main's first-parent
history, fetched from the tree's origin into KEEP/RUN by default (`first_parent`); ``--challenger-key`` is the
deployment's challenger-key/v1 record (labelled ``challenger-key current``), sent with the run. CR_SCAN_ARGS and
CR_SCAN_CMD may not name the scan's coins (``--draw``, ``--challenger``, ``--peer``) themselves.

**The endpoint.** It is started by hand on `PEER` (`HAND_START`) ahead of the session's ``--at``, and it
listens within about 35 s. Its records stay in its serve dir on h200-2: requests.jsonl (prompt and streamed ids) and
advice_trace.jsonl (prompt lengths, sampling, rows, emitted ids) are request values, and this runbook neither fetches
nor publishes them. The endpoint step's advice (`verity-vllm advice`) needs both, so ``endpoint`` runs it there, on a
copy whose requests.jsonl is rebuilt from `USERS` (each engine request's id and its warden user, ``{"engine_id",
"user"}`` a line, in requests.jsonl's order) and the certified I/O, each request's prompt and streamed ids as the
warden's cells carry them and its finish (`finish`): the endpoint's trace is checked against the certified I/O, not the
prover's record of it. The certified I/O is ``io``'s file, written in KEEP on `AUDITOR`: it reaches the served node
over a private route (``endpoint --io -`` on an ssh stdin, or a ``--send`` the step moves out of the run dir), never
through a run's custody, and the served node already holds its values. The check then runs on the prover's node, and
the chain's report says so (`chain_record.WARDEN_SERVED_STUBBED`). Its art (``--endpoint ART``, ``endpoint``'s
``$OUT``) is report.json, windows.jsonl and serve.json.

The runs (the lane's gate key is on `AUDITOR`, a CPU node; KEEP is the lane's gate directory there):

    research run --on AUDITOR --source . --project verity --as network-accounting --cwd source --timeout 90m \\
      -- bash -lc 'python3 benchmarks/network_traces/served_session.py session --out "$OUT" --at UTC'
    research run --on AUDITOR --source . --project verity --as network-accounting --cwd source --timeout 30m \\
      -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && uv run --no-sync python \\
      benchmarks/network_traces/served_session.py io --run RUN'
    research run --on PEER --source . --project verity --as network-accounting --cwd source --timeout 30m \\
      --send certified-io.json -- bash -lc 'python3 benchmarks/network_traces/served_session.py endpoint \\
      --from SERVE_DIR --io "$RESEARCH_RUN_DIR/inputs/certified-io.json" --out "$OUT"'
    research run --on AUDITOR --source . --project verity --as network-accounting --cwd clone --timeout 270m \\
      --store-read --send challenger-key.json -- env CHECK_SLOTS=/workspace/research/locks/network/slots python3 \\
      tools/verity/check/slot.py --by network-accounting --wait-max 1200 --run-max 14400 -- nice -n 19 bash -c \\
      'python3 benchmarks/network_traces/served_session.py chain --run RUN --endpoint ART \\
      --draw 8 --coins gate --challenger-key "$RESEARCH_RUN_DIR/inputs/challenger-key.json"'

(``--store-read`` gives the job the read-only store key its fetch of the endpoint's art needs; ``--run-max`` leaves room
over a dry chain of 16 requests on the gate's coins, which held its slot about 9,500 s, an hour of it the challenger's
session; ``--coins verifier``, without ``--send`` and ``--challenger-key``, is the fallback);

and the dry rehearsal, on `AUDITOR`, whose KEEP its dry chain reads:

    research run --on AUDITOR --pool network --source . --project verity --as LANE --cwd source \\
      --timeout 45m -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && uv run --no-sync python \\
      benchmarks/network_traces/served_session.py session --gate local --vocab 256 --prompts 16 --requests 16 \\
      --height 10 --keep DIR --out "$OUT"'
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, UTC
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "live_challenger"))
import auditor as A  # noqa: E402

#: the scan's class (`scan_zk.W63`): at most KI prompt ids in, at most KO ids out a request
KI, KO = 1024, 64
#: the session's prompt ids a request, which reach the warden in time (KI's don't)
PROMPT_IDS = 256
#: the session's requests: what `PEER`'s free disk seals, within 72 GB even if every request runs all KO ids
#: (the endpoint's sizing: 636.5 MB a full-length request)
REQUESTS = 113
#: where the sessions run, kept as data (``site.json``): the served peer, the gate (USER@HOST) and the lane whose key it
#: takes, the auditor's node, the gate key's file under ~/.ssh and the lane's gate directory there
SITE: dict[str, str] = json.loads((HERE / "site.json").read_text())
PEER, PORT, GATE, LANE = SITE["peer"], 18782, SITE["gate"], SITE["lane"]
#: the auditor's side, where the chain runs and the gate's coins for its draw arrive
AUDITOR = SITE["auditor"]
KEY = Path.home() / ".ssh" / SITE["gate_key"]
KEEP = Path(SITE["keep"])
#: Llama 3's regular token ids; 128,000 and up are its special tokens
VOCAB = 128_000
#: the minutes past `auditor.warden_minutes` a session asks for
MARGIN = 3
LATE_S, POLL_S = 60, 15
SSH = ("ssh", "-T", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ServerAliveInterval=15")
#: the flags the endpoint serves the session with: the warden's id form, in its release order, each window sealed
#: without the attention interiors (about 1 GB a request at 256 ids)
ENDPOINT = "verity-vllm serve-openai --host 127.0.0.1 --port {port} --protocol ids --id-payload 503 --admit release-order " \
           "--scheme rows-sha512 --seal --retain-exclude hidden_m1"
#: serve-openai's default --max-model-len, which the endpoint's flags name only past it
CONTEXT = 512
#: the endpoint's map from its engine requests to the warden's, in its serve dir: each engine request's engine_id and
#: its warden user (null for a body that sent none), a line in requests.jsonl's order: requests.jsonl less its values
USERS = "request_users.jsonl"
#: the endpoint's records its art publishes beside the endpoint step's report.json, from the serve dir
RECORDS = ("windows.jsonl", "serve.json")
#: what the endpoint step reads in the serve dir, which stays there
SERVED = ("advice_trace.jsonl", "requests.jsonl", "windows.jsonl", "serve.json", USERS)
#: the certified I/O file `io` writes in KEEP/RUN and the endpoint step reads
CERTIFIED = "certified-io.json"
#: a record that holds a request's values: a scheduler trace's record of these kinds (`verity_vllm.engine.step_trace`)
TRACED = ("arrive", "step", "finish")
#: or a record with one of these fields: the trace's, requests.jsonl's and the certified I/O's request values
VALUED = ("prompt_len", "sampling", "seed", "rows", "emitted", "prompt_ids", "streamed_ids", "prompt", "output")
HAND_START = ("serve-openai run directly, not examples/vllm/serve.sh: serve.sh's serve-openai line (--required-manifest "
              "ROW/manifest.json, the int8 B4 row's Build of record r20261026-182623-0d2a, with vy-dc-h200-2's own taps as "
              "--hidden-so and --norm-tap-so) plus the endpoint's flags, and --disk-floor-gb 40 (serve-openai's own default "
              "under --seal is 300 GB; 72 GB of seals then needs about 112 GB free at the start); serve.sh refuses on "
              "h200-2 under its 300 GB disk floor "
              "and sends its own smoke and load requests into the records. VERITY_STAGING_BOUNDED stays unset: bounded "
              "staging, on by default under windowed staging, is what --retain-exclude needs, and the collector refuses "
              "VERITY_STAGING_BOUNDED=0 beside it. Its --copy-records-to copies requests.jsonl and advice_trace.jsonl "
              "too, so it is left off or points at a private dir on h200-2: the serve dir's records stay there, and "
              "the endpoint's art is `endpoint`'s $OUT, its report.json, windows.jsonl and serve.json")


class Refused(Exception):
    """A session this runbook won't start, and why."""
    code = 2


class Late(Refused):
    """Launched more than `LATE_S` past ``--at``: nothing is sent."""
    code = 3


def prompts(n: int, ids: int, vocab: int, seed: bytes) -> list[list[int]]:
    """`n` prompts of `ids` token ids below `vocab`: prompt j's id i is SHA-512(seed ‖ u32be j ‖ u32be i)'s first 8
    bytes, mod `vocab`."""
    return [[int.from_bytes(hashlib.sha512(seed + j.to_bytes(4, "big") + i.to_bytes(4, "big")).digest()[:8], "big")
             % vocab for i in range(ids)] for j in range(n)]


def words(a: argparse.Namespace) -> list[str]:
    return ["warden-rehearse", "--peer", a.peer, "--port", str(a.port), "--minutes", str(a.minutes),
            "--requests", str(a.requests), "--height", str(a.height), "--scan", "--prompts", "--tokens", str(a.tokens),
            *(["--window-per-group"] if getattr(a, "window_per_group", False) else [])]


def scratch_host(d: Path, peer: str) -> A.Host:
    (d / "var").mkdir(parents=True, exist_ok=True)
    (d / "peers").write_text(f"{peer} research@127.0.0.1\n")
    return A.Host(root=d / "var", peers=d / "peers", pins=d / "pins.jsonl")


def check(a: argparse.Namespace, data: bytes) -> dict:
    """The request as this tree's gate records it, or Refused with the gate's reason or the scan class's."""
    if not 1 <= a.prompt_ids <= KI:
        raise Refused(f"--prompt-ids {a.prompt_ids}: the scan's class takes 1 to KI = {KI} prompt ids a request")
    if not 1 <= a.tokens <= KO:
        raise Refused(f"--tokens {a.tokens}: the scan's class answers 1 to KO = {KO} ids a request")
    if not 2 <= a.vocab <= 1 << A.WARDEN_PROMPT_ID_BITS:
        raise Refused(f"--vocab {a.vocab}: 2 to 2^{A.WARDEN_PROMPT_ID_BITS}, the scan's ids")
    with tempfile.TemporaryDirectory() as d:
        try:
            req = A.parse_warden(words(a)[1:], scratch_host(Path(d), a.peer))
            A.warden_prompts(req, data)
        except A.Refused as e:
            raise Refused(f"the gate refuses it: {e}") from None
    return req


def plan(a: argparse.Namespace, data: bytes) -> dict:
    req = check(a, data)
    session = (f"python3 benchmarks/network_traces/served_session.py session --out \"$OUT\" --peer {a.peer} "
               f"--port {a.port} --prompts {a.prompts} --requests {a.requests} --height {a.height} "
               f"--minutes {a.minutes}" + (f" --at {a.at}" if a.at else ""))
    ctx = a.prompt_ids + a.tokens
    return {"request": shlex.join(words(a)), "minutes": a.minutes,
            "least_minutes": A.warden_minutes(a.requests, a.height),
            "prompts": req["prompts"], "prompts_bytes": len(data),
            "endpoint": ENDPOINT.format(port=a.port) + (f" --max-model-len {ctx}" if ctx > CONTEXT else ""),
            "hand_start": HAND_START,
            "runs": {
                "session": "research run --on vy-nebius-cpu-2 --source . --project verity --as network-accounting "
                           f"--cwd source --timeout {a.minutes + 60}m -- bash -lc '{session}'",
                "io": "research run --on vy-nebius-cpu-2 --source . --project verity --as network-accounting "
                      "--cwd source --timeout 30m -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && uv run "
                      "--no-sync python benchmarks/network_traces/served_session.py io --run RUN'",
                "endpoint": f"research run --on {a.peer} --source . --project verity --as network-accounting "
                            f"--cwd source --timeout 30m --send {CERTIFIED} -- bash -lc 'python3 "
                            "benchmarks/network_traces/served_session.py endpoint --from SERVE_DIR --io "
                            f"\"$RESEARCH_RUN_DIR/inputs/{CERTIFIED}\" --out \"$OUT\" --port {a.port}'",
                "chain": "research run --on vy-nebius-cpu-2 --source . --project verity --as network-accounting "
                         "--cwd clone --timeout 270m --store-read --send challenger-key.json -- "
                         "env CHECK_SLOTS=/workspace/research/locks/network/slots "
                         "python3 tools/verity/check/slot.py --by network-accounting --wait-max 1200 --run-max 14400 "
                         "-- nice -n 19 bash -c 'python3 benchmarks/network_traces/served_session.py chain --run RUN "
                         "--endpoint ART --draw 8 --coins gate "
                         "--challenger-key \"$RESEARCH_RUN_DIR/inputs/challenger-key.json\"'"}}


def keep_dir(keep: Path, out: Path) -> Path:
    keep, out = keep.resolve(), out.resolve()
    if keep == out or out in keep.parents:
        raise Refused(f"--keep {keep} is inside --out, which custody publishes: give a directory outside it")
    return keep


def public(full: bytes) -> bytes:
    """What the gate's `fetch` gives any key but the run's own lane's: the bundle less ``witness/``."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(full), mode="r:gz") as tar, tarfile.open(fileobj=buf, mode="w:gz") as out:
        for m in tar:
            if A.WITNESS not in m.name.split("/") and (m.isfile() or m.isdir()):
                out.addfile(m, tar.extractfile(m) if m.isfile() else None)
    return buf.getvalue()


def members(bundle: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tar:
        return {m.name: tar.extractfile(m).read() for m in tar if m.isfile()}


def outcome(bundle: bytes) -> dict:
    """The session's public outcome, from the bundle's ``end.json`` and ``out/deployment.json``."""
    files = {name.partition("/")[2]: data for name, data in members(bundle).items()}
    end = json.loads(files["end.json"])
    dep = json.loads(files["out/deployment.json"]) if "out/deployment.json" in files else {}
    n, params = dep.get("windows"), dep.get("params") or {}
    span = n * params["window"] * params["bucket_ms"] / 1000 if n is not None and params else None
    return {"run": end.get("run"), "rc": end.get("rc"), "ended": end.get("ended"), "ended_utc": end.get("ended_utc"),
            "pinned": end.get("pinned"), "verdict": end.get("verdict"), "origin_unix": dep.get("origin_unix"),
            "windows": n, "session_s": span}


def needles(ps: list[list[int]]) -> list[bytes]:
    """Each prompt's first 16 ids as JSON writes them, compact (the prompts') or with json.dumps's default separators."""
    return [sep.join(map(str, p[:16])).encode() for p in ps if p for sep in (",", ", ")]


def leaked(out: Path, needles: list[bytes]) -> list[str]:
    """Each file of `out` that holds any of `needles`, ``bundle.tgz`` read through its members."""
    found = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            data = p.read_bytes()
            parts = [data, *members(data).values()] if p.name == "bundle.tgz" else [data]
            if any(n in d for d in parts for n in needles):
                found.append(p.relative_to(out).as_posix())
    return found


def wait_until(at: str, *, now=time.time, sleep=time.sleep) -> None:
    t = datetime.fromisoformat(at.replace("Z", "+00:00")).timestamp()
    if now() > t + LATE_S:
        raise Late(f"past --at {at} by more than {LATE_S} s: nothing sent")
    if now() < t:
        print(f"waiting until {at}", flush=True)
        sleep(t - now())


def remote(a: argparse.Namespace, data: bytes, out: Path, *, sleep=time.sleep) -> bytes:
    """The session through the gate at `a.gate`: the lane's fetch of its run."""
    ssh = [*SSH, "-i", str(a.key), a.gate]
    got = subprocess.run([*ssh, shlex.join(words(a))], input=data, capture_output=True)
    (out / "start.json").write_bytes(got.stdout)
    try:
        run = json.loads(got.stdout)["run"]
    except (ValueError, KeyError, TypeError):
        raise Refused(f"the gate didn't start the session (exit {got.returncode}): {got.stdout[:400]!r} "
                      f"{got.stderr[-400:]!r}") from None
    print(f"gate run {run}", flush=True)
    while True:
        got = subprocess.run([*ssh, f"status {run}"], capture_output=True)
        if got.returncode == 0:
            (out / "status.json").write_bytes(got.stdout)
            if json.loads(got.stdout).get("end") is not None:
                break
        sleep(POLL_S)
    return subprocess.run([*ssh, f"fetch {run}"], capture_output=True, check=True).stdout


def pinned() -> str:
    try:
        return A.pinned_commit()
    except A.Refused:
        got = subprocess.run(["git", "-C", str(TREE), "rev-parse", "HEAD"], capture_output=True, text=True)
        return got.stdout.strip() or "unpinned"


def local(a: argparse.Namespace, data: bytes, keep: Path) -> bytes:
    """The dry rehearsal: this tree's gate on a scratch host under `keep`, the stand-in as its tunnel, or ``a.tunnel``
    (``(ssh, port) -> argv``, a process listening on ``port``) when given."""
    host = scratch_host(keep / "local-gate", a.peer)
    rc, start = A.gate(["--ssh", shlex.join(words(a)), "--key", LANE], host, now=datetime.now(UTC),
                       token=secrets.token_hex(2), own=lambda p: None, certifier_active=lambda r: False,
                       start_certifier=lambda r: None, read=lambda n: data[:n])
    if rc != 0:
        raise Refused(f"the gate didn't start the session (exit {rc}): {start[:400]!r}")
    run = json.loads(start)["run"]
    print(f"gate run {run} (local)", flush=True)
    endpoint = keep / run / "endpoint"
    endpoint.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    stub = [sys.executable, str(HERE / "demo_stub.py"), "serve", "--prompt-ids", str(a.prompt_ids), "--tokens",
            str(a.tokens), "--vocab", str(a.vocab), "--out", str(endpoint)]
    A.certifier_run(host, run, pinned=pinned(), now=lambda: datetime.now(UTC),
                    tunnel_argv=getattr(a, "tunnel", None) or (lambda ssh, port: [*stub, "--port", str(port)]))
    return A.certifier_fetch(run, host, witness=True)


def session(a: argparse.Namespace, data: bytes, ps: list[list[int]]) -> int:
    out = a.out.resolve()
    keep = keep_dir(a.keep, out)
    keep.mkdir(parents=True, exist_ok=True)
    keep.chmod(0o700)
    if a.at:
        wait_until(a.at)
    full = local(a, data, keep) if a.gate == "local" else remote(a, data, out)
    shown = public(full)
    doc = outcome(shown)
    run = doc["run"]
    (keep / run).mkdir(exist_ok=True)
    (keep / run).chmod(0o700)
    kept = keep / run / "fetch.tar.gz"
    kept.write_bytes(full)
    kept.chmod(0o600)
    (out / "bundle.tgz").write_bytes(shown)
    doc |= {"gate": a.gate, "request": shlex.join(words(a)), "prompts": {"count": len(ps), "ids": sum(map(len, ps))},
            "keep": str(kept)}
    (out / "session.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    found = leaked(out, needles(ps))
    for rel in found:
        (out / rel).unlink()
    print(json.dumps(doc | {"leaked_removed": found}, sort_keys=True), flush=True)
    if found:
        return 4
    return 0 if doc["rc"] == 0 and (doc["verdict"] or {}).get("accepted") is True else (doc["rc"] or 1)


def served_peer(fetch: Path) -> str | None:
    """The node the kept gate run served on: its ``request.json``'s peer."""
    with tarfile.open(fetch, "r:gz") as tar:
        for m in tar:
            if m.isfile() and m.name.count("/") == 1 and m.name.endswith("/request.json"):
                return json.loads(tar.extractfile(m).read()).get("peer")
    return None


def first_parent(tree: Path, dest: Path) -> Path:
    """main's first-parent history from `tree`'s origin, written to `dest`."""
    git = ["git", "-C", str(tree)]
    shallow = subprocess.run([*git, "rev-parse", "--is-shallow-repository"], capture_output=True, text=True)
    unshallow = ["--unshallow"] if shallow.stdout.strip() == "true" else []
    for refspec in ("+refs/heads/main:refs/remotes/origin/main", "+refs/remotes/origin/main:refs/remotes/origin/main"):
        if subprocess.run([*git, "fetch", "-q", *unshallow, "origin", refspec], capture_output=True).returncode == 0:
            break
    got = subprocess.run([*git, "rev-list", "--first-parent", "refs/remotes/origin/main"], capture_output=True, text=True)
    if got.returncode or not got.stdout.strip():
        raise Refused(f"no main first-parent history from {tree}'s origin ({got.stderr.strip()[-200:]}): give --main FILE")
    dest.write_text(got.stdout)
    return dest


def input_file(path: Path, what: str) -> Path:
    if not path.is_file():
        raise Refused(f"no {what} {path}: send it with the run (--send) and give its path under the run's inputs/")
    return path


def coin_args(a: argparse.Namespace, fetch: Path) -> list[str]:
    """The scan's draw arguments: none (every unit), ``--draw K`` on the verifier's coins, or K on the gate's, drawn on
    the auditor's side (`--coin-peer`)."""
    if a.coins == "verifier":
        return [] if a.draw is None else ["--draw", str(a.draw)]
    if a.draw is None:
        raise Refused("--coins gate draws --draw K of the scan's units: give K")
    served = served_peer(fetch)
    if served is None:
        raise Refused(f"{fetch} holds no request.json naming the node it served on, so the coin peer can't be checked")
    if a.coin_peer in (served, PEER):
        raise Refused(f"--coin-peer {a.coin_peer} is the served node, the prover's: the gate key and the warden run's "
                      f"witness are the auditor's, so the draw runs where the chain does, {AUDITOR} by default")
    if a.challenger_key is None:
        raise Refused("--coins gate needs --challenger-key, the deployment's challenger-key/v1 record "
                      "(challenger-key current)")
    if not a.key.is_file():
        raise Refused(f"no gate key {a.key}: the draw runs on the node that holds the lane's gate key, {AUDITOR}")
    key = input_file(a.challenger_key, "challenger key record")
    main = input_file(a.main, "--main") if a.main else first_parent(TREE, a.keep / a.run / "scan-main.txt")
    return ["--draw", str(a.draw), "--challenger", f"gate:{a.gate}", "--main", str(main), "--peer", a.coin_peer,
            "--challenger-key", str(key), "--gate-key", str(a.key)]


def certified_io(fetch: Path) -> tuple[dict, dict]:
    """The kept warden run's deployment.json and certified I/O (`warden_chain.io`), read in a scratch directory beside
    the fetch."""
    sys.path[:0] = [str(TREE), str(HERE)]
    import warden_chain as WC
    with tempfile.TemporaryDirectory(dir=fetch.parent) as d:
        with tarfile.open(fetch, "r:gz") as tar:
            tar.extractall(d, filter="data")
        try:
            run = next(p for p in Path(d).iterdir() if p.is_dir())
            return WC.deployment(run)[0], WC.io(run)
        except SystemExit as e:
            raise Refused(f"{fetch}: {e}") from None


def certified(a: argparse.Namespace) -> int:
    fetch = a.keep / a.run / "fetch.tar.gz"
    if not fetch.is_file():
        raise Refused(f"no kept fetch {fetch}: the session's run writes it")
    dep, io = certified_io(fetch)
    if not io["accepted"]:
        raise Refused(f"the kept warden run's I/O doesn't read whole off its cells: {io['violations'][:3]}")
    dest = a.keep / a.run / CERTIFIED
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"deployment": dep, "io": io}, f)
    dest.chmod(0o600)
    print(json.dumps({"io": str(dest), "warden_run": dep["run_id"], "requests": io["requests"],
                      "tokens": io["tokens"]}), flush=True)
    return 0


def finish(out: list[int], max_tokens: int, eos: set[int]) -> dict:
    """serve-openai's finish of an answer `out`: the model's EOS when it ends at one or short of its max_tokens, else
    its length."""
    stop = (bool(out) and out[-1] in eos) or len(out) < max_tokens
    return {"reason": "stop" if stop else "length", "stop_reason": None}


def rebuild(src: Path, dest: Path, records: list[dict], eos: list[int]) -> Path:
    """The endpoint's records as its step reads them: `dest` (0700, made afresh) with `src`'s `SERVED` records but
    requests.jsonl, rebuilt from `USERS` and the certified I/O's ``records``."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(mode=0o700, parents=True)
    for f in SERVED:
        if f != "requests.jsonl":
            shutil.copyfile(src / f, dest / f)
    by, lines = {r["user"]: r for r in records}, []
    for m in map(json.loads, filter(str.strip, (src / USERS).read_text().splitlines())):
        if (r := by.pop(m["user"], None)) is None:
            raise Refused(f"{USERS}: engine request {m['engine_id']}'s user {m['user']!r} is no certified request left "
                          "(none sent, not the warden's, or an earlier line's)")
        lines.append(json.dumps({"engine_id": m["engine_id"], "user": m["user"], "prompt_ids": r["prompt"],
                                 "streamed_ids": r["output"], "finish": finish(r["output"], r["max_tokens"], set(eos))}))
    fd = os.open(dest / "requests.jsonl", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("".join(line + "\n" for line in lines))
    return dest


def served_as(serve: dict, port: int) -> list[str]:
    """Each of `ENDPOINT`'s flags that serve.json shows the endpoint didn't serve with, beside what it recorded."""
    words, off = shlex.split(ENDPOINT.format(port=port))[2:], []
    host, at = (serve.get("address") or [None, None])[:2]
    for i, w in enumerate(words):
        if not w.startswith("--"):
            continue
        v = words[i + 1] if i + 1 < len(words) and not words[i + 1].startswith("--") else None
        flag = w[2:]
        have = {"host": host, "port": at, "seal": serve.get("sealed")}.get(flag, serve.get(flag.replace("-", "_")))
        if (have is None) if v is None else str(have) != v:
            off.append(f"{w}{' ' + v if v else ''} (serve.json: {have!r})")
    return off


def valued(data: bytes, marks: list[bytes]) -> str | None:
    """Why `data`, a JSON document or JSON lines, isn't published: it holds one of ``marks`` (a request's prompt ids,
    or a request-valued field's name at any depth), or a record (the document, or one of its lines) that is a trace's
    `TRACED` record or has a `VALUED` field; None when it holds none."""
    if any(m in data for m in marks):
        return "a request's prompt ids or a request-valued field"
    try:
        docs = [json.loads(data)]
    except ValueError:
        try:
            docs = [json.loads(x) for x in data.splitlines() if x.strip()]
        except ValueError:
            return "a line that isn't JSON, so it isn't read for request values"
    for d in docs:
        if isinstance(d, dict) and d.get("kind") in TRACED:
            return f"a scheduler trace's {d['kind']} record"
        if isinstance(d, dict) and (fields := sorted(set(d) & set(VALUED))):
            return f"a record with {', '.join(fields)}"
    return None


def endpoint(a: argparse.Namespace) -> int:
    src, out = a.src.resolve(), a.out.resolve()
    keep = (a.keep or src.with_name(src.name + ".endpoint")).resolve()
    run = out.parent
    given = None if a.io == "-" else Path(a.io).resolve()

    def within(p: Path, d: Path | None) -> bool:
        return d is not None and (p == d or d in p.parents)

    for d, what in ((out, "--out"), (run, "--out's directory, the run dir"), (TREE, "the tree")):
        if within(keep, d):
            if given is not None and within(given, run):
                given.unlink()
            raise Refused(f"--keep {keep} is inside {what}, which is published or shared: give a private directory on "
                          "the served node")
    keep.mkdir(mode=0o700, parents=True, exist_ok=True)
    keep.chmod(0o700)
    if given is not None and within(given, run):
        given = Path(shutil.move(given, keep / CERTIFIED))
        given.chmod(0o600)
    if within(out, TREE):
        raise Refused(f"--out {out} is in the tree: give the run's $OUT, expanded on the node (inside bash -lc '...')")
    if missing := [f for f in SERVED if not (src / f).is_file()]:
        raise Refused(f"--from {src}: no {', '.join(missing)}: give the endpoint's serve dir, its --out")
    if off := served_as(json.loads((src / "serve.json").read_text()), a.port):
        raise Refused(f"the endpoint didn't serve with the session's flags: {'; '.join(off)}. Start it again with the "
                      "plan's endpoint and hand_start")
    doc = json.load(sys.stdin) if given is None else json.loads(given.read_text())
    dep, io = doc["deployment"], doc["io"]
    if not io.get("accepted"):
        raise Refused(f"--io: the warden run's I/O doesn't read whole off its cells: {io.get('violations', [])[:3]}")
    eos = [int(t) for t in a.eos_ids.split(",")]
    ep = rebuild(src, keep / "records", io["records"], eos)
    sys.path[:0] = [str(TREE), str(TREE / "integrations" / "vllm"), str(HERE)]
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, sys.path))
    import chain_record as CR
    rep = CR.served_endpoint(dep, io, ep, eos) | {"at": CR.SERVED_NODE}
    step = CR.warden_endpoint_step(rep)
    rep["verdict"] = step["verdict"]
    publish = {"report.json": (json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n").encode(),
               **{f: (src / f).read_bytes() for f in RECORDS}}
    ps = [json.loads(x).get("prompt_ids") or [] for x in (src / "requests.jsonl").read_text().splitlines() if x.strip()]
    marks = [*needles(ps + [r["prompt"] for r in io["records"]]),
             *(f'"{k}"'.encode() for k in ("prompt_len", "sampling", "emitted", "prompt_ids", "streamed_ids"))]
    if held := [f"{f}: {why}" for f, d in publish.items() if (why := valued(d, marks))]:
        raise Refused(f"{'; '.join(held)}, which the endpoint's art never publishes: nothing written")
    out.mkdir(parents=True, exist_ok=True)
    for f, d in publish.items():
        (out / f).write_bytes(d)
    print(json.dumps({"verdict": rep["verdict"], "why": step["why"], "warden_run": rep["warden_run"],
                      "advice": step["advice"], "published": sorted(publish), "kept": str(keep)}), flush=True)
    return 0 if rep["verdict"] == "accepted" else 1


def chain(a: argparse.Namespace) -> int:
    if not (HERE / "leak_guard.py").is_file():
        raise Refused("this tree has no record leak guard (leak_guard.py), which chain_record.sh runs on its "
                      "way out: run the chain from a tree with it")
    fetch = a.keep / a.run / "fetch.tar.gz"
    if not fetch.is_file():
        raise Refused(f"no kept fetch {fetch}: the session's run writes it")
    endpoint = a.endpoint or (str(a.keep / a.run / "endpoint") if (a.keep / a.run / "endpoint").is_dir() else None)
    if endpoint and Path(endpoint).is_dir() and any(Path(endpoint).rglob("advice_trace.jsonl")) and not a.eos_ids:
        raise Refused("--endpoint's records need --eos-ids, the model's EOS ids (the stand-in's is its --vocab)")
    env = dict(os.environ)
    env.update({"CR_PATH": "warden", "CR_GATES": str(fetch), "CR_MEM_GB": "96", "CR_N2_DRAW": "8",
                "CR_TIE_WINDOWS": "8"})
    for kv in a.set:
        k, eq, v = kv.partition("=")
        if not (eq and k.startswith("CR_")):
            raise Refused(f"{kv!r}: CR_K=V, a chain_record.sh variable")
        if k in ("CR_SCAN_ARGS", "CR_SCAN_CMD") and any(w in v for w in ("--draw", "--challenger", "--peer")):
            raise Refused(f"{k}: the scan's draw and its coins are --draw and --coins's, which keep the draw off the "
                          "prover's node")
        env[k] = v
    coins = coin_args(a, fetch)
    if any(c.isspace() for w in coins for c in w):
        raise Refused(f"{coins}: chain_record.sh splits CR_SCAN_ARGS on whitespace, so give paths without it")
    if scan := " ".join(coins):
        env["CR_SCAN_ARGS"] = f"{scan} {env.get('CR_SCAN_ARGS', '')}".strip()
    if endpoint:
        env.update({"CR_ENDPOINT": endpoint, **({"CR_EOS_IDS": a.eos_ids} if a.eos_ids else {})})
    return subprocess.run(["bash", str(HERE / "chain_record.sh")], cwd=TREE, env=env).returncode


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="served_session.py", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "session"):
        q = sub.add_parser(name)
        q.add_argument("--peer", default=PEER)
        q.add_argument("--port", type=int, default=PORT)
        q.add_argument("--prompts", type=int, default=64)
        q.add_argument("--requests", type=int, default=REQUESTS)
        q.add_argument("--height", type=int, default=12)
        q.add_argument("--minutes", type=int)
        q.add_argument("--prompt-ids", type=int, default=PROMPT_IDS)
        q.add_argument("--tokens", type=int, default=KO)
        q.add_argument("--vocab", type=int, default=VOCAB)
        q.add_argument("--at")
    q = sub.choices["session"]
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--gate", default=GATE, help="USER@HOST, the gate's; or local, the dry rehearsal")
    q.add_argument("--key", type=Path, default=KEY)
    q.add_argument("--keep", type=Path, default=KEEP)
    q = sub.add_parser("io")
    q.add_argument("--run", required=True)
    q.add_argument("--keep", type=Path, default=KEEP)
    q = sub.add_parser("endpoint")
    q.add_argument("--from", dest="src", type=Path, required=True, help="the endpoint's serve dir, serve-openai's --out")
    q.add_argument("--io", required=True, help=f"io's {CERTIFIED}, or - for stdin")
    q.add_argument("--out", type=Path, required=True,
                   help="the run's $OUT: its directory, the run dir, is published, so --keep and an --io in it are refused "
                        "and moved out")
    q.add_argument("--eos-ids", default="128001")
    q.add_argument("--port", type=int, default=PORT)
    q.add_argument("--keep", type=Path, help="private, 0700: by default SERVE_DIR.endpoint beside the serve dir")
    q = sub.add_parser("chain")
    q.add_argument("--run", required=True)
    q.add_argument("--keep", type=Path, default=KEEP)
    q.add_argument("--endpoint", help="endpoint's art (its report.json), or a dry rehearsal's endpoint records")
    q.add_argument("--eos-ids")
    q.add_argument("--draw", type=int, help="the scan's K drawn units (every unit by default)")
    q.add_argument("--coins", choices=("verifier", "gate"), default="verifier")
    q.add_argument("--coin-peer", default=AUDITOR, help="with --coins gate: this node, as the gate's peers file names it")
    q.add_argument("--challenger-key", type=Path)
    q.add_argument("--main", type=Path, help="the verifier's git rev-list --first-parent origin/main")
    q.add_argument("--gate", default=GATE)
    q.add_argument("--key", type=Path, default=KEY)
    q.add_argument("set", nargs="*", help="CR_K=V")
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    try:
        if a.cmd == "chain":
            return chain(a)
        if a.cmd == "io":
            return certified(a)
        if a.cmd == "endpoint":
            return endpoint(a)
        if a.minutes is None:
            a.minutes = A.warden_minutes(a.requests, a.height) + MARGIN
        if a.cmd == "session" and a.gate == "local" and a.vocab > 256:
            raise Refused(f"--gate local: the stand-in serves byte ids, --vocab at most 256, not {a.vocab}")
        ps = prompts(a.prompts, a.prompt_ids, a.vocab, secrets.token_bytes(32))
        data = json.dumps(ps, separators=(",", ":")).encode()
        doc = plan(a, data)
        if a.cmd == "plan":
            print(json.dumps(doc, indent=1))
            return 0
        print(f"{datetime.now(UTC):%FT%TZ}: {doc['request']}", flush=True)
        return session(a, data, ps)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
