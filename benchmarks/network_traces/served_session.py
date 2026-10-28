"""served-session: the session of record, the network certifier in front of a served model, as `research run` steps
(nothing reaches a node otherwise). The gate's scan rehearsal (`vy-challenger-gate warden-rehearse --scan --prompts`,
`benchmarks/live_challenger/auditor.py`) runs against the endpoint on the peer's loopback, by default vy-dc-h200-2's
`verity-vllm serve-openai` on 127.0.0.1:18782, inside the scan's class (`scan_zk.W63`: at most KI = 1,024 prompt ids and
KO = 64 ids out a request). The session's prompts are of `PROMPT_IDS` = 256 ids, since prompts of 1,024 reach the warden
late against its D = 2.7 s (about 1.2 s of ingress and 1.2 to 4 s of committed prefill), and each is answered
greedily with at most 64 ids: the base Llama-3.2-1B ends at its one EOS id, 128001, after 1 to 5 ids in 15 to 27% of
requests. The network chain's run of record (`chain_record.sh`'s warden path) then proves the session.

    python served_session.py plan [FLAGS]
    python served_session.py session --out DIR [FLAGS] [--gate USER@HOST | --gate local] [--key KEY] [--keep DIR]
        [--at UTC]
    python served_session.py chain --run RUN [--keep DIR] [--endpoint DIR|ART] [--eos-ids IDS] [--draw K]
        [--coins verifier | --coins gate --challenger-key RECORD [--coin-peer NAME] [--main FILE]] [CR_K=V ...]

FLAGS: [--peer vy-dc-h200-2] [--port 18782] [--prompts 64] [--requests 113] [--height 12] [--minutes M]
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
  - ``--gate USER@HOST`` (vy-challenger-1's gate, `GATE`, under the lane's gate key ``--key``): the request with the
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
* **chain** runs chain_record.sh's warden path on the kept fetch (CR_PATH=warden, CR_GATES=KEEP/RUN/fetch.tar.gz), with
  the endpoint's records (``--endpoint``; a dry rehearsal's KEEP/RUN/endpoint by default) and the model's EOS ids
  (``--eos-ids``: 128001 for the served model, the stand-in's ``--vocab`` for a dry rehearsal's), N2's period drawn
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

**The endpoint.** It is started by hand on vy-dc-h200-2 (`HAND_START`) ahead of the session's ``--at``, and it
listens within about 35 s. Its records art (``--endpoint ART``) is advice_trace.jsonl, windows.jsonl, serve.json and
`USERS`, each engine request's id and its warden user (``{"engine_id", "user"}`` a line, in requests.jsonl's order).
requests.jsonl, whose prompt ids are request values, stays in h200-2's serve dir: this runbook neither fetches
nor publishes it. The endpoint step's advice (`verity-vllm advice`) needs it, the Commit record, so ``chain`` rebuilds
it (`endpoint_records`) from `USERS` and the kept warden run's certified I/O (`warden_chain.io`), each request's prompt
and streamed ids as the warden's cells carry them and its finish (`finish`), in KEEP/RUN/endpoint-rebuilt (0700, outside
the run dir). chain_record.sh's copy of it is in its scratch, removed on exit, and its leak guard reads the record
against the warden's witness; the endpoint's trace is then checked against the certified I/O, not the prover's record
of it. An endpoint that holds requests.jsonl (a dry rehearsal's stand-in's) is read as it is.

The runs (the lane's gate key is on vy-nebius-cpu-2; KEEP is the lane's gate directory there):

    research run --on vy-nebius-cpu-2 --source . --project verity --as network-accounting --cwd source --timeout 90m \\
      -- bash -lc 'python3 benchmarks/network_traces/served_session.py session --out "$OUT" --at 2026-10-28T12:00:00Z'
    research run --on vy-nebius-cpu-2 --source . --project verity --as network-accounting --cwd clone --timeout 270m \\
      --store-read --send challenger-key.json -- env CHECK_SLOTS=/workspace/research/locks/network/slots python3 \\
      tools/verity/check/slot.py --by network-accounting --wait-max 1200 --run-max 14400 -- nice -n 19 bash -c \\
      'python3 benchmarks/network_traces/served_session.py chain --run RUN --endpoint ART \\
      --eos-ids 128001 --draw 8 --coins gate --challenger-key "$RESEARCH_RUN_DIR/inputs/challenger-key.json"'

(``--store-read`` gives the job the read-only store key its fetch of the records art needs; ``--run-max`` leaves room
over a dry chain of 16 requests on the gate's coins, which held its slot about 9,500 s, an hour of it the challenger's
session; ``--coins verifier``, without ``--send`` and ``--challenger-key``, is the fallback);

and the dry rehearsal, on `AUDITOR`, whose KEEP its dry chain reads:

    research run --on vy-nebius-cpu-2 --pool network --source . --project verity --as LANE --cwd source \\
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
#: the session's requests: what vy-dc-h200-2's free disk seals, within 72 GB even if every request runs all KO ids
#: (the endpoint's sizing: 636.5 MB a full-length request)
REQUESTS = 113
PEER, PORT, GATE, LANE = "vy-dc-h200-2", 18782, "research@204.12.178.84", "network-accounting"
#: the auditor's side, where the chain runs and the gate's coins for its draw arrive
AUDITOR = "vy-nebius-cpu-2"
KEY = Path.home() / ".ssh" / "vy-gate-network-accounting"
KEEP = Path("/workspace/research/lanes/network-accounting/gate")
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
#: the endpoint's record in its records artifact: each engine request's engine_id and its warden user (null for a body
#: that sent none), a line in requests.jsonl's order: requests.jsonl less its values
USERS = "request_users.jsonl"
HAND_START = ("serve-openai run directly, not examples/vllm/serve.sh: serve.sh's serve-openai line (--required-manifest "
              "ROW/manifest.json, the int8 B4 row's Build of record r20261026-182623-0d2a, with vy-dc-h200-2's own taps as "
              "--hidden-so and --norm-tap-so) plus the endpoint's flags, and --disk-floor-gb 40 (serve-openai's own default "
              "under --seal is 300 GB; 72 GB of seals then needs about 112 GB free at the start); serve.sh refuses on "
              "h200-2 under its 300 GB disk floor "
              "and sends its own smoke and load requests into the records. Its --copy-records-to copies requests.jsonl "
              "too, so it points at a private dir on h200-2, and the records art is the four public records copied from "
              "there: request_users.jsonl, advice_trace.jsonl, windows.jsonl and serve.json")


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
                "chain": "research run --on vy-nebius-cpu-2 --source . --project verity --as network-accounting "
                         "--cwd clone --timeout 270m --store-read --send challenger-key.json -- "
                         "env CHECK_SLOTS=/workspace/research/locks/network/slots "
                         "python3 tools/verity/check/slot.py --by network-accounting --wait-max 1200 --run-max 14400 "
                         "-- nice -n 19 bash -c 'python3 benchmarks/network_traces/served_session.py chain --run RUN "
                         "--endpoint ART --eos-ids 128001 --draw 8 --coins gate "
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
    """Each prompt's first 16 ids as the prompts' JSON writes them."""
    return [json.dumps(p[:16], separators=(",", ":"))[1:-1].encode() for p in ps]


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


def certified_io(fetch: Path) -> dict:
    """The kept warden run's certified I/O (`warden_chain.io`), read in a scratch directory beside the fetch."""
    sys.path[:0] = [str(TREE), str(HERE)]
    import warden_chain as WC
    with tempfile.TemporaryDirectory(dir=fetch.parent) as d:
        with tarfile.open(fetch, "r:gz") as tar:
            tar.extractall(d, filter="data")
        try:
            return WC.io(next(p for p in Path(d).iterdir() if p.is_dir()))
        except SystemExit as e:
            raise Refused(f"{fetch}: {e}") from None


def finish(out: list[int], max_tokens: int, eos: set[int]) -> dict:
    """serve-openai's finish of an answer `out`: the model's EOS when it ends at one or short of its max_tokens, else
    its length."""
    stop = (bool(out) and out[-1] in eos) or len(out) < max_tokens
    return {"reason": "stop" if stop else "length", "stop_reason": None}


def endpoint_records(src: str, dest: Path, fetch: Path, eos: list[int]) -> str:
    """The endpoint's records as the chain reads them: `src` when it holds requests.jsonl, else `dest` (0700) with
    `src`'s records (an art: id, a directory or a tarball) and requests.jsonl rebuilt beside its `USERS`."""
    path = Path(src)
    if path.is_dir() and any(path.rglob("requests.jsonl")):
        return src
    dest.mkdir(mode=0o700, parents=True, exist_ok=True)
    if src.startswith("art:"):
        got = subprocess.run(["research", "data", "fetch", src, "--to", str(dest)], capture_output=True, text=True)
        if got.returncode:
            raise Refused(f"research data fetch {src} (exit {got.returncode}): {got.stderr.strip()[-300:]} (a `research "
                          "run` job reads the store's remote only when launched with --store-read)")
    elif path.is_dir():
        if path.resolve() != dest.resolve():
            shutil.copytree(path, dest, dirs_exist_ok=True)
    elif path.is_file():
        with tarfile.open(path) as tar:
            tar.extractall(dest, filter="data")
    dest.chmod(0o700)
    ep = next((p.parent for p in dest.rglob("advice_trace.jsonl")), None)
    if ep is None or not (ep / USERS).is_file():
        raise Refused(f"--endpoint {src}: no requests.jsonl, and no {USERS} beside an advice_trace.jsonl to rebuild it "
                      "from: give the endpoint's records artifact")
    io = certified_io(fetch)
    if not io["accepted"]:
        raise Refused(f"the kept warden run's I/O doesn't read whole off its cells: {io['violations'][:3]}")
    by, lines = {r["user"]: r for r in io["records"]}, []
    for m in map(json.loads, filter(str.strip, (ep / USERS).read_text().splitlines())):
        if (r := by.pop(m["user"], None)) is None:
            raise Refused(f"{USERS}: engine request {m['engine_id']}'s user {m['user']!r} is no certified request left "
                          "(none sent, not the warden's, or an earlier line's)")
        lines.append(json.dumps({"engine_id": m["engine_id"], "user": m["user"], "prompt_ids": r["prompt"],
                                 "streamed_ids": r["output"], "finish": finish(r["output"], r["max_tokens"], set(eos))}))
    rebuilt = ep / "requests.jsonl"
    rebuilt.write_text("".join(line + "\n" for line in lines))
    rebuilt.chmod(0o600)
    print(f"chain: requests.jsonl rebuilt in {ep}: {len(lines)} requests from {USERS} and the certified I/O", flush=True)
    return str(dest)


def chain(a: argparse.Namespace) -> int:
    if not (HERE / "leak_guard.py").is_file():
        raise Refused("this tree has no record leak guard (leak_guard.py), which chain_record.sh runs on its "
                      "way out: run the chain from a tree with it")
    fetch = a.keep / a.run / "fetch.tar.gz"
    if not fetch.is_file():
        raise Refused(f"no kept fetch {fetch}: the session's run writes it")
    endpoint = a.endpoint or (str(a.keep / a.run / "endpoint") if (a.keep / a.run / "endpoint").is_dir() else None)
    if endpoint and not a.eos_ids:
        raise Refused("--endpoint needs --eos-ids, the model's EOS ids (the stand-in's is its --vocab)")
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
        eos = [int(t) for t in a.eos_ids.split(",")]
        env.update({"CR_ENDPOINT": endpoint_records(endpoint, a.keep / a.run / "endpoint-rebuilt", fetch, eos),
                    "CR_EOS_IDS": a.eos_ids})
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
    q = sub.add_parser("chain")
    q.add_argument("--run", required=True)
    q.add_argument("--keep", type=Path, default=KEEP)
    q.add_argument("--endpoint", help=f"the endpoint's records: its artifact (with {USERS}), a directory or a tarball")
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
