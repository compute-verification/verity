"""certifier-piece: the network certifier's piece of `served_verdicts.sh` (its ``CERTIFIER_CMD``), and the front that puts
the active warden in front of serve-openai there (its ``CERTIFIER_FRONT``).

    python certifier_piece.py front PORT SERVE_DIR OUT [--gate USER@HOST | --gate local] [--key KEY] [--peer NAME]
        [--prompts 4] [--prompt-ids 24] [--tokens 4] [--vocab 128000] [--height 10] [--minutes M] [--warmup N]
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
  defaults are the window circuit's class 4,32,4 (4 slots, prompts of at most 32 ids, 4 outputs). ``--warmup N``
  first sends N plain completions straight to PORT (`warmup`: no warden, no release header), one at a time, each of
  the group's shape and awaited, so the endpoint's first window (H1's first reservation, the kernels' first compile)
  is behind it; it then waits until serve-openai records each in `served_session.USERS`, which it does only once the
  request's window has closed. So each warm-up is a window of one request, closed before the gate is asked, which the
  pick passes over for the group's whenever the group's window holds two or more; one that fails or isn't recorded
  within `WARMUP_CLOSE_S` refuses the front. A warm-up doesn't make a slow endpoint pass: the scan's schedule gives a
  request D·τ = 2.7 s from its admission to its first egress cell, by which a short answer must be whole, and an
  endpoint slower than that fails each request late. The gate's fetch,
  ``witness/`` with it, is kept under SERVE_DIR/`PRIVATE`/RUN/fetch.tar.gz (0700, the serve dir stays on the node);
  OUT gets ``session.json`` and ``bundle.tgz``, the public outcome. ``--gate local`` is the dry run's (this tree's gate in this process, a
  `relay` to PORT its tunnel): a warden on the prover's node is no auditor's device, so it never stands for the gate.
  Exit as `served_session.session`'s.
* **piece** writes window WINDOW's core answer to DIR/answer.json (`verity/certifier/v0`, `NetworkCheck.Period`'s rows
  form): accept only when the chain accepted and ``network-period check`` accepted the period, whose answer it then is.
  1. The kept warden run: exactly one fetch under SERVE_DIR/`PRIVATE`. With none the run's load didn't go through the
     front (it wasn't set, or the gate, its tunnel or ssh failed).
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
     (``--lean``, else built in verity/core/protocols/network/lean); its answer is the piece's.
  DIR also gets the chain's public report, job and leak-guard files and the check's three input files in DIR/period,
  the case's only copy (its params alone run to tens of megabytes on a served window); the record's leak guard
  (`leak_guard.py`) then reads DIR against the warden run's witness. Only the protocol rejects: a step that can't run
  (no kept run, a build, a tool or the chain failing without a report, an input missing, the leak guard failing, or
  SIGTERM from the piece's time limit) is `NotRun`, and the piece then refuses: DIR keeps only its log and gets
  DIR/`REFUSED` naming why, with no answer, which `served_verdicts.py` reads as the certifier missing. ``--draw K`` is the scan's K drawn units (every
  unit by default); a CR_K=V goes to the chain; ``--chain-cmd`` replaces the chain (a test's, which writes the report
  and the scan's answers under RESEARCH_RUN_DIR/out).
* **relay**: a TCP relay from 127.0.0.1:``--listen`` to 127.0.0.1:``--to``, the local gate's tunnel.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
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
#: a piece's refusal to run, which `served_verdicts.py` reads as the subprotocol missing
REFUSED = "refused.txt"
LEAN_DIR = TREE / "verity" / "core" / "protocols" / "network" / "lean"
#: the window circuit's class (4 slots, prompts of at most 32 ids, 4 outputs) and the scan's lanes
PROMPTS, PROMPT_IDS, TOKENS = 4, 24, 4
HEIGHT = 10
#: how long a warm-up request may take, and how long the front waits for serve-openai to record the warm-ups'
#: windows closed
WARMUP_REQUEST_S, WARMUP_CLOSE_S = 600, 600


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
        a.minutes = SS.A.warden_group_minutes(1, a.tokens, a.height) + SS.MARGIN
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
        return SS.session(a, data, ps)
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
        return subprocess.run(argv, cwd=TREE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def period_exe(lean: str | None) -> str:
    if lean:
        return lean
    got = subprocess.run(["lake", "build", "network-period"], cwd=LEAN_DIR, capture_output=True, text=True)
    exe = LEAN_DIR / ".lake" / "build" / "bin" / "network-period"
    if got.returncode or not exe.is_file():
        raise NotRun(f"lake build network-period (exit {got.returncode}): {(got.stdout + got.stderr)[-400:]}")
    return str(exe)


def period_check(exe: str, case: dict, d: Path) -> dict:
    import scan_period as SP
    try:
        return SP.lean_check(exe, case, d)
    except (SystemExit, OSError, ValueError) as e:
        raise NotRun(f"network-period check: {type(e).__name__}: {e}") from None


def case_of(warden: Path, scan: Path) -> dict:
    import scan_period as SP
    try:
        return SP.rows_case(warden, scan)
    except (SystemExit, OSError, KeyError, ValueError) as e:
        raise NotRun(f"the period's rows case: {type(e).__name__}: {e}") from None


def publish(chain_out: Path, d: Path) -> None:
    for rel in ("report/report.json", "report/report.md", "job.json", "leak-guard.json", "certs.json",
                "endpoint/report.json", "scan/summary.json", "scan/answers.json", "n2/summary.json", "tie/summary.json"):
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


def guard(d: Path, unpacked: Path | None, quarantine: Path) -> int:
    """The record's leak guard on DIR against the warden run's witness (0 when nothing of it was found)."""
    if unpacked is None:
        return 0
    quarantine.mkdir(mode=0o700, parents=True, exist_ok=True)
    got = subprocess.run([sys.executable, str(HERE / "leak_guard.py"), str(d), str(quarantine), "--gates",
                          str(unpacked), "--report", str(quarantine / "piece-guard.json")],
                         capture_output=True, text=True)
    print(f"leak guard (exit {got.returncode}): {got.stdout.strip()[-400:]}", flush=True)
    return got.returncode


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
    raise NotRun(f"{signal.Signals(signum).name}: the piece's time limit (CERTIFIER_TIMEOUT) ended it before a verdict")


def piece(a: argparse.Namespace) -> int:
    serve_dir, d = a.serve_dir.resolve(), a.dir.resolve()
    d.mkdir(parents=True, exist_ok=True)
    for kv in a.set:
        if not kv.startswith("CR_") or "=" not in kv:
            print(f"refused: {kv!r}: CR_K=V, a chain_record.sh variable", file=sys.stderr)
            return 2
    was, unpacked = signal.signal(signal.SIGTERM, ended), None
    try:
        if not any((serve_dir / PRIVATE).glob("*/fetch.tar.gz")):
            raise NotRun(f"no warden run kept under {serve_dir / PRIVATE}: the run's load didn't go through the "
                         "certifier's front (CERTIFIER_FRONT unset, or the gate, its tunnel or ssh failed)")
        exe = period_exe(a.lean)
        chain = lambda keep, run, ep, n: run_chain(keep, run, ep, a.eos_ids, a.draw, sized(n, a.set),  # noqa: E731
                                                   a.chain_cmd)
        got, unpacked = decide(serve_dir, a.window, d, chain=chain, check=lambda c, p: period_check(exe, c, p))
        (d / "answer.json").write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
        g = guard(d, unpacked, serve_dir / PRIVATE / "quarantine")
        if g not in (0, 3) or not (d / "answer.json").is_file():
            raise NotRun(f"the leak guard failed (exit {g}): DIR withheld")
    except NotRun as e:
        return refuse(d, str(e))
    except (OSError, ValueError, tarfile.TarError) as e:
        traceback.print_exc()
        return refuse(d, f"the piece failed on a file: {type(e).__name__}: {e}")
    finally:
        if unpacked is not None:
            rm(unpacked)
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
    f.add_argument("--warmup", type=int, default=0, metavar="N",
                   help="plain requests of the group's shape straight to PORT first, each in a window of its own")
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
