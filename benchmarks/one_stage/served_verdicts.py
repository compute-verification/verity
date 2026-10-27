"""The verdicts on one served window (`served_verdicts.sh`): the window it audits, the subprotocols after
sampled proofs, and the one record of every verdict (the block record).

    served_verdicts.py window SERVE_DIR [--slots N] [--window K]
    served_verdicts.py gate [--gate CMD] [--lane vllm] [--main FILE]
    served_verdicts.py pieces --serve-run DIR --window K --out DIR [--node NAME]
    served_verdicts.py pouw SERVE_DIR WINDOW DIR [--gate CMD] [--key-sha512 HEX] [--main FILE] [--python PY]
    served_verdicts.py record --serve-run DIR --window K --sampled-proofs DIR --pieces DIR --out DIR [--node NAME]

``window``: the sealed window to audit, from a serve-openai run's ``seals.jsonl``, ``windows.jsonl`` and ``requests.jsonl``
(`pick`): sealed without error, its Commit passed, each of its requests checked and in no other window (V3b's window
circuit refuses a request carried across windows), at most ``--slots`` of them (its class's); the most requests, then the
earliest. ``--window K`` holds K to the same rules. Prints ``{window, requests, users, steps, refused}``, ``steps`` the
window's ``[lo, hi)`` in the run's committed engine steps.

``pieces``: each subprotocol after sampled proofs (`PIECES`: PoUW, the bandwidth audit, PoUS, the network certifier) runs
its piece where one is named (``POUW_CMD``, ``BANDWIDTH_CMD``, ``POUS_CMD``, ``CERTIFIER_CMD``: a command run in the tree as
``bash -c CMD _ SERVE_DIR WINDOW DIR``, which writes its core answer to ``DIR/answer.json``), one at a time, each within
``<COMPONENT>_TIMEOUT`` (else ``PIECE_TIMEOUT``, default `PIECE_TIMEOUT`), else prints a ``MISSING`` line naming it, the
command it lacks and its owner. Each run leaves ``DIR/piece.json`` (`RAN`: the node it ran on, its exit, its limit). A piece
that ran but wrote no answer is its error, unless it refused to run (``DIR/refused.txt``, `REFUSED`, naming why), which
leaves it missing; no verdict is made for a piece that didn't run. A development piece writes ``DIR/development.json``
(`DEVELOPMENT`) in place of a core answer: its verdict is printed as a development run's, and it credits nothing. A piece
that audited another server than the node it ran on says so in ``DIR/served.json`` (`SERVED`: ``{server, source}``,
``source`` ``fixture`` for an answer brought from a recorded run).

``gate``: before serving with H1's coins from infra's gate (``--gate``, default ``H1_GATE``): refused while the key's lane
(``--lane``) has a period open, which a reserve would join; prints one ``GATE`` line with the gate's latest pin and,
with ``--main``, whether that pin is on the verifier's first-parent line of main (a session pinned off it is refused).

``pouw``: PoUW's development piece (``POUW_CMD``): window K's H1 outcome (``windows.jsonl``'s ``pouw_h1``) and record
(``h1/w<K>/pouw-h1-calls.json``); with the auditor host's challenger's coins, the window's session as the gate's ``fetch``
gives it, the auditor's ledger and pins log as the gate answers now and main's first-parent history (``--main``),
checked by `pouw_h1_gpu.check_challenger` under the key the deployment pins (``--key-sha512``). Its verdict is accept when
H1's tables are tied to the window's Commit, its coins are the session the ledger opened and ended for it, and its record
is one manifest under them; core's credit refuses the window and is shown beside it (`DEV_WHY`).

``record``: sampled proofs' answer from the records audit `prove_path/run.sh` (``PP_RECORDS=1``) kept under
``--sampled-proofs`` (its ``NAME`` dir: ``summary.json``, ``window/audit.json``, ``windows/rank0/registration.json``), read
as `served_integrity.answer` reads it; each other subprotocol's from
``--pieces/<component>/answer.json`` (`piece_row`), else a ``missing`` row naming why. Each answer is about the server
`piece_server` names: sampled proofs' and a piece's that ran here are the record's node's, the wipe's and any other's the
one its ``served.json`` names. They go into the window report (``verdicts.json``, `row_of`'s rows, with each one's server)
and from it the block record (``block-record.json``, `block_record.from_period`, held by `block_record.check`), which
composes only the answers about the record's node and carries the others as ``elsewhere``; an answer that names no server
isn't carried. An accepted answer whose outputs are not its leg's certificate (`block_record.leg`) is its subprotocol's
error. `block_record` carries only an accepted subprotocol's certificate, so where one about this node rejects or errs the
composed verdict is ``reject`` and no block record is written; it is ``accept`` only when every subprotocol accepted here.
Prints one ``RECORD`` line, naming each missing subprotocol and why, each rejecting one, and each one elsewhere.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import tarfile
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _p in (Path(__file__).resolve().parent, ROOT / "examples" / "h200" / "interfaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

#: the window report's format, kept from the all-protocols runner whose report it was
REPORT_FORMAT = "verity/all-protocols-window/v1"
#: sampled proofs' owner, its row's
INTEGRITY_OWNER = "compute-accounting"

#: each subprotocol after sampled proofs, by its runner name: (its name, its component, its verifier's owner, what its piece
#: does), in the chain's order
PIECES: dict[str, tuple[str, str, str, str]] = {
    "work": ("PoUW", "pouw", "compute-accounting",
             "H1's credit on the served window, its coins from vy-challenger-1's gate (H1_GATE)"),
    "bandwidth": ("the bandwidth audit", "bandwidth", "memory-accounting",
                  "the remote bandwidth audit of the served window's store, against the bandwidth verifier on vy-auditor-2"),
    "space": ("PoUS", "pous", "memory-accounting", "PoUS over the served window's int8 weights"),
    "communication": ("the network certifier", "certifier", "network-accounting",
                      "the network certifier in front of the serving endpoint"),
}
#: the record's subprotocol outside this chain: serve() runs its piece (WIPE_CMD) on the card before its engine loads
WIPE = ("the wipe", "wipe", "memory-accounting", "the wipe of the serving card before its engine loads")
NAMES = {"integrity": "sampled proofs", **{k: v[0] for k, v in PIECES.items()}, "wipe": WIPE[0]}
#: a piece's limit unless ``<COMPONENT>_TIMEOUT`` or ``PIECE_TIMEOUT`` sets one, as `timeout` reads it
PIECE_TIMEOUT = "60m"
#: what `pieces` leaves beside a piece it ran: the node it ran on, its exit and its limit
RAN = "piece.json"
#: a piece's refusal to run, naming why: it leaves its subprotocol missing
REFUSED = "refused.txt"
#: the server a piece's answer is about, where it isn't the node `pieces` ran it on (the wipe's always)
SERVED = "served.json"
#: a development piece's document, written in place of its core answer
DEVELOPMENT = "development.json"
#: why a development piece's verdict credits nothing, so its subprotocol's row in the record stays missing
DEV_WHY = ("a development run: H1's coins come from vy-challenger-1's gate under the lane's development key, and a window "
           "counts for credit only under the lane's own key; core's credit also needs the window in a period whose open "
           "commits its draw K, rows and weights")
#: H1's record of a window (`pouw_h1.CALLS_FILE`) under the serve dir
H1_CALLS = "h1/w{k}/pouw-h1-calls.json"


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def _refusal(k: int, seal: Mapping[str, Any], line: Mapping[str, Any] | None, mine: Sequence[Mapping[str, Any]],
             slots: int) -> str | None:
    """Why window ``k`` can't be audited, None when it can."""
    if "error" in seal:
        return f"its seal failed: {seal['error']}"[:300]
    if not line or not line.get("passed"):
        return "its Commit did not pass"
    if not mine:
        return "no request"
    carried = sorted(r.get("user") or r.get("engine_id") for r in mine if r.get("windows") != [k])
    if carried:
        return f"requests carried across windows: {carried}"
    unchecked = sorted(r.get("user") or r.get("engine_id") for r in mine if not (r.get("check") or {}).get("passed"))
    if unchecked:
        return f"requests whose check failed: {unchecked}"
    if len(seal.get("requests") or ()) != len(mine):
        return f"{len(seal.get('requests') or ())} sealed requests, {len(mine)} with a record"
    if len(mine) > slots:
        return f"{len(mine)} requests, more than the class's {slots} slots"
    return None


def pick(serve_dir: Path, slots: int, want: int | None = None) -> dict[str, Any]:
    """The window to audit (module docstring); ValueError when none qualifies, or ``want`` doesn't."""
    seals = {int(s["window"]): s for s in _lines(serve_dir / "seals.jsonl")}
    lines = {int(w["window"]): w for w in _lines(serve_dir / "windows.jsonl")}
    reqs = _lines(serve_dir / "requests.jsonl")
    refused: dict[int, str] = {}
    ok: list[tuple[int, list[dict[str, Any]]]] = []
    for k in sorted(seals):
        mine = [r for r in reqs if k in (r.get("windows") or ())]
        why = _refusal(k, seals[k], lines.get(k), mine, slots)
        if why is None:
            ok.append((k, mine))
        else:
            refused[k] = why
    if want is not None:
        if want not in seals:
            raise ValueError(f"window {want} has no seal (sealed: {sorted(seals)})")
        if want in refused:
            raise ValueError(f"window {want}: {refused[want]}")
        ok = [x for x in ok if x[0] == want]
    if not ok:
        raise ValueError(f"no sealed window to audit: {json.dumps(refused, sort_keys=True)}")
    k, mine = max(ok, key=lambda x: (len(x[1]), -x[0]))
    lo = sum(int((w.get("commit") or {}).get("steps") or 0) for j, w in lines.items() if j < k)
    hi = lo + int((lines[k].get("commit") or {}).get("steps") or 0)
    return {"window": k, "requests": len(mine), "users": sorted(r.get("user") or "" for r in mine), "steps": [lo, hi],
            "refused": {str(j): why for j, why in refused.items()}}


def _unset(comp: str, what: str) -> str:
    return f"{comp.upper()}_CMD unset: {what}"


def pieces(serve_dir: Path, window: int, out: Path, env: Mapping[str, str] | None = None,
           node: str | None = None) -> dict[str, dict[str, Any]]:
    """Each piece of `PIECES` run where ``<COMPONENT>_CMD`` names it, under `timeout` at its limit, else named missing;
    one line each on stdout, its verdict the one `piece_row` gives the record."""
    env = dict(os.environ if env is None else env)
    node = node or socket.gethostname()
    got: dict[str, dict[str, Any]] = {}
    for key, (name, comp, owner, what) in PIECES.items():
        cmd = env.get(f"{comp.upper()}_CMD")
        if not cmd:
            got[comp] = {"verdict": "missing", "why": _unset(comp, what)}
            print(f"MISSING {name} ({comp.upper()}_CMD unset; {owner}): {what}", flush=True)
            continue
        d = out / comp
        d.mkdir(parents=True, exist_ok=True)
        limit = env.get(f"{comp.upper()}_TIMEOUT") or env.get("PIECE_TIMEOUT") or PIECE_TIMEOUT
        with open(d / "piece.log", "w") as log:
            rc = subprocess.run(["timeout", "-k", "1m", limit, "bash", "-c", cmd, "_", str(serve_dir), str(window),
                                 str(d)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, check=False).returncode
        (d / RAN).write_text(json.dumps({"server": node, "rc": rc, "limit": limit}, sort_keys=True) + "\n")
        label, row = name.upper(), piece_row(key, out)
        if "development" in row:
            got[comp], label = {**development_line(row["development"]), "rc": rc}, f"{label} (development run, credits nothing)"
        else:
            got[comp] = {**{k: row[k] for k in ("verdict", "reason", "why") if row.get(k) is not None}, "rc": rc}
        print(f"{label} {json.dumps(got[comp], sort_keys=True)}", flush=True)
    return got


def development_line(doc: Mapping[str, Any]) -> dict[str, Any]:
    """A development piece's line (`pouw`'s document): its verdict, why it rejects, the window's session and how each
    check came out, core's credit refusal, and that it credits nothing."""
    check, credit = doc.get("check") or {}, doc.get("credit") or {}
    return {"verdict": doc.get("verdict"), "development": True, "credited": False, "reasons": doc.get("reasons"),
            "coin_source": doc.get("coin_source"), "session": (doc.get("session") or {}).get("session"),
            "tied": doc.get("tied"), "coins": check.get("reason", "unchecked"), "manifest": check.get("manifest", "unchecked"),
            "gate_passed": doc.get("gate_passed"), "credit": credit.get("window")}


def _gate_out(gate: str, *words: str) -> bytes:
    """The gate's answer to `words` through its command `gate`, the words one shell-quoted line as `commit_coins.Gate`
    sends them; RuntimeError naming what it said when it refuses."""
    p = subprocess.run([*shlex.split(gate), shlex.join(words)], capture_output=True, check=False)
    if p.returncode:
        said = (p.stdout + p.stderr).decode(errors="replace").strip()
        raise RuntimeError(f"the gate refused {words[0]} (exit {p.returncode}): {said[-400:]}")
    return p.stdout


def gate_ready(gate: str, lane: str, main: Path | None = None) -> tuple[bool, str]:
    """Whether this run may reserve its windows' sessions at `gate` (the ``gate`` subcommand), and its ``GATE`` line."""
    from verity_vllm.protocols.pouw.h1 import window as PHW

    ledger = _gate_out(gate, "ledger").decode().splitlines()
    pins = [json.loads(x) for x in _gate_out(gate, "pins").decode().splitlines() if x.strip()]
    pin, period = (pins[-1] if pins else {}), PHW.open_period(ledger, lane)
    on = None if main is None else pin.get("commit") in main.read_text().split()
    where = "" if on is None else f", {'on' if on else 'NOT on'} the verifier's first-parent line of main"
    line = (f"GATE {'refused' if period else 'ready'}: lane {lane}'s open period "
            f"{period and period.get('period')}{' (a reserve would join it)' if period else ''}; the ledger's "
            f"{len(ledger)} entries; pinned {str(pin.get('commit'))[:12]} (main {str(pin.get('main'))[:12]}, "
            f"{pin.get('pinned_utc')}){where}")
    return period is None, line


def check_challenger(calls: Path, tree: Path, ledger: Path, pins: Path, main: Path | None, key: str, out: Path,
                     python: str) -> dict[str, Any]:
    """`pouw_h1_gpu.py --check-challenger` (run by `python` in integrations/vllm, so ``tests`` is that package's) of the
    window's record `calls` against its session's tree `tree`; its ``pouw_h1_challenger.json``."""
    out.mkdir(parents=True, exist_ok=True)
    argv = [python, "-m", "tests.protocols.pouw_h1_gpu", "--check-challenger", str(calls), str(tree), "--key-sha512", key,
            "--ledger", str(ledger), "--pins", str(pins), *(["--main", str(main)] if main else []), "--out", str(out)]
    p = subprocess.run(argv, cwd=ROOT / "integrations" / "vllm", capture_output=True, text=True, check=False)
    (out / "check.log").write_text(p.stdout + p.stderr)
    try:
        return json.loads((out / "pouw_h1_challenger.json").read_text())
    except (OSError, ValueError) as e:
        raise RuntimeError(f"pouw_h1_gpu --check-challenger wrote no verdict (exit {p.returncode}): "
                           f"{(p.stdout + p.stderr).strip()[-400:]}") from e


Check = Callable[[Path, Path, Path, Path, "Path | None", str, Path], Mapping[str, Any]]


def pouw(serve_dir: Path, window: int, out: Path, *, gate: str | None, key: str | None, main: Path | None,
         check: Check) -> dict[str, Any]:
    """PoUW's development piece on window `window` (module docstring), its document written to ``out/development.json``;
    `check` is `check_challenger` with its Python bound."""
    out.mkdir(parents=True, exist_ok=True)
    line = next((w for w in _lines(serve_dir / "windows.jsonl") if int(w["window"]) == window), {})
    h1 = line.get("pouw_h1") or {}
    calls = serve_dir / H1_CALLS.format(k=window)
    doc: dict[str, Any] = {"window": window, "credited": False, "why": DEV_WHY, "coin_source": h1.get("coin_source"),
                           "tied": h1.get("tied"), "steps_served": h1.get("steps_served"), "session": h1.get("session"),
                           "gate_passed": (h1.get("gate") or {}).get("gate_passed"), "check": None, "credit": None}
    reasons = []
    if not h1 or "error" in h1:
        reasons.append(f"H1's window: {h1.get('error') or 'its line has no pouw_h1 (served without --protocols)'}")
    elif h1.get("tied") is not True:
        reasons.append(f"ROOTS1: H1's tables are not tied to the window's Commit ({(h1.get('roots1') or {}).get('reason')})")
    if h1 and "error" not in h1 and h1.get("coin_source") != "challenger":
        reasons.append(f"coins: {h1.get('coin_source')}, not the auditor host's challenger's, so nothing checks them")
    elif h1.get("coin_source") == "challenger":
        try:
            if not (gate and key):
                raise ValueError("no gate or no key pin to check the window's session by (H1_GATE, H1_KEY_SHA512)")
            sid = json.loads(calls.read_text())["commitment"]["session"]["session"]
            tgz, tree = out / "session.tgz", out / "session"
            tgz.write_bytes(_gate_out(gate, "fetch", sid))
            with tarfile.open(tgz) as t:
                t.extractall(tree, filter="data")
            (out / "ledger.jsonl").write_bytes(_gate_out(gate, "ledger"))
            (out / "pins.jsonl").write_bytes(_gate_out(gate, "pins"))
            got = check(calls, tree / sid, out / "ledger.jsonl", out / "pins.jsonl", main, key, out / "check")
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, tarfile.TarError) as e:
            reasons.append(f"coins: {type(e).__name__}: {e}"[:600])
        else:
            doc["check"] = {k: got.get(k) for k in ("ok", "reason", "manifest", "rounds", "coin_source")}
            doc["credit"] = {k: (got.get("credit") or {}).get(k) for k in ("window", "manifest", "k", "drawn", "eps")}
            reasons += [f"{k}: {got[r]}" for k, r in (("coins", "reason"), ("manifest", "manifest")) if got.get(r)]
    doc.update(verdict="reject" if reasons else "accept", reasons=reasons)
    (out / DEVELOPMENT).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    return doc


def check_answer(a: Any) -> dict[str, Any]:
    """``a`` when it is an answer in the shapes `Answer.ofJson` reads (`pouw.core_check.answer_of`, the Python mirror of
    the core interface's decoder): an accept has no reason and both its public outputs and its profile, a refusal a
    reason and neither. Anything else is a ValueError, which the row reports as the subprotocol's error."""
    from verity.core.protocols.pouw import core_check as CC

    try:
        CC.answer_of(a)
    except CC.Refused as e:
        raise ValueError(f"not a core answer: {e}") from None
    return dict(a)


def epsilon(a: Mapping[str, Any]) -> str | None:
    """A sampled protocol's bound on wrong units (``public.bound``) over its checked level's size (`Core.Integrity`), or
    None for an answer that has no such bound."""
    public, profile = a.get("public"), a.get("profile")
    levels = profile.get("levels") if isinstance(profile, Mapping) else None
    if not isinstance(public, Mapping) or not isinstance(levels, list) or not levels or not isinstance(levels[-1], Mapping):
        return None
    bound, size = public.get("bound"), levels[-1].get("size")
    return str(Fraction(bound, size)) if isinstance(bound, int) and isinstance(size, int) and size > 0 else None


def row_of(owner: str, a: Any, out: Path, run: str | None = None) -> dict[str, Any]:
    """A subprotocol's report row from its answer (checked to be a core answer first); ``run`` is the recording run."""
    a = check_answer(a)
    runs = json.loads((out / "runs.json").read_text()) if (out / "runs.json").exists() else []
    accepted = a["verdict"] == "accept"
    return {"protocol": a["protocol"], "owner": owner, "verdict": a["verdict"], "reason": a["reason"],
            "commitment": a["commitment"], "quantity": a["public"] if accepted else None,
            "delta": (a["profile"] or {}).get("delta") if isinstance(a["profile"], dict) else None,
            "epsilon": epsilon(a) if accepted else None, "runs": [*([run] if run else []), *runs], "answer": a}


def sampled_proofs_row(audit_dir: Path, run: str | None = None) -> dict[str, Any]:
    """Sampled proofs' row from the records audit kept under ``audit_dir`` (module docstring)."""
    import served_integrity as SI

    reg = audit_dir / "windows" / "rank0" / "registration.json"
    try:
        a = SI.answer(audit_dir, json.loads(reg.read_text()) if reg.exists() else None)
        return row_of(INTEGRITY_OWNER, a, audit_dir, run)
    except (RuntimeError, ValueError) as e:
        return {"verdict": "error", "owner": INTEGRITY_OWNER, "reason": f"{type(e).__name__}: {e}"[:2000]}


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def piece_row(key: str, pieces_dir: Path, run: str | None = None) -> dict[str, Any]:
    """A subprotocol's runner row from its piece's answer; a development piece's is ``missing`` with its document when it
    accepts (it credits nothing), else ``reject``; a piece that ran (`RAN`) and wrote neither is its ``error``, unless it
    refused to run (`REFUSED`); otherwise ``missing``, naming why."""
    _name, comp, owner, what = PIECES.get(key, WIPE)
    d = pieces_dir / comp
    f, dev, ran = d / "answer.json", d / DEVELOPMENT, _json(d / RAN)
    if not f.exists() and dev.exists():
        doc = _json(dev)
        if not isinstance(doc, dict):
            return {"verdict": "error", "owner": owner, "reason": f"its piece's {DEVELOPMENT} is not a JSON object"}
        if doc.get("verdict") == "accept":
            return {"verdict": "missing", "owner": owner, "why": DEV_WHY, "development": doc}
        return {"verdict": "reject", "owner": owner, "reason": "; ".join(doc.get("reasons") or ())[:2000],
                "development": doc}
    if not f.exists():
        if (d / REFUSED).exists():
            said = (d / REFUSED).read_text(errors="replace").strip()
            return {"verdict": "missing", "owner": owner, "why": f"its piece refused to run: {said}"[:2000]}
        if isinstance(ran, dict):
            late = f", out of time at {ran.get('limit')}" if ran.get("rc") in (124, 137) else ""
            return {"verdict": "error", "owner": owner,
                    "reason": f"its piece ran (exit {ran.get('rc')}{late}) and wrote no answer; its log is {comp}/piece.log"}
        if key == "wipe" and (d / "piece.log").exists():
            return {"verdict": "missing", "owner": owner, "why": f"its piece wrote no answer; its log is {comp}/piece.log"}
        return {"verdict": "missing", "owner": owner, "why": _unset(comp, what)}
    try:
        return row_of(owner, json.loads(f.read_text()), f.parent, run)
    except ValueError as e:
        return {"verdict": "error", "owner": owner, "reason": f"{type(e).__name__}: {e}"[:2000]}


def piece_server(key: str, pieces_dir: Path) -> tuple[str | None, str]:
    """The server a piece's answer is about and the answer's source (``real`` or ``fixture``): its `SERVED` record's, else,
    but for the wipe, the node `pieces` ran it on (`RAN`); None where neither names one."""
    comp = PIECES.get(key, WIPE)[1]
    served, ran = _json(pieces_dir / comp / SERVED), _json(pieces_dir / comp / RAN)
    at = served if isinstance(served, dict) else ran if isinstance(ran, dict) and key != "wipe" else {}
    return at.get("server"), (served.get("source") if isinstance(served, dict) else None) or "real"


def record(rows: Mapping[str, Mapping[str, Any]], *, window: int, steps: tuple[int, int], node: str,
           run: str | None, servers: Mapping[str, tuple[str | None, str]] | None = None
           ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """The window report over ``rows`` (by runner name), its block record (None when a subprotocol about this node
    rejected, erred or was undecided), and the ``RECORD`` line's summary. ``servers``: each piece's `piece_server`, by
    runner name; sampled proofs' answer is about ``node``, and a piece's that names no server isn't carried."""
    from verity_e2e_demo import block_record as BR

    legs = {v: k for k, v in BR.RUNNER_KEYS.items()} | {"wipe": "wipe"}
    at = {k: (None, "real") for k in rows} | dict(servers or {}) | {"integrity": (node, "real")}
    rows = {k: dict(r) for k, r in rows.items()}
    for k, r in rows.items():
        if r.get("verdict") == "accept" and at[k][0] is not None:
            try:
                BR.leg(legs[k], r, server=at[k][0], runs=[], source=at[k][1])
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as e:
                rows[k] = {**r, "verdict": "error", "quantity": None, "epsilon": None,
                           "reason": f"its answer is not the block record's {legs[k]} leg: {type(e).__name__}: {e}"[:2000]}
    report = {"format": REPORT_FORMAT, "window": window, "node": node, "run": run, "protocols": rows,
              "servers": {k: at[k][0] for k in rows}}
    verdicts = {NAMES[k]: r.get("verdict") for k, r in rows.items()}
    missing = {NAMES[k]: r.get("why") for k, r in rows.items() if r.get("verdict") == "missing"}
    elsewhere = {NAMES[k]: at[k][0] for k in rows if at[k][0] not in (None, node)}
    unserved = sorted(NAMES[k] for k, r in rows.items() if r.get("verdict") != "missing" and at[k][0] is None)
    dev = {"development": {NAMES[k]: r["development"]["verdict"] for k, r in rows.items() if "development" in r}}
    dev = dev if dev["development"] else {}
    named = {"verdicts": verdicts, "missing": missing, "elsewhere": elsewhere, **({"unserved": unserved} if unserved else {}),
             **dev}
    bad = sorted(NAMES[k] for k, r in rows.items()
                 if r.get("verdict") not in ("accept", "missing") and NAMES[k] not in elsewhere)
    if bad:
        composed = "reject" if any(rows[k].get("verdict") in ("reject", "error") for k in rows
                                   if NAMES[k] in bad) else "pending"
        line = {"composed": composed, "by": bad, **named, "block_record": None,
                "why": "block_record carries an accepted subprotocol's certificate only"}
        return report, None, line
    extra = {legs[k]: {"row": r, "server": at[k][0], "runs": r.get("runs") or [run], "source": at[k][1]}
             for k, r in rows.items() if k != "integrity" and r.get("verdict") == "accept" and at[k][0] is not None}
    composing = {**report, "protocols": {k: r if k == "integrity" else {"verdict": "missing"} for k, r in rows.items()}}
    rec = BR.from_period(composing, steps=steps, deployment=BR.REFERENCE, extra=extra)
    line = {"composed": rec["composed"]["verdict"], **named, "record_missing": rec["composed"]["missing"],
            "check": BR.check(rec), "steps": list(steps), "node": node,
            "deployment": rec["monitoring_period"]["deployment"]["id"]}
    return report, rec, line


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("window")
    w.add_argument("serve_dir", type=Path)
    w.add_argument("--slots", type=int, default=8)
    w.add_argument("--window", type=int)
    g = sub.add_parser("gate")
    g.add_argument("--gate", default=os.environ.get("H1_GATE"))
    g.add_argument("--lane", default="vllm")
    g.add_argument("--main", type=Path, default=os.environ.get("MAIN_FILE") or None)
    o = sub.add_parser("pouw")
    o.add_argument("serve_dir", type=Path)
    o.add_argument("window", type=int)
    o.add_argument("out", type=Path)
    o.add_argument("--gate", default=os.environ.get("H1_GATE"))
    o.add_argument("--key-sha512", default=os.environ.get("H1_KEY_SHA512"))
    o.add_argument("--main", type=Path, default=os.environ.get("MAIN_FILE") or None)
    o.add_argument("--python", default=os.environ.get("PY") or sys.executable)
    p = sub.add_parser("pieces")
    r = sub.add_parser("record")
    for q in (p, r):
        q.add_argument("--serve-run", type=Path, required=True)
        q.add_argument("--window", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    r.add_argument("--sampled-proofs", type=Path, required=True)
    r.add_argument("--pieces", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    for q in (p, r):
        q.add_argument("--node", default=socket.gethostname())
    a = ap.parse_args(argv)
    if a.cmd == "window":
        try:
            print(json.dumps(pick(a.serve_dir, a.slots, a.window), sort_keys=True))
        except ValueError as e:
            print(f"served_verdicts.py window: {e}", file=sys.stderr)
            return 1
        return 0
    if a.cmd == "gate":
        if not a.gate:
            print("GATE refused: no gate command (--gate, H1_GATE)", flush=True)
            return 1
        try:
            ok, line = gate_ready(a.gate, a.lane, a.main)
        except (OSError, RuntimeError, ValueError) as e:
            ok, line = False, f"GATE refused: {type(e).__name__}: {e}"[:800]
        print(line, flush=True)
        return 0 if ok else 1
    if a.cmd == "pouw":
        doc = pouw(a.serve_dir, a.window, a.out, gate=a.gate, key=a.key_sha512, main=a.main,
                   check=lambda *x: check_challenger(*x, python=a.python))
        print(f"POUW-DEV {json.dumps(development_line(doc), sort_keys=True, default=str)}", flush=True)
        return 0
    if a.cmd == "pieces":
        a.out.mkdir(parents=True, exist_ok=True)
        got = pieces(a.serve_run, a.window, a.out, node=a.node)
        return 0 if all(g["verdict"] in ("accept", "missing") for g in got.values()) else 1
    steps = pick(a.serve_run, 1 << 30, a.window)["steps"]
    run = os.environ.get("RESEARCH_RUN_ID")
    keys = [*PIECES, "wipe"]
    rows = {"integrity": sampled_proofs_row(a.sampled_proofs, run), **{k: piece_row(k, a.pieces, run) for k in keys}}
    report, rec, line = record(rows, window=a.window, steps=(steps[0], steps[1]), node=a.node, run=run,
                               servers={k: piece_server(k, a.pieces) for k in keys})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "verdicts.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    if rec is not None:
        (a.out / "block-record.json").write_text(json.dumps(rec, separators=(",", ":"), sort_keys=True) + "\n")
    print("RECORD " + json.dumps(line, sort_keys=True), flush=True)
    return 0 if rec is not None and not line["check"] else 1


if __name__ == "__main__":
    sys.exit(main())
