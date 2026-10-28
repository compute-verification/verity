"""The verdicts on one served window (`served_verdicts.sh`): the window it audits, the subprotocols after
sampled proofs, and the one record of every verdict (the block record).

    served_verdicts.py window SERVE_DIR [--slots N] [--window K]
    served_verdicts.py gate [--gate CMD] [--lane vllm] [--main FILE]
    served_verdicts.py pieces --serve-run DIR --window K --out DIR [--node NAME]
    served_verdicts.py pouw SERVE_DIR WINDOW DIR [--gate CMD] [--key-sha512 HEX] [--main FILE] [--python PY]
    served_verdicts.py record --serve-run DIR --window K --sampled-proofs DIR --pieces DIR --out DIR [--node NAME]
                                [--cases DIR]... [--onsite EXE] [--remote EXE]

``window``: the sealed window to audit, from a serve-openai run's ``seals.jsonl``, ``windows.jsonl`` and ``requests.jsonl``
(`pick`): sealed without error, its Commit passed, each of its requests checked and in no other window (V3b's window
circuit refuses a request carried across windows), at most ``--slots`` of them (its class's); the most requests, then the
earliest. ``--window K`` holds K to the same rules. Prints ``{window, requests, users, steps, refused}``, ``steps`` the
window's ``[lo, hi)`` in the run's committed engine steps.

``pieces``: each subprotocol after sampled proofs (`PIECES`: PoUW, the bandwidth audit, PoUS, the network certifier) runs
its piece where one is named (``POUW_CMD``, ``BANDWIDTH_CMD``, ``POUS_CMD``, ``CERTIFIER_CMD``: a command run in the tree as
``bash -c CMD _ SERVE_DIR WINDOW DIR``, which writes its core answer to ``DIR/answer.json``, the public one, and its case,
what its Lean component reads, with its own answer on it, to its private directory, ``SERVE_DIR/private/<component>``,
or to ``DIR`` when the case holds no witness), one at a time, each within ``<COMPONENT>_TIMEOUT`` (else
``PIECE_TIMEOUT``, default `PIECE_TIMEOUT`). Each run leaves ``DIR/piece.json`` (`RAN`: the node it ran on, its exit, its
limit). A piece that isn't named didn't run: its answer, written to ``DIR/answer.json``, is ``undecided`` with the reason
``not run`` (`NOT_RUN`), and a ``NOT RUN`` line names it, the command it lacks and its owner. So is one that refused to
run (``DIR/refused.txt``, `REFUSED`, naming why): only the protocol's answer rejects. A piece that ran and wrote neither
is an error, and undecided. A development piece writes ``DIR/development.json`` (`DEVELOPMENT`) in place of a core
answer: its verdict is printed as a development run's, and it credits nothing, so its answer is ``undecided`` when it
accepts. A piece that audited another server than the node it ran on says so in ``DIR/served.json`` (`SERVED`:
``{server, source}``, ``source`` ``fixture`` for an answer brought from a recorded run); one that names no server there
leaves its answer about the node it ran on.

``gate``: before serving with H1's coins from infra's gate (``--gate``, default ``H1_GATE``): refused while the key's lane
(``--lane``) has a period open, which a reserve would join; prints one ``GATE`` line with the gate's latest pin and,
with ``--main``, whether that pin is on the verifier's first-parent line of main (a session pinned off it is refused).

``pouw``: PoUW's development piece (``POUW_CMD``): window K's H1 outcome (``windows.jsonl``'s ``pouw_h1``) and record
(``h1/w<K>/pouw-h1-calls.json``); with the auditor host's challenger's coins, the window's session as the gate's ``fetch``
gives it, the auditor's ledger and pins log as the gate answers now and main's first-parent history (``--main``),
checked by `pouw_h1_gpu.check_challenger` under the key the deployment pins (``--key-sha512``). Its verdict is accept when
H1's tables are tied to the window's Commit, its coins are the session the ledger opened and ended for it, and its record
is one manifest under them; core's credit refuses the window and is shown beside it (`DEV_WHY`).

``record``: every subprotocol's core answer (`verity.core.protocols.answer`). Sampled proofs' is from the records audit
`prove_path/run.sh` (``PP_RECORDS=1``) kept under ``--sampled-proofs`` (its ``NAME`` dir: ``summary.json``,
``window/audit.json``, ``windows/rank0/registration.json``), read as `served_integrity.answer` reads it; each other
subprotocol's is ``--pieces/<component>/answer.json`` (`piece_row`), and ``undecided`` ``not run`` where its piece wrote
nothing or refused to run. Each answer is about the server `piece_server` names: sampled proofs' the record's node, a
piece's the one its ``served.json`` names, else the node it ran on, and the wipe's the one its ``served.json`` names.
They are the window report's rows (``verdicts.json``, `row_of`'s, with each one's server); an accepted answer whose
outputs are not its leg's certificate (`block_record.leg`) is its subprotocol's error, and undecided. Beside them the
report says each subprotocol's answer in one clause (`said`: why, unless it accepted here, and for an accept here what
it holds beyond its verdict, `terms`), which decides nothing.
The window's verdict is a Lean verdict, onsite's or remote's, and nothing else: each piece's case directory
(`case_dirs`: its private directory under the serve dir, else its public one; and each ``--cases``) that holds a core
answer and its case (``case.json``, as `composite.py` reads it) is that answer's protocol's case (`found`), and when the
cases make up onsite (`NEEDS`: its seven components, the inference-only policy's among them) `composite.py onsite` runs ``onsite check`` on
them, else when they make up remote `composite.py remote` runs ``remote check`` (`lean_verdict`). Its ``--private-out``,
the composed case, its evidence and every answer, is ``SERVE_DIR/private/composite/w<K>`` (0700), which no record
copies; its ``--public-out`` is held to composite.py's public form (`public_form`: verdicts, commitments and params
digests) before it is published to ``--out/composite``, and the window's verdict is Lean's public outcome there
(``{protocol, verdict, reason}``, the reason its refusal's name). Cases that make up neither leave the window
``undecided``, its reason naming what each product lacks and why.
The block record (``block-record.json``, `block_record.from_period`, held by `block_record.check`) is built from the
report, its ``composed.onsite`` the window's verdict, its legs the answers accepted about the record's node and the
others carried as ``elsewhere`` (an answer that names no server isn't carried), unless the window or a subprotocol's
answer about this node rejects (``rejected``: a rejection fails the run whether or not a product composed it). Prints
one ``RECORD`` line, naming each subprotocol's verdict (the development piece's `DEV_VERDICT`), each one not run and
why, each rejecting or erring one, each one elsewhere, and the window's verdict, and exits 1 when there is no block
record, a piece erred, the Lean check couldn't answer, or the block record doesn't hold.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity.core.protocols import answer as CA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "examples" / "h200" / "interfaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import composite as C  # noqa: E402

#: the window report's format: every row a core answer, and the window's verdict onsite's or remote's Lean answer
REPORT_FORMAT = "verity/all-protocols-window/v2"
#: sampled proofs' owner, its row's
INTEGRITY_OWNER = "compute-accounting"
#: each subprotocol's core protocol, by its runner name
PROTOCOLS = {"integrity": "verity/sampled-proofs/one-stage/v1", "work": "verity/pouw/v1",
             "bandwidth": "verity/remote-bandwidth/v0", "space": "verity/pous/p2/v4/run",
             "communication": "verity/certifier/v0", "wipe": "verity/wipe/v0"}
#: the pieces each product needs, in the order the window's verdict tries them (`composite.COMPONENTS`:
#: `Core.Onsite.components`, `Core.Remote.components`)
NEEDS = {p: C.COMPONENTS[p] for p in ("onsite", "remote")}
#: each product's executable (`lake build onsite`, `lake build remote` in its package)
LEAN = {p: ROOT / "verity" / "core" / "protocols" / p / ".lake" / "build" / "bin" / p for p in C.PRODUCTS}
#: an answer's file and its case's, in a piece's directory
ANSWER, CASE = "answer.json", "case.json"
#: the reason a subprotocol whose piece didn't run is undecided
NOT_RUN = "not run"
#: the serve dir's private directory (0700): each piece's case under its component's name, and the composite's
PRIVATE = "private"
#: composite.py's ``--public-out``: all the record publishes of the composite
PUBLIC_FILES = ("composite.json", "verdicts.txt")
#: the reason an accepting development piece is undecided
DEV_UNDECIDED = "a development run: it credits nothing"

#: each subprotocol after sampled proofs, by its runner name: (its name, its component, its verifier's owner, what its piece
#: does), in the chain's order
PIECES: dict[str, tuple[str, str, str, str]] = {
    "work": ("PoUW", "pouw", "compute-accounting",
             "H1's credit on the served window, its coins from vy-challenger-1's gate (H1_GATE)"),
    "bandwidth": ("the bandwidth audit", "bandwidth", "memory-accounting",
                  ("a remote bandwidth session over this node's registered store, against the auditor's verifier on "
                   "vy-auditor-2; served_verdicts.sh sets it where the tree has benchmarks/one_stage/served_bandwidth.sh "
                   "and this node the verifier's key and the store")),
    "space": ("PoUS", "pous", "memory-accounting",
              ("PoUS over the served model's int8 weights on the serving card, against the memory challenger on "
               "vy-auditor-2; served_verdicts.sh sets it where the tree has benchmarks/pous/served_piece.sh and this node "
               "the challenger's key")),
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
#: a piece's refusal to run, naming why (a rehearsal the gate's check didn't pass, a step that can't run): its piece
#: didn't run, so its subprotocol is undecided, "not run"; only the protocol's answer rejects
REFUSED = "refused.txt"
#: the server a piece's answer is about, where it isn't the node `pieces` ran it on (the wipe's always)
SERVED = "served.json"
#: a development piece's document, written in place of its core answer
DEVELOPMENT = "development.json"
#: why a development piece's verdict credits nothing, so its subprotocol's answer is undecided when it accepts
DEV_WHY = ("development run, credits nothing: H1's coins come from vy-challenger-1's gate under the lane's development key, "
           "and a window counts for credit only under the lane's own key; core's credit also needs the window in a period "
           "whose open commits its draw K, rows and weights")
#: a development piece's verdict in the ``RECORD`` line, in place of its row's ``undecided``
DEV_VERDICT = "development run, credits nothing"
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
    """Each piece of `PIECES` run where ``<COMPONENT>_CMD`` names it, under `timeout` at its limit, else answered
    ``undecided`` ``not run``; one line each on stdout, its verdict the one `piece` gives the record."""
    env = dict(os.environ if env is None else env)
    node = node or socket.gethostname()
    got: dict[str, dict[str, Any]] = {}
    for key, (name, comp, owner, what) in PIECES.items():
        d = out / comp
        d.mkdir(parents=True, exist_ok=True)
        cmd = env.get(f"{comp.upper()}_CMD")
        if not cmd:
            (d / ANSWER).write_text(CA.Answer.undecided(PROTOCOLS[key], NOT_RUN).canonical() + "\n")
            got[comp] = {"verdict": "undecided", "reason": NOT_RUN, "why": _unset(comp, what), "owner": owner}
            print(f"NOT RUN {name} ({comp.upper()}_CMD unset; {owner}): {what}", flush=True)
            continue
        limit = env.get(f"{comp.upper()}_TIMEOUT") or env.get("PIECE_TIMEOUT") or PIECE_TIMEOUT
        with open(d / "piece.log", "w") as log:
            rc = subprocess.run(["timeout", "-k", "1m", limit, "bash", "-c", cmd, "_", str(serve_dir), str(window),
                                 str(d)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, check=False).returncode
        (d / RAN).write_text(json.dumps({"server": node, "rc": rc, "limit": limit}, sort_keys=True) + "\n")
        label, (row, _) = name.upper(), piece(key, out)
        if "development" in row:
            got[comp], label = {**development_line(row["development"]), "rc": rc}, f"{label} (development run, credits nothing)"
        else:
            got[comp] = {**{k: row[k] for k in ("verdict", "reason", "why", "error") if row.get(k) is not None},
                         "rc": rc}
            if (row["verdict"], row.get("reason")) == ("undecided", NOT_RUN):
                label = f"NOT RUN {name}"
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


def row_of(owner: str, a: Any, out: Path | None, run: str | None = None) -> dict[str, Any]:
    """A subprotocol's report row from its answer (checked to be a core answer first); ``run`` is the recording run."""
    a = check_answer(a)
    runs = json.loads((out / "runs.json").read_text()) if out is not None and (out / "runs.json").exists() else []
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
        refused = CA.Answer.reject(PROTOCOLS["integrity"], f"{type(e).__name__}: {e}"[:2000])
        return row_of(INTEGRITY_OWNER, refused.to_json(), None, run)


def _json(path: Path) -> Any:
    """``path``'s JSON, None where there's no such file; one that isn't JSON raises ValueError."""
    return json.loads(path.read_text()) if path.exists() else None


def _undecided(key: str, reason: str, d: Path | None, run: str | None, **kw: Any) -> dict[str, Any]:
    """Subprotocol ``key``'s row on an ``undecided`` answer whose reason is ``reason``, with ``kw`` beside it."""
    a = CA.Answer.undecided(PROTOCOLS[key], reason[:2000]).to_json()
    return {**row_of(PIECES.get(key, WIPE)[2], a, d, run), **kw}


def piece(key: str, pieces_dir: Path, run: str | None = None) -> tuple[dict[str, Any], tuple[str | None, str]]:
    """A subprotocol's `piece_row` and `piece_server`; a piece that left a file that isn't JSON is its ``error``, and
    undecided, about no server."""
    try:
        return piece_row(key, pieces_dir, run), piece_server(key, pieces_dir)
    except ValueError as e:
        return (_undecided(key, f"its piece left a file that isn't JSON: {type(e).__name__}: {e}", None, run, error=True),
                (None, "real"))


def piece_row(key: str, pieces_dir: Path, run: str | None = None) -> dict[str, Any]:
    """A subprotocol's runner row from its piece's answer: ``undecided`` ``not run``, naming why, where the piece wrote
    nothing and didn't run, or refused to run (`REFUSED`); ``undecided`` and an ``error`` where it ran (`RAN`) and wrote
    no answer, or one that isn't a core answer, since only the protocol's answer rejects; a development piece's is
    ``undecided`` with its document when it accepts (it credits nothing), else ``reject``."""
    _name, comp, owner, what = PIECES.get(key, WIPE)
    pid, d = PROTOCOLS[key], pieces_dir / comp
    f, dev, ran = d / ANSWER, d / DEVELOPMENT, _json(d / RAN)
    if not f.exists() and dev.exists():
        doc = _json(dev)
        if not isinstance(doc, dict):
            return _undecided(key, f"its piece's {DEVELOPMENT} is not a JSON object", d, run, error=True)
        if doc.get("verdict") == "accept":
            refused = (doc.get("credit") or {}).get("window")
            if isinstance(refused, list) and len(refused) == 2:
                refused = f"({refused[0]}) {refused[1]}"
            why = DEV_WHY if refused is None else (f"{DEV_WHY}; core's credit refuses the window "
                                                   f"{refused if isinstance(refused, str) else json.dumps(refused)}")
            return _undecided(key, DEV_UNDECIDED, d, run, why=why[:2000], development=doc)
        why = "; ".join(doc.get("reasons") or ())[:2000] or "a development run that rejected"
        return {**row_of(owner, CA.Answer.reject(pid, why).to_json(), d, run), "development": doc}
    if not f.exists():
        if (d / REFUSED).exists():
            said = (d / REFUSED).read_text(errors="replace").strip()
            return _undecided(key, NOT_RUN, d, run, why=f"its piece refused to run: {said}"[:2000])
        if isinstance(ran, dict):
            late = f", out of time at {ran.get('limit')}" if ran.get("rc") in (124, 137) else ""
            return _undecided(key, f"its piece ran (exit {ran.get('rc')}{late}) and wrote no answer; its log is "
                                   f"{comp}/piece.log", d, run, error=True)
        if key == "wipe" and (d / "piece.log").exists():
            return _undecided(key, f"its piece wrote no answer; its log is {comp}/piece.log", d, run)
        return _undecided(key, NOT_RUN, d, run, why=_unset(comp, what))
    try:
        row = row_of(owner, json.loads(f.read_text()), d, run)
    except ValueError as e:
        return _undecided(key, f"its answer isn't a core answer: {type(e).__name__}: {e}", d, run, error=True)
    return {**row, "why": _unset(comp, what)} if (row["verdict"], row["reason"]) == ("undecided", NOT_RUN) else row


def piece_server(key: str, pieces_dir: Path) -> tuple[str | None, str]:
    """The server a piece's answer is about and the answer's source (``real`` or ``fixture``): the server its `SERVED`
    record names, else, but for the wipe, the node `pieces` ran it on (`RAN`); None where neither names one."""
    comp = PIECES.get(key, WIPE)[1]
    served, ran = _json(pieces_dir / comp / SERVED), _json(pieces_dir / comp / RAN)
    served = served if isinstance(served, dict) else {}
    ran = ran if isinstance(ran, dict) and key != "wipe" else {}
    return served.get("server") or ran.get("server"), served.get("source") or "real"


def _ratio(e: Fraction, deltas: Sequence[Fraction]) -> str:
    """``e`` exactly, as ``d + 2^-n`` where it is one of ``deltas`` plus a power of two."""
    for d in deltas:
        r = e - d
        if r > 0 and r.numerator == 1 and r.denominator & (r.denominator - 1) == 0:
            return f"{d} + 2^-{r.denominator.bit_length() - 1}"
    return str(e)


def terms(r: Mapping[str, Any], leg: Mapping[str, Any] | None) -> list[str]:
    """What an accepted answer holds beyond its verdict: where its profile says when its weights were held and what share
    of the checkpoint they are, those; and each δ of the block record's at which its leg's quantity is not established."""
    profile = (r.get("answer") or {}).get("profile")
    w = profile.get("weights") if isinstance(profile, Mapping) else None
    out = []
    if isinstance(w, Mapping) and w.get("held"):
        out.append(f"its weights {w['held']}, not concurrently with the block")
    if isinstance(w, Mapping) and w.get("share"):
        outside = (profile.get("outside") or {}).get("bytes") or {}
        out.append(f"they are {float(Fraction(w['share'])):.1%} of the checkpoint ({w.get('covers', 'its weights')})"
                   + (f"; {', '.join(sorted(outside))} are outside its claim" if outside else ""))
    if leg:
        no = [d for d, x in leg["at"].items() if not x["holds"]]
        if no:
            from verity_e2e_demo import block_record as BR
            out.append(f"its quantity is not established at δ = {' or '.join(no)}: the run established it except with "
                       f"{_ratio(Fraction(leg['error']), BR.DELTAS)}")
    return out


def case_dirs(serve_dir: Path, window: int, pieces_dir: Path, extra: Sequence[Path] = ()) -> list[tuple[str, Path]]:
    """Where each piece keeps its case, by a label: its private directory under the serve dir (`PRIVATE`; PoUS's the
    one its ``pous.json`` names, else ``private/pous/w<K>``, the wipe's ``private/wipe``, any other's
    ``private/<component>``) when that holds a case and its answer, else its public directory, where a piece whose case
    holds no witness keeps it (the network certifier's, read by its leak guard); then each ``extra`` directory
    (``--cases``), or its subdirectories, as composite.py reads one (the inference-only policy's is one)."""
    priv = serve_dir / PRIVATE
    out: list[tuple[str, Path]] = []
    for comp in (*(v[1] for v in PIECES.values()), WIPE[1]):
        d, p = pieces_dir / comp, priv / comp
        if comp == "pous":
            try:
                p = Path(json.loads((d / "pous.json").read_text())["private"])
            except (OSError, ValueError, KeyError, TypeError):
                p = priv / comp / f"w{window}"
        out.append((comp, p if C.ran(p) else d))
    return out + [(d.name, d) for d in C.case_dirs(extra)]


def found(dirs: Sequence[tuple[str, Path]]) -> tuple[dict[str, Path], dict[str, str]]:
    """The cases among ``dirs`` (`case_dirs`'s) by their answers' protocols: each directory holding a core answer and its
    case (``answer.json``, ``case.json``); and for each protocol answered with no case, or by more than one directory,
    why it has no piece, by its verdict alone (an answer's reason can be witness). A directory with no answer is no
    protocol's piece (its row says why), and one whose answer isn't a core answer is none either: ``why`` says so under
    its label."""
    by: dict[str, list[tuple[str, Path, CA.Answer]]] = {}
    why: dict[str, str] = {}
    for label, d in dirs:
        if not (d / ANSWER).is_file():
            continue
        try:
            a = CA.of_json(json.loads((d / ANSWER).read_text()))
        except ValueError as e:
            why[label] = f"its {ANSWER} is no core answer ({type(e).__name__})"
            continue
        by.setdefault(a.protocol, []).append((label, d, a))
    got: dict[str, Path] = {}
    for pid, xs in by.items():
        label, d, a = xs[0]
        if len(xs) > 1:
            why[pid] = "answered by more than one piece: " + ", ".join(x for x, _, _ in xs)
        elif (d / CASE).is_file():
            got[pid] = d
        else:
            why[pid] = f"{a.verdict.value}; {label} holds no {CASE}"
    return got, why


def missing(got: Mapping[str, Path], why: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """What each product of `NEEDS` lacks of ``got`` (`found`'s), and why."""
    return {p: {i: why.get(i, "no piece answers it") for i in ids if i not in got} for p, ids in NEEDS.items()}


def lacking(lacks: Mapping[str, Mapping[str, str]]) -> str:
    """An undecided window's reason: what each product lacks, and why."""
    return "; ".join(f"{C.PRODUCTS[p]} lacks " + ", ".join(f"{i} ({w})" for i, w in m.items()) for p, m in lacks.items())


#: a product's exit status for each verdict (`Core.Protocol.main`)
EXIT = {"accept": 0, "reject": 1, "undecided": 2}


def public_form(product: str, rec: Any) -> dict[str, Any]:
    """``rec`` when it is composite.py's public record (`composite.public_record`) of ``product`` on a whole case, with
    the Lean verdict and nothing else: its product, the composite's commitment, each case's id, params digest,
    commitment and outcome, the product's outcome (`composite.outcome`) and its stated exceptions. Anything else is a ValueError, and none of
    it is published."""
    hexes = lambda x: isinstance(x, str) and len(x) % 2 == 0 and all(c in "0123456789abcdef" for c in x)  # noqa: E731
    outcome = lambda o: (isinstance(o, dict) and {"id", "verdict"} <= set(o) <= {"id", "verdict", "refusal"}  # noqa: E731
                         and all(isinstance(v, str) for v in o.values()))
    if not (isinstance(rec, dict)
            and sorted(rec) == ["cases", "commitment", "exceptions", "missing", "product", "verdict"]):
        raise ValueError(f"not composite.py's public record: {sorted(rec) if isinstance(rec, dict) else type(rec)}")
    if rec["product"] != C.PRODUCTS[product] or not hexes(rec["commitment"]):
        raise ValueError("its product or commitment isn't the composite's")
    if rec["exceptions"] != C.exceptions(product):
        raise ValueError("its stated exceptions aren't the product's")
    if rec["missing"] != []:
        raise ValueError(f"composite.py found missing what the pieces answer: {rec['missing']}")
    if not (outcome(rec["verdict"]) and rec["verdict"]["id"] == C.PRODUCTS[product]
            and rec["verdict"]["verdict"] in EXIT):
        raise ValueError(f"no Lean verdict of {C.PRODUCTS[product]}: {rec['verdict']}")
    for c in rec["cases"] if isinstance(rec["cases"], list) else [None]:
        rest = {k: v for k, v in c.items() if k not in ("params_sha512", "commitment")} if isinstance(c, dict) else None
        if not (rest is not None and hexes(c.get("params_sha512")) and hexes(c.get("commitment")) and outcome(rest)):
            raise ValueError("a case's entry isn't its id, params digest, commitment and outcome")
    return rec


def lean_verdict(product: str, got: Mapping[str, Path], exe: Path, private: Path, public: Path) -> dict[str, Any]:
    """``composite.py PRODUCT`` on its cases of ``got`` (`NEEDS`, each a ``--cases``), ``--exe`` ``exe``: the composed
    case, its evidence and every answer go to ``--private-out`` ``private/out`` (0700, with composite.py's own output),
    and ``--public-out`` to ``public``, which is held to `PUBLIC_FILES` and `public_form` and to the exit status Lean's
    verdict gives; its public record. RuntimeError when it is anything else, whose message is public: it carries none
    of composite.py's output but its own refusal (``composite.py: ...``)."""
    if not exe.is_file():
        raise RuntimeError(f"no {product} executable at {exe} (lake build {product} in verity/core/protocols/{product})")
    shutil.rmtree(public, ignore_errors=True)
    C.write_private(private, {})
    out = private / "out"
    argv = [sys.executable, str(HERE / "composite.py"), product, "--private-out", str(out), "--public-out", str(public),
            "--exe", str(exe)]
    for i in NEEDS[product]:
        argv += ["--cases", str(got[i])]
    p = subprocess.run(argv, capture_output=True, text=True, check=False)
    C.write_private(private, {"composite.log": (p.stdout + p.stderr).encode()})
    said = next((f": {x[:600]}" for x in reversed(p.stderr.splitlines()) if x.startswith("composite.py:")), "")
    files = sorted(f.name for f in public.iterdir()) if public.is_dir() else []
    if files != sorted(PUBLIC_FILES):
        raise RuntimeError(f"composite.py {product} exited {p.returncode} with public output {files}{said}")
    try:
        rec = public_form(product, json.loads((public / "composite.json").read_text()))
    except ValueError as e:
        raise RuntimeError(f"composite.py {product} exited {p.returncode}: {e}"[:2000]) from None
    if p.returncode != EXIT[rec["verdict"]["verdict"]]:
        raise RuntimeError(f"composite.py {product} exited {p.returncode} on Lean's {rec['verdict']['verdict']}")
    return rec


def said(rows: Mapping[str, Mapping[str, Any]], at: Mapping[str, tuple[str | None, str]], node: str,
         legs: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """Each subprotocol's answer in one clause: its verdict and, unless it accepted about ``node``, why; an accept about
    ``node`` with its `terms` (``legs``: each accepted one's block record leg, by runner name). It decides nothing."""
    out = []
    for k, r in rows.items():
        v, server = r.get("verdict"), at[k][0]
        if "development" in r and v == "undecided":
            s = r.get("why")
        elif (v, r.get("reason")) == ("undecided", NOT_RUN):
            s = f"{NOT_RUN}: {r.get('why')}"
        elif server not in (None, node):
            s = f"{v} about {server}, another server: carried, not composed"
        elif v == "accept":
            t = terms(r, (legs or {}).get(k))
            s = ("accept" if server is not None else "accept, naming no server: not carried") + (f" ({'; '.join(t)})" if t else "")
        else:
            s = f"{v}{' (an error)' if r.get('error') else ''}: {r.get('reason')}"
        out.append(f"{NAMES[k]}: {s}")
    return "; ".join(out)


def record(rows: Mapping[str, Mapping[str, Any]], *, window: int, steps: tuple[int, int], node: str,
           run: str | None, cases: Sequence[tuple[str, Path]], private: Path, out: Path,
           lean: Mapping[str, Path] | None = None, servers: Mapping[str, tuple[str | None, str]] | None = None
           ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """The window report over ``rows`` (by runner name), whose verdict is onsite's or remote's Lean verdict on the
    ``cases`` (`case_dirs`'s; `lean_verdict`, with each product's executable in ``lean``, `LEAN` by default, its private
    files under ``private``) or ``undecided`` naming what each lacks; its block record (None when the window or a row
    about this node rejects); and the ``RECORD`` line's summary. Of the composite only its public record is published,
    to ``out/composite``, once `lean_verdict` has held it to its form. ``servers``: each piece's `piece_server`, by
    runner name; sampled proofs' answer is about ``node``, and a piece's that names no server isn't carried."""
    from verity_e2e_demo import block_record as BR

    legs = {v: k for k, v in BR.RUNNER_KEYS.items()} | {"wipe": "wipe"}
    at = {k: (None, "real") for k in rows} | dict(servers or {}) | {"integrity": (node, "real")}
    rows = {k: dict(r) for k, r in rows.items()}
    built: dict[str, dict[str, Any]] = {}
    for k, r in rows.items():
        if r.get("verdict") == "accept" and at[k][0] is not None:
            try:
                built[k] = BR.leg(legs[k], r, server=at[k][0], runs=[], source=at[k][1])
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as e:
                why = f"its answer is not the block record's {legs[k]} leg: {type(e).__name__}: {e}"
                rows[k] = {**r, **_undecided(k, why, None, run, error=True), "owner": r.get("owner"), "runs": r.get("runs")}
    got, why = found(cases)
    lacks = missing(got, why)
    product = next((p for p, m in lacks.items() if not m), None)
    composite, error = None, None
    shutil.rmtree(out / "composite", ignore_errors=True)
    if product is not None:
        try:
            composite = lean_verdict(product, got, Path((lean or LEAN)[product]), private, private / "public")
        except RuntimeError as e:
            error = str(e)[:2000]
        else:
            (out / "composite").mkdir(parents=True)
            for f in PUBLIC_FILES:
                shutil.copyfile(private / "public" / f, out / "composite" / f)
    if composite is not None:
        v = composite["verdict"]
        verdict = {"protocol": v["id"], "verdict": v["verdict"], "reason": v.get("refusal")}
    elif product is not None:
        verdict = CA.Answer.undecided(C.PRODUCTS[product], f"the Lean check: {error}").to_json()
    else:
        verdict = CA.Answer.undecided(C.PRODUCTS["onsite"], lacking(lacks)).to_json()
    report = {"format": REPORT_FORMAT, "window": window, "node": node, "run": run, "protocols": rows,
              "servers": {k: at[k][0] for k in rows}, "said": said(rows, at, node, built),
              "lean": {"product": product and C.PRODUCTS[product], "cases": sorted(got),
                       "missing": {C.PRODUCTS[p]: m for p, m in lacks.items()}, "composite": composite, "error": error},
              "verdict": verdict}
    elsewhere = {NAMES[k]: at[k][0] for k in rows if at[k][0] not in (None, node)}
    unserved = sorted(NAMES[k] for k, r in rows.items() if (r["verdict"] != "undecided" or r.get("error"))
                      and at[k][0] is None)
    dev = {"development": {NAMES[k]: r["development"]["verdict"] for k, r in rows.items() if "development" in r}}
    line = {"verdict": verdict["verdict"], "protocol": verdict["protocol"], "reason": verdict["reason"],
            "lean": {"product": report["lean"]["product"], "verdict": composite and composite["verdict"]["verdict"],
                     "error": error, "missing": {C.PRODUCTS[p]: sorted(m) for p, m in lacks.items()}},
            "verdicts": {NAMES[k]: DEV_VERDICT if "development" in r and r["verdict"] == "undecided" else r["verdict"]
                         for k, r in rows.items()},
            "rejected": [NAMES[k] for k, r in rows.items() if r["verdict"] == "reject" and NAMES[k] not in elsewhere],
            "errors": [NAMES[k] for k, r in rows.items() if r.get("error") and NAMES[k] not in elsewhere],
            "not_run": {NAMES[k]: r.get("why") for k, r in rows.items()
                        if (r["verdict"], r["reason"]) == ("undecided", NOT_RUN)},
            "elsewhere": elsewhere, **({"unserved": unserved} if unserved else {}),
            **(dev if dev["development"] else {})}
    if verdict["verdict"] == "reject" or line["rejected"]:
        return report, None, {**line, "block_record": None,
                              "why": "block_record carries an accepted subprotocol's certificate only"}
    extra = {legs[k]: {"row": r, "server": at[k][0], "runs": r.get("runs") or [run], "source": at[k][1]}
             for k, r in rows.items() if k != "integrity" and r.get("verdict") == "accept" and at[k][0] is not None}
    composing = {**report, "protocols": {k: r if k == "integrity" else {"verdict": "undecided"} for k, r in rows.items()}}
    rec = BR.from_period(composing, steps=steps, deployment=BR.REFERENCE, extra=extra)
    return report, rec, {**line, "record_missing": rec["composed"]["missing"], "check": BR.check(rec),
                         "steps": list(steps), "node": node, "deployment": rec["monitoring_period"]["deployment"]["id"]}


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
    r.add_argument("--cases", type=Path, action="append", default=[],
                   help="a further private case directory, or a directory of them (the inference-only policy's)")
    for prod in C.PRODUCTS:
        r.add_argument(f"--{prod}", type=Path, default=LEAN[prod], help=f"{prod}'s executable")
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
        return 0 if all(g["verdict"] in ("accept", "undecided") and not g.get("error") for g in got.values()) else 1
    steps = pick(a.serve_run, 1 << 30, a.window)["steps"]
    run = os.environ.get("RESEARCH_RUN_ID")
    keys = [*PIECES, "wipe"]
    got = {k: piece(k, a.pieces, run) for k in keys}
    rows = {"integrity": sampled_proofs_row(a.sampled_proofs, run), **{k: row for k, (row, _) in got.items()}}
    a.out.mkdir(parents=True, exist_ok=True)
    report, rec, line = record(rows, window=a.window, steps=(steps[0], steps[1]), node=a.node, run=run,
                               cases=case_dirs(a.serve_run, a.window, a.pieces, a.cases),
                               private=a.serve_run / PRIVATE / "composite" / f"w{a.window}", out=a.out,
                               lean={p: getattr(a, p) for p in C.PRODUCTS},
                               servers={k: at for k, (_, at) in got.items()})
    (a.out / "verdicts.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    if rec is not None:
        (a.out / "block-record.json").write_text(json.dumps(rec, separators=(",", ":"), sort_keys=True) + "\n")
    print("RECORD " + json.dumps(line, sort_keys=True), flush=True)
    ok = not (line.get("check") or line["rejected"] or line["errors"] or report["lean"]["error"])
    return 0 if rec is not None and ok else 1


if __name__ == "__main__":
    sys.exit(main())
