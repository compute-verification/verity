"""The bandwidth audit's answer on one served window (`served_verdicts.sh`'s BANDWIDTH_CMD):

    served_bandwidth.py collect --serve-dir SERVE_DIR --window K --out DIR [--private P] [--gate CMD]
    served_bandwidth.py answer --session FILE --serve-dir SERVE_DIR --window K --out DIR [--r-bytes-per-s R]
                       [--started UTC --ended UTC] [--gpu FILE]

``collect`` (`collect`), the composed run's piece: PoUB-P2's session runs inside the PoUS piece, over the ``C*`` that piece
encoded and holds (`benchmarks/pous/served_piece.py`, `held_bandwidth.py`), which writes the bandwidth piece's public
answer and summary to ``DIR`` (``answer.json``, ``bandwidth.json``) and its case to ``SERVE_DIR/private/bandwidth``; the
composed run starts PoUS's piece before this one (`served_verdicts.PIECES`), so ``collect`` only reads what it wrote:

    BANDWIDTH_CMD='"$PY" benchmarks/one_stage/served_bandwidth.py collect --serve-dir "$1" --window "$2" --out "$3"'

(`served_verdicts.sh`'s default once POUS_CMD is set).

An answer that isn't window K's is replaced by an undecided one saying whose it is, and where the PoUS piece wrote none
nothing is written, so the record names the bandwidth audit missing (`served_verdicts.piece_row`). An accept is credited
only by the auditor's host (`held_to`), since what the PoUS piece wrote came through the prover's process: the bandwidth
party's own answer of its first session under window K's binding as that host kept it (`host_answer`, the gate's
``party_log bw answers AUDIT`` through ``--gate``, by default `BW_GATE`; `p2_party.session`'s ``audit``), the binding
derived from the serve dir as PoUS's piece binds it (`benchmarks/pous/served_window.py`), never from what that piece
wrote. That answer must be an accept saying what the public one does (`agrees`: the public outputs
`RemoteAudit.P2Protocol` gives its case, and γ against the public answer's R), and the prover's copy of it, the
session's record (``P/session.json``'s ``verifier_record``, ``P`` the PoUS piece's ``--bw-private``, by default
``SERVE_DIR/private/bandwidth``), must be it. A later session under the same binding is a retry, which never counts. Any
other (no answer on the host, another, a record that isn't the host's, no gate) is replaced by an undecided answer saying
which and naming no value; nothing the host kept is written. Exit 0 on accept, 1 on any other answer, 2 on none.

``answer``: the session of `served_bandwidth.sh`, over a store of the node's own on its disk.

The remote bandwidth session's record (`benchmarks/pous/p2_v1/bandwidth.py session`'s ``session.json``) as a core answer
in ``DIR/answer.json`` (`answer`): the verifier's answer, held to the verifier's own record of it where the session has
one, in the core interface's form (`p2_party.core_answer`), with γ against ``R``. The session runs on the card after
window K was served, beside it: serving is not hooked, so its reads are the session's own, not the engine's reads of the
window's steps (`SCOPE`); where window K isn't served the answer says that instead (`UNSERVED`). An accept has no reason
in the core form, so its profile says so; any other answer says so in its reason. ``DIR/served.json`` names the window
(`served`), the session's times and card, the verifier and the store.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from verity.core.protocols.remote_bandwidth import p2_case, p2_party  # noqa: E402

#: the served window's binding, the audit PoUS's piece runs its session and its bandwidth party under
SERVED_WINDOW = HERE.parent / "pous" / "served_window.py"
#: the auditor's host's gate under the card's bandwidth key, whose `bw_party` the PoUS piece starts the party with
#: (`benchmarks/pous/served_piece.py`'s BW_PARTY), and whose `party_log bw answers AUDIT` gives that party's own answers
BW_GATE = "ssh -i ~/.ssh/vy-auditor2-bw -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 research@10.96.0.36"
#: an H200's specified HBM read rate, bytes a second
H200_BYTES_PER_S = 4_800_000_000_000
#: what the session certifies of the served window
SCOPE = ("beside the served window: the session ran on the card after window {k} was served, over the node's registered "
         "store, and was not interleaved with the engine's reads (serving is not hooked), so it certifies the session's "
         "own reads, not the window's")
#: what it certifies when the window it was asked about wasn't served
UNSERVED = ("with no served window: window {k} of {serve_dir} is not served ({error}), so the session ran on the card over "
            "the node's registered store with no window served before it, and certifies its own reads only")


def served(serve_dir: Path, window: int) -> dict[str, Any]:
    """Window `window` of the serve dir as `served_verdicts.pick` holds it (its steps and requests), or why it can't."""
    import served_verdicts as S

    try:
        w = S.pick(serve_dir, 1 << 30, window)
    except (OSError, ValueError, KeyError, TypeError) as e:
        return {"window": window, "serve_dir": str(serve_dir), "error": f"{type(e).__name__}: {e}"[:600]}
    return {"window": window, "serve_dir": str(serve_dir), "steps": w["steps"], "requests": w["requests"]}


def scope_of(window: Mapping[str, Any]) -> str:
    """What the session certifies of `window`, as `served` found it."""
    if "error" in window:
        return UNSERVED.format(k=window["window"], serve_dir=window["serve_dir"], error=window["error"])
    return SCOPE.format(k=window["window"])


def answer(session: Mapping[str, Any], scope: str, r_bytes_per_s: int | None) -> dict[str, Any]:
    """The core answer of the session's record `session`, saying `scope` (module docstring)."""
    a, record = dict(session.get("answer") or {}), session.get("verifier_record")
    if "protocol" not in a:
        a = {"protocol": p2_party.PROTOCOL, "registration": (session.get("registration") or {}).get("digest"), **a}
    if record is not None and record != session.get("answer"):
        a = {"protocol": p2_party.PROTOCOL, "registration": a.get("registration"), "verdict": "disagreement",
             "reason": f"the verifier's own record of its answer ({record.get('verdict')}) is not the session's "
                       f"({a.get('verdict')})"}
    out = p2_party.core_answer(a, r_bytes_per_s)
    if out["verdict"] == "accept":
        out["profile"]["scope"] = scope
    else:
        out["reason"] = f"{out['reason']}; {scope}"
    return out


def _served_window():
    """`benchmarks/pous/served_window.py`, loaded by its path."""
    import importlib.util

    if "served_window" not in sys.modules:
        spec = importlib.util.spec_from_file_location("served_window", SERVED_WINDOW)
        module = sys.modules["served_window"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return sys.modules["served_window"]


def host_answer(gate: str, audit: bytes) -> dict[str, Any] | None:
    """The bandwidth party's own answer of its first session held by the PoUS session of `audit` (the window's binding), as
    the auditor's host kept it (`gate`'s ``party_log bw answers AUDIT``), None when it kept none; a later one under `audit`
    is a retry, which never counts. RuntimeError when the gate refuses, ValueError on an answer out of form."""
    import served_verdicts as S

    doc = json.loads(S._gate_out(gate, "party_log", "bw", "answers", audit.hex()))
    if not isinstance(doc, dict) or not isinstance(doc.get("answers"), list):
        raise ValueError("the gate's party_log bw answers is not {\"answers\": [...]}")
    kept = [x for x in doc["answers"] if isinstance(x, dict) and x.get("audit") == audit.hex()]
    return kept[0] if kept else None


def agrees(a: Mapping[str, Any], host: Mapping[str, Any]) -> bool:
    """Whether the public accept `a` says what the bandwidth party's own answer `host` does: an accept, `a`'s public
    outputs those `RemoteAudit.P2Protocol` gives the case the party kept with it (`p2_case.answer`) and, where `a` has
    them, γ against its R as the PoUS piece adds it (`held_bandwidth.public_answer`)."""
    if host.get("verdict") != "accept":
        return False
    try:
        # the case's public outputs are its session's answer's alone: the case gives only the commitment
        public = p2_case.answer({"commitment": ""}, host)["public"]
        r = a["public"].get("r_bytes_per_s")
        if r is not None:
            public |= {"gamma": str(1 - Fraction(public["q_bytes"] * 10**9, r * public["span_ns"])), "r_bytes_per_s": r}
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return a["public"] == public


def held_to(a: Mapping[str, Any], serve_dir: Path, window: int, private: Path, gate: str) -> str | None:
    """None when the accept `a` is the bandwidth party's own answer of window `window`'s PoUS session as the auditor's host
    kept it (module docstring): the host's (`host_answer`, under the binding derived here from `serve_dir`,
    `served_window.of_serve_dir`) is an accept that `a` `agrees` with, and the session's record in `private` is it;
    else why not, naming no value."""
    try:
        bind = _served_window().of_serve_dir(serve_dir, window)
    except ValueError as e:
        return f"window {window}'s binding can't be derived: {e}"
    try:
        host = host_answer(gate, bind)
    except (OSError, RuntimeError, ValueError) as e:
        print(f"BANDWIDTH the auditor's host: {type(e).__name__}: {e}"[:800], file=sys.stderr, flush=True)
        return "the auditor's host gave no answers of its bandwidth party (its gate refused or wasn't reached)"
    if host is None:
        return f"the auditor's host kept no answer of a bandwidth session under window {window}'s binding"
    if not agrees(a, host):
        return f"the auditor's host kept another answer than this accept for window {window}'s session"
    try:
        doc = json.loads((private / "session.json").read_text())
    except (OSError, ValueError) as e:
        return f"there is no record of the session in {private} ({type(e).__name__})"
    if not isinstance(doc, dict) or doc.get("verifier_record") != host:
        return "the session's record of the bandwidth party's answer is not the one the auditor's host kept"
    return None


def _undecided(out: Path, reason: str) -> dict[str, Any]:
    """The undecided answer saying `reason`, written over ``out/answer.json``."""
    a = {"commitment": None, "profile": None, "protocol": p2_party.PROTOCOL, "public": None, "reason": reason,
         "verdict": "undecided"}
    (out / "answer.json").write_text(json.dumps(a, indent=1, sort_keys=True) + "\n")
    return a


def collect(out: Path, window: int, serve_dir: Path, private: Path | None = None,
            gate: str = BW_GATE) -> dict[str, Any] | None:
    """The answer the PoUS piece's bandwidth session wrote to `out` for `window` (module docstring), an accept held to
    the bandwidth party's own answer of the window's PoUS session as the auditor's host behind `gate` kept it, and the
    session's record in `private` to that (`held_to`; by default ``serve_dir/private/bandwidth``), None when it wrote
    none; ValueError when what it wrote isn't a core answer of the remote bandwidth audit."""
    import served_verdicts as S

    files = [out / f for f in ("answer.json", "bandwidth.json")]
    if not all(f.exists() for f in files):
        return None
    a, summary = (json.loads(f.read_text()) for f in files)
    S.check_answer(a)
    if a.get("protocol") != p2_party.PROTOCOL:
        raise ValueError(f"the answer in {out} is {a.get('protocol')!r}'s, not the remote bandwidth audit's")
    if summary.get("window") != window:
        return _undecided(out, f"the bandwidth answer in {out} is window {summary.get('window')}'s, not window "
                               f"{window}'s: the PoUS piece ran no bandwidth session for window {window}")
    private = serve_dir / "private" / "bandwidth" if private is None else private
    if a["verdict"] == "accept" and (why := held_to(a, serve_dir, window, private, gate)) is not None:
        return _undecided(out, f"the bandwidth answer in {out} is not of window {window}'s PoUS session: {why}")
    return a


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--serve-dir", type=Path, required=True)
    c.add_argument("--window", type=int, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--private", type=Path, help="the PoUS piece's --bw-private, where its bandwidth session's record "
                                                "is (default SERVE_DIR/private/bandwidth)")
    c.add_argument("--gate", default=BW_GATE, help="the shell command that reaches the bandwidth party's host's gate, "
                                                   "where its own answers are read (default BW_GATE: vy-auditor-2's, "
                                                   "under the key the PoUS piece's --bw-party uses)")
    p = sub.add_parser("answer")
    p.add_argument("--session", type=Path, required=True, help="bandwidth.py session's session.json")
    p.add_argument("--serve-dir", type=Path, required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--r-bytes-per-s", type=int, default=H200_BYTES_PER_S, help="R, the read rate γ is against")
    p.add_argument("--started", help="when the session started (UTC)")
    p.add_argument("--ended", help="when it ended (UTC)")
    p.add_argument("--gpu", type=Path, help="nvidia-smi's line of the card the session ran on")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "collect":
        try:
            got = collect(a.out, a.window, a.serve_dir, a.private, a.gate)
        except ValueError as e:
            print(f"BANDWIDTH error: {e}", flush=True)
            return 1
        if got is None:
            print(f"MISSING the bandwidth audit: the PoUS piece wrote no bandwidth answer to {a.out} for window "
                  f"{a.window} (its session runs over the store the PoUS piece holds, POUS_CMD, before this piece)",
                  flush=True)
            return 2
        print(f"BANDWIDTH {json.dumps({k: got[k] for k in ('verdict', 'reason', 'public')}, sort_keys=True)}", flush=True)
        return 0 if got["verdict"] == "accept" else 1
    try:
        doc = json.loads(a.session.read_text())
    except (OSError, ValueError) as e:
        doc = {"answer": {"verdict": "missing", "reason": f"no session record: {type(e).__name__}: {e}"[:600]}}
    window = served(a.serve_dir, a.window)
    scope = scope_of(window)
    out = answer(doc, scope, a.r_bytes_per_s)
    (a.out / "answer.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    gpu = a.gpu.read_text().strip() if a.gpu is not None and a.gpu.exists() else None
    (a.out / "served.json").write_text(json.dumps({
        "served": window, "scope": scope, "session": {
            "started": a.started, "ended": a.ended, "gpu": gpu, "prover_host": doc.get("prover_host"),
            "verifier": doc.get("verifier"), "verifier_host": (doc.get("answer") or {}).get("host"),
            "store": doc.get("store"), "registration": (doc.get("registration") or {}).get("digest"),
            "request": doc.get("request"), "verdict_line": doc.get("verdict_line")}}, indent=1, sort_keys=True) + "\n")
    print(f"BANDWIDTH {json.dumps({k: out[k] for k in ('verdict', 'reason', 'public')}, sort_keys=True)}", flush=True)
    return 0 if out["verdict"] == "accept" else 1


if __name__ == "__main__":
    sys.exit(main())
