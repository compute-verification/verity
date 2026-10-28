"""The bandwidth audit's answer on one served window (`served_verdicts.sh`'s BANDWIDTH_CMD):

    served_bandwidth.py collect --serve-dir SERVE_DIR --window K --out DIR
    served_bandwidth.py answer --session FILE --serve-dir SERVE_DIR --window K --out DIR [--r-bytes-per-s R]
                       [--started UTC --ended UTC] [--gpu FILE]

``collect`` (`collect`), the composed run's piece: PoUB-P2's session runs inside the PoUS piece, over the ``C*`` that piece
encoded and holds (`benchmarks/pous/served_piece.py`, `held_bandwidth.py`), which writes the bandwidth piece's public
answer and summary to ``DIR`` (``answer.json``, ``bandwidth.json``) and its case to ``SERVE_DIR/private/bandwidth``; the
composed run starts PoUS's piece before this one (`served_verdicts.PIECES`), so ``collect`` only reads what it wrote:

    BANDWIDTH_CMD='"$PY" benchmarks/one_stage/served_bandwidth.py collect --serve-dir "$1" --window "$2" --out "$3"'

(`served_verdicts.sh`'s default once POUS_CMD is set).

An answer that isn't window K's is replaced by an undecided one saying whose it is, and where the PoUS piece wrote none
nothing is written, so the record names the bandwidth audit missing (`served_verdicts.piece_row`). Exit 0 on accept, 1 on
any other answer, 2 on none.

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
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from verity.core.protocols.remote_bandwidth import p2_party  # noqa: E402

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


def collect(out: Path, window: int) -> dict[str, Any] | None:
    """The answer the PoUS piece's bandwidth session wrote to `out` for `window` (module docstring), None when it wrote
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
        a = {"commitment": None, "profile": None, "protocol": p2_party.PROTOCOL, "public": None,
             "reason": f"the bandwidth answer in {out} is window {summary.get('window')}'s, not window {window}'s: the "
                       f"PoUS piece ran no bandwidth session for window {window}", "verdict": "undecided"}
        (out / "answer.json").write_text(json.dumps(a, indent=1, sort_keys=True) + "\n")
    return a


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--serve-dir", type=Path, required=True)
    c.add_argument("--window", type=int, required=True)
    c.add_argument("--out", type=Path, required=True)
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
            got = collect(a.out, a.window)
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
