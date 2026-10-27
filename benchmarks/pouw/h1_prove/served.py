"""served.py: the units a served H1 window's own draw coin selects (vllm's `pouw_h1` window, its z table one leaf a
unit of its ``tile``), as PoUW's verifier draws them (core's `credit.drawn`).

    python served.py --calls CALLS.json --session DIR --ledger windows.jsonl [--ledger ...] --pin SHA512 --pins LOG
                     --main FILE --descriptor PROGRAM[.gz] [--out DRAWN.json]

CALLS.json is the window's ``pouw-h1-calls.json``; DIR the challenger session's directory as the auditor's gate fetched it
(``fetch/<session>/out``), never the prover's bundle; each ``--ledger`` a published head of the auditor's signed ledger;
``--pin`` the SHA-512 of the challenger's public key, the verifier's own configuration; ``--pins`` the challenger host's
pins log as infra published it, a line a pin; ``--main`` the verifier's own ``git rev-list --first-parent origin/main``;
PROGRAM the window Program's descriptor (the rehearsal's ``window-program.json.gz``), whose SHA-512 must be the circuit the
window's period-open signed.

1. The window's draw coin is the verifier's challenger's, or nothing is drawn: the session the signed ledger opens and
   ends for the window's root, under the pinned key (`remote.trusted_session`), run from a commit the pins log records
   on main's first-parent line (`challenger.pinned_session`: a session the log doesn't record is refused, though this
   draw credits nothing), and in it the window's one draw coin,
   opened after its `draw_root` over its manifest, last receipts and every round's root and coin
   (`pouw_h1_challenger.draw_reason`), over a manifest whose ``calls_root`` binds its calls and served trees' entries
   (`pouw_h1_challenger.manifest_reason`). The window's salt and round coins are its served values' checks (vllm's
   verify), not the draw's.
2. The draw is core's (`pouw_h1_challenger.drawn_of`, `credit.drawn` on the served window's arguments: stream the window's
   root, registration its draw root, receipt ``{"receipts": …}``), over the layout of the descriptor its period committed
   (`credit.layout_of`), at the K that period's open committed (`credit.committed_k`): one work stratum a K.
3. Each drawn unit is located in the window's calls: unit u is call i's unit u - base_i, its row block b and column block
   c at b · (n / C) + c for the window's R × C ``tile`` (vllm's z/y records' order), and the call's z table's leaf
   u - base_i.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
from pathlib import Path

from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs import remote
from verity.core.protocols.sampled_proofs.challenger import CoinError, LiveOpening, ledger_head, pinned_session
from verity_vllm.protocols.pouw.h1 import challenger as PHC


def locate(calls: list[dict], tile: tuple[int, int], u: int) -> dict:
    """Unit `u` of the window: its call, and its row and column block in that call's z table of `tile` units."""
    i = bisect.bisect_right([c["base"] for c in calls], u) - 1
    if i < 0 or not 0 <= u - calls[i]["base"] < calls[i]["units"]:
        raise SystemExit(f"unit {u} is in no call of the window")
    c = calls[i]
    local = u - c["base"]
    per_row = c["n"] // tile[1]
    return {"unit": u, "call": c["index"], "module": c["module"], "step": c["step"], "k": c["k"], "n": c["n"],
            "row_block": local // per_row, "col_block": local % per_row, "z_leaf": local,
            "z_table": c["tables"]["z"], "a_table": c["tables"]["a"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--ledger", action="append", required=True)
    ap.add_argument("--pin", required=True)
    ap.add_argument("--pins", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--descriptor", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    doc = json.loads(Path(a.calls).read_text())
    commits, calls = doc["commitment"], doc["calls"]
    root, tile = commits["session"]["root"], tuple(commits["tile"])
    pins = [x for x in Path(a.pins).read_text().split("\n") if x]
    try:
        key = PHC.pinned_key(Path(a.session) / "key.json", a.pin)
        trusted, received = remote.trusted_session(a.session, a.ledger, {"root": root}, key, PHC.lms_verify)
        lines = ledger_head([remote.ledger_lines(p) for p in a.ledger], key, PHC.lms_verify)
        audited = pinned_session(lines, {"root": root}, key, pins, remote.main_lines(a.main), PHC.lms_verify)
    except CoinError as e:
        raise SystemExit(f"REFUSED: the challenger's session is not trusted: {e}") from e
    if (why := PHC.draw_reason(commits, trusted, received.get(Wd.DRAW, ()))) is not None:
        raise SystemExit(f"REFUSED: {why}")
    if (why := PHC.manifest_reason(doc)) is not None:
        raise SystemExit(f"REFUSED: {why}")
    if sum(c["units"] for c in calls) != commits["units"] or any(
            c["base"] != sum(x["units"] for x in calls[:j]) for j, c in enumerate(calls)):
        raise SystemExit("REFUSED: the calls' units are not the window's units in order")
    if audited.period is None:
        raise SystemExit("REFUSED: the window is in no period of the auditor's ledger, so no period-open committed its "
                         "circuit and draw K")
    raw = Path(a.descriptor).read_bytes()
    descriptor = gzip.decompress(raw) if a.descriptor.endswith(".gz") else raw
    if hashlib.sha512(descriptor).hexdigest() != audited.circuit:
        raise SystemExit(f"REFUSED: the descriptor's SHA-512 is not {str(audited.circuit)[:16]}..., the circuit period "
                         f"{audited.period}'s open signed")
    k, why = CR.committed_k(lines, audited.period)
    units = None
    if why is None:
        units, why = PHC.drawn_of(doc, {"root": root}, k, descriptor)
    if why is not None:
        raise SystemExit(f"REFUSED: {why[0]}: {why[1]}")
    layout, _ = CR.layout_of(descriptor)
    strata, _ = CR.strata(layout)
    CR.law(strata, k)
    coin = LiveOpening.from_json(commits["draw"]["coin"]).coin
    out = {"window": root, "draw_root": commits["draw"]["root"], "coin": coin.hex(), "session": audited.session,
           "period": audited.period, "circuit": audited.circuit, "k": k,
           "key": {"stream": "the window's root", "registration": "the draw root",
                   "receipt": {"receipts": commits["draw"].get("receipts")}},
           "strata": [{"template": s["template"], "units": sum(hi - lo for lo, hi in s["units"]), "work": s["work"],
                       "k": s["k"]} for s in strata],
           "units": [locate(calls, tile, u) for u in units]}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({**{x: out[x] for x in ("window", "draw_root", "coin", "session", "period", "k", "strata")},
                      "drawn": len(out["units"]),
                      "units": [{x: v for x, v in u.items() if x not in ("z_table", "a_table")} for u in out["units"][:8]]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
