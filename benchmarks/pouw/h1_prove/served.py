"""served.py: the units a served H1 window's own draw coin selects (vllm's `pouw_h1` window, its z table one leaf a
unit of its ``tile``).

    python served.py --calls CALLS.json --session DIR --ledger windows.jsonl [--ledger ...] --pin SHA512 --lean FLOCK_VERIFY
                     [--k 1] [--out DRAWN.json]

CALLS.json is the window's ``pouw-h1-calls.json``; DIR the challenger session's directory as the auditor's gate fetched it
(``fetch/<session>/out``), never the prover's bundle; each ``--ledger`` a published head of the auditor's signed ledger;
``--pin`` the SHA-512 of the challenger's public key, the verifier's own configuration.

1. The window's draw coin is the verifier's challenger's, or nothing is drawn: the session the signed ledger opens and
   ends for the window's root, under the pinned key (`remote.trusted_session`), and in it the window's one draw coin,
   opened after its `draw_root` over its manifest, last receipts and every round's root and coin
   (`pouw_h1_challenger.draw_reason`). The window's salt and round coins are its served values' checks (vllm's verify),
   not the draw's.
2. The draw coin selects ``--k`` of the window's ``units`` by h1_prove's rule (`run.draw_units`, Lean ``flock-verify
   draw``), bound to the window's root and its draw root, the registration the coin followed.
3. Each drawn unit is located in the window's calls: unit u is call i's unit u - base_i, its row block b and column block
   c at b · (n / C) + c for the window's R × C ``tile`` (vllm's z/y records' order), and the call's z table's leaf
   u - base_i.

Prints and writes ``{window, draw_root, coin, population, k, units: [{unit, call, module, k, n, row_block, col_block,
z_table, a_table}]}``.
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run import draw_units  # noqa: E402
from verity.core.protocols.pouw import window as Wd  # noqa: E402
from verity.core.protocols.sampled_proofs import remote  # noqa: E402
from verity.core.protocols.sampled_proofs.challenger import CoinError, LiveOpening  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_challenger as PHC  # noqa: E402


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
    ap.add_argument("--lean", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    doc = json.loads(Path(a.calls).read_text())
    commits, calls = doc["commitment"], doc["calls"]
    root = commits["session"]["root"]
    try:
        key = PHC.pinned_key(Path(a.session) / "key.json", a.pin)
        trusted, received = remote.trusted_session(a.session, a.ledger, {"root": root}, key, PHC.lms_verify)
    except CoinError as e:
        raise SystemExit(f"REFUSED: the challenger's session is not trusted: {e}") from e
    if (why := PHC.draw_reason(commits, trusted, received.get(Wd.DRAW, ()))) is not None:
        raise SystemExit(f"REFUSED: {why}")
    if sum(c["units"] for c in calls) != commits["units"] or any(
            c["base"] != sum(x["units"] for x in calls[:j]) for j, c in enumerate(calls)):
        raise SystemExit("REFUSED: the calls' units are not the window's units in order")
    draw = commits["draw"]
    coin = LiveOpening.from_json(draw["coin"]).coin
    units = draw_units(a.lean, coin, root, draw["root"], commits["units"], a.k)
    out = {"window": root, "draw_root": draw["root"], "coin": coin.hex(), "population": commits["units"], "k": a.k,
           "rule": "h1_prove run.draw_units (DRAW_DOMAIN, Lean flock-verify draw)",
           "units": [locate(calls, tuple(commits["tile"]), u) for u in units]}
    text = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
