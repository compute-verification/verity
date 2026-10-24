"""served.py: the units a served H1 window's own draw coin selects (vllm's `pouw_h1` window, its z table one leaf a
unit of its ``tile``), by PoUW's work law over h1window's templates.

    python served.py --calls CALLS.json --session DIR --ledger windows.jsonl [--ledger ...] --pin SHA512 --pins LOG
                     --main FILE [--k K] [--out DRAWN.json]

``--k`` is the work law's budget K (1: each template's floor, one unit a K).

CALLS.json is the window's ``pouw-h1-calls.json``; DIR the challenger session's directory as the auditor's gate fetched it
(``fetch/<session>/out``), never the prover's bundle; each ``--ledger`` a published head of the auditor's signed ledger;
``--pin`` the SHA-512 of the challenger's public key, the verifier's own configuration; ``--pins`` the challenger host's
pins log as infra published it, a line a pin; ``--main`` the verifier's own ``git rev-list --first-parent origin/main``.

1. The window's draw coin is the verifier's challenger's, or nothing is drawn: the session the signed ledger opens and
   ends for the window's root, under the pinned key (`remote.trusted_session`), run from a commit the pins log records
   on main's first-parent line (`challenger.pinned_session`: a session the log doesn't record is refused, though this
   draw credits nothing), and in it the window's one draw coin,
   opened after its `draw_root` over its manifest, last receipts and every round's root and coin
   (`pouw_h1_challenger.draw_reason`). The window's salt and round coins are its served values' checks (vllm's verify),
   not the draw's.
2. The strata are h1window's, one a template (one a K: `h1window.template` over `h1window.tile_unit`, the
   `h1zy.PouwH1TileZy` unit), each a K's units in the window's order, a unit's W_ref its R·C·K multiply-accumulates.
   The draw is `plan.draw` under `plan.WorkLaw` (k_s = min(n_s, max(f_s, ⌈K·W_s/W⌉)), every floor 1) over the window's
   whole work W, each K its own layout (`plan.check_work_strata` takes one work stratum a layout), from the coin's key
   under `plan.DRAW_DOMAIN` and `core_check.draw_key_context`. The result is in the form `core_check.unit_draw` reads.
3. Each drawn unit is located in the window's calls: unit u is call i's unit u - base_i, its row block b and column block
   c at b · (n / C) + c for the window's R × C ``tile`` (vllm's z/y records' order), and the call's z table's leaf
   u - base_i.

Assumed, until the served window's draw arguments are fixed (PRF-132): the population is the window's units in vllm's
order (call, row block, column block), the key's stream is the window's root, its registration the draw root the coin
followed, and its receipt ``{"session": S}``, S the ledger's session for the window.
"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

from verity.ml.circuits.codec import descriptor_id
from verity.ml.randomness import derive
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.circuit import h1window as HW
from verity.core.protocols.pouw.circuit import plan as PL
from verity.core.protocols.pouw.circuit.partition import Unit
from verity.core.protocols.sampled_proofs import remote
from verity.core.protocols.sampled_proofs.challenger import CoinError, LiveOpening, ledger_head, pinned_session
from verity_vllm.protocol_options import pouw_h1_challenger as PHC


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


def layouts(calls: list[dict], tile: tuple[int, int]) -> list[PL.Layout]:
    """One layout a K: its template's stratum, every unit of its calls, each tile its own closure."""
    r, c = tile
    ks = sorted({x["k"] for x in calls})
    tid = {k: descriptor_id(HW.template(k, HW.tile_unit(k, tile))) for k in ks}
    n = sum(x["units"] for x in calls)
    kof = [0] * n
    for x in calls:
        kof[x["base"]:x["base"] + x["units"]] = [x["k"]] * x["units"]
    units = tuple(Unit(u, "tile", tid[kof[u]], 0, 0) for u in range(n))
    out = []
    for k in ks:
        mine = tuple(u for u in range(n) if kof[u] == k)
        st = PL.StratumSpec(tid[k], "tile", mine, r * c * k * len(mine))
        out.append(PL.Layout(None, units, (st,), {u: (u,) for u in mine}, w_ref=r * c * k))
    return out


def draw(key, lays: list[PL.Layout], k: int) -> dict:
    """The work law's draw over the window's layouts, in `core_check.unit_draw`'s form."""
    law = PL.WorkLaw(k, window_work=sum(lay.work() for lay in lays))
    strata, units = [], []
    for lay in lays:
        p = PL.draw(key, lay, law)
        s = lay.strata[0]
        strata.append({"template": s.name, "units": s.ranges(), "work": s.work // len(s.units),
                       "floor": law.floor(s.name), "k": p.sizes[s.name]})
        units += p.drawn
    d = {"law": "work", "k": k, "population": len(lays[0].units), "strata": strata, "units": sorted(units)}
    CC.unit_draw({"draw": d})
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--ledger", action="append", required=True)
    ap.add_argument("--pin", required=True)
    ap.add_argument("--pins", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--k", type=int, default=1)
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
        sid = pinned_session(lines, {"root": root}, key, pins, remote.main_lines(a.main), PHC.lms_verify).session
    except CoinError as e:
        raise SystemExit(f"REFUSED: the challenger's session is not trusted: {e}") from e
    if (why := PHC.draw_reason(commits, trusted, received.get(Wd.DRAW, ()))) is not None:
        raise SystemExit(f"REFUSED: {why}")
    if sum(c["units"] for c in calls) != commits["units"] or any(
            c["base"] != sum(x["units"] for x in calls[:j]) for j, c in enumerate(calls)):
        raise SystemExit("REFUSED: the calls' units are not the window's units in order")
    coin = LiveOpening.from_json(commits["draw"]["coin"]).coin
    receipt = {"session": sid}
    dkey = derive(coin, PL.DRAW_DOMAIN, CC.draw_key_context(root, commits["draw"]["root"], receipt))
    d = draw(dkey, layouts(calls, tile), a.k)
    out = {"window": root, "draw_root": commits["draw"]["root"], "coin": coin.hex(), "session": sid,
           "key": {"domain": PL.DRAW_DOMAIN, "stream": "the window's root", "registration": "the draw root",
                   "receipt": receipt},
           "draw": d, "units": [locate(calls, tile, u) for u in d["units"]]}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({**{k: out[k] for k in ("window", "draw_root", "coin", "session", "key")},
                      "law": f"work:{d['k']}", "population": d["population"],
                      "strata": [{"k_of_template": s["k"], "units": sum(hi - lo for lo, hi in s["units"]),
                                  "work": s["work"]} for s in d["strata"]],
                      "units": [{k: v for k, v in u.items() if k not in ("z_table", "a_table")} for u in out["units"]]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
