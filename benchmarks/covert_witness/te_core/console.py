"""The panels cut to the console's panels route's limits, and one panel's PUT.

    python -m te_core.console DIR    # DIR: te_core.plots' panels; writes DIR/console/

A body is at most `BODY_MAX` bytes (compact JSON, no NaN) and a note at most `NOTE_MAX` characters. Each panel keeps
its first cut that fits (`CUTS`, in order, the first the whole panel) with numbers to 4 significant digits (losses 6),
and its note is the panel's short note (`SHORT`), the cut's, and `COMMON`; the uncut panel is DIR/<name>.json beside
it. The console's one training panel is `training/frontier-branches` (te_core.panels, which PUTs it with `put`):
te_core.plots' five panels are off it, so nothing here PUTs them. `put` PUTs one panel to `PANELS_URL`/<id> with a
Cursor OIDC token (research.slack's `oidc_token`, imported only there) and returns its HTTP status.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from .plots import NAMES, frame

PANELS_URL = "https://website-docs-sage.vercel.app/api/panels"
BODY_MAX = 65536
NOTE_MAX = 500
CELL_MAX = 200
REF_TARGET = -1.0
BAND_FEW = ("c1e15/k1e5", "c1e15/k1e6", "c1e14/k1e4")
LOSSES = ("loss", "loss_lo", "loss_hi", "l0")
COMMON = ("Formalism v2: a frozen tail always charged. basis: fit = rho c/(c-d) by Fit; target = Slowdown's bound, "
          "unproved, from A1/A1' upper bounds. Delta, u(1), a: witness upper bounds; tau a stub. One seed. Uncut: the "
          "panel's source.")
SHORT = {
    "p1-loss-vs-compute": "Loss (val bpb) against compute after theta over every checkpoint of every valid v2.3 record: "
                          "honest |C| (gates), covert |C| rho at c=4096w k=10kB, covert-policy at other (c, k).",
    "p2-slowdown-vs-gain": "S = W*/H at c=4096w k=10kB per start and target, H and W* over every checkpoint (S >= 1); "
                           "x0, s_star: the claim's bounds; *_known: the attack side (tightness). g null until F.",
    "p3-policy-plane": "S = W*/H over te_core.policy's (c, k) grid and each start's marked shapes, k in bits, H and W* "
                       "over every checkpoint of every valid v2.3 record.",
    "p4-decomposition": "Per algorithm, its least-|C|rho checkpoint per target at c=4096w k=10kB: p = |C|/H, s = rho, "
                        "S = p s; x0, s_star bounds, *_known the attack side.",
    "a2-efficiency-floor": "Least compute to each target among checkpoints with Delta <= delta_bits; fit_*_k_bits: the "
                           "least k where Fit holds at c=4096w (running theta', training).",
}


def body(p: dict) -> bytes:
    return json.dumps(p, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode()


def refused(p: dict) -> list[str]:
    """What the route refuses in `p`'s rows: a row has one cell per column, each a number, a string of at most
    `CELL_MAX` characters, or null (a boolean is none of them)."""
    def ok(v):
        return v is None or isinstance(v, (int, float)) and not isinstance(v, bool) or \
            isinstance(v, str) and len(v) <= CELL_MAX
    cols, out = p["columns"], []
    for i, r in enumerate(p["rows"]):
        if len(r) != len(cols):
            out.append(f"row {i} has {len(r)} cells, not {len(cols)}")
        out += [f"row {i}'s {c} is {v!r:.40}" for c, v in zip(cols, r) if not ok(v)]
    return out


def compact(p: dict, keep=None, drop: tuple = (), pick=None) -> dict:
    """`p` with the rows `keep` takes, of those the ones `pick` picks, without the columns `drop`, numbers to 4
    significant digits (losses 6)."""
    cols = [c for c in p["columns"] if c not in drop]

    def val(c, v):
        if isinstance(v, float):
            return round(v, 6) if c in LOSSES else float(f"{v:.4g}")
        return v
    ds = [d for d in frame(p) if keep is None or keep(d)]
    ds = pick(ds) if pick else ds
    return p | {"columns": cols, "rows": [[val(c, d[c]) for c in cols] for d in ds]}


def thinned(step: float):
    """A pick: each P1 curve at the points that lower its loss by `step` or reach a deeper target, its ends kept."""
    def pick(ds: list[dict]) -> list[dict]:
        groups: dict[tuple, list] = defaultdict(list)
        for d in ds:
            groups[(d["start_id"], d["curve"], d["policy"])].append(d)
        out = []
        for g in groups.values():
            kept = [g[0]]
            for i, d in enumerate(g[1:], 1):
                if i == len(g) - 1 or d["target_pct"] != kept[-1]["target_pct"] or \
                        d["loss"] <= kept[-1]["loss"] * (1 - step):
                    kept.append(d)
            out += kept
        return out
    return pick


def front(x: str, y: str):
    """A pick: per start and target, the rows no other beats in both `x` and `y` (smaller is better), in order."""
    def pick(ds: list[dict]) -> list[dict]:
        groups: dict[tuple, list] = defaultdict(list)
        for i, d in enumerate(ds):
            groups[(d["start_id"], d["target_pct"])].append(i)
        kept = set()
        for ix in groups.values():
            best = float("inf")
            for i in sorted(ix, key=lambda i: (ds[i][x], ds[i][y])):
                if ds[i][y] is not None and ds[i][y] < best:
                    best = ds[i][y]
                    kept.add(i)
        return [d for i, d in enumerate(ds) if i in kept]
    return pick


def band(policies: tuple):
    return lambda d: d["curve"] != "covert-policy" or d["policy"] in policies


def ref(d: dict) -> bool:
    return d["target_pct"] == REF_TARGET


LEAN = ("loss_lo", "loss_hi")
CUTS = {
    "p1-loss-vs-compute": [
        ("", None, LEAN, None),
        (" Points 0.25% apart in loss.", None, LEAN, thinned(0.0025)),
        (f" Band: {', '.join(BAND_FEW)}; points 0.25% apart.", band(BAND_FEW), LEAN, thinned(0.0025)),
        (" No band; points 0.25% apart.", band(()), LEAN, thinned(0.0025)),
        (" No band; points 0.5% apart.", band(()), LEAN, thinned(0.005)),
        (" No band; points 1% apart.", band(()), (*LEAN, "start", "checkpoint"), thinned(0.01))],
    "p3-policy-plane": [
        ("", None, (), None),
        (" Here -1% only.", ref, (), None),
        (" Here -1% only, no record, g, k_bytes or p (= S/rho).", ref, ("record", "g", "k_bytes", "p"), None)],
    "p4-decomposition": [
        ("", None, (), None),
        (" Per start and target, the (p, s) front only.", None, (), front("p", "s"))],
    "a2-efficiency-floor": [("", None, (), None)],
    "p2-slowdown-vs-gain": [("", None, (), None)],
}


def cut(p: dict) -> dict:
    """`p` cut to the route's limits: its first cut (`CUTS`) whose body fits, with the console's note."""
    name = p["id"].split("/", 1)[1]
    for why, keep, drop, pick in CUTS[name]:
        q = compact(p, keep, drop, pick)
        note = f"{SHORT[name]}{why} {COMMON}"
        q["note"] = note if len(note) <= NOTE_MAX else note[:NOTE_MAX - 1] + "…"
        if len(body(q)) <= BODY_MAX:
            return q
    raise ValueError(f"{p['id']}: no cut fits {BODY_MAX} bytes (the last is {len(body(q))})")


def write(src: Path, out: Path) -> dict[str, dict]:
    """Each of `src`'s panels (te_core.plots' DIR/<name>.json) cut, written to `out`/<name>.json."""
    out.mkdir(parents=True, exist_ok=True)
    made = {}
    for name in NAMES:
        made[name] = cut(json.loads((src / f"{name}.json").read_text()))
        (out / f"{name}.json").write_bytes(body(made[name]))
    return made


def put(p: dict) -> str:
    import urllib.error
    import urllib.request

    from research.slack import oidc_token
    req = urllib.request.Request(f"{PANELS_URL}/{p['id']}", body(p), method="PUT",  # noqa: S310 (a fixed https URL)
                                 headers={"Authorization": f"Bearer {oidc_token(PANELS_URL)}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            return str(r.status)
    except urllib.error.HTTPError as e:
        return f"{e.code} {e.read()[:400].decode(errors='replace')}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir", type=Path, help="te_core.plots' panels")
    a = ap.parse_args(argv)
    for p in write(a.dir, a.dir / "console").values():
        print(f"{p['id']}: {len(body(p))} bytes, {len(p['rows'])} rows, note {len(p['note'])} characters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
