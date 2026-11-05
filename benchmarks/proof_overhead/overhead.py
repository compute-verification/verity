"""Proof overhead against a proof block's output width, and what the exfiltration bound makes of each curve.

    python overhead.py --flock RUN_OUT [RUN_OUT ...] [--card h200-sxm-141gb] --out DIR

Each ``RUN_OUT`` is a `benchmarks/flock_s8_block` run's ``out/``, a statement's fields merged across them (a ``STEPS=gpu`` run's
from the run that staged it); their ``STATEMENTS`` hold two instance counts for each shape:
``K:N:R`` (a block of one int8 row against R rows, packed, every instance's own rows: R·K MACs, 32·R output bits) or
``K:N:PxC`` (a tile of C rows of x against P rows of W, one u16 leaf an s8: P·C·K MACs, 32·P·C output bits). Per shape:

- **Measured.** The per-block GPU seconds are the slope between the two statements' median prove times; the overhead is those
  seconds times the card's peak s8 MACs a second, over the block's MACs.
- **The layout.** Flock's prover pays for cells: a block of 2^k_log cells per instance (`layout`, `verity_flock.circuit`'s
  rule, checked against each record's k_log) holding the rows' SHA-512 compressions (``sha512x3`` slots, three a 2^18 slot,
  each row's L padded bytes L / 128 of them after its constant prefix block), one ``hm96`` slot of 2^18 a row (its last
  compression, the salt's two and ``b || c``), the units' slots (2^ceil(log2 useful rows) each) and a 2^14 mask. The
  commitment's share is its slots' share of the used cells. One rate, seconds per block cell, is fitted over every shape in
  relative error, and each shape's residual against it is reported.
- **The floor.** The units' own slots at the fitted rate, the commitment opened for free: what any commitment scheme can at
  best reach with this prover. ``served_pcs`` measures what a linear-claim commitment's opening and serving actually cost.
- **Exfiltration.** The 3-pager's model (`exfil_bound`) at each shape: proof blocks of 2·MACs FLOP and the block's output bits
  at the curve's overhead, one year of world inference, a 1% verification budget.

Writes ``overhead.json`` and, with matplotlib, ``overhead_vs_width.png``, ``share_vs_width.png`` and ``exfil_vs_width.png``
to DIR.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SLOT_LOG, MASK_LOG, PER_SLOT = 18, 14, 3


def _summary(out: Path) -> dict:
    spec = importlib.util.spec_from_file_location("s8_summary", ROOT / "benchmarks/flock_s8_block/summary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main(out)


def peak_macs(card: str) -> float:
    sys.path.insert(0, str(ROOT))
    from verity import hardware
    c = hardware.load(card)
    return c["rate"]["tensor"]["value"] * c["sm"]["count"]["value"] * c["clock"]["sm"]["value"] * 1e6


def _pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


def layout(stage: dict) -> dict:
    """One instance's cells by kind, from a staged record's ``stage`` (its ports' u16 words, its one output port's ``w_out``
    words, a unit's useful rows, its tile and ``vus_per_block``), placed as `verity_flock.circuit._ranges` places them."""
    rows = [2 * w for w in stage["ports"]] + [2 * stage["w_out"]]
    comps = sum(-(-b // 128) for b in rows)
    units = math.prod(stage["tile"]) if stage.get("tile") else 1
    ul = max(7, (stage["unit_rows"] - 1).bit_length())
    g = stage["vus_per_block"]
    sha, hm = -(-g * comps // PER_SLOT), _pow2(g * len(rows))
    at = 0
    for count, sl in sorted([(hm, SLOT_LOG), (g * units, ul), (1, MASK_LOG)], key=lambda t: -(_pow2(t[0]) << t[1])):
        span = _pow2(count) << sl
        at = -(-at // span) * span + span
    at = (-(-at >> SLOT_LOG) << SLOT_LOG) + (sha << SLOT_LOG)
    k_log = max(20, (at - 1).bit_length())
    return {"k_log": k_log, "block": (1 << k_log) / g, "sha": (sha << SLOT_LOG) / g, "hm96": (hm << SLOT_LOG) / g,
            "unit": units << ul, "compressions": comps}


SHAPE = re.compile(r"k(\d+)-n(\d+)-(?:r(\d+)|t(\d+)x(\d+))")


def blocks(summary: dict) -> list[dict]:
    by: dict[tuple, list] = {}
    for name, st in summary["statements"].items():
        m = SHAPE.fullmatch(name)
        t = (st.get("gpu") or {}).get("prove_total_s")
        if m and t is not None:
            k, n, r, p, c = (int(x) if x else None for x in m.groups())
            by.setdefault((k, r, p, c), []).append((n, t, st))
    out = []
    for (k, r, p, c), pts in sorted(by.items(), key=lambda kv: tuple(x or 0 for x in kv[0])):
        if len(pts) < 2:
            continue
        (n1, t1, st), (n2, t2, _) = sorted(pts, key=lambda x: x[0])[0], sorted(pts, key=lambda x: x[0])[-1]
        stage = st["stage"]
        if p is not None and stage.get("tile") != [p, c]:
            raise SystemExit(f"overhead: k{k} t{p}x{c} staged as tile {stage.get('tile')}, not the shape its name says")
        lay = layout(stage)
        if lay["k_log"] != stage["k_log"]:
            raise SystemExit(f"overhead: k{k} {r or f'{p}x{c}'}: the layout model gives k_log {lay['k_log']}, the record "
                             f"{stage['k_log']}; bring `layout` in line with verity_flock.circuit")
        outs = r if r is not None else p * c
        per = (t2 - t1) / (n2 - n1)
        out.append({"k": k, "shape": f"r{r}" if r is not None else f"t{p}x{c}", "family": "rows" if r is not None else "tiles",
                    "width_bits": 32 * outs, "macs": outs * k, "block_s": per, "fixed_s": t1 - per * n1, "n": [n1, n2],
                    "prove_s": [t1, t2], "ands": stage["ands"], **{f"cells_{x}": lay[x] for x in ("block", "sha", "hm96", "unit")},
                    "k_log": lay["k_log"], "compressions": lay["compressions"]})
    return out


# The 3-pager's model (its Figure 1): all inference worldwide for a year, DeepSeek-V4-Pro at 4,500 tokens a request, a
# 2.5 MB output cap a request, the verifier's replay share q and per-unit proving share s within a budget beta, the
# adversary's best corruption at a 1% evasion threshold over the horizon.
FLOP_REQ = 4500 * 2 * 49e9
REQUESTS = int(1e16 * 104 / 49)
W_REQ = 20_000_000
EVADE = math.log(100)


def _lgc(n, k):
    k = np.asarray(k, dtype=float)
    return (math.lgamma(n + 1) - np.vectorize(math.lgamma)(k + 1) - np.vectorize(math.lgamma)(n - k + 1)) / math.log(2)


def _hull(x, y):
    h = []
    for p in sorted(zip(x, y)):
        while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) >= 0:
            h.pop()
        h.append(p)
    return np.array(h)


def exfil_bound(flop_pu: float, bits: int, h: float, beta: float = 0.01) -> float:
    """The most bytes the adversary exfiltrates undetected, at the verifier's best replay share."""
    n = int(FLOP_REQ / flop_pu)
    mgrid = np.unique(np.r_[np.round(np.geomspace(1, n, 400)), n])
    js = np.unique(np.r_[np.round(np.geomspace(1, REQUESTS, 4000)), REQUESTS])
    lgc_n, lgc_r = _lgc(n, mgrid), _lgc(REQUESTS, js)
    value = np.minimum(W_REQ, lgc_n + bits * mgrid)
    best = math.inf
    for q in beta * np.geomspace(1e-4, 1 - 1e-4, 60):
        s = min(1.0, (beta - q) / (q * h))
        rho = (np.full_like(mgrid, -math.log1p(-q)) if s >= 1
               else -np.log1p(q * np.expm1(mgrid * math.log1p(-s))))
        hull = _hull(rho, value)
        xb = EVADE / js
        ok = xb >= hull[0, 0]
        env = np.interp(xb, hull[:, 0], hull[:, 1])
        best = min(best, min(REQUESTS * W_REQ / 8, float(((js * env + lgc_r)[ok]).max()) / 8))
    return best


LABELS = {"rows": "1 token × R weight rows, s8 packed", "tiles": "T tokens × T weight rows, one u16 leaf an s8"}


def _plot(a, bs: list[dict], res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = [(f, [b for b in bs if b["family"] == f and (f == "rows" or b["shape"].split("x")[0][1:] == b["shape"].split("x")[1])])
            for f in ("rows", "tiles")]
    odd = [b for b in bs if b["family"] == "tiles" and b["shape"].split("x")[0][1:] != b["shape"].split("x")[1]]

    def curves(ax, key, scale=1.0, floor=None):
        for (f, fb), mk in zip(fams, "os"):
            if not fb:
                continue
            w = [b["width_bits"] for b in fb]
            line, = ax.loglog(w, [b[key] * scale for b in fb], mk + "-", label=f"{LABELS[f]}, K = {fb[0]['k']}: measured")
            if floor:
                ax.loglog(w, [b[floor] * scale for b in fb], mk + "--", color=line.get_color(), alpha=0.6,
                          label=f"{LABELS[f]}: commitment free (floor)")
        for b in odd:
            ax.loglog([b["width_bits"]], [b[key] * scale], "D", color="k", label=f"tile {b['shape'][1:]} (tokens × rows = C × P)")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    curves(ax, "overhead", floor="overhead_floor")
    ax.set_xlabel("Proof block output width (bits)")
    ax.set_ylabel("Proof overhead (GPU-s × peak s8 MAC/s ÷ block MACs)")
    ax.set_title(f"C-Flock --zk, hm96 rows hashed in the circuit, {a.card}")
    ax.legend(fontsize=7)
    fig.savefig(a.out / "overhead_vs_width.png", dpi=200, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (f, fb), mk in zip(fams, "os"):
        if fb:
            ax.semilogx([b["width_bits"] for b in fb], [100 * b["commit_share"] for b in fb], mk + "-", label=LABELS[f])
    for b in odd:
        ax.semilogx([b["width_bits"]], [100 * b["commit_share"]], "D", color="k", label=f"tile {b['shape'][1:]}")
    ax.set_xlabel("Proof block output width (bits)")
    ax.set_ylabel("Commitment's share of a block's used cells (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Where a block's witness goes: SHA-512 and hm96 slots vs the block's own units")
    ax.legend(fontsize=7)
    fig.savefig(a.out / "share_vs_width.png", dpi=200, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    curves(ax, "exfil_bytes", 1e-9, floor="exfil_bytes_floor")
    ax.set_xlabel("Proof block output width (bits)")
    ax.set_ylabel("Exfiltration capacity, one year of world inference (GB)")
    ax.set_title("3-pager model at a 1% verification budget")
    ax.legend(fontsize=7)
    fig.savefig(a.out / "exfil_vs_width.png", dpi=200, bbox_inches="tight")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--flock", type=Path, nargs="+", required=True, help="one or more flock_s8_block runs' out/")
    p.add_argument("--card", default="h200-sxm-141gb")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    peak = peak_macs(a.card)
    merged: dict[str, dict] = {}
    for out in a.flock:  # a STEPS=gpu run's statements carry no stage record: another run's out/ does
        for name, st in _summary(out)["statements"].items():
            merged.setdefault(name, {}).update({k: v for k, v in st.items() if v is not None})
    bs = blocks({"statements": merged})
    if len(bs) < 2:
        print("overhead: fewer than two shapes with two instance counts each", file=sys.stderr)
        return 1
    v = np.array([b["cells_block"] / b["block_s"] for b in bs])
    per_cell = float(v.sum() / (v @ v))  # least squares in relative error, so a small block counts as much as a large one
    for b in bs:
        used = b["cells_sha"] + b["cells_hm96"] + b["cells_unit"]
        b["residual"] = b["block_s"] / (per_cell * b["cells_block"]) - 1
        b["overhead"] = b["block_s"] * peak / b["macs"]
        b["commit_share"] = (b["cells_sha"] + b["cells_hm96"]) / used
        b["fill"] = used / b["cells_block"]
        b["overhead_floor"] = per_cell * b["cells_unit"] * peak / b["macs"]
        b["exfil_bytes"] = exfil_bound(2 * b["macs"], b["width_bits"], b["overhead"])
        b["exfil_bytes_floor"] = exfil_bound(2 * b["macs"], b["width_bits"], b["overhead_floor"])
    res = {"card": a.card, "peak_s8_macs_per_s": peak, "s_per_block_cell": per_cell, "blocks": bs}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "overhead.json").write_text(json.dumps(res, indent=1))
    if importlib.util.find_spec("matplotlib") is None:
        print("overhead: matplotlib isn't installed, so no plots", file=sys.stderr)
        return 0
    _plot(a, bs, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
