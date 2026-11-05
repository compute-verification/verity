"""Proof overhead against a proof block's output width, per proof system and commitment scheme, and what the exfiltration
bound makes of each curve.

    python overhead.py --flock RUN_OUT [--served-pcs JSONL] [--card h200-sxm-141gb] --out DIR

``RUN_OUT`` is a `benchmarks/flock_s8_block` run's ``out/`` whose ``STATEMENTS`` hold two instance counts at each rows-per-block
R (``K:N:R``, one K). A block of R rows computes R·K int8 MACs and outputs 32·R bits. Per R:

- **Flock, hm96 rows in the circuit (measured).** The per-block GPU seconds are the slope between the two statements' median
  prove times, and the overhead is those seconds times the card's peak s8 MACs a second, over the block's MACs.
- **The split.** The block's own ANDs are its record's ``ands``. Its hm96 openings' ANDs are modeled from the row format:
  each instance's rows (its activation row of K bytes, its weight row of R·K bytes, its output row of 4R bytes; packed words,
  own rows) cost ceil((L + 145) / 128) + 6 SHA-512 compressions of `SHA512_ANDS` each. A two-rate least-squares fit of the
  per-block seconds over both counts gives seconds per AND of each kind.
- **Flock, a commitment it opens by linear claims (modeled).** The block's own ANDs at the fitted rate, plus a per-block
  opening cost: ``served_pcs``'s marginal per-claim opening seconds (CPU) over ``--cpu-per-gpu``, or 0 without its file.
- **Exfiltration.** The 3-pager's model (`exfil_bound`) at each width: proof units of 2·R·K FLOP and 32·R output bits at the
  curve's overhead, one year of world inference, a 1% verification budget.

Writes ``overhead.json`` and, with matplotlib, ``overhead_vs_width.png`` and ``exfil_vs_width.png`` to DIR.
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
SHA512_ANDS = 58_120


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


def hm96_compressions(row_bytes: int) -> int:
    return math.ceil((row_bytes + 145) / 128) + 6


def hash_ands(k: int, r: int) -> int:
    return SHA512_ANDS * sum(hm96_compressions(b) for b in (k, r * k, 4 * r))


def blocks(summary: dict) -> list[dict]:
    by_r: dict[tuple[int, int], list] = {}
    for name, st in summary["statements"].items():
        m = re.fullmatch(r"k(\d+)-n(\d+)-r(\d+)", name)
        t = (st.get("gpu") or {}).get("prove_total_s")
        if m and t is not None:
            k, n, r = map(int, m.groups())
            by_r.setdefault((k, r), []).append((n, t, st))
    out = []
    for (k, r), pts in sorted(by_r.items()):
        if len(pts) < 2:
            continue
        (n1, t1, st), (n2, t2, _) = sorted(pts)[0], sorted(pts)[-1]
        per = (t2 - t1) / (n2 - n1)
        out.append({"k": k, "rows": r, "width_bits": 32 * r, "macs": r * k, "block_s": per, "fixed_s": t1 - per * n1,
                    "ands": st.get("ands"), "hash_ands": hash_ands(k, r), "n": [n1, n2], "prove_s": [t1, t2]})
    return out


def fit_rates(bs: list[dict]) -> tuple[float, float]:
    a = np.array([[b["ands"], b["hash_ands"]] for b in bs], dtype=float)
    y = np.array([b["block_s"] for b in bs])
    sol, *_ = np.linalg.lstsq(a, y, rcond=None)
    return float(max(sol[0], 0.0)), float(max(sol[1], 0.0))


def open_cost(path: Path | None, cpu_per_gpu: float) -> float:
    """The marginal per-claim opening seconds from served_pcs (the largest window, its lowest rate), on the GPU."""
    if path is None:
        return 0.0
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip().startswith("{")]
    pcs = [r for r in rows if r["kind"] == "pcs"]
    m = max(r["m"] for r in pcs)
    rate = min(r["log_inv_rate"] for r in pcs if r["m"] == m)
    at = {r["claims"]: r["open_s"] for r in pcs if r["m"] == m and r["log_inv_rate"] == rate}
    lo, hi = min(at), max(at)
    return max(at[hi] - at[lo], 0.0) / (hi - lo) / cpu_per_gpu if hi > lo else 0.0


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--flock", type=Path, required=True)
    p.add_argument("--served-pcs", type=Path)
    p.add_argument("--card", default="h200-sxm-141gb")
    p.add_argument("--cpu-per-gpu", type=float, default=48.0, help="CPU over GPU prove time for the same work (d826: 0.205 s / 4.23 ms)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    peak = peak_macs(a.card)
    bs = blocks(_summary(a.flock))
    if len(bs) < 2:
        print("overhead: fewer than two widths with two instance counts each", file=sys.stderr)
        return 1
    per_and, per_hash_and = fit_rates(bs)
    opening = open_cost(a.served_pcs, a.cpu_per_gpu)
    for b in bs:
        b["overhead_hm96"] = b["block_s"] * peak / b["macs"]
        b["hash_share"] = b["hash_ands"] * per_hash_and / (b["ands"] * per_and + b["hash_ands"] * per_hash_and)
        b["block_s_native"] = b["ands"] * per_and + opening
        b["overhead_native"] = b["block_s_native"] * peak / b["macs"]
        b["exfil_bytes_hm96"] = exfil_bound(2 * b["macs"], b["width_bits"], b["overhead_hm96"])
        b["exfil_bytes_native"] = exfil_bound(2 * b["macs"], b["width_bits"], b["overhead_native"])
    res = {"card": a.card, "peak_s8_macs_per_s": peak, "s_per_and": per_and, "s_per_hash_and": per_hash_and,
           "native_opening_s_per_block": opening, "blocks": bs}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "overhead.json").write_text(json.dumps(res, indent=1))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0
    w = [b["width_bits"] for b in bs]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(w, [b["overhead_hm96"] for b in bs], "o-", label="Flock, hm96 rows opened in the circuit (measured)")
    ax.loglog(w, [b["overhead_native"] for b in bs], "s--", label="Flock, commitment opened by linear claims (modeled)")
    ax.set_xlabel("Proof block output width (bits)")
    ax.set_ylabel("Proof overhead (GPU-seconds × peak MAC/s ÷ block MACs)")
    ax.set_title(f"int8 matmul blocks, K = {bs[0]['k']}, {a.card}")
    ax.legend(fontsize=8)
    fig.savefig(a.out / "overhead_vs_width.png", dpi=200, bbox_inches="tight")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(w, [b["exfil_bytes_hm96"] / 1e9 for b in bs], "o-", label="Flock, hm96 in the circuit")
    ax.loglog(w, [b["exfil_bytes_native"] / 1e9 for b in bs], "s--", label="Flock, linear-claim commitment (modeled)")
    ax.set_xlabel("Proof block output width (bits)")
    ax.set_ylabel("Exfiltration capacity, one year of world inference (GB)")
    ax.set_title("3-pager model at a 1% verification budget")
    ax.legend(fontsize=8)
    fig.savefig(a.out / "exfil_vs_width.png", dpi=200, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
