"""Summarize hcpre_phase.py's rows with the locked scorer's arithmetic: `python3 hcpre_phase_score.py [--scoring DIR] OUT...`.

Each part's row goes through `scoring.point_stats` (DIR holds the scorer's `scoring.py`, default this directory). Printed per
part: the paired slowdown (and the worst round's), both sides' median ms, the filler's rate over the tensor peak at the
alone clock, beside/alone and receipts. Then three prefill entries through `scoring.coverage`, at the scorer's MACs:

- `v2 (lin)`: the scorer's own entry (`prefill_entry`) from the `lin` row: measured linears, FA2 and non-GEMM constants
  stretched by the linears' slowdown and credited at the linears' filler rate;
- `lin + both`: the linears and the rest of the step measured as two parts, times and filler work summed;
- `step`: the whole step measured in one graph, the filler's work over it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def entry(S, st_parts):
    t0 = sum(st["alone_ms"] for st in st_parts)
    t1 = sum(st["arm_ms"] for st in st_parts)
    f = sum(st["fill_rate"] * st["arm_ms"] for st in st_parts) * 1e-3
    mhz = sum(st["alone_mhz"] * st["alone_ms"] for st in st_parts) / t0
    n = S.PREFILL_M * S.share.LIN_TOK + S.share.causal(S.PREFILL_M)
    return S.coverage({"t0_ms": t0, "t1_ms": t1, "serving_macs": n, "filler_macs": f, "mhz": mhz,
                       "slowdowns": {"prefill": t1 / t0}})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", default=str(Path(__file__).resolve().parent))
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.scoring)
    import scoring as S

    print(f"scorer {S.SCORER}")
    for d in a.dirs:
        rows = [json.loads(ln) for ln in (Path(d) / "phase.jsonl").read_text().splitlines() if ln.startswith("{")]
        for r in rows:
            if r.get("mode") == "categories":
                print("categories (ms alone): " + ", ".join(f"{k} {v:.2f}" for k, v in r["ms"].items()))
        st = {}
        print("| part | shape | slowdown (rounds max) | alone ms | arm ms | filler of peak | beside/alone | receipts (bad) |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r.get("mode") != "dscore":
                continue
            p = r["kind"].removeprefix("phase-")
            s = st[p] = S.point_stats(r)
            fp = s["fill_rate"] / S.cap(s["alone_mhz"])
            bo = s["beside_over_alone"]
            print(f"| {p} | W{r['W']} NC{r['NC']} S{r['S']} 2^{r['chunk_log2']} map {r['map']} | {s['slowdown']:.4f} "
                  f"({max(s['slowdown_rounds']):.4f}) | {s['alone_ms']:.2f} | {s['arm_ms']:.2f} | {fp:.3f} | "
                  f"{'–' if bo is None else f'{bo:.3f}'} | {s['rec_checked']} ({s['rec_bad']}) |")
        print("| prefill entry | slowdown | coverage | filler share |")
        print("|---|---|---|---|")
        out = []
        if "lin" in st:
            out.append(("v2 (lin)", S.coverage(S.prefill_entry(st["lin"]))))
        if "lin" in st and "both" in st:
            out.append(("lin + both", entry(S, [st["lin"], st["both"]])))
        if "step" in st:
            out.append(("step", entry(S, [st["step"]])))
        for name, e in out:
            print(f"| {name} | {e['time']:.4f} | {e['coverage']:.3f} | {e['filler_share']:.3f} |")


if __name__ == "__main__":
    main()
