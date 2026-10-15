"""The ephemeral γ terms at the checkpoint's depth, from fetched runs: per model, mode, batch and context, the plain
lm_head's share of a step's reads and of R·T, the rest's (everything unregistered but the lm_head) eager and fused, μ_eph
= rest / (R·T), writes over reads, and μ_eph with the writes' bus time counted at R, (rest + writes) / (R·T).

    uv run python benchmarks/ephemeral/gamma.py OUT_DIR... [--json PATH]

Each OUT_DIR is a run's `out/` (`research fetch RUN --all`).  Its profiled arms are attributed afresh (`attribute.py`),
and its timed arms (`<model>-L0-<mode>-time.json`) give T and R.  A point's bytes come from the best arm that has it: one
at full depth, then per CUDA graph rather than per kernel; a later OUT_DIR wins a tie.  Bytes from a per-kernel arm are
marked `*`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attribute as A  # noqa: E402

MODES = {"0": "eager", "c": "fused"}


def at_depth(p: dict) -> tuple[float, float, float]:
    """A point's reads, rest and writes at the checkpoint's depth."""
    f = p.get("full_depth")
    if f:
        return f["read"], f["ephemeral_ex_lm"], f["write"]
    return p["read"], p["unreg_encoded_lm"], p["write"]


def collect(dirs: list[Path]) -> tuple[dict, dict, list[str]]:
    best, times, fails = {}, {}, []
    for d in dirs:
        result, f, _ = A.attribute_dir(d)
        fails += [f"{d}: {x}" for x in f]
        for p in result:
            if "fail" in p:
                continue
            model, depth, mode, how = p["tag"].split("-")
            key = (model, MODES[mode], p["batch"], p["context"])
            rank = (depth != "L0", how != "graph")
            if key not in best or rank <= best[key][0]:
                best[key] = (rank, p)
        for tf in sorted(d.glob("*-L0-*-time.json")):
            model, _, mode, _ = tf.stem.split("-")
            arm = json.loads(tf.read_text())
            times[(model, MODES[mode])] = ({(q["batch"], q["context"]): q["step_ms"] / 1e3 for q in arm["points"]
                                            if "step_ms" in q}, arm["read_bytes_per_s"], arm["budget"]["lm_head_bytes"])
    return best, times, fails


def rows_of(best: dict, times: dict) -> list[dict]:
    out = []
    for model, _, b, ctx in sorted({(k[0], "", k[2], k[3]) for k in best}, key=lambda k: (-len(k[0]), k[0], k[2], k[3])):
        row = {"model": model, "batch": b, "context": ctx}
        for mode in MODES.values():
            hit = best.get((model, mode, b, ctx))
            if not hit:
                continue
            rd, rest, wr = at_depth(hit[1])
            t, R, lm = times.get((model, mode), ({}, None, hit[1]["lm_head"]))
            T = t.get((b, ctx))
            row[mode] = {"source": hit[1]["tag"], "per_kernel": hit[0][1], "read": rd, "rest": rest, "write": wr,
                         "lm_head": lm, "T_s": T, "R": R, "rest_share": rest / rd, "lm_share": lm / rd,
                         "lm_time": lm / (R * T) if T else None, "mu_eph": rest / (R * T) if T else None,
                         "mu_eph_writes": (rest + wr) / (R * T) if T else None, "write_over_read": wr / rd}
        out.append(row)
    return out


def pc(x: float | None, d: int = 2) -> str:
    return "–" if x is None else f"{x:.{d}%}"


def table(rows: list[dict], model: str) -> str:
    lines = ["| B | ctx | T eager / fused (ms) | plain lm_head, % of reads (% of R·T) | rest, eager | rest, fused | "
             "μ_eph eager / fused (% of R·T) | writes / reads, eager / fused | μ_eph with writes, eager / fused |",
             "| --- " * 9 + "|"]
    for r in rows:
        if r["model"] != model:
            continue
        e, f = r.get("eager", {}), r.get("fused", {})
        ts = [x.get("T_s") for x in (e, f)]
        t = " / ".join(f"{x * 1e3:.2f}" if x else "–" for x in ts) if any(ts) else "–"
        base = e or f
        lm = pc(base["lm_share"], 1) + (f" ({pc(base['lm_time'], 1)})" if base.get("lm_time") else "")
        rest = [f"{pc(x['rest_share'])}{'*' if x['per_kernel'] else ''} ({x['rest'] / 1e6:.0f} MB)" if x else "–"
                for x in (e, f)]
        mu, muw = ((" / ".join(pc(x.get(k), 3) for x in (e, f)) if any(x.get(k) for x in (e, f)) else "–")
                   for k in ("mu_eph", "mu_eph_writes"))
        wr = " / ".join(pc(x["write_over_read"], 1) if x else "–" for x in (e, f))
        lines.append(f"| {r['batch']} | {r['context'] // 1024}k | {t} | {lm} | {rest[0]} | {rest[1]} | {mu} | {wr} | "
                     f"{muw} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    best, times, fails = collect(a.dirs)
    rows = rows_of(best, times)
    for model in dict.fromkeys(r["model"] for r in rows):
        print(f"\n**{model}**\n\n{table(rows, model)}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))
    for x in fails:
        print("FAIL", x)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
