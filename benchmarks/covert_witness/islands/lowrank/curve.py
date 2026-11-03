"""The lowrank direction's A2 points from its collected runs: per run, |E| (Delta), Delta_grad (the fp32 gradient and the
optimizer's fp32 state; the straight-through fp32 latents beside it as `+lat`), the start and, for each
target drop (-0.5, -1, -2, -5% of the start's bpb), the least training FLOPs (2 x MACs) of a checkpoint reaching it, marked
`c` when that checkpoint ends the run's schedule (a complete schedule) and `k` when it is inside one (an upper bound).

    python curve.py [--state ~/.research/lowrank/v3/state.jsonl ~/.research/lowrank/night2/state.jsonl ...] [--json]

Reads each run's fetched `out/` under ~/.research/runs (launch.py --collect puts it there), and keeps the fields it reads
in ~/.research/lowrank/cache/RUN.json, so a run whose big local files `research data custody RUN --publish` removed
still has its point. A tool of the agent VM, never run by the harness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path.home() / ".research" / "runs"
CACHE = Path.home() / ".research" / "lowrank" / "cache"
TARGETS = (0.5, 1.0, 2.0, 5.0)


def load(run: str) -> tuple[dict, dict] | None:
    out, cached = RUNS / run / "out", CACHE / f"{run}.json"
    try:
        rj = json.loads((out / "run" / "run.json").read_text())
        meta = json.loads((out / "meta.json").read_text())
    except (OSError, ValueError):
        if not cached.exists():
            return None
        c = json.loads(cached.read_text())
        return c["run"], c["meta"]
    keep = ("step", "macs", "val_bpb", "e_bytes")
    island = rj.get("island") or meta.get("class")
    rj = {"invalid": rj.get("invalid"),
          "checkpoints": [{**{k: c.get(k) for k in keep}, "score": {"start_bpb": (c.get("score") or {}).get("start_bpb")}}
                          for c in rj.get("checkpoints", [])]}
    meta = {"start": meta["start"], "island": island,
            "notes": {"delta_grad": meta.get("notes", {}).get("delta_grad", {})}}
    if not cached.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps({"run": rj, "meta": meta}))
    return rj, meta


def point(run: str, note: str) -> dict | None:
    got = load(run)
    if got is None:
        return None
    rj, meta = got
    cks = [c for c in rj.get("checkpoints", []) if c.get("val_bpb") is not None]
    if not cks or rj.get("invalid"):
        return None
    l0 = cks[0]["score"]["start_bpb"]
    last = max(c["step"] for c in cks)
    notes = meta.get("notes", {})
    best = {}
    for t in TARGETS:
        hit = [c for c in cks if c["val_bpb"] <= l0 * (1 - t / 100)]
        if hit:
            c = min(hit, key=lambda c: c["macs"])
            best[t] = (2 * c["macs"], "c" if c["step"] == last else "k")
    fin = max(cks, key=lambda c: c["step"])
    dg = notes.get("delta_grad", {})
    state = sum(dg.get(k, 0) for k in ("grad_bits", "adam_bits", "muon_bits")) // 8 if dg else None
    return {"run": run, "start": meta["start"], "island": meta.get("island") or meta.get("class"),
            "mechanism": note.split(":")[0].split()[0],
            "E_bytes": max(c.get("e_bytes", 0) for c in cks), "delta_grad_bytes": state,
            "latent_bytes": dg.get("latent_bits", 0) // 8 if dg else None,
            "l0": l0, "final_bpb": fin["val_bpb"], "final_drop_pct": 100 * (1 - fin["val_bpb"] / l0),
            "final_flops": 2 * fin["macs"], "best_drop_pct": 100 * (1 - min(c["val_bpb"] for c in cks) / l0),
            "targets": {str(t): v for t, v in best.items()}, "note": note}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state", type=Path, nargs="*",
                    default=[Path.home() / ".research" / "lowrank" / n / "state.jsonl" for n in ("v3", "night2")])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    pts, runs = [], set()
    for st in (s for s in a.state if s.exists()):
        for x in st.read_text().splitlines():
            s = json.loads(x)
            if "row" in s and s.get("run") and s["run"] not in runs:
                p = point(s["run"], s["key"].split(";")[1].strip())
                if p:
                    pts.append(p)
                    runs.add(s["run"])
    pts.sort(key=lambda p: (p["start"], p["E_bytes"]))
    if a.json:
        print(json.dumps(pts, indent=1))
        return 0
    for p in pts:
        cells = " ".join(f"-{t}%:" + (f"{p['targets'][str(t)][0]:.2e}{p['targets'][str(t)][1]}" if str(t) in p["targets"]
                                      else "-") for t in TARGETS)
        dg = f"{p['delta_grad_bytes'] / 1e6:.1f}MB" if p["delta_grad_bytes"] else "?"
        lat = f"{p['latent_bytes'] / 1e6:.1f}MB" if p["latent_bytes"] else "-"
        print(f"{p['run']} {p['start']} {p['island']:6} E {p['E_bytes'] / 1e3:7.1f}kB dgrad {dg:>7} +lat {lat:>6} "
              f"final {p['final_bpb']:.4f} ({-p['final_drop_pct']:+.2f}%) at {p['final_flops']:.2e} "
              f"best {-p['best_drop_pct']:+.2f}%  {cells}  {p['mechanism']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
