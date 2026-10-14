"""A run's sessions in a few lines, from its `verifier.json` and `prover.json` (`run.sh`'s outputs): each honest
session's verdict, γ terms and costs, and for each skip fraction F the sessions accepted against the rate the draws
predict, Π_t C(U_t − F·N·L_t, s) / C(U_t, s) over the windows drawn, and the draws that landed on a dropped block.

    python summary.py OUT [--json]
    python summary.py OUT OUT ... [--json]      the skip sessions of several runs, pooled
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def miss(units: int, bad: int, s: int) -> float:
    """The chance that s distinct draws of `units` miss all `bad` of them."""
    if bad > units - min(s, units):
        return 0.0
    return math.exp(sum(math.log1p(-bad / (units - k)) for k in range(min(s, units))))


def summarize(out: Path) -> dict:
    v = json.loads((out / "verifier.json").read_text())
    p = json.loads((out / "prover.json").read_text())
    N, s = v["registration"]["blocks"], None
    honest, skips = [], defaultdict(list)
    for vs, ps in zip(v["sessions"], p["sessions"]):
        s = vs["samples"]
        passes = [w["passes"] for w in vs["windows_seen"]]
        row = {"k": vs["k"], "accepted": vs["accepted"], "failure": vs["failure"], "passes": passes,
               "verify_cpu_s": sum(vs["verify_cpu_ns"].values()) / 1e9, "draws": sum(len(d) for d in vs["draws"].values()),
               "on_dropped": sum(o.get("on_dropped", 0) for o in ps["opened"]),
               "opening_bytes": sum(o["bytes"] for o in ps["opened"]), "device": ps.get("device")}
        if ps["mode"] == "honest":
            lg = vs.get("ledger", {})
            row.update({"gamma_spec": lg.get("spec", {}).get("gamma"), "gamma_measured": lg.get("measured", {}).get("gamma"),
                        "terms_spec": lg.get("spec", {}).get("terms"), "horizon_spec": lg.get("spec", {}).get("horizon"),
                        "inputs": lg.get("inputs"), "sketch_failure_log2": lg.get("sketch_failure", {}).get("log2"),
                        "slack_ms_min": min(w["slack_ns"] for w in vs["windows_seen"] if w["slack_ns"] is not None) / 1e6,
                        "passes_per_s": sum(passes) / (vs["windows"] * vs["window_ns"] / 1e9)})
            honest.append(row)
        else:
            dropped = len(ps["dropped"])
            # The chance this session is accepted: every one of its W windows drawn clean, at the passes each made
            # (a window a rejected session never reached, at the mean of those it did).
            mean = round(sum(passes) / len(passes))
            row["p_accept"] = math.prod(miss(L * N, L * dropped, s)
                                        for L in passes + [mean] * (vs["windows"] - len(passes)))
            row["dropped"] = dropped
            if vs["failure"] is not None:
                at = vs["failure"]["at"]
                row["rejected_on_dropped"] = at is not None and at[1] in set(ps["dropped"])
                row["rejected_window"] = vs["failure"]["t"]
            skips[ps["fraction"]].append(row)
    table = []
    for f, rows in sorted(skips.items()):
        n, acc = len(rows), sum(r["accepted"] for r in rows)
        expect = sum(r["p_accept"] for r in rows)
        var = sum(r["p_accept"] * (1 - r["p_accept"]) for r in rows)
        draws, hits = sum(r["draws"] for r in rows), sum(r["on_dropped"] for r in rows)
        table.append({"fraction": f, "log2_fraction": math.log2(f), "dropped_blocks": rows[0]["dropped"], "sessions": n,
                      "accepted": acc, "expected_accepted": expect, "sd": math.sqrt(var),
                      "predicted_reject_rate": 1 - expect / n, "observed_reject_rate": 1 - acc / n,
                      "draws": draws, "draws_on_dropped": hits, "hit_rate": hits / draws if draws else None,
                      "rejections_all_on_dropped": all(r.get("rejected_on_dropped", True) for r in rows),
                      "failures": sorted({r["failure"]["failure"] for r in rows if r["failure"]}),
                      "rejected_windows": [r.get("rejected_window") for r in rows], "rows": rows})
    return {"blocks": N, "bytes": N * 4096, "samples": s, "calibration": v["calibration"], "rtt": v["rtt"],
            "allowance_ms": v["allowance_ns"] / 1e6, "verifier_process_cpu_s": v["process_cpu_ns"] / 1e9,
            "wire": v["wire"], "honest": honest, "skip": table}


def pooled(summaries: list[dict]) -> list[dict]:
    """The skip rows of several runs' summaries pooled by fraction, sessions whose rejection was not at a draw (a late
    root) left out of the rate and counted apart."""
    by = defaultdict(lambda: {"sessions": 0, "accepted": 0, "expected": 0.0, "var": 0.0, "draws": 0, "hits": 0,
                              "other_failures": 0})
    for r in summaries:
        for t in r["skip"]:
            b = by[t["fraction"]]
            b["draws"] += t["draws"]
            b["hits"] += t["draws_on_dropped"]
            for row in t["rows"]:
                if row["failure"] is not None and row["failure"]["failure"] != "receipt":
                    b["other_failures"] += 1
                    continue
                b["sessions"] += 1
                b["accepted"] += row["accepted"]
                b["expected"] += row["p_accept"]
                b["var"] += row["p_accept"] * (1 - row["p_accept"])
    out = []
    for f, b in sorted(by.items()):
        n, sd = b["sessions"], math.sqrt(b["var"])
        out.append({"fraction": f, "log2_fraction": math.log2(f), "sessions": n, "accepted": b["accepted"],
                    "rejected": n - b["accepted"], "predicted_rejected": n - b["expected"], "sd": sd,
                    "z": (b["expected"] - b["accepted"]) / sd if sd else None,
                    "draws": b["draws"], "draws_on_dropped": b["hits"], "expected_on_dropped": b["draws"] * f,
                    "hit_rate": b["hits"] / b["draws"], "other_failures": b["other_failures"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, nargs="+")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if len(a.out) > 1:
        rows = pooled([summarize(o) for o in a.out])
        if a.json:
            print(json.dumps(rows, indent=1))
        for t in rows:
            print(f"skip 2^{t['log2_fraction']:.0f}: {t['rejected']}/{t['sessions']} rejected, predicted "
                  f"{t['predicted_rejected']:.2f} ± {t['sd']:.2f} (z = {t['z']:.2f}); draws on dropped "
                  f"{t['draws_on_dropped']}/{t['draws']} against {t['expected_on_dropped']:.1f} expected; "
                  f"{t['other_failures']} rejected otherwise")
        return 0
    r = summarize(a.out[0])
    if a.json:
        print(json.dumps(r, indent=1))
        return 0
    print(f"store {r['bytes'] / 2 ** 30:.0f} GiB ({r['blocks']} blocks), s = {r['samples']}, A = {r['allowance_ms']:.2f} ms")
    for h in r["honest"]:
        print(f"honest {h['k']}: accepted={h['accepted']} passes/s={h['passes_per_s']:.1f} gamma(spec)={h['gamma_spec']} "
              f"slack_min={h['slack_ms_min']:.2f} ms verify_cpu={h['verify_cpu_s']:.2f} s")
    for t in r["skip"]:
        print(f"skip 2^{t['log2_fraction']:.0f} ({t['dropped_blocks']} blocks): {t['accepted']}/{t['sessions']} accepted, "
              f"expected {t['expected_accepted']:.2f} ± {t['sd']:.2f}; draws on dropped {t['draws_on_dropped']}/{t['draws']} "
              f"= {t['hit_rate']:.3g} (F = {t['fraction']:.3g}); every rejection on a dropped block: "
              f"{t['rejections_all_on_dropped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
