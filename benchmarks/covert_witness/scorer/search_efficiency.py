"""The search's efficiency table, kept current during a night (coordinator 19:15Z, night 2): per start and target, H(l*),
the least 2|C| over every checkpoint of every valid v2.3 record (the coordinator's 19:50Z rule, which replaced the
complete-schedule rule of 14:42Z and 14:57Z; te_core 3.5.3 frontier.target with every checkpoint counted as an end), and
per Delta bin (the record's class) and per direction the least compute to the target with p = |C|/H. No rho, W* or S:
the v1 quantities come from the analytic re-score (benchmarks/covert_witness/analysis/)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scorer as S

NIGHT2 = "2026-11-02T19:15Z"
BINS = ("e100kb", "e1mb", "shallow", "reference", "seed")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/cursor/artifacts/te-night2-efficiency.md")


def night(row: dict | None) -> str:
    return "n2" if row and (row.get("launched") or "") >= NIGHT2 else "n1"


def best(on: list[dict], l_star: float) -> dict | None:
    t = S.te_pin("frontier").target([x | {"schedule_end": True} for x in on], l_star, S.KEYS,
                                    reference=lambda x: x["reference"])
    if not t["honest"]:
        return None
    o = t["honest_of"]
    return {"macs": t["honest"], "id": o["id"], "cp": o["checkpoint"], "bound": t["bound"]}


def main() -> dict:
    rows = {x["id"]: x for x in (json.loads(l) for l in (S.STATE / "index.jsonl").read_text().splitlines() if l.strip())}
    subs = [x for x in json.loads((S.STATE / "submissions.json").read_text()) if x.get("regime") == "current"]
    for x in subs:
        r = rows.get(x["id"]) or {}
        x["bin"], x["night"] = r.get("class") or "?", night(r)
    l0 = {s: v for s in S.STARTS if (v := S.regime_start_bpb(subs, s)) is not None}
    n2 = sorted({x["id"] for x in subs if x["night"] == "n2"})
    out = {"at": S.now(), "frontier": f"te_core {S.te_pin('').VERSION} ({S.TE_PIN})", "pin": S.pin_of(), "starts": l0,
           "records": len({x["id"] for x in subs}), "night2_records": len(n2), "table": {}}
    md = ["# Training efficiency, night 2: the search's efficiency table", "",
          f"Built {out['at']} by the scorer (`benchmarks/covert_witness/scorer/search_efficiency.py` on "
          f"`cursor/te-scorer-516a`). It covers {out['records']} valid v2.3 records (pin 11d96c1b874 and its "
          f"measurement commit), {len(n2)} of them launched in night 2 (from 19:15Z Mon).", "",
          "- **H(l\\*)** is the least training FLOPs (2|C|) over every valid v2.3 record of either night that reaches "
          "l\\* = start × (1 − 0.5%, 1%, 2%, 5%) at any checkpoint (the coordinator's 19:50Z rule), by te_core "
          "3.5.3's `frontier.target`.",
          "- **Per Δ bin** (the record's class: e100kb Δ ≤ 100 kB, e1mb ≤ 1 MB, shallow, reference any Δ) the least "
          "FLOPs to the target at any checkpoint, and p = that over H. A bin's best is an upper bound on A2's "
          "curve at that Δ until the v1 Δ of each record replaces the class.",
          "- `n2` marks a night-2 record.",
          "- **ρ, W\\* and S are not here**: they come from the v1 re-score (Fit, the Slowdown theorem's ρ as target), "
          "not the harness's retired ρ.", ""]
    for s, v in l0.items():
        on = [x for x in subs if x["start"] == s]
        md += [f"## {s} (start {v:.4f} bpb)", "",
               "| target | l\\* | H FLOPs (setter, direction) | " + " | ".join(BINS) + " |",
               "|---|---|---|" + "---|" * len(BINS)]
        rows_s = []
        for f in S.TARGETS:
            ls = v * (1 - f)
            h = best(on, ls)
            hx = h and next(x for x in on if x["id"] == h["id"] and x["cp"] == h["cp"])
            cell = {"drop": f, "l_star": ls, "H": h and 2 * h["macs"], "H_id": h and h["id"], "H_cp": h and h["cp"],
                    "H_direction": hx and hx["direction"], "H_night": hx and hx["night"], "H_bound": h and h["bound"],
                    "bins": {}, "directions": {}}
            for b in BINS:
                bb = best([x for x in on if x["bin"] == b], ls)
                cell["bins"][b] = bb and {"flops": 2 * bb["macs"], "id": bb["id"], "cp": bb["cp"], "bound": bb["bound"],
                                          "p": bb["macs"] / h["macs"]}
            for d in sorted({x["direction"] for x in on}):
                bd = best([x for x in on if x["direction"] == d], ls)
                cell["directions"][d] = bd and {"flops": 2 * bd["macs"], "id": bd["id"], "cp": bd["cp"],
                                                "bound": bd["bound"], "p": bd["macs"] / h["macs"]}
            rows_s.append(cell)

            def show(c):
                if not c:
                    return "—"
                tag = " n2" if next(x for x in on if x["id"] == c["id"])["night"] == "n2" else ""
                return f"{c['flops']:.3g} (p {c['p']:.2f}, {S.short(c['id'])}{tag})"
            hs = (f"{cell['H']:.3g} ({S.short(h['id'])}@{h['cp']}, {cell['H_direction']}"
                  + (" n2" if cell["H_night"] == "n2" else "") + ")") if h else "none"
            md.append(f"| −{100 * f:g}% | {ls:.4f} | {hs} | " + " | ".join(show(cell["bins"][b]) for b in BINS) + " |")
        out["table"][s] = rows_s
        md.append("")
    md += ["## Each direction's least FLOPs to each target (p = over H)", "",
           "| start | direction | " + " | ".join(f"−{100 * f:g}%" for f in S.TARGETS) + " |",
           "|---|---|" + "---|" * len(S.TARGETS)]
    for s, rs in out["table"].items():
        for d in sorted({d for c in rs for d in c["directions"]}):
            cs = [c["directions"].get(d) for c in rs]
            if any(cs):
                md.append(f"| {s} | {d} | " + " | ".join(
                    f"{c['flops']:.3g} (p {c['p']:.2f})" if c else "—" for c in cs) + " |")
    OUT.write_text("\n".join(md) + "\n")
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


if __name__ == "__main__":
    print(main()["night2_records"])
