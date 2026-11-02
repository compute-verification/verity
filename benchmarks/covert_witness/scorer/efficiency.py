"""The final EFFICIENCY table (coordinator 16:35Z, 17:08Z): per (start, target), honest FLOPs over every valid v2.3
record's complete schedules (te_core 3.5.3 frontier.target, the 14:57Z rule), and each finalist's FLOPs to the target
on the final shard per seed (mean and spread), beside its search value. No rho, W* or W*/honest: pending te_core 4.0."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scorer as S  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/cursor/artifacts/te-final-efficiency-table.md")
rep = S.final_table(write=True)
target = S.te_pin("frontier").target
rows = {x["id"]: x for x in (json.loads(l) for l in (S.STATE / "index.jsonl").read_text().splitlines() if l.strip())}
res = [json.loads(l) for l in (S.FINAL / "results.jsonl").read_text().splitlines() if l.strip()]
launches = [json.loads(l) for l in (S.FINAL / "launches.jsonl").read_text().splitlines() if l.strip()]
ok = {r["run"] for r in rep["reruns"] if not r["problems"]}
subs = []
for r in res:
    x = rows.get(r.get("id"))
    if r.get("run") not in ok or not x:
        continue
    subs += [{"id": x["id"], "run": r["run"], "seed": r.get("seed"), "cp": round(t.get("fraction") or 0, 3),
              "start": x["start"], "macs": t["macs"], "val_bpb": t["bpb"], "rho": t["rho"], "direction": x["direction"],
              "reference": S.is_reference(x), "fork_ladder": S.fork_ladder(x)}
             for t in r["ledger"] if t.get("te_kind") == "submission" and t.get("rho") and t.get("valid")]
S.schedule_ends(subs, key=lambda s: s["run"])
search = [x for x in json.loads((S.STATE / "submissions.json").read_text()) if x.get("regime") == "current"]
l0, l0s = rep["final_starts"], rep["search_starts"]
fmt, short = S.fmt, S.short

planned: dict[str, dict] = {}
for la in launches:
    if la.get("kind") == "rerun" and la.get("id") in rows:
        planned.setdefault(la["id"], {})[la.get("seed")] = la.get("run")
for r in rep["reruns"]:
    planned.setdefault(r["id"], {})
state = {r["run"]: r["state"] for r in res if r.get("run")}
for rn in list(state):
    p = S.RUNS / rn / "status.json"
    if p.exists():
        state[rn] = json.loads(p.read_text()).get("state") or state[rn]
CANCELLED = {"r20261102-154002-5e28", "r20261102-154011-4e3b", "r20261102-154019-50ed", "r20261102-154028-2290",
             "r20261102-155000-ad4f", "r20261102-155007-ef3c", "r20261102-155015-c6de", "r20261102-155025-2520",
             "r20261102-160316-c0ac", "r20261102-160633-fb6b", "r20261102-160642-0381", "r20261102-160650-936c"}
for rn in CANCELLED & set(state):
    state[rn] = "cancelled 17:09Z (vy-nebius-2's GPU lease never granted)"
for r in rep["reruns"]:
    if r["problems"] in (["run running"], ["run failed"]) and state.get(r["run"]) != "running":
        r["problems"] = [f"run {state[r['run']]}"]
probs = {r["run"]: r["problems"] for r in rep["reruns"]}
superseded = {r["run"] for r in res if r.get("superseded")}
NOT_LAUNCHED = ["228a65b1c940", "ae5177c91dbf", "cce098238b2f", "5129d3cca07c"]
for e in NOT_LAUNCHED:
    full = next((i for i in rows if short(i) == f"art:{e}"), None)
    if full:
        planned.setdefault(full, {})


def reach(on, l_star):
    t = target(on, l_star, S.KEYS, reference=lambda x: x["reference"])
    return (t["honest"], t["bound"], (t["honest_of"] or {}).get("checkpoint")) if t["honest"] else (None, "", None)


def g(v):
    return f"{2 * v:.3g}"


out = {"at": S.now(), "frontier": rep["frontier"], "pin": rep["pin"], "final_starts": l0, "search_starts": l0s,
       "starts": {}}
md = [
    "# Training efficiency, first night: the final table (efficiency only)", "",
    f"Built {out['at']} by the scorer (`cursor/te-scorer-516a`, `benchmarks/covert_witness/scorer/efficiency.py`) on the coordinator's 16:35Z "
    "and 17:08Z lines.", "",
    "- **honest(l\\*)** is the least training FLOPs (2|C|) over every valid v2.3 record of any class or direction that "
    "reaches l\\* by a complete schedule (the 14:57Z rule), by `te_core.frontier.target` of te_core 3.5.3 "
    "(`cursor/te-core-313-on-pin-3a8f`, 34ef146f1c3). It is read on the search set, where every record was scored. "
    "A complete schedule is a run's last checkpoint or a fork of a ref-optim fork ladder.",
    "- **l\\*** is the start's bpb on that split times (1 − 0.5%, 1%, 2%, 5%). The search slices and the final shard "
    "have their own start bpb (below), so a finalist is compared at the same drop.",
    "- **A finalist's FLOPs to the target on the final shard** is, per seed (101, 102, 103), the least 2|C| over that "
    "rerun's complete schedules that reach the final shard's l\\*, by the same `target`. A cell shows the mean over the "
    "seeds that reach, the min–max where the seeds differ, and how many of the seeds run reach. A checkpoint's FLOPs "
    "don't depend on the seed, so the seeds differ only in which checkpoint reaches; the bpb spread is in each "
    "finalist's best drop. `search` is the same quantity on the search "
    "record (one seed). `≠` marks where the final shard disagrees with the search slices: a different reach, or FLOPs "
    "more than 5% apart (a different checkpoint).",
    "- A rerun counts only if it passed every check a search record passes, on the final split, with clean read-outs "
    "(te_core 3.5.3's `cw.readout_leaks`).",
    "- FLOPs are the pin's count (harness v2.3, te_core 3.5.1's op table, re-ingested from each run's own `run.json`).",
    "- **ρ, W\\* and W\\*/honest: pending te_core 4.0.** The pinned count defines updated MACs by changed weights, "
    "whereas T2 is about deep activations (θ's MACs on update-dependent activations count), and β was priced at p unless "
    "a family was declared. The toolsmith is re-scoring every v2.3 record under the dependence definition and a "
    "circuit-measured β (`cursor/te-core-dependence-4207`). This table prints none of them.", "",
    "| start | search start bpb | final-shard start bpb |", "|---|---|---|"]
md += [f"| {s} | {l0s.get(s, float('nan')):.4f} | {l0.get(s, float('nan')):.4f} |" for s in S.STARTS if s in l0s]
md.append("")

for s in S.STARTS:
    if s not in l0s:
        continue
    son = [x for x in search if x["start"] == s]
    fin = sorted({e for e in planned if rows[e]["start"] == s})
    st = {"targets": [], "finalists": {}}
    md += [f"## {s}", "", "| target | l\\* search / final | honest FLOPs (setter, direction) | least finalist on the final "
           "shard (seed mean) |", "|---|---|---|---|"]
    for i, f in enumerate(S.TARGETS):
        ts = S.target_row(son, f, l0s[s] * (1 - f))
        tf = rep["final"].get(s, {}).get("targets", [None] * 4)[i] if s in l0 else None
        st["targets"].append({"drop": f, "l_star_search": ts["l_star"], "l_star_final": l0[s] * (1 - f) if s in l0 else None,
                              "honest_flops": ts["honest_macs"] and 2 * ts["honest_macs"], "honest_id": ts["honest_id"],
                              "honest_cp": ts["honest_cp"], "honest_direction": ts["honest_direction"],
                              "bound": ts["bound"],
                              "final_least_flops": tf and tf["honest_macs"] and 2 * tf["honest_macs"],
                              "final_least_id": tf and tf["honest_id"], "final_least_cp": tf and tf["honest_cp"]})
        h = (f"{g(ts['honest_macs'])} ({short(ts['honest_id'])}@{ts['honest_cp']}, {ts['honest_direction']}"
             + (f", {ts['bound']}" if ts["bound"] else "") + ")") if ts["honest_macs"] else "no record reaches it"
        lf = (f"{g(tf['honest_macs'])} ({short(tf['honest_id'])}@{tf['honest_cp']})"
              + (f", {tf['bound']}" if tf["bound"] else "")) if tf and tf["honest_macs"] else "none reaches it"
        md.append(f"| −{100 * f:g}% | {ts['l_star']:.4f} / {l0[s] * (1 - f):.4f} | {h} | {lf} |")
    md += ["", "Each finalist's FLOPs to the target: final shard mean [min–max if the seeds differ] (seeds reaching/seeds "
           "run); search. Its best drop is the most its complete schedules fall below the split's start bpb: final mean "
           "[min–max over seeds] / search.", "",
           "| finalist | direction | best drop final / search | " + " | ".join(f"−{100 * f:g}%" for f in S.TARGETS)
           + " |", "|---|---|---|" + "---|" * len(S.TARGETS)]
    for e in fin:
        seeds = sorted({x["seed"] for x in subs if x["id"] == e})
        row = {"direction": rows[e]["direction"], "seeds": seeds, "cells": []}
        sbest = min((x["val_bpb"] for x in son if x["id"] == e and x.get("schedule_end")), default=None)
        fbest = [min(x["val_bpb"] for x in subs if x["id"] == e and x["seed"] == sd and x.get("schedule_end"))
                 for sd in seeds]
        sd_ = sbest and 100 * (1 - sbest / l0s[s])
        fd = [100 * (1 - b / l0[s]) for b in fbest]
        row |= {"best_drop_search_pct": sd_, "best_drop_final_pct": fd}
        drop = ((f"{sum(fd) / len(fd):.2f}% [{min(fd):.2f}–{max(fd):.2f}]" if fd else "—")
                + (f" / {sd_:.2f}%" if sd_ is not None else " / —"))
        cells = []
        for f in S.TARGETS:
            ls, lf = l0s[s] * (1 - f), l0[s] * (1 - f)
            sv, sb, scp = reach([x for x in son if x["id"] == e], ls)
            per = {sd: reach([x for x in subs if x["id"] == e and x["seed"] == sd], lf) for sd in seeds}
            hit = [v for v, _, _ in per.values() if v]
            c = {"search_flops": sv and 2 * sv, "search_bound": sb, "search_cp": scp,
                 "final_flops": {sd: v and 2 * v for sd, (v, _, _) in per.items()},
                 "final_bound": sorted({b for _, b, _ in per.values() if b})}
            dis = ""
            if seeds:
                if bool(sv) != bool(hit) or (sv and len(hit) < len(seeds)):
                    dis = "≠"
                elif sv and hit and abs(sum(hit) / len(hit) / sv - 1) > 0.05:
                    dis = "≠"
            c["disagrees"] = bool(dis)
            row["cells"].append(c)
            spread = f" [{g(min(hit))}–{g(max(hit))}]" if hit and max(hit) > min(hit) * 1.0001 else ""
            fs = ("—" if not seeds else f"{g(sum(hit) / len(hit))}{spread} ({len(hit)}/{len(seeds)})"
                  if hit else f"none (0/{len(seeds)})")
            fs += " ub" if c["final_bound"] else ""
            cells.append(f"{fs}; search {g(sv) if sv else 'none'}{' ub' if sb else ''} {dis}".strip())
        runs = planned.get(e, {})
        miss = [f"{sd} {', '.join(probs.get(rn) or [state.get(rn, 'not collected')])}" for sd, rn in sorted(runs.items())
                if rn not in superseded and sd not in seeds]
        miss += [f"{sd} not launched" for sd in S.FINAL_SEEDS if sd not in seeds and sd not in runs]
        if len(seeds) < len(S.FINAL_SEEDS):
            row["missing_seeds"] = miss
        st["finalists"][e] = row
        tag = "" if not miss else (" (no final rerun: " if not seeds else " (seeds missing: ") + ", ".join(miss) + ")"
        md.append(f"| {short(e)}{tag} | {rows[e]['direction']} | {drop} | " + " | ".join(cells) + " |")
    out["starts"][s] = st
    md.append("")

bad = [r for r in rep["reruns"] if r["problems"]]
md += ["## Reruns", "",
       f"- {len(ok)} reruns valid on the final shard ({len({r['id'] for r in rep['reruns'] if not r['problems']})} "
       f"entries), {len(bad)} with problems: "
       + ("; ".join(f"{short(r['id'])} seed {r['seed']} ({r['run']}): {', '.join(r['problems'])}" for r in bad) or "none")
       + ".",
       f"- {len(rep['superseded'])} superseded (a later rerun of the same entry and seed replaced them).",
       "- `ub` marks an upper bound: only a mid-schedule checkpoint reaches the target (te_core's 'upper bound "
       "(checkpoint)').", ""]
out["reruns"] = rep["reruns"]
OUT.write_text("\n".join(md) + "\n")
OUT.with_suffix(".json").write_text(json.dumps(out, indent=1, default=str) + "\n")
print(OUT)
