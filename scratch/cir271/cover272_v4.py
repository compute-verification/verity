"""cover272_v4.py COVER.json PLAN.json RUNS_DIR RUN[:LIST[:NODE[:running]]]...: CIR-272's second pass, `circuits/request-cover/v4`,
written over COVER.json, which must be v3 (`art:de7ccd2119367c0264ca57c1be3d88774345f1ea13a8f91baf09817c4a188888`, the record
NET-390 quoted; `git show 9a0d813ec490:scratch/cir271/cover.json`).

PLAN.json is p2.py's (scratch/cir272/p2-plan.json on cursor/circuits-cir272-bound-6f04): for each of the 159 tags v3 left
refused or not run (141 of the card's types and the 18 extras) its estimate, whether it runs, and its list. Each RUN is a
second-pass run of bound.sh, its files fetched under RUNS_DIR/RUN/out; ``running`` marks one that had not ended when the record
was written. A tag's second-pass outcome, from its own run's files:

    accepted          B.verify.held exit 0 and accepted true, and B.verify.no-program and B.verify.other-def exit 1 and accepted
                      false; state own-definition or boolean-version (stage.json's definition against the block's)
    held-control-off  B.verify.held accepted but a control did not refuse
    refused           the verifier (ks-form, statement or verify) or the stager refused; `text` holds their own lines
    not-run           why: estimate-past-64GB (p2.py: its stager over 62 GB or its Lean check over 60 GB by the estimate, or a
                      measurement that did; not staged), not-reached (its run's budget), past-limit (its own limit, `timeout`),
                      stage-past-62GB (the stager stopped), lean-gate-past-60GB (staged; 4.42 x circuit GiB over 60 GB, the Lean
                      checks not run), lean-past-62GB (a Lean check stopped past 62 GB resident)
    running           its run had not ended

The v3 record of a second-pass tag is kept under its ``v3`` key. bound_totals counts the card's 254 types by outcome, and
bound_extras the 18 extras; cover_check.py recomputes both."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

TREE = "33d7c7dfd8c23f5db0c93584aeb7a96391defde9"
CHANGES = [
    "form.py: the list as run stages census:<the lookup's boolean_version> where names.jsonl gives a zk_refusal",
    "session: serve and prove get --tables DIR when the stage wrote tables/",
    "a row's fifth column is its own limit in seconds, which the budget check uses too",
    "each Lean check is stopped past 62 GB resident, as the stager is",
    "Lean checks run when 4.42 x circuit.txt's GiB is at most 60 GB (job.sh's 6 x + 4 GB is logged beside it)",
]

cover_path, plan_path, runs_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
cover, plan = json.loads(cover_path.read_text()), json.loads(plan_path.read_text())
assert cover["schema"] == "circuits/request-cover/v3", cover["schema"]
RUNS = []
for a in sys.argv[4:]:
    run, lst, node, *st = a.split(":") + ["", ""]
    RUNS.append({"run": run, "list": lst, "node": node, "state": "running" if "running" in st else "ended"})
J = lambda p: json.loads(p.read_text()) if p.exists() else None  # noqa: E731


def tm(p: Path) -> dict:
    s = p.read_text(errors="replace") if p.exists() else ""
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", s)
    e = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", s)
    secs = round(sum(float(v) * 60 ** i for i, v in enumerate(reversed(e[1].split(":")))), 1) if e else None
    return {"maxrss_gb": round(int(m[1]) / 2 ** 20, 2) if m else None, "wall_s": secs}


def v_of(v: dict, k: str) -> dict | None:
    r = v.get(k)
    if r is None:
        return None
    vd = r.get("verdict") or {}
    return {"exit": r["exit"], "accepted": vd.get("accepted"), "why": vd.get("why") or r.get("setup") or r.get("ks_form") or r.get("said"),
            "units": ((vd.get("partition") or {}) if isinstance(vd.get("partition"), dict) else {}).get("note", "")[:200] or None,
            "wall": r.get("wall"), "maxrss_gb": round(r["maxrss_kb"] / 2 ** 20, 2) if r.get("maxrss_kb") else None}


by_block = {b["block"]: b for b in cover["blocks"]}
T = cover["bound_runs"]["types"]
assert sorted(plan) == sorted(t for t, r in T.items() if r["outcome"] != "accepted"), "the plan is not v3's 159 tags"
seen: dict[str, dict] = {}
for R in RUNS:
    out = runs_dir / R["run"] / "out"
    log = (out / "job.log").read_text(errors="replace").splitlines()
    nr = {l.split("\t")[0]: l.split("\t")[2] for l in (out / "not-reached.tsv").read_text().splitlines() if l} \
        if (out / "not-reached.tsv").exists() and R["state"] == "ended" else {}
    form = {l.split("\t")[0]: l.split("\t") for l in (out / "form.tsv").read_text().splitlines() if l}
    names = {n["tag"]: n for n in map(json.loads, (out / "names.jsonl").read_text().splitlines()) if n.get("tag")}
    for tag in [l.split("\t")[0] for l in (out / "types.tsv").read_text().splitlines() if l]:
        b = by_block[plan[tag]["block"]]
        ce = names[tag].get("census") or {}
        r = {"run": R["run"], "node": R["node"], "list": R["list"], "tree": TREE, "staged_unit": form[tag][2],
             "lookup": {"kind": ce.get("kind"), "zk_refusal": ce.get("zk_refusal"), "error": ce.get("error") or names[tag].get("error"),
                        "boolean_version": ce.get("boolean_version")},
             "lines": [l[:600] for l in log if re.match(rf"^\S+ {tag}: ", l) or l.startswith(f"STAGED {tag} ")
                       or (l.startswith("VERDICTS ") and l.split(" ")[1] == tag)]}
        d = out / tag
        if tag in nr:
            r.update(outcome="not-run", why="not-reached", text=nr[tag])
        elif not (d / "done").exists():
            assert R["state"] == "running", (R["run"], tag)
            r.update(outcome="running", text="started" if d.exists() else "not started")
        else:
            done, st, v = J(d / "done"), J(d / "B" / "stage.json"), J(d / "verdicts.json") or {}
            err = (d / "B" / "stage.err").read_text(errors="replace").strip().splitlines() if (d / "B" / "stage.err").exists() else []
            stopped = sorted(p.read_text().strip() for p in (d / "lean").glob("*.stopped")) if (d / "lean").exists() else []
            r.update(done=done, stage={"definition": st and st["definition"], "word": st and st["word"], "ands": st and st.get("ands"),
                                       "k_log": st and st.get("k_log"), "circuit_bytes": st and st.get("circuit_bytes"),
                                       "statement": st and st.get("statement"), **tm(d / "B" / "stage.time")},
                     verdicts={k: v_of(v, k) for k in ("B.statement.held", "B.statement.no-program", "B.statement.other-def", "B.ks-form",
                                                        "B.verify.held", "B.verify.no-program", "B.verify.other-def")})
            h, np_, od, ks = (r["verdicts"][k] for k in ("B.verify.held", "B.verify.no-program", "B.verify.other-def", "B.ks-form"))
            if h and h["exit"] == 0 and h["accepted"] is True:
                ok = all(c and c["exit"] == 1 and c["accepted"] is False for c in (np_, od))
                r["outcome"] = "accepted" if ok else "held-control-off"
                sd = st["definition"]
                r["state"] = "own-definition" if sd == b["definition"] else "boolean-version" if sd == b["boolean_version"] else "other"
                assert r["state"] != "other", (tag, sd, b["definition"], b["boolean_version"])
            elif done["timeout"]:
                r.update(outcome="not-run", why="past-limit", text=(d / "timeout").read_text().strip())
            elif (d / "lean-not-run").exists():
                r.update(outcome="not-run", why="lean-gate-past-60GB", lean_need_gb=int((d / "lean-not-run").read_text()),
                         text=next(l for l in r["lines"] if "Lean not run" in l))
            elif any(l.startswith("stopped: the stager's resident memory") for l in err):
                r.update(outcome="not-run", why="stage-past-62GB", text=next(l for l in err if l.startswith("stopped:")))
            elif stopped:
                r.update(outcome="not-run", why="lean-past-62GB", text="; ".join(stopped))
            else:
                texts = [f"ks-form: {ks['why']}"] if ks and ks["exit"] != 0 else []
                texts += [l.split(": ", 1)[1] for l in r["lines"] if "session FAILED" in l or "staging refused" in l]
                if h and h["accepted"] is False:
                    texts.append(f"verify.held: {h['why']}")
                if not st:
                    texts.append(f"stager: {err[-1] if err else ''}")
                r.update(outcome="refused", text=texts)
        assert tag not in seen or seen[tag]["outcome"] == "not-run" and seen[tag]["why"] == "not-reached", tag
        seen[tag] = r

for tag, p in plan.items():
    old = T[tag]
    if not p["run"]:
        r = {"outcome": "not-run", "why": "estimate-past-64GB", "text": p["estimate"]["basis"]}
    else:
        assert tag in seen, f"{tag} is planned to run but in none of the runs given"
        r = seen[tag]
    T[tag] = {**{k: old[k] for k in ("tag", "block", "blocks", "k_log", "extra")}, **r, "estimate": p["estimate"], "plan": {
        k: p.get(k) for k in ("run", "order", "list", "limit_s")}, "v3": {k: old.get(k) for k in ("run", "outcome", "why", "text", "state",
                                                                                           "verdicts", "stage", "needs", "zk_refusal")}}

for b in cover["blocks"]:
    r = next((x for x in T.values() if x["block"] == b["block"]), None)
    if r is None:
        continue
    b["cir272"] = {k: r[k] for k in ("tag", "run", "outcome", "state", "why", "text") if k in r}
    if b["on_card"]:
        o = r["outcome"]
        b["bound_outcome"] = (f"accepted-{r['state']}" if o == "accepted" else f"not-run-{r['why']}" if o == "not-run" else o)
on = [b for b in cover["blocks"] if b["on_card"]]
total = sum(b["blocks"] for b in on)
cnt = Counter(b["bound_outcome"] for b in on)
cover["schema"] = "circuits/request-cover/v4"
cover["second_pass"] = {"issue": "CIR-272", "tree": TREE, "branch": "cursor/circuits-cir272-bound-6f04", "plan": "scratch/cir272/p2-plan.json",
                        "runs": RUNS, "threads": 8, "mem_gb": 64, "stage_stop_gb": 62, "lean_gate_gb": 60, "lean_stop_gb": 62,
                        "budget_s": 21000, "runner_changes": CHANGES, "v3": "art:de7ccd2119367c0264ca57c1be3d88774345f1ea13a8f91baf09817c4a188888"}
cover["bound_totals"] = {s: [n, sum(b["blocks"] for b in on if b["bound_outcome"] == s)] for s, n in sorted(cnt.items())}
groups = {"accepted": [s for s in cnt if s.startswith("accepted")], "refused": ["refused"], "held-control-off": ["held-control-off"],
          "not-run": [s for s in cnt if s.startswith("not-run")], "running": ["running"]}
cover["bound_shares"] = {g: {"types": sum(cnt[s] for s in ss), "blocks": (n := sum(b["blocks"] for b in on if b["bound_outcome"] in ss)),
                             "share": round(n / total, 6)} for g, ss in groups.items()}
ex = Counter(f"{r['outcome']}{'/' + r['why'] if r.get('why') else ''}" for r in T.values() if r["extra"])
cover["bound_extras"] = dict(sorted(ex.items()))
cover_path.write_text(json.dumps(cover, indent=1) + "\n")
print(json.dumps({"bound_totals": cover["bound_totals"], "bound_shares": cover["bound_shares"], "extras": cover["bound_extras"]}, indent=1))
