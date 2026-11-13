"""cover272_v4.py [--v5|--v6|--v7] COVER.json PLAN.json RUNS_DIR RUN:LIST:NODE:MEM_GB[:running]... (in launch order): CIR-272's second pass,
(with --v5 and scratch/cir272/p3-plan.json, the third pass's lists i and j too, as `circuits/request-cover/v5`; with --v6 and p4-plan.json, list k's extras too, under `beside_extras`, as v6; with --v7 and p5-plan.json, the fifth pass's Boolean versions of the 74 types not run, beside them, with `fifth_pass.by_family`, as v7), `circuits/request-cover/v4`,
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
                      checks not run), lean-past-62GB (a Lean check stopped past 62 GB resident), session-past-ulimit,
                      cancelled (its run, given as RUN:LIST:NODE:MEM_GB:cancelled, was cancelled before the tag ended)
    running           its run had not ended

The v3 record of a second-pass tag is kept under its ``v3`` key. A tag that a job with more memory ran again after a smaller
job's limit stopped it takes the larger job's record, the smaller's kept under ``smaller_jobs``. bound_totals counts the card's 254 types by outcome, and
bound_extras the 18 extras; cover_check.py recomputes both."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

TREE = "33d7c7dfd8c23f5db0c93584aeb7a96391defde9"  # runs a and b; c and d at 82a2979f9601, f and g at a03c53314c89, h at 9b8895ab4a16 (scratch/cir272 only)
CHANGES = [
    "form.py: the list as run stages census:<the lookup's boolean_version> where names.jsonl gives a zk_refusal",
    "session: serve and prove get --tables DIR when the stage wrote tables/",
    "a row's fifth column is its own limit in seconds, which the budget check uses too",
    "each Lean check is stopped past 62 GB resident, as the stager is",
    "Lean checks run when 4.42 x circuit.txt's GiB is at most 60 GB (job.sh's 6 x + 4 GB is logged beside it)",
]
LISTS = {
    "a": "p2.py: half of the types v3 left, cheapest first (scratch/cir272/types-p2a.tsv)",
    "b": "p2.py: the other half (types-p2b.tsv)",
    "c": "p2b.py at 07:5xZ: the DsaMqaLogits types past A's and B's estimated reach (types-p2c.tsv). t211 (N=37) and t212 (N=38) "
         "lost their sessions at serve's 32 GB ulimit -v, t211's sides resident at 26.2 GB (serve) and 36.6 GB (prove) [measured], "
         "t212 at 26.5 and 37.0 GB, so the unreached N=39-45 need a job above 64 GB [inferred]",
    "d": "p2b.py: the 14 types the 64 GB rule left out in v3, at --mem-gb 160 (types-p2d.tsv)",
    "e": "p2b.py: the 18 extras for the first slot A or B freed (types-p2e.tsv); not launched, since A's and B's lists held them "
         "too and B reached 8 of them",
    "f": "after A and B ended: the 10 extras and t171, t173, t175, t177, t179, which A and B listed, did not reach, and no live run held (types-p2f.tsv)",
    "g": "the rest of those: t172, t174, t176, t178, t202 (types-p2g.tsv)",
    "h": "after D ended: the 5 types of list d that D did not reach, the next --mem-gb 160 run, with a 15000 s budget so that it ends "
         "before the record is due (types-p2h.tsv)",
}

V4 = "art:fd2160d5541b057b59c6fb8fc1a105ff2c3f9939cd6a078352cd988c55eee67e"
V7 = sys.argv[1] == "--v7"
V6 = sys.argv[1] in ("--v6", "--v7")
V5 = sys.argv[1] in ("--v5", "--v6", "--v7")
if V5:
    sys.argv.pop(1)
if V6:
    LISTS["k"] = ("fourth pass: two extras beside listed types, SparseMlaHeadScores_v2 at NVIS=63 (beside t161) then NVIS=21 (beside "
                  "t113), FIRST=true, each named directly in its row; --mem-gb 64, limits 3600 and 2400 s, budget 6600 s (types-p4k.tsv)")
    LISTS["i"] = "third pass: t019 (its Boolean version, 9000 s), then t032 (6000 s); --mem-gb 160, budget 15600 s (types-p3i.tsv)"
    LISTS["j"] = "third pass: t031 (its Boolean version, 9000 s); --mem-gb 160, budget 9600 s (types-p3j.tsv)"
cover_path, plan_path, runs_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
cover, plan = json.loads(cover_path.read_text()), json.loads(plan_path.read_text())
split = plan.pop("_split", None)
third = plan.pop("_third", None)
fourth = plan.pop("_fourth", None)
fifth = plan.pop("_fifth", None)
assert V5 == (third is not None), "--v5 takes scratch/cir272/p3-plan.json, and only it"
assert V6 == (fourth is not None), "--v6 takes scratch/cir272/p4-plan.json, and only it"
assert V7 == (fifth is not None), "--v7 takes scratch/cir272/p5-plan.json, and only it"
BESIDE = {**(fourth or {}).get("beside", {}), **(fifth or {}).get("beside", {})}
if V7:
    LISTS.update(fifth["lists"])
assert cover["schema"] == "circuits/request-cover/v3", cover["schema"]
RUNS = []
for a in sys.argv[4:]:
    run, lst, node, mem, *st = a.split(":") + ["", ""]
    RUNS.append({"run": run, "list": lst, "node": node, "mem_gb": int(mem),
                 "state": "running" if "running" in st else "cancelled" if "cancelled" in st else "ended"})
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
        b = by_block[plan[BESIDE.get(tag, tag)]["block"]]
        ce = names[tag].get("census") or {}
        r = {"run": R["run"], "node": R["node"], "list": R["list"], "mem_gb": R["mem_gb"], "tree": TREE, "staged_unit": form[tag][2],
             "lookup": {"kind": ce.get("kind"), "zk_refusal": ce.get("zk_refusal"), "error": ce.get("error") or names[tag].get("error"),
                        "boolean_version": ce.get("boolean_version")},
             "lines": [l[:600] for l in log if re.match(rf"^\S+ {tag}: ", l) or l.startswith(f"STAGED {tag} ")
                       or (l.startswith("VERDICTS ") and l.split(" ")[1] == tag)]}
        d = out / tag
        if tag in nr:
            r.update(outcome="not-run", why="not-reached", text=nr[tag])
        elif not (d / "done").exists() and R["state"] == "cancelled":
            r.update(outcome="not-run", why="cancelled", text=f"its run was cancelled {'while it staged' if d.exists() else 'before it'}")
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
            peaks = [r["stage"]["maxrss_gb"]] + [x["maxrss_gb"] for x in r["verdicts"].values() if x] + \
                [round(x["maxrss_kb"] / 2 ** 20, 2) for f in ("prove.time", "serve.time") if (x := J(d / "B" / f)) and x.get("maxrss_kb")]
            r["peak_gb"] = max((x for x in peaks if x is not None), default=None)
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
                text = next(l for l in r["lines"] if "Lean not run" in l)
                gate = re.search(r"over this job's (\d+) GB for Lean", text)
                r.update(outcome="not-run", why=f"lean-gate-past-{gate[1] if gate else 60}GB", lean_need_gb=int((d / "lean-not-run").read_text()),
                         text=text)
            elif any(l.startswith("stopped: the stager's resident memory") for l in err):
                text = next(l for l in err if l.startswith("stopped:"))
                stop = re.search(r"passed this job's (\d+) GB", text)
                r.update(outcome="not-run", why=f"stage-past-{stop[1] if stop else 62}GB", text=text)
            elif stopped:
                r.update(outcome="not-run", why="lean-past-62GB", text="; ".join(stopped))
            elif oom := [(side, l, (J(d / "B" / f"{side}.time") or {}).get("cmd", "")) for side, log_ in (("serve", "serve.log"), ("prove", "prove.err"))
                         if (d / "B" / log_).exists() for l in (d / "B" / log_).read_text(errors="replace").splitlines()
                         if l.startswith("memory allocation of")]:
                side, line, cmd = oom[0]
                lim = re.search(r"ulimit -v (\d+)", cmd)
                r.update(outcome="not-run", why="session-past-ulimit",
                         text=f"{side}: {line} (its ulimit -v {int(lim.group(1)) // 2 ** 20 if lim else '?'} GB)")
            else:
                texts = [f"ks-form: {ks['why']}"] if ks and ks["exit"] != 0 else []
                texts += [l.split(": ", 1)[1] for l in r["lines"] if "session FAILED" in l or "staging refused" in l]
                if h and h["accepted"] is False:
                    texts.append(f"verify.held: {h['why']}")
                if not st:
                    texts.append(f"stager: {err[-1] if err else ''}")
                r.update(outcome="refused", text=texts)
        reached = not (r.get("why") in ("not-reached", "cancelled") or r.get("text") == "not started")
        if tag in seen:
            prev = seen[tag]
            bigger = (re.fullmatch(r"stage-past-\d+GB|lean-gate-past-\d+GB|lean-past-62GB|session-past-ulimit", prev.get("why") or "")
                      and R["mem_gb"] > prev["mem_gb"])
            assert bigger or not (reached and prev["_reached"]), f"{tag} staged in two runs: {prev['run']} and {R['run']}"
            if not reached and prev["_reached"]:
                continue
            if bigger and reached:
                r["smaller_jobs"] = [*prev.get("smaller_jobs", []), {k: v for k, v in prev.items() if k not in ("lines", "_reached", "smaller_jobs")}]
        seen[tag] = {**r, "_reached": reached}

for r in seen.values():
    r.pop("_reached")

for tag, p in plan.items():
    old = T[tag]
    if not p["run"]:
        r = {"outcome": "not-run", "why": "estimate-past-64GB", "text": p["estimate"]["basis"]}
    else:
        assert tag in seen, f"{tag} is planned to run but in none of the runs given"
        r = seen[tag]
    T[tag] = {**{k: old[k] for k in ("tag", "block", "blocks", "k_log", "extra")}, **r, "estimate": p["estimate"],
        **({"estimate_160": p["estimate_160"]} if "estimate_160" in p else {}), "plan": {
        k: p.get(k) for k in ("run", "order", "list", "limit_s", "mem_gb")}, "v3": {k: old.get(k) for k in ("run", "outcome", "why", "text", "state",
                                                                                           "verdicts", "stage", "needs", "zk_refusal")}}

def beside(tag: str) -> dict | None:
    if tag in seen:
        return seen[tag]
    if not V7:
        return None
    if tag in (nr5 := fifth.get("not_run", {})):
        return {"outcome": "not-run", **nr5[tag]}
    return {"outcome": "queued", "text": "in no fifth-pass run yet"}


if V6:
    cover["beside_extras"] = {
        tag: {"tag": tag, "extra": "beside", **r, **({"staged_unit_named": f"census:{fifth['boolean_version'][tag]}"} if V7 else {}),
              **({"estimate": fifth["estimates"][tag]} if V7 and tag in fifth.get("estimates", {}) else {}),
              "earlier_form": {k: T[of].get(k) for k in ("tag", "block", "blocks", "k_log", "staged_unit", "outcome", "why")}}
        for tag, of in BESIDE.items() if (r := beside(tag)) is not None}


def by_family(bx: dict) -> dict:
    """The fifth pass's table: per family of the earlier forms, its types' Boolean versions by outcome, with their sizes."""
    fam: dict[str, dict] = {}
    for r in bx.values():
        f = fam.setdefault(re.match(r"(\w+?)_v\d+\{", r["earlier_form"]["block"])[1], {"types": 0, "blocks": 0, "outcomes": Counter(), "rows": []})
        f["types"] += 1
        f["blocks"] += r["earlier_form"]["blocks"]
        f["outcomes"][f"{r['outcome']}{'/' + r['why'] if r.get('why') else ''}"] += 1
        f["rows"].append(r)
    out = {}
    for name, f in sorted(fam.items(), key=lambda x: -x[1]["blocks"]):
        st = [r["stage"] for r in f["rows"] if (r.get("stage") or {}).get("ands")]
        rng = lambda k: [min(s[k] for s in st), max(s[k] for s in st)] if st else None  # noqa: E731
        pk = max((r for r in f["rows"] if r.get("peak_gb") is not None), key=lambda r: r["peak_gb"], default=None)
        out[name] = {"types": f["types"], "blocks": f["blocks"], "outcomes": dict(sorted(f["outcomes"].items())), "staged": len(st),
                     "ands": rng("ands"), "k_log": rng("k_log"), "circuit_bytes": rng("circuit_bytes"),
                     "largest_peak_gb": pk and [pk["peak_gb"], pk["tag"]]}
    return out

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
cover["schema"] = ("circuits/request-cover/v7" if V7 else "circuits/request-cover/v6" if V6 else "circuits/request-cover/v5" if V5
                   else "circuits/request-cover/v4")
if V7:
    cover["fifth_pass"] = {**{k: v for k, v in fifth.items() if k not in ("beside", "boolean_version")}, "plan": "scratch/cir272/p5-plan.json",
                           "runs": [r for r in RUNS if r["list"].startswith("p5")], "tree": "4be1de7019db (bound.sh as at 33d7c7dfd8c2)",
                           "supersedes": "art:546ff3480ca9dd2abe6e019e1cd89e3a9bb2b908570a0d7a1f527915556b8fbb",
                           "by_family": by_family(cover["beside_extras"])}
if V6:
    cover["fourth_pass"] = {**fourth, "plan": "scratch/cir272/p4-plan.json", "runs": [r for r in RUNS if r["list"] == "k"],
                            "mem_gb": 64, "budget_s": {"k": 6600}, "tree": "5427f12eabf1 (bound.sh as at 33d7c7dfd8c2)",
                            "supersedes": "art:3d708c34f04b952ef68d1f4b39a390c38cb8898c627354055f442661bc200e75"}
if V5:
    cover["third_pass"] = {**third, "plan": "scratch/cir272/p3-plan.json", "runs": [r for r in RUNS if r["list"] in third["lists"]],
                           "mem_gb": 160, "stage_stop_gb": 156, "lean_gate_gb": 150, "budget_s": {"i": 15600, "j": 9600},
                           "tree": "94c4c61b1768 (bound.sh as at 33d7c7dfd8c2)", "supersedes": V4}
cover["second_pass"] = {"issue": "CIR-272", "tree": TREE, "branch": "cursor/circuits-cir272-bound-6f04", "plan": "scratch/cir272/p2-plan.json",
                        "runs": RUNS, "threads": 8, "mem_gb": 64, "stage_stop_gb": 62, "lean_gate_gb": 60, "lean_stop_gb": 62,
                        "budget_s": {"a-g": 21000, "h": 15000}, "runner_changes": CHANGES, "split": split, "lists": LISTS, "v3": "art:de7ccd2119367c0264ca57c1be3d88774345f1ea13a8f91baf09817c4a188888"}
cover["bound_totals"] = {s: [n, sum(b["blocks"] for b in on if b["bound_outcome"] == s)] for s, n in sorted(cnt.items())}
groups = {"accepted": [s for s in cnt if s.startswith("accepted")], "refused": ["refused"], "held-control-off": ["held-control-off"],
          "not-run": [s for s in cnt if s.startswith("not-run")], "running": ["running"]}
cover["bound_shares"] = {g: {"types": sum(cnt[s] for s in ss), "blocks": (n := sum(b["blocks"] for b in on if b["bound_outcome"] in ss)),
                             "share": round(n / total, 6)} for g, ss in groups.items()}
ex = Counter(f"{r['outcome']}{'/' + r['why'] if r.get('why') else ''}" for r in T.values() if r["extra"])
cover["bound_extras"] = dict(sorted(ex.items()))
cover_path.write_text(json.dumps(cover, indent=1) + "\n")
print(json.dumps({"bound_totals": cover["bound_totals"], "bound_shares": cover["bound_shares"], "extras": cover["bound_extras"]}, indent=1))
