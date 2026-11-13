"""cover272_build.py COVER.json TYPES.json RUNS_DIR: CIR-272's record, `circuits/request-cover/v3`, written over COVER.json (v2).

For each tag of TYPES.json (scratch/cir272/types.json on cursor/circuits-cir272-bound-6f04: the 240 card types without a
bound accepted statement and the 18 extra units), its outcome read from its own run's files under RUNS_DIR/RUN/out/TAG
(`done`, `timeout`, `lean-not-run`, B/stage.json, B/stage.err, B/stage.time, verdicts.json) and the run's job.log and
not-reached.tsv:

    accepted          B.verify.held exit 0 and accepted true, and B.verify.no-program and B.verify.other-def exit 1 and
                      accepted false; state own-definition (stage.json's definition is the block's Definition) or
                      boolean-version (it is the block's Boolean version)
    held-control-off  B.verify.held accepted but a control did not refuse
    refused           the verifier (ks-form, statement or verify) or the stager refused; `text` holds their own lines
    not-run           why: not-reached (the run's budget), past-2700s (the type's limit), stage-past-62GB (the stager stopped
                      at 62 GB resident under the job's 64), lean-past-64GB (staged; the Lean checks' peak by job.sh's rule,
                      6 x circuit.txt + 4 GB, over 64 GB, so they were not run)

Every not-run type gets `needs`: measured figures where its own run has them, else its nearest measured sibling's (same
family, nearest k_log), marked inferred. The card's 254 types get `bound_outcome` (the 14 earlier bound types keep theirs as
`accepted-earlier`), and `bound_totals` counts types and blocks by it; cover_check.py recomputes them."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

RUNS = {"first": ("r20261112-214915-789b", "vy-nebius-cpu-15"), "a": ("r20261112-215857-32d2", "vy-nebius-cpu-7"),
        "b": ("r20261112-215921-f101", "vy-nebius-cpu-15")}
TREE = "fc665e18ac040bf6d0685d3b98f4873421b22492"
LIMIT_S, STAGE_GB, MEM_GB = 2700, 62, 64

cover_path, types_path, runs_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
cover, index = json.loads(cover_path.read_text()), json.loads(types_path.read_text())
assert cover["schema"] == "circuits/request-cover/v2", cover["schema"]
J = lambda p: json.loads(p.read_text()) if p.exists() else None  # noqa: E731


def tm(p: Path) -> dict:
    s = p.read_text(errors="replace") if p.exists() else ""
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", s)
    e = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", s)
    secs = None
    if e:
        parts = [float(x) for x in e[1].split(":")]
        secs = round(sum(v * 60 ** i for i, v in enumerate(reversed(parts))), 1)
    return {"maxrss_gb": round(int(m[1]) / 2 ** 20, 2) if m else None, "wall_s": secs}


def lines(log: list[str], tag: str) -> list[str]:
    return [l[:600] for l in log if re.match(rf"^\S+ {tag}: ", l) or l.startswith(f"STAGED {tag} ")
            or (l.startswith("VERDICTS ") and l.split(" ")[1] == tag)]


def v_of(v: dict, k: str) -> dict | None:
    r = v.get(k)
    if r is None:
        return None
    vd = r.get("verdict") or {}
    return {"exit": r["exit"], "accepted": vd.get("accepted"), "why": vd.get("why") or r.get("setup") or r.get("ks_form") or r.get("said"),
            "units": ((vd.get("partition") or {}) if isinstance(vd.get("partition"), dict) else {}).get("note", "")[:200] or None,
            "wall": r.get("wall"), "maxrss_gb": round(r["maxrss_kb"] / 2 ** 20, 2) if r.get("maxrss_kb") else None}


by_block = {b["block"]: b for b in cover["blocks"]}
recs: dict[str, dict] = {}
for where, (run, node) in RUNS.items():
    out = runs_dir / run / "out"
    log = (out / "job.log").read_text(errors="replace").splitlines()
    nr = {l.split("\t")[0]: l.rstrip("\n").split("\t")[2] for l in (out / "not-reached.tsv").read_text().splitlines() if l}
    names = {n["tag"]: n.get("error") or (n.get("census") or {}).get("error")
             for n in map(json.loads, (out / "names.jsonl").read_text().splitlines()) if n.get("tag")}
    for tag, ix in index.items():
        if ix["list"] != where:
            continue
        b = by_block[ix["block"]]
        r: dict = {"tag": tag, "run": run, "node": node, "tree": TREE, "list": where, "block": ix["block"], "blocks": ix["blocks"],
                   "k_log": ix["k_log"], "extra": ix["extra"], "earlier_form": ix["form"], "state": None, "text": None}
        d = out / tag
        if names.get(tag):
            r["names_refusal"] = f"names.jsonl (the stager's census lookup, serving_rows.template): {names[tag]}"
        if tag in nr:
            assert not d.exists(), tag
            r.update(outcome="not-run", why="not-reached", text=nr[tag])
            recs[tag] = r
            continue
        done, st, v = J(d / "done"), J(d / "B" / "stage.json"), J(d / "verdicts.json") or {}
        err = (d / "B" / "stage.err").read_text(errors="replace").strip().splitlines() if (d / "B" / "stage.err").exists() else []
        r.update(done=done, stage={"definition": st and st["definition"], "word": st and st["word"], "ands": st and st.get("ands"),
                                   "k_log": st and st.get("k_log"), "circuit_bytes": st and st.get("circuit_bytes"),
                                   "statement": st and st.get("statement"), **tm(d / "B" / "stage.time")},
                 verdicts={k: v_of(v, k) for k in ("B.statement.held", "B.statement.no-program", "B.statement.other-def", "B.ks-form",
                                                    "B.verify.held", "B.verify.no-program", "B.verify.other-def")},
                 lines=lines(log, tag))
        h, np_, od, ks = (r["verdicts"][k] for k in ("B.verify.held", "B.verify.no-program", "B.verify.other-def", "B.ks-form"))
        if h and h["exit"] == 0 and h["accepted"] is True:
            ok = all(c and c["exit"] == 1 and c["accepted"] is False for c in (np_, od))
            r["outcome"] = "accepted" if ok else "held-control-off"
            sd = st["definition"]
            r["state"] = "own-definition" if sd == b["definition"] else "boolean-version" if sd == b["boolean_version"] else "other"
            assert r["state"] != "other", (tag, sd, b["definition"], b["boolean_version"])
        elif done["timeout"]:
            r.update(outcome="not-run", why="past-2700s", text=(d / "timeout").read_text().strip())
        elif (d / "lean-not-run").exists():
            r.update(outcome="not-run", why="lean-past-64GB", lean_need_gb=int((d / "lean-not-run").read_text()),
                     text=next(l for l in r["lines"] if "Lean not run" in l))
        elif any(l.startswith("stopped: the stager's resident memory") for l in err):
            r.update(outcome="not-run", why="stage-past-62GB", text=next(l for l in err if l.startswith("stopped:")))
        else:
            texts = [f"ks-form: {ks['why']}"] if ks and ks["exit"] != 0 else []
            texts += [l.split(": ", 1)[1] for l in r["lines"] if "session FAILED" in l or "staging refused" in l]
            if h and h["accepted"] is False:
                texts.append(f"verify.held: {h['why']}")
            if not st:
                texts.append(f"stager: {err[-1] if err else ''}")
            r.update(outcome="refused", text=texts)
        recs[tag] = r
assert len(recs) == len(index) == 258

# what one more run of a not-run type would need: its own measured lines, else its nearest measured sibling's (inferred)
fam = lambda n: n.split("{")[0]  # noqa: E731
measured = [r for r in recs.values() if r.get("stage") and r["stage"]["definition"]]
# SparseMlaHeadScores_v1 at FIRST=true: circuit.txt GiB, the stager's peak GB and its wall seconds, each a least-squares line
# in NVIS over the measured staged types of tonight's runs; Lean's peak over circuit GiB, the largest of the accepted types
# whose circuit is over 5 GiB
hs = [(int(re.search(r"NVIS=(\d+)", m["block"])[1]), m["stage"]["circuit_bytes"] / 2 ** 30, m["stage"]["maxrss_gb"], m["stage"]["wall_s"])
      for m in measured if m["block"].startswith("SparseMlaHeadScores_v1") and "FIRST=true" in m["block"] and "NVIS=64," not in m["block"]]


def line(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - k * mx, k


HS_FIT = [line([h[0] for h in hs], [h[i] for h in hs]) for i in (1, 2, 3)] if len(hs) >= 3 else None
HS_FROM = sorted(h[0] for h in hs)
LEAN_RATIO = max(max((v or {}).get("maxrss_gb") or 0 for v in m["verdicts"].values()) / (m["stage"]["circuit_bytes"] / 2 ** 30)
                 for m in measured if m.get("outcome") == "accepted" and m["stage"]["circuit_bytes"] > 5 * 2 ** 30)
for r in recs.values():
    if r["outcome"] != "not-run":
        continue
    own = r.get("stage") or {}
    if own.get("definition"):
        cb = own["circuit_bytes"]
        r["needs"] = {"source": "measured, its own run", "stage_s": own["wall_s"], "stage_gb": own["maxrss_gb"], "circuit_bytes": cb,
                      "lean_gb_by_rule": cb * 6 // 2 ** 30 + 4 if cb else None}
        continue
    if r["why"] in ("past-2700s", "stage-past-62GB"):
        r["needs"] = {"source": "measured, its own run", "stage_s": f"over {r['done']['seconds']}" if r["why"] == "past-2700s" else (r.get("stage") or {}).get("wall_s"),
                      "stage_gb": f"over {STAGE_GB}" if r["why"] == "stage-past-62GB" else (r.get("stage") or {}).get("maxrss_gb")}
    sib = sorted((m for m in measured if fam(m["block"]) == fam(r["block"])),
                 key=lambda m: (abs((m["k_log"] or 0) - (r["k_log"] or 0)), -(m["stage"]["circuit_bytes"] or 0)))
    nv = re.search(r"NVIS=(\d+),FIRST=true", r["block"])
    if r["block"].startswith("SparseMlaHeadScores_v1") and nv and HS_FIT:
        n = int(nv[1])
        gib, gb, s = (a + k * n for a, k in HS_FIT)
        r.setdefault("needs", {}).update(fit={"from": HS_FROM, "nvis": n, "circuit_gib": round(gib, 2), "stage_gb": round(gb, 2),
                                              "stage_s": round(s), "lean_gb_by_rule": int(gib * 6) + 4,
                                              "lean_gb_by_measured_ratio": round(gib * LEAN_RATIO, 1)}, inferred=True)
        continue
    if not sib:
        old = [x for x in cover["blocks"] if fam(x["block"]) == fam(r["block"]) and x["staging_run"]]
        r.setdefault("needs", {}).update(earlier=[{"block": x["block"], "k_log": x["k_log"], "run": x["staging_run"]["run"],
                                                   "stage_gb": x["staging_run"]["peak_gib"], "tree": x["staging_run"].get("tree")} for x in old] or None,
                                         inferred=True)
        continue
    if sib:
        m = sib[0]
        cb = m["stage"]["circuit_bytes"]
        r.setdefault("needs", {}).update(sibling={"tag": m["tag"], "block": m["block"], "k_log": m["k_log"], "outcome": m["outcome"],
                                                  "why": m.get("why"), "seconds": m["done"]["seconds"], "stage_gb": m["stage"]["maxrss_gb"],
                                                  "circuit_bytes": cb, "lean_gb_by_rule": cb * 6 // 2 ** 30 + 4 if cb else None},
                                         inferred=True)
    else:
        r.setdefault("needs", {}).update(sibling=None, inferred=True)

# the card's 254 types by outcome
for b in cover["blocks"]:
    r = next((x for x in recs.values() if x["block"] == b["block"]), None)
    b["cir272"] = r and {k: r[k] for k in ("tag", "run", "node", "outcome", "state", "why", "text") if k in r}
    if b["on_card"]:
        if b["form"] == "bound":
            b["bound_outcome"] = "accepted-earlier"
        else:
            o = r["outcome"]
            b["bound_outcome"] = (f"accepted-{r['state']}" if o == "accepted" else f"not-run-{r['why']}" if o == "not-run" else o)
on = [b for b in cover["blocks"] if b["on_card"]]
total = sum(b["blocks"] for b in on)
cnt = Counter(b["bound_outcome"] for b in on)
cover["schema"] = "circuits/request-cover/v3"
cover["bound_runs"] = {"issue": "CIR-272", "tree": TREE, "branch": "cursor/circuits-cir272-bound-6f04",
                       "runs": {w: {"run": r, "node": n} for w, (r, n) in RUNS.items()}, "type_limit_s": LIMIT_S,
                       "stage_stop_gb": STAGE_GB, "mem_gb": MEM_GB, "threads": 8, "units": 1,
                       "accepted_rule": "B.verify.held exit 0 and accepted true; B.verify.no-program and B.verify.other-def exit 1, accepted false",
                       "proofs_runs": "none named on CIR-272 (its comments read 13 Nov 04:3xZ)",
                       "types": recs}
cover["bound_totals"] = {s: [n, sum(b["blocks"] for b in on if b["bound_outcome"] == s)] for s, n in sorted(cnt.items())}
groups = {"accepted": ("accepted-earlier", "accepted-own-definition", "accepted-boolean-version"), "refused": ("refused",),
          "not-run": tuple(s for s in cnt if s.startswith("not-run")), "held-control-off": ("held-control-off",)}
cover["bound_shares"] = {g: {"types": sum(cnt[s] for s in ss), "blocks": (n := sum(b["blocks"] for b in on if b["bound_outcome"] in ss)),
                             "share": round(n / total, 6)} for g, ss in groups.items()}
cover_path.write_text(json.dumps(cover, indent=1) + "\n")
print(json.dumps({"bound_totals": cover["bound_totals"], "bound_shares": cover["bound_shares"],
                  "extras": dict(Counter(f"{r['outcome']} {r.get('why')}" for r in recs.values() if r["extra"]))}, indent=1))
