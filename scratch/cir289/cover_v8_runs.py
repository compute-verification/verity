"""cover_v8_runs.py BASE.json CARD.json CARD_PATH CARD_TREE OUT.json --base ART --forms FORMS.jsonl RUN_DIR... [--partial]:
CIR-289's request cover record, `circuits/request-cover/v8`, over a sampling plan's card from a v7 or v8 record and the
bound.sh runs over the kinds it does not hold.

As scratch/cir285/cover_v8.py (each entry held to the card; each run's kind a `bound_runs.types` entry by cover272_v4.py's
rule; the stored totals recomputed), but over several runs, and a run's row may name a kind's Boolean version (`_v2`)
where the card names the kind itself. FORMS (the census lookup of each card kind, names.py's lines) maps each Boolean
version back to its kind:
- where the lookup refuses the kind's listed form under --zk (`zk_refusal`), the Boolean version is the kind's own
  statement, recorded as cover_v8.py records form.py's swap: `accepted-boolean-version`;
- elsewhere the listed form was not run: the kind's entry is `not-run-boolean-only` with a `not-run` run under its own
  tag, and the Boolean version's run is a `beside_extras` entry (`v2` + tag) whose `earlier_form` is that tag.
--partial skips the check that every card kind has an entry, to try the builder before every run is in."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
for a in ("base_json", "card_json", "card_path", "card_tree", "out"):
    ap.add_argument(a)
ap.add_argument("run_dirs", nargs="+")
ap.add_argument("--base", required=True)
ap.add_argument("--forms", required=True)
ap.add_argument("--partial", action="store_true")
A = ap.parse_args()

c = json.loads(Path(A.base_json).read_text())
card = json.loads(Path(A.card_json).read_text())
assert c["schema"] in ("circuits/request-cover/v7", "circuits/request-cover/v8"), c["schema"]
defs = card["definitions"]
J = lambda p: json.loads(p.read_text()) if p.exists() else None  # noqa: E731
lookup = {r["census"]["name"]: r["census"] for r in map(json.loads, Path(A.forms).read_text().splitlines())}
of_boolean = {ce["boolean_version"]: name for name, ce in lookup.items() if ce.get("boolean_version")}

by_block = {b["block"]: b for b in c["blocks"]}
assert len(by_block) == len(c["blocks"]), "two entries of one block type"
sync = {"off": [], "on": [], "counts": []}
for b in c["blocks"]:
    was = {k: b[k] for k in ("on_card", "prefill", "decode", "blocks")}
    d = defs.get(b["block"])
    if d is None:
        b.update(on_card=False, prefill=0, decode=0, blocks=0)
    else:
        b.update(on_card=True, prefill=d["prefill"], decode=d["decode"], blocks=d["prefill"] + d["decode"])
    now = {k: b[k] for k in was}
    if now != was:
        b["recut"] = {"was": was, "card": card["source"]}
        sync["off" if was["on_card"] and not now["on_card"] else "on" if now["on_card"] and not was["on_card"] else "counts"].append(
            {"block": b["block"], "was": was, "now": now})


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


def one_run(run_dir: Path) -> list[tuple[str, str, dict, dict | None]]:
    """Each row of the run as (tag, staged name, its run record, its stage.json)."""
    job = json.loads((run_dir / "job.json").read_text())
    RUN, SRC = job["run"]["id"], job["computation"]["tree_sha"]
    out = run_dir / "out"
    log = (out / "job.log").read_text(errors="replace").splitlines()
    nr = {l.split("\t")[0]: l.split("\t")[2] for l in (out / "not-reached.tsv").read_text().splitlines() if l}
    form = {l.split("\t")[0]: l.split("\t") for l in (out / "form.tsv").read_text().splitlines() if l}
    names = {n["tag"]: n for n in map(json.loads, (out / "names.jsonl").read_text().splitlines()) if n.get("tag")}
    rows = []
    for tag in [l.split("\t")[0] for l in (out / "types.tsv").read_text().splitlines() if l]:
        listed = form[tag][1]
        assert listed.startswith("census:"), listed
        ce = names[tag].get("census") or {}
        d = out / tag
        r = {"run": RUN, "list": run_dir.name, "tree": SRC, "staged_unit": form[tag][2],
             "lookup": {"kind": ce.get("kind"), "zk_refusal": ce.get("zk_refusal"), "error": ce.get("error") or names[tag].get("error"),
                        "boolean_version": ce.get("boolean_version")},
             "lines": [l[:600] for l in log if re.match(rf"^\S+ {tag}: ", l) or l.startswith(f"STAGED {tag} ")
                       or (l.startswith("VERDICTS ") and l.split(" ")[1] == tag)]}
        st = None
        if tag in nr:
            r.update(outcome="not-run", why="not-reached", text=nr[tag])
        else:
            done, st, v = J(d / "done"), J(d / "B" / "stage.json"), J(d / "verdicts.json") or {}
            assert done is not None, f"{tag}: the run did not end it"
            err = (d / "B" / "stage.err").read_text(errors="replace").strip().splitlines() if (d / "B" / "stage.err").exists() else []
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
                ok = all(x and x["exit"] == 1 and x["accepted"] is False for x in (np_, od))
                r["outcome"] = "accepted" if ok else "held-control-off"
            elif done["timeout"]:
                r.update(outcome="not-run", why="past-limit", text=(d / "timeout").read_text().strip())
            else:
                texts = [f"ks-form: {ks['why']}"] if ks and ks["exit"] != 0 else []
                texts += [l.split(": ", 1)[1] for l in r["lines"] if "session FAILED" in l or "staging refused" in l]
                if h and h["accepted"] is False:
                    texts.append(f"verify.held: {h['why']}")
                if not st:
                    texts.append(f"stager: {err[-1] if err else ''}")
                r.update(outcome="refused", text=texts)
        rows.append((tag, listed[len("census:"):], r, st))
    return rows


T = c["bound_runs"]["types"]
bx = c.setdefault("beside_extras", {})
added = []
for rd in A.run_dirs:
    for tag, staged, r, st in one_run(Path(rd)):
        name = staged if staged in defs else of_boolean.get(staged)
        assert name in defs, f"{tag}: {staged} is neither a card kind nor the Boolean version of one"
        assert name not in by_block, f"{tag}: {name} already has an entry"
        assert tag not in T and f"v2{tag}" not in bx, f"{tag} is already a tag"
        ce = lookup[name]
        dd = defs[name]
        blocks = dd["prefill"] + dd["decode"]
        boolean_only = staged != name and not ce.get("zk_refusal")
        o = r["outcome"]
        if o == "accepted":
            r["state"] = ("own-definition" if r["staged_unit"] == f"census:{name}" else
                          "boolean-version" if r["staged_unit"] == f"census:{ce.get('boolean_version')}" else "other")
            assert r["state"] != "other", (tag, st["definition"], st["word"], name, ce.get("boolean_version"))
        entry = {"block": name, "on_card": True, "family": dd.get("family"), "prefill": dd["prefill"], "decode": dd["decode"],
                 "blocks": blocks, "k_log": dd.get("k_log"), "boolean_version": ce.get("boolean_version"), "staged": f"census:{staged}",
                 "zk_refusal": ce.get("zk_refusal"), "error": None}
        if boolean_only:
            T[tag] = {"tag": tag, "block": name, "blocks": blocks, "k_log": dd.get("k_log"), "extra": False, "earlier_form": None,
                      "state": None, "outcome": "not-run", "why": "boolean-only",
                      "text": "the listed form not run: its Boolean version was run instead, on the lead's word (CIR-289)"}
            bx[f"v2{tag}"] = {"tag": f"v2{tag}", "extra": True, **r, "staged_unit_named": staged,
                              "earlier_form": {"tag": tag, "block": name, "blocks": blocks, "k_log": dd.get("k_log"), "staged_unit": None,
                                               "outcome": "not-run", "why": "boolean-only"}}
            entry.update(definition=None, accepted="none", form=None, statements=None,
                         cir272={"run": r["run"], "outcome": "not-run", "why": "boolean-only", "tag": tag},
                         bound_outcome="not-run-boolean-only")
        else:
            T[tag] = {"tag": tag, "block": name, "blocks": blocks, "k_log": dd.get("k_log"), "extra": False, "earlier_form": None,
                      "state": r.get("state"), "text": r.get("text"), **r}
            entry.update(definition=st and st["definition"], accepted=r.get("state") if o == "accepted" else "none",
                         form="bound" if o == "accepted" else None,
                         statements=[{"run": r["run"], "tag": tag, "verdict": "accepted", "definition": st["definition"], "units": 1, "form": "bound",
                                      "handed": "the partition object and the Program (B.verify.held)", "device": "cpu", "tree": r["tree"],
                                      "state": r["state"]}] if o == "accepted" else None,
                         cir272={k: r[k] for k in ("run", "outcome", "state", "why", "text") if k in r} | {"tag": tag},
                         bound_outcome=f"accepted-{r['state']}" if o == "accepted" else f"not-run-{r['why']}" if o == "not-run" else o)
        c["blocks"].append(entry)
        by_block[name] = entry
        added.append({"tag": tag, "block": name, "run": r["run"], "outcome": entry["bound_outcome"],
                      "boolean_version": bx[f"v2{tag}"]["outcome"] if boolean_only else None})

on = [b for b in c["blocks"] if b["on_card"]]
on_names = {b["block"]: b for b in on}
kept, moved = {}, dict(c.get("beside_extras_off_card") or {})
for x, r in bx.items():
    b = on_names.get(r["earlier_form"]["block"])
    if b is None:
        moved[x] = r
    else:
        if r["earlier_form"]["blocks"] != b["blocks"]:
            r["earlier_form"]["recut"] = {"was_blocks": r["earlier_form"]["blocks"], "card": card["source"]}
            r["earlier_form"]["blocks"] = b["blocks"]
        kept[x] = r
c["beside_extras"], c["beside_extras_off_card"] = kept, moved


def by_family(bxs: dict) -> dict:
    fam: dict[str, dict] = {}
    for r in bxs.values():
        f = fam.setdefault(re.match(r"(\w+?)_v\d+\{", r["earlier_form"]["block"])[1], {"types": 0, "blocks": 0, "outcomes": Counter(), "rows": []})
        f["types"] += 1
        f["blocks"] += r["earlier_form"]["blocks"]
        f["outcomes"][f"{r['outcome']}{'/' + r['why'] if r.get('why') else ''}"] += 1
        f["rows"].append(r)
    res = {}
    for name, f in sorted(fam.items(), key=lambda x: -x[1]["blocks"]):
        st = [r["stage"] for r in f["rows"] if (r.get("stage") or {}).get("ands")]
        rng = lambda k: [min(s[k] for s in st), max(s[k] for s in st)] if st else None  # noqa: E731
        pk = max((r for r in f["rows"] if r.get("peak_gb") is not None), key=lambda r: r["peak_gb"], default=None)
        res[name] = {"types": f["types"], "blocks": f["blocks"], "outcomes": dict(sorted(f["outcomes"].items())), "staged": len(st),
                     "ands": rng("ands"), "k_log": rng("k_log"), "circuit_bytes": rng("circuit_bytes"),
                     "largest_peak_gb": pk and [pk["peak_gb"], pk["tag"]]}
    return res


c["fifth_pass"]["by_family"] = by_family(c["beside_extras"])
total = sum(b["blocks"] for b in on)
cnt = Counter(b["bound_outcome"] for b in on)
c["bound_totals"] = {s: [n, sum(b["blocks"] for b in on if b["bound_outcome"] == s)] for s, n in sorted(cnt.items())}
groups = {"accepted": [s for s in cnt if s.startswith("accepted")], "refused": ["refused"], "held-control-off": ["held-control-off"],
          "not-run": [s for s in cnt if s.startswith("not-run")], "running": ["running"]}
c["bound_shares"] = {g: {"types": sum(cnt[s] for s in ss), "blocks": (n := sum(b["blocks"] for b in on if b["bound_outcome"] in ss)),
                         "share": round(n / total, 6)} for g, ss in groups.items()}

card_blocks = sum(d["prefill"] + d["decode"] for d in defs.values())
missing = sorted(set(defs) - set(on_names))
if not A.partial:
    assert not missing and set(on_names) == set(defs), (missing[:5], sorted(set(on_names) - set(defs))[:5])
    assert total == card_blocks, (total, card_blocks)
prev = c.get("v8")
c["schema"] = "circuits/request-cover/v8"
c["tree"] = " ".join(sorted({T[a["tag"]].get("tree") or bx[f"v2{a['tag']}"]["tree"] for a in added}))
c["card"] = {"path": A.card_path, "source": card["source"], "tree": A.card_tree, "types": len(defs), "blocks": card_blocks}
c["v8"] = {"issue": "CIR-289", "base": A.base, "runs": sorted({a["run"] for a in added}),
           "builder": "scratch/cir289/cover_v8_runs.py on cursor/circuits-cir285-card-scratch-6f04",
           "card_sync": sync, "added": added, "extras_off_card": sorted(moved), "previous": prev}
Path(A.out).write_text(json.dumps(c, indent=1) + "\n")
print(json.dumps({"card": c["card"], "off": len(sync["off"]), "on": len(sync["on"]), "counts": len(sync["counts"]), "added": len(added),
                  "missing": len(missing), "extras_kept": len(kept), "extras_moved": len(moved), "bound_totals": c["bound_totals"],
                  "bound_shares": c["bound_shares"]}, indent=1))
