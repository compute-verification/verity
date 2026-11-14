"""cover_v8.py BASE.json CARD.json CARD_PATH CARD_TREE RUN_DIR OUT.json --base ART [--base-note TEXT]: CIR-285's request cover
record over a sampling plan's card, `circuits/request-cover/v8`: v7 with `card`, the card it is of.

BASE is a v7 record (`art:300d2f82...`, or PLANUNITS's copy of it with the recut card's 30 whole rows and 14 unit rows
edited). CARD is the card (its `definitions`), CARD_PATH its path in the tree and CARD_TREE the commit it is taken at.
RUN_DIR is a fetched bound.sh run (scratch/cir272/bound.sh, its `out/` and `job.json`) over types the record does not hold.

1. Every entry is held to the card: one the card names is on the card with the card's prefill and decode counts; one it
   does not name is off the card with no blocks. Each entry that changes keeps what it was under `recut`, and
   `v8.card_sync` lists them.
2. Each type of the run gets a `bound_runs.types` entry by cover272_v4.py's rule for a tag's outcome (accepted only when
   B.verify.held has exit 0 and accepted true and B.verify.no-program and B.verify.other-def exit 1 and accepted false),
   and a `blocks` entry on the card.
3. A Boolean version beside a type no longer on the card moves from `beside_extras` to `beside_extras_off_card`; one beside
   a type on it takes its type's count from the card.
4. `fifth_pass.by_family`, `bound_totals` and `bound_shares` are recomputed as cover272_v4.py computes them."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
for a in ("base_json", "card_json", "card_path", "card_tree", "run_dir", "out"):
    ap.add_argument(a)
ap.add_argument("--base", required=True, help="BASE's art: id")
ap.add_argument("--base-note", default=None)
A = ap.parse_args()

c = json.loads(Path(A.base_json).read_text())
card = json.loads(Path(A.card_json).read_text())
assert c["schema"] == "circuits/request-cover/v7", c["schema"]
defs = card["definitions"]
J = lambda p: json.loads(p.read_text()) if p.exists() else None  # noqa: E731

# 1. every entry held to the card
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


# 2. the run's types, by cover272_v4.py's rule
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


run_dir = Path(A.run_dir)
job = json.loads((run_dir / "job.json").read_text())
RUN, SRC = job["run"]["id"], job["computation"]["tree_sha"]
out = run_dir / "out"
log = (out / "job.log").read_text(errors="replace").splitlines()
nr = {l.split("\t")[0]: l.split("\t")[2] for l in (out / "not-reached.tsv").read_text().splitlines() if l}
form = {l.split("\t")[0]: l.split("\t") for l in (out / "form.tsv").read_text().splitlines() if l}
names = {n["tag"]: n for n in map(json.loads, (out / "names.jsonl").read_text().splitlines()) if n.get("tag")}
T = c["bound_runs"]["types"]
added = []
for tag in [l.split("\t")[0] for l in (out / "types.tsv").read_text().splitlines() if l]:
    listed = form[tag][1]
    assert listed.startswith("census:"), listed
    name = listed[len("census:"):]
    assert name in defs, f"{tag}: {name} is not on the card"
    assert name not in by_block, f"{tag}: {name} already has an entry"
    assert tag not in T, f"{tag} is already a tag of bound_runs.types"
    ce = names[tag].get("census") or {}
    d = out / tag
    r = {"run": RUN, "list": "cir285", "tree": SRC, "staged_unit": form[tag][2],
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
            r["state"] = ("own-definition" if st["word"] == name else
                          "boolean-version" if st["definition"] == ce.get("boolean_version") else "other")
            assert r["state"] != "other", (tag, st["definition"], st["word"], name, ce.get("boolean_version"))
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
    dd = defs[name]
    blocks = dd["prefill"] + dd["decode"]
    T[tag] = {"tag": tag, "block": name, "blocks": blocks, "k_log": dd.get("k_log"), "extra": False, "earlier_form": None,
              "state": r.get("state"), "text": r.get("text"), **r}
    o = r["outcome"]
    entry = {"block": name, "on_card": True, "family": dd.get("family"), "prefill": dd["prefill"], "decode": dd["decode"],
             "blocks": blocks, "k_log": dd.get("k_log"), "definition": st and st["definition"],
             "boolean_version": ce.get("boolean_version"), "staged": listed, "zk_refusal": ce.get("zk_refusal"),
             "accepted": r.get("state") if o == "accepted" else "none", "form": "bound" if o == "accepted" else None,
             "statements": [{"run": RUN, "tag": tag, "verdict": "accepted", "definition": st["definition"], "units": 1, "form": "bound",
                             "handed": "the partition object and the Program (B.verify.held)", "device": "cpu", "tree": SRC,
                             "state": r["state"]}] if o == "accepted" else None,
             "error": None, "cir272": {k: r[k] for k in ("run", "outcome", "state", "why", "text") if k in r} | {"tag": tag},
             "bound_outcome": f"accepted-{r['state']}" if o == "accepted" else f"not-run-{r['why']}" if o == "not-run" else o}
    c["blocks"].append(entry)
    by_block[name] = entry
    added.append({"tag": tag, "block": name, "outcome": entry["bound_outcome"]})

# 3. the Boolean versions beside types no longer on the card
on = [b for b in c["blocks"] if b["on_card"]]
on_names = {b["block"]: b for b in on}
kept, moved = {}, dict(c.get("beside_extras_off_card") or {})
for x, r in (c.get("beside_extras") or {}).items():
    b = on_names.get(r["earlier_form"]["block"])
    if b is None:
        moved[x] = r
    else:
        if r["earlier_form"]["blocks"] != b["blocks"]:
            r["earlier_form"]["recut"] = {"was_blocks": r["earlier_form"]["blocks"], "card": card["source"]}
            r["earlier_form"]["blocks"] = b["blocks"]
        kept[x] = r
c["beside_extras"], c["beside_extras_off_card"] = kept, moved


# 4. the stored totals, as cover272_v4.py computes them
def by_family(bx: dict) -> dict:
    fam: dict[str, dict] = {}
    for r in bx.values():
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
assert set(on_names) == set(defs), (sorted(set(defs) - set(on_names))[:5], sorted(set(on_names) - set(defs))[:5])
assert total == card_blocks, (total, card_blocks)
c["schema"] = "circuits/request-cover/v8"
c["tree"] = SRC
c["card"] = {"path": A.card_path, "source": card["source"], "tree": A.card_tree, "types": len(defs), "blocks": card_blocks}
c["v8"] = {"issue": "CIR-285", "base": A.base, "base_note": A.base_note, "run": RUN, "run_tree": SRC,
           "builder": "scratch/cir285/cover_v8.py on cursor/circuits-cir285-card-scratch-6f04",
           "card_sync": sync, "added": added, "extras_off_card": sorted(moved)}
Path(A.out).write_text(json.dumps(c, indent=1) + "\n")
print(json.dumps({"card": c["card"], "off": len(sync["off"]), "on": len(sync["on"]), "counts": len(sync["counts"]), "added": len(added),
                  "extras_kept": len(kept), "extras_moved": len(moved), "bound_totals": c["bound_totals"],
                  "bound_shares": c["bound_shares"]}, indent=1))
