"""cover_build.py NODE_OUT ACCEPTED.json: `cover.json` (the record) and `cover.md` (made from it) beside this script.

NODE_OUT is `cover_list.sh`'s $OUT (list, ranks, pins, cuts, names, roots); ACCEPTED.json is `cover_accepted.py`'s. The census
card is main's `benchmarks/sampling_plan/glm53_blocks.json`; Friday's select and its blocks are read at the commit vllm names
(`FRIDAY_AT`), where they are registered and carded, since main has neither.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
node, acc_path = Path(sys.argv[1]), Path(sys.argv[2])
MAIN = "5f0c008aa0403ac275f5626828c534f93d987048"
LIST_RUN, LEAD_RUN = node.parent.name, "r20261112-201113-d1f1"
AUDIT_RUN = "r20261112-134838-1b6c"
BRANCH_259 = "31b1b56aec2a"
FRIDAY_AT = "f461ac0edeb2"
FRIDAY = "GreedySelect_v1{V=154880,N=7744}"

J = lambda p: json.loads(Path(p).read_text())  # noqa: E731
lst = J(node / "list.json")["by"]["eager"]
ranks, pins, cuts = J(node / "ranks.json"), J(node / "pins.json")["templates"], J(node / "cuts.json")["templates"]
names, roots = J(node / "names.json"), J(node / "roots.json")["templates"]
card = J(ROOT / "benchmarks/sampling_plan/glm53_blocks.json")["definitions"]
card_f = json.loads(subprocess.check_output(["git", "-C", str(ROOT), "show", f"{FRIDAY_AT}:benchmarks/sampling_plan/glm53_blocks.json"]))
card_f = card_f["definitions"]
acc = [a for a in J(acc_path) if a.get("word") or a.get("unit")]
audit = J(Path.home() / f".research/runs/{AUDIT_RUN}/out/sparse-mla-units/summary.json")
audit_ids = set(audit["rank0"]["audit"]["wrong_units"]["per_template"]) if audit["rank0"]["audit"]["accepted"] else set()

RANK = {"yes": 4, "boolean-version": 3, "refused": 2, "no-verdict": 1, "none": 0}


def verdict_of(word: str, def_id: str | None) -> tuple[str, str | None, str | None, list]:
    """The best Lean verdict on a statement of `word` in the runs read: (accepted, accepted_definition, run, every hit)."""
    hits = [a for a in acc if word in (a.get("word"), (a.get("unit") or "").removeprefix("census:"))]
    best = ("none", None, None)
    for a in hits:
        v = {"accepted": "yes" if a["definition"] == def_id else "boolean-version", "refused": "refused"}.get(a["verdict"], "no-verdict")
        if RANK[v] > RANK[best[0]]:
            best = (v, a["definition"] if v != "yes" else None, a["run"])
    if word in audit_ids and RANK["yes"] > RANK[best[0]]:
        best = ("yes", None, AUDIT_RUN)
    return (*best, [f"{a['run']}/{a['tag']}: {a['verdict']} at {a['definition']}" for a in hits])


def staged_name(tid: str) -> str | None:
    """`stage.py`'s name for tid at its binding: a table's name at these statics, else `census:tid` when the census template
    resolves it (dry: `serving_rows.template` and `chunk_rows.statement`, no staging run)."""
    t = names["templates"].get(tid) or names["blocks"].get(tid)
    if not t:
        return None
    exact = [h for h in t["tables"] if h["exact"] and h["table"] in ("UNITS", "PARTS", "CENSUS", "READS")]
    if exact:
        return exact[0]["name"]
    return f"census:{tid}" if t["census"]["ok"] else None


rows = []
for r in sorted(lst["rows"], key=lambda r: (-r["n"], r["id"])):
    tid, p = r["id"], pins[r["id"]]
    other = p["family"] + (p["boolean_version"].get("family_pins") or [])
    pin = "this-binding" if p["exact"] else ("other-binding" if other else "none")
    units = [u[0] for u in cuts[tid]["units"]]
    cut = sorted(d for d in set(units) | {tid} if d in card and r["spec_base"] in card[d]["of"])
    staged = staged_name(tid)
    if staged is None:
        staged = next((s for s in (staged_name(d) for d in cut) if s), None)
    accepted, acc_def, run, hits = verdict_of(tid, p["def_id"])
    cut_hits = {u: verdict_of(u, None)[3] for u in sorted(set(units) - {tid})}
    rows.append({
        "template": tid, "definition": p["def_id"], "bindings": r["bindings"], "instances": r["n"], "in_list": True,
        "pin": pin, "pin_binding": None if pin != "other-binding" else sorted(x[1] for x in other)[0],
        "cut": cut or None, "staged": staged,
        "accepted": accepted, "accepted_definition": acc_def, "run": run, "source": "read" if run else None,
        "notes": {
            "pin_kinds": p["exact"] or None, "pins_at_other_bindings": len(other) or None,
            "boolean_version": p["boolean_version"].get("id"), "catalog_root": roots[tid]["root"],
            "staged_basis": None if staged is None else ("table" if not staged.startswith("census:") else "census template resolves (dry)"),
            "statements": hits or None,
            "cut_unit_statements": {u: h for u, h in cut_hits.items() if h} or None,
        },
    })

# Friday's select (vllm on NET-390) and its blocks, at the commit that registers and cards them
fr = [(FRIDAY, {"V": 154880, "N": 7744})]
fr += [(k, None) for k in sorted(card_f, key=lambda k: (k.startswith("GreedyCombine"), len(k), k)) if "TokenSelect_v1" in card_f[k]["of"]
       and k.startswith(("GreedyChunk_v2", "GreedyCombine_v2"))]
for tid, b in fr:
    if b is None:
        b = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in tid[tid.index("{") + 1:-1].split(",")}
    accepted, acc_def, run, hits = verdict_of(tid, tid)
    rows.append({
        "template": tid, "definition": tid, "bindings": b, "instances": None, "in_list": False,
        "pin": "this-binding" if tid == FRIDAY else "none", "pin_binding": None,
        "cut": sorted(k for k in card_f if k.startswith(("GreedyChunk_v2{N=7744,", "GreedyCombine_v2{V=154880,N=7744}"))) if tid == FRIDAY else [tid],
        "staged": None, "accepted": accepted, "accepted_definition": acc_def, "run": run, "source": "read" if run else None,
        "notes": {"friday": "Friday's binding, not in the list",
                  "where": f"registered, pinned (digests) and carded at {FRIDAY_AT} (cursor/proofs-gather-served-select-b88b), "
                           f"not on main {MAIN[:12]}: main's registry has no GreedySelect_v1, main's stage.py no name for it",
                  "card": f"{FRIDAY_AT}'s glm53_blocks.json (of TokenSelect_v1)", "statements": hits or None},
    })

ls = [x for x in rows if x["in_list"]]
cover = {
    "schema": "circuits/request-cover/v1", "tree": MAIN, "list_run": LIST_RUN, "lead_run": LEAD_RUN,
    "request": {"model": "GLM-5.3", "target": "H200, TP8, block FP8 (vLLM)", "class": "G", "ranks": 8, "rank": 0,
                "prompt_ids": 256, "outputs": 32, "decode_steps": lst["T"], "execution": "eager (compiled equal)", "sampling": "greedy"},
    "totals": {"templates": len(ls), "instances": sum(x["instances"] for x in ls)},
    "list_check": {"templates": lst["templates"], "instances": lst["instances"]},
    "ranks": {k: {"templates": v["templates"], "instances": v["instances"], "only_here": v["only_here"] or None,
                  "only_rank0": v["only_rank0"] or None, "count_differs": v["count_differs"] or None} for k, v in ranks.items()},
    "sources": {"cir259_runs": sorted({a["run"] for a in J(acc_path)}), "cir259_branch": BRANCH_259, "audit_of_record": AUDIT_RUN,
                "audit_templates": sorted(audit_ids), "card": "main benchmarks/sampling_plan/glm53_blocks.json",
                "card_blocks": len(card), "friday_at": FRIDAY_AT},
    "rows": rows,
}
assert cover["totals"] == cover["list_check"], (cover["totals"], cover["list_check"])
(HERE / "cover.json").write_text(json.dumps(cover, indent=1, sort_keys=False) + "\n")
subprocess.check_call([sys.executable, str(HERE / "cover_md.py")])
