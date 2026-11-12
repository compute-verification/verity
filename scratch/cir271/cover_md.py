"""cover.md from cover.json (beside this script); never edited by hand."""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
c = json.loads((HERE / "cover.json").read_text())
k = json.loads(subprocess.check_output([sys.executable, str(HERE / "cover_check.py"), str(HERE / "cover.json")]))
rows = c["rows"]
fam = lambda t: t.partition("{")[0]  # noqa: E731
code = lambda s: "none" if s is None else "`" + str(s).replace("|", "\\|") + "`"  # noqa: E731

out = [f"# Request cover: {c['request']['model']} rank 0, class {c['request']['class']}",
       "",
       f"Made by `cover_md.py` from `cover.json` (`{c['schema']}`). Tree: main `{c['tree'][:12]}`. List: `{c['list_run']}` "
       f"(recomputing `{c['lead_run']}`): {c['totals']['templates']} templates, {c['totals']['instances']:,} instances. "
       f"Request: {c['request']['prompt_ids']} prompt ids, {c['request']['outputs']} outputs ({c['request']['decode_steps']} decode steps), "
       f"{c['request']['execution']}, {c['request']['sampling']}, {c['request']['target']}.",
       "",
       "Ranks 1 to 7 against rank 0:",
       ""]
for r, v in c["ranks"].items():
    out.append(f"- rank {r}: {v['templates']} templates, {v['instances']:,} instances; only here {code(v['only_here'])}; "
               f"only on rank 0 {code(v['only_rank0'])}; counts differing {code(v['count_differs'])}")
out += ["", "## Counts (templates / instances, list rows only)", "", "| column | state | templates | instances |", "|---|---|---:|---:|"]
for col, states in k["counts"].items():
    for s, (t, n) in states.items():
        out.append(f"| {col} | {s} | {t} | {n:,} |")
out += ["", f"No accepted statement (neither `yes` nor `boolean-version`): {k['uncovered'][0]} templates, {k['uncovered'][1]:,} instances.",
        "", "## Without an accepted statement, by family, largest first", "",
        "| family | templates | instances | pin | cut | staged | cut units with an accepted statement |", "|---|---:|---:|---|---|---|---|"]
def acc_units(rs) -> str:
    us = {u for r in rs for u, h in (r["notes"]["cut_unit_statements"] or {}).items() if any(": accepted at " in x for x in h)}
    return ", ".join(f"`{u.partition('{')[0]}`" for u in sorted(us)) + (f" ({len(us)})" if us else "none")


by = defaultdict(list)
for r in rows:
    if r["in_list"] and r["accepted"] not in ("yes", "boolean-version"):
        by[fam(r["template"])].append(r)
for f, rs in sorted(by.items(), key=lambda kv: -sum(r["instances"] for r in kv[1])):
    agg = lambda key: ", ".join(sorted({str(key(r)) for r in rs}))  # noqa: E731
    out.append(f"| `{f}` | {len(rs)} | {sum(r['instances'] for r in rs):,} | {agg(lambda r: r['pin'])} | "
               f"{agg(lambda r: 'cut' if r['cut'] else 'none')} | "
               f"{agg(lambda r: 'none' if r['staged'] is None else ('census' if r['staged'].startswith('census:') else r['staged']))} | "
               f"{acc_units(rs)} |")
out += ["", "## Every row", "",
        "| # | template | definition | instances | pin | cut | staged | accepted | run | source |", "|---:|---|---|---:|---|---|---|---|---|---|"]
for i, r in enumerate(rows, 1):
    d = "" if r["definition"] == r["template"] else code(r["definition"])
    pin = r["pin"] + (f" ({code(r['pin_binding'])})" if r["pin_binding"] else "")
    cut = "none" if not r["cut"] else (code(r["cut"][0]) + (f" +{len(r['cut']) - 1}" if len(r["cut"]) > 1 else ""))
    accd = r["accepted"] + (f" ({code(r['accepted_definition'])})" if r["accepted_definition"] else "")
    tag = "" if r["in_list"] else " *(Friday's binding, not in the list)*"
    out.append(f"| {i} | {code(r['template'])}{tag} | {d} | {'' if r['instances'] is None else format(r['instances'], ',')} | {pin} | {cut} | "
               f"{code(r['staged']) if r['staged'] else 'none'} | {accd} | {r['run'] or ''} | {r['source'] or ''} |")
(HERE / "cover.md").write_text("\n".join(out) + "\n")
print(f"cover.md: {len(rows)} rows")
