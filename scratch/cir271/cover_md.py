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
if "blocks" in c:
    kb = k["blocks"]
    out += ["", "## By block type (main's card)", "",
            f"Block run `{c['block_run']}`. Card: {kb['types']} block types, {kb['blocks']:,} blocks ({kb['prefill']:,} prefill, "
            f"{kb['decode']:,} decode); {kb['extra_units']} rows more for units of list templates the card does not cut.", "",
            "| column | state | block types | blocks |", "|---|---|---:|---:|"]
    for col, states in kb["counts"].items():
        for s, (t, n) in states.items():
            out.append(f"| {col} | {s} | {t} | {n:,} |")
    fc = c["first_cut"]
    out += ["", "First cut (the coordinator's join by exact name): " + "; ".join(f"{s} {t} types, {n:,} blocks" for s, (t, n) in fc.items()) + ".",
            "", "## Unaccepted block types over 3,000,000 blocks", "", "| block | blocks | lacks |", "|---|---:|---|"]
    for x in c["lacks_over_3m"]:
        out.append(f"| {code(x['block'])} | {x['blocks']:,} | {'; '.join(x['lacks'])} |")
    out += ["", "## The card's five places", ""]
    for t in c["card_places"]["templates"]:
        out.append(f"- {code(t['template'])}: {t['instances_rank0']:,} instances at rank 0; the card's names {code(t['card_names'])} "
                   f"({t['card_blocks']:,} blocks); whole, it would add {t['whole_would_add']:,} (8 ranks × instances)")
        for u in t["registered_cuts"]:
            out.append(f"  - {code(u['unit'])} ×{u['per_instance']}: registered {u['registered']}, file {code(u['owner_file'])}, "
                       f"pinned {u['pinned']}, staged {code(u['staged'])}, would add {u['would_add']:,}")
    out += ["", "The card's last three commits on main:", ""] + [f"    {x}" for x in c["card_places"]["card_log"] if x]
    out += ["", "## Every block type", "",
            "| # | block | blocks | list templates | pinned | staged | staging run (GiB) | accepted | form | run | units | device | tree |",
            "|---:|---|---:|---|---|---|---|---|---|---|---:|---|---|"]
    for i, b in enumerate(sorted(c["blocks"], key=lambda b: (-b["blocks"], b["block"])), 1):
        lt = b["list_templates"] or []
        lts = "none" if not lt else f"{code(lt[0][0].partition('{')[0])} ({lt[0][1]:,})" + (f" +{len(lt) - 1}" if len(lt) > 1 else "")
        sr = b["staging_run"]
        srs = f"{sr['run']} ({sr['peak_gib']})" if sr else ("failed: " + ", ".join(f"{a['run']} rc {a['rc']} ({a['peak_gib']})" for a in b["staging_attempts"]) if b["staging_attempts"] else "none")
        ar = b["accepted_run"] or {}
        tag = "" if b["on_card"] else " *(not on the card)*"
        out.append(f"| {i} | {code(b['block'])}{tag} | {b['blocks']:,} | {lts} | {b['pinned']} | {code(b['staged']) if b['staged'] else 'none'} | "
                   f"{srs} | {b['accepted']} | {b['form'] or ''} | {ar.get('run', '')} | {ar.get('units', '')} | {ar.get('device', '')} | {ar.get('tree', '')} |")
(HERE / "cover.md").write_text("\n".join(out) + "\n")
print(f"cover.md: {len(rows)} rows" + (f", {len(c['blocks'])} block types" if "blocks" in c else ""))
