"""Render the inference-only policy's literature index as Markdown on stdout: the parameters with the findings each estimate merges, the
board's slowdown brackets, and every source with its quoted findings, by topic. Never committed (README.md, "Rendering").

    python benchmarks/nci/render.py > /tmp/nci-index.md
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arguments  # noqa: E402
import catalog  # noqa: E402
import slowdown  # noqa: E402


def _num(x: float) -> str:
    return f"{x:.3g}"


def render() -> str:
    sources, parameters = catalog.load("sources"), catalog.load("parameters")
    rows, cals = catalog.load("rows"), catalog.load("calibrations")
    fs = catalog.findings(sources)
    by_source = {s["id"]: s for s in sources}
    out = ["# NCI literature index", "",
           f"{len(sources)} sources, {len(fs)} findings, {len(parameters)} parameters, {len(rows)} board rows.", ""]

    out += arguments.render()
    out += ["## Parameters", ""]
    for p in parameters:
        out += [f"### `{p['id']}` ({p['symbol']}, {p['unit']})", "", p["meaning"], ""]
        for e in p.get("estimates", []):
            out.append(f"- **{e['id']}**: {e['context']}. Low {_num(e['low'])}, central {_num(e['central'])}, "
                       f"high {_num(e['high'])} ({e['confidence']} confidence).")
            out.append(f"  - Merge: {e['merge']}")
            for ref in e["evidence"]:
                f = fs[ref]
                src = by_source[ref.split('/')[0]]
                out.append(f"  - [{ref}]({src['url']}) {f['claim']}")
        out.append("")

    out += ["## Board", ""]
    for c in cals:
        out += [f"### Calibration `{c['id']}`: K = {_num(c['K'])} bits, Gamma = {_num(c['Gamma'])} gates, "
                f"b_tok = {_num(c['b_tok'])}", "", c["description"], "",
                "| Row | Thread theorem | Partial-sum law | Upper (construction) |", "|---|---|---|---|"]
        for b in slowdown.brackets(rows, [c], parameters):
            chain = _num(b["chain"]) if b["chain"] is not None else "-"
            out.append(f"| `{b['row']}` | {_num(b['lower'])} | {chain} | {_num(b['upper'])} ({b['upper_basis']}) |")
        out += ["", f"Both lower bounds are proven: `{slowdown.THREAD_THEOREM}` and `{slowdown.CHAIN_THEOREM}`, the "
                "second on one forward chain of the row's reference circuit.", ""]
    for r in rows:
        out += [f"- **`{r['id']}`**: {r['method']}, {r['configuration']}. State: {r['state']}."
                + (f" Sources: {', '.join(r['sources'])}." if r.get("sources") else "")]
    out.append("")

    out += ["## Sources by topic", ""]
    for topic in sorted(catalog.TOPICS):
        here = [s for s in sources if topic in s["topics"]]
        if not here:
            continue
        out += [f"### {topic} ({len(here)})", ""]
        for s in sorted(here, key=lambda s: (-s["year"], s["id"])):
            out.append(f"- **{s['id']}**: {s['authors']}, [{s['title']}]({s['url']}), {s['venue']} ({s['verified']}).")
            for f in s["findings"]:
                qs = "; ".join(f"{q['param']} = {q['value']} {q['unit']} ({q['condition']})" for q in f["quantities"])
                out.append(f"  - `{f['id']}` [{f['grade']}] {f['claim']}" + (f" Quantities: {qs}." if qs else ""))
                out.append(f"    > {f['quote']} ({f['location']})")
                out.append(f"    Bears on: {f['bears_on']}")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    sys.stdout.write(render())
