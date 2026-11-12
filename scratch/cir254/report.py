"""CIR-254: the phase table of one staging, or two side by side, from `stage_profile.py`'s records.

    python report.py DIR/CASE.json [DIR/CASE.json]     Markdown to stdout
    python report.py --json DIR/CASE.json              the phases as JSON

A phase is a run of marks (`PHASES`): its seconds are theirs summed, its resident memory the last one's, its high-water mark
their largest.  The counting the profiler does itself (``count wires``, ``count layout``) is left out of every figure.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

GIB = 2**30
#: (phase, what it is, the marks that end inside it)
PHASES = [
    ("read off the Definition", "bind, description (`program.json`), ANDs, Flock block, word view",
     ("imports", "bind", "description", "definition_ands", "unit_block", "word_block", "> stager", "> refuse_past_range",
      "< refuse_past_range", "> unit_lowering", "> unit_ports", "< unit_ports", "> unit_circuit", "> lower", "> standalone",
      "< standalone", "> lower_gates")),
    ("lower the gates", "`ir_lower.lower_gates`: every gate to linear forms and ANDs, each gate's word kept in `wires`",
     ("< lower_gates",)),
    ("drop the words", "`ir_lower.lower` returns: `wires` freed, the circuit's outputs set", ("< lower", "< unit_circuit")),
    ("live cone", "`Circuit.live` in `ir_lower.layout`", ("> layout", "> live", "< live")),
    ("lay out the rows", "`ir_lower.layout`: rows A and B as lists of columns, beside the circuit", ("< layout",)),
    ("drop the circuit", "`unit_lowering` returns: the lowered circuit freed, the layout kept", ("< unit_lowering",)),
    ("slot circuits", "`cache_slot_circuits`: sha512x3 and hm96, the same for every unit", ("cache_slot_circuits", "layout#", "live#")),
    ("zero instance", "`circuit._dummy`: the laid-out rows run on the all-zero instance", ("> compose_to", "> compose", "> _dummy", "< _dummy")),
    ("write the circuit", "`circuit._write_netlist` and the rest of `compose`: the file, 1,024 rows at a time",
     ("> _write_netlist", "< _write_netlist", "< compose")),
    ("read it back", "`circuit._file_digests`: the pin and the class digest", ("> _file_digests", "< _file_digests", "< compose_to")),
    ("meta", "`meta_of`, the nets' counts, `ks_form_why`", ("meta_of", "< stager", "meta", "ks_form_why")),
    ("drop the stager", "the stager and its layout freed", ("del stager", "end")),
]
SKIP = ("count wires", "count layout")


def phase_of(mark: str) -> int | None:
    if mark in SKIP:
        return None
    for k, (_n, _w, marks) in enumerate(PHASES):
        if mark in marks:
            return k
    for k, (_n, _w, marks) in enumerate(PHASES):
        if any(m in mark for m in marks if not m.startswith(("<", ">"))):
            return k
    raise KeyError(mark)


def phases(rec: dict) -> list[dict]:
    out = [{"phase": n, "what": w, "seconds": 0.0, "rss": None, "hwm": 0, "py_in_use": None, "glibc_in_use": None, "traced": None}
           for n, w, _ in PHASES]
    order = []
    for m in rec["marks"]:
        k = phase_of(m["mark"])
        if k is None:
            continue
        if k not in order:
            order.append(k)
        p = out[k]
        p["seconds"] += m["dt"]
        p["rss"], p["hwm"] = m["rss"], max(p["hwm"], m["hwm"], m["rss"])
        if m.get("py"):
            p["py_in_use"], p["py_arenas"] = m["py"].get("in_use"), m["py"].get("arenas")
        if m.get("glibc"):
            p["glibc_in_use"], p["glibc_free"], p["glibc_mmapped"] = m["glibc"]["in_use"], m["glibc"]["free"], m["glibc"]["mmapped"]
        if "traced" in m:
            p["traced"], p["traced_peak"] = m["traced"], max(p.get("traced_peak", 0), m["traced_peak"])
    peak = 0
    got = []
    for k in order:
        peak = max(peak, out[k]["hwm"])
        out[k]["so_far"] = peak
        out[k]["seconds"] = round(out[k]["seconds"], 1)
        got.append(out[k])
    return got


def g(b) -> str:
    return "" if b is None else f"{b / GIB:,.2f}"


def table(recs: list[dict]) -> str:
    cols = [phases(r) for r in recs]
    names = [p["phase"] for p in max(cols, key=len)]
    head = "| phase |" + "".join(f" {r['id']}: s | at end, GiB | high-water in it | so far |" for r in recs)
    lines = [head, "|---|" + "---:|---:|---:|---:|" * len(recs)]
    for n in names:
        row = f"| {n} |"
        for c in cols:
            p = next((p for p in c if p["phase"] == n), None)
            row += " | | | |" if p is None else f" {p['seconds']:,.0f} | {g(p['rss'])} | {g(p['hwm'])} | {g(p['so_far'])} |"
        lines.append(row)
    row = "| **whole run** |"
    for r, c in zip(recs, cols):
        row += f" **{sum(p['seconds'] for p in c):,.0f}** | | | **{g(max(p['hwm'] for p in c))}** |"
    lines.append(row)
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    as_json = "--json" in argv
    recs = [json.loads(Path(a).read_text()) for a in argv if not a.startswith("--")]
    if as_json:
        print(json.dumps({r["name"]: phases(r) for r in recs}, indent=1))
    else:
        print(table(recs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
