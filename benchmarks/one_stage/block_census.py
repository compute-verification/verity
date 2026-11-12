"""The proof-block census of a served Commit: every Program row of every rank and request step (`m1_families.py`'s census
of each rank, sized with ``--estimate``) cut into proof blocks, and the proof blocks counted per Definition, for the prefill
step (step 0) and the decode steps, per token and per request.

A row's cut is its own Definition, or else a unit template for each slot of its split (`serving_rows.unit_templates`):
a ``CUT`` row's batched members, a ``PARTS`` row's parts, a ``CHAINS`` row's key blocks, ends, column ends and combines and
a ``STAGES`` row's stages are a slot each; a
``COORDS``, ``COLS`` or ``SLICES`` row has one slot, its units listed coarse to fine, and a ``KCHUNKS`` row's slot is its
coordinate or its K-chunks of one width. A unit template fits when it binds, Flock has a piece for every leaf, its block
(`m1_families.block`) is at most `m1_families.K_MAX` and its outputs fit a public size class (`partition.unit_class`). The
row itself is taken when it fits, else each slot's coarsest fitting option. A ``COLS`` or ``WIDE_COORDS`` row, and a
split's ``COLS`` or ``WIDE_COORDS`` slot (the block-scaled FP8 GEMM's coordinates of c outputs), instead take the fitting
width of least leakage-weighted proving cost (`width_pick`, the row's own width among them). A row
with a slot that has none is a gap, with its finest option's reason. A row whose Definition has no gate (a gather) or no
input (a constant) is no proof block.

Per proof block: its output bits (its returned fields') and size class, its input bits, its estimated ANDs (an upper
bound, `m1_families.estimate`), its served MACs and the bytes of the rows it opens (a row per parameter and per returned
field). Beside them, every row's own estimate unsplit (the circuit's ANDs and MACs, whatever the cut).

    python benchmarks/one_stage/block_census.py --sized sized-rank0.json [sized-rank1.json ...] --out DIR
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from m1_families import K_MAX
from verity.protocols.spot.optimize import row_ands, unit_class
from verity.definitions.call_scope import spec_base, spec_statics
from verity_flock import ir_lower as IL
from verity_vllm.pipeline import serving_rows as ST

PHASES = ("prefill", "decode")
#: the bits a sampled block's position adds to what its outputs reveal: the planner's location bits, log2(e R n) for R replay
#: blocks of n units of a type (`optimize.two_stage_design`), at the scale of the world's compute
LOCATION_BITS = 80
#: a ``COLS`` row's column blocks and a ``WIDE_COORDS`` row's coordinates are no smaller than 2^WIDTH_FLOOR rows while a wider
#: width fits: below it a block is mostly its rows' hashing slots (Flock's 2^18-row slots), so a narrower one proves about as
#: much and adds a hiding leaf a row to every token's commitments
WIDTH_FLOOR = 22


def misfit(e: dict) -> str | None:
    """Why the sized unit ``e`` (`m1_families.sized`) is no proof block, or None when it fits."""
    if "error" in e:
        return e["error"]
    if e.get("unpieced"):
        return IL.no_piece(e["unpieced"][:3])
    b = e["block"]
    if b["rows_log"] > K_MAX:
        return f"its rows alone need a 2^{b['rows_log']} block"
    if b["est_log"] > K_MAX:
        return f"a 2^{b['est_log']} block at {e['ands']:,} ANDs"
    if unit_class(sum(b["out_bits"])) is None:
        return f"{sum(b['out_bits']):,} output bits, past the widest size class"
    return None


def slots(spec: str, units: list[dict]) -> list[list[list[dict]]]:
    """The slots of a split of row ``spec`` over its sized units (``units``, the row's unit templates past itself): each a
    list of options coarse to fine, an option the units it takes.  A ``PARTS`` row's split is its ``PARTS`` cut when every part
    has units, else its ``FINAL`` cut (the norm whole: the census names no op_path to tell a layer's norm from the final one)."""
    base = spec_base(spec)
    if base in ST.PARTS:
        for c in (ST.PARTS[base], ST.FINAL.get(base, ())):
            got = [[[u] for u in units if spec_base(u["spec"]) == p[0]] for p in c]
            if all(got):
                return got
    if base in ST.CUT or base in ST.CHAINS or base in ST.STAGES:
        return [[[u]] for u in units]
    if base in ST.KCHUNKS:
        whole, chunks = [], defaultdict(list)
        for u in units:
            if spec_base(u["spec"]) == ST.KCHUNKS[base]:
                chunks[spec_statics(u["spec"])["K"]].append(u)
            else:
                whole.append([u])
        return [whole + list(chunks.values())]
    return [[[u] for u in units]] if units else []


def leak_cost(units: list[dict]) -> float:
    """What the sampled-proofs planner pays to prove ``units`` (each taken ``n`` times), up to one factor: a type's samples
    grow with its blocks' outputs at their size class plus `LOCATION_BITS`, and each sample's proving with its block's rows."""
    return sum(u["n"] * (unit_class(sum(u["block"]["out_bits"])) + LOCATION_BITS) * 2 ** u["block"]["est_log"] for u in units)


def widths(spec: str) -> bool:
    """Whether a row or unit of ``spec`` comes in widths `width_pick` chooses between: a ``COLS`` family's column blocks or a
    ``WIDE_COORDS`` family's coordinates."""
    return spec_base(spec) in ST.COLS or spec_base(spec) in ST.WIDE_COORDS


def width_pick(opts: list[list[dict]]) -> list[dict] | None:
    """A ``COLS`` or ``WIDE_COORDS`` slot's option (its widths coarse to fine): of the fitting ones no smaller than
    2^`WIDTH_FLOOR` rows, the one of least `leak_cost`, the coarser on a tie; the coarsest fitting one when none is that large;
    None when none fits."""
    fit = [o for o in opts if all(misfit(u) is None for u in o)]
    big = [o for o in fit if all(u["block"]["est_log"] >= WIDTH_FLOOR for u in o)]
    return min(big, key=leak_cost) if big else next(iter(fit), None)


def cut(spec: str, f: dict) -> tuple[str, list[dict], str | None]:
    """Row family ``f``'s cut: ``row``, ``split``, ``gap`` (with why), ``no gates`` or ``const``, and its units, each taken
    ``n`` times a row."""
    e = f["estimate"]
    if f.get("word_gates") == 0:
        return "no gates", [], None
    if (f.get("ports") is not None and not f["ports"]["in"]) or ("block" in e and not sum(e["block"]["in_bits"])):
        return "const", [], None
    why = misfit(e)
    whole = [{**e, "spec": spec, "n": 1}]
    ss = slots(spec, e.get("units", []))
    if widths(spec) and (pick := width_pick([whole, *(ss[0] if ss else [])])) is not None:
        return ("row" if pick == whole else "split"), pick, None
    if why is None:
        return "row", whole, None
    got = []
    for opts in ss:
        wide = all(widths(u["spec"]) for o in opts for u in o)
        pick = width_pick(opts) if wide else next((o for o in opts if all(misfit(u) is None for u in o)), None)
        if pick is None:
            worst = next((u for u in (opts[-1] if opts else []) if misfit(u) is not None), None)
            return "gap", [], f"{why}; finest split {worst['spec']}: {misfit(worst)}" if worst else why
        got += pick
    return ("split", got, None) if ss else ("gap", [], why)


def _per_unit(u: dict) -> dict:
    b = u["block"]
    out = sum(b["out_bits"])
    nbytes = [-(-x // 8) for x in b["in_bits"]] + [2 * -(-x // 16) for x in b["out_bits"]]
    return {"out_bits": out, "class_bits": unit_class(out), "in_bits": sum(b["in_bits"]), "ands": u["ands"],
            "macs": u.get("macs", 0), "reads": u.get("reads", 0), "asserts": u.get("asserts", 0), "rows": nbytes,
            "row_bytes": sum(nbytes), "open_ands": sum(row_ands(x) for x in nbytes), "k_log": b["est_log"],
            "rows_log": b["rows_log"]}


#: the summed quantities of a population of proof blocks, each a proof block's field times its count
SUMS = ("out_bits", "class_bits", "ands", "macs", "row_bytes", "open_ands")


def census(docs: list[dict]) -> dict:
    """The census of the sized rank censuses ``docs``: every proof-block Definition with its counts per phase, the totals
    per phase, per token and per request, the per-rank totals, and the gaps."""
    defs: dict[str, dict] = {}
    gaps: dict[str, dict] = {}
    other: dict[str, dict] = {}
    undescribed: dict[str, int] = defaultdict(int)
    rows = {p: defaultdict(int) for p in PHASES}
    ranks: dict[int, dict] = {}
    lengths = set()
    for doc in docs:
        rk = ranks.setdefault(int(doc["rank"]), {p: defaultdict(int) for p in PHASES})
        for req in doc["requests"].values():
            lengths.add((req["prompt_len"], req["steps"]))
            for spec, f in req["families"].items():
                kind, units, why = cut(spec, f)
                e = f["estimate"]
                by = {p: sum(n for s, n in f["steps"].items() if (int(s) == 0) == (p == "prefill")) for p in PHASES}
                if f.get("undescribed"):
                    undescribed[spec] += f["undescribed"]
                for p in PHASES:
                    rows[p]["rows"] += by[p]
                    rows[p][f"rows {kind}"] += by[p]
                    rows[p]["circuit_ands"] += by[p] * e.get("ands", 0)
                    rows[p]["circuit_macs"] += by[p] * e.get("macs", 0)
                if kind == "gap":
                    g = gaps.setdefault(spec, {"family": spec_base(spec), "why": why, "ands": e.get("ands"), **dict.fromkeys(PHASES, 0)})
                    for p in PHASES:
                        g[p] += by[p]
                        rows[p]["gap_ands"] += by[p] * e.get("ands", 0)
                elif kind in ("no gates", "const"):
                    o = other.setdefault(spec, {"family": spec_base(spec), "kind": kind, **dict.fromkeys(PHASES, 0)})
                    for p in PHASES:
                        o[p] += by[p]
                for u in units:
                    d = defs.get(u["spec"])
                    if d is None:
                        d = defs[u["spec"]] = {"family": spec_base(u["spec"]), **_per_unit(u), "of": {},
                                               **dict.fromkeys(PHASES, 0)}
                    d["of"][spec_base(spec)] = kind
                    for p in PHASES:
                        d[p] += by[p] * u["n"]
                        for k in SUMS:
                            rk[p][k] += by[p] * u["n"] * d[k]
                        rk[p]["blocks"] += by[p] * u["n"]
    if len(lengths) != 1:
        raise ValueError(f"the ranks' requests are of {sorted(lengths)} (prompt, steps): one request shape a census")
    (prompt, steps), = lengths
    tot = {p: {"blocks": sum(d[p] for d in defs.values()), **{k: sum(d[p] * d[k] for d in defs.values()) for k in SUMS},
               **rows[p]} for p in PHASES}
    tokens = {"prefill": prompt, "decode": steps - 1}
    per_token = {p: {k: v / tokens[p] for k, v in tot[p].items()} for p in PHASES}
    request = {k: tot["prefill"].get(k, 0) + tot["decode"].get(k, 0) for k in {*tot["prefill"], *tot["decode"]}}
    return {"ranks": sorted(ranks), "prompt_len": prompt, "steps": steps, "tokens": tokens,
            "totals": tot, "per_token": per_token, "per_request": dict(sorted(request.items())),
            "per_rank": {r: {p: dict(v[p]) for p in PHASES} for r, v in sorted(ranks.items())},
            "definitions": dict(sorted(defs.items(), key=lambda kv: -(kv[1]["prefill"] + kv[1]["decode"]) * kv[1]["ands"])),
            "gaps": dict(sorted(gaps.items(), key=lambda kv: -(kv[1]["prefill"] + kv[1]["decode"]))),
            "not_blocks": other, "undescribed": dict(undescribed)}


def _by_family(rows: dict[str, dict]) -> list[tuple[str, list[dict]]]:
    """``rows`` grouped by family (a primitive's bracketed constant dropped), the most rows first."""
    out: dict[str, list[dict]] = defaultdict(list)
    for s, r in rows.items():
        out[re.sub(r"\[[^\]]*\]", "", spec_base(s))].append(r)
    return sorted(out.items(), key=lambda kv: (-sum(r["prefill"] + r["decode"] for r in kv[1]), kv[0]))


def gap_families(gaps: dict[str, dict]) -> dict[str, dict]:
    """The gaps by row family: its Definitions, its rows per phase in one request, their ANDs and why its most rows'
    Definition fits no cut."""
    out = {}
    for fam, gs in _by_family(gaps):
        top = max(gs, key=lambda g: g["prefill"] + g["decode"])
        out[fam] = {"definitions": len(gs), "prefill": sum(g["prefill"] for g in gs), "decode": sum(g["decode"] for g in gs),
                    "ands": sum((g["prefill"] + g["decode"]) * (g["ands"] or 0) for g in gs), "why": top["why"]}
    return out


def _n(x: float) -> str:
    if isinstance(x, float) and not x.is_integer():
        return f"{x:,.1f}" if abs(x) < 1e6 else f"{x:.3e}"
    return f"{int(x):,}" if abs(x) < 1e12 else f"{x:.3e}"


def table(c: dict) -> str:
    """The census as markdown: the totals, one line per proof-block Definition, the gaps and the rows that are no proof
    block."""
    t, pt = c["tokens"], c["per_token"]
    keys = (("blocks", "proof blocks"), ("out_bits", "output bits"), ("class_bits", "output bits at size class"),
            ("ands", "proof blocks' ANDs (estimate)"), ("open_ands", "ANDs opening their rows"),
            ("row_bytes", "opened row bytes"), ("circuit_ands", "the circuit's ANDs (rows unsplit)"),
            ("circuit_macs", "served MACs"), ("rows", "Program rows"), ("rows gap", "rows in a gap"),
            ("gap_ands", "their ANDs"))
    out = [f"Ranks {c['ranks'][0]}-{c['ranks'][-1]}, one request: a prefill step of {t['prefill']} prompt tokens and "
           f"{t['decode']} decode steps.", "",
           "| quantity | prefill token | decode token | request |", "| --- | ---: | ---: | ---: |"]
    for k, name in keys:
        out.append(f"| {name} | {_n(pt['prefill'].get(k, 0))} | {_n(pt['decode'].get(k, 0))} | {_n(c['per_request'].get(k, 0))} |")
    out += ["", "| proof-block Definition | of | prefill token | decode token | out bits | class | in bits | ANDs | MACs | "
            "opened bytes | block |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s, d in c["definitions"].items():
        of = ", ".join(f"{k} ({v})" for k, v in sorted(d["of"].items()))
        out.append(f"| `{s}` | {of} | {_n(d['prefill'] / t['prefill'])} | {_n(d['decode'] / t['decode'])} | {d['out_bits']:,} | "
                   f"{d['class_bits']:,} | {d['in_bits']:,} | {d['ands']:,} | {d['macs']:,} | {d['row_bytes']:,} | 2^{d['k_log']} |")
    out += ["", "Gaps: rows no cut fits, by family (each Definition in `census.json`).", "",
            "| row family | Definitions | prefill rows | decode rows | their ANDs | why (its most rows' Definition) |",
            "| --- | ---: | ---: | ---: | ---: | --- |"]
    for fam, g in gap_families(c["gaps"]).items():
        out.append(f"| `{fam}` | {g['definitions']} | {g['prefill']:,} | {g['decode']:,} | {_n(g['ands'])} | {g['why']} |")
    out += ["", "No proof block: rows of no gate or no input, by family.", "",
            "| row family | kind | Definitions | prefill rows | decode rows |", "| --- | --- | ---: | ---: | ---: |"]
    for fam, os_ in _by_family(c["not_blocks"]):
        out.append(f"| `{fam}` | {os_[0]['kind']} | {len(os_)} | {sum(o['prefill'] for o in os_):,} | {sum(o['decode'] for o in os_):,} |")
    if c["undescribed"]:
        out += ["", "Rows the serving rows can't address (`describe`), counted above: "
                + ", ".join(f"`{s}` {n:,}" for s, n in sorted(c["undescribed"].items()))]
    return "\n".join(out) + "\n"


def card(c: dict) -> dict:
    """The planner's card (`benchmarks/sampling_plan/glm53.py` reads it): per proof-block Definition its counts in one
    request, output bits, ANDs, MACs, opened rows' bytes and staged block log2, and the gaps by row family
    (`gap_families`)."""
    keep = ("family", "prefill", "decode", "out_bits", "ands", "macs", "rows", "k_log")
    return {"prompt_len": c["prompt_len"], "decode_steps": c["tokens"]["decode"], "ranks": len(c["ranks"]),
            "definitions": {s: {k: d[k] for k in keep} | {"of": sorted(d["of"])} for s, d in c["definitions"].items()},
            "gaps": gap_families(c["gaps"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sized", nargs="+", required=True, help="each rank's `m1_families.py --estimate` output")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    c = census([json.loads(Path(p).read_text()) for p in a.sized])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "census.json").write_text(json.dumps(c, indent=1, default=str) + "\n")
    (a.out / "table.md").write_text(table(c))
    (a.out / "card.json").write_text(json.dumps(card(c), indent=1) + "\n")
    r = c["per_request"]
    print(f"BLOCKS ranks {c['ranks']}: {r['blocks']:,} proof blocks a request ({len(c['definitions'])} Definitions), "
          f"{r['out_bits']:,} output bits, {r['ands']:.3e} ANDs, {r['circuit_macs']:.3e} MACs; "
          f"{len(c['gaps'])} gap Definitions, {r.get('rows gap', 0):,} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
