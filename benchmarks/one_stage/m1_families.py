"""M1's family sizes: a census file's families (each Program row of the request steps by its Definition, with its unit templates,
whether the rows commit it, its port words and its Definition's word gates), each with its Flock size.
``--estimate`` adds each family's Flock size to the census: every distinct primitive (or Boolean Definition) of the row's
Definition lowered once on its own (`ir_lower.lower` on fresh inputs), its ANDs, table reads and assertions times how many
times the Definition holds it, with its served MACs and each port's bits. Its ANDs are the sum of each leaf's ANDs,
lowered alone, times how often the Definition holds it. Nothing is taken off for constants or unread outputs. Table reads
are counted apart (``reads``), so a statement with its reads inline, whose ``ands`` counts their rows
(`ir_lower.layout`), can exceed it: the estimate is not that statement's ``ands`` and is not compared with it.
With it, M0's block for the unit (`block`): the in-circuit hashing of its rows alone bounds it from below, so a family whose
``rows_log`` passes `K_MAX` is refused whatever its ANDs.  The same is computed for each family's split units (its unit
templates past the row itself: ``units``), and for each ``--also`` Definition (``candidates``: a split's unit main lacks).

    PYTHONPATH=...:<tree>/verity/protocols/proof/flock/prover/python python benchmarks/one_stage/m1_families.py --estimate families.json --out sized.json \\
      [--also 'CublasSplitKGemmF32Out_v1{K=6144,N=1,..}' ...] [--also-file SPECS.txt]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from verity.commitments.periods.interface import K_MAX

#: the statement limit the brief measures against
LIMIT_LOG = 26


def _leaf_cost(fn, IL) -> dict:
    from verity.circuits.boolean.forms import Circuit

    C = Circuit()
    params, _ret = fn.signature()
    ins = [C.input(f"a{i}.{j}", w) for i, (_n, t) in enumerate(params) for j, w in enumerate(t.leaf_widths())]
    outs = IL.lower(C, fn, ins)
    C.output("o", [b for w in outs for b in w])
    c = C.counts()
    return {"ands": c["ands"], "reads": c["lookups"], "asserts": c["asserts"]}


def estimate(fn, memo: dict, leaves: dict) -> dict:
    """fn's ANDs, table reads and assertions as the sum of its leaves' (a primitive or a Boolean Definition, each lowered once
    by `_leaf_cost`), each times how often fn holds it; ``unpieced`` the leaves Flock has no piece for; ``macs`` its served
    MACs as the census counts them (`census.PRODUCTS` a tensor-core step, a family's `census.FAMILY_COUNTS` rule over it;
    informational, as the census's are)."""
    from verity_flock import ir_lower as IL

    from verity.circuits.census import PRODUCTS, _rule, family
    from verity.circuits.defs import PrimitiveDefinition
    from verity.circuits.program import is_input

    if fn.id in memo:
        return memo[fn.id]
    boolean = callable(getattr(getattr(fn, "defn", fn), "build", None))
    if isinstance(fn, PrimitiveDefinition) or boolean:
        fam = getattr(fn, "family", None)
        if is_input(fn) or fam in ("Input", "Const"):
            r = {"ands": 0, "reads": 0, "asserts": 0, "unpieced": []}
        else:
            try:
                r = {**_leaf_cost(fn, IL), "unpieced": []}
            except IL.NoPiece:
                r = {"ands": 0, "reads": 0, "asserts": 0, "unpieced": [fn.id]}
        r["macs"] = PRODUCTS.get(family(fn), 0)
        leaves[fn.id] = {k: v for k, v in r.items() if k != "unpieced"}
    else:
        r = {"ands": 0, "reads": 0, "asserts": 0, "macs": 0, "unpieced": []}
        for nd in fn.body.nodes:
            if is_input(nd.fn):
                continue
            n = nd.n if nd.form in ("batch", "scan") else 1
            s = estimate(nd.fn, memo, leaves)
            for k in ("ands", "reads", "asserts", "macs"):
                r[k] += n * s[k]
            r["unpieced"] = sorted(set(r["unpieced"]) | set(s["unpieced"]))
        if family(fn) in PRODUCTS:
            r["macs"] = PRODUCTS[family(fn)]
        r["macs"] = (_rule(fn) or {}).get("served_macs", r["macs"])
    memo[fn.id] = r
    return r


def block(fn, ands: int | None) -> dict:
    """The log2 of M0's block for fn's unit as `m1_unit.py` stages it (a ``row/v2`` bit row per parameter, an output row per
    returned field), placed by Flock's own rule (`circuit._ranges`): its rows' compressions, the hm96 slots, the mask and
    the unit's slot, its table reads' slots left out, one unit to the block (the stager packs more while the block stays
    within 2^22). ``rows_log`` takes the unit's slot at its least, a lower bound for any number of ANDs; ``est_log`` sizes
    it at the estimate's ANDs plus the port bits. The unit fits a block at most `K_MAX`. ``in_bits`` and ``out_bits``: each
    parameter's and each returned field's bits."""
    from verity_flock import circuit as FC

    params, ret = fn.signature()
    fields = [f[1] if isinstance(f, tuple) else f for f in ret.fields] if hasattr(ret, "fields") else [ret]
    ins, outs = [sum(t.leaf_widths()) for _n, t in params], [sum(f.leaf_widths()) for f in fields]
    bits = ins + [16 * -(-b // 16) for b in outs]
    ports = [FC.Port(f"p{i}", -(-b // 16), b) for i, b in enumerate(bits)]
    n_sha = FC._row_slots(ports)[0]

    def log(unit_rows: int) -> int:
        return FC._ranges([("unit", FC.SHA_LOG, n_sha, {"packed": True}), ("unit", FC.HM_LOG, FC._pow2(len(ports)), {}),
                           ("unit", FC._slot_log(unit_rows), 1, {}), ("mask", FC.MASK_LOG, 1, {})])[1]

    return {"row_bits": sum(bits), "in_bits": ins, "out_bits": outs, "compressions": sum(p.comps for p in ports),
            "rows_log": log(1), "est_log": None if ands is None else log(ands + sum(bits))}


def sized(spec: str, memo: dict) -> dict:
    """`estimate` and `block` of the Definition ``spec`` binds to (`serving_rows._bind`), or the reason it has none."""
    import math

    from verity_vllm.pipeline import serving_rows as SRW

    t = time.time()
    try:
        sp = SRW._bind(spec)
        e = estimate(sp, memo, {}) if sp is not None else {"error": "does not bind"}
    except Exception as ex:  # noqa: BLE001 -- reported, not fatal
        e = {"error": f"{type(ex).__name__}: {ex}"[:300]}
    if "ands" in e:
        e["log2_ands"] = round(math.log2(e["ands"]), 2) if e["ands"] else None
        e["under_2^26"] = e["ands"] < 2**LIMIT_LOG
        e["block"] = b = block(sp, e["ands"])
        e["fits_block"] = b["est_log"] <= K_MAX
        e["refused_block"] = b["rows_log"] > K_MAX
    e["seconds"] = round(time.time() - t, 1)
    return e


def _say(rid: str, rows: int | str, spec: str, e: dict) -> None:
    print(f"M1S {rid} {rows:>5} {spec[:90]:90s} ands={e.get('ands')} log2={e.get('log2_ands')} reads={e.get('reads')} "
          f"asserts={e.get('asserts')} unpieced={e.get('unpieced')} block={e.get('block')} {e.get('error', '')}", flush=True)


def sizes(census: str, out: str, also: list[str] = ()) -> int:
    """Each family's estimate, each of its split units' (its `unit_templates` past the row itself, ``units``: per row ``n``
    of them), and with ``also`` each of those Definitions' (``candidates``: the units of a split main doesn't have)."""
    import sys

    sys.setrecursionlimit(100000)
    doc = json.loads(Path(census).read_text())
    memo: dict = {}
    seen: dict[str, dict] = {}
    for rid, req in doc["requests"].items():
        for spec, f in req["families"].items():
            f["estimate"] = e = sized(spec, memo)
            _say(rid, f["rows"], spec, e)
            e["units"] = []
            for u, n in f.get("unit_templates", [])[1:]:
                if u not in seen:
                    seen[u] = sized(u, memo)
                    _say(rid, f"x{n}", u, seen[u])
                e["units"].append({"spec": u, "n": n, **seen[u]})
    doc["candidates"] = {}
    for s in also:
        doc["candidates"][s] = sized(s, memo)
        _say("candidate", "-", s, doc["candidates"][s])
    Path(out).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", required=True, metavar="CENSUS", help="add each family's estimated size to this census's output")
    ap.add_argument("--also", action="append", default=[], metavar="SPEC",
                    help="with --estimate: also size this Definition (a split's unit that main doesn't have); repeatable")
    ap.add_argument("--also-file", default=None, metavar="PATH", help="with --estimate: --also for each line of this file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    also = a.also + ([s for s in Path(a.also_file).read_text().splitlines() if s.strip()] if a.also_file else [])
    return sizes(a.estimate, a.out, also)


if __name__ == "__main__":
    raise SystemExit(main())
