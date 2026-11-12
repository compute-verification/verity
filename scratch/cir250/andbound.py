"""CIR-250 scratch (never for main): the census's AND estimate (`m1_families.estimate`) beside the staged unit's ANDs, by cause.

    PYTHONPATH=benchmarks/one_stage python scratch/cir250/andbound.py --scratch DIR --out DIR

Prints ``AB <tag> {json}`` lines. For each spec: the estimate's leaves (ANDs, reads, asserts alone, and how often the
Definition holds each, walked as `estimate` walks), then the unit lowered as the stager lowers it (`class_statement.unit_ports`
and `unit_circuit`, then `ir_lower.layout`) with every AND and table read tagged by the estimate leaf whose piece or builder
made it (`forms.Circuit.phase`; `ir_lower.PIECES` and `ir_lower._boolean_activation` wrapped in this process only), so that
staged = estimate + (occurrences - multiplicity) L + (calls - occurrences) L + (created - calls L) + (live - created) + read rows.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.setrecursionlimit(100000)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "one_stage"))

import m1_families as MF  # noqa: E402
from verity.circuits.boolean.forms import Circuit  # noqa: E402
from verity.circuits.defs import PrimitiveDefinition  # noqa: E402
from verity.circuits.program import is_input  # noqa: E402
from verity_flock import class_statement as CS  # noqa: E402
from verity_flock import ir_lower as IL  # noqa: E402
from verity_vllm.pipeline import chunk_rows as CR  # noqa: E402
from verity_vllm.pipeline import serving_rows as SRW  # noqa: E402

COMBINE = 'SparseMlaCombine_v1{NSPLIT=5,DV=64,INV={"fn":"Fa3InvSum_v1"}}'
HEAD64 = ('SparseMlaHeadBlock_v1{DR=64,DV=512,NVIS=64,FIRST=false,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v1"},'
          'SCALE={"f64":"0x1.0000000000000p-4"}}')
SMALL_HEADS = [
    ('SparseMlaHeadBlock_v1{DR=16,DV=16,NVIS=16,FIRST=false,BN=16,DOT={"fn":"HopperBF16WgmmaDot16_v1"},'
     'SCALE={"f64":"0x1.0000000000000p-4"}}'),
    ('SparseMlaHeadBlock_v1{DR=64,DV=64,NVIS=16,FIRST=false,BN=16,DOT={"fn":"HopperBF16WgmmaDot16_v1"},'
     'SCALE={"f64":"0x1.0000000000000p-4"}}'),
]
CARD = ROOT / "benchmarks" / "sampling_plan" / "glm53_blocks.json"
T0 = time.time()


def say(tag: str, **kw) -> None:
    kw["t"] = round(time.time() - T0, 1)
    kw["rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    print(f"AB {tag} " + json.dumps(kw, default=str), flush=True)


def is_leaf(fn) -> bool:
    return isinstance(fn, PrimitiveDefinition) or callable(getattr(getattr(fn, "defn", fn), "build", None))


def walk(fn, k: int, mult: Counter, objs: dict, forms: Counter) -> None:
    """How often ``fn`` holds each leaf, as `m1_families.estimate` multiplies: a ``batch`` or ``scan`` node its ``n``, any
    other node once, input nodes skipped, a leaf (a primitive or a Boolean Definition with a builder) not entered."""
    if is_leaf(fn):
        mult[fn.id] += k
        objs[fn.id] = fn
        return
    for nd in fn.body.nodes:
        if is_input(nd.fn):
            continue
        forms[nd.form] += 1
        walk(nd.fn, k * (nd.n if nd.form in ("batch", "scan") else 1), mult, objs, forms)


_READ_ROWS: dict = {}


def read_rows(C, g: int, bits) -> tuple[int, int]:
    """`ir_lower._Read`'s rows for read ``g`` at its live value ``bits`` (as `ir_lower.layout` lays it inline), and how many of
    those bits it wires to a constant."""
    t, idx, _w = C.lookups[g]
    key = (t, len(idx), tuple(sorted(bits)))
    if key not in _READ_ROWS:
        r = IL._Read(C, g, sorted(bits), 0)
        _READ_ROWS[key] = (r.rows, len(r.const_bits))
    return _READ_ROWS[key]


def leaf_alone(fn) -> dict:
    """The leaf lowered once on fresh inputs, as `m1_families._leaf_cost` lowers it, then laid out by `ir_lower.layout`."""
    C = Circuit()
    params, _ret = fn.signature()
    ins = [C.input(f"a{i}.{j}", w) for i, (_n, t) in enumerate(params) for j, w in enumerate(t.leaf_widths())]
    outs = IL.lower(C, fn, ins)
    C.output("o", [b for w in outs for b in w])
    c = C.counts()
    lay = IL.layout(C, list(C.inputs), ["o"])
    return {"ands": c["ands"], "reads": c["lookups"], "asserts": c["asserts"], "layout_ands": lay.counts["ands"],
            "read_rows": lay.counts["read_rows"], "tables": sorted({t for t, _i, _w in C.lookups}),
            "index_bits": sorted({len(i) for _t, i, _w in C.lookups})}


def estimate_table(spec: str) -> dict:
    """`m1_families.sized` on ``spec``, with its leaves: each leaf's cost alone, its multiplicity, their products' sum."""
    card = json.loads(CARD.read_text())["definitions"].get(spec, {})
    e = MF.sized(spec, {})
    sp = SRW._bind(spec)
    leaves: dict = {}
    e2 = MF.estimate(sp, {}, leaves)
    mult, objs, forms = Counter(), {}, Counter()
    walk(sp, 1, mult, objs, forms)
    rows, tot = [], Counter()
    for lid, m in sorted(mult.items(), key=lambda kv: -kv[1] * leaves.get(kv[0], {}).get("ands", 0)):
        lc = leaves.get(lid, {"ands": 0, "reads": 0, "asserts": 0})
        r = {"leaf": lid, "mult": m, "ands": lc["ands"], "reads": lc["reads"], "asserts": lc["asserts"],
             "ands_x_mult": m * lc["ands"], "reads_x_mult": m * lc["reads"], "asserts_x_mult": m * lc["asserts"]}
        if lc["reads"]:
            a = leaf_alone(objs[lid])
            r.update({"alone_read_rows": a["read_rows"], "alone_layout_ands": a["layout_ands"], "tables": a["tables"],
                      "index_bits": a["index_bits"], "read_rows_x_mult": m * a["read_rows"]})
            tot["read_rows_x_mult"] += m * a["read_rows"]
        tot["ands"] += m * lc["ands"]
        tot["reads"] += m * lc["reads"]
        tot["asserts"] += m * lc["asserts"]
        rows.append(r)
    out = {"spec": spec, "bound": sp.id, "sized_ands": e.get("ands"), "sized_reads": e.get("reads"),
           "sized_asserts": e.get("asserts"), "estimate_again": {k: e2[k] for k in ("ands", "reads", "asserts")},
           "card_ands": card.get("ands"), "card_k_log": card.get("k_log"), "sum_leaves": dict(tot),
           "sum_equals_sized": tot["ands"] == e.get("ands"), "sized_equals_card": e.get("ands") == card.get("ands"),
           "forms": dict(forms), "leaves": len(rows), "unpieced": e.get("unpieced")}
    say("estimate", **out)
    for r in rows:
        say("leaf", spec=spec[:40], **r)
    return {"mult": mult, "leaves": leaves, "objs": objs, "sum": tot, "sized": e.get("ands")}


class _Tagger:
    """`ir_lower.PIECES` and `_boolean_activation` wrapped: the gate `lower_gates` is at, and each piece or builder call
    run with ``C.phase`` its estimate leaf (the outermost scope below the unit's root that `estimate` takes as a leaf, else
    the gate's primitive)."""

    def __init__(self):
        self.cur = None
        self.calls: Counter = Counter()
        self.base_pieces, self.base_ba = IL.PIECES, IL._boolean_activation
        tagger = self

        class Proxy:
            def get(self, key, default=None):
                f = tagger.base_pieces.get(key)
                return default if f is None else tagger.tagged(f, "piece")

            def wrap(self, key, build):
                return tagger.tagged(build, "build")

            def __contains__(self, k):
                return k in tagger.base_pieces

            def __getitem__(self, k):
                return tagger.base_pieces[k]

        self.proxy = Proxy()

    def ba(self, g):
        self.cur = g
        return self.base_ba(g)

    def tagged(self, fn, kind):
        def w(C, *a):
            lab = leaf_label(self.cur)[0]
            self.calls[(lab, kind)] += 1
            if C.phase == "misc":
                C.phase = lab
                try:
                    return fn(C, *a)
                finally:
                    C.phase = "misc"
            return fn(C, *a)
        return w

    def __enter__(self):
        IL.PIECES, IL._boolean_activation = self.proxy, self.ba
        return self

    def __exit__(self, *exc):
        IL.PIECES, IL._boolean_activation = self.base_pieces, self.base_ba


def leaf_label(g) -> tuple[str, int | None]:
    chain, s = [], getattr(g, "scope", None)
    if s is None:
        return g.prim.id, None
    while s is not None:
        chain.append(s)
        s = s.parent
    for sc in reversed(chain):
        if is_leaf(sc.spec):
            return sc.spec.id, sc.offset
    return g.prim.id, None


def occurrences(fn) -> Counter:
    """Each estimate leaf's occurrences in the unrolled unit (`ir_lower.standalone`'s call): a Boolean leaf's activations, a
    primitive's gates; input gates skipped."""
    _prog, call = IL.standalone(fn)
    occ, seen = Counter(), set()
    for i in range(call.lo, call.hi):
        g = call.circuit.gate(i)
        if getattr(g.prim, "family", None) == "Input":
            continue
        lab, off = leaf_label(g)
        if off is None:
            occ[lab] += 1
        elif (lab, off) not in seen:
            seen.add((lab, off))
            occ[lab] += 1
    return occ


def unit_attribution(fn, name: str, rows_spec, est: dict, label: str) -> dict:
    t = time.time()
    ports, _out_words, row_words = CS.unit_ports(fn, name, rows_spec)
    with _Tagger() as tg:
        C = CS.unit_circuit(fn, ports, CS._UnitCircuit)
    built = round(time.time() - t, 1)
    cnt = C.counts()
    in_names, out_names = [f"in{j}" for j in range(len(ports.ins))], [f"out{o}" for o in range(len(ports.outs))]
    lay = IL.layout(C, in_names, out_names)
    say("unit", spec=label, name=name, row_words=row_words, built_s=built, circuit_counts=cnt, layout_counts=lay.counts)
    live = C.live()
    created, live_c, g_of = Counter(), Counter(), {}
    for v, k in enumerate(C.kind):
        if k == "and":
            created[C.phase_of[v]] += 1
            if v in live:
                live_c[C.phase_of[v]] += 1
        elif k == "lookup":
            g_of.setdefault(C.data[v][0], v)
    bits: dict[int, set] = {}
    for v in live:
        if C.kind[v] == "lookup":
            g, j = C.data[v]
            bits.setdefault(g, set()).add(j)
    rr, nreads, detail = Counter(), Counter(), Counter()
    for g, bs in bits.items():
        ph = C.phase_of[g_of[g]]
        rows, const = read_rows(C, g, bs)
        rr[ph] += rows
        nreads[ph] += 1
        t_, idx, w = C.lookups[g]
        detail[(ph, t_, len(idx), w, len(bs), const, rows)] += 1
    for k, n in sorted(detail.items(), key=lambda kv: -kv[0][-1] * kv[1]):
        say("read", spec=label, leaf=k[0], table=k[1], index_bits=k[2], width=k[3], live_bits=k[4], const_bits=k[5],
            rows_each=k[6], reads=n, rows=k[6] * n)
    occ = occurrences(fn)
    calls = Counter()
    kinds: dict[str, set] = {}
    for (lab, kind), n in tg.calls.items():
        calls[lab] += n
        kinds.setdefault(lab, set()).add(kind)
    mult, leaves = est["mult"], est["leaves"]
    labs = sorted(set(mult) | set(occ) | set(calls) | set(created) | set(rr), key=lambda x: str(x))
    T = Counter()
    for lab in labs:
        L = leaves.get(lab, {}).get("ands", 0)
        m, o, c = mult.get(lab, 0), occ.get(lab, 0), calls.get(lab, 0)
        parts = {"estimate": m * L, "form": (o - m) * L, "folded_whole": (c - o) * L, "per_call": created[lab] - c * L,
                 "dead": live_c[lab] - created[lab], "read_rows": rr[lab]}
        staged = live_c[lab] + rr[lab]
        assert sum(parts.values()) == staged, (lab, parts, staged)
        for k, v in parts.items():
            T[k] += v
        T["staged"] += staged
        if any(parts[k] for k in parts if k != "estimate") or m != o:
            say("cause", spec=label, leaf=lab, L=L, mult=m, occ=o, calls=c, kinds=sorted(kinds.get(lab, ())),
                created=created[lab], live=live_c[lab], reads=nreads[lab], staged=staged, **parts)
    T["staged_layout"] = lay.counts["ands"]
    T["read_rows_layout"] = lay.counts["read_rows"]
    T["reads_layout"] = lay.counts["reads"]
    T["asserts_layout"] = lay.counts["asserts"]
    T["identity_holds"] = T["staged"] == lay.counts["ands"]
    say("attribution", spec=label, **T)
    out = {"totals": dict(T), "per_leaf": {lab: {"mult": mult.get(lab, 0), "occ": occ.get(lab, 0),
                                                 "staged": live_c[lab] + rr[lab],
                                                 "estimate": mult.get(lab, 0) * leaves.get(lab, {}).get("ands", 0),
                                                 "read_rows": rr[lab]} for lab in labs}}
    del C, lay, live
    gc.collect()
    return out


def run_slots(fn, name: str, rows_spec, label: str) -> None:
    t = time.time()
    low, _ow = CS.unit_lowering(fn, name, rows_spec, reads="slots")
    c = getattr(low, "counts", None)
    if not isinstance(c, dict):
        c = low.layout.counts
    stages = getattr(getattr(low, "tail", None), "stages", ())
    lays = [("unit", low.layout), *((f"stage{i}", st.layout) for i, st in enumerate(stages))]
    say("slots", spec=label, seconds=round(time.time() - t, 1), split=type(low).__name__, counts=c,
        caps=getattr(low, "caps", None), nets=[{"net": nm, **{k: int(ly.counts[k]) for k in ("ands", "rows", "read_rows")}}
                                               for nm, ly in lays])
    del low
    gc.collect()


def run_stage(spec: str, scratch: Path, out: Path, label: str) -> None:
    d = scratch / label
    t = time.time()
    p = subprocess.run([sys.executable, str(ROOT / "benchmarks" / "served_units" / "stage.py"), f"census:{spec}", "1", str(d)],
                       cwd=ROOT, capture_output=True, text=True)
    (out / f"{label}.stage.stderr").write_text(p.stderr[-20000:])
    rec = json.loads((d / "stage.json").read_text()) if (d / "stage.json").exists() else None
    if rec is not None:
        (out / f"{label}.stage.json").write_text(json.dumps(rec, indent=1))
    say("stage", spec=label, rc=p.returncode, seconds=round(time.time() - t, 1),
        record=None if rec is None else {k: rec.get(k) for k in ("n", "definition", "ands", "nets", "lookups", "statement",
                                                                  "net_and_limits", "k_log", "unit_log", "circuit_sha512",
                                                                  "maxrss_gb", "seconds")},
        stderr_tail=p.stderr[-1500:] if p.returncode else "")
    subprocess.run(["rm", "-rf", str(d)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="all")
    a = ap.parse_args()
    scratch, out = Path(a.scratch), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    say("start", source=os.environ.get("RESEARCH_SOURCE_SHA"), head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                                                   capture_output=True, text=True).stdout.strip())
    res: dict = {}

    def step(name, f, *args):
        try:
            return f(*args)
        except Exception as ex:  # noqa: BLE001 -- reported, the next step runs
            import traceback

            say("error", step=name, error=f"{type(ex).__name__}: {ex}"[:600], tb=traceback.format_exc()[-2500:])
            return None

    est_c = step("estimate combine", estimate_table, COMBINE)
    est_h = step("estimate head64", estimate_table, HEAD64)
    if est_c is not None:
        tp = SRW.template(COMBINE)
        in_rows, _out_rows = CR.unit_rows(tp)
        say("combine-template", id=tp.id, word=tp.word, layout=tp.layout, fn=tp.fn.id, statement=CR.statement(tp),
            in_rows=[[r, s, b, ps] for r, s, b, ps in in_rows])
        rows_spec = [(r, ps) for r, _s, _b, ps in in_rows]
        res["combine"] = step("unit combine", unit_attribution, tp.fn, "unit/serving-rows", rows_spec, est_c, "combine")
    small = {}
    for i, spec in enumerate(SMALL_HEADS):
        e = step(f"estimate head{i}", estimate_table, spec)
        if e is None:
            continue
        try:
            fn = SRW.template(spec).fn
        except Exception as ex:  # noqa: BLE001 -- not a served template at these statics: the bound Definition
            say("head-template-fallback", spec=spec, why=f"{type(ex).__name__}: {ex}"[:300])
            fn = SRW._bind(spec)
        rows_spec = [(nm, [nm]) for nm, _t in fn.signature()[0]]
        say("head-template", spec=spec, fn=fn.id, rows=rows_spec)
        small[spec] = step(f"unit head{i}", unit_attribution, fn, f"m1/{fn.id.split('{')[0]}", rows_spec, e, f"head{i}")
    if est_h is not None:
        for spec, r in small.items():
            if r is None:
                continue
            pred = Counter()
            for lab, m in est_h["mult"].items():
                pl = r["per_leaf"].get(lab)
                if pl and pl["occ"]:
                    d = pl["staged"] - pl["estimate"]
                    pred["excess"] += m * d / pl["occ"]
                    pred["read_rows"] += m * pl["read_rows"] / pl["occ"]
                elif est_h["leaves"].get(lab, {}).get("ands"):
                    pred["unseen_leaf_ands"] += m * est_h["leaves"][lab]["ands"]
            say("head64-extrapolated", from_spec=spec, excess=round(pred["excess"]), read_rows=round(pred["read_rows"]),
                unseen=pred["unseen_leaf_ands"], estimate=est_h["sized"], predicted_staged=round(est_h["sized"] + pred["excess"]))
    if est_c is not None:
        tp = SRW.template(COMBINE)
        rows_spec = [(r, ps) for r, _s, _b, ps in CR.unit_rows(tp)[0]]
        step("slots combine", run_slots, tp.fn, "unit/serving-rows", rows_spec, "combine")
        step("stage combine", run_stage, COMBINE, scratch, out, "combine")
    (out / "result.json").write_text(json.dumps(res, default=str))
    say("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
