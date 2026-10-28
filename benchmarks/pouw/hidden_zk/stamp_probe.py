"""Scratch probe (removed before review): how the flat H1 unit's ANDs fall into Boolean activations and pieces, whether
one input-form pattern always gives one AND block, and how liveness varies. ``python stamp_probe.py K R C OUT.json``."""
import collections
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

k, r, c = (int(x) for x in sys.argv[1:4])
out_path = Path(sys.argv[4])
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "h1_prove"))

from verity_flock import ir_lower as IL
from verity_flock import class_statement as CS
from verity.core.protocols.pouw.circuit import h1window as HW

acts = []  # (spec id, n0, n1, pattern key, inputs' vars (global, first-seen order), outs)
orig = IL._lower_activation


def lower_activation(C, s, build, wires, pieces):
    n0 = len(C.kind)
    rec = {}

    def b2(C_, *els):
        rec["els"] = els
        o = build(C_, *els)
        rec["out"] = o
        return o

    ok = orig(C, s, b2, wires, pieces)
    if ok:
        acts.append(("A", s.spec.id, n0, len(C.kind), rec["els"], rec["out"], s.spec.signature()[1]))
    return ok


IL._lower_activation = lower_activation
piece_calls = []
orig_get = IL._Pieces.get


def get(self, key, default=None):
    fn = orig_get(self, key, default)
    if fn is None or not callable(fn):
        return fn

    def p2(C_, *ops):
        n0 = len(C_.kind)
        o = fn(C_, *ops)
        piece_calls.append(("P", key, n0, len(C_.kind), ops, o, None))
        return o
    return p2


IL._Pieces.get = get
got = {}
orig_layout = IL.layout


def layout(C, *a, **kw):
    got["C"] = C
    t = time.time()
    lay = orig_layout(C, *a, **kw)
    got["layout_s"] = time.time() - t
    return lay


IL.layout = layout
import h1unit as U

t0 = time.time()
shape = U.Shape(k, r, c)
template = shape.definition()
rows = HW.statement_rows(template, U.ports(template))
low, _ = CS.unit_lowering(template, "probe", [(p, ps) for p, _s, _b, ps in rows])
t1 = time.time()
C = got["C"]
live = C.live()
t2 = time.time()
kind = C.kind
n_and = kind.count("and")
res = {"shape": [k, r, c], "lower_s": t1 - t0, "layout_s": got["layout_s"], "live_s": t2 - t1,
       "maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, "kinds": dict(collections.Counter(kind)),
       "ands": n_and, "live_ands": sum(1 for v in live if kind[v] == "and"), "asserts": len(C.asserts),
       "useful": low.layout.useful, "nnz_a": low.layout.counts["nnz_a"], "nnz_b": low.layout.counts["nnz_b"]}
calls = sorted(acts + piece_calls, key=lambda x: x[2])
covered = sum(x[3] - x[2] for x in calls)
res["vars_in_calls"] = covered
res["vars_total"] = len(kind)
res["n_inputs"] = sum(len(v) for v in C.inputs.values())
prev, overlap = 0, 0
for x in calls:
    if x[2] < prev:
        overlap += 1
    prev = max(prev, x[3])
res["overlapping_calls"] = overlap


def flat_bits(els):
    out = []
    for e in els:
        if isinstance(e, (list, tuple)):
            out.extend(e)
        else:
            out.append(e)
    return out


def pattern(c_):
    _tag, sid, n0, n1, els, outs, _ret = c_
    bits = flat_bits(els)
    ren, order = {}, []
    key = []
    for w in bits:
        vs = sorted(w.s)
        for v in vs:
            if v not in ren:
                ren[v] = len(ren)
                order.append(v)
        key.append((tuple(ren[v] for v in vs), w.c))
    P = len(ren)
    foreign = 0

    def name(v):
        nonlocal foreign
        if v in ren:
            return ren[v]
        if n0 <= v < n1:
            return P + (v - n0)
        foreign += 1
        return -1 - v

    body = []
    for v in range(n0, n1):
        if kind[v] != "and":
            body.append((kind[v],))
            continue
        a, b = C.data[v]
        body.append((tuple(sorted(name(x) for x in a.s)), a.c, tuple(sorted(name(x) for x in b.s)), b.c))
    o = flat_bits(outs) if isinstance(outs, (list, tuple)) else [outs]
    oo = []
    for w in o:
        if hasattr(w, "s"):
            oo.append((tuple(sorted(name(x) for x in w.s)), w.c))
        else:
            oo.append(("?", repr(type(w))))
    th = hashlib.sha256(repr((body, oo)).encode()).hexdigest()[:16]
    lv = hashlib.sha256(bytes(1 if v in live else 0 for v in range(n0, n1))).hexdigest()[:16]
    nlive = sum(1 for v in range(n0, n1) if v in live)
    kh = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    mono = all(a < b for a, b in zip(order, order[1:]))
    return kh, th, lv, nlive, foreign, P, mono, len(bits)


per = collections.defaultdict(lambda: {"n": 0, "ands": 0, "live": 0, "keys": collections.defaultdict(set),
                                       "lives": collections.defaultdict(set), "foreign": 0, "P": set(), "mono": 0,
                                       "bits": set()})
seq = []
for x in calls:
    kh, th, lv, nlive, foreign, P, mono, nb = pattern(x)
    e = per[(x[0], x[1])]
    e["n"] += 1
    e["ands"] += x[3] - x[2]
    e["live"] += nlive
    e["keys"][kh].add(th)
    e["lives"][kh].add(lv)
    e["foreign"] += foreign
    e["P"].add(P)
    e["mono"] += mono
    e["bits"].add(nb)
    if seq and seq[-1][0] == x[1]:
        seq[-1][1] += 1
    else:
        seq.append([x[1], 1])
res["specs"] = {f"{t}:{sid}": {"n": e["n"], "ands": e["ands"], "live": e["live"], "keys": len(e["keys"]),
                               "max_templates_per_key": max(len(v) for v in e["keys"].values()),
                               "max_lives_per_key": max(len(v) for v in e["lives"].values()), "foreign": e["foreign"],
                               "P": sorted(e["P"])[:8], "monotone": e["mono"], "bits": sorted(e["bits"])[:8]}
                for (t, sid), e in sorted(per.items(), key=lambda kv: -kv[1]["ands"])}
res["runs"] = len(seq)
res["run_seq_head"] = seq[:200]
res["run_seq_tail"] = seq[-60:]
out_path.write_text(json.dumps(res, indent=1))
print(json.dumps({k_: v for k_, v in res.items() if k_ not in ("specs", "run_seq_head", "run_seq_tail")}))
for sid, e in list(res["specs"].items())[:40]:
    print(f"{sid:58s} n={e['n']:7d} ands={e['ands']:10d} live={e['live']:10d} keys={e['keys']:5d} "
          f"tpk={e['max_templates_per_key']} lpk={e['max_lives_per_key']} foreign={e['foreign']} P={e['P'][:3]} mono={e['monotone']}")
