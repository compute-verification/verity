"""The units a served Program spends under property/v1 (`verity.ml.circuits.stacking`), by callee and class: the honest budget
`share.py` multiplies by a served rate.

`units(fn)` runs property/v1's pass over the body of `fn` (a Program's root, or a bound Definition) and sums each node's tally by
its callee's id: instances, units by class, and the gates property/v1 refuses. On a Program it also sums them by step (the
census/v1 steps at `ends`) and checks every step against `stacking.analyze`, so the table reads property/v1's own numbers.

Property/v1 refuses an unlisted primitive. `PROVISIONAL` classes the exact-int8 profile's eight (vLLM's
`program/registry/int8_exact.py`) by the table's own rules until property/v1 lists them: FP arithmetic and casts are roundings,
`LutIndex`'s shift and clamp is one rounding, and `Exp2Lut127` reads one of 256 constant entries, a gather of 8 fresh index bits.
Each instance is charged, as property/v1 refuses each. Their units are kept under ``provisional:<class>``, so a reader sees what
rests on them. Anything else unlisted stays refused.

`fit(count, points, checks)` writes a callee whose cost grows with context, an attention window of S keys and N query rows (the
rows at keys S - N + 1 .. S), as units(N, S) = r0 N + b_q R + b_kv S + w0 + w_start [N = S] with R = N S - N (N - 1) / 2, the
rows' keys, and [N = S] a window from position 0 (its first row reads one key: in the int8 window that row's max is its score,
whose range is proven, so its subtraction is affine). It solves the coefficients from as many shapes exactly and refuses a fit
that misses any check shape (the Program's own calls among them).

    python units.py DESCRIPTOR.json[.gz] --out units.json   # a Build's Program, decoded through vLLM's registry
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from verity.ml.circuits import stacking as P
from verity.ml.circuits.census import census, family, step_ends
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import is_input

#: the exact-int8 profile's primitives property/v1 doesn't list yet: each one's class and units an instance
PROVISIONAL: dict[str, tuple[str, int]] = {
    **dict.fromkeys(("F32Rint_v1", "F32ToS8Rzi_v1", "S8ToF32_v1", "F32MulWide_v1", "F64Mul_v1", "F64ToU32RniSat_v1",
                     "LutIndex_v1"), ("roundings", 1)),
    "Exp2Lut127_v1": ("gathers", 8),
}
KEYS = P.BUDGETED + tuple(f"provisional:{c}" for c in P.BUDGETED)


def charged(t: P.Tally) -> tuple[Counter, Counter]:
    """A tally's units by `KEYS` (property/v1's, and the provisional classes'), and the gates left refused by ``code:primitive``."""
    u: Counter = Counter({c: n for c, n in t.units.items() if n})
    refused: Counter = Counter()
    for (code, fid), n in t.refused.items():
        if code == "unlisted" and fid in PROVISIONAL:
            cls, per = PROVISIONAL[fid]
            u[f"provisional:{cls}"] += n * per
        else:
            refused[f"{code}:{fid}"] += n
    return u, refused


def units(fn: Any, ends: Sequence[int] | None = None, dyn: Sequence[bool] | None = None, rng: Sequence[Any] | None = None) -> dict:
    """``callees``: each callee of `fn`'s body by id, with its family, instances, units and refused gates; ``total``. On a Program
    (anything with a ``root``: its run inputs are its dynamic values) also ``steps``, the units of each census/v1 step at `ends`
    (default: one step), checked against `stacking.analyze`. A Definition's parameters depend on a run input as `dyn` says
    (default: all) and take the value ranges `rng` (default: any word); gather index bits its parameters carry go under
    ``(parameters)``, and ``arg_rng`` holds each call's argument ranges."""
    program = fn if hasattr(fn, "root") else None
    if program is not None:
        body, pdyn, fid = program.root.body, (), program.root.id
    else:
        fn = bind(fn) if isinstance(fn, CompositeDefinition) else fn
        body, fid = fn.body, fn.id
        pdyn = tuple(dyn) if dyn is not None else (True,) * len(body.params)
    rows: dict[str, dict] = {}
    for nd in body.nodes:
        if not is_input(nd.fn):
            r = rows.setdefault(nd.fn.id, {"family": family(nd.fn), "statics": dict(getattr(nd.fn, "bindings", {})), "instances": 0,
                                           "units": Counter(), "refused": Counter()})
            r["instances"] += nd.n if nd.form == "batch" else 1
    ends = list(ends) if ends is not None else [len(body.nodes)]
    steps = [Counter() for _ in ends]
    up: dict = {}
    dyns: list[bool] = []
    segs: list = []
    prng = tuple(rng) if rng is not None else (None,) * len(pdyn)
    for k, t in P._Pass().walk(body, pdyn, prng, fid, segs, dyns, up):
        u, refused = charged(t)
        r = rows[body.nodes[k].fn.id]
        r["units"].update(u)
        r["refused"].update(refused)
        steps[min(bisect.bisect_right(ends, k), len(ends) - 1)].update(u)
    if up:
        rows["(parameters)"] = {"family": "(parameters)", "statics": {}, "instances": 0, "refused": Counter(),
                                "units": Counter(gathers=sum(bits for _refs, bits, _kind in up.values()))}
    out = {"callees": {k: {**r, "units": dict(r["units"]), "refused": dict(r["refused"])} for k, r in rows.items()},
           "total": dict(sum((Counter(r["units"]) for r in rows.values()), Counter()))}
    if program is not None:
        a = P.analyze(program, census(program, ends))
        for s, (mine, theirs) in enumerate(zip(steps, a.steps)):
            listed = {c: mine.get(c, 0) for c in P.BUDGETED}
            if listed != theirs["units"]:
                raise AssertionError(f"step {s}: the callees' units {listed} are not stacking.analyze's {theirs['units']}")
        out["steps"] = [dict(s) for s in steps]
    out["arg_dyn"] = {k: [P._dyn(a.refs, pdyn, dyns) for a in nd.args] for k, nd in enumerate(body.nodes) if not is_input(nd.fn)}
    if program is None:
        out["arg_rng"] = {k: [P._read(a.refs, prng, segs) for a in nd.args] for k, nd in enumerate(body.nodes) if not is_input(nd.fn)}
    return out


def count(fn: Any, dyn: Sequence[bool], rng: Sequence[Any] | None = None) -> dict[str, int]:
    """One call's units by `KEYS` (refused gates other than the provisional ones raise: their cost isn't known)."""
    u, refused = charged(P.count(bind(fn) if isinstance(fn, CompositeDefinition) else fn, dyn, rng))
    if refused:
        raise ValueError(f"{fn.id}: property/v1 refuses {dict(refused)}")
    return dict(u)


def rows_keys(N: int, S: int) -> int:
    return N * S - N * (N - 1) // 2


def basis(N: int, S: int) -> tuple[int, int, int, int, int]:
    return (N, rows_keys(N, S), S, 1, int(N == S))


COEFFICIENTS = ("r0", "b_q", "b_kv", "w0", "w_start")


def _solve(rows: list[tuple[int, ...]], rhs: list[Fraction]) -> list[Fraction]:
    n = len(rows)
    m = [[Fraction(x) for x in r] + [rhs[i]] for i, r in enumerate(rows)]
    for c in range(n):
        p = next((i for i in range(c, n) if m[i][c] != 0), None)
        if p is None:
            raise ValueError(f"the fit's shapes {rows} don't determine {COEFFICIENTS[c]}")
        m[c], m[p] = m[p], m[c]
        for i in range(n):
            if i != c and m[i][c] != 0:
                f = m[i][c] / m[c][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit(count_at: Callable[[int, int], Mapping[str, int]], points: Sequence[tuple[int, int]],
        checks: Iterable[tuple[int, int, Mapping[str, int] | None]] = ()) -> dict[str, dict[str, Fraction]]:
    """units(N, S) = r0 N + b_q R + b_kv S + w0 + w_start [N = S] for each key, solved from the five `points` (N, S) that `count_at`
    counts, then held to every check (N, S, units), counted by `count_at` when units is None. Raises when a check misses."""
    if len(points) != len(COEFFICIENTS):
        raise ValueError(f"the fit takes {len(COEFFICIENTS)} shapes, not {len(points)}")
    seen = [dict(count_at(N, S)) for N, S in points]
    keys = sorted({k for s in seen for k in s})
    co = {k: dict(zip(COEFFICIENTS, _solve([basis(N, S) for N, S in points], [Fraction(s.get(k, 0)) for s in seen]))) for k in keys}
    for N, S, want in checks:
        got = dict(want) if want is not None else dict(count_at(N, S))
        model = {k: v for k in sorted(set(keys) | set(got)) if (v := evaluate(co, N, S).get(k, Fraction(0))) or k in got}
        if any(model.get(k, 0) != got.get(k, 0) for k in set(model) | set(got)):
            raise AssertionError(f"the fit misses (N={N}, S={S}): it gives {_ints(model)}, the count {got}")
    return co


def evaluate(co: Mapping[str, Mapping[str, Fraction]], N: int, S: int) -> dict[str, Fraction]:
    b = basis(N, S)
    return {k: sum((c[name] * x for name, x in zip(COEFFICIENTS, b)), Fraction(0)) for k, c in co.items()}


def _ints(d: Mapping[str, Fraction]) -> dict:
    return {k: int(v) if Fraction(v).denominator == 1 else str(v) for k, v in d.items()}


def template(fn: Any) -> Any:
    """The registry's composite a decoded callee was bound from (its Python body builder), to bind at other statics."""
    t = REGISTRY.defs.get(fn.base_id)
    if not isinstance(t, CompositeDefinition):
        raise ValueError(f"{fn.base_id} is not a registered composite (is its registry module loaded?)")
    return t


def rebound(fn: Any, **statics: Any) -> Any:
    """A decoded callee's Definition with `statics` replacing its bindings."""
    return bind(template(fn), **{**fn.bindings, **statics})


def components(rows: Mapping[str, dict], *, token: str, sample: Iterable[str], attention: str) -> dict:
    """A Program's callees as the per-step model's parts: ``tokens`` (instances of the `token` family, one a computed token row),
    ``samples`` (of the last `sample` family), ``token`` and ``sample`` units each (every callee but the attention's is a token's,
    except the `sample` families'), and the attention calls by (N, S) with their units a call."""
    sample = tuple(sample)
    fams = Counter()
    for r in rows.values():
        fams[r["family"]] += r["instances"]
    tokens, samples = fams[token], fams[sample[-1]]
    tok: Counter = Counter()
    smp: Counter = Counter()
    calls = []
    for cid, r in rows.items():
        if r["family"] == attention:
            calls.append((cid, r))
        elif r["family"] in sample:
            smp.update(r["units"])
        else:
            tok.update(r["units"])
    return {"tokens": tokens, "samples": samples, "token": {k: Fraction(v, tokens) for k, v in tok.items()},
            "sample": {k: Fraction(v, samples) for k, v in smp.items()}, "calls": calls}


def load(path: Path) -> Any:
    """A Build's descriptor (``.json`` or ``.json.gz``) decoded through vLLM's registry."""
    from verity.ml.circuits.codec import decode_program
    from verity_vllm.program.registry.catalog import load as registry

    reg, errors = registry()
    for e in errors:
        print(f"[units] registry module unavailable ({e})", file=sys.stderr)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return decode_program(json.loads(raw), reg)


def jsonable(o: Any) -> Any:
    if isinstance(o, Fraction):
        return int(o) if o.denominator == 1 else str(o)
    if isinstance(o, Mapping):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


#: the fit's shapes, and the extra shapes it is held to beside the Program's own calls
FIT_POINTS = ((1, 64), (1, 128), (2, 128), (4, 256), (4, 4))
FIT_CHECKS = ((1, 2048), (3, 77), (8, 512), (1, 1), (6, 6), (33, 33))


def attention_fit(program: Any, rows: Mapping[str, dict], fam: str, arg_dyn: Mapping[int, list[bool]],
                  **statics: Any) -> tuple[dict, dict]:
    """The attention family's fit at the Program's own statics (or at `statics`, e.g. a TP rank's heads), held to the Program's
    calls when `statics` is empty. Returns the coefficients and the statics fitted."""
    nodes = program.root.body.nodes
    k0 = next(k for k, nd in enumerate(nodes) if not is_input(nd.fn) and family(nd.fn) == fam)
    fn, dyn = nodes[k0].fn, arg_dyn[k0]
    checks = [(N, S, None) for N, S in FIT_CHECKS]
    if not statics:
        for nd in nodes:
            if not is_input(nd.fn) and family(nd.fn) == fam:
                r = rows[nd.fn.id]
                want = {k: v // r["instances"] for k, v in r["units"].items()}
                checks.append((nd.fn.bindings["N"], nd.fn.bindings["S"], want))
        checks = list(dict.fromkeys((N, S, json.dumps(w, sort_keys=True) if w else None) for N, S, w in checks))
        checks = [(N, S, json.loads(w) if w else None) for N, S, w in checks]
    co = fit(lambda N, S: count(rebound(fn, N=N, S=S, **statics), dyn), FIT_POINTS, checks)
    return co, {k: v for k, v in {**fn.bindings, **statics}.items() if k not in ("N", "S")}


def bind_body(fn: Any) -> Any:
    return (bind(fn) if isinstance(fn, CompositeDefinition) else fn).body


def by_family(fn: Any, dyn: Sequence[bool], expand: Sequence[str] = (), every: bool = False,
              rng: Sequence[Any] | None = None) -> dict[str, dict[str, int]]:
    """One call of `fn`'s units by the family of each callee of its body, a callee of an `expand` family (`every`: any composite)
    by its own callees' at the ranges its call gives it."""
    if isinstance(fn, PrimitiveDefinition):
        return {family(fn): count(fn, dyn, rng)}
    u = units(fn, dyn=dyn, rng=rng)
    first: dict[str, int] = {}
    for k, nd in enumerate(bind_body(fn).nodes):
        if not is_input(nd.fn):
            first.setdefault(nd.fn.id, k)
    out: dict[str, Counter] = {}
    body = bind_body(fn)
    for cid, r in u["callees"].items():
        k = first.get(cid)
        if k is not None and (r["family"] in expand or (every and not isinstance(body.nodes[k].fn, PrimitiveDefinition))):
            for f, v in by_family(body.nodes[k].fn, u["arg_dyn"][k], expand, every, u["arg_rng"][k]).items():
                out.setdefault(f, Counter()).update({c: n * r["instances"] for c, n in v.items()})
        else:
            out.setdefault(r["family"], Counter()).update(r["units"])
    return {f: dict(v) for f, v in out.items() if v}


def by_primitive(fn: Any, dyn: Sequence[bool]) -> dict[str, int]:
    """One call of `fn`'s units by ``primitive|class``, every composite expanded; refused when they don't sum to `count`'s (a
    callee whose units depend on the ranges its caller gives it)."""
    out = {f"{p}|{c}": n for p, u in by_family(fn, dyn, every=True).items() for c, n in u.items()}
    whole, parts = count(fn, dyn), Counter()
    for k, n in out.items():
        parts[k.split("|")[1]] += n
    if dict(parts) != whole:
        raise AssertionError(f"{fn.id}: its primitives' units {dict(parts)} are not its own {whole}")
    return out


def per_key(fn: Any, dyn: Sequence[bool], S0: int = 128) -> dict[str, dict[str, Fraction]]:
    """A one-row attention window's units a key, by the family of each callee of its body: the slope between S0 and 2 S0 keys."""
    a, b = by_family(rebound(fn, N=1, S=S0), dyn), by_family(rebound(fn, N=1, S=2 * S0), dyn)
    out = {}
    for f in sorted(set(a) | set(b)):
        d = {k: Fraction(b.get(f, {}).get(k, 0) - a.get(f, {}).get(k, 0), S0) for k in set(a.get(f, {})) | set(b.get(f, {}))}
        if any(d.values()):
            out[f] = {k: v for k, v in d.items() if v}
    return out


def first_calls(program: Any, arg_dyn: Mapping[int, list[bool]]) -> dict[str, tuple[Any, list[bool]]]:
    """Each callee's first call in the Program's root, by id and by family: its Definition and which of its arguments depend on
    a run input."""
    out: dict[str, tuple[Any, list[bool]]] = {}
    for k, nd in enumerate(program.root.body.nodes):
        if not is_input(nd.fn):
            out.setdefault(nd.fn.id, (nd.fn, arg_dyn[k]))
            out.setdefault(family(nd.fn), (nd.fn, arg_dyn[k]))
    return out


#: a family whose callees pair one to one with another family's at the same K: the int8 linear's row quantizer, read by its product
PAIRED = {"Int8RowQuant_v1": "Int8ScaledMm_v1"}


def modeled(program: Any, rows: Mapping[str, dict], spec: Mapping[str, Any], arg_dyn: Mapping[int, list[bool]], *,
            token: str, attention: str) -> dict[str, dict]:
    """The Program's callees as another model's (`spec`): each callee `spec["statics"]` names (by id, else by family) bound at
    those statics and counted; a layer's callee (a multiple of ``layers[0]`` instances a token row) repeated ``layers[1]`` times a
    token row instead; every callee on each of ``ranks`` ranks (``instances`` stays one rank's, ``units`` is every rank's) but a
    family ``once`` names, computed once; and a `PAIRED` family's callees again from its partner's, one at the partner's new K
    an instance (each partner a K of its own: the 70B's o and qkv linears share none of the 8B's K = 4096). The attention's calls
    are left out (`attention_fit` at ``spec["attention"]`` prices them)."""
    fns = first_calls(program, arg_dyn)
    tokens = sum(r["instances"] for r in rows.values() if r["family"] == token)
    (L0, L1), ranks = spec["layers"], spec["ranks"]
    once = set(spec.get("once", ()))
    for fam, by in PAIRED.items():
        mine, theirs = Counter(), Counter()
        for r in rows.values():
            if r["family"] in (fam, by):
                (mine if r["family"] == fam else theirs)[r["statics"]["K"]] += r["instances"]
        if mine != theirs:
            raise ValueError(f"{fam}'s instances by K {dict(mine)} don't pair with {by}'s {dict(theirs)}")
    out: dict[str, dict] = {}
    for cid, r in rows.items():
        if r["family"] == attention or r["family"] in PAIRED:
            continue
        fn, dyn = fns[cid]
        st = spec["statics"].get(cid) or spec["statics"].get(r["family"])
        n = r["instances"]
        per = count(rebound(fn, **st), dyn) if st else {k: Fraction(v, n) for k, v in r["units"].items()}
        m = n * L1 // L0 if n % (tokens * L0) == 0 else n
        rk = 1 if r["family"] in once else ranks
        out[cid] = {"family": r["family"], "statics": {**r["statics"], **(st or {})}, "instances": m, "ranks": rk,
                    "refused": r["refused"], "units": {k: v * m * rk for k, v in per.items()}}
    for fam, by in PAIRED.items():
        if fam not in fns:
            continue
        fn, dyn = fns[fam]
        for r in [r for r in out.values() if r["family"] == by]:
            K = r["statics"]["K"]
            o = out.setdefault(f"{fam}{{K={K}}}", {"family": fam, "statics": {"K": K}, "instances": 0, "ranks": r["ranks"],
                                                   "refused": {}, "units": Counter()})
            o["instances"] += r["instances"]
            o["units"] = Counter(o["units"]) + Counter({k: v * r["instances"] * r["ranks"] for k, v in count(rebound(fn, K=K), dyn).items()})
    return {k: {**r, "units": dict(r["units"])} for k, r in out.items()}


#: the families an int8 linear's activation quantizer is read from, and the inner family `levers` breaks one into
LINEAR = ("Int8Linear_v1", "Int8RowQuant_v1")
LINEAR_INNER = ("Int8RowQuant_v1",)


def levers(program: Any, rows: Mapping[str, dict], arg_dyn: Mapping[int, list[bool]], *, token: str, sample: Sequence[str],
           attention: str, spec: Mapping[str, Any] | None = None) -> dict:
    """What the levers act on, from the Program's own Definitions at `rows`' statics (a modeled variant's rows, with `spec`'s
    attention statics): the attention window's units a key by inner family (a K' quantized once at the KV write removes
    ``Int8StaticQuant``'s; a V' stored once, with a static scale, removes ``Int8ColumnQuant``'s), each int8 linear's activation
    quantizer's units a call by inner family (a static activation scale removes ``Int8RowScale``'s), the LM head as an int8
    linear (``Int8Linear_v1`` at the BF16 ``Gemm_v2``'s K and N), and ``primitives``: the units a token row and a sampled row
    spend by ``primitive|class``, and the attention window's coefficients by the same keys (`fit`, exact)."""
    fns = first_calls(program, arg_dyn)
    fn, dyn = fns[attention]
    att = (spec or {}).get("attention") or {}
    win = rebound(fn, **att) if att else fn
    out: dict[str, Any] = {"attention_per_key": per_key(win, dyn), "linear": {}}
    tokens = sum(r["instances"] for r in rows.values() if r["family"] == token)
    samples = sum(r["instances"] for r in rows.values() if r["family"] == sample[-1])
    prims: dict[str, Counter] = {"token": Counter(), "sample": Counter()}
    for cid, r in rows.items():
        if r["family"] in LINEAR:
            f, d = fns[r["family"]]
            out["linear"][cid] = {"statics": r["statics"], "by_family": by_family(rebound(f, **r["statics"]), d, LINEAR_INNER)}
        elif r["family"] == "Gemm_v2":
            K, N = r["statics"]["K"], r["statics"]["N"]
            out["lm_head_int8"] = {"K": K, "N": N, "units": count(bind(REGISTRY.defs["Int8Linear_v1"], K=K, N=N), [True, False, False])}
        if r["family"] in (attention, "(parameters)"):
            continue
        f, d = fns.get(cid) or fns[r["family"]]
        g = f if r["statics"] == dict(getattr(f, "bindings", {})) else rebound(f, **r["statics"])
        part, n = ("sample", samples) if r["family"] in sample else ("token", tokens)
        prims[part].update({p: Fraction(v * r["instances"] * r.get("ranks", 1), n) for p, v in by_primitive(g, d).items()})
    out["primitives"] = {k: dict(v) for k, v in prims.items()}
    out["primitives"]["attention"] = fit(lambda N, S: by_primitive(rebound(win, N=N, S=S), dyn), FIT_POINTS,
                                         [(N, S, None) for N, S in FIT_CHECKS])
    return out


def describe(program: Any, rows: Mapping[str, dict], *, token: str, sample: Sequence[str], attention: str,
             att: tuple[dict, dict], layers: int, extra: Mapping[str, Any] | None = None) -> dict:
    parts = components(rows, token=token, sample=sample, attention=attention)
    tok = Counter(parts["token"])
    for k, v in ((extra or {}).get("token") or {}).items():
        tok[k] += Fraction(v)
    return {"schema": "verity-stacked-share/units/v1", "classes": P.digest(), "provisional": PROVISIONAL,
            "program": {"function": program.root.id, "root_nodes": len(program.root.body.nodes)},
            "tokens": parts["tokens"], "samples": parts["samples"], "token": dict(tok), "sample": parts["sample"],
            "attention": {"family": attention, "statics": att[1], "coefficients": att[0], "calls": layers},
            "extra": dict(extra or {}),
            "callees": {k: dict(r) for k, r in rows.items()},
            "total": dict(sum((Counter(r["units"]) for r in rows.values()), Counter()))}


def check_primitives(doc: Mapping[str, Any]) -> None:
    """The primitives' units sum to the document's by class: a token row's (but the spec's extra), a sampled row's, and each
    attention coefficient."""
    def classes(d: Mapping[str, Any]) -> dict:
        out: Counter = Counter()
        for k, v in d.items():
            out[k.split("|")[1]] += v
        return {k: v for k, v in out.items() if v}
    pr = doc["levers"]["primitives"]
    tok = Counter(doc["token"])
    tok.subtract({k: Fraction(v) for k, v in ((doc.get("extra") or {}).get("token") or {}).items()})
    att = {c: {k: v for k, v in co.items() if v} for c, co in doc["attention"]["coefficients"].items()}
    mine = {}
    for k, co in pr["attention"].items():
        for name, v in co.items():
            if v:
                mine.setdefault(k.split("|")[1], Counter())[name] += v
    for what, a, b in (("token", classes(pr["token"]), {k: v for k, v in tok.items() if v}),
                       ("sample", classes(pr["sample"]), {k: v for k, v in doc["sample"].items() if v}),
                       ("attention", {c: dict(v) for c, v in mine.items()}, att)):
        if a != b:
            raise AssertionError(f"the primitives' {what} units {_ints(a) if what != 'attention' else a} are not the document's {b}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("descriptor", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--token", default="Embedding_v1", help="the family with one instance a computed token row")
    ap.add_argument("--sample", default="Gemm_v2,TokenSelect_v1", help="the families computed once a sampled row (last: one a sample)")
    ap.add_argument("--attention", default="Int8LutAttentionWindow_v1")
    ap.add_argument("--as", dest="spec", type=Path, help="a modeled variant's spec (JSON: label, layers, ranks, once, statics, "
                    "attention, extra), written to --as-out")
    ap.add_argument("--as-out", type=Path)
    a = ap.parse_args(argv)
    program = load(a.descriptor)
    sample = a.sample.split(",")
    ends = step_ends(program, [sample[-1]])
    u = units(program, ends)
    dyn = {int(k): v for k, v in u["arg_dyn"].items()}
    att = attention_fit(program, u["callees"], a.attention, dyn)
    calls = {r["instances"] for r in u["callees"].values() if r["family"] == a.attention}
    if len(calls) != 1:
        raise ValueError(f"{a.attention}'s callees have {sorted(calls)} instances each, not one count a step (a layer each)")
    doc = describe(program, u["callees"], token=a.token, sample=sample, attention=a.attention, att=att, layers=calls.pop())
    doc["program"]["descriptor"] = str(a.descriptor)
    doc["steps"] = u["steps"]
    doc["levers"] = levers(program, u["callees"], dyn, token=a.token, sample=sample, attention=a.attention)
    check_primitives(doc)
    a.out.write_text(json.dumps(jsonable(doc), indent=1, default=str))
    print(json.dumps(jsonable({k: doc[k] for k in ("tokens", "samples", "token", "sample", "attention", "total")}), indent=1,
                     default=str))
    if a.spec:
        spec = json.loads(a.spec.read_text())
        rows = modeled(program, u["callees"], spec, dyn, token=a.token, attention=a.attention)
        att2 = attention_fit(program, u["callees"], a.attention, dyn, **spec["attention"])
        m = describe(program, rows, token=a.token, sample=sample, attention=a.attention, att=att2,
                     layers=spec["layers"][1] * spec["ranks"], extra=spec.get("extra"))
        m["program"]["modeled"] = {k: spec[k] for k in ("label", "layers", "ranks", "once", "statics", "attention") if k in spec}
        m["levers"] = levers(program, rows, dyn, token=a.token, sample=sample, attention=a.attention, spec=spec)
        check_primitives(m)
        a.as_out.write_text(json.dumps(jsonable(m), indent=1, default=str))
        print(json.dumps(jsonable({k: m[k] for k in ("tokens", "samples", "token", "sample", "attention", "total")}), indent=1,
                         default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
