"""The units a served Program spends under property/v1 (`verity.ml.circuits.stacking`), by callee and class: the honest budget
`share.py` multiplies by a served rate.

`units(fn)` runs property/v1's pass over the body of `fn` (a Program's root, or a bound Definition) and sums each node's tally by
its callee's id: instances, units by class, and the gates property/v1 refuses. On a Program it also sums them by step (the
census/v1 steps at `ends`) and checks every step against `stacking.analyze`, so the table reads property/v1's own numbers.

The attention of one request-step-layer (`AttentionWindow`) is the exact-int8 profile's two calls: `Int8KvQuant_v1{S, D}` once a KV head
(G of them) over the window's S keys, then `Int8LutAttentionRow_v1{D, s, SCALE}` once a (row, head), H heads a row, for its N last
rows s = S - N + 1 .. S. `fit(count, points, checks)` writes it as units(N, S) = r0 N + b_q R + b_kv S + w0 + w_start [N = S]
with R = N S - N (N - 1) / 2, the rows' keys, and [N = S] a window from position 0 (its first row reads one key: that row's max
is its score, whose range is proven, so its subtraction is affine). It solves the coefficients from as many shapes exactly and
refuses a fit that misses any check shape (the Program's own windows among them).

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

def charged(t: P.Tally) -> tuple[Counter, Counter]:
    """A tally's units by class, and its refused gates by ``code:primitive``."""
    u: Counter = Counter({c: n for c, n in t.units.items() if n})
    refused: Counter = Counter({f"{code}:{fid}": n for (code, fid), n in t.refused.items()})
    return u, refused


def units(fn: Any, ends: Sequence[int] | None = None, dyn: Sequence[bool] | None = None, rng: Sequence[Any] | None = None) -> dict:
    """``callees``: each callee of `fn`'s body by id, with its family, instances, units and refused gates; ``total``. On a Program
    (anything with a ``root``: its run inputs are its dynamic values) also ``steps``, the units of each census/v1 step at `ends`
    (default: one step), checked against `stacking.analyze`, and ``reasons``, property/v1's refusals but the budgets' (`judge`'s
    ``operand-bits``, ``s8-window``, ``uncovered``). A Definition's parameters depend on a run input as `dyn` says
    (default: all) and take the value ranges `rng` (default: any word); gather index bits its parameters carry go under
    ``(parameters)``, and ``arg_rng`` holds each call's argument ranges (a scan's: its first iteration's, its carry dependent on
    a run input when any argument is, as the walk has it)."""
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
            r["instances"] += nd.n if nd.form in ("batch", "scan") else 1
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
        v = P.judge(a, c=dict.fromkeys(P.BUDGETED, 0), windows=[P.Window(range(len(a.steps)), 1, 1)])
        out["reasons"] = [r for r in v.reasons if r["code"] not in P.BUDGETED]
    out["arg_dyn"] = {}
    for k, nd in enumerate(body.nodes):
        if not is_input(nd.fn):
            ad = [P._dyn(a.refs, pdyn, dyns) for a in nd.args]
            out["arg_dyn"][k] = [any(ad)] + ad[1:] if nd.form == "scan" else ad
    if program is None:
        out["arg_rng"] = {k: [P._read(a.refs, prng, segs) for a in nd.args] for k, nd in enumerate(body.nodes) if not is_input(nd.fn)}
    return out


def count(fn: Any, dyn: Sequence[bool], rng: Sequence[Any] | None = None) -> dict[str, int]:
    """One call's units by class (a refused gate raises: its cost isn't known)."""
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


def components(rows: Mapping[str, dict], *, token: str, sample: Iterable[str], attention: Sequence[str]) -> dict:
    """A Program's callees as the per-step model's parts: ``tokens`` (instances of the `token` family, one a computed token row),
    ``samples`` (of the last `sample` family), and ``token`` and ``sample`` units each (every callee but the `attention`
    families' is a token's, except the `sample` families')."""
    sample = tuple(sample)
    fams = Counter()
    for r in rows.values():
        fams[r["family"]] += r["instances"]
    tokens, samples = fams[token], fams[sample[-1]]
    tok: Counter = Counter()
    smp: Counter = Counter()
    for r in rows.values():
        if r["family"] in attention:
            continue
        (smp if r["family"] in sample else tok).update(r["units"])
    return {"tokens": tokens, "samples": samples, "token": {k: Fraction(v, tokens) for k, v in tok.items()},
            "sample": {k: Fraction(v, samples) for k, v in smp.items()}}


def load(path: Path) -> Any:
    """A Build's descriptor (``.json`` or ``.json.gz``) decoded through vLLM's registry."""
    from verity.ml.circuits.codec import decode_program
    from verity.ml.definitions.registry.catalog import load as registry

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


#: the attention's families: one call a KV head over a request-step-layer's window of S keys, and one a (row, head)
KV, ROW = "Int8KvQuant_v1", "Int8LutAttentionRow_v1"
ATTENTION = (KV, ROW)


class AttentionWindow:
    """One request-step-layer's attention over its window of S keys and its N last rows: G calls of `kv` at S, and H calls of
    `row` at each row's keys s = S - N + 1 .. S, each with its first call's run dependence. ``units``, ``primitives`` and
    ``families`` give its units by class, by ``primitive|class`` (every composite expanded) and by the family of each callee of
    the two Definitions' bodies; each call's are counted once a shape."""

    def __init__(self, kv: tuple[Any, list[bool]], row: tuple[Any, list[bool]], G: int, H: int) -> None:
        (self.kv, self.kv_dyn), (self.row, self.row_dyn), self.G, self.H = kv, row, G, H
        self.memo: dict[tuple, dict] = {}

    def _at(self, how: str, which: str, s: int) -> dict:
        key = (how, which, s)
        if key not in self.memo:
            fn, dyn = (self.kv, self.kv_dyn) if which == "kv" else (self.row, self.row_dyn)
            g = rebound(fn, S=s)
            self.memo[key] = (count(g, dyn) if how == "units" else by_primitive(g, dyn) if how == "primitives"
                              else by_family(g, dyn))
        return self.memo[key]

    def _sum(self, how: str, N: int, S: int) -> dict:
        out: Counter = Counter({k: self.G * v for k, v in self._at(how, "kv", S).items()})
        for s in range(S - N + 1, S + 1):
            out.update({k: self.H * v for k, v in self._at(how, "row", s).items()})
        return {k: v for k, v in out.items() if v}

    def units(self, N: int, S: int) -> dict:
        return self._sum("units", N, S)

    def primitives(self, N: int, S: int) -> dict:
        return self._sum("primitives", N, S)

    def families(self, S: int) -> dict:
        """A one-row window's units by family, the `families` of each call merged."""
        out: dict[str, Counter] = {}
        for which, m in (("kv", self.G), ("row", self.H)):
            for f, v in self._at("families", which, S).items():
                out.setdefault(f, Counter()).update({k: m * n for k, n in v.items()})
        return {f: dict(v) for f, v in out.items()}

    def statics(self) -> dict:
        return {"G": self.G, "H": self.H, KV: {k: v for k, v in self.kv.bindings.items() if k != "S"},
                ROW: {k: v for k, v in self.row.bindings.items() if k != "S"}}


def window(program: Any, rows: Mapping[str, dict], arg_dyn: Mapping[int, list[bool]], G: int, H: int,
           spec: Mapping[str, Any] | None = None) -> tuple[AttentionWindow, int]:
    """The Program's attention as an `AttentionWindow` of G KV heads and H heads (or `spec`'s: one TP rank's), and the calls a
    request-step makes (its layers, ``L``). Held to the Program: every `KV` callee is G L instances and every `ROW` callee H L
    (a window's S keys once a step, each row's keys once), and each callee's units an instance are the Definition's own at its
    statics (so the window's counts hold at the Program's ranges)."""
    fns = first_calls(program, arg_dyn)
    w = AttentionWindow(fns[KV], fns[ROW], G, H)
    Ls = set()
    for cid, r in rows.items():
        if r["family"] not in ATTENTION:
            continue
        which, m = ("kv", G) if r["family"] == KV else ("row", H)
        if r["instances"] % m:
            raise ValueError(f"{cid}: {r['instances']} instances, not a multiple of {m}")
        Ls.add(r["instances"] // m)
        per = {k: Fraction(v, r["instances"]) for k, v in r["units"].items() if v}
        if per != w._at("units", which, r["statics"]["S"]):
            raise AssertionError(f"{cid}: its units an instance {_ints(per)} are not its Definition's "
                                 f"{w._at('units', which, r['statics']['S'])}")
    if len(Ls) != 1:
        raise ValueError(f"the attention callees give {sorted(Ls)} calls a request-step (G={G}, H={H}), not one")
    if spec:
        w = AttentionWindow(fns[KV], fns[ROW], spec["G"], spec["H"])
    return w, Ls.pop()


def windows_of(rows: Mapping[str, dict]) -> list[tuple[int, int]]:
    """One request's windows (N, S) in its Program: each `KV` callee's S, its rows the keys its step adds (a prefill from 0 all
    S of them, a decode its last), held to the `ROW` callees' keys (each row's once)."""
    kv = sorted(r["statics"]["S"] for r in rows.values() if r["family"] == KV)
    out, prev = [], 0
    for S in kv:
        out.append((S - prev, S))
        prev = S
    keys = sorted(r["statics"]["S"] for r in rows.values() if r["family"] == ROW)
    if keys != list(range(1, prev + 1)):
        raise ValueError(f"the rows' keys {keys[:3]}..{keys[-3:]} are not 1..{prev}: not one request's windows")
    return out


def attention_fit(w: AttentionWindow, own: Iterable[tuple[int, int]] = ()) -> dict:
    """The window's fit (`fit`), held to `FIT_CHECKS` and the Program's own windows `own`."""
    checks = [(N, S, None) for N, S in dict.fromkeys(tuple(FIT_CHECKS) + tuple(own))]
    return fit(w.units, FIT_POINTS, checks)


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


def per_key(w: AttentionWindow, S0: int = 128) -> dict[str, dict[str, Fraction]]:
    """A one-row attention window's units a key, by the family of each callee of its Definitions' bodies: the slope between S0
    and 2 S0 keys."""
    a, b = w.families(S0), w.families(2 * S0)
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
PAIRED = {"Int8RowQuant_v1": "ScaledMmS8_v1"}


def spec_statics(statics: Mapping[str, Mapping[str, Any]], cid: str, r: Mapping[str, Any], used: set[str]) -> dict | None:
    """The statics `statics` (a spec's) binds the callee `cid` (row `r`) at: its id's entry, else the one ``Family{k=v,..}`` key
    whose family and statics it has, else its family's; the key read goes in `used`."""
    hits = [cid] if cid in statics else []
    if not hits:
        for key in statics:
            fam, _, rest = key.partition("{")
            if rest and fam == r["family"] and all(str(r["statics"].get(k)) == v
                                                   for k, v in (kv.split("=", 1) for kv in rest.rstrip("}").split(","))):
                hits.append(key)
    if not hits and r["family"] in statics:
        hits = [r["family"]]
    if len(hits) > 1:
        raise ValueError(f"{cid}: the spec's keys {hits} all name it")
    used.update(hits)
    return dict(statics[hits[0]]) if hits else None


def modeled(program: Any, rows: Mapping[str, dict], spec: Mapping[str, Any], arg_dyn: Mapping[int, list[bool]], *,
            token: str) -> dict[str, dict]:
    """The Program's callees as another model's (`spec`): each callee `spec["statics"]` names (`spec_statics`; a key that names
    none is refused) bound at those statics and counted; a layer's callee (a multiple of ``layers[0]`` instances a token row)
    repeated ``layers[1]`` times a token row instead; every callee on each of ``ranks`` ranks (``instances`` stays one rank's,
    ``units`` is every rank's) but a family ``once`` names, computed once; and a `PAIRED` family's callees again from its
    partner's, one at the partner's new K an instance (each partner a K of its own: the 70B's o and qkv linears share none of
    the 8B's K = 4096). The attention's calls are left out (`window` at ``spec["attention"]`` prices them)."""
    fns = first_calls(program, arg_dyn)
    tokens = sum(r["instances"] for r in rows.values() if r["family"] == token)
    (L0, L1), ranks = spec["layers"], spec["ranks"]
    once = set(spec.get("once", ()))
    used: set[str] = set()
    for fam, by in PAIRED.items():
        mine, theirs = Counter(), Counter()
        for r in rows.values():
            if r["family"] in (fam, by):
                (mine if r["family"] == fam else theirs)[r["statics"]["K"]] += r["instances"]
        if mine != theirs:
            raise ValueError(f"{fam}'s instances by K {dict(mine)} don't pair with {by}'s {dict(theirs)}")
    out: dict[str, dict] = {}
    for cid, r in rows.items():
        if r["family"] in ATTENTION or r["family"] in PAIRED:
            continue
        fn, dyn = fns[cid]
        st = spec_statics(spec["statics"], cid, r, used)
        n = r["instances"]
        per = count(rebound(fn, **st), dyn) if st else {k: Fraction(v, n) for k, v in r["units"].items()}
        m = n * L1 // L0 if n % (tokens * L0) == 0 else n
        rk = 1 if r["family"] in once else ranks
        out[cid] = {"family": r["family"], "statics": {**r["statics"], **(st or {})}, "instances": m, "ranks": rk,
                    "refused": r["refused"], "units": {k: v * m * rk for k, v in per.items()}}
    if set(spec["statics"]) - used:
        raise ValueError(f"the spec's statics {sorted(set(spec['statics']) - used)} name no callee of the Program")
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


def levers(program: Any, rows: Mapping[str, dict], arg_dyn: Mapping[int, list[bool]], w: AttentionWindow, *, token: str,
           sample: Sequence[str]) -> dict:
    """What the levers act on, from the Program's own Definitions at `rows`' statics (a modeled variant's rows, with its window
    `w`): the attention window's units a key and a call by inner family (``attention_per_key``, ``attention_per_call``: a K'
    quantized once at the KV write takes ``Int8StaticQuant``'s from every key a step reads to ``key_write``, its own cost, once a
    token row; a V' stored once with a static scale takes ``Int8ColumnQuant``'s to ``key_write`` again), each int8 linear's
    activation quantizer's units a call by inner family (a static activation scale removes ``Int8RowScale``'s), the LM head as
    an int8 linear (``Int8Linear_v1`` at the BF16 ``Gemm_v2``'s K and N), and ``primitives``: the units a token row and a sampled
    row spend by ``primitive|class``, and the attention window's coefficients by the same keys (`fit`, exact)."""
    fns = first_calls(program, arg_dyn)
    slope = per_key(w)
    a = w.families(128)
    call = {f: {k: v for k, v in ((k, Fraction(n) - 128 * slope.get(f, {}).get(k, 0)) for k, n in a[f].items()) if v}
            for f in ("Int8StaticQuant_v1", "Int8ColumnQuant_v1") if f in a}
    write = {k: w.G * v for k, v in count(bind(REGISTRY.defs["Int8StaticQuant_v1"], K=w.kv.bindings["D"]), [True, False]).items()}
    if slope.get("Int8StaticQuant_v1") != write or call.get("Int8StaticQuant_v1"):
        raise AssertionError(f"K' costs {slope.get('Int8StaticQuant_v1')} a key and {call.get('Int8StaticQuant_v1')} a call, "
                             f"not one write's {write} a key")
    out: dict[str, Any] = {"attention_per_key": slope, "attention_per_call": call, "key_write": write, "linear": {}}
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
        if r["family"] in ATTENTION + ("(parameters)",):
            continue
        f, d = fns.get(cid) or fns[r["family"]]
        g = f if r["statics"] == dict(getattr(f, "bindings", {})) else rebound(f, **r["statics"])
        part, n = ("sample", samples) if r["family"] in sample else ("token", tokens)
        prims[part].update({p: Fraction(v * r["instances"] * r.get("ranks", 1), n) for p, v in by_primitive(g, d).items()})
    out["primitives"] = {k: dict(v) for k, v in prims.items()}
    out["primitives"]["attention"] = fit(w.primitives, FIT_POINTS, [(N, S, None) for N, S in FIT_CHECKS])
    return out


def describe(program: Any, rows: Mapping[str, dict], *, token: str, sample: Sequence[str], w: AttentionWindow,
             co: Mapping[str, Mapping[str, Fraction]], layers: int, extra: Mapping[str, Any] | None = None) -> dict:
    parts = components(rows, token=token, sample=sample, attention=ATTENTION)
    tok = Counter(parts["token"])
    for k, v in ((extra or {}).get("token") or {}).items():
        tok[k] += Fraction(v)
    return {"schema": "verity-stacked-share/units/v2", "classes": P.digest(),
            "program": {"function": program.root.id, "root_nodes": len(program.root.body.nodes)},
            "tokens": parts["tokens"], "samples": parts["samples"], "token": dict(tok), "sample": parts["sample"],
            "attention": {"families": list(ATTENTION), "statics": w.statics(), "coefficients": co, "calls": layers},
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
    ap.add_argument("--heads", type=int, default=32, help="H: the query heads of a row")
    ap.add_argument("--kv-heads", type=int, default=8, help="G: the KV heads of a request-step-layer")
    ap.add_argument("--as", dest="spec", type=Path, help="a modeled variant's spec (JSON: label, layers, ranks, once, statics, "
                    "attention: one rank's H and G, extra), written to --as-out")
    ap.add_argument("--as-out", type=Path)
    a = ap.parse_args(argv)
    program = load(a.descriptor)
    sample = a.sample.split(",")
    ends = step_ends(program, [sample[-1]])
    u = units(program, ends)
    dyn = {int(k): v for k, v in u["arg_dyn"].items()}
    w, layers = window(program, u["callees"], dyn, a.kv_heads, a.heads)
    co = attention_fit(w, windows_of(u["callees"]))
    doc = describe(program, u["callees"], token=a.token, sample=sample, w=w, co=co, layers=layers)
    doc["program"]["descriptor"] = str(a.descriptor)
    doc["steps"], doc["reasons"] = u["steps"], u["reasons"]
    doc["levers"] = levers(program, u["callees"], dyn, w, token=a.token, sample=sample)
    check_primitives(doc)
    a.out.write_text(json.dumps(jsonable(doc), indent=1, default=str))
    print(json.dumps(jsonable({k: doc[k] for k in ("tokens", "samples", "token", "sample", "attention", "reasons", "total")}),
                     indent=1, default=str))
    if a.spec:
        spec = json.loads(a.spec.read_text())
        rows = modeled(program, u["callees"], spec, dyn, token=a.token)
        w2, _ = window(program, u["callees"], dyn, a.kv_heads, a.heads, spec=spec["attention"])
        m = describe(program, rows, token=a.token, sample=sample, w=w2, co=attention_fit(w2),
                     layers=spec["layers"][1] * spec["ranks"], extra=spec.get("extra"))
        m["program"]["modeled"] = {k: spec[k] for k in ("label", "layers", "ranks", "once", "statics", "attention") if k in spec}
        m["levers"] = levers(program, rows, dyn, w2, token=a.token, sample=sample)
        check_primitives(m)
        a.as_out.write_text(json.dumps(jsonable(m), indent=1, default=str))
        print(json.dumps(jsonable({k: m[k] for k in ("tokens", "samples", "token", "sample", "attention", "total")}), indent=1,
                         default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
