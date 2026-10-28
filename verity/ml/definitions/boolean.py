"""The served Program on bits: the Boolean version of each Definition a derived Program calls, and how far a Program is from
pure bits.

A Definition is Boolean (`verity.ml.circuits.boolean.is_boolean`) when it computes on 1-bit wires with And/Xor/Not gates and constant bits
only, through every sub-Call.  A Boolean Definition names the word Definition it computes (`verity.ml.circuits.boolean.trace.word_of`: its
``word``, a Definition, or for one with statics a function of its bindings, in which case it carries its word's name).  Those
links are the one table this module reads: `boolean_version(fn)` is the registered Boolean Definition whose word is ``fn``,
looked up among every module of `verity.ml.circuits.ops.boolean` and of `verity.ml.definitions.registry` (`registry.catalog`), and
`NoBooleanVersion` names each Definition it finds none for.  A function-valued static is bound to its own Boolean version; a
static the word has no counterpart for, to the Boolean version of the word Definition the Boolean Definition's ``word_statics``
names for it.  A word primitive made on demand (`prims.const`'s `Const<w>[0x..]`, `GatherBf16x{V}`, `GatherI32x{V}`) has its
Boolean version made on demand too (`_families`, by id: `made_on_demand`), since no module can register one per constant or
vocabulary size ahead.

`sigma_lift(word, version)` is a word composite as a registered Boolean Definition: at each binding, its word body with each Call
the Boolean version of its Definition, over the bits each word operand occupies (`lift`'s replay); `boolean_version` serves it
only where every Definition that body Calls has one (`registry.boolean_sigma`).

`lift(fn)` is a derived Program's function (`frontend.derive`'s flat `Derived[...]`) on bits: the same Calls in the same order,
each of the Boolean version of its Definition, over parameters and values whose every w-bit word is an element
``Array<w, Value<1>>`` (`boolean.bit_type`).  Word leaf j of a collection is element j of its bit form, so node indices and the
runtime correspondence carry over unchanged.  `frontend.boolean.derived(report)` is a word derivation's report with its Program on
bits, which the Build takes when `TargetProfile.ir == "boolean"`; the default (word) derives exactly as before.  The Build writes a lifted
Program's word Program beside it (`WORD_DESCRIPTOR`), and the readers that work in word units (the workload composition, the
Match) read a derive dir through `word_descriptor`.

`purity(program)` lists every Definition a Program reaches that is not Boolean (a word-id MUFU sub-Call inside a Boolean
version counts): zero when the Program is bits throughout.  `dry_run(program)` reads a word Program and reports what its Boolean
Program would reach today: each derived Call's Boolean version or its gap, and the same count over the would-be Program
(`verity-vllm boolean-purity`, `program.boolean_purity`).
"""
from __future__ import annotations

import bisect
import gzip
import importlib
import json
import pkgutil
import re
from pathlib import Path
from typing import Any, Callable

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import REGISTRY, Body, Builder, CompositeDefinition, FunctionDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from verity.ml.circuits.refs import Coll, PartLog, concat, prefix_runs, runs
from verity.ml.circuits.types import Array, Record, Tuple, Type, Value

SCHEMA = "verity-vllm/boolean-purity/v1"
#: a Boolean derive dir's word Program, the one its Program is σ-checked against node by node (its digest the derive report's
#: ``ir.word_digest``, result.json's ``program.word_digest``)
WORD_DESCRIPTOR = "word_descriptor.json.gz"


def word_descriptor(d: str | Path) -> Path:
    """Derive dir ``d``'s descriptor in word units: its word Program's (`WORD_DESCRIPTOR`) for a Boolean derive, else its own."""
    p = Path(d) / WORD_DESCRIPTOR
    return p if p.exists() else Path(d) / "descriptor.json.gz"


def write_word(d: str | Path, program, word_digest: str) -> str:
    """Write a lifted Program's word Program into derive dir ``d`` (`WORD_DESCRIPTOR`) and return its digest, refused unless it is
    ``word_digest`` (the derive's ``ir.word_digest``)."""
    from verity.ml.circuits.codec import encode_program, program_digest

    desc = encode_program(program.word)
    digest = program_digest(desc)
    if digest != word_digest:
        raise ValueError(f"the word Program's descriptor is {digest[:16]}, not the derive's word digest {word_digest[:16]}")
    with gzip.open(Path(d) / WORD_DESCRIPTOR, "wb") as f:
        f.write(json.dumps(desc, sort_keys=True).encode())
    return digest


class NoBooleanVersion(LookupError):
    """``missing``: word Definition id -> why no Boolean version serves it."""

    def __init__(self, missing: dict[str, str]) -> None:
        self.missing = dict(sorted(missing.items()))
        names = sorted({_base(k) for k in self.missing})
        super().__init__(f"no Boolean version of {', '.join(names)}: " + "; ".join(f"{k}: {v}" for k, v in self.missing.items()))


def _base(fid: str) -> str:
    """`Name_vN` of a specialization id; a lowered root `Root[<program fn id>]_v1` as `Root[<its Name_vN>]`."""
    if fid.startswith("Root[") and fid.endswith("]_v1"):
        return f"Root[{_base(fid[5:-4])}]"
    return fid.split("{", 1)[0]


def _fid(fn: Any) -> str:
    return descriptor_id(bind(fn) if isinstance(fn, CompositeDefinition) else fn)


# ---- the table: word id -> Boolean Definition, from the Boolean Definitions' own `word` links ---------------------------------

_TABLE: list = []


def _load() -> tuple[dict[str, list], dict[str, list], list[str]]:
    """(exact, by_name, errors): Boolean Definitions whose word is one Definition, keyed by its id; those whose word is a function
    of their bindings, keyed by their name; the modules that failed to import."""
    if _TABLE:
        return _TABLE[0]
    from verity.ml.definitions.registry.catalog import load as load_registry
    _, errors = load_registry()
    errors = list(errors)
    core = importlib.import_module("verity.ml.circuits.ops.boolean")
    for m in sorted(pkgutil.iter_modules(core.__path__), key=lambda m: m.name):
        try:
            importlib.import_module(f"verity.ml.circuits.ops.boolean.{m.name}")
        except Exception as e:  # noqa: BLE001  (named in every gap: a module that fails to import serves nothing)
            errors.append(f"verity.ml.circuits.ops.boolean.{m.name}: {type(e).__name__}: {e}")
    exact: dict[str, list] = {}
    by_name: dict[str, list] = {}
    for d in list(REGISTRY.defs.values()):
        w = getattr(d, "word", None)
        if not isinstance(d, CompositeDefinition) or w is None:
            continue
        if isinstance(w, FunctionDefinition):
            exact.setdefault(_fid(w), []).append(d)
        elif callable(w):
            by_name.setdefault(d.name, []).append(d)
    _TABLE.append((exact, by_name, errors))
    return _TABLE[0]


def load() -> list[str]:
    """Every module that registers a Boolean Definition imported (`verity.ml.circuits.ops.boolean`'s and the integration's registry),
    so a Boolean Definition's id resolves in `REGISTRY`; the modules that failed to import."""
    return list(_load()[2])


_VERSIONS: dict[str, Any] = {}


def boolean_version(fn: FunctionDefinition) -> FunctionDefinition:
    """The Boolean Definition whose word is ``fn``: the one registered Boolean Definition linked to it, its newest version when
    several versions of one name are, with ``bit_type`` of ``fn``'s signature as its own.  `NoBooleanVersion` otherwise."""
    fid = _fid(fn)
    hit = _VERSIONS.get(fid)
    if hit is not None:
        if isinstance(hit, NoBooleanVersion):
            raise hit
        return hit
    try:
        hit = _find(fn, fid)
    except NoBooleanVersion as e:
        _VERSIONS[fid] = e
        raise
    _VERSIONS[fid] = hit
    return hit


def _statics_on_bits(kw: dict) -> dict:
    """A word Definition's bindings with each function-valued static replaced by its Boolean version (a Boolean Definition with
    statics binds Boolean functions, `DotBf16_v3{K, DOT}`, and names its word through them)."""
    def on_bits(v):
        if isinstance(v, CompositeDefinition):
            v = bind(v)
        if isinstance(v, FunctionDefinition):
            return v if BL.is_boolean(v) else boolean_version(v)
        if isinstance(v, (list, tuple)):
            return type(v)(on_bits(x) for x in v)
        if isinstance(v, dict):
            return {k: on_bits(x) for k, x in v.items()}
        return v
    return {k: on_bits(v) for k, v in kw.items()}


def _families() -> tuple:
    """The Boolean versions made on demand, as their word Definitions are (a family by id): (word id pattern, its Boolean
    version's factory)."""
    if not _FAMILIES:
        from verity.ml.circuits.ops.boolean import scalar as SC
        from verity.ml.definitions.registry import boolean_gather as BGA
        _FAMILIES.extend([
            (re.compile(r"^Const([1-9][0-9]*)\[0x([0-9a-f]+)\]_v1$"), lambda m: SC.const_word(int(m[1]), int(m[2], 16))),
            (re.compile(r"^GatherBf16x(\d+)_v1$"), lambda m: BGA.gather_bf16(int(m[1]))),
            (re.compile(r"^GatherI32x(\d+)_v1$"), lambda m: BGA.gather_i32(int(m[1]))),
        ])
    return tuple(_FAMILIES)


_FAMILIES: list = []


def made_on_demand(fid: str) -> FunctionDefinition | None:
    """The Boolean version `_families` makes for the word Definition ``fid`` names (`Const32[0x00000000]_v1`,
    `GatherI32x16_v1`, ...), bound; None when no family made on demand has ``fid``."""
    for pat, make in _families():
        if (m := pat.match(fid)) is not None:
            d = make(m)
            return bind(d) if isinstance(d, CompositeDefinition) else d
    return None


def _find(fn: FunctionDefinition, fid: str) -> FunctionDefinition:
    if isinstance(fn, PrimitiveDefinition) and BL.is_boolean(fn):
        return fn
    exact, by_name, errors = _load()
    cands = [bind(d) for d in exact.get(fid, ())]
    if not cands and (d := made_on_demand(fid)) is not None:
        cands.append(d)
    bindings = getattr(fn, "bindings", None) or {}
    notes: list[str] = []
    if isinstance(fn, SpecializedDefinition) and bindings:
        from verity.ml.circuits.boolean.trace import word_of
        for d in by_name.get(fn.name, ()):
            # ``word_statics``: the statics its word has no counterpart for, as the word Definitions they compute, each bound to its
            # Boolean version (`RMSNormFusedCuda_v3`'s RSQRT computes `RsqrtApprox_v1`)
            extra = getattr(d, "word_statics", {})
            if not set(d.statics) <= set(bindings) | set(extra):
                continue
            raw = {k: bindings[k] for k in d.statics if k in bindings}
            try:
                more = _statics_on_bits({k: extra[k] for k in d.statics if k not in bindings})
            except NoBooleanVersion as e:
                notes.append(f"{d.name}_v{d.version}'s {', '.join(k for k in d.statics if k not in bindings)}: {e}")
                continue
            for on_bits in True, False:
                try:
                    sp = bind(d, **(_statics_on_bits(raw) if on_bits else raw), **more)
                    if _fid(word_of(sp)) == fid:
                        cands.append(sp)
                        break
                except (LookupError, TypeError, ValueError):
                    continue
    if not cands:
        why = "; ".join(notes) or "no registered Boolean Definition names it as its word"
        if errors:
            why += f" (registry modules that failed to import: {errors})"
        raise NoBooleanVersion({fid: why})
    names = {c.name for c in cands}
    if len(names) > 1:
        raise NoBooleanVersion({fid: f"ambiguous: {sorted(_fid(c) for c in cands)} all name it as their word"})
    best = max(cands, key=lambda c: c.version)
    params, ret = fn.signature()
    bparams, bret = best.signature()
    want = (tuple(BL.bit_type(t) for _, t in params), BL.bit_type(ret))
    if (tuple(t for _, t in bparams), bret) != want:
        raise NoBooleanVersion({fid: f"{_fid(best)}'s signature {[t for _, t in bparams]} -> {bret!r} is not the bit form "
                                     f"{list(want[0])} -> {want[1]!r} of the word signature"})
    if getattr(getattr(best, "defn", best), "sigma_of", None) is not None:
        _, missing = _sigma_callees(fn)
        if missing:
            names = ", ".join(sorted({_base(k) for k in missing}))
            raise NoBooleanVersion({fid: f"{_fid(best)} is σ of it, and its body Calls {names} with no Boolean version",
                                    **missing})
    return best


# ---- the lift: a derived function on bits ----------------------------------------------------------------------------------------

def _bits(t: Type) -> int:
    if isinstance(t, Value):
        return t.w
    if isinstance(t, Array):
        return t.n * _bits(t.elem)
    if isinstance(t, Tuple):
        return sum(_bits(f) for f in t.fields)
    if isinstance(t, Record):
        return sum(_bits(f) for _, f in t.fields)
    raise TypeError(t)


def bit_offsets(t: Type) -> Callable[[int], int]:
    """Word leaf index j of ``t`` (0..leaves) -> the index of its first bit in ``bit_type(t)``: each leaf of width w becomes w
    consecutive bits in leaf order, so a run of word leaves is one run of bits."""
    if not t.leaves:
        return lambda j: 0
    w = t.homogeneous_width()
    if w is not None:
        return lambda j: j * w
    if isinstance(t, Array):
        L, b, inner = t.elem.leaves, _bits(t.elem), bit_offsets(t.elem)
        return lambda j: (j // L) * b + inner(j % L)
    fields = list(t.fields) if isinstance(t, Tuple) else [f for _, f in t.fields]
    starts, bstarts, inners, pos, bpos = [], [], [], 0, 0
    for f in fields:
        starts.append(pos)
        bstarts.append(bpos)
        inners.append(bit_offsets(f))
        pos += f.leaves
        bpos += _bits(f)

    def off(j: int) -> int:
        k = bisect.bisect_right(starts, j) - 1
        return bstarts[k] + inners[k](j - starts[k])

    return off


def _substitute(refs, env: dict, offs: dict, logs: dict | None = None):
    """`replay.substitute` from word leaves to bits: each word run of ``(space, idx)`` is the run of bits it occupies in
    ``env[(space, idx)]``; the runs a `PartLog` shares are mapped once per log."""
    rs, log, s = prefix_runs(refs)

    def part(space, idx, lo, hi):
        o = offs[(space, idx)]
        return env[(space, idx)].slice(o(lo), o(hi))

    if log is None or logs is None or s == 0:
        parts = [part(*r) for r in rs]
        return concat(parts) if parts else refs
    hit = logs.get(id(log))
    if hit is None or hit[0] is not log:
        hit = logs[id(log)] = (log, PartLog())
    sub = hit[1]
    if len(sub) < s:
        for r in log.shared_runs(len(sub), s):
            sub.append(part(*r))
    return sub.cut(s, [part(*r) for r in rs[s:]])


def _bit_replayer(body: Body, ptypes: list[Type], callee: dict[int, FunctionDefinition]):
    def replay(B: Builder, S, *args: Coll) -> Coll:
        env: dict[tuple[str, int], Any] = {}
        offs: dict[tuple[str, int], Callable[[int], int]] = {}
        for i, (a, t) in enumerate(zip(args, ptypes)):
            env[("p", i)], offs[("p", i)] = a.refs, bit_offsets(t)
        logs: dict[int, tuple[Any, Any]] = {}
        for k, node in enumerate(body.nodes):
            fn = callee[id(node.fn)]
            colls = [Coll(BL.bit_type(a.type), _substitute(a.refs, env, offs, logs)) for a in node.args]
            if node.form in ("call", "primitive"):
                out = B.call(fn, *colls, name=node.name)
            elif node.form == "batch":
                out = B.batch(fn, *colls, axes=node.axes, n=node.n, name=node.name)
            else:
                out = B.scan(fn, colls[0], xs=colls[1:1 + node.n_xs], shared=colls[1 + node.n_xs:], n=node.n, name=node.name)
            env[("n", k)], offs[("n", k)] = out.refs, bit_offsets(node.out)
        return Coll(BL.bit_type(body.ret_type), _substitute(body.ret, env, offs))

    return replay


def _sigma_callees(fn: FunctionDefinition) -> tuple[dict[int, FunctionDefinition], dict[str, str]]:
    """σ on ``fn``'s body: each Definition it Calls -> its Boolean version (by id of the callee object), and the gaps."""
    callee: dict[int, FunctionDefinition] = {}
    missing: dict[str, str] = {}
    for n in fn.body.nodes:
        if id(n.fn) in callee:
            continue
        try:
            callee[id(n.fn)] = boolean_version(n.fn)
        except NoBooleanVersion as e:
            missing.update(e.missing)
            callee[id(n.fn)] = n.fn
    return callee, missing


def lift(fn: SpecializedDefinition) -> SpecializedDefinition:
    """``fn`` (a derived function: an unregistered composite whose body Calls registered Definitions) on bits, with the static
    ``IR="boolean"`` beside its own.  `NoBooleanVersion` names every Definition its body Calls that has no Boolean version."""
    callee, missing = _sigma_callees(fn)
    if missing:
        raise NoBooleanVersion(missing)
    params, ret = fn.signature()
    bparams, bret = tuple((n, BL.bit_type(t)) for n, t in params), BL.bit_type(ret)
    defn = CompositeDefinition(fn.name, fn.version, (*fn.defn.statics, "IR"), lambda S: (bparams, bret),
                               _bit_replayer(fn.body, [t for _, t in params], callee),
                               doc=f"{fn.defn.doc}; on bits (each Call the Boolean version of its Definition)",
                               conformance=fn.defn.conformance, register=False)
    return bind(defn, **fn.bindings, IR="boolean")


def _word_static(v: Any) -> Any:
    """A Boolean Definition's binding as its word's: a Boolean Definition as the word Definition it computes (`trace.word_of`)."""
    from verity.ml.circuits.boolean.trace import word_of
    if isinstance(v, CompositeDefinition):
        v = bind(v)
    if isinstance(v, FunctionDefinition) and getattr(getattr(v, "defn", v), "word", None) is not None:
        return word_of(v)
    if isinstance(v, (list, tuple)):
        return type(v)(_word_static(x) for x in v)
    if isinstance(v, dict):
        return {k: _word_static(x) for k, x in v.items()}
    return v


def sigma_lift(word: CompositeDefinition, version: int, register: bool = True) -> CompositeDefinition:
    """The Boolean Definition ``{word.name}_v{version}``, σ of the word composite ``word``: at each binding, ``word``'s body at
    that binding with each Call the Boolean version of its Definition (`boolean_version`, so the newest registered one at the
    time the body is built), over the bits each word operand occupies (`lift`'s replay).  A function-valued static binds a
    Boolean Definition (`_statics_on_bits`) and is read as its word (`trace.word_of`); the word view is ``word`` at the bindings
    so read.  `boolean_version` serves it only where every Definition the word body Calls has a Boolean version, and names
    those that have none otherwise."""
    statics = tuple(word.statics)

    def at(S) -> SpecializedDefinition:
        return bind(word, **{k: _word_static(getattr(S, k)) for k in statics})

    def signature(S):
        params, ret = at(S).signature()
        return tuple((n, BL.bit_type(t)) for n, t in params), BL.bit_type(ret)

    def body(B: Builder, S, *args: Coll) -> Coll:
        w = at(S)
        callee, missing = _sigma_callees(w)
        if missing:
            raise NoBooleanVersion(missing)
        params, _ = w.signature()
        return _bit_replayer(w.body, [t for _, t in params], callee)(B, S, *args)

    refuse = (lambda b: word.refuse({k: _word_static(x) for k, x in b.items()})) if word.refuse is not None else None
    d = CompositeDefinition(word.name, version, statics, signature, body,
                            doc=f"σ({word.name}_v{word.version}): its body on bits, each Call the Boolean version of its "
                                "Definition", conformance=word.conformance, refuse=refuse, register=register)
    word_view = (lambda b: bind(word, **{k: _word_static(x) for k, x in b.items()})) if statics else bind(word)
    d.word = word_view  # type: ignore[attr-defined]
    d.sigma_of = word  # type: ignore[attr-defined]
    return d


def _merged(rs, offs: dict | None = None) -> list[tuple[str, int, int, int]]:
    """Runs ``rs`` with adjacent ones of one source joined; with ``offs``, each word run as the bits it occupies."""
    out: list[list] = []
    for space, idx, lo, hi in rs:
        if offs is not None:
            lo, hi = offs[(space, idx)](lo), offs[(space, idx)](hi)
        if out and out[-1][0] == space and out[-1][1] == idx and out[-1][3] == lo:
            out[-1][3] = hi
        elif hi > lo:
            out.append([space, idx, lo, hi])
    return [tuple(r) for r in out]


def sigma_check(word: SpecializedDefinition, bits: SpecializedDefinition) -> dict:
    """``bits`` against σ(``word``), node by node and without `lift`: each node Calls the Boolean version of the word node's
    Definition in the same form (n, axes, scan inputs, name), and each operand and the return are the word's runs on the bits they
    occupy (`bit_offsets`), in the bit types.  {"nodes", "definitions" (σ's entries), "mismatches" (the first 8), "ok"}."""
    (wp, wr), (bp, br) = word.signature(), bits.signature()
    bad = [] if ([BL.bit_type(t) for _, t in wp], BL.bit_type(wr)) == ([t for _, t in bp], br) else ["signature: not the word's on bits"]
    wb, bb = word.body, bits.body
    if len(wb.nodes) != len(bb.nodes):
        bad.append(f"{len(bb.nodes)} nodes, the word body {len(wb.nodes)}")
    offs: dict = {("p", i): bit_offsets(t) for i, (_, t) in enumerate(wp)}
    sigma: dict[int, str] = {}
    one = {"primitive": "call"}  # a word primitive's Call is a Call of its Boolean composite
    for k, (w, b) in enumerate(zip(wb.nodes, bb.nodes)):
        if id(w.fn) not in sigma:
            v = boolean_version(w.fn)
            sigma[id(w.fn)] = descriptor_id(bind(v) if isinstance(v, CompositeDefinition) else v)
        if b.fn is not w.fn and descriptor_id(b.fn) != sigma[id(w.fn)]:
            bad.append(f"node {k}: Calls {descriptor_id(b.fn)}, σ({descriptor_id(w.fn)}) = {sigma[id(w.fn)]}")
        bf, wf = one.get(b.form, b.form), one.get(w.form, w.form)
        if (bf, b.n, b.axes, b.name, getattr(b, "n_xs", 0)) != (wf, w.n, w.axes, w.name, getattr(w, "n_xs", 0)) \
                or len(b.args) != len(w.args):
            bad.append(f"node {k}: form {b.form}/{b.n}/{b.axes}/{b.name}, the word's {w.form}/{w.n}/{w.axes}/{w.name}")
        for g, (wa, ba) in enumerate(zip(w.args, b.args)):
            if ba.type != BL.bit_type(wa.type) or _merged(runs(ba.refs)) != _merged(runs(wa.refs), offs):
                bad.append(f"node {k} operand {g}: not the word operand's bits")
        offs[("n", k)] = bit_offsets(w.out)
    if _merged(runs(bb.ret)) != _merged(runs(wb.ret), offs):
        bad.append("return: not the word return's bits")
    return {"nodes": len(wb.nodes), "definitions": len(set(sigma.values())), "mismatches": bad[:8], "mismatch_n": len(bad),
            "ok": not bad}


def boolean_function(fn: SpecializedDefinition) -> SpecializedDefinition:
    """A derived Program's function on bits: the flat derived function lifted (`lift`), a registered one (a family's claim) by its
    Boolean version."""
    if isinstance(fn, SpecializedDefinition) and not fn.defn.register and fn.name.startswith("Derived["):
        return lift(fn)
    v = boolean_version(fn)
    return bind(v) if isinstance(v, CompositeDefinition) else v


# ---- purity: the Definitions a Program reaches that are not Boolean ------------------------------------------------------------

def _reachable(roots) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    stack = list(roots)
    while stack:
        fn = stack.pop()
        fid = descriptor_id(fn)
        if fid in seen:
            continue
        seen[fid] = fn
        if not isinstance(fn, PrimitiveDefinition):
            stack.extend(n.fn for n in fn.body.nodes)
    return seen


def _non_boolean(defs: dict[str, Any]) -> dict:
    """The not-Boolean ones of ``defs`` grouped by Definition (`Name_vN`): how many specializations, and the not-Boolean
    Definitions each one Calls directly (the reason it is not Boolean beyond its own signature)."""
    bad = {fid: fn for fid, fn in defs.items() if not BL.is_boolean(fn)}
    by: dict[str, dict] = {}
    for fid, fn in sorted(bad.items()):
        e = by.setdefault(_base(fid), {"id": _base(fid), "kind": "primitive" if isinstance(fn, PrimitiveDefinition) else "composite",
                                       "specializations": 0, "example": fid, "calls": set()})
        e["specializations"] += 1
        if not isinstance(fn, PrimitiveDefinition):
            e["calls"] |= {_base(descriptor_id(n.fn)) for n in fn.body.nodes if descriptor_id(n.fn) in bad}
    rows = [{**e, "calls": sorted(e["calls"])} for _, e in sorted(by.items())]
    return {"definitions": len(rows), "specializations": len(bad), "list": rows}


def purity(program) -> dict:
    """Every Definition ``program`` reaches (a `Program` or a decoded descriptor) that is not Boolean, by `Name_vN`."""
    defs = _reachable([program.root])
    return {"schema": SCHEMA, "mode": "built", "reached": {"definitions": len({_base(f) for f in defs}), "specializations": len(defs)},
            "non_boolean": _non_boolean(defs)}


def _is_input(n) -> bool:
    return n.form == "batch" and isinstance(n.fn, PrimitiveDefinition) and getattr(n.fn, "family", None) == "Input"


def derived_calls(program) -> list:
    """The Calls the derive rules emitted: the root's non-Input nodes of a derived Program, each request's body under a
    workload Program (`WorkloadRequest_v1` components)."""
    out = []
    for n in program.root.body.nodes:
        if _is_input(n):
            continue
        if n.fn.name == "WorkloadRequest":
            out.extend(n.fn.body.nodes)
        else:
            out.append(n)
    return out


def dry_run(program) -> dict:
    """What ``program`` (a word Program) reaches on bits today: each derived Call's Definition with its Boolean version or its gap,
    and the not-Boolean Definitions of the would-be Boolean Program -- each gap's whole word subtree, what each Boolean version
    still reaches that is not Boolean, and the wrappers above them (the root, each request component) while any remain."""
    calls: dict[str, dict] = {}
    would: dict[str, Any] = {}
    gaps: dict[str, str] = {}
    for n in derived_calls(program):
        fid = descriptor_id(n.fn)
        e = calls.setdefault(_base(fid), {"word": _base(fid), "boolean": set(), "specializations": set(), "calls": 0})
        e["calls"] += 1
        if fid in e["specializations"]:
            continue
        e["specializations"].add(fid)
        try:
            v = boolean_version(n.fn)
        except NoBooleanVersion as x:
            gaps.update(x.missing)
            would.update(_reachable([n.fn]))
            continue
        e["boolean"].add(_base(_fid(v)))
        would.update(_reachable([bind(v) if isinstance(v, CompositeDefinition) else v]))
    rows = []
    for _, e in sorted(calls.items()):
        rows.append({"word": e["word"], "boolean": sorted(e["boolean"]) or None, "specializations": len(e["specializations"]),
                     "calls": e["calls"], "gap": next((gaps[f] for f in sorted(e["specializations"]) if f in gaps), None)})
    nb = _non_boolean(would)
    wrappers = sorted({_base(descriptor_id(n.fn)) for n in program.root.body.nodes if n.fn.name == "WorkloadRequest"}
                      | {_base(descriptor_id(program.root))}) if nb["definitions"] else []
    count = nb["definitions"] + len(wrappers)
    return {"schema": SCHEMA, "mode": "dry-run", "derived_calls": rows,
            "gaps": sorted({r["word"] for r in rows if r["gap"]}),
            "non_boolean": {"definitions": count, "specializations": nb["specializations"] + len(wrappers),
                            "list": nb["list"] + [{"id": w, "kind": "wrapper"} for w in wrappers]}}


def load_program(path: Path):
    from verity.ml.circuits.codec import decode_program, program_digest
    from verity.ml.definitions.registry.catalog import load as load_registry
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        desc = json.load(f)
    registry, _ = load_registry()
    return decode_program(desc, registry), program_digest(desc)


__all__ = ["NoBooleanVersion", "boolean_version", "made_on_demand", "bit_offsets", "lift", "sigma_lift", "sigma_check",
           "boolean_function", "purity", "dry_run", "derived_calls", "load_program", "word_descriptor", "write_word"]
