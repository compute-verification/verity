"""Every circuit template the circuit-check catalog reaches at its bindings that calls an opener (`OPENERS`) either gates
what it returns on the opener's verdict or returns a verdict alone, and each that does neither is an entry of
``allowlists/repo_openers_gated.json`` naming what consumes its verdict today; the list only shrinks.  The walk takes each
template once per shape (its statics, each size only as zero or not), at its least binding, by its own nodes alone; a
traced template, or one whose code reaches no opener's module, is a leaf.  With each opener's verdict cut to 0 and every
parameter unknown, a reaching template's body is evaluated in three values (0, 1, unknown) with each parameter's
dependence carried beside its value: a returned element that comes out constant is gated, one that depends on no
parameter reaching an opener's opened value or path is a verdict, and any other is ungated.  A call that reaches no opener
on parameters all unknown is not entered.  A template the walk can't decide (a body that doesn't build or compile, a
shape whose next binding classes otherwise, a module that names an opener its imports don't reach) fails.
``openers_gated.json`` is the record of the templates that call an opener at each binding a circuit uses: an
application's, read off the description of its epoch circuit as the adapter builds it at each GPU card's target, and the
tests' and trials' (`TESTED`); each is evaluated at that very binding, with its closure digest, and says where it is used.
``--write`` writes it.  The tests that walk are slow: no cheaper test finds an opener, which is seen only in the bodies
that call it."""
from __future__ import annotations

import ast
import heapq
import importlib.util
import itertools
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
LIST = Path(__file__).resolve().parent / "allowlists" / "repo_openers_gated.json"
KIND = "ungated"
FIELDS = ("file", "kind", "symbol", "detail")
OPENERS = {"FrameV3Read_v1": ("value", "path"), "FrameV3ReadAt_v1": ("value", "path"),
           "MerkleRead_v1": ("leaf", "path"), "LmsVerify_v1": ("message", "C", "y", "path")}
GATED, VERDICT, UNGATED = "gated", "verdict", "ungated"
RECORD = Path(__file__).resolve().parent / "openers_gated.json"
APPLICATIONS = ROOT / "applications"
#: the bindings of a template that calls an opener in the circuits tests and processor trials encode, by the module
#: defining it: each key as the circuit's description names it
TESTED: dict[str, tuple[str, ...]] = {
    "verity.commitments.gates.embedding": ("EmbeddingRead_v4{V=256,H=64}", "EmbeddingRead_v4{V=20,H=8}"),
}


def _imports() -> tuple[Any, Any, Any, Any]:
    sys.path[:0] = [p for p in (str(ROOT / "tools/verity/circuit_check/src"),) if p not in sys.path]
    from circuit_check import bodies as BD
    from circuit_check import by_calls as BC
    from circuit_check import targets as T
    from verity.circuits import defs as D
    return BD, T, D, BC


class Imports:
    """Whether a module's code can reach an opener's module: its imports, function-level ones included, and its packages',
    closed over; ``mentions`` the modules whose source names an opener without that."""

    def __init__(self) -> None:
        from verity.commitments.gates import frame_v3, lms, merkle

        defs = (frame_v3.FrameV3Read, frame_v3.FrameV3ReadAt, merkle.MerkleRead, lms.LmsVerify)
        assert {f"{d.name}_v{d.version}" for d in defs} == set(OPENERS)
        self.names = tuple(d.name for d in defs)
        self.targets = {d.body.__module__ for d in defs}
        self.edges: dict[str, set[str]] = {}
        self.memo: dict[str, bool] = {}
        self.mentions: set[str] = set()

    @staticmethod
    def origin(name: str) -> str | None:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError, AttributeError):
            return None
        return spec.origin if spec is not None and spec.origin and spec.origin.endswith(".py") else None

    def imports(self, mod: str) -> set[str]:
        if mod not in self.edges:
            self.edges[mod] = out = {".".join(mod.split(".")[:i]) for i in range(1, mod.count(".") + 1)}
            if (path := self.origin(mod)) is not None:
                pkg = mod.split(".") if path.endswith("__init__.py") else mod.split(".")[:-1]
                for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
                    if isinstance(node, ast.Import):
                        out.update(a.name for a in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        m = ".".join((pkg[:len(pkg) - node.level + 1] if node.level else [])
                                     + (node.module.split(".") if node.module else []))
                        out.update([m] if m else [])
                        out.update(f"{m}.{a.name}" for a in node.names if m and self.origin(f"{m}.{a.name}"))
        return self.edges[mod]

    def reaches(self, mod: str) -> bool:
        if mod not in self.memo:
            seen, stack = set(), [mod]
            while stack and not seen & self.targets:
                m = stack.pop()
                if m not in seen:
                    seen.add(m)
                    stack.extend(self.imports(m))
            self.memo[mod] = bool(seen & self.targets)
            path = self.origin(mod)
            if not self.memo[mod] and path is not None and any(n in Path(path).read_text(encoding="utf-8") for n in self.names):
                self.mentions.add(mod)
        return self.memo[mod]

    def can_reach(self, v: Any) -> bool:
        """Whether a template, or a static value, can call an opener: its body's module reaches one, or a static does."""
        D = _imports()[2]
        if isinstance(v, D.SpecializedDefinition):
            return self.reaches(v.defn.body.__module__) or self.can_reach(list(v.bindings.values()))
        if isinstance(v, D.CompositeDefinition):
            return self.reaches(v.body.__module__)
        if isinstance(v, (tuple, list)):
            return any(self.can_reach(x) for x in v)
        if isinstance(v, dict):
            return any(self.can_reach(x) for x in v.values())
        return callable(v) and isinstance(getattr(v, "__module__", None), str) and self.reaches(v.__module__)


def _shape(v: Any, D: Any) -> str:
    if isinstance(v, D.SpecializedDefinition):
        return shape(v)
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return repr(v)
    if isinstance(v, int):
        return "0" if v == 0 else "n"
    if isinstance(v, float):
        return "f"
    if isinstance(v, (tuple, list)):
        return "(" + ",".join(_shape(x, D) for x in v) + ")"
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{_shape(x, D)}" for k, x in sorted(v.items())) + "}"
    return str(getattr(v, "id", repr(v)))


def shape(fn: Any) -> str:
    """A template's shape: its circuit template and its statics, each size taken only as zero or not."""
    D = _imports()[2]
    return fn.base_id + "{" + ",".join(f"{k}={_shape(v, D)}" for k, v in fn.bindings.items()) + "}"


def volume(v: Any) -> int:
    """The product of a binding's sizes, nested statics' included: the walk takes each shape at its least."""
    D = _imports()[2]
    if isinstance(v, D.SpecializedDefinition):
        return volume(list(v.bindings.values()))
    if isinstance(v, bool):
        return 1
    if isinstance(v, int):
        return max(v, 1)
    if isinstance(v, (tuple, list)):
        out = 1
        for x in v:
            out *= volume(x)
        return out
    if isinstance(v, dict):
        return volume(list(v.values()))
    return 1


def roots() -> list[Any]:
    """The catalog's roots as `targets.catalog` makes them, without building a model's body."""
    T = _imports()[1]
    out = list(T._registered())
    for units, make, _kind, _label in T.catalog_units():
        out += T._produce(units, make)
    return out


def walk(start: list[Any]) -> tuple[dict[str, list[tuple[int, Any]]], dict[str, set[str]], dict[str, str], list]:
    """Each shape reached from ``start``, with its bindings by size; each built shape's callee shapes; each that couldn't
    be opened, with why; and each body over 5 s to build.  Smallest binding first over the whole walk, one body per shape
    (an opener's none), each its own nodes alone (`by_calls.unlaid`), so no callee's body is built for it.  A traced
    template's (`boolean.trace`: its body is `trace.emit`'s, a body of primitive gates and of calls of traced templates) is
    none either: it reaches no opener; nor is one whose code reaches no opener's module (`Imports`), where its module
    doesn't name an opener."""
    _BD, _T, D, BC = _imports()
    from verity.circuits.boolean import trace
    imports = Imports()
    members: dict[str, list[tuple[int, Any]]] = {}
    callees: dict[str, set[str]] = {}
    undecided: dict[str, str] = {}
    slow: list[tuple[float, str]] = []
    seen: set[str] = set()
    heap: list[tuple[int, int, Any]] = []
    order = itertools.count()

    def push(fn: Any) -> None:
        if isinstance(fn, D.CompositeDefinition) and not fn.statics:
            fn = D.bind(fn)
        fid = getattr(fn, "id", None)
        if fid is None or fid in seen:
            return
        seen.add(fid)
        if isinstance(fn, D.SpecializedDefinition):
            heapq.heappush(heap, (volume(fn), next(order), fn))
        elif not isinstance(fn, D.PrimitiveDefinition) and type(fn).__name__ != "ScalarReference":
            undecided[fid] = f"a {type(fn).__name__}, neither a primitive gate nor a specialized template"

    for fn in start:
        push(fn)
    while heap:
        size, _, fn = heapq.heappop(heap)
        k = shape(fn)
        members.setdefault(k, []).append((size, fn))
        if k in callees or k in undecided:
            continue
        if fn.base_id in OPENERS or fn.defn.body.__module__ == trace.__name__ or not imports.can_reach(fn):
            callees[k] = set()
            continue
        t = time.monotonic()
        try:
            kids = list({id(n.fn): n.fn for n in BC.unlaid(fn).nodes}.values())
        except Exception as e:  # noqa: BLE001
            undecided[k] = f"{fn.id[:160]}: its body doesn't build: {type(e).__name__}: {str(e)[:160]}"
            continue
        if (dt := time.monotonic() - t) > 5:
            slow.append((round(dt, 1), fn.id[:160]))
        callees[k] = {shape(c) for c in kids if isinstance(c, D.SpecializedDefinition)}
        for c in kids:
            push(c)
    for mod in sorted(imports.mentions):
        undecided[mod] = "its source names an opener, though its imports reach none: a lookup by name the walk can't follow"
    return members, callees, undecided, slow


def reaching(members: dict[str, list[tuple[int, Any]]], callees: dict[str, set[str]]) -> set[str]:
    reach = {k for k, ms in members.items() if ms[0][1].base_id in OPENERS}
    changed = True
    while changed:
        changed = False
        for k, cs in callees.items():
            if k not in reach and cs & reach:
                reach.add(k)
                changed = True
    return reach


class Gating:
    """The three-valued run of one template over compiled bodies: a wire is (value, mask), value an int or None (unknown),
    mask the root's parameters it depends on, as a bitset."""

    def __init__(self, reach: set[str]) -> None:
        from verity.circuits import boolean as BL
        self.BD = _imports()[0]
        self.AND = BL.AND
        self.reach = reach
        self.memo: dict[tuple, tuple[tuple, tuple, int]] = {}
        self.raised: set[str] = set()

    def body(self, spec: Any) -> Any:
        return self.BD.compile_body(spec)

    def opener_mask(self, sub: Any, ws: list[int], M: list[int]) -> int:
        names = OPENERS[sub.spec.base_id]
        base, s = 0, 0
        for name, t in sub.spec.body.params:
            if name in names:
                for w in ws[base:base + t.leaves]:
                    s |= M[w]
            base += t.leaves
        return s

    def activation(self, sub: Any, vals: tuple, masks: tuple) -> tuple[tuple, tuple, int]:
        if sub.spec.base_id in OPENERS:
            return (0,) * len(sub.ret), (0,) * len(sub.ret), -1
        if shape(sub.spec) not in self.reach and sub.P and all(v is None for v in vals):
            m = 0
            for x in masks:
                m |= x
            return (None,) * len(sub.ret), (m,) * len(sub.ret), 0
        key = (sub.spec.id, vals, masks)
        hit = self.memo.get(key)
        if hit is None:
            hit = self.memo[key] = self.run(sub, list(vals), list(masks))
        return hit

    def run(self, body: Any, V: list, M: list) -> tuple[tuple, tuple, int]:
        BD = self.BD
        V.extend([None] * (body.size - body.P))
        M.extend([0] * (body.size - body.P))
        S = 0
        for plan in body.plans:
            kind = plan[0]
            if kind == "prim":
                _, prim, n, sa, _cols, _wide, out = plan
                fam = getattr(prim, "family", None)
                lazy = getattr(prim, "evaluate_lazy", None) is not None
                for m in range(n):
                    ws = BD._member(sa, m)
                    ops = [V[w] for w in ws]
                    mask = 0
                    for w in ws:
                        mask |= M[w]
                    val = None
                    if fam == "Input" or lazy:
                        val = None
                    elif all(o is not None for o in ops):
                        try:
                            val = prim.evaluate(*ops)
                        except Exception as e:  # noqa: BLE001
                            self.raised.add(f"{getattr(prim, 'id', prim)}: {type(e).__name__}")
                    elif prim is self.AND and any(o == 0 for o in ops):
                        val = 0
                    V[out + m] = val
                    M[out + m] = 0 if val is not None else mask
                continue
            if kind == "call":
                _, sub, ws, out = plan
                members = [(ws, out)]
            elif kind == "batch":
                _, sub, sa, n, out = plan
                R = len(sub.ret)
                members = [(BD._member(sa, m), out + m * R) for m in range(n)]
            else:
                _, sub, init, xs, shared, n, c, y, out = plan
                carry_v = [V[w] for w in init]
                carry_m = [M[w] for w in init]
                for m in range(n):
                    ws = [w for xw, L in xs for w in xw[m * L:(m + 1) * L]] + list(shared)
                    vals = tuple(carry_v + [V[w] for w in ws])
                    masks = tuple(carry_m + [M[w] for w in ws])
                    rv, rm, s = self.activation(sub, vals, masks)
                    S |= self.opener_mask(sub, list(init) + ws, M) if s == -1 else s
                    carry_v, carry_m = list(rv[:c]), list(rm[:c])
                    V[out + c + m * y:out + c + (m + 1) * y] = rv[c:]
                    M[out + c + m * y:out + c + (m + 1) * y] = rm[c:]
                V[out:out + c], M[out:out + c] = carry_v, carry_m
                continue
            for ws, o in members:
                rv, rm, s = self.activation(sub, tuple(V[w] for w in ws), tuple(M[w] for w in ws))
                S |= self.opener_mask(sub, ws, M) if s == -1 else s
                V[o:o + len(rv)] = rv
                M[o:o + len(rm)] = rm
        return tuple(V[w] for w in body.ret), tuple(M[w] for w in body.ret), S

    def classify(self, spec: Any) -> list[tuple[str, str]]:
        """(returned element, class) of each element of ``spec``'s return."""
        body = self.body(spec)
        params, ret = spec.signature()
        V: list = []
        M: list = []
        for i, (_, t) in enumerate(params):
            V += [None] * t.leaves
            M += [1 << i] * t.leaves
        rv, rm, S = self.run(body, V, M)
        fields = list(getattr(ret, "fields", (ret,)))
        out, lo = [], 0
        for k, t in enumerate(fields):
            vs, ms = rv[lo:lo + t.leaves], rm[lo:lo + t.leaves]
            lo += t.leaves
            if all(v is not None for v in vs):
                c = GATED
            elif not any(m & S for m in ms):
                c = VERDICT
            else:
                c = UNGATED
            out.append((str(k), c))
        return out


def parse(doc: object) -> dict[str, dict]:
    assert isinstance(doc, dict) and isinstance(doc.get("entries"), list), f"{LIST.name}: {{'entries': [...]}} expected"
    out: dict[str, dict] = {}
    for e in doc["entries"]:
        assert isinstance(e, dict) and tuple(e) == FIELDS, f"{LIST.name}: an entry is {{{', '.join(FIELDS)}}}: {e}"
        assert e["kind"] == KIND, f"{LIST.name}: an entry's kind is {KIND!r}: {e}"
        assert all(isinstance(e[k], str) and e[k] for k in FIELDS), f"{LIST.name}: every field is a nonempty string: {e}"
        assert e["symbol"] not in out, f"{LIST.name}: duplicate entry {e}"
        out[e["symbol"]] = e
    return out


def allowed() -> dict[str, dict]:
    return parse(json.loads(LIST.read_text(encoding="utf-8")))


def classes(start: list[Any] | None = None) -> tuple[dict[str, tuple[Any, list[tuple[str, str]]]], dict[str, str], dict]:
    """Each reaching shape's least binding with its classes, what couldn't be decided, and the walk's numbers.  A reaching
    shape's next binding, up to 64 times the least's size, is classified too and must agree."""
    BD = _imports()[0]
    t0 = time.monotonic()
    members, callees, undecided, slow = walk(roots() if start is None else start)
    built = time.monotonic() - t0
    reach = reaching(members, callees)
    G = Gating(reach)
    got: dict[str, tuple[Any, list[tuple[str, str]]]] = {}
    compared = 0
    for k in sorted(reach):
        ms = members[k]
        if ms[0][1].base_id in OPENERS:
            continue
        try:
            got[k] = (ms[0][1], G.classify(ms[0][1]))
            if len(ms) > 1 and ms[1][0] <= 64 * ms[0][0]:
                compared += 1
                if (other := G.classify(ms[1][1])) != got[k][1]:
                    undecided[k] = f"its classes change with its sizes: {ms[0][1].id[:120]} {got[k][1]}, {ms[1][1].id[:120]} {other}"
        except Exception as e:  # noqa: BLE001
            undecided[k] = f"its body doesn't compile: {type(e).__name__}: {str(e)[:160]}"
        finally:
            G.memo.clear()
            BD._BODIES.clear()
    stats = {"bindings": sum(len(ms) for ms in members.values()), "shapes": len(members), "built": len(callees),
             "reaching": len(reach), "openers": sorted(k for k in reach if members[k][0][1].base_id in OPENERS),
             "compared": compared, "build_s": round(built), "classify_s": round(time.monotonic() - t0 - built),
             "slowest_builds": sorted(slow, reverse=True)[:10], "unknown_where_evaluate_raised": sorted(G.raised)}
    return got, undecided, stats


def report(got: dict, undecided: dict, stats: dict) -> str:
    lines = [json.dumps(stats)]
    for _k, (fn, cs) in sorted(got.items()):
        worst = UNGATED if any(c == UNGATED for _, c in cs) else VERDICT if any(c == VERDICT for _, c in cs) else GATED
        lines.append(f"{worst:8} {fn.id[:220]}  [{', '.join(f'{k}:{c}' for k, c in cs)}]")
    lines += [f"UNDECIDED {k[:220]}: {v}" for k, v in sorted(undecided.items())]
    return "\n".join(lines)


def evaluated(fn: Any) -> dict:
    """``fn``'s entry of the record, from a walk and a three-valued run of that very binding; refused (ValueError) unless
    every output is gated or a verdict."""
    from verity.circuits import codec as C
    from verity.circuits.program import Program

    _BD, T, D, _BC = _imports()
    T.load_registries()
    members, callees, undecided, _slow = walk([fn])
    if undecided:
        raise ValueError(f"{fn.id}: undecided: {undecided}")
    cs = Gating(reaching(members, callees)).classify(fn)
    if any(c == UNGATED for _, c in cs):
        raise ValueError(f"{fn.id}: ungated outputs {cs}: not a record entry")
    params, ret = fn.signature()
    shell = D.CompositeDefinition(f"Show[{fn.id}]", 1, (), lambda S: (tuple(params), ret), lambda B, S, *a: B.call(fn, *a),
                                  register=False)
    key = C.descriptor_id(fn)
    return {"template": fn.base_id, "key": key, "closure_digest": C.closure_digest(C.encode_program(Program(shell)), key),
            "class": VERDICT if any(c == VERDICT for _, c in cs) else GATED, "outputs": [list(x) for x in cs],
            "evaluated": True}


def callers(desc: dict) -> list[str]:
    """The keys of ``desc``'s definitions that call an opener and that its root reaches through its nodes' callees and word
    primitives' Boolean definitions, sorted: the verifier's reading (`docket.ungated_openers` with no gated pair)."""
    defs, calls, boolean = desc["definitions"], desc["callees"], desc.get("boolean", {})
    seen, out, todo = set(), [], [desc["root"]]
    while todo:
        k = todo.pop()
        if k in seen:
            continue
        seen.add(k)
        e = defs[k]
        if e["id"] in OPENERS:
            continue
        kids = ([calls[n["fn"]] for n in e["body"]["nodes"]] if e["form"] == "composite"
                else [boolean[e["id"]]] if e["id"] in boolean else [])
        if any(defs[c]["id"] in OPENERS for c in kids):
            out.append(k)
        todo += kids
    return sorted(out)


def targets() -> dict[str, Any]:
    """Each GPU card's target (`target_profile.card_facts`), by its cards' names, one per compute capability and SM count."""
    from verity_vllm.config import CARD_FILES
    from verity_vllm.program.frontend.target_profile import TargetProfile, card_facts

    got: dict[tuple, list[str]] = {}
    for p in sorted(CARD_FILES.glob("*.toml")):
        try:
            cc, sms, _name = card_facts(p.stem)
        except ValueError:
            continue
        got.setdefault((cc, sms), []).append(p.stem)
    return {"+".join(names): TargetProfile(compute_capability=cc, num_sms=sms) for (cc, sms), names in got.items()}


def epoch(params: dict, target: Any) -> tuple[dict, dict[str, Any]]:
    """The description of an application's epoch circuit at ``target`` as the serve's close builds it
    (`window_circuit.served_lowering`), no weights read, every slot of its class taken by a request with the class's
    prompt and outputs, and its definitions by key; refused (`Unexpressible`, `UnregisteredTarget`) where the adapter can't
    lower it."""
    import verity_vllm.pipeline.advice_code.format as I1
    from verity.circuits import codec as C
    from verity_vllm.engine.served_seed import DEFAULT_SEED
    from verity_vllm.inputs import serves
    from verity_vllm.pipeline import advice as ADV
    from verity_vllm.program.predict import window_circuit as WC
    from verity_vllm.program.predict.phi import phi_of_serve
    from verity_vllm.program.predict.steps import Registered

    s = serves.served(params["model"], params["revision"], params["precision"], params["slots"], params["window-steps"])
    serve = {"model": s.model, "revision": s.revision, "execution": target.execution_label, "target": target.to_json()}
    reg = Registered.of(phi_of_serve(serve), DEFAULT_SEED, ADV.CAP_BITS)
    sampling = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": None, "max_tokens": s.outputs}
    steps = [[I1.Row(i, "prefill", 0, s.prompt) for i in range(s.slots)]]
    steps += [[I1.Row(i, "decode", s.prompt + j, 1) for i in range(s.slots)] for j in range(s.outputs - 1)]
    advice = I1.Advice(tuple(I1.RequestAdvice(f"q{i}", s.prompt, s.outputs, "length") for i in range(s.slots)),
                       tuple(I1.StepAdvice(tuple(st)) for st in steps), (0,) * (s.slots * s.outputs))
    ingress = {f"q{i}": {"prompt": [1] * s.prompt, "model": s.model, "user": f"q{i}", **sampling} for i in range(s.slots)}
    _compiled, low = WC.served_lowering(reg, WC.WindowClass(s.slots, s.prompt, s.outputs), advice, ingress)
    program, _query = low.circuit.program()
    return C.encode_program(program), C.definitions(program)


def applications() -> dict[str, dict]:
    """Each application's bindings of a template that calls an opener, by target its epoch circuit builds at, with each
    binding's Definition and closure digest in that circuit's description, and the targets that refuse it, with why."""
    import tomllib

    from verity.circuits import codec as C
    from verity_vllm.program.frontend.targets import UnregisteredTarget
    from verity_vllm.program.predict.common import Unexpressible

    _imports()[1].load_registries()
    out: dict[str, dict] = {}
    for path in sorted(APPLICATIONS.glob("*.toml")):
        params = tomllib.loads(path.read_text(encoding="utf-8"))["workload"]["params"]
        app: dict[str, dict] = {"built": {}, "refused": {}}
        for name, target in targets().items():
            try:
                desc, defs = epoch(params, target)
            except (Unexpressible, UnregisteredTarget) as e:
                app["refused"][name] = f"{type(e).__name__}: {e}"
                continue
            app["built"][name] = {k: (defs[k], C.closure_digest(desc, k)) for k in callers(desc)}
        out[path.relative_to(ROOT).as_posix()] = app
    return out


def bound(module: str, key: str) -> Any:
    """The binding ``key`` names of a template ``module`` registers, its statics integers; refused unless it binds as
    ``key``."""
    import importlib

    from verity.circuits.codec import descriptor_id

    D = _imports()[2]
    importlib.import_module(module)
    base, _, rest = key.partition("{")
    statics = dict(kv.split("=", 1) for kv in rest.rstrip("}").split(",")) if rest else {}
    fn = D.bind(D.REGISTRY[base], **{k: int(v) for k, v in statics.items()})
    if (got := descriptor_id(fn)) != key:
        raise ValueError(f"{key}: binds as {got}")
    return fn


def record() -> dict:
    apps = applications()
    used: dict[str, set[str]] = {}
    fns: dict[str, Any] = {}
    digests: dict[str, str] = {}
    for where, app in apps.items():
        for got in app["built"].values():
            for k, (fn, digest) in got.items():
                used.setdefault(k, set()).add(where)
                fns[k], digests[k] = fn, digest
    for module, keys in TESTED.items():
        for k in keys:
            used.setdefault(k, set()).add("tests")
            if k not in fns:
                fns[k] = bound(module, k)
    entries = []
    for k in sorted(used):
        e = evaluated(fns[k])
        if e["key"] != k or digests.get(k, e["closure_digest"]) != e["closure_digest"]:
            raise ValueError(f"{k}: evaluated as {e['key']} with closure digest {e['closure_digest']}, the circuit's {digests[k]}")
        entries.append({**e, "used_by": sorted(used[k])})
    return {"rule": "circuit templates that call an opener and gate on its verdict or return a verdict alone, at each binding "
                    "an application's epoch circuit or a test's uses, each evaluated at that binding "
                    "(tests/lint/test_repo_openers_gated.py --write)",
            "applications": {where: {"built": {t: sorted(got) for t, got in app["built"].items()}, "refused": app["refused"]}
                             for where, app in apps.items()},
            "entries": entries}


@pytest.mark.slow
def test_no_new_ungated_opener() -> None:
    got, undecided, stats = classes()
    print(report(got, undecided, stats))
    listed = allowed()
    ungated = {fn.base_id for fn, cs in got.values() if any(c == UNGATED for _, c in cs)}
    new = sorted(ungated - set(listed))
    assert not undecided, ("templates the walk can't decide whether they reach an opener or gate its verdict: "
                           + "; ".join(f"{k[:160]}: {v}" for k, v in sorted(undecided.items())))
    assert not new, (f"circuit templates whose returned rows don't gate on their opener's verdict: {new}. Gate each returned "
                     f"row on the verdict (`EmbeddingRead_v4`'s shape), or return the verdict alone")
    stale = sorted(set(listed) - ungated)
    if stale:
        warnings.warn(f"{LIST.name}: no longer ungated, delete their entries: {stale}", stacklevel=1)


def test_the_list_is_well_formed() -> None:
    for e in allowed().values():
        assert (ROOT / e["file"]).is_file(), f"{LIST.name}: {e['file']} isn't a file"
    for bad in ({"entries": [{"file": "a.py", "kind": KIND, "symbol": "X_v1"}]},
                {"entries": [{"file": "a.py", "kind": "gated", "symbol": "X_v1", "detail": "d"}]}):
        with pytest.raises(AssertionError):
            parse(bad)


@pytest.mark.slow
def test_a_gated_read_and_an_ungated_one() -> None:
    D = _imports()[2]
    from verity.commitments.gates import embedding as E
    v4, v2 = D.bind(E.EmbeddingRead_v4, V=2, H=1), D.bind(E.EmbeddingRead, V=2, H=1)
    got, undecided, stats = classes([v4, D.bind(E.EmbeddingRead, V=2, H=16), v2])
    assert not undecided and stats["compared"] == 1
    assert {fn.id: cs for fn, cs in got.values()} == {v4.id: [("0", GATED), ("1", GATED)], v2.id: [("0", UNGATED), ("1", VERDICT)]}


@pytest.mark.slow
def test_the_record_is_the_walks() -> None:
    assert json.loads(RECORD.read_text(encoding="utf-8")) == record(), (
        f"{RECORD.name} is not what evaluating each binding gives: python3 {Path(__file__).relative_to(ROOT)} --write")


if __name__ == "__main__" and sys.argv[1:] == ["--write"]:
    RECORD.write_text(json.dumps(record(), indent=2) + "\n", encoding="utf-8")
