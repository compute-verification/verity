"""The host plan of the call-boundary source (`call_boundary_source`): each `call_boundaries` identity of a required manifest as a target
of a module body's row template, and the template's evaluation.

The rows are each request's Program (the one request of a request manifest; `components[r]` of a global one): `instances.json.gz` of the
Build dir whose program digest is the request's (under `--program-dir` / `--programs-root`), grouped by the population the manifest was
built with, each identity checked against its Calls (rows, width, element range, dtype).  A row of module body M reads a Value of M's own
body at the same step (evaluated here), a literal (evaluated once), a weight field (the model's parameter or buffer of that name, wrapper
prefix stripped, in its own width), a Value of M's weight-only Calls of another step (a norm's `weight + 1` the Build made once: taken
into the body and evaluated from the weights, its words the same at every step) or a Value produced outside M.  The outside Values are M's forward arguments: the k-th distinct
(producer module, fx node, hint, port) M's rows at the step read, in Program order, is the k-th tensor among the arguments (positional,
then keyword), holding the producer rows read in ascending order, as [rows, port elements] words of the port's width.  A body that reads
anything else -- a Program input other than a weight, a Value of another step, a module under M, a slice across two ports or two fields --
leaves its identities uncovered, named.

A body's step-shared Calls -- those reading only weights, literals and each other (a norm's `weight + 1`) -- are evaluated once per step,
whatever its rows, and each of their targets is committed as one row (with several requests, the binding map's split of a member neither
per token nor per request: the whole row to each, whose step-shared Calls must be of one form).  The other Calls split into groups
connected through M's own Values or one producer row of an outside Value.  R groups of one form
(the same Definitions and statics, dataflow, literals and weight slices, group j reading block j of every outside Value's rows) are a row
template repeated R times -- a token's rows, or a sequence's; any other body is one group (R = 1).  Target t's Calls are f per group and it
is committed as [R, f * width].  At an engine step of several requests (the global manifest's `engine_steps` composition) the forward
arguments pack the rows of every request of the step: each request's template of M must commit the same targets from the same argument
shapes, and a target binds the way the binding map splits a packed member -- per token (every request's R = its token rows at the step) or
per request (every R = 1); any other body is uncovered.  Where the requests' templates are equal the packed rows are one batch of the
template (in the forward pre-hook); otherwise each request's rows are evaluated with its own template, split in the runtime's row order.
"""
from __future__ import annotations

import bisect
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

FAMILY = "call_boundaries"
#: the manifest's word dtype -> the numpy word it is evaluated as and the torch dtype it is committed as
WORD_DTYPES = {"bf16": (np.uint16, "bfloat16"), "f32": (np.uint32, "float32"), "w8": (np.uint8, "uint8"), "i64": (np.uint64, "int64")}
#: argument words per instance above which a Definition is its row kernel's alone: the reference holds an instance as Python ints (a
#: 256000 x 2304 logits weight is ~20 GB of them), so an instance the kernel declines is an error, and a Definition no row kernel claims
#: is uncovered
REFERENCE_LEAVES = 1 << 24


class Uncovered(Exception):
    """A body the rule cannot evaluate from M's forward arguments, named."""


@dataclass(frozen=True)
class Operand:
    """A Value produced outside M that M's rows read: one forward argument, `per_row` producer rows per template row, `elems` words of
    `width` bits each."""

    where: str = field(compare=False)      # `<producer module>/<fx node>[/<hint>]:<port>`, as the request's Program names it
    per_row: int
    elems: int
    width: int


@dataclass(frozen=True)
class Target:
    """One `call_boundaries` identity: per template row (once per step when `shared`), the Calls at positions `at` produce its rows, each
    the leaves [lo, hi)."""

    name: str                              # the committed tensor name `<op_path>/<member>`
    at: tuple[int, ...]
    lo: int
    hi: int
    dtype: str
    shared: bool = False


@dataclass(frozen=True)
class Template:
    """One row of a body: per position (Program order) its Definition and per argument group its slices -- ("v", position, lo, hi) a Value
    of the row, ("s", shared position, lo, hi) a step-shared Value, ("l", literal spec, lo, hi), ("o", operand, row of the block, lo, hi)
    a forward argument's words, ("w", field, lo, hi) a weight field's; `levels` = [(spec, positions)] per dependency level.  `shared_*`:
    the step's Calls that read only weights, literals and each other (a norm's `weight + 1`), evaluated once whatever the rows."""

    specs: tuple[str, ...]
    recipes: tuple[tuple[tuple[tuple, ...], ...], ...]
    levels: tuple[tuple[tuple[str, tuple[int, ...]], ...], ...]
    operands: tuple[Operand, ...]
    targets: tuple[Target, ...]
    shared_specs: tuple[str, ...] = ()
    shared_recipes: tuple[tuple[tuple[tuple, ...], ...], ...] = ()
    shared_levels: tuple[tuple[tuple[str, tuple[int, ...]], ...], ...] = ()


@dataclass(frozen=True)
class Part:
    """One request's rows of a body at an engine step: its template, `rows` times."""

    request: str
    template: Template
    rows: int

    def operand_rows(self, k: int) -> int:
        return self.template.operands[k].per_row * self.rows


@dataclass(frozen=True)
class Body:
    """What the source evaluates of module body M at one engine step: one part per request of the step, in declaration order."""

    module: str                            # M's implementation path
    engine_step: int
    parts: tuple[Part, ...]

    @property
    def packed(self) -> bool:
        """Every request's template is the first's: the packed rows are one batch of it."""
        return all(p.template == self.parts[0].template for p in self.parts)

    @property
    def targets(self) -> tuple[Target, ...]:
        return self.parts[0].template.targets

    @property
    def operands(self) -> tuple[Operand, ...]:
        return self.parts[0].template.operands


@dataclass
class Plan:
    program_dirs: dict[str, str] = field(default_factory=dict)             # request -> its Build dir
    bodies: dict[tuple[int, str], Body] = field(default_factory=dict)     # (engine step, module) -> body
    literals: dict[str, np.ndarray] = field(default_factory=dict)         # literal spec -> its leaves
    fields: dict[str, tuple[int, int]] = field(default_factory=dict)      # weight field (implementation name) -> (elements, width)
    definitions: dict[str, tuple[Any, int]] = field(default_factory=dict)
    kernel_only: set[str] = field(default_factory=set)                    # Definitions over REFERENCE_LEAVES words per instance
    composition: dict[int, dict[str, int]] = field(default_factory=dict)  # engine step -> {request: token rows}: a global manifest's
    order: dict[str, str] = field(default_factory=dict)                   # workload request id -> `r<i>` (the binding map's)
    by_index: dict[int, str] = field(default_factory=dict)               # workload index -> `r<i>` (a warm-up request stands in by index)
    identities: int = 0
    uncovered: list[str] = field(default_factory=list)
    bytes_per_token: int = 0
    tensors_per_step: int = 0

    @property
    def covered(self) -> int:
        return sum(len(b.parts) * len(b.targets) for b in self.bodies.values())


def build_dirs(*roots: str | None) -> dict[str, str]:
    """Program digest -> the Build dir (holding `instances.json.gz`) under `roots`, each a Build dir itself or a directory of them."""
    out: dict[str, str] = {}
    for root in roots:
        if not root:
            continue
        for p in sorted({os.path.join(root, "result.json"), *glob.glob(os.path.join(root, "**", "result.json"), recursive=True)}):
            d = os.path.dirname(p)
            if not (os.path.exists(p) and os.path.exists(os.path.join(d, "instances.json.gz"))):
                continue
            art = os.path.join(d, "artifact.json")
            try:
                got = (_load(art) if os.path.exists(art) else {}).get("program_digest") or (_load(p).get("program") or {}).get("digest")
            except (OSError, ValueError):
                continue
            if got:
                out.setdefault(str(got), d)
    return out


def request_programs(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Request -> the digest of its Program: a global manifest's `components`, else the manifest's one request."""
    comps = manifest.get("components")
    if isinstance(comps, Mapping) and comps:
        return {str(r): str((c or {}).get("program_digest") or "") for r, c in comps.items()}
    return {str((manifest.get("requests") or ["r0"])[0]): str(manifest.get("program_digest") or "")}


def _load(p: str) -> Any:
    with open(p) as f:
        return json.load(f)


def _label(r: Mapping[str, Any]) -> str:
    return f"{r.get('op_path')}/{r.get('output_member')} of {r.get('request_id')} @ engine step {r.get('engine_step')}"


# ---------------------------------------------------------------------------------------------------------------------------------- plan
def plan(manifest: Mapping[str, Any], ids: Sequence[Mapping[str, Any]], *roots: str | None) -> Plan:
    """Identities `ids` of `manifest` as targets of the bodies the source evaluates, or uncovered with the reason; the Programs are the
    Build dirs under `roots`."""
    p = Plan(identities=len(ids))
    requests, ranks, progs = list(manifest.get("requests") or []), int(manifest.get("ranks") or 1), request_programs(manifest)
    steps = manifest.get("engine_steps") or []
    if steps and isinstance(steps[0], Mapping):
        p.composition = {int(e["engine_step"]): {str(m["request_id"]): int(m["rows"]) for m in e.get("members") or ()} for e in steps}
    p.order = {str(d["workload_request_id"]): str(d["request_id"]) for d in manifest.get("requests_declared") or () if d.get("workload_request_id")}
    p.by_index = {int(d["index"]): str(d["request_id"]) for d in manifest.get("requests_declared") or () if d.get("index") is not None}
    why = (f"a manifest of {ranks} ranks (the source serves one rank)" if ranks > 1 else
           f"a manifest of {len(requests)} requests without per-request components and composition"
           if len(requests) > 1 and (len(progs) != len(requests) or not p.composition) else None)
    if why:
        p.uncovered = [f"{_label(r)}: {why}" for r in ids]
        return p
    dirs = build_dirs(*roots)
    mine: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in ids:
        rid = next(iter(progs)) if len(progs) == 1 else str(r.get("request_id"))
        if rid in progs:
            mine[rid].append(r)
        else:
            p.uncovered.append(f"{_label(r)}: the manifest declares no Program of request {rid}")
    parts: dict[tuple[int, str], dict[str, Part]] = defaultdict(dict)
    for rid, rs in mine.items():
        d = dirs.get(progs[rid])
        if d is None:
            p.uncovered.extend(f"{_label(r)}: no Build dir under {list(roots)} holds request {rid}'s Program {progs[rid][:16]}" for r in rs)
            continue
        p.program_dirs[rid] = d
        for key, part in _plan_request(p, manifest, rid, rs, d).items():
            parts[key][rid] = part
    for (e, m), got in sorted(parts.items()):
        try:
            p.bodies[(e, m)] = _merge(p, e, m, got)
        except Uncovered as u:
            p.uncovered.extend(f"{t.name} of {q.request} @ engine step {e}: {u}" for q in got.values() for t in q.template.targets)
    p.bytes_per_token, p.tensors_per_step = _arena_bound(manifest, p)
    return p


def _plan_request(p: Plan, manifest: Mapping[str, Any], rid: str, ids: Sequence[Mapping[str, Any]],
                  program_dir: str) -> dict[tuple[int, str], Part]:
    """Request `rid`'s identities as parts of its bodies, from its Program in `program_dir`."""
    from verity_vllm.correspondence.reader_for_query import Correspondence
    from verity_vllm.query import norm_scales as NS
    from verity_vllm.query.program_view import from_instances, iter_instance_rows
    from verity_vllm.query.required import Q_WORD_ID, population

    path = os.path.join(program_dir, "instances.json.gz")
    P = from_instances(path)
    res = _load(os.path.join(program_dir, "result.json"))
    corr = Correspondence.of(P, program_dir, implementation_paths=Correspondence.implementation_paths_of(res))
    pop = (NS.POLICY.population if NS.policy_of(manifest) else population)(P, corr, query=Q_WORD_ID)
    impl = corr.implementation_path
    groups: dict[tuple[int, str, str], tuple[Any, list[int]]] = {}
    for cid in pop.producers():
        for cm in pop.members_of(cid):
            if cm.family == FAMILY:
                groups.setdefault((pop.step_of(cid), impl(pop.program_path(cid)), cm.member), (P.call(cid).port(cm.port), []))[1].append(cid)
    wanted: dict[tuple[int, int, str], list[tuple[str, tuple[int, ...], int, int, str]]] = defaultdict(list)
    for r in ids:
        g = groups.get((int(r["step"]), str(r["op_path"]), str(r["output_member"])))
        why = _identity_fault(r, g)
        if why:
            p.uncovered.append(f"{_label(r)}: {why}")
            continue
        port, cids = g
        wanted[(int(r["engine_step"]), int(r["step"]), corr.module_of(cids[0]))].append(
            (f"{r['op_path']}/{r['output_member']}", tuple(cids), port.lo, port.hi, str(r["dtype"])))
    modules = {m for _e, _s, m in wanted}
    calls: dict[tuple[str, int], list[int]] = defaultdict(list)
    for c in P.calls:
        m = corr.module_of(c.id)
        if m in modules:
            calls[(m, pop.step_of(c.id))].append(c.id)
    keep = {c for cs in calls.values() for c in cs} | set(pop.literals)
    raw = {int(r["i"]): r for r in iter_instance_rows(path) if int(r["i"]) in keep}
    weights = next((i for i, q in enumerate(P.params) if q in pop.record_params), None)
    wports = sorted(P.params[weights].port_table, key=lambda q: q.lo) if weights is not None else []
    for q in wports:
        if p.fields.setdefault(impl(q.name), (q.elems, int(q.width or 0))) != (q.elems, int(q.width or 0)):
            p.uncovered.extend(f"{_label(r)}: weight field {impl(q.name)} is {q.elems} Value<{q.width}> in {rid}'s Program, "
                               f"{p.fields[impl(q.name)]} in another's" for r in ids)
            return {}
    ctx = _Context(P, corr, pop, raw, weights, wports, p)
    out: dict[tuple[int, str], Part] = {}
    for (e, s, m), targets in sorted(wanted.items()):
        try:
            out[(e, impl(m))] = ctx.part(rid, m, s, calls[(m, s)], targets)
        except Uncovered as u:
            p.uncovered.extend(f"{t[0]} of {rid} @ engine step {e}: {u}" for t in targets)
    return out


def _definition(spec: str) -> tuple[Any, list[Any]]:
    """The registered Definition a Call's spec names and its parameters."""
    from verity.ml.circuits.evaluation.reference import signature_of
    from verity_vllm.query.call_scope import specialization
    from verity_vllm.query.manifest.format import spec_family
    from verity_vllm.query.program_view import spec_statics

    d = specialization(spec_family(spec), spec_statics(spec))
    return d, signature_of(d)[0]


def _identity_fault(r: Mapping[str, Any], g: tuple[Any, list[int]] | None) -> str | None:
    if int(r.get("invocation") or 0) or r.get("program_op_path"):
        return f"invocation {r.get('invocation')} of a shared module ({r.get('program_op_path')}): one forward, several bodies"
    if g is None:
        return "no Call of the Program produces it under the population of record"
    port, cids = g
    rows, width = int(r.get("rows") or 0), int(r.get("width") or 0)
    if (rows, width) != (len(cids), port.elems) or [int(x) for x in r.get("element_range") or ()] != [0, rows * width]:
        return f"the identity is {rows} x {width} over {r.get('element_range')}, its Calls {len(cids)} x {port.elems}"
    if str(r.get("dtype")) not in WORD_DTYPES or np.dtype(WORD_DTYPES[str(r["dtype"])][0]).itemsize * 8 != int(port.width or 0):
        return f"dtype {r.get('dtype')} is not the port's Value<{port.width}> word"
    return None


def _merge(p: Plan, e: int, m: str, got: Mapping[str, Part]) -> Body:
    """The body of M at engine step `e` from each request's part: every request of the step (the declared composition) has one, with the
    same targets and argument shapes, rows per token or per request (the binding map's splits of a packed member)."""
    declared = p.composition.get(e)
    if declared is None:
        if p.composition or len(got) != 1:
            raise Uncovered(f"engine step {e} is not in the manifest's composition")
        return Body(m, e, tuple(got.values()))
    if set(got) != set(declared):
        raise Uncovered(f"{m} has Program rows of {sorted(got)} at the step; the step runs {sorted(declared)}")
    parts = tuple(got[r] for r in declared)

    def shape(t: Template) -> tuple:
        return tuple((x.name, len(x.at), x.hi - x.lo, x.dtype, x.shared) for x in t.targets), tuple((o.elems, o.width) for o in t.operands)

    if any(shape(q.template) != shape(parts[0].template) for q in parts):
        raise Uncovered(f"the requests of the step commit other targets of {m} or read other arguments (fx nodes of another local step?)")
    if any((q.template.shared_specs, q.template.shared_recipes) != (parts[0].template.shared_specs, parts[0].template.shared_recipes)
           for q in parts):
        raise Uncovered(f"{m}: the requests' step-shared Calls differ (other weights or literals): one committed row cannot be each's")
    if len(parts) > 1 and not (all(q.rows == declared[q.request] for q in parts) or all(q.rows == 1 for q in parts)):
        raise Uncovered(f"{m}: rows (request, rows, token rows) {[(q.request, q.rows, declared[q.request]) for q in parts]} are neither "
                        f"each request's token rows nor one per request")
    return Body(m, e, parts)


class _Context:
    """The Program facts `part` reads, once per request."""

    def __init__(self, P: Any, corr: Any, pop: Any, raw: dict[int, dict], weights: int | None, wports: list[Any], plan: Plan) -> None:
        self.P, self.corr, self.pop, self.raw, self.weights, self.wports, self.plan = P, corr, pop, raw, weights, wports, plan
        self.wlo = [q.lo for q in wports]

    def port_at(self, cid: int, lo: int, hi: int) -> Any:
        for q in self.P.call(cid).port_table:
            if q.lo <= lo and hi <= q.hi:
                return q
        raise Uncovered(f"a slice [{lo}, {hi}) of Call {cid} spans two ports")

    def field_at(self, lo: int, hi: int) -> Any:
        k = bisect.bisect_right(self.wlo, lo) - 1
        if k < 0 or hi > self.wports[k].hi:
            raise Uncovered(f"a weights slice [{lo}, {hi}) spans two fields")
        return self.wports[k]

    def definition(self, cid: int) -> str:
        r = self.raw[cid]
        spec = str(r["spec"])
        if str(r.get("form") or "call") != "call":
            raise Uncovered(f"Call {cid} ({spec}) is a {r.get('form')} row")
        if spec not in self.plan.definitions:
            from verity_vllm.program.kernels import kernel_registry, rows

            try:
                d, params = _definition(spec)
            except Exception as e:  # noqa: BLE001
                raise Uncovered(f"Call {cid}: {spec} does not resolve to a registered Definition ({e!r:.80})") from None
            leaves = sum(pt.leaves for pt in params)
            if leaves > REFERENCE_LEAVES:
                why = kernel_registry.covers(d, rows.KERNEL)
                if why:
                    raise Uncovered(f"Call {cid}: {spec} reads {leaves} words per instance, past the reference's bound, and {why}")
                self.plan.kernel_only.add(spec)
            self.plan.definitions[spec] = (d, len(params))
        if self.plan.definitions[spec][1] != len(r.get("args") or ()):
            raise Uncovered(f"Call {cid}: {spec} takes {self.plan.definitions[spec][1]} arguments, the row gives {len(r.get('args') or ())}")
        return spec

    def literal(self, cid: int) -> str:
        from verity.ml.circuits.evaluation import evaluate

        spec = self.definition(cid)
        if spec not in self.plan.literals:
            self.plan.literals[spec] = np.asarray(evaluate(self.plan.definitions[spec][0]), dtype=np.uint64)
        return spec

    def weight_only(self, a: int, m: str, memo: dict[int, bool]) -> bool:
        """Call `a` is of body M and reads only weight fields, literals and other such Calls: its words are the same at every step."""
        if a not in memo:
            memo[a] = False
            r = self.raw.get(a)
            memo[a] = r is not None and self.corr.module_of(a) == m and str(r.get("form") or "call") == "call" and all(
                (kind == "p" and int(x) == self.weights) or int(x) in self.pop.literals or (kind == "n" and self.weight_only(int(x), m, memo))
                for g in r.get("args") or () for kind, x, _lo, _hi in g)
        return memo[a]

    def part(self, rid: str, m: str, s: int, calls: list[int], targets: list[tuple]) -> Part:
        pop, corr, lits = self.pop, self.corr, self.pop.literals
        inside, memo, stack = set(calls), {}, list(calls)
        while stack:                                   # M's weight-only Calls of another step (a norm's `weight + 1`, made once): its own
            for g in self.raw[stack.pop()].get("args") or ():
                for kind, a, _lo, _hi in g:
                    a = int(a)
                    if kind == "n" and a not in inside and a not in lits and pop.step_of(a) != s and self.weight_only(a, m, memo):
                        inside.add(a)
                        stack.append(a)
        keys: dict[tuple[str, str, str, str], tuple[Any, set[int]]] = {}
        for c in sorted(inside):
            for g in self.raw[c].get("args") or ():
                for kind, a, lo, hi in g:
                    a = int(a)
                    if kind == "p":
                        if a != self.weights:
                            raise Uncovered(f"Call {c} of {corr.implementation_path(m)} reads Program input {self.P.params[a].name!r}")
                        self.field_at(int(lo), int(hi))
                    elif a not in inside and a not in lits:
                        pm, n = corr.module_of(a), corr.name(a)
                        if pop.step_of(a) != s:
                            raise Uncovered(f"Call {c} reads Call {a} of step {pop.step_of(a)} (under {corr.implementation_path(pm)})")
                        if pm[: len(m) + 1] == m + ".":
                            raise Uncovered(f"Call {c} reads a Value of {corr.implementation_path(pm)}, a module under the body's")
                        q = self.port_at(a, int(lo), int(hi))
                        keys.setdefault((pm, n.node, n.hint, q.name), (q, set()))[1].add(a)
        operands, feeds = [], {}
        for (pm, node, hint, pname), (q, rows) in keys.items():
            feeds.update({(a, pname): (len(operands), i) for i, a in enumerate(sorted(rows))})
            operands.append((f"{corr.implementation_path(pm)}/{node}{'/' + hint if hint else ''}:{pname}", len(rows), q.elems, int(q.width or 0)))
        need: set[int] = set()
        stack = [c for t in targets for c in t[1]]
        while stack:
            c = stack.pop()
            if c in need:
                continue
            need.add(c)
            stack.extend(int(a) for g in self.raw[c].get("args") or () for kind, a, _lo, _hi in g
                         if kind == "n" and int(a) in inside and int(a) not in lits)
        order = sorted(need)
        recipes = {c: self.recipe(c, inside, feeds) for c in order}
        specs = {c: self.definition(c) for c in order}
        tpl, rows = _template(order, recipes, specs, operands, targets)
        return Part(rid, tpl, rows)

    def recipe(self, c: int, inside: set[int], feeds: dict[tuple[int, str], tuple[int, int]]) -> list[list[tuple]]:
        """Per argument group of Call `c`, its slices: ("v", Call, lo, hi) a Value of the body, ("l", spec, lo, hi) a literal, ("o",
        operand, row, lo, hi) a forward argument's words, ("w", field, lo, hi) a weight field's."""
        out = []
        for g in self.raw[c].get("args") or ():
            sl = []
            for kind, a, lo, hi in g:
                a, lo, hi = int(a), int(lo), int(hi)
                if kind == "p":
                    q = self.field_at(lo, hi)
                    sl.append(("w", self.corr.implementation_path(q.name), lo - q.lo, hi - q.lo))
                elif a in self.pop.literals:
                    sl.append(("l", self.literal(a), lo, hi))
                elif a in inside:
                    sl.append(("v", a, lo, hi))
                else:
                    q = self.port_at(a, lo, hi)
                    k, i = feeds[(a, q.name)]
                    sl.append(("o", k, i, lo - q.lo, hi - q.lo))
            out.append(sl)
        return out


# ------------------------------------------------------------------------------------------------------------------------------ template
def _template(order: list[int], recipes: dict[int, list[list[tuple]]], specs: dict[int, str], operands: list[tuple],
              targets: list[tuple]) -> tuple[Template, int]:
    """The body's row template and its rows R (module docstring): the step-shared Calls once, then R groups of one form, else the rest of
    the body as one group."""
    shared = _step_shared(order, recipes, targets)
    spos = {c: i for i, c in enumerate(shared)}
    own = [c for c in order if c not in spos]
    rec = {c: [[("s", spos[x[1]], *x[2:]) if x[0] == "v" and x[1] in spos else x for x in g] for g in recipes[c]] for c in own}
    sat = {t[0]: tuple(spos[c] for c in t[1]) for t in targets if t[1][0] in spos}
    rtargets = [t for t in targets if t[0] not in sat]
    root = {c: c for c in own}

    def find(x: int) -> int:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    first: dict[tuple[int, int], int] = {}
    for c in own:
        for g in rec[c]:
            for x in g:
                o = x[1] if x[0] == "v" else first.setdefault((x[1], x[2]), c) if x[0] == "o" else None
                if o is not None:
                    root[find(c)] = find(o)
    groups: dict[int, list[int]] = defaultdict(list)
    for c in own:
        groups[find(c)].append(c)
    rows = sorted(groups.values(), key=lambda cs: cs[0])
    got = _row_template(rows, rec, specs, operands, rtargets) if len(rows) > 1 else None
    R = len(rows) if got is not None else 1
    if got is None:
        pos = {c: i for i, c in enumerate(own)}
        whole = _form(own, 0, [n for _w, n, _e, _b in operands], rec, specs)
        assert whole is not None
        got = whole, [n for _w, n, _e, _b in operands], [tuple(pos[c] for c in t[1]) for t in rtargets]
    form, per, ats = got
    sform = _form(shared, 0, [], recipes, specs)
    assert sform is not None
    return _assemble(form, sform, operands, per, {**dict(zip((t[0] for t in rtargets), ats)), **sat}, set(sat), targets), R


def _step_shared(order: list[int], recipes: dict[int, list[list[tuple]]], targets: list[tuple]) -> list[int]:
    """The Calls of `order` that read only weights, literals and each other; none when a target has both such Calls and others."""
    shared: set[int] = set()
    for c in order:
        if all(x[0] in ("l", "w") or (x[0] == "v" and x[1] in shared) for g in recipes[c] for x in g):
            shared.add(c)
    if any(len({c in shared for c in t[1]}) > 1 for t in targets):
        return []
    return [c for c in order if c in shared]


def _row_template(rows: list[list[int]], recipes: dict[int, list[list[tuple]]], specs: dict[int, str], operands: list[tuple],
                  targets: list[tuple]) -> tuple[tuple, list[int], list[tuple[int, ...]]] | None:
    """(form, operand rows per template row, target positions) of R = len(rows) groups of one form, or None."""
    R = len(rows)
    if any(n % R for _w, n, _e, _b in operands) or any(len(t[1]) % R for t in targets):
        return None
    per = [n // R for _w, n, _e, _b in operands]
    form, ats = None, None
    for j, cs in enumerate(rows):
        f = _form(cs, j, per, recipes, specs)
        pos = {c: i for i, c in enumerate(cs)}
        at = [tuple(pos.get(c, -1) for c in t[1][j * (len(t[1]) // R):(j + 1) * (len(t[1]) // R)]) for t in targets]
        if f is None or (form is not None and (f != form or at != ats)) or any(-1 in a for a in at):
            return None
        form, ats = f, at
    return form, per, ats


def _form(cs: list[int], j: int, per: list[int], recipes: dict[int, list[list[tuple]]], specs: dict[int, str]) -> tuple | None:
    """Group j's Calls `cs` canonical: positions for its own Values, operand rows relative to block j (None: a read outside it)."""
    pos = {c: i for i, c in enumerate(cs)}
    rec = []
    for c in cs:
        gs = []
        for g in recipes[c]:
            sl = []
            for x in g:
                if x[0] == "v":
                    sl.append(("v", pos[x[1]], x[2], x[3]))
                elif x[0] == "o":
                    i = x[2] - j * per[x[1]]
                    if not 0 <= i < per[x[1]]:
                        return None
                    sl.append(("o", x[1], i, x[3], x[4]))
                else:
                    sl.append(tuple(x))
            gs.append(tuple(sl))
        rec.append(tuple(gs))
    return tuple(specs[c] for c in cs), tuple(rec)


def _levels(form: tuple) -> tuple[tuple[tuple[str, tuple[int, ...]], ...], ...]:
    specs, recipes = form
    lv: list[int] = []
    for rec in recipes:
        lv.append(1 + max((lv[x[1]] for g in rec for x in g if x[0] == "v"), default=-1))
    levels: list[dict[str, list[int]]] = [defaultdict(list) for _ in range(1 + max(lv, default=-1))]
    for i, n in enumerate(lv):
        levels[n][specs[i]].append(i)
    return tuple(tuple((s, tuple(ps)) for s, ps in d.items()) for d in levels)


def _assemble(form: tuple, sform: tuple, operands: list[tuple], per: list[int], at: Mapping[str, tuple[int, ...]], shared: set[str],
              targets: list[tuple]) -> Template:
    return Template(*form, _levels(form), tuple(Operand(w, k, e, b) for (w, _n, e, b), k in zip(operands, per)),
                    tuple(Target(t[0], at[t[0]], t[2], t[3], t[4], t[0] in shared) for t in targets), *sform, _levels(sform))


def _arena_bound(manifest: Mapping[str, Any], plan: Plan) -> tuple[int, int]:
    """(bytes per token row, tensors per step): the most any engine step commits per token row (its `input_token_ids` rows, over the
    step's requests), and the most targets of one step."""
    tokens: dict[int, int] = defaultdict(int)
    for r in (manifest or {}).get("identities") or []:
        if r.get("family") == "input_token_ids" and "engine_step" in r:
            tokens[int(r["engine_step"])] += int(r.get("rows") or 0)
    nbytes: dict[int, int] = defaultdict(int)
    count: dict[int, int] = defaultdict(int)
    for (e, _m), b in plan.bodies.items():
        rows = sum(q.rows for q in b.parts)
        for t in b.targets:
            nbytes[e] += (1 if t.shared else rows) * len(t.at) * (t.hi - t.lo) * np.dtype(WORD_DTYPES[t.dtype][0]).itemsize
            count[e] += 1
    return max((-(-n // max(1, tokens.get(e, 1))) for e, n in nbytes.items()), default=0), max(count.values(), default=0)


# ------------------------------------------------------------------------------------------------------------------------------- record
def to_json(p: Plan) -> dict[str, Any]:
    """`p` as JSON: each distinct template once (the bodies' parts index them), the literals' words as little-endian u64 in base64, each
    Definition by its spec and arity (`of_json` resolves it again)."""
    import base64

    index: dict[tuple, int] = {}
    templates: list[dict[str, Any]] = []

    def tpl(t: Template) -> int:
        k = (t, tuple(o.where for o in t.operands))
        if k not in index:
            index[k] = len(templates)
            templates.append({"specs": t.specs, "recipes": t.recipes, "levels": t.levels,
                              "operands": [[o.where, o.per_row, o.elems, o.width] for o in t.operands],
                              "targets": [[x.name, x.at, x.lo, x.hi, x.dtype, x.shared] for x in t.targets],
                              "shared_specs": t.shared_specs, "shared_recipes": t.shared_recipes, "shared_levels": t.shared_levels})
        return index[k]

    bodies = [[e, m, [[q.request, tpl(q.template), q.rows] for q in b.parts]] for (e, m), b in p.bodies.items()]
    return {"program_dirs": p.program_dirs, "templates": templates, "bodies": bodies,
            "literals": {s: [list(a.shape), base64.b64encode(np.ascontiguousarray(a, dtype="<u8").tobytes()).decode()] for s, a in p.literals.items()},
            "fields": {n: list(v) for n, v in p.fields.items()}, "definitions": {s: n for s, (_d, n) in p.definitions.items()},
            "kernel_only": sorted(p.kernel_only), "composition": [[e, c] for e, c in p.composition.items()], "order": p.order,
            "by_index": [[i, r] for i, r in p.by_index.items()], "identities": p.identities, "uncovered": p.uncovered,
            "bytes_per_token": p.bytes_per_token, "tensors_per_step": p.tensors_per_step}


def _tuples(x: Any) -> Any:
    return tuple(_tuples(y) for y in x) if isinstance(x, list) else x


def of_json(doc: Mapping[str, Any]) -> Plan:
    """The Plan `to_json` wrote; each Definition resolved from its spec (ValueError when one resolves to another arity)."""
    import base64

    templates = [Template(_tuples(t["specs"]), _tuples(t["recipes"]), _tuples(t["levels"]),
                          tuple(Operand(str(w), int(k), int(e), int(b)) for w, k, e, b in t["operands"]),
                          tuple(Target(str(n), _tuples(at), int(lo), int(hi), str(dt), bool(sh)) for n, at, lo, hi, dt, sh in t["targets"]),
                          _tuples(t["shared_specs"]), _tuples(t["shared_recipes"]), _tuples(t["shared_levels"])) for t in doc["templates"]]
    definitions: dict[str, tuple[Any, int]] = {}
    for spec, n in doc["definitions"].items():
        d, params = _definition(spec)
        if len(params) != int(n):
            raise ValueError(f"call-boundary plan: {spec} takes {len(params)} arguments here, {n} in the plan")
        definitions[spec] = (d, int(n))
    return Plan(program_dirs={str(r): str(d) for r, d in doc["program_dirs"].items()},
                bodies={(int(e), str(m)): Body(str(m), int(e), tuple(Part(str(r), templates[int(i)], int(n)) for r, i, n in parts))
                        for e, m, parts in doc["bodies"]},
                literals={s: np.frombuffer(base64.b64decode(b), dtype="<u8").astype(np.uint64).reshape(shape)
                          for s, (shape, b) in doc["literals"].items()},
                fields={str(n): (int(e), int(w)) for n, (e, w) in doc["fields"].items()}, definitions=definitions,
                kernel_only=set(doc["kernel_only"]), composition={int(e): {str(r): int(n) for r, n in c.items()} for e, c in doc["composition"]},
                order={str(k): str(v) for k, v in doc["order"].items()}, by_index={int(i): str(r) for i, r in doc["by_index"]},
                identities=int(doc["identities"]), uncovered=[str(u) for u in doc["uncovered"]], bytes_per_token=int(doc["bytes_per_token"]),
                tensors_per_step=int(doc["tensors_per_step"]))


# ---------------------------------------------------------------------------------------------------------------------------- evaluation
def evaluate_template(plan: Plan, tpl: Template, R: int, operands: Sequence[np.ndarray], weight: Callable[[str], np.ndarray],
                      shared: dict | None = None) -> dict[str, np.ndarray]:
    """Target name -> its [R, f * width] words ([1, f * width] for a step-shared target): `tpl` over R rows, level by level, one
    `evaluate_batch` per (level, Definition) over every row (a registered row kernel of `verity_vllm.program.kernels.rows` where one is,
    the reference for the rest and for what a kernel declines; a `plan.kernel_only` Definition by its row kernel alone, a declined
    instance a ValueError).  `operands[k]` = forward argument k's [R * per_row, elems] words; `weight(name)` = a weight field's words, read
    as a broadcast view (never copied per row); `shared` = the step-shared Values by form, kept across steps by the caller (weights and
    literals do not change within a Commit)."""
    ops = [np.asarray(o).reshape(R, tpl.operands[k].per_row, -1) for k, o in enumerate(operands)]
    key = (tpl.shared_specs, tpl.shared_recipes)
    sv = (shared or {}).get(key)
    if sv is None:
        sv = _run(plan, tpl.shared_levels, tpl.shared_recipes, 1, {}, ops, weight)
        if shared is not None:
            shared[key] = sv
    vals = _run(plan, tpl.levels, tpl.recipes, R, sv, ops, weight)
    return {t.name: np.concatenate([(sv if t.shared else vals)[i][:, t.lo:t.hi] for i in t.at], axis=1).astype(WORD_DTYPES[t.dtype][0])
            for t in tpl.targets}


def _run(plan: Plan, levels: tuple, recipes: tuple, R: int, sv: Mapping[int, np.ndarray], ops: Sequence[np.ndarray],
         weight: Callable[[str], np.ndarray]) -> dict[int, np.ndarray]:
    """Position -> [R, leaves] of the Calls of `recipes` over R rows; `sv` the step-shared Values ("s") they read."""
    from verity.ml.circuits.evaluation import evaluate_batch
    from verity_vllm.program.kernels import rows  # noqa: F401  -- registers the row kernels evaluate_batch picks

    vals: dict[int, np.ndarray] = {}

    def part(x: tuple) -> np.ndarray:
        if x[0] == "v":
            return vals[x[1]][:, x[2]:x[3]]
        if x[0] == "o":
            return ops[x[1]][:, x[2], x[3]:x[4]]
        src = sv[x[1]][0] if x[0] == "s" else plan.literals[x[1]] if x[0] == "l" else weight(x[1])
        return np.broadcast_to(src[x[2]:x[3]], (R, x[3] - x[2]))

    def arg(g: tuple) -> np.ndarray:
        parts = [part(x) for x in g]
        return parts[0] if len(parts) == 1 else np.concatenate([q.astype(np.uint64) for q in parts], axis=1)

    for level in levels:
        for spec, poss in level:
            groups = list(zip(*(recipes[i] for i in poss)))
            arrays = ([arg(g[0]) for g in groups] if len(poss) == 1 else
                      [np.stack([arg(x) for x in g], axis=1).reshape(R * len(poss), -1) for g in groups])
            d = plan.definitions[spec][0]
            out = (_kernel_rows(d, spec, arrays) if spec in plan.kernel_only else evaluate_batch(d, *arrays)).reshape(R, len(poss), -1)
            for n, i in enumerate(poss):
                vals[i] = out[:, n]
    return vals


def _kernel_rows(d: Any, spec: str, arrays: Sequence[np.ndarray]) -> np.ndarray:
    """`d` on every row of `arrays` by its row kernel alone (its batch form where it has one: `kernel_registry.register`); an
    instance the kernel declines is a ValueError naming it."""
    from verity.ml.circuits.evaluation import kernels
    from verity.ml.circuits.evaluation.batch import type_dtype, type_shape
    from verity.ml.circuits.evaluation.reference import signature_of, statics_of
    from verity_vllm.program.kernels.kernel_registry import Declined, instance
    from verity_vllm.program.kernels.rows import KERNEL

    params, ret = signature_of(d)
    statics = statics_of(d)
    args = [np.asarray(a).reshape(len(a), *type_shape(pt)).astype(type_dtype(pt), copy=False) for a, pt in zip(arrays, params)]
    n = len(args[0])
    got = kernels(d)[KERNEL].fn(*args, **statics)
    vals, declined = (None, np.ones(n, dtype=bool)) if got is NotImplemented else got
    if declined is not None and np.any(declined):
        i = int(np.flatnonzero(declined)[0])
        try:
            instance(d, KERNEL)(*(a[i] for a in args), **statics)
            why = "declined by its batch form"
        except (Declined, ValueError) as e:
            why = str(e)
        raise ValueError(f"call boundaries: the row kernel of {spec} declines an instance ({why}); at "
                         f"{sum(pt.leaves for pt in params)} words per instance the reference is not run") from None
    return np.asarray(vals).reshape(n, -1).astype(type_dtype(ret))


__all__ = ["FAMILY", "WORD_DTYPES", "Uncovered", "Operand", "Target", "Template", "Part", "Body", "Plan", "build_dirs", "request_programs",
           "plan", "to_json", "of_json", "evaluate_template"]
