"""Call boundaries (`call_boundaries`, Q_word: a Value two Calls of one module body exchange) as an acquisition source, evaluated on the
host.  Per module body M and committed step, a forward pre-hook reads M's inputs, evaluates M's rows in Program order with the IR's
evaluator (`verity.evaluation.evaluate_batch`: a registered row kernel of `verity_vllm.program.kernels.rows` where one is, the reference
for the rest and for what a kernel declines -- exact, NaN payloads included) and commits every required interior Value as `<M>/<fx node>/<member>`, [rows, width] words
over the identity's element range.  Attached whenever the required manifest names `call_boundaries` identities; a manifest naming none
refuses it (exit 3).  No kernel changes: the served kernels run unchanged, and the Calls that consume these Values check the served
outputs (the replay evaluates them from the words committed here).

The rows are the request Program's: `instances.json.gz` of the Build dir whose program digest is the manifest's (`--program-dir`, else
the one under `--programs-root`), grouped by the population the manifest was built with, each identity checked against its Calls (rows,
width, element range, dtype).  A row of M reads a Value of M's own body at the same step (evaluated here), a literal (evaluated once), a
weight field (the model's parameter or buffer of that name, wrapper prefix stripped, in its own width) or a Value produced outside M.
The outside Values are M's forward arguments: the k-th distinct (producer module, fx node, hint, port) M's rows at the step read, in
Program order, is the k-th tensor among the arguments (positional, then keyword), holding the producer rows read in ascending order, as
[rows, port elements] words of the port's width.  An argument of another size, row width or word width is a collector error naming the
site.  A body that reads anything else -- a Program input other than a weight, a Value of another step, a module under M, a slice
across two ports or two fields -- leaves its identities uncovered and the Commit is refused naming them, as is a manifest of several
requests or ranks (the forward arguments then batch other rows than the Program's).
"""
from __future__ import annotations

import bisect
import glob
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

FAMILY = "call_boundaries"
#: the manifest's word dtype -> the numpy word it is evaluated as and the torch dtype it is committed as
WORD_DTYPES = {"bf16": (np.uint16, "bfloat16"), "f32": (np.uint32, "float32"), "w8": (np.uint8, "uint8"), "i64": (np.uint64, "int64")}
_SIGNED = {1: "uint8", 2: "int16", 4: "int32", 8: "int64"}


class Uncovered(Exception):
    """A body the rule cannot evaluate from M's forward arguments, named."""


@dataclass(frozen=True)
class Target:
    """One `call_boundaries` identity: `calls` produce its rows in order, each the leaves [lo, hi) of its output."""

    name: str                       # the committed tensor name `<op_path>/<member>`
    calls: tuple[int, ...]
    lo: int
    hi: int
    dtype: str


@dataclass(frozen=True)
class Operand:
    """A Value produced outside M that M's rows read: one forward argument, [rows, elems] words of `width` bits."""

    where: str                      # `<producer module>/<fx node>[/<hint>]:<port>`
    rows: tuple[int, ...]           # the producer Calls, ascending
    lo: int                         # the port's first leaf in the producer's output
    elems: int
    width: int


@dataclass
class Body:
    """What the source evaluates of module body M at one engine step: `levels` = [(spec, Calls)] per dependency level, Program order."""

    module: str                     # M's implementation path
    engine_step: int
    operands: tuple[Operand, ...]
    levels: list[list[tuple[str, tuple[int, ...]]]]
    targets: tuple[Target, ...]


@dataclass
class Plan:
    program_dir: str
    bodies: dict[tuple[int, str], Body] = field(default_factory=dict)     # (engine step, module) -> body
    recipes: dict[int, list[list[tuple]]] = field(default_factory=dict)   # Call -> per argument group its slices, resolved
    specs: dict[int, str] = field(default_factory=dict)
    literals: dict[int, np.ndarray] = field(default_factory=dict)
    fields: list[tuple[str, int, int]] = field(default_factory=list)       # weight fields: (implementation name, elements, width)
    definitions: dict[str, Any] = field(default_factory=dict)
    identities: int = 0
    uncovered: list[str] = field(default_factory=list)
    bytes_per_token: int = 0
    tensors_per_step: int = 0

    @property
    def covered(self) -> int:
        return sum(len(b.targets) for b in self.bodies.values())


#: other sources' claims on `call_boundaries` identities (identity -> the claiming source's name, or None): a claimed identity is that
#: source's to acquire (a tap of a Value the runtime already holds, like a pre-bias GEMM output) and is never evaluated here
CLAIMS: list[Callable[[Mapping[str, Any]], str | None]] = []


def claimed(r: Mapping[str, Any]) -> str | None:
    """The source that claims identity `r` (`CLAIMS`), or None: this source evaluates it."""
    return next((c for f in CLAIMS if (c := f(r))), None)


def manifest_sites(manifest: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """The `call_boundaries` identities of a manifest this source evaluates: every one no other source claims."""
    return [r for r in (manifest or {}).get("identities") or [] if r.get("family") == FAMILY and claimed(r) is None]


def program_dir_of(manifest: Mapping[str, Any], *roots: str | None) -> str | None:
    """The Build dir under `roots` (a Build dir itself, or a directory of them) whose Program is the manifest's (`program_digest`)."""
    want = str(manifest.get("program_digest") or "")
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
            if want and got == want:
                return d
    return None


def _load(p: str) -> Any:
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------------------------------- plan
def plan_of(manifest: Mapping[str, Any], program_dir: str) -> Plan:
    """Every `call_boundaries` identity of `manifest` as a target of a body the source evaluates, or uncovered with the reason."""
    from verity_vllm.correspondence.reader_for_query import Correspondence
    from verity_vllm.query import norm_scales as NS
    from verity_vllm.query.program_view import from_instances, iter_instance_rows
    from verity_vllm.query.required import Q_WORD_ID, population

    ids = manifest_sites(manifest)
    plan = Plan(program_dir, identities=len(ids))
    requests, ranks = list(manifest.get("requests") or []), int(manifest.get("ranks") or 1)
    if len(requests) > 1 or ranks > 1:
        plan.uncovered = [f"{_label(r)}: a manifest of {len(requests)} requests on {ranks} ranks (the source serves one request on one rank)"
                          for r in ids]
        return plan
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
    wanted: dict[tuple[int, int, str], list[Target]] = defaultdict(list)
    for r in ids:
        g = groups.get((int(r["step"]), str(r["op_path"]), str(r["output_member"])))
        why = _identity_fault(r, g)
        if why:
            plan.uncovered.append(f"{_label(r)}: {why}")
            continue
        port, cids = g
        wanted[(int(r["engine_step"]), int(r["step"]), corr.module_of(cids[0]))].append(
            Target(f"{r['op_path']}/{r['output_member']}", tuple(cids), port.lo, port.hi, str(r["dtype"])))
    modules = {m for _e, _s, m in wanted}
    calls: dict[tuple[str, int], list[int]] = defaultdict(list)
    for c in P.calls:
        m = corr.module_of(c.id)
        if m in modules:
            calls[(m, pop.step_of(c.id))].append(c.id)
    keep = {c for cs in calls.values() for c in cs} | set(pop.literals)
    raw = {int(r["i"]): r for r in iter_instance_rows(path) if int(r["i"]) in keep}
    weights = next((i for i, p in enumerate(P.params) if p in pop.record_params), None)
    wports = sorted(P.params[weights].port_table, key=lambda q: q.lo) if weights is not None else []
    plan.fields = [(impl(q.name), q.elems, int(q.width or 0)) for q in wports]
    ctx = _Context(P, corr, pop, raw, weights, [q.lo for q in wports], wports, plan)
    for (e, s, m), targets in sorted(wanted.items()):
        try:
            plan.bodies[(e, impl(m))] = ctx.body(m, s, e, calls[(m, s)], targets)
        except Uncovered as u:
            plan.uncovered.extend(f"{t.name} @ engine step {e}: {u}" for t in targets)
    plan.bytes_per_token, plan.tensors_per_step = _arena_bound(manifest, plan)
    return plan


def _label(r: Mapping[str, Any]) -> str:
    return f"{r.get('op_path')}/{r.get('output_member')} @ engine step {r.get('engine_step')}"


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


class _Context:
    """The Program facts `body` reads, once per plan."""

    def __init__(self, P: Any, corr: Any, pop: Any, raw: dict[int, dict], weights: int | None, wlo: list[int], wports: list[Any],
                 plan: Plan) -> None:
        self.P, self.corr, self.pop, self.raw, self.weights, self.wlo, self.wports, self.plan = P, corr, pop, raw, weights, wlo, wports, plan

    def port_at(self, cid: int, lo: int, hi: int) -> Any:
        for q in self.P.call(cid).port_table:
            if q.lo <= lo and hi <= q.hi:
                return q
        raise Uncovered(f"a slice [{lo}, {hi}) of Call {cid} spans two ports")

    def field_at(self, lo: int, hi: int) -> int:
        k = bisect.bisect_right(self.wlo, lo) - 1
        if k < 0 or hi > self.wports[k].hi:
            raise Uncovered(f"a weights slice [{lo}, {hi}) spans two fields")
        return k

    def definition(self, cid: int) -> str:
        from verity_vllm.query.call_scope import specialization
        from verity_vllm.query.manifest.format import spec_family
        from verity_vllm.query.program_view import spec_statics

        r = self.raw[cid]
        spec = str(r["spec"])
        if str(r.get("form") or "call") != "call":
            raise Uncovered(f"Call {cid} ({spec}) is a {r.get('form')} row")
        if spec not in self.plan.definitions:
            try:
                d = specialization(spec_family(spec), spec_statics(spec))
                n = len(d.signature()[0])
            except Exception as e:  # noqa: BLE001
                raise Uncovered(f"Call {cid}: {spec} does not resolve to a registered Definition ({e!r:.80})") from None
            self.plan.definitions[spec] = (d, n)
        if self.plan.definitions[spec][1] != len(r.get("args") or ()):
            raise Uncovered(f"Call {cid}: {spec} takes {self.plan.definitions[spec][1]} arguments, the row gives {len(r.get('args') or ())}")
        self.plan.specs[cid] = spec
        return spec

    def literal(self, cid: int) -> None:
        from verity.evaluation import evaluate

        if cid not in self.plan.literals:
            self.definition(cid)
            self.plan.literals[cid] = np.asarray(evaluate(self.plan.definitions[self.plan.specs[cid]][0]), dtype=np.uint64)

    def body(self, m: str, s: int, e: int, calls: list[int], targets: list[Target]) -> Body:
        pop, corr, lits = self.pop, self.corr, self.pop.literals
        inside = set(calls)
        keys: dict[tuple[str, str, str, str], tuple[Any, set[int]]] = {}
        for c in calls:
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
            k = len(operands)
            operands.append(Operand(f"{corr.implementation_path(pm)}/{node}{'/' + hint if hint else ''}:{pname}", tuple(sorted(rows)),
                                    q.lo, q.elems, int(q.width or 0)))
            feeds.update({(a, pname): (k, i) for i, a in enumerate(operands[-1].rows)})
        need: set[int] = set()
        stack = [c for t in targets for c in t.calls]
        while stack:
            c = stack.pop()
            if c in need:
                continue
            need.add(c)
            stack.extend(int(a) for g in self.raw[c].get("args") or () for kind, a, _lo, _hi in g
                         if kind == "n" and int(a) in inside and int(a) not in lits)
        level: dict[int, int] = {}
        for c in sorted(need):
            self.plan.recipes[c] = self.recipe(c, inside, feeds)
            level[c] = 1 + max((level[x[1]] for g in self.plan.recipes[c] for x in g if x[0] == "v" and x[1] in level), default=-1)
            self.definition(c)
        levels: list[dict[str, list[int]]] = [defaultdict(list) for _ in range(1 + max(level.values(), default=-1))]
        for c in sorted(need):
            levels[level[c]][self.plan.specs[c]].append(c)
        return Body(corr.implementation_path(m), e, tuple(operands), [[(sp, tuple(cs)) for sp, cs in lv.items()] for lv in levels], tuple(targets))

    def recipe(self, c: int, inside: set[int], feeds: dict[tuple[int, str], tuple[int, int]]) -> list[list[tuple]]:
        """Per argument group of Call `c`, its slices: ("v", Call, lo, hi) an evaluated Value or literal, ("o", operand, row, lo, hi) a
        forward argument's words, ("w", field, lo, hi) a weight field's."""
        out = []
        for g in self.raw[c].get("args") or ():
            sl = []
            for kind, a, lo, hi in g:
                a, lo, hi = int(a), int(lo), int(hi)
                if kind == "p":
                    k = self.field_at(lo, hi)
                    sl.append(("w", k, lo - self.wports[k].lo, hi - self.wports[k].lo))
                elif a in self.pop.literals or a in inside:
                    if a in self.pop.literals:
                        self.literal(a)
                    sl.append(("v", a, lo, hi))
                else:
                    q = self.port_at(a, lo, hi)
                    k, i = feeds[(a, q.name)]
                    sl.append(("o", k, i, lo - q.lo, hi - q.lo))
            out.append(sl)
        return out


def _arena_bound(manifest: Mapping[str, Any], plan: Plan) -> tuple[int, int]:
    """(bytes per token row, tensors per step): the most any engine step commits per token row (its `input_token_ids` rows), and the
    most targets of one step."""
    tokens: dict[int, int] = {}
    for r in (manifest or {}).get("identities") or []:
        if r.get("family") == "input_token_ids" and "engine_step" in r:
            tokens[int(r["engine_step"])] = max(tokens.get(int(r["engine_step"]), 0), int(r.get("rows") or 0))
    nbytes: dict[int, int] = defaultdict(int)
    count: dict[int, int] = defaultdict(int)
    for (e, _m), b in plan.bodies.items():
        for t in b.targets:
            nbytes[e] += len(t.calls) * (t.hi - t.lo) * np.dtype(WORD_DTYPES[t.dtype][0]).itemsize
            count[e] += 1
    return max((-(-n // max(1, tokens.get(e, 1))) for e, n in nbytes.items()), default=0), max(count.values(), default=0)


# ---------------------------------------------------------------------------------------------------------------------------- evaluation
def evaluate_body(plan: Plan, body: Body, operands: Sequence[np.ndarray], weight: Callable[[int], np.ndarray]) -> dict[str, np.ndarray]:
    """Target name -> its [rows, width] words: `body`'s Calls evaluated level by level, one `evaluate_batch` per (level, Definition).
    `operands[k]` = forward argument k's [rows, elems] words; `weight(k)` = weight field k's words."""
    from verity.evaluation import evaluate_batch
    from verity_vllm.program.kernels import rows  # noqa: F401  -- registers the row kernels evaluate_batch picks

    vals: dict[int, np.ndarray] = dict(plan.literals)

    def gather(sl: list[tuple]) -> np.ndarray:
        parts = [vals[x[1]][x[2]:x[3]] if x[0] == "v" else operands[x[1]][x[2], x[3]:x[4]] if x[0] == "o" else weight(x[1])[x[2]:x[3]]
                 for x in sl]
        return parts[0] if len(parts) == 1 else np.concatenate([p.astype(np.uint64) for p in parts])

    for level in body.levels:
        for spec, cids in level:
            groups = zip(*(plan.recipes[c] for c in cids))
            arrays = [gather(g[0])[None] if len(cids) == 1 else np.stack([gather(x) for x in g]) for g in groups]
            out = evaluate_batch(plan.definitions[spec][0], *arrays)
            for c, row in zip(cids, out.reshape(len(cids), -1)):
                vals[c] = row
    return {t.name: np.stack([vals[c][t.lo:t.hi] for c in t.calls]).astype(WORD_DTYPES[t.dtype][0]) for t in body.targets}


def words(torch: Any, t: Any) -> np.ndarray:
    """A tensor's words in its own width, flat, on the host."""
    x = t.detach().contiguous()
    return x.view(getattr(torch, _SIGNED[x.element_size()])).cpu().numpy().view(f"u{x.element_size()}").reshape(-1)


def operand_words(torch: Any, body: Body, tensors: Sequence[Any]) -> list[np.ndarray]:
    """The body's operands from M's tensor arguments (module docstring); ValueError naming the site on any other argument."""
    if len(tensors) < len(body.operands):
        raise ValueError(f"call boundaries: {body.module} ran with {len(tensors)} tensor arguments at engine step {body.engine_step}; its "
                         f"body reads {len(body.operands)} outside Values ({[o.where for o in body.operands][:3]})")
    out = []
    for k, (o, t) in enumerate(zip(body.operands, tensors)):
        shape = tuple(int(d) for d in t.shape)
        if math.prod(shape) != len(o.rows) * o.elems or not shape or shape[-1] != o.elems or t.element_size() * 8 != o.width:
            raise ValueError(f"call boundaries: {body.module} argument {k} is {shape} {t.dtype} at engine step {body.engine_step}; the body "
                             f"reads {o.where} as {len(o.rows)} rows x {o.elems} Value<{o.width}>")
        out.append(words(torch, t).reshape(len(o.rows), o.elems))
    return out


# -------------------------------------------------------------------------------------------------------------------------------- source
class CallBoundarySource:
    short_name = "call_boundary"

    def __init__(self, committer: Any, model: Any, plan: Plan) -> None:
        import torch

        if not plan.bodies:
            raise ValueError("call boundaries: no module body to evaluate")
        self.torch, self.c, self.model, self.plan = torch, committer, model, plan
        self.modules = sorted({m for _e, m in plan.bodies})
        self._handles: list[Any] = []
        self._held: list[Any] = []
        self._ran: set[tuple[int, str]] = set()
        self._values: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        self._fields: dict[int, np.ndarray] = {}
        self.stats: dict[str, Any] = {"tensors": 0, "bytes": 0, "bodies": 0, "modules": len(self.modules), "errors": 0, "seconds": 0.0}
        self.active = False

    def install(self) -> None:
        mods = dict(self.model.named_modules())
        missing = sorted(set(self.modules) - set(mods))
        if missing:
            raise RuntimeError(f"call boundaries: the manifest's module bodies are not modules of the model: {missing[:4]}")
        for op in self.modules:
            self._handles.append(mods[op].register_forward_pre_hook(self._pre(op), with_kwargs=True))
        self.active = True

    def uninstall(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._held.clear()
        self._values.clear()
        self.active = False

    def on_step_begin(self) -> None:
        self._held.clear()
        self._ran.clear()
        self._values.clear()

    def error(self, msg: str) -> None:
        self.stats["errors"] += 1
        self.c.errors.append(msg[:300])

    def weight(self, k: int) -> np.ndarray:
        """Weight field k's words: the model's parameter or buffer of its implementation name, in the field's declared width."""
        if k not in self._fields:
            name, elems, width = self.plan.fields[k]
            ts = dict(self.model.named_buffers())
            ts.update(dict(self.model.named_parameters()))
            t = ts.get(name)
            if t is None or t.numel() != elems or t.element_size() * 8 != width:
                raise ValueError(f"call boundaries: weight field {name} is {None if t is None else (tuple(t.shape), t.dtype)} in the model; "
                                 f"the Program declares {elems} Value<{width}>")
            self._fields[k] = words(self.torch, t)
        return self._fields[k]

    def _pre(self, op: str) -> Any:
        me, torch = self, self.torch

        def pre(_module: Any, args: tuple, kwargs: dict) -> None:
            job = me.c._cur
            body = me.plan.bodies.get((int(job.step), op)) if job is not None else None
            if body is None:
                return None
            if (body.engine_step, op) in me._ran:
                me.error(f"call boundaries: {op} ran twice at engine step {body.engine_step}; the Program has one body")
                return None
            me._ran.add((body.engine_step, op))
            try:
                me.take(body, [a for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor)], job)
            except ValueError as e:
                me.error(str(e))
            return None

        return pre

    def take(self, body: Body, tensors: Sequence[Any], job: Any) -> None:
        torch = self.torch
        t0 = time.perf_counter()
        out = evaluate_body(self.plan, body, operand_words(torch, body, tensors), self.weight)
        self._values[(body.module, body.engine_step)] = out
        dev = tensors[0].device if tensors else torch.device("cpu")
        n = 0
        for t in body.targets:
            w = out[t.name]
            v = torch.from_numpy(w.view(_SIGNED[w.itemsize])).view(getattr(torch, WORD_DTYPES[t.dtype][1])).to(dev)
            self._held.append(v)
            self.c._acquire(t.name, v, job)
            n += int(w.nbytes)
        self.stats["tensors"] += len(body.targets)
        self.stats["bytes"] += n
        self.stats["bodies"] += 1
        self.stats["seconds"] += time.perf_counter() - t0
        spans = getattr(self.c, "spans", None)
        if spans is not None:
            spans.add("acquisition", time.perf_counter() - t0, step=job.step, nbytes=n)

    def values(self, module: str, engine_step: int) -> dict[str, np.ndarray]:
        """The words this step committed for module body `module` (target name -> [rows, width]); {} before its pre-hook ran."""
        return dict(self._values.get((module, int(engine_step)), {}))

    def activation_scales(self, linear: str, engine_step: int) -> np.ndarray | None:
        """The f32 words ([rows, K/G], u32) of the one 32-bit call-boundary Value of `linear`'s body at `engine_step` -- a block-scaled
        FP8 linear's group scales x_s, as committed; None when its body commits no such Value or several."""
        f32 = [w for n, w in self.values(linear, engine_step).items() if w.dtype == np.uint32]
        return f32[0] if len(f32) == 1 else None

    def extra_step_bytes(self, envelope: Mapping[str, Any], model: Any) -> dict[str, int]:
        return {"per_token_bytes": self.plan.bytes_per_token, "per_seq_bytes": 0, "tensors_per_step": self.plan.tensors_per_step}

    def declaration_classes(self) -> list[dict]:
        return [{"class": FAMILY, "how": "bound-as-leaf", "rule": None, "count": self.stats["tensors"], "bytes": self.stats["bytes"],
                 "tensors": self.stats["tensors"], "bodies": self.stats["bodies"], "errors": self.stats["errors"],
                 "evidence": (f"call-boundary source: `<module>/<fx node>/<member>` of {self.plan.covered} identities in {len(self.modules)} "
                              f"module bodies, evaluated on the host from each module's forward arguments by verity.evaluation.evaluate_batch "
                              f"(Program {self.plan.program_dir}); the consuming Calls' replay checks the served outputs")}]


def attach(com: Any, enabled: bool, manifest: Mapping[str, Any] | None, required_classes: tuple[str, ...], *, explicit_classes: bool = False,
           model: Any = None, programs: Sequence[str | None] = ()) -> tuple[tuple[str, ...], CallBoundarySource | None]:
    """Attach the source to `com` when `enabled`; the required manifest must name `call_boundaries` identities, `programs` (the Build dir,
    the programs root) must hold its Program and every identity must be covered, else the Commit is refused (exit 3).  Returns the
    required classes with `call_boundaries` added unless they were given explicitly."""
    if not enabled:
        return required_classes, None
    if not manifest_sites(manifest):
        print("[delta] REFUSE: the call-boundary source but the required-value manifest names no call_boundaries identity", flush=True)
        raise SystemExit(3)
    pdir = program_dir_of(manifest or {}, *programs)
    if pdir is None:
        print(f"[delta] REFUSE: call boundaries: no Build dir under {list(programs)} holds the manifest's Program "
              f"{str((manifest or {}).get('program_digest'))[:16]} (--program-dir / --programs-root)", flush=True)
        raise SystemExit(3)
    plan = plan_of(manifest or {}, pdir)
    if plan.uncovered:
        print(f"[delta] REFUSE: call boundaries: {len(plan.uncovered)} of {plan.identities} identities have no host evaluation: "
              f"{plan.uncovered[:4]}", flush=True)
        raise SystemExit(3)
    src = CallBoundarySource(com, model if model is not None else getattr(com, "model", None), plan)
    com.attach_source(src)
    print(f"[delta] call-boundary source attached: {plan.covered} identities in {len(src.modules)} module bodies ({pdir}), "
          f"{plan.bytes_per_token} bytes per token -> committer {com.name}", flush=True)
    if not explicit_classes and FAMILY not in required_classes:
        return (*required_classes, FAMILY), src
    return required_classes, src


__all__ = ["FAMILY", "WORD_DTYPES", "CLAIMS", "Uncovered", "Target", "Operand", "Body", "Plan", "CallBoundarySource", "claimed", "manifest_sites", "program_dir_of",
           "plan_of", "evaluate_body", "words", "operand_words", "attach"]
