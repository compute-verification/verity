"""Call boundaries (`call_boundaries`, Q_word: a Value two Calls of one module body exchange) as an acquisition source, evaluated on the
host.  Per module body M and committed engine step, a forward pre-hook reads M's inputs and evaluates M's rows with the IR's evaluator
(`verity.ml.circuits.evaluation.evaluate_batch`: a registered row kernel of `verity_vllm.program.kernels.rows` where one is, the reference for the rest
and for what a kernel declines -- exact, NaN payloads included); at the step's end every required interior Value is committed as
`<M>/<fx node>/<member>`, [rows, width] words over the identity's element range.  Attached whenever the required manifest names
`call_boundaries` identities; a manifest naming none refuses it (exit 3).  No kernel changes: the served kernels run unchanged, and the
Calls that consume these Values check the served outputs (the replay evaluates them from the words committed here).

What a body reads, and how its rows are grouped into a row template, is `call_boundary_plan`'s; a body the rule cannot evaluate leaves
its identities uncovered and the Commit is refused naming them, as is a manifest of several ranks.  An argument of another size, row width
or word width is a collector error naming the site.  A global manifest (several requests, `engine_steps` composition): the arguments at
engine step e pack the rows of every request of e in the runtime's order (`step_rows[e]`, the binding map's; each engine row id mapped to
`r<i>` by `binding.logical_request`), which must be the declared composition -- the same requests, each with its declared token rows --
else a collector error naming the step and nothing of it is committed.  A body whose requests share one template is evaluated packed in
the pre-hook, so its words (a block-scaled FP8 linear's x_s) are there during the forward; otherwise each request's rows are split out in
the runtime's order and evaluated with its own template at the step's end.  Each target is committed packed under the identity's name,
rows in the runtime's order: per token or one per request, the two splits of the binding map.  A run that is never a record (the
Commit's warm-up instrumented run, `placeholder`) commits every target as zero words of its shape and evaluates nothing: its root is
discarded, and the steps still lay out the record's tensors for the staging it sizes and learns.
"""
from __future__ import annotations

import math
import re
import time
from typing import Any, Callable, Collection, Mapping, Sequence

import numpy as np

from verity_vllm.acquire.sources import call_boundary_plan as CP
from verity_vllm.acquire.sources.call_boundary_plan import FAMILY, WORD_DTYPES, Body, Plan

_SIGNED = {1: "uint8", 2: "int16", 4: "int32", 8: "int64"}
WARMUP_ID = re.compile(r"warmup-(\d+)(?:-[0-9a-f]{8})?")

#: other sources' claims on `call_boundaries` identities (identity -> the claiming source's `short_name`, or None): a claimed identity is
#: that source's to acquire (a tap of a Value the runtime already holds, like a pre-bias GEMM output) and is never evaluated here
CLAIMS: list[Callable[[Mapping[str, Any]], str | None]] = []


def claimed(r: Mapping[str, Any]) -> str | None:
    """The source that claims identity `r` (`CLAIMS`), or None: this source evaluates it."""
    return next((c for f in CLAIMS if (c := f(r))), None)


def unattached(manifest: Mapping[str, Any] | None, attached: Collection[str]) -> list[str]:
    """The `call_boundaries` identities of `manifest` whose claimant is not among `attached` (the sources the Commit attaches), each named
    with it: no source acquires them."""
    return [f"{CP._label(r)}: claimed by {s}, a source the Commit does not attach" for r in (manifest or {}).get("identities") or []
            if r.get("family") == FAMILY and (s := claimed(r)) is not None and s not in attached]


def manifest_sites(manifest: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """The `call_boundaries` identities of a manifest this source evaluates: every one no other source claims."""
    return [r for r in (manifest or {}).get("identities") or [] if r.get("family") == FAMILY and claimed(r) is None]


def plan_of(manifest: Mapping[str, Any], *roots: str | None) -> Plan:
    """Every identity of `manifest_sites(manifest)` as a target of a body the source evaluates, or uncovered with the reason; each
    request's Program is the Build dir under `roots` (a Build dir, or a directory of them) whose program digest is the request's."""
    return CP.plan(manifest, manifest_sites(manifest), *roots)


def words(torch: Any, t: Any) -> np.ndarray:
    """A tensor's words in its own width, flat, on the host."""
    x = t.detach().contiguous()
    return x.view(getattr(torch, _SIGNED[x.element_size()])).cpu().numpy().view(f"u{x.element_size()}").reshape(-1)


def operand_words(torch: Any, body: Body, tensors: Sequence[Any]) -> list[np.ndarray]:
    """The body's operands from M's tensor arguments, every request's rows packed (module docstring); ValueError naming the site on any
    other argument."""
    if len(tensors) < len(body.operands):
        raise ValueError(f"call boundaries: {body.module} ran with {len(tensors)} tensor arguments at engine step {body.engine_step}; its "
                         f"body reads {len(body.operands)} outside Values ({[o.where for o in body.operands][:3]})")
    out = []
    for k, (o, t) in enumerate(zip(body.operands, tensors)):
        n = sum(q.operand_rows(k) for q in body.parts)
        shape = tuple(int(d) for d in t.shape)
        if math.prod(shape) != n * o.elems or not shape or shape[-1] != o.elems or t.element_size() * 8 != o.width:
            raise ValueError(f"call boundaries: {body.module} argument {k} is {shape} {t.dtype} at engine step {body.engine_step}; the body "
                             f"reads {o.where} as {n} rows x {o.elems} Value<{o.width}>")
        out.append(words(torch, t).reshape(n, o.elems))
    return out


def zeros(body: Body) -> dict[str, np.ndarray]:
    """Every target of `body` as zero words of the shape and word width its evaluation has: [rows, f * width], one row when step-shared."""
    rows = sum(q.rows for q in body.parts)
    return {t.name: np.zeros((1 if t.shared else rows, len(t.at) * (t.hi - t.lo)), WORD_DTYPES[t.dtype][0]) for t in body.targets}


# -------------------------------------------------------------------------------------------------------------------------------- source
class CallBoundarySource:
    short_name = "call_boundary"

    def __init__(self, committer: Any, model: Any, plan: Plan) -> None:
        import torch

        if not plan.bodies:
            raise ValueError("call boundaries: no module body to evaluate")
        self.torch, self.c, self.model, self.plan = torch, committer, model, plan
        self.modules = sorted({m for _e, m in plan.bodies})
        self._at: dict[int, list[Body]] = {}
        for (e, _m), b in sorted(plan.bodies.items()):
            self._at.setdefault(e, []).append(b)
        self._handles: list[Any] = []
        self._held: list[Any] = []
        self._ran: dict[tuple[int, str], Any] = {}                           # (engine step, module) -> the arguments' device
        self._args: dict[tuple[int, str], list[np.ndarray]] = {}             # a body evaluated per request: its packed operand words
        self._values: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        self._fields: dict[str, np.ndarray] = {}
        self._shared: dict = {}                                               # step-shared Values by form, once per Commit
        self.stats: dict[str, Any] = {"tensors": 0, "bytes": 0, "bodies": 0, "modules": len(self.modules), "errors": 0, "seconds": 0.0,
                                      "placeholder_bodies": 0}
        self.active = False
        #: set around a run that is never a record (the Commit's warm-up instrumented run, `taps.warmup`): every target is committed as
        #: zero words of its shape, so the steps lay out the same tensors, and nothing is evaluated
        self.placeholder = False
        self._names: frozenset[str] | None = None

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
        self.on_step_begin()
        self.active = False

    def on_step_begin(self) -> None:
        self._held.clear()
        self._ran.clear()
        self._args.clear()
        self._values.clear()

    def error(self, msg: str) -> None:
        self.stats["errors"] += 1
        self.c.errors.append(msg[:300])

    def weight(self, name: str) -> np.ndarray:
        """Weight field `name`'s words: the model's parameter or buffer of that implementation name, in the field's declared width, moved
        to the host once per Commit."""
        if name not in self._fields:
            elems, width = self.plan.fields[name]
            ts = dict(self.model.named_buffers())
            ts.update(dict(self.model.named_parameters()))
            t = ts.get(name)
            if t is None or t.numel() != elems or t.element_size() * 8 != width:
                raise ValueError(f"call boundaries: weight field {name} is {None if t is None else (tuple(t.shape), t.dtype)} in the model; "
                                 f"the Program declares {elems} Value<{width}>")
            self._fields[name] = words(self.torch, t)
        return self._fields[name]

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
            tensors = [a for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor)]
            me._ran[(body.engine_step, op)] = tensors[0].device if tensors else torch.device("cpu")
            try:
                me.take(body, tensors)
            except ValueError as e:
                me.error(str(e))
            return None

        return pre

    def take(self, body: Body, tensors: Sequence[Any]) -> None:
        """The body's operands at its pre-hook, evaluated now when its requests share one template, else held for the step's end."""
        t0 = time.perf_counter()
        ops = operand_words(self.torch, body, tensors)
        if self.placeholder:
            self._values[(body.module, body.engine_step)] = zeros(body)
            self.stats["placeholder_bodies"] += 1
        elif body.packed:
            tpl = body.parts[0].template
            self._values[(body.module, body.engine_step)] = CP.evaluate_template(self.plan, tpl, sum(q.rows for q in body.parts), ops,
                                                                                 self.weight, self._shared)
        else:
            self._args[(body.engine_step, body.module)] = ops
        self.stats["seconds"] += time.perf_counter() - t0

    def _order(self, e: int) -> list[str] | None:
        """The requests of engine step `e` in the runtime's row order, which must be the declared composition; None (a collector error
        naming the step) otherwise.  A request manifest's one request needs no row order."""
        if not self.plan.composition:
            return [self._at[e][0].parts[0].request]
        rows = (getattr(self.c, "step_rows", None) or {}).get(e)
        if rows is None:
            self.error(f"call boundaries: no row order recorded at engine step {e} (step_rows); its packed rows cannot be split")
            return None
        from verity_vllm.commit.binding import logical_request

        got = [(self._request(logical_request(rid, self.plan.order)), int(hi) - int(lo)) for rid, lo, hi in rows]
        declared = self.plan.composition.get(e) or {}
        if sorted(got) != sorted(declared.items()):
            self.error(f"call boundaries: engine step {e} ran (request, rows) {got} in this order; the manifest declares "
                       f"{sorted(declared.items())}: nothing of the step is committed")
            return None
        return [r for r, _n in got]

    def _request(self, rid: str) -> str:
        """An engine row id no workload id resolves: a warm-up request (`vllm_adapter.throwaway_requests`: `warmup-<i>`, vLLM's
        `-<8 hex>` suffix or not) is workload request i's stand-in -- same prompt length, same schedule -- so it takes that request's
        rows.  Anything else stays raw (a collector error naming it)."""
        m = WARMUP_ID.fullmatch(rid)
        return self.plan.by_index.get(int(m.group(1)), rid) if m else rid

    def per_request(self, body: Body, order: Sequence[str], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
        """Each request's rows of the packed operands `ops`, in the runtime's `order`, evaluated with its own template; the targets
        packed back in that order, a step-shared target once (the requests' step-shared Calls are one form, `call_boundary_plan._merge`)."""
        parts = {q.request: q for q in body.parts}
        off, outs = [0] * len(ops), []
        for r in order:
            q = parts[r]
            mine = []
            for k, o in enumerate(ops):
                n = q.operand_rows(k)
                mine.append(o[off[k]:off[k] + n])
                off[k] += n
            outs.append(CP.evaluate_template(self.plan, q.template, q.rows, mine, self.weight, self._shared))
        return {t.name: outs[0][t.name] if t.shared else np.concatenate([o[t.name] for o in outs]) for t in body.targets}

    def on_step_end(self, job: Any) -> None:
        """Check the step's row order against the declared composition, evaluate the per-request bodies, and commit every target."""
        e = int(job.step)
        if e not in self._at:
            return
        t0 = time.perf_counter()
        order = self._order(e)
        torch, n = self.torch, 0
        for b in self._at[e]:
            if (e, b.module) not in self._ran:
                self.error(f"call boundaries: {b.module} did not run at engine step {e}; the Program has its body there")
                continue
            out = self._values.get((b.module, e))
            if out is None and order is not None and (e, b.module) in self._args:
                try:
                    out = self._values[(b.module, e)] = self.per_request(b, order, self._args.pop((e, b.module)))
                except ValueError as err:
                    self.error(str(err))
                    continue
            if out is None or order is None:
                continue
            for t in b.targets:
                w = out[t.name]
                v = torch.from_numpy(w.view(_SIGNED[w.itemsize])).view(getattr(torch, WORD_DTYPES[t.dtype][1])).to(self._ran[(e, b.module)])
                self._held.append(v)
                self.c._acquire(t.name, v, job)
                n += int(w.nbytes)
            self.stats["tensors"] += len(b.targets)
            self.stats["bodies"] += 1
        self.stats["bytes"] += n
        self.stats["seconds"] += time.perf_counter() - t0
        spans = getattr(self.c, "spans", None)
        if spans is not None:
            spans.add("acquisition", time.perf_counter() - t0, step=e, nbytes=n)

    def targets_at(self, engine_step: int) -> dict[str, int]:
        """The names this source commits at `engine_step`, once each: the targets of the Program's bodies there.  A body's fx node names
        count on through the run (`logits_processor/div` at step 0, `div_1` at step 1), so no two steps share one."""
        return {t.name: 1 for b in self._at.get(int(engine_step), ()) for t in b.targets}

    def target_names(self) -> frozenset[str]:
        if self._names is None:
            self._names = frozenset(t.name for bs in self._at.values() for b in bs for t in b.targets)
        return self._names

    def split_step_names(self, names: Mapping[str, int], engine_step: int, padding_only: bool) -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
        """X-03 for this source's names at one step: (the step's other names, for the step-0 multiplicity rule; name -> (committed, targets)
        where the committed names differ from `targets_at` -- by multiplicity, or as a set at a padding-only step)."""
        mine = self.target_names()
        got = {n: c for n, c in names.items() if n in mine}
        want = self.targets_at(engine_step)
        keys = set(got) ^ set(want) if padding_only else {n for n in set(got) | set(want) if got.get(n, 0) != want.get(n, 0)}
        return {n: c for n, c in names.items() if n not in mine}, {n: (got.get(n, 0), want.get(n, 0)) for n in keys}

    def values(self, module: str, engine_step: int) -> dict[str, np.ndarray]:
        """The words of module body `module` at `engine_step` (target name -> [rows, f * width], every request's rows in the runtime's
        order); {} before they are evaluated (its pre-hook; the step's end for a body evaluated per request)."""
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
                              f"module bodies, evaluated on the host from each module's forward arguments by verity.ml.circuits.evaluation.evaluate_batch "
                              f"(the Programs of {sorted(self.plan.program_dirs)}); the consuming Calls' replay checks the served outputs")}]


def attach(com: Any, enabled: bool, manifest: Mapping[str, Any] | None, required_classes: tuple[str, ...], *, explicit_classes: bool = False,
           model: Any = None, programs: Sequence[str | None] = (), plan: Plan | None = None) -> tuple[tuple[str, ...], CallBoundarySource | None]:
    """Attach the source to `com` when `enabled`; the required manifest must name `call_boundaries` identities, `programs` (the Build dir,
    the programs root) must hold each request's Program and every identity must be covered, else the Commit is refused (exit 3).  `plan`:
    `plan_of` the manifest over `programs`, derived already (the Commit's call-boundary plan), else derived here.  Returns the required
    classes with `call_boundaries` added unless they were given explicitly."""
    if not enabled:
        return required_classes, None
    if not manifest_sites(manifest):
        print("[delta] REFUSE: the call-boundary source but the required-value manifest names no call_boundaries identity", flush=True)
        raise SystemExit(3)
    plan = plan if plan is not None else plan_of(manifest or {}, *programs)
    if plan.uncovered or not plan.bodies:
        print(f"[delta] REFUSE: call boundaries: {len(plan.uncovered)} of {plan.identities} identities have no host evaluation "
              f"(--program-dir / --programs-root {list(programs)}): {plan.uncovered[:4]}", flush=True)
        raise SystemExit(3)
    src = CallBoundarySource(com, model if model is not None else getattr(com, "model", None), plan)
    com.attach_source(src)
    print(f"[delta] call-boundary source attached: {plan.covered} identities in {len(src.modules)} module bodies of {len(plan.program_dirs)} "
          f"request Programs, {plan.bytes_per_token} bytes per token -> committer {com.name}", flush=True)
    if not explicit_classes and FAMILY not in required_classes:
        return (*required_classes, FAMILY), src
    return required_classes, src


__all__ = ["FAMILY", "WORD_DTYPES", "CLAIMS", "CallBoundarySource", "claimed", "unattached", "manifest_sites", "plan_of", "words",
           "operand_words", "zeros", "attach"]
