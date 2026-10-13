"""Each step's attention routes on one rank, recorded from the call that decides them, and the step's committed names held to the
capture map's names for those routes (`verity_vllm.commit.capture_map`).

A capture map whose model routes its attention per step names the decision in `route_site`: a method of each attention layer
(`module`, `method`) whose answer on the layer's attention metadata picks the routes (`true`, `false`), and the routes a step's
decode tokens take whatever the answer (`decode_tokens`, when the metadata's `num_decode_tokens` is positive).  `find_map` picks the
map that describes a model (its `model_class` is the model's class and its target's tensor-parallel size the rank's world); the
source wraps the method on every attention layer and records, per step and layer, the routes taken, and whether the step had
prefill tokens (its phase).  The pinned vLLM's sparse-attention MLA layer asks once per step before it attends, and again in
`forward_impl` when the step has prefill tokens; both answers read the same metadata.

A model that routes on more than one decision per step names each call that decides one in `route_site.sites` (`CM.route_sites`):
a site is a method of a module of every layer that has it (`module`, `method`), the decision it belongs to (`decision`), and how a
call answers (`answer`): `called` (every call answers true), `return` (its return's truth), `none_returned` (it returned None, or a
tuple of Nones) or an expression over the map's symbols and `M`, the leading dimension of the call's first tensor argument; an answer
picks `true` or `false`, and a site marked `prefill` makes a step that answered true at it a prefill step.  A step's routes are the
union over its decisions; a decision's layers are those that have one of its sites, and each of them must record the same routes on
the step.  DeepSeek-V4-Pro's map decides its mHC route at the decoder layer's token count, its indexer route at the indexer's return
(None on the short path) and its attention rows at FlashMLA's `_forward_decode` and `_forward_prefill` calls.

`identity_check` reads two rules from it.  `split_step_names`: the step's names that the module hooks, the sampler wrap, the
all-reduce-norm source, the attention source and the mHC source bind (`BINDERS`; the rank-tagged partials and the MoE planes have
rules of their own) are exactly the map's `committer_names` for the step's routes and phase, once each, and no module-hook name (`<module>`,
`<module>/<k>`, `<module>/in`) of a module the map lists is outside them.  Those names leave the step-0 multiplicity rule, which a
step on another route than step 0 would fail, and the module rule: `step_modules`, the hooked modules a step must cover by a name of
their own, are the ones the map has no entry for.  A fused norm, or `kv_b_proj` on a sparse step, is held to the map's names, not to
"every hooked module ran".  A step whose layers recorded no route, or different ones, fails; a padding-only step (nobody ran it) is
held to the routes of the run's first step.  `capture-map ids --route R --binder B --names` prints the names a step on route R
commits.

A tap the map marks `target_only` is bound only on the map's target device (`on_target`: the CUDA device's compute capability and
SM count are the target's): there a step without it fails, and elsewhere no step may commit it.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping

import torch

from verity_vllm.acquire.sources import attention_source, mhc_source
from verity_vllm.commit import capture_map as CM
from verity_vllm.config import MANIFESTS
from verity_vllm.engine.hooks import Hooks

MAPS = MANIFESTS / "capture-maps"
BINDERS = ("module_hook", "sampler", "allreduce_norm", attention_source.BINDER, mhc_source.BINDER)
#: binders whose sources hold their names to rules of their own (`partial_source`, `moe_source`)
OWN_RULES = ("tp_partial", "moe")
NO_ROUTE = "<no route recorded>"
SPLIT = "<layers took different routes>"
PHASES = ("prefill", "decode")
_HOOK_NAME = re.compile(r"^[^/]+(/(\d+|in))?$")


def world_of(model: Any) -> int:
    """The tensor-parallel size a rank's model was built for: the largest `tp_size` its modules hold (1 when none holds one)."""
    return max((m.tp_size for m in model.modules() if isinstance(getattr(m, "tp_size", None), int)), default=1)


def find_map(model: Any, world: int | None = None) -> dict[str, Any] | None:
    """The capture map under `manifests/capture-maps/` whose `model_class` is `model`'s class and whose target runs `world` ranks
    (default `world_of(model)`); None when no map describes the model."""
    cls = f"{type(model).__module__}.{type(model).__qualname__}"
    maps = [m for m in (CM.load(p) for p in sorted(MAPS.glob("*.json"))) if m.get("model_class") == cls]
    if not maps:
        return None
    world = world_of(model) if world is None else world
    return next((m for m in maps if m["target"]["tensor_parallel_size"] == world), None)


def unbuilt(cmap: dict[str, Any]) -> set[str]:
    """The binders `cmap` names that no source builds (neither `BINDERS` nor `OWN_RULES`)."""
    return set(cmap.get("binders", {})) - {*BINDERS, *OWN_RULES}


def on_target(cmap: dict[str, Any]) -> bool:
    """Whether this process's CUDA device is the map's target: its compute capability and SM count."""
    if not torch.cuda.is_available():
        return False
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    t = cmap["target"]
    return [p.major, p.minor] == list(t["compute_capability"]) and p.multi_processor_count == t["num_sms"]


def known_modules(cmap: dict[str, Any], env: dict[str, int]) -> set[str]:
    """Every module name the map has an entry for, at the layers of its entry (a shared module at the first)."""
    out: set[str] = set()
    for e in cmap["modules"]:
        if not e["pattern"]:
            continue
        layers: list[int | None] = [None]
        if "{" + CM.LAYER_VAR + "}" in e["pattern"]:
            layers = list(CM.layer_set(cmap, e.get("when", "all"), env))
        out.update(CM.at_layer(e["pattern"], i) for i in (layers[:1] if e.get("shared") else layers))
    return out


class RouteSource:
    short_name = "routes"

    def __init__(self, com: Any, model: Any, cmap: dict[str, Any], *, rank: int = 0, world: int | None = None,
                 config: Mapping[str, Any] | None = None, target: bool | None = None) -> None:
        if not cmap.get("route_site"):
            raise ValueError(f"capture map {cmap.get('id')}: no route_site")
        self.sites = CM.route_sites(cmap)
        self.c, self.cmap, self.site, self.rank = com, cmap, self.sites[0], int(rank)
        self.on_target = on_target(cmap) if target is None else bool(target)
        world = world_of(model) if world is None else world
        self.env = CM.symbols(cmap, dict(config) if config is not None else model.config.to_dict(), world)
        #: per site, layer -> the module whose method it is; a decision's layers are those of its sites
        self.site_layers: list[dict[int, Any]] = []
        self.decisions: dict[str | None, set[int]] = {}
        for site in self.sites:
            at = re.compile("^" + re.escape(site["module"]).replace(re.escape("{" + CM.LAYER_VAR + "}"), r"(\d+)") + "$")
            found = {int(k.group(1)): m for n, m in model.named_modules()
                     if (k := at.match(n)) and callable(getattr(m, site["method"], None))}
            if not found:
                raise ValueError(f"capture map {cmap.get('id')}: the model has no {site['module']} with a {site['method']} method")
            self.site_layers.append(found)
            self.decisions.setdefault(site.get("decision"), set()).update(found)
        self.layers: dict[int, Any] = self.site_layers[0]
        self.known = known_modules(cmap, self.env)
        self._want: dict[tuple[tuple[str, ...] | None, str], dict[str, int]] = {}
        self.mine = {n for p in PHASES for n in CM.committer_names(cmap, self.env, phase=p, rank=self.rank, binders=BINDERS)}
        self._hooks = Hooks()
        #: step -> decision -> layer -> the routes it took
        self.routes: dict[int, dict[str | None, dict[int, set[str]]]] = {}
        self.prefill: dict[int, bool] = {}
        self.calls = 0
        self.attention: Any = None

    def names_at(self, routes: tuple[str, ...] | None, phase: str) -> dict[str, int]:
        """The names `BINDERS` commit at a step on `routes` (every route when None) in `phase`, once each, the `target_only` ones
        only on the target; the rank-tagged ones are the partial source's."""
        key = (routes, phase)
        if key not in self._want:
            got = CM.committer_names(self.cmap, self.env, phase=phase, routes=routes, rank=self.rank, binders=BINDERS,
                                     on_target=self.on_target)
            self._want[key] = {n: 1 for n, (tap, _layer) in got.items() if "{r}" not in tap["commit"]["name"]}
        return self._want[key]

    def install(self) -> None:
        for k, (site, found) in enumerate(zip(self.sites, self.site_layers)):
            for layer, m in sorted(found.items()):
                self._hooks.wrap(m, site["method"], self._decide(layer) if "answer" not in site else self._observe(k, layer))

    def uninstall(self) -> None:
        self._hooks.uninstall()

    def on_step_begin(self) -> None:
        pass

    def begin_run(self) -> None:
        self.routes, self.prefill, self.calls = {}, {}, 0

    def _decide(self, layer: int) -> Any:
        me = self

        def make(orig: Any) -> Any:
            def decide(attn_metadata: Any, *args: Any, **kwargs: Any) -> Any:
                out = orig(attn_metadata, *args, **kwargs)
                me.record(layer, bool(out), attn_metadata)
                return out
            return decide
        return make

    def record(self, layer: int, answer: bool, metadata: Any) -> None:
        """One answer of `layer`'s route method at the committer's current step (none outside a step: warm-up, profiling)."""
        job = getattr(self.c, "_cur", None)
        if job is None:
            return
        step = int(job.step)
        routes = set(self.site["true" if answer else "false"])
        if int(getattr(metadata, "num_decode_tokens", 0) or 0) > 0:
            routes.update(self.site.get("decode_tokens", ()))
        self.routes.setdefault(step, {}).setdefault(self.site.get("decision"), {}).setdefault(layer, set()).update(routes)
        self.prefill[step] = self.prefill.get(step, False) or getattr(metadata, "prefill", None) is not None
        self.calls += 1

    def _observe(self, k: int, layer: int) -> Any:
        me = self

        def make(orig: Any) -> Any:
            def observe(*args: Any, **kwargs: Any) -> Any:
                out = orig(*args, **kwargs)
                me.observe(k, layer, args, kwargs, out)
                return out
            return observe
        return make

    def answer(self, site: Mapping[str, Any], args: tuple, kwargs: Mapping[str, Any], out: Any) -> bool:
        """What one call of `site`'s method answers (module docstring)."""
        how = site["answer"]
        if how == "called":
            return True
        if how == "return":
            return bool(out)
        if how == "none_returned":
            return out is None or (isinstance(out, tuple) and all(v is None for v in out))
        first = next((a for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor)), None)
        if first is None or first.dim() == 0:
            raise ValueError(f"route site {site['module']}.{site['method']}: {how!r} reads M, and the call has no tensor argument")
        return bool(CM.evaluate(how, {**self.env, "M": int(first.shape[0])}))

    def observe(self, k: int, layer: int, args: tuple, kwargs: Mapping[str, Any], out: Any) -> None:
        """One call of site `k` on `layer` at the committer's current step (none outside a step)."""
        job = getattr(self.c, "_cur", None)
        if job is None:
            return
        site, step = self.sites[k], int(job.step)
        yes = self.answer(site, args, kwargs, out)
        self.routes.setdefault(step, {}).setdefault(site.get("decision"), {}).setdefault(layer, set()).update(
            site["true"] if yes else site.get("false", ()))
        self.prefill[step] = self.prefill.get(step, False) or (yes and bool(site.get("prefill")))
        self.calls += 1

    def step_routes(self, engine_step: int, padding_only: bool = False) -> tuple[tuple[str, ...] | None, str]:
        """(the routes every layer took at `engine_step`, over every decision, and its phase), or (None, why not)."""
        step = min(self.routes) if padding_only and self.routes else int(engine_step)
        rec = self.routes.get(step)
        if not rec:
            return None, NO_ROUTE
        taken: set[str] = set()
        for d, layers in self.decisions.items():
            got = rec.get(d) or {}
            why = "" if d is None else f" ({d})"
            if not got:
                return None, NO_ROUTE[:-1] + why + ">"
            if set(got) != layers:
                return None, f"<{len(layers - set(got))} of {len(layers)} layers recorded no route{why}>"
            one = {frozenset(r) for r in got.values()}
            if len(one) != 1:
                return None, SPLIT[:-1] + why + ">"
            taken |= next(iter(one))
        return tuple(sorted(taken)), "prefill" if self.prefill.get(step) else "decode"

    def split_step_names(self, names: Mapping[str, int], engine_step: int, padding_only: bool) -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
        """X-03 for the names the map binds through `BINDERS` at one step: (the step's other names, for the module rule and the
        step-0 multiplicity rule; name -> (committed, the map's count on the step's routes) where they differ -- by multiplicity, or
        as a set at a padding-only step).  A step without one route for every layer reports why under that reason's name."""
        got = {n: c for n, c in names.items() if n in self.mine or (_HOOK_NAME.match(n) and n.split("/", 1)[0] in self.known)}
        rest = {n: c for n, c in names.items() if n not in got}
        routes, phase = self.step_routes(engine_step, padding_only)
        if routes is None:
            return rest, {phase: (sum(got.values()), 0)}
        want = self.names_at(routes, phase)
        keys = set(got) ^ set(want) if padding_only else {n for n in set(got) | set(want) if got.get(n, 0) != want.get(n, 0)}
        return rest, {n: (got.get(n, 0), want.get(n, 0)) for n in keys}

    def step_modules(self, expected: set[str]) -> set[str]:
        """The hooked modules (`expected`) a step must cover by names of their own: those the map has no entry for and binds no name
        of (`split_step_names` holds the others to the map's names)."""
        held = self.known | {n.split("/", 1)[0] for n in self.mine}
        return {m for m in expected if m not in held}

    @property
    def stats(self) -> dict[str, Any]:
        steps = {s: self.step_routes(s) for s in sorted(self.routes)}
        return {"map": self.cmap.get("id"), "layers": len(self.layers), "calls": self.calls, "steps": len(steps), "on_target": self.on_target,
                "steps_by_routes": dict(Counter(f"{why} {'+'.join(r)}" if r else why for r, why in steps.values())),
                **({"attention": self.attention.stats} if self.attention is not None else {})}

    def declaration_classes(self) -> list[dict]:
        return []


def attach(com: Any, model: Any, rank: int = 0) -> RouteSource | None:
    """Attach the source to `com` when a capture map with a `route_site` describes `model`, and the attention source
    (`attention_source`) and the mHC source (`mhc_source`) beside it when the map binds taps through them; None when no map
    describes the model, and while the map names a binder no source builds (`unbuilt`), whose taps nothing would commit while the
    source releases the modules the map knows from the module rule."""
    cmap = find_map(model)
    if cmap is None or not cmap.get("route_site") or unbuilt(cmap):
        return None
    src = RouteSource(com, model, cmap, rank=rank)
    attn = attention_source.AttentionSource(com, src) if attention_source.binds(cmap) else None
    mhc = mhc_source.MhcSource(com, src, model) if mhc_source.binds(cmap) else None
    com.attach_source(src)
    if attn is not None:
        com.attach_source(attn)
        src.attention = attn
    if mhc is not None:
        com.attach_source(mhc)
    return src
