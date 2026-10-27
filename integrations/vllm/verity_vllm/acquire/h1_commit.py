"""The Commit in H1's serving engine (ROOTS1, VLLM-73), installed: the committer, the int8 source at each linear H1 records
and the tap that keeps what the Commit acquired for H1 (`protocol_options.pouw_h1_commit.CommitTap`). Here in ``acquire``
because it builds the int8 source; `pouw_h1_commit` holds the tap and the values and imports nothing from ``acquire``.
On a server that commits already (serve-openai's rows-sha512 collector), `tap` puts H1's tap on that Commit instead.

A MoE layer's routed experts (`engine.int8_experts.ExpertSite`, VLLM-80) are no modules: each expert projection's site is
named as H1 names it (`pouw_h1_hooks.expert_names`, ``prefix.expert.projection``), and its Int8RowQuant_v1 output is
acquired under that name inside the call (`ExpertGemm.apply`), as `Int8ExpertsMethod` runs every expert on every forward:
an expert no token routed to is a call of zero rows, acquired as its zero rows [0, K] and scales [0]."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from verity_vllm.acquire.sources.int8_source import Int8Source
from verity_vllm.protocols.pouw.h1.commit import CommitTap, Sites, _EveryStep, site
from verity_vllm.protocols.pouw.h1.ranks import RankExtension, RankRun, RankTap


class H1Source(Int8Source):
    """`Int8Source` at H1's linears: a named module's site under its module's forward, and each routed expert's under its
    call (``experts``, name -> `ExpertSite`)."""

    def __init__(self, com: Any, model: Any, modules: Sequence[str], experts: dict[str, Any]) -> None:
        sites = Sites([*modules, *experts])
        super().__init__(com, model, sites, [])
        # the modules whose forward opens their site's scope; an expert's call opens its own (`_scoped`)
        self.modules = sorted(modules)
        self.quant, self._at, self.experts = sites, _EveryStep(self._at), dict(experts)

    def install(self) -> None:
        from verity_vllm.engine import hooks
        super().install()
        for name, s in self.experts.items():
            self._hooks.append(hooks.wrap(s.quant_method, "apply", lambda orig, name=name: self._scoped(name, orig)))

    def _scoped(self, name: str, orig: Any) -> Any:
        """Expert site ``name``'s ``apply`` with the source's scope at ``name``, so its ``row_quant`` is acquired as the
        site's; a call of zero rows runs no ``row_quant`` and is acquired as its zero rows, before the call."""
        me = self

        def apply(layer: Any, x: Any, bias: Any = None) -> Any:
            job = me.c._cur
            k = int(x.shape[-1])
            if job is not None and x.numel() == 0:
                key = (int(job.step), name)
                if key in me._seen:
                    me.error(f"int8: {name} was called twice at engine step {key[0]}; the Program has one Int8RowQuant_v1 site")
                else:
                    me._seen.add(key)
                    me._take(f"{site(name)}/0", x.new_empty((0, k), dtype=me.torch.int8), job, "quant")
                    me._take(f"{site(name)}/1", x.new_empty(0, dtype=me.torch.float32), job, "quant")
            me._scope.append(name)
            try:
                return orig(layer, x, bias)
            finally:
                me._scope.pop()
        return apply


def _linears(model: Any, layers: Sequence[Any]) -> tuple[list[str], dict[str, Any]]:
    """H1's ``layers``: the modules' names in ``model`` and the routed experts by their H1 names. ValueError on a module
    the model doesn't name."""
    from verity_vllm.protocols.pouw.h1.hooks import expert_names
    names = expert_names(list(layers))
    experts = {n: layers[w] for w, n in names.items()}
    ids = {id(m) for w, m in enumerate(layers) if w not in names}
    modules = [name for name, m in model.named_modules() if id(m) in ids]
    if len(modules) != len(ids):
        raise ValueError(f"{len(ids) - len(modules)} of H1's linears are not named modules of the model")
    return modules, experts


def source(com: Any, model: Any, layers: Sequence[Any]) -> Any:
    """The int8 source (`H1Source`) on committer ``com``, at the Int8RowQuant_v1 site of each of ``layers`` (H1's
    linears, `pouw_h1_hooks.layers`) at every engine step: a linear by its name in ``model``, as `pouw.install` names it,
    and a routed expert by its H1 name (`pouw_h1_hooks.expert_names`)."""
    modules, experts = _linears(model, layers)
    return H1Source(com, model, modules, experts)


def tap(com: Any, model: Any, layers: Sequence[Any], steps: Sequence[int], tap: CommitTap | None = None) -> CommitTap:
    """H1's tap (``tap``, else a new `CommitTap`) on the Commit already serving ``model`` (serve-openai's rows-sha512
    collector, `pipeline.serve_seal.committer`), in place of a Commit of its own (`attach`): the step's input ids are its
    step-input port's (`served_record.StepInputPort`) and each of H1's ``layers``' x_q and s_a are its int8 source's
    (`Int8Source`, the bench row's manifest's sites, under their op paths), so H1's A tables are tied to the values the
    served window commits. ValueError when the Commit has no port or no int8 source, when ``layers`` hold a routed
    expert (only H1's own source acquires those), or when the source doesn't acquire some linear at each engine step of
    ``steps``, the window's. ``tap.untap()`` removes it."""
    from verity_vllm.acquire.sources.int8_source import Int8Source
    from verity_vllm.commit.served_record import PORT_NAMES, StepInputPort

    sources = list(getattr(com, "sources", ()))
    src = next((s for s in sources if isinstance(s, Int8Source) and not isinstance(s, H1Source)), None)
    if not any(isinstance(s, StepInputPort) for s in sources):
        raise ValueError(f"the Commit {getattr(com, 'name', com)} has no step-input port: no step's input ids for H1")
    if src is None:
        raise ValueError(f"the Commit {getattr(com, 'name', com)} has no int8 source: no linear's x_q and s_a for H1")
    modules, experts = _linears(model, layers)
    if experts:
        raise ValueError(f"H1's linears hold {len(experts)} routed experts, {sorted(experts)[:2]}: the served Commit's int8 "
                         "source acquires no expert's site")
    if missing := [(e, m) for e in steps for m in modules if (e, m) not in src.quant]:
        raise ValueError(f"the Commit's int8 source acquires {len(missing)} of H1's (engine step, linear) sites nowhere, "
                         f"{missing[:3]}: its manifest names no Int8RowQuant_v1 site there")
    tap = tap if tap is not None else CommitTap()
    names = set(modules)
    tap.embed, tap.port = None, PORT_NAMES[0]
    tap.ops = {op: site(m) for (_e, m), op in src.quant.items() if m in names}
    tap.tap(com)
    return tap


def fused_embed(com: Any, model: Any, embed: str) -> Any:
    """Where vLLM never calls embedding ``embed``, a forward pre-hook on its parent acquiring that forward's input ids as
    ``<embed>/in``, the name the committer's pre-hook on the embedding would; None where it does call it. A model whose
    table is whole on the rank (``replicated_embed``: vLLM's DeepseekV32Model, GLM-5.3's, at TP1) gathers it fused with
    layer 0's norm (`fused_embed_norm`) from the embedding's weight, so the embedding's own pre-hook never runs. The parent
    is called as ``(input_ids, positions, intermediate_tensors, inputs_embeds)``; with ``inputs_embeds`` it calls the
    embedding itself, and the committer's pre-hook takes its ids."""
    parent = model.get_submodule(embed.rpartition(".")[0])
    if not getattr(parent, "replicated_embed", False):
        return None

    def pre(module: Any, args: tuple, kwargs: dict) -> None:
        job = com._cur
        got = dict(zip(("input_ids", "positions", "intermediate_tensors", "inputs_embeds"), args), **kwargs)
        ids = got.get("input_ids")
        if job is None or got.get("inputs_embeds") is not None or got.get("intermediate_tensors") is not None:
            return
        try:
            if hasattr(ids, "shape"):
                com._acquire(f"{embed}/in", ids, job)
        except Exception as e:  # noqa: BLE001
            com.errors.append(f"{embed}/in: {e!r}"[:300])

    return parent.register_forward_pre_hook(pre, with_kwargs=True)


def attach(engine: Any, layers: Sequence[Any], tap: CommitTap | None = None, **committer: Any
           ) -> tuple[Any, Any, CommitTap]:
    """The Commit on ``engine`` (`engine.build.build_engine`'s, or a rank's, `RankCommit.h1rank_commit`), installed: a
    `NativeHostCommitter` (``committer`` its keywords), the int8 source (`source`), its `CommitTap` (``tap``, given its
    ``embed`` here, else a new one) and, on a model that gathers its embedding fused, the hook that acquires the step's
    ids (`fused_embed`). Returns the committer, the source and the tap; ``committer.close()`` and ``tap.untap()`` remove
    them."""
    from verity_vllm.commit.committer.native_host import NativeHostCommitter

    com = NativeHostCommitter(engine, **committer)
    src = source(com, com.model, layers)
    com.attach_source(src)
    tap = tap if tap is not None else CommitTap()
    tap.embed = next((n for n, m in com._modules if com._acquires_input(n, m)), None)
    if tap.embed is None:
        raise ValueError("the committer acquires no module's input: no embedding gives the step's token ids")
    tap.tap(com)
    com.install()
    if (h := fused_embed(com, com.model, tap.embed)) is not None:
        com._handles.append(h)
    return com, src, tap


class RankCommit(RankExtension):
    """`pouw_h1_ranks.RankExtension` with the Commit on the rank (`h1rank_commit`), the ``worker_extension_cls``
    `pouw_h1_ranks.COMMIT_EXTENSION`; here because it attaches the int8 source. The holder releases it with the rank
    (`RankExtension.h1rank_release`)."""

    def _h1rank_engine(self, model: Any, cp: Mapping[str, Any]) -> Any:
        """What the committer reads of an engine (`attach`), on this rank: its model, runner, vLLM config and the
        checkpoint's pin ``cp`` (repo, revision)."""
        return SimpleNamespace(llm=SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=self.vllm_config)),  # type: ignore[attr-defined]
                               runner=self.model_runner, model=model, cp=dict(cp))  # type: ignore[attr-defined]

    def h1rank_commit(self, cp: Mapping[str, Any] | None = None, **committer: Any) -> dict[str, Any]:
        """The Commit on this rank (`attach`, ``committer`` the committer's keywords, its ``run_id`` named with the rank's
        first linear), its tap a `pouw_h1_ranks.RankTap` naming each of the rank's H1 linears as the window does; returns
        the committer's name and the module whose input ids it acquires. A site the rank's source acquires that the
        window has no name for (a routed expert's included) is refused: under its rank-local name it would meet no call,
        or another rank's site of the same name."""
        from verity_vllm.protocols import pouw
        st = self._h1rank  # type: ignore[attr-defined]
        run: RankRun = st["run"]
        model, layers = self._h1rank_layers()
        names = {name: run.modules[run.lo + int(i)] for i, name in pouw._named(model, layers).items()}
        committer["run_id"] = f"{committer.get('run_id', 'h1')}/rank{run.lo}"
        com, src, tap = attach(self._h1rank_engine(model, cp or {}), layers, tap=RankTap(names=names), **committer)
        if unnamed := sorted({*src.modules, *src.experts} - set(names)):
            com.close()
            tap.untap()
            raise ValueError(f"rank [{run.lo}, {run.hi}): the Commit acquires {len(unnamed)} sites the window has no name "
                             f"for, {unnamed[:4]}")
        st["commit"] = (com, tap)
        return {"lo": run.lo, "committer": com.name, "embed": tap.embed}


__all__ = ["H1Source", "RankCommit", "attach", "fused_embed", "source", "tap"]
