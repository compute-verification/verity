"""The Commit in H1's serving engine (ROOTS1, VLLM-73), installed: the committer, the int8 source at each linear H1 records
and the tap that keeps what the Commit acquired for H1 (`protocol_options.pouw_h1_commit.CommitTap`). Here in ``acquire``
because it builds the int8 source; `pouw_h1_commit` holds the tap and the values and imports nothing from ``acquire``.

A MoE layer's routed experts (`engine.int8_experts.ExpertSite`, VLLM-80) are no modules: each expert projection's site is
named as H1 names it (`pouw_h1_hooks.expert_names`, ``prefix.expert.projection``), and its Int8RowQuant_v1 output is
acquired under that name inside the call (`ExpertGemm.apply`), as `Int8ExpertsMethod` runs every expert on every forward:
an expert no token routed to is a call of zero rows, acquired as its zero rows [0, K] and scales [0]."""
from __future__ import annotations

from typing import Any, Sequence

from verity_vllm.acquire.sources.int8_source import Int8Source
from verity_vllm.protocol_options.pouw_h1_commit import CommitTap, Sites, _EveryStep, site


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


def source(com: Any, model: Any, layers: Sequence[Any]) -> Any:
    """The int8 source (`H1Source`) on committer ``com``, at the Int8RowQuant_v1 site of each of ``layers`` (H1's
    linears, `pouw_h1_hooks.layers`) at every engine step: a linear by its name in ``model``, as `pouw.install` names it,
    and a routed expert by its H1 name (`pouw_h1_hooks.expert_names`)."""
    from verity_vllm.protocol_options.pouw_h1_hooks import expert_names
    names = expert_names(list(layers))
    experts = {n: layers[w] for w, n in names.items()}
    ids = {id(m) for w, m in enumerate(layers) if w not in names}
    modules = [name for name, m in model.named_modules() if id(m) in ids]
    if len(modules) != len(ids):
        raise ValueError(f"{len(ids) - len(modules)} of H1's linears are not named modules of the model")
    return H1Source(com, model, modules, experts)


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


def attach(engine: Any, layers: Sequence[Any], **committer: Any) -> tuple[Any, Any, CommitTap]:
    """The Commit on ``engine`` (`engine.build.build_engine`'s), installed: a `NativeHostCommitter` (``committer`` its
    keywords), the int8 source (`source`), its `CommitTap` and, on a model that gathers its embedding fused, the hook
    that acquires the step's ids (`fused_embed`). Returns the committer, the source and the tap; ``committer.close()``
    and ``tap.untap()`` remove them."""
    from verity_vllm.commit.committer.native_host import NativeHostCommitter

    com = NativeHostCommitter(engine, **committer)
    src = source(com, com.model, layers)
    com.attach_source(src)
    tap = CommitTap(next((n for n, m in com._modules if com._acquires_input(n, m)), None))
    if tap.embed is None:
        raise ValueError("the committer acquires no module's input: no embedding gives the step's token ids")
    tap.tap(com)
    com.install()
    if (h := fused_embed(com, com.model, tap.embed)) is not None:
        com._handles.append(h)
    return com, src, tap


__all__ = ["H1Source", "attach", "fused_embed", "source"]
