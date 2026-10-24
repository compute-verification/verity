"""The Commit in H1's serving engine (ROOTS1, VLLM-73), installed: the committer, the int8 source at each linear H1 records
and the tap that keeps what the Commit acquired for H1 (`protocol_options.pouw_h1_commit.CommitTap`). Here in ``acquire``
because it builds the int8 source; `pouw_h1_commit` holds the tap and the values and imports nothing from ``acquire``."""
from __future__ import annotations

from typing import Any, Sequence

from verity_vllm.acquire.sources.int8_source import Int8Source
from verity_vllm.protocol_options.pouw_h1_commit import CommitTap, Sites, _EveryStep


def source(com: Any, model: Any, layers: Sequence[Any]) -> Any:
    """The int8 source (`Int8Source`) on committer ``com``, at the Int8RowQuant_v1 site of each of ``layers`` (H1's
    linears, `pouw_h1_hooks.layers`, by their names in ``model``, as `pouw.install` names them) at every engine step."""
    ids = {id(m) for m in layers}
    sites = Sites([name for name, m in model.named_modules() if id(m) in ids])
    if len(sites.modules) != len(ids):
        raise ValueError(f"{len(ids) - len(sites.modules)} of H1's linears are not named modules of the model")
    src = Int8Source(com, model, sites, [])
    src.quant, src._at = sites, _EveryStep(src._at)
    return src


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


__all__ = ["attach", "fused_embed", "source"]
