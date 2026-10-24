"""The Commit in H1's serving engine (ROOTS1, VLLM-73), installed: the committer, the int8 source at each linear H1 records
and the tap that keeps what the Commit acquired for H1 (`protocol_options.pouw_h1_commit.CommitTap`). Here in ``acquire``
because it builds the int8 source; `pouw_h1_commit` holds the tap and the values and imports nothing from ``acquire``."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from verity_vllm.acquire.sources.int8_source import Int8Source
from verity_vllm.protocol_options.pouw_h1_commit import CommitTap, Sites, _EveryStep
from verity_vllm.protocol_options.pouw_h1_ranks import RankExtension, RankRun, RankTap


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


def attach(engine: Any, layers: Sequence[Any], tap: CommitTap | None = None, **committer: Any
           ) -> tuple[Any, Any, CommitTap]:
    """The Commit on ``engine`` (`engine.build.build_engine`'s, or a rank's, `RankCommit.h1rank_commit`),
    installed: a `NativeHostCommitter` (``committer`` its keywords), the int8 source (`source`) and its `CommitTap`
    (``tap``, given its ``embed`` here, else a new one). Returns the committer, the source and the tap;
    ``committer.close()`` and ``tap.untap()`` remove them."""
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
        the committer's name and the module whose input ids it acquires."""
        from verity_vllm.protocol_options import pouw
        st = self._h1rank  # type: ignore[attr-defined]
        run: RankRun = st["run"]
        model, layers = self._h1rank_layers()
        names = {name: run.modules[run.lo + int(i)] for i, name in pouw._named(model, layers).items()}
        committer["run_id"] = f"{committer.get('run_id', 'h1')}/rank{run.lo}"
        com, _src, tap = attach(self._h1rank_engine(model, cp or {}), layers, tap=RankTap(names=names), **committer)
        st["commit"] = (com, tap)
        return {"lo": run.lo, "committer": com.name, "embed": tap.embed}


__all__ = ["RankCommit", "attach", "source"]
