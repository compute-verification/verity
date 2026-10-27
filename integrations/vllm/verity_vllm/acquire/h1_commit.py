"""The Commit in H1's serving engine, installed: the committer (`committer`, the windowed collector under
rows-sha512: hm96-sha512 rows in frame-v3-sha512 trees, 64-byte roots), the int8 source at each linear H1 records, the
step-input port that gives each step's input ids, and the tap that keeps what the Commit acquired for H1
(`protocol_options.pouw_h1_commit.CommitTap`). Here in ``acquire`` because it builds the int8 source; `pouw_h1_commit`
holds the tap and the values and imports nothing from ``acquire``. On a server that commits already (serve-openai's
rows-sha512 collector), `tap` puts H1's tap on that Commit instead.

A MoE layer's routed experts (`engine.int8_experts.ExpertSite`) are no modules: each expert projection's site is
named as H1 names it (`pouw_h1_hooks.expert_names`, ``prefix.expert.projection``), and its Int8RowQuant_v1 output is
acquired under that name inside the call (`ExpertGemm.apply`), as `Int8ExpertsMethod` runs every expert on every forward:
an expert no token routed to is a call of zero rows, acquired as its zero rows [0, K] and scales [0]."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from verity_vllm.acquire.sources.int8_source import Int8Source
from verity_vllm.protocols.pouw.h1.commit import SCHEME, CommitTap, Sites, _EveryStep, site
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
    tap.port = PORT_NAMES[0]
    tap.ops = {op: site(m) for (_e, m), op in src.quant.items() if m in names}
    tap.tap(com)
    return tap


def committer(engine: Any, run_id: str) -> Any:
    """H1's committer on ``engine``, its run ``run_id``: the declared Commit's (`pipeline.commit`'s native_collect_v2b, built
    as `pipeline.served_match` builds it), the windowed collector under rows-sha512 (`pouw_h1_commit.SCHEME`). It retains
    its values nowhere unless VERITY_RETAIN says otherwise: H1 reads them from its tap, and the windowed collector retains
    none on the device."""
    from verity_vllm import settings
    from verity_vllm.commit.committer.native_collect import NativeCollectCommitter

    settings.setdefault("VERITY_RETAIN", "none")
    return NativeCollectCommitter(engine, run_id=run_id, arena_bytes=None, wrap=True, gather=True, native_hooks=True,
                                  auto_blocks=True, native_worker=True, commit_scheme=SCHEME)


#: what `envelope` reads of the engine's effective config
ENVELOPE = ("max_num_batched_tokens", "max_num_seqs", "max_model_len")


def envelope(engine: Any) -> dict[str, int]:
    """The workload the committer sizes its staging for (`NativeHostCommitter.size_arena`), from what ``engine`` executes
    (`engine.build.Engine.effective`, read back from its vLLM config): its batch, its token budget (a step's prompt rows at
    most, chunked prefill scheduling the rest) and its model length. ValueError when the engine doesn't say."""
    facts = getattr(engine, "effective", None) or {}
    if missing := [k for k in ENVELOPE if facts.get(k) is None]:
        raise ValueError(f"H1's Commit: the engine's effective config gives no {missing}, so its staging can't be sized")
    tokens = int(facts["max_num_batched_tokens"])
    return {"batch": int(facts["max_num_seqs"]), "max_prompt": tokens, "max_new": 1, "max_num_batched_tokens": tokens,
            "max_model_len": int(facts["max_model_len"])}


def port_tokens(com: Any, tokens: int) -> int:
    """The step-input port's T for a token budget of ``tokens``: rounded up to whole chunks of the committer's i64 ids,
    the only T `served_record.StepInputPort` takes, so H1's T = 64 rows is a port of 512 ids."""
    from verity_vllm.commit.served_record import StepInputPort

    per = int(com.chunk) // StepInputPort.IB
    return -(-int(tokens) // per) * per


def attach(engine: Any, layers: Sequence[Any], tap: CommitTap | None = None, run_id: str = "h1"
           ) -> tuple[Any, Any, CommitTap]:
    """The Commit on ``engine`` (`engine.build.build_engine`'s, or a rank's, `RankCommit.h1rank_commit`), installed: H1's
    committer (`committer`), refused unless it commits under rows-sha512; the int8 source (`source`); the step-input port,
    the step's one copy of its input ids (no module's input is acquired beside it), over the engine's token budget
    (`port_tokens`); the staging and its window ring sized for the engine's workload (`envelope`); and its `CommitTap`
    (``tap``, else a new one) on the port. Returns the committer, the source and the tap; ``committer.close()`` and
    ``tap.untap()`` remove them."""
    from verity_vllm.commit.served_record import PORT_NAMES, StepInputPort

    work = envelope(engine)
    com = committer(engine, run_id)
    if (got := getattr(com, "commit_scheme", None)) != SCHEME:
        com.close()
        raise ValueError(f"H1's committer {com.name} commits under {got}, not {SCHEME} (64-byte roots)")
    src = source(com, com.model, layers)
    com.attach_source(src)
    com.attach_source(StepInputPort(com, port_tokens(com, work["max_num_batched_tokens"])))
    com.plan_inputs = set()
    com.enable_windows(com.size_arena(work))
    tap = tap if tap is not None else CommitTap()
    tap.port = PORT_NAMES[0]
    tap.tap(com)
    com.install()
    return com, src, tap


class RankCommit(RankExtension):
    """`pouw_h1_ranks.RankExtension` with the Commit on the rank (`h1rank_commit`), the ``worker_extension_cls``
    `pouw_h1_ranks.COMMIT_EXTENSION`; here because it attaches the int8 source. The holder releases it with the rank
    (`RankExtension.h1rank_release`)."""

    def _h1rank_engine(self, model: Any, cp: Mapping[str, Any]) -> Any:
        """What the committer reads of an engine (`attach`), on this rank: its model, runner, vLLM config (and of it what
        `envelope` reads, as `engine.build.Engine.effective` holds it) and the checkpoint's pin ``cp`` (repo, revision)."""
        vc = self.vllm_config  # type: ignore[attr-defined]
        effective = {"max_num_batched_tokens": vc.scheduler_config.max_num_batched_tokens,
                     "max_num_seqs": vc.scheduler_config.max_num_seqs, "max_model_len": vc.model_config.max_model_len}
        return SimpleNamespace(llm=SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=vc)), runner=self.model_runner,  # type: ignore[attr-defined]
                               model=model, cp=dict(cp), effective=effective)

    def h1rank_commit(self, cp: Mapping[str, Any] | None = None, run_id: str = "h1") -> dict[str, Any]:
        """The Commit on this rank (`attach`, its run ``run_id`` named with the rank's first linear), its tap a
        `pouw_h1_ranks.RankTap` naming each of the rank's H1 linears as the window does; returns the committer's name, its
        scheme and the port it acquires the step's input ids under. A site the rank's source acquires that the window has
        no name for (a routed expert's included) is refused: under its rank-local name it would meet no call, or another
        rank's site of the same name."""
        from verity_vllm.protocols import pouw
        st = self._h1rank  # type: ignore[attr-defined]
        run: RankRun = st["run"]
        model, layers = self._h1rank_layers()
        names = {name: run.modules[run.lo + int(i)] for i, name in pouw._named(model, layers).items()}
        com, src, tap = attach(self._h1rank_engine(model, cp or {}), layers, tap=RankTap(names=names),
                               run_id=f"{run_id}/rank{run.lo}")
        if unnamed := sorted({*src.modules, *src.experts} - set(names)):
            com.close()
            tap.untap()
            raise ValueError(f"rank [{run.lo}, {run.hi}): the Commit acquires {len(unnamed)} sites the window has no name "
                             f"for, {unnamed[:4]}")
        st["commit"] = (com, tap)
        return {"lo": run.lo, "committer": com.name, "commit_scheme": com.commit_scheme, "port": tap.port}


__all__ = ["ENVELOPE", "H1Source", "RankCommit", "attach", "committer", "envelope", "port_tokens", "source", "tap"]
