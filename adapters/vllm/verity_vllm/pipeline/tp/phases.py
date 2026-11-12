"""`tp-commit`'s phases on its row's timeline (`pipeline.timeline`: ``VERITY_TIMELINE``, else nothing), so a held-idle reading can say
where a TP Commit's leased GPU time goes.  The single-GPU Commit's names (`pipeline.commit`) for the same steps, each span ended (``ok``)
however its step ends:

* ``engine.build``: the engine, through its ranks' answer to the probe (``engine``);
* ``prep.warmup_control<i>``: each control warm-up (``warmups``);
* ``prep.committer_setup``: the ranks' committers, which register the weights and, on the ranks, digest the weights of record for the
  weights pin (no span of their own);
* ``prep.warmup_instrumented``: the warm-up instrumented run (`pipeline.tp.warmup`);
* ``work.pair<p>.<arm>``: each measured run, an instrumented one from its ranks' install to the end of their finalize, which holds the
  GPUs for their checks and sampled replay.  A leased last run's finalize gives the GPUs back on the way (`pipeline.tp.lease`), so its
  span runs past the lease's end, where the lease's reader clips it;
* ``validate.value_check``: the value check's extra instrumented run.

Never a ``gpu_lease.*`` span: the reader takes a Commit window with one for a one-GPU lease, and reads a TP Commit's lease, on every
rank's GPU, from ``gpu_lease.started`` / ``gpu_lease.released``.

Nothing here depends on the world size but ``tp``: at one rank, vLLM's worker runs in this process (`engine.env`), and
`collective_rpc` calls it here.  A Commit with a timeline of its own passes it (``tl``), with its restart reason and its log tag."""
from __future__ import annotations

import time
from typing import Any, Callable

from verity_vllm.pipeline.timeline import Timeline

TL = Timeline.from_env("tp_commit", hot=False)
RESTART_REASON = "fresh process (one tp-commit per Commit stage)"


def engine(build: Callable[[], Any], tp: int, rpc_timeout: float, *, model: str, enforce_eager: Any, tl: Timeline | None = None,
           restart_reason: str = RESTART_REASON, **fields: Any) -> tuple[Any, float, list]:
    """``engine.build`` on ``tl`` (``TL`` when None): the engine (``build()``) to its ranks' answer to the probe, ``tp`` of them or
    SystemExit (fail closed) -> (the LLM, the build's seconds, the ranks' answers).  ``restart_reason``: why this Commit built an
    engine rather than reused one, which `timeline.report` counts."""
    with (TL if tl is None else tl).span("engine.build", tp=tp, **fields) as end:
        t0 = time.perf_counter()
        llm = build()
        build_s = time.perf_counter() - t0
        end.update(build_s=round(build_s, 3), model=model, enforce_eager=enforce_eager, engine_reused=False, restart_reason=restart_reason)
        topo = llm.collective_rpc("tp2_probe", timeout=rpc_timeout)
        if len(topo) != tp:
            raise SystemExit(f"fail-closed: {len(topo)} ranks answered the probe, tp={tp}")
    return llm, build_s, topo


def warmups(n: int, run: Callable[[int], dict], *, tl: Timeline | None = None, tag: str = "tp2commit") -> None:
    """The ``n`` control warm-ups, ``run(i)``, each in ``prep.warmup_control<i>`` on ``tl`` (``TL`` when None), each logged under
    ``tag``."""
    for i in range(n):
        with (TL if tl is None else tl).span(f"prep.warmup_control{i}"):
            r = run(i)
        print(f"[{tag}] warm-up control {i}: {r['wall_s']:.3f}s {r['iterations']} it", flush=True)


__all__ = ["RESTART_REASON", "TL", "engine", "warmups"]
