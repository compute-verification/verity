"""What a Commit attaches beside the record's sources.  The single-rank Commit (`attach`) attaches from the capture plan's classes
(`program.predict.capture_plan.of_build`, taken as data): the exact-int8 quantizers' results (`int8_source`, which claims its
`call_boundary` classes from the call-boundary source); a plan with a call boundary no source claims, or a `scale_products` class, is
refused by name.  `attached` names what `attach` attaches for a plan, constructing nothing.  The norm-scale tap is not attached on
this road.

The rank workers of a TP Commit still read their rows of the merged manifest: `pipeline.tp.commit` hands `for_ranks(...)` to
`pipeline.tp.rank_worker.tp2_commit_make` through the RPC, each rank calls it with its own committer, and `require_on_every_rank`
fails the Commit closed unless every rank attached every tap that is on (`SOURCES`: the router softmax, a rank's vocabulary
range; the norm-scale source reads no manifest and is attached by neither side).  Each rank also attaches the route source (`route_source`) when a capture map describes its model; when the rank's
rows name `DeepGemmFp8Linear_v1` Calls, the block-FP8 linears' input quantization (`linear_quant_source`); when they name the MoE
align Definitions, the Marlin experts' align schedule (`moe_align_source`); and the narrowed-rows source (`narrowed_source`) whenever
its rows are narrowed.  The route, linear quant, align and narrowed sources: once one rank has one, every rank must have it."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from verity_vllm.acquire.sources import (
    call_boundary_source,
    int8_source,
    linear_quant_source,
    moe_align_source,
    narrowed_source,
    route_source,
    router_softmax_source,
    vocab_range_source,
)
from verity_vllm.acquire.sources.call_boundary_plan import SCALE_PRODUCTS
from verity_vllm.acquire.sources.hidden_source import require_hidden_on_every_rank
from verity_vllm.config import option
from verity_vllm.log import init_logger
from verity_vllm.query import router_softmax as RS
from verity_vllm.query import vocab_range as VR

logger = init_logger(__name__)

#: (the TapSources field, the rank result's key, the source module, the manifest family, the source's class)
SOURCES = (("router_tap_so", "router_softmax_source", router_softmax_source, RS.FAMILY, router_softmax_source.RouterSoftmaxSource),
           ("vocab_range", "vocab_range_source", vocab_range_source, VR.FAMILY, vocab_range_source.VocabRangeSource))


@dataclass(frozen=True, kw_only=True)
class TapSources:
    """A Commit's tap options (`pipeline.commit`, `pipeline.tp.commit`); each refuses a required manifest without its family (exit 3)."""

    router_tap_so: str | None = option("--router-tap-so", default=None, help="the router-softmax tap op (ops/pod_router_tap.sh): commit "
                                       "`moe/L<k>/router_softmax` for every router_softmax identity of the required manifest "
                                       "(acquire.sources.router_softmax_source)")
    vocab_range: bool = option("--vocab-range", default=False, action="store_true", help="commit `<embedding module>/vocab_range` for every "
                               "vocab_range identity of the required manifest (acquire.sources.vocab_range_source)")

    def given(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f, *_ in SOURCES if getattr(self, f)}


def attach(com: Any, plan: Sequence[Mapping[str, Any]], required_classes: tuple[str, ...], *, explicit_classes: bool = False,
           model: Any = None) -> tuple[str, ...]:
    """Attach to `com` the sources of the capture plan's classes (`program.predict.capture_plan.of_build`): the int8 source when the
    plan names Int8RowQuant call boundaries or Int8KvQuant module outputs.  A plan with a `call_boundary` class no source claims, or a
    `scale_products` class, is refused by name (exit 3): the call-boundary source's host plan reads a row manifest's identities and
    the scale-products source its block geometry, neither of which the capture plan carries.  Returns the required classes with the
    attached sources' capture classes added unless the caller gave them explicitly."""
    refused = [*(f"{c['op_path']}/{c['member']}: a call boundary no attached source claims (the call-boundary source's host plan "
                 "is not on the capture plan)" for c in call_boundary_source.sites(plan)),
               *(f"{c['op_path']}/{c['member']}: scale products (the capture plan carries no block geometry)"
                 for c in plan if c["capture"] == SCALE_PRODUCTS)]
    if refused:
        logger.error("[delta] REFUSE: %d classes of the capture plan have no acquisition source: %s", len(refused), refused[:4])
        raise SystemExit(3)
    required_classes, _src = int8_source.attach(com, int8_source.applies(plan), plan, required_classes,
                                                explicit_classes=explicit_classes, model=model)
    return required_classes


def warmup(com: Any, on: bool) -> None:
    """Around the Commit's warm-up instrumented run, which is never a record (its root is discarded): bounded staging learns only
    (`learn_only`) and every source with placeholder words (the call-boundary source's `placeholder`) commits them instead of evaluating."""
    if getattr(com, "bounded", False):
        com.learn_only = on
    for src in getattr(com, "sources", None) or ():
        if hasattr(src, "placeholder"):
            src.placeholder = on


def attached(plan: Sequence[Mapping[str, Any]]) -> list[str]:
    """The `short_name`s of the sources `attach` attaches for a capture plan's classes, constructing none: the int8 source when the plan
    names a class of its."""
    return [int8_source.SHORT_NAME] if int8_source.applies(plan) else []


def attach_rank(com: Any, model: Any, manifest_path: str | None, rank: int, **given: Any) -> dict[str, Any]:
    """A TP rank's side: attach each tap that is on with the rank's rows of the merged manifest, and the route source when a capture
    map describes the rank's model (`route_source`, absent when none does), per source its stats, or its refusal."""
    man = json.loads(Path(manifest_path).read_text()) if manifest_path else {}
    rows = man.get("identities")
    mine = dict(man, identities=[r for r in (rows if rows else ()) if int(r.get("rank", 0)) == int(rank)])
    out: dict[str, Any] = {}
    for f, key, mod, _fam, _cls in SOURCES:
        if not given.get(f):
            continue
        try:
            _classes, src = mod.attach(com, given[f], mine, (), explicit_classes=True, model=model)
            out[key] = {"stats": dict(src.stats)}
        except (SystemExit, ValueError, RuntimeError) as e:
            out[key] = {"error": repr(e)[:200]}
    try:
        rsrc = route_source.attach(com, model, rank, mine["identities"])
        if rsrc is not None:
            out["route_source"] = {"stats": dict(rsrc.stats)}
    except (ValueError, RuntimeError, AttributeError) as e:
        out["route_source"] = {"error": repr(e)[:200]}
    try:
        lq = linear_quant_source.attach(com, model, mine)
        if lq is not None:
            out["linear_quant_source"] = {"sites": len(lq.sites)}
    except (ValueError, RuntimeError, AttributeError) as e:
        out["linear_quant_source"] = {"error": repr(e)[:200]}
    try:
        ma = moe_align_source.attach(com, model, mine)
        if ma is not None:
            out["moe_align_source"] = {"layers": len(ma.layers)}
    except (ValueError, RuntimeError, AttributeError) as e:
        out["moe_align_source"] = {"error": repr(e)[:200]}
    try:
        nr = narrowed_source.attach(com, mine, model=model)
        if nr is not None:
            out["narrowed_source"] = {"sites": len(nr.sites)}
    except (SystemExit, ValueError, RuntimeError, AttributeError) as e:
        out["narrowed_source"] = {"error": repr(e)[:200]}
    return out


def for_ranks(taps: TapSources) -> Any:
    """What `tp2_commit_make` calls on every rank: `attach_rank` with the taps that are on bound (picklable), also when none is, since
    every rank arms the prescribed split count and attaches the route source."""
    return partial(attach_rank, **taps.given())


def _recorded(rank: Mapping[str, Any], key: str) -> Any:
    """What a rank recorded of the source `key` when it made its committer (`attach_rank`'s result); None where it recorded none."""
    sources = rank.get("tap_sources")
    return sources.get(key) if sources else None


def require_on_every_rank(made: list[dict[str, Any]], required_classes: tuple[str, ...], versions: dict[str, Any], taps: TapSources, *,
                          explicit_classes: bool) -> tuple[str, ...]:
    """The TP driver's side, once every rank made its committer: every tap that is on attached on every rank (else SystemExit, fail
    closed), its family added to the required classes unless they were given, and each rank's state recorded in `versions`; the route
    source (when any rank's model has a capture map), the linear quant source and the MoE align source (when any rank's rows name their
    Definitions) and the narrowed-rows source (when any rank's rows are narrowed) attached on every rank (else SystemExit); the attention
    hidden class every rank reports
    (`hidden_source.require_hidden_on_every_rank`)."""
    required_classes = require_hidden_on_every_rank(made, required_classes, versions, explicit_classes=explicit_classes)
    for key in ("route_source", "linear_quant_source", "moe_align_source", "narrowed_source"):
        states = {int(m["rank"]): _recorded(m, key) for m in made}
        if any(states.values()):
            bad = sorted(r for r, v in states.items() if not v or v.get("error"))
            if bad:
                raise SystemExit(f"fail-closed: the {key.replace('_', ' ')} did not attach on ranks {bad} of {len(made)}: "
                                 f"{[states[r] for r in bad][:2]}")
            versions[key] = states
    for f, key, _mod, fam, _cls in SOURCES:
        if not getattr(taps, f):
            continue
        states = {int(m["rank"]): _recorded(m, key) for m in made}
        bad = sorted(r for r, v in states.items() if not v or v.get("error"))
        if bad or not states:
            raise SystemExit(f"fail-closed: the {fam} source did not attach on ranks {bad} of {len(made)}: {[states[r] for r in bad][:2]}")
        if not explicit_classes and fam not in required_classes:
            required_classes = (*required_classes, fam)
        versions[key] = states
        versions["required_classes"] = list(required_classes)
        logger.info("[tp2commit] %s source attached on every rank (%s); required classes %s", fam, states[min(states)], list(required_classes))
    return required_classes


__all__ = ["SOURCES", "TapSources", "attach", "warmup", "attached", "attach_rank", "for_ranks", "require_on_every_rank"]
