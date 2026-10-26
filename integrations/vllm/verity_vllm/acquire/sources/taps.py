"""The taps a Commit attaches beside the record's sources: the norm scales (`norm_scale_source`), the router softmax
(`router_softmax_source`) and a TP rank's vocabulary range (`vocab_range_source`).  On by default; the row passes each one's option only
when its manifest names the tap's family (`pipeline.row_records.applicable_taps`), and NORM_TAP=0 etc. leave it off, for A/B only.  Whenever
the manifest names them: the blockwise FP8 GEMM's scale products (`scale_products_source`) and the exact-int8 quantizers' results
(`int8_source`, which claims its call boundaries from the call-boundary source); the single-rank Commit only.  One entry point
for the single-rank Commit (`pipeline.commit`: `attach`) and for the rank workers of a TP Commit: `pipeline.tp.commit` hands
`for_ranks(...)` to `pipeline.tp.rank_worker.tp2_commit_make` through the RPC, each rank calls it with its own committer and its rows of the
merged manifest, and `require_on_every_rank` fails the Commit closed unless every rank attached every tap that is on.  Each rank also
attaches the route source (`route_source`) when a capture map describes its model, and beside it the attention sources
(`attention_source`, which binds the sparse-MLA call boundaries of the rank's rows to its `attn` tap, and `dsv4_attention_source`,
which binds DeepSeek-V4's attention body); when the rank's rows name
`DeepGemmFp8Linear_v1` Calls, the block-FP8 linears' input quantization (`linear_quant_source`); when they name the MoE align
Definitions, the Marlin experts' align schedule (`moe_align_source`); the rank worker attaches the MoE block source (`moe_source`,
which claims the Marlin block's call boundaries when the manifest names no plane); and under tp-commit ``--memory-map`` the KV-slots source
(`kv_slots_source`, for the rank's memory map, never the record).  The route, linear quant and align sources: once one rank has
one, every rank must have it.  The single-rank `attach` also attaches the call-boundary source (`call_boundary_source`) whenever the
manifest names `call_boundaries` identities; `attached` names what it attaches for a row's manifest, constructing nothing (the CPU
gate's `call-boundaries`, `acquire.gate`).  Both `attach` and every rank's `attach_rank` arm a derived-S Program's prescribed split
count (`prescribe_idle_splits`), taps or none, and attach the narrowed-rows source (`narrowed_source`: the keys' and values' columns of
the last layer's prompt rows before the last) whenever the manifest names narrowed rows; once one rank has it, every rank must."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

from verity_vllm.acquire.sources import (attention_source, call_boundary_source, dsv4_attention_source, int8_source,
                                         kv_slots_source, linear_quant_source, moe_align_source, narrowed_source, norm_scale_source,
                                         route_source, router_softmax_source, scale_products_source, vocab_range_source)
from verity_vllm.acquire.sources.call_boundary_plan import Plan
from verity_vllm.acquire.sources import moe_source
from verity_vllm.config import option
from verity.ml.definitions.registry.topp_geometry import splits_for_v1
from verity_vllm.query import norm_scales as NS
from verity_vllm.query import router_softmax as RS
from verity_vllm.query import scale_products as SP
from verity_vllm.query import vocab_range as VR

#: (the TapSources field, the rank result's key, the source module, the manifest family, the source's class)
SOURCES = (("norm_tap_so", "norm_scale_source", norm_scale_source, NS.FAMILY, norm_scale_source.NormScaleSource),
           ("router_tap_so", "router_softmax_source", router_softmax_source, RS.FAMILY, router_softmax_source.RouterSoftmaxSource),
           ("vocab_range", "vocab_range_source", vocab_range_source, VR.FAMILY, vocab_range_source.VocabRangeSource))


@dataclass(frozen=True, kw_only=True)
class TapSources:
    """A Commit's tap options (`pipeline.commit`, `pipeline.tp.commit`); each refuses a required manifest without its family (exit 3)."""

    norm_tap_so: str | None = option("--norm-tap-so", default=None, help="the norm-scale tap op (ops/pod_norm_tap.sh): commit `<norm module>/"
                                     "norm_scale` for every norm_scales identity of the required manifest (acquire.sources.norm_scale_source)")
    router_tap_so: str | None = option("--router-tap-so", default=None, help="the router-softmax tap op (ops/pod_router_tap.sh): commit "
                                       "`moe/L<k>/router_softmax` for every router_softmax identity of the required manifest "
                                       "(acquire.sources.router_softmax_source)")
    vocab_range: bool = option("--vocab-range", default=False, action="store_true", help="commit `<embedding module>/vocab_range` for every "
                               "vocab_range identity of the required manifest (acquire.sources.vocab_range_source)")

    def given(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f, *_ in SOURCES if getattr(self, f)}


def attach(com: Any, manifest: Mapping[str, Any] | None, required_classes: tuple[str, ...], taps: TapSources, *,
           explicit_classes: bool = False, model: Any = None, programs: Sequence[str | None] = (),
           call_boundary_plan: Plan | None = None) -> tuple[str, ...]:
    """Attach each tap that is on to `com` (each refuses a manifest that does not name its family, exit 3), the call-boundary source when
    the manifest names call boundaries (its Programs found under `programs`, its plan `call_boundary_plan` when derived already) and the
    scale-products source when the manifest names its family; returns the required classes with their families added unless the caller
    gave them explicitly."""
    for f, _key, mod, _fam, _cls in SOURCES:
        required_classes, _src = mod.attach(com, getattr(taps, f), manifest, required_classes, explicit_classes=explicit_classes, model=model)
    required_classes, _src = call_boundary_source.attach(com, bool(call_boundary_source.manifest_sites(manifest)), manifest, required_classes,
                                                         explicit_classes=explicit_classes, model=model, programs=programs,
                                                         plan=call_boundary_plan)
    required_classes, _src = scale_products_source.attach(com, bool(SP.manifest_sites(manifest)), manifest, required_classes,
                                                          explicit_classes=explicit_classes, model=model)
    required_classes, _src = int8_source.attach(com, int8_source.applies(manifest), manifest, required_classes,
                                                explicit_classes=explicit_classes, model=model)
    narrowed_source.attach(com, manifest, model=model)
    prescribe_idle_splits(com, manifest)
    return required_classes


def prescribe_idle_splits(com: Any, manifest: Mapping[str, Any] | None) -> bool:
    """For a derived-S Program (its manifest names `sampler_splits`), a sampling step on which vLLM ran no split kernel (every row at
    top_p 1) binds the prescribed S `SplitsFor_v1(rows, num_SMs)` (`commit.committer.topp_splits.step_splits`); True when armed.  A
    manifest without the family (greedy, or B1's constant S) arms nothing, and a step whose split kernels ran binds their S either way."""
    if hasattr(com, "prescribe_idle_splits") and int(((manifest or {}).get("populations") or {}).get("by_family", {}).get("sampler_splits", 0) or 0):
        com.prescribe_idle_splits = splits_for_v1
        return True
    return False


def warmup(com: Any, on: bool) -> None:
    """Around the Commit's warm-up instrumented run, which is never a record (its root is discarded): bounded staging learns only
    (`learn_only`) and every source with placeholder words (the call-boundary source's `placeholder`) commits them instead of evaluating."""
    if getattr(com, "bounded", False):
        com.learn_only = on
    for src in getattr(com, "sources", None) or ():
        if hasattr(src, "placeholder"):
            src.placeholder = on


def attached(manifest: Mapping[str, Any] | None) -> list[str]:
    """The `short_name`s of the sources a row's Commit attaches through `attach` for its manifest of record, constructing none: each tap
    whose family the manifest names (the row passes a tap's option exactly then, `pipeline.row_records.applicable_taps`, and rebuilds a
    manifest of other tap policies) and, on one rank, the call-boundary source when the manifest names call boundaries no other source
    claims and the scale-products source when it names scale products (a TP Commit attaches neither); on several ranks, each attention
    source (`attention_source`, `dsv4_attention_source`) when a capture map that binds through it describes the manifest's model at its
    ranks (`route_source.manifest_map`: every rank attaches it, `attach_rank`), the MoE align source when the manifest names the
    align Definitions, and the MoE block source when the manifest names a call boundary it claims and no `moe_block_stream` plane (every
    rank attaches it, binding at the block's Calls: `pipeline.tp.rank_worker`, `moe_source.at_calls`); on any number of ranks, the
    narrowed-rows source when the manifest names narrowed rows."""
    ids = (manifest or {}).get("identities") or []
    fams = {r.get("family") for r in ids}
    one = int((manifest or {}).get("ranks") or 1) == 1
    cmap = None if one else route_source.manifest_map(manifest)
    return [*(cls.short_name for _f, _key, _mod, fam, cls in SOURCES if fam in fams),
            *([call_boundary_source.CallBoundarySource.short_name] if one and call_boundary_source.manifest_sites(manifest) else []),
            *([scale_products_source.ScaleProductsSource.short_name] if one and SP.manifest_sites(manifest) else []),
            *([int8_source.SHORT_NAME] if one and int8_source.applies(manifest) else []),
            *(src.short_name for mod, src in ((attention_source, attention_source.AttentionSource),
                                              (dsv4_attention_source, dsv4_attention_source.Dsv4AttentionSource))
              if cmap is not None and mod.binds(cmap)),
            *([moe_align_source.MoeAlignSource.short_name] if not one and moe_align_source.manifest_names_it(manifest) else []),
            *([moe_source.MoeBlockSource.short_name]
              if not one and not moe_source.binds_planes(manifest) and any(moe_source.claims(r) for r in ids) else []),
            *([narrowed_source.SHORT_NAME] if narrowed_source.applies(manifest) else [])]


def attach_rank(com: Any, model: Any, manifest_path: str | None, rank: int, *, kv_slots: bool = False, **given: Any) -> dict[str, Any]:
    """A TP rank's side: attach each tap that is on with the rank's rows of the merged manifest, and the route source when a capture
    map describes the rank's model (`route_source`, absent when none does), per source its stats, or its refusal; and arm the
    prescribed split count as `attach` does (`prescribe_idle_splits`: "SplitsFor_v1" when armed).  With ``kv_slots`` (tp-commit
    ``--memory-map``) also the KV-slots source, never the record: the rank's map reads it (`pipeline.tp.rank_replay.memory_map`)."""
    man = json.loads(Path(manifest_path).read_text()) if manifest_path else {}
    mine = dict(man, identities=[r for r in man.get("identities") or [] if int(r.get("rank", 0)) == int(rank)])
    out: dict[str, Any] = {"prescribe_idle_splits": "SplitsFor_v1"} if prescribe_idle_splits(com, mine) else {}
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
    if kv_slots:
        try:
            out["kv_slots_source"] = {"layers": len(kv_slots_source.attach(com, model).layers)}
        except (ValueError, RuntimeError, AttributeError) as e:
            out["kv_slots_source"] = {"error": repr(e)[:200]}
    return out


def for_ranks(taps: TapSources, *, kv_slots: bool = False) -> Any:
    """What `tp2_commit_make` calls on every rank: `attach_rank` with the taps that are on bound (picklable), also when none is, since
    every rank arms the prescribed split count and attaches the route source; ``kv_slots`` bound when on."""
    return partial(attach_rank, **taps.given(), **({"kv_slots": True} if kv_slots else {}))


def require_on_every_rank(made: list[dict[str, Any]], required_classes: tuple[str, ...], versions: dict[str, Any], taps: TapSources, *,
                          explicit_classes: bool) -> tuple[str, ...]:
    """The TP driver's side, once every rank made its committer: every tap that is on attached on every rank (else SystemExit, fail
    closed), its family added to the required classes unless they were given, and each rank's state recorded in `versions`; the route
    source (when any rank's model has a capture map), the linear quant source and the MoE align source (when any rank's rows name their
    Definitions) and the narrowed-rows source (when any rank's rows are narrowed) attached on every rank (else SystemExit); the attention
    hidden class every rank reports
    (`hidden_source.require_hidden_on_every_rank`)."""
    from verity_vllm.acquire.sources.hidden_source import require_hidden_on_every_rank

    required_classes = require_hidden_on_every_rank(made, required_classes, versions, explicit_classes=explicit_classes)
    for key in ("route_source", "linear_quant_source", "moe_align_source", "narrowed_source"):
        states = {int(m["rank"]): (m.get("tap_sources") or {}).get(key) for m in made}
        if any(states.values()):
            bad = sorted(r for r, v in states.items() if not v or v.get("error"))
            if bad:
                raise SystemExit(f"fail-closed: the {key.replace('_', ' ')} did not attach on ranks {bad} of {len(made)}: "
                                 f"{[states[r] for r in bad][:2]}")
            versions[key] = states
    for f, key, _mod, fam, _cls in SOURCES:
        if not getattr(taps, f):
            continue
        states = {int(m["rank"]): (m.get("tap_sources") or {}).get(key) for m in made}
        bad = sorted(r for r, v in states.items() if not v or v.get("error"))
        if bad or not states:
            raise SystemExit(f"fail-closed: the {fam} source did not attach on ranks {bad} of {len(made)}: {[states[r] for r in bad][:2]}")
        if not explicit_classes and fam not in required_classes:
            required_classes = (*required_classes, fam)
        versions[key] = states
        versions["required_classes"] = list(required_classes)
        print(f"[tp2commit] {fam} source attached on every rank ({states[min(states)]}); required classes {list(required_classes)}", flush=True)
    return required_classes


__all__ = ["SOURCES", "TapSources", "attach", "prescribe_idle_splits", "warmup", "attached", "attach_rank", "for_ranks", "require_on_every_rank"]
