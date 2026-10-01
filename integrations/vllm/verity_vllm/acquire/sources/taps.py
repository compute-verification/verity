"""The taps a Commit attaches beside the record's sources: the norm scales (`norm_scale_source`), the router softmax
(`router_softmax_source`) and a TP rank's vocabulary range (`vocab_range_source`).  On by default; the row passes each one's option only
when its manifest names the tap's family (`pipeline.row_records.applicable_taps`), and NORM_TAP=0 etc. leave it off, for A/B only.  Whenever
the manifest names them: the blockwise FP8 GEMM's scale products (`scale_products_source`; the single-rank Commit only).  One entry point
for the single-rank Commit (`pipeline.commit`: `attach`) and for the rank workers of a TP Commit: `pipeline.tp.commit` hands
`for_ranks(...)` to `engine.rank_worker.tp2_commit_make` through the RPC, each rank calls it with its own committer and its rows of the
merged manifest, and `require_on_every_rank` fails the Commit closed unless every rank attached every tap that is on.  The single-rank
`attach` also attaches the call-boundary source (`call_boundary_source`) whenever the manifest names `call_boundaries` identities, and
just before it the served taps (`served_tap`: the logits tap, `logits_tap_source`, with `--logits-tap 1`; the norm body tap,
`norm_body_tap_source`, with `--norm-body-tap 1`) whenever they claim some of them, whose words the call-boundary source commits;
`attached` names what it attaches for a row's manifest, constructing nothing (the CPU gate's `call-boundaries`, `acquire.gate`)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

from verity_vllm.acquire.sources import (call_boundary_source, logits_tap_source, norm_body_tap_source, norm_scale_source, router_softmax_source,
                                         scale_products_source, vocab_range_source)
from verity_vllm.acquire.sources.call_boundary_plan import Plan
from verity_vllm.config import option
from verity_vllm.program.registry.topp_split import splits_for_v1
from verity_vllm.query import norm_scales as NS
from verity_vllm.query import router_softmax as RS
from verity_vllm.query import scale_products as SP
from verity_vllm.query import vocab_range as VR

#: (the TapSources field, the rank result's key, the source module, the manifest family, the source's class)
SOURCES = (("norm_tap_so", "norm_scale_source", norm_scale_source, NS.FAMILY, norm_scale_source.NormScaleSource),
           ("router_tap_so", "router_softmax_source", router_softmax_source, RS.FAMILY, router_softmax_source.RouterSoftmaxSource),
           ("vocab_range", "vocab_range_source", vocab_range_source, VR.FAMILY, vocab_range_source.VocabRangeSource))
#: the served taps (`served_tap`), each (its module, its class), in the order they attach
SERVED = ((logits_tap_source, logits_tap_source.LogitsTapSource), (norm_body_tap_source, norm_body_tap_source.NormBodyTapSource))


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
    logits_tap: int = logits_tap_source.option_field()
    norm_body_tap: int = norm_body_tap_source.option_field()

    def given(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f, *_ in SOURCES if getattr(self, f)}


def claiming(opts: Any) -> None:
    """The served taps' claims on call boundaries (`call_boundary_source.CLAIMS`) from a command's `--logits-tap` and `--norm-body-tap`."""
    logits_tap_source.claiming(bool(getattr(opts, "logits_tap", 0)))
    norm_body_tap_source.claiming(bool(getattr(opts, "norm_body_tap", 0)))


def attach(com: Any, manifest: Mapping[str, Any] | None, required_classes: tuple[str, ...], taps: TapSources, *,
           explicit_classes: bool = False, model: Any = None, programs: Sequence[str | None] = (),
           call_boundary_plan: Plan | None = None) -> tuple[str, ...]:
    """Attach each tap that is on to `com` (each refuses a manifest that does not name its family, exit 3), the served taps when they claim
    call boundaries, the call-boundary source when the manifest names others or a served tap is attached (its Programs found under
    `programs`, its plan `call_boundary_plan` when derived already; a plan of identities a served tap commits refuses, exit 3) and the
    scale-products source when the manifest names its family; returns the required classes with their families added unless the caller
    gave them explicitly."""
    claiming(taps)
    for f, _key, mod, _fam, _cls in SOURCES:
        required_classes, _src = mod.attach(com, getattr(taps, f), manifest, required_classes, explicit_classes=explicit_classes, model=model)
    served = []
    for mod, _cls in SERVED:
        required_classes, tap = mod.attach(com, manifest, required_classes, explicit_classes=explicit_classes, model=model)
        served += [tap] if tap is not None else []
    required_classes, _cb = call_boundary_source.attach(com, bool(call_boundary_source.manifest_sites(manifest) or served), manifest,
                                                        required_classes, explicit_classes=explicit_classes, model=model, programs=programs,
                                                        plan=call_boundary_plan, taps=served)
    required_classes, _src = scale_products_source.attach(com, bool(SP.manifest_sites(manifest)), manifest, required_classes,
                                                          explicit_classes=explicit_classes, model=model)
    if hasattr(com, "prescribe_idle_splits") and int(((manifest or {}).get("populations") or {}).get("by_family", {}).get("sampler_splits", 0) or 0):
        com.prescribe_idle_splits = splits_for_v1            # a derived-S Program: a step vLLM ran without top-p binds its prescribed S
    return required_classes


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
    manifest of other tap policies) and, on one rank, each served tap when it claims call boundaries it can commit, the call-boundary
    source when the manifest names call boundaries no other source claims or a served tap is attached, and the scale-products source when
    it names scale products (a TP Commit attaches none of them)."""
    fams = {r.get("family") for r in (manifest or {}).get("identities") or []}
    one = int((manifest or {}).get("ranks") or 1) == 1
    served = [cls.short_name for mod, cls in SERVED if one and (ids := mod.manifest_sites(manifest)) and mod.refusal(manifest, ids) is None]
    return [*(cls.short_name for _f, _key, _mod, fam, cls in SOURCES if fam in fams), *served,
            *([call_boundary_source.CallBoundarySource.short_name] if one and (call_boundary_source.manifest_sites(manifest) or served) else []),
            *([scale_products_source.ScaleProductsSource.short_name] if one and SP.manifest_sites(manifest) else [])]


def attach_rank(com: Any, model: Any, manifest_path: str | None, rank: int, **given: Any) -> dict[str, Any]:
    """A TP rank's side: attach each tap that is on with the rank's rows of the merged manifest; per source its stats, or its refusal."""
    man = json.loads(Path(manifest_path).read_text()) if manifest_path else {}
    mine = dict(man, identities=[r for r in man.get("identities") or [] if int(r.get("rank", 0)) == int(rank)])
    out: dict[str, Any] = {}
    for f, key, mod, _fam, _cls in SOURCES:
        if not given.get(f):
            continue
        try:
            _classes, src = mod.attach(com, given[f], mine, (), explicit_classes=True, model=model)
            out[key] = {"stats": dict(src.stats)}
        except (SystemExit, ValueError, RuntimeError) as e:
            out[key] = {"error": repr(e)[:200]}
    return out


def for_ranks(taps: TapSources) -> Any:
    """What `tp2_commit_make` calls on every rank (`attach_rank` with the taps bound; picklable), or None when no tap is on."""
    return partial(attach_rank, **taps.given()) if taps.given() else None


def require_on_every_rank(made: list[dict[str, Any]], required_classes: tuple[str, ...], versions: dict[str, Any], taps: TapSources, *,
                          explicit_classes: bool) -> tuple[str, ...]:
    """The TP driver's side, once every rank made its committer: every tap that is on attached on every rank (else SystemExit, fail
    closed), its family added to the required classes unless they were given, and each rank's state recorded in `versions`."""
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


__all__ = ["SOURCES", "SERVED", "TapSources", "claiming", "attach", "warmup", "attached", "attach_rank", "for_ranks", "require_on_every_rank"]
