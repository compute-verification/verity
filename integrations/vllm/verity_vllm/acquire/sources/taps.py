"""The taps a Commit attaches beside the record's sources: the norm scales (`norm_scale_source`), the router softmax
(`router_softmax_source`) and a TP rank's vocabulary range (`vocab_range_source`).  On by default; the row passes each one's option only
when its manifest names the tap's family (`pipeline.row_records.applicable_taps`), and NORM_TAP=0 etc. leave it off, for A/B only.  One
entry point for the single-rank Commit (`pipeline.commit`: `attach`) and for the rank workers of a TP Commit: `pipeline.tp.commit` hands
`for_ranks(...)` to `engine.rank_worker.tp2_commit_make` through the RPC, each rank calls it with its own committer and its rows of the
merged manifest, and `require_on_every_rank` fails the Commit closed unless every rank attached every tap that is on."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping

from verity_vllm.acquire.sources import norm_scale_source, router_softmax_source, vocab_range_source
from verity_vllm.config import option
from verity_vllm.query import norm_scales as NS
from verity_vllm.query import router_softmax as RS
from verity_vllm.query import vocab_range as VR

#: (the TapSources field, the rank result's key, the source module, the manifest family)
SOURCES = (("norm_tap_so", "norm_scale_source", norm_scale_source, NS.FAMILY),
           ("router_tap_so", "router_softmax_source", router_softmax_source, RS.FAMILY),
           ("vocab_range", "vocab_range_source", vocab_range_source, VR.FAMILY))


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
           explicit_classes: bool = False, model: Any = None) -> tuple[str, ...]:
    """Attach each tap that is on to `com` (each refuses a manifest that does not name its family, exit 3); returns the required classes
    with the taps' families added unless the caller gave them explicitly."""
    for f, _key, mod, _fam in SOURCES:
        required_classes, _src = mod.attach(com, getattr(taps, f), manifest, required_classes, explicit_classes=explicit_classes, model=model)
    return required_classes


def attach_rank(com: Any, model: Any, manifest_path: str | None, rank: int, **given: Any) -> dict[str, Any]:
    """A TP rank's side: attach each tap that is on with the rank's rows of the merged manifest; per source its stats, or its refusal."""
    man = json.loads(Path(manifest_path).read_text()) if manifest_path else {}
    mine = dict(man, identities=[r for r in man.get("identities") or [] if int(r.get("rank", 0)) == int(rank)])
    out: dict[str, Any] = {}
    for f, key, mod, _fam in SOURCES:
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
    for f, key, _mod, fam in SOURCES:
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


__all__ = ["SOURCES", "TapSources", "attach", "attach_rank", "for_ranks", "require_on_every_rank"]
