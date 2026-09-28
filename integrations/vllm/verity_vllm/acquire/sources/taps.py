"""The taps a Commit attaches beside the record's sources.  Opt-in, never the record: the router softmax (`router_softmax_source`) and a TP
rank's vocabulary range (`vocab_range_source`).  Whenever the manifest names them: the blockwise FP8 GEMM's scale products
(`scale_products_source`; the single-rank Commit only).  One entry point for the single-rank Commit (`pipeline.commit`: `attach`) and for the
rank workers of a TP Commit: `pipeline.tp.commit` hands `for_ranks(...)` to `engine.rank_worker.tp2_commit_make` through the RPC, each
rank calls it with its own committer and its rows of the merged manifest, and `require_on_every_rank` fails the Commit closed unless
every rank attached every tap that is on."""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Mapping

from verity_vllm.acquire.sources import router_softmax_source, scale_products_source, vocab_range_source
from verity_vllm.query import router_softmax as RS
from verity_vllm.query import scale_products as SP
from verity_vllm.query import vocab_range as VR

#: (the rank result's key, the source module, the manifest family)
SOURCES = (("router_softmax_source", router_softmax_source, RS.FAMILY), ("vocab_range_source", vocab_range_source, VR.FAMILY))


def attach(com: Any, manifest: Mapping[str, Any] | None, required_classes: tuple[str, ...], *, router_tap_so: str | None = None,
           vocab_range: bool = False, explicit_classes: bool = False, model: Any = None) -> tuple[str, ...]:
    """Attach each tap that is on to `com` (each refuses a manifest that does not name its family, exit 3), and the scale-products source
    when the manifest names its family; returns the required classes with the taps' families added unless the caller gave them explicitly."""
    for (_key, mod, _fam), arg in zip(SOURCES, (router_tap_so, vocab_range)):
        required_classes, _src = mod.attach(com, arg, manifest, required_classes, explicit_classes=explicit_classes, model=model)
    required_classes, _src = scale_products_source.attach(com, bool(SP.manifest_sites(manifest)), manifest, required_classes,
                                                          explicit_classes=explicit_classes, model=model)
    return required_classes


def attach_rank(com: Any, model: Any, manifest_path: str | None, rank: int, *, router_tap_so: str | None = None,
                vocab_range: bool = False) -> dict[str, Any]:
    """A TP rank's side: attach each tap that is on with the rank's rows of the merged manifest; per source its stats, or its refusal."""
    man = json.loads(Path(manifest_path).read_text()) if manifest_path else {}
    mine = dict(man, identities=[r for r in man.get("identities") or [] if int(r.get("rank", 0)) == int(rank)])
    out: dict[str, Any] = {}
    for (key, mod, _fam), arg in zip(SOURCES, (router_tap_so, vocab_range)):
        if not arg:
            continue
        try:
            _classes, src = mod.attach(com, arg, mine, (), explicit_classes=True, model=model)
            out[key] = {"stats": dict(src.stats)}
        except (SystemExit, ValueError, RuntimeError) as e:
            out[key] = {"error": repr(e)[:200]}
    return out


def for_ranks(router_tap_so: str | None = None, vocab_range: bool = False) -> Any:
    """What `tp2_commit_make` calls on every rank (`attach_rank` with the taps bound; picklable), or None when no tap is on."""
    return partial(attach_rank, router_tap_so=router_tap_so, vocab_range=vocab_range) if (router_tap_so or vocab_range) else None


def require_on_every_rank(made: list[dict[str, Any]], required_classes: tuple[str, ...], versions: dict[str, Any], *, router_tap_so: str | None,
                          vocab_range: bool, explicit_classes: bool) -> tuple[str, ...]:
    """The TP driver's side, once every rank made its committer: every tap that is on attached on every rank (else SystemExit, fail
    closed), its family added to the required classes unless they were given, and each rank's state recorded in `versions`."""
    for (key, _mod, fam), on in zip(SOURCES, (router_tap_so, vocab_range)):
        if not on:
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


__all__ = ["SOURCES", "attach", "attach_rank", "for_ranks", "require_on_every_rank"]
