"""Advice-driven serving-rows windows, for network accounting under arbitrary vLLM usage: a TP rank's window over the Programs a
registered compiler makes from the advice (everything serving made unpredictable, network-accounting's I1), with the registration
binding both.

The binding is two keys of the registration's window provenance (`binding`, `tp.serving_rows.BINDING`), beside the served Programs:

    compiler   {"name", "digest"}   the registered compiler: model and code fixed before the run
    advice     {"digest", "bits"}   the advice it read: the digest of its encoding, and its length in bits, the leak

Core's ``verity/registration/v1`` record carries the window's provenance as one public item (``window``), which the record's digest
covers and the coins commit to before the draw, so the draw comes after both are fixed and the record gains no field.  Both are
public: the compiler is registered, and the certifier charges the advice as the leak.

* **The prover** commits the window from a rank's sealed bundle over the compiler's Programs for the advice (`commit_window`:
  `tp.serving_rows.from_bundle` with ``programs`` and ``binding``).  Advice naming a request the Commit didn't serve fails there by
  name: the committed store holds none of its values.
* **The verifier** derives its own window from the record, the advice and its own copy of the compiler alone (`window_of_advice`:
  `tp.serving_rows.window_of`), and `tp.serving_rows.source` holds the registration's window to it, ``compiler`` and ``advice``
  among its codes.  It reads no committed value.

A compiler is what the window path reads of it: ``name``, ``digest`` and ``load(rank, world, requests)``, the rank's
`rank_replay.Loaded` for the requests the advice states, each ``{"request_id", "arrive_step", "prompt_len", "max_tokens"}`` with
its output length as ``max_tokens``.  `Traced` is today's stand-in: the Build's Programs of record, by the manifest's lookup of
record.  The ahead-of-time compiler takes its place under its own registered digest.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity.ml.circuits.codec import canonical_json

TRACED = "verity-vllm/advice-windows/traced-of-record/v0"
_DIGEST = re.compile(r"^[0-9a-f]{128}$")


@dataclass(frozen=True)
class Traced:
    """The stand-in compiler: each request's Program of record, from ``programs`` (rank -> the rank's Build dir) by the manifest's
    lookup of record (`rank_replay.load`), refused unless its digest is in ``of_record`` (rank -> digests).  Its digest names the
    rule and every rank's Programs of record, so a window over another Build's Programs names another compiler."""

    programs: Mapping[str, str]
    of_record: Mapping[str, Sequence[str]]
    name: str = TRACED

    @classmethod
    def for_commit(cls, commit_dir: str | os.PathLike, build_root: str | os.PathLike) -> Traced:
        """The stand-in over a TP Commit's Programs of record (its record's ``of_record``), each rank's under ``build_root/rank<r>``."""
        from verity_vllm.pipeline.tp import serving_rows as SRT

        rec = SRT.record(commit_dir)
        return cls({str(r): os.path.join(str(build_root), f"rank{r}") for r in range(int(rec["world"]))}, rec["of_record"])

    @property
    def digest(self) -> str:
        return hashlib.sha512(canonical_json({"name": self.name, "of_record": {str(r): sorted(map(str, d)) for r, d in
                                                                              sorted(self.of_record.items())}})).hexdigest()

    def load(self, rank: int, world: int, requests: Sequence[Mapping[str, Any]]) -> Any:
        from verity_vllm.pipeline.tp import rank_replay as RR

        return RR.load({"programs": {str(rank): self.programs[str(rank)]}, "of_record": {str(k): list(v) for k, v in self.of_record.items()},
                        "requests": [dict(q) for q in requests]}, int(rank), int(world), definitions=False)


def binding(compiler: Any, advice_digest: str, advice_bits: int) -> dict[str, Any]:
    """The window provenance's binding (`tp.serving_rows.BINDING`): the registered compiler's name and digest, and the advice's
    digest and length in bits."""
    for what, d in (("compiler", compiler.digest), ("advice", advice_digest)):
        if not (isinstance(d, str) and _DIGEST.match(d)):
            raise ValueError(f"the {what}'s digest is a SHA-512, 128 hex digits, not {d!r}")
    if isinstance(advice_bits, bool) or not isinstance(advice_bits, int) or advice_bits < 0:
        raise ValueError(f"the advice's length is a whole number of bits, not {advice_bits!r}")
    return {"compiler": {"name": str(compiler.name), "digest": compiler.digest}, "advice": {"digest": advice_digest, "bits": advice_bits}}


def _loaded(commit_dir: str | os.PathLike, rank: int, compiler: Any, requests: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], Any]:
    from verity_vllm.pipeline.tp import serving_rows as SRT

    rec, rank = SRT.record(commit_dir), int(rank)
    if not 0 <= rank < int(rec["world"]):
        raise ValueError(f"rank {rank} is not a rank of the world of {rec['world']}")
    return rec, compiler.load(rank, int(rec["world"]), requests)


def window_of_advice(commit_dir: str | os.PathLike, rank: int, partition_file: str | os.PathLike, compiler: Any,
                     requests: Sequence[Mapping[str, Any]], bound: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """The verifier's own window of rank ``rank`` of the TP Commit ``commit_dir``: the population over ``compiler``'s Programs for the
    advice's ``requests``, and what `tp.serving_rows.source` holds the registration to, ``bound`` (`binding`) among it."""
    from verity_vllm.pipeline.tp import serving_rows as SRT

    rec, p = _loaded(commit_dir, rank, compiler, requests)
    return SRT.window_of(rec, rank, partition_file, p, dict(bound))


def commit_window(commit_dir: str | os.PathLike, rank: int, partition_file: str | os.PathLike, out_dir: str | os.PathLike, compiler: Any,
                  requests: Sequence[Mapping[str, Any]], bound: Mapping[str, Any], **kw: Any) -> dict[str, Any]:
    """Rank ``rank``'s stand-in window into ``out_dir`` from its sealed bundle (`tp.serving_rows.from_bundle`, ``kw`` its options) over
    ``compiler``'s Programs for the advice's ``requests``, ``bound`` (`binding`) in its provenance."""
    from verity_vllm.pipeline.tp import serving_rows as SRT

    _rec, p = _loaded(commit_dir, rank, compiler, requests)
    return SRT.from_bundle(commit_dir, rank, partition_file, out_dir, programs=p, binding=dict(bound), **kw)
