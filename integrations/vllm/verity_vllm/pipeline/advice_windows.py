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
`rank_replay.Loaded` for the requests the advice states (`requests_of`), each ``{"request_id", "arrive_step", "prompt_len",
"max_tokens"}`` with its output length as ``max_tokens``.  `Traced` is today's stand-in: the Build's Programs of record, by the
manifest's lookup of record.  The ahead-of-time compiler takes its place under its own registered digest.

The advice is I1's (``codec``, default `I1`): `of_advice` reads its bytes, which must be its canonical encoding, and binds their
SHA-512 and the advice's length, ``total_bits`` (the session's content advice).  `advice_of_workload` is the stand-in's advice, the prover's, for a fixed-length
workload; a served run's is ADVICE-SRC's, from the scheduler's step trace.

    verity-vllm advice-windows COMMIT_OUT --advice FILE --partition F --out O [--ranks 0,1] [--build-root B]
        [--write-from-workload WORKLOAD [--drop r1 ...]]
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.ml.circuits.codec import canonical_json
from verity_vllm.config import option

TRACED = "verity-vllm/advice-windows/traced-of-record/v0"
I1 = "verity.core.protocols.certifier.advice"
_DIGEST = re.compile(r"^[0-9a-f]{128}$")


def codec_of(codec: Any = None) -> Any:
    return codec if codec is not None else importlib.import_module(I1)


def requests_of(advice: Any) -> list[dict[str, Any]]:
    """The compiler's requests from I1 advice: each request's id, its arrival (the first step with a row of it), its prompt length
    and its output length as ``max_tokens``.  A request with no row is refused: the advice names work it never scheduled."""
    first: dict[int, int] = {}
    for s, step in enumerate(advice.steps):
        for row in step.rows:
            first.setdefault(int(row.request), s)
    if unscheduled := [q.id for i, q in enumerate(advice.requests) if i not in first]:
        raise ValueError(f"the advice names requests no step schedules: {unscheduled}")
    return [{"request_id": str(q.id), "arrive_step": first[i], "prompt_len": int(q.prompt_len), "max_tokens": int(q.output_len)}
            for i, q in enumerate(advice.requests)]


def of_advice(compiler: Any, data: bytes, codec: Any = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The advice's bytes read by ``codec`` (`codec_of`): the compiler's requests (`requests_of`) and the binding (`binding`), the
    SHA-512 of the bytes, refused unless they are the advice's canonical encoding, and its length, ``codec.total_bits``."""
    c = codec_of(codec)
    advice = c.decode(data)
    if c.encode(advice) != data:
        raise ValueError("the advice's bytes are not its canonical encoding")
    return requests_of(advice), binding(compiler, hashlib.sha512(data).hexdigest(), int(c.total_bits(advice)))


def advice_of_workload(doc: Mapping[str, Any], codec: Any = None, drop: Sequence[str] = ()) -> Any:
    """The stand-in's advice, the prover's: a fixed-length workload's requests (``r<position>``, as ``tp-commit`` names them, less
    ``drop``) at their declared arrivals, each a whole-prompt prefill row, then one decode row a step until its ``max_tokens``,
    finished by length; a step's rows in the runner's order (decode rows first), the requests numbered in order of their first
    row, and each kept output released in its step's bucket.  A step the workload leaves idle is refused: the advice's steps
    are the forwards that ran, so its arrivals would not be the Commit's engine steps."""
    c = codec_of(codec)
    reqs = [(f"r{i}", int(q.get("arrive_step") or 0), len(q["prompt_token_ids"]), int(q["max_tokens"])) for i, q in enumerate(doc["requests"])]
    reqs = [q for q in reqs if q[0] not in set(drop)]
    steps: list[list[tuple[int, str, int, int]]] = [[] for _ in range(max((a + t for _, a, _, t in reqs), default=0))]
    for i, (_rid, a, lp, t) in enumerate(reqs):
        steps[a].append((i, "prefill", 0, lp))
        for j in range(1, t):
            steps[a + j].append((i, "decode", lp + j - 1, 1))
    if idle := [s for s, rows in enumerate(steps) if not rows]:
        raise ValueError(f"the workload leaves engine steps {idle} idle: the advice's steps are the forwards that ran")
    steps = [sorted(rows, key=lambda x: (x[3], x[0])) for rows in steps]
    order = list(dict.fromkeys(x[0] for rows in steps for x in rows))
    index = {i: k for k, i in enumerate(order)}
    return c.Advice(tuple(c.RequestAdvice(reqs[i][0], reqs[i][2], reqs[i][3], "length") for i in order),
                    tuple(c.StepAdvice(tuple(c.Row(index[i], kind, start, n) for i, kind, start, n in rows)) for rows in steps),
                    tuple(s for s, rows in enumerate(steps) for _ in rows))


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


@dataclasses.dataclass(frozen=True, kw_only=True)
class Options:
    """verity-vllm advice-windows COMMIT_OUT --advice FILE --partition F --out O: a deferred TP Commit's stand-in windows, ``O/rank<r>/``
    each, from its ranks' sealed bundles over the stand-in compiler's Programs (`Traced`) for the advice (`commit_window`)."""
    commit_out: str = option("commit_out", help="the deferred TP Commit's --out (runs.jsonl, versions.json, replay_deferred.json, its bundles)")
    advice: str = option("--advice", help="the advice: I1's canonical encoding")
    partition: str = option("--partition", help="the verifier's partition file: {templates, law, scope?}")
    out: str = option("--out", help="the windows' directory: rank<r>/ each")
    build_root: str = option("--build-root", help="the stand-in compiler's Programs: each rank's Programs of record in rank<r>/")
    ranks: str | None = option("--ranks", default=None, help="comma-separated ranks (default: every rank of the world)")
    write_from_workload: str | None = option("--write-from-workload", default=None,
                                             help="first write --advice as the stand-in's advice for this fixed-length workload (`advice_of_workload`)")
    drop: list[str] = option("--drop", action="append", default=[], help="with --write-from-workload: leave this request out (repeatable)")
    workers: int | None = option("--workers", type=int, default=None, help="processes hashing the rows (default: min(16, CPUs))")


def main(o: Options) -> int:
    from verity_vllm.pipeline.tp import commit_replay as CR
    from verity_vllm.pipeline.tp import serving_rows as SRT

    try:
        c = codec_of()
        if o.write_from_workload:
            Path(o.advice).write_bytes(c.encode(advice_of_workload(json.loads(Path(o.write_from_workload).read_text()), c, o.drop)))
        rec = SRT.record(o.commit_out)
        compiler = Traced.for_commit(o.commit_out, o.build_root)
        reqs, bound = of_advice(compiler, Path(o.advice).read_bytes(), c)
        for r in [int(x) for x in o.ranks.split(",")] if o.ranks else range(int(rec["world"])):
            got = commit_window(o.commit_out, r, o.partition, Path(o.out) / f"rank{r}", compiler, reqs, bound, workers=o.workers)
            print(f"[advice-windows] rank {r}: stand-in window of {got['population']} units over {len(reqs)} requests, advice "
                  f"{bound['advice']['digest'][:16]} ({bound['advice']['bits']} bits), compiler {compiler.digest[:16]}, registration "
                  f"{str(got['registration_digest'])[:16]}", flush=True)
    except (CR.Refused, LookupError, ValueError, RuntimeError) as e:
        print(f"[advice-windows] REFUSED: {e}", flush=True)
        return 2
    return 0
