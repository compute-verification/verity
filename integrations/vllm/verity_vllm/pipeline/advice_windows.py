"""Advice-driven serving-rows windows, for network accounting under arbitrary vLLM usage: a TP rank's window over the Programs a
registered compiler makes from the advice (everything serving made unpredictable, network-accounting's I1), with the registration
binding both.

The binding is two keys of the registration's window provenance (`binding`, `tp.serving_rows.BINDING`), beside the served Programs:

    compiler   {"name", "digest"}   the registered compiler: model and code fixed before the run
    advice     {"root", "bits"}     the advice it read, private: the root of its hiding commitment, and its length in bits

Core's ``verity/registration/v1`` record carries the window's provenance as one public item (``window``), which the record's digest
covers and the coins commit to before sampling, so the sample comes after both are fixed and the record gains no field.  The advice
is not published: its canonical encoding is committed as hiding rows (`commit_advice`: ``ROW_BITS``-bit rows,
zero-padded, under `verity.ml.commitments.advice`, hm96-sha512 leaves in one frame-v3-sha512 tree, its domain bound to the
compiler), and only the root and the length are public.  The rows and their salts are the prover's (`opening`).

* **The prover** commits the window from a rank's sealed bundle over the compiler's Programs for the advice (`commit_window`:
  `tp.serving_rows.from_bundle` with ``programs`` and ``binding``).  Advice naming a request the Commit didn't serve fails there by
  name: the committed store holds none of its values.
* **The verifier** derives its own window from the record, the advice's opening and its own copy of the compiler alone
  (`of_opening`, `window_of_advice`: `tp.serving_rows.window_of`), holding the root it recomputes from the opening, and
  `tp.serving_rows.source` holds the registration's window to it, ``compiler`` and ``advice`` among its codes.  It reads no
  committed value.  The opening shows the verifier the advice: that is the named gap, until the wiring between the compiler's
  public units is proved in zero knowledge instead.

A compiler is what the window path reads of it: ``name``, ``digest``, ``transfer_gates`` (``()`` if it has none) and
``load(rank, world, requests)``, the
rank's `rank_replay.Loaded` for the requests the advice states (`requests_of`), each ``{"request_id", "arrive_step",
"prompt_len", "max_tokens"}`` with its output length as ``max_tokens``.  `Traced` is today's stand-in: the Build's Programs of
record, by the manifest's lookup of record.  The ahead-of-time compiler takes its place under its own registered digest.

**Transfer gates**: the units holding transfer gates, the I/O shell's ingress parse and egress render units, are
proved with selection All, never sampled; every other unit is sampled as before.  The compiler lists them (``transfer_gates``: the
templates whose units hold them, which its digest covers), and a ``work:K`` window samples by the work table
`serving_rows.work_table_of` derives from that list and the window's own counts: each transfer-gate template works 0 with its
floor all its units, so its whole stratum is in the sample, beside the rest's strata at the stratified share of K.  Core's work law
(`draw.work_k`) is the All stratum, with no new verifier form; both sides derive the table, the registration's law is expanded
under it (R1), and the Stream registers with it.  A partition that leaves a transfer-gate template out, or samples it by another law,
is refused by name.  Until COMPILER lists the I/O shell's units, the stand-in's list is given (``--transfer-gate``), a template
C-Flock proves today: the toy Commit's ``RoPEHead_v1{D=64}`` (its egress render, ``TokenSelect_v1``, has no C-Flock lowering).

The advice is I1's (``codec``, default `I1`): `of_advice` reads its bytes, which must be its canonical encoding, commits them and
binds the root and the advice's length, ``total_bits``.  `advice_of_workload` is the stand-in's advice, the prover's, for a
fixed-length workload; a served run's is ADVICE-SRC's, from the scheduler's step trace.

    verity-vllm advice-windows COMMIT_OUT --advice FILE --opening FILE --partition F --out O --build-root B [--ranks 0,1]
        [--transfer-gate TEMPLATE ...] [--write-from-workload WORKLOAD [--drop r1 ...]]
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
from verity.ml.commitments import advice as AC
from verity.ml.windows import interface as IF
from verity_vllm.config import option, workload_arg

TRACED = "verity-vllm/advice-windows/traced-of-record/v0"
I1 = "verity.core.protocols.certifier.advice"
SCHEME = IF.ADVICE_SCHEME
ROW_BITS = IF.ADVICE_ROW_BITS
_DIGEST = re.compile(r"^[0-9a-f]{128}$")


def codec_of(codec: Any = None) -> Any:
    return codec if codec is not None else importlib.import_module(I1)


def _rows(advice: Any) -> dict[int, list[tuple[int, Any]]]:
    rows: dict[int, list[tuple[int, Any]]] = {i: [] for i in range(len(advice.requests))}
    for s, step in enumerate(advice.steps):
        for row in step.rows:
            rows.setdefault(int(row.request), []).append((s, row))
    return rows


def refusals(advice: Any) -> dict[int, str]:
    """Why each request the stand-in can't compile is refused, by request index.  A request's Program of record runs its whole prompt
    in one row from position 0, then one decode row a step until its last kept output; the one row async scheduling runs past that
    output samples nothing kept and is no Program row.  So a request with no row, a prefix-cache hit, a chunked prefill, a
    preemption or a gap is refused by name (COMPILER's v0 rules, `predict.steps.shape_of`)."""
    out: dict[int, str] = {}
    for i, rows in _rows(advice).items():
        q = advice.requests[i]
        lp, t = int(q.prompt_len), int(q.output_len)
        if not rows:
            out[i] = "no step schedules it"
            continue
        s0, first = rows[0]
        if first.cache_hit or first.start:
            out[i] = f"a prefix-cache hit of {first.cache_hit or first.start} tokens at step {s0}"
        elif first.n != lp:
            out[i] = f"a chunked prefill, {first.n} of its {lp} prompt tokens at step {s0}"
        elif any(r.preempted for _, r in rows):
            out[i] = "a preemption"
        elif bad := next(((s, r) for k, (s, r) in enumerate(rows[1:], 1) if (s, r.start, r.n) != (s0 + k, lp + k - 1, 1)), None):
            out[i] = f"its row at step {bad[0]} is (start, n) = ({bad[1].start}, {bad[1].n}), not the next decode"
        elif not t <= len(rows) <= t + 1:
            out[i] = f"{len(rows)} rows for {t} kept outputs"
    return out


def requests_of(advice: Any) -> list[dict[str, Any]]:
    """The compiler's requests from I1 advice: each request's id, its arrival (the first step with a row of it), its prompt length
    and its output length as ``max_tokens``.  A request the stand-in can't compile (`refusals`) refuses the advice."""
    if refused := refusals(advice):
        why = "; ".join(f"{advice.requests[i].id}: {w}" for i, w in sorted(refused.items())[:4])
        raise ValueError(f"the stand-in compiles {len(advice.requests) - len(refused)} of the advice's {len(advice.requests)} "
                         f"requests ({why}{'; …' if len(refused) > 4 else ''})")
    rows = _rows(advice)
    return [{"request_id": str(q.id), "arrive_step": rows[i][0][0], "prompt_len": int(q.prompt_len), "max_tokens": int(q.output_len)}
            for i, q in enumerate(advice.requests)]


def named(compiler: Any) -> dict[str, str]:
    """A compiler as the registration names it: its name and digest."""
    return {"name": str(compiler.name), "digest": compiler.digest}


def commitment_binding(compiler: Any) -> bytes:
    """The advice tree's domain binding (`interface.advice_tree_binding`): the scheme and the registered compiler, so a root
    opens only under the compiler it names."""
    return IF.advice_tree_binding(named(compiler))


advice_rows = IF.advice_rows


def commit_advice(compiler: Any, data: bytes, salts: Sequence[bytes] | None = None) -> AC.Registration:
    """The advice's hiding commitment (`interface.commit_advice`: `advice_rows` under `commitment_binding`), fresh salts unless
    ``salts`` is given."""
    return IF.commit_advice(named(compiler), data, salts)


def of_advice(compiler: Any, data: bytes, codec: Any = None,
              salts: Sequence[bytes] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any], AC.Registration]:
    """The advice's bytes read by ``codec`` (`codec_of`), refused unless they are its canonical encoding: the compiler's requests
    (`requests_of`), the binding (`binding`: the root of `commit_advice` and the length, ``codec.total_bits``) and the commitment,
    whose rows and salts are the prover's."""
    c = codec_of(codec)
    advice = c.decode(data)
    if c.encode(advice) != data:
        raise ValueError("the advice's bytes are not its canonical encoding")
    reg = commit_advice(compiler, data, salts)
    return requests_of(advice), binding(compiler, reg.root.hex(), int(c.total_bits(advice))), reg


def report(compiler: Any, data: bytes, T: int, codec: Any = None) -> dict[str, Any]:
    """Served advice through the advice side of the window path, for a run with no TP Commit to window: I1's round trip, the binding
    a registration would carry (fresh salts), the advice's charge per egress window of ``T`` buckets (``codec.bits_by_window``),
    and the requests the stand-in refuses (`refusals`), by id.  Nothing of the advice itself."""
    c = codec_of(codec)
    advice = c.decode(data)
    if c.encode(advice) != data:
        raise ValueError("the advice's bytes are not its canonical encoding")
    reg, refused = commit_advice(compiler, data), refusals(advice)
    return {"roundtrip": True, "bytes": len(data), "rows": len(reg.salts), "row_bits": ROW_BITS,
            "binding": binding(compiler, reg.root.hex(), int(c.total_bits(advice))), "T": int(T),
            "bits_by_window": {str(w): int(b) for w, b in sorted(c.bits_by_window(advice, int(T)).items())},
            "requests": len(advice.requests), "compiled": len(advice.requests) - len(refused),
            "refused": {str(advice.requests[i].id): why for i, why in sorted(refused.items())}}


def opening(data: bytes, reg: AC.Registration) -> dict[str, Any]:
    """The advice's opening: its encoding and every row's salt, the prover's private file.  Handing it to the verifier stands in
    for the proof that the compiler's units are wired as the committed advice says (the named gap)."""
    return IF.advice_opening(data, reg)


def of_opening(compiler: Any, doc: Mapping[str, Any], codec: Any = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The verifier's side of `opening`: the requests of the advice it opens and the binding it recomputes (`of_advice` under the
    opening's salts), which `tp.serving_rows.source` holds the registration's to.  A malformed opening is a ValueError."""
    if not isinstance(doc, Mapping) or doc.get("scheme") != SCHEME:
        raise ValueError(f"an advice opening is a {SCHEME} document")
    try:
        data, salts = bytes.fromhex(doc["advice"]), [bytes.fromhex(s) for s in doc["salts"]]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"an advice opening holds the advice and its salts in hex: {e}") from e
    reqs, bound, _reg = of_advice(compiler, data, codec, salts)
    return reqs, bound


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
    lookup of record (`rank_replay.load`), refused unless its digest is in ``of_record`` (rank -> digests), and ``transfer_gates``,
    the templates it lists as holding transfer gates (module docstring).  Its digest names the rule, every rank's Programs of
    record and that list, so a window over another Build's Programs, or proving other units whole, names another compiler."""

    programs: Mapping[str, str]
    of_record: Mapping[str, Sequence[str]]
    name: str = TRACED
    transfer_gates: tuple[str, ...] = ()

    @classmethod
    def for_commit(cls, commit_dir: str | os.PathLike, build_root: str | os.PathLike, transfer_gates: Sequence[str] = ()) -> Traced:
        """The stand-in over a TP Commit's Programs of record (its record's ``of_record``), each rank's under ``build_root/rank<r>``."""
        from verity_vllm.pipeline.tp import serving_rows as SRT

        rec = SRT.record(commit_dir)
        return cls({str(r): os.path.join(str(build_root), f"rank{r}") for r in range(int(rec["world"]))}, rec["of_record"],
                   transfer_gates=tuple(dict.fromkeys(map(str, transfer_gates))))

    @property
    def digest(self) -> str:
        return hashlib.sha512(canonical_json({"name": self.name, "of_record": {str(r): sorted(map(str, d)) for r, d in
                                                                              sorted(self.of_record.items())},
                                              "transfer_gates": sorted(self.transfer_gates)})).hexdigest()

    def load(self, rank: int, world: int, requests: Sequence[Mapping[str, Any]]) -> Any:
        from verity_vllm.pipeline.tp import rank_replay as RR

        return RR.load({"programs": {str(rank): self.programs[str(rank)]}, "of_record": {str(k): list(v) for k, v in self.of_record.items()},
                        "requests": [dict(q) for q in requests]}, int(rank), int(world), definitions=False)


CAPPED = "verity/prove-path/capped-compiler/v0"
#: ℓ, the public cap on the advice's encoding in bits: 1 MiB, far above a session's few hundred requests (I1 spends tens of bits a
#: row); `predict.steps.CAP_BITS`
CAP_BITS = 1 << 23


@dataclass(frozen=True)
class Capped:
    """A registered compiler with ℓ, the public cap on the advice's length in bits. Its digest names the inner compiler's and ℓ,
    so the registration's ``compiler`` fixes ℓ before sampling as it fixes the code; `load` is the inner compiler's."""

    inner: Any
    cap_bits: int = CAP_BITS
    name: str = CAPPED

    def __post_init__(self) -> None:
        if isinstance(self.cap_bits, bool) or not isinstance(self.cap_bits, int) or self.cap_bits < 8:
            raise ValueError(f"the cap ℓ is a whole number of bits, at least 8, not {self.cap_bits!r}")

    @property
    def digest(self) -> str:
        return hashlib.sha512(canonical_json({"name": self.name, "cap_bits": self.cap_bits,
                                              "compiler": {"name": str(self.inner.name), "digest": self.inner.digest}})).hexdigest()

    @property
    def transfer_gates(self) -> tuple[str, ...]:
        """The inner compiler's transfer-gate templates; the inner's digest, which this one names, names them."""
        return tuple(getattr(self.inner, "transfer_gates", ()))

    def load(self, rank: int, world: int, requests: Any) -> Any:
        return self.inner.load(rank, world, requests)


def binding(compiler: Any, advice_root: str, advice_bits: int) -> dict[str, Any]:
    """The window provenance's binding (`tp.serving_rows.BINDING`): the registered compiler's name and digest, and the root of the
    advice's hiding commitment (`commit_advice`) and its length in bits."""
    for what, d in (("compiler's digest", compiler.digest), ("advice's root", advice_root)):
        if not (isinstance(d, str) and _DIGEST.match(d)):
            raise ValueError(f"the {what} is 64 bytes, 128 hex digits, not {d!r}")
    if isinstance(advice_bits, bool) or not isinstance(advice_bits, int) or advice_bits < 0:
        raise ValueError(f"the advice's length is a whole number of bits, not {advice_bits!r}")
    return {"compiler": {"name": str(compiler.name), "digest": compiler.digest}, "advice": {"root": advice_root, "bits": advice_bits}}


def _loaded(commit_dir: str | os.PathLike, rank: int, compiler: Any, requests: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], Any]:
    from verity_vllm.pipeline.tp import serving_rows as SRT

    rec, rank = SRT.record(commit_dir), int(rank)
    if not 0 <= rank < int(rec["world"]):
        raise ValueError(f"rank {rank} is not a rank of the world of {rec['world']}")
    return rec, compiler.load(rank, int(rec["world"]), requests)


def transfer_gates_of(compiler: Any) -> tuple[str, ...]:
    """The templates ``compiler`` lists as holding transfer gates, ``()`` for a compiler that lists none."""
    return tuple(getattr(compiler, "transfer_gates", ()))


def window_of_advice(commit_dir: str | os.PathLike, rank: int, partition_file: str | os.PathLike, compiler: Any,
                     requests: Sequence[Mapping[str, Any]], bound: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """The verifier's own window of rank ``rank`` of the TP Commit ``commit_dir``: the population over ``compiler``'s Programs for the
    advice's ``requests``, and what `tp.serving_rows.source` holds the registration to, ``bound`` (`binding`) among it."""
    from verity_vllm.pipeline.tp import serving_rows as SRT

    rec, p = _loaded(commit_dir, rank, compiler, requests)
    return SRT.window_of(rec, rank, partition_file, p, dict(bound), transfer_gates_of(compiler))


def commit_window(commit_dir: str | os.PathLike, rank: int, partition_file: str | os.PathLike, out_dir: str | os.PathLike, compiler: Any,
                  requests: Sequence[Mapping[str, Any]], bound: Mapping[str, Any], **kw: Any) -> dict[str, Any]:
    """Rank ``rank``'s stand-in window into ``out_dir`` from its sealed bundle (`tp.serving_rows.from_bundle`, ``kw`` its options) over
    ``compiler``'s Programs for the advice's ``requests``, ``bound`` (`binding`) in its provenance."""
    from verity_vllm.pipeline.tp import serving_rows as SRT

    _rec, p = _loaded(commit_dir, rank, compiler, requests)
    return SRT.from_bundle(commit_dir, rank, partition_file, out_dir, programs=p, binding=dict(bound),
                           transfer_gates=transfer_gates_of(compiler), **kw)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Options:
    """verity-vllm advice-windows COMMIT_OUT --advice FILE --opening FILE --partition F --out O: a deferred TP Commit's stand-in
    windows, ``O/rank<r>/`` each, from its ranks' sealed bundles over the stand-in compiler's Programs (`Traced`, listing the
    ``--transfer-gate`` templates) for the advice (`commit_window`), their registrations binding the advice's hiding commitment,
    whose opening goes to ``--opening``."""
    commit_out: str = option("commit_out", help="the deferred TP Commit's --out (runs.jsonl, versions.json, replay_deferred.json, its bundles)")
    advice: str = option("--advice", help="the advice: I1's canonical encoding (the prover's, private)")
    opening: str = option("--opening", help="write the advice's opening here (its encoding and salts, `opening`): the prover's, private")
    partition: str = option("--partition", help="the verifier's partition file: {templates, law, scope?}")
    out: str = option("--out", help="the windows' directory: rank<r>/ each")
    build_root: str = option("--build-root", help="the stand-in compiler's Programs: each rank's Programs of record in rank<r>/")
    ranks: str | None = option("--ranks", default=None, help="comma-separated ranks (default: every rank of the world)")
    transfer_gate: list[str] = option("--transfer-gate", action="append", default=[],
                                      help="a template the stand-in compiler lists as holding transfer gates, every unit of it proved "
                                           "(repeatable; the partition's law is then work:K)")
    write_from_workload: str | None = option("--write-from-workload", type=workload_arg, default=None,
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
        compiler = Traced.for_commit(o.commit_out, o.build_root, o.transfer_gate)
        data = Path(o.advice).read_bytes()
        reqs, bound, reg = of_advice(compiler, data, c)
        Path(o.opening).write_text(json.dumps(opening(data, reg)) + "\n")
        for r in [int(x) for x in o.ranks.split(",")] if o.ranks else range(int(rec["world"])):
            got = commit_window(o.commit_out, r, o.partition, Path(o.out) / f"rank{r}", compiler, reqs, bound, workers=o.workers)
            print(f"[advice-windows] rank {r}: stand-in window of {got['population']} units over {len(reqs)} requests, advice root "
                  f"{bound['advice']['root'][:16]} ({bound['advice']['bits']} bits), compiler {compiler.digest[:16]}, registration "
                  f"{str(got['registration_digest'])[:16]}", flush=True)
    except (CR.Refused, LookupError, ValueError, RuntimeError) as e:
        print(f"[advice-windows] REFUSED: {e}", flush=True)
        return 2
    return 0
