"""The advice a registered compiler reads (everything serving made unpredictable, network-accounting's I1), committed under
hiding rows, and the binding a window's registration carries of it.

The binding is two keys of the registration's window provenance (`binding`), beside the served Programs:

    compiler   {"name", "digest"}   the registered compiler: model and code fixed before the run
    advice     {"root", "bits"}     the advice it read, private: the root of its hiding commitment, and its length in bits

Core's ``verity/registration/v1`` record carries the window's provenance as one public item (``window``), which the record's digest
covers and the coins commit to before sampling, so the sample comes after both are fixed and the record gains no field.  The advice
is not published: its canonical encoding is committed as hiding rows (`commit_advice`: ``ROW_BITS``-bit rows,
zero-padded, under `verity.commitments.advice`, hm96-sha512 leaves in one frame-v3-sha512 tree, its domain bound to the
compiler), and only the root and the length are public.  The rows and their salts are the prover's (`opening`).

A compiler is what the binding reads of it: ``name``, ``digest`` and ``transfer_gates`` (``()`` if it has none), the templates
whose units hold transfer gates (the I/O shell's ingress parse and egress render units), proved with selection All, never sampled,
which its digest covers.  `Capped` registers one with ℓ, the public cap on the advice's length.

The advice is I1's (``codec``, default `I1`): `of_advice` reads its bytes, which must be its canonical encoding, commits them and
binds the root and the advice's length, ``total_bits``; `opened` recomputes that binding from an opening without the stand-in's
requests, for a compiler that compiles the advice itself.  `advice_of_workload` is the stand-in's advice, the prover's, for a
fixed-length workload; a served run's is ADVICE-SRC's, from the scheduler's step trace.
"""
from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity.circuits.codec import canonical_json
from verity.commitments import advice as AC
from verity.commitments.periods import interface as IF

I1 = "verity_vllm.pipeline.advice_code.format"
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


def commit_advice(compiler: Any, data: bytes, salts: Sequence[bytes] | None = None) -> AC.Committed:
    """The advice's hiding commitment (`interface.commit_advice`: `advice_rows` under `commitment_binding`), fresh salts unless
    ``salts`` is given."""
    return IF.commit_advice(named(compiler), data, salts)


def of_advice(compiler: Any, data: bytes, codec: Any = None,
              salts: Sequence[bytes] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any], AC.Committed]:
    """The advice's bytes read by ``codec`` (`codec_of`), refused unless they are its canonical encoding: the compiler's requests
    (`requests_of`), the binding (`binding`: the root of `commit_advice` and the length, ``codec.total_bits``) and the commitment,
    whose rows and salts are the prover's."""
    c = codec_of(codec)
    advice = _read(c, data)
    reg = commit_advice(compiler, data, salts)
    return requests_of(advice), binding(compiler, reg.root.hex(), int(c.total_bits(advice))), reg


def _read(c: Any, data: bytes) -> Any:
    advice = c.decode(data)
    if c.encode(advice) != data:
        raise ValueError("the advice's bytes are not its canonical encoding")
    return advice


def report(compiler: Any, data: bytes, T: int, codec: Any = None) -> dict[str, Any]:
    """Served advice through the advice side of the window path, for a run with no TP Commit to window: I1's round trip, the binding
    a registration would carry (fresh salts), the advice's charge per egress window of ``T`` buckets (``codec.bits_by_window``),
    and the requests the stand-in refuses (`refusals`), by id.  Nothing of the advice itself."""
    c = codec_of(codec)
    advice = _read(c, data)
    reg, refused = commit_advice(compiler, data), refusals(advice)
    return {"roundtrip": True, "bytes": len(data), "rows": len(reg.salts), "row_bits": ROW_BITS,
            "binding": binding(compiler, reg.root.hex(), int(c.total_bits(advice))), "T": int(T),
            "bits_by_window": {str(w): int(b) for w, b in sorted(c.bits_by_window(advice, int(T)).items())},
            "requests": len(advice.requests), "compiled": len(advice.requests) - len(refused),
            "refused": {str(advice.requests[i].id): why for i, why in sorted(refused.items())}}


def opening(data: bytes, reg: AC.Committed) -> dict[str, Any]:
    """The advice's opening: its encoding and every row's salt, the prover's private file.  Handing it to the verifier stands in
    for the proof that the compiler's units are wired as the committed advice says (the named gap)."""
    return IF.advice_opening(data, reg)


def of_opening(compiler: Any, doc: Mapping[str, Any], codec: Any = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The verifier's side of `opening`: the requests of the advice it opens (`requests_of`) and the binding it recomputes
    (`opened`), which a window's registration is held to.  A malformed opening is a ValueError."""
    advice, bound = opened(compiler, doc, codec)
    return requests_of(advice), bound


def opened(compiler: Any, doc: Mapping[str, Any], codec: Any = None) -> tuple[Any, dict[str, Any]]:
    """The advice an opening opens, read by ``codec`` and refused unless its bytes are its canonical encoding, and the binding it
    recomputes under the opening's salts (`binding`), without the stand-in's requests: a compiler that compiles the advice itself
    (I2) refuses what it can't run.  A malformed opening is a ValueError."""
    if not isinstance(doc, Mapping) or doc.get("scheme") != SCHEME:
        raise ValueError(f"an advice opening is a {SCHEME} document")
    try:
        data, salts = bytes.fromhex(doc["advice"]), [bytes.fromhex(s) for s in doc["salts"]]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"an advice opening holds the advice and its salts in hex: {e}") from e
    c = codec_of(codec)
    advice = _read(c, data)
    return advice, binding(compiler, commit_advice(compiler, data, salts).root.hex(), int(c.total_bits(advice)))


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


CAPPED = "verity/prove-path/capped-compiler/v0"
#: ℓ when a registration names none, the public cap on the advice's encoding in bits: 1 MiB, far above a session's few hundred
#: requests (I1 spends tens of bits a row)
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
    """The window provenance's binding: the registered compiler's name and digest, and the root of the
    advice's hiding commitment (`commit_advice`) and its length in bits."""
    for what, d in (("compiler's digest", compiler.digest), ("advice's root", advice_root)):
        if not (isinstance(d, str) and _DIGEST.match(d)):
            raise ValueError(f"the {what} is 64 bytes, 128 hex digits, not {d!r}")
    if isinstance(advice_bits, bool) or not isinstance(advice_bits, int) or advice_bits < 0:
        raise ValueError(f"the advice's length is a whole number of bits, not {advice_bits!r}")
    return {"compiler": {"name": str(compiler.name), "digest": compiler.digest}, "advice": {"root": advice_root, "bits": advice_bits}}
