"""A span of a served window (`boundary_rows`) and its boundary rows as u16 words: the request (`Span`) and its class
(P, O), `b.ingress`'s fields and the sampling inputs they read, `b.egress`'s finish codes, and the rows of each a span
commits."""
from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

MIN_CAP = 2
#: `b.ingress`'s fields before its ids, each a u32 (``u64``: two, low first) of the request's own sampling inputs
FIELDS = (("lp", "u32"), ("max_tokens", "u32"), ("min_tokens", "u32"), ("ignore_eos", "u32"), ("temperature", "f32"),
          ("top_p", "f32"), ("top_k", "i32"), ("min_p", "f32"), ("presence_penalty", "f32"), ("frequency_penalty", "f32"),
          ("repetition_penalty", "f32"), ("seeded", "u32"), ("seed", "u64"))
#: the sampling inputs `FIELDS` reads off a request's SamplingParams
SAMPLING = tuple(k for k, _ in FIELDS if k not in ("lp", "seeded"))
HEAD_U32 = sum(2 if t == "u64" else 1 for _, t in FIELDS)
#: each field's first u32 among `HEAD_U32`
FIELD_AT = {k: sum(2 if t == "u64" else 1 for _, t in FIELDS[:i]) for i, (k, _) in enumerate(FIELDS)}
#: `b.egress`'s finish codes; 0 is a request still running when the serve stopped
FINISH = ("stop", "length", "abort", "error")


def cap(n: int) -> int:
    """The least power of two at least `MIN_CAP` and ``n``."""
    return max(MIN_CAP, 1 << max(0, int(n) - 1).bit_length())


def class_size(v: Any) -> bool:
    """Whether ``v`` is a class's size, of prompt ids or of outputs (`Span.shape`)."""
    return type(v) is int and v >= MIN_CAP and v & (v - 1) == 0


@dataclass(frozen=True)
class Span:
    """One request as the engine core served it: its engine id, prompt ids, sampling inputs (`SAMPLING`), the ids it was
    served, how it finished (a `FINISH` name, None while running) and, where the serve knows it, its release: the
    network certifier's run and the request's index N in that run's release order."""

    engine_id: str
    prompt: tuple[int, ...]
    sampling: Mapping[str, Any]
    served: tuple[int, ...] = ()
    finish: str | None = None
    release: tuple[str, int] | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Its class (P, O)."""
        return cap(len(self.prompt)), cap(len(self.served))

    def to_json(self) -> dict[str, Any]:
        return {"engine_id": self.engine_id, "prompt": list(self.prompt), "sampling": dict(self.sampling),
                "served": list(self.served), "finish": self.finish,
                **({} if self.release is None else {"release": list(self.release)})}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Span:
        release = d.get("release")
        return cls(str(d["engine_id"]), tuple(d["prompt"]), dict(d["sampling"]), tuple(d["served"]), d["finish"],
                   None if release is None else (str(release[0]), int(release[1])))



def sampling_of(params: Any) -> dict[str, Any]:
    """A request's `SAMPLING` inputs off its SamplingParams (None where it has none)."""
    out = {}
    for k in SAMPLING:
        v = getattr(params, k, None)
        out[k] = v if v is None or isinstance(v, (bool, int, float)) else float(v)
    return out


def _u32(name: str, kind: str, v: Any) -> list[int]:
    if kind == "f32":
        return [struct.unpack("<I", struct.pack("<f", float(v or 0.0)))[0]]
    x, bits = int(v or 0), 64 if kind == "u64" else 32
    lo = -(1 << (bits - 1)) if kind in ("i32", "u64") else 0
    if not lo <= x < 1 << bits:
        raise ValueError(f"{name} = {x} doesn't fit b.ingress's {kind}")
    x &= (1 << bits) - 1
    return [x & 0xFFFFFFFF, x >> 32] if kind == "u64" else [x]


def _words(u32s: Sequence[int]) -> np.ndarray:
    a = np.asarray(u32s, dtype=np.int64)
    if a.size and (a.min() < 0 or a.max() >= 1 << 32):
        raise ValueError("an id is a u32")
    return np.stack([a & 0xFFFF, a >> 16], axis=-1).astype(np.uint16).reshape(-1)


def ingress_row(span: Span, prompt_cap: int) -> np.ndarray:
    """Span's `b.ingress` row at prompt size P, as u16 words."""
    if len(span.prompt) > prompt_cap:
        raise ValueError(f"{span.engine_id}: {len(span.prompt)} prompt ids past the class's {prompt_cap}")
    s = {**{k: None for k in SAMPLING}, **span.sampling}
    vals = {**s, "lp": len(span.prompt), "seeded": s["seed"] is not None}
    head = [w for k, t in FIELDS for w in _u32(k, t, vals[k])]
    return _words(head + list(span.prompt) + [0] * (prompt_cap - len(span.prompt)))


def egress_row(span: Span, output_cap: int, head: bool = True) -> np.ndarray:
    """Span's `b.egress` row at output size O, as u16 words: the ids served n, the finish and the O ids; without ``head``
    (a serve that registers a stop rule), the O ids alone."""
    if len(span.served) > output_cap:
        raise ValueError(f"{span.engine_id}: {len(span.served)} ids served past the class's {output_cap}")
    ids = [*span.served, *[0] * (output_cap - len(span.served))]
    if not head:
        return _words(ids)
    if span.finish is not None and span.finish not in FINISH:
        raise ValueError(f"{span.engine_id}: finish {span.finish!r} is none of {FINISH}")
    finish = 0 if span.finish is None else 1 + FINISH.index(span.finish)
    return _words([len(span.served), finish, *ids])


def id_rows(span: Span, outputs: int) -> np.ndarray:
    """Span's rows of its replay block's return in a class of O outputs, (O, 2) u16: row j the id of output j, and past
    the ids served the last one again, as the block's instances past them are copies of its last
    (`interior_rows.Span.padded`). A span served no id has no instance to copy, and its rows are zero: no replay
    reproduces it."""
    if len(span.served) > outputs:
        raise ValueError(f"{span.engine_id}: {len(span.served)} ids served past the class's {outputs}")
    last = span.served[-1] if span.served else 0
    return _words([*span.served] + [last] * (outputs - len(span.served))).reshape(outputs, 2)
