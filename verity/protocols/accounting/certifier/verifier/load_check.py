"""The check of the weights' load: the manifest a shape gives, and the load's registrations read against it."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from verity.protocols.accounting.certifier.verifier import transfer_rows as TR
from verity.protocols.accounting.certifier.verifier.certificate import Direction
from verity.protocols.proofs.spot.verifier.one_stage import registered as G

FORMAT = "verity-network-certifier/load/v1"
LINK, KIND, WINDOW = "s0/uplink", "load", 0
SCHEMA = G.SCHEMA_V2


@dataclass(frozen=True)
class Shape:
    """Interface 1's `Shape` (``benchmarks.torch_gpu.p2_stage.served_weights``)."""

    name: str
    width: int
    layers: int
    heads: int
    head_dim: int
    kv_heads: int
    mlp: int
    vocab: int
    context: int
    row_codes: int


#: a linear's row the 8,192 codes a FAMILIES ``wrow`` part reads (interface 1's)
H200 = Shape("cmp92-h200-mqa-w16384-l52", 16384, 52, 256, 64, 1, 65536, 256, 32, 8192)
TINY = Shape("tiny-h32-l1", 32, 1, 2, 16, 1, 64, 256, 8, 16)
SHAPES = {"h200": H200, "tiny": TINY}


@dataclass(frozen=True)
class Value:
    """One registered value of the load: interface 1's tensor ``name``, its bytes ``[offset, offset + nbytes)`` of the
    weights stream, in ``leaves`` rows of ``words`` u16 words, each row ``schema``'s bit row of `reg_words` leaves of
    `word_bits`; and how interface 1 generates it, ``rows`` generation rows of ``kind`` elements (``s8``, ``scale`` with
    exponent ``exponent``, ``gain``, ``embed``)."""

    name: str
    offset: int
    nbytes: int
    words: int
    kind: str = "s8"
    rows: int = 1
    exponent: int = 0
    schema: str = SCHEMA

    @property
    def row_bytes(self) -> int:
        return 2 * self.words

    @property
    def leaves(self) -> int:
        return self.nbytes // self.row_bytes

    @property
    def word_bits(self) -> int:
        """Its leaves' width read by a program named by its digest: 16, a Boolean program's one-bit leaves rounded up,
        an f32 scale two 16-bit leaves, its low half first (interface 1's `Tensor.word_bits`)."""
        return 16

    @property
    def reg_words(self) -> int:
        """Its leaves of `word_bits` a registered row (the same row bytes)."""
        return 16 * self.words // self.word_bits


def manifest(shape: Shape) -> tuple[Value, ...]:
    """Interface 1's tensors in stream order, each a value: ``tok`` (H, V), a coordinate over the tokens a row of V
    words, ``pos`` (1 word), then per layer ``attn_norm``, ``wq``, ``wk``, ``wv``, ``wo``, ``mlp_norm``, ``wup``, ``wdown``,
    each linear its s8 rows of ``row_codes`` codes then its f32 ``.scale`` (2 words a row), a gain 1 word a row; then
    ``final_norm``, ``head`` and ``head.scale``."""
    s, out, off = shape, [], 0

    def add(name: str, nbytes: int, words: int, kind: str, rows: int = 1, exponent: int = 0,
            schema: str = SCHEMA) -> None:
        nonlocal off
        if nbytes % (2 * words):
            raise ValueError(f"{name}: {nbytes} bytes are no whole rows of {words} words")
        out.append(Value(name, off, nbytes, words, kind, rows, exponent, schema))
        off += nbytes

    def linear(name: str, n: int, k: int, shift: int = 0) -> None:
        add(name, n * k, s.row_codes // 2, "s8", n)
        add(name + ".scale", 4 * n, 2, "scale", 1, -(7 + _log2(k) // 2) + shift)

    add("tok", 2 * s.width * s.vocab, s.vocab, "embed", s.width)
    add("pos", 2 * s.width * s.context, 1, "embed", s.width)
    q, kv = s.heads * s.head_dim, s.kv_heads * s.head_dim
    for i in range(s.layers):
        p = f"l{i}."
        add(p + "attn_norm", 2 * s.width, 1, "gain")
        linear(p + "wq", q, s.width, -(_log2(s.head_dim) // 2))
        linear(p + "wk", kv, s.width)
        linear(p + "wv", kv, s.width)
        linear(p + "wo", s.width, q)
        add(p + "mlp_norm", 2 * s.width, 1, "gain")
        linear(p + "wup", s.mlp, s.width)
        linear(p + "wdown", s.width, s.mlp)
    add("final_norm", 2 * s.width, 1, "gain")
    linear("head", s.vocab, s.width)
    return tuple(out)


def _log2(n: int) -> int:
    return n.bit_length() - 1


def shape_named(name: object) -> Shape:
    for s in SHAPES.values():
        if s.name == name:
            return s
    raise ValueError(f"no shape {name!r}: one of {[s.name for s in SHAPES.values()]}")


def check_load(doc: Mapping[str, Any], key: Any, session: bytes) -> tuple[dict[str, Any], list[str]]:
    """A ``load.json`` read as the verifier reads it: each of its shape's values a registration that
    `transfer_rows.check_registration` accepts under `key` and `session` for the value it expects, at the document's
    program and link. Returns the count and the problems."""
    try:
        shape = shape_named(doc.get("shape"))
        if doc.get("format") != FORMAT or doc.get("window") != WINDOW or not isinstance(doc.get("link"), str):
            raise ValueError(f"not a load's registrations ({FORMAT}, window {WINDOW})")
    except ValueError as e:
        return {"values": 0, "registered": 0}, [f"load: {e}"]
    values, regs = manifest(shape), doc.get("values")
    if not isinstance(regs, list) or len(regs) != len(values):
        return {"values": len(values), "registered": 0}, [f"load: not one registration of each of {len(values)} values"]
    problems, ok = [], 0
    for v, reg in zip(values, regs):
        try:
            TR.check_registration(reg, key, session, link=doc["link"], direction=Direction.INGRESS, window=WINDOW,
                                  kind=KIND, value=v.name, program=doc.get("program"), words=v.reg_words, rows=v.leaves,
                                  word_bits=v.word_bits, schema=v.schema)
            ok += 1
        except ValueError as e:
            problems.append(f"load {v.name}: {e}")
    return {"values": len(values), "registered": ok}, problems
