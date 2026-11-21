"""Supplied inputs (PROTOCOL.md §5.4): an input of C whose value the verifier computes, from public things only, by a
function C names under its digest.

A supplied parameter lowers to `SuppliedInput<w>_v1` nodes (`SUPPLIED_ID`), and its `inputs` entry carries its mark:
``[name, n, {"supplied": fn, "args": {...}, "layout": layout, "bits": bits}]``. ``fn`` is a function of `FUNCTIONS`, which
says what it reads (`READS`), the origin its leaves take (`ORIGINS`), its static arguments and its value's length in bytes at
them; ``layout`` (`LAYOUTS`) reads those bytes as the parameter's leaves; ``bits`` is the value's length in bits.

This module holds a mark's form (`mark_of`), how a block's parameter over part of a supplied input is marked (`part`), and
the order rule the verifier holds every evaluation to (`check_order`). The functions' bodies are the verifier's, not this
module's.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["FUNCTIONS", "Function", "LAYOUTS", "ORIGINS", "READS", "SUPPLIED", "SUPPLIED_ID", "SuppliedError", "check_order",
           "layout_fits", "leaves", "mark_of", "part"]

SUPPLIED = "supplied"
#: the ids that denote `SuppliedInput<w>`: a node is a supplied-input node exactly when its callee's id matches
SUPPLIED_ID = re.compile(r"^SuppliedInput([1-9][0-9]*)_v1$")
#: what a function reads: a live coin (the proof service's, or a trusted device's signed one) opened against a commitment
#: released before it, the sample drawn on live coins, a commitment registered before the coin, and the registration's
#: constants (C's own args among them)
READS = ("coin", "sample", "commitment", "constant")
#: a supplied leaf's origin (§12): `coin` where the value is a live coin itself, `supplied` for any other function's
ORIGINS = ("coin", "supplied")
#: how a value's bytes are the parameter's leaves of w bits:
#: `le`, the bytes as one little-endian bit string (bit t is bit t mod 8 of byte t // 8), leaf i its bits [i w, (i + 1) w)
#: least significant first: a row's own order (u16 words as little-endian pairs, a byte string's bits in byte order);
#: `halves`, each big-endian 32-bit half's 32 / w leaves least significant first (w dividing 32);
#: `digest`, each big-endian 64-bit word's 64 bits least significant first (w = 1, whole 64-byte digests)
LAYOUTS = ("le", "halves", "digest")
WIDTHS = (1, 8, 16, 32, 64)
MARK_KEYS = frozenset({"supplied", "args", "layout", "bits"})


class SuppliedError(ValueError):
    """A refusal of a supplied input, its code first (`E_SUPPLIED_*`)."""

    def __init__(self, code: str, what: str) -> None:
        super().__init__(f"{code}: {what}")
        self.code = code


@dataclass(frozen=True)
class Function:
    """A public function a supplied input names: ``reads`` what it reads (`READS`), ``origin`` its leaves' (`ORIGINS`),
    ``args`` its static arguments by name, each a natural or a string fixed in C, and ``nbytes`` its value's bytes at
    them (None: no value at those args)."""
    name: str
    origin: str
    reads: tuple[str, ...]
    args: tuple[str, ...]
    nbytes: Callable[[Mapping[str, Any]], int | None]


def _fixed(n: int) -> Callable[[Mapping[str, Any]], int]:
    return lambda _args: n


def _bits(f: Callable[[Mapping[str, Any]], int]) -> Callable[[Mapping[str, Any]], int | None]:
    """A value of ``f(args)`` bits, in whole bytes."""
    def nbytes(args: Mapping[str, Any]) -> int | None:
        if not all(type(v) is int for v in args.values()):
            return None
        b = f(args)
        return b // 8 if type(b) is int and b > 0 and b % 8 == 0 else None
    return nbytes


#: every function a supplied input may name, by name, arguments and value length: the same closed list as the Lean
#: registry's (`Spot/Supplied.lean` of the proofs' verifier), which holds each function's body and so its definition (§5.4). A name is versioned: a
#: function whose value changes takes a new name. Its arguments are fixed in C, and no length here is one deployment's. A
#: value is short: what is a function of a unit's index and a short value is computed inside the unit, or inside one unit
#: of the block whose rows its other units read by wire.
FUNCTIONS: dict[str, Function] = {f.name: f for f in (
    # storage session: round j's coin opened against the released session commitment, u32be(j) ‖ u64be(i_j)
    Function("pous.session.round_coin/v1", "supplied", ("coin", "commitment"), ("round",), _fixed(12)),
    # storage session: the signed header and context, as the session commitment releases them; the header's compact form
    # is `head` bytes, fixed by the registered blocks, k, limit and block bytes, read as whole u16 words
    Function("pous.session.header/v1", "supplied", ("commitment",), ("head",), _bits(lambda a: 8 * (a["head"] + a["head"] % 2))),
    Function("pous.session.context/v1", "supplied", ("commitment",), (), _fixed(106)),
    # storage session: m, the digest the memory attester signed, released in the clear beside its signature, which the
    # verifier checks under the pinned key and the session's digest unit recomputes from its committed salt and records
    Function("pous.session.message/v1", "supplied", ("commitment",), (), _fixed(64)),
    # storage setup: the memory attester's salt, drawn after the weights' commitment and before the store's; a unit's
    # prefix is the keys' tag, the framed protocol id, the framed prime's id and this salt framed, zero-padded to 128
    # bytes, the first three registered constants
    Function("pous.setup.salt/v1", "supplied", ("coin",), (), _fixed(24)),
    # bandwidth audit: pass `pass`'s seed in window `window`, from the window's live challenge, the pass's number and the
    # chain over the previous pass's root and the appends' head: the named exception to live coins (seeds inside a
    # window). The pass's one expanding unit computes the sketch's midstate, the receipt tree's domain and the sign rows
    Function("poub.read.pass_seed/v1", "supplied", ("coin", "commitment"), ("window", "pass"), _fixed(64)),
    # bandwidth audit: the pass root the prover released for pass `pass` of window `window`, before the openings' coin
    Function("poub.read.pass_root/v1", "supplied", ("commitment",), ("window", "pass"), _fixed(64)),
    # useful work: the execution phase's live salt, bound to the verifier's stream and prover and the record's window
    Function("window.salt/v1", "coin", ("coin",), (), _fixed(64)),
    # useful work: the epoch's live salt of Pearl+ (the first 32 bytes of its salt stream)
    Function("pouw.pearlplus.salt/v1", "coin", ("coin",), (), _fixed(32)),
    # the root the registration holds for registered value `value`: the verifier's pin of it, never a prover's row
    Function("registered.root/v1", "supplied", ("commitment",), ("value",), _fixed(64)),
    # the root the record names for port `port`, registered before the coin
    Function("record.root/v1", "supplied", ("commitment",), ("port",), _fixed(64)),
    # the first position after every unit of the log's records, a u64 big-endian: a unit's position adds its index
    Function("log.base/v1", "supplied", ("commitment",), (), _fixed(8)),
    # a constant u16 word
    Function("const.u16/v1", "supplied", ("constant",), ("value",), lambda a: 2 if a["value"] < 1 << 16 else None),
)}


def _arg(v: Any) -> bool:
    return (isinstance(v, str) and v != "") or (type(v) is int and v >= 0)


def layout_fits(layout: str, w: int, nbytes: int) -> bool:
    """Whether ``nbytes`` bytes read by ``layout`` are whole leaves of ``w`` bits."""
    if layout == "le":
        return w in WIDTHS and 8 * nbytes % w == 0
    if layout == "halves":
        return 0 < w <= 32 and 32 % w == 0 and nbytes % 4 == 0
    return layout == "digest" and w == 1 and nbytes % 64 == 0


def leaves(layout: str, data: bytes, w: int) -> list[int]:
    """``data``, a function's value, as the parameter's leaves of ``w`` bits by ``layout`` (`LAYOUTS`); ValueError where
    they are not whole leaves (`layout_fits`)."""
    if layout not in LAYOUTS or not layout_fits(layout, w, len(data)):
        raise ValueError(f"{len(data)} bytes read by {layout!r} are no whole leaves of {w} bits")
    if layout == "le":
        x = int.from_bytes(data, "little")
        return [(x >> (w * i)) & ((1 << w) - 1) for i in range(8 * len(data) // w)]
    step, word = (4, w) if layout == "halves" else (8, 1)
    return [(h >> (word * j)) & ((1 << word) - 1) for i in range(0, len(data), step)
            for h in (int.from_bytes(data[i:i + step], "big"),) for j in range(8 * step // word)]


def _of(x: Any) -> bool:
    return (isinstance(x, list) and len(x) == 3 and isinstance(x[0], str) and all(type(v) is int for v in x[1:])
            and 0 <= x[1] < x[2])


def mark_of(entry: Sequence[Any]) -> Mapping[str, Any] | None:
    """An `inputs` entry's mark, or None for an unmarked ``[name, n]``; `E_SUPPLIED_MARK` for any other form."""
    if len(entry) == 2:
        return None
    m = entry[2] if len(entry) == 3 else None
    if not (isinstance(m, Mapping) and set(m) - {"of"} == MARK_KEYS and _of(m.get("of", ["", 0, 1]))
            and isinstance(m["supplied"], str)
            and isinstance(m["args"], Mapping) and all(isinstance(k, str) and _arg(v) for k, v in m["args"].items())
            and m["layout"] in LAYOUTS and type(m["bits"]) is int and m["bits"] > 0):
        raise SuppliedError("E_SUPPLIED_MARK", f"input {entry[0]!r}: a mark is {{supplied, args, layout, bits}}, a part's "
                                               f"with of = [input, lo, hi], not "
                                               f"{entry[2:]!r}")
    return m


def part(name: str, mark: Mapping[str, Any], lo: int, hi: int, w: int) -> dict[str, Any]:
    """The mark of a block's parameter over leaves ``[lo, hi)`` of C's supplied input ``name`` (§5.4: how `restrict`
    reports it): C's mark with ``of`` naming the input and the range, and ``bits`` the part's."""
    if not (0 <= lo < hi and hi * w <= mark["bits"]):
        raise SuppliedError("E_SUPPLIED_WIDTH", f"leaves [{lo}, {hi}) of {w} bits are not in {name!r}'s {mark['bits']} bits")
    return {**mark, "bits": (hi - lo) * w, "of": [name, lo, hi]}


def check_order(reads: Mapping[str, int], coin: int, coin_name: str = "the coin") -> None:
    """Refuse (`E_SUPPLIED_ORDER`) an evaluation that reads a commitment not registered before the coin it reads:
    ``reads`` each commitment's place in the record, ``coin`` the coin's."""
    if late := sorted(c for c, at in reads.items() if at >= coin):
        raise SuppliedError("E_SUPPLIED_ORDER", f"{late} registered after {coin_name}, which a supplied value reads")
