"""Supplied inputs (PROTOCOL.md §5.4): an input of C whose value the verifier computes, from public things only, by a
function C names under its digest; and a public output C holds to one.

A supplied parameter lowers to `SuppliedInput<w>_v1` nodes (`SUPPLIED_ID`), and its `inputs` entry carries its mark:
``[name, n, {"supplied": fn, "args": {...}, "layout": layout, "bits": bits}]``. ``fn`` is a function of `FUNCTIONS`, which
says what it reads (`READS`), the origin its leaves take (`ORIGINS`), its static arguments and its value's length in bytes at
them; ``layout`` (`LAYOUTS`) reads those bytes as the parameter's leaves; ``bits`` is the value's length in bits. A root
`ret` row ``[lo, hi, {"equals": [name, at, n]}]`` holds each of its leaves to a leaf of a supplied input (`check_held`).

This module holds what the decoder checks of a mark and of a held row, how a block's parameter over part of a supplied
input is marked (`part`), and the order rule the verifier holds every evaluation to (`check_order`). The functions' bodies
are the verifier's, not this module's.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["FUNCTIONS", "Function", "LAYOUTS", "ORIGINS", "READS", "SUPPLIED", "SUPPLIED_ID", "SuppliedError", "check_entry",
           "check_held", "check_order", "layout_fits", "mark_of", "part"]

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
    """A refusal of a supplied input or a held output, its code first (`E_SUPPLIED_*`, `E_HELD`, or `E_INPUTS_MIXED`)."""

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
        try:
            b = f(args)
        except TypeError:
            return None
        return b // 8 if type(b) is int and b > 0 and b % 8 == 0 else None
    return nbytes


def _k32(f: Callable[[Mapping[str, Any]], int | None]) -> Callable[[Mapping[str, Any]], int | None]:
    """``f``, where the argument ``k`` is a multiple of 32; no value elsewhere."""
    return lambda args: f(args) if type(args["k"]) is int and args["k"] % 32 == 0 else None


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
    # useful work, Pearl+: column `col`'s `tn` lines of 32 E4M3 bytes, 256 bits a line
    Function("pouw.pearlplus.eb/v1", "supplied", ("coin", "commitment"), ("col", "tn"), _bits(lambda a: 256 * a["tn"])),
    # useful work, Pearl+: `k` lines of 32 E4M3 bytes, 256 bits a line, `k` a multiple of 32
    Function("pouw.pearlplus.fa/v1", "supplied", ("coin", "commitment"), ("k",), _k32(_bits(lambda a: 256 * a["k"]))),
    Function("pouw.pearlplus.fb/v1", "supplied", ("coin", "commitment"), ("k",), _k32(_bits(lambda a: 256 * a["k"]))),
    # useful work, Pearl+: a 32 by 32 Gram matrix of FP32 words as u32, 32768 bits whatever `k` (a multiple of 32) and `dev`
    Function("pouw.pearlplus.gram/v1", "supplied", ("coin", "commitment"), ("k", "dev"), _k32(_fixed(4096))),
    # a constant word of `bits` bits (a multiple of 8, up to 64) holding `value`: what a held output is held to (an
    # opening's verdict, a check's ok)
    Function("const.uint/v1", "supplied", ("constant",), ("value", "bits"),
             _bits(lambda a: a["bits"] if a["bits"] <= 64 and a["value"] < 1 << a["bits"] else 0)),
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


def check_entry(entry: Sequence[Any], nodes: Sequence[tuple[str, int, int]],
                functions: Mapping[str, Function] = FUNCTIONS) -> Mapping[str, Any] | None:
    """Refuse (`SuppliedError`) an `inputs` entry whose mark does not hold of its input nodes ``nodes``, each ``(binding,
    width, leaves)``: a marked entry's nodes are all supplied (`E_INPUTS_MIXED`: a registered or run input under a mark,
    so a value the verifier makes is never opened as the prover's registration), its function is one ``functions`` holds
    (`E_SUPPLIED_FN`), its args are that function's (`E_SUPPLIED_ARGS`), and its leaves are one width, the value's bits
    by its layout (`E_SUPPLIED_WIDTH`); an unmarked entry holds no supplied node (`E_INPUTS_MIXED`). Returns the mark."""
    name, m = entry[0], mark_of(entry)
    supplied = [b == SUPPLIED for b, _w, _n in nodes]
    if m is None:
        if any(supplied):
            raise SuppliedError("E_INPUTS_MIXED", f"input {name!r}: supplied input nodes and no mark naming their function")
        return None
    if not all(supplied):
        raise SuppliedError("E_INPUTS_MIXED", f"input {name!r}: marked supplied, and its nodes are "
                                              f"{sorted({b for b, _w, _n in nodes})}")
    f = functions.get(m["supplied"])
    if f is None:
        raise SuppliedError("E_SUPPLIED_FN", f"input {name!r}: {m['supplied']!r} is no supplied function")
    if set(m["args"]) != set(f.args):
        raise SuppliedError("E_SUPPLIED_ARGS", f"input {name!r}: {f.name} takes the args {sorted(f.args)}, not "
                                               f"{sorted(m['args'])}")
    nbytes = f.nbytes(m["args"])
    if nbytes is None:
        raise SuppliedError("E_SUPPLIED_ARGS", f"input {name!r}: {f.name} has no value at {dict(m['args'])}")
    widths = {w for _b, w, _n in nodes}
    leaves = sum(n for _b, _w, n in nodes)
    w = next(iter(widths)) if len(widths) == 1 else None
    lo, hi = m["of"][1:] if "of" in m else (0, leaves)
    if (w is None or leaves * w != m["bits"] or hi - lo != leaves or hi * w > 8 * nbytes
            or not layout_fits(m["layout"], w, nbytes) or "of" not in m and m["bits"] != 8 * nbytes):
        raise SuppliedError("E_SUPPLIED_WIDTH", f"input {name!r}: {f.name}'s value is {8 * nbytes} bits, read by "
                                                f"{m['layout']} as whole leaves of one width; the mark says {m['bits']} and "
                                                f"its nodes hold {leaves} leaves of widths {sorted(widths)}")
    return m


def check_held(row: Sequence[Any], width: int, inputs: Mapping[str, tuple[Mapping[str, Any], int, int]]) -> None:
    """Refuse (`E_HELD`) a root `ret` row ``[lo, hi, {"equals": [name, at, n]}]`` of leaves of ``width`` bits unless
    ``name`` is a supplied input of ``inputs`` (by name: its mark, leaf width and leaf count), of that width, ``n`` at least
    1 dividing ``hi - lo``, and ``[at, at + n)`` its leaves: leaf ``lo + i`` is held to its leaf ``at + i mod n``."""
    lo, hi, port = row
    eq = port.get("equals") if isinstance(port, Mapping) else None
    ok = isinstance(eq, list) and len(eq) == 3 and isinstance(eq[0], str) and all(type(x) is int for x in eq[1:])
    got = inputs.get(eq[0]) if ok else None
    if not (ok and got is not None and got[1] == width and eq[2] >= 1 and (hi - lo) % eq[2] == 0 and eq[1] >= 0
            and eq[1] + eq[2] <= got[2]):
        raise SuppliedError("E_HELD", f"output row [{lo}, {hi}) of {width}-bit leaves: held to {eq!r}, which is no run of "
                                      "leaves of a supplied input of that width whose length divides the row's")


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
