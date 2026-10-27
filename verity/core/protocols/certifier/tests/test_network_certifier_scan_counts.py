"""`ScanCountsRequests` (`Properties.NetworkCertifier.PerServerIngress`), tried against the scan's reference
(`ingress_scan.step`): the conjecture that one link-window's frame streams the scan accepts number at most
`2^windowBits` (every unit correct), plus `faultyIngressBits` for each faulty ingress unit. The bound has three parts,
each checked here:

* **A request's bits** (`request_bits`): its ordinal (u32: the scan checks only q > `last`), its content-length (u32:
  free while the request is unfinished), `max_tokens` and `seed` (u32), `temperature` and `top_p` (10^7 values each),
  `model` and `user` (a quote, up to 62 printable ASCII characters but `"` and `\\`, a quote, spaces), and its prompt ids,
  at most K0 in its first cell and K in each other of its C cells, each below 2^IDB. Each count is the field's check in
  `io_request` (`_u32`, `_ratio`, `_name`) or the scan's (`step`), tried at its edges. The bound doesn't read the
  class's KI, past which the scan refuses a prompt: every toy class here sets KI to every id its cells hold
  (`toy_class`), so that check never binds, and a smaller KI only refuses more.
* **A bucket's bits** (`bucket_bits`): each slot's block is absent, or d ≤ r data cells then PADs, r + 2 cases.
* **A window's requests** (`window_requests`): a slot is held D + 1 buckets from its admission, so a T-bucket window
  from the empty state admits at most ⌈T/(D+1)⌉ a slot.

The enumeration (`_reach`) takes every row over a cell alphabet (each ordinal's PAD and the cells of three requests on
it: one cell, two, and one longer than C, plus a zero cell and two malformed cells the scan must refuse) from every
state the scan reaches at toy classes, and checks that each accepted row is the only accepted row with its timing and
its requests' cells (so a stream carries nothing but those), that no request takes more than C cells, and that no
window admits more than `window_requests`. From a forged state (a faulty ingress unit's, its row anything), a slot
admits at most two requests more than ⌈T/(D+1)⌉, the `2·lanes·request_bits` of `faulty_ingress_bits`. A cell is at
least 387 bytes (the request's head and one prompt field), so the enumeration reads the cells' structure, and the
per-field tests their contents."""
from __future__ import annotations

import dataclasses
import itertools

import pytest

from verity.core.protocols.certifier import ingress_scan as IS
from verity.core.protocols.certifier import io_request as Q
from verity.core.protocols.certifier import warden as W
from verity.core.protocols.certifier.active import (
    HEADER,
    ID_NAME_BYTES,
    PAD,
    id_request,
)

B = 512
#: the first prompt field's offset in a request's first cell (`io_request.layout(B).start`)
PROMPT_START = 367
#: the printable ASCII characters `io_request._name` admits in a name
NAME_CHARS = [c for c in range(0x20, 0x7F) if c not in b'"\\']
#: a name field's values: up to 62 characters between its quotes
NAME_BITS = (sum(len(NAME_CHARS) ** k for k in range(ID_NAME_BYTES - 1)) - 1).bit_length()
#: content-length, max_tokens, temperature, top_p, seed, model and user
HEAD_BITS = 32 + 32 + IS.RATIO_BITS + IS.RATIO_BITS + 32 + 2 * NAME_BITS
ORDINAL_BITS = 32
#: w63 (`benchmarks/network_traces/scan_zk.py`'s `W63`): its class and its 32-bucket link-window
W63 = IS.Class(B=512, r=4, lanes=4, C=47, IDB=17, EVERY=16, WITHIN=12, D=27, KO=64, KI=1024)
W63_T = 32


def toy_class(**kw) -> IS.Class:
    """The class `kw` names with KI every prompt id its C cells hold (`prompt_room`), so the scan's KI never binds."""
    c = IS.Class(**kw, KI=1)
    return dataclasses.replace(c, KI=c.prompt_room)


def clog2(n: int) -> int:
    return (n - 1).bit_length()


def request_ids(cls: IS.Class) -> int:
    return cls.K0 + (cls.C - 1) * cls.K


def request_bits(cls: IS.Class) -> int:
    return request_ids(cls) * cls.IDB + HEAD_BITS + ORDINAL_BITS


def bucket_bits(cls: IS.Class) -> int:
    return cls.lanes * clog2(cls.r + 2)


def window_requests(cls: IS.Class, T: int) -> int:
    return cls.lanes * -(-T // (cls.D + 1))


def window_bits(cls: IS.Class, T: int) -> int:
    return window_requests(cls, T) * request_bits(cls) + T * bucket_bits(cls)


def faulty_ingress_bits(cls: IS.Class) -> int:
    return 8 * cls.S * cls.B + 2 * cls.lanes * request_bits(cls)


def test_w63_terms():
    """The terms at w63, as `Properties.NetworkCertifier.PerServerIngress` and `benchmarks/rho` read them."""
    lay = Q.layout(B)
    assert lay.start == PROMPT_START and (W63.K, W63.K0) == (45, 12)
    assert (lay.room - PROMPT_START) // 11 == W63.K0 and lay.room // 11 == W63.K
    assert [k for _, _, k in lay.spans] == ["content-length", "u32", "ratio", "ratio", "u32", "name", "name"]
    assert (NAME_BITS, IS.RATIO_BITS, HEAD_BITS) == (406, 24, 956)
    assert (request_ids(W63), request_bits(W63)) == (2082, 36_382)
    assert (bucket_bits(W63), window_requests(W63, W63_T), window_bits(W63, W63_T)) == (12, 8, 291_440)
    assert faulty_ingress_bits(W63) == 65_536 + 291_056


def test_field_counts():
    """Each head field's check admits exactly the values its bits count: a u32 in ten characters (spaces, then its
    digits), a ratio in eight (a digit, `.`, six digits), a name a quote, up to 62 of the 93 characters, a quote and
    spaces."""
    def u32(v: int) -> bytes:
        return b"%10d" % v

    assert Q._u32(u32(0)) and Q._u32(u32(2**32 - 1)) and not Q._u32(b"4294967296")
    assert not Q._u32(b"0000000001") and not Q._u32(b"     1   2") and not Q._u32(b"          ")
    assert Q._ratio(b"0.000000") and Q._ratio(b"9.999999") and not Q._ratio(b"10.00000") and not Q._ratio(b" 1.00000")
    assert len(NAME_CHARS) == 93
    def name(s: bytes) -> bytes:
        return (b'"' + s + b'"').ljust(ID_NAME_BYTES)

    assert Q._name(name(b"")) and Q._name(name(bytes(NAME_CHARS[:62]))) and Q._name(name(b"a  "))
    assert not Q._name(name(bytes(NAME_CHARS[:63])[:63]))
    for c in NAME_CHARS:
        assert Q._name(name(bytes([c]) * 62))
    for c in (*range(0x20), 0x22, 0x5C, *range(0x7F, 0x100)):
        assert not Q._name(name(bytes([c]))), hex(c)


def _cells(cls: IS.Class, raw: bytes, q: int) -> list[bytes]:
    c = W.cells(raw, cls.B, q)
    return [c[i:i + cls.B] for i in range(0, len(c), cls.B)]


def _req(cls: IS.Class, prompt, **kw) -> bytes:
    return id_request(model=kw.pop("model", ""), user=kw.pop("user", ""), prompt=list(prompt),
                      max_tokens=kw.pop("max_tokens", 1), frame_bytes=cls.B, offset=0, **kw)


def _row(cls: IS.Class, blocks) -> bytes:
    cells = [c for b in blocks for c in b]
    return b"".join(cells + [bytes(cls.B)] * (cls.S - len(cells)))


def _scan(cls: IS.Class, rows) -> IS.State | None:
    st = IS.empty(cls)
    for r in rows:
        if (st := IS.step(cls, r, st)) is None:
            return None
    return st


def _one(cls: IS.Class, cells, q: int) -> list[bytes]:
    """A window of one lane's blocks, `cells` r at a time, the last padded."""
    cells = list(cells)
    rows = []
    while cells:
        blk, cells = cells[:cls.r], cells[cls.r:]
        rows.append(_row(cls, [blk + [W.cell(PAD, q, b"", cls.B)] * (cls.r - len(blk))]))
    return rows


CLS = toy_class(B=B, r=2, lanes=1, C=3, IDB=17, EVERY=1, WITHIN=1, D=1, KO=1)


def test_request_fields_free():
    """The scan takes a request whatever its head's values and names, and any prompt ids below 2^IDB; it refuses an id
    of 2^IDB."""
    names = bytes(NAME_CHARS[:62]).decode()
    raw = _req(CLS, [2**17 - 1] * 3, model=names, user=names[::-1], max_tokens=2**32 - 1, seed=2**32 - 1,
               temperature=9.999999, top_p=0.000001)
    assert _scan(CLS, _one(CLS, _cells(CLS, raw, 1), 1)) is not None
    assert _scan(CLS, _one(CLS, _cells(CLS, _req(CLS, [2**17]), 1), 1)) is None


def test_request_ids_in_c_cells():
    """A request of K0 + (C − 1)·K ids is C cells, one more id is C + 1, and the scan refuses its last cell."""
    n = request_ids(CLS)
    assert len(_cells(CLS, _req(CLS, [0] * n), 1)) == CLS.C
    assert len(more := _cells(CLS, _req(CLS, [0] * (n + 1)), 1)) == CLS.C + 1
    assert _scan(CLS, _one(CLS, more[:CLS.C], 1)) is not None and _scan(CLS, _one(CLS, more, 1)) is None


def test_unfinished_content_length_free():
    """An unfinished request's content-length is any u32 past the bytes it has sent: its 32 bits."""
    cells = _cells(CLS, _req(CLS, [0] * request_ids(CLS)), 1)[:CLS.C - 1]
    at = HEADER.size + Q.layout(B).spans[0][0]
    for cl in ((CLS.C - 1) * Q.layout(B).room - Q.BODY_AT + 1, 2**32 - Q.BODY_AT - 1):
        first = cells[0][:at] + b"%10d" % cl + cells[0][at + 10:]
        assert _scan(CLS, _one(CLS, [first, *cells[1:]], 1) + [_row(CLS, [])] * 2) is not None, cl


def test_ordinals_free_above_last():
    """A new connection's ordinal is any u32 above `last`: its 32 bits."""
    cls = toy_class(B=B, r=1, lanes=2, C=3, IDB=17, EVERY=1, WITHIN=1, D=1, KO=1)
    def one(q: int) -> bytes:
        return _row(cls, [_cells(cls, _req(cls, [0]), q)])

    gap = _row(cls, [])
    assert _scan(cls, [_row(cls, [_cells(cls, _req(cls, [0]), q) for q in (7, 2**32 - 1)])]) is not None
    assert _scan(cls, [one(7), gap, one(8)]) is not None
    assert _scan(cls, [one(7), gap, one(5)]) is None and _scan(cls, [one(7), gap, gap, one(7)]) is None


#: toy classes and their windows: one lane of one slot, one lane of two, two lanes of one
TOYS = ((toy_class(B=B, r=1, lanes=1, C=2, IDB=1, EVERY=1, WITHIN=1, D=1, KO=1), 5, (1, 2, 3, 5)),
        (toy_class(B=B, r=2, lanes=1, C=2, IDB=1, EVERY=2, WITHIN=1, D=2, KO=1), 4, (1, 2, 3)),
        (toy_class(B=B, r=1, lanes=2, C=2, IDB=1, EVERY=1, WITHIN=1, D=1, KO=1), 3, (1, 2, 3, 4)))


def _alphabet(cls: IS.Class, ordinals) -> tuple[list[bytes], set[bytes]]:
    """Each ordinal's PAD and the cells of three requests on it (one cell, two, and more than C), a zero cell; and two
    malformed cells the scan must refuse: a PAD with a payload byte, and a request's `DATA` cell marked `END`."""
    out = [bytes(cls.B)]
    for q in ordinals:
        out.append(W.cell(PAD, q, b"", cls.B))
        for n in (1, cls.K0 + 1, request_ids(cls) + 1):
            out += _cells(cls, _req(cls, [0] * n), q)[:cls.C + 1]
    pad = W.cell(PAD, ordinals[0], b"", cls.B)
    data = _cells(cls, _req(cls, [0] * (cls.K0 + 1)), ordinals[0])[0]
    return out, {pad[:-1] + b"\x01", bytes([W.END]) + data[1:]}


def _rows(cls: IS.Class, ordinals) -> tuple[list[bytes], set[bytes]]:
    cells, bad = _alphabet(cls, ordinals)
    return [b"".join(cs) for cs in itertools.product([*cells, *bad], repeat=cls.S)], bad


def _sig(cls: IS.Class, row: bytes, nxt: IS.State):
    """An accepted row's timing and its requests' cells: each slot's ordinal and data cells, None where it has no
    block."""
    out = [None] * cls.NR
    for p in range(cls.lanes):
        blk = [row[(p * cls.r + i) * cls.B:(p * cls.r + i + 1) * cls.B] for i in range(cls.r)]
        q = HEADER.unpack_from(blk[0])[1]
        if q:
            j = next(j for j, s in enumerate(nxt.slots) if s.live and s.q == q)
            assert out[j] is None
            out[j] = (q, tuple(c for c in blk if c[0] != PAD))
    return tuple(out)


def _reach(cls: IS.Class, T: int, start: IS.State, rows, bad: set[bytes]):
    """Every state the scan reaches from `start` in T rows of `rows`, each with the most requests admitted on a path to
    it, in total and a slot; each state's accepted rows checked unique by their timing and requests' cells, and free
    of the `bad` cells."""
    level = {tuple(IS.encode(cls, start)): (start, (0, (0,) * cls.NR))}
    for _ in range(T):
        nxt_level = {}
        for st, (tot, per) in level.values():
            seen = {}
            for r in rows:
                if (nx := IS.step(cls, r, st)) is None:
                    continue
                assert not any(r[i * cls.B:(i + 1) * cls.B] in bad for i in range(cls.S)), "a malformed cell"
                sig = _sig(cls, r, nx)
                assert seen.setdefault(sig, r) == r, "two rows with one timing and the same requests' cells"
                assert all(s.cells <= cls.C for s in nx.slots)
                new = [j for j in range(cls.NR) if sig[j] and not any(o.live and o.q == sig[j][0] for o in st.slots)]
                got = (tot + len(new), tuple(a + (j in new) for j, a in enumerate(per)))
                key = tuple(IS.encode(cls, nx))
                old = nxt_level.get(key, (nx, (0, (0,) * cls.NR)))[1]
                nxt_level[key] = (nx, (max(old[0], got[0]), tuple(map(max, old[1], got[1]))))
        level = nxt_level
    return level


@pytest.mark.parametrize("toy", range(len(TOYS)))
def test_window_enumeration(toy):
    """Every window of rows over the alphabet, from the empty state: no two accepted rows share a timing and requests'
    cells, no request takes more than C cells, and a window admits at most ⌈T/(D+1)⌉ requests a slot."""
    cls, T, ordinals = TOYS[toy]
    level = _reach(cls, T, IS.empty(cls), *_rows(cls, ordinals))
    per = -(-T // (cls.D + 1))
    assert max(t for _, (t, _) in level.values()) == window_requests(cls, T)
    assert max(max(p) for _, (_, p) in level.values()) == per


def _forged(cls: IS.Class, t: int, ordinal: int) -> list[IS.State]:
    """The faulty proof block's S_t, as the scan's structure sees it: `last` 0, each slot free, or live on `ordinal` with an
    unfinished request at no cells, held or not, or done and held."""
    now = t + 1
    kinds = [IS.Slot(), IS.Slot(live=1, q=ordinal, left=2**31, admitted=now),
             IS.Slot(live=1, done=1, q=ordinal, admitted=now)]
    if now > cls.D:
        kinds.append(IS.Slot(live=1, q=ordinal, left=2**31, admitted=now - cls.D - 1))
    out = []
    for ks in itertools.product(kinds, repeat=cls.NR):
        slots = [IS.Slot(**{**vars(k), "out": [0] * cls.KO, "ids": [0] * (cls.C * cls.K)}) for k in ks]
        out.append(IS.State(now, now % cls.EVERY, 0, slots))
    return out


#: one lane of one slot at D = 3 and T = 8, where a faulty proof block's forged state reaches ⌈T/(D+1)⌉ + 2
FAULTY = (toy_class(B=B, r=1, lanes=1, C=2, IDB=1, EVERY=1, WITHIN=1, D=3, KO=1), 8, (1, 2, 3, 5))


@pytest.mark.parametrize("toy", [TOYS[0], FAULTY], ids=["toy0", "tight"])
def test_faulty_unit_two_more_a_slot(toy):
    """A faulty ingress unit at bucket t (its row anything, its S_t forged): with the requests admitted before t and
    the forged state's own, a slot carries at most ⌈T/(D+1)⌉ + 2 requests in the window, and at `FAULTY` exactly
    that."""
    cls, T, ordinals = toy
    rows, bad = _rows(cls, ordinals)
    per, most = -(-T // (cls.D + 1)), 0
    for t in range(T):
        before = max(p[0] for _, (_, p) in _reach(cls, t, IS.empty(cls), rows, bad).values())
        for st in _forged(cls, t, ordinals[-1]):
            after = max(p[0] for _, (_, p) in _reach(cls, T - t - 1, st, rows, bad).values())
            most = max(most, before + int(st.slots[0].live and not st.slots[0].done) + after)
    assert most <= per + 2 and (toy is not FAULTY or most == per + 2)
