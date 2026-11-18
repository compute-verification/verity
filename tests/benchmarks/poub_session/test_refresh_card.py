"""The card's refresh path with no card: a refresh plan's pieces, the bound offsets and payload digests against the
codec, the light ledger's books against `audit_prover.Store`'s, and the re-key's two kv.cu calls, faked at the ctypes
boundary by the codec's own word functions, against `codec.encode(..., bound=True, coin=...)`."""
from __future__ import annotations

import ctypes
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "verity/protocols/resource_accounting/memory/poub/prover/poub_session"))

torch = pytest.importorskip("torch")

import device as dv  # noqa: E402
from device import BLOCK, Piece, rb, rcodec  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.prover import audit_prover  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.prover.rota import Schedule  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier import audit  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier.store import PAYLOAD_LABEL, store_salt  # noqa: E402

KEY = rcodec.key(bytes(range(32)))
COIN = bytes(range(100, 132))


def payload(j: int) -> bytes:
    raw = bytearray(b"".join(hashlib.sha512(b"payload" + j.to_bytes(8, "big") + c.to_bytes(4, "big")).digest()
                             for c in range(BLOCK // 64)))
    for w in range(127, BLOCK, 128):
        raw[w] &= 0x7F
    return bytes(raw)


def test_pieces_split_each_run_where_it_leaves_the_registered_store():
    plan = rb.Plan(((0, 4), (8, 4), (16, 4)), 20)
    assert dv.pieces(plan, 16, 4, 8) == (Piece(False, 0, 4, 0, 0), Piece(False, 8, 4, 4, 8), Piece(True, 0, 4, 8, 16))
    assert dv.pieces(rb.Plan(((12, 8),), 20), 16, 4, 8) == (Piece(False, 12, 4, 0, 12), Piece(True, 0, 4, 4, 16))
    assert dv.pieces(rb.Plan.whole(16), 16, 4, 0) == (Piece(False, 0, 16, 0, 0),)
    got = dv.pieces(plan, 16, 4, 8)
    ps = list(range(plan.length))
    assert dv.DeviceStore.indices(got, ps).tolist() == [plan.block(p) for p in ps]


def test_pieces_refuse_what_rs_pass_at_cannot_read():
    with pytest.raises(ValueError, match="whole chunks"):
        dv.pieces(rb.Plan(((0, 6),), 20), 16, 4, 8)
    with pytest.raises(ValueError, match="whole chunks"):
        dv.pieces(rb.Plan(((0, 4), (6, 2)), 20), 16, 4, 8)
    assert dv.pieces(rb.Plan(((2, 4),), 20), 16, 4, 8) == (Piece(False, 2, 4, 0, 2),)
    with pytest.raises(ValueError, match="past the spare area"):
        dv.pieces(rb.Plan(((16, 8),), 24), 16, 4, 4)


@pytest.mark.parametrize("first", [0, 5, 1 << 40])
def test_bound_offsets_are_the_codecs_binding(first):
    digests = [hashlib.sha512(bytes([u])).digest() for u in range(5)]
    got = dv.bound_offsets(KEY, first, digests, COIN)
    s = KEY.codec.field.word_bytes
    assert len(got) == s * len(digests)
    for u, d in enumerate(digests):
        assert got[s * u:s * (u + 1)] == rcodec.binding(d, first + u, KEY, COIN).to_bytes(s, "little")
    with pytest.raises(ValueError, match="coin"):
        dv.bound_offsets(KEY, first, digests, COIN[:-1])


def test_payload_digests_are_the_codecs():
    blocks = [payload(j) for j in range(3)]
    assert dv.payload_digests(b"".join(blocks)) == b"".join(rcodec.digest(m) for m in blocks)


def registration(payloads: list[bytes]) -> tuple[audit.Registration, list[bytes]]:
    node, key = hashlib.sha512(b"node").digest(), bytes(range(32))
    salts = [store_salt(key, PAYLOAD_LABEL, j) for j in range(len(payloads))]
    root = audit.payload_tree(node, payloads, salts).commitment.root
    return audit.Registration(node, len(payloads), root, bytes(range(32)), rcodec.DEFAULT, rb.DEFAULT, False,
                              rb.LAYOUT, 16), salts


@pytest.fixture(scope="module")
def books():
    payloads = [payload(j) for j in range(8)]
    reg, salts = registration(payloads)
    return payloads, reg, audit_prover.Store(reg, payloads, salts)


def test_ledger_keeps_the_reference_stores_books(books):
    payloads, reg, ref = books
    done: set[int] = set()
    ledger = dv.Ledger(reg, dv.payload_digests(b"".join(payloads)), done.__contains__)
    a, b = ref.rekey(("registered", 2), 2, 0), ledger.rekey(("registered", 2), 2, 0)
    assert (a.first, a.digests, a.origin, a.root, a.slot) == (b.first, b.digests, b.origin, b.root, b.slot)
    assert (ledger.extent, ledger.head) == (ref.extent, ref.head)
    assert [ledger.origin_position(j) for j in range(10)] == [ref.origin_position(j) for j in range(10)]
    assert ledger.digest(9) == rcodec.digest(payloads[3]) and ledger.digest(1) == rcodec.digest(payloads[1])
    assert ledger.blocks[8] is None and ledger.blocks[1] is not None and 8 in ledger.pending
    done.add(8)
    assert ledger.blocks[9] is not None and not ledger.pending
    with pytest.raises(ValueError, match="registered run"):
        ledger.rekey(("append", 0), 2, 1)
    with pytest.raises(ValueError, match="registered run"):
        ledger.rekey(("registered", 7), 2, 1)
    with pytest.raises(TypeError, match="on the device"):
        ledger.encode(8, COIN)


def test_a_refresh_plan_holds_a_rekey_once_its_encode_lands(books):
    payloads, reg, _ = books
    done: set[int] = set()
    ledger = dv.Ledger(reg, dv.payload_digests(b"".join(payloads)), done.__contains__)
    refresh = audit_prover.Refresh(ledger, Schedule(8, 4, 3, 0))
    (k, origin, count), = refresh.due(1)
    a = ledger.rekey(origin, count, k)
    refresh.appended(a)
    moved = set(range(a.first, a.first + a.count))
    assert not moved & set(refresh.plan(1))
    done.add(a.first)
    assert moved <= set(refresh.plan(1))
    assert set(refresh.plan(1)) == (set(range(8)) - set(range(origin[1], origin[1] + count))) | moved


class FakeKv:
    """kv.cu's decode and bound encode on host memory, word by word by the codec's own functions."""

    def __init__(self, key: rcodec.Key) -> None:
        self.key, self.s, self.calls = key, key.codec.field.word_bytes, []

    def kv_tlw(self) -> int:
        return self.key.codec.field.block_words

    def _words(self, at: int, n: int) -> list[int]:
        raw = ctypes.string_at(at, n * self.s)
        return [int.from_bytes(raw[self.s * w:self.s * (w + 1)], "little") for w in range(n)]

    def _put(self, at: int, words: list[int]) -> int:
        out = b"".join(x.to_bytes(self.s, "little") for x in words)
        ctypes.memmove(at, out, len(out))
        return 0

    def kv_decode(self, src, dst, n, base, k, grid, xr, stream) -> int:
        self.calls.append(("kv_decode", n, base, k))
        return self._put(dst, [rcodec.decode_word(y, base + w, self.key) for w, y in enumerate(self._words(src, n))])

    def kv_encode_bound(self, src, dst, n, base, k, b, grid, stream) -> int:
        self.calls.append(("kv_encode_bound", n, base, k))
        tlw = self.kv_tlw()
        offs = self._words(b, n // tlw)
        return self._put(dst, [rcodec.encode_word(m, base + w, self.key, offs[w // tlw])
                               for w, m in enumerate(self._words(src, n))])


def test_rekey_through_kv_calls_is_the_codecs_bound_encode():
    kv = object.__new__(dv.Kv)
    kv.so, kv.bits, kv.masked = FakeKv(KEY), KEY.codec.field.n, KEY.codec.masked
    kv.rounds, kv.words = KEY.rounds, KEY.codec.field.block_words
    kv.torch = SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: SimpleNamespace(cuda_stream=0)))
    origin, first = 5, 1 << 21
    payloads = [payload(origin + u) for u in range(2)]
    src = torch.from_numpy(np.frombuffer(b"".join(rcodec.encode(m, origin + u, KEY) for u, m in enumerate(payloads)),
                                         dtype=np.uint8).copy())
    dst = torch.zeros_like(src)
    kv.decode(src, dst, origin, 2)
    assert dst.numpy().tobytes() == b"".join(payloads)
    offsets = dv.bound_offsets(KEY, first, [rcodec.digest(m) for m in payloads], COIN)
    kv.encode_bound(dst, dst, first, 2, torch.from_numpy(np.frombuffer(offsets, dtype=np.uint8).copy()))
    got = dst.numpy().tobytes()
    for u, m in enumerate(payloads):
        assert got[u * BLOCK:(u + 1) * BLOCK] == rcodec.encode(m, first + u, KEY, bound=True, coin=COIN)
    w = kv.words
    assert kv.so.calls == [("kv_decode", 2 * w, origin * w, KEY.rounds), ("kv_encode_bound", 2 * w, first * w, KEY.rounds)]
    with pytest.raises(ValueError, match="offsets"):
        kv.encode_bound(dst, dst, first, 2, torch.zeros(10, dtype=torch.uint8))


def test_the_backend_rekeys_into_the_spare_area_under_the_coin(books):
    payloads, reg, _ = books
    calls = []
    backend = object.__new__(dv.GpuBackend)
    backend.refresh_blocks, backend.key = 0, KEY
    backend.ledger = None
    with pytest.raises(ValueError, match="--refresh-gib"):
        backend.rekey(("registered", 0), 2, 0)
    assert not backend.deferred
    backend.refresh_blocks = 4
    backend.ledger = dv.Ledger(reg, dv.payload_digests(b"".join(payloads)), lambda first: False)
    backend.store = SimpleNamespace(rekey=lambda *a: calls.append(("rekey", *a)),
                                    launch=lambda i, after, plan=None, appends=None: calls.append(
                                        ("launch", i, plan, appends)))
    assert backend.deferred
    with pytest.raises(ValueError, match="awaits its coin"):
        backend.encode_append(8, COIN)
    a = backend.rekey(("registered", 4), 2, 0)
    backend.launch(0, None, rb.Plan.whole(8))
    backend.encode_append(a.first, COIN)
    assert calls[0] == ("launch", 0, rb.Plan.whole(8), backend.ledger.head) and backend.ledger.head is not None
    assert calls[1] == ("rekey", 4, 8, 2, dv.bound_offsets(KEY, 8, a.digests, COIN))
    with pytest.raises(ValueError, match="awaits its coin"):
        backend.encode_append(a.first, COIN)
