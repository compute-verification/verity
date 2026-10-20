"""The credited read a build installs from the environment (`engine.int8_credited_env`) on the CPU, over
`test_int8_window`'s stand-in for paged.cu: the variable's parsing and refusals, and the ``encoded`` mode's settling, where
the slots a step's windows flagged are encoded in place at their codec words and unflagged before the next step's window of
each layer, and read back decoded to the plain read's bits, but for a slot the next step's write refilled, which stays
flagged and is read as written."""
from __future__ import annotations

import types

import pytest
import torch

from tests.engine.test_int8_window import D, G, PAGES, SLOTS, FakePaged, _credited, _plain, _step, _xor
from verity_vllm.engine import int8_credited_env as E


class _Paged(FakePaged):
    @staticmethod
    def stream(torch_) -> int:
        return 0


class _Lib:
    """kv_encode as the fake codec's XOR on the slot ``p`` points at, in whichever cache holds it."""

    shape = {"wb": 128}

    def __init__(self, caches: list[torch.Tensor]) -> None:
        self.caches, self.encoded = caches, []

    def call(self, fn, p, q, words, index, rounds, zero, st) -> None:
        assert (fn, p, rounds, zero) == ("kv_encode", q, E.ROUNDS, 0)
        for layer, c in enumerate(self.caches):
            base, block = c.data_ptr(), c.stride(2) * c.element_size()
            if base <= p < base + c.numel() * c.element_size():
                rows = c.transpose(1, 2).reshape(-1, G, 2 * D)
                rows[(p - base) // block] = _xor(rows[(p - base) // block])
                self.encoded.append((layer, (p - base) // block, index))
                return
        raise AssertionError(f"kv_encode at {p:#x}: no cache holds it")


def test_the_variable_names_a_mode_and_a_library():
    assert E.requested(None) is None and E.requested("") is None
    assert E.requested("encoded:/scratch/paged.so") == ("encoded", "/scratch/paged.so")
    for bad in ("encoded", "encoded:", "plain:/scratch/paged.so", "/scratch/paged.so"):
        with pytest.raises(ValueError, match=E.ENV):
            E.requested(bad)
    assert E.install(types.SimpleNamespace(), None) is None


def test_encoded_mode_settles_a_steps_writes_before_the_next_steps_windows():
    lens, rows = [7, 20], [7, 1]
    cache, query, meta = _step(lens, rows, seed=5)
    ks = torch.tensor([0.02, 0.03])
    want = _plain(cache, query, meta, ks)
    caches = [cache, cache.clone()]
    nslots, words = PAGES * SLOTS, 64
    bases = {layer: (7 + layer * nslots) * words for layer in range(2)}
    flags = [torch.zeros(nslots, dtype=torch.uint8) for _ in caches]
    lib = _Lib(caches)
    read = E.SettlingRead(_Paged(), lib, bases, flags, words)
    written = sorted(int(meta.block_table[r, p // SLOTS]) * SLOTS + p % SLOTS for r, n in enumerate(lens) for p in range(n))
    sm = torch.tensor(written + [-1])
    plain_rows = cache.transpose(1, 2).reshape(nslots, G, 2 * D).clone()

    for layer, c in enumerate(caches):                          # step 1: the prefill's writes, flagged and read as they are
        assert torch.equal(_credited(read, c, query, meta, ks, layer, slot_mapping=sm).view(torch.int16), want.view(torch.int16))
    assert lib.encoded == [] and all(f[written].eq(1).all() and int(f.sum()) == len(written) for f in flags)

    refill = written[3]                                         # step 2 writes one of step 1's slots anew (a reused block)
    fresh = (torch.randn(G, 2 * D, generator=torch.Generator().manual_seed(9)) * 3).to(torch.bfloat16)
    want2 = _plain(_with_row(cache, refill, fresh), query, meta, ks)
    meta2 = types.SimpleNamespace(**vars(meta))
    for layer, c in enumerate(caches):                          # each layer's write precedes its window
        c.transpose(1, 2).reshape(nslots, G, 2 * D)[refill] = fresh
        got = _credited(read, c, query, meta2, ks, layer, slot_mapping=torch.tensor([refill, -1]))
        assert torch.equal(got.view(torch.int16), want2.view(torch.int16))
    settled = [s for s in written if s != refill]
    assert read.settled == 2 * len(settled) and all(int(f.sum()) == 1 and int(f[refill]) == 1 for f in flags)
    assert lib.encoded == [(layer, s, bases[layer] + s * words) for layer in range(2) for s in settled]
    for c in caches:
        enc = c.transpose(1, 2).reshape(nslots, G, 2 * D)
        assert torch.equal(enc[settled].view(torch.int16), _xor(plain_rows[settled]).view(torch.int16))
        assert torch.equal(enc[refill].view(torch.int16), fresh.view(torch.int16))


def _with_row(cache: torch.Tensor, slot: int, row: torch.Tensor) -> torch.Tensor:
    out = cache.transpose(1, 2).clone()
    out.reshape(-1, G, 2 * D)[slot] = row
    return out.transpose(1, 2)
