"""The int8 attention's credited read (`engine.int8_window`) on the CPU over a stand-in for paged.cu's `window_offsets` and
`window` (a torch gather with a fake codec): the windows read once a step and once a layer give `paged_attention`'s bits, the
step's KV writes are flagged plaintext, a slot read without its flag decodes to garbage, and the read is the named ops the int8
fold binds (`verity_int8::credited_step` once a step, `verity_int8::credited_window` once a layer)."""
from __future__ import annotations

import types
from collections import Counter

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.engine.test_int8_attention import _window
from verity_vllm.engine import int8_ops
from verity_vllm.engine.int8_window import CreditedRead, LayerKeys, read_of

PAGES, SLOTS, G, D, H = 12, 16, 2, 32, 4
MASK = 0x5A5A


def _xor(t: torch.Tensor) -> torch.Tensor:
    return (t.view(torch.int16) ^ MASK).view(torch.bfloat16)


class FakePaged:
    """paged.py's two launchers as torch gathers: a slot not flagged plaintext is decoded (XOR) when ``encoded``."""

    def __init__(self) -> None:
        self.offsets_calls = 0

    def window_offsets(self, lib, torch_, lens, batch, wo) -> None:
        self.offsets_calls += 1
        wo[0] = 0
        wo[1:] = lens[:batch].clamp(min=0).cumsum(0)

    def layout(self, cache, kv_heads, wb, dims):
        return types.SimpleNamespace(bs=cache.shape[2])

    def window(self, lib, torch_, cache, lay, store_base, layers, layer, encoded, k, lens, bt, wo, batch, outk, outv, order=0,
               bj=None, plain=None) -> None:
        assert (layers, layer, order) == (1, 0, 1)
        rows = cache.transpose(1, 2)                                      # [pages][slots][G][2D]
        for r in range(batch):
            n = int(lens[r])
            pos = torch.arange(n)
            slot = bt[r][pos // lay.bs].long() * lay.bs + pos % lay.bs
            kv = rows.reshape(-1, G, 2 * D)[slot]
            if encoded:
                dec = _xor(kv)
                keep = torch.zeros(n, dtype=torch.bool) if plain is None else plain[slot].bool()
                kv = torch.where(keep[:, None, None], kv, dec)
            outk[int(wo[r]):int(wo[r]) + n], outv[int(wo[r]):int(wo[r]) + n] = kv[..., :D], kv[..., D:]


def _step(lens, rows, seed):
    """Windows in an LBNHC cache under a shuffled block table, the step's queries and its metadata."""
    windows = [_window(s, H, G, D, seed=seed + i) for i, s in enumerate(lens)]
    cache = torch.zeros(PAGES, SLOTS, G, 2 * D, dtype=torch.bfloat16)
    order = torch.randperm(PAGES, generator=torch.Generator().manual_seed(seed))
    table, at = torch.zeros(len(lens), 4, dtype=torch.int32), 0
    for r, (_, k, v) in enumerate(windows):
        for b in range(-(-k.shape[0] // SLOTS)):
            table[r, b] = order[at]
            at += 1
            sl = slice(b * SLOTS, min((b + 1) * SLOTS, k.shape[0]))
            cache[table[r, b], :sl.stop - sl.start] = torch.cat([k[sl], v[sl]], -1)
    query = torch.cat([q[s - n:] for (q, _, _), s, n in zip(windows, lens, rows)])
    qsl = torch.tensor([0, *torch.tensor(rows).cumsum(0).tolist()], dtype=torch.int32)
    meta = types.SimpleNamespace(query_start_loc=qsl, seq_lens=torch.tensor(lens, dtype=torch.int32), block_table=table)
    return cache.transpose(1, 2), query, meta


def _plain(cache, query, meta, ks):
    out = torch.full_like(query, float("nan"))
    kc, vc = cache.transpose(1, 2).split(D, dim=-1)
    int8_ops.paged_attention(query, kc, vc, meta.query_start_loc, meta.seq_lens, meta.block_table, ks, D ** -0.5, out)
    return out


def _credited(read, cache, query, meta, ks, layer=0, slot_mapping=None):
    out = torch.full_like(query, float("nan"))
    meta.slot_mapping = slot_mapping
    int8_ops.credited_attention(read, query, cache, layer, meta, G, ks, D ** -0.5, out)
    return out


def test_an_identity_store_reads_todays_bits():
    lens, rows = [5, 40, 23, 1, 17], [5, 40, 1, 1, 1]          # two prefills, three decode steps
    cache, query, meta = _step(lens, rows, seed=3)
    ks = torch.tensor([0.02, 0.031])
    paged = FakePaged()
    read = CreditedRead(paged, types.SimpleNamespace(shape={"wb": 128}), lambda layer: LayerKeys(0), encoded=False)
    want = _plain(cache, query, meta, ks)
    for layer in range(3):                                      # three layers of one step: one offsets, three windows
        assert torch.equal(_credited(read, cache, query, meta, ks, layer).view(torch.int16), want.view(torch.int16))
    assert (paged.offsets_calls, read.steps, read.windows) == (1, 1, 3)
    meta2 = types.SimpleNamespace(**vars(meta))
    _credited(read, cache, query, meta2, ks)
    assert (paged.offsets_calls, read.steps) == (2, 2)


def test_encoded_slots_decode_and_flagged_slots_read_as_they_are():
    lens, rows = [33, 9], [33, 1]
    cache, query, meta = _step(lens, rows, seed=11)
    ks = torch.tensor([0.017, 0.024])
    want = _plain(cache, query, meta, ks)
    nslots = PAGES * SLOTS
    plain = torch.zeros(nslots, dtype=torch.uint8)
    plain[::3] = 1
    flat = cache.transpose(1, 2).reshape(nslots, G, 2 * D)
    enc = torch.where(plain.bool()[:, None, None], flat, _xor(flat)).reshape(PAGES, SLOTS, G, 2 * D).transpose(1, 2)
    read = CreditedRead(FakePaged(), types.SimpleNamespace(shape={"wb": 128}), lambda layer: LayerKeys(0, plain=plain))
    assert torch.equal(_credited(read, enc, query, meta, ks).view(torch.int16), want.view(torch.int16))

    # the step's KV write put plaintext in the decode rows' last slots: the read flags them before it reads
    last = [int(meta.block_table[r, (s - 1) // SLOTS]) * SLOTS + (s - 1) % SLOTS for r, s in enumerate(lens)]
    flat_enc = enc.transpose(1, 2).reshape(nslots, G, 2 * D).clone()
    for s in last:
        flat_enc[s] = flat[s]
        plain[s] = 0
    enc2 = flat_enc.reshape(PAGES, SLOTS, G, 2 * D).transpose(1, 2)
    meta2 = types.SimpleNamespace(**vars(meta))
    got = _credited(read, enc2, query, meta2, ks, slot_mapping=torch.tensor([last[0], -1, last[1]]))
    assert torch.equal(got.view(torch.int16), want.view(torch.int16)) and all(int(plain[s]) == 1 for s in last)

    unflagged = CreditedRead(FakePaged(), types.SimpleNamespace(shape={"wb": 128}), lambda layer: LayerKeys(0))
    assert not torch.equal(_credited(unflagged, enc, query, meta, ks).view(torch.int16), want.view(torch.int16))


class _Ops(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.seen: Counter = Counter()

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        self.seen[str(func.overloadpacket)] += 1
        return func(*args, **(kwargs or {}))


def test_the_read_is_one_step_op_a_step_and_one_window_op_a_layer():
    lens, rows = [7, 20], [7, 1]
    cache, query, meta = _step(lens, rows, seed=5)
    ks = torch.tensor([0.02, 0.03])
    read = CreditedRead(FakePaged(), types.SimpleNamespace(shape={"wb": 128}), lambda layer: LayerKeys(0), encoded=False)
    assert read_of(read.handle) is read
    with _Ops() as ops:
        for layer in range(2):
            _credited(read, cache, query, meta, ks, layer)
    seen = {k: n for k, n in ops.seen.items() if k.startswith("verity_int8.")}
    assert seen == {"verity_int8.credited_step": 1, "verity_int8.credited_window": 2, "verity_int8.kv_quant_window_batch": 2,
                    "verity_int8.window_rows_batch": 2}, seen
