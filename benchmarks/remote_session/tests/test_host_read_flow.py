"""`host_read.py` end to end over a store of CPU tensors, with no GPU: every row is gated and held to `pass` before it
is timed, a row whose pass writes one wrong receipt is named and not timed, and a pass whose host part writes nothing
must be refused or nothing is timed. The store's pass here is the protocol's own receipter, so `host_read.reference`
(the float32 product and the receipt's hash) is held to it too."""
from __future__ import annotations

import contextlib
import hashlib
import json
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

torch = pytest.importorskip("torch")

import host_read  # noqa: E402
from device import BLOCK, SLOTS, rb  # noqa: E402


def words(digest: bytes):
    """A digest as the device holds it: 8 words."""
    return torch.from_numpy(np.frombuffer(digest, ">u8").astype(np.uint64).view(np.int64).copy())


class Store:
    """What `host_read.py` reads of a `DeviceStore`: one shard and one host segment on one device, whose pass writes
    each block's receipt, a node over each 16 and a root over the nodes it finds in its buffers."""

    def __init__(self) -> None:
        self.torch, self.n, self.base, self.on_host = torch, [32, 16], [0, 32], [False, True]
        self.gpus = [SimpleNamespace(index=0, numa=0, sms=4, host_main=None)]
        self.seg_gpu, self.gpu_segs, self.blocks, self.hbm_blocks = [0, 0], [[0, 1]], 48, 32
        self.chunk, self.h0, self.hc, self.host_ctas = 16, 1, 1, 16
        gen = torch.Generator().manual_seed(7)
        self.data_g = [torch.randint(0, 256, (n * BLOCK,), dtype=torch.uint8, generator=gen) for n in self.n]
        self.ring = [{"roots": torch.zeros((SLOTS + 1, 8), dtype=torch.int64),
                      "pks": torch.zeros((SLOTS, 4), dtype=torch.int32),
                      "host": {"pks": torch.zeros((SLOTS, 4), dtype=torch.int32)}}]
        self.seg_ring = [{"ctr": torch.zeros(4, dtype=torch.int32), "ys": torch.zeros((n, 8), dtype=torch.int64),
                          "kept": torch.zeros((SLOTS, n // 16, 8), dtype=torch.int64),
                          "host": torch.zeros((SLOTS, n // 16, 8), dtype=torch.int64)} for n in self.n]
        self.root_host = torch.zeros((SLOTS, 8), dtype=torch.int64)
        self.offload_jobs: queue.Queue = queue.Queue()
        self.launched, self.pass_gpu_ms, self.pass_host_ms, self.plain_level_matches = 0, [], [], {}
        self.seed, self.cache, self.lib, self.host_calls = b"", {}, None, []

    @contextlib.contextmanager
    def on(self, gpu, stream=None):
        yield stream

    def sync(self) -> None:
        pass

    def close(self) -> None:
        pass

    @staticmethod
    def ready(handle):
        return handle

    def receipts(self, q: int):
        """Segment q's receipts under the pass's seed, by the protocol's receipter."""
        if (self.seed, q) not in self.cache:
            r, raw = rb.receipter(rb.SKETCH_SHA512, self.seed), self.data_g[q].numpy().tobytes()
            self.cache[self.seed, q] = torch.stack([words(r(self.base[q] + j, raw[j * BLOCK:(j + 1) * BLOCK]))
                                                    for j in range(self.n[q])])
        return self.cache[self.seed, q]

    @staticmethod
    def nodes(ys):
        return torch.stack([words(hashlib.sha512(ys[k:k + 16].numpy().tobytes()).digest())
                            for k in range(0, ys.shape[0], 16)])

    def host_pass(self, q: int, pk, slot: int) -> int:
        self.host_calls.append(self.host_ctas)
        return self.lib.so.rs_pass(q, slot, 8)

    def root(self, slot: int) -> bytes:
        return hashlib.sha512(b"".join(x["kept"][slot].numpy().tobytes() for x in self.seg_ring)).digest()

    def run_pass(self, seed: bytes):
        slot, self.seed = self.launched % SLOTS, seed
        self.launched += 1
        self.ring[0]["pks"][slot].fill_(7)
        self.host_pass(1, self.ring[0]["pks"][slot], slot)
        self.lib.so.rs_pass(0, slot, 0)
        root = self.root(slot)
        self.ring[0]["roots"][slot].copy_(words(root))
        return root, (seed, self.ring[0]["pks"][slot].clone(), [x["kept"][slot].clone() for x in self.seg_ring],
                      words(root))

    def plain_root(self, seed: bytes, expected) -> bytes:
        self.seed = seed
        true = [self.nodes(self.receipts(q)) for q in range(2)]
        self.plain_level_matches = {1: all(bool(torch.equal(a, b)) for a, b in zip(true, expected[2]))}
        return hashlib.sha512(b"".join(t.numpy().tobytes() for t in true)).digest()

    def start_chain(self, challenge: bytes, previous) -> None:
        self.challenge = challenge

    def chain(self, passes: int):
        for i in range(passes):
            self.pass_host_ms.append(0.5)
            yield i, hashlib.sha512(self.challenge + bytes([i])).digest(), None, 3_000_000 * i


class Library:
    """The store's session library: a segment's part of a pass, with one receipt wrong on a host segment read at
    one of `wrong`'s receipt CTAs."""

    def __init__(self, store: Store, wrong: tuple[int, ...]) -> None:
        self.store, self.wrong = store, wrong

    def write(self, q: int, slot: int) -> int:
        s, ys = self.store, self.store.receipts(q)
        if s.on_host[q] and s.host_ctas in self.wrong:
            ys = ys.clone()
            ys[3, 0] ^= 1
        s.seg_ring[q]["ys"].copy_(ys)
        s.seg_ring[q]["kept"][slot].copy_(s.nodes(ys))
        return 0

    def rs_pass(self, q: int, slot: int, consumers: int) -> int:
        return self.write(q, slot)


class Backend:
    def __init__(self, wrong: tuple[int, ...] = ()) -> None:
        self.torch, self.store, self.gated = torch, Store(), []
        self.lib = self.store.lib = SimpleNamespace(so=Library(self.store, wrong), call=lambda *args: None)
        self.host = {"host_ctas": 16, "host_consumers": 8}
        self.blocks = self.store.blocks

    def set_payload_key(self, key: bytes) -> None:
        pass

    def payload_root(self, node: bytes) -> bytes:
        return bytes(rb.DIGEST_BYTES)

    def gate(self, registration) -> dict:
        self.gated.append(self.host["host_ctas"])
        return {"ok": True}

    def read_ms(self, segments: list[int]) -> float:
        return 2.0


class Event:
    def __init__(self, enable_timing: bool = False) -> None:
        pass

    def record(self, stream=None) -> None:
        pass

    def elapsed_time(self, other) -> float:
        return 1.5


def run(tmp_path, monkeypatch, backend: Backend) -> tuple[int, dict]:
    monkeypatch.setattr(host_read.device, "from_arguments", lambda a, log=print: backend)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "no device")
    monkeypatch.setattr(sys, "argv", ["host_read.py", "--lib", "session.so", "--kv", "kv.so", "--out", str(tmp_path),
                                      "--host-gib", "1", "--ctas", "8", "16", "32", "--reps", "2",
                                      "--chain-passes", "4", "--budget-s", "1e9"])
    rc = host_read.main()
    return rc, json.loads((tmp_path / "host_read.json").read_text())


def test_every_row_is_gated_and_held_to_pass_then_timed(tmp_path, monkeypatch):
    backend = Backend()
    rc, doc = run(tmp_path, monkeypatch, backend)
    assert rc == 0 and doc["ok"] and doc["cut"] == []
    assert list(doc["rows"]) == ["pass", "pass@8", "pass@32"] and all(r["ok"] for r in doc["rows"].values())
    assert backend.gated == [16, 8, 32]
    assert doc["rows"]["pass"]["equal"]["host_receipts_reference"] == [True]
    assert doc["rows"]["pass@8"]["equal"] == {"plain_root": True, "plain_levels": True, "root": True, "top": True,
                                              "receipts": [True, True], "kept": [True, True],
                                              "kept_host": [True, True], "ok": True}
    assert doc["control"] == {k: {"root_same": False, "host_receipts_same": [False], "host_kept_same": [False],
                                  "refused": True} for k in ("pass", "pass@8")}
    for arm in ("alone", "beside"):
        assert list(doc[arm]) == list(doc["rows"])
        assert all(x["wrong"] == 0 and x["ms"]["n"] == 2 and x["bytes_per_s"] == 16 * BLOCK / 1.5e-3
                   for x in doc[arm].values())
    assert doc["beside_reads"] == 3
    assert all(c["roots_are_pass_s"] and c["passes"] == 4 and c["pass_ms"]["p50"] == 3.0 for c in doc["chain"].values())
    assert (doc["chunk_blocks"], doc["step_bytes"], doc["host_bytes"], doc["hbm_bytes"]) == (
        16, 16 * BLOCK, 16 * BLOCK, 32 * BLOCK)
    assert doc["host_segments"][0]["bytes"] == 16 * BLOCK and not doc["encoded"]
    # a timed rep runs each row at its own CTA count
    assert set(backend.store.host_calls[-3:]) == {16, 8, 32}


def test_a_row_with_one_wrong_receipt_is_named_and_not_timed(tmp_path, monkeypatch):
    rc, doc = run(tmp_path, monkeypatch, Backend(wrong=(8,)))
    assert rc == 6 and not doc["ok"]
    bad = doc["rows"]["pass@8"]
    assert not bad["ok"] and bad["equal"]["receipts"] == [True, False] and bad["equal"]["kept"] == [True, False]
    assert not bad["equal"]["plain_root"] and not bad["equal"]["root"]
    assert doc["rows"]["pass@32"]["ok"] and list(doc["alone"]) == ["pass", "pass@32"] == list(doc["chain"])


def test_a_pass_that_is_wrong_itself_times_nothing(tmp_path, monkeypatch):
    rc, doc = run(tmp_path, monkeypatch, Backend(wrong=(16,)))
    assert rc == 6 and list(doc["rows"]) == ["pass"] and "alone" not in doc and "control" not in doc
    assert doc["rows"]["pass"]["equal"]["host_receipts_reference"] == [False]


def test_a_control_that_is_not_refused_times_nothing(tmp_path, monkeypatch):
    backend = Backend()
    # A store whose host part never goes through the library, so the control's library changes nothing.
    monkeypatch.setattr(backend.store, "host_pass", lambda q, pk, slot: Library(backend.store, ()).write(q, slot))
    rc, doc = run(tmp_path, monkeypatch, backend)
    assert rc == 6 and "alone" not in doc and "ok" not in doc
    assert doc["control"]["pass"] == {"root_same": True, "host_receipts_same": [True], "host_kept_same": [True],
                                      "refused": False}
