"""`InPlaceBackend`: the remote bandwidth session's prover over vLLM's own tensors, the weights and the KV pool where
the binder registered them, with no copy of the store (`benchmarks/remote_session/inplace.py`'s `InPlace`, from
`Region.of_backing` over each registered region's backing, and session.cu).

~~~text
hello         the payload tree on the GPU from the tensors' bytes (`rs_payload_leaves`, block j's salt
              `device.store_salt(key, PAYLOAD_LABEL, j)`), its levels kept on the host for the openings
registration  every registered block encoded in place by kv.cu under the seed's key (`KvDevice.encode_in_place`),
              checked against `codec.encode` at each region's first block and the store's last, and its encode under
              an append's bound offset on a scratch block (`_gate_bound`), so an append's encode is the device's alone
append        the step's completed blocks at their slots (`InPlace.append`), committed by their digests under the
              device's salts; once the coin is back, their slots' bytes (still the payloads appended, or the session
              fails) are encoded in place under b_j(c_a) and the memory map records their writes (deferred residency)
pass          run when launched, after the engine step it follows (`step_of`), over the tensors as they are: each
              plan position read through the map, except a block the step's forward rewrote, read from the copy taken
              before the forward (`Store.on_write`), and a block withheld (servedrop), read as zeros
opening       its subtree recomputed from the store under the same map, a block rewritten since the pass read from the
              earliest copy taken at or after the pass's step; a subtree that still differs from the pass's raises
              `Stale` (a block rewritten after its receipt with no copy)
~~~

A registered tensor stays where it is for the session (`Binder.moved`, checked before every pass: a tensor moved
fails the session by name).  The tensors hold the codec's encoding from registration on.  The weights are decoded at
use (`decode_at_use.DecodeAtUse`: each module's into a scratch for its forward) and decoded back where they are when
the session closes, so the Commit's replay reads the weights it committed.  With paged.cu's build (``paged_lib``), the
int8 attention reads its cache through a credited read (`int8_window.CreditedRead`, installed at registration and
kept after the session): each layer's slots decoded under their offsets, a slot the engine wrote read as it is until
its append's coin is in and the session encodes it, which sets the slot's offset (b_j(c_a) moved from the append's
index j to the slot's own, pg_window's index: b_j + τ · block_words · (j − slot)) and clears its flag.  Without it,
attention reads the KV pool encoded, and the model's output under a session is not the model's after the first
encoded append.  A re-key would append a block whose
payload the map reads from its slot, which holds its encoding: the in-place session refuses it, so its lifetime cap
must outlast its windows.
"""
from __future__ import annotations

import bisect
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from verity.core.protocols.remote_bandwidth import audit, codec as rcodec, receipts as rb
from verity_vllm import config
from verity_vllm.protocols.remote_bandwidth.decode_at_use import DecodeAtUse, encoded_weights, kv_decoder
from verity_vllm.protocols.remote_bandwidth.interface import Stale
from verity_vllm.protocols.remote_bandwidth.registry import Store
from verity_vllm.protocols.remote_bandwidth.remote import KvDevice, StoreBackend, session_module

BLOCK = rcodec.BLOCK_BYTES
ROW = 64
CHUNK = 1 << 16
STEP_BITS = 24


@dataclass
class Saved:
    """The bytes of the pool's blocks at ``slots`` (ascending) before step ``step``'s forward wrote them: ``host``,
    and while it is the newest, ``device``."""

    step: int
    slots: np.ndarray
    host: np.ndarray
    device: Any = None


class Appends:
    """The session's appends as `audit.Store` keeps them, without the store's blocks: each committed by its digests
    (`audit.append_tree`), its payloads pending until its coin arrives (``coins``)."""

    def __init__(self, registration: audit.Registration) -> None:
        self.registration = registration
        self.appends: list[audit.Append] = []
        self.firsts: list[int] = []
        self.pending: dict[int, tuple[bytes, ...]] = {}
        self.coins: dict[int, bytes] = {}
        self.head: bytes | None = None
        self.extent = registration.blocks

    def append(self, payloads: Sequence[bytes], salts: Sequence[bytes]) -> audit.Append:
        reg, first = self.registration, self.extent
        digests = tuple(rcodec.digest(m) for m in payloads)
        a = audit.Append(first, digests, tuple(salts), audit.append_tree(reg.digest, first, digests, salts))
        self.appends.append(a)
        self.firsts.append(first)
        self.extent += a.count
        self.pending[first] = tuple(payloads)
        self.head = audit.append_head(self.head, a.first, a.count, a.root)
        return a

    def of(self, j: int) -> audit.Append:
        return self.appends[bisect.bisect_right(self.firsts, j) - 1]


class InPlaceBackend(StoreBackend):
    """`StoreBackend` over the store's tensors in place (the module's docstring): ``backings`` are the registered
    regions' backings in store order (`Store.backing`), ``session_lib`` session.cu's build and ``kv_lib`` kv.cu's at
    the codec's field and mask, ``moved()`` the registered tensors no longer where they were (`Binder.moved`).
    ``model``'s registered weights are decoded at use; with ``paged_lib`` (paged.cu's build at the codec's field), its
    int8 attention reads the KV pool through `credited` (with no model, the read is built and not installed)."""

    def __init__(self, store: Store, engine: Callable[[int], None], backings: Sequence[Any], *, session_lib: str,
                 kv_lib: str, codec: rcodec.Codec = rcodec.DEFAULT, moved: Callable[[], list[str]] = list,
                 path: Any = None, randbytes: Callable[[int], bytes] = os.urandom, kept_from: int = 6,
                 calibration_passes: int = 256, calibration_warm: int = 16, warmup_ms: float = 500.0,
                 model: Any = None, paged_lib: str | None = None) -> None:
        import torch

        super().__init__(store, engine, randbytes=randbytes, path=path, device=KvDevice(kv_lib, codec))
        session_module("prover")
        self.dev = session_module("device", config.REMOTE_SESSION)
        ipm = session_module("inplace", config.REMOTE_SESSION)
        self.torch, self.codec, self.moved = torch, codec, moved
        self.calibration_passes, self.calibration_warm, self.warmup_ms = calibration_passes, calibration_warm, warmup_ms
        self.model, self.paged_lib, self.decoded, self.credited = model, paged_lib, None, None
        self._sample: dict[int, bytes] = {}
        self._weight_names: set[str] = set()
        self.kv_tables: dict[int, tuple[Any, Any]] = {}
        self.lib = self.dev.Lib(str(session_lib), torch)
        self.key = randbytes(32)
        lay = store.layout()
        if len(backings) != len(lay.regions):
            raise ValueError(f"a backing a region: {len(backings)} for {len(lay.regions)} regions")
        self.backings = list(backings)
        self.regions = [ipm.Region.of_backing(r.first, b) for r, b in zip(lay.regions, backings)]
        backend = self

        class Mapped(ipm.InPlace):
            def map(self, plan):
                return backend._patched(*super().map(plan))

            def tree(self, ys, c0, real, width):
                return backend._subtree(super().tree(ys, c0, real, width), c0, width)

        self.ip = Mapped(torch, self.lib, self.regions, self.key, kept_from=kept_from)
        self.cuda = self.ip.device
        self._firsts = np.array([r.first for r in self.regions], np.int64)
        self._table = np.array([[r.addr, r.plane_blocks, r.planes, r.page_stride, r.plane_stride]
                                for r in self.regions], np.int64)
        self.zero = torch.zeros(BLOCK, dtype=torch.uint8, device=self.cuda)
        self.app_slots: list[int] = []
        self._app_dev = None
        self._dropped_dev = None
        self.saved: list[Saved] = []
        self._windex: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._mode: tuple | None = None
        self._stale: list[int] = []
        self._held: list[Any] = []
        self.pass_ns: list[int] = []
        self.stats = {"payload_s": 0.0, "encode_s": 0.0, "saved_blocks": 0, "pass_patched": 0, "open_patched": 0,
                      "appended": 0, "encoded_appends": 0}
        store.on_write = self._before_write

    # --- the tensors' blocks ------------------------------------------------------------------------------------

    def _addresses(self, slots: np.ndarray) -> np.ndarray:
        """Each registered block's device address (`Region.address`)."""
        slots = np.asarray(slots, np.int64)
        r = np.searchsorted(self._firsts, slots, side="right") - 1
        addr, pb, planes, ps, ks = self._table[r].T
        x = slots - self._firsts[r]
        q = np.where(pb > 0, x // np.maximum(pb, 1), 0)
        paged = addr + q // planes * ps + q % planes * ks + BLOCK * (x % np.maximum(pb, 1))
        return np.where(pb > 0, paged, addr + BLOCK * x)

    def _gather(self, slots: np.ndarray):
        """The blocks at ``slots`` as they are, an (n, 4096) uint8 tensor on the device (`rs_gather64`)."""
        torch, n = self.torch, len(slots)
        out = torch.empty((max(n, 1) * BLOCK // ROW, 8), dtype=torch.int64, device=self.cuda)
        if n:
            rows = (self._addresses(slots)[:, None] + np.arange(0, BLOCK, ROW)).reshape(-1)
            self.lib.call("rs_gather64", torch.from_numpy(rows).to(self.cuda).data_ptr(), len(rows), out.data_ptr())
        return out.view(torch.uint8).view(-1, BLOCK)[:n]

    def _kv_rows(self, slots: np.ndarray):
        """The KV blocks at ``slots`` as rows of their storages: [(storage's (blocks, 4096) view, rows, which)]."""
        torch, addr, out = self.torch, self._addresses(slots), []
        r = np.searchsorted(self._firsts, slots, side="right") - 1
        for key in sorted({id(self.backings[i].flat) for i in set(r.tolist())}):
            which = np.flatnonzero(np.array([id(self.backings[i].flat) == key for i in r]))
            flat = self.backings[r[which[0]]].flat
            off = addr[which] - flat.data_ptr()
            if getattr(self.backings[r[which[0]]], "physical", None) is None or (off % BLOCK).any():
                raise ValueError("an appended block is a KV block at a 4096-byte boundary of its storage")
            view = flat[:flat.numel() // BLOCK * BLOCK].view(-1, BLOCK)
            out.append((view, torch.from_numpy(off // BLOCK).to(self.cuda), torch.from_numpy(which).to(self.cuda)))
        return out

    def _scatter(self, slots: np.ndarray, data) -> None:
        for view, rows, which in self._kv_rows(slots):
            view[rows] = data[which]

    def _chunks(self):
        """The registered store in codec order, a chunk at a time: (first j, contiguous uint8 blocks, write back)."""
        torch = self.torch
        for r, b in zip(self.regions, self.backings):
            if not r.plane_blocks:
                flat = b.flat[:r.blocks * BLOCK]
                for u in range(0, r.blocks, CHUNK):
                    yield r.first + u, flat[u * BLOCK:min(r.blocks, u + CHUNK) * BLOCK], lambda d: None
                continue
            per = r.planes * r.plane_blocks
            pages = r.blocks // per
            view = torch.as_strided(b.flat, (pages, r.planes, r.plane_blocks * BLOCK), (r.page_stride, r.plane_stride, 1),
                                    b.base)
            step = max(1, CHUNK // per)
            for p0 in range(0, pages, step):
                part = view[p0:p0 + step]
                yield r.first + p0 * per, part.contiguous().view(-1), lambda d, part=part: part.copy_(d.view(part.shape))

    def _words_below_p(self, data) -> None:
        f = self.codec.field
        top = data.view(self.torch.int16).view(-1, f.word_bytes // 2)[:, -1]
        if bool((top == -1).any()):
            words = data.view(-1, f.word_bytes)[top == -1].cpu().numpy()
            if any(int.from_bytes(w.tobytes(), "little") >= f.p for w in words):
                raise ValueError(f"a registered block holds a {f.word_bytes}-byte word ≥ p: no {self.codec.name} "
                                 "payload")

    # --- registration ---------------------------------------------------------------------------------------------

    def payload_root(self, node: bytes) -> bytes:
        torch, dev, N = self.torch, self.dev, self.blocks
        t0 = time.perf_counter()
        width = 1 << max(N - 1, 0).bit_length()
        dom = audit.payload_domain(node, N)
        self.lib.set_tree(bytes(128), BLOCK, BLOCK, dom.domain_id, dev.SALT_DOMAIN + self.key + dev.PAYLOAD_LABEL, N)
        level = torch.empty((width, 8), dtype=torch.int64, device=self.cuda)
        for j0, data, _ in self._chunks():
            self._words_below_p(data)
            self.lib.call("rs_payload_leaves", data.data_ptr(), j0, data.numel() // BLOCK, level.data_ptr() + ROW * j0)
        if N < width:
            self.lib.call("rb_pads", N, width - N, level.data_ptr() + ROW * N)
        up = torch.empty((max(width - 1, 1), 8), dtype=torch.int64, device=self.cuda)
        self.payload_levels = [x.cpu().numpy() for x in dev.climb(self.lib, level, 0, width, up)]
        del level, up
        self.stats["payload_s"] = time.perf_counter() - t0
        return dev.digests(self.payload_levels[-1])[0]

    def payload_path(self, j: int) -> tuple[bytes, ...]:
        lv = self.payload_levels
        return tuple(self.dev.digests(np.stack([lv[h][(j >> h) ^ 1] for h in range(len(lv) - 1)]))) if len(lv) > 1 else ()

    def encode(self, registration: audit.Registration) -> dict:
        self.registration, N = registration, self.blocks
        sample = np.array(sorted({r.first for r in self.regions} | {N - 1}), np.int64)
        before = self._gather(sample).cpu().numpy()
        t0 = time.perf_counter()
        self.device.set_key(registration.key)
        for j0, data, back in self._chunks():
            self.device.encode_in_place(data, j0)
            back(data)
        self.torch.cuda.synchronize()
        self.stats["encode_s"] = secs = time.perf_counter() - t0
        after = self._gather(sample).cpu().numpy()
        for j, m, e in zip(sample.tolist(), before, after):
            if e.tobytes() != rcodec.encode(m.tobytes(), j, registration.key):
                raise RuntimeError(f"kv.cu's block {j}, encoded in place, is not codec.encode's")
        self._gate_bound(registration.key)
        self.gates["device_encode"] = {"registered": N, "appends": 0, "blocks_checked": len(sample) + 1,
                                       "in_place": True, "bound": "a scratch block at registration"}
        self.audit = Appends(registration)
        self._sample = {j: m.tobytes() for j, m in zip(sample.tolist(), before)}
        self._decode_at_use()
        self._credited_read(registration.key)
        return {"encode_bytes_per_s": N * BLOCK / secs}

    def _decode_at_use(self) -> None:
        if self.model is None:
            return
        f = self.codec.field
        weights = encoded_weights(self.store.layout(), self.model, f.word_bytes)
        if not weights:
            return
        self._weight_names = {w.name for w in weights}
        self.decoded = DecodeAtUse(self.model, weights, kv_decoder(self.device.lib, self.codec.rounds, f.word_bytes))
        self.decoded.install()
        self.gates["decode_at_use"] = {"weights": len(weights), "modules": len(self.decoded.modules),
                                       "scratch_bytes": self.decoded.scratch.numel(), "decoded_back": None}

    def _credited_read(self, key: rcodec.Key) -> None:
        """The int8 attention's read of the encoded pool: paged.cu under the registration's key, each KV layer's
        offsets (zero: the seed's key, as registration encoded it) and plaintext flags on the device."""
        if self.paged_lib is None:
            return
        from verity_vllm.engine.int8_window import CreditedRead
        from verity_vllm.protocols.remote_bandwidth import int8_kv

        torch, P, f = self.torch, self.device.paged, self.device.field
        lib = P.PagedLib(f.name, str(self.paged_lib))
        if lib.mask != self.codec.masked or lib.kmax < self.codec.rounds:
            raise ValueError(f"{self.paged_lib} is paged.cu at {f.name}, masked={lib.mask}, at most {lib.kmax} rounds: "
                             f"not {self.codec.name}")
        self.device.set_key(key, lib)
        layout = self.store.layout()
        bases = int8_kv.store_bases(layout, word_bytes=self.codec.field.word_bytes)
        blocks = {int(r.name.split("/", 1)[1]): r.blocks for r in layout.regions
                  if r.kind == "kv" and r.name.startswith("kv/")}
        self.kv_tables = {
            layer: (torch.zeros((n, f.limbs), dtype=torch.int32, device=self.cuda),
                    torch.zeros(n, dtype=torch.uint8, device=self.cuda)) for layer, n in blocks.items()}
        tables = self.kv_tables
        self.credited = CreditedRead(P, lib, int8_kv.layer_keys(bases, bj=lambda layer: tables[layer][0],
                                                                plain=lambda layer: tables[layer][1]),
                                     rounds=self.codec.rounds)
        layers = []
        if self.model is not None:
            from verity_vllm.engine.int8_attention_vllm import install_credited_read

            layers = install_credited_read(self.model, self.credited)
        self.gates["credited_read"] = {
            "layers": len(tables), "installed": len(layers), "slots": sum(blocks.values()),
            "table_bytes": sum(b.numel() * 4 + p.numel() for b, p in tables.values()), "settled": 0}

    def _settle(self, first: int, slots: np.ndarray, offsets: Sequence[int]) -> None:
        """The credited read's offsets and flags of an append's KV slots, once they hold its encoding: block
        ``first + u`` at store slot ``slots[u]`` under b = ``offsets[u]`` at index j is read at the slot's own index,
        under b + τ · block_words · (j − slot)."""
        if not self.kv_tables:
            return
        torch, P, f = self.torch, self.device.paged, self.device.field
        tau, bw, p = self.registration.key.tau, self.codec.field.block_words, self.codec.field.p
        layout = self.store.layout()
        r = np.searchsorted(self._firsts, slots, side="right") - 1
        for ri in sorted(set(r.tolist())):
            reg = layout.regions[ri]
            if reg.kind != "kv" or not reg.name.startswith("kv/"):
                raise ValueError(f"an appended block at slot {slots[r == ri][0]} is in {reg.name}, not a KV layer")
            bj, plain = self.kv_tables[int(reg.name.split("/", 1)[1])]
            at = np.flatnonzero(r == ri)
            x = torch.from_numpy(slots[at] - reg.first).to(self.cuda)
            bj[x] = P.offsets(torch, [(offsets[u] + tau * bw * (first + u - int(slots[u]))) % p for u in at.tolist()], f)
            plain[x] = 0
        self.gates["credited_read"]["settled"] += len(slots)

    def _gate_bound(self, key: rcodec.Key) -> None:
        """kv.cu's encode under an append's bound offset, in place, against `codec.encode`'s, on a scratch block at
        the store's first appended index: the appends' encodes are then the device's alone, off the session's
        critical path."""
        f, j, coin = self.codec.field, self.blocks, self.randbytes(rcodec.COIN_BYTES)
        words = np.frombuffer(self.randbytes(BLOCK), np.uint8).reshape(-1, f.word_bytes).copy()
        words[:, -1] &= 0x7F
        m = words.tobytes()
        data = self.torch.frombuffer(bytearray(m), dtype=self.torch.uint8).to(self.cuda)
        self.device.encode_in_place(data, j, [rcodec.binding(rcodec.digest(m), j, key, coin)])
        if data.cpu().numpy().tobytes() != rcodec.encode(m, j, key, bound=True, coin=coin):
            raise RuntimeError(f"kv.cu's block {j}, encoded in place under a bound offset, is not codec.encode's")

    # --- serving ----------------------------------------------------------------------------------------------------

    def append_salts(self, first: int, count: int) -> list[bytes]:
        return [self.dev.store_salt(self.key, self.dev.PAYLOAD_LABEL, first + u) for u in range(count)]

    def append(self, payloads, salts, slots=()) -> audit.Append:
        if self.trace is not None and self.trace.rekeyed:
            raise RuntimeError("the memory map re-keyed a block, whose payload it reads from its slot, which holds its "
                               "encoding in place: an in-place session's lifetime cap outlasts its windows")
        a = self.audit.append(payloads, salts)
        if self.ip.append([int(x) for x in slots]) != a.first:
            raise AssertionError("InPlace's next index is not the append's")
        self.app_slots.extend(int(x) for x in slots)
        self._app_dev = None
        self.stats["appended"] += a.count
        return a

    def encode_append(self, first: int, coin: bytes) -> None:
        payloads = self.audit.pending.pop(first, None)
        if payloads is None:
            raise ValueError(f"no append from block {first} awaits its coin")
        a, N, k, phases = self.audit.of(first), self.blocks, self.registration.key, self.phases
        slots = np.array(self.app_slots[first - N:first - N + a.count], np.int64)
        with phases.time("encode.check"):
            data = self._gather(slots)
            want = self.torch.frombuffer(bytearray(b"".join(payloads)), dtype=self.torch.uint8).to(self.cuda)
            if bad := self.torch.nonzero((data != want.view(-1, BLOCK)).any(dim=1)).flatten().tolist():
                raise RuntimeError(f"slot {slots[bad[0]]} no longer holds block {first + bad[0]}'s payload when its "
                                   "append's coin arrives: rewritten before its encode")
        with phases.time("encode.bind"):
            offsets = [rcodec.binding(d, first + u, k, coin) for u, d in enumerate(a.digests)]
        with phases.time("encode.device"):
            self.device.encode_in_place(data.reshape(-1), first, offsets)
            self._scatter(slots, data)
            self._settle(first, slots, offsets)
            self.torch.cuda.synchronize()
        self.audit.coins[first] = coin
        self.stats["encoded_appends"] += 1
        self.gates["device_encode"]["appends"] += 1
        if self.trace is not None:
            self.trace.encoded(first)

    def _before_write(self, step: int, keys: np.ndarray) -> None:
        """`Store.on_write`: the pool blocks ``keys`` the step's forward is about to write, copied first."""
        if not len(keys):
            return
        p, store = self.store.pool, self.store
        ids = p.ids[keys // p.per_page, keys % p.per_page]
        slots = np.sort(np.concatenate([store.region(rn).first + ids for rn in p.regions]).astype(np.int64))
        with self.phases.time("copy"):
            data = self._gather(slots)
            host = data.cpu().numpy()
        if self.saved:
            self.saved[-1].device = None
        self.saved.append(Saved(step, slots, host, data))
        self._windex = None
        self.stats["saved_blocks"] += len(slots)

    def holds(self, slot: int, j: int, digest: bytes) -> bool:
        block = self._gather(np.array([slot])).cpu().numpy()[0].tobytes()
        coin = self.audit.coins.get(self.audit.of(j).first)
        m = block if coin is None else rcodec.decode(block, j, self.registration.key, digest, coin)
        return m is not None and rcodec.digest(m) == digest

    # --- calibration ------------------------------------------------------------------------------------------------

    def _whole_passes(self, n: int | None = None, ns: int | None = None) -> tuple[list[int], list[int]]:
        """Passes over the whole registered store, each seeded from the root before as a session chains them, until
        ``n`` of them or ``ns`` have run: each root's time and each root's gap, the host's time from the root to where
        a session sends it (the plan's change against the last sent, and the message)."""
        prover, plan, challenge = session_module("prover"), rb.Plan.whole(self.blocks), self.randbytes(32)
        prev, at, gaps, t0 = bytes(64), [], [], time.perf_counter_ns()
        while (n is not None and len(at) < n) or (ns is not None and time.perf_counter_ns() - t0 < ns):
            root, _ = self.ip.run_pass(rb.seed(challenge, len(at), prev), plan)
            self.torch.cuda.synchronize()
            at.append(time.perf_counter_ns())
            msg = {"type": "root", "t": 0, "i": len(at), "root": root.hex(), "step": len(at),
                   "change": [[list(r) for r in side] for side in prover.plan_change(plan, plan)]}
            json.dumps(msg)
            gaps.append(time.perf_counter_ns() - at[-1])
            prev = root
        return at, gaps

    def warmup(self, ns: int) -> dict:
        """Untimed passes before the session's first window (`ProverSide`'s warm-up), so its first passes don't pay the
        device's ramp."""
        at, _ = self._whole_passes(ns=ns)
        return {"passes": len(at), "ms": (at[-1] - at[0]) / 1e6 if at else 0.0}

    def calibrate(self) -> dict:
        """`device.Device.calibrate`'s report from this store read in place, before the session: the bare read of every
        registered block where it is (`rs_mapped_read`, the best of 6 after the first), then ``calibration_passes``
        passes over the whole store chained as a session's (the first ``calibration_warm`` untimed): each root's
        interval from the one before (the allowance's pass) and each root's gap to its send, at the percentiles the
        verifier's allowance reads (`device.percentiles`).  An engine step in lockstep, the appends' round trips and
        their encodes are not in it (a paced pass's step is: `with_step`); the session's record times them
        (`phases_ms`)."""
        torch, N = self.torch, self.blocks
        addrs, _ = self.ip.map(rb.Plan.whole(N))
        xr = torch.zeros(1, dtype=torch.int64, device=self.cuda)
        ctas = 16 * torch.cuda.get_device_properties(self.cuda).multi_processor_count
        reads = []
        for _ in range(7):
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            self.lib.call("rs_mapped_read", addrs.data_ptr(), N, xr.data_ptr(), ctas)
            torch.cuda.synchronize()
            reads.append(time.perf_counter_ns() - t0)
        del addrs
        read_ms = min(reads[1:]) / 1e6
        self._whole_passes(n=self.calibration_warm)
        at, gaps = self._whole_passes(n=self.calibration_passes)
        intervals, gaps = sorted(b - a for a, b in zip(at, at[1:])), sorted(gaps[1:])
        pct = self.dev.percentiles
        bps = N * BLOCK / (read_ms / 1e3)
        return self.with_step({"pass_ns_max": intervals[-1], "pass_ns_median": intervals[len(intervals) // 2],
                "pass_ns": pct(intervals), "gap_ns_max": gaps[-1], "gap_ns_median": gaps[len(gaps) // 2],
                "gap_ns": pct(gaps), "passes": len(intervals), "warm_passes": self.calibration_warm,
                "read_ms": read_ms, "read_bytes_per_s": bps,
                "read_tiers": {"hbm": {"bytes": N * BLOCK, "ms": read_ms, "bytes_per_s": bps}},
                "tau": intervals[len(intervals) // 2] / 1e6 / read_ms,
                "max_over_median": intervals[-1] / intervals[len(intervals) // 2],
                "in_place": True, "device": torch.cuda.get_device_name(self.cuda)})

    # --- passes and openings ----------------------------------------------------------------------------------------

    def arm(self, dropped) -> None:
        super().arm(dropped)
        self._dropped_dev = (self.torch.tensor(sorted(self.dropped), dtype=self.torch.int64, device=self.cuda)
                             if self.dropped else None)

    def launch(self, i: int, after: dict | None, plan: rb.Plan | None = None) -> dict:
        ticket = super().launch(i, after, plan)
        prev = audit.chained(ticket["previous"] if after is None else after["root"], ticket["head"])
        ticket["root"], ticket["handle"] = self.run_pass(rb.seed(ticket["challenge"], i, prev), plan, ticket["step"])
        return ticket

    def wait(self, ticket: dict):
        return ticket["root"], ticket["handle"]

    def run_pass(self, seed: bytes, plan: rb.Plan | None = None, step: int | None = None):
        with self.phases.time("moved"):
            if moved := self.moved():
                raise RuntimeError(f"registered tensors moved since registration ({', '.join(moved[:8])}): the pass "
                                   "reads the store at their registered addresses, so the session fails")
            plan = plan or rb.Plan.whole(self.blocks)
            runs = np.frombuffer(plan.encode(), dtype=">u8", offset=4).reshape(-1, 2).astype(np.int64)
            for first, payloads in self.audit.pending.items():
                if ((runs[:, 0] < first + len(payloads)) & (runs[:, 0] + runs[:, 1] > first)).any():
                    raise RuntimeError(f"the plan holds the append from {first}, not yet encoded (deferred residency)")
        self._mode = ("pass", step)
        t0 = time.perf_counter_ns()
        with self.phases.time("pass"):
            root, handle = self.ip.run_pass(seed, plan)
            self.torch.cuda.synchronize()
        self.pass_ns.append(time.perf_counter_ns() - t0)
        self._mode, self._held = None, []
        self.phases.close()
        return root, (handle, step)

    def _slots_of(self, js):
        torch, N = self.torch, self.blocks
        if not self.app_slots:
            return js
        if self._app_dev is None or len(self._app_dev) != len(self.app_slots):
            self._app_dev = torch.tensor(self.app_slots, dtype=torch.int64, device=self.cuda)
        return torch.where(js < N, js, self._app_dev[(js - N).clamp(0, len(self.app_slots) - 1)])

    def _patched(self, addrs, js):
        """The map with the session's substitutions (the module's docstring), by the mode it is read in."""
        torch, mode = self.torch, self._mode
        if mode is not None and mode[0] == "pass":
            if self.saved and self.saved[-1].step == mode[1] and self.saved[-1].device is not None:
                s = self.saved[-1]
                targets = torch.from_numpy(s.slots).to(self.cuda)
                sl = self._slots_of(js)
                at = torch.searchsorted(targets, sl).clamp(max=len(targets) - 1)
                hit = targets[at] == sl
                addrs[hit] = s.device.data_ptr() + BLOCK * at[hit]
                self.stats["pass_patched"] += int(hit.sum())
        elif mode is not None and mode[0] == "open":
            self._patch_open(addrs, js, *mode[1:])
        if self._dropped_dev is not None:
            d = self._dropped_dev
            at = torch.searchsorted(d, js).clamp(max=len(d) - 1)
            addrs[d[at] == js] = self.zero.data_ptr()
        return addrs, js

    def _write_index(self):
        if self._windex is None:
            if not self.saved:
                self._windex = (np.zeros(0, np.int64),) * 3
            else:
                key = np.concatenate([s.slots << STEP_BITS | s.step for s in self.saved])
                which = np.concatenate([np.full(len(s.slots), k) for k, s in enumerate(self.saved)])
                row = np.concatenate([np.arange(len(s.slots)) for s in self.saved])
                o = np.argsort(key, kind="stable")
                self._windex = (key[o], which[o], row[o])
        return self._windex

    def _patch_open(self, addrs, js, step: int, handle, positions: Sequence[int]) -> None:
        """An opening's subtrees read each block rewritten since the pass from the earliest copy taken at or after the
        pass's step, before the write that followed it."""
        key, which, row = self._write_index()
        if not len(key) or step is None:
            return
        n, sub = handle.plan.length, rb.PER_LEAF << handle.h0
        pos = np.concatenate([np.arange(s, min(s + sub, n)) for s in sorted({p // sub * sub for p in positions})])
        pos_d = self.torch.from_numpy(pos).to(self.cuda)
        sl = self._slots_of(js[pos_d]).cpu().numpy()
        q = sl << STEP_BITS | step
        at = np.searchsorted(key, q)
        ok = at < len(key)
        ok[ok] = (key[at[ok]] >> STEP_BITS) == sl[ok]
        if not ok.any():
            return
        rows = np.stack([self.saved[w].host[r] for w, r in zip(which[at[ok]], row[at[ok]])])
        buf = self.torch.from_numpy(rows).to(self.cuda)
        self._held.append(buf)
        addrs[pos_d[self.torch.from_numpy(np.flatnonzero(ok)).to(self.cuda)]] = (
            buf.data_ptr() + BLOCK * self.torch.arange(len(rows), device=self.cuda))
        self.stats["open_patched"] += len(rows)

    def _subtree(self, nodes, c0: int, width: int):
        mode = self._mode
        if mode is not None and mode[0] == "open":
            h = mode[2]
            if width == 1 << h.h0 and not self.torch.equal(nodes[2 * width - 2], h.kept[c0 >> h.h0]):
                self._stale.append(c0)
        return nodes

    def open_window(self, items) -> list[list[tuple]]:
        out = [self.open(handle, positions) for handle, positions in items]
        self._held = []
        return out

    def open(self, handle, positions: list[int]) -> list[tuple]:
        h, step = handle
        self._mode, self._stale = ("open", step, h, positions), []
        try:
            got = self.ip.open(h, positions)
        finally:
            self._mode = None
        if self._stale:
            sub = rb.PER_LEAF << h.h0
            gone = sorted({h.plan.block(p) for p in positions if p // sub * sub // rb.PER_LEAF in self._stale})
            raise Stale(f"the pass at step {step}: blocks {gone[:8]} were rewritten after its receipts, with no copy "
                        "taken before the write")
        return [(*g, *self.committed(h.plan.block(p))) for g, p in zip(got, positions)]

    def committed(self, j: int) -> tuple[bytes, tuple[bytes, ...], bytes]:
        if j < self.blocks:
            return self.dev.store_salt(self.key, self.dev.PAYLOAD_LABEL, j), self.payload_path(j), b""
        a = self.audit.of(j)
        return a.salts[j - a.first], a.tree.open(j).path, a.digests[j - a.first]

    def close(self) -> None:
        """The session is over: the engine's later writes are copied no more, and the weights are decoded back where
        they are (what the Commit's replay reads), checked at each weight region's first block against its bytes at
        registration."""
        if self.store.on_write == self._before_write:
            self.store.on_write = None
        self.saved, self._windex = [], None
        if self.credited is not None:
            self.gates["credited_read"].update(steps=self.credited.steps, windows=self.credited.windows)
        if self.decoded is not None and self.decoded.handles:
            self.decoded.remove()
            self.decoded.decode_in_place()
            firsts = np.array([j for j in sorted(self._sample) if j < self.blocks and self._is_weight(j)], np.int64)
            got = self._gather(firsts).cpu().numpy()
            bad = [int(j) for j, g in zip(firsts, got) if g.tobytes() != self._sample[int(j)]]
            self.gates["decode_at_use"].update(decoded_back=not bad, blocks_checked=len(firsts),
                                               calls=self.decoded.calls)
            if bad:
                raise RuntimeError(f"weight block {bad[0]}, decoded back after the session, is not its bytes at "
                                   "registration")

    def _is_weight(self, j: int) -> bool:
        r = self.store.layout().regions[int(np.searchsorted(self._firsts, j, side="right")) - 1]
        return r.kind == "weights" and r.name in self._weight_names

    @classmethod
    def for_commit(cls, session_lib: str, paged_lib: str | None = None) -> Callable[[Any, Any], InPlaceBackend]:
        """`CommitSession`'s ``backend_of``: the backend over its store, in lockstep with its run (with ``paged_lib``,
        the int8 attention's credited read)."""
        def build(cs, path) -> InPlaceBackend:
            backings = [cs.store.backing(r.name) for r in cs.store.layout().regions]
            return cls(cs.store, cs.lockstep, backings, session_lib=session_lib, kv_lib=cs.kv_lib, codec=cs.codec,
                       moved=cs.binder.moved, path=path, model=cs.binder.model, paged_lib=paged_lib)
        return build

    def session_stats(self) -> dict:
        ms = sorted(x / 1e6 for x in self.pass_ns)
        at = lambda q: ms[min(len(ms) - 1, int(q * len(ms)))] if ms else None
        return {"in_place": True, "passes": len(ms), "pass_ms": {"p50": at(0.5), "p99": at(0.99), "max": at(1.0),
                                                                 "mean": sum(ms) / len(ms) if ms else None},
                "store_blocks": self.ip.blocks, "saved_steps": len(self.saved), "phases_ms": self.phases.summary(),
                "intervals_ms": [{k: round(v / 1e6, 2) for k, v in x.items()} for x in self.phases.closed],
                **self.stats}
