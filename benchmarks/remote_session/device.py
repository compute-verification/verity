"""The prover's GPU backend: an encoded store on one GPU, its payload tree, chained sketch-sha512 passes over it with
the device reader (`../codec_tensor/reader.cu` in `session.cu`) and the openings of what the verifier draws.

~~~text
payload    N blocks of seeded random bytes, each 128-byte word's top byte below 0x80 (so below p1024 and p512), its
           tree the driver's `audit.payload_tree` with salt_j = SHAKE256(D(store salt) ‖ key ‖ PAYLOAD_LABEL ‖ u64be j)
store      E = the codec over the payload in place, by kv.cu (`../word_codec/kv.cu`) at the registration's key
pass       sketch-sha512 receipts of every block in store order (`rb_sketch`, `rb_sketch_hash`), the leaves under
           salt_c = SHAKE256(D(store salt) ‖ key ‖ seed ‖ u64be c) (`rb_leaves`) and the whole tree (`rb_tree`); its
           leaves and tree are kept for the window's openings, its receipts are not
opening    the drawn leaf's 16 blocks gathered from the store, sketched and hashed again under the pass's signs
           (`rs_sketch_hash_idx`), on a stream and a second copy of the library of its own (a library's pass constants
           are global), with the paths from the kept trees
skip       the dropped blocks zeroed on the device, their true blocks kept on the host for openings
~~~

Gates, before anything is timed (`Device.gates`): the payload root against `audit.payload_tree` and blocks of the
store against `codec.encode` (at the store's last index too), a pass's root against `receipts.tree` over the
reference's receipts, and two sessions over a small store under the registration's key, one honest and accepted by
`audit.Session`, one with half its blocks dropped and rejected at a receipt.
"""
from __future__ import annotations

import ctypes
import hashlib
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "codec_tensor"))
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.remote_bandwidth import audit, coins, receipts as rb  # noqa: E402
from verity.core.protocols.remote_bandwidth import codec as rcodec  # noqa: E402
from verity.ml.commitments.rowleaf import sha512_row_v2_prefix  # noqa: E402

import sketch_receipt as skr  # noqa: E402
from prover import PAYLOAD_LABEL  # noqa: E402

BLOCK = rcodec.BLOCK_BYTES
SALT_DOMAIN = b"verity/store/salt/v0\0"
PER_LEAF = rb.PER_LEAF


def store_salt(key: bytes, label: bytes, c: int) -> bytes:
    return hashlib.shake_256(SALT_DOMAIN + key + label + struct.pack(">Q", c)).digest(rb.SALT_BYTES)


def digests(words) -> list[bytes]:
    """Digests from rows of 8 state words (an n x 8 int64 tensor or array)."""
    a = np.asarray(words.cpu().numpy() if hasattr(words, "cpu") else words).astype(np.uint64).astype(">u8")
    return [r.tobytes() for r in a.reshape(-1, 8)]


class Lib:
    def __init__(self, path: str, torch) -> None:
        self.so, self.torch = ctypes.CDLL(path), torch
        v, u64, i, b = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p
        self.so.rb_set.argtypes = [b, b, b, b, b, b, b, b, i, b, i, u64, v]
        self.so.rb_leaves.argtypes = [v, u64, u64, v, v, v]
        self.so.rb_tree.argtypes = [v, u64, u64, v, v]
        self.so.rb_read.argtypes = [v, u64, v, i, v]
        self.so.rb_sketch_set.argtypes = [b, v]
        self.so.rb_sketch.argtypes = [v, v, u64, v, v, i, v]
        self.so.rb_sketch_hash.argtypes = [v, u64, u64, v, v]
        self.so.rs_payload_leaves.argtypes = [v, u64, u64, v, v]
        self.so.rs_sketch_hash_idx.argtypes = [v, v, u64, v, v]

    def call(self, fn: str, *args) -> None:
        rc = getattr(self.so, fn)(*args, self.torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"{fn}: {rc}")

    def set_tree(self, receipt_prefix: bytes, row_bytes: int, last_row_bytes: int, domain_id: bytes, salt_in: bytes,
                 positions: int) -> None:
        self.call("rb_set", receipt_prefix, sha512_row_v2_prefix(8 * row_bytes), sha512_row_v2_prefix(8 * last_row_bytes),
                  rb.HM96.salt_prefix, rb.HM96.leaf_prefix(rb.KEY), rb.KEY, domain_id, rb.SCHEMA.encode(),
                  len(rb.SCHEMA), salt_in, len(salt_in), positions)


class Kv:
    """kv.cu's encode at the library's width, masked or not."""

    def __init__(self, path: str, torch) -> None:
        self.so, self.torch = ctypes.CDLL(path), torch
        v, q, i = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int
        self.so.kv_set_key.argtypes = [v, i, v]
        self.so.kv_set_mask.argtypes = [v]
        self.so.kv_encode.argtypes = [v, v, q, q, i, i, v]
        self.bits, self.masked = self.so.kv_bits(), bool(self.so.kv_mask())

    def call(self, fn: str, *args) -> None:
        rc = getattr(self.so, fn)(*args)
        if rc:
            raise RuntimeError(f"{fn}: {rc}")

    def set_key(self, key: rcodec.Key) -> None:
        f = key.codec.field
        if f.n != self.bits or key.codec.masked != self.masked:
            raise ValueError(f"kv.so is built for {self.bits} bits, masked={self.masked}, not {key.codec.name}")
        s = f.word_bytes
        c = ctypes.create_string_buffer(b"".join(x.to_bytes(s, "little") for x in key.c), s * key.rounds)
        t = ctypes.create_string_buffer(key.tau.to_bytes(s, "little"), s)
        self.call("kv_set_key", ctypes.cast(c, ctypes.c_void_p), key.rounds, ctypes.cast(t, ctypes.c_void_p))
        if self.masked:
            m = ctypes.create_string_buffer(key.mask.to_bytes(s, "little"), s)
            self.call("kv_set_mask", ctypes.cast(m, ctypes.c_void_p))
        self.rounds, self.words = key.rounds, f.block_words

    def encode(self, data, j0: int, n: int) -> None:
        """Blocks j0 … j0 + n − 1, at `data`, in place."""
        self.call("kv_encode", data.data_ptr(), data.data_ptr(), n * self.words, j0 * self.words, self.rounds, 0,
                  self.torch.cuda.current_stream().cuda_stream)


def levels_of(torch, dig, up) -> list:
    out, n, at = [dig], len(dig), 0
    while n > 1:
        n //= 2
        out.append(up[at:at + n])
        at += n
    return out


def paths(torch, levels: list, idx: list[int]) -> list[tuple[bytes, ...]]:
    """Each leaf's path in `idx`, sibling by sibling up to the root, in one copy from the device."""
    if len(levels) == 1:
        return [() for _ in idx]
    rows = torch.cat([lv[torch.tensor([(c >> h) ^ 1 for c in idx], device=lv.device)]
                      for h, lv in enumerate(levels[:-1])])
    flat = digests(rows)
    H, n = len(levels) - 1, len(idx)
    return [tuple(flat[h * n + q] for h in range(H)) for q in range(n)]


class DeviceStore:
    """N blocks on the GPU: the payload, its tree, then the encoded store, and the passes over it."""

    def __init__(self, torch, lib: Lib, opener: Lib, kv: Kv, blocks: int, key: bytes, generator) -> None:
        nl = blocks // PER_LEAF
        if blocks % PER_LEAF or nl & (nl - 1):
            raise ValueError(f"{blocks} blocks: the harness takes N = 16 · 2^h, so every tree is whole")
        self.torch, self.lib, self.opener, self.kv, self.blocks, self.key = torch, lib, opener, kv, blocks, key
        self.data = torch.empty(blocks * BLOCK, dtype=torch.uint8, device="cuda")
        chunk = 1 << 18
        for j in range(0, blocks, chunk):
            m = min(chunk, blocks - j)
            v = self.data[j * BLOCK:(j + m) * BLOCK]
            v.view(torch.int32).copy_(torch.randint(-2 ** 31, 2 ** 31 - 1, (m * BLOCK // 4,), dtype=torch.int32,
                                                    device="cuda", generator=generator))
            v.view(-1, 128)[:, 127] &= 0x7F
        self.plan = rb.Plan.whole(blocks)
        self.sk = torch.empty((blocks, skr.ROWS), dtype=torch.int32, device="cuda")
        self.ys = torch.empty((blocks, 8), dtype=torch.int64, device="cuda")
        self.dropped: dict[int, bytes] = {}
        self.pass_gpu_ms: list[float] = []

    def block(self, j: int) -> bytes:
        return self.dropped[j] if j in self.dropped else self.data[j * BLOCK:(j + 1) * BLOCK].cpu().numpy().tobytes()

    def host_blocks(self, j0: int, n: int) -> list[bytes]:
        raw = self.data[j0 * BLOCK:(j0 + n) * BLOCK].cpu().numpy().tobytes()
        return [raw[q * BLOCK:(q + 1) * BLOCK] for q in range(n)]

    def payload_tree(self, node: bytes) -> bytes:
        torch, N = self.torch, self.blocks
        dom = audit.payload_domain(node, N)
        self.lib.set_tree(bytes(128), BLOCK, BLOCK, dom.domain_id, SALT_DOMAIN + self.key + PAYLOAD_LABEL, N)
        dig = torch.empty((N, 8), dtype=torch.int64, device="cuda")
        self.lib.call("rs_payload_leaves", self.data.data_ptr(), 0, N, dig.data_ptr())
        up = torch.empty((N - 1, 8), dtype=torch.int64, device="cuda")
        self.lib.call("rb_tree", dig.data_ptr(), N, 0, up.data_ptr())
        self.payload_levels = levels_of(torch, dig, up)
        self.payload_root = digests(self.payload_levels[-1])[0]
        return self.payload_root

    def encode(self, key: rcodec.Key, chunk: int = 1 << 16) -> float:
        """The store from its payload, in place; the encode's rate in bytes a second of GPU time."""
        torch = self.torch
        self.kv.set_key(key)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for j in range(0, self.blocks, chunk):
            m = min(chunk, self.blocks - j)
            self.kv.encode(self.data[j * BLOCK:], j, m)
        b.record()
        torch.cuda.synchronize()
        return self.blocks * BLOCK / (a.elapsed_time(b) / 1e3)

    def run_pass(self, seed: bytes):
        torch, N, nl = self.torch, self.blocks, self.blocks // PER_LEAF
        signs = torch.from_numpy(skr.signs(seed)).cuda()
        dom = rb.domain(seed, self.plan)
        self.lib.set_tree(rb.prefix(seed), rb.DIGEST_BYTES * PER_LEAF, rb.DIGEST_BYTES * PER_LEAF, dom.domain_id,
                          SALT_DOMAIN + self.key + seed, N)
        self.lib.call("rb_sketch_set", skr.prefix(seed))
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        self.lib.call("rb_sketch", self.data.data_ptr(), None, N, signs.data_ptr(), self.sk.data_ptr(), 0)
        self.lib.call("rb_sketch_hash", self.sk.data_ptr(), 0, N, self.ys.data_ptr())
        dig = torch.empty((nl, 8), dtype=torch.int64, device="cuda")
        self.lib.call("rb_leaves", self.ys.data_ptr(), 0, nl, dig.data_ptr(), None)
        up = torch.empty((max(nl - 1, 1), 8), dtype=torch.int64, device="cuda")
        if nl > 1:
            self.lib.call("rb_tree", dig.data_ptr(), nl, 0, up.data_ptr())
        b.record()
        root = digests((up[nl - 2] if nl > 1 else dig[0]).cpu())[0]
        self.pass_gpu_ms.append(a.elapsed_time(b))
        return root, (seed, signs, dig, up if nl > 1 else up[:0])

    def arm(self, dropped: set[int]) -> None:
        """Restore the blocks dropped before, then drop `dropped`: zeroed, their true blocks kept on the host."""
        for j, blk in self.dropped.items():
            self.data[j * BLOCK:(j + 1) * BLOCK].copy_(self.torch.frombuffer(bytearray(blk), dtype=self.torch.uint8))
        self.dropped = {}
        for j in sorted(dropped):
            self.dropped[j] = self.data[j * BLOCK:(j + 1) * BLOCK].cpu().numpy().tobytes()
            self.data[j * BLOCK:(j + 1) * BLOCK].zero_()
        self.torch.cuda.synchronize()

    def open(self, handle, positions: list[int]) -> list[tuple]:
        """Each position's (block, leaf, leaf salt, receipt path, payload salt, payload path), on the opener's stream
        and library."""
        torch = self.torch
        seed, signs, dig, up = handle
        cs = sorted({p // PER_LEAF for p in positions})
        js = torch.tensor([PER_LEAF * c + r for c in cs for r in range(PER_LEAF)], dtype=torch.int64, device="cuda")
        blocks = self.data.view(-1, BLOCK)[js].contiguous()
        self.opener.call("rb_sketch_set", skr.prefix(seed))
        sk = torch.empty((len(js), skr.ROWS), dtype=torch.int32, device="cuda")
        ys = torch.empty((len(js), 8), dtype=torch.int64, device="cuda")
        self.opener.call("rb_sketch", blocks.data_ptr(), None, len(js), signs.data_ptr(), sk.data_ptr(), 0)
        self.opener.call("rs_sketch_hash_idx", sk.data_ptr(), js.data_ptr(), len(js), ys.data_ptr())
        ys = digests(ys)
        leaf_of = {c: b"".join(ys[PER_LEAF * q:PER_LEAF * (q + 1)]) for q, c in enumerate(cs)}
        rpath = dict(zip(cs, paths(torch, levels_of(torch, dig, up), cs)))
        ppath = paths(torch, self.payload_levels, positions)
        ptensor = torch.tensor(positions, dtype=torch.int64, device="cuda")
        raw = self.data.view(-1, BLOCK)[ptensor].cpu().numpy()
        out = []
        for q, p in enumerate(positions):
            c = p // PER_LEAF
            blk = self.dropped.get(p) or raw[q].tobytes()
            out.append((blk, leaf_of[c], store_salt(self.key, seed, c), rpath[c], store_salt(self.key, PAYLOAD_LABEL, p),
                        ppath[q]))
        return out


class Device:
    """The prover's backend (`prover.ProverSide`): the store, its gates and calibration, its passes and openings."""

    def __init__(self, torch, lib_path: str, kv_path: str, blocks: int, *, seed: int, log=print,
                 gate_blocks: int = 256, calibration_passes: int = 40) -> None:
        self.torch, self.log, self.blocks = torch, log, blocks
        self.gate_blocks, self.calibration_passes = gate_blocks, calibration_passes
        self.lib = Lib(lib_path, torch)
        self.tmp = tempfile.mkdtemp(prefix="rs-opener.")
        twin = shutil.copy(lib_path, str(Path(self.tmp) / "session-opener.so"))
        self.opener = Lib(twin, torch)
        self.kv = Kv(kv_path, torch)
        self.gen = torch.Generator(device="cuda")
        self.gen.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.salt_key = self.rng.bytes(32)
        self.main_stream = torch.cuda.Stream()
        self.open_stream = torch.cuda.Stream()
        self.gates: dict = {}
        with torch.cuda.stream(self.main_stream):
            self.store = DeviceStore(torch, self.lib, self.opener, self.kv, blocks, self.salt_key, self.gen)
        torch.cuda.synchronize()

    def payload_root(self, node: bytes) -> bytes:
        self.node = node
        with self.torch.cuda.stream(self.main_stream):
            root = self.store.payload_tree(node)
        self.torch.cuda.synchronize()
        return root

    def encode(self, registration: audit.Registration) -> dict:
        self.gates = self.gate(registration)
        self.log(f"device: gates {self.gates}")
        if not self.gates["ok"]:
            raise RuntimeError(f"a gate failed, nothing is timed: {self.gates}")
        with self.torch.cuda.stream(self.main_stream):
            rate = self.store.encode(registration.key)
        self.torch.cuda.synchronize()
        last = self.blocks - 1
        self.gates["encoded_last_block"] = self.store.block(last) == self.reference_last
        if not self.gates["encoded_last_block"]:
            raise RuntimeError("the store's last block is not the codec's encode of its payload")
        self.encode_bytes_per_s = rate
        return {"encode_bytes_per_s": rate, "blocks": self.blocks, "bytes": self.blocks * BLOCK}

    def calibrate(self) -> dict:
        torch, s = self.torch, self.store
        xr = torch.zeros(1, dtype=torch.int64, device="cuda")
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        reads = []
        with torch.cuda.stream(self.main_stream):
            for _ in range(6):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                self.lib.call("rb_read", s.data.data_ptr(), s.blocks * BLOCK, xr.data_ptr(), 16 * sms)
                b.record()
                b.synchronize()
                reads.append(a.elapsed_time(b))
            prev, pass_ns, gap_ns = bytes(64), [], []
            challenge = self.rng.bytes(32)
            t_end = None
            for i in range(self.calibration_passes):
                t0 = time.perf_counter_ns()
                seed = rb.seed(challenge, i, prev)
                t1 = time.perf_counter_ns()
                prev, _ = s.run_pass(seed)
                t2 = time.perf_counter_ns()
                if t_end is not None:
                    gap_ns.append(t1 - t_end)
                pass_ns.append(t2 - t1)
                t_end = t2
        gpu = s.pass_gpu_ms[-self.calibration_passes:]
        s.pass_gpu_ms.clear()
        read_ms = min(reads[1:])
        return {"pass_ns_max": max(pass_ns[1:]), "pass_ns_median": sorted(pass_ns)[len(pass_ns) // 2],
                "gap_ns_max": max(gap_ns[1:]), "gap_ns_median": sorted(gap_ns)[len(gap_ns) // 2],
                "pass_gpu_ms_median": sorted(gpu)[len(gpu) // 2], "read_ms": read_ms,
                "read_bytes_per_s": s.blocks * BLOCK / (read_ms / 1e3),
                "tau": sorted(gpu)[len(gpu) // 2] / read_ms, "device": torch.cuda.get_device_name(0)}

    def arm(self, dropped: set[int]) -> None:
        with self.torch.cuda.stream(self.main_stream):
            self.store.arm(dropped)
        self.store.pass_gpu_ms.clear()

    def run_pass(self, seed: bytes):
        with self.torch.cuda.stream(self.main_stream):
            return self.store.run_pass(seed)

    def open(self, handle, positions: list[int]) -> list[tuple]:
        with self.torch.cuda.stream(self.open_stream):
            return self.store.open(handle, positions)

    def session_stats(self) -> dict:
        gpu = self.store.pass_gpu_ms
        return {"passes": len(gpu), "pass_gpu_ms_sum": sum(gpu), "pass_gpu_ms_median": sorted(gpu)[len(gpu) // 2]
                if gpu else None, "pass_gpu_ms_max": max(gpu) if gpu else None}

    def gate(self, registration: audit.Registration) -> dict:
        torch, key, out = self.torch, registration.key, {}
        last = self.blocks - 1
        self.reference_last = rcodec.encode(self.store.block(last), last, key)
        g = torch.Generator(device="cuda")
        g.manual_seed(int(self.rng.integers(1 << 62)))
        with torch.cuda.stream(self.main_stream):
            small = DeviceStore(torch, self.lib, self.opener, self.kv, self.gate_blocks, self.salt_key, g)
            payloads = small.host_blocks(0, small.blocks)
            root = small.payload_tree(self.node)
            salts = [store_salt(self.salt_key, PAYLOAD_LABEL, j) for j in range(small.blocks)]
            out["payload_root"] = root == audit.payload_tree(self.node, payloads, salts).commitment.root
            small.encode(key)
            encoded = small.host_blocks(0, small.blocks)
        torch.cuda.synchronize()
        out["encode"] = all(encoded[j] == rcodec.encode(payloads[j], j, key) for j in (0, 1, small.blocks - 1))
        out["decode"] = all(rcodec.decode(encoded[j], j, key) == payloads[j] for j in range(small.blocks))
        reg = audit.Registration(registration.node, small.blocks, root, registration.codec_seed, registration.codec,
                                 registration.receipt, registration.bind)
        seed = rb.seed(self.rng.bytes(32), 0, reg.digest)
        with torch.cuda.stream(self.main_stream):
            got, _ = small.run_pass(seed)
        r = skr.SketchReceipter(seed)
        ys = [r(j, encoded[j], bytes(BLOCK)) for j in range(small.blocks)]
        ref = rb.tree(seed, small.plan, ys, [store_salt(self.salt_key, seed, c) for c in range(small.blocks // PER_LEAF)])
        out["pass_root"] = got == ref.commitment.root
        out["session_honest"] = self.gate_session(small, reg, set())
        out["session_skip_half"] = self.gate_session(small, reg, set(range(0, small.blocks, 2)))
        out["ok"] = (out["payload_root"] and out["encode"] and out["decode"] and out["pass_root"]
                     and out["session_honest"]["accepted"] and out["session_skip_half"]["failure"] == "receipt")
        del small
        torch.cuda.empty_cache()
        return out

    def gate_session(self, small: DeviceStore, reg: audit.Registration, dropped: set[int]) -> dict:
        """Two windows of two passes each over `small` under a clock the gate turns, checked by `audit.Session`."""
        now = [0]
        params = audit.Params(reg.blocks, 1000, 2, 24, 2.0 ** -40, codec=reg.codec, receipt=reg.receipt)
        session = audit.Session(params, reg, None, coins.Challenger(reg.digest, 0, params.coins), clock=lambda: now[0])
        with self.torch.cuda.stream(self.main_stream):
            small.arm(dropped)
        coin, prev, kept, verdicts = session.start(), reg.digest, {}, []
        for t in range(2):
            for i in range(2):
                with self.torch.cuda.stream(self.main_stream):
                    prev, kept[t, i] = small.run_pass(rb.seed(coin.coin, i, prev))
                now[0] += 1
                if session.receive(t, i, prev) is not None:
                    return {"accepted": False, "failure": session.failed.failure}
            coin = session.boundary()
            now[0] += 1
        for t in range(2):
            if t == 1:
                now[0] += 10_000
            draws = session.close(t)
            if isinstance(draws, audit.Verdict):
                return {"accepted": False, "failure": draws.failure}
            openings = []
            for i, p in draws:
                with self.torch.cuda.stream(self.open_stream):
                    f = small.open(kept[t, i], [p])[0]
                openings.append(audit.Opening(i, p, f[0], f[1], f[2], tuple(f[3]), f[4], tuple(f[5]), b""))
            v = session.verify(t, openings)
            verdicts.append(v)
            if not v.accepted:
                return {"accepted": False, "failure": v.failure, "at": v.at, "dropped": v.at[1] in dropped}
        with self.torch.cuda.stream(self.main_stream):
            small.arm(set())
        return {"accepted": True, "failure": None, "draws": sum(len(w.draws) for w in session.windows)}


def add_arguments(ap) -> None:
    ap.add_argument("--lib", required=True, help="session.so")
    ap.add_argument("--kv", required=True, help="kv.so at the registration's codec (p1024 masked)")
    ap.add_argument("--gib", type=float, default=16.0, help="the store's size; N = 16 · 2^h blocks")
    ap.add_argument("--gate-blocks", type=int, default=256)
    ap.add_argument("--calibration-passes", type=int, default=40)


def from_arguments(a, log=print) -> Device:
    import torch

    n = int(a.gib * 2 ** 30) // BLOCK
    blocks = PER_LEAF << ((n // PER_LEAF).bit_length() - 1)
    return Device(torch, a.lib, a.kv, blocks, seed=a.seed, log=log, gate_blocks=a.gate_blocks,
                  calibration_passes=a.calibration_passes)
