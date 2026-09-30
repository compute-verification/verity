"""pouw_hash/h2_bench.py's gate on a host stand-in for the device: buffers laid out as pearl_c/fixture.py lays out the kernels'
(tree_bytes, digests, leaves: run.py's check holds the card to them), under -h1 and -h2, pass; a byte flipped in any hashed
buffer fails the check that covers it."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verity.commitments.merkle import b3s_level_key, blake3_node_key
from verity_pouw.schemes import pearl_c as C
from verity_pouw.serving import HASH_FORMATS

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load("h2_bench", ROOT / "pouw_hash" / "h2_bench.py")
H = _load("pearl_c_h100_fixture", ROOT / "pearl_c" / "fixture.py")
R = B.R


class HostDev:
    """The Dev calls the gate makes (read, copy, sync) over host buffers at stand-in addresses."""

    def __init__(self):
        self.mem: dict[int, bytearray] = {}
        self.top = 1 << 20

    def put(self, data: bytes) -> int:
        p, self.top = self.top, self.top + ((len(data) + 4095) // 4096 + 1) * 4096
        self.mem[p] = bytearray(data)
        return p

    def at(self, p: int, n: int) -> tuple[bytearray, int]:
        for base, buf in self.mem.items():
            if base <= p and p + n <= base + len(buf):
                return buf, p - base
        raise AssertionError(f"read outside any buffer: {p:#x} + {n}")

    def read(self, p: int, n: int) -> bytes:
        buf, o = self.at(p, n)
        return bytes(buf[o:o + n])

    def copy(self, dst: int, src: int, n: int) -> None:
        buf, o = self.at(dst, n)
        buf[o:o + n] = self.read(src, n)

    def sync(self) -> None:
        pass


def pipeline(hashing: str, m: int = 32, n: int = 256, k: int = 256):
    rng = np.random.default_rng(7)
    dev, rb, tiles = HostDev(), 4 * k, R.cdiv(m, R.TILE) * (n // R.TILE)
    doms = B.domains(hashing, m, n, tiles)
    xa, xb = rng.integers(0, 2**32, (m, k), dtype=np.uint32), rng.integers(0, 2**32, (n, k), dtype=np.uint32)
    c, u = rng.integers(0, 2**32, (m, n), dtype=np.uint32), rng.integers(0, 2**32, (m, n), dtype=np.uint32)
    ra, rbs = [r.astype("<u4").tobytes() for r in xa], [r.astype("<u4").tobytes() for r in xb]
    dg = H.digests(c, u)
    lv = H.leaves(dg, m, n)
    bufs = {"xa": b"".join(ra), "xb": b"".join(rbs), "c": c.astype("<u4").tobytes(), "u": u.astype("<u4").tobytes(),
            "digests": b"".join(dg), "digests2": bytes(32 * len(dg)), "leaves": b"".join(lv),
            "tree_a": H.tree_bytes(doms["a"], ra, R.ROW_SCHEMA), "tree_b": H.tree_bytes(doms["b"], rbs, R.ROW_SCHEMA),
            "tree_t": H.tree_bytes(doms["t"], lv, R.TILE_SCHEMA)}
    for t, count in (("a", m), ("b", n), ("t", tiles)):
        bufs[f"keys_{t}"] = b"".join(blake3_node_key(doms[t].domain_id, lv_) for lv_ in range(max(R.depth(count), 1)))
    for t in ("a", "b"):
        bufs[f"skeys_{t}"] = b"".join(b3s_level_key(doms[t].domain_id, lv_) for lv_ in range(R.b3s_levels(rb)))
    b = {nm: dev.put(v) for nm, v in bufs.items()}
    size = {nm: len(v) for nm, v in bufs.items()}

    def hash_msg():
        buf, o = dev.at(b["digests"], size["digests"])
        buf[o:o + size["digests"]] = b"".join(H.digests(np.frombuffer(dev.read(b["c"], size["c"]), "<u4").reshape(m, n),
                                                        np.frombuffer(dev.read(b["u"], size["u"]), "<u4").reshape(m, n)))

    def diff(x, y):
        gx, gy = dev.read(b[x], size[x]), dev.read(b[y], size[y])
        return sum(gx[i:i + 4] != gy[i:i + 4] for i in range(0, size[x], 4))

    pipe = SimpleNamespace(b=b, size=size, m_real=m, n=n, row_bytes=rb, tiles=tiles, hashing=hashing, diff=diff,
                           extra=lambda: {"hash_msg": hash_msg}, root=lambda t: b[f"tree_{t}"] + size[f"tree_{t}"] - 32)
    return dev, pipe, doms


def test_tables_are_the_schemes():
    assert B.DIGEST_KEYS == {"k_msg": C.H1_KEYS["message"], "k_leaf": C.H1_KEYS["leaf"]}
    assert B.TREE_HASH == {h: HASH_FORMATS[h].tree_hash for h in ("h1", "h2")}


@pytest.mark.parametrize("hashing", ["h1", "h2"])
def test_gate_accepts_the_kernels_layout(hashing):
    dev, pipe, doms = pipeline(hashing)
    t = B.gate(dev, pipe, doms, random.Random(0))
    assert not t.failed(), {nm: c for nm, c in t.items() if c["failed"]}
    want = {"node_keys_a", "leaves_a", "nodes_a", "root_a", "node_keys_b", "leaves_b", "nodes_b", "root_b", "digests",
            "tile_leaves", "node_keys_t", "leaves_t", "nodes_t", "root_t", "digests_vs_hash_msg"}
    assert set(t) == want | ({"skeys_a", "skeys_b"} if hashing == "h2" else set())


@pytest.mark.parametrize("hashing", ["h1", "h2"])
@pytest.mark.parametrize("buf,offset,check", [
    ("tree_a", 32 * 31 + 5, "leaves_a"), ("tree_a", 32 * (32 + 3), "nodes_a"), ("tree_b", 32 * 511 - 1, "nodes_b"),
    ("keys_t", 40, "node_keys_t"), ("skeys_b", 33, "skeys_b"), ("digests", 32 * 255, "digests"),
    ("leaves", 3, "tile_leaves"), ("tree_t", 32 * 2 + 1, "leaves_t")])
def test_gate_rejects_a_flip(hashing, buf, offset, check):
    if check.startswith("skeys") and hashing == "h1":
        pytest.skip("-h1 has no segment keys")
    dev, pipe, doms = pipeline(hashing)
    mem, o = dev.at(pipe.b[buf] + offset, 1)
    mem[o] ^= 0x01
    t = B.gate(dev, pipe, doms, random.Random(0))
    assert check in t.failed(), t.failed()
