"""hash_decode.cuh's served launches on the host (pearl_c_sm120/hash_decode_host.cpp built with g++ for the variants
pearl_c_sm120.cu builds, its PD_VARIANT_LIST), as pearl_c_sm120/run.py's `Pipeline` launches them at decode when its caller
gives pd's counter (``pd_count``, as pouw_pearl_c_device does): ``commit_a`` (pd_commit: A's row digests, A's tree, seed_A
with line_a, each row's seed and line key, and E_A's lines) and ``hash_leaf`` (pd_tiles), at 48 real rows of 64 (A's pads and
E_A's filler lines) and at 64 of 64, in each served variant, every output poisoned (0xA5) before.  Each output is held to
Pearl-C -h3's Python (`pearl_c.row_digest`, A's `audit.HM96` tree under the unit's index, `pearl_c.row_seed` with
`pearl_c.line_key`, `pearl_c.unit_seed_a`, `pearl_c.sample_line`, the tile leaves `pearl_c.leaf`) and to the tile tree the
pipeline builds over the leaves under `audit.transcript_domain`; pd_commit's counter is left zero.  The no-write control, the
poison with no pd launch, must fail every comparison.  run.py's pd constants are hash_decode.cuh's."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

KERNEL = Path(__file__).resolve().parents[1] / "pearl_c_sm120"
_spec = importlib.util.spec_from_file_location("pearl_c_sm120_run", KERNEL / "run.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

from verity.ml.commitments.merkle import tree_levels  # noqa: E402
from verity.core.protocols.pouw import audit  # noqa: E402
from verity.core.protocols.pouw.audit import ROW_SCHEMA, TILE_SCHEMA, activation_binding, rows_domain  # noqa: E402
from verity.core.protocols.pouw.protocol import Matmul, Shape  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_c as C  # noqa: E402

#: pearl_c_sm120.cu's PD_VARIANT_LIST, the variants the cubin builds
SERVED_LIST = re.search(r"#define PD_VARIANT_LIST\(X\) (.*)", (KERNEL / "pearl_c_sm120.cu").read_text()).group(1).strip()
SERVED = tuple(int(x) for x in re.findall(r"X\((\d+)\)", SERVED_LIST))
SRC = (KERNEL / "hash_decode.cuh").read_text()


def const(name: str) -> int:
    return int(re.search(rf"\b{name} = (\d+)", SRC).group(1))


class Host:
    """run.py's `Dev` over the host build: buffers are numpy arrays, and a launch runs its grid in order (`pd_run`).  pd_setup's
    attribute reads get the RTX PRO 6000's opt-in shared memory and no static shared memory; each launch is logged."""

    dev, sms, OPTIN = 0, 188, 101376

    def __init__(self, lib):
        self.lib, self.bufs, self.log = lib, {}, []

    def alloc(self, n: int) -> int:
        a = np.zeros(max(n, 16), np.uint8)
        self.bufs[a.ctypes.data] = a
        return a.ctypes.data

    def put(self, data: bytes) -> int:
        p = self.alloc(len(data))
        self.bufs[p][:len(data)] = np.frombuffer(data, np.uint8)
        return p

    def array(self, p: int) -> np.ndarray:
        return self.bufs[p]

    def read(self, p: int, n: int) -> bytes:
        return self.bufs[p][:n].tobytes()

    def free(self, p: int) -> None:
        self.bufs.pop(p, None)

    def sync(self) -> None:
        pass

    def fn(self, name: str) -> str:
        return name

    def ck(self, f: str, *a) -> None:
        assert f == "cuFuncSetAttribute" and a[1] == R.MAX_DYN, (f, a)

    def get(self, f: str, *a) -> int:
        return {"cuDeviceGetAttribute": self.OPTIN, "cuFuncGetAttribute": 0}[f]

    def launch(self, name: str, grid, block: int, smem: int = 0, p=(), i=(), stream=None) -> None:
        self.log.append((name, grid, block, smem, tuple(i)))
        a = R.Args()
        for j, v in enumerate(p):
            a.p[j] = v
        for j, v in enumerate(i):
            a.i[j] = v
        assert isinstance(grid, int) and self.lib.pd_run(name.encode(), grid, block, ctypes.byref(a)) == 0, name


def build(directory: Path) -> ctypes.CDLL:
    """hash_decode_host.cpp built into `directory` for the served variants, with its `pd_run`."""
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++ for the host build of hash_decode.cuh")
    lib = directory / "hash_decode_host.so"
    subprocess.run([gxx, "-O2", "-std=c++20", "-pthread", "-shared", "-fPIC", "-Wno-unknown-pragmas",
                    f"-DPD_VARIANT_LIST(X)={SERVED_LIST}", "-o", str(lib), str(KERNEL / "hash_decode_host.cpp")], check=True)
    cdll = ctypes.CDLL(str(lib))
    cdll.pd_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    return cdll


@pytest.fixture(scope="module")
def host(tmp_path_factory) -> Host:
    return Host(build(tmp_path_factory.mktemp("hash-decode-host")))


def test_run_py_launches_the_cubins_variants_with_hash_decode_cuhs_constants():
    assert R.PD_SERVED == SERVED
    assert R.PD_PAIR_SMEM == 2 * 8 * const("PD_PAIR_WORDS") and R.PD_G == const("PD_G")
    assert R.PD_MAX_ROWS == const("PD_MAX_ROWS") and R.PD_MAX_K == 2 * 256 * (const("PD_KW_BLOCKS") - 1)
    assert (R.PD_WHOLE, R.PD_GROUP) == (const("PD_WHOLE"), const("PD_GROUP"))
    assert all(v & const("PD_PAIR") and v & R.PD_WHOLE and not v & const("PD_STAMP") for v in SERVED)
    assert R.pd_variant(48, 384, 1024) == 851 and R.pd_variant(64, 384, 1024) == 979 and R.pd_variant(32, 28672, 4096) == 979
    assert R.pd_variant(R.PD_MAX_ROWS + 1, 128, 1024) is None and R.pd_variant(32, 128, 2 * R.PD_MAX_K) is None


def levels(dom, values: list[bytes], schema, salt_key: bytes, nonce: int) -> bytes:
    """An `audit.HM96` tree's levels as the kernels lay them out, each value salted by `audit.hm96_salt`."""
    salted = [v + audit.hm96_salt(salt_key, nonce, i) for i, v in enumerate(values)]
    return b"".join(b"".join(lv) for lv in tree_levels(dom, [dom.leaf(i, i, schema, v) for i, v in enumerate(salted)]))


@pytest.mark.parametrize("m, mr, n", [(64, 48, 384), (64, 64, 384)])
def test_pd_writes_pearl_c_h3s_bytes_and_the_no_write_control_fails(host, m, mr, n):
    k, w, idx = 1024, 3, 5
    rng = np.random.default_rng(mr)
    a_rows = [rng.integers(0, 256, 4 * k, dtype=np.uint8).tobytes() for _ in range(mr)] + [bytes(4 * k)] * (m - mr)
    c, u = (rng.integers(0, 2**32, (m, n), dtype=np.uint64).astype(np.uint32) for _ in range(2))
    salt, salt_key, root_b = bytes(range(32)), bytes(range(100, 132)), hashlib.sha512(b"root_b").digest()
    tiles = R.cdiv(mr, R.TILE) * (n // R.TILE)
    dom_a = rows_domain(activation_binding(salt, idx), idx, mr, audit.HM96)
    epoch = audit.Epoch(audit.Workload((Matmul(idx, Shape(mr, k, n), w),)), {w: root_b}, salt)
    dom_t = audit.transcript_domain(epoch, tiles, audit.HM96)
    keys = {"salt_key": salt_key, "hm": audit.hm96_kernel_constants(), "unit": salt + root_b + idx.to_bytes(8, "little"),
            "dom_a": dom_a.domain_id, "dom_b": bytes(64), "dom_t": dom_t.domain_id, "line_b": bytes(32)}
    bufs = {name: host.alloc(size) for name, size in R.Pipeline.sizes(m, n, k, mr).items()}
    bufs["pd_count"] = host.alloc(16)
    for name, words in (("c", c), ("u", u)):
        host.array(bufs[name])[:m * n * 4] = np.frombuffer(words.astype("<u4").tobytes(), np.uint8)
    pipe = R.Pipeline(host, m, n, k, "sm120", {name: host.put(v) for name, v in keys.items()}, host.put(b"".join(a_rows)), 0,
                      bufs=bufs, m_real=mr)
    assert pipe.pd == R.pd_variant(mr, n, k) and host.pd_smem == dict.fromkeys(SERVED, host.OPTIN)
    by = {name: step for name, _, step in pipe.steps()}
    assert "commit_a" in by and not set(R.Pipeline.FUSED["commit_a"]) & set(by)

    digests = [C.row_digest(r) for r in a_rows[:mr]]
    tree_a = levels(dom_a, a_rows[:mr], ROW_SCHEMA, salt_key, idx)
    seeds = [C.row_seed(salt, d, root_b, idx, r, "h3") for r, d in enumerate(digests)]
    seed_a = C.unit_seed_a(salt, tree_a[-64:], root_b, idx, "h3")
    leaves = [C.leaf(C.Checked(c[r0:mr][:64, c0:c0 + 64].tolist(), u[r0:mr][:64, c0:c0 + 64].tolist()), "h3")
              for r0 in range(0, mr, 64) for c0 in range(0, n, 64)]
    want = {"rowdig_a": b"".join(digests), "tree_a": tree_a, "seed_a": bytes(seed_a), "line_a": C.line_key(seed_a),
            "seeds_a": b"".join(seeds), "lkeys_a": b"".join(C.line_key(s) for s in seeds),
            "ea_codes": b"".join(bytes(C.sample_line(C.S512Seed(seeds[t]), 0, 0, 0) if t < mr else
                                       C.sample_line(C.S512Seed(seed_a), 0, 0, t)) for t in range(m)),
            "leaves": b"".join(leaves), "tree_t": levels(dom_t, leaves, TILE_SCHEMA, salt_key, audit.HM96_TRANSCRIPT)}
    assert set(want) == {*R.PD_OUTPUTS, "tree_t"} and len(want["tree_a"]) == pipe.size["tree_a"]
    assert len(want["tree_t"]) == pipe.size["tree_t"] and len(want["ea_codes"]) == pipe.size["ea_codes"]

    for v in (None, *SERVED):
        for x in want:
            host.array(bufs[x])[:] = R.POISON
        host.log.clear()
        if v is not None:
            pipe.pd = v
            by["commit_a"]()
            by["hash_leaf"]()
        for step in ("pad_t", "tree_leaves_t", "tree_t"):
            if step in by:
                by[step]()
        got = {x: host.read(bufs[x], len(want[x])) == want[x] for x in want}
        if v is None:
            assert got == dict.fromkeys(want, False), "the no-write control passed"
            continue
        assert got == dict.fromkeys(want, True), (v, got)
        assert host.read(bufs["pd_count"], 16) == bytes(16)
        launches = {name: (grid, block, smem, i) for name, grid, block, smem, i in host.log}
        commit, tile = launches[f"pd_commit_{v}"], launches[f"pd_tiles_{v}"]
        assert commit == (mr, 32 * (R.cdiv(k // 256, 32) * 2 + 2), host.OPTIN, (mr, k, 0, m, 0))
        group = bool(v & R.PD_GROUP)
        assert tile == ((R.cdiv(mr, 64) * R.cdiv(n // 64, R.PD_G), 32 * R.PD_G, R.PD_PAIR_SMEM // 2) if group
                        else (tiles, 32, 0)) + ((mr, n),)


@pytest.mark.parametrize("m, mr", [(64, 48), (64, 64)])
def test_pd_under_h4_reads_a_bf16_halves_in_place_and_keys_the_padding_rows_on_the_pad_seed(host, m, mr):
    """Under h4 (`a_tap`) ``commit_a`` is pd_commit's mode 1 in narrow mode over A's FP32 words (each a BF16 widened) and
    ``tail_a`` is pd_tree: the row digests are the BF16 rows' (the Commit tap's row table, under `audit.tap_binding`'s
    domain), the tree, row seeds, seed_A and line_a are the format's over them, and the padding rows' E_A lines are
    `pearl_c.PAD_SEED`'s.  The no-write control fails every comparison."""
    k, n, idx = 1024, 384, 5
    rng = np.random.default_rng(mr + 1)
    bf = [rng.integers(0, 1 << 16, k, dtype=np.uint16) for _ in range(mr)] + [np.zeros(k, np.uint16)] * (m - mr)
    xa = b"".join((r.astype(np.uint32) << 16).astype("<u4").tobytes() for r in bf)
    rows = [r.astype("<u2").tobytes() for r in bf[:mr]]
    salt, salt_key, root_b = bytes(range(32)), bytes(range(100, 132)), hashlib.sha512(b"root_b").digest()
    dom_a = rows_domain(audit.tap_binding(salt, idx), idx, mr, audit.HM96)
    keys = {"salt_key": salt_key, "hm": audit.hm96_kernel_constants(), "unit": salt + root_b + idx.to_bytes(8, "little"),
            "dom_a": dom_a.domain_id, "dom_b": bytes(64), "dom_t": bytes(64), "line_b": bytes(32), "pad": C.line_key(C.PAD_SEED)}
    bufs = {name: host.alloc(size) for name, size in R.Pipeline.sizes(m, n, k, mr).items()}
    bufs["pd_count"] = host.alloc(16)
    pipe = R.Pipeline(host, m, n, k, "sm120", {name: host.put(v) for name, v in keys.items()}, host.put(xa), 0, bufs=bufs,
                      m_real=mr, a_tap=True)
    by = {name: step for name, _, step in pipe.steps()}
    assert pipe.FUSED is R.Pipeline.TAP_FUSED and {"commit_a", "tail_a"} <= set(by)
    assert not {x for parts in R.Pipeline.TAP_FUSED.values() for x in parts} & set(by)

    digests = [C.row_digest(r) for r in rows]
    tree_a = levels(dom_a, rows, ROW_SCHEMA, salt_key, idx)
    seeds = [C.row_seed(salt, d, root_b, idx, r, "h4") for r, d in enumerate(digests)]
    seed_a = C.unit_seed_a(salt, tree_a[-64:], root_b, idx, "h4")
    want = {"rowdig_a": b"".join(digests), "tree_a": tree_a, "seed_a": bytes(seed_a), "line_a": C.line_key(seed_a),
            "seeds_a": b"".join(seeds), "lkeys_a": b"".join(C.line_key(s) for s in seeds),
            "ea_codes": b"".join(bytes(C.sample_line(C.S512Seed(seeds[t]), 0, 0, 0) if t < mr else
                                       C.sample_line(C.PAD_SEED, 0, 0, t)) for t in range(m))}
    for v in (None, *SERVED):
        for x in want:
            host.array(bufs[x])[:] = R.POISON
        host.log.clear()
        if v is not None:
            pipe.pd = v
            by["commit_a"]()
            by["tail_a"]()
        got = {x: host.read(bufs[x], len(want[x])) == want[x] for x in want}
        if v is None:
            assert got == dict.fromkeys(want, False), "the no-write control passed"
            continue
        assert got == dict.fromkeys(want, True), (v, got)
        assert host.read(bufs["pd_count"], 16) == bytes(16)
        launches = {name: (grid, block, smem, i) for name, grid, block, smem, i in host.log}
        assert launches[f"pd_commit_{v}"] == (mr, 32 * (R.cdiv(k // 512, 32) * 2 + 2), host.OPTIN, (mr, k // 2, 1, m, 1))
        assert launches[f"pd_tree_{v}"] == (1, 128, host.OPTIN, (mr, 0, 0, m))
