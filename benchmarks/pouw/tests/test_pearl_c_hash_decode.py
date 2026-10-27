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
import math
import re
import shutil
import subprocess
import sys
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
        self.lib, self.bufs, self.log, self.args = lib, {}, [], {}

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

    def at(self, p: int, n: int) -> np.ndarray:
        """The n bytes at address p, inside whichever buffer holds them."""
        base = next(b for b, a in self.bufs.items() if b <= p and p + n <= b + len(a))
        return self.bufs[base][p - base:p - base + n]

    def launch(self, name: str, grid, block: int, smem: int = 0, p=(), i=(), stream=None) -> None:
        self.log.append((name, grid, block, smem, tuple(i)))
        self.args[name] = (grid, block, smem, tuple(p), tuple(i))
        if name == "lines":
            self.lines(p, i)
            return
        a = R.Args()
        for j, v in enumerate(p):
            a.p[j] = v
        for j, v in enumerate(i):
            a.i[j] = v
        assert isinstance(grid, int) and self.lib.pd_run(name.encode(), grid, block, ctypes.byref(a)) == 0, name

    def lines(self, p, i) -> None:
        """pearl_c_sm120.cu's `lines` at E_A, which the host build lacks: line t < i4 is line 0 under its own key (p4 + 32 t),
        the rest line t under p0's, each drawn by `C.sample_line`'s -h3 rule from the key."""
        key, codes, t32, t128, own = (*p, 0, 0, 0)[:5]
        side, factor, count, norm, rows = i
        assert (side, factor, norm, t32, t128) == (0, 0, R.NORM_256, 0, 0)
        for t in range(count):
            kw = self.at(own + 32 * t if t < rows else key, 32).tobytes()
            raw = C.s512("line", kw + bytes([side, factor]) + (0 if t < rows else t).to_bytes(4, "little"))
            xs = [(-1 if b >> 7 else 1) * ((b & 0x7F) + 1) for b in raw]
            scale = C.P.bf16_div(C.P.f32_to_bf16(C.P.NOISE_NORM * 32), C.P.f32_to_bf16(float(math.isqrt(sum(x * x for x in xs) * 1024))))
            self.at(codes + 32 * t, 32)[:] = [C.P.f32_to_fp8(C.P.bf16_to_f32(C.P.bf16_mul(C.P.f32_to_bf16(float(x)), scale))) for x in xs]


def build(directory: Path, *defines: str) -> ctypes.CDLL:
    """hash_decode_host.cpp built into `directory` for the served variants, with its `pd_run`, `defines` defined."""
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++ for the host build of hash_decode.cuh")
    lib = directory / "hash_decode_host.so"
    subprocess.run([gxx, "-O2", "-std=c++20", "-pthread", "-shared", "-fPIC", "-Wno-unknown-pragmas",
                    f"-DPD_VARIANT_LIST(X)={SERVED_LIST}", *(f"-D{d}" for d in defines), "-o", str(lib),
                    str(KERNEL / "hash_decode_host.cpp")], check=True)
    cdll = ctypes.CDLL(str(lib))
    cdll.pd_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    return cdll


@pytest.fixture(scope="module", params=["sm120", "sm90"])
def host(request, tmp_path_factory) -> Host:
    """The sm_120 build, and the build with pearl_c_sm90.cu's PD_SM90 (hash_decode.cuh's PD_SM90_FORMS), the same bytes."""
    h = Host(build(tmp_path_factory.mktemp(f"hash-decode-host-{request.param}"), *(["PD_SM90"] * (request.param == "sm90"))))
    h.sm90 = request.param == "sm90"
    return h


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


@pytest.mark.parametrize("form", ["whole", "split", "split-nofill"])
@pytest.mark.parametrize("m, mr, n", [(64, 48, 384), (64, 64, 384)])
def test_pd_writes_pearl_c_h3s_bytes_and_the_no_write_control_fails(host, m, mr, n, form):
    """Each form of A's commitment: ``commit_a`` (pd_commit in mode 0), or `split_a`'s ``rows_a`` (pd_commit in mode 1)
    then ``tree_seed_a`` (pd_tree, the same variant), which writes the same bytes; without `fillers` pd_tree writes no
    padding rows' E_A lines, which keep the poison, and every other byte is the same."""
    k, w, idx = 1024, 3, 5
    split, fillers = form != "whole", form != "split-nofill"
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
                      bufs=bufs, m_real=mr, split_a=split, fillers=fillers)
    commits = {v & ~R.PD_GROUP for v in SERVED}
    assert pipe.pd == R.pd_variant(mr, n, k) and host.pd_smem == host.pd_tree_smem == dict.fromkeys(commits, host.OPTIN)
    by = {name: step for name, _, step in pipe.steps()}
    names = ("rows_a", "tree_seed_a") if split else ("commit_a",)
    assert set(names) <= set(by) and not {"commit_a", "rows_a", "tree_seed_a"} - set(names) & set(by)
    assert not {x for f in names for x in R.Pipeline.FUSED[f]} & set(by)

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
            for f in names:
                by[f]()
            by["hash_leaf"]()
        for step in ("pad_t", "tree_leaves_t", "tree_t"):
            if step in by:
                by[step]()
        got = {x: host.read(bufs[x], len(want[x])) == want[x] for x in want}
        if v is None:
            assert got == dict.fromkeys(want, False), "the no-write control passed"
            continue
        ea = host.read(bufs["ea_codes"], len(want["ea_codes"]))
        if not fillers:
            got["ea_codes"] = ea[:32 * mr] == want["ea_codes"][:32 * mr] and set(ea[32 * mr:]) <= {R.POISON}
        assert got == dict.fromkeys(want, True), (v, got)
        assert host.read(bufs["pd_count"], 16) == bytes(16)
        launches = {name: (grid, block, smem, i) for name, grid, block, smem, i in host.log}
        commit, tile = launches[f"pd_commit_{v & ~R.PD_GROUP}"], launches[f"pd_tiles_{v}"]
        assert commit == (mr, 32 * (R.cdiv(k // 256, 32) * 2 + 2), host.OPTIN, (mr, k, int(split), m))
        if split:
            assert launches[f"pd_tree_{v & ~R.PD_GROUP}"] == (1, 128, host.OPTIN, (mr, 0, 0, m))
            assert [x[0] for x in host.log][:2] == [f"pd_commit_{v & ~R.PD_GROUP}", f"pd_tree_{v & ~R.PD_GROUP}"]
        group = bool(v & R.PD_GROUP)
        assert tile == ((R.cdiv(mr, 64) * R.cdiv(n // 64, R.PD_G), 32 * R.PD_G, R.PD_PAIR_SMEM // 2) if group
                        else (tiles, 32, 0)) + ((mr, n),)


@pytest.mark.parametrize("form", ["serial", "split", "split-nofill"])
@pytest.mark.parametrize("m, mr", [(64, 48), (64, 64)])
def test_a_split_commitment_outside_pd_writes_the_unsplit_steps_bytes(host, m, mr, form):
    """Outside pd (no counter: the generic kernels, as at a prefill's rows), A's commitment unsplit (A's pads, row digests,
    level 0, levels, seed_A, the row seeds, then the forming's E_A lines) or `split_a`'s two steps over the same kernels
    (`run.Pipeline.FUSED_GENERIC`): ``rows_a``, the pads, row digests, row seeds and the real rows' E_A lines, reading no
    line_a, then ``tree_seed_a``, A's level 0, levels and seed_A with line_a, and with `fillers` every row's E_A line again.
    Each output holds Pearl-C -h3's bytes; without `fillers` the padding rows' E_A lines keep the poison."""
    n, k, idx = 384, 1024, 5
    split, fillers = form != "serial", form != "split-nofill"
    rng = np.random.default_rng(100 + mr)
    a_rows = [rng.integers(0, 256, 4 * k, dtype=np.uint8).tobytes() for _ in range(mr)] + [bytes(4 * k)] * (m - mr)
    salt, salt_key, root_b = bytes(range(32)), bytes(range(100, 132)), hashlib.sha512(b"root_b").digest()
    dom_a = rows_domain(activation_binding(salt, idx), idx, mr, audit.HM96)
    keys = {"salt_key": salt_key, "hm": audit.hm96_kernel_constants(), "unit": salt + root_b + idx.to_bytes(8, "little"),
            "dom_a": dom_a.domain_id, "dom_b": bytes(64), "dom_t": bytes(64), "line_b": bytes(32)}
    bufs = {name: host.alloc(size) for name, size in R.Pipeline.sizes(m, n, k, mr).items()}
    pipe = R.Pipeline(host, m, n, k, "sm120", {name: host.put(v) for name, v in keys.items()}, host.put(b"".join(a_rows)), 0,
                      bufs=bufs, m_real=mr, split_a=split, fillers=fillers)
    assert pipe.pd is None
    by = {name: (phase, step) for name, phase, step in pipe.steps()}
    if split:
        assert pipe.FUSED == {**R.Pipeline.FUSED, **R.Pipeline.FUSED_GENERIC} and R.Pipeline.FUSED["rows_a"] != pipe.FUSED["rows_a"]
        assert [x for x in by if by[x][0] == "commit"] == ["rows_a", "tree_seed_a"]
        assert not {x for f in ("rows_a", "tree_seed_a") for x in pipe.FUSED[f]} & set(by)
        names = ("rows_a", "tree_seed_a")
    else:
        assert "FUSED" not in vars(pipe)
        names = ("pad_a",) * (mr < m) + ("rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a", "lines_ea")
        assert {x for x in names if by[x][0] == "commit"} == set(names) - {"pad_a", "lines_ea"} and by["lines_ea"][0] == "forming"

    digests = [C.row_digest(r) for r in a_rows[:mr]]
    tree_a = levels(dom_a, a_rows[:mr], ROW_SCHEMA, salt_key, idx)
    seeds = [C.row_seed(salt, d, root_b, idx, r, "h3") for r, d in enumerate(digests)]
    seed_a = C.unit_seed_a(salt, tree_a[-64:], root_b, idx, "h3")
    want = {"rowdig_a": b"".join(digests), "tree_a": tree_a, "seed_a": bytes(seed_a), "line_a": C.line_key(seed_a),
            "seeds_a": b"".join(seeds), "lkeys_a": b"".join(C.line_key(s) for s in seeds),
            "ea_codes": b"".join(bytes(C.sample_line(C.S512Seed(seeds[t]), 0, 0, 0) if t < mr else
                                       C.sample_line(C.S512Seed(seed_a), 0, 0, t)) for t in range(m))}
    for x in want:
        host.array(bufs[x])[:] = R.POISON
    host.log.clear()
    for x in names:
        by[x][1]()
    got = {x: host.read(bufs[x], len(want[x])) == want[x] for x in want}
    if not fillers:
        ea = host.read(bufs["ea_codes"], len(want["ea_codes"]))
        got["ea_codes"] = ea[:32 * mr] == want["ea_codes"][:32 * mr] and set(ea[32 * mr:]) <= {R.POISON}
    assert got == dict.fromkeys(want, True), got
    lines = [i for name, _, _, _, i in host.log if name == "lines"]
    if split:
        rows_a = [x[0] for x in host.log].index("hm_leaves")
        assert lines[0] == (0, 0, mr, R.NORM_256, mr) and ("lines", R.cdiv(mr, 128), 128, 0, lines[0]) in host.log[:rows_a]
        assert lines[1:] == [(0, 0, m, R.NORM_256, mr)] * (fillers and mr < m)
    else:
        assert lines == [(0, 0, m, R.NORM_256, mr)]


def test_pd_commits_zero_form_writes_its_bytes_without_reading_the_zero_rows(host):
    """pd_commit_t's Z form (pd_commit_zero_V, as pearl_c_sm90.cu's pd_commit_g takes it) at 64 rows whose last 16 are zero,
    given `grouped.zero_state` at p14 and the first zero row at i4, writes pd_commit_V's every byte in each served variant,
    and still does with those rows nonzero: it takes the zero row's digest (`pearl_c.row_digest`) for them unread."""
    if "pearl_c_sm90_grouped" not in sys.modules:
        spec = importlib.util.spec_from_file_location("pearl_c_sm90_grouped", KERNEL.parent / "pearl_c_sm90" / "grouped.py")
        sys.modules[spec.name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[spec.name])
    G = sys.modules["pearl_c_sm90_grouped"]
    m, mr, n, k, idx = 64, 48, 384, 1024, 5
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, (m, 4 * k), dtype=np.uint8)
    a[mr:] = 0
    salt, salt_key, root_b = bytes(range(32)), bytes(range(100, 132)), hashlib.sha512(b"root_b").digest()
    dom_a = rows_domain(activation_binding(salt, idx), idx, m, audit.HM96)
    keys = {"salt_key": salt_key, "hm": audit.hm96_kernel_constants(), "unit": salt + root_b + idx.to_bytes(8, "little"),
            "dom_a": dom_a.domain_id, "dom_b": bytes(64), "dom_t": bytes(64), "line_b": bytes(32)}
    bufs = {name: host.alloc(size) for name, size in R.Pipeline.sizes(m, n, k, m).items()}
    bufs["pd_count"] = host.alloc(16)
    rows = host.put(a.tobytes())
    pipe = R.Pipeline(host, m, n, k, "sm120", {name: host.put(v) for name, v in keys.items()}, rows, 0, bufs=bufs, m_real=m)
    by = {name: step for name, _, step in pipe.steps()}
    out = [x for x in R.PD_OUTPUTS if x != "leaves"]
    zero = host.put(G.zero_state(k))

    def commit(name, *z):
        for x in out:
            host.array(bufs[x])[:] = R.POISON
        if z:
            grid, block, smem, p, i = host.args[f"pd_commit_{pipe.pd & ~R.PD_GROUP}"]
            assert len(p) <= 14 and len(i) == 4
            host.launch(name, grid, block, smem, p=(*p, *[0] * (14 - len(p)), z[0]), i=(*i, z[1]))
        else:
            by["commit_a"]()
        assert host.read(bufs["pd_count"], 16) == bytes(16)
        return {x: host.read(bufs[x], pipe.size[x]) for x in out}

    for v in SERVED:
        pipe.pd = v
        want = commit(f"pd_commit_{v}")
        assert want["rowdig_a"][64 * mr:] == C.row_digest(bytes(4 * k)) * (m - mr)
        assert commit(f"pd_commit_zero_{v}", zero, mr) == want and commit(f"pd_commit_zero_{v}", 0, mr) == want, v
        host.array(rows)[4 * k * mr:4 * k * m] = 0x5A
        assert commit(f"pd_commit_zero_{v}", zero, mr) == want, v
        assert commit(f"pd_commit_zero_{v}", zero, m)["rowdig_a"] != want["rowdig_a"]
        host.array(rows)[4 * k * mr:4 * k * m] = 0


def test_pd_commit_reads_a_calls_bf16_rows_in_place_of_their_fp32_words(host):
    """PD_SM90_FORMS's pd_commit in mode 1 (``rows_a``) given a call's BF16 rows at p15, i5 of them (its rows past zero),
    writes every byte it writes from their FP32 widening at p0, with p0 then holding other words, which it doesn't read; the
    sm_120 build reads p0 whatever p15 holds.  The BF16 buffer holds the i5 rows alone."""
    m, mr, n, k, idx = 64, 48, 384, 1024, 5
    rng = np.random.default_rng(7)
    halves = rng.integers(0, 2**16, (mr, k), dtype=np.uint16)
    words = np.zeros((m, k), np.uint32)
    words[:mr] = halves.astype(np.uint32) << 16
    salt, salt_key, root_b = bytes(range(32)), bytes(range(100, 132)), hashlib.sha512(b"root_b").digest()
    dom_a = rows_domain(activation_binding(salt, idx), idx, m, audit.HM96)
    keys = {"salt_key": salt_key, "hm": audit.hm96_kernel_constants(), "unit": salt + root_b + idx.to_bytes(8, "little"),
            "dom_a": dom_a.domain_id, "dom_b": bytes(64), "dom_t": bytes(64), "line_b": bytes(32)}
    bufs = {name: host.alloc(size) for name, size in R.Pipeline.sizes(m, n, k, m).items()}
    bufs["pd_count"] = host.alloc(16)
    rows, other = host.put(words.astype("<u4").tobytes()), host.put(bytes([0x5A]) * (4 * k * m))
    b16 = host.put(halves.astype("<u2").tobytes())
    pipe = R.Pipeline(host, m, n, k, "sm120", {name: host.put(v) for name, v in keys.items()}, rows, 0, bufs=bufs, m_real=m,
                      split_a=True)
    by = {name: step for name, _, step in pipe.steps()}
    out = [x for x in R.PD_OUTPUTS if x != "leaves"]

    def commit(*bf16):
        for x in out:
            host.array(bufs[x])[:] = R.POISON
        if bf16:
            name = next(x[0] for x in reversed(host.log) if x[0].startswith("pd_commit_"))
            grid, block, smem, p, i = host.args[name]
            assert len(p) == 14 and len(i) == 4 and i[2] == 1
            host.launch(name, grid, block, smem, p=(other, *p[1:], 0, bf16[0]), i=(*i, 0, bf16[1]))
        else:
            by["rows_a"]()
        assert host.read(bufs["pd_count"], 16) == bytes(16)
        return {x: host.read(bufs[x], pipe.size[x]) for x in out}

    for v in SERVED:
        pipe.pd = v
        want = commit()
        assert want["rowdig_a"][:64 * mr] == b"".join(C.row_digest(r.tobytes()) for r in words[:mr].astype("<u4"))
        assert want["rowdig_a"][64 * mr:] == C.row_digest(bytes(4 * k)) * (m - mr)
        got = commit(b16, mr)
        assert (got == want) == host.sm90, v
