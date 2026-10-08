"""The vLLM device path for Pearl-C on sm_120 (`verity_vllm.protocol_options.pouw_pearl_c_device`) against GPU 1's pipeline
(``benchmarks/pouw/pearl_c_sm120``) on the CPU, with a recording stand-in for the device; and the pass verifier
(``benchmarks/pouw/pearl_c_vllm/verify_run.py``) on an honest transcript that the scheme's reference prover builds."""
from __future__ import annotations

import ctypes
import importlib.util
import json
import math
import random
import shutil
import sys
import types
from fractions import Fraction
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.ml.commitments import MerkleTree
from verity.ml.commitments.merkle import tree_levels
from verity.core.protocols.pouw.audit import (
    HM96,
    HM96_TRANSCRIPT,
    HM96_WEIGHTS,
    ROW_SCHEMA,
    TILE_SCHEMA,
    Epoch,
    Prover,
    Verifier,
    Workload,
    activation_binding,
    commit_rows,
    hm96_kernel_constants,
    hm96_salt,
    opened_value,
    rows_domain,
    transcript_domain,
    weight_binding,
    weights_root,
)
from verity.core.protocols.pouw.coins import Challenger, CoinCommitment
from verity.core.protocols.pouw.protocol import Context, DomainError, Matmul, Shape
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes import pearl_c_work as PW
from verity.core.protocols.pouw.schemes import pearl_kw as P
from verity_vllm.protocol_options import pouw_pearl_c_device as PPD

KERNEL = Path(__file__).resolve().parents[1] / "pearl_c_sm120"
SCHEME = "pearl-c-sm120-v1-h3"
SEED = bytes(range(32))
SALT_KEY = bytes(range(100, 132))
RUN = "r20260930-120000-test"


def salt_coin(roots: dict[int, bytes], epoch: int = 0, entropy: int = 7):
    """An epoch's coin commitment and its salt coin, opened after `roots`' weights root, from a seeded challenger."""
    ch = Challenger(epoch, random.Random(entropy).randbytes)
    return ch.commitment, ch.open("salt", weights_root(roots))


def published(path: Path) -> CoinCommitment:
    """The epoch's coin commitment as its challenger published it beside the passes (e2e.py's ``coins.json``)."""
    return CoinCommitment.from_json(json.loads((path.parent / "coins.json").read_text()))


def coins_args(path: Path) -> list[str]:
    return ["--coins", str(path.parent / "coins.json")]


def load(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


R = load("pearl_c_sm120_run", KERNEL / "run.py")
F = load("pearl_c_sm120_fixture", KERNEL / "fixture.py")
VR = load("pearl_c_vllm_verify_run", KERNEL.parent / "pearl_c_vllm" / "verify_run.py")
SD = load("pearl_c_vllm_served_debit", KERNEL.parent / "pearl_c_vllm" / "served_debit.py")


def bf16_words(rng, rows: int, k: int, scale: float = 1.0) -> np.ndarray:
    """FP32 words of BF16 values (the only rows the vLLM path commits: BF16 widened)."""
    return (rng.standard_normal((rows, k)).astype(np.float32) * scale).view(np.uint32) & 0xFFFF0000


def stats(words: np.ndarray) -> tuple:
    """`stats_s5`'s α and (s, ρ) per row as the kernel writes them, from the scheme's own `s5_stats`."""
    st = [C.s5_stats([int(v) for v in row]) for row in words]
    alpha = torch.from_numpy(np.array([s[2] for s in st], np.uint32).view(np.uint8).copy())
    aux = torch.from_numpy(np.array([(s[0], s[1]) for s in st], np.uint32).view(np.uint8).copy())
    return alpha, aux


# ---- the records' rules -------------------------------------------------------------------------------------------

def test_screen_decides_the_per_row_rules_as_row_passes():
    rng, k = np.random.default_rng(7), 1024
    words = bf16_words(rng, 12, k)
    words[1] = 0                                                   # the noise floor: ρ·α = 0
    for r, count in ((2, 17), (3, 16), (4, 40)):                   # liveness: 64 · dead > k fails, so 17 of 1,024 dead do
        pos = [p for p in range(k) if p % 8][:count]
        words[r, pos] = np.float32(1e4).view(np.uint32) & 0xFFFF0000
    words[5, 3] = np.float32(200.0).view(np.uint32)                # one large entry: s over 444·ρ fails the floor only
    x = torch.from_numpy(words.view(np.float32).copy())
    alpha, aux = stats(words)
    fails, unsure = PPD.screen(x, alpha, aux)
    got = PPD.excluded(fails, unsure, x)
    want = tuple(i for i, row in enumerate(words) if not PW.row_passes([int(v) for v in row]))
    assert got == want and 1 in got and 2 in got and 4 in got and 0 not in got


def test_an_unsure_row_is_decided_exactly():
    """A row whose dead count sits on the bound, with entries inside the screen's band, goes to `live_row`."""
    k = 1024
    base = np.ones(k, np.float32)
    thr = 64.0 * float((base[::8] ** 2).sum()) / len(range(0, k, 8))
    row = base.copy()
    pos = [p for p in range(k) if p % 8][:17]
    row[pos[:16]] = np.float32(np.sqrt(thr) * 2)
    row[pos[16]] = np.float32(np.sqrt(thr))                        # exactly at the threshold: dead, and inside the band
    words = (row.view(np.uint32) & 0xFFFF0000)[None]
    x = torch.from_numpy(words.view(np.float32).copy())
    fails, unsure = PPD.screen(x, *stats(words))
    assert bool(unsure[0]) and not bool(fails[0])
    assert PPD.excluded(fails, unsure, x) == (() if PW.row_passes([int(v) for v in words[0]]) else (0,))


# ---- the per-call keys --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", sorted(PPD.SCHEMES))
def test_call_keys_are_the_fixtures_keys_of(scheme):
    s = PPD.scheme_of(scheme)
    roots = {0: bytes([1]) * 32, 1: bytes([2]) * 32}
    work = Workload((Matmul(0, Shape(16, 1024, 256), 0), Matmul(1, Shape(16, 1024, 128), 1)))
    epoch = Epoch.unverified(work, roots, SEED)
    for index, w, m16 in ((0, 0, 32), (5, 1, 8192), (127, 0, 16)):
        mm = Matmul(index, Shape(m16, 1024, 256 if w == 0 else 128), w)
        kk = F.keys_of(s, epoch, mm, C._labelled(epoch.salt + roots[w], "seed-B"))
        assert PPD.call_keys(F, epoch.salt, roots[w], index, m16, s.commitment_hash) == kk["dom_a"] + kk["unit"]


# ---- the run on a recording stand-in for the device ---------------------------------------------------------------

class RecordingDev:
    """What `run.Pipeline` and the run touch of `run.Dev`, recording each launch (name, stream, pointers, ints)."""

    sms = 188

    def __init__(self):
        self.stream, self.log, self.n = None, [], 0

    def launch(self, name, grid, block, smem=0, p=(), i=(), stream=None):
        self.log.append((name, self.stream if stream is None else stream, tuple(p), tuple(i)))

    def ck(self, *_):
        pass

    def fn(self, name):
        return name

    def alloc(self, n):
        self.n += 1
        return self.n

    def free(self, p):
        pass

    def sync(self):
        pass

    def read(self, p, n):
        return bytes(n)

    def tmap(self, ptr, depth, rows, box_rows=128):
        return bytes(128)


@pytest.fixture
def device_run():
    rng = np.random.default_rng(3)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    kernel = PPD.Kernel(R, F, RecordingDev(), "0" * 64)
    roots = PPD.weight_roots(kernel, weights, SCHEME, salt_key=SALT_KEY)
    declared = Workload(tuple(Matmul(w, Shape(16, 1024, t.shape[0]), w) for w, t in weights.items()))
    coins = salt_coin(roots)
    salt = Epoch.start(declared, roots, *coins).salt

    def make(hashing=True, schedule="serial", flag_rows=PPD.FLAG_ROWS):
        return PPD.PearlCDeviceRun(PPD.scheme_of(SCHEME), f"pouw:{SCHEME}", kernel, weights, roots, *coins, hashing=hashing,
                                   capacity=1 << 12, schedule=schedule, flag_rows=flag_rows, salt_key=SALT_KEY)
    return make, kernel, weights, roots, salt


PER_CALL = ["s512_rows", "hm_leaves", "s512_frame_leaves", "s512_tree_small", "s512_seeds", "s512_seeds", "lines", "stats_s5",
            "eprime", "form_s5", "words", "pa", "gemm_pearlc_d", "s512_tile_leaves"]


def test_each_call_runs_the_pipelines_call_steps_at_its_offset_and_keys(device_run):
    make, kernel, weights, roots, salt = device_run
    run = make()
    dev = kernel.dev
    for w, m in ((0, 20), (1, 32), (0, 32)):
        dev.log.clear()
        x = torch.randn(m, 1024).to(torch.bfloat16)
        before = run.cur.tiles
        y = run.gemm(w, x, weights[w], None)
        assert y.shape == (m, weights[w].shape[0]) and y.dtype == torch.bfloat16
        names = [n for n, *_ in dev.log if not n.endswith("_smem")]
        assert names == PER_CALL, names
        by = {n: (p, i) for n, _, p, i in dev.log}
        assert by["s512_tile_leaves"][0][2] == run.leaves.data_ptr() + 64 * before
        assert by["gemm_pearlc_d"][0][4] == y.data_ptr()
        sh = run.shapes[(32, weights[w].shape[0], 1024)]
        index = len(run.cur.matmuls) - 1
        assert PPD._bytes(sh.keys[:168]) == PPD.call_keys(F, salt, roots[w], index, 32, run.tree_hash)
        assert bool((sh.xa[m:] == 0).all()) and torch.equal(sh.xa[:m], x.float())
    assert [u.index for u in run.cur.matmuls] == [0, 1, 2] and run.cur.tiles == 4 + 2 + 4
    dev.log.clear()
    out = run.end()
    assert [n for n, *_ in dev.log][-4:] == ["s512_pads", "hm_leaves", "s512_frame_leaves", "s512_tree_small"]
    assert dev.log[-2][3] == (0, 10)
    assert out["units"] == 3 and out["tiles"] == 10 and len(out["transcript_root"]) == 128
    assert run.commits["transcript_root"] == out["transcript_root"]


def test_a_call_over_whole_128_row_tiles_runs_the_128_tile_chain(device_run):
    """At m a multiple of 128 the chain runs on 128 x 128 tiles (gemm_pearlc, on the mainloop), writing C~, U and y, and the
    tile leaves read them."""
    make, kernel, weights, *_ = device_run
    run = make()
    kernel.dev.log.clear()
    y = run.gemm(0, torch.randn(128, 1024).to(torch.bfloat16), weights[0], None)
    names = [n for n, *_ in kernel.dev.log if not n.endswith("_smem")]
    assert run.pipes[(0, 128, frozenset(), True, "serial", False)][0].bm == 128
    by = {n: p for n, _, p, _ in kernel.dev.log}
    assert names[-2:] == ["gemm_pearlc", "s512_tile_leaves"] and by["gemm_pearlc"][4] == y.data_ptr()
    assert by["s512_tile_leaves"][:2] == tuple(by["gemm_pearlc"][2:4])


def test_nohash_forms_and_multiplies_without_hashing(device_run):
    make, kernel, weights, *_ = device_run
    run = make(hashing=False)
    kernel.dev.log.clear()
    run.gemm(0, torch.randn(16, 1024).to(torch.bfloat16), weights[0], None)
    assert [n for n, *_ in kernel.dev.log if not n.endswith("_smem")] == ["stats_s5", "eprime", "form_s5", "words", "pa",
                                                                          "gemm_pearlc_d"]
    assert run.end() == {"units": 1, "tiles": 0, "hashing": False, "schedule": "serial"}


def test_a_pass_screens_at_most_its_flag_rows(device_run):
    make, _, weights, _, _ = device_run
    run = make(flag_rows=48)
    assert run.flag_buf.shape == (48, 2)
    run.begin()
    x = torch.randn(32, 1024).to(torch.bfloat16)
    run.gemm(0, x, weights[0], None)
    with pytest.raises(RuntimeError, match="over its capacity .* 48 screened rows"):
        run.gemm(1, x, weights[1], None)


def test_a_retained_pass_writes_what_the_verifier_opens(device_run, tmp_path):
    make, _, weights, roots, salt = device_run
    run = make()
    run.begin(retain=str(tmp_path))
    run.gemm(0, torch.randn(20, 1024).to(torch.bfloat16), weights[0], None)
    run.gemm(1, torch.randn(32, 1024).to(torch.bfloat16), weights[1], None)
    out = run.end()
    man = json.loads(Path(out["manifest"]).read_text())
    assert man["schema"] == PPD.MANIFEST_SCHEMA and man["scheme"] == SCHEME and man["salt"] == salt.hex()
    assert [(u["m"], u["real_rows"], u["weight"]) for u in man["matmuls"]] == [(32, 20, 0), (32, 32, 1)]
    assert man["commitment"] == {"root": out["transcript_root"], "count": 6}
    assert {w: v["root"] for w, v in man["weights"].items()} == {str(w): r.hex() for w, r in roots.items()}
    assert (tmp_path / man["matmuls"][0]["rows"]["path"]).stat().st_size == 2 * 32 * 1024
    assert (tmp_path / man["weights"]["0"]["rows"]["path"]).stat().st_size == 2 * 256 * 1024


def test_a_retained_pass_starts_from_poisoned_outputs_and_omit_drops_a_step(device_run, tmp_path):
    """The call runs as vLLM runs it, under inference mode, so its shape's buffers are inference tensors; the retained pass
    begins outside it (window r20260930-141053-b3c0's verify pass failed there)."""
    make, kernel, weights, *_ = device_run
    run = make()
    with torch.inference_mode():
        run.gemm(0, torch.randn(32, 1024).to(torch.bfloat16), weights[0], None)
    assert all(t.is_inference() for sh in run.shapes.values() for t in sh.b.values())
    run.end()
    run.begin(retain=str(tmp_path))
    assert bool((run.leaves == 0xA5).all()) and all(bool((t == 0xA5).all()) for sh in run.shapes.values() for t in sh.b.values())
    run.omit = frozenset({"hash_leaf"})
    kernel.dev.log.clear()
    run.gemm(0, torch.randn(32, 1024).to(torch.bfloat16), weights[0], None)
    assert "s512_tile_leaves" not in [n for n, *_ in kernel.dev.log] and "gemm_pearlc_d" in [n for n, *_ in kernel.dev.log]


def test_the_e2e_gate_fails_a_device_that_writes_nothing(device_run):
    """On the recording stand-in nothing is written, so every pass commits the poison: the repeat holds, and only the negative
    control (the call with its tile leaves omitted) can tell. It must, for every shape."""
    make, _, weights, *_ = device_run
    E2E = load("pearl_c_vllm_e2e", KERNEL.parent / "pearl_c_vllm" / "e2e.py")
    rows = E2E.gate(make(), weights)
    calls = [r for r in rows if r["check"] == "call"]
    assert [(r["n"], r["m"]) for r in calls] == [(256, 8192), (256, 32), (128, 8192), (128, 32)]
    assert all(r["commitment_repeats"] and not r["omitted_leaves_rejected"] and not r["ok"] for r in calls)
    (check,) = [r for r in rows if r["check"] == "screen"]
    assert check["ok"] and check["flags_equal"] and check["screen"] == "torch"


# ---- the trees, their hash kernels run on the host ----------------------------------------------------------------

TWIN = load("simt_twin", KERNEL.parent / "pearl_c" / "simt_twin.py")


class TwinDev(RecordingDev):
    """`RecordingDev` that also runs each hash kernel `simt_twin` builds, thread for thread, at the addresses the pipeline
    passes: the run's buffers are CPU tensors, so the trees are really computed.  The GEMMs, forming and seed_line_a don't run."""

    def launch(self, name, grid, block, smem=0, p=(), i=(), stream=None):
        super().launch(name, grid, block, smem, p, i, stream)
        if name in TWIN.KERNELS:
            a = TWIN.Args()
            for j, v in enumerate(p):
                a.p[j] = v
            for j, v in enumerate(i):
                a.i[j] = v
            getattr(TWIN.build(), f"run_{name}")(ctypes.byref(a), grid, block, smem)


@pytest.mark.parametrize("scheme", sorted(PPD.SCHEMES))
def test_the_device_paths_keys_and_tree_hash_are_the_schemes(scheme):
    """The constants the pipeline reads (the prover's salt key, then `audit.hm96_kernel_constants`) and a run's tree hash are
    the scheme's: hm96-sha512's.  Without the prover's 32-byte key the constants are refused."""
    s = PPD.scheme_of(scheme)
    kernel = PPD.Kernel(R, F, RecordingDev(), "0" * 64)
    for bad in (None, bytes(31)):
        with pytest.raises(ValueError, match="salt key"):
            PPD._constants(s, torch.device("cpu"), bad)
    t = PPD._constants(s, torch.device("cpu"), SALT_KEY)
    assert PPD._bytes(t) == SALT_KEY.ljust(256, b"\0") + hm96_kernel_constants()
    assert PPD._constant_keys(t) == {"salt_key": t.data_ptr(), "hm": t.data_ptr() + 256}
    rng = np.random.default_rng(1)
    weights = {0: torch.from_numpy(bf16_words(rng, 128, 1024).view(np.float32).copy()).to(torch.bfloat16)}
    roots = PPD.weight_roots(kernel, weights, scheme, salt_key=SALT_KEY)
    run = PPD.PearlCDeviceRun(s, f"pouw:{scheme}", kernel, weights, roots, *salt_coin(roots), capacity=16, salt_key=SALT_KEY)
    assert run.tree_hash == s.commitment_hash == C.TREE_HASH["h3"] == HM96 and PPD._bytes(run.const) == PPD._bytes(t)


@pytest.mark.slow
@pytest.mark.skipif(not (shutil.which("g++") or shutil.which("c++")), reason="simt_twin needs a C++ compiler")
@pytest.mark.parametrize("scheme", sorted(PPD.SCHEMES))
def test_every_tree_the_device_path_commits_is_its_hashing_formats(scheme):
    """B's roots, each call's A root over its m₁₆ rows (filler zero), and the pass's tile tree over the leaves the calls wrote,
    their kernels run by `simt_twin`: each is `MerkleTree`'s root in hm96-sha512 (frame-v3-sha512 over values salted by the
    prover's key).  Each call's row digests are s512_rows', each `pearl_c.row_digest` of its row; its row seeds are s512_seeds'
    over them, the scheme's `row_seeds`, with their line keys, which E_A's lines read for the m₁₆ committed rows; and seed_A is
    the scheme's."""
    s = PPD.scheme_of(scheme)
    h, nb, key = s.commitment_hash, 64, SALT_KEY
    rng = np.random.default_rng(5)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16) for w, n in ((0, 256), (1, 128))}
    rows = lambda x: [r.tobytes() for r in x.float().numpy()]  # noqa: E731
    kernel = PPD.Kernel(R, F, TwinDev(), "0" * 64)
    roots = PPD.weight_roots(kernel, weights, scheme, salt_key=key)
    assert roots == {w: commit_rows(weight_binding(w), -2, rows(t), h, key, HM96_WEIGHTS + w).commitment.root
                     for w, t in weights.items()}
    declared = Workload(tuple(Matmul(w, Shape(16, 1024, t.shape[0]), w) for w, t in weights.items()))
    coins = salt_coin(roots)
    salt = Epoch.start(declared, roots, *coins).salt
    run = PPD.PearlCDeviceRun(s, f"pouw:{scheme}", kernel, weights, roots, *coins, capacity=1 << 12, salt_key=key)
    calls = [(0, torch.from_numpy(bf16_words(rng, 20, 1024).view(np.float32).copy()).to(torch.bfloat16)),
             (1, torch.from_numpy(bf16_words(rng, 32, 1024).view(np.float32).copy()).to(torch.bfloat16))]
    for w, x in calls:
        kernel.dev.log.clear()
        run.gemm(w, x, weights[w], None)
        names = [n for n, *_ in kernel.dev.log if not n.endswith("_smem")]
        tree = ["s512_rows", "hm_leaves", "s512_frame_leaves", "s512_tree_small", "s512_seeds", "s512_seeds", "lines"]
        assert names[:len(tree)] == tree, names
        sh, index = run.shapes[(32, weights[w].shape[0], 1024)], len(run.cur.matmuls) - 1
        a = torch.zeros((32, 1024), dtype=torch.float32)
        a[:x.shape[0]] = x.float()
        mm = Matmul(index, Shape(32, 1024, weights[w].shape[0]), w)
        ctx = Context(salt, mm, PPD._bytes(run.roots_buf[nb * index:nb * index + nb]), roots[w])
        want = s.row_seeds(ctx, rows(a), range(32))
        assert PPD._bytes(sh.b["rowdig_a"][:64 * 32]) == b"".join(C.row_digest(r) for r in rows(a))
        assert PPD._bytes(sh.b["seeds_a"][:32 * 32]) == b"".join(want)
        assert PPD._bytes(sh.b["lkeys_a"][:32 * 32]) == b"".join(C.line_key(z) for z in want)
        assert PPD._bytes(sh.b["seed_a"][:32]) == s.noise(ctx).seed_a
        (lines_ea,) = [(p, i) for n, _, p, i in kernel.dev.log if n == "lines"]
        assert lines_ea[0][4] == sh.b["lkeys_a"].data_ptr() and lines_ea[1][2:5] == (sh.mp, R.NORM_256, 32)
        (_, row_seeds) = [(p, i) for n, _, p, i in kernel.dev.log if n == "s512_seeds"]
        assert row_seeds[0][0] == sh.b["rowdig_a"].data_ptr() and row_seeds[1][2] == 1
    for index, (w, x) in enumerate(calls):
        a = torch.zeros((32, 1024), dtype=torch.float32)
        a[:x.shape[0]] = x.float()
        want = commit_rows(activation_binding(salt, index), index, rows(a), h, key, index).commitment.root
        assert PPD._bytes(run.roots_buf[nb * index:nb * index + nb]) == want, index
    count = run.cur.tiles
    assert count == 4 + 2
    leaves = rng.integers(0, 256, nb * count, dtype=np.uint8).tobytes()      # no GEMM ran: every tile's leaf is the same
    run.leaves[:nb * count] = torch.frombuffer(bytearray(leaves), dtype=torch.uint8)
    kernel.dev.log.clear()
    out = run.end()
    assert [n for n, *_ in kernel.dev.log][-2] == "s512_frame_leaves"
    dt = transcript_domain(Epoch(Workload(tuple(run.cur.matmuls)), roots, salt), count, h)
    tree = MerkleTree(dt, {i: leaves[nb * i:nb * i + nb] + hm96_salt(key, HM96_TRANSCRIPT, i) for i in range(count)},
                      lambda _: TILE_SCHEMA)
    assert out["transcript_root"] == tree.commitment.root.hex()


# ---- the verifier on an honest pass -------------------------------------------------------------------------------

def laid_out(domain, leaves: list[bytes]) -> bytes:
    return b"".join(b"".join(level) for level in tree_levels(domain, leaves))


def bf16_file(rows: list[bytes]) -> bytes:
    return b"".join((np.frombuffer(r, "<u4") >> 16).astype("<u2").tobytes() for r in rows)


#: the reference verifier's Skip debit has no rule without promotion, so v2 (sm120-unpromoted) isn't provable here
@pytest.fixture(params=sorted(s for s in PPD.SCHEMES if "unpromoted" not in s))
def honest_pass(tmp_path, request):
    """Two matmuls of a test instance (k = 128, 16 x 16 tiles), one with 12 filler rows and one liveness-failing row, proved by
    the scheme's reference prover and written in the device run's layout, for each promoted device scheme (hm96 trees, their
    salts under the prover's key, which the manifest retains; the unpromoted record's debit isn't stated, so pearl_c_work can't
    audit it).  The key is fixed: -h3's row seeds follow A's salted roots, and under some keys
    (about 2 in 15) a credited row of a drawn tile is over its own cap, which the law and verify_run both reject."""
    scheme = request.param
    s = PPD.scheme_of(scheme, k_min=32, tile=(16, 16))
    rng, k = np.random.default_rng(11), 128
    wrows = {0: bf16_words(rng, 16, k), 1: bf16_words(rng, 32, k)}
    a0 = bf16_words(rng, 32, k)
    a0[20:] = 0
    a0[5, [1, 2, 3]] = np.float32(1e4).view(np.uint32) & 0xFFFF0000
    acts = {0: a0, 1: bf16_words(rng, 16, k)}
    rows = lambda x: [r.tobytes() for r in x]
    work = Workload((Matmul(0, Shape(32, k, 16), 0), Matmul(1, Shape(16, k, 32), 1)))
    prover = Prover(s, {w: rows(x) for w, x in wrows.items()}, salt_key=bytes(32))
    salted = lambda v, nonce, i: v + hm96_salt(prover.salt_key, nonce, i)  # noqa: E731
    coins = salt_coin(prover.weight_roots)
    epoch = Epoch.start(work, prover.weight_roots, *coins)
    tr = prover.prove(epoch, {i: rows(x) for i, x in acts.items()})
    splits = {w: PW.Split(k) for w in wrows}
    records = PW.declare(s, epoch, {i: rows(x) for i, x in acts.items()}, {w: rows(x) for w, x in wrows.items()}, splits,
                         real_rows={0: 20})
    assert records[0].excluded_a == (5,)
    (tmp_path / "calls").mkdir()
    (tmp_path / "weights").mkdir()
    weights = {}
    for w, x in wrows.items():
        d = rows_domain(weight_binding(w), -2, len(x), s.commitment_hash)
        weights[str(w)] = {"n": len(x), "k": k, "root": prover.weight_roots[w].hex(), "excluded_b": list(records[0 if w == 0 else 1].excluded_b),
                           "rows": PPD._write(tmp_path / "weights" / f"{w}.rows.bf16", bf16_file(rows(x)), digest=False),
                           "tree": PPD._write(tmp_path / "weights" / f"{w}.tree",
                                              laid_out(d, [d.leaf(j, j, ROW_SCHEMA, salted(r, HM96_WEIGHTS + w, j))
                                                           for j, r in enumerate(rows(x))]))}
    matmuls = []
    for u, rec, root in zip(work.matmuls, records, tr.activation_roots):
        x = rows(acts[u.index])
        d = rows_domain(activation_binding(epoch.salt, u.index), u.index, u.shape.m, s.commitment_hash)
        matmuls.append({"index": u.index, "m": u.shape.m, "k": k, "n": u.shape.n, "weight": u.weight, "real_rows": rec.real_rows,
                        "k_kept": rec.k_kept, "excluded_a": list(rec.excluded_a), "root_a": root.hex(),
                        "rows": PPD._write(tmp_path / "calls" / f"{u.index}.rows.bf16", bf16_file(x), digest=False),
                        "tree": PPD._write(tmp_path / "calls" / f"{u.index}.tree",
                                           laid_out(d, [d.leaf(i, i, ROW_SCHEMA, salted(r, u.index, i)) for i, r in enumerate(x)]))})
    count = tr.commitment.count
    leaves = [prover.digest(t) for t in range(count)]
    dt = transcript_domain(epoch, count, s.commitment_hash)
    head = {"schema": PPD.MANIFEST_SCHEMA, "run": RUN, "entry": f"pouw:{scheme}", "scheme": scheme,
            "instance": {"k_min": 32, "tile": [16, 16]},
            "coins": coins[0].to_json(), "salt_opening": coins[1].to_json(), "salt": epoch.salt.hex(),
            "records_root": PW.records_root(epoch.salt, epoch.weights_root, splits, records).hex(),
            "commitment": {"root": tr.commitment.root.hex(), "count": count},
            "leaves": PPD._write(tmp_path / "leaves", b"".join(leaves)),
            "tree_t": PPD._write(tmp_path / "tree_t", laid_out(dt, [dt.leaf(i, i, TILE_SCHEMA, salted(v, HM96_TRANSCRIPT, i))
                                                                    for i, v in enumerate(leaves)]))}
    head["hm96_salt_key"] = prover.salt_key.hex()
    path = PPD.write_manifest(tmp_path, head, weights, matmuls)
    (tmp_path / "coins.json").write_text(json.dumps(coins[0].to_json()))
    return path, s, epoch, prover, tr, splits, records


@pytest.mark.slow
def test_verify_run_accepts_an_honest_pass_as_the_work_law_does(honest_pass):
    path, s, epoch, prover, tr, splits, records = honest_pass
    source = b"verifier" * 4
    got = VR.verify(path, 3, 1, "test", source, coins=published(path))
    want = PW.audit_replay(s, epoch, prover, tr, splits, records, 0, PW.RecordedDraw(1, source), 3)
    assert got["accepted"] and want.audit.accepted, got.get("line") or got.get("error")
    assert tuple(got["tiles"]) == want.audit.selected and got["exclusion_tiles"] == 1
    assert got["line"].startswith(f"ACCEPT {s.name} pouw:{s.name}: pearl_c_work's tiles (3 drawn, 1 for excluded rows)")
    assert got["transcript"] == f"{RUN}:{path} sha256:{got['transcript_sha256']}" and got["transcript"] in got["line"]
    assert VR.main([str(path), *coins_args(path), "--jobs", "1", "--commit", "0" * 7]) == 0
    assert VR.main([str(path), *coins_args(path), "--jobs", "1", "--commit", "0" * 7, "--expect-reject"]) == 1
    fast = VR.verify(path, 3, 1, "test", source, fast=True, cross_check=1, coins=published(path))
    assert fast["replay"] == "batch" and fast["cross_checked"] == fast["tiles"] and "cross_check_failed" not in fast
    assert {k: fast[k] for k in ("accepted", "tiles", "verdicts")} == {k: got[k] for k in ("accepted", "tiles", "verdicts")}
    assert C.form_v1 is SD.FT._SCALAR["form_v1"]
    assert VR.main([str(path), *coins_args(path), "--jobs", "2", "--commit", "0" * 7, "--fast", "--cross-check", "1"]) == 0


@pytest.mark.slow
def test_verify_run_rejects_a_wrong_weight_row_and_undeclared_exclusions(honest_pass):
    path, *_ = honest_pass
    man = json.loads(path.read_text())
    rows = path.parent / man["weights"]["0"]["rows"]["path"]
    good = rows.read_bytes()
    bad = bytearray(good)
    for j in range(16):
        bad[2 * 128 * j + 1] ^= 0x01
    rows.write_bytes(bytes(bad))
    r = VR.verify(path, 3, 1, "test", b"verifier" * 4, coins=published(path))
    assert not r["accepted"] and "weight opening" in r["line"] and r["line"].startswith("REJECT") and RUN in r["line"]
    fast = VR.verify(path, 3, 1, "test", b"verifier" * 4, fast=True, cross_check=1, coins=published(path))
    assert {k: fast[k] for k in ("accepted", "verdicts")} == {k: r[k] for k in ("accepted", "verdicts")}
    assert VR.main([str(path), *coins_args(path), "--jobs", "1", "--commit", "0" * 7, "--expect-reject"]) == 0
    rows.write_bytes(good)
    man["matmuls"][0]["excluded_a"] = []
    path.write_text(json.dumps(man))
    r = VR.verify(path, 3, 1, "test", b"verifier" * 4, coins=published(path))
    assert not r["accepted"] and "records are not the ones" in r["line"]


@pytest.mark.slow
def test_h3_screens_check_holds_a_pass_to_the_reference(honest_pass):
    """h3_screen's check on a pass the reference proved: every tile's leaf is the reference's, and a call's dumped row digests,
    row seeds, seed_A and E_A lines (16 padding rows under seed_A) are; a wrong line or row digest is caught."""
    path, s, epoch, prover, tr, splits, records = honest_pass
    HS = load("pearl_c_vllm_h3_screen", KERNEL.parent / "pearl_c_vllm" / "h3_screen.py")
    d = path.parent
    assert HS._tiles(d, 1) == {"tiles": tr.commitment.count, "failed": 0, "failures": []}
    for u in epoch.workload.matmuls:
        ctx = Context(epoch.salt, u, tr.activation_roots[u.index], epoch.weight_roots[u.weight])
        data = VR.Files(d)(json.loads(path.read_text())["matmuls"][u.index]["rows"])
        rows = [VR.f32_row(data, u.shape.k, i) for i in range(u.shape.m)]
        seeds, seed_a = s.row_seeds(ctx, rows, range(u.shape.m)), s.noise(ctx).seed_a
        lines = [C.sample_line(x, 0, 0, 0) for x in seeds] + [C.sample_line(seed_a, 0, 0, r) for r in range(u.shape.m, u.shape.m + 16)]
        (d / f"call{u.index}.seeds_a").write_bytes(b"".join(seeds))
        (d / f"call{u.index}.seed_a").write_bytes(seed_a)
        (d / f"call{u.index}.ea_codes").write_bytes(b"".join(bytes(x) for x in lines))
        (d / f"call{u.index}.rowdig_a").write_bytes(b"".join(C.row_digest(r) for r in rows))
    got = HS._calls(d)
    assert got["ok"] and [c["padded"] - c["m16"] for c in got["calls"]] == [16, 16]
    (d / "call1.ea_codes").write_bytes(bytes(32) + (d / "call1.ea_codes").read_bytes()[32:])
    assert HS._calls(d)["calls"][1]["ea_rows_wrong"] == [0]
    (d / "call1.rowdig_a").write_bytes(bytes(64) + (d / "call1.rowdig_a").read_bytes()[64:])
    assert not HS._calls(d)["calls"][1]["row_digests"]
def test_verify_run_reads_a_v1_manifest_only_as_a_recorded_replay(honest_pass):
    """A v1 manifest (a pass recorded before epoch coins) is refused unless ``--recorded``; read as a recorded pass it needs
    a beacon its salt is from; a manifest of another schema is refused either way."""
    path, *_ = honest_pass
    man = json.loads(path.read_text())
    v1 = path.parent / "manifest-v1.json"
    for schema, recorded, why in ((PPD.MANIFEST_SCHEMA_RECORDED, False, "recorded before epoch coins"),
                                  (PPD.MANIFEST_SCHEMA_RECORDED, True, "recorded pass has no beacon"),
                                  ("verity-vllm/pouw-pearl-c-run/v0", True, f"not {PPD.MANIFEST_SCHEMA}")):
        v1.write_text(json.dumps({**man, "schema": schema}))
        r = VR.verify(v1, 3, 1, "test", b"verifier" * 4, coins=published(path), recorded=recorded)
        assert not r["accepted"] and r["line"].startswith("REJECT") and why in r["line"], (schema, recorded, r["line"])
    v1.unlink()


def test_a_pass_whose_salt_coin_is_foreign_early_mislabelled_or_tampered_is_refused():
    """A pass's epoch re-starts only from the epoch's own salt coin under the commitment the verifier holds, opened after
    the weights' root, with the salt it derives: a coin of another challenger, one opened after another record, the draw
    coin's label, a flipped coin byte, a claimed commitment that isn't the published one, or a salt that isn't the coin's is
    refused.  A pass recorded with a beacon (a v1 manifest) is read only as a replay of a recorded pass, and a manifest of
    any other schema not at all."""
    roots = {0: bytes([1]) * 32, 1: bytes([2]) * 32}
    commitment, opening = salt_coin(roots)
    epoch = Epoch.start(Workload((Matmul(0, Shape(16, 1024, 256), 0), Matmul(1, Shape(16, 1024, 128), 1))), roots, commitment,
                        opening)
    man = {"schema": PPD.MANIFEST_SCHEMA, "matmuls": [{"index": 0, "m": 16, "k": 1024, "n": 256, "weight": 0}, {"index": 1, "m": 16, "k": 1024, "n": 128, "weight": 1}],
           "weights": {str(w): {"root": r.hex()} for w, r in roots.items()},
           "coins": commitment.to_json(), "salt_opening": opening.to_json(), "salt": epoch.salt.hex()}
    got, how = PPD.manifest_epoch(man, commitment)
    assert got == epoch and how == "coin"
    foreign_c, foreign = salt_coin(roots, entropy=8)
    tampered = opening.to_json()
    tampered["coin"] = bytes([opening.coin[0] ^ 1]).hex() + tampered["coin"][2:]
    bad = {"foreign": {"salt_opening": foreign.to_json()},
           "early": {"salt_opening": {**opening.to_json(), "after": weights_root({0: roots[0], 1: bytes(32)}).hex()}},
           "mislabelled": {"salt_opening": {**opening.to_json(), "label": "draw"}},
           "tampered": {"salt_opening": tampered},
           "claimed": {"coins": foreign_c.to_json(), "salt_opening": foreign.to_json()},
           "salt": {"salt": "00" * 32}}
    why = {"foreign": "does not open", "early": "not opened after", "mislabelled": "'draw' coin", "tampered": "does not open",
           "claimed": "not the epoch's", "salt": "not its coin's"}
    for name, change in bad.items():
        with pytest.raises(DomainError, match=why[name]):
            PPD.manifest_epoch({**man, **change}, commitment)
    with pytest.raises(DomainError, match="the verifier holds"):
        PPD.manifest_epoch(man, None)
    with pytest.raises(DomainError, match="no salt coin"):
        PPD.manifest_epoch({k: v for k, v in man.items() if k not in ("coins", "salt_opening")}, commitment)
    recorded = {k: v for k, v in man.items() if k not in ("coins", "salt_opening")}
    recorded |= {"schema": PPD.MANIFEST_SCHEMA_RECORDED, "beacon": SEED.hex(),
                 "salt": Epoch.unverified(epoch.workload, roots, SEED).salt.hex()}
    with pytest.raises(DomainError, match="recorded before epoch coins: only a replay of a recorded pass reads it"):
        PPD.manifest_epoch(recorded, commitment)
    assert PPD.manifest_epoch(recorded, None, recorded=True)[1] == "recorded beacon"
    with pytest.raises(DomainError, match="recorded pass has no beacon"):
        PPD.manifest_epoch({**man, "schema": PPD.MANIFEST_SCHEMA_RECORDED}, commitment, recorded=True)
    with pytest.raises(DomainError, match="not its recorded beacon's"):
        PPD.manifest_epoch({**recorded, "salt": "00" * 32}, None, recorded=True)
    assert PPD.manifest_epoch(man, commitment, recorded=True) == (epoch, "coin")
    for schema in ("verity-vllm/pouw-pearl-c-run/v0", None):
        for flag in (False, True):
            with pytest.raises(DomainError, match=f"schema {schema!r}, not {PPD.MANIFEST_SCHEMA}"):
                PPD.manifest_epoch({**man, "schema": schema}, commitment, recorded=flag)
    with pytest.raises(DomainError, match="recorded before epoch coins"):
        PPD.manifest_epoch({**recorded, "schema": PPD.MANIFEST_SCHEMA_RECORDED, **{k: man[k] for k in ("coins", "salt_opening")}},
                           commitment)


@pytest.mark.slow
def test_served_debit_records_the_tile_rules_own_debit_and_cap(honest_pass, tmp_path):
    path, s, epoch, prover, tr, splits, records = honest_pass
    out = tmp_path / "out"
    assert SD.main([str(path), *coins_args(path), "--tiles", "3", "--jobs", "1", "--out", str(out)]) == 0
    assert PW._debit_cap is SD._DEBIT_CAP and PW.D.chain_debit is SD._CHAIN_DEBIT
    (got,) = json.loads((out / "summary.json").read_text())["passes"]
    assert got["scheme"] == s.name and got["cap"] == str(s.cap) and got["openings_failed"] == 0
    assert set(got["strata"]) == {"m32-k128-n16", "m16-k128-n32"} and got["tiles_total"] == tr.commitment.count
    assert got["strata"]["m32-k128-n16"]["credited_macs"] == (20 - 1) * 16 * 128
    tiles = [json.loads(line) for line in (out / f"{path.parent.name}.tiles.jsonl").read_text().splitlines()]
    assert len(tiles) == got["sampled"] and got["measured"] == sum("share" in t for t in tiles) > 0
    refs = Verifier(PW._TileRuleView(s), epoch).refs
    for t in (t for t in tiles if "share" in t):
        ref, rec = refs[t["tile"]], records[t["matmul"]]
        op = prover.open(epoch, ref.id)
        a, b = [opened_value(s.commitment_hash, o.value) for o in op.a], [opened_value(s.commitment_hash, o.value) for o in op.b]
        u = epoch.workload.matmuls[ref.matmul]
        ctx = Context(epoch.salt, u, tr.activation_roots[ref.matmul], epoch.weight_roots[u.weight])
        ops = s.encode(ctx, s.noise(ctx), a, b, ref.tile)
        rows, cols = rec.credited(ref.tile)
        debit, cap = PW._debit_cap(s, rec, ref.tile, ops, rows, cols)
        credit = PW.cell_credit(s, rec) * len(rows) * len(cols)
        assert (t["debit_exact"], t["ratio"], t["credit_exact"]) == (str(debit), float(debit / cap), str(credit))
        assert {key: t[key] for key in t if key.startswith("share")} == {"share": float(debit / credit)}
        pos_a, pos_b = {i: p for p, i in enumerate(ref.tile.rows)}, {j: p for p, j in enumerate(ref.tile.cols)}
        a_codes, b_codes = [ops.a.codes[pos_a[i]] for i in rows], [ops.b.codes[pos_b[j]] for j in cols]
        assert (t["atoms"], t["later_zero_promotions"]) == PW.D.chain_debit(a_codes, b_codes, s.device)
        assert t["ok"] == (PW.check_opened(s, rec, ref.tile, a, b, ops) is None)
    assert got["later_zero_promotions"] == sum(t.get("later_zero_promotions", 0) for t in tiles)


@pytest.mark.slow
def test_served_debit_per_input_bounds_each_input_on_its_own(honest_pass, tmp_path):
    """Per-input mode (lean's ruling on #923): each chosen input is one matmul of the pass, measured in full or by
    work-weighted draws, with its own failures, `W`, `w_max` and a simultaneous upper limit; nothing is pooled."""
    path, s, epoch, prover, tr, splits, records = honest_pass
    out = tmp_path / "out"
    assert SD.main([str(path), *coins_args(path), "--per-input", "1", "--census-max", "4", "--draws", "3", "--jobs", "1", "--out", str(out)]) == 0
    doc = json.loads((out / "inputs.json").read_text())
    refs = Verifier(PW._TileRuleView(s), epoch).refs
    assert doc["m"] == len(doc["inputs"]) == 2 and doc["delta_per_input"] == 0.05 / 2
    for row in doc["inputs"]:
        ids = [r.id for r in refs if r.matmul == row["matmul"]]
        area = [len(refs[t].tile.rows) * len(refs[t].tile.cols) for t in ids]
        assert (row["NT"], row["W"], row["w_max"]) == (len(ids), sum(area), max(area))
        assert row["drawn"] == (len(ids) if row["method"] == "census" else 3) and row["openings_failed"] == 0
        assert 0 < row["upper"] <= 1 and (row["k"], row["n"]) == (records[row["matmul"]].matmul.shape.k, records[row["matmul"]].matmul.shape.n)
    assert doc["cited_upper"] == max(r["upper"] for r in doc["inputs"])
    lines = (out / f"{path.parent.name}.input-tiles.jsonl").read_text().splitlines()
    for d in ("calls", "weights"):
        shutil.rmtree(path.parent / d)
    again = tmp_path / "again"
    args = [str(path), "--per-input", "1", "--census-max", "4", "--draws", "3", "--jobs", "1", "--resumed-only", *coins_args(path)]
    assert SD.main([*args, "--out", str(again), "--resume", str(out / f"{path.parent.name}.input-tiles.jsonl")]) == 0
    assert json.loads((again / "inputs.json").read_text())["inputs"] == doc["inputs"]
    census = next(r for r in doc["inputs"] if r["method"] == "census")
    drop = next(x for x in lines if json.loads(x)["matmul"] == census["matmul"])
    (tmp_path / "short.jsonl").write_text("\n".join(x for x in lines if x != drop) + "\n")
    with pytest.raises(ValueError, match="unmeasured"):
        SD.main([*args, "--out", str(tmp_path / "short"), "--resume", str(tmp_path / "short.jsonl")])


@pytest.mark.slow
def test_served_debit_fresh_draws_measure_each_input_under_salts_of_its_own(honest_pass, tmp_path):
    """`--fresh-draws`: the per-input mode's inputs, each by N work-weighted trials under a salt of its own derived from the
    source, every trial the tile rule on the committed rows under that salt, bounded by Clopper–Pearson; a resume with the
    same source measures nothing again, and another source's trials are refused."""
    path, s, epoch, prover, tr, splits, records = honest_pass
    out, src = tmp_path / "out", "ab" * 32
    args = [str(path), "--per-input", "1", "--fresh-draws", "4", "--fresh-source", src, "--jobs", "1", "--cross-check", "2",
            *coins_args(path)]
    assert SD.main([*args, "--out", str(out)]) == 0
    doc = json.loads((out / "inputs.json").read_text())
    assert doc["fresh_source"] == src and doc["fresh_draws"] == 4 and doc["m"] == 2
    lines = [json.loads(x) for x in (out / f"{path.parent.name}.fresh-tiles.jsonl").read_text().splitlines()]
    assert len(lines) == 8 and len({t["salt"] for t in lines}) == 8 and epoch.salt.hex() not in {t["salt"] for t in lines}
    fails = [t for t in lines if not t["ok"]]
    assert sum(bool(t.get("cross_checked")) for t in lines) == 4 and all(t["openings_ok"] for t in lines)
    # an honest row can exceed its own cap (-h3's row cap, k 128 here), and the served path volunteers no rows
    assert all("the row cap" in t["reason"] for t in fails) and (not fails or PW.row_cap(s))
    for row in doc["inputs"]:
        failures = sum(t["matmul"] == row["matmul"] for t in fails)
        assert row["method"] == "fresh" and row["drawn"] == 4 and row["failures"] == failures
        assert row["upper"] == SD.clopper_pearson_upper(failures, 4, 0.025) and row["bound_method"].startswith("fresh")
    t = lines[0]
    ref = Verifier(PW._TileRuleView(s), epoch).refs[t["tile"]]
    op = prover.open(epoch, ref.id)
    a, b = [opened_value(s.commitment_hash, o.value) for o in op.a], [opened_value(s.commitment_hash, o.value) for o in op.b]
    u = epoch.workload.matmuls[ref.matmul]
    ctx = Context(bytes.fromhex(t["salt"]), u, tr.activation_roots[ref.matmul], epoch.weight_roots[u.weight])
    ops = s.encode(ctx, s.noise(ctx), a, b, ref.tile)
    rows, cols = records[ref.matmul].credited(ref.tile)
    debit, cap = PW._debit_cap(s, records[ref.matmul], ref.tile, ops, rows, cols)
    assert t["debit_exact"] == str(debit) and t["ratio"] == float(debit / cap)
    again = tmp_path / "again"
    resume = ["--resume", str(out / f"{path.parent.name}.fresh-tiles.jsonl")]
    assert SD.main([*args, "--out", str(again), *resume]) == 0
    assert json.loads((again / "inputs.json").read_text())["inputs"] == doc["inputs"]
    with pytest.raises(ValueError, match="another salt"):
        SD.main([*args[:6], "cd" * 32, *args[7:], "--out", str(tmp_path / "other"), *resume])


def test_fresh_trials_draw_exactly_by_work_from_a_source_fixed_in_the_command(tmp_path):
    """Lean's conditions on the fresh-trial ε: draw d is `key.uniform(W, d)` over the input's cumulative credited cells
    (exactly proportional to `TR.Wref` within the input), under a domain naming the table and a context naming the input,
    and the source is given in the command, never drawn inside the run."""
    from verity.ml.randomness import derive

    src, job, row = bytes(range(32)), {"sha256": "ab" * 32}, {"matmul": 7, "area": {40: 64 * 64, 41: 64 * 64, 42: 32 * 64}}
    got = SD.fresh_trials(job, row, 50, src)
    key = derive(src, "verity/pouw/fp8-served-table-fresh-trials/v1", {"manifest": job["sha256"], "matmul": 7, "use": "tiles"})
    cum = [4096, 8192, 10240]
    assert [t for t, *_ in got] == [[40, 41, 42][sum(key.uniform(10240, d) >= c for c in cum)] for d in range(50)]
    assert [d for _, _, d, _ in got] == list(range(50)) and len({s for *_, s in got}) == 50
    assert SD.fresh_trials(job | {"sha256": "cd" * 32}, row, 50, src) != got
    with pytest.raises(SystemExit):
        SD.main(["m.json", "--per-input", "1", "--fresh-draws", "4", "--out", str(tmp_path)])


ST = load("pearl_c_vllm_served_table", KERNEL.parent / "pearl_c_vllm" / "served_table.py")


def test_the_served_table_cites_each_models_worst_input_and_refuses_what_it_cannot_cite(tmp_path):
    """Every input is re-bounded at δ = 0.05/m, m every input the table measures: both models' and a model's fresh trials."""
    row = lambda run, i, nt, fails=0: {"pass": "decode", "run": run, "manifest_sha256": "ab", "matmul": i, "k": 4096, "n": 4096,
                                       "NT": nt, "W": nt * 4096, "w_max": 4096, "method": "census", "drawn": nt,
                                       "openings_failed": 0, "failures": fails, "share_max": 5e-5}
    doc = lambda run, nts: {"m": len(nts), "inputs": [row(run, i, nt) for i, nt in enumerate(nts)]}
    verify = lambda run, label, ok: {"manifest": f"/p/{run}/{label}/manifest.json", "accepted": ok, "transcript": f"{run}:/p sha256:00",
                                     "transcript_sha256": "00", "tiles": [1, 2], "line": f"{'ACCEPT' if ok else 'REJECT'} x", "replay": "batch",
                                     "cross_checked": [1]}
    trial = lambda i, fails: row("rq", i, 1) | {"method": "fresh", "drawn": 320, "failures": fails}
    files = {}
    for name, obj in {"q": doc("rq", [64, 32]), "l": doc("rl", [48]), "v1": verify("rq", "decode", True),
                      "v2": verify("rq", "control", False), "bad": verify("rq", "control", True),
                      "fresh": {"fresh_draws": 320, "fresh_source": "cd", "m": 2, "inputs": [trial(0, 0), trial(1, 1)]}}.items():
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps(obj))
    out = tmp_path / "out"
    assert ST.main(["--model", "qwen", str(files["q"]), str(files["v1"]), str(files["v2"]), "--model", "llama", str(files["l"]),
                    "--fresh", "qwen", str(files["fresh"]), "--out", str(out)]) == 0
    table = json.loads((out / "table.json").read_text())
    q, l = table["models"]
    assert table["m"] == 5 and q["delta_per_input"] == 0.01 and (q["m_model"], q["failures"]) == (2, 0)
    assert q["epsilon"] == pytest.approx(math.log(100) / 32) and q["epsilon_input"]["matmul"] == 1
    assert l["epsilon"] == pytest.approx(math.log(100) / 48)
    assert q["fresh"]["epsilon"] == SD.clopper_pearson_upper(1, 320, 0.01) and q["fresh"]["trials"] == 640 and "fresh" not in l
    assert [v["accepted"] for v in q["verify"]] == [True, False] and "| qwen | `rq` | 2 |" in (out / "table.md").read_text()
    assert q["tile_draws"] == 96 and "m = 5" in (out / "table.md").read_text()
    assert ST.main(["--model", "qwen", str(files["q"]), "--absent", "llama", "its pass is gone", "--out", str(out)]) == 0
    table = json.loads((out / "table.json").read_text())
    assert table["m"] == 2 and table["models"][0]["epsilon"] == pytest.approx(math.log(40) / 32)
    assert table["models"][1] == {"model": "llama", "absent": "its pass is gone"}
    assert "| llama | — | — | — | — | — | none | its pass is gone |" in (out / "table.md").read_text()
    assert ST.main(["--model", "qwen", str(files["q"]), "--model", "llama", str(files["l"]), "--pending", "llama", "1", "rp",
                    "--fresh", "qwen", str(files["fresh"]), "--out", str(out)]) == 0
    table = json.loads((out / "table.json").read_text())
    assert table["m"] == 6 and table["models"][1]["fresh_pending"] == {"m_model": 1, "run": "rp"}
    assert table["models"][1]["epsilon"] == pytest.approx(math.log(120) / 48) and "pending (`rp`)" in (out / "table.md").read_text()
    with pytest.raises(SystemExit):
        ST.main(["--model", "qwen", str(files["q"]), "--pending", "qwen", "2", "rp", "--fresh", "qwen", str(files["fresh"]),
                 "--out", str(out)])
    with pytest.raises(ValueError, match="committed list"):
        files["fresh"].write_text(json.dumps({"fresh_draws": 320, "fresh_source": "cd", "m": 1, "inputs": [trial(0, 0)]}))
        ST.main(["--model", "qwen", str(files["q"]), "--fresh", "qwen", str(files["fresh"]), "--out", str(out)])
    with pytest.raises(ValueError, match="control"):
        ST.main(["--model", "qwen", str(files["q"]), str(files["bad"]), "--out", str(out)])
    with pytest.raises(ValueError, match="one served pass"):
        files["q"].write_text(json.dumps(doc("rq", [64]) | {"inputs": [row("rq", 0, 64), row("rl", 1, 64)]}))
        ST.main(["--model", "qwen", str(files["q"]), "--out", str(out)])


def test_the_served_table_labels_a_windows_fraction_as_the_replay_diagnostic_beside_proving_cost(tmp_path):
    """A window's row comes from its accepting audit: credited against claimed GEMMs, capacity, the certified fraction with
    its uncertified end, the overhead over cuBLASLt and a proving-cost cell that points to hidden-zk until it has a number;
    a rejecting audit is refused."""
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"m": 1, "inputs": [{"pass": "decode", "run": "rq", "manifest_sha256": "ab", "matmul": 0,
                                                     "k": 4096, "n": 4096, "NT": 64, "W": 64, "w_max": 1, "method": "census",
                                                     "drawn": 64, "openings_failed": 0, "failures": 0, "share_max": 5e-5}]}))
    audit = lambda n, cap, ok=True: {
        "ok": ok, "scheme": "pearl-c-sm120-v1-h3", "shape": [8192, 8192, 8192], "t_s": "t0", "t_d": "t1", "verifier_commit": "c",
        "gemms": {"0": n // 2, "1": n - n // 2}, "tiles": [{"ok": True}] * 3,
        "controls": [{"label": "tampered", "ok": False}, {"label": "wrong salt", "ok": False}],
        "checks": [{"check": "gamma", "ok": ok}],
        "fraction": {"credited_gemms": n, "capacity_gemms": cap, "uncertified_fraction": n / cap, "fraction": 0.98 * n / cap,
                     "gamma": "1/100", "eps_s": "1/100", "slowdown": cap / n}}
    paths = []
    for i, doc in enumerate((audit(5_500, 10_000.4), audit(5_400, 10_000.4))):
        paths.append(tmp_path / f"a{i}.json")
        paths[-1].write_text(json.dumps(doc))
    out = tmp_path / "t"
    assert ST.main(["--model", "qwen", str(inputs), "--window", "w0", "r0", str(paths[0]), "--window", "w1", "r1", str(paths[1]),
                    "--out", str(out)]) == 0
    w0, w1 = json.loads((out / "table.json").read_text())["windows"]
    assert (w0["credited_gemms"], w0["claimed_gemms"], w0["tiles_accepted"], w0["proving_cost"]) == (5_500, 5_500, 3, None)
    assert w0["controls"] == {"tampered": "REJECT", "wrong salt": "REJECT"} and w1["fraction"] == pytest.approx(0.98 * 0.54, 1e-4)
    md = (out / "table.md").read_text()
    assert "## Replay diagnostic: salt-to-deadline windows" in md and "a diagnostic, not a verification" in md
    assert ("| `w0` | `r0` | pearl-c-sm120-v1-h3 | t0 | 5,500 / 5,500 | 10,000 | 53.90% [0.5390, 0.5500] | 1.818 | "
            "— (hidden-zk) | 3 / 3 | tampered REJECT, wrong salt REJECT |") in md
    assert "spans 52.92% to 53.90% (spread 0.98%, 98.0 basis points)" in md
    assert ST.main(["--model", "qwen", str(inputs), "--window", "w0", "r0", str(paths[0]),
                    "--proving-cost", "pearl-c-sm120-v1-h3", "$1.00 per GEMM", "--out", str(out)]) == 0
    assert "| $1.00 per GEMM | 3 / 3 |" in (out / "table.md").read_text()
    paths[1].write_text(json.dumps(audit(5_400, 10_000.4, ok=False)))
    with pytest.raises(ValueError, match="did not accept"):
        ST.main(["--model", "qwen", str(inputs), "--window", "w1", "r1", str(paths[1]), "--out", str(out)])


FH = load("pearl_c_vllm_forming_headroom", KERNEL.parent / "pearl_c_vllm" / "forming_headroom.py")


def test_forming_headroom_on_a_recorded_line(tmp_path):
    """A real decode line (Qwen3-8B's down projection, pass r20261003-210254-47a3) recomputes at Prices.sm120, and at the
    cited Prices.sm120Loop its verdict flips only past 11.0% salt-dead elements, by hand."""
    line = {"tile": 1769828, "matmul": 7179, "m": 32, "k": 12288, "n": 4096, "credited_rows": 32, "credited_cols": 64,
            "ok": True, "atoms": 14, "later_zero_promotions": 0, "debit_exact": "476", "cap": 27426.816}
    path = tmp_path / "q" / "decode.input-tiles.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps(line) + "\n" + json.dumps(line | {"tile": 1, "atoms": 200, "debit_exact": "6800"}) + "\n")
    cell = 12288 + Fraction(12288 * 1047, 125 * 128) + 96 + 128 + 120 - Fraction(1047, 125)
    want = ((cell * 2048 / 1000 - (32 + Fraction(1047, 500)) * 14) * 4096 / (40 * 64)) / (32 * 12288)
    assert FH.dead_to_flip(line) == want and 0.1099 < want < 0.1100
    out = tmp_path / "fh.json"
    assert FH.main([str(path), "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["tiles"] == 2 and doc["unit_rows_covered_by_rho_D"] and doc["inputs"][0]["worst_tile"] == 1
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"m": 1, "inputs": [{"pass": "decode", "run": "rq", "manifest_sha256": "ab", "matmul": 7179,
                                                     "k": 12288, "n": 4096, "NT": 64, "W": 64, "w_max": 1, "method": "census",
                                                     "drawn": 64, "openings_failed": 0, "failures": 0, "share_max": 5e-5}]}))
    assert ST.main(["--model", "qwen", str(inputs), "--forming", "qwen", str(out), "--out", str(tmp_path / "t")]) == 0
    assert json.loads((tmp_path / "t" / "table.json").read_text())["models"][0]["forming"]["tiles"] == 2
    assert "flips only if at least" in (tmp_path / "t" / "table.md").read_text()
    quality = {"run": "rp", "model": "/m/q", "dataset": {}, "windows": 2, "window_tokens": 9, "gates": {"fp8": True, "v1": True},
               "summary": {"bf16": {"perplexity": 7.5, "top1_agreement_vs_bf16": 1.0, "greedy_token_agreement_vs_bf16": 1.0},
                           "pearlc-v1": {"scheme": "pearl-c-sm120-v1-h3", "perplexity": 7.75, "top1_agreement_vs_bf16": 0.9,
                                         "greedy_token_agreement_vs_bf16": 0.6}}}
    qpath = tmp_path / "quality.json"
    qpath.write_text(json.dumps(quality))
    assert ST.main(["--model", "qwen", str(inputs), "--quality", "qwen", str(qpath), "--out", str(tmp_path / "t")]) == 0
    md = (tmp_path / "t" / "table.md").read_text()
    assert "| qwen | `rp` | pearlc-v1 | pearl-c-sm120-v1-h3 | 7.7500 | 0.9000 | 0.6000 |" in md
    assert "| qwen | `rp` | bf16 | — | 7.5000 |" in md and "differ only in their hashes" in md
    qpath.write_text(json.dumps(quality | {"gates": {"fp8": True, "v1": False}}))
    with pytest.raises(ValueError, match="gates"):
        ST.main(["--model", "qwen", str(inputs), "--quality", "qwen", str(qpath), "--out", str(tmp_path / "t")])
    flagged = line | {"atoms": 200, "debit_exact": "6800"}
    assert FH.dead_to_flip(flagged) < want - Fraction(32 * 3, 32 * 12288)
    path.write_text(json.dumps(line | {"debit_exact": "477"}) + "\n")
    with pytest.raises(ValueError, match="not Prices.sm120"):
        FH.main([str(path), "--out", str(out)])


FD = load("pearl_c_vllm_forming_domain", KERNEL.parent / "pearl_c_vllm" / "forming_domain.py")


def test_forming_domain_on_a_synthetic_pass(tmp_path, monkeypatch):
    """The domain check's numpy rows agree with s5_stats and an all-Fraction count on every credited row, skip filler and
    excluded rows, and find a row with α·s > 444 (no stride energy: ρ = 0, α·s = 448, so every element of it is loud and its
    unit fails liveness too), a row through the subnormal fallback, and a unit whose loud elements break liveness at 8ρ_max.
    Its pool forks: under forkserver (Python 3.14's default) a worker can't import this file's module to unpickle a task."""
    rng = np.random.default_rng(7)
    k = 256
    bf16 = lambda a: (np.asarray(a, np.float32).view(np.uint32) >> 16).astype("<u2")
    a = rng.normal(0, 1, (64, k)).astype(np.float32)
    a[5, ::8] = 0
    a[7, 8] = 2.0 ** -70
    b = rng.normal(0, 1, (4, k)).astype(np.float32)
    b[:, 1:80:8] = 40.0
    split = tmp_path / "decode"
    (split / "calls").mkdir(parents=True)
    mats = []
    for i, (rows, real, excl) in enumerate([(a, 60, [3, 5 + 64]), (b, 4, [])]):
        data = bf16(rows).tobytes()
        (split / "calls" / f"{i}.rows.bf16").write_bytes(data)
        mats.append({"index": i, "m": rows.shape[0], "k": k, "n": 64, "real_rows": real, "excluded_a": excl,
                     "rows": {"path": f"calls/{i}.rows.bf16", "bytes": len(data)}})
    (split / "manifest.json").write_text(json.dumps({"matmuls": mats}))

    def by_fractions(words):
        s_w, _, alpha_w, _ = C.s5_stats(words)
        alpha, s = Fraction(P.f32_of_bits(alpha_w)), Fraction(P.f32_of_bits(s_w))
        loud = sum(1 for w in words if abs(Fraction(P.f32_of_bits(w))) * alpha >= 2 * (448 - alpha * s))
        return alpha * s, loud

    want = []
    for rows, real, excl in [(a, 60, {3}), (b, 4, set())]:
        words = (bf16(rows).astype(np.uint32) << 16)
        want.append([by_fractions(words[i].tolist()) for i in range(real) if i not in excl])
    unit = FD.check_matmul((str(split), mats[0], list(range(64))))
    assert unit["m_cred"] == 59 and unit["sampled"] == 59 and unit["fallback"] == 1 and unit["exact"] >= 1
    assert unit["rows_out"] == sum(1 for a_s, _ in want[0] if a_s > 444) == 1 and unit["min_margin_row"] == 5
    assert unit["loud"] == sum(n for _, n in want[0]) and unit["max_out_per_tile"] == 1
    out = tmp_path / "fd.json"
    methods, get_context = [], FD.multiprocessing.get_context
    monkeypatch.setattr(FD.multiprocessing, "get_context", lambda method=None: methods.append(method) or get_context(method))
    assert FD.main([str(tmp_path), "--splits", "decode", "--jobs", "2", "--out", str(out)]) == 0
    assert methods == ["fork"]
    doc = json.loads(out.read_text())
    r = doc["splits"]["decode"]
    assert r["rows_checked"] == 63 and r["rows_out_of_domain"] == 1 and r["units_failing_live"] == 2
    assert r["min_slack"] == 4 * k - 64 * sum(n for _, n in want[1]) < 0
    assert r["hn_fail"] == 1 and not (doc["hfloor_holds"] or doc["hlive_holds"] or doc["hn_holds"])


def test_clopper_pearson_upper():
    assert SD.clopper_pearson_upper(0, 320, 0.05 / 16) == pytest.approx(1 - (0.05 / 16) ** (1 / 320))
    assert SD.clopper_pearson_upper(0, 320, 0.05 / 16) < 0.02 < math.log(20 * 16) / 64
    up = SD.clopper_pearson_upper(3, 100, 0.05)
    p_at = lambda p: sum(math.comb(100, i) * p ** i * (1 - p) ** (100 - i) for i in range(4))
    assert p_at(up) == pytest.approx(0.05, abs=1e-9) and 0.03 < up < 0.1
    assert SD.clopper_pearson_upper(5, 5, 0.05) == 1.0


def test_served_debit_input_bounds():
    """The per-input limits (lean's rulings on #923): the work-weighted rule of three for a census and for n draws at no
    failure, Bernstein no tighter than the observed rate, and no bound for a sampled input with failures."""
    row = {"method": "census", "failures": 0, "failures_weighted": 0, "W": 64 * 4096, "w_max": 4096}
    assert SD.input_bound(row, 0.05)[0] == pytest.approx(math.log(20) / 64)
    a = 4096 / (2048 * 4096)
    drawn = {"method": "sampled", "failures": 0, "W": 2048 * 4096, "w_max": 4096, "drawn": 64}
    assert SD.input_bound(drawn, 0.05)[0] == pytest.approx(math.log(20) * a / (1 - math.exp(-64 * a)))
    assert SD.input_bound(drawn, 0.05)[0] > math.log(20) / 64 > SD.input_bound(drawn | {"drawn": 10 ** 6}, 0.05)[0]
    assert SD.input_bound(drawn | {"drawn": 10 ** 6}, 0.05)[0] == pytest.approx(math.log(20) * a)
    assert SD.input_bound(drawn | {"failures": 1, "failures_weighted": 4096}, 0.05) == (1.0, "sampled with failures: measure this input in full")
    up = SD._bernstein_upper(3 * 4096, 4096, 64 * 4096, 0.05)
    assert 3 / 64 < up < 1
    assert SD.input_bound(row | {"failures": 3, "failures_weighted": 3 * 4096}, 0.05)[0] == up


def test_served_debit_rebound_recomputes_an_inputs_json(tmp_path):
    """`--rebound` rewrites a recorded inputs.json's bounds from its counts alone, at δ = 0.05/m, with no manifest."""
    rows = [{"pass": "decode", "run": "r", "manifest_sha256": "0", "matmul": i, "weight": i, "m": 1, "k": 4096, "n": 4096,
             "NT": nt, "W": nt * 4096, "w_max": 4096, "method": method, "drawn": drawn, "openings_failed": 0, "failures": 0,
             "failures_weighted": 0, "share_max": None, "reasons": [], "upper": 0.0, "bound_method": "old"}
            for i, (nt, method, drawn) in enumerate([(64, "census", 64), (2048, "sampled", 64)])]
    src = tmp_path / "inputs.json"
    src.write_text(json.dumps({"seed": 7, "census_max": 128, "draws": 64, "inputs": rows}))
    assert SD.main(["--rebound", str(src), "--out", str(tmp_path / "out")]) == 0
    doc = json.loads((tmp_path / "out" / "inputs.json").read_text())
    assert doc["m"] == 2 and doc["delta_per_input"] == 0.025 and doc["seed"] == 7
    assert [r["upper"] for r in doc["inputs"]] == [SD.input_bound(r, 0.025)[0] for r in rows]
    assert doc["cited_upper"] == doc["inputs"][1]["upper"] and doc["cited_input"]["matmul"] == 1


def _debit_codes(rng, rows: int, k: int, zero_first: bool) -> list[list[int]]:
    """E4M3 code rows whose words flag atoms: small codes with some large 32-code slices (the small ones' atoms then round
    away), and the first promotion group zero when `zero_first` (a later promotion into a +0 total)."""
    c = rng.integers(1, 0x10, (rows, k))
    for t in range(0, k, 32):
        if rng.random() < 0.3:
            c[:, t:t + 32] = rng.integers(0x50, 0x7E, (rows, 32))
    if zero_first:
        c[:, :128] = 0
    return [[int(x) for x in row] for row in c | rng.integers(0, 2, (rows, k)) << 7]


def test_the_batch_debit_is_the_scalar_debit_and_a_nan_code_is_the_scalars_rejection():
    FT, s = SD.FT, PPD.scheme_of(SCHEME)
    rng, seen = np.random.default_rng(3), set()
    for trial in range(8):
        a, b = _debit_codes(rng, 3, 384, trial % 2 == 0), _debit_codes(rng, 4, 384, trial % 2 == 0)
        want = PW.D.chain_debit(a, b, s.device)
        assert FT.chain_debit(a, b, s.device) == want
        seen.add(want[1] > 0)
        assert want[0] > 0
    assert seen == {True, False}
    a[0][5] = 0x7F
    with pytest.raises(Exception) as want:
        PW.D.chain_debit(a, b, s.device)
    with pytest.raises(type(want.value), match=str(want.value)):
        FT.chain_debit(a, b, s.device)


def test_the_batch_forms_encode_and_check_a_tile_as_the_scalar_ones():
    """`fast_tile.install`'s forms (forming, the peel factors, G, C̃ and U, the debit, native BLAKE3) give a tile's encoding,
    check and debit field for field as the scalar reference, and its undo puts every scalar function back."""
    FT, s = SD.FT, PPD.scheme_of(SCHEME)
    rng = np.random.default_rng(5)
    shape = Shape(8, 256, 64)
    mm, tile = Matmul(0, shape, 0), s.tiles(shape)[0]
    a = [x.tobytes() for x in bf16_words(rng, len(tile.rows), shape.k)]
    b = [x.tobytes() for x in bf16_words(rng, len(tile.cols), shape.k, 0.02)]
    ctx = Context(bytes(range(32)), mm, bytes(32), bytes(range(1, 33)))

    def run():
        ops = s.encode(ctx, s.noise(ctx), a, b, tile)
        return ops, s.checked(ctx, ops), PW.D.chain_debit(ops.a.codes, ops.b.codes, s.device)

    before = (C.form_v1, C.gram, C.peel_factors_a, C.peel_factors_b, C._basis, PW.D.chain_debit, type(s).checked, C.blake3)
    undo = FT.install(s)
    assert C.form_v1 is FT.form_v1 and PW.D.chain_debit is FT.chain_debit
    fast = run()
    undo()
    assert (C.form_v1, C.gram, C.peel_factors_a, C.peel_factors_b, C._basis, PW.D.chain_debit, type(s).checked, C.blake3) == before
    ref = run()
    for side in ("a", "b"):
        for field in ("codes", "alpha", "beta", "s", "rho", "ep", "x"):
            assert getattr(getattr(fast[0], side), field) == getattr(getattr(ref[0], side), field), (side, field)
    assert (fast[0].pa, fast[0].pb) == (ref[0].pa, ref[0].pb)
    assert (fast[1].c, fast[1].u) == (ref[1].c, ref[1].u) and fast[2] == ref[2]
    with pytest.raises(ValueError):
        FT.install(PPD.scheme_of("pearl-c-sm120-unpromoted-v1-h3"))


@pytest.mark.slow
def test_served_debits_batch_replay_and_resume_record_the_scalar_lines(honest_pass, tmp_path):
    """The batch replay writes the scalar replay's tile lines (all of them cross-checked here), and `--resume` on them
    measures nothing again and writes the same summary."""
    path, *_ = honest_pass
    lines = {}
    for name, extra in (("scalar", ["--scalar"]), ("batch", ["--cross-check", "1"])):
        out = tmp_path / name
        assert SD.main([str(path), *coins_args(path), "--tiles", "3", "--jobs", "1", "--out", str(out), *extra]) == 0
        lines[name] = [json.loads(x) for x in (out / f"{path.parent.name}.tiles.jsonl").read_text().splitlines()]
    drop = ("seconds", "replay", "cross_checked", "resumed_from")
    strip = lambda ts: sorted(({k: v for k, v in t.items() if k not in drop} for t in ts), key=lambda t: t["tile"])
    assert strip(lines["scalar"]) == strip(lines["batch"]) and all(t["cross_checked"] for t in lines["batch"])
    assert {t["replay"] for t in lines["batch"]} == {"batch"} and {t["replay"] for t in lines["scalar"]} == {"scalar"}
    out = tmp_path / "resumed"
    src = tmp_path / "batch" / f"{path.parent.name}.tiles.jsonl"
    assert SD.main([str(path), *coins_args(path), "--tiles", "3", "--jobs", "1", "--out", str(out), "--resume", str(src)]) == 0
    again = [json.loads(x) for x in (out / f"{path.parent.name}.tiles.jsonl").read_text().splitlines()]
    assert strip(again) == strip(lines["batch"]) and all(t["resumed_from"] == str(src) for t in again)
    summary = lambda d: {k: v for k, v in json.loads((d / "summary.json").read_text())["passes"][0].items() if k != "cpu_seconds"}
    assert summary(out) == summary(tmp_path / "batch")
    unstamped = tmp_path / "unstamped.jsonl"
    unstamped.write_text("".join(json.dumps({k: v for k, v in t.items() if k != "manifest_sha256"}) + "\n" for t in lines["batch"]))
    assert SD.resumed([str(unstamped)], lines["batch"][0]["manifest_sha256"]) == {}
    assert len(SD.resumed([f"{unstamped}={lines['batch'][0]['manifest_sha256']}"], lines["batch"][0]["manifest_sha256"])) == len(lines["batch"])


def test_served_debits_share_is_the_tile_rules_debit_over_its_rev1_credit():
    """served_debit's credit is the tile rule's cap over ρ, `creditDevRev1`'s share of the credited cells, with the first
    promotion add of every word out once: every share it records is `tileDebitRev1` over that credit, none with the add
    taken out a second time."""
    s = PPD.scheme_of(SCHEME)
    rng, shape, rows, cols = np.random.default_rng(11), Shape(64, 512, 48), 3, 2
    rec = PW.UnitRecord(Matmul(0, shape, 0), shape.m, shape.k, (), ())
    codes = lambda r: [[0] * shape.k] + [[int(c) for c in rng.integers(0, 0x70, shape.k) | (rng.integers(0, 2, shape.k) << 7)]
                                         for _ in range(r - 1)]
    debit, cap = PW.tile_cap(s, rec, codes(rows), codes(cols))
    credit = Fraction(s.credit_of(Shape(1, shape.k, shape.n)), shape.n) * rows * cols
    assert s.first_adds(Shape(1, shape.k, shape.n)) == s.prices.fadd * shape.n > 0 and debit > 0
    got = SD.tile_numbers(s, debit, cap)
    assert (got["debit_exact"], got["credit_exact"], got["ratio"]) == (str(debit), str(credit), float(debit / cap))
    assert {key: got[key] for key in got if key.startswith("share")} == {"share": float(debit / credit)}


# ---- the panel rows -----------------------------------------------------------------------------------------------

def e2e_result(control_accepted: bool = False, interleave: str = "rep", failure: str | None = None,
               graphed: bool = False) -> tuple[dict, dict]:
    reps = lambda mhz: {"prefill_8192": [{"clock": {"sm_mhz": mhz}}] * 3, "decode_b32": [{"clock": {"sm_mhz": mhz - 3}}] * 3}
    split = {key: {"shape": shape, "fp8_s": 1.0, "nohash_s": 1.5, "slowdown_vs_fp8": 2.5, "gemm_x_fp8": 1.2, "hashing_x_fp8": 1.0,
                   "else_x_fp8": 0.3, "slowdown_vs_bf16": 1.6} for key, shape in (("prefill_8192", "e2e-llama31-8b-vllm-m8192"),
                                                                                 ("decode_b32_step", "e2e-llama31-8b-vllm-m32"))}
    e2e = {"scheme": SCHEME, "run": RUN, "results": {"pearlc": {"reps": reps(2085)}, "fp8": {"reps": reps(2090)}},
           "summary": {"gates": {"arm": True}, "split": split, "interleave": interleave, "resident_failure": failure,
                       "verify_pass": {"prefill": {"same_commitment_as_timed": True}, "decode": {"same_commitment_as_timed": True}}}}
    if graphed:
        e2e["results"]["fp8-graphs"] = {"reps": reps(2080)}
        for x in split.values():
            x.update(fp8_graphs_s=0.5, slowdown_vs_fp8_graphs=5.0)
    t = lambda p: f"{RUN}:/workspace/pouw/mvp-e2e/passes/{RUN}/{p}/manifest.json sha256:{'ab' * 32}"
    verdicts = {p: {"accepted": p != "control" or control_accepted, "verifier_commit": "c7033d67", "transcript": t(p),
                    "line": f"{'ACCEPT' if p != 'control' or control_accepted else 'REJECT'} {SCHEME} pouw:{SCHEME}: ...; transcript {t(p)}"}
                for p in ("prefill", "decode", "control")}
    return e2e, verdicts


def test_panel_rows_are_what_panel_py_takes_for_a_measured_row():
    PR = load("pearl_c_vllm_panel_rows", KERNEL.parent / "pearl_c_vllm" / "panel_rows.py")
    rows = PR.rows(*e2e_result(), "panel.py")
    args = [dict(zip(r[3::2], r[4::2])) for r in (x[:-2] if "--decode-method" in x else x for x in rows)]
    assert [(a["--phase"], a["--shape"], a["--version"]) for a in args] == [
        ("prefill", "e2e-llama31-8b-vllm-m8192", "v1-h3"), ("decode", "e2e-llama31-8b-vllm-m32", "v1-h3")]
    for a in args:
        assert a["--kind"] == "measured" and a["--run"] == RUN and RUN in a["--transcript"] and "accept" in a["--verifier-accept"].lower()
        assert "reject" in a["--negative-control"].lower() and RUN in a["--negative-control"]
        assert a["--sm-clock-arm"] and a["--sm-clock-base"] and a["--slowdown"] == "2.5000" and a["--hash-free"] == "1.5000"
    assert rows[1][-2:] == ["--decode-method", "dependent-chain"] and "interleaved rep by rep" in args[0]["--description"]
    blocks = PR.rows(*e2e_result(interleave="blocks", failure="arm: OutOfMemoryError"), "panel.py")
    assert "ABAB blocks" in blocks[0][blocks[0].index("--description") + 1] and "OutOfMemoryError" in blocks[1][blocks[1].index("--description") + 1]
    with pytest.raises(SystemExit, match="negative control was not rejected"):
        PR.rows(*e2e_result(control_accepted=True), "panel.py")


def test_with_graphed_fp8_decode_heads_over_it_and_the_eager_row_is_beside_it():
    """The root's 21:33Z ruling: decode's headline row is over stock FP8 under CUDA graphs, the eager row beside it at the
    shape's -eager name, in the same attempt; both behind the same gate, and prefill keeps its one row."""
    PR = load("pearl_c_vllm_panel_rows", KERNEL.parent / "pearl_c_vllm" / "panel_rows.py")
    rows = PR.rows(*e2e_result(graphed=True), "panel.py")
    args = [dict(zip(r[3::2], r[4::2])) for r in (x[:-2] if "--decode-method" in x else x for x in rows)]
    assert [(a["--phase"], a["--shape"]) for a in args] == [
        ("prefill", "e2e-llama31-8b-vllm-m8192"), ("decode", "e2e-llama31-8b-vllm-m32"), ("decode", "e2e-llama31-8b-vllm-m32-eager")]
    head, eager = args[1], args[2]
    assert (head["--slowdown"], head["--hash-free"], eager["--slowdown"], eager["--hash-free"]) == ("5.0000", "3.0000", "2.5000", "1.5000")
    assert set(head["--sm-clock-base"].split(",")) == {"2077"} and set(eager["--sm-clock-base"].split(",")) == {"2087"}
    assert head["--description"].startswith("Headline, over stock FP8 as vLLM serves it") and rows[1][-2:] == rows[2][-2:]
    with pytest.raises(SystemExit, match="negative control was not rejected"):
        PR.rows(*e2e_result(control_accepted=True, graphed=True), "panel.py")


# ---- the e2e's schedule and its assembly --------------------------------------------------------------------------

def test_the_schedule_alternates_the_baseline_rep_by_rep_or_in_abab_blocks():
    E2E = load("pearl_c_vllm_e2e", KERNEL.parent / "pearl_c_vllm" / "e2e.py")
    (one,) = E2E.schedule(3, "rep")
    assert one == list(E2E.CYCLE) * 3 and one[0::2] == ["fp8"] * 9 and one.count("pearlc") == 3
    blocks = E2E.schedule(3, "blocks")
    assert [{E2E.GROUP[m] for m in b} for b in blocks] == [{"base"}, {"arm"}, {"base"}, {"arm"}]
    assert sum(b.count("fp8") for b in blocks) == 3 and sum(b.count("pearlc") for b in blocks) == 3


def test_with_a_statement_each_pearlc_rep_is_followed_by_a_pearlc_zk_rep_in_the_arm_engine():
    E2E = load("pearl_c_vllm_e2e", KERNEL.parent / "pearl_c_vllm" / "e2e.py")
    (one,) = E2E.schedule(2, "rep", zk=True)
    assert one.count("pearlc-zk") == 2 and all(one[i + 1] == "pearlc-zk" for i, m in enumerate(one) if m == "pearlc")
    assert [m for m in one if m != "pearlc-zk"] == list(E2E.CYCLE) * 2 and E2E.GROUP["pearlc-zk"] == "arm"
    blocks = E2E.schedule(3, "blocks", zk=True)
    assert sum(b.count("pearlc-zk") for b in blocks) == 3 and all(E2E.GROUP[m] == "arm" for m in blocks[1] + blocks[3])
    assert E2E.schedule(3, "rep", graphed=True) == E2E.schedule(3, "rep", graphed=True, zk=False)
    run = types.SimpleNamespace(statement=types.SimpleNamespace(enabled=False), entry="pouw:test")
    args = types.SimpleNamespace(schedule="serial")
    assert E2E.arm_of("pearlc-zk", run, args).spec.hashing and run.statement.enabled
    assert E2E.arm_of("pearlc", run, args).spec.hashing and not run.statement.enabled
    E2E.arm_of("pearlc-zk", run, args)
    assert not E2E.arm_of("pearlc-nohash", run, args).spec.hashing and not run.statement.enabled
    E2E.arm_of("pearlc-zk", run, args)
    E2E.arm_of("bf16", run, args)
    assert not run.statement.enabled


def test_a_sweep_points_decode_batch_and_prompt_name_its_key_and_its_graph_sizes():
    """Off the panel's decode (32 prompts of 64 tokens) the key names the batch and the prompt, so `panel_rows.py` never reads
    a sweep point as its decode row; the graphed baseline captures the point's batch."""
    E2E = load("pearl_c_vllm_e2e_sweep", KERNEL.parent / "pearl_c_vllm" / "e2e.py")
    E2E.configure(types.SimpleNamespace(batch=32, prompt=64))
    assert E2E.decode_key() == "decode_b32" and E2E.CAPTURE_SIZES == [1, 2, 4, 8, 16, 24, 32]
    assert E2E.SHAPES == {"prefill_8192": "e2e-llama31-8b-vllm-m8192", "decode_b32_step": "e2e-llama31-8b-vllm-m32"}
    E2E.configure(types.SimpleNamespace(batch=128, prompt=2048))
    assert E2E.decode_key() == "decode_b128_p2048" and E2E.CAPTURE_SIZES == [1, 2, 4, 8, 16, 24, 32, 128]
    assert E2E.SHAPES["decode_b128_p2048_step"] == "e2e-llama31-8b-vllm-m128-p2048" and len(E2E.SHAPES) == 2
    rec = lambda m, ph: {"mode": m, "phase": ph, "rep": 0, "s": 1.0, "gemm_s": 0.0, "step_s": 0.01, "gemm_step_s": 0.0}
    t = E2E.timing([rec(m, ph) for m in E2E.MODES for ph in ("prefill", "decode")])
    assert set(t["fp8"]["timing"]) == {"prefill_8192_s", "prefill_8192_gemm_s", "decode_b128_p2048_step_s", "decode_b128_p2048_step_gemm_s"}
    assert list(E2E.split(t)) == ["prefill_8192", "decode_b128_p2048_step"]
    ns = [6144, 4096, 28672, 4096] * 32                               # Llama-3.1-8B's linears
    E2E.configure(types.SimpleNamespace(batch=32, prompt=64))
    assert E2E.pass_capacity(128, ns) == (PPD.LEAF_CAPACITY, PPD.FLAG_ROWS)
    E2E.configure(types.SimpleNamespace(batch=256, prompt=64))       # 256 x (64 + 65) tokens, 128 calls each: over 4M rows
    assert E2E.pass_capacity(128, ns) == (PPD.LEAF_CAPACITY, 2 * PPD.FLAG_ROWS)
    E2E.configure(types.SimpleNamespace(batch=1, prompt=64))
    assert E2E.decode_key() == "decode_b1" and E2E.CAPTURE_SIZES == [1]


def test_the_noise_sweep_pairs_each_pearl_c_rep_with_the_stock_reps_around_it(tmp_path):
    NS = load("pearl_c_vllm_noise_sweep", KERNEL.parent / "pearl_c_vllm" / "noise_sweep.py")
    fp8 = [10.0, 12.0, 14.0, 20.0, 22.0, 24.0]                       # two cycles of three fp8 reps
    steps = {"fp8": fp8, "pearlc": [13.2, 25.3], "pearlc-nohash": [14.3, 25.3], "bf16": [9.0, 9.0]}
    reps = lambda m: [{"rep": i, "step_s": s / 1e3, "clock": {"sm_mhz": 2100}} for i, s in enumerate(steps[m])]
    e2e = {"run": "r1", "interleave": "rep", "summary": {"gates": {"arm": True}},
           "results": {m: {"reps": {"prefill_8192": [], "decode_b8": reps(m)}} for m in steps}}
    r = NS.run_ratios(e2e)["ratios"]["fp8"]
    assert r["pearlc"] == pytest.approx([1.2, 25.3 / 21]) and r["pearlc-nohash"] == pytest.approx([1.1, 25.3 / 23])
    assert r["hashing"] == pytest.approx([0.1, 25.3 / 21 - 25.3 / 23])
    for d in ("eager-b8.r1", "eager-b8.r2.t2"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "e2e.json").write_text(json.dumps(e2e))
    (tmp_path / "eager-b8.r3").mkdir()
    (tmp_path / "eager-b8.r3" / "e2e.json").write_text(json.dumps({"error": "base: preempted", "results": {}}))
    doc = NS.summarize([tmp_path])
    p = doc["points"]["eager-b8"]
    assert len(p["runs"]) == 2 and p["failed"][0]["error"] == "base: preempted" and set(p["rows"]) == {"fp8"}
    row = p["rows"]["fp8"]
    assert row["pearlc-nohash"]["median"] == pytest.approx((1.1 + 25.3 / 23) / 2) and row["flag"] is None
    assert (row["pearlc-nohash"]["n"], row["pearlc-nohash"]["runs"]) == (4, 2)
    assert "| eager-b8 | fp8 | 17.00 | 1.100 [1.100, 1.100] n=4/2 |" in NS.table(doc)
    steps["pearlc-nohash"] = [10.0, 20.0]                             # faster than stock FP8: flagged, not a win
    for d in ("eager-b8.r1", "eager-b8.r2.t2"):
        (tmp_path / d / "e2e.json").write_text(json.dumps({**e2e, "results": {m: {"reps": {"decode_b8": reps(m)}} for m in steps}}))
    assert "assessor" in NS.summarize([tmp_path])["points"]["eager-b8"]["rows"]["fp8"]["flag"]


def test_a_wikitext_window_starts_with_the_configs_bos_when_the_tokenizer_names_none(monkeypatch):
    """Qwen3-8B's tokenizer has no BOS (vLLM refused its windows' None id); its config's `bos_token_id` stands in."""
    E2E = load("pearl_c_vllm_e2e", KERNEL.parent / "pearl_c_vllm" / "e2e.py")

    class Tok:
        def __init__(self, bos):
            self.bos_token_id = bos

        def __call__(self, text, add_special_tokens):
            return {"input_ids": list(range(100, 100 + 2 * E2E.WINDOW))}

    configs = {"qwen": 151643, "llama": 128000, "none": None}
    transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda m: Tok(128000 if m == "llama" else None)),
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda m: types.SimpleNamespace(bos_token_id=configs[m])))
    table = types.SimpleNamespace(column=lambda c: types.SimpleNamespace(to_pylist=lambda: ["a", "b"]))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=lambda *a, **k: "x"))
    parquet = types.SimpleNamespace(read_table=lambda p: table)
    monkeypatch.setitem(sys.modules, "pyarrow", types.SimpleNamespace(parquet=parquet))
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    qwen = E2E.wikitext_windows("qwen", None)
    assert [w[0] for w in qwen] == [151643, 151643] and all(len(w) == E2E.WINDOW for w in qwen)
    assert [w[0] for w in E2E.wikitext_windows("llama", 1)] == [128000]
    with pytest.raises(ValueError, match="BOS"):
        E2E.wikitext_windows("none", None)


def test_the_controller_runs_each_rep_on_the_same_prompt_and_warms_each_engine_once():
    E2E = load("pearl_c_vllm_e2e", KERNEL.parent / "pearl_c_vllm" / "e2e.py")
    E2E.clocks = lambda: {"sm_mhz": 2090}
    p = {"warm": [1], "prefill": [[10], [11], [12]], "decode": [[2]] * 32}

    class Fake:
        def __init__(self, group, serial):
            self.group, self.serial, self.log = group, serial, []

        def ask(self, **cmd):
            self.log.append((cmd["op"], cmd["mode"], cmd["prompt"][0]))
            return {"s": 1.0, "gemm_s": 0.5, "step_s": 0.01, "gemm_step_s": 0.004, "commitment": {}}

    workers, records, warm = {"base": Fake("base", 1), "arm": Fake("arm", 2)}, [], set()
    E2E.measure(workers, E2E.schedule(3, "rep")[0], p, records, warm)
    pre = [(r["mode"], r["rep"], r["clock"]["sm_mhz"]) for r in records if r["phase"] == "prefill"]
    assert [m for m, _, _ in pre] == list(E2E.CYCLE) * 3 and [i for m, i, _ in pre if m == "pearlc"] == [0, 1, 2]
    assert [i for m, i, _ in pre if m == "fp8"] == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    timed = [x for x in workers["arm"].log if x[0] == "prefill" and x[2] != 1]
    assert [x[2] for x in timed if x[1] == "pearlc"] == [10, 11, 12]
    assert [x[0] for x in workers["base"].log if x[2] == 1] == ["prefill", "decode"] and len(warm) == 8
    t = E2E.timing(records)
    assert set(t) == set(E2E.MODES) and t["pearlc"]["timing"]["prefill_8192_s"] == 1.0


def test_the_smoke_starts_both_engines_and_records_a_failure_to_stay_resident(tmp_path, monkeypatch):
    E2E = load("pearl_c_vllm_e2e", KERNEL.parent / "pearl_c_vllm" / "e2e.py")
    monkeypatch.setattr(E2E, "memory_mib", lambda: 30000)

    class Fake:
        live: ClassVar[list] = []
        refuse_arm = False

        def __init__(self, ctx, args, group, coins):
            if group == "arm" and Fake.refuse_arm:
                raise RuntimeError("arm: OutOfMemoryError: CUDA out of memory")
            self.group, self.ready = group, {"gate": [{"ok": True}]} if group == "arm" else {}
            Fake.live.append(self)

        def ask(self, **cmd):
            return {"tokens": [[1, 2, 3]] if cmd["mode"] != "pearlc" else [[1, 2, 4]], "texts": [cmd["mode"]]}

        def close(self):
            Fake.live.remove(self)

    monkeypatch.setattr(E2E, "Worker", Fake)
    args = types.SimpleNamespace(out=tmp_path / "smoke.json")
    E2E.smoke(args)
    got = json.loads(args.out.read_text())
    assert got["resident"] and got["gate_ok"] and "error" not in got and not Fake.live
    assert [s["step"].split(":")[0] for s in got["steps"]][1::2] == ["fp8", "bf16"] and len(got["steps"]) == 5
    assert got["token_agreement_vs_bf16"] == {"fp8": 1.0, "pearlc": 2 / 3}
    Fake.refuse_arm = True
    E2E.smoke(args)
    got = json.loads(args.out.read_text())
    assert not got["resident"] and "OutOfMemoryError" in got["error"] and len(got["steps"]) == 2 and not Fake.live


# ---- the decode profile -------------------------------------------------------------------------------------------

def test_the_profile_maps_gpu_1s_kernels_to_their_categories(tmp_path):
    PF = load("pearl_c_vllm_profile", KERNEL.parent / "pearl_c_vllm" / "profile_decode.py")
    assert [PF.kernel_category(k) for k in ("gemm_pearlc", "gemm_pearlc_d", "form_s5", "stats_s5", "gemm_words", "s512_rows",
                                            "hm_leaves", "flash_fwd_kernel", "paged_attention")] == [
        "gemm", "gemm", "forming", "forming", "forming", "hashing", "hashing", "other", "other"]
    csvf = tmp_path / "k.csv"
    csvf.write_text('"Time (%)","Total Time (ns)","Instances","Avg (ns)","Name"\n'
                    '50,2000000,10,200000,"gemm_pearlc"\n30,1200000,40,30000,"s512_tile_leaves"\n20,800000,5,160000,"flash_fwd"\n')
    got = PF.nsys_kernels(csvf)
    assert got["gpu_ms_by_category"] == {"gemm": 2.0, "hashing": 1.2, "other": 0.8} and got["top_kernels"][0]["kernel"] == "gemm_pearlc"


def test_the_served_profile_counts_stock_fp8s_cutlass_3_kernel_as_gemm(monkeypatch):
    """vLLM's FP8 linear on sm_120 is CUTLASS 3's `device_kernel`, the stock arm's matmul, not `other`."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    SP = load("pearl_c_vllm_served_profile", KERNEL.parent / "pearl_c_vllm" / "served_profile.py")
    assert [SP.category(k) for k in ("device_kernel", "Kernel2", "gemm_pearlc_d", "words", "s512_tile_leaves", "flash_fwd_splitkv_kernel",
                                     "some_device_kernel_x")] == [
        "gemm", "gemm", "pearlc:gemm", "pearlc:forming", "pearlc:tile_hash", "attention", "other"]


def test_the_profile_times_each_calls_sections_and_steps_and_puts_the_module_back(device_run):
    from verity_vllm.protocol_options import pouw_pearl_c_device as PPD2
    PF = load("pearl_c_vllm_profile", KERNEL.parent / "pearl_c_vllm" / "profile_decode.py")
    make, _, weights, *_ = device_run
    before = (PPD2._steps, PPD2.call_keys, PPD2.PearlCDeviceRun.gemm)
    prof = PF.Profile()
    undo = PF.instrument(prof)
    try:
        run = make()
        prof.begin("pearlc-steady", True)
        run.begin()
        for m in (128, 16, 16):
            run.gemm(0, torch.randn(m, 1024).to(torch.bfloat16), weights[0], None)
            run.gemm(1, torch.randn(m, 1024).to(torch.bfloat16), weights[1], None)
        run.end()
        prof.end()
    finally:
        undo()
    assert (PPD2._steps, PPD2.call_keys, PPD2.PearlCDeviceRun.gemm) == before
    rep = prof.report()["pearlc-steady"]
    assert rep["prefill"]["forwards"] == 1 and rep["decode"]["forwards"] == 2 and rep["decode"]["pearlc_calls"] == 4
    host = rep["decode"]["host_ms"]
    assert {"call_keys", "_put", "screen", "launch gemm", "launch forming", "launch hashing", "gemm wrapper (all of it)"} <= set(host)
    assert set(rep["decode"]["split_ms"]) == {"gemm", "forming", "hashing", "other"}


# ---- step 1 of the decode plan: the host path, byte for byte the reference's ---------------------------------------

def test_native_keys_are_call_keys_byte_for_byte():
    import random
    tree_hash = HM96
    rng = random.Random(5)
    for _ in range(40):
        salt, root = rng.randbytes(32), rng.randbytes(32)
        keys = PPD._Keys(F, salt, tree_hash)
        assert keys.native is not None
        for index, m16 in ((0, 16), (1, 32), (127, 8192), (rng.randrange(1 << 20), 16 * rng.randrange(1, 600)), ((1 << 40) + 3, 48)):
            assert keys(root, index, m16) == PPD.call_keys(F, salt, root, index, m16, tree_hash)
    keys.native = None
    assert keys(root, 9, 64) == PPD.call_keys(F, salt, root, 9, 64, tree_hash)
    with pytest.raises(ValueError, match="don't frame tree hash 'sha256'"):
        PPD._Keys(F, salt, "sha256")


def test_the_upload_ring_writes_each_slot_as_given_through_its_wraparound():
    ring = PPD._Uploads(torch.device("cpu"), slots=4, width=16)
    dst = torch.zeros(16, dtype=torch.uint8)
    for i in range(11):
        data = bytes([i]) * (5 + i % 3)
        ring.put(dst, data)
        assert bytes(dst[:len(data)].tolist()) == data
    assert ring.i == 11 % 4


def test_the_host_path_commits_what_the_reference_path_commits(device_run, tmp_path):
    """The same calls through native keys and through `call_keys` itself (the pure path): the same keys uploaded for every
    call, the same launches, and the same retained manifest (records, roots, commitment)."""
    make, kernel, weights, *_ = device_run
    xs = [(w, torch.from_numpy(bf16_words(np.random.default_rng(20 + i), m, 1024).view(np.float32).copy()).to(torch.bfloat16))
          for i, (w, m) in enumerate(((0, 20), (1, 32), (0, 128), (1, 16)))]
    got = {}
    for name, native in (("native", True), ("reference", False)):
        run = make()
        if not native:
            run.keys.native = None
        kernel.dev.log.clear()
        run.begin(retain=str(tmp_path / name), run=RUN)
        keys = []
        for w, x in xs:
            run.gemm(w, x, weights[w], None)
            sh = run.shapes[(-(-x.shape[0] // 16) * 16, weights[w].shape[0], 1024)]
            keys.append(PPD._bytes(sh.keys[:168]))
        out = run.end()
        man = json.loads(Path(out["manifest"]).read_text())
        launches = [(n, i) for n, _, _, i in kernel.dev.log if not n.endswith("_smem")]   # run.py probes once a process
        got[name] = (keys, launches, {k: man[k] for k in ("records_root", "commitment", "salt")},
                     [{k: u[k] for k in ("root_a", "excluded_a", "real_rows")} for u in man["matmuls"]])
    assert got["native"] == got["reference"]
    assert got["native"][0][0] == PPD.call_keys(F, make().salt, make().roots[0], 0, 32, HM96)


def interpreted_triton(monkeypatch) -> None:
    """Triton's CPU interpreter for the kernels made from here, or a skip."""
    import os
    if "triton" in sys.modules and os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("triton is already imported without its interpreter")
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    pytest.importorskip("triton")


def edge_rows(k: int = 1024) -> tuple:
    """Twelve rows at the screen's edges (zero, dead counts either side of k/64, a large outlier) as FP32 words: (words, x,
    α, aux)."""
    words = bf16_words(np.random.default_rng(7), 12, k)
    words[1] = 0
    for r, count in ((2, 17), (3, 16), (4, 40)):
        words[r, [p for p in range(k) if p % 8][:count]] = np.float32(1e4).view(np.uint32) & 0xFFFF0000
    words[5, 3] = np.float32(200.0).view(np.uint32)
    return words, torch.from_numpy(words.view(np.float32).copy()), *stats(words)


def test_the_fused_screen_decides_the_rows_as_the_torch_screen_and_the_rules(monkeypatch):
    """The Triton kernel under Triton's CPU interpreter, on rows at the rules' edges: its excluded rows are `screen`'s and
    `pearl_c_work.row_passes`'s."""
    interpreted_triton(monkeypatch)
    words, x, alpha, aux = edge_rows()
    fused = torch.zeros((12, 2), dtype=torch.uint8)
    PPD.screen_into(fused, x, alpha, aux, fused=True)
    ref = torch.zeros((12, 2), dtype=torch.uint8)
    PPD.screen_into(ref, x, alpha, aux, fused=False)
    want = tuple(i for i, row in enumerate(words) if not PW.row_passes([int(v) for v in row]))
    assert PPD.excluded(fused[:, 0].bool(), fused[:, 1].bool(), x) == PPD.excluded(ref[:, 0].bool(), ref[:, 1].bool(), x) == want
    assert torch.equal(fused, ref)



@pytest.mark.parametrize("rows, tiles", [(12, 40), (5, 1), (0, 33)])
def test_a_graphs_scatter_screens_its_rows_into_the_pass_and_copies_its_leaves_and_root_at_the_slots_offsets(monkeypatch, rows,
                                                                                                           tiles):
    """A graph's last launch under Triton's CPU interpreter: the flags it writes at the slot's flag offset are `screen_into`'s
    for its first `rows` rows, its leaves and A root land at their offsets, and no other byte of the pass's buffers changes."""
    interpreted_triton(monkeypatch)
    from verity_vllm.protocol_options import pouw_pearl_c_screen as PS
    _, x, alpha, aux = edge_rows()
    flags = torch.zeros((12, 2), dtype=torch.uint8)
    PS.screen_into(flags, x, alpha, aux, fused=True)
    gen = torch.Generator().manual_seed(rows + tiles)
    leaves, root = (torch.randint(0, 256, (n,), dtype=torch.uint8, generator=gen) for n in (32 * tiles, 32))
    bufs = [torch.full((n,), 0xA5, dtype=torch.uint8) for n in (1 << 16, 4096, 32 * 64)]
    want = [t.clone() for t in bufs]
    want[0][3200:3200 + 32 * tiles], want[1][74:74 + 2 * rows], want[2][288:320] = leaves, flags[:rows].flatten(), root
    PS.scatter_into(x, alpha, aux, rows, bufs[1], leaves, bufs[0], root, bufs[2], torch.tensor([3200, 74, 288]))
    assert all(torch.equal(a, b) for a, b in zip(bufs, want, strict=True))

# ---- step 2 of the decode plan: the deferred schedule -------------------------------------------------------------

@pytest.mark.parametrize("schedule, main", [("deferred", 2), ("deferred-caller", 0)])
def test_the_deferred_schedule_hashes_tiles_on_the_side_stream_from_rotating_sets(device_run, schedule, main):
    """Under ``deferred`` the call's other steps run on the run's own main lane (the recording stand-in's handle 2), as measured;
    under ``deferred-caller`` on the caller's stream (0).  The tile hashing is on the side lane (1) either way."""
    make, kernel, weights, *_ = device_run
    run = make(schedule=schedule)
    kernel.dev.log.clear()
    for _ in range(3):
        run.gemm(0, torch.randn(32, 1024).to(torch.bfloat16), weights[0], None)
    log = [(n, st, p) for n, st, p, _ in kernel.dev.log if not n.endswith("_smem")]
    assert {st for n, st, _ in log if n == "s512_tile_leaves"} == {run.side.raw} == {1}
    assert {st for n, st, _ in log if n != "s512_tile_leaves"} == {main}
    sets = run.shapes[(32, 256, 1024)].sets
    ptr = lambda j: (sets[j]["c"].data_ptr(), sets[j]["u"].data_ptr())
    gemms = [p[2:4] for n, _, p in log if n == "gemm_pearlc_d"]
    leaves = [p[:2] for n, _, p in log if n == "s512_tile_leaves"]
    assert gemms == leaves == [ptr(0), ptr(1), ptr(0)] and ptr(0) != ptr(1)
    run.begin()
    run.poison()
    assert all(bool((t == 0xA5).all()) for st in sets for t in st.values())


@pytest.mark.parametrize("deferred", ["deferred", "deferred-caller"])
def test_the_deferred_schedule_commits_what_the_serial_one_commits(device_run, tmp_path, deferred):
    make, kernel, weights, *_ = device_run
    xs = [(w, torch.from_numpy(bf16_words(np.random.default_rng(40 + i), m, 1024).view(np.float32).copy()).to(torch.bfloat16))
          for i, (w, m) in enumerate(((0, 32), (1, 16), (0, 32), (0, 128), (1, 32), (0, 128)))]
    got = {}
    for name, schedule in (("serial", "serial"), ("deferred", deferred)):
        run = make(schedule=schedule)
        kernel.dev.log.clear()
        run.begin(retain=str(tmp_path / name), run=RUN)
        keys = []
        for w, x in xs:
            run.gemm(w, x, weights[w], None)
            keys.append(PPD._bytes(run.shapes[(-(-x.shape[0] // 16) * 16, weights[w].shape[0], 1024)].keys[:168]))
        out = run.end()
        man = json.loads(Path(out["manifest"]).read_text())
        launches = sorted((n, i) for n, _, _, i in kernel.dev.log if not n.endswith("_smem"))
        got[name] = (keys, launches, {k: man[k] for k in ("records_root", "commitment", "salt")},
                     [{k: u[k] for k in ("root_a", "excluded_a", "real_rows", "m", "n")} for u in man["matmuls"]])
    assert got["deferred"] == got["serial"]


# ---- the hashing format, from the one table --------------------------------------------------------------------------

def test_the_device_path_commits_with_its_schemes_hashing_format(device_run):
    """Every tree's hash is the scheme's format's (`verity.core.protocols.pouw.serving.HASH_FORMATS`), not a
    literal: -h3's hm96-sha512, keyed by the prover's salt key; and the path runs only its own schemes."""
    from verity.core.protocols.pouw.serving import HASH_FORMATS, SchemeId, hash_format

    make, *_ = device_run
    run = make()
    fmt = hash_format(run.scheme)
    assert fmt is HASH_FORMATS["h3"] and run.tree_hash == fmt.tree_hash == run.scheme.commitment_hash == HM96
    assert PPD._bytes(run.const[:32]) == SALT_KEY and set(HASH_FORMATS) == {"h0", "h3"}
    assert not hasattr(PPD, "TREE_HASH") and all(SchemeId.parse(n).name == n for n in PPD.SCHEMES)
    assert not hasattr(PPD, "tree_hash") and not hasattr(PPD, "variant")
    assert C.PearlC(forming="v1", hashing="h0").commitment_hash == "sha256"
    with pytest.raises(KeyError, match="this device path runs"):
        PPD.scheme_of("pearl-c-sm120-v1-h0")


# ---- the schedule as data ---------------------------------------------------------------------------------------------

def test_a_schedule_this_path_or_the_format_cannot_run_is_refused():
    """-h3's row seeds come from A's rows, not its root, so its format lets A's tree leave the main lane; this path moves only
    the tile hashing, which reads C~ and U, the buffers each shape rotates."""
    from verity.core.protocols.pouw.serving import Schedule
    s = PPD.scheme_of(SCHEME)
    with pytest.raises(ValueError, match="this path runs only tile_hash on a side lane, not a_tree$"):
        PPD.schedule_of(Schedule("early-a", frozenset({"a_tree", "tile_hash"}), ring=2), s)
    with pytest.raises(ValueError, match="this path runs only tile_hash on a side lane, not screen"):
        PPD.schedule_of(Schedule("screen-aside", frozenset({"tile_hash", "screen"}), ring=2), s)
    with pytest.raises(ValueError, match="a side lane needs ring >= 2"):
        PPD.schedule_of(Schedule("one-set", frozenset({"tile_hash"}), ring=1), s)
    assert PPD.rotated(PPD.schedule_of("deferred", s)) == ("c", "u") and PPD.rotated(PPD.SERIAL) == ()


def test_the_schedule_changes_between_passes_and_the_ring_grows_poisoned(device_run):
    make, kernel, weights, *_ = device_run
    run = make()
    run.gemm(0, torch.randn(32, 1024).to(torch.bfloat16), weights[0], None)
    sh = run.shapes[(32, 256, 1024)]
    assert len(sh.sets) == 1 and run.side is None and run.details["schedule"]["name"] == "serial"
    run.end()
    run.begin(retain=None, schedule="deferred")
    run.poison()
    kernel.dev.log.clear()
    run.gemm(0, torch.randn(32, 1024).to(torch.bfloat16), weights[0], None)
    assert len(sh.sets) == 2 and all(bool((t == 0xA5).all()) for t in sh.sets[1].values())
    assert run.details["schedule"] == {"name": "deferred", "side": ["tile_hash"], "ring": 2, "main": -(1 << 10), "side_lane": 0}
    assert {st for n, st, _, _ in kernel.dev.log if n == "s512_tile_leaves"} == {1} and run.end()["schedule"] == "deferred"


def test_the_gates_schedule_check_compares_with_serial_and_keeps_the_runs_schedule(device_run):
    make, _, weights, *_ = device_run
    E2E = load("pearl_c_vllm_e2e", KERNEL.parent / "pearl_c_vllm" / "e2e.py")
    run = make(schedule="deferred")
    got = E2E.schedule_check(run, weights)
    assert got["ok"] and got["schedule"] == "deferred" and set(got) >= {"serial", "deferred", "priorities"}
    assert run.schedule.name == "deferred" and got["priorities"]["main"] == -(1 << 10)


# ---- the quality run ----------------------------------------------------------------------------------------------

def test_the_unpromoted_scheme_runs_gpu_1s_unpromoted_kernels(device_run):
    _, kernel, weights, roots, salt = device_run
    s = PPD.scheme_of("pearl-c-sm120-unpromoted-v1-h3")
    assert s.device.name == "sm120-unpromoted" and "pearl-c-sm120-unpromoted-v1-h3" in PPD.SCHEMES
    run = PPD.PearlCDeviceRun(s, "pouw:pearl-c-sm120-unpromoted-v1-h3", kernel, weights, roots, *salt_coin(roots), capacity=1 << 12,
                              salt_key=SALT_KEY)
    kernel.dev.log.clear()
    run.gemm(0, torch.randn(32, 1024).to(torch.bfloat16), weights[0], None)
    names = [n for n, *_ in kernel.dev.log if not n.endswith("_smem")]
    assert "gemm_pearlc_d_u" in names and "words_u" in names and "gemm_pearlc_d" not in names


class _Logprob:
    def __init__(self, logprob, rank):
        self.logprob, self.rank = logprob, rank


def test_prompt_logprobs_sums_every_token_after_the_first_and_keeps_the_top_1():
    E2E = load("pearl_c_vllm_e2e", KERNEL.parent / "pearl_c_vllm" / "e2e.py")
    windows = [[1, 5, 6, 7], [1, 8, 9]]

    def timed(prompts, sp):
        outs = []
        for p in prompts:
            w = p["prompt_token_ids"]
            lps = [None] + [{t: _Logprob(-0.5, 1)} if t % 2 else {t: _Logprob(-2.0, 3), 42: _Logprob(-0.1, 1)} for t in w[1:]]
            outs.append(types.SimpleNamespace(prompt_logprobs=lps))
        return 0.0, 0.0, {}, outs
    got = E2E.prompt_logprobs(None, lambda **kw: kw, timed, windows)
    assert got["tokens"] == 5 and got["nll"] == pytest.approx(0.5 + 2.0 + 0.5 + 2.0 + 0.5) and got["top1"] == [5, 42, 7, 42, 9]
    parts = {"bf16": {"nll": 5.0, "tokens": 5, "top1": [5, 42, 7, 42, 9], "chat": {"tokens": [[1, 2, 3]]}},
             "pearlc-v1": {"scheme": "pearl-c-sm120-v1-h3", "nll": 6.0, "tokens": 5, "top1": [5, 42, 7, 1, 9], "chat": {"tokens": [[1, 2, 4]]}}}
    q = E2E.quality_summary(parts)
    assert q["bf16"]["perplexity"] == pytest.approx(np.exp(1.0)) and q["pearlc-v1"]["perplexity"] == pytest.approx(np.exp(1.2))
    assert q["pearlc-v1"]["top1_agreement_vs_bf16"] == 0.8 and q["pearlc-v1"]["greedy_token_agreement_vs_bf16"] == pytest.approx(2 / 3)


def test_the_quality_run_does_a_group_a_chunk_and_skips_the_ones_done(tmp_path, monkeypatch):
    E2E = load("pearl_c_vllm_e2e", KERNEL.parent / "pearl_c_vllm" / "e2e.py")
    monkeypatch.setattr(E2E, "wikitext_windows", lambda model, limit: [[1, 2, 3]])
    started = []

    class Fake:
        live: ClassVar[list] = []

        def __init__(self, ctx, args, group, coins):
            started.append((group, args.scheme, args.batch_tokens))
            self.ready = {"gate": [{"ok": True}], "committer": "x"} if group == "arm" else {}

        def ask(self, **cmd):
            if cmd["op"] == "ppl":
                return {"nll": 2.0 if cmd["mode"] == "pearlc" else 1.0, "tokens": 2, "top1": [2, 3]}
            return {"tokens": [[1, 2]], "texts": ["t"]}

        def close(self):
            pass

    monkeypatch.setattr(E2E, "Worker", Fake)
    args = types.SimpleNamespace(out=tmp_path / "quality.json", model="m", ppl_windows=None, chunk_min=0.0)
    assert [E2E.perplexity(args) for _ in range(3)] == [99, 99, 0]
    assert started == [("base", None, 2048), ("arm", "pearl-c-sm120-v1-h3", 2048), ("arm", "pearl-c-sm120-unpromoted-v1-h3", 2048)]
    doc = json.loads(args.out.read_text())
    assert set(doc["summary"]) == {"fp8", "bf16", "pearlc-v1", "pearlc-v2"} and all(doc["gates"].values())
    assert doc["summary"]["pearlc-v2"]["scheme"] == "pearl-c-sm120-unpromoted-v1-h3"
    assert E2E.perplexity(args) == 0 and len(started) == 3
    (tmp_path / "quality-v2.json").unlink()

    class Ungated(Fake):
        def __init__(self, *a):
            super().__init__(*a)
            self.ready["gate"] = [{"ok": False}]
    monkeypatch.setattr(E2E, "Worker", Ungated)
    with pytest.raises(SystemExit, match="gate failed"):
        E2E.perplexity(args)


# ---- one install path: the run behind the Stack ------------------------------------------------------------------------

def _stack_sites(weights):
    from verity_vllm.linear.api import LinearSite
    return [LinearSite(w, w, f"model.layers.{w}", t.shape[0], t.shape[1], "bf16") for w, t in weights.items()]


class _Layer:
    def __init__(self, weight):
        self.weight = weight


def test_the_run_behind_the_stack_commits_what_the_run_commits_alone(device_run, tmp_path):
    """e2e's install path (`Stack` + `PearlCExecutor`, the arm from `arm_of`) retains the same manifest and files as the run
    driven directly, byte for byte."""
    import dataclasses

    from verity_vllm.linear.api import Stack
    make, _, weights, *_ = device_run
    E2E = load("pearl_c_vllm_e2e", KERNEL.parent / "pearl_c_vllm" / "e2e.py")
    xs = [(w, torch.from_numpy(bf16_words(np.random.default_rng(60 + i), m, 1024).view(np.float32).copy()).to(torch.bfloat16))
          for i, (w, m) in enumerate(((0, 32), (1, 16), (0, 128), (1, 20)))]
    direct = make()
    direct.begin(retain=str(tmp_path / "direct"), run=RUN)
    for w, x in xs:
        direct.gemm(w, x, weights[w], None)
    want = direct.end()
    run = make()
    sites = _stack_sites(weights)
    stack = Stack(sites, PPD.PearlCExecutor(run))
    assert stack.refusals() == [] and stack.executor.schedules == {"serial", "deferred", "deferred-caller"}
    stack.activate(E2E.arm_of("pearlc", run, types.SimpleNamespace(schedule="serial")))
    with stack.pass_(dataclasses.replace(stack.active.spec, retain=str(tmp_path / "stack"), run_id=RUN)):
        for w, x in xs:
            stack.apply(sites[w], _Layer(weights[w]), x, None, lambda *a: pytest.fail("the stack called vLLM's own apply"))
    got = stack.passes[-1]
    assert {k: v for k, v in got.items() if k != "manifest"} == {k: v for k, v in want.items() if k != "manifest"}
    a, b = Path(want["manifest"]).parent, Path(got["manifest"]).parent
    assert (a / "manifest.json").read_text() == (b / "manifest.json").read_text()
    assert all((a / f).read_bytes() == (b / f).read_bytes() for f in ("leaves", "tree_t", "calls/0.rows.bf16", "weights/0.tree"))


def test_the_modes_are_arms_and_the_negative_control_is_a_pass_control(device_run):
    from verity_vllm.linear.api import Control, Stack
    make, kernel, weights, *_ = device_run
    E2E = load("pearl_c_vllm_e2e", KERNEL.parent / "pearl_c_vllm" / "e2e.py")
    run, args = make(), types.SimpleNamespace(schedule="deferred")
    arms = {m: E2E.arm_of(m, run, args) for m in ("bf16", "pearlc-nohash", "pearlc")}
    assert arms["bf16"].executor is None and arms["pearlc"].executor == arms["pearlc-nohash"].executor == run.entry
    assert (arms["pearlc"].spec.hashing, arms["pearlc-nohash"].spec.hashing) == (True, False)
    assert {a.spec.schedule for m, a in arms.items() if m != "bf16"} == {"deferred"}
    sites = _stack_sites(weights)
    stack = Stack(sites, PPD.PearlCExecutor(run))
    stack.activate(arms["bf16"])
    assert stack.apply(sites[0], _Layer(weights[0]), "x", None, lambda layer, x, bias: ("stock", x)) == ("stock", "x")
    stack.activate(arms["pearlc"])
    kernel.dev.log.clear()
    spec = arms["pearlc"].spec.__class__(controls=(Control("omit", E2E.CONTROL_OMITS),), schedule="deferred")
    with stack.pass_(spec):
        assert run.omit == E2E.CONTROL_OMITS
        stack.apply(sites[0], _Layer(weights[0]), torch.randn(32, 1024).to(torch.bfloat16), None, None)
    names = {n for n, *_ in kernel.dev.log}
    assert "s512_tile_leaves" not in names and "s512_rows" not in names and run.omit == frozenset()
    assert all(c <= set(PPD.CALL_STEPS) for c in E2E.CONTROLS.values())
