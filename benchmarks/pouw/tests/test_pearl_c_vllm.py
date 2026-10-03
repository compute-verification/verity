"""The vLLM device path for Pearl-C on sm_120 (`verity_vllm.protocol_options.pouw_pearl_c_device`) against GPU 1's pipeline
(``benchmarks/pouw/pearl_c_sm120``) on the CPU, with a recording stand-in for the device; and the pass verifier
(``benchmarks/pouw/pearl_c_vllm/verify_run.py``) on an honest transcript that the scheme's reference prover builds."""
from __future__ import annotations

import ctypes
import importlib.util
import json
import math
import shutil
import sys
import types
from fractions import Fraction
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.commitments import MerkleTree
from verity.commitments.blake3 import keyed
from verity.commitments.merkle import tree_levels
from verity_pouw.audit import (
    ROW_SCHEMA,
    TILE_SCHEMA,
    Epoch,
    Prover,
    Verifier,
    Workload,
    activation_binding,
    commit_rows,
    rows_domain,
    transcript_domain,
    weight_binding,
)
from verity_pouw.protocol import Context, Matmul, Shape
from verity_pouw.schemes import pearl_c as C
from verity_pouw.schemes import pearl_c_work as PW
from verity_vllm.protocol_options import pouw_pearl_c_device as PPD

KERNEL = Path(__file__).resolve().parents[1] / "pearl_c_sm120"
SCHEME = "pearl-c-sm120-v1-h1"
BEACON = bytes(range(32))
RUN = "r20260930-120000-test"


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
    epoch = Epoch.unverified(work, roots, BEACON)
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
    roots = PPD.weight_roots(kernel, weights, SCHEME)
    declared = Workload(tuple(Matmul(w, Shape(16, 1024, t.shape[0]), w) for w, t in weights.items()))
    salt = Epoch.unverified(declared, roots, BEACON).salt

    def make(hashing=True, schedule="serial"):
        return PPD.PearlCDeviceRun(PPD.scheme_of(SCHEME), f"pouw:{SCHEME}", kernel, weights, roots, salt, BEACON, hashing=hashing,
                                   capacity=1 << 12, schedule=schedule)
    return make, kernel, weights, roots, salt


PER_CALL = ["hash_node_keys", "hash_rows_w", "hash_tree_small", "seed_line_a", "lines", "stats_s5", "eprime", "form_s5", "words",
            "pa", "gemm_pearlc_d", "hash_msg_v", "hash_leaf_p"]


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
        assert by["hash_leaf_p"][0][1] == run.leaves.data_ptr() + 32 * before
        assert by["gemm_pearlc_d"][0][4] == y.data_ptr()
        sh = run.shapes[(32, weights[w].shape[0], 1024)]
        index = len(run.cur.matmuls) - 1
        assert PPD._bytes(sh.keys[:136]) == PPD.call_keys(F, salt, roots[w], index, 32, run.tree_hash)
        assert bool((sh.xa[m:] == 0).all()) and torch.equal(sh.xa[:m], x.float())
    assert [u.index for u in run.cur.matmuls] == [0, 1, 2] and run.cur.tiles == 4 + 2 + 4
    dev.log.clear()
    out = run.end()
    assert [n for n, *_ in dev.log][-2:] == ["hash_tree_leaves", "hash_tree_small"] and dev.log[-2][3] == (10,)
    assert out["units"] == 3 and out["tiles"] == 10 and len(out["transcript_root"]) == 64
    assert run.commits["transcript_root"] == out["transcript_root"]


def test_a_call_over_whole_128_row_tiles_runs_gpu_1s_fused_path(device_run):
    """At m a multiple of 128 the pipeline is fused: gemm_pearlc_b writes y and the message digests, and hash_msg isn't a step."""
    make, kernel, weights, *_ = device_run
    run = make()
    kernel.dev.log.clear()
    y = run.gemm(0, torch.randn(128, 1024).to(torch.bfloat16), weights[0], None)
    names = [n for n, *_ in kernel.dev.log if not n.endswith("_smem")]
    assert run.pipes[(0, 128, frozenset(), True, "serial", False)][0].fused and "hash_msg_v" not in names
    assert names[-2:] == ["gemm_pearlc_b", "hash_leaf_p"] and {n: p for n, _, p, _ in kernel.dev.log}["gemm_pearlc_b"][4] == y.data_ptr()


def test_nohash_forms_and_multiplies_without_hashing(device_run):
    make, kernel, weights, *_ = device_run
    run = make(hashing=False)
    kernel.dev.log.clear()
    run.gemm(0, torch.randn(16, 1024).to(torch.bfloat16), weights[0], None)
    assert [n for n, *_ in kernel.dev.log if not n.endswith("_smem")] == ["stats_s5", "eprime", "form_s5", "words", "pa",
                                                                          "gemm_pearlc_d"]
    assert run.end() == {"units": 1, "tiles": 0, "hashing": False, "schedule": "serial"}


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
    assert "hash_leaf_p" not in [n for n, *_ in kernel.dev.log] and "hash_msg_v" in [n for n, *_ in kernel.dev.log]


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
    """The constants the pipeline reads (the message and leaf keys, then frame-b3s's templates) and a run's tree hash are the
    scheme's: its digest keys are the ones its own digests are keyed by.  A scheme without keys (h0) is refused."""
    s = PPD.scheme_of(scheme)
    kernel = PPD.Kernel(R, F, RecordingDev(), "0" * 64)
    const = PPD._bytes(PPD._constants(kernel, s, torch.device("cpu")))
    msg = bytes(range(256))
    assert keyed(const[:32], msg) == C.message_digest(msg, s.hashing) and const[32:64] == s.digest_keys["leaf"]
    assert const[64:] == R.b3s_template(ROW_SCHEMA) + R.b3s_template(TILE_SCHEMA)
    rng = np.random.default_rng(1)
    weights = {0: torch.from_numpy(bf16_words(rng, 128, 1024).view(np.float32).copy()).to(torch.bfloat16)}
    roots = PPD.weight_roots(kernel, weights, scheme)
    run = PPD.PearlCDeviceRun(s, f"pouw:{scheme}", kernel, weights, roots, bytes(32), BEACON, capacity=16)
    assert run.tree_hash == s.commitment_hash == C.TREE_HASH[s.hashing] and PPD._bytes(run.const) == const
    with pytest.raises(ValueError, match="no keys"):
        PPD._constants(kernel, C.PearlC(forming="v1", device=s.device), torch.device("cpu"))


def test_the_pipelines_h2_templates_bind_the_audits_schemas():
    assert (R.ROW_SCHEMA, R.TILE_SCHEMA) == (ROW_SCHEMA, TILE_SCHEMA)
    for schema in (ROW_SCHEMA, TILE_SCHEMA):
        t, n = R.b3s_template(schema), R.b3s_template_len(schema)
        assert len(t) == 64 and t[0] == 5 and t[29:31] == len(schema).to_bytes(2, "little") and t[31:n] == schema.encode()
        assert not any(t[1:29]) and not any(t[n:])


@pytest.mark.slow
@pytest.mark.skipif(not (shutil.which("g++") or shutil.which("c++")), reason="simt_twin needs a C++ compiler")
@pytest.mark.parametrize("scheme,rows_form", [(s, "w") for s in sorted(PPD.SCHEMES)] + [("pearl-c-sm120-v1-h2", "s")])
def test_every_tree_the_device_path_commits_is_its_hashing_formats(scheme, rows_form):
    """B's roots, each call's A root over its m₁₆ rows (filler zero), and the pass's tile tree over the leaves the calls wrote,
    their kernels run by `simt_twin`: each is `MerkleTree`'s root in the scheme's tree hash (frame-b3 or frame-b3s)."""
    hashing = PPD.scheme_of(scheme).hashing
    h = PPD.scheme_of(scheme).commitment_hash
    rng = np.random.default_rng(5)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16) for w, n in ((0, 256), (1, 128))}
    rows = lambda x: [r.tobytes() for r in x.float().numpy()]  # noqa: E731
    kernel = PPD.Kernel(R, F, TwinDev(), "0" * 64)
    roots = PPD.weight_roots(kernel, weights, scheme)
    assert roots == {w: commit_rows(weight_binding(w), -2, rows(t), h).commitment.root for w, t in weights.items()}
    declared = Workload(tuple(Matmul(w, Shape(16, 1024, t.shape[0]), w) for w, t in weights.items()))
    salt = Epoch.unverified(declared, roots, BEACON).salt
    run = PPD.PearlCDeviceRun(PPD.scheme_of(scheme), f"pouw:{scheme}", kernel, weights, roots, salt, BEACON, capacity=1 << 12,
                              rows_form=rows_form)
    calls = [(0, torch.from_numpy(bf16_words(rng, 20, 1024).view(np.float32).copy()).to(torch.bfloat16)),
             (1, torch.from_numpy(bf16_words(rng, 32, 1024).view(np.float32).copy()).to(torch.bfloat16))]
    for w, x in calls:
        kernel.dev.log.clear()
        run.gemm(w, x, weights[w], None)
        names = [n for n, *_ in kernel.dev.log if not n.endswith("_smem")]
        want_rows = ["hash_rows_b3s" + ("_stats" if rows_form == "s" else "")] if hashing == "h2" else ["hash_rows_w"]
        assert names[:len(want_rows) + 2] == ["hash_node_keys", *want_rows, "hash_tree_small"], names
        assert ("stats_s5" in names) == (rows_form == "w")
        if rows_form == "s":
            a = torch.zeros((32, 1024), dtype=torch.float32)
            a[:x.shape[0]] = x.float()
            alpha, aux = stats(a.numpy().view(np.uint32))
            sh = run.shapes[(32, weights[w].shape[0], 1024)]
            assert torch.equal(sh.b["alpha_a"][:4 * 32], alpha) and torch.equal(sh.b["aux_a"][:8 * 32], aux.reshape(-1))
    for index, (w, x) in enumerate(calls):
        a = torch.zeros((32, 1024), dtype=torch.float32)
        a[:x.shape[0]] = x.float()
        want = commit_rows(activation_binding(salt, index), index, rows(a), h).commitment.root
        assert PPD._bytes(run.roots_buf[32 * index:32 * index + 32]) == want, index
    count = run.cur.tiles
    assert count == 4 + 2
    leaves = rng.integers(0, 256, 32 * count, dtype=np.uint8).tobytes()      # no GEMM ran: every tile's leaf is the same
    run.leaves[:32 * count] = torch.frombuffer(bytearray(leaves), dtype=torch.uint8)
    kernel.dev.log.clear()
    out = run.end()
    assert [n for n, *_ in kernel.dev.log][-2] == ("hash_leaves_b3s" if hashing == "h2" else "hash_tree_leaves")
    dt = transcript_domain(Epoch(Workload(tuple(run.cur.matmuls)), roots, salt), count, h)
    tree = MerkleTree(dt, {i: leaves[32 * i:32 * i + 32] for i in range(count)}, lambda _: TILE_SCHEMA)
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
    the scheme's reference prover and written in the device run's layout, for each promoted device scheme (-h1 and -h2 trees;
    the unpromoted record's debit isn't stated, so pearl_c_work can't audit it)."""
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
    prover = Prover(s, {w: rows(x) for w, x in wrows.items()})
    epoch = Epoch.unverified(work, prover.weight_roots, BEACON)
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
                                              laid_out(d, [d.leaf(j, j, ROW_SCHEMA, r) for j, r in enumerate(rows(x))]))}
    matmuls = []
    for u, rec, root in zip(work.matmuls, records, tr.activation_roots):
        x = rows(acts[u.index])
        d = rows_domain(activation_binding(epoch.salt, u.index), u.index, u.shape.m, s.commitment_hash)
        matmuls.append({"index": u.index, "m": u.shape.m, "k": k, "n": u.shape.n, "weight": u.weight, "real_rows": rec.real_rows,
                        "k_kept": rec.k_kept, "excluded_a": list(rec.excluded_a), "root_a": root.hex(),
                        "rows": PPD._write(tmp_path / "calls" / f"{u.index}.rows.bf16", bf16_file(x), digest=False),
                        "tree": PPD._write(tmp_path / "calls" / f"{u.index}.tree",
                                           laid_out(d, [d.leaf(i, i, ROW_SCHEMA, r) for i, r in enumerate(x)]))})
    count = tr.commitment.count
    leaves = [prover.digest(t) for t in range(count)]
    dt = transcript_domain(epoch, count, s.commitment_hash)
    head = {"schema": PPD.MANIFEST_SCHEMA, "run": RUN, "entry": f"pouw:{scheme}", "scheme": scheme,
            "instance": {"k_min": 32, "tile": [16, 16]},
            "beacon": BEACON.hex(), "salt": epoch.salt.hex(),
            "records_root": PW.records_root(epoch.salt, epoch.weights_root, splits, records).hex(),
            "commitment": {"root": tr.commitment.root.hex(), "count": count},
            "leaves": PPD._write(tmp_path / "leaves", b"".join(leaves)),
            "tree_t": PPD._write(tmp_path / "tree_t", laid_out(dt, [dt.leaf(i, i, TILE_SCHEMA, v) for i, v in enumerate(leaves)]))}
    path = PPD.write_manifest(tmp_path, head, weights, matmuls)
    return path, s, epoch, prover, tr, splits, records


@pytest.mark.slow
def test_verify_run_accepts_an_honest_pass_as_the_work_law_does(honest_pass):
    path, s, epoch, prover, tr, splits, records = honest_pass
    source = b"verifier" * 4
    got = VR.verify(path, 3, 1, "test", source)
    want = PW.audit_replay(s, epoch, prover, tr, splits, records, 0, PW.Beacon(1, source), 3)
    assert got["accepted"] and want.audit.accepted, got.get("line") or got.get("error")
    assert tuple(got["tiles"]) == want.audit.selected and got["exclusion_tiles"] == 1
    assert got["line"].startswith(f"ACCEPT {s.name} pouw:{s.name}: pearl_c_work's tiles (3 drawn, 1 for excluded rows)")
    assert got["transcript"] == f"{RUN}:{path} sha256:{got['transcript_sha256']}" and got["transcript"] in got["line"]
    assert VR.main([str(path), "--jobs", "1", "--commit", "0" * 7]) == 0
    assert VR.main([str(path), "--jobs", "1", "--commit", "0" * 7, "--expect-reject"]) == 1


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
    r = VR.verify(path, 3, 1, "test", b"verifier" * 4)
    assert not r["accepted"] and "weight opening" in r["line"] and r["line"].startswith("REJECT") and RUN in r["line"]
    assert VR.main([str(path), "--jobs", "1", "--commit", "0" * 7, "--expect-reject"]) == 0
    rows.write_bytes(good)
    man["matmuls"][0]["excluded_a"] = []
    path.write_text(json.dumps(man))
    r = VR.verify(path, 3, 1, "test", b"verifier" * 4)
    assert not r["accepted"] and "records are not the ones" in r["line"]


@pytest.mark.slow
def test_served_debit_records_the_tile_rules_own_debit_and_cap(honest_pass, tmp_path):
    path, s, epoch, prover, tr, splits, records = honest_pass
    out = tmp_path / "out"
    assert SD.main([str(path), "--tiles", "3", "--jobs", "1", "--out", str(out)]) == 0
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
        a, b = [o.value for o in op.a], [o.value for o in op.b]
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
    assert SD.main([str(path), "--per-input", "1", "--census-max", "4", "--draws", "3", "--jobs", "1", "--out", str(out)]) == 0
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
        ("prefill", "e2e-llama31-8b-vllm-m8192", "v1-h1"), ("decode", "e2e-llama31-8b-vllm-m32", "v1-h1")]
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

        def __init__(self, ctx, args, group, beacon):
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
    assert [PF.kernel_category(k) for k in ("gemm_pearlc_b", "gemm_pearlc_cu", "form_s5", "stats_s5", "gemm_words", "hash_rows_w",
                                            "seed_line_a", "flash_fwd_kernel", "paged_attention")] == [
        "gemm", "gemm", "forming", "forming", "forming", "hashing", "hashing", "other", "other"]
    csvf = tmp_path / "k.csv"
    csvf.write_text('"Time (%)","Total Time (ns)","Instances","Avg (ns)","Name"\n'
                    '50,2000000,10,200000,"gemm_pearlc_b"\n30,1200000,40,30000,"hash_leaf_p"\n20,800000,5,160000,"flash_fwd"\n')
    got = PF.nsys_kernels(csvf)
    assert got["gpu_ms_by_category"] == {"gemm": 2.0, "hashing": 1.2, "other": 0.8} and got["top_kernels"][0]["kernel"] == "gemm_pearlc_b"


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

@pytest.mark.parametrize("tree_hash", ["blake3", "blake3-s256"])
def test_native_keys_are_call_keys_byte_for_byte(tree_hash):
    import random
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
            keys.append(PPD._bytes(sh.keys[:136]))
        out = run.end()
        man = json.loads(Path(out["manifest"]).read_text())
        launches = [(n, i) for n, _, _, i in kernel.dev.log if not n.endswith("_smem")]   # run.py probes once a process
        got[name] = (keys, launches, {k: man[k] for k in ("records_root", "commitment", "salt")},
                     [{k: u[k] for k in ("root_a", "excluded_a", "real_rows")} for u in man["matmuls"]])
    assert got["native"] == got["reference"]
    assert got["native"][0][0] == PPD.call_keys(F, make().salt, make().roots[0], 0, 32, "blake3")


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
    assert {st for n, st, _ in log if n in ("hash_msg_v", "hash_leaf_p")} == {run.side.raw} == {1}
    assert {st for n, st, _ in log if n not in ("hash_msg_v", "hash_leaf_p")} == {main}
    sets = run.shapes[(32, 256, 1024)].sets
    ptr = lambda j: (sets[j]["c"].data_ptr(), sets[j]["u"].data_ptr())
    gemms = [p[2:4] for n, _, p in log if n == "gemm_pearlc_d"]
    msgs = [p[:2] for n, _, p in log if n == "hash_msg_v"]
    assert gemms == msgs == [ptr(0), ptr(1), ptr(0)] and ptr(0) != ptr(1)
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
            keys.append(PPD._bytes(run.shapes[(-(-x.shape[0] // 16) * 16, weights[w].shape[0], 1024)].keys[:136]))
        out = run.end()
        man = json.loads(Path(out["manifest"]).read_text())
        launches = sorted((n, i) for n, _, _, i in kernel.dev.log if not n.endswith("_smem"))
        got[name] = (keys, launches, {k: man[k] for k in ("records_root", "commitment", "salt")},
                     [{k: u[k] for k in ("root_a", "excluded_a", "real_rows", "m", "n")} for u in man["matmuls"]])
    assert got["deferred"] == got["serial"]


# ---- the hashing format, from the one table --------------------------------------------------------------------------

def test_the_device_path_commits_with_its_schemes_hashing_format(device_run):
    """The message and leaf keys the kernels read and every tree's hash are the scheme's format's
    (`verity_pouw.serving.HASH_FORMATS`), not literals, and -h1's bytes are what they were."""
    from verity.commitments.blake3 import blake3
    from verity_pouw.serving import HASH_FORMATS, SchemeId, hash_format

    make, *_ = device_run
    run = make()
    fmt = hash_format(run.scheme)
    assert fmt is HASH_FORMATS["h1"] and run.tree_hash == fmt.tree_hash == run.scheme.commitment_hash == "blake3"
    assert PPD._bytes(run.const[:64]) == fmt.message_key() + fmt.leaf_key()
    assert C.H1_KEYS == {"message": blake3(b"pearl-c/h1/message"), "leaf": blake3(b"pearl-c/h1/leaf")}
    assert not hasattr(PPD, "TREE_HASH") and all(SchemeId.parse(n).name == n for n in PPD.SCHEMES)
    assert not hasattr(PPD, "tree_hash") and not hasattr(PPD, "variant")
    assert C.PearlC(forming="v1", hashing="h0").commitment_hash == "sha256"
    with pytest.raises(KeyError, match="this device path runs"):
        PPD.scheme_of("pearl-c-sm120-v1-h3")


# ---- the schedule as data ---------------------------------------------------------------------------------------------

def test_a_schedule_this_path_or_the_format_cannot_run_is_refused():
    """-h1's forming reads seed_A from A's root, so A's tree stays on the main lane; and this path moves only the tile hashing."""
    from verity_pouw.serving import Schedule
    s = PPD.scheme_of(SCHEME)
    with pytest.raises(ValueError, match="a_tree can't leave the main lane under h1"):
        PPD.schedule_of(Schedule("early-a", frozenset({"a_tree", "tile_hash"}), ring=2), s)
    with pytest.raises(ValueError, match="this path runs only tile_hash on a side lane, not screen"):
        PPD.schedule_of(Schedule("screen-aside", frozenset({"tile_hash", "screen"}), ring=2), s)
    assert PPD.rotated(PPD.schedule_of("deferred", s)) == ("c", "digests", "u") and PPD.rotated(PPD.SERIAL) == ()


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
    assert {st for n, st, _, _ in kernel.dev.log if n == "hash_leaf_p"} == {1} and run.end()["schedule"] == "deferred"


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
    s = PPD.scheme_of("pearl-c-sm120-unpromoted-v1-h1")
    assert s.device.name == "sm120-unpromoted" and "pearl-c-sm120-unpromoted-v1-h1" in PPD.SCHEMES
    run = PPD.PearlCDeviceRun(s, "pouw:pearl-c-sm120-unpromoted-v1-h1", kernel, weights, roots, salt, BEACON, capacity=1 << 12)
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
             "pearlc-v1": {"scheme": "pearl-c-sm120-v1-h1", "nll": 6.0, "tokens": 5, "top1": [5, 42, 7, 1, 9], "chat": {"tokens": [[1, 2, 4]]}}}
    q = E2E.quality_summary(parts)
    assert q["bf16"]["perplexity"] == pytest.approx(np.exp(1.0)) and q["pearlc-v1"]["perplexity"] == pytest.approx(np.exp(1.2))
    assert q["pearlc-v1"]["top1_agreement_vs_bf16"] == 0.8 and q["pearlc-v1"]["greedy_token_agreement_vs_bf16"] == pytest.approx(2 / 3)


def test_the_quality_run_does_a_group_a_chunk_and_skips_the_ones_done(tmp_path, monkeypatch):
    E2E = load("pearl_c_vllm_e2e", KERNEL.parent / "pearl_c_vllm" / "e2e.py")
    monkeypatch.setattr(E2E, "wikitext_windows", lambda model, limit: [[1, 2, 3]])
    started = []

    class Fake:
        live: ClassVar[list] = []

        def __init__(self, ctx, args, group, beacon):
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
    assert started == [("base", None, 2048), ("arm", "pearl-c-sm120-v1-h1", 2048), ("arm", "pearl-c-sm120-unpromoted-v1-h1", 2048)]
    doc = json.loads(args.out.read_text())
    assert set(doc["summary"]) == {"fp8", "bf16", "pearlc-v1", "pearlc-v2"} and all(doc["gates"].values())
    assert doc["summary"]["pearlc-v2"]["scheme"] == "pearl-c-sm120-unpromoted-v1-h1"
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
    assert "hash_leaf_p" not in names and "hash_rows_w" not in names and run.omit == frozenset()
