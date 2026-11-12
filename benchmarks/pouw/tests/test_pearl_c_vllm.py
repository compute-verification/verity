"""The vLLM device path for Pearl-C on sm_120 (`verity_vllm.protocols.pouw.pearl_c`) against GPU 1's pipeline
(``benchmarks/pouw/pearl_c_sm120``) on the CPU, with a recording stand-in for the device; and the pass verifier
(``benchmarks/pouw/pearl_c_vllm/verify_run.py``) on an honest transcript that the scheme's reference prover builds."""
from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import math
import random
import shlex
import shutil
import socket
import sys
import threading
import types
from fractions import Fraction
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from verity.protocols.pouw.prover.schemes import pearl_c_u as PCU
from verity.protocols.spot.verifier.one_stage import registration as REG
from verity_vllm.protocols.pouw.pearl_c import host as PH
from verity_vllm.protocols.pouw.pearl_c import served_y as SY
from verity_vllm.protocols.pouw.pearl_c import window as PWR

from verity.protocols.pouw.verifier import core_check as CC

torch = pytest.importorskip("torch")

from verity.commitments import MerkleTree
from verity.commitments.merkle import tree_levels
from verity.protocols.pouw.verifier.audit import (
    HM96,
    HM96_TRANSCRIPT,
    HM96_WEIGHTS,
    ROW_SCHEMA,
    TILE_SCHEMA,
    Epoch,
    Workload,
    activation_binding,
    commit_rows,
    hm96_kernel_constants,
    hm96_salt,
    opened_value,
    rows_domain,
    salt_of,
    transcript_domain,
    weight_binding,
    weights_root,
)
from verity.protocols.pouw.verifier import salt as PS
from verity.protocols.pouw.verifier import window as Wd
from verity.protocols.pouw.verifier.coins import Challenger, CoinCommitment
from verity.protocols.pouw.verifier.credit import passes_distinct
from verity.protocols.pouw.verifier.protocol import Context, DomainError, Matmul, Shape
from verity.protocols.pouw.prover.schemes import pearl_c as C
from verity.protocols.pouw.prover.schemes import pearl_c_work as PW
from verity.protocols.pouw.prover.schemes import pearl_kw as P
from verity.protocols.pouw.verifier.serving import SchemeId
from verity.protocols.spot.device import challenger as Ch
from verity_vllm.protocols.pouw import pearl_c as PPD

# the work sampling design and the clear audit are a reference tool's (verity/protocols/pouw/clear), imported by name
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity" / "protocols" / "pouw" / "clear"))
import clear_work as CW  # noqa: E402
from clear_audit import Prover, Verifier  # noqa: E402

KERNEL = Path(__file__).resolve().parents[1] / "pearl_c_sm120"
SCHEME = "pearl-c-sm120-v1-h3"
SEED = bytes(range(32))
SALT_KEY = bytes(range(100, 132))
RUN = "r20260930-120000-test"


def salt_coin(roots: dict[int, bytes], epoch: int = 0, entropy: int = 7):
    """An execution phase's coin commitment and its salt coin, opened after `roots`' weights root, from a seeded challenger."""
    ch = Challenger(epoch, random.Random(entropy).randbytes)
    return ch.commitment, ch.open("salt", weights_root(roots))


def published(path: Path) -> CoinCommitment:
    """The execution phase's coin commitment as its challenger published it beside the passes (e2e.py's ``coins.json``)."""
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
#: the device path's schemes whose device record is a variant of KERNEL's pipeline (pearl_c_sm90 runs h100's)
KERNEL_SCHEMES = sorted(s for s in PPD.SCHEMES if SchemeId.parse(s).device in R.VARIANT_G)


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
    want = tuple(i for i, row in enumerate(words) if not CW.row_passes([int(v) for v in row]))
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
    assert PPD.excluded(fails, unsure, x) == (() if CW.row_passes([int(v) for v in words[0]]) else (0,))


# ---- the per-call keys --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", KERNEL_SCHEMES)
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
    """What `run.Pipeline` and the run touch of `run.Dev`, recording each launch (name, stream, pointers, ints).  Its attribute
    reads (`run.pd_setup`'s) get the RTX PRO 6000's opt-in shared memory and no static shared memory."""

    sms, dev = 188, 0

    def __init__(self):
        self.stream, self.log, self.n = None, [], 0

    def launch(self, name, grid, block, smem=0, p=(), i=(), stream=None):
        self.log.append((name, self.stream if stream is None else stream, tuple(p), tuple(i)))

    def ck(self, *_):
        pass

    def get(self, f, *_):
        return {"cuDeviceGetAttribute": 101376, "cuFuncGetAttribute": 0}[f]

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


def zeroed_y(monkeypatch) -> None:
    """The recording stand-in runs no GEMM, so a call's y is whatever ``torch.empty`` left: zeroed, two runs of the same
    calls retain the same y, and the same y roots."""
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: empty(*a, **kw).zero_())


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

    def make(hashing=True, schedule="serial", flag_rows=PPD.FLAG_ROWS, **kw):
        return PPD.PearlCDeviceRun(PPD.scheme_of(SCHEME), f"pouw:{SCHEME}", kernel, weights, roots, *coins, hashing=hashing,
                                   capacity=1 << 12, schedule=schedule, flag_rows=flag_rows, salt_key=SALT_KEY, **kw)
    return make, kernel, weights, roots, salt


#: a decode call's launches (m₁₆ = 32 of 64 rows: run.pd_variant's 851): A's commitment and E_A's lines in one, decode's split
#: stats and words kernels, the tile leaves
PER_CALL = ["pd_commit_851", "stats_r", "eprime", "form_s5", "words_k", "pa", "gemm_pearlc_d", "pd_tiles_851"]


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
        assert by["pd_tiles_851"][0][2] == run.leaves.data_ptr() + 64 * before      # the tile leaves, the call's last
        assert by["gemm_pearlc_d"][0][4] == y.data_ptr()
        sh = run.shapes[(32, weights[w].shape[0], 1024)]
        assert by["pd_commit_851"][0][9] == sh.count.data_ptr() and by["pd_commit_851"][1] == (32, 1024, 0, sh.mp)
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
    assert run.pipes[(0, 128, frozenset(), True, "serial", False, False)][0].bm == 128
    by = {n: p for n, _, p, _ in kernel.dev.log}
    assert names[-2:] == ["gemm_pearlc", "pd_tiles_979"] and by["gemm_pearlc"][4] == y.data_ptr()
    assert by["pd_tiles_979"][:2] == tuple(by["gemm_pearlc"][2:4])


def test_nohash_forms_and_multiplies_without_hashing(device_run):
    make, kernel, weights, *_ = device_run
    run = make(hashing=False)
    kernel.dev.log.clear()
    run.gemm(0, torch.randn(16, 1024).to(torch.bfloat16), weights[0], None)
    assert [n for n, *_ in kernel.dev.log if not n.endswith("_smem")] == ["stats_r", "eprime", "form_s5", "words_k", "pa",
                                                                          "gemm_pearlc_d"]
    assert run.end() == {"units": 1, "base": 0, "tiles": 0, "hashing": False, "schedule": "serial"}


def test_passes_number_their_calls_across_the_epoch(device_run):
    """Each pass's calls continue the execution phase's indices (`pouw_pass.call_index`), its buffers positional: three calls, then
    two in the next pass take 3 and 4.  A check that repeats a pass from its index (`begin`'s ``at``) shares its indices,
    and the pass after continues past every index the execution phase has used, never from the repeat's end; so does a call after
    `finish`."""
    make, _, weights, *_ = device_run
    run = make()
    x = torch.randn(16, 1024).to(torch.bfloat16)
    for w in (0, 1, 0):
        run.gemm(w, x, weights[w], None)
    first = run.end_pass()
    for w in (1, 0):
        run.gemm(w, x, weights[w], None)
    assert first["base"] == 0 and [u.index for u in run.cur.matmuls] == [3, 4] and run.cur.base == 3
    second = run.end_pass()
    assert second["base"] == 3 and run.epoch_calls() == 5
    run.begin(at=0)
    run.gemm(0, x, weights[0], None)
    assert [u.index for u in run.cur.matmuls] == [0]
    run.repeat_at = 3
    run.end_pass()
    assert run.cur.base == 3 and run.repeat_at is None
    run.end_pass()
    assert run.cur.base == 5
    run.gemm(1, x, weights[1], None)
    run.finish()
    assert run.cur.base is None and run.epoch_calls() == 6
    run.gemm(0, x, weights[0], None)
    assert run.cur.base == 6 and [u.index for u in run.cur.matmuls] == [6]


def test_the_epochs_call_at_2_to_the_32_is_refused_on_the_device(device_run):
    """A call whose index would reach `pearl_c.CALL_RANGE` (2^32, the row seed's range) raises before it runs, naming the
    fix: the execution phase ends and the install takes a new salt."""
    make, kernel, weights, *_ = device_run
    run = make()
    x = torch.randn(16, 1024).to(torch.bfloat16)
    run.begin(at=C.CALL_RANGE - 1)
    run.gemm(0, x, weights[0], None)
    assert [u.index for u in run.cur.matmuls] == [C.CALL_RANGE - 1]
    kernel.dev.log.clear()
    with pytest.raises(RuntimeError, match="2\\^32 calls.*reinstall under a new salt"):
        run.gemm(1, x, weights[1], None)
    assert not kernel.dev.log and len(run.cur.matmuls) == 1


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
    assert man["schema"] == CC.MANIFEST_SCHEMA and man["scheme"] == SCHEME and man["salt"] == salt.hex()
    assert "session" not in man, "in-process coins: no live session to name"
    assert [(u["m"], u["real_rows"], u["weight"]) for u in man["matmuls"]] == [(32, 20, 0), (32, 32, 1)]
    assert man["commitment"] == {"root": out["transcript_root"], "count": 6}
    assert {w: v["root"] for w, v in man["weights"].items()} == {str(w): r.hex() for w, r in roots.items()}
    assert (tmp_path / man["matmuls"][0]["rows"]["path"]).stat().st_size == 2 * 32 * 1024
    assert (tmp_path / man["weights"]["0"]["rows"]["path"]).stat().st_size == 2 * 256 * 1024
    for u in man["matmuls"]:
        y, tree = (tmp_path / u["y"]["rows"]["path"]).read_bytes(), (tmp_path / u["y"]["tree"]["path"]).read_bytes()
        root, want = SY.commit_y(y, u["n"], salt, u["index"], SALT_KEY)
        assert len(y) == 2 * u["real_rows"] * u["n"] and tree == want and u["y"]["root"] == root.hex() == tree[-64:].hex()


def test_a_retained_call_given_its_y_tree_retains_that_tree_and_never_makes_one(device_run, tmp_path, monkeypatch):
    """A caller that commits y with its own launches (the overhead arm) hands `_retained` its tree (``y_tree``): the pass
    retains those bytes, its y root their last node, and makes no tree of its own; so a tree its launches wrote nothing
    into is retained as the poison it held, and the verifier sees what the launches left."""
    make, _, weights, _, salt = device_run
    run = make()
    run.begin(retain=str(tmp_path))
    retained, given = run._retained, {}

    def with_tree(cur, mm, m, flags, xa, tree, y=None, y_tree=None):
        honest = SY.commit_y(PPD._bytes(y.contiguous().view(torch.uint8)), y.shape[1], salt, mm.index, SALT_KEY)[1]
        given[mm.index] = bytes([0xA5]) * len(honest) if given else honest
        return retained(cur, mm, m, flags, xa, tree, y=y, y_tree=torch.frombuffer(bytearray(given[mm.index]), dtype=torch.uint8))
    monkeypatch.setattr(run, "_retained", with_tree)
    monkeypatch.setattr(SY, "y_tree", lambda *_: pytest.fail("a given y tree is never made again"))
    run.gemm(0, torch.randn(20, 1024).to(torch.bfloat16), weights[0], None)
    run.gemm(1, torch.randn(32, 1024).to(torch.bfloat16), weights[1], None)
    man = json.loads(Path(run.end()["manifest"]).read_text())
    for u in man["matmuls"]:
        tree = (tmp_path / u["y"]["tree"]["path"]).read_bytes()
        assert tree == given[u["index"]] and u["y"]["root"] == tree[-64:].hex()
    assert man["matmuls"][1]["y"]["root"] == "a5" * 64


def test_a_retained_calls_files_are_written_behind_it_and_the_passs_end_joins_them(device_run, tmp_path, monkeypatch,
                                                                                    request):
    """`host.write_behind`: when a retained call returns, its entries name its files and sizes and its bytes are a copy of
    their own, which the call's buffers no longer reach; the files (A's rows and tree, y's rows and tree) are written behind
    it (here only once the buffers have changed), each digest landing in its entry, and the pass's end joins them before
    its manifest names them.  A write that fails fails the pass's end, which then writes no manifest."""
    make, _, weights, *_ = device_run
    gate, write = threading.Event(), PH._write_host
    request.addfinalizer(gate.set)

    def gated(*a):
        gate.wait()
        write(*a)
    monkeypatch.setattr(PH, "_write_host", gated)
    run = make()
    run.begin(retain=str(tmp_path / "ok"))
    x = torch.randn(32, 1024).to(torch.bfloat16)
    y = run.gemm(0, x, weights[0], None)
    served = y.contiguous().view(torch.uint8).numpy().tobytes()
    (sh,) = run.shapes.values()
    index = run.cur.matmuls[0].index
    (f,) = run.cur.files
    assert {k: f[k] for k in ("rows", "tree")} == {"rows": {"path": f"calls/{index}.rows.bf16", "bytes": 2 * 32 * 1024},
                                                   "tree": {"path": f"calls/{index}.tree", "bytes": f["tree"]["bytes"]}}
    assert f["y"]["rows"] == {"path": f"y/{index}.rows.bf16", "bytes": 2 * 32 * 256}
    assert len(run.cur.writes) == 1 and not any((tmp_path / "ok" / g["path"]).exists() for g in (f["rows"], f["y"]["rows"]))
    with torch.inference_mode():                              # the call's buffers are made as vLLM's calls make them
        sh.xa.fill_(7.0)
        sh.buf("tree_a").fill_(7)
        y.fill_(7.0)
    gate.set()
    man = json.loads(Path(run.end()["manifest"]).read_text())
    (u,) = man["matmuls"]
    assert run.details["passes"][-1]["manifest"] and not run.cur.writes
    assert (tmp_path / "ok" / u["rows"]["path"]).read_bytes() == x.view(torch.uint8).numpy().tobytes()
    tree = (tmp_path / "ok" / u["tree"]["path"]).read_bytes()
    assert u["tree"]["sha256"] == hashlib.sha256(tree).hexdigest() and tree != bytes([7]) * len(tree)
    assert (tmp_path / "ok" / u["y"]["rows"]["path"]).read_bytes() == served
    ytree = (tmp_path / "ok" / u["y"]["tree"]["path"]).read_bytes()
    assert u["y"]["tree"]["sha256"] == hashlib.sha256(ytree).hexdigest()
    assert u["y"]["root"] == ytree[-64:].hex() == SY.commit_y(served, 256, run.salt, index, run.salt_key)[0].hex()

    def fails(path, *_):
        raise OSError(f"no space left for {path.name}")
    monkeypatch.setattr(PH, "_write_host", fails)
    run.begin(retain=str(tmp_path / "full"))
    run.gemm(0, x, weights[0], None)
    with pytest.raises(OSError, match="no space left"):
        run.end()
    assert not (tmp_path / "full" / "manifest.json").exists() and not run.cur.writes


def test_a_retained_pass_starts_from_poisoned_outputs_and_omit_drops_a_step(device_run, tmp_path):
    """The call runs as vLLM runs it, under inference mode, so its shape's buffers are inference tensors; the retained pass
    begins outside it (execution phase 's verify pass failed there)."""
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
    assert "pd_tiles_851" not in [n for n, *_ in kernel.dev.log] and "gemm_pearlc_d" in [n for n, *_ in kernel.dev.log]


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


# ---- one pass per engine step, through the PoUW option (`Option.end_pass`) ---------------------------------------------

class _Linear:
    """A stand-in for vLLM's ``LinearBase``: its weight and its own ``quant_method``, which the PoUW option executes."""

    def __init__(self, weight):
        self.weight, self.quant_method = weight, types.SimpleNamespace(apply=None)


class _Model:
    def __init__(self, weights):
        self.layers = [_Linear(weights[w]) for w in sorted(weights)]

    def modules(self):
        return iter(self.layers)


class _Engine:
    """A stand-in for vLLM's ``llm_engine``: each `step` runs its step's calls, (weight, rows), through the model's linears,
    as the step's forward does, on rows sampled from `seed`."""

    def __init__(self, model, steps, seed=0):
        self.model, self.steps, self.rng, self.at = model, steps, np.random.default_rng(seed), 0

    def step(self):
        for w, m in self.steps[self.at]:
            layer = self.model.layers[w]
            x = torch.from_numpy(bf16_words(self.rng, m, layer.weight.shape[1]).view(np.float32).copy()).to(torch.bfloat16)
            layer.quant_method.apply(layer, x)
        self.at += 1
        return []


def _served(monkeypatch, kernel, weights, more=None, **run):
    """`weights` served under the PoUW option as `verity_vllm.LLM` installs it (`protocol_options.install`), on `kernel`,
    with the execution phase's coins from a seeded challenger, the further pouw settings `more` and the hm96 key `SALT_KEY` (an
    install samples a fresh one, so two installs compared byte for byte share it); `run` overrides the device run's buffer
    sizes."""
    import functools

    from verity_vllm import protocols as PO
    from verity_vllm.protocols import pouw
    monkeypatch.setattr(pouw, "linear_types", lambda: (_Linear,))
    monkeypatch.setattr(PPD.Kernel, "load", staticmethod(lambda *a: kernel))
    monkeypatch.setattr(PPD, "new_salt_key", lambda scheme: SALT_KEY)
    if run:
        monkeypatch.setattr(PPD, "PearlCDeviceRun", functools.partial(PPD.PearlCDeviceRun, **run))
    model = _Model(weights)
    settings = {"pouw": {"coins": Challenger(0, random.Random(7).randbytes), "device": "cuda", "kernel": "k", **(more or {})}}
    comp = PO.install(model, PO.Context(stage="serve", enabled=(f"pouw:{SCHEME}",), settings=settings))
    return model, comp, comp.get("pouw")


def _per_step(engine, end):
    """`engine.step` wrapped so that `end` closes each step's pass; each step's commitment, with the bytes it commits (the
    tile leaves, A's roots and the screen's flags), in order."""
    from verity_vllm.engine.hooks import Hooks
    hooks, out = Hooks(), []

    def make(original):
        def step(*a, **kw):
            run, got, served = end(original, a, kw)
            out.append({**got, "leaves": PPD._bytes(run.leaves[:run.nb * got["tiles"]]),
                        "roots_a": PPD._bytes(run.roots_buf[:run.nb * got["units"]]),
                        "flags": PPD._bytes(run.flag_buf[:got.get("screened_rows", 0)].contiguous().view(-1))})
            return served
        return step
    hooks.wrap(engine, "step", make)
    return hooks, out


def _reach_in(option):
    """The per-step pass as benchmarks' harness made it before `Option.end_pass` (`verity_bench`'s ``_pass_per_step``): the
    device run found through the option's private ``release``, a pass begun before each engine step and ended after it."""
    run = option.release.__self__

    def end(original, a, kw):
        run.begin()
        served = original(*a, **kw)
        return run, run.end(), served
    return end


def _end_pass(option):
    def end(original, a, kw):
        served = original(*a, **kw)
        return option.end_pass.__self__, option.end_pass(), served
    return end


#: engine steps, each its calls (weight, rows): prefill chunks, decodes, a step with no forward, a call over 128 rows
SERVED_STEPS = [[(0, 40), (1, 40)], [(0, 3), (1, 3)], [], [(1, 130), (0, 17), (1, 17)], [(0, 1), (1, 1)], [(0, 64)]]


@pytest.mark.parametrize("dev", ["recording", pytest.param("twin", marks=[
    pytest.mark.slow, pytest.mark.skipif(not (shutil.which("g++") or shutil.which("c++")), reason="simt_twin needs a C++ compiler")])])
def test_end_pass_commits_each_engine_step_as_the_per_step_reach_in_did(monkeypatch, tmp_path, dev):
    """The PoUW option's `end_pass` at each engine step's end commits what the harness's reach-in (begin before the step, end
    after it) committed: per step, the same pass commitment (transcript root, A roots' digest, units, tiles, screened,
    failing and unsure rows), the same tile leaves, A roots and screen flags.  On ``twin`` the hash kernels run on the host
    (`TwinDev`), so the roots are real; on ``recording`` nothing is written and only the counts and the flags differ by
    step.  After close the passes are the steps' own: `end_pass` leaves the next pass open and empty, so `finish` ends
    nothing twice."""
    rng = np.random.default_rng(13)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    got, pd = {}, HD.build(tmp_path) if dev == "twin" else None
    for name, end in (("reach-in", _reach_in), ("end_pass", _end_pass)):
        kernel = PPD.Kernel(R, F, TwinDev(pd) if dev == "twin" else RecordingDev(), "0" * 64)
        model, comp, option = _served(monkeypatch, kernel, weights)
        engine = _Engine(model, SERVED_STEPS)
        hooks, out = _per_step(engine, end(option))
        for _ in SERVED_STEPS:
            engine.step()
        hooks.uninstall()
        got[name] = out, [dict(p) for p in option.details["passes"]], dict(option.commitment())
        comp.close()
        if name == "end_pass":
            assert len(option.details["passes"]) == len(SERVED_STEPS)
    (a, a_passes, a_commits), (b, b_passes, b_commits) = got["reach-in"], got["end_pass"]
    assert [(p["units"], p["tiles"], p.get("screened_rows", 0)) for p in b] == [
        (2, 6, 80), (2, 6, 6), (0, 0, 0), (3, 6 + 4 + 2, 164), (2, 6, 2), (1, 4, 64)]
    assert a == b and a_passes == b_passes and a_commits == b_commits
    if dev == "twin":
        assert len({p["activation_roots"] for p in b if p["units"]}) == 5


@pytest.mark.slow
def test_a_long_served_run_ending_a_pass_each_engine_step_never_overflows(monkeypatch):
    """A pass holds 32 tiles and 512 screened rows, and each step takes 24 tiles and 400 rows: with `end_pass` at each step's
    end, 256 steps (6,144 tiles) each commit in a pass of their own; without it, the install's one pass overflows at the
    second step, as served Pearl-C did within a few prefill steps (the default buffers hold 8M tiles and 4M rows). Its
    length is the point: a few steps would not show a buffer that grows across passes."""
    rng = np.random.default_rng(17)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    steps = [[(0, 200), (1, 200)]] * 256
    kernel = PPD.Kernel(R, F, RecordingDev(), "0" * 64)
    model, comp, option = _served(monkeypatch, kernel, weights, capacity=32, flag_rows=512)
    engine = _Engine(model, steps)
    hooks, out = _per_step(engine, _end_pass(option))
    for _ in steps:
        engine.step()
    hooks.uninstall()
    comp.close()
    assert [(p["units"], p["tiles"], p["screened_rows"]) for p in out] == [(2, 24, 400)] * len(steps)
    assert len(option.details["passes"]) == len(steps)
    model, comp, option = _served(monkeypatch, kernel, weights, capacity=32, flag_rows=512)
    engine = _Engine(model, steps)
    engine.step()
    with pytest.raises(RuntimeError, match="over its capacity"):
        engine.step()
    comp.close()


# ---- one pass per engine step, ended by the PoUW option itself at the model runner's step ----------------------------

class _Runner:
    """A stand-in for vLLM's ``GPUModelRunner``: ``execute_model`` runs one step's forward, `_Engine.step`'s calls."""

    engine: ClassVar[_Engine | None] = None

    def execute_model(self, scheduler_output=None):
        return type(self).engine.step()


def _runners(monkeypatch, *modules):
    """vLLM's model runner modules as this process holds them once an engine is built (`pouw_pass.MODEL_RUNNERS`), each
    with a fresh `_Runner` subclass, each module's a subclass of the one before; the classes, in order."""
    out, base = [], _Runner
    for mod in modules or ("vllm.v1.worker.gpu.model_runner",):
        base = type("GPUModelRunner", (base,), {})
        monkeypatch.setitem(sys.modules, mod, types.SimpleNamespace(GPUModelRunner=base))
        out.append(base)
    return out


class _Core:
    """A stand-in for vLLM's in-process engine core: each `step` schedules and calls the runner's ``execute_model`` once,
    and nothing else touches the PoUW option (no harness wrapper, no ``end_pass``)."""

    def __init__(self, runner, engine):
        self.runner = runner
        type(runner).engine = engine

    def step(self):
        return self.runner.execute_model(object())


def _hooked(option):
    def end(original, a, kw):
        served = original(*a, **kw)
        return option.end_pass.__self__, option.details["passes"][-1], served
    return end


@pytest.mark.parametrize("dev", ["recording", pytest.param("twin", marks=[
    pytest.mark.slow, pytest.mark.skipif(not (shutil.which("g++") or shutil.which("c++")), reason="simt_twin needs a C++ compiler")])])
def test_the_model_runner_hook_commits_each_engine_step_as_end_pass_did(monkeypatch, tmp_path, dev):
    """Served with a model runner in the process, the PoUW option ends its pass when ``execute_model`` returns, so plain
    serving commits per step exactly what `Option.end_pass` at each step's end committed without one: per step the same
    commitment, tile leaves, A roots and screen flags, the same passes and the same commitment after close.  A caller that
    still calls ``end_pass`` after each hooked step (benchmarks' harness) ends nothing: it gets the step's commitment back
    and the passes stay the steps' own."""
    rng = np.random.default_rng(13)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    got, pd = {}, HD.build(tmp_path) if dev == "twin" else None
    for name in ("end_pass", "hook", "hook and end_pass"):
        with monkeypatch.context() as mp:
            kernel = PPD.Kernel(R, F, TwinDev(pd) if dev == "twin" else RecordingDev(), "0" * 64)
            runners = _runners(mp) if name != "end_pass" else []
            model, comp, option = _served(mp, kernel, weights)
            engine = _Engine(model, SERVED_STEPS)
            stepper = _Core(runners[0](), engine) if runners else engine
            hooks, out = _per_step(stepper, _hooked(option) if name == "hook" else _end_pass(option))
            for _ in SERVED_STEPS:
                stepper.step()
            hooks.uninstall()
            got[name] = out, [dict(p) for p in option.details["passes"]], dict(option.commitment())
            assert option.details["pass_ends"].startswith("the caller's end_pass" if name == "end_pass" else "each engine step")
            comp.close()
            assert not runners or "execute_model" not in vars(runners[0])
            assert len(option.details["passes"]) == len(SERVED_STEPS)
    assert [(p["units"], p["tiles"], p.get("screened_rows", 0)) for p in got["hook"][0]] == [
        (2, 6, 80), (2, 6, 6), (0, 0, 0), (3, 6 + 4 + 2, 164), (2, 6, 2), (1, 4, 64)]
    assert got["end_pass"] == got["hook"] == got["hook and end_pass"]
    if dev == "twin":
        assert len({p["activation_roots"] for p in got["hook"][1] if p["units"]}) == 5


def test_plain_serving_with_a_model_runner_never_overflows(monkeypatch):
    """`test_a_long_served_run_ending_a_pass_each_engine_step_never_overflows` with nothing but the engine core's steps:
    256 steps of 24 tiles and 400 rows, under a pass of 32 tiles and 512 rows, each commit in a pass of their own."""
    rng = np.random.default_rng(17)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    steps = [[(0, 200), (1, 200)]] * 256
    (runner,) = _runners(monkeypatch)
    model, comp, option = _served(monkeypatch, PPD.Kernel(R, F, RecordingDev(), "0" * 64), weights, capacity=32, flag_rows=512)
    core = _Core(runner(), _Engine(model, steps))
    for _ in steps:
        core.step()
    comp.close()
    assert [(p["units"], p["tiles"], p["screened_rows"]) for p in option.details["passes"]] == [(2, 24, 400)] * len(steps)


def test_a_manual_end_pass_ends_only_what_no_engine_step_ended(monkeypatch):
    """The two ways a pass ends can't end one twice.  After a hooked step, ``end_pass`` returns that step's commitment and
    ends nothing, however often it is called, and arguments it is given apply to the open pass.  A call made outside any
    step is the caller's to end: ``end_pass`` ends it.  With V1's and V2's runner modules both loaded, V2's class a
    subclass of V1's, each step still ends one pass, and a forward that raises leaves its calls in the open pass."""
    rng = np.random.default_rng(19)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    v1, v2 = _runners(monkeypatch, "vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner")
    model, comp, option = _served(monkeypatch, PPD.Kernel(R, F, RecordingDev(), "0" * 64), weights)
    run, passes = option.end_pass.__self__, option.details["passes"]
    engine = _Engine(model, [[(0, 40)], [(1, 3)], [(0, 5), (1, 5)], [(0, 7)]])
    core = _Core(v2(), engine)
    core.step()
    assert [(p["units"], p["tiles"]) for p in passes] == [(1, 4)]
    assert option.end_pass() is passes[-1] and option.end_pass() is passes[-1] and len(passes) == 1
    assert run.end_pass(run="r-next") is passes[-1] and (run.cur.run, run.cur.stepped, len(passes)) == ("r-next", True, 1)
    core.step()
    layer = model.layers[0]
    layer.quant_method.apply(layer, torch.zeros(16, 1024, dtype=torch.bfloat16))
    assert option.end_pass()["units"] == 1 and len(passes) == 3
    engine.steps[2] = [(0, 5), (1, 5), (2, 5)]
    with pytest.raises(IndexError):
        core.step()
    assert (len(passes), len(run.cur.matmuls)) == (3, 2)
    engine.at = 3
    core.step()
    comp.close()
    assert [p["units"] for p in passes] == [1, 1, 1, 3]
    assert "execute_model" not in vars(v1) and "execute_model" not in vars(v2)


@pytest.mark.parametrize("dev", ["recording", pytest.param("twin", marks=[
    pytest.mark.slow, pytest.mark.skipif(not (shutil.which("g++") or shutil.which("c++")), reason="simt_twin needs a C++ compiler")])])
def test_each_served_step_is_a_pass_retained_on_its_own(monkeypatch, tmp_path, dev):
    """Plain serving (a model runner, nothing else ending passes) under ``transcript_dir``, three steps: each pass is one
    step's forward over every weight, retained in ``pass-NNNNN`` in the order the passes began (the one the install began
    first among them), its manifest's commitment the one it published and its calls the step's, numbered on from the last
    pass's (the execution phase's call indices, so the passes together hold each index once), its weights' files hard links to the
    first pass's; the pass left open and empty after the last step writes nothing.  On ``twin`` the hash kernels run, so
    A's roots are real and differ pass to pass."""
    rng = np.random.default_rng(23)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    steps = [[(0, 200), (1, 200)], [(0, 120), (1, 120)], [(0, 3), (1, 3)]]
    kernel = PPD.Kernel(R, F, TwinDev(HD.build(tmp_path)) if dev == "twin" else RecordingDev(), "0" * 64)
    (runner,) = _runners(monkeypatch)
    root = tmp_path / "passes"
    model, comp, option = _served(monkeypatch, kernel, weights, {"transcript_dir": str(root)}, capacity=32, flag_rows=512)
    core = _Core(runner(), _Engine(model, steps))
    for _ in steps:
        core.step()
    comp.close()
    passes = option.details["passes"]
    assert [(p["units"], p["tiles"], p["screened_rows"]) for p in passes] == [(2, 24, 400), (2, 12, 240), (2, 6, 6)]
    assert [Path(p["manifest"]) for p in passes] == [root / f"pass-{i:05d}" / "manifest.json" for i in range(3)]
    assert sorted(d.name for d in root.iterdir()) == [f"pass-{i:05d}" for i in range(3)]
    mans = []
    for i, (p, step) in enumerate(zip(passes, steps)):
        man = json.loads(Path(p["manifest"]).read_text())
        mans.append(man)
        assert p["base"] == 2 * i
        assert [(u["index"], u["weight"], u["real_rows"]) for u in man["matmuls"]] == [(2 * i + j, w, m) for j, (w, m) in enumerate(step)]
        assert man["commitment"] == {"root": p["transcript_root"], "count": p["tiles"]} and sorted(man["weights"]) == ["0", "1"]
    assert passes_distinct(mans) is None
    for f in ("weights/0.rows.bf16", "weights/0.tree", "weights/1.rows.bf16", "weights/1.tree"):
        stats = [(root / f"pass-{i:05d}" / f).stat() for i in range(3)]
        assert len({s.st_ino for s in stats}) == 1 and stats[0].st_nlink == 3, f
    if dev == "twin":
        assert len({p["activation_roots"] for p in passes}) == 3


#: `_retained_files`' digest of the three steps below as the retention path wrote them when each call's copies, its record,
#: y's root and the pass's files and manifest were all waited for in its step
RETAINED_DIGEST = "6ea82b92968e7213c16a075e8c0d7910dcbe5ea32f8892f937b85165e3a8722d"


def _retained_files(root: Path) -> dict[str, str]:
    """Every file under `root` and its bytes' SHA-256."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def test_a_step_completed_behind_it_retains_the_bytes_a_waiting_end_retains(monkeypatch, tmp_path):
    """`PassEnds.retain_behind`: three served steps, each pass's files, records and manifest completed behind its step
    (`retained_ends`, which the install's close waits for), retain every byte that ends completing them in the step do, the
    same files as the path that waited for each call's copies (`RETAINED_DIGEST`); each file is the owner's alone (0600)
    and each directory of a pass too (0700)."""
    zeroed_y(monkeypatch)
    rng = np.random.default_rng(23)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    steps = [[(0, 200), (1, 200)], [(0, 120), (1, 120)], [(0, 3), (1, 3)]]
    got, cls = {}, PPD.PearlCDeviceRun
    for behind in (False, True):
        monkeypatch.setattr(cls, "retain_behind", behind)
        (runner,) = _runners(monkeypatch)
        root = tmp_path / f"behind-{behind}"
        model, comp, option = _served(monkeypatch, PPD.Kernel(R, F, RecordingDev(), "0" * 64), weights,
                                      {"transcript_dir": str(root)}, capacity=32, flag_rows=512)
        core = _Core(runner(), _Engine(model, steps))
        for _ in steps:
            core.step()
        comp.close()
        assert all(p["manifest"] and p["excluded_rows"] is not None for p in option.details["passes"])
        got[behind] = _retained_files(root)
        modes = {p.stat().st_mode & 0o777 for p in root.rglob("*")}
        assert modes == {0o600, 0o700}, sorted(map(oct, modes))
    assert got[True] == got[False] and len(got[True]) == 3 * (1 + 2 + 2 * 2 * 2 + 2 * 2)
    assert hashlib.sha256(json.dumps(got[True], sort_keys=True).encode()).hexdigest() == RETAINED_DIGEST


def test_a_pass_poisons_again_only_the_shapes_a_call_took_since_the_last_poison(monkeypatch, tmp_path):
    """`PearlCDeviceRun.poison`: each retained pass begins with every shape's buffers at the poison byte, but a shape no call
    took since the last poison holds it already and isn't filled again; one a call took is."""
    rng = np.random.default_rng(37)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    steps = [[(0, 200), (1, 200)], [(0, 3), (1, 3)], [(0, 3), (1, 3)]]
    (runner,) = _runners(monkeypatch)
    model, comp, option = _served(monkeypatch, PPD.Kernel(R, F, RecordingDev(), "0" * 64), weights,
                                  {"transcript_dir": str(tmp_path)}, capacity=32, flag_rows=512)
    run, core, fills, fill = option.end_pass.__self__, _Core(runner(), _Engine(model, steps)), [], torch.Tensor.fill_
    monkeypatch.setattr(torch.Tensor, "fill_", lambda t, v: (fills.append(t.data_ptr()), fill(t, v))[1])

    def buffers(rows):
        return [t for (m16, _, _), sh in run.shapes.items() if (m16 > 16) == rows for t in run._buffers(sh)]
    core.step()
    big = buffers(True)
    assert big and {t.data_ptr() for t in big} <= set(fills)
    fills.clear()
    core.step()
    assert not {t.data_ptr() for t in big} & set(fills) and all(bool((t == 0xA5).all()) for t in big)
    assert {t.data_ptr() for t in buffers(False)} <= set(fills)
    core.step()
    comp.close()


def test_the_record_decided_from_the_rows_bytes_is_the_devices(monkeypatch):
    """`screen.excluded_host`, which a device's retained call decides its record by from the bytes its rows' write lands
    (the flags and A's rows as BF16), excludes the rows `screen.excluded` does from the flags and A's FP32 words."""
    S = importlib.import_module("verity_vllm.protocols.pouw.pearl_c.screen")
    rng = np.random.default_rng(31)
    words = bf16_words(rng, 64, 1024).view(np.uint32)
    words[::3, 1::8] = 0x47000000                             # 2^15 off the screen's stride: rows live_row rejects
    x = torch.from_numpy(words.view(np.float32).copy())
    flags = torch.from_numpy(rng.integers(0, 2, (64, 2), dtype=np.uint8))
    want = S.excluded(flags[:, 0].bool(), flags[:, 1].bool(), x)
    rows = x.to(torch.bfloat16).view(torch.uint8).numpy().view("<u2").reshape(64, 1024)
    assert S.excluded_host(flags.numpy(), rows) == want and want and set(want) > set(np.flatnonzero(flags[:, 0].numpy()))


def test_the_pinned_ring_hands_out_disjoint_regions_and_frees_them_in_order(monkeypatch):
    """`host._Ring`: regions handed out are disjoint from every live one, a region given back out of order is free only
    once every earlier one is, and a take that doesn't fit before the oldest live region (or past the end) waits."""
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, pin_memory=False, **kw: empty(*a, **kw))
    ring = PH._Ring(1000)
    a, b, c = (ring.take(n) for n in (300, 300, 300))
    assert [r[:2] for r, _ in (a, b, c)] == [[0, 300], [300, 600], [600, 900]] and ring.take(200) is None
    ring.give(b[0])
    assert ring.take(200) is None                             # b is free only once a is
    ring.give(a[0])
    d = ring.take(400)
    assert d[0][:2] == [0, 400] and ring.take(250) is None    # the end past 900 is too short, 400..600 is c's way
    ring.give(c[0])
    e = ring.take(500)
    assert e[0][:2] == [400, 900] and d[1].numel() == 400
    ring.give(d[0])
    ring.give(e[0])
    assert not ring.live and ring.take(1000)[0][:2] == [0, 1000]


def test_a_step_over_one_pass_is_refused_before_its_forward(monkeypatch):
    """`pouw_pass.step_tokens`: weights of n 256 and 128 take 4 + 2 tiles a 64-row tile, so a pass of 32 tiles holds 5 tile
    rows (320 tokens) and one of 512 screened rows 256 tokens over its two calls a step: a step is held to 256 tokens
    (``details["max_step_tokens"]``).  A step scheduled over it is refused before its forward runs, leaving the open pass
    empty and nothing committed; a step at the limit commits as one pass, within both caps."""
    rng = np.random.default_rng(29)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    (runner,) = _runners(monkeypatch)
    model, comp, option = _served(monkeypatch, PPD.Kernel(R, F, RecordingDev(), "0" * 64), weights, capacity=32, flag_rows=512)
    assert option.details["max_step_tokens"] == 256
    r = runner()
    type(r).engine = _Engine(model, [[(0, 257), (1, 257)], [(0, 256), (1, 256)]])
    with pytest.raises(RuntimeError, match=r"a step of 257 tokens is over one pass \(256 tokens a step"):
        r.execute_model(types.SimpleNamespace(total_num_scheduled_tokens=257))
    assert not option.end_pass.__self__.cur.matmuls and option.details["passes"] == []
    type(r).engine.at = 1
    r.execute_model(types.SimpleNamespace(total_num_scheduled_tokens=256))
    comp.close()
    assert [(p["units"], p["tiles"], p["screened_rows"]) for p in option.details["passes"]] == [(2, 24, 512)]


# ---- the trees, their hash kernels run on the host ----------------------------------------------------------------

TWIN = load("simt_twin", KERNEL.parent / "pearl_c" / "simt_twin.py")
HD = load("pearl_c_hash_decode_test", Path(__file__).with_name("test_pearl_c_hash_decode.py"))


class TwinDev(RecordingDev):
    """`RecordingDev` that also runs each hash kernel `simt_twin` builds, thread for thread, and hash_decode.cuh's (`pd`, its
    host build, with the header tables hash_fast.cuh's s512_heads writes there too, and in hash_prefill.cuh's build chain
    "pl"'s), at the addresses the pipeline passes: the
    run's buffers are CPU tensors, so the trees are really computed.  The GEMMs, forming and seed_line_a don't run."""

    def __init__(self, pd):
        super().__init__()
        self.pd = pd

    def launch(self, name, grid, block, smem=0, p=(), i=(), stream=None):
        super().launch(name, grid, block, smem, p, i, stream)
        if name in TWIN.KERNELS:
            a = TWIN.Args()
            for j, v in enumerate(p):
                a.p[j] = v
            for j, v in enumerate(i):
                a.i[j] = v
            assert getattr(TWIN.build(), f"run_{name}")(ctypes.byref(a), grid, block, smem) == 0, name
        if name.startswith(("pd_", "pl_")) or name == "s512_heads":
            a = R.Args()
            for j, v in enumerate(p):
                a.p[j] = v
            for j, v in enumerate(i):
                a.i[j] = v
            assert self.pd.pd_run(name.encode(), grid, block, ctypes.byref(a)) == 0, name


@pytest.mark.parametrize("scheme", KERNEL_SCHEMES)
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
@pytest.mark.parametrize("scheme", KERNEL_SCHEMES)
def test_every_tree_the_device_path_commits_is_its_hashing_formats(scheme, tmp_path):
    """B's roots, each call's A root over its m₁₆ rows (filler zero), and the pass's tile tree over the leaves the calls wrote,
    their kernels run by `simt_twin` and hash_decode.cuh's host build: each is `MerkleTree`'s root in hm96-sha512
    (frame-v3-sha512 over values salted by the prover's key).  A decode call's commitment is pd_commit's: each row digest
    `pearl_c.row_digest` of its row, the row seeds the scheme's `row_seeds` with their line keys, seed_A the scheme's, and E_A's
    lines the m₁₆ committed rows' under their own keys and the filler rows' under line_a; its tile leaves (pd_tiles') over
    the zero C~ and U are `pearl_c.leaf`'s."""
    s = PPD.scheme_of(scheme)
    h, nb, key = s.commitment_hash, 64, SALT_KEY
    rng = np.random.default_rng(5)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16) for w, n in ((0, 256), (1, 128))}
    rows = lambda x: [r.tobytes() for r in x.float().numpy()]  # noqa: E731
    kernel = PPD.Kernel(R, F, TwinDev(HD.build(tmp_path)), "0" * 64)
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
        assert names[0] == "pd_commit_851" and names[-1] == "pd_tiles_851", names
        sh, index = run.shapes[(32, weights[w].shape[0], 1024)], len(run.cur.matmuls) - 1
        a = torch.zeros((32, 1024), dtype=torch.float32)
        a[:x.shape[0]] = x.float()
        mm = Matmul(index, Shape(32, 1024, weights[w].shape[0]), w)
        ctx = Context(salt, mm, PPD._bytes(run.roots_buf[nb * index:nb * index + nb]), roots[w])
        want, seed_a = s.row_seeds(ctx, rows(a), range(32)), s.noise(ctx).seed_a
        assert PPD._bytes(sh.b["rowdig_a"][:64 * 32]) == b"".join(C.row_digest(r) for r in rows(a))
        assert PPD._bytes(sh.b["seeds_a"][:32 * 32]) == b"".join(want)
        assert PPD._bytes(sh.b["lkeys_a"][:32 * 32]) == b"".join(C.line_key(z) for z in want)
        assert PPD._bytes(sh.b["seed_a"][:32]) == seed_a
        assert PPD._bytes(sh.b["ea_codes"][:32 * sh.mp]) == b"".join(
            bytes(C.sample_line(C.S512Seed(want[t]), 0, 0, 0) if t < 32 else C.sample_line(C.S512Seed(seed_a), 0, 0, t))
            for t in range(sh.mp))
        (commit,) = [(p, i) for n, _, p, i in kernel.dev.log if n == "pd_commit_851"]
        assert commit[0][4] == sh.b["lkeys_a"].data_ptr() and commit[0][12] == sh.b["ea_codes"].data_ptr()
        assert commit[1] == (32, 1024, 0, sh.mp)
    for index, (w, x) in enumerate(calls):
        a = torch.zeros((32, 1024), dtype=torch.float32)
        a[:x.shape[0]] = x.float()
        want = commit_rows(activation_binding(salt, index), index, rows(a), h, key, index).commitment.root
        assert PPD._bytes(run.roots_buf[nb * index:nb * index + nb]) == want, index
    count = run.cur.tiles
    assert count == 4 + 2
    zero = C.leaf(C.Checked([[0] * 64] * 32, [[0] * 64] * 32), "h3")
    assert PPD._bytes(run.leaves[:nb * count]) == zero * count
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


def reference_y(s, ctx: Context, a_rows: list[bytes], b_rows: list[bytes]) -> bytes:
    """The call's served y as the kernels store it, row-major BF16: each tile's checked U unscaled (`pearl_c_u.y`)."""
    y = np.zeros((ctx.matmul.shape.m, ctx.matmul.shape.n), "<u2")
    for tile in s.tiles(ctx.matmul.shape):
        ops = s.encode(ctx, s.noise(ctx), [a_rows[i] for i in tile.rows], [b_rows[j] for j in tile.cols], tile)
        if ops is None:
            continue
        ch = s.checked(ctx, ops)
        for x, i in enumerate(tile.rows):
            for z, j in enumerate(tile.cols):
                y[i, j] = PCU.y(ch.u[x][z], ops.a.alpha[x], ops.b.alpha[z])
    return y.tobytes()


#: the reference verifier's Skip debit has no rule without promotion, so v2 (sm120-unpromoted) isn't provable here
PROMOTED = sorted(s for s in PPD.SCHEMES if "unpromoted" not in s)


@pytest.fixture(params=PROMOTED)
def honest_pass(tmp_path, request):
    return proved_pass(tmp_path, request.param)


def proved_pass(tmp_path: Path, scheme: str, indices: tuple[int, int] = (0, 1), key: bytes | None = None,
                live: int | None = None, seed: int = 11, spike: float | None = None, bad_b: int | None = None,
                tile: tuple[int, int] = (16, 16), zero_a: int | None = None):
    """Two matmuls of a test instance (k = 128, 16 x 16 tiles), one with 12 filler rows and one liveness-failing row, proved by
    the scheme's reference prover and written in the device run's layout, for each promoted device scheme (hm96 trees, their
    salts under the prover's key, which the manifest retains; the unpromoted record's debit isn't stated, so clear_work can't
    audit it).  The key is fixed per scheme unless `key` names one: -h3's row seeds follow A's salted roots, and under some
    keys (about 2 in 15) a credited row of a sampled tile is over its own cap, which the sampling design and verify_run both reject.  On the
    h100 record (Hopper's 14-bit atom) 2 of the first 8 keys are, the zero key among them: at k = 128 a row's cap admits
    one flagged atom-word (at k = 2,048 and 6,144 the worst of 96 Gaussian rows each is at 0.27 and 0.11 of its cap).
    The calls have the execution phase's indices `indices`, and every pass built here is of the one execution phase (the same weights, key and
    salt coin), as passes of one install are.  With `live` the salt coin is a live session's, opened after salt epoch
    `live`'s root over the weights' root (`live_coin`), and the manifest names that epoch (``pass_index``).  `seed` draws
    the rows: under another, the weights are others, and so is the salt the one coin gives them.  With `spike`, credited
    row 9 of the first call holds one entry `spike` times its every-8th entries' RMS (ρ's positions), off them.  With
    `bad_b`, that row of the second call's weight fails liveness as the first call's row 5 does, and is excluded.  `tile`
    is the instance's tile: at 64 x 64 each weight has 64 rows and each matmul is one tile, partial down its rows, k_min 32
    keeping it a test instance.  With `zero_a`, that real row of the first call is all +0: it fails liveness and is
    excluded, its U is +0 and so is every word of its served y."""
    if key is None:
        key = bytes([1]) * 32 if SchemeId.parse(scheme).device == "h100" else bytes(32)
    s = PPD.scheme_of(scheme, k_min=32, tile=tile)
    rng, k = np.random.default_rng(seed), 128
    n0, n1 = max(16, tile[1]), max(32, tile[1])
    wrows = {0: bf16_words(rng, n0, k), 1: bf16_words(rng, n1, k)}
    a0 = bf16_words(rng, 32, k)
    a0[20:] = 0
    a0[5, [1, 2, 3]] = np.float32(1e4).view(np.uint32) & 0xFFFF0000
    if spike is not None:
        a0[9, 1] = np.float32(spike * np.sqrt(np.mean(a0[9, ::8].view(np.float32) ** 2))).view(np.uint32) & 0xFFFF0000
    if bad_b is not None:
        wrows[1][bad_b, [1, 2, 3]] = np.float32(1e4).view(np.uint32) & 0xFFFF0000
    if zero_a is not None:
        a0[zero_a] = 0
    i0, i1 = indices
    acts = {i0: a0, i1: bf16_words(rng, 16, k)}
    rows = lambda x: [r.tobytes() for r in x]
    work = Workload((Matmul(i0, Shape(32, k, n0), 0), Matmul(i1, Shape(16, k, n1), 1)))
    prover = Prover(s, {w: rows(x) for w, x in wrows.items()}, salt_key=key)
    salted = lambda v, nonce, i: v + hm96_salt(prover.salt_key, nonce, i)  # noqa: E731
    if live is None:
        coins = salt_coin(prover.weight_roots)
        epoch = Epoch.start(work, prover.weight_roots, *coins)
    else:
        coins = live_coin(PS.epoch_open_root(weights_root(prover.weight_roots), live))
        epoch = PPD.epoch_of(work, prover.weight_roots, *coins, live)
    tr = prover.prove(epoch, {i: rows(x) for i, x in acts.items()})
    splits = {w: PW.Split(k) for w in wrows}
    records = CW.declare(s, epoch, {i: rows(x) for i, x in acts.items()}, {w: rows(x) for w, x in wrows.items()}, splits,
                         real_rows={i0: 20})
    assert records[0].excluded_a == tuple(sorted({5} | ({zero_a} - {None})))
    assert bad_b is None or records[1].excluded_b == (bad_b,)
    (tmp_path / "calls").mkdir(parents=True)
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
        y = reference_y(s, Context(epoch.salt, u, root, epoch.weight_roots[u.weight]), x, rows(wrows[u.weight]))
        y = y[:2 * rec.real_rows * u.shape.n]
        matmuls.append({"index": u.index, "m": u.shape.m, "k": k, "n": u.shape.n, "weight": u.weight, "real_rows": rec.real_rows,
                        "k_kept": rec.k_kept, "excluded_a": list(rec.excluded_a), "root_a": root.hex(),
                        "rows": PPD._write(tmp_path / "calls" / f"{u.index}.rows.bf16", bf16_file(x), digest=False),
                        "tree": PPD._write(tmp_path / "calls" / f"{u.index}.tree",
                                           laid_out(d, [d.leaf(i, i, ROW_SCHEMA, salted(r, u.index, i)) for i, r in enumerate(x)])),
                        "y": SY.retain_y(tmp_path, u.index, y, u.shape.n, epoch.salt, prover.salt_key)})
    count = tr.commitment.count
    leaves = [prover.digest(t) for t in range(count)]
    dt = transcript_domain(epoch, count, s.commitment_hash)
    head = {"schema": CC.MANIFEST_SCHEMA, "run": RUN, "entry": f"pouw:{scheme}", "scheme": scheme,
            "instance": {"k_min": 32, "tile": list(tile)},
            "coins": coins[0].to_json(), "salt_opening": coins[1].to_json(), "salt": epoch.salt.hex(),
            **({} if live is None else {"pass_index": live}),
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
    want = CW.audit_replay(s, epoch, prover, tr, splits, records, 0, CW.RecordedDraw(1, source), 3)
    assert got["accepted"] and want.audit.accepted, got.get("line") or got.get("error")
    assert tuple(got["tiles"]) == want.audit.selected and got["exclusion_tiles"] == 1
    assert got["line"].startswith(f"ACCEPT {s.name} pouw:{s.name}: clear_work's tiles (3 drawn, 1 for excluded rows)")
    assert got["transcript"] == f"{RUN}:{path} sha256:{got['transcript_sha256']}" and got["transcript"] in got["line"]
    assert VR.main([str(path), *coins_args(path), "--jobs", "1", "--commit", "0" * 7]) == 0
    assert VR.main([str(path), *coins_args(path), "--jobs", "1", "--commit", "0" * 7, "--expect-reject"]) == 1
    fast = VR.verify(path, 3, 1, "test", source, fast=True, cross_check=1, coins=published(path))
    assert fast["replay"] == "batch" and fast["cross_checked"] == fast["tiles"] and "cross_check_failed" not in fast
    assert {k: fast[k] for k in ("accepted", "tiles", "verdicts")} == {k: got[k] for k in ("accepted", "tiles", "verdicts")}
    by_place = VR.verify(path, 3, 1, "test", source, fast=True, cross_check=2, coins=published(path))
    assert by_place["accepted"] and by_place["cross_checked"] == by_place["tiles"][::2]
    assert C.form_v1 is SD.FT._SCALAR["form_v1"]
    assert VR.main([str(path), *coins_args(path), "--jobs", "2", "--commit", "0" * 7, "--fast", "--cross-check", "1"]) == 0


def test_verify_run_counts_the_tiles_it_drew_where_no_tile_is_credited(honest_pass, monkeypatch):
    """A pass with no credited cell, as a no-write control whose rows are all excluded, samples no tile (`clear_work.draw`), so
    its line and counts name none sampled and every exclusion tile, not the samples asked for."""
    path, s, *_ = honest_pass
    monkeypatch.setattr(VR.CW, "draw", lambda *_: [])
    got = VR.verify(path, 3, 1, "test", b"verifier" * 4, coins=published(path))
    assert got["accepted"] and (got["drawn"], got["exclusion_tiles"], len(got["tiles"])) == (0, 1, 1), got.get("line")
    assert got["line"].startswith(f"ACCEPT {s.name} pouw:{s.name}: clear_work's tiles (0 drawn, 1 for excluded rows)")


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


def served_verify(d: Path, out: Path, *more: str) -> tuple[int, dict, dict]:
    """``served_passes.py verify`` on serve dir `d` at 3 draws, a test instance's (`proved_pass`'s): its exit, its document
    and its core answer."""
    SP = load("pearl_c_vllm_served_passes", KERNEL.parent / "pearl_c_vllm" / "served_passes.py")
    rc = SP.main(["verify", "--passes", str(d), "--out", str(out), "--jobs", "1", "--draws", "3", "--test-instance", *more])
    answer = json.loads((out.parent / "answer.json").read_text())
    from verity.protocols.pouw.verifier import core_check as CC
    CC.answer_of(answer)
    return rc, json.loads(out.read_text()), answer


@pytest.mark.slow
def test_served_passes_verify_accepts_only_when_every_pass_with_calls_is_accepted(honest_pass, tmp_path):
    """``served_passes.py verify`` draws over the passes with calls that the serve's served.json names, skipping the empty
    one the run left open: it accepts an honest pass and credits its MACs at Pearl's assumption in a core answer, and the
    same pass with a weight row flipped (opened on the excluded row's tile, whatever the draw) fails the whole run.  A
    test instance's verdict credits nothing."""
    path, *_ = honest_pass
    served = {"run": RUN, "passes": [{"pass": 0, "units": 2, "tiles": 4, "manifest": str(path)},
                                     {"pass": 1, "units": 0, "tiles": 0}]}
    (path.parent / "served.json").write_text(json.dumps(served))
    rc, doc, answer = served_verify(path.parent, tmp_path / "verify.json")
    assert rc == 0 and answer["verdict"] == "accept" and doc["line"].startswith("ACCEPT verity/pouw/pearl-plus/v1"), doc["line"]
    work = answer["public"]["work"]
    assert work["credited"] == 0 < work["macs"] and work["credited_tiles"] > 0 and answer["profile"]["delta"] == "1/100"
    assert answer["profile"]["test_instance"] and "a test instance's, credited nothing" in doc["line"]
    assert len(answer["profile"]["tiles"]) == 4 and {i for i, _ in answer["profile"]["tiles"]} == {0}
    man = json.loads(path.read_text())
    assert answer["public"]["passes"][0]["y"] == [u["y"]["root"] for u in man["matmuls"]]
    assert "y committed and checked on 2 of 2 calls" in doc["line"]
    rows = path.parent / man["weights"]["0"]["rows"]["path"]
    bad = bytearray(rows.read_bytes())
    bad[1] ^= 0x01
    rows.write_bytes(bytes(bad))
    rc, doc, answer = served_verify(path.parent, tmp_path / "verify.json")
    assert rc == 1 and answer["verdict"] == "reject" and "weight opening" in answer["reason"] and doc["line"].startswith("REJECT")


@pytest.mark.slow
def test_a_served_y_that_is_not_the_unscale_of_its_u_rejects_the_run(tmp_path):
    """A pass whose y is not what its U unscales to fails the run, whichever tile is drawn: committed as it is (a new root,
    so the run's commitment changes too) each drawn tile names a word that isn't its U's unscale, and left under the root
    the pass published, the rows don't open."""
    path = proved_pass(tmp_path, PROMOTED[0])[0]
    man = json.loads(path.read_text())
    good = VR.verify_passes([path], 3, 1, "test", published(path), b"verifier" * 4, test_instance=True)
    assert good["answer"]["verdict"] == "accept", good["line"]
    for u in man["matmuls"]:
        y = bytearray((path.parent / u["y"]["rows"]["path"]).read_bytes())
        y[0::2] = bytes(b ^ 0x01 for b in y[0::2])
        u["y"] = SY.retain_y(path.parent, u["index"], bytes(y), u["n"], bytes.fromhex(man["salt"]), bytes.fromhex(man["hm96_salt_key"]))
    path.write_text(json.dumps(man))
    lied = VR.verify_passes([path], 3, 1, "test", published(path), b"verifier" * 4, test_instance=True)
    assert lied["answer"]["verdict"] == "reject" and "is not the unscale of the tile's U" in lied["answer"]["reason"], lied["line"]
    assert lied["commitment"] != good["commitment"]
    for u, old in zip(man["matmuls"], good["answer"]["public"]["passes"][0]["y"]):
        u["y"]["root"] = old
    path.write_text(json.dumps(man))
    moved = VR.verify_passes([path], 3, 1, "test", published(path), b"verifier" * 4, test_instance=True)
    assert moved["answer"]["verdict"] == "reject" and "y opening" in moved["answer"]["reason"], moved["line"]


@pytest.mark.slow
def test_a_pass_whose_y_trees_hold_the_poison_is_rejected(tmp_path):
    """The no-write control on y: a pass whose y trees are what launches that wrote nothing into them leave, the 0xA5
    poison, each root its last node (as `_retained` retains a given ``y_tree``), its y rows and everything else honest, is
    rejected at its sampled tiles' y openings, by the one-pass verifier the harness runs and by the passes' verifier."""
    path = proved_pass(tmp_path, PROMOTED[0])[0]
    source = b"verifier" * 4
    assert VR.verify(path, 3, 1, "test", source, coins=published(path))["accepted"]
    man = json.loads(path.read_text())
    for u in man["matmuls"]:
        file = path.parent / u["y"]["tree"]["path"]
        poison = bytes([0xA5]) * file.stat().st_size
        u["y"] = {**u["y"], "root": poison[-64:].hex(), "tree": PPD._write(file, poison)}
    path.write_text(json.dumps(man))
    one = VR.verify(path, 3, 1, "test", source, coins=published(path))
    assert not one["accepted"] and "y opening" in one["line"], one["line"]
    passes = VR.verify_passes([path], 3, 1, "test", published(path), source, test_instance=True)
    assert passes["answer"]["verdict"] == "reject" and "y opening" in passes["answer"]["reason"], passes["line"]


def short_y(path: Path, pos: int, drop: int) -> None:
    """Call `pos`'s y rows file less its last `drop` bytes, committed as the verifier's opener reads it (`VR.f32_row`: row
    i is the words the file holds of its bytes [2ni, 2n(i + 1)), a ragged row's whole ones here) under a new root, which the
    manifest then names: a prover's own y whose last real row is short, and opens."""
    man = json.loads(path.read_text())
    u = man["matmuls"][pos]
    key, n, real = bytes.fromhex(man["hm96_salt_key"]), u["n"], u["real_rows"]
    file = path.parent / u["y"]["rows"]["path"]
    data = file.read_bytes()
    data = data[:len(data) - drop]
    rows = [data[2 * n * i:2 * n * (i + 1)] for i in range(real)]
    d = SY.y_domain(bytes.fromhex(man["salt"]), u["index"], real)
    tree = laid_out(d, [d.leaf(i, i, ROW_SCHEMA, VR.f32_row(r[:len(r) & ~1], len(r), 0) + hm96_salt(key, SY.HM96_Y + u["index"], i))
                        for i, r in enumerate(rows)])
    u["y"] = {"root": tree[-64:].hex(), "rows": PPD._write(file, data, digest=False),
              "tree": PPD._write(path.parent / u["y"]["tree"]["path"], tree)}
    path.write_text(json.dumps(man))


@pytest.mark.slow
def test_a_served_y_row_that_ends_before_its_tiles_columns_rejects_the_run(tmp_path):
    """A y row shorter than its tile is refused by its length, and never read past its end.  The opening binds a row's
    value and not how many words it has, so a prover can commit one: here the last real row of the first call, all +0
    (its A row is zero, so it is excluded and its tile is opened whatever the draw), less its last 4 words, the tree and
    root over the row as it is.  The row opens, and each word it lacks is one whose unscale is +0, the word a read past the
    end of the Lean's row gives: the run is refused naming the row's length, as it is on a row that is absent altogether."""
    path = proved_pass(tmp_path, PROMOTED[0], zero_a=19)[0]
    verdict = lambda: VR.verify_passes([path], 3, 1, "test", published(path), b"verifier" * 4, test_instance=True)  # noqa: E731
    good = verdict()
    assert good["answer"]["verdict"] == "accept", good["line"]
    man = json.loads(path.read_text())
    y = np.frombuffer((path.parent / man["matmuls"][0]["y"]["rows"]["path"]).read_bytes(), "<u2").reshape(20, 16)
    assert not y[19].any() and y[18].all()
    for drop, held in ((4, 12), (12, 0)):
        short_y(path, 0, 2 * drop)
        short = verdict()
        assert short["answer"]["verdict"] == "reject" and short["commitment"] != good["commitment"]
        assert short["answer"]["reason"].endswith(f"y[19]: the opened row holds {held} words, the tile's columns end at 16"), short["line"]


def test_check_y_holds_each_opened_row_to_its_tiles_columns():
    """`check_y` on rows it is handed, the cases of the Lean's `#guard` on `Rule.yReason`: row 7 over columns 2 and 3, U
    +0 at α = 1 on both, so every cell's unscale is +0.  A row that covers the columns passes; one that ends at column 3,
    or before the tile, or holds half a word past column 2, is refused by its length (and nothing raises); a whole row
    is still held to the unscale, word by word."""
    real, index, salt, key = 8, 3, bytes(64), bytes(32)
    d = SY.y_domain(salt, index, real)
    tile = types.SimpleNamespace(rows=(7,), cols=(2, 3))
    side = lambda n: types.SimpleNamespace(alpha=[PCU.ONE] * n)  # noqa: E731
    v = types.SimpleNamespace(checked=types.SimpleNamespace(u=[[0, 0]]), ops=types.SimpleNamespace(a=side(1), b=side(2)))
    assert PCU.y(0, PCU.ONE, PCU.ONE) == 0

    def reason(row: bytes) -> str | None:
        values = [(row if i == 7 else bytes(16)) + hm96_salt(key, SY.HM96_Y + index, i) for i in range(real)]
        tree = laid_out(d, [d.leaf(i, i, ROW_SCHEMA, x) for i, x in enumerate(values)])
        p = types.SimpleNamespace(scheme=types.SimpleNamespace(commitment_hash=HM96, forming="v1", name="stand-in"),
                                  epoch=types.SimpleNamespace(salt=salt, workload=types.SimpleNamespace(
                                      matmuls=[types.SimpleNamespace(index=index)])),
                                  man={"matmuls": [{"real_rows": real, "y": {"root": tree[-64:].hex()}}]})
        return VR.check_y(p, 0, tile, v, (VR.opening(tree, real, 7, values[7], 64),))

    words = lambda *w: np.array(w, "<u4").tobytes()  # noqa: E731
    assert reason(words(5, 5, 0, 0)) is None
    assert reason(words(5, 5, 0)) == "y[7]: the opened row holds 3 words, the tile's columns end at 4"
    assert reason(words(5, 5, 0) + b"\0\0") == "y[7]: the opened row holds 3 words, the tile's columns end at 4"
    assert reason(b"") == "y[7]: the opened row holds 0 words, the tile's columns end at 4"
    assert reason(words(5, 5, 0, 1 << 16)) == "y[7, 3] is not the unscale of the tile's U"


def _serve_dir(tmp_path: Path, passes: list[Path]) -> Path:
    """A serve's directory naming `passes` (manifests) in order, as ``served_passes.py serve`` writes it."""
    d = tmp_path / "served"
    d.mkdir(exist_ok=True)
    (d / "served.json").write_text(json.dumps({"run": RUN, "passes": [{"pass": i, "units": 2, "tiles": 4, "manifest": str(p)}
                                                                      for i, p in enumerate(passes)]}))
    (d / "coins.json").write_text((passes[0].parent / "coins.json").read_text())
    return d


def test_served_passes_verify_refuses_two_passes_of_one_salt_sharing_a_call_index(tmp_path):
    """Two honest passes of one execution phase whose calls are 0, 1 and 1, 2: call 1's seeds, noise and tickets are the same in
    both, so the run is refused (``repeated-call``) before any pass is replayed, whether served.json or ``--manifest``
    names them."""
    a, b = (proved_pass(tmp_path / name, PROMOTED[0], ix)[0] for name, ix in (("a", (0, 1)), ("b", (1, 2))))
    assert json.loads(a.read_text())["salt"] == json.loads(b.read_text())["salt"]
    for d, more in ((_serve_dir(tmp_path, [a, b]), ()), (a.parent, ("--manifest", str(a), "--manifest", str(b)))):
        rc, doc, answer = served_verify(d, tmp_path / d.name / "verify.json", *more)
        assert rc == 1 and answer["verdict"] == "reject" and "verdicts" not in doc and doc["refused"][0] == "repeated-call"
        assert "passes 0 and 1 both hold call index 1" in doc["refused"][1]
        assert [x.split(" sha256:")[0].split(":", 1)[1] for x in doc["passes"]] == [str(a), str(b)]


@pytest.mark.slow
def test_served_passes_verify_accepts_an_epochs_passes_numbered_on(tmp_path):
    """Two honest passes of one execution phase numbered on, 0, 1 then 2, 3, as the device numbers a serve's passes: one draw over
    both, each pass's exclusion tile opened, accepted, its MACs over both (a test instance's, credited nothing)."""
    a, b = (proved_pass(tmp_path / name, PROMOTED[0], ix)[0] for name, ix in (("a", (0, 1)), ("b", (2, 3))))
    rc, doc, answer = served_verify(_serve_dir(tmp_path, [a, b]), tmp_path / "verify.json")
    assert rc == 0 and answer["verdict"] == "accept" and "refused" not in doc
    assert answer["profile"]["exclusion_tiles"] == 2 and len(answer["public"]["passes"]) == 2
    one = VR.credit_at_assumption([VR.load(a, published(a))], 3, Fraction(1, 100))
    assert answer["public"]["work"]["macs"] == 2 * one["macs"]


@pytest.mark.slow
def test_verify_passes_reads_a_test_instance_only_as_one_and_a_runs_passes_of_one_salt(tmp_path):
    """A test instance's pass (k_min 32, at 16 x 16 tiles or at the served 64 x 64) is refused as a served run's, whose
    instance is k_min 1,024 at 64 x 64, and accepted as a test instance's, crediting nothing.  Passes of two salts (other
    weights under the one coin) are refused, though each holds calls 0 and 1: `passes_distinct` holds call indices apart
    per salt."""
    a = proved_pass(tmp_path / "a", PROMOTED[0])[0]
    wide = proved_pass(tmp_path / "wide", PROMOTED[0], tile=(64, 64))[0]
    for path, tile in ((a, 16), (wide, 64)):
        coins, source = published(path), b"verifier" * 4
        served = VR.verify_passes([path], 3, 1, "test", coins, source)
        assert served["answer"]["reason"] == (f"ValueError: pass 0's instance is k_min 32 at {tile} x {tile} tiles, not the "
                                              "served 1024 at 64 x 64: a test instance is read only as one, crediting "
                                              "nothing") and "verdicts" not in served
        test = VR.verify_passes([path], 3, 1, "test", coins, source, test_instance=True)
        work = test["answer"]["public"]["work"]
        assert test["answer"]["verdict"] == "accept" and work["credited"] == 0 < work["macs"], test["line"]
    coins, source = published(a), b"verifier" * 4
    b = proved_pass(tmp_path / "b", PROMOTED[0], seed=12)[0]
    assert json.loads(a.read_text())["salt"] != json.loads(b.read_text())["salt"]
    for kw in ({}, {"test_instance": True}):
        two = VR.verify_passes([a, b], 3, 1, "test", coins, source, **kw)
        assert two["answer"]["reason"] == "ValueError: the passes are not all of one salt" and "refused" not in two


def test_a_runs_credit_at_pearls_assumption_is_its_credited_macs_less_epsilon(tmp_path):
    """`verify_run.epsilon` is the least ε (on its 10^-9 grid) with (1 − ε)^t ≤ δ, and, with X_O draws over the
    inert-work bound, with P[Binomial(t, ε) ≤ X_O] ≤ δ; a run's credit is at most (1 − ρ)(1 − ε) of its credited cells'
    MACs (each at |K|), the excluded row's and the filler rows' cells credited nothing."""
    eps = VR.epsilon(459, Fraction(1, 100))
    assert (1 - eps) ** 459 <= Fraction(1, 100) < (1 - eps + Fraction(2, 10 ** 9)) ** 459 and abs(eps - Fraction(1, 100)) < Fraction(1, 10 ** 4)

    def miss(e: Fraction, x: int) -> Fraction:
        return sum(math.comb(459, i) * e ** i * (1 - e) ** (459 - i) for i in range(x + 1))

    for x in (1, 3):
        ex = VR.epsilon(459, Fraction(1, 100), x)
        assert miss(ex, x) <= Fraction(1, 100) < miss(ex - Fraction(2, 10 ** 9), x) and ex > eps
    path = proved_pass(tmp_path, PROMOTED[0])[0]
    p = VR.load(path, published(path))
    got = VR.credit_at_assumption([p], 459, Fraction(1, 100))
    assert got["macs"] == ((20 - 1) * 16 + 16 * 32) * 128 and got["epsilon"] == str(eps)
    assert 0 < got["credited"] <= (1 - p.scheme.cap) * (1 - eps) * got["macs"]


def test_served_passes_verify_credits_the_timed_call_its_passes_less_the_runs_whole_deduction(tmp_path):
    """Where served.json says which passes the timed call made, an accepted verdict also carries the credit over those
    alone (`served_passes.timed_credit`): the run's credited MACs less the warm-up pass's credited cells' MACs at (1 − ρ),
    rounded up, so the run's deduction for ε falls wholly on the timed pass."""
    SP = load("pearl_c_vllm_served_passes", KERNEL.parent / "pearl_c_vllm" / "served_passes.py")
    a, b = (proved_pass(tmp_path / name, PROMOTED[0], ix)[0] for name, ix in (("a", (0, 1)), ("b", (2, 3))))
    served = {"run": RUN, "passes": [{"pass": 0, "timed": False, "units": 2, "tiles": 4, "manifest": str(a)},
                                     {"pass": 1, "timed": True, "units": 2, "tiles": 4, "manifest": str(b)}]}
    work = VR.credit_at_assumption([VR.load(x, published(x)) for x in (a, b)], SP.DRAWS, Fraction(1, 100))
    warm = VR.pass_macs(VR.load(a, published(a)))
    got = SP.timed_credit(served, work, published(a))
    assert warm == work["macs"] // 2 and got["passes"] == 1 and got["macs"] == work["macs"] - warm
    assert got["credited"] == work["credited"] - math.ceil((1 - Fraction(work["rho"])) * warm) > 0
    few = VR.credit_at_assumption([VR.load(x, published(x)) for x in (a, b)], 3, Fraction(1, 100))
    assert few["credited"] < (1 - Fraction(few["rho"])) * warm and SP.timed_credit(served, few, published(a))["credited"] == 0
    assert SP.timed_credit({**served, "passes": [{**p, "timed": True} for p in served["passes"]]}, work, published(a)) == {
        "passes": 2, "macs": work["macs"], "credited": work["credited"]}


def test_served_passes_verify_refuses_a_serve_that_retained_no_pass(tmp_path):
    """A serve run with ``--no-retain`` wrote no pass's manifest, so ``verify`` refuses it (``unretained``), replays
    nothing, and its core answer rejects with that reason."""
    d, out = tmp_path / "served", tmp_path / "verify.json"
    d.mkdir()
    (d / "served.json").write_text(json.dumps({"run": RUN, "retained": False,
                                               "passes": [{"pass": i, "units": 2, "tiles": 4, "manifest": None, "macs": None}
                                                          for i in range(3)]}))
    rc, doc, answer = served_verify(d, out)
    assert rc == 1 and doc["refused"][0] == "unretained" and "verdicts" not in doc and doc["line"].startswith("REJECT")
    assert answer["verdict"] == "reject" and answer["reason"].startswith("unretained: the serve retained no pass")


def test_a_seeded_serve_draws_the_same_coins_and_salt_key_and_takes_no_retention_or_live_coins(tmp_path, monkeypatch):
    """`serve --seeded`'s streams (`served_passes.seeded`) give one label's bytes however they are drawn, so two seeded
    installs (`seed_install`) hold challengers of one coin commitment and take one hm96 salt key (`pearl_c.new_salt_key`,
    which the install's weight commitment calls); ``--seeded`` is refused beside retention and beside the challenger gate,
    before anything is built."""
    SP = load("pearl_c_vllm_served_passes", KERNEL.parent / "pearl_c_vllm" / "served_passes.py")
    a, b = SP.seeded("coins"), SP.seeded("coins")
    assert a(3) + a(70) + a(1) == b(74) and SP.seeded("salt key")(32) != SP.seeded("coins")(32)
    monkeypatch.setattr(PPD, "new_salt_key", PPD.new_salt_key)
    one, two = {"coins": None}, {}
    SP.seed_install(one)
    SP.seed_install(two)
    assert one["coins"].commitment.to_json() == two["coins"].commitment.to_json()
    assert PPD.new_salt_key(SCHEME) == SP.seeded("salt key")(32)
    argv = ["serve", "--seeded", "--passes", str(tmp_path), "--out", str(tmp_path / "served.json")]
    with pytest.raises(SystemExit, match="takes --no-retain and no PEARLC_CHALLENGER_GATE"):
        SP.main(argv)
    monkeypatch.setenv("PEARLC_CHALLENGER_GATE", "ssh research@gate")
    with pytest.raises(SystemExit, match="takes --no-retain and no PEARLC_CHALLENGER_GATE"):
        SP.main([*argv, "--no-retain"])
    assert not (tmp_path / "served.json").exists()


def test_served_passes_same_holds_two_seeded_serves_of_one_load_to_the_same_passes_naming_no_value(tmp_path, capsys):
    """`same`: two seeded serves of one load (spec, arm, requests, tokens and batched tokens) and one salt are the same
    when every pass's `KEEP` fields are; a pass that differs is named with its fields, never their values, and a serve
    with another pass count fails.  Serves that aren't both seeded and unretained, that served different loads or that
    drew different salts are refused, and so are two whose passes committed nothing, which would otherwise be the same
    for want of any root."""
    SP = load("pearl_c_vllm_served_passes", KERNEL.parent / "pearl_c_vllm" / "served_passes.py")
    passes = [{"pass": i, "timed": i > 0, "units": 64, "base": 64 * i, "tiles": 5888, "screened_rows": 8192,
               "failing_rows": 0, "unsure_rows": 0, "excluded_rows": 0, "transcript_root": f"{7 + i:0128x}",
               "activation_roots": [f"{100 + 3 * i + j:0128x}" for j in range(3)], "schedule": "serial"} for i in range(3)]

    def doc(**kw):
        return {"spec": "s.toml", "arm": "pouw:pearl-c-h100-v1-h3", "requests": 8, "prompt_tokens": 1024,
                "output_tokens": 1, "engine": {"max_num_batched_tokens": 8192}, "retained": False, "seeded": True,
                "coins": {"source": "seeded", "salt": "ab" * 32}, "passes": passes, **kw}

    def same(a, b):
        paths = [tmp_path / "a.json", tmp_path / "b.json"]
        for p, d in zip(paths, (a, b)):
            p.write_text(json.dumps(d))
        rc = SP.main(["same", *map(str, paths), "--out", str(tmp_path / "same.json")])
        text = (tmp_path / "same.json").read_text()
        assert json.loads(text)["line"] == capsys.readouterr().out.strip()
        return rc, json.loads(text), text

    rc, got, _ = same(doc(), doc())
    assert rc == 0 and got["ok"] and got["passes"] == [3, 3] and got["calls"] == 192 and got["differ"] == []
    assert got["line"].startswith("same: two seeded serves of one load (8 requests of 1024 tokens, 1 out, 8192 a step): "
                                  "3 passes (3 with calls), 192 calls") and got["line"].endswith("-> OK")
    empty = {**passes[0], "pass": 3, "units": 0, "tiles": 0, "transcript_root": None, "activation_roots": None}
    rc, got, _ = same(doc(passes=[*passes, empty]), doc(passes=[*passes, empty]))
    assert rc == 0 and got["ok"] and "4 passes (3 with calls), 192 calls" in got["line"]
    other = [dict(p) for p in passes]
    other[1]["activation_roots"] = [other[1]["activation_roots"][0], f"{999:0128x}", other[1]["activation_roots"][2]]
    rc, got, text = same(doc(), doc(passes=other))
    assert rc == 1 and got["differ"] == [{"pass": 1, "fields": ["activation_roots"]}] and got["line"].endswith("-> FAIL")
    assert f"{999:0128x}" not in text and passes[0]["transcript_root"] not in text
    rc, got, _ = same(doc(), doc(passes=passes[:2]))
    assert rc == 1 and got["differ"] == [] and "3 passes against 2" in got["line"]
    for b, why in ((doc(seeded=False), "aren't both seeded"), (doc(retained=True), "aren't both seeded"),
                   (doc(requests=16), "different loads"), (doc(engine={"max_num_batched_tokens": 4096}), "different loads"),
                   (doc(coins={"source": "seeded", "salt": "cd" * 32}), "different salts")):
        rc, got, _ = same(doc(), b)
        assert rc == 1 and not got["ok"] and why in got["refused"] and got["line"].startswith("same: refused: "), why
    rootless = [{**p, "transcript_root": None, "activation_roots": None} for p in passes]
    for a in (doc(passes=rootless), doc(passes=[{**passes[0], "activation_roots": []}, *passes[1:]]), doc(passes=[empty]),
              doc(passes=[])):
        rc, got, _ = same(a, a)
        assert rc == 1 and not got["ok"] and "committed nothing to compare" in got["refused"], a["passes"]


def test_served_passes_counts_a_passs_linear_macs_and_verify_run_its_credited_cells_macs(tmp_path):
    """`pearl_c.window.macs` (served.json's) is a pass's real rows × n × k, and `verify_run.pass_macs` the MACs of the cells
    `clear_work` credits (`UnitRecord.credited` over the scheme's tiles, at kept K), `credit_at_assumption`'s ``macs`` over
    the pass alone: a filler row and an excluded A row each take their cells out, and an excluded B row its block of 64
    columns' (the band rule)."""
    from verity_vllm.protocols.pouw.pearl_c.window import macs

    path = proved_pass(tmp_path, PROMOTED[0])[0]
    man = json.loads(path.read_text())
    p = VR.load(path, published(path))
    assert macs(str(path)) == sum(u["real_rows"] * u["n"] * u["k"] for u in man["matmuls"]) and macs(None) is None
    assert any(u["excluded_a"] for u in man["matmuls"]) and any(u["real_rows"] < u["m"] for u in man["matmuls"])
    assert VR.pass_macs(p) == VR.credit_at_assumption([p], 459, Fraction(1, 100))["macs"] < macs(str(path))
    r = p.records[0]
    blocks = {j // PW.BAND for j in (*r.excluded_b, *r.voluntary_b)}
    b = min(j for j in range(r.matmul.shape.n) if j // PW.BAND not in blocks)
    block = sum(1 for j in range(r.matmul.shape.n) if j // PW.BAND == b // PW.BAND)
    more = dataclasses.replace(p, records=[dataclasses.replace(r, excluded_b=(*r.excluded_b, b)), *p.records[1:]])
    assert not r.voluntary_a and block > 1
    assert VR.pass_macs(more) == VR.pass_macs(p) - (r.real_rows - len(r.excluded_a)) * block * r.k_kept


def test_verify_run_refuses_a_call_index_at_2_to_the_32_and_one_held_twice(tmp_path):
    """A pass whose call index is 2^32 (`pearl_c.CALL_RANGE`, past the row seed's range) is refused ``call-range``, and one
    holding an index twice ``repeated-call``, before anything is opened."""
    path = proved_pass(tmp_path, PROMOTED[0])[0]
    man = json.loads(path.read_text())
    for index, code in ((C.CALL_RANGE, "call-range"), (0, "repeated-call")):
        man["matmuls"][1]["index"] = index
        path.write_text(json.dumps(man))
        r = VR.verify(path, 3, 1, "test", b"verifier" * 4, coins=published(path))
        assert not r["accepted"] and r["refused"][0] == code and r["line"].startswith("REJECT") and code in r["line"], r


def _replay_every_tile(p) -> dict[int, str]:
    """Every tile of pass `p` through the verifier's check (`verify_run.replay`): the failing tiles' reasons by tile id."""
    got, _ = VR.replay([p], [(0, ref.matmul, ref.id) for ref in p.verifier.refs], 1)
    return {tid: reason for _, tid, ok, reason, *_ in got if not ok}


def test_a_salt_dead_element_puts_its_row_over_the_cap(tmp_path, monkeypatch):
    """P3's forming term in the tile rule (`pearl_c_debit.forming_elements`): credited row 9 holds one entry 300 times its
    RMS, which keeps the per-row rules and is salt-dead, inert work of fs·|cols|/n = 64 over the row's bound of 41 at
    k = 128 with no flagged atom, so its tile, and it alone, fails; without the term every tile passes."""
    path = proved_pass(tmp_path, "pearl-c-h100-v1-h3", spike=300.0)[0]
    p = VR.load(path, published(path))
    assert _replay_every_tile(p) == {0: "A row 9's replayed debit 64 exceeds its cap 41 (the row cap: ρ·its credited "
                                        "cells' credit)"}
    monkeypatch.setattr(CW.D, "forming_elements", lambda flags, dead, device=None: 0)
    assert _replay_every_tile(p) == {}


def test_volunteer_pass_gives_up_the_rows_over_their_caps_and_every_tile_then_passes(tmp_path):
    """V-EX's remedy on a retained pass before its sampling coin: under the h100 record's zero key a credited row is over its
    own cap (`proved_pass`), so the verifier rejects its tile.  `volunteer_pass` finds the same tiles from the words' counts,
    gives the row up, and rewrites the records and their root alone: the commitment, A's roots and the files are the pass's,
    every tile then passes the verifier's check, and a second run changes nothing."""
    VP = load("pearl_c_vllm_volunteer_pass", KERNEL.parent / "pearl_c_vllm" / "volunteer_pass.py")
    path = proved_pass(tmp_path, "pearl-c-h100-v1-h3", key=bytes(32))[0]
    coins, before = published(path), json.loads(path.read_text())
    p = VR.load(path, coins)
    failed = _replay_every_tile(p)
    assert failed and all("replayed debit" in r and "the row cap" in r for r in failed.values()), failed
    out = VP.volunteer(path, coins, 1)
    after = json.loads(path.read_text())
    assert {x["tile_id"] for x in out["over"]} == set(failed)
    assert out["ok"] and not out["left"] and out["voluntary_a"] and 0 < out["credit_kept"] < 1, out["line"]
    assert after["records_root"] == out["records_root"]["after"] != before["records_root"]
    assert {k: v for k, v in after.items() if k not in ("records_root", "matmuls", "weights")} == \
        {k: v for k, v in before.items() if k not in ("records_root", "matmuls", "weights")}
    for side in ("matmuls", "weights"):
        strip = lambda xs: [{k: v for k, v in x.items() if not k.startswith("voluntary_")}  # noqa: E731
                            for x in (xs.values() if isinstance(xs, dict) else xs)]
        assert strip(after[side]) == strip(before[side])
    assert [u.get("voluntary_a", []) for u in after["matmuls"]] == [out["voluntary_a"].get(str(u["index"]), []) for u in after["matmuls"]]
    assert _replay_every_tile(VR.load(path, coins)) == {}
    again = VP.volunteer(path, coins, 1)
    assert again["ok"] and again["records_root"]["before"] == again["records_root"]["after"] == out["records_root"]["after"]


def test_the_give_up_runs_exactly_while_the_verdict_rejects_a_drawn_credited_row_over_its_cap(tmp_path, capsys):
    """The pipelines give the rows over a debit cap up before the coin exactly where ``served_passes.py volunteer-needed``
    says yes, so it says yes exactly where the verdict they call rejects a run on a drawn tile with a credited row over its
    cap: on a pass with such rows (`proved_pass` under the h100 record's zero key) and a coin that draws them, ``verify``
    (the run's verdict, `served_pearlplus.sh`'s) and `verify_run.verify` (a pass at a time, `glm53_passes.py`'s) reject iff
    it says yes, under their own rule and each ``--over-cap`` they take.  A verifier whose over-cap rule changes moves the
    give-up onto or off the prover's path with it; one that takes no ``over_cap`` refuses any."""
    SP = load("pearl_c_vllm_served_passes", KERNEL.parent / "pearl_c_vllm" / "served_passes.py")
    path = proved_pass(tmp_path, "pearl-c-h100-v1-h3", key=bytes(32))[0]
    coins = published(path)
    p = VR.load(path, coins)
    over = _replay_every_tile(p)
    assert over and all("replayed debit" in r for r in over.values()), over
    d, source = _serve_dir(tmp_path, [path]), b"verifier" * 4
    (d / SP.COIN).write_text(json.dumps({"commitment": VR.run_commitment([p]).hex(), "source": source.hex()}))
    takes = {}
    for verdict, fn in (("run", VR.verify_passes), ("pass", VR.verify)):
        takes[verdict] = "over_cap" in inspect.signature(fn).parameters
        for rule in (None, *(SP.OVER_CAP if takes[verdict] else ())):
            flag = () if rule is None else ("--over-cap", rule)
            capsys.readouterr()
            assert SP.main(["volunteer-needed", "--verdict", verdict, *flag]) == 0
            yes, took = capsys.readouterr().out.split()
            assert yes in ("yes", "no") and took in SP.OVER_CAP and took == (rule or took)
            if verdict == "run":
                _, doc, answer = served_verify(d, tmp_path / "verify.json", "--draws", "32", *flag)
                drawn = {v["tile"] for v in doc["verdicts"]}
                rejects = answer["verdict"] == "reject" and "replayed debit" in answer["reason"]
            else:
                r = VR.verify(path, 32, 1, "test", source, coins=coins, **({} if rule is None else {"over_cap": rule}))
                drawn, rejects = set(r["tiles"]), not r["accepted"] and "replayed debit" in r["line"]
            assert drawn & set(over), (verdict, rule, sorted(drawn), sorted(over))
            assert (yes == "yes") == rejects, (
                f"volunteer-needed --verdict {verdict} says {yes} under over-cap rule {took}, but the verdict "
                f"{'rejects' if rejects else 'does not reject'} a run on a drawn row over its cap: served_passes.over_cap_rule "
                "must name the rule the verdict takes")
    for verdict in (v for v, t in takes.items() if not t):
        capsys.readouterr()
        assert SP.main(["volunteer-needed", "--verdict", verdict, "--over-cap", "reject"]) == 2
        assert "takes no over_cap" in capsys.readouterr().err


class _StandInDebit:
    """`debit.Debit` without a card: `fast_tile.word_debits` (module `ft`) on the whole of A′ by B̃, with the words `bad`
    names in the first call marked non-finite and their counts wrong, and one added to the first call's atoms at `off`."""

    def __init__(self, ft, device, bad=(), off=None):
        self.ft, self.device, self.bad, self.off, self.calls = ft, device, set(bad), off, []

    def words(self, a, b):
        self.calls.append((a.shape, b.shape))
        atoms, zeros = (x.astype(np.uint16) for x in self.ft.word_debits(a.tolist(), b.tolist(), self.device))
        bad = np.zeros(atoms.shape, bool)
        if len(self.calls) == 1:
            for i, j in self.bad:
                bad[i, j], atoms[i, j] = True, atoms[i, j] + 7
            if self.off is not None:
                atoms[self.off] += 1
        return atoms, zeros, bad


def test_volunteer_pass_on_the_device_counts_every_tile_as_the_replay_and_gives_up_the_same_rows(tmp_path):
    """`volunteer_pass.words_device` on a stand-in for the card (`_StandInDebit`): each call's A′ (its real rows) and each
    weight's B̃ formed once as `encode` forms a tile's, every word's counts cut into the tiles in `scheme.tiles`' order with
    filler rows zero, give every tile's counts as the CPU replay (`words`) gives them, so the pass gives up the same rows to
    the same records root.  A tile with a word the device marks non-finite takes the replay's counts, and a count unlike
    the replay's on a cross-checked tile stops the step before the records change."""
    VP = load("pearl_c_vllm_volunteer_pass", KERNEL.parent / "pearl_c_vllm" / "volunteer_pass.py")
    path = proved_pass(tmp_path, "pearl-c-h100-v1-h3", key=bytes(32))[0]
    coins, before = published(path), path.read_text()
    p = VP.VR.load(path, coins)
    want = VP.words(p, 1)
    cubin = tmp_path / "debit_sm90.cubin"
    cubin.write_bytes(b"stand-in")
    # call 0's row 17 is in its second row tile (tile id 1), which no cross-check at every 2^20th tile replays
    stand = _StandInDebit(VP.FT, p.scheme.device, bad=[(17, 3)])
    VP._DEBIT[str(cubin.resolve())] = stand
    VP._B_CODES.clear()
    try:
        got, on = VP.words_device(VP.VR.load(path, coins), 1, cubin, cross_check=1 << 20)
        assert got.keys() == want.keys()
        assert all(np.array_equal(got[t][0], want[t][0]) and np.array_equal(got[t][1], want[t][1]) for t in want)
        assert [a for a, _ in stand.calls] == [(r.real_rows, r.matmul.shape.k) for r in p.records]
        assert on["bad_tiles"] == 1 and on["replayed"] == 2 and on["cross_check"] == 1 << 20
        VP._DEBIT[str(cubin.resolve())] = _StandInDebit(VP.FT, p.scheme.device, off=(0, 0))
        with pytest.raises(RuntimeError, match="differ from the CPU replay's on 1 of 1 tiles"):
            VP.volunteer(path, coins, 1, cubin, 1 << 20)
        assert path.read_text() == before
        VP._DEBIT[str(cubin.resolve())] = _StandInDebit(VP.FT, p.scheme.device)
        out = VP.volunteer(path, coins, 1, cubin, 1)
        assert out["ok"] and out["device"]["replayed"] == len(want) and "counted on the device" in out["line"]
        dev_root = out["records_root"]["after"]
    finally:
        VP._DEBIT.clear()
        VP._B_CODES.clear()
    path.write_text(before)
    cpu = VP.volunteer(path, coins, 1)
    assert cpu["records_root"]["after"] == dev_root != cpu["records_root"]["before"]
    assert (cpu["voluntary_a"], cpu["voluntary_b"]) == (out["voluntary_a"], out["voluntary_b"])


def test_the_run_verdict_credits_no_given_up_row_and_refuses_a_give_up_past_v_ex_or_its_published_root(tmp_path):
    """A pass whose rows over a cap are given up (`volunteer_pass`) is accepted by the run's verdict, which credits no cell
    of a given-up A row's band of 64 rows or a given-up B row's block of 64 columns.  More given-up rows than V-EX allows (⌈m/64⌉ of a call's A), or records changed after the
    pass published their root, the verdict refuses before it takes its coin."""
    VP = load("pearl_c_vllm_volunteer_pass", KERNEL.parent / "pearl_c_vllm" / "volunteer_pass.py")
    path = proved_pass(tmp_path, "pearl-c-h100-v1-h3", key=bytes(32))[0]
    coins = published(path)
    assert VP.volunteer(path, coins, 1)["ok"]
    good = VR.verify_passes([path], 3, 1, "test", coins, b"verifier" * 4, test_instance=True)
    assert good["answer"]["verdict"] == "accept", good["line"]
    p = VR.load(path, coins)
    assert any(r.voluntary_a or r.voluntary_b for r in p.records)
    def band_rule(r):
        bands, blocks = {i // PW.BAND for i in r.voluntary_a}, {j // PW.BAND for j in (*r.excluded_b, *r.voluntary_b)}
        rows = sum(1 for i in range(r.real_rows) if i not in r.excluded_a and i // PW.BAND not in bands)
        return r.k_kept * rows * sum(1 for j in range(r.matmul.shape.n) if j // PW.BAND not in blocks)
    kept = sum(band_rule(r) for r in p.records)
    whole = sum(r.k_kept * (r.real_rows - len(r.excluded_a)) * (r.matmul.shape.n - len(r.excluded_b)) for r in p.records)
    assert good["answer"]["public"]["work"]["macs"] == kept < whole
    assert good["answer"]["public"]["work"] == {**VR.credit_at_assumption([p], 3, VR.DELTA), "credited": 0}
    man = json.loads(path.read_text())
    u = next(u for u in man["matmuls"] if u.get("voluntary_a"))
    for given, why in (([i for i in range(u["real_rows"]) if i not in u["excluded_a"]][:CW.voluntary_cap(u["m"]) + 1],
                        "more voluntary rows than V-EX allows"),
                       ([], "not the ones whose root the pass published")):
        path.write_text(json.dumps({**man, "matmuls": [{**x, "voluntary_a": given} if x is u else x for x in man["matmuls"]]}))
        r = VR.verify_passes([path], 3, 1, "test", coins, b"verifier" * 4, test_instance=True)
        assert r["answer"]["verdict"] == "reject" and why in r["answer"]["reason"] and "source" not in r, r["line"]


def window_record(*paths: Path, version: int = 2) -> dict:
    """Pearl+'s window record W over passes `paths`, pass s its round s, as the tie writes it: a v2 record at W's
    ``version`` whose registered values are each call's A rows (m16 rows of K FP32 words, A's domain) and y (its real rows
    of N, y's domain), then at version 2 each weight's B rows once (its n rows of K FP32 words at PoUW's root, its
    weights-tree domain), beside a value of another kind the verdict doesn't read. At version 1 its header states no
    version and it registers no weight, as a W written before W registered them (the run of record's) does."""
    reg, weights = [], {}
    for s, path in enumerate(paths):
        man = json.loads(path.read_text())
        salt = bytes.fromhex(man["salt"])
        for c, u in enumerate(man["matmuls"]):
            a = rows_domain(activation_binding(salt, u["index"]), u["index"], u["m"], HM96)
            reg += [{"value": f"pearl-a/{s}/{c}", "root": u["root_a"], "leaves": u["m"], "words": u["k"], "word_bits": 32,
                     "domain": a.domain_id.hex()},
                    {"value": f"pearl-y/{s}/{c}", "root": u["y"]["root"], "leaves": u["real_rows"], "words": u["n"],
                     "word_bits": 32, "domain": SY.y_domain(salt, u["index"], u["real_rows"]).domain_id.hex()}]
            weights.setdefault(u["weight"], {
                "value": f"pearl-b/{u['weight']}", "root": man["weights"][str(u["weight"])]["root"], "leaves": u["n"],
                "words": u["k"], "word_bits": 32,
                "domain": rows_domain(weight_binding(u["weight"]), -2, u["n"], HM96).domain_id.hex()})
    return {"format": REG.FORMAT_V2, "window": {"kind": "served", **({"version": 2} if version == 2 else {})},
            "registered": [*reg, *(weights[w] for w in sorted(weights) if version == 2),
                           {"value": "input-ids/0", "root": "00", "leaves": 1, "words": 1, "domain": "00"}]}


def test_the_run_verdict_holds_its_window_record_to_each_calls_a_and_y_and_each_weights_b_root_and_domain(tmp_path):
    """``served_passes.py verify --tie W`` accepts a run whose window record registers each call's A rows and y at the roots
    the verdict credits, under the domains it recomputes from the salt and the call's index, and each weight's B rows at
    the root the epoch commits it under (PoUW's weights tree), and names W's digest in its answer.  A W whose root, domain
    (another salt's, call's or weight's), rows, words or word bits differ for any call or weight, which misses one, names
    one twice or one the run doesn't hold, or which is not a v2 record, rejects the run before the coin.  So does a
    version-2 W that registers every call right and no weight: it reads as untied, by `UNTIED_WEIGHTS`, through
    ``served_passes.py verify`` too; and a W stating another version, or a version-1 one registering a weight.  A W that
    states no version (version 1, the run of record's: written before W registered the weights) and registers every call
    right is held to the calls alone and accepts, its tie on the calls' tables naming its weights untied
    (`WEIGHTS_UNTIED`), in the answer and the verdict's line."""
    path = proved_pass(tmp_path, PROMOTED[0])[0]
    coins, w, man = published(path), window_record(path), json.loads(path.read_text())
    (wfile := tmp_path / "w.json").write_text(json.dumps(w))
    rc, doc, answer = served_verify(path.parent, tmp_path / "verify.json", "--manifest", str(path), "--tie", str(wfile))
    assert rc == 0 and answer["public"]["tie"] == {"record": REG.digest(w), "tables": 6}, doc["line"]
    assert f"tied to W {REG.digest(w)[:16]} on its 6 tables" in doc["line"] and doc["tie_record"] == str(wfile)
    salt, (u0, u1) = bytes.fromhex(man["salt"]), man["matmuls"]
    b0, b1 = (next(e for e in w["registered"] if e["value"] == f"pearl-b/{x}") for x in (0, 1))
    n1 = b1["leaves"]
    legacy = {**w, "registered": [e for e in w["registered"] if not e["value"].startswith("pearl-b/")]}
    old = window_record(path, version=1)

    def at(v: str, **kw) -> dict:
        return {**w, "registered": [{k: x for k, x in {**e, **kw}.items() if x is not None} if e["value"] == v else e
                                    for e in w["registered"]]}
    cases = [(at("pearl-a/0/0", root=u1["root_a"]), "W's pearl-a/0/0: its root not the call's"),
             (at("pearl-y/0/1", domain=SY.y_domain(bytes(32), u1["index"], u1["real_rows"]).domain_id.hex()),
              "W's pearl-y/0/1: its domain"),
             (at("pearl-a/0/1", domain=rows_domain(activation_binding(salt, u0["index"]), u0["index"], u1["m"], HM96)
                 .domain_id.hex()), "W's pearl-a/0/1: its domain"),
             (at("pearl-a/0/0", leaves=u0["real_rows"]), "W's pearl-a/0/0: its leaves"),
             (at("pearl-y/0/0", words=u0["k"]), "W's pearl-y/0/0: its words"),
             (at("pearl-y/0/0", word_bits=None), "W's pearl-y/0/0: its word_bits"),
             ({**w, "registered": [e for e in w["registered"] if e["value"] != "pearl-y/0/1"]}, "W registers no pearl-y/0/1"),
             ({**w, "registered": [*w["registered"], {**w["registered"][0], "value": "pearl-a/1/0"}]},
              "W registers pearl-a/1/0, no call of the run's"),
             ({**w, "registered": [*w["registered"], w["registered"][0]]}, "W registers pearl-a/0/0 twice"),
             (at("pearl-b/0", root=man["weights"]["1"]["root"]), "W's pearl-b/0: its root not the weight's"),
             (at("pearl-b/1", domain=rows_domain(weight_binding(0), -2, n1, HM96).domain_id.hex()),
              "W's pearl-b/1: its domain"),
             (at("pearl-b/0", leaves=n1), "W's pearl-b/0: its leaves"),
             ({**w, "registered": [e for e in w["registered"] if e["value"] != "pearl-b/1"]}, "W registers no pearl-b/1"),
             ({**w, "registered": [*w["registered"], b0]}, "W registers pearl-b/0 twice"),
             ({**w, "registered": [*w["registered"], {**b0, "value": "pearl-b/2"}]},
              "W registers pearl-b/2, no weight of the run's"),
             (legacy, f"{VR.UNTIED_WEIGHTS}, so the run's rootB is no W's (pearl-b/0 first)"),
             ({**old, "registered": [*old["registered"], b0]},
              f"{VR.V1_WEIGHTS} pearl-b/0: a W that registers the weights states version 2"),
             ({**w, "window": {"kind": "served", "version": 1}}, PWR.UNKNOWN_VERSION),
             ({**w, "window": {"kind": "served", "version": 3}}, PWR.UNKNOWN_VERSION),
             ({**w, "format": REG.FORMAT}, f"W is no {REG.FORMAT_V2} record")]
    for bad, why in cases:
        r = VR.verify_passes([path], 3, 1, "test", coins, b"verifier" * 4, window=bad, test_instance=True)
        assert r["answer"]["verdict"] == "reject" and why in r["answer"]["reason"] and "source" not in r, (why, r["line"])
    (lfile := tmp_path / "legacy.json").write_text(json.dumps(legacy))
    rc, doc, answer = served_verify(path.parent, tmp_path / "legacy-verify.json", "--manifest", str(path), "--tie", str(lfile))
    assert rc != 0 and answer["verdict"] == "reject" and VR.UNTIED_WEIGHTS in answer["reason"], doc["line"]
    untied = {"record": REG.digest(old), "tables": 4, "weights": VR.WEIGHTS_UNTIED}
    r = VR.verify_passes([path], 3, 1, "test", coins, b"verifier" * 4, window=old, test_instance=True)
    assert r["answer"]["verdict"] == "accept" and r["answer"]["public"]["tie"] == untied, r["line"]
    (ofile := tmp_path / "v1.json").write_text(json.dumps(old))
    rc, doc, answer = served_verify(path.parent, tmp_path / "v1-verify.json", "--manifest", str(path), "--tie", str(ofile))
    assert rc == 0 and answer["public"]["tie"] == untied, doc["line"]
    assert f"on its 4 tables, {VR.WEIGHTS_UNTIED}" in doc["line"]


def test_retain_writes_w_over_the_retained_timed_rounds_and_the_tied_verdict_credits_them(tmp_path):
    """``retain --window-record W`` on a serve of a warm-up pass, a pass without calls and two timed passes: it drops the
    warm-up's round 0, retains rounds 1 and 2 (places among served.json's passes with calls), and writes W over those two
    under their numbers, so ``verify --tie W`` accepts the run tied to W's digest on its 8 tables and credits the timed
    passes alone.  The warm-up retained, W can't tie it (W registers none of round 0, and `pearl_c.window.write` refuses
    to); once the coin is drawn ``retain`` refuses to write W again."""
    SP = load("pearl_c_vllm_served_passes", KERNEL.parent / "pearl_c_vllm" / "served_passes.py")
    warm, a, b = (proved_pass(tmp_path / name, PROMOTED[0], ix)[0] for name, ix in (("w", (0, 1)), ("a", (2, 3)), ("b", (4, 5))))
    d = _serve_dir(tmp_path, [warm, a, b])
    served = json.loads((d / "served.json").read_text())
    served["passes"] = [{**served["passes"][0], "timed": False}, {"pass": 1, "units": 0, "tiles": 0},
                        *({**p, "pass": p["pass"] + 1, "timed": True} for p in served["passes"][1:])]
    (d / "served.json").write_text(json.dumps(served))
    wfile, coins = tmp_path / "w.json", published(warm)
    assert SP.main(["retain", "--passes", str(d)]) == 0 and not wfile.exists()
    assert json.loads((d / SP.RETENTION).read_text())["rounds"] == [0, 1, 2]
    assert SP.main(["retain", "--passes", str(d), "--window-record", str(wfile)]) == 0
    got = json.loads((d / SP.RETENTION).read_text())
    assert got["rounds"] == [1, 2] and [(x["round"], x["why"]) for x in got["dropped"]] == [(0, SP.WARM_UP)]
    w = json.loads(wfile.read_text())
    calls = [e["value"] for e in w["registered"] if e["value"].startswith(("pearl-a/", "pearl-y/"))]
    assert sorted({v.split("/")[1] for v in calls}) == ["1", "2"] and len(calls) == 8
    assert [e["value"] for e in w["registered"]][8:] == ["pearl-b/0", "pearl-b/1"]
    r = VR.verify_passes([warm, a, b], 3, 1, "test", coins, b"verifier" * 4, window=w, test_instance=True)
    assert r["answer"]["verdict"] == "reject" and "W registers no pearl-a/0/0" in r["answer"]["reason"], r["line"]
    rc, doc, answer = served_verify(d, tmp_path / "verify.json", "--tie", str(wfile))
    assert rc == 0 and answer["public"]["tie"] == {"record": REG.digest(w), "tables": 10}, doc["line"]
    assert (answer["public"]["rounds"], answer["public"]["dropped"], doc["timed"]["passes"]) == ([1, 2], [0], 2)
    before = wfile.read_text()
    assert SP.main(["retain", "--passes", str(d), "--window-record", str(wfile)]) == 1 and wfile.read_text() == before


def test_served_passes_volunteer_gives_up_each_retained_passs_rows_over_a_cap_and_refuses_an_unretained_serve(tmp_path):
    """``served_passes.py volunteer``, the prover's step between the serve and the verifier's coin, runs `volunteer_pass` on
    each pass with calls that served.json names: a pass with a credited row over its cap (`proved_pass` under the h100
    record's zero key) is left with none and its records' root is rewritten, so every tile then passes the verifier's check;
    its document names the node it ran on; a serve that retained no pass is refused."""
    SP = load("pearl_c_vllm_served_passes", KERNEL.parent / "pearl_c_vllm" / "served_passes.py")
    path = proved_pass(tmp_path, "pearl-c-h100-v1-h3", key=bytes(32))[0]
    d, out = _serve_dir(tmp_path, [path]), tmp_path / "volunteer.json"
    before = json.loads(path.read_text())["records_root"]
    assert SP.main(["volunteer", "--passes", str(d), "--out", str(out), "--jobs", "1"]) == 0
    doc = json.loads(out.read_text())
    assert doc["ok"] and [p["manifest"] for p in doc["passes"]] == [str(path)] and doc["over"] and not doc["left"]
    assert (doc["node"], doc["done"], doc["of"]) == (socket.gethostname().split(".")[0], True, 1)
    assert doc["voluntary_a"] + doc["voluntary_b"] > 0
    assert json.loads(path.read_text())["records_root"] == doc["passes"][0]["records_root"]["after"] != before
    assert _replay_every_tile(VR.load(path, published(path))) == {}
    (d / "served.json").write_text(json.dumps({**json.loads((d / "served.json").read_text()), "retained": False}))
    assert SP.main(["volunteer", "--passes", str(d), "--out", str(out), "--jobs", "1"]) == 1


def test_served_passes_volunteer_on_the_device_gives_up_the_cpus_rows_and_names_its_cubin_and_cross_check(tmp_path):
    """``served_passes.py volunteer --device`` (a stand-in for the card, `_StandInDebit`) gives up the rows the CPU replay
    gives up, to the same records root, and its document names the debit cubin, the cross-check rate (``--device-check``,
    else `volunteer_pass.CROSS_CHECK`) and the tiles it replayed on the CPU too; the CPU's document has no ``device``."""
    SP = load("pearl_c_vllm_served_passes", KERNEL.parent / "pearl_c_vllm" / "served_passes.py")
    VP = load("volunteer_pass", KERNEL.parent / "pearl_c_vllm" / "volunteer_pass.py")
    path = proved_pass(tmp_path, "pearl-c-h100-v1-h3", key=bytes(32))[0]
    d, out, before = _serve_dir(tmp_path, [path]), tmp_path / "volunteer.json", path.read_text()
    cubin = tmp_path / "debit_sm90.cubin"
    cubin.write_bytes(b"stand-in")
    device = VP.VR.load(path, published(path)).scheme.device
    got = {}
    try:
        for check in ("2", None):
            path.write_text(before)
            VP._DEBIT[str(cubin.resolve())] = _StandInDebit(VP.FT, device)
            VP._B_CODES.clear()
            more = ["--device-check", check] if check else []
            assert SP.main(["volunteer", "--passes", str(d), "--out", str(out), "--jobs", "1", "--device", str(cubin),
                            *more]) == 0
            got[check] = json.loads(out.read_text()), json.loads(path.read_text())["records_root"]
    finally:
        VP._DEBIT.clear()
        VP._B_CODES.clear()
    path.write_text(before)
    assert SP.main(["volunteer", "--passes", str(d), "--out", str(out), "--jobs", "1"]) == 0
    cpu, root = json.loads(out.read_text()), json.loads(path.read_text())["records_root"]
    assert cpu["device"] is None and cpu["voluntary_a"] + cpu["voluntary_b"] > 0
    for check, (doc, dev_root) in got.items():
        rate = int(check) if check else VP.CROSS_CHECK
        assert doc["ok"] and dev_root == root and doc["tiles"] == cpu["tiles"]
        assert {k: doc[k] for k in ("voluntary_a", "voluntary_b", "over", "left")} == {
            k: cpu[k] for k in ("voluntary_a", "voluntary_b", "over", "left")}
        assert {k: doc["device"][k] for k in ("cubin", "cubin_sha256", "cross_check", "replayed", "bad_tiles")} == {
            "cubin": str(cubin), "cubin_sha256": hashlib.sha256(b"stand-in").hexdigest(), "cross_check": rate,
            "replayed": len(range(0, doc["tiles"], rate)), "bad_tiles": 0}
        assert doc["passes"][0]["device"]["replayed"] == doc["device"]["replayed"]


def test_served_passes_verify_credits_only_the_passes_retained_before_its_coin_and_refuses_a_retention_after_it(tmp_path):
    """``retain`` drops passes before the coin (one ``--drop``, and with ``--volunteer`` each pass that step didn't finish or
    left over a cap): ``verify`` then reads the retained passes alone, each at its round, credits only them, and records
    its coin against their commitment (``coin.json``).  The same retention again takes the same source, never another;
    ``retain`` and ``volunteer`` refuse once the coin is drawn, and a retention changed behind them is refused
    (``after-coin``) before any replay, whichever way it changed."""
    SP = load("pearl_c_vllm_served_passes", KERNEL.parent / "pearl_c_vllm" / "served_passes.py")
    a, b = (proved_pass(tmp_path / name, PROMOTED[0], ix)[0] for name, ix in (("a", (0, 1)), ("b", (2, 3))))
    d = _serve_dir(tmp_path, [a, b])
    vol = tmp_path / "volunteer.json"
    for passes, why in (([{"manifest": str(a), "ok": True, "left": []}], "did not finish before the coin"),
                        ([{"manifest": str(a), "ok": True, "left": []}, {"manifest": str(b), "ok": False, "left": [7]}],
                         "1 tiles left over a cap")):
        vol.write_text(json.dumps({"passes": passes}))
        assert SP.main(["retain", "--passes", str(d), "--volunteer", str(vol)]) == 0
        got = json.loads((d / SP.RETENTION).read_text())
        assert got["rounds"] == [0] and [x["round"] for x in got["dropped"]] == [1] and why in got["dropped"][0]["why"]
    assert SP.main(["retain", "--passes", str(d), "--drop", "2"]) == 1
    assert SP.main(["retain", "--passes", str(d), "--drop", "0"]) == 0
    rc, doc, answer = served_verify(d, tmp_path / "verify.json")
    one = VR.credit_at_assumption([VR.load(b, published(b))], 3, Fraction(1, 100))
    work = answer["public"]["work"]
    assert rc == 0 and work == {**one, "credited": 0}, doc["line"]
    assert (answer["public"]["rounds"], answer["public"]["dropped"], doc["dropped"]) == ([1], [0], [0])
    assert doc["passes"][0].split(" sha256:")[0].split(":", 1)[1] == str(b) and "1 passes dropped before the coin (rounds 0)" in doc["line"]
    assert doc["timed"] == {"passes": 1, "macs": one["macs"], "credited": 0}
    coin = json.loads((d / SP.COIN).read_text())
    assert (coin["commitment"], coin["source"], coin["rounds"]) == (answer["commitment"], doc["source"], [1])
    again = served_verify(d, tmp_path / "again.json")[1]
    assert again["source"] == doc["source"] and again["answer"] == answer
    before = (d / SP.RETENTION).read_text()
    assert SP.main(["retain", "--passes", str(d)]) == 1 and (d / SP.RETENTION).read_text() == before
    assert SP.main(["volunteer", "--passes", str(d), "--out", str(vol), "--jobs", "1"]) == 1
    for rounds in ([0, 1], [0]):
        (d / SP.RETENTION).write_text(json.dumps({"rounds": rounds}))
        rc, doc, answer = served_verify(d, tmp_path / "verify.json")
        assert rc == 1 and doc["refused"][0] == "after-coin" and "source" not in doc and "verdicts" not in doc, doc["line"]
    (d / SP.RETENTION).write_text(json.dumps({"rounds": [1, 0]}))
    assert "not distinct rounds of the served run's 2, in order" in served_verify(d, tmp_path / "verify.json")[2]["reason"]
    assert json.loads((d / SP.COIN).read_text()) == coin


def test_the_run_verdict_refuses_a_give_up_made_after_its_coin(tmp_path):
    """V-EX's give-up is the prover's before the coin: a pass whose records change after the verifier recorded its coin
    (`volunteer_pass` run past ``served_passes.py volunteer``'s refusal) is another run, and the verdict refuses it
    (``after-coin``) rather than draw again over it."""
    VP = load("pearl_c_vllm_volunteer_pass", KERNEL.parent / "pearl_c_vllm" / "volunteer_pass.py")
    path = proved_pass(tmp_path, "pearl-c-h100-v1-h3", key=bytes(32))[0]
    coins, record = published(path), tmp_path / "coin.json"
    first = VR.verify_passes([path], 3, 1, "test", coins, record=record, test_instance=True)
    assert "source" in first and json.loads(record.read_text())["commitment"] == first["commitment"]
    assert VP.volunteer(path, coins, 1)["ok"]
    late = VR.verify_passes([path], 3, 1, "test", coins, record=record, test_instance=True)
    assert late["answer"]["verdict"] == "reject" and late["refused"][0] == "after-coin" and "source" not in late, late["line"]
    assert VR.verify_passes([path], 3, 1, "test", coins, test_instance=True)["answer"]["verdict"] == "accept"


def test_the_run_verdict_takes_its_source_from_its_draw_once_and_never_asks_a_draw_again(tmp_path):
    """``verify_passes``' ``draw(C, asking)``: its source is the draw's coin on the run's commitment C, recorded (v2) with
    the draw before any replay and reused by a later verdict without drawing; a draw that fails before it asks rejects
    and may be asked again, while one that fails after ``asking`` marked the record is refused (``asked-once``) and
    never redrawn."""
    VP = load("pearl_c_vllm_volunteer_pass", KERNEL.parent / "pearl_c_vllm" / "volunteer_pass.py")
    path = proved_pass(tmp_path, "pearl-c-h100-v1-h3", key=bytes(32))[0]
    coins = published(path)
    assert VP.volunteer(path, coins, 1)["ok"]
    root = VR.run_commitment([VR.load(path, coins)])
    calls = []

    def draw(c, asking):
        calls.append(c)
        asking({"reserved": {"session": "s20261104T090000Z-dr01"}})
        return b"\x5e" * 64, {"from": VR.DRAW_CHALLENGER, "session": {"session": "s20261104T090000Z-dr01"}}

    record = tmp_path / "coin.json"
    first = VR.verify_passes([path], 3, 1, "test", coins, record=record, draw=draw, test_instance=True)
    held = json.loads(record.read_text())
    assert first["answer"]["verdict"] == "accept" and first["draw"] == "challenger" and calls == [root]
    assert (held["schema"], held["commitment"], held["source"], held["draw"]["from"]) == (
        VR.COIN_SCHEMA, root.hex(), "5e" * 64, "challenger")
    again = VR.verify_passes([path], 3, 1, "test", coins, record=record, draw=draw, test_instance=True)
    assert again["answer"] == first["answer"] and calls == [root]

    def before(c, asking):
        raise Ch.CoinError("lane vllm's period p1 is open")

    def after(c, asking):
        asking({"reserved": {"session": "s20261104T090000Z-dr02"}})
        raise Ch.CoinError("the gate's open timed out")

    other = tmp_path / "other" / "coin.json"
    r = VR.verify_passes([path], 3, 1, "test", coins, record=other, draw=before, test_instance=True)
    assert r["answer"]["verdict"] == "reject" and "p1 is open" in r["line"] and not other.exists()
    assert not other.with_name(other.name + VR.ASKED).exists()
    r = VR.verify_passes([path], 3, 1, "test", coins, record=other, draw=after, test_instance=True)
    assert r["answer"]["verdict"] == "reject" and "timed out" in r["line"] and not other.exists()
    assert json.loads(other.with_name(other.name + VR.ASKED).read_text())["commitment"] == root.hex()
    r = VR.verify_passes([path], 3, 1, "test", coins, record=other, draw=draw, test_instance=True)
    assert r["refused"][0] == "asked-once" and r["answer"]["verdict"] == "reject" and calls == [root]
    node = VR.verify_passes([path], 3, 1, "test", coins, record=tmp_path / "node" / "coin.json", test_instance=True)
    assert node["draw"] == "node" and json.loads((tmp_path / "node" / "coin.json").read_text())["draw"] == {
        "from": "node", "claim": VR.NODE_CLAIM}


def test_the_live_draw_reserves_a_withheld_draw_session_outside_any_period_and_opens_it_on_the_runs_commitment(monkeypatch):
    """`bench_arm.LiveDraw` (``served_passes.py verify`` under PEARLC_CHALLENGER_GATE/PEER/LANE): with no period of the
    lane's open, it reserves a draw session withheld (``reserve --peer P --scheme S --families [["pouw/draw",1]]
    --withhold``), marks the record with the reserve before its open, and returns the session's ``pouw/draw`` coin 0
    after C with the record's draw.  It refuses before reserving while the lane's period is open, and before its open
    when the gate reserves it plain or into a period; `live_draw` is None with neither setting and refused with one."""
    from verity_vllm.protocols.pouw.h1 import challenger as PHC
    from verity_vllm.protocols.pouw.h1 import window as PHW

    BA = load("pearl_c_vllm_bench_arm", KERNEL.parent / "pearl_c_vllm" / "bench_arm.py")
    gate, root = "ssh -i KEY -o BatchMode=yes research@gate", b"\xc1" * 64
    reserved = {"session": "s20261104T090000Z-dr01", "seq": 9, "peer": "vy-dc-h200", "by": "vllm", "period": None,
                "withheld": True}
    ledger, asked, opened, marks = [], [], [], []

    def run(argv):
        assert argv[:-1] == shlex.split(gate)
        asked.append(shlex.split(argv[-1]))
        if asked[-1][0] == "ledger":
            return 0, "".join(json.dumps(e) + "\n" for e in ledger)
        return 0, json.dumps(reserved)

    class Session:
        def __init__(self, entry, at, got, call):
            self.reserved, self.challenger = got, Wd.draw_challenger(0, random.Random(7).randbytes)
            self.session = self.commitment = None

        def record(self, c):
            opened.append((self.reserved, c))
            self.session, self.commitment = {"session": self.reserved["session"], "root": c.hex()}, self.challenger.commitment
            return self.challenger.open(Wd.DRAW, 0, c)

    monkeypatch.setattr(PHW, "DrawSession", Session)
    coin, drew = BA.LiveDraw(gate, "vy-dc-h200", "vllm", run)(root, marks.append)
    want = {k: v for k, v in reserved.items() if k != "withheld"}
    assert asked == [["ledger"], ["reserve", "--peer", "vy-dc-h200", "--scheme", PHC.SCHEME, "--families",
                                  PHC.draw_families(), "--withhold"]]
    assert marks == [{"reserved": want}] and opened == [(want, root)]
    assert drew["from"] == "challenger" and (drew["label"], drew["index"]) == (Wd.DRAW, 0) and drew["reserved"] == want
    opening = Ch.LiveOpening.from_json(drew["opening"])
    assert coin == opening.coin and len(coin) == 64 and opening.after == root
    assert Wd.draw_epoch_reason(Ch.LiveCommitment.from_json(drew["commitment"])) is None
    assert drew["session"] == {"session": reserved["session"], "root": root.hex()}

    for case, why in (("plain", "plain, not withheld"), ("period", "joined period p2"), ("open", "period p1 is open")):
        for held in (asked, marks, opened, ledger):
            held.clear()
        reserved.update(withheld=True, period=None)
        if case == "plain":
            reserved["withheld"] = False
        elif case == "period":
            reserved["period"] = "p2"
        else:
            ledger.append({"kind": "period-open", "session": None, "body": {"period": "p1", "lane": "vllm"}})
        with pytest.raises(Ch.CoinError, match=why):
            BA.LiveDraw(gate, "vy-dc-h200", "vllm", run)(root, marks.append)
        assert marks == [] and opened == [] and asked[0] == ["ledger"] and len(asked) == (1 if case == "open" else 2)

    assert BA.live_draw(None, None, "vllm") is None
    for one in ((gate, None), (None, "vy-dc-h200")):
        with pytest.raises(ValueError, match="together"):
            BA.live_draw(*one, "vllm", run)


def test_the_live_draw_after_the_runs_pearl_c_period_is_the_lanes_first_reserve_since_its_close_or_asks_no_coin(monkeypatch):
    """Given the Pearl-C period P the run's salt session joined (``served.json``'s ``coins.reserved.period``), the draw
    is lane vllm's first reserve after P's close, the one a compose takes (`pouw_h1_window.after_close`): it is refused
    before reserving while the ledger doesn't close P or the lane (``for``, else ``by``) reserved since, and before its
    coin is asked when another reserve of the lane reached the gate first; another lane's reserve doesn't count."""
    from verity_vllm.protocols.pouw.h1 import window as PHW

    BA = load("pearl_c_vllm_bench_arm", KERNEL.parent / "pearl_c_vllm" / "bench_arm.py")
    gate, root, sid = "ssh -i KEY -o BatchMode=yes research@gate", b"\xc1" * 64, "s20261104T120000Z-dr01"
    reserve = lambda s, **b: {"kind": "reserve", "session": s, "body": {"by": "vllm", **b}}  # noqa: E731
    salt = [{"kind": "period-open", "session": None, "body": {"period": "pc1", "lane": "vllm"}},
            reserve("s20261104T110000Z-salt", period="pc1")]
    closed = {"kind": "period-close", "session": None, "body": {"period": "pc1"}}
    ledger, asked, marks, opened, first = [], [], [], [], []

    def run(argv):
        asked.append(shlex.split(argv[-1])[0])
        if asked[-1] == "ledger":
            return 0, "".join(json.dumps(e) + "\n" for e in ledger)
        ledger.extend(first)
        ledger.append(reserve(sid, withheld=True))
        return 0, json.dumps({"session": sid, "seq": len(ledger), "peer": "vy-dc-h200", "by": "vllm", "period": None,
                              "withheld": True})

    class Session:
        def __init__(self, entry, at, got, call):
            self.reserved, self.challenger = got, Wd.draw_challenger(0, random.Random(7).randbytes)
            self.session = self.commitment = None

        def record(self, c):
            opened.append(c)
            self.session, self.commitment = {"session": self.reserved["session"], "root": c.hex()}, self.challenger.commitment
            return self.challenger.open(Wd.DRAW, 0, c)

    monkeypatch.setattr(PHW, "DrawSession", Session)
    assert BA.live_draw(gate, "vy-dc-h200", "vllm", run, period="pc1").period == "pc1"
    for case, held_ledger, race, why in (
            ("first", [*salt, closed], [], None),
            ("another lane's", [*salt, closed, reserve("s-pous", by="pous")], [reserve("s-pous2", by="pous")], None),
            ("open", salt, [], "lane vllm's period pc1 is open"),
            ("not in the ledger", [], [], "doesn't close the run's Pearl-C period pc1"),
            ("reserved since", [*salt, closed, reserve("s-h1")], [],
             "lane vllm reserved session s-h1 after its Pearl-C period pc1 closed"),
            ("for the lane", [*salt, closed, reserve("s-for", by="admin", **{"for": "vllm"})], [],
             "reserved session s-for after"),
            ("raced", [*salt, closed], [reserve("s-h1")], f"draw session {sid} is not lane vllm's first reserve after its "
                                                          "Pearl-C period pc1 closed \\(that is s-h1\\): refused before its "
                                                          "coin is asked")):
        ledger[:], first[:] = held_ledger, race
        for held in (asked, marks, opened):
            held.clear()
        draw = BA.LiveDraw(gate, "vy-dc-h200", "vllm", run, period="pc1")
        if why is None:
            coin, drew = draw(root, marks.append)
            assert asked == ["ledger", "reserve", "ledger"] and opened == [root] and drew["reserved"]["session"] == sid
            assert marks == [{"reserved": drew["reserved"]}] and len(coin) == 64, case
            continue
        with pytest.raises(Ch.CoinError, match=why):
            draw(root, marks.append)
        assert marks == [] and opened == [], case
        assert asked == (["ledger", "reserve", "ledger"] if case == "raced" else ["ledger"]), case


def test_a_tie_over_a_retention_holds_each_retained_pass_at_its_round_and_passes_over_the_dropped_rounds(tmp_path):
    """With round 0 dropped before the coin, W over both rounds ties the retained pass at its round 1, its dropped round's
    entries crediting nothing; without the drop those entries are a call the run doesn't hold, and the pass at round 0
    misses its own."""
    a, b = (proved_pass(tmp_path / name, PROMOTED[0], ix)[0] for name, ix in (("a", (0, 1)), ("b", (2, 3))))
    coins, w = published(b), window_record(a, b)
    r = VR.verify_passes([b], 3, 1, "test", coins, b"verifier" * 4, window=w, rounds=[1], dropped=(0,), test_instance=True)
    assert r["answer"]["verdict"] == "accept" and r["answer"]["public"]["tie"]["tables"] == 6, r["line"]
    for rounds, dropped, why in (([1], (), "W registers pearl-a/0/0, no call of the run's"),
                                 ([0], (1,), "W's pearl-a/0/0: its root")):
        r = VR.verify_passes([b], 3, 1, "test", coins, b"verifier" * 4, window=w, rounds=rounds, dropped=dropped, test_instance=True)
        assert r["answer"]["verdict"] == "reject" and why in r["answer"]["reason"] and "source" not in r, (why, r["line"])


def test_a_tie_refuses_two_passes_that_commit_one_weight_under_two_roots(tmp_path):
    """W registers each weight's B rows once, so a tie over passes whose epochs commit weight 0 under two roots (other
    weights, seed 12) has no one entry to hold them to.  `verify_passes` never gets here (one salt is one weights root,
    checked before the tie), so the tie is called directly."""
    a, b = proved_pass(tmp_path / "a", PROMOTED[0])[0], proved_pass(tmp_path / "b", PROMOTED[0], (2, 3), seed=12)[0]
    passes = [VR.load(x, published(x)) for x in (a, b)]
    assert passes[0].epoch.weight_roots[0] != passes[1].epoch.weight_roots[0]
    with pytest.raises(ValueError, match="pass 1 commits weight 0's B rows under another root or shape than an earlier"):
        VR.tie(window_record(a, b), passes)


def test_volunteer_words_gives_up_one_a_row_a_band_and_a_b_row_once_the_a_count_is_spent():
    """Of two credited rows of one band over their row caps, the one further over is given up as an A row, and the band
    rule leaves the other no credit, so it needs no B row.  With V-EX's ⌈m/64⌉ = 2 A rows of a 128-row unit already
    spent, the row furthest over gives up the B row holding its flagged atom-word, whose block of 64 columns is the
    weight's every column, and no tile is left over."""
    s = PPD.scheme_of("pearl-c-h100-v1-h3", k_min=32, tile=(16, 16))
    rec = PW.UnitRecord(Matmul(7, Shape(128, 128, 32), 2), 128, 128)
    tiles = s.tiles(rec.matmul.shape)
    first = next(t for t, x in enumerate(tiles) if x.rows[0] == 0 and x.cols[0] == 0)

    def counts(*flags):
        out = {(0, t): (np.zeros((16, 16), np.int64), np.zeros((16, 16), np.int64)) for t in range(len(tiles))}
        for i, j, n in flags:
            out[(0, first)][0][i, j] = n
        return out

    two = counts((0, 3, 2), (1, 5, 1))
    assert {x["tile"]: [r[0] for r in x["rows"]] for x in CW.over_words(s, [rec], two)} == {first: [0, 1]}
    va, vb = CW.volunteer_words(s, [rec], two)
    assert (va, vb) == ({7: [0]}, {})
    assert CW.over_words(s, [dataclasses.replace(rec, voluntary_a=(0,))], two) == []
    spent = dataclasses.replace(rec, voluntary_a=(64, 100))
    va, vb = CW.volunteer_words(s, [spent], two)
    assert (va, vb) == ({7: [64, 100]}, {2: [3]})
    assert CW.over_words(s, [dataclasses.replace(spent, voluntary_b=(3,))], two) == []


def _volunteer_words_one_row_at_a_time(scheme, records, words):
    """`clear_work.volunteer_words` with its A count stated one row at a time: every tile of the unit rescanned after each
    A row given up, the row furthest over then given up, each tile's credit `UnitRecord.credited`'s (the band rule)."""
    _, tol, units = CW._word_units(scheme, records, words)
    va = {r.matmul.index: set(r.voluntary_a) for r in records}
    vb = {}
    for r in records:
        vb.setdefault(r.matmul.weight, set(r.voluntary_b))

    def credited(r, tile):
        cur = dataclasses.replace(r, voluntary_a=tuple(sorted(va[r.matmul.index])),
                                  voluntary_b=tuple(sorted(vb[r.matmul.weight])))
        rows, cols = (set(x) for x in cur.credited(tile))
        return [x for x, i in enumerate(tile.rows) if i in rows], [z for z, j in enumerate(tile.cols) if j in cols]

    def a_left(r):
        return len(va[r.matmul.index]) < CW.voluntary_cap(r.matmul.shape.m)

    def b_left(r):
        return len(vb[r.matmul.weight]) < CW.voluntary_cap(r.matmul.shape.n)

    changed = True
    while changed:
        changed = False
        for _, r, _, row_cell, tiles in units if CW.row_cap(scheme) else ():
            while a_left(r):
                excess = {}
                for _, tile, w in tiles:
                    rows, cols = credited(r, tile)
                    if rows and cols:
                        s, limit = w[np.ix_(rows, cols)].sum(axis=1), row_cell * len(cols) * tol
                        for x in np.flatnonzero(s > math.floor(limit)):
                            excess[tile.rows[rows[x]]] = excess.get(tile.rows[rows[x]], 0) + int(s[x]) - limit
                if not excess:
                    break
                va[r.matmul.index].add(max(excess, key=lambda i: (excess[i], -i)))
                changed = True
        for _, r, _, row_cell, tiles in units if CW.row_cap(scheme) else ():
            for _, tile, w in tiles:
                while b_left(r):
                    rows, cols = credited(r, tile)
                    if not (rows and cols):
                        break
                    sub = w[np.ix_(rows, cols)]
                    x = int(np.argmax(sub.sum(axis=1)))
                    z = int(np.argmax(sub[x]))
                    if int(sub[x].sum()) <= row_cell * len(cols) * tol or int(sub[x, z]) <= row_cell * tol:
                        break
                    vb[r.matmul.weight].add(tile.cols[cols[z]])
                    changed = True
        for _, r, cell, _, tiles in units:
            for _, tile, w in tiles:
                while True:
                    rows, cols = credited(r, tile)
                    if not (rows and cols):
                        break
                    sub = w[np.ix_(rows, cols)]
                    debit, nr, nc = int(sub.sum()), len(rows), len(cols)
                    if debit <= cell * nr * nc * tol:
                        break
                    options = []
                    if a_left(r) and nr > 1:
                        options += [(debit - int(d) - cell * (nr - 1) * nc, "A", tile.rows[rows[x]])
                                    for x, d in enumerate(sub.sum(axis=1))]
                    if b_left(r) and nc > 1:
                        options += [(debit - int(d) - cell * nr * (nc - 1), "B", tile.cols[cols[z]])
                                    for z, d in enumerate(sub.sum(axis=0))]
                    if not options:
                        break
                    _, side, row = min(options, key=lambda o: o[0])
                    (va[r.matmul.index] if side == "A" else vb[r.matmul.weight]).add(row)
                    changed = True
    return {i: sorted(s) for i, s in va.items() if s}, {w: sorted(s) for w, s in vb.items() if s}


def test_volunteer_words_ranks_a_units_rows_over_their_caps_in_one_scan():
    """`volunteer_words` gives up a unit's A rows over their caps from one scan of its tiles, one a band, since an A row
    given up takes its band out and leaves every other band's rows' sums and caps as they were: on seeded counts with more rows over their caps than ⌈m/64⌉ allows, filler
    rows, and B rows over theirs, across three units, it gives up the A and B rows the one-row-at-a-time statement does."""
    s = PPD.scheme_of("pearl-c-h100-v1-h3", k_min=32, tile=(16, 16))
    rng = np.random.default_rng(7)
    most = 0
    for trial in range(24):
        recs = [PW.UnitRecord(Matmul(u, Shape(192, 128, 48), 10 + u), int(rng.integers(150, 193)), 128) for u in range(3)]
        words = {(p, t): (np.where(rng.random((16, 16)) < 0.04, rng.integers(1, 4, (16, 16)), 0),
                          np.where(rng.random((16, 16)) < 0.02, rng.integers(1, 3, (16, 16)), 0))
                 for p, r in enumerate(recs) for t in range(len(s.tiles(r.matmul.shape)))}
        got = CW.volunteer_words(s, recs, words)
        assert got == _volunteer_words_one_row_at_a_time(s, recs, words), trial
        most = max(most, *(len(v) for v in got[0].values()))
    assert most == CW.voluntary_cap(192) == 3


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
    """A v1 manifest (a pass recorded before the execution phase's coins) is refused unless ``--recorded``; read as a recorded pass it needs
    a beacon its salt is from; a manifest of another schema is refused either way."""
    path, *_ = honest_pass
    man = json.loads(path.read_text())
    v1 = path.parent / "manifest-v1.json"
    for schema, recorded, why in ((CC.MANIFEST_RECORDED, False, "recorded before epoch coins"),
                                  (CC.MANIFEST_RECORDED, True, "recorded pass has no beacon"),
                                  ("verity-vllm/pouw-pearl-c-run/v0", True, f"not {CC.MANIFEST_SCHEMA}")):
        v1.write_text(json.dumps({**man, "schema": schema}))
        r = VR.verify(v1, 3, 1, "test", b"verifier" * 4, coins=published(path), recorded=recorded)
        assert not r["accepted"] and r["line"].startswith("REJECT") and why in r["line"], (schema, recorded, r["line"])
    v1.unlink()


def test_a_pass_whose_salt_coin_is_foreign_early_mislabelled_or_tampered_is_refused():
    """A pass's execution phase re-starts only from the execution phase's own salt coin under the commitment the verifier holds, opened after
    the weights' root, with the salt it derives: a coin of another challenger, one opened after another record, the sampling
    coin's label, a flipped coin byte, a claimed commitment that isn't the published one, or a salt that isn't the coin's is
    refused.  A pass recorded with a beacon (a v1 manifest) is read only as a replay of a recorded pass, and a manifest of
    any other schema not at all."""
    roots = {0: bytes([1]) * 32, 1: bytes([2]) * 32}
    commitment, opening = salt_coin(roots)
    epoch = Epoch.start(Workload((Matmul(0, Shape(16, 1024, 256), 0), Matmul(1, Shape(16, 1024, 128), 1))), roots, commitment,
                        opening)
    man = {"schema": CC.MANIFEST_SCHEMA, "matmuls": [{"index": 0, "m": 16, "k": 1024, "n": 256, "weight": 0}, {"index": 1, "m": 16, "k": 1024, "n": 128, "weight": 1}],
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
    recorded |= {"schema": CC.MANIFEST_RECORDED, "beacon": SEED.hex(),
                 "salt": Epoch.unverified(epoch.workload, roots, SEED).salt.hex()}
    with pytest.raises(DomainError, match="recorded before epoch coins: only a replay of a recorded pass reads it"):
        PPD.manifest_epoch(recorded, commitment)
    assert PPD.manifest_epoch(recorded, None, recorded=True)[1] == "recorded beacon"
    with pytest.raises(DomainError, match="recorded pass has no beacon"):
        PPD.manifest_epoch({**man, "schema": CC.MANIFEST_RECORDED}, commitment, recorded=True)
    with pytest.raises(DomainError, match="not its recorded beacon's"):
        PPD.manifest_epoch({**recorded, "salt": "00" * 32}, None, recorded=True)
    assert PPD.manifest_epoch(man, commitment, recorded=True) == (epoch, "coin")
    for schema in ("verity-vllm/pouw-pearl-c-run/v0", None):
        for flag in (False, True):
            with pytest.raises(DomainError, match=f"schema {schema!r}, not {CC.MANIFEST_SCHEMA}"):
                PPD.manifest_epoch({**man, "schema": schema}, commitment, recorded=flag)
    with pytest.raises(DomainError, match="recorded before epoch coins"):
        PPD.manifest_epoch({**recorded, "schema": CC.MANIFEST_RECORDED, **{k: man[k] for k in ("coins", "salt_opening")}},
                           commitment)


def live_coin(after: bytes, entropy: int = 7):
    """A live session's coin commitment, Pearl-C's salt coin alone (`window.SALT_FAMILIES`), and that coin opened after
    `after`, from a seeded in-process live challenger (the auditor host's session as `pouw_h1_window.open_session`
    returns it)."""
    ch = Ch.LiveChallenger(Wd.LIVE_SCHEME, 0, Wd.SALT_FAMILIES, random.Random(entropy).randbytes)
    return ch.commitment, ch.open(Wd.SALT, 0, after)


def test_the_device_run_takes_a_live_sessions_salt_opened_after_its_epochs_root(device_run, tmp_path):
    """A live session's salt coin opened after salt execution phase k's root (`salt.epoch_open_root` of the weights' root W and
    k, the run's `pass_index`) salts the device run's execution phase (`audit.salt_of` over W, as H1's), and the run records the
    session's coins and k; a retained pass's manifest names k (``pass_index``) and the session the install carries by its
    id, ledger seq, root and commitment alone (never the gate's hello secret or the rest of its answer, which core's
    check of the pass reads from the ledger). One opened after another root (another execution phase's, the bare W, or W before
    the weights were committed), a run naming no execution phase number, one bound to another session's commitment, a
    session of H1's layout (salt, sample and round coins, which a Pearl-C pass leaves unreleased, so its session never
    ends complete), and a live opening against an in-process commitment or the reverse are refused."""
    _, kernel, weights, roots, _ = device_run
    root = weights_root(roots)
    r = PS.epoch_open_root(root, 1)

    def make(c, o, k=1):
        return PPD.PearlCDeviceRun(PPD.scheme_of(SCHEME), f"pouw:{SCHEME}", kernel, weights, roots, c, o,
                                   capacity=1 << 12, salt_key=SALT_KEY, pass_index=k)
    c, o = live_coin(r)
    run = make(c, o)
    assert run.epoch.salt == salt_of(c, o.coin, root) and run.commits["coins"] == c.to_json()
    assert run.commits["salt_opening"] == o.to_json() and run.commits["pass_index"] == 1
    named = {"session": "s20261020T110800Z-0001", "seq": 4, "root": r.hex(), "commitment": c.root.hex()}
    run.commits["session"] = {**named, "port": 7811, "by": "vllm", "secret": "ab" * 32}
    run.begin(retain=str(tmp_path))
    for w in (0, 1):
        run.gemm(w, torch.randn(20, 1024).to(torch.bfloat16), weights[w], None)
    man = json.loads(Path(run.end()["manifest"]).read_text())
    assert man["session"] == named and man["coins"] == c.to_json() and man["salt"] == run.epoch.salt.hex()
    assert man["pass_index"] == 1 and PPD.manifest_epoch(man, c)[0].salt == run.epoch.salt
    for k in (0, 2):
        with pytest.raises(DomainError, match="not opened after"):
            make(c, o, k)
        with pytest.raises(DomainError, match="not opened after"):
            PPD.manifest_epoch({**man, "pass_index": k}, c)
    for k in (None, True, -1):
        with pytest.raises(DomainError, match="salt epoch's number"):
            make(c, o, k)
    with pytest.raises(DomainError, match="salt epoch's number"):
        PPD.manifest_epoch({k: v for k, v in man.items() if k != "pass_index"}, c)
    with pytest.raises(DomainError, match="not opened after"):
        make(*live_coin(root))
    with pytest.raises(DomainError, match="not opened after"):
        make(*live_coin(PS.epoch_open_root(weights_root({0: roots[0], 1: bytes([9]) * len(roots[1])}), 1)))
    with pytest.raises(DomainError, match="does not open against"):
        make(live_coin(r, entropy=8)[0], o)
    with pytest.raises(DomainError, match="not a Pearl-C salt session's"):
        ch = Wd.live_challenger(0, 1, random.Random(7).randbytes)
        make(ch.commitment, ch.open(Wd.SALT, 0, r))
    in_process = salt_coin(roots)
    for pair in ((in_process[0], o), (c, in_process[1])):
        with pytest.raises(DomainError, match="live coin commitment, and nothing else"):
            make(*pair)


def test_a_retained_pass_salted_by_a_live_session_restarts_only_under_that_sessions_commitment():
    roots = {0: bytes([1]) * 32, 1: bytes([2]) * 32}
    work = Workload((Matmul(0, Shape(16, 1024, 256), 0), Matmul(1, Shape(16, 1024, 128), 1)))
    c, o = live_coin(PS.epoch_open_root(weights_root(roots), 0))
    epoch = PPD.epoch_of(work, roots, c, o, 0)
    man = {"schema": CC.MANIFEST_SCHEMA, "matmuls": [{"index": 0, "m": 16, "k": 1024, "n": 256, "weight": 0},
                                                      {"index": 1, "m": 16, "k": 1024, "n": 128, "weight": 1}],
           "weights": {str(w): {"root": r.hex()} for w, r in roots.items()},
           "coins": c.to_json(), "salt_opening": o.to_json(), "salt": epoch.salt.hex(), "pass_index": 0}
    assert PPD.manifest_epoch(man, c) == (epoch, "coin")
    with pytest.raises(DomainError, match="not the epoch's"):
        PPD.manifest_epoch(man, live_coin(weights_root(roots), entropy=8)[0])
    with pytest.raises(DomainError, match="not the epoch's"):
        PPD.manifest_epoch(man, salt_coin(roots)[0])
    early = {**man, "salt_opening": {**o.to_json(), "after": weights_root({0: roots[0], 1: bytes(32)}).hex()}}
    with pytest.raises(DomainError, match="not opened after"):
        PPD.manifest_epoch(early, c)


def test_a_runs_coins_file_is_a_live_sessions_commitment_or_an_in_process_challengers(tmp_path):
    """`verify_run.epoch_coins`, the coins file every step of a served run reads (``served_passes.py volunteer`` and
    ``verify``, ``verify_run.py --coins``, ``volunteer_pass.py --coins``): a live session's commitment when its scheme is
    `window.LIVE_SCHEME`, under which a pass salted by that session restarts its epoch, else an in-process challenger's;
    a pass restarts under its own epoch's commitment only."""
    live = proved_pass(tmp_path / "live", PROMOTED[0], live=0)[0]
    plain = proved_pass(tmp_path / "plain", PROMOTED[0])[0]
    c, cp = VR.epoch_coins(live.parent / "coins.json"), VR.epoch_coins(plain.parent / "coins.json")
    assert isinstance(c, Ch.LiveCommitment) and isinstance(cp, CoinCommitment) and c.to_json()["scheme"] == Wd.LIVE_SCHEME
    man = json.loads(live.read_text())
    assert man["pass_index"] == 0 and man["coins"] == c.to_json()
    assert VR.load(live, c).epoch.salt.hex() == man["salt"] != json.loads(plain.read_text())["salt"]
    assert VR.load(plain, cp).epoch.salt.hex() == json.loads(plain.read_text())["salt"]
    for path, other in ((live, cp), (plain, c)):
        with pytest.raises(DomainError, match="not the epoch's"):
            VR.load(path, other)


@pytest.mark.slow
def test_served_passes_verify_accepts_a_run_salted_by_a_live_session(tmp_path):
    """A served run whose epoch's salt coin is a live session's (``serve`` with the challenger gate): ``verify`` reads its
    coins file as that session's commitment and accepts its honest pass."""
    path = proved_pass(tmp_path, PROMOTED[0], live=0)[0]
    (path.parent / "served.json").write_text(json.dumps({"run": RUN, "passes": [{"pass": 0, "units": 2, "tiles": 4,
                                                                                 "manifest": str(path)}]}))
    rc, doc, answer = served_verify(path.parent, tmp_path / "verify.json")
    assert rc == 0 and answer["verdict"] == "accept" and doc["line"].startswith("ACCEPT"), doc["line"]


def test_live_coins_reserve_a_salt_only_session_whose_settings_the_install_takes():
    """`bench_arm.live_coins` (``served_passes.py serve``'s and ``served_step.py``'s): given the gate and this node's peer
    name, the PoUW settings drop their in-process coins and take a salt-only session the gate reserves under the lane once
    its ledger shows the lane's open period (``reserve --peer P --scheme S --families [["pouw/salt",1]] --epoch N``): the
    challenger on this node, the reserve answer, the gate's open and that period, which `pouw.Settings` reads and the
    adapter's refusals (`pouw.refusals`) refuse nothing of.  Neither given leaves the settings, and their coins
    in-process, untouched; one alone is refused, and a reserve while the lane's open period is H1's is refused before it
    is asked."""
    from verity_vllm.protocols import pouw
    from verity_vllm.protocols.pouw.h1 import window as PHW

    BA = load("pearl_c_vllm_bench_arm", KERNEL.parent / "pearl_c_vllm" / "bench_arm.py")
    gate = "ssh -i KEY -o BatchMode=yes research@gate"
    reserved = {"session": "s20261103T140000Z-0001", "seq": 7, "peer": "vy-dc-h200", "by": "vllm",
                "expires_utc": "2026-11-03T16:00:00Z"}
    ledger, asked = [], []

    def run(argv):
        assert argv[:-1] == shlex.split(gate)
        asked.append(shlex.split(argv[-1]))
        if asked[-1][0] == "ledger":
            return 0, "".join(json.dumps(e) + "\n" for e in ledger)
        return 0, json.dumps(reserved)

    base = {"device": "cuda", "kernel": "k", "transcript_dir": "passes", "coins": Challenger(0, random.Random(7).randbytes)}
    assert BA.live_coins(base, None, None, 5, "vllm", run) == (base, None) and asked == []
    got, answer = BA.live_coins(base, gate, "vy-dc-h200", 5, "vllm", run)
    assert answer == reserved and "coins" not in got and "coins" in base
    assert {k: got[k] for k in ("device", "kernel", "transcript_dir", "challenger", "session", "period")} == {
        "device": "cuda", "kernel": "k", "transcript_dir": "passes", "challenger": "127.0.0.1:0", "session": reserved,
        "period": None}
    assert asked == [["ledger"], ["reserve", "--peer", "vy-dc-h200", "--scheme", Wd.LIVE_SCHEME, "--families",
                                  json.dumps([list(f) for f in Wd.SALT_FAMILIES], separators=(",", ":")), "--epoch", "5"]]
    settings = pouw.Settings.of(got)
    assert (settings.coins, settings.challenger, settings.session, settings.period) == (None, "127.0.0.1:0", reserved, None)
    serving = types.SimpleNamespace(stage="serve", world=1, enforce_eager=True, settings_of=lambda name: got,
                                    filled_of=lambda name: ())
    assert pouw.refusals(SCHEME, serving) == ()
    for one in ((gate, None), (None, "vy-dc-h200")):
        with pytest.raises(ValueError, match="together"):
            BA.live_coins(base, *one, 5, "vllm", run)
    ledger.append({"kind": "period-open", "session": None, "body": {"period": "p1", "lane": "vllm",
                                                                     **dict.fromkeys(PHW.H1_PERIOD, 1)}})
    asked.clear()
    with pytest.raises(ValueError, match="refused before reserving a salt-only session: the lane's open period p1 is H1's"):
        BA.live_coins(base, gate, "vy-dc-h200", 5, "vllm", run)
    assert asked == [["ledger"]]


@pytest.mark.slow
def test_served_debit_records_the_tile_rules_own_debit_and_cap(honest_pass, tmp_path):
    path, s, epoch, prover, tr, splits, records = honest_pass
    out = tmp_path / "out"
    assert SD.main([str(path), *coins_args(path), "--tiles", "3", "--jobs", "1", "--out", str(out)]) == 0
    assert CW._debit_cap is SD._DEBIT_CAP and CW.D.tile_debit is SD._TILE_DEBIT
    (got,) = json.loads((out / "summary.json").read_text())["passes"]
    assert got["scheme"] == s.name and got["cap"] == str(s.cap) and got["openings_failed"] == 0
    assert set(got["strata"]) == {"m32-k128-n16", "m16-k128-n32"} and got["tiles_total"] == tr.commitment.count
    assert got["strata"]["m32-k128-n16"]["credited_macs"] == (20 - 1) * 16 * 128
    tiles = [json.loads(line) for line in (out / f"{path.parent.name}.tiles.jsonl").read_text().splitlines()]
    assert len(tiles) == got["sampled"] and got["measured"] == sum("share" in t for t in tiles) > 0
    refs = Verifier(CW._TileRuleView(s), epoch).refs
    for t in (t for t in tiles if "share" in t):
        ref, rec = refs[t["tile"]], records[t["matmul"]]
        op = prover.open(epoch, ref.id)
        a, b = [opened_value(s.commitment_hash, o.value) for o in op.a], [opened_value(s.commitment_hash, o.value) for o in op.b]
        u = epoch.workload.matmuls[ref.matmul]
        ctx = Context(epoch.salt, u, tr.activation_roots[ref.matmul], epoch.weight_roots[u.weight])
        ops = s.encode(ctx, s.noise(ctx), a, b, ref.tile)
        rows, cols = rec.credited(ref.tile)
        debit, cap = CW._debit_cap(s, rec, ref.tile, ops, rows, cols)
        credit = CW.cell_credit(s, rec) * len(rows) * len(cols)
        assert (t["debit_exact"], t["ratio"], t["credit_exact"]) == (str(debit), float(debit / cap), str(credit))
        assert {key: t[key] for key in t if key.startswith("share")} == {"share": float(debit / credit)}
        pos_a, pos_b = {i: p for p, i in enumerate(ref.tile.rows)}, {j: p for p, j in enumerate(ref.tile.cols)}
        a_codes, b_codes = [ops.a.codes[pos_a[i]] for i in rows], [ops.b.codes[pos_b[j]] for j in cols]
        formed = [(ops.a.alpha[pos_a[i]], ops.a.x[pos_a[i]], ops.a.beta[pos_a[i]]) for i in rows]
        assert (t["atoms"], t["later_zero_promotions"], t["forming_elements"]) == CW.D.tile_debit(a_codes, b_codes, formed, s.device)
        assert t["ok"] == (CW.check_opened(s, rec, ref.tile, a, b, ops) is None)
    assert got["later_zero_promotions"] == sum(t.get("later_zero_promotions", 0) for t in tiles)


@pytest.mark.slow
def test_served_debit_per_input_bounds_each_input_on_its_own(honest_pass, tmp_path):
    """Per-input mode (lean's ruling on): each chosen input is one matmul of the pass, measured in full or by
    work-weighted samples, with its own failures, `W`, `w_max` and a simultaneous upper limit; nothing is pooled."""
    path, s, epoch, prover, tr, splits, records = honest_pass
    out = tmp_path / "out"
    assert SD.main([str(path), *coins_args(path), "--per-input", "1", "--census-max", "4", "--draws", "3", "--jobs", "1", "--out", str(out)]) == 0
    doc = json.loads((out / "inputs.json").read_text())
    refs = Verifier(CW._TileRuleView(s), epoch).refs
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
    assert all("the row cap" in t["reason"] for t in fails) and (not fails or CW.row_cap(s))
    for row in doc["inputs"]:
        failures = sum(t["matmul"] == row["matmul"] for t in fails)
        assert row["method"] == "fresh" and row["drawn"] == 4 and row["failures"] == failures
        assert row["upper"] == SD.clopper_pearson_upper(failures, 4, 0.025) and row["bound_method"].startswith("fresh")
    t = lines[0]
    ref = Verifier(CW._TileRuleView(s), epoch).refs[t["tile"]]
    op = prover.open(epoch, ref.id)
    a, b = [opened_value(s.commitment_hash, o.value) for o in op.a], [opened_value(s.commitment_hash, o.value) for o in op.b]
    u = epoch.workload.matmuls[ref.matmul]
    ctx = Context(bytes.fromhex(t["salt"]), u, tr.activation_roots[ref.matmul], epoch.weight_roots[u.weight])
    ops = s.encode(ctx, s.noise(ctx), a, b, ref.tile)
    rows, cols = records[ref.matmul].credited(ref.tile)
    debit, cap = CW._debit_cap(s, records[ref.matmul], ref.tile, ops, rows, cols)
    assert t["debit_exact"] == str(debit) and t["ratio"] == float(debit / cap)
    again = tmp_path / "again"
    resume = ["--resume", str(out / f"{path.parent.name}.fresh-tiles.jsonl")]
    assert SD.main([*args, "--out", str(again), *resume]) == 0
    assert json.loads((again / "inputs.json").read_text())["inputs"] == doc["inputs"]
    with pytest.raises(ValueError, match="another salt"):
        SD.main([*args[:6], "cd" * 32, *args[7:], "--out", str(tmp_path / "other"), *resume])


def test_fresh_trials_draw_exactly_by_work_from_a_source_fixed_in_the_command(tmp_path):
    """Lean's conditions on the fresh-trial ε: sample d is `key.uniform(W, d)` over the input's cumulative credited cells
    (exactly proportional to `TR.Wref` within the input), under a domain naming the table and a context naming the input,
    and the source is given in the command, never taken inside the run."""
    from verity.commitments.randomness import derive

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
    """An execution phase's row comes from its accepting audit: credited against claimed GEMMs, capacity, the certified fraction with
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
    """A real decode line (Qwen3-8B's down projection, pass) recomputes at Prices.sm120, and at the
    cited Prices.sm120Loop its verdict flips only past 11.0% salt-dead elements, by hand."""
    line = {"tile": 1769828, "matmul": 7179, "m": 32, "k": 12288, "n": 4096, "credited_rows": 32, "credited_cols": 64,
            "ok": True, "atoms": 14, "later_zero_promotions": 0, "debit_exact": "476", "cap": 27426.816}
    path = tmp_path / "q" / "decode.input-tiles.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps(line) + "\n" + json.dumps(line | {"tile": 1, "atoms": 200, "debit_exact": "6800"}) + "\n")
    cell = 12288 + Fraction(12288 * 1047, 125 * 128) + 96 + 128 + 120 - Fraction(1047, 125)
    want = ((cell * 2048 / 1000 - (32 + Fraction(1047, 500)) * 14) * 4096 / (40 * 64)) / (32 * 12288)
    assert FH.dead_to_flip(line) == want and 0.1099 < want < 0.1100
    with pytest.raises(ValueError, match="recorded with the forming term"):
        FH.check_line(line | {"forming_elements": 0})
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
    """The per-input limits (lean's rulings on): the work-weighted rule of three for a census and for n samples at no
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
        want = CW.D.chain_debit(a, b, s.device)
        assert FT.chain_debit(a, b, s.device) == want
        seen.add(want[1] > 0)
        assert want[0] > 0
    assert seen == {True, False}
    a[0][5] = 0x7F
    with pytest.raises(Exception) as want:
        CW.D.chain_debit(a, b, s.device)
    with pytest.raises(type(want.value), match=str(want.value)):
        FT.chain_debit(a, b, s.device)


def test_the_batch_tile_debit_is_the_scalar_tile_debit():
    """`fast_tile.tile_debit` is `pearl_c_debit.tile_debit`, forming elements included: on rows with salt-dead elements
    (accumulators past both ends' saturation, and at 0), a zero row whose every block every word flags, and a row whose
    accumulator overflows, where both reject alike."""
    FT, s = SD.FT, PPD.scheme_of(SCHEME)
    rng, seen = np.random.default_rng(7), set()
    one = 0x3F800000
    for trial in range(6):
        a, b = _debit_codes(rng, 3, 384, trial % 2 == 0), _debit_codes(rng, 4, 384, trial % 2 == 0)
        a[1] = [0] * 384
        mags = 10.0 ** rng.uniform(-3, 5, (3, 384)) * rng.choice((-1, 1), (3, 384))
        x = [[int(w) for w in row.astype(np.float32).view(np.uint32)] for row in mags]
        x[0][:7] = [0, 0x80000000, 1, 0x00800000, 0x3F000000, 0x45000000, 0xC5000000]
        beta = [int(np.float32(v).view(np.uint32)) for v in rng.uniform(0.01, 2.0, 3)]
        formed = [(one, x[i], beta[i]) for i in range(3)]
        want = CW.D.tile_debit(a, b, formed, s.device)
        assert FT.tile_debit(a, b, formed, s.device) == want
        flags = [[CW.D.word_flags(r, c, s.device) for c in b] for r in a]
        dead = [CW.D.salt_dead(*f, s.device) for f in formed]
        assert want[:2] == CW.D.chain_debit(a, b, s.device) and want[2] == sum(
            1 for i in range(3) for l in range(384) if dead[i][l] or all(f[l // 32] for f in flags[i]))
        seen.add(0 < sum(dead[0]) < 384)
    assert seen == {True}
    formed[0] = (0x7F000000, x[0], beta[0])
    with pytest.raises(Exception) as want:
        CW.D.tile_debit(a, b, formed, s.device)
    with pytest.raises(type(want.value), match=str(want.value)):
        FT.tile_debit(a, b, formed, s.device)


def test_the_batch_debit_word_by_word_is_each_words_scalar_debit():
    """`fast_tile.word_debits` is `pearl_c_debit.word_debit` on every word, and its sums `chain_debit`'s on any rows and
    columns, so the remedy's row and column debits are the verifier's."""
    FT, s = SD.FT, PPD.scheme_of(SCHEME)
    rng = np.random.default_rng(3)
    for trial in range(4):
        a, b = _debit_codes(rng, 3, 384, trial % 2 == 0), _debit_codes(rng, 4, 384, trial % 2 == 0)
        atoms, zeros = FT.word_debits(a, b, s.device)
        assert atoms.shape == zeros.shape == (3, 4) and atoms.sum() > 0
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                flags, z = CW.D.word_debit(x, y, s.device)
                assert (atoms[i, j], zeros[i, j]) == (sum(flags), z)
        assert (atoms[[0, 2]][:, [1, 3]].sum(), zeros[[0, 2]][:, [1, 3]].sum()) == CW.D.chain_debit([a[0], a[2]], [b[1], b[3]], s.device)


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
        formed = list(zip(ops.a.alpha, ops.a.x, ops.a.beta))
        return (ops, s.checked(ctx, ops), CW.D.chain_debit(ops.a.codes, ops.b.codes, s.device),
                CW.D.tile_debit(ops.a.codes, ops.b.codes, formed, s.device))

    debits = (CW.D.chain_debit, CW.D.tile_debit)
    before = (C.form_v1, C.gram, C.peel_factors_a, C.peel_factors_b, C._basis, *debits, type(s).checked, C.blake3)
    undo = FT.install(s)
    assert C.form_v1 is FT.form_v1 and (CW.D.chain_debit, CW.D.tile_debit) == (FT.chain_debit, FT.tile_debit)
    fast = run()
    undo()
    debits = (CW.D.chain_debit, CW.D.tile_debit)
    assert (C.form_v1, C.gram, C.peel_factors_a, C.peel_factors_b, C._basis, *debits, type(s).checked, C.blake3) == before
    ref = run()
    for side in ("a", "b"):
        for field in ("codes", "alpha", "beta", "s", "rho", "ep", "x"):
            assert getattr(getattr(fast[0], side), field) == getattr(getattr(ref[0], side), field), (side, field)
    assert (fast[0].pa, fast[0].pb) == (ref[0].pa, ref[0].pb)
    assert (fast[1].c, fast[1].u) == (ref[1].c, ref[1].u) and fast[2:] == ref[2:]
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
    debit, cap = CW.tile_cap(s, rec, codes(rows), codes(cols), [(0x3F800000, [0] * shape.k, 0x3F800000)] * rows)
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
    """Decode's headline row is over stock FP8 under CUDA graphs, the eager row beside it at the
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


def test_a_prefill_point_off_8192_tokens_names_its_key():
    """``--prefill`` off the panel's 8,192 tokens keys the prefill row by its length, so `panel_rows.py` (``prefill_8192``)
    never reads it as the panel's prefill row."""
    E2E = load("pearl_c_vllm_e2e_prefill", KERNEL.parent / "pearl_c_vllm" / "e2e.py")
    E2E.configure(types.SimpleNamespace(batch=32, prompt=64, prefill=2560))
    assert (E2E.prefill_key(), E2E.PREFILL) == ("prefill_2560", 2560)
    assert E2E.SHAPES == {"prefill_2560": "e2e-llama31-8b-vllm-m2560", "decode_b32_step": "e2e-llama31-8b-vllm-m32"}
    assert len(E2E.prompts(1)["prefill"][0]) == 2560
    rec = lambda m, ph: {"mode": m, "phase": ph, "rep": 0, "s": 1.0, "gemm_s": 0.0, "step_s": 0.01, "gemm_step_s": 0.0}
    t = E2E.timing([rec(m, ph) for m in E2E.MODES for ph in ("prefill", "decode")])
    assert set(t["pearlc"]["timing"]) == {"prefill_2560_s", "prefill_2560_gemm_s", "decode_b32_step_s", "decode_b32_step_gemm_s"}
    assert set(t["pearlc"]["reps"]) == {"prefill_2560", "decode_b32"} and list(E2E.split(t)) == ["prefill_2560", "decode_b32_step"]
    E2E.configure(types.SimpleNamespace(batch=32, prompt=64))
    assert E2E.SHAPES == {"prefill_8192": "e2e-llama31-8b-vllm-m8192", "decode_b32_step": "e2e-llama31-8b-vllm-m32"}


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
    """Qwen3-8B's tokenizer has no BOS (vLLM refused its execution phases' None id); its config's `bos_token_id` stands in."""
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


def test_the_whole_step_arm_captures_vllms_graphs_at_the_decode_batch_or_captures_sizes(monkeypatch):
    E2E = load("pearl_c_vllm_e2e", KERNEL.parent / "pearl_c_vllm" / "e2e.py")
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=lambda **kw: kw))
    args = types.SimpleNamespace(model="m", revision=None, mem_base=0.2, mem_arm=0.3, whole_step=True, capture=None)
    got = E2E.engine(args, "arm")
    assert not got["enforce_eager"] and got["compilation_config"] == {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                                                      "cudagraph_capture_sizes": [E2E.BATCH]}
    args.capture = [8, 32]
    assert E2E.engine(args, "arm")["compilation_config"]["cudagraph_capture_sizes"] == [8, 32]
    args.whole_step = False
    assert E2E.engine(args, "arm")["compilation_config"] == {"mode": 0}


# ---- the decode profile -------------------------------------------------------------------------------------------

def test_the_profile_maps_gpu_1s_kernels_to_their_categories(tmp_path):
    PF = load("pearl_c_vllm_profile", KERNEL.parent / "pearl_c_vllm" / "profile_decode.py")
    assert [PF.kernel_category(k) for k in ("gemm_pearlc", "gemm_pearlc_d", "form_s5", "stats_s5", "stats_r", "words_k", "gemm_words",
                                            "s512_rows", "hm_leaves", "pd_commit_851", "pd_tiles_979", "flash_fwd_kernel",
                                            "paged_attention")] == [
        "gemm", "gemm", "forming", "forming", "forming", "forming", "forming", "hashing", "hashing", "hashing", "hashing", "other",
        "other"]
    csvf = tmp_path / "k.csv"
    csvf.write_text('"Time (%)","Total Time (ns)","Instances","Avg (ns)","Name"\n'
                    '50,2000000,10,200000,"gemm_pearlc"\n30,1200000,40,30000,"s512_tile_leaves"\n20,800000,5,160000,"flash_fwd"\n')
    got = PF.nsys_kernels(csvf)
    assert got["gpu_ms_by_category"] == {"gemm": 2.0, "hashing": 1.2, "other": 0.8} and got["top_kernels"][0]["kernel"] == "gemm_pearlc"


def test_the_served_profile_counts_stock_fp8s_cutlass_3_kernel_as_gemm(monkeypatch):
    """vLLM's FP8 linear on sm_120 is CUTLASS 3's `device_kernel`, the stock arm's matmul, not `other`."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    SP = load("pearl_c_vllm_served_profile", KERNEL.parent / "pearl_c_vllm" / "served_profile.py")
    assert [SP.category(k) for k in ("device_kernel", "Kernel2", "gemm_pearlc_d", "words", "words_k", "stats_r", "s512_tile_leaves",
                                     "flash_fwd_splitkv_kernel", "some_device_kernel_x")] == [
        "gemm", "gemm", "pearlc:gemm", "pearlc:forming", "pearlc:forming", "pearlc:forming", "pearlc:tile_hash", "attention", "other"]


def test_the_served_profile_names_hash_fasts_kernels_and_gives_a_fold_its_segment_launchs_step(monkeypatch):
    """hash_fast.cuh's segment kernels are their step's, and a ``s512_folds`` launch, which A's row digests and the tile leaves
    both run, is the step of the segment launch before it on its stream.  A Triton ``kernel``, which stock FP8 launches too, isn't
    the records' screen (``screen``, ``scatter``)."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    SP = load("pearl_c_vllm_served_profile", KERNEL.parent / "pearl_c_vllm" / "served_profile.py")
    assert [SP.category(k) for k in ("s512_row_segs", "s512_tile_segs", "s512_heads", "s512_folds", "lines", "s512_seeds", "scatter",
                                     "kernel")] == [
        "pearlc:row_digests", "pearlc:tile_hash", "pearlc:setup", "pearlc:folds", "pearlc:lines_a", "pearlc:seeds", "pearlc:screen",
        "other"]
    rows = [(0, 1, "s512_row_segs", 7), (1, 2, "s512_tile_segs", 9), (2, 3, "s512_folds", 7), (3, 4, "s512_folds", 9),
            (4, 5, "s512_folds", 11), (5, 6, "s512_tile_segs", 7), (6, 7, "s512_folds", 7)]
    assert [r[4] for r in SP._folds(rows)] == ["pearlc:row_digests", "pearlc:tile_hash", "pearlc:row_digests", "pearlc:tile_hash",
                                               "pearlc:folds", "pearlc:tile_hash", "pearlc:tile_hash"]
    # hash_wide.cuh's launches (rows past 128 segments) are the same steps
    assert [SP.category(k) for k in ("s512_row_segs_w", "s512_heads_w", "s512_folds_w")] == [
        "pearlc:row_digests", "pearlc:setup", "pearlc:folds"]
    assert [r[4] for r in SP._folds([(0, 1, "s512_row_segs_w", 7), (1, 2, "s512_folds_w", 7)])] == ["pearlc:row_digests"] * 2


def test_the_served_profile_times_each_passs_end_and_splits_a_steps_hashing(monkeypatch, tmp_path):
    """A pass's end (its transcript tree, after the generate) is timed apart from the generate, its tree kernels counted as the
    transcript tree's; the decode step's split is (dn − d1) / 64 of both."""
    import sqlite3
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    SP = load("pearl_c_vllm_served_profile", KERNEL.parent / "pearl_c_vllm" / "served_profile.py")
    ms, g = 1_000_000, 5 << 24
    names = ["s512_row_segs", "s512_folds", "s512_tile_segs", "gemm_pearlc", "form_s5", "flash_fwd", "s512_pads", "s512_nodes",
             "lines", "screen"]
    db = tmp_path / "p.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
    con.executemany("INSERT INTO StringIds VALUES (?, ?)", list(enumerate(names)))
    con.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, globalTid INTEGER, textId INTEGER)")
    nvtx = [(90, 260, "pass"), (100, 200, "gen|pearlc|prefill|1"), (300, 315, "pass"), (300, 310, "gen|pearlc|d1|1"),
            (400, 530, "pass"), (400, 500, "gen|pearlc|dn|1"), (600, 700, "gen|pearlc|prefill|0")]
    con.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, NULL)", [(s * ms, e * ms, t, g) for s, e, t in nvtx])
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, shortName INTEGER, globalPid INTEGER, "
                "streamId INTEGER)")
    k = [(110, 111, 0, 7), (111, 111.5, 1, 7), (120, 122, 2, 9), (122, 122.5, 1, 9), (130, 134, 3, 7), (135, 136, 4, 7),
         (140, 141, 5, 7), (150, 150.5, 8, 7), (151, 151.25, 9, 7),
         (210, 210.5, 6, 7), (211, 212, 7, 7),                       # the prefill pass's end
         (305, 306, 3, 7), (311, 312, 7, 7),                         # d1, and its pass's end
         (405, 469, 3, 7), (510, 575, 7, 7)]                         # dn, and its pass's end
    con.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)", [(int(s * ms), int(e * ms), n, g, st) for s, e, n, st in k])
    con.commit()
    doc = SP.analyze(db)
    pre = doc["phases"]["pearlc"]["prefill"]
    assert pre["n"] == 1 and pre["pass_end_ms"] == pytest.approx(60.0)
    assert pre["by_category_ms"] == pytest.approx({"pearlc:row_digests": 1.5, "pearlc:tile_hash": 2.5, "pearlc:gemm": 4.0, "pearlc:forming": 1.0,
                                                   "attention": 1.0, "pearlc:lines_a": 0.5, "pearlc:screen": 0.25})
    assert pre["pass_end_by_category_ms"] == pytest.approx({"pearlc:a_tree": 1.5})
    assert pre["hash_split_ms"] == pytest.approx({"tile_leaves": 2.5, "a_rows": 2.0, "transcript_tree": 1.5, "screen": 0.25,
                                                  "forming_gemm": 5.0, "hashing": 6.25})
    step = doc["phases"]["pearlc"]["step"]
    assert step["pass_end_ms"] == pytest.approx((30 - 5) / 64) and step["by_category_ms"]["pearlc:gemm"] == pytest.approx(63 / 64)
    assert step["hash_split_ms"]["transcript_tree"] == pytest.approx(1.0) and step["hash_split_ms"]["hashing"] == pytest.approx(1.0)


def test_the_profile_times_each_calls_sections_and_steps_and_puts_the_module_back(device_run):
    from verity_vllm.protocols.pouw import pearl_c as PPD2
    PF = load("pearl_c_vllm_profile", KERNEL.parent / "pearl_c_vllm" / "profile_decode.py")
    make, _, weights, *_ = device_run
    before = (PPD2._steps, PPD2.call_keys, PPD2.PearlCDeviceRun.gemm, PPD2.PearlCDeviceRun.end)
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
    assert (PPD2._steps, PPD2.call_keys, PPD2.PearlCDeviceRun.gemm, PPD2.PearlCDeviceRun.end) == before
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


def test_eager_calls_leave_every_filler_row_zero_as_m_moves_within_a_shape(device_run):
    make, _, weights, *_ = device_run
    run = make()
    for m in (20, 31, 17, 32, 25, 18):
        x = torch.randn(m, 1024).to(torch.bfloat16)
        run.gemm(0, x, weights[0], None)
        sh = run.shapes[(32, 256, 1024)]
        assert sh.dirty == m and bool((sh.xa[m:] == 0).all()) and torch.equal(sh.xa[:m], x.float())
    with torch.inference_mode():
        sh.xa[18:30] = 3.0                                         # a replay's rows, which this host doesn't see
    run.eager_only = False
    run.gemm(0, torch.randn(17, 1024).to(torch.bfloat16), weights[0], None)
    assert bool((sh.xa[17:] == 0).all())


def test_each_calls_a_root_lands_at_its_position_and_its_keys_take_the_epochs_index(device_run):
    """Two steps of the same three calls: each call's A root lands at its position in its pass's buffer, and its keys are
    those of its index in the execution phase, so the second step's calls are 3, 4 and 5 and share no keys with the first's."""
    make, _, weights, roots, salt = device_run
    run = make()
    for step in range(2):
        for i, w in enumerate((0, 1, 0)):
            sh = run._shape(16, weights[w].shape[0], 1024)
            tree = sh.b["tree_a"]
            with torch.inference_mode():                           # A's tree, which the device would write
                tree.copy_((torch.arange(len(tree)) * 7 + 3 * i + step).to(torch.uint8))
            run.gemm(w, torch.randn(8, 1024).to(torch.bfloat16), weights[w], None)
            size = run._pipe(w, sh)[0].size["tree_a"]
            assert PPD._bytes(run.roots_buf[64 * i:64 * (i + 1)]) == PPD._bytes(tree[size - 64:size])
            assert PPD._bytes(sh.keys[:168]) == PPD.call_keys(F, salt, roots[w], 3 * step + i, 16, run.tree_hash)
        assert [u.index for u in run.cur.matmuls] == [3 * step, 3 * step + 1, 3 * step + 2]
        run.end_pass()
    assert run.cur.base == run.epoch_calls() == 6


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


def test_the_screen_launches_tritons_compiled_kernel_directly_only_where_triton_would_launch_the_same(monkeypatch):
    """After Triton's own launch at a specialization, `launch` calls its compiled kernel with the same arguments; another
    alignment compiles again, and a launch hook (Triton's chains hold none by default) or a CPU tensor takes Triton's path."""
    # the module itself: under the layout move its package re-exports a function named as the module
    PS = importlib.import_module("verity_vllm.protocols.pouw.pearl_c.screen")
    enter, exit_ = types.SimpleNamespace(calls=[]), types.SimpleNamespace(calls=[])
    knobs = types.SimpleNamespace(runtime=types.SimpleNamespace(debug=False, add_stages_inspection_hook=None, launch_enter_hook=enter,
                                                                launch_exit_hook=exit_),
                                  compilation=types.SimpleNamespace(instrumentation_mode=""))
    monkeypatch.setitem(sys.modules, "triton", types.SimpleNamespace(knobs=knobs))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch._C, "_cuda_getCurrentRawStream", lambda device: 77, raising=False)
    monkeypatch.setattr(PS, "_COMPILED", {})
    log = []

    class Kernel:
        pre_run_hooks: ClassVar[list] = []

        def __getitem__(self, grid):
            def own(*args, BLOCK):
                log.append(("triton", grid, args[1].data_ptr() % 16, BLOCK))
                return types.SimpleNamespace(run=lambda *a: log.append(("direct", *a)), function="fn", packed_metadata="meta")
            return own
    x, kernel, out = type("OnDevice", (), {"is_cuda": True})(), Kernel(), torch.zeros(64, dtype=torch.uint8)
    assert out.data_ptr() % 16 == 0
    for o in (0, 0, 2, 2, 16):
        PS.launch(kernel, 3, (x, out[o:], 4096, 1.0), 1024)
    assert [e[0] for e in log] == ["triton", "direct", "triton", "direct", "direct"]
    assert log[1][:10] == ("direct", 3, 1, 1, 77, "fn", "meta", None, None, None) and log[1][10] is x
    assert log[1][12:] == (4096, 1.0, 1024) and log[1][11].data_ptr() == out.data_ptr()
    enter.calls.append(print)
    PS.launch(kernel, 3, (x, out, 4096, 1.0), 1024)
    enter.calls.clear()
    PS.launch(kernel, 3, (out, out, 4096, 1.0), 1024)
    PS.launch(kernel, 3, (x, out, 4096, 2.0), 1024)
    assert [e[0] for e in log[5:]] == ["triton", "triton", "triton"]


def test_the_fused_screen_decides_the_rows_as_the_torch_screen_and_the_rules(monkeypatch):
    """The Triton kernel under Triton's CPU interpreter, on rows at the rules' edges: its excluded rows are `screen`'s and
    `clear_work.row_passes`'s."""
    interpreted_triton(monkeypatch)
    words, x, alpha, aux = edge_rows()
    fused = torch.zeros((12, 2), dtype=torch.uint8)
    PPD.screen_into(fused, x, alpha, aux, fused=True)
    ref = torch.zeros((12, 2), dtype=torch.uint8)
    PPD.screen_into(ref, x, alpha, aux, fused=False)
    want = tuple(i for i, row in enumerate(words) if not CW.row_passes([int(v) for v in row]))
    assert PPD.excluded(fused[:, 0].bool(), fused[:, 1].bool(), x) == PPD.excluded(ref[:, 0].bool(), ref[:, 1].bool(), x) == want
    assert torch.equal(fused, ref)



@pytest.mark.parametrize("rows, tiles", [(12, 40), (5, 1), (0, 33)])
def test_a_graphs_scatter_screens_its_rows_into_the_pass_and_copies_its_leaves_and_root_at_the_slots_offsets(monkeypatch, rows,
                                                                                                           tiles):
    """A graph's last launch under Triton's CPU interpreter: the flags it writes at the slot's flag offset are `screen_into`'s
    for its first `rows` rows, its leaves and A root land at their offsets, and no other byte of the pass's buffers changes."""
    interpreted_triton(monkeypatch)
    PS = importlib.import_module("verity_vllm.protocols.pouw.pearl_c.screen")
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


def test_a_grouped_scatter_screens_each_flag_pair_at_its_row_of_the_map(monkeypatch):
    """A grouped call's scatter under Triton's CPU interpreter, with a row map: flag pair i is `screen_into`'s for row at[i]
    (of x, α and (s, ρ)), at the slot's flag offset, and the leaves and A root land as without one."""
    interpreted_triton(monkeypatch)
    from verity_vllm.protocols.pouw.pearl_c import screen as PS
    _, x, alpha, aux = edge_rows()
    flags = torch.zeros((12, 2), dtype=torch.uint8)
    PS.screen_into(flags, x, alpha, aux, fused=True)
    at = torch.tensor([5, 0, 11, 2, 4, 3])
    leaves, root = torch.arange(64, dtype=torch.uint8), torch.arange(100, 132, dtype=torch.uint8)
    bufs = [torch.full((n,), 0xA5, dtype=torch.uint8) for n in (256, 64, 128)]
    want = [t.clone() for t in bufs]
    want[0][32:96], want[1][10:10 + 2 * len(at)], want[2][64:96] = leaves, flags[at].flatten(), root
    PS.scatter_into(x, alpha, aux, len(at), bufs[1], leaves, bufs[0], root, bufs[2], torch.tensor([32, 10, 64]), at)
    assert all(torch.equal(a, b) for a, b in zip(bufs, want, strict=True))
    assert len({tuple(r) for r in flags[at].tolist()}) > 1

# ---- step 2 of the decode plan: the deferred schedule -------------------------------------------------------------

@pytest.mark.parametrize("schedule, main", [("deferred", 2), ("deferred-caller", 0)])
def test_the_deferred_schedule_hashes_tiles_on_the_side_stream_from_rotating_sets(device_run, schedule, main):
    """Under ``deferred`` the call's other steps run on the run's own main lane (the recording stand-in's handle 2), as measured;
    under ``deferred-caller`` on the caller's stream (0).  The tile hashing is on the run's side lanes in turn (1, then 3)
    either way."""
    make, kernel, weights, *_ = device_run
    run = make(schedule=schedule)
    kernel.dev.log.clear()
    for _ in range(3):
        run.gemm(0, torch.randn(32, 1024).to(torch.bfloat16), weights[0], None)
    log = [(n, st, p) for n, st, p, _ in kernel.dev.log if not n.endswith("_smem")]
    sides = [lane.raw for lane in run.sides]
    assert [(n, st) for n, st, _ in log if st in sides] == [("pd_tiles_851", 1), ("pd_tiles_851", 3), ("pd_tiles_851", 1)]
    assert sides == [1, 3] and {st for n, st, _ in log if st not in sides} == {main}
    sets = run.shapes[(32, 256, 1024)].sets
    ptr = lambda j: (sets[j]["c"].data_ptr(), sets[j]["u"].data_ptr())
    gemms = [p[2:4] for n, _, p in log if n == "gemm_pearlc_d"]
    leaves = [p[:2] for n, _, p in log if n == "pd_tiles_851"]
    assert gemms == leaves == [ptr(0), ptr(1), ptr(0)] and ptr(0) != ptr(1)
    out = [p[2] for n, _, p in log if n == "pd_tiles_851"]             # into the pass's leaves, 4 tiles a call
    assert out == [run.leaves.data_ptr() + 64 * 4 * j for j in range(3)]
    run.begin()
    run.poison()
    assert all(bool((t == 0xA5).all()) for st in sets for t in st.values())


def test_calls_take_the_side_lanes_in_turn_whatever_their_shape(device_run):
    """Two shapes' calls, each shape on its own ring, alternate the run's side lanes call by call (1, 3, 1, 3), so a call's
    side steps run beside the next call's whatever its shape; each call hashes its own shape's set's C~ and U."""
    make, kernel, weights, *_ = device_run
    run = make(schedule="deferred")
    kernel.dev.log.clear()
    for w in (0, 1, 0, 1):
        run.gemm(w, torch.randn(32, 1024).to(torch.bfloat16), weights[w], None)
    tiles = [(st, tuple(p[:2])) for n, st, p, _ in kernel.dev.log if n.startswith("pd_tiles_")]
    cu = [(lambda s: (s["c"].data_ptr(), s["u"].data_ptr()))(run.shapes[(32, weights[w].shape[0], 1024)].sets[j])
          for j in (0, 1) for w in (0, 1)]
    assert tiles == list(zip((1, 3, 1, 3), cu, strict=True)) and len(set(cu)) == 4


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
    """Every tree's hash is the scheme's format's (`verity.protocols.pouw.verifier.serving.HASH_FORMATS`), not a
    literal: -h3's hm96-sha512, keyed by the prover's salt key; and the path runs only its own schemes."""
    from verity.protocols.pouw.verifier.serving import HASH_FORMATS, SchemeId, hash_format

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


def test_each_device_records_kernels_refuse_their_own_grain():
    """pearl_c_sm120's kernels take k a multiple of 1,024 and n of 128, pearl_c_sm90's (h100's) k of 256 and n of 64, inside
    the scheme's domain: an n = 2,624, k = 1,536 linear (GLM-5.3's at TP 8) runs on sm_90a and not on sm_120, n = 160 on
    neither, k = 512 is under both schemes' k_min, and k = 256 is h100's census k and under sm_120's grain."""
    shapes = {0: (2624, 1536), 1: (160, 6144), 2: (7168, 512), 3: (4096, 4096), 4: (7168, 256)}
    sm90 = PPD.refusals("pearl-c-h100-v1-h3", shapes)
    assert [r.split(":")[0] for r in sm90] == ["linear 1", "linear 2"] and "k a multiple of 256 and n of 64" in sm90[0]
    assert "1024 <= k" in sm90[1]
    sm120 = PPD.refusals(SCHEME, shapes)
    assert [r.split(":")[0] for r in sm120] == ["linear 0", "linear 1", "linear 2", "linear 4"]
    assert "k a multiple of 1,024 and n of 128" in sm120[0]


# ---- the schedule as data ---------------------------------------------------------------------------------------------

#: pearl_c_sm90's pipeline (pearl-c-h100-v1-h3's), whose calls split A's commitment (`run.Pipeline.FUSED`'s tree_seed_a,
#: past pd's rows `FUSED_GENERIC`'s)
R90 = load("pearl_c_sm90_run", KERNEL.parent / "pearl_c_sm90" / "run.py")
F90 = load("pearl_c_sm90_fixture", KERNEL.parent / "pearl_c_sm90" / "fixture.py")
#: hash_prefill.cuh's host build (`build`), with chain "pl"'s kernels beside hash_decode.cuh's
P90 = load("pearl_c_sm90_prefill_test", Path(__file__).with_name("test_pearl_c_sm90_prefill.py"))
H100_SCHEME = "pearl-c-h100-v1-h3"


class H100Dev(RecordingDev):
    """`RecordingDev` with an H100's 132 SMs and opt-in shared memory."""

    sms = 132

    def get(self, f, *_):
        return {"cuDeviceGetAttribute": 232448, "cuFuncGetAttribute": 0}[f]


@pytest.fixture
def sm90_run():
    rng = np.random.default_rng(3)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    kernel = PPD.Kernel(R90, F90, H100Dev(), "0" * 64)
    roots = PPD.weight_roots(kernel, weights, H100_SCHEME, salt_key=SALT_KEY)
    coins = salt_coin(roots)

    def make(schedule="deferred-a"):
        return PPD.PearlCDeviceRun(PPD.scheme_of(H100_SCHEME), f"pouw:{H100_SCHEME}", kernel, weights, roots, *coins,
                                   capacity=1 << 12, schedule=schedule, salt_key=SALT_KEY)
    return make, kernel, weights, roots


def test_a_schedule_this_path_the_kernel_or_the_format_cannot_run_is_refused(device_run, sm90_run):
    """-h3's row seeds come from A's rows, not its root, so its format lets A's tree leave the main lane; this path moves the
    tile hashing, which reads C~ and U, and A's tree where the kernel's pipeline splits A's commitment (sm_90a's and
    sm_120's), each with the buffers it touches rotated.  The screen stays on the caller."""
    from verity.protocols.pouw.verifier.serving import Schedule
    make, kernel, *_ = device_run
    s = PPD.scheme_of(SCHEME)
    assert PPD.schedule_of("deferred-a", s).name == "deferred-a" and PPD.splits(kernel)
    assert make(schedule="deferred-a").schedule.name == "deferred-a"
    whole = types.SimpleNamespace(run=types.SimpleNamespace(Pipeline=types.SimpleNamespace(FUSED={"commit_a": ()})))
    with pytest.raises(ValueError, match="this kernel's pipeline doesn't split A's commitment"):
        PPD.schedule_of(Schedule("early-a", frozenset({"a_tree", "tile_hash"}), ring=2), s, whole)
    with pytest.raises(ValueError, match="doesn't split A's commitment"):
        PPD.schedule_of("deferred-a", s, whole)
    with pytest.raises(ValueError, match="this path runs only a_tree, tile_hash on a side lane, not screen"):
        PPD.schedule_of(Schedule("screen-aside", frozenset({"tile_hash", "screen"}), ring=2), s)
    with pytest.raises(ValueError, match="a side lane needs ring >= 2"):
        PPD.schedule_of(Schedule("one-set", frozenset({"tile_hash"}), ring=1), s)
    assert PPD.rotated(PPD.schedule_of("deferred", s)) == ("c", "segdig_t", "u") and PPD.rotated(PPD.SERIAL) == ()
    assert PPD.rotated(PPD.SCHEDULES["deferred-a"]) == ("c", "keys", "line_a", "rowdig_a", "seed_a", "segdig_t", "tree_a",
                                                     "u")
    assert PPD.PearlCExecutor(sm90_run[0]()).schedules == {"serial", "deferred", "deferred-caller", "deferred-a"}


def test_deferred_a_builds_as_tree_beside_the_tile_hashing_from_rotating_sets(sm90_run):
    """Under ``deferred-a`` a decode call's main lane (the recording stand-in's handle 2) runs pd_commit in mode 1 (``rows_a``:
    each real row's digest, leaf, seed, line key and E_A line), the forming and the GEMMs; below a full tile of rows the
    call's side lane runs pd_tree (A's levels, seed_A and line_a, and no padding rows' E_A lines, though m₁₆ = 32 has
    them), then the tile leaves, on the run's side lanes in turn (1, then 3); from a full tile (64) the call's side lane
    runs the tile leaves and the side lane after it pd_tree (tiles on 1, then 3; the tree on 3, then 1); no further
    stream.  The lanes use the ring's set for the call: A's tree, seed_A, line_a, the keys, C~ and U; E_A's lines are the
    shape's own, which only the main lane touches.  The call's keys are uploaded into its set's, and A's root is copied
    to the pass from its set."""
    make, kernel, weights, roots = sm90_run
    run = make()
    sh = run._shape(32, 256, 1024)
    size, nb = sh.size["tree_a"], run.nb
    for j, st in enumerate(sh.sets):                   # stand-in roots, since the recording stand-in writes nothing
        st["tree_a"][size - nb:size].fill_(0x11 * (j + 1))
    kernel.dev.log.clear()
    for _ in range(3):
        run.gemm(0, torch.randn(20, 1024).to(torch.bfloat16), weights[0], None)
    log = [(n, st, p, i) for n, st, p, i in kernel.dev.log if not n.endswith("_smem")]
    sides = [lane.raw for lane in run.sides]
    assert [(n, st) for n, st, _, _ in log if st in sides] == [(n, st) for lane in (1, 3, 1)
                                                               for n, st in (("pd_tree_851", lane), ("pd_tiles_851", lane))]
    assert sides == [1, 3] and {st for _, st, _, _ in log if st not in sides} == {2}
    full = make()
    kernel.dev.log.clear()
    for _ in range(3):
        full.gemm(0, torch.randn(64, 1024).to(torch.bfloat16), weights[0], None)
    assert [(n, st) for n, st, *_ in kernel.dev.log if st in sides and n.startswith("pd_")] == [
        (n, st) for tree, tiles in ((3, 1), (1, 3), (3, 1)) for n, st in (("pd_tree_851", tree), ("pd_tiles_979", tiles))]
    by = {name: [(p, i) for n, _, p, i in log if n == name] for name in ("pd_commit_851", "pd_tree_851", "form_s5_ep",
                                                                          "gemm_pearlc_d", "pd_tiles_851")}
    ptr = lambda j, x: (sh.keys if j == 0 and x == "keys" else sh.sets[j][x]).data_ptr()
    for c, j in enumerate((0, 1, 0)):
        (cp, ci), (tp, ti) = by["pd_commit_851"][c], by["pd_tree_851"][c]
        assert ci == (32, 1024, 1, 64) and ti == (32, 0, 0, 64)
        assert cp[2] == tp[2] == ptr(j, "tree_a") and cp[12] == sh.b["ea_codes"].data_ptr() == by["form_s5_ep"][c][0][2]
        assert tp[12] == 0
        assert (tp[10], tp[11]) == (ptr(j, "seed_a"), ptr(j, "line_a"))
        assert cp[6] == tp[6] == ptr(j, "keys") and cp[5] == tp[5] == ptr(j, "keys") + nb
        assert by["gemm_pearlc_d"][c][0][2:4] == by["pd_tiles_851"][c][0][:2] == (ptr(j, "c"), ptr(j, "u"))
    assert len({ptr(j, x) for j in (0, 1) for x in PPD.rotated(run.schedule)}) == 16
    for j, index in ((0, 2), (1, 1)):
        keys = sh.keys if j == 0 else sh.sets[j]["keys"]
        assert PPD._bytes(keys[:168]) == PPD.call_keys(F90, run.salt, roots[0], index, 32, run.tree_hash)
    assert [set(PPD._bytes(run.roots_buf[nb * c:nb * (c + 1)])) for c in range(3)] == [{0x11}, {0x22}, {0x11}]
    assert run.end()["schedule"] == "deferred-a"


def test_the_served_whole_capture_launches_each_calls_a_tree_as_one_cta_on_the_tree_stream(sm90_run, monkeypatch):
    """Inside a whole capture under the served arm's options (``whole_defer``, ``whole_defer_a``, ``whole_fork_stats``,
    ``whole_tree_sm``) each decode call of m = 32 splits A's commitment: pd_commit in mode 1 (``rows_a``, a CTA a real row)
    on the capturing stream, then A's tree in a launch of its own, pd_tree's one CTA, the tree stream's only launch, over the
    same tree, both at PD_WHOLE's shared memory; the forward's last call joins the tree stream.  Under ``whole_pd_shared`` too
    the capture's key says so and every launch is the same but pd_commit's and pd_tree's shared memory, half an SM's, and
    the pipelines are left without it.  Without ``whole_defer_a`` the graphs made split aren't replayed, and graphs made
    unsplit launch pd_commit in mode 0 alone.  The graphs are made as `_replay` makes them, a stand-in in the scatter's place
    (a Triton launch)."""
    G = PPD.graphs
    make, kernel, weights, _ = sm90_run
    run = make(schedule="serial")
    run.graphs, run.order = True, {w: i for i, w in enumerate(weights)}
    run.key_slots = torch.zeros((len(weights), G.SLOT_BYTES), dtype=torch.uint8)
    for x in ("whole_defer", "whole_defer_a", "whole_fork_stats", "whole_tree_sm"):
        setattr(run, x, True)
    log, launched = [], []

    class Lane:
        def __init__(self, name, raw):
            self.name, self.cuda_stream = name, raw

        def wait_stream(self, other):
            log.append(f"{self.name} waits for {other.name}")

        def wait_event(self, event):
            log.append(f"{self.name} waits for {event.at}'s mark")

    class Mark:
        def record(self, stream):
            self.at = stream.name

    class On:
        def __init__(self, s):
            self.s = s

        def __enter__(self):
            self.held, current[0] = current[0], self.s

        def __exit__(self, *exc):
            current[0] = self.held

    current, made = [Lane("main", 0)], iter((Lane("side", 20), Lane("tree", 30), Lane("stats", 40)))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: current[0])
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: next(made))
    monkeypatch.setattr(torch.cuda, "stream", On)
    monkeypatch.setattr(torch.cuda, "Event", Mark)
    monkeypatch.setattr(G, "_capturing", lambda: True)
    record = kernel.dev.launch

    def launch(name, grid, block, smem=0, p=(), i=(), stream=None):
        if not name.endswith("_smem"):                     # the GEMMs' probes, at their first use alone
            launched.append((name, grid, block, smem, kernel.dev.stream if stream is None else stream, tuple(p), tuple(i)))
        record(name, grid, block, smem, p, i, stream)
    monkeypatch.setattr(kernel.dev, "launch", launch)
    shapes = {w: run._shape(32, run.weights[w].n, 1024) for w in weights}
    xs = {w: torch.randn(32, 1024).to(torch.bfloat16) for w in weights}

    def graphs():
        run._graphs.clear()
        for w, sh in shapes.items():
            pipe = run._pipe(w, sh)[0]
            g = G._Graph(torch.zeros((sh.mp, sh.n), dtype=torch.bfloat16), torch.zeros(run.nb * 8, dtype=torch.uint8), "graph")
            g.steps = [*run._call_steps(pipe), lambda: log.append(f"scatter on {current[0].name}")]
            g.cuts, g.tree_at, g.stats_at = run._call_cuts(pipe), run._call_tree(pipe), run._call_stats(pipe)
            run._graphs[run._graph_key(w, sh, 32)[0]] = g

    def forward():
        launched.clear()
        log.clear()
        run.whole_capture_begin(run.whole_key(False))
        for w in weights:
            run.gemm(w, xs[w], weights[w], None)
        assert [c.w for c in run.whole_capture_end()] == list(weights)
        return list(launched)

    def pd(got):
        return [x for x in got if x[0].startswith(("pd_commit_", "pd_tree_"))]

    graphs()
    split = forward()
    pipes = [run._pipe(w, sh)[0] for w, sh in shapes.items()]
    assert all(p.split_a and p.pd_a == 851 for p in pipes) and "pd_shared" not in run.whole_key(False)
    whole, half = 232448, (232448 - R90.CTA_RESERVED_SMEM) // 2
    assert [x[:5] + x[6:] for x in pd(split)] == [x for _ in weights for x in (
        ("pd_commit_851", 32, 128, whole, 0, (32, 1024, 1, 64)), ("pd_tree_851", 1, 128, whole, 30, (32, 0, 0, 64)))]
    assert all(c[5][2] == t[5][2] == p.b["tree_a"] for c, t, p in zip(pd(split)[::2], pd(split)[1::2], pipes, strict=True))
    assert {x[0] for x in split if x[4] == 30} == {"pd_tree_851"} and log.count("tree waits for main") == len(weights)
    assert log[-2:] == ["main waits for side", "main waits for tree"]

    run.whole_pd_shared = True
    assert "pd_shared" in run.whole_key(False)
    shared = forward()
    assert [x[3] for x in pd(shared)] == [half] * 2 * len(weights) and not any(p.pd_shared for p in pipes)
    assert [x[:3] + x[4:] for x in shared] == [x[:3] + x[4:] for x in split]
    assert [x for x in shared if x not in pd(shared)] == [x for x in split if x not in pd(split)]

    run.whole_pd_shared = run.whole_defer_a = run.whole_tree_sm = False
    with pytest.raises(RuntimeError, match="no graph yet"):
        forward()
    run.whole_capture_end()
    graphs()
    unsplit = pd(forward())
    assert [(x[0], x[4], x[6][2], x[3]) for x in unsplit] == [("pd_commit_851", 0, 0, whole)] * len(weights)


def test_deferred_a_runs_sm120s_split_on_the_lanes(device_run, tmp_path):
    """sm_120's pipeline splits A's commitment as sm_90a's does (`run.Pipeline`'s ``split_a``), so under ``deferred-a`` a
    decode call's main lane (handle 2) runs pd_commit in mode 1 and a side lane pd_tree, both on pd_commit's variant (851,
    also where the tiles take 979), writing no padding rows' E_A lines, over the ring's set for the call: below a full
    tile of rows the call's side lane runs the tree, then the tile leaves (lanes 1, 3, 1); from a full tile the side lane
    after the call's runs the tree.  A retained pass's record keeps the tree of the call's set."""
    make, kernel, weights, *_ = device_run
    run = make(schedule="deferred-a")
    sh = run._shape(32, 256, 1024)
    kernel.dev.log.clear()
    for _ in range(3):
        run.gemm(0, torch.randn(20, 1024).to(torch.bfloat16), weights[0], None)
    log = [(n, st, p, i) for n, st, p, i in kernel.dev.log if not n.endswith("_smem")]
    sides = [lane.raw for lane in run.sides]
    assert sides == [1, 3] and [(n, st) for n, st, _, _ in log if st in sides] == [
        (n, lane) for lane in (1, 3, 1) for n in ("pd_tree_851", "pd_tiles_851")]
    assert {st for _, st, _, _ in log if st not in sides} == {2}
    ptr = lambda j, x: (sh.keys if j == 0 and x == "keys" else sh.sets[j][x]).data_ptr()  # noqa: E731
    commits = [(p, i) for n, _, p, i in log if n == "pd_commit_851"]
    trees = [(p, i) for n, _, p, i in log if n == "pd_tree_851"]
    for (cp, ci), (tp, ti), j in zip(commits, trees, (0, 1, 0), strict=True):
        assert ci == (32, 1024, 1, 64) and ti == (32, 0, 0, 64) and tp[12] == 0
        assert cp[2] == tp[2] == ptr(j, "tree_a") and (tp[10], tp[11]) == (ptr(j, "seed_a"), ptr(j, "line_a"))
        assert cp[6] == tp[6] == ptr(j, "keys") and cp[12] == sh.b["ea_codes"].data_ptr()
    full = make(schedule="deferred-a")
    kernel.dev.log.clear()
    for _ in range(3):
        full.gemm(0, torch.randn(64, 1024).to(torch.bfloat16), weights[0], None)
    assert [(n, st) for n, st, *_ in kernel.dev.log if st in sides and n.startswith("pd_")] == [
        (n, st) for tree, tiles in ((3, 1), (1, 3), (3, 1)) for n, st in (("pd_tree_851", tree), ("pd_tiles_979", tiles))]
    kept_run = make(schedule="deferred-a")
    kept_run.begin(retain=str(tmp_path / "kept"), run=RUN)
    kept = []
    kept_run._retained = lambda cur, mm, m, flags, xa, tree, y=None: kept.append(tree.data_ptr())
    kernel.dev.log.clear()
    kept_run.gemm(0, torch.randn(20, 1024).to(torch.bfloat16), weights[0], None)
    pd = [(n, st, p, i) for n, st, p, i in kernel.dev.log if n.startswith("pd_") and not n.endswith("_smem")]
    assert [(n, st) for n, st, _, _ in pd] == [("pd_commit_851", 2), ("pd_tree_851", 1), ("pd_tiles_851", 1)]
    assert pd[0][3][2] == 1 and kept == [pd[1][2][2]] and pd[1][2][12] == 0


def test_an_eager_forward_under_whole_eager_runs_a_prefills_calls_with_as_tree_and_tiles_aside(device_run):
    """With ``whole_eager`` a forward a whole-step arm runs eagerly (`eager_forward`, before the pass's first call) runs under
    that schedule.  Under ``deferred-a``, at a prefill's rows (m₁₆ = 304, outside pd), sm_120's pipeline splits A's
    commitment over the generic kernels (`run.Pipeline.FUSED_GENERIC`): the main lane (handle 2) runs ``rows_a`` (A's pads,
    the row digests, the row seeds and the real rows' E_A lines, keyed per row and reading no line_a), the forming and the
    GEMMs, and never A's level 0; from a full tile of rows the side lane after the call's runs A's level 0 over the row
    digests of the call's set, its levels and seed_A, and the call's side lane the tile leaves.  A second eager_forward,
    after a call, changes nothing; the next pass is back on its own schedule, and without whole_eager nothing moves.  A
    level 0 reads its unit index (its salt nonce) from its own set's keys."""
    make, kernel, weights, *_ = device_run
    assert make().whole_eager is None and (r := make()) and (r.eager_forward() or r.schedule.name) == "serial"
    run = make(whole_eager="deferred-a")
    assert run.schedule.name == "serial" and run.whole_eager.name == "deferred-a"
    run.eager_forward()
    assert run.schedule.name == "deferred-a"
    sh = run._shape(304, 256, 1024)
    kernel.dev.log.clear()
    for _ in range(2):
        run.gemm(0, torch.randn(300, 1024).to(torch.bfloat16), weights[0], None)
    run.eager_forward()
    assert run.schedule.name == "deferred-a"
    log = [(n, st, p, i) for n, st, p, i in kernel.dev.log if not n.endswith("_smem")]
    sides = [lane.raw for lane in run.sides]
    assert sides == [1, 3] and {st for _, st, _, _ in log if st not in sides} == {2}
    main = [n for n, st, _, _ in log if st == 2]
    rows_a = ["s512_pads", "s512_row_segs", "s512_folds", "s512_seeds", "lines"]
    assert main[:5] == rows_a and main.count("lines") == 2 and "hm_leaves" not in main and "gemm_pearlc_d" in main
    levels = (R.pow2(304) // R.S512_SMALL).bit_length() - 1        # s512_nodes down to S512_SMALL nodes, then tree_small
    tree = ["hm_leaves", "s512_frame_leaves"] + ["s512_nodes"] * levels + ["s512_tree_small", "s512_seeds"]
    on = lambda lane: [n for n, st, _, _ in log if st == lane]  # noqa: E731
    assert on(3) == tree + ["s512_tile_segs", "s512_folds"] and on(1) == ["s512_tile_segs", "s512_folds"] + tree
    ptr = lambda j, x: sh.sets[j][x].data_ptr()  # noqa: E731
    lines = [(p, i) for n, st, p, i in log if n == "lines"]
    assert all(i == (0, 0, 304, R.NORM_256, 304) and p[0] == 0 for p, i in lines)
    folds = [p for n, st, p, i in log if n == "s512_folds" and st == 2]
    leaves = [p for n, st, p, i in log if n == "hm_leaves"]
    for c, j in enumerate((0, 1)):
        assert folds[c][1] == leaves[c][0] == ptr(j, "rowdig_a") and leaves[c][1] == ptr(j, "tree_a")
        keys = sh.keys if j == 0 else sh.sets[j]["keys"]
        assert leaves[c][5] == keys.data_ptr() + run.nb + 96      # A's unit index, its salt nonce: the set's own keys
    assert run.end()["schedule"] == "deferred-a"
    run.begin()
    assert run.schedule.name == "serial" and run._eager_from is None


def test_a_prefills_rows_past_mainloop_rows_pad_to_the_mainloop_tile(device_run):
    """A call of more than `run.MAINLOOP_ROWS` rows pads to the 128-row tile, so its chain runs on the mainloop
    (``gemm_pearlc``, its operands by tensor map); at fewer rows the 64-row tile and the 64 x 64 chain."""
    make, kernel, weights, *_ = device_run
    run = make()
    assert run._shape(R.MAINLOOP_ROWS, 256, 1024).mp == R.MAINLOOP_ROWS and run._shape(304, 256, 1024).mp == 320
    assert run._shape(R.MAINLOOP_ROWS + 16, 256, 1024).mp == R.MAINLOOP_ROWS + 128
    for m, chain in ((600, "gemm_pearlc"), (300, "gemm_pearlc_d")):
        kernel.dev.log.clear()
        run.gemm(0, torch.randn(m, 1024).to(torch.bfloat16), weights[0], None)
        names = [n for n, *_ in kernel.dev.log]
        assert chain in names and {"gemm_pearlc", "gemm_pearlc_d"} - {chain} - set(names) == {"gemm_pearlc", "gemm_pearlc_d"} - {chain}


def test_deferred_a_runs_as_tree_aside_in_a_retained_pass_too_and_writes_no_padding_lines(sm90_run, tmp_path):
    """A retained pass's record reads A's tree once the side lane is done with it, so there too pd_commit runs in mode 1 on
    the main lane and pd_tree on a side lane, and the record keeps the tree of the call's set.  pd_tree writes no padding
    rows' lines, at m₁₆ = 32 (which has them) as at 64, the kernels' tile."""
    make, kernel, weights, _ = sm90_run
    run = make()
    run.begin(retain=str(tmp_path / "kept"), run=RUN)
    kept = []
    run._retained = lambda cur, mm, m, flags, xa, tree, y=None: kept.append(tree.data_ptr())
    kernel.dev.log.clear()
    run.gemm(0, torch.randn(20, 1024).to(torch.bfloat16), weights[0], None)
    pd = [(n, st, p, i) for n, st, p, i in kernel.dev.log if n.startswith("pd_") and not n.endswith("_smem")]
    assert [(n, st) for n, st, _, _ in pd] == [("pd_commit_851", 2), ("pd_tree_851", 1), ("pd_tiles_851", 1)]
    assert pd[0][3][2] == 1 and kept == [pd[1][2][2]] and pd[1][2][12] == 0
    run.begin()
    kernel.dev.log.clear()
    run.gemm(0, torch.randn(64, 1024).to(torch.bfloat16), weights[0], None)
    (tree,) = [(p, i) for n, _, p, i in kernel.dev.log if n == "pd_tree_851"]
    assert tree[1] == (64, 0, 0, 64) and tree[0][12] == 0


def test_deferred_a_starts_as_tree_right_after_the_per_row_keying_at_every_m16(sm90_run):
    """A call starts A's tree on its side lane right after the main lane's pd_commit (the pipeline's ``lane_cuts``: one main
    step, then one side step), with or without padding rows (m₁₆ = 32, 64), since the split pipeline writes no padding rows'
    E_A lines (``fillers`` off), so pd_tree writes nothing the forming or the GEMM read.  Under ``deferred`` A's tree stays
    on the main lane, in pd_commit's mode 0, which writes them."""
    make, *_ = sm90_run
    run = make()
    for m16 in (32, 64):
        sh = run._shape(m16, 256, 1024)
        pipe, _, side = run._pipe(0, sh)
        assert pipe.lane_cuts == (1, 1) and len(side) == 2 and "a_tree" in run._side_groups(sh)
        assert pipe.split_a and not pipe.fillers
    run.begin(schedule="deferred")
    sh = run._shape(32, 256, 1024)
    pipe, _, side = run._pipe(0, sh)
    assert pipe.lane_cuts == (1, 0) and len(side) == 1 and run._side_groups(sh) == {"tile_hash"}
    assert not pipe.split_a and pipe.fillers


def test_deferred_a_runs_as_tree_aside_only_where_the_call_keeps_its_step(sm90_run, monkeypatch):
    """A call runs A's tree on its side lane (and starts it behind the main lane's per-row keying) only where its side steps
    hold the tree: the e2e control's omits (A's commitment and the tile leaves) leave its side lane no step, so it starts no
    tree there and its call still runs; omitting the tile leaves alone keeps the tree aside."""
    E2E = load("pearl_c_vllm_e2e", KERNEL.parent / "pearl_c_vllm" / "e2e.py")
    make, kernel, weights, _ = sm90_run
    run = make()
    seen, lanes = [], run._lanes
    monkeypatch.setattr(run, "_lanes", lambda *a, **k: seen.append((len(a[4]), a[2].lane_cuts, a[8])) or lanes(*a, **k))
    for omit in (E2E.CONTROL_OMITS, frozenset({"hash_leaf"}), frozenset()):
        run.begin()
        run.omit = omit
        kernel.dev.log.clear()
        run.gemm(0, torch.randn(20, 1024).to(torch.bfloat16), weights[0], None)
        assert any(n == "gemm_pearlc_d" for n, *_ in kernel.dev.log)
    run.omit = frozenset()
    assert seen == [(0, (0, 0), False), (1, (1, 1), True), (2, (1, 1), True)]


def test_deferred_a_at_a_prefills_rows_runs_as_tree_aside_over_chain_pl(sm90_run):
    """Past pd's rows (m₁₆ = 304, a prefill's), sm_90a's pipeline splits A's commitment over chain "pl"'s launches
    (`run.Pipeline.FUSED_GENERIC`), as sm_120's over its generic kernels: the main lane runs ``rows_a`` (A's pads, the row
    digests, the row seeds and the real rows' E_A lines, keyed per row and reading no line_a) and never A's level 0, and
    from a full tile of rows the side lane after the call's runs A's level 0, its levels and seed_A."""
    make, kernel, weights, _ = sm90_run
    run = make()
    sh = run._shape(304, 256, 1024)
    pipe, _, _ = run._pipe(0, sh)
    assert pipe.pd is None and pipe.pl and pipe.split_a and not pipe.fillers and "a_tree" in run._side_groups(sh)
    assert pipe.FUSED == {**R90.Pipeline.FUSED, **R90.Pipeline.FUSED_GENERIC}
    kernel.dev.log.clear()
    run.gemm(0, torch.randn(300, 1024).to(torch.bfloat16), weights[0], None)
    log = [(n, st, p, i) for n, st, p, i in kernel.dev.log if not n.endswith("_smem")]
    sides = [lane.raw for lane in run.sides]
    main = [n for n, st, _, _ in log if st not in sides]
    v = R90.PL_VARIANT
    assert main[:5] == ["s512_pads", "s512_row_segs", "s512_folds", f"pl_seeds_{v}", "lines"] and main.count("lines") == 1
    assert not {f"pl_leaves_{v}", f"pl_levels_{v}", f"pl_seed_a_{v}"} & set(main) and "gemm_pearlc_d" in main
    levels = -(-(R90.pow2(304).bit_length() - 1) // R90.PL_LEVELS)
    tree = [f"pl_leaves_{v}"] + [f"pl_levels_{v}"] * levels + [f"pl_seed_a_{v}"]
    assert [n for n, st, _, _ in log if st == sides[1]][:len(tree)] == tree
    (lines,) = [(p, i) for n, _, p, i in log if n == "lines"]
    assert lines[0][0] == 0 and lines[1] == (0, 0, 304, R90.NORM_256, 304)


class H100TwinDev(TwinDev):
    """`TwinDev` with an H100's 132 SMs and opt-in shared memory, for pearl_c_sm90's pipeline."""

    sms = 132

    def get(self, f, *_):
        return {"cuDeviceGetAttribute": 232448, "cuFuncGetAttribute": 0}[f]


@pytest.mark.slow
@pytest.mark.skipif(not (shutil.which("g++") or shutil.which("c++")), reason="simt_twin needs a C++ compiler")
@pytest.mark.parametrize("arch", ["sm90", "sm120"])
def test_deferred_a_commits_what_the_serial_one_commits_and_leaves_the_padding_lines(tmp_path, arch):
    """Under ``deferred-a`` (A's commitment split, its tree on the side lane, no padding rows' E_A lines) a retained pass
    commits what the serial one commits, on sm_90a's pipeline and on sm_120's, its trees computed on the host
    (`H100TwinDev` or `TwinDev`: hash_decode.cuh's host build and simt_twin): the records root, the commitment and each
    call's A root, at m₁₆ with padding rows (20 and 32 rows of the kernels' 64) and without (64), and the committed rows'
    E_A lines.  The padding rows' lines are serial's under line_a and deferred-a's the pass's poison, which nothing
    commits or serves."""
    pd = HD.build(tmp_path)
    rng = np.random.default_rng(23)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    xs = [(w, torch.from_numpy(bf16_words(rng, m, 1024).view(np.float32).copy()).to(torch.bfloat16))
          for w, m in ((0, 20), (1, 32), (0, 64), (1, 20), (0, 32))]
    scheme = H100_SCHEME if arch == "sm90" else SCHEME
    kernel = (PPD.Kernel(R90, F90, H100TwinDev(pd), "0" * 64) if arch == "sm90" else
              PPD.Kernel(R, F, TwinDev(pd), "0" * 64))
    roots = PPD.weight_roots(kernel, weights, scheme, salt_key=SALT_KEY)
    got = {}
    for schedule in ("serial", "deferred-a"):
        run = PPD.PearlCDeviceRun(PPD.scheme_of(scheme), f"pouw:{scheme}", kernel, weights, roots,
                                  *salt_coin(roots), capacity=1 << 12, schedule=schedule, salt_key=SALT_KEY)
        run.begin(retain=str(tmp_path / schedule), run=RUN)
        lines = []
        for w, x in xs:
            run.gemm(w, x, weights[w], None)
            m16 = -(-x.shape[0] // 16) * 16
            sh = run.shapes[(m16, weights[w].shape[0], 1024)]
            ea = PPD._bytes(sh.b["ea_codes"])[:sh.size["ea_codes"]]
            lines.append((ea[:32 * m16], ea[32 * m16:]))
        man = json.loads(Path(run.end()["manifest"]).read_text())
        got[schedule] = ({k: man[k] for k in ("records_root", "commitment", "salt")},
                         [{k: u[k] for k in ("root_a", "excluded_a", "real_rows", "m", "n")} for u in man["matmuls"]], lines)
    (serial, s_units, s_lines), (deferred, d_units, d_lines) = got["serial"], got["deferred-a"]
    assert deferred == serial and d_units == s_units and len({u["root_a"] for u in s_units}) == len(xs)
    assert [a for a, _ in d_lines] == [a for a, _ in s_lines]
    assert [len(p) for _, p in d_lines] == [32 * 32, 32 * 32, 0, 32 * 32, 32 * 32]
    assert all(set(p) == {0xA5} for _, p in d_lines if p) and all(set(p) != {0xA5} for _, p in s_lines if p)


@pytest.mark.slow
@pytest.mark.skipif(not (shutil.which("g++") or shutil.which("c++")), reason="simt_twin needs a C++ compiler")
@pytest.mark.parametrize("arch", ["sm90", "sm120"])
def test_deferred_a_at_a_prefills_rows_commits_what_the_serial_one_commits(tmp_path, arch):
    """Past pd's rows (a prefill's: A's commitment split over the generic kernels, `run.Pipeline.FUSED_GENERIC`, on sm_90a
    over chain "pl"'s), two calls of each shape in a pass under ``deferred-a`` take the ring's two sets, each call's A tree
    under its own set's keys, and the pass commits what the serial one commits (`H100TwinDev` over hash_prefill.cuh's host
    build, or `TwinDev`: the trees computed on the host): the records root, the commitment and each call's A root, beside
    a pd call."""
    pd = P90.build(tmp_path) if arch == "sm90" else HD.build(tmp_path)
    rng = np.random.default_rng(29)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    xs = [(w, torch.from_numpy(bf16_words(rng, m, 1024).view(np.float32).copy()).to(torch.bfloat16))
          for w, m in ((0, 300), (1, 290), (0, 300), (1, 20), (1, 290))]
    scheme = H100_SCHEME if arch == "sm90" else SCHEME
    kernel = (PPD.Kernel(R90, F90, H100TwinDev(pd), "0" * 64) if arch == "sm90" else
              PPD.Kernel(R, F, TwinDev(pd), "0" * 64))
    roots = PPD.weight_roots(kernel, weights, scheme, salt_key=SALT_KEY)
    got = {}
    for schedule in ("serial", "deferred-a"):
        run = PPD.PearlCDeviceRun(PPD.scheme_of(scheme), f"pouw:{scheme}", kernel, weights, roots, *salt_coin(roots),
                                  capacity=1 << 12, schedule=schedule, salt_key=SALT_KEY)
        run.begin(retain=str(tmp_path / schedule), run=RUN)
        for w, x in xs:
            run.gemm(w, x, weights[w], None)
        if schedule == "deferred-a":
            assert [run.shapes[(304, 256, 1024)].turn, run.shapes[(304, 128, 1024)].turn] == [2, 2]
        man = json.loads(Path(run.end()["manifest"]).read_text())
        got[schedule] = ({k: man[k] for k in ("records_root", "commitment", "salt")},
                         [{k: u[k] for k in ("root_a", "excluded_a", "real_rows", "m", "n")} for u in man["matmuls"]])
    assert got["deferred-a"] == got["serial"] and len({u["root_a"] for u in got["serial"][1]}) == len(xs)


def test_a_lane_graphs_replay_makes_the_eager_calls_driver_calls_in_its_order(sm90_run, monkeypatch):
    """A replayed call (`_replay_lanes`) keeps `_lanes`'s events: the main lane waits for the side steps of its set's last
    call and uploads the slot, then waits for the caller and replays its graph; the side lane after the call's replays
    A's tree behind the main graph's own event (where A's tree is aside, from a full tile of rows), and the call's side
    lane the rest behind the main lane's end, then waits for the tree; below a full tile the call's side lane replays the
    tree, then the rest, with no event between them; the caller waits for the main lane.  A set's first call waits for
    nothing, and its side lane's end becomes the set's event."""
    from verity_vllm.protocols.pouw.pearl_c import host as PPH
    from verity_vllm.protocols.pouw.pearl_c import lanes as PPL
    make, *_ = sm90_run
    run = make()
    calls = []

    class Driver:
        def __getattr__(self, name):
            return lambda *a: calls.append((name, *a))

    monkeypatch.setattr(PPH, "_event", lambda: "slot")
    monkeypatch.setattr(PPL, "_event", lambda: "done0")
    run.driver = Driver()
    side0, side1 = (types.SimpleNamespace(raw=raw, event=f"side{raw // 2}", tree_event=f"tree{raw // 2}") for raw in (1, 3))
    main, sh = types.SimpleNamespace(raw=2, event="main"), types.SimpleNamespace(done=[None, "done1"])
    up, i = run.uploads, run.uploads.i

    def src(k):
        return up.host_ptr + (i + k) % up.slots * up.width
    run._replay_lanes(types.SimpleNamespace(execs=(10, 11, 12), slot_ptr=4096, ev_tree="cut"), sh, 1, side1, side0, 0, main,
                      b"k" * 200)
    run._replay_lanes(types.SimpleNamespace(execs=(20, 0, 22), slot_ptr=8192, ev_tree=None), sh, 0, side0, None, 0, None,
                      b"k" * 200)
    run._replay_lanes(types.SimpleNamespace(execs=(30, 31, 32), slot_ptr=4096, ev_tree="cut2"), sh, 1, side0, side0, 0, main,
                      b"k" * 200)
    assert calls == [
        ("wait", 2, "done1"), ("copy_in", 4096, src(0), 200, 2), ("record", "slot", 2), ("record", "main", 0),
        ("wait", 2, "main"), ("launch", 10, 2), ("record", "side1", 2), ("wait", 1, "cut"), ("launch", 11, 1),
        ("record", "tree1", 1), ("wait", 3, "side1"), ("launch", 12, 3), ("wait", 3, "tree1"), ("record", "done1", 3),
        ("record", "main", 2), ("wait", 0, "main"),
        ("copy_in", 8192, src(1), 200, 0), ("record", "slot", 0), ("launch", 20, 0), ("record", "side0", 0),
        ("wait", 1, "side0"), ("launch", 22, 1), ("record", "done0", 1),
        ("wait", 2, "done1"), ("copy_in", 4096, src(2), 200, 2), ("record", "slot", 2), ("record", "main", 0),
        ("wait", 2, "main"), ("launch", 30, 2), ("record", "side0", 2), ("wait", 1, "cut2"), ("launch", 31, 1),
        ("wait", 1, "side0"), ("launch", 32, 1), ("record", "done1", 1), ("record", "main", 2), ("wait", 0, "main")]
    assert sh.done == ["done0", "done1"] and bytes(up.view[i, :200]) == b"k" * 200


def test_a_lane_graphs_main_steps_run_stats_and_the_screen_on_a_fork_beside_the_per_row_keying(sm90_run, monkeypatch):
    """A lane graph's main steps (`_main_steps`) run ``stats_a``, which reads only A's words, and the screen's scatter where
    A's tree is aside, on the run's fork lane: it starts behind an event recorded on the lane at its start, and the lane
    waits for stats_a before E' (``form_ep_a``, under FORM_EP) and for the whole fork at its end, so the per-row keying's
    end moves one step (the fork's).  Where A's tree stays on the main lane the scatter copies A's root, so it runs last on
    the lane; without the driver (no graphs) the steps are as they were."""
    from verity_vllm.protocols.pouw.pearl_c import graphs as PPG
    make, kernel, *_ = sm90_run
    run = make()
    calls = []

    class Driver:
        def __getattr__(self, name):
            return lambda *a: calls.append((name, *a))

    monkeypatch.setattr(PPG, "_event", iter(["start", "first", "end", "start2", "first2", "end2"]).__next__)
    run.driver = Driver()
    pipe, *_ = run._pipe(0, run._shape(32, 256, 1024))
    assert pipe.main_names == ("rows_a", "stats_a", "form_ep_a", "af", "pa", "gemm_pearlc")
    # from a cubin with form_s5_epw, A'F_B's windows are in the forming launch (run.WORDS_FORM)
    monkeypatch.setattr(kernel.dev, "has", lambda name: name == "form_s5_epw", raising=False)
    other = make()
    other.driver = Driver()
    words, *_ = other._pipe(0, other._shape(32, 256, 1024))
    assert words.main_names == ("rows_a", "stats_a", "form_epw_a", "pa", "gemm_pearlc")
    monkeypatch.delattr(kernel.dev, "has")

    def step(name):
        return lambda: calls.append((name, kernel.dev.stream))
    steps = [step(x) for x in pipe.main_names]
    kernel.dev.stream = 2
    got, shift = run._main_steps(pipe, steps, step("screen"), True)
    for f in got:
        f()
    assert shift == 1 and kernel.dev.stream == 2 and run.fork.raw == -2
    assert calls == [("record", "start", 2), ("wait", -2, "start"), ("stats_a", -2), ("record", "first", -2), ("screen", -2),
                     ("record", "end", -2), ("rows_a", 2), ("wait", 2, "first"), *((x, 2) for x in pipe.main_names[2:]),
                     ("wait", 2, "end")]
    calls.clear()
    got, _ = run._main_steps(pipe, steps, step("screen"), False)
    for f in got:
        f()
    assert calls[:3] == [("record", "start2", 2), ("wait", -2, "start2"), ("stats_a", -2)]
    assert calls[-2:] == [("wait", 2, "end2"), ("screen", 2)] and ("screen", -2) not in calls
    run.driver = None
    assert run._main_steps(pipe, steps, None, True) == (steps, 0)


def test_the_lane_graphs_driver_calls_take_raw_handles_and_raise_with_the_drivers_code():
    from verity_vllm.protocols.pouw.pearl_c import host as PPH
    got = []

    def cu(rc=0):
        def fn(name):
            def f(*a):
                got.append((name, *a))
                return rc
            return f
        return types.SimpleNamespace(**{name: fn(name) for name in PPH._Driver.SIGS})

    lib = cu()
    D, ev, stream = PPH._Driver(lib), types.SimpleNamespace(cuda_event=77), 9
    D.launch(5, stream)
    D.record(ev, stream)
    D.wait(stream, ev)
    D.copy_in(1 << 40, 4096, 200, stream)
    assert got == [("cuGraphLaunch", 5, 9), ("cuEventRecord", 77, 9), ("cuStreamWaitEvent", 9, 77, 0),
                   ("cuMemcpyHtoDAsync_v2", 1 << 40, 4096, 200, 9)]
    assert lib.cuStreamWaitEvent.argtypes == [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    assert lib.cuMemcpyHtoDAsync_v2.argtypes[0] is ctypes.c_uint64 and lib.cuGraphLaunch.restype is ctypes.c_int
    with pytest.raises(RuntimeError, match="cuGraphLaunch failed: CUDA driver error 400"):
        PPH._Driver(cu(400)).launch(5, 9)


def test_the_schedule_changes_between_passes_and_the_ring_grows_poisoned(device_run):
    make, kernel, weights, *_ = device_run
    run = make()
    run.gemm(0, torch.randn(32, 1024).to(torch.bfloat16), weights[0], None)
    sh = run.shapes[(32, 256, 1024)]
    assert len(sh.sets) == 1 and run.sides == [] and run.details["schedule"]["name"] == "serial"
    run.end()
    run.begin(retain=None, schedule="deferred")
    run.poison()
    kernel.dev.log.clear()
    run.gemm(0, torch.randn(32, 1024).to(torch.bfloat16), weights[0], None)
    assert len(sh.sets) == 2 and all(bool((t == 0xA5).all()) for t in sh.sets[1].values())
    assert run.details["schedule"] == {"name": "deferred", "side": ["tile_hash"], "ring": 2, "main": -(1 << 10), "side_lane": 0}
    assert {st for n, st, _, _ in kernel.dev.log if n == "pd_tiles_851"} == {1} and run.end()["schedule"] == "deferred"


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


def test_the_quality_run_on_sm90a_runs_its_one_pearl_c_scheme(tmp_path, monkeypatch):
    """On pearl_c_sm90's kernels the quality run's Pearl-C group is ``pearl-c-h100-v1-h3`` alone, since sm_90a has no
    unpromoted record, so it finishes after two groups instead of asking for an sm_120 scheme on an H200."""
    E2E = load("pearl_c_vllm_e2e_sm90", KERNEL.parent / "pearl_c_vllm" / "e2e.py")
    monkeypatch.setattr(E2E, "wikitext_windows", lambda model, limit: [[1, 2, 3]])
    started = []

    class Fake:
        live: ClassVar[list] = []

        def __init__(self, ctx, args, group, coins):
            started.append((group, args.scheme))
            self.ready = {"gate": [{"ok": True}], "committer": "x"} if group == "arm" else {}

        def ask(self, **cmd):
            return {"nll": 1.0, "tokens": 2, "top1": [2, 3]} if cmd["op"] == "ppl" else {"tokens": [[1, 2]], "texts": ["t"]}

        def close(self):
            pass

    monkeypatch.setattr(E2E, "Worker", Fake)
    args = types.SimpleNamespace(out=tmp_path / "quality.json", model="m", ppl_windows=None, chunk_min=0.0,
                                 kernel=str(KERNEL.parent / "pearl_c_sm90"))
    assert [E2E.perplexity(args) for _ in range(2)] == [99, 0]
    assert started == [("base", None), ("arm", "pearl-c-h100-v1-h3")]
    doc = json.loads(args.out.read_text())
    assert set(doc["summary"]) == {"fp8", "bf16", "pearlc-v1"} and doc["summary"]["pearlc-v1"]["scheme"] == "pearl-c-h100-v1-h3"


def test_a_tag_names_another_model_in_the_shape_names():
    """``--tag`` puts another model's name in both phases' shape names, so a Llama 3.2 1B row is never read as the panel's
    Llama-3.1-8B one; without it the names are the panel's."""
    E2E = load("pearl_c_vllm_e2e_tag", KERNEL.parent / "pearl_c_vllm" / "e2e.py")
    E2E.configure(types.SimpleNamespace(batch=32, prompt=64, tag="llama32-1b"))
    assert E2E.SHAPES == {"prefill_8192": "e2e-llama32-1b-vllm-m8192", "decode_b32_step": "e2e-llama32-1b-vllm-m32"}
    E2E.configure(types.SimpleNamespace(batch=32, prompt=64))
    assert E2E.SHAPES == {"prefill_8192": "e2e-llama31-8b-vllm-m8192", "decode_b32_step": "e2e-llama31-8b-vllm-m32"}


def test_a_base_models_chat_prompts_are_their_text():
    """A base model (the demo's Llama 3.2 1B) has no chat template, so its chat prompts are their own text instead of the
    engine failing; an instruct model's are its template's turns, and any other refusal still raises."""
    E2E = load("pearl_c_vllm_e2e_chat", KERNEL.parent / "pearl_c_vllm" / "e2e.py")

    class Tokenizer:
        def __init__(self, error=None):
            self.error = error

        def apply_chat_template(self, turns, tokenize, add_generation_prompt):
            if self.error:
                raise ValueError(self.error)
            return f"<user>{turns[0]['content']}<assistant>"

    base = Tokenizer("Cannot use chat template functions because tokenizer.chat_template is not set")
    assert E2E.chat(base, ["a", "b"]) == ["a", "b"]
    assert E2E.chat(Tokenizer(), ["a"]) == ["<user>a<assistant>"]
    with pytest.raises(ValueError, match="other"):
        E2E.chat(Tokenizer("some other refusal"), ["a"])


# ---- one install path: the run behind the Stack ------------------------------------------------------------------------

def _stack_sites(weights):
    from verity_vllm.protocols.linear.api import LinearSite
    return [LinearSite(w, w, f"model.layers.{w}", t.shape[0], t.shape[1], "bf16") for w, t in weights.items()]


class _Layer:
    def __init__(self, weight):
        self.weight = weight


def test_the_run_behind_the_stack_commits_what_the_run_commits_alone(device_run, tmp_path, monkeypatch):
    """e2e's install path (`Stack` + `PearlCExecutor`, the arm from `arm_of`) retains the same manifest and files as the run
    driven directly, byte for byte."""
    import dataclasses

    from verity_vllm.protocols.linear.api import Stack
    zeroed_y(monkeypatch)
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
    assert stack.refusals() == [] and stack.executor.schedules == {"serial", "deferred", "deferred-caller", "deferred-a"}
    stack.activate(E2E.arm_of("pearlc", run, types.SimpleNamespace(schedule="serial")))
    with stack.pass_(dataclasses.replace(stack.active.spec, retain=str(tmp_path / "stack"), run_id=RUN)):
        for w, x in xs:
            stack.apply(sites[w], _Layer(weights[w]), x, None, lambda *a: pytest.fail("the stack called vLLM's own apply"))
    got = stack.passes[-1]
    assert {k: v for k, v in got.items() if k != "manifest"} == {k: v for k, v in want.items() if k != "manifest"}
    a, b = Path(want["manifest"]).parent, Path(got["manifest"]).parent
    assert (a / "manifest.json").read_text() == (b / "manifest.json").read_text()
    assert all((a / f).read_bytes() == (b / f).read_bytes()
               for f in ("leaves", "tree_t", "calls/0.rows.bf16", "weights/0.tree", "y/0.rows.bf16", "y/0.tree"))


def test_the_modes_are_arms_and_the_negative_control_is_a_pass_control(device_run):
    from verity_vllm.protocols.linear.api import Control, Stack
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
    assert not names & {"s512_tile_segs", "s512_row_segs", "s512_tile_leaves", "s512_rows", "pd_commit_851", "pd_tiles_851"}
    assert run.omit == frozenset()
    assert all(c <= set(PPD.CALL_STEPS) for c in E2E.CONTROLS.values())


# ---- the served step's breakdown ----------------------------------------------------------------------------------

def _served_step_profile(path: Path) -> Path:
    """An nsys SQLite export's tables as `served_step.analyze` reads them: a prefill step and three decode steps of 8
    requests, each with three Pearl-C calls (pd_commit, the GEMM, and the tile leaves on a stream of their own, each its
    own launch inside the call), an attention kernel launched between calls and a pass end whose transcript tree is
    ``s512_nodes``; 10 ms cycles, each step 9 ms."""
    import sqlite3
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
    con.execute("CREATE TABLE NVTX_EVENTS (start INT, end INT, text TEXT, globalTid INT, textId INT)")
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT, shortName INT, globalPid INT, streamId INT, "
                "correlationId INT)")
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INT, end INT, nameId INT, globalTid INT, correlationId INT)")
    names = {1: "pd_commit_851", 2: "gemm_pearlc_d", 3: "flash_fwd", 4: "cudaLaunchKernel", 5: "pd_tiles_851", 6: "s512_nodes"}
    con.executemany("INSERT INTO StringIds VALUES (?, ?)", names.items())
    tid, ms, corr = 7, 1_000_000, iter(range(1, 1000))

    def kernel(at, start, end, name, stream=7):
        k = next(corr)
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 4, ?, ?)", (at, at + 10_000, tid, k))
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, 0, ?, ?)", (start, end, name, stream, k))
    for i in range(4):
        s = i * 10 * ms
        con.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, 'step', ?, NULL)", (s, s + 9 * ms, tid))
        con.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, NULL)", (s + ms, s + 8 * ms, f"exec|{8 if i else 40}|8", tid))
        for j in range(3):
            c = s + ms + j * 2 * ms
            con.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, 'pc', ?, NULL)", (c, c + ms, tid))
            kernel(c, c + 100_000, c + 400_000, 1)
            kernel(c + 20_000, c + 400_000, c + 500_000, 2)
            kernel(c + 40_000, c + 450_000, c + 600_000, 5, 9)
        kernel(s + 6_500_000, s + 7 * ms, s + 7 * ms + 200_000, 3)
        con.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, 'pend', ?, NULL)", (s + 7 * ms, s + 8 * ms, tid))
        kernel(s + 7_100_000, s + 7_500_000, s + 7_600_000, 6)
    con.commit()
    return path


def test_the_served_step_profile_splits_each_decode_step_into_kernels_and_host_ranges(tmp_path):
    S = load("pearl_c_vllm_served_step", KERNEL.parent / "pearl_c_vllm" / "served_step.py")
    doc = S.analyze(_served_step_profile(tmp_path / "served.sqlite"), 8)
    assert (doc["batch"], doc["steps"], doc["decode_steps"], doc["full"]["n"], doc["other"]["n"]) == (8, 4, 3, 3, 1)
    f = doc["full"]
    got = {k: round(f[k], 6) for k in ("cycle_ms", "step_ms", "gpu_busy_ms", "kernel_sum_ms", "pc_ms", "pc_n", "pend_ms",
                                        "exec_ms", "launches", "api_ms", "streams", "pc_api_ms", "pc_launches",
                                        "pend_launches")}
    assert got == {"cycle_ms": 10.0, "step_ms": 9.0, "gpu_busy_ms": 1.8, "kernel_sum_ms": 1.95, "pc_ms": 3.0, "pc_n": 3,
                   "pend_ms": 1.0, "exec_ms": 7.0, "launches": 11, "api_ms": 0.11, "streams": 2, "pc_api_ms": 0.09,
                   "pc_launches": 9, "pend_launches": 1}
    assert {c: round(v, 6) for c, v in f["by_category_ms"].items()} == {
        "pearlc:a_commit": 0.9, "pearlc:gemm": 0.3, "pearlc:tile_hash": 0.45, "attention": 0.2, "pearlc:a_tree": 0.1}
    assert {k: {c: round(v, 6) for c, v in cs.items()} for k, cs in f["by_source_ms"].items()} == {
        "pc": {"pearlc:a_commit": 0.9, "pearlc:gemm": 0.3, "pearlc:tile_hash": 0.45}, "vllm": {"attention": 0.2},
        "pend": {"pearlc:a_tree": 0.1}}
    assert [(k["kernel"], k["per_step"]) for k in doc["top_kernels_full"][:2]] == [("pd_commit_851", 3), ("pd_tiles_851", 3)]


def test_the_served_steps_breakdown_puts_each_arms_kernels_and_host_time_in_its_rows(tmp_path):
    S = load("pearl_c_vllm_served_step", KERNEL.parent / "pearl_c_vllm" / "served_step.py")
    pearl, base = tmp_path / "pouw_pearl-c-sm120-v1-h3", tmp_path / "stock-fp8_compiled"
    for d in (pearl, base):
        d.mkdir()
    (pearl / "step-profile.json").write_text(json.dumps(S.analyze(_served_step_profile(tmp_path / "served.sqlite"), 8)))
    (pearl / "reps.json").write_text(json.dumps({"gate": {"passed": True}, "reps": [
        {"metrics": {"inter_token_latency_p50_s": 0.010}}, {"metrics": {"inter_token_latency_p50_s": 0.012}}]}))
    (base / "step-profile.json").write_text(json.dumps({"full": {"n": 3, "cycle_ms": 2.0, "gpu_busy_ms": 1.5,
                                                                 "by_category_ms": {"gemm": 1.0, "attention": 0.4}}}))
    (base / "reps.json").write_text(json.dumps({"gate": {"passed": True}, "reps": [
        {"metrics": {"inter_token_latency_p50_s": 0.002}}]}))
    doc, md = S.table([pearl, base], "stock-fp8_compiled")
    p = doc["arms"][pearl.name]
    assert {k: round(v, 6) for k, v in p["kernels"].items()} == {
        "stock forward (vLLM's kernels)": 0.2, "Pearl-C GEMM kernels": 0.3, "Pearl-C forming kernels": 0.0,
        "hashing and A's trees": 1.35, "pass end (transcript tree, coins)": 0.1}
    assert (round(p["host_gap_ms"], 6), round(p["host"]["Pearl-C calls' Python"], 6), p["gate"]) == (8.2, 2.91, True)
    assert p["harness_ms"] == {"inter_token_latency_p50_s": 11.0}
    assert doc["arms"][base.name]["kernels"]["stock forward (vLLM's kernels)"] == 1.4
    assert "| cycle / stock-fp8_compiled | 5.00 | 1.00 |" in md
    assert "| inter_token_latency_p50_s / stock-fp8_compiled | 5.50 | 1.00 |" in md


def test_an_arm_whose_profile_has_no_nvtx_ranges_is_listed_as_missing_not_tabled(tmp_path):
    import sqlite3
    S = load("pearl_c_vllm_served_step", KERNEL.parent / "pearl_c_vllm" / "served_step.py")
    db = tmp_path / "eager.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
    con.commit()
    bare = S.analyze(db, 8)
    assert bare["error"] == "no NVTX ranges in the profile" and "full" not in bare
    pearl, eager = tmp_path / "pouw_pearl-c-sm120-v1-h3", tmp_path / "stock-fp8_eager"
    for d in (pearl, eager):
        d.mkdir()
    (pearl / "step-profile.json").write_text(json.dumps(S.analyze(_served_step_profile(tmp_path / "served.sqlite"), 8)))
    (eager / "step-profile.json").write_text(json.dumps(bare))
    doc, md = S.table([pearl, eager], pearl.name)
    assert list(doc["arms"]) == [pearl.name]
    assert doc["missing"] == {eager.name: "no NVTX ranges in the profile"}
    assert f"{eager.name}: no NVTX ranges in the profile" in md


def test_the_served_steps_host_records_are_the_timed_reps_decode_steps():
    S = load("pearl_c_vllm_served_step", KERNEL.parent / "pearl_c_vllm" / "served_step.py")
    ms = 1_000_000
    steps = [{"phase": "measure", "rep": 0, "t0": 0, "t1": 9 * ms, "tokens": 8, "reqs": 8, "pc_ns": 3 * ms, "pc_n": 3},
             {"phase": "measure", "rep": 0, "t0": 0, "t1": 4 * ms, "tokens": 5, "reqs": 5, "pc_ns": ms, "pc_n": 3},
             {"phase": "measure", "rep": 0, "t0": 0, "t1": 20 * ms, "tokens": 40, "reqs": 8},
             {"phase": "warmup", "rep": 0, "t0": 0, "t1": 9 * ms, "tokens": 8, "reqs": 8}]
    got = S.summarize_host(steps, 8)
    assert got["full"] == {"n": 1, "step_ms": 9.0, "pc_ms": 3.0, "pc_n": 3}
    assert got["decode"]["n"] == 2 and got["decode"]["pc_ms"] == 2.0
    assert got["steps"] == {"measured": 3, "decode": 2, "total_ms": 33.0, "decode_ms": 13.0}
