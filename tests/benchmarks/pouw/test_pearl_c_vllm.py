"""The vLLM device path for Pearl-C on sm_120 (`verity.protocols.accounting.compute.pouw.prover.serving.pearl_c`) against GPU 1's pipeline
(``verity/protocols/accounting/compute/pouw/prover/kernels/pearl_c_sm120``) on the CPU, with a recording stand-in for the device; and the pass verifier
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
import re
import shlex
import shutil
import socket
import sys
import threading
import time
import types
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from verity.commitments import docket_digest as DD
from verity.commitments import domains as DM
from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c_u as PCU
from verity_torch.pouw.pearl_c import ends as ENDS
from verity_torch.pouw.pearl_c import host as PH
from verity_torch.pouw.pearl_c import passes as PP
from verity_torch.pouw.pearl_c import served_y_device as SYD
from verity.protocols.accounting.compute.pouw.prover.serving.pearl_c import served_y as SY
from verity.protocols.accounting.compute.pouw.prover.serving.pearl_c import window as PWR

from verity.protocols.accounting.compute.pouw.verifier import core_check as CC
from verity.commitments import MerkleTree
from verity.commitments.merkle import tree_levels
from verity.protocols.accounting.compute.pouw.verifier.audit import (
    HM96,
    HM96_SALT,
    HM96_TRANSCRIPT,
    HM96_WEIGHTS,
    ROW_SCHEMA,
    TILE_SCHEMA,
    Epoch,
    Workload,
    activation_binding,
    commit_rows,
    fresh_salts,
    hm96_kernel_constants,
    hm96_salt,
    opened_value,
    rows_domain,
    salt_of,
    transcript_domain,
    weight_binding,
    weights_root,
)
from verity.protocols.accounting.compute.pouw.verifier import salt as PS
from verity.protocols.accounting.compute.pouw.verifier import window as Wd
from verity.protocols.accounting.compute.pouw.verifier.coins import Challenger, CoinCommitment
from verity.protocols.accounting.compute.pouw.verifier.credit import passes_distinct
from verity.protocols.accounting.compute.pouw.verifier.protocol import Context, DomainError, Matmul, Shape, call_place
from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c as C
from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c_work as PW
from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_kw as P
from verity.protocols.accounting.compute.pouw.verifier.serving import SchemeId
from verity.protocols.proof.spot.verifier.challenger import challenger as Ch
from verity.protocols.accounting.compute.pouw.prover.serving import pearl_c as PPD
from verity.protocols.accounting.compute.pouw.prover import schemes as pouw_schemes
from verity.protocols.accounting.compute.pouw.verifier.kernel_schemes import names as kernel_names
from verity_torch.pouw.pearl_c import run_device as PPDR

torch = pytest.importorskip("torch")

# the work sampling design and the clear audit are a reference tool's (verity/protocols/accounting/compute/pouw/prover/clear), imported by name
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity" / "protocols" / "accounting" / "compute" / "pouw" / "prover" / "clear"))
import clear_work as CW  # noqa: E402
from clear_audit import Prover, Verifier  # noqa: E402

KERNEL = Path(__file__).resolve().parents[3] / "verity" / "protocols" / "accounting" / "compute" / "pouw" / "prover" / "kernels" / "pearl_c_sm120"
POUW = (Path(__file__).resolve().parents[3] / "benchmarks" / "pouw")
SCHEME = "pearl-c-sm120-v1-h3"
SEED = bytes(range(32))


def salts_b(weights) -> dict[int, bytes]:
    """Each weight's B-row salts as a registration would hand them, seeded so two runs over the same weights share them."""
    return {w: hashlib.shake_256(b"salts_b %d" % w).digest(HM96_SALT * t.shape[0]) for w, t in weights.items()}
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
F = load("pearl_c_sm120_fixture", POUW / "pearl_c_sm120" / "fixture.py")
VR = load("pearl_c_vllm_verify_run", POUW / "pearl_c_vllm" / "verify_run.py")
SD = load("pearl_c_vllm_served_inert_work", POUW / "pearl_c_vllm" / "served_inert_work.py")
#: the device path's schemes whose device record is a variant of KERNEL's pipeline (pearl_c_sm90 runs h100's)
KERNEL_SCHEMES = sorted(s for s in kernel_names("pearl-c") if SchemeId.parse(s).device in R.VARIANT_G)


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
    fails, unsure = PPDR.screen(x, alpha, aux)
    got = PPDR.excluded(fails, unsure, x)
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
    fails, unsure = PPDR.screen(x, *stats(words))
    assert bool(unsure[0]) and not bool(fails[0])
    assert PPDR.excluded(fails, unsure, x) == (() if CW.row_passes([int(v) for v in words[0]]) else (0,))


# ---- the per-call keys --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", KERNEL_SCHEMES)
def test_call_keys_are_the_fixtures_keys_of(scheme):
    s = pouw_schemes.get(scheme)
    roots = {0: bytes([1]) * 32, 1: bytes([2]) * 32}
    work = Workload((Matmul(0, Shape(16, 1024, 256), 0), Matmul(1, Shape(16, 1024, 128), 1)))
    epoch = Epoch.unverified(work, roots, SEED)
    for index, w, m16 in ((0, 0, 32), (5, 1, 8192), (127, 0, 16)):
        mm = Matmul(index, Shape(m16, 1024, 256 if w == 0 else 128), w)
        kk = F.keys_of(s, epoch, mm, C._labelled(epoch.salt + roots[w], "seed-B"))
        assert PPDR.call_keys(epoch.salt, roots[w], index, m16, s.commitment_hash) == kk["dom_a"] + kk["unit"]


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
    kernel = PPD.Kernel(R, RecordingDev(), "0" * 64)
    roots = PPDR.weight_roots(kernel, weights, SCHEME, weight_salts=salts_b(weights))
    declared = Workload(tuple(Matmul(w, Shape(16, 1024, t.shape[0]), w) for w, t in weights.items()))
    coins = salt_coin(roots)
    salt = Epoch.start(declared, roots, *coins).salt

    def make(hashing=True, schedule="serial", flag_rows=PPD.FLAG_ROWS, **kw):
        return PPDR.PearlCDeviceRun(pouw_schemes.get(SCHEME), f"pouw:{SCHEME}", kernel, weights, roots, *coins, hashing=hashing,
                                   capacity=1 << 12, schedule=schedule, flag_rows=flag_rows,
                                   weight_salts=salts_b(weights), **kw)
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
        assert PPDR._bytes(sh.keys[:168]) == PPDR.call_keys(salt, roots[w], index, 32, run.tree_hash)
        assert bool((sh.xa[m:] == 0).all()) and torch.equal(sh.xa[:m], x.float())
    assert [u.index for u in run.cur.matmuls] == [0, 1, 2] and run.cur.tiles == 4 + 2 + 4
    dev.log.clear()
    out = run.end()
    assert [n for n, *_ in dev.log][-4:] == ["s512_pads", "hm_leaves", "s512_frame_leaves", "s512_tree_small"]
    assert dev.log[-2][3] == (0, 10)
    assert out["units"] == 3 and out["tiles"] == 10 and len(out["transcript_root"]) == 128
    assert run.commits["transcript_root"] == out["transcript_root"]


def test_the_transcripts_leaves_are_hashed_under_its_salts_with_their_length_stated(device_run, monkeypatch):
    """A pass's end hands `hm_level0` the transcript's salts buffer and its length, `HM_SALT` bytes a tile, so the kernels'
    own refusal of a wrong length guards it, and its hm_leaves launch reads that buffer."""
    make, kernel, weights, *_ = device_run
    run = make()
    run.gemm(0, torch.randn(20, 1024).to(torch.bfloat16), weights[0], None)
    given, level0 = [], R.Pipeline.hm_level0

    def spy(p, tag, *a, **kw):
        given.append((tag, kw))
        return level0(p, tag, *a, **kw)
    monkeypatch.setattr(R.Pipeline, "hm_level0", staticmethod(spy))
    kernel.dev.log.clear()
    tiles = run.end()["tiles"]
    (tag, kw), = given
    assert tag == "t" and kw["salt_bytes"] == R.HM_SALT * tiles
    assert [p[2] for n, _, p, _ in kernel.dev.log if n == "hm_leaves"] == [kw["salts"]]


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


def test_a_calls_index_is_its_round_and_its_position_and_no_caller_chooses_one(device_run):
    """A call's index is its place (`passes.call_index`): its pass's round among the epoch's passes with calls times
    `ROUND_CALLS`, plus its position in the pass.  Three calls, then two in the next pass, take round 1's positions 0 and 1;
    a pass without a call takes no round.  A call after an ended pass is refused by name (`NO_OPEN_PASS`) before it
    takes a round or a base, until a `begin`; `finish` ends a pass with calls and leaves a fresh open one.
    A caller's own index is refused by name, at `begin` and at `repeat_at`, and the keys a call uploads carry the derived
    index."""
    make, kernel, weights, roots, salt = device_run
    run = make()
    x = torch.randn(16, 1024).to(torch.bfloat16)
    for w in (0, 1, 0):
        run.gemm(w, x, weights[w], None)
    assert [u.index for u in run.cur.matmuls] == [0, 1, 2]
    first = run.end_pass()
    assert run.cur.base is None and run.rounds() == 1
    for w in (1, 0):
        run.gemm(w, x, weights[w], None)
    assert first["base"] == 0 and [u.index for u in run.cur.matmuls] == [PP.ROUND_CALLS, PP.ROUND_CALLS + 1]
    sh = run._shape(16, 256, 1024)
    assert PPDR._bytes(sh.keys[:168]) == PPDR.call_keys(salt, roots[0], PP.ROUND_CALLS + 1, 16, run.tree_hash)
    second = run.end_pass()
    assert second["base"] == PP.ROUND_CALLS and run.rounds() == 2
    run.end_pass()
    run.begin()
    assert run.rounds() == 2 and "base" not in run.end()
    for at in (0, 2 * PP.ROUND_CALLS, PP.ROUND_CALLS + 1):
        with pytest.raises(ValueError, match="which the device derives and no caller chooses"):
            run.begin(at=at)
        with pytest.raises(ValueError, match="repeat_at.*no caller chooses"):
            run.repeat_at = at
    assert run.repeat_at is None and run.rounds() == 2 and not hasattr(run, "epoch_calls") and not hasattr(run, "pass_base")
    with pytest.raises(PH.SaltsRefused, match="NO_OPEN_PASS") as refused:
        run.gemm(1, x, weights[1], None)
    assert refused.value.name == "NO_OPEN_PASS" and run.rounds() == 2 and run.cur.base is None and not run.cur.matmuls
    run.begin()
    run.gemm(1, x, weights[1], None)
    run.finish()
    assert run.cur.base is None and run.rounds() == 3
    run.begin()
    run.gemm(0, x, weights[0], None)
    assert run.cur.base == 3 * PP.ROUND_CALLS and [u.index for u in run.cur.matmuls] == [3 * PP.ROUND_CALLS]


def test_twins_give_a_pass_the_same_indices_salt_and_keys(device_run):
    """What a check that runs a pass twice does in place of a chosen index: each on a twin of the run
    (`PearlCDeviceRun.twin`), a device run of its own over the same install at the run's hashing, whose rounds number from
    0 whatever the run's have reached, so the two passes' calls have the same places, salt and keys."""
    make, _, weights, *_ = device_run
    x = torch.randn(16, 1024).to(torch.bfloat16)
    run = make(hashing=False)
    for w in (0, 1):
        run.gemm(w, x, weights[w], None)
    run.end_pass()
    got = []
    for _ in range(2):
        twin = run.twin()
        for w in (1, 0):
            twin.gemm(w, x, weights[w], None)
        got.append(([u.index for u in twin.cur.matmuls], twin.salt, twin.hashing, PPDR._bytes(twin._shape(16, 256, 1024).keys[:168])))
        twin.finish()
    assert got[0] == got[1] and got[0][:3] == ([0, 1], run.salt, False) and run.rounds() == 1


def test_the_epochs_last_round_is_its_last_and_a_rounds_calls_stay_inside_it(device_run):
    """An epoch holds `EPOCH_ROUNDS` rounds, so every index is below `protocol.CALL_RANGE` (2^32, the row seed's range):
    the last round's first call has index 2^32 - `ROUND_CALLS`, and a call of the round after raises before it runs,
    naming the fix: the execution phase ends and the install takes a new salt.  A round's position never reaches
    `ROUND_CALLS`."""
    make, kernel, weights, *_ = device_run
    assert PP.ROUND_CALLS * PP.EPOCH_ROUNDS == C.CALL_RANGE and PP.ROUND_CALLS == PPD.CALL_CAPACITY
    run = make()
    x = torch.randn(16, 1024).to(torch.bfloat16)
    run._rounds = PP.EPOCH_ROUNDS - 1
    run.gemm(0, x, weights[0], None)
    assert [u.index for u in run.cur.matmuls] == [C.CALL_RANGE - PP.ROUND_CALLS]
    run.end_pass()
    kernel.dev.log.clear()
    with pytest.raises(RuntimeError, match="rounds.*2\\^32 calls.*reinstall under a new salt"):
        run.gemm(1, x, weights[1], None)
    assert not kernel.dev.log
    with pytest.raises(RuntimeError, match="would pass its 65536 calls"):
        PP.call_index(make(), PP.ROUND_CALLS + 1)


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
        root, want = SY.commit_y(y, u["n"], salt, u["index"], split_salts((tmp_path / u["y"]["salts"]["path"]).read_bytes()))
        assert len(y) == 2 * u["real_rows"] * u["n"] and tree == want and u["y"]["root"] == root.hex() == tree[-64:].hex()


def test_a_passs_salts_are_its_keys_stream_in_fill_order_and_its_end_drops_the_key(device_run, tmp_path, monkeypatch):
    """A's salts and the tiles' (`host.fresh_into`) are the leaves of the pass's own keyed stream from leaf 0, in the order
    they were filled, under a key `derive` made from fresh bytes at its first fill; its end drops the key, so no object
    holds it after, a fill then is refused by name, and the next pass draws another."""
    import gc
    make, _, weights, _, _ = device_run
    keys, derive = [], PH.derive
    monkeypatch.setattr(PH, "derive", lambda *a: keys.append(derive(*a)) or keys[-1])
    run = make()
    run.begin(retain=str(tmp_path / "first"))
    ended = run.cur
    assert ended.salts.key is None and not keys, "a pass draws its key at its first fill"
    run.gemm(0, torch.randn(20, 1024).to(torch.bfloat16), weights[0], None)
    key = ended.salts.key
    run.gemm(1, torch.randn(32, 1024).to(torch.bfloat16), weights[1], None)
    out = run.end()
    man = json.loads(Path(out["manifest"]).read_text())
    files = [u["salts"]["path"] for u in man["matmuls"]] + [man["salts_t"]["path"]]
    got = b"".join((tmp_path / "first" / f).read_bytes() for f in files)
    assert len(got) == HM96_SALT * (32 + 32 + 6) and len(keys) == 1
    assert got == b"".join([key.block(PH.SALT_INDEX, c) for c in range(3 * (32 + 32 + 6))])  # inlined: no cell holds key
    del out
    gc.collect()
    holders = [r for r in gc.get_referrers(key) if r is not keys and not isinstance(r, types.FrameType)]
    assert ended.ended and ended.salts.key is None and not holders, "an ended pass's key is held by no object"
    with pytest.raises(PH.SaltsRefused, match="NO_OPEN_PASS") as refused:
        run.gemm(0, torch.randn(16, 1024).to(torch.bfloat16), weights[0], None)
    assert refused.value.name == "NO_OPEN_PASS"
    run.begin()
    run.gemm(0, torch.randn(16, 1024).to(torch.bfloat16), weights[0], None)
    assert len(keys) == 2 and run.cur.salts.key is keys[1] and keys[1].value != key.value and run.cur.salts.next == 16


def test_a_retained_pass_refuses_a_weight_tree_whose_root_moved_after_the_build(device_run, tmp_path):
    """A retained pass writes each weight's tree from the device beside the build's root: a tree_b whose root bytes were
    written after the build is refused by the weight's name before any of the pass's last files is written, and an
    untouched one is retained."""
    make, _, weights, roots, _ = device_run
    for touched in (True, False):
        run = make()
        d = tmp_path / str(touched)
        run.begin(retain=str(d))
        run.gemm(0, torch.randn(20, 1024).to(torch.bfloat16), weights[0], None)
        if touched:
            run.weights[1].b["tree_b"][-64:] ^= 0x5A
            with pytest.raises(ValueError, match=r"weight 1's tree on the device roots at [0-9a-f]{16}, not its build's "
                                                  f"{roots[1].hex()[:16]}: its tree_b was written after the build"):
                run.end()
            assert not (d / "weights").exists() and not (d / "leaves").exists()
        else:
            man = json.loads(Path(run.end()["manifest"]).read_text())
            assert (d / man["weights"]["1"]["tree"]["path"]).read_bytes()[-64:] == roots[1]


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_the_weight_build_refuses_a_weight_that_is_not_bfloat16(device_run, dtype):
    """Pearl+ commits a weight's FP32 words and retains its rows as BF16, so a weight in another format (whose words BF16
    does not hold) is refused by name and dtype, by `weight_roots` and by a run's own build, before either allocates."""
    make, kernel, weights, roots, _ = device_run
    bad = {0: weights[0].to(getattr(torch, dtype)), 1: weights[1]}
    refusal = f"weight 0 is torch.{dtype}: Pearl\\+ commits a weight's FP32 words and retains its rows as BF16"
    with pytest.raises(ValueError, match=refusal):
        PPDR.weight_roots(kernel, bad, SCHEME, weight_salts=salts_b(bad))
    with pytest.raises(ValueError, match=refusal):
        PPDR.PearlCDeviceRun(pouw_schemes.get(SCHEME), f"pouw:{SCHEME}", kernel, bad, roots, *salt_coin(roots),
                             capacity=1 << 12, weight_salts=salts_b(bad))


def test_a_retained_call_given_its_y_tree_retains_that_tree_and_never_makes_one(device_run, tmp_path, monkeypatch):
    """A caller that commits y with its own launches (the overhead arm) hands `_retained` its tree (``y_tree``): the pass
    retains those bytes, its y root their last node, and makes no tree of its own; so a tree its launches wrote nothing
    into is retained as the poison it held, and the verifier sees what the launches left."""
    make, _, weights, _, salt = device_run
    run = make()
    run.begin(retain=str(tmp_path))
    retained, given = run._retained, {}

    def with_tree(cur, mm, m, flags, xa, tree, salts, y=None, y_tree=None, y_salts=None):
        ys = fresh_salts(y.shape[0])
        honest = SY.commit_y(PPDR._bytes(y.contiguous().view(torch.uint8)), y.shape[1], salt, mm.index, ys)[1]
        given[mm.index] = bytes([0xA5]) * len(honest) if given else honest
        return retained(cur, mm, m, flags, xa, tree, salts, y=y,
                        y_tree=torch.frombuffer(bytearray(given[mm.index]), dtype=torch.uint8), y_salts=torch.frombuffer(bytearray(b"".join(ys)), dtype=torch.uint8))
    monkeypatch.setattr(run, "_retained", with_tree)
    monkeypatch.setattr(SYD, "y_tree", lambda *_: pytest.fail("a given y tree is never made again"))
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
    assert u["y"]["root"] == ytree[-64:].hex() == SY.commit_y(served, 256, run.salt, index, split_salts(
        (tmp_path / "ok" / u["y"]["salts"]["path"]).read_bytes()))[0].hex()

    def fails(path, *_):
        raise OSError(f"no space left for {path.name}")
    monkeypatch.setattr(PH, "_write_host", fails)
    run.begin(retain=str(tmp_path / "full"))
    run.gemm(0, x, weights[0], None)
    with pytest.raises(OSError, match="no space left"):
        run.end()
    assert not (tmp_path / "full" / "manifest.json").exists() and not run.cur.writes


def test_a_retained_pass_starts_from_poisoned_outputs_and_omit_drops_a_step(device_run, tmp_path):
    """The call runs as vLLM runs it, under inference mode, and its shape's buffers are still normal tensors; the retained
    pass begins outside it (execution phase 's verify pass failed there)."""
    make, kernel, weights, *_ = device_run
    run = make()
    with torch.inference_mode():
        run.gemm(0, torch.randn(32, 1024).to(torch.bfloat16), weights[0], None)
    assert not any(t.is_inference() for sh in run.shapes.values() for t in sh.b.values())
    run.end()
    run.begin(retain=str(tmp_path))
    assert bool((run.leaves == 0xA5).all()) and all(bool((t == 0xA5).all()) for sh in run.shapes.values() for t in sh.b.values())
    run.omit = frozenset({"hash_leaf"})
    kernel.dev.log.clear()
    run.gemm(0, torch.randn(32, 1024).to(torch.bfloat16), weights[0], None)
    assert "pd_tiles_851" not in [n for n, *_ in kernel.dev.log] and "gemm_pearlc_d" in [n for n, *_ in kernel.dev.log]


def test_the_e2e_gate_fails_a_device_that_writes_nothing(device_run):
    """On the recording stand-in nothing is written, so every pass's tile digests are the poison: the repeat holds, and
    the fresh salts' differing roots must tell, for every shape."""
    make, _, weights, *_ = device_run
    E2E = load("pearl_c_vllm_e2e", POUW / "pearl_c_vllm" / "e2e.py")
    rows = E2E.gate(make(), weights)
    calls = [r for r in rows if r["check"] == "call"]
    assert [(r["n"], r["m"]) for r in calls] == [(256, 8192), (256, 32), (128, 8192), (128, 32)]
    assert all(r["digests_repeat"] and not r["salts_fresh"] and not r["ok"] for r in calls)
    (check,) = [r for r in rows if r["check"] == "screen"]
    assert check["ok"] and check["flags_equal"] and check["screen"] == "torch"


def test_the_call_check_compares_under_the_salt():
    """`call_verdict`: with every leaf's salt fresh, the same call twice passes on equal tile digests under different
    roots; equal roots (salts not fresh) and differing digests (a call that isn't deterministic) each fail; y beyond `ERR` fails; a call without hashing checks
    y alone."""
    E2E = load("pearl_c_vllm_e2e", POUW / "pearl_c_vllm" / "e2e.py")
    a, b = ({"transcript_root": r, "digests": d} for r, d in ((b"r1", b"d"), (b"r2", b"d")))
    assert E2E.call_verdict([a, b], 0.05)["ok"]
    assert not E2E.call_verdict([a, b], 0.09)["ok"]
    got = E2E.call_verdict([a, {**b, "transcript_root": b"r1"}], 0.05)
    assert not got["ok"] and not got["salts_fresh"] and got["digests_repeat"]
    got = E2E.call_verdict([a, {**b, "digests": b"e"}], 0.05)
    assert not got["ok"] and not got["digests_repeat"]
    assert E2E.call_verdict([{"units": 1}], 0.05)["ok"] and not E2E.call_verdict([{"units": 1}], 0.09)["ok"]


def test_the_whole_check_compares_a_pass_without_hashing():
    """`unsalted` of a pass ended without hashing, which has no transcript root, no tiles and no leaves (the gate's second
    pass, `--whole-step`): its units and tiles under the digest of nothing, with `run.leaves` unread."""
    import hashlib
    from types import SimpleNamespace
    E2E = load("pearl_c_vllm_e2e", POUW / "pearl_c_vllm" / "e2e.py")
    ended = {"units": 3, "base": 0, "tiles": 0, "hashing": False, "schedule": "serial"}
    assert E2E.unsalted(SimpleNamespace(nb=64), ended) == {"digests": hashlib.sha512(b"").hexdigest(), "units": 3, "tiles": 0}
    with pytest.raises(KeyError, match="transcript_root"):
        E2E.under_salt(SimpleNamespace(nb=64), ended)


def test_the_whole_retained_check_names_the_first_call_and_field_that_differ(tmp_path):
    """`retained_differences`: two retained passes the same but for one call's A rows names that call, its matmul and
    its entry's differing field, and the one file on disk; two passes the same are ok."""
    from verity.protocols.accounting.compute.pouw.verifier.protocol import Matmul, Shape
    E2E = load("pearl_c_vllm_e2e", POUW / "pearl_c_vllm" / "e2e.py")
    mms = [Matmul(i, Shape(16, 2048, 3072), 7 + i) for i in range(3)]
    files = [{"rows": {"path": f"calls/{i}.rows.bf16", "bytes": 4, "sha256": "a"}} for i in range(3)]
    moved = [*files[:1], {**files[1], "rows": {**files[1]["rows"], "sha256": "b"}}, *files[2:]]
    dirs = [tmp_path / "eager", tmp_path / "replayed"]
    for d, rows in zip(dirs, (b"x", b"y"), strict=True):
        (d / "calls").mkdir(parents=True)
        (d / "calls" / "1.rows.bf16").write_bytes(rows)
    want, got = ([None] * 3, files, {"tiles": 3}, mms), ([None] * 3, moved, {"tiles": 3}, mms)
    assert E2E.retained_differences(want, got, dirs) == {
        "ok": False, "disk": {"differ": 1, "first": ["calls/1.rows.bf16"]},
        "call_files": {"calls": [3, 3], "first": 1, "matmul": {"index": 1, "shape": [16, 2048, 3072], "weight": 8},
                       "fields": ["rows.sha256"]}}
    assert E2E.retained_differences(want, want, [dirs[0], dirs[0]]) == {"ok": True}


def test_the_whole_retained_check_sets_aside_what_fresh_salts_make_differ(tmp_path):
    """Each pass draws its own salts, so two retained passes of the same calls differ in A's tree and salts and y's root,
    tree and salts, in their entries and on disk: `retained_differences` passes them, and still names a y rows file that
    differs."""
    from verity.protocols.accounting.compute.pouw.verifier.protocol import Matmul, Shape
    E2E = load("pearl_c_vllm_e2e", POUW / "pearl_c_vllm" / "e2e.py")
    mms = [Matmul(0, Shape(16, 2048, 3072), 7)]

    def call(tag: str, y_rows: str = "r") -> dict:
        return {"rows": {"path": "calls/0.rows.bf16", "bytes": 4},
                "tree": {"path": "calls/0.tree", "bytes": 8, "sha256": tag}, "salts": {"path": "calls/0.salts", "bytes": 4},
                "y": {"root": tag, "rows": {"path": "y/0.rows.bf16", "bytes": 4, "sha256": y_rows},
                      "tree": {"path": "y/0.tree", "bytes": 8, "sha256": tag}, "salts": {"path": "y/0.salts", "bytes": 4}}}
    dirs = [tmp_path / "eager", tmp_path / "replayed"]
    for d, salt in zip(dirs, (b"s", b"t"), strict=True):
        for sub in ("calls", "y"):
            (d / sub).mkdir(parents=True)
            (d / sub / "0.rows.bf16").write_bytes(b"rows")
            for name in ("0.tree", "0.salts"):
                (d / sub / name).write_bytes(salt)
    want = ([None], [call("a")], {"digests": "d", "units": 1, "tiles": 1}, mms)
    assert E2E.retained_differences(want, ([None], [call("b")], want[2], mms), dirs) == {"ok": True}
    (dirs[1] / "y" / "0.rows.bf16").write_bytes(b"rowz")
    got = E2E.retained_differences(want, ([None], [call("b", "q")], want[2], mms), dirs)
    assert got["disk"] == {"differ": 1, "first": ["y/0.rows.bf16"]} and got["call_files"]["fields"] == ["y.rows.sha256"]
    assert not E2E.retained_differences(want, ([None], [call("a")], {**want[2], "digests": "e"}, mms), dirs)["ok"]


def test_a_retained_pass_is_read_after_its_end_fills_its_records_in():
    """On the device a call's record is filled in behind it, by the time `end` joins the pass's writes: the whole check
    reads the pass's records after its end, without its manifest."""
    E2E = load("pearl_c_vllm_e2e", POUW / "pearl_c_vllm" / "e2e.py")
    cur = types.SimpleNamespace(records=["first", None], files=[{}, {}], matmuls=[0, 1], retain="dir")

    def end():
        assert cur.retain is None
        cur.records[1] = "behind"
        return {"tiles": 2, "manifest": "m"}

    run = types.SimpleNamespace(cur=cur, end=end)
    assert E2E.retained_end(run, lambda _, e: {"tiles": e["tiles"]}) == (["first", "behind"], [{}, {}], {"tiles": 2}, [0, 1])


# ---- one pass per engine step, through Pearl+ attached by name (`pouw.scheme.Attached`) ---------------------------------

def seeded_device_salts(monkeypatch, seed=11):
    """Every salt the device run draws (A's and the tiles', `host.fresh_into`, wherever imported; y's) from streams seeded
    with `seed`, so two runs making the same calls draw the same salts and commit the same bytes."""
    import sys

    rng, ys = random.Random(f"{seed} fresh_into"), random.Random(f"{seed} y")

    def fresh_into(dst, n, raw=None, *, salts):
        dst[:HM96_SALT * n].copy_(torch.frombuffer(bytearray(rng.randbytes(HM96_SALT * n)), dtype=torch.uint8))
    for name in ("ends", "graphs", "grouped", "host", "lanes_device", "run_device", "whole_calls"):
        mod = sys.modules.get(f"verity_torch.pouw.pearl_c.{name}")
        if hasattr(mod, "fresh_into"):
            monkeypatch.setattr(mod, "fresh_into", fresh_into)
    monkeypatch.setattr(SYD, "fresh_salts", lambda m: [ys.randbytes(HM96_SALT) for _ in range(m)])


class _Steps:
    """Pearl+ attached by name (`pouw.scheme.Attached`) over `weights` on `kernel`, as an engine's exchange drives it: the
    weights registered, then each `step` its calls (weight, rows) as the step's blocks (`BlockDue`, through
    `Attached.kernel`) between its `StepBegan` (its slots the step's most rows) and its `StepComputed` and `StepEnded`, on
    rows sampled from `seed`.  Its coins are in-process from a seeded challenger, the weights' B salts `salts_b`, every other
    fresh salt `seeded_device_salts`'; ``more`` are further Pearl+ settings and ``run`` overrides the device run's buffer
    sizes."""

    def __init__(self, monkeypatch, kernel, weights, steps, more=None, seed=0, **run):
        import functools

        from verity import hooks as H
        from verity.protocols.accounting.compute.pouw import scheme as PP
        from verity.protocols.accounting.compute.pouw.prover import serving as SV
        from verity.protocols.accounting.compute.pouw.verifier import verifier

        monkeypatch.setattr(PPD.Kernel, "load", staticmethod(lambda *a: kernel))
        seeded_device_salts(monkeypatch)
        by_rows = {weights[w].shape[0]: b for w, b in salts_b(weights).items()}
        monkeypatch.setattr(SV, "fresh_salts", lambda m: [by_rows[m][i * HM96_SALT:(i + 1) * HM96_SALT] for i in range(m)],
                            raising=False)
        monkeypatch.setattr(verifier, "coins", random.Random(7).randbytes)
        monkeypatch.setattr(PPDR, "PearlCDeviceRun", functools.partial(PPDR.PearlCDeviceRun, **run))
        self.H, self.weights, self.steps, self.rng, self.at = H, weights, steps, np.random.default_rng(seed), 0
        self.scheme = PP.Attached({"settings": {"kernel": "k", "coins": "in-process", **(more or {})}})
        self.scheme.registered(H.RegisteredValues(tuple((f"l{w}.weight", t) for w, t in weights.items()), 0,
                                                  {f"l{w}.weight": w for w in weights}))

    def begin(self):
        calls = self.steps[self.at]
        self.scheme.step(self.H.StepBegan(self.at, slots=max((m for _, m in calls), default=0)))
        return calls

    def step(self):
        for w, m in self.begin():
            weight = self.weights[w]
            x = torch.from_numpy(bf16_words(self.rng, m, weight.shape[1]).view(np.float32).copy()).to(torch.bfloat16)
            due = self.H.BlockDue(f"l{w}", w, {"m": m}, "bf16", {"x": x, "weight": weight}, self.at)
            self.scheme.kernel(due, lambda _: None)
        self.scheme.step(self.H.StepComputed(self.at))
        self.scheme.step(self.H.StepEnded(self.at))
        self.at += 1

    def release(self):
        """The scheme released, as at serving's end; the run's details."""
        self.scheme.release()
        return self.scheme.life["summary"]


@pytest.mark.parametrize("dev", ["recording", pytest.param("twin", marks=[
    pytest.mark.slow, pytest.mark.skipif(not (shutil.which("g++") or shutil.which("c++")), reason="simt_twin needs a C++ compiler")])])
def test_each_served_step_is_a_pass_retained_on_its_own(monkeypatch, tmp_path, dev):
    """Pearl+ attached by name under ``transcript_dir``, three engine steps: each pass is one
    step's forward over every weight, retained in ``pass-NNNNN`` in the order the passes began (the one the install began
    first among them), its manifest's commitment the one it published and its calls the step's, numbered on from the last
    pass's (the execution phase's call indices, so the passes together hold each index once), its weights' files hard links to the
    first pass's; the pass left open and empty after the last step writes nothing.  On ``twin`` the hash kernels run, so
    A's roots are real and differ pass to pass."""
    rng = np.random.default_rng(23)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    steps = [[(0, 200), (1, 200)], [(0, 120), (1, 120)], [(0, 3), (1, 3)]]
    kernel = PPD.Kernel(R90, H100TwinDev(HD.build(tmp_path)) if dev == "twin" else H100Dev(), "0" * 64)
    root = tmp_path / "passes"
    served = _Steps(monkeypatch, kernel, weights, steps, {"transcript_dir": str(root)}, capacity=32, flag_rows=512)
    for _ in steps:
        served.step()
    passes = served.release()["passes"]
    assert [(p["units"], p["tiles"], p["screened_rows"]) for p in passes] == [(2, 24, 400), (2, 12, 240), (2, 6, 6)]
    assert [Path(p["manifest"]) for p in passes] == [root / f"pass-{i:05d}" / "manifest.json" for i in range(3)]
    assert sorted(d.name for d in root.iterdir() if d.is_dir()) == [f"pass-{i:05d}" for i in range(3)]
    mans = []
    for i, (p, step) in enumerate(zip(passes, steps)):
        man = json.loads(Path(p["manifest"]).read_text())
        mans.append(man)
        assert p["base"] == PP.ROUND_CALLS * i
        assert [(u["index"], u["weight"], u["real_rows"]) for u in man["matmuls"]] == [
            (PP.ROUND_CALLS * i + j, w, m) for j, (w, m) in enumerate(step)]
        assert man["commitment"] == {"root": p["transcript_root"], "count": p["tiles"]} and sorted(man["weights"]) == ["0", "1"]
    assert passes_distinct(mans) is None
    for f in ("weights/0.rows.bf16", "weights/0.tree", "weights/1.rows.bf16", "weights/1.tree"):
        stats = [(root / f"pass-{i:05d}" / f).stat() for i in range(3)]
        assert len({s.st_ino for s in stats}) == 1 and stats[0].st_nlink == 3, f
    if dev == "twin":
        assert len({p["activation_roots"] for p in passes}) == 3


#: `_retained_files`' digest of the three steps below as the retention path wrote them when each call's copies, its record,
#: y's root and the pass's files and manifest were all waited for in its step, under Pearl+'s kernel scheme
#: (``pearl-c-h100-v1-h3``), every fresh salt `seeded_device_salts`'
RETAINED_DIGEST = "16837623d625907f37e5ab18ca09160501095583d8f5dfc596a6a6302cd2c737"


def _retained_files(root: Path) -> dict[str, str]:
    """Every file of `root`'s passes (``pass-NNNNN``) and its bytes' SHA-256."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("pass-*/**/*"))
            if p.is_file()}


def test_a_step_completed_behind_it_retains_the_bytes_a_waiting_end_retains(monkeypatch, tmp_path):
    """`PassEnds.retain_behind`: three served steps, each pass's files, records and manifest completed behind its step
    (`retained_ends`, which the run's finish waits for), retain every byte that ends completing them in the step do, the
    same files as the path that waited for each call's copies (`RETAINED_DIGEST`); each file is the owner's alone (0600)
    and each directory of a pass too (0700)."""
    zeroed_y(monkeypatch)
    rng = np.random.default_rng(23)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    steps = [[(0, 200), (1, 200)], [(0, 120), (1, 120)], [(0, 3), (1, 3)]]
    got, cls = {}, PPDR.PearlCDeviceRun
    for behind in (False, True):
        monkeypatch.setattr(cls, "retain_behind", behind)
        root = tmp_path / f"behind-{behind}"
        served = _Steps(monkeypatch, PPD.Kernel(R90, H100Dev(), "0" * 64), weights, steps,
                        {"transcript_dir": str(root)}, capacity=32, flag_rows=512)
        for _ in steps:
            served.step()
        assert all(p["manifest"] and p["excluded_rows"] is not None for p in served.release()["passes"])
        got[behind] = _retained_files(root)
        modes = {p.stat().st_mode & 0o777 for p in root.glob("pass-*/**/*")} | {p.stat().st_mode & 0o777 for p in root.glob("pass-*")}
        assert modes == {0o600, 0o700}, sorted(map(oct, modes))
    assert got[True] == got[False] and len(got[True]) == 3 * (1 + 3 + 2 * 2 * 3 + 2 * 3)
    assert hashlib.sha256(json.dumps(got[True], sort_keys=True).encode()).hexdigest() == RETAINED_DIGEST


def test_a_pass_poisons_again_only_the_shapes_a_call_took_since_the_last_poison(monkeypatch, tmp_path):
    """`PearlCDeviceRun.poison`: each retained pass begins with every shape's buffers at the poison byte, but a shape no call
    took since the last poison holds it already and isn't filled again; one a call took is."""
    rng = np.random.default_rng(37)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    steps = [[(0, 200), (1, 200)], [(0, 3), (1, 3)], [(0, 3), (1, 3)]]
    served = _Steps(monkeypatch, PPD.Kernel(R90, H100Dev(), "0" * 64), weights, steps,
                    {"transcript_dir": str(tmp_path)}, capacity=32, flag_rows=512)
    fills, fill = [], torch.Tensor.fill_
    monkeypatch.setattr(torch.Tensor, "fill_", lambda t, v: (fills.append(t.data_ptr()), fill(t, v))[1])

    def buffers(rows):
        run = served.scheme.run
        return [t for (m16, _, _), sh in run.shapes.items() if (m16 > 16) == rows for t in run._buffers(sh)]
    served.step()
    big = buffers(True)
    assert big and {t.data_ptr() for t in big} <= set(fills)
    fills.clear()
    served.step()
    assert not {t.data_ptr() for t in big} & set(fills) and all(bool((t == 0xA5).all()) for t in big)
    assert {t.data_ptr() for t in buffers(False)} <= set(fills)
    served.step()
    served.release()


def test_the_record_decided_from_the_rows_bytes_is_the_devices(monkeypatch):
    """`screen.excluded_host`, which a device's retained call decides its record by from the bytes its rows' write lands
    (the flags and A's rows as BF16), excludes the rows `screen.excluded` does from the flags and A's FP32 words."""
    S = importlib.import_module("verity_torch.pouw.pearl_c.screen")
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


def test_an_attached_pearl_plus_after_its_first_step_holds_the_run_the_harness_gate_reads(monkeypatch):
    """`Attached.run` after the first step with tokens is the device run `verity_bench.vllm.runner._pearl_c_gate` checks,
    with every name that gate and `bench_arm.PearlCArm.gate` read on it, its ``weights_root`` in the commits the run
    itself made."""
    rng = np.random.default_rng(41)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    cls = PPDR.PearlCDeviceRun
    served = _Steps(monkeypatch, PPD.Kernel(R90, H100Dev(), "0" * 64), weights, [[(0, 8), (1, 8)]])
    assert served.scheme.run is None
    served.step()
    run = served.scheme.run
    assert isinstance(run, cls) and isinstance(run.schedule.name, str) and run.omit == frozenset()
    assert all(callable(getattr(run, f)) for f in ("begin", "end", "gemm", "poison", "rounds"))
    assert isinstance(run.details, dict) and run.scheme is not None
    assert {w: x.tensor.data_ptr() for w, x in run.weights.items()} == {w: t.data_ptr() for w, t in weights.items()}
    assert run.commits["weights_root"] == run.epoch.weights_root.hex() and "registration" in run.commits
    assert run.commits.get("coin_source", "in-process") == "in-process"
    served.release()


def test_a_step_over_one_pass_is_refused_before_its_forward(monkeypatch):
    """`passes.step_tokens`: weights of n 256 and 128 take 4 + 2 tiles a 64-row tile, so a pass of 32 tiles holds 5 tile
    rows (320 tokens) and one of 512 screened rows 256 tokens over its two calls a step: a step is held to 256 tokens
    (``details["max_step_tokens"]``).  Pearl+ refuses a step begun over it before any block, leaving the open pass empty
    and nothing committed; a step at the limit commits as one pass, within both caps."""
    rng = np.random.default_rng(29)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16)
               for w, n in ((0, 256), (1, 128))}
    served = _Steps(monkeypatch, PPD.Kernel(R90, H100Dev(), "0" * 64), weights,
                    [[(0, 257), (1, 257)], [(0, 256), (1, 256)]], capacity=32, flag_rows=512)
    with pytest.raises(RuntimeError, match=r"a step of 257 tokens is over one pass \(256 tokens a step"):
        served.begin()
    run = served.scheme.run
    assert run.details["max_step_tokens"] == 256 and not run.cur.matmuls and run.details["passes"] == []
    served.at = 1
    served.step()
    assert [(p["units"], p["tiles"], p["screened_rows"]) for p in served.release()["passes"]] == [(2, 24, 512)]


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
    """The constants the pipeline reads (`audit.hm96_kernel_constants`, and no salt key: a tree's salts are its buffer's
    bytes) and a run's tree hash are the scheme's: hm96-sha512's."""
    s = pouw_schemes.get(scheme)
    kernel = PPD.Kernel(R, RecordingDev(), "0" * 64)
    t = PPDR._constants(torch.device("cpu"))
    assert PPDR._bytes(t) == hm96_kernel_constants()
    assert PPDR._constant_keys(t) == {"hm": t.data_ptr()}
    rng = np.random.default_rng(1)
    weights = {0: torch.from_numpy(bf16_words(rng, 128, 1024).view(np.float32).copy()).to(torch.bfloat16)}
    roots = PPDR.weight_roots(kernel, weights, scheme, weight_salts=salts_b(weights))
    run = PPDR.PearlCDeviceRun(s, f"pouw:{scheme}", kernel, weights, roots, *salt_coin(roots), capacity=16,
                               weight_salts=salts_b(weights))
    assert run.tree_hash == s.commitment_hash == C.TREE_HASH["h3"] == HM96 and PPDR._bytes(run.const) == PPDR._bytes(t)


@pytest.mark.slow
@pytest.mark.skipif(not (shutil.which("g++") or shutil.which("c++")), reason="simt_twin needs a C++ compiler")
@pytest.mark.parametrize("scheme", KERNEL_SCHEMES)
def test_every_tree_the_device_path_commits_is_its_hashing_formats(scheme, tmp_path, monkeypatch):
    """B's roots, each call's A root over its m₁₆ rows (filler zero), and the pass's tile tree over the leaves the calls wrote,
    their kernels run by `simt_twin` and hash_decode.cuh's host build: each is `MerkleTree`'s root in hm96-sha512
    (frame-v3-sha512 over values salted with the salts the run was handed or drew: B's, each call's A's, the tiles').  A decode call's commitment is pd_commit's: each row digest
    `pearl_c.row_digest` of its row, the row seeds the scheme's `row_seeds` with their line keys, seed_A the scheme's, and E_A's
    lines the m₁₆ committed rows' under their own keys and the filler rows' under line_a; its tile leaves (pd_tiles') over
    the zero C~ and U are `pearl_c.leaf`'s."""
    s = pouw_schemes.get(scheme)
    h, nb = s.commitment_hash, 64
    rng = np.random.default_rng(5)
    weights = {w: torch.from_numpy(bf16_words(rng, n, 1024).view(np.float32).copy()).to(torch.bfloat16) for w, n in ((0, 256), (1, 128))}
    rows = lambda x: [r.tobytes() for r in x.float().numpy()]  # noqa: E731
    kernel = PPD.Kernel(R, TwinDev(HD.build(tmp_path)), "0" * 64)
    key = salts_b(weights)
    roots = PPDR.weight_roots(kernel, weights, scheme, weight_salts=key)
    assert roots == {w: commit_rows(weight_binding(w), -2, rows(t), h,
                                    [key[w][i:i + HM96_SALT] for i in range(0, len(key[w]), HM96_SALT)]).commitment.root
                     for w, t in weights.items()}
    declared = Workload(tuple(Matmul(w, Shape(16, 1024, t.shape[0]), w) for w, t in weights.items()))
    coins = salt_coin(roots)
    salt = Epoch.start(declared, roots, *coins).salt
    run = PPDR.PearlCDeviceRun(s, f"pouw:{scheme}", kernel, weights, roots, *coins, capacity=1 << 12, weight_salts=key)
    calls = [(0, torch.from_numpy(bf16_words(rng, 20, 1024).view(np.float32).copy()).to(torch.bfloat16)),
             (1, torch.from_numpy(bf16_words(rng, 32, 1024).view(np.float32).copy()).to(torch.bfloat16))]
    a_salts = []
    for w, x in calls:
        kernel.dev.log.clear()
        run.gemm(w, x, weights[w], None)
        names = [n for n, *_ in kernel.dev.log if not n.endswith("_smem")]
        assert names[0] == "pd_commit_851" and names[-1] == "pd_tiles_851", names
        sh, index = run.shapes[(32, weights[w].shape[0], 1024)], len(run.cur.matmuls) - 1
        got = PPDR._bytes(sh.buf("salts_a")[:HM96_SALT * 32])
        a_salts.append([got[i:i + HM96_SALT] for i in range(0, len(got), HM96_SALT)])
        a = torch.zeros((32, 1024), dtype=torch.float32)
        a[:x.shape[0]] = x.float()
        mm = Matmul(index, Shape(32, 1024, weights[w].shape[0]), w)
        ctx = Context(salt, mm, PPDR._bytes(run.roots_buf[nb * index:nb * index + nb]), roots[w])
        want, seed_a = s.row_seeds(ctx, rows(a), range(32)), s.noise(ctx).seed_a
        assert PPDR._bytes(sh.b["rowdig_a"][:64 * 32]) == b"".join(C.row_digest(r) for r in rows(a))
        assert PPDR._bytes(sh.b["seeds_a"][:32 * 32]) == b"".join(want)
        assert PPDR._bytes(sh.b["lkeys_a"][:32 * 32]) == b"".join(C.line_key(z) for z in want)
        assert PPDR._bytes(sh.b["seed_a"][:32]) == seed_a
        assert PPDR._bytes(sh.b["ea_codes"][:32 * sh.mp]) == b"".join(
            bytes(C.sample_line(C.S512Seed(want[t]), 0, 0, 0) if t < 32 else C.sample_line(C.S512Seed(seed_a), 0, 0, t))
            for t in range(sh.mp))
        (commit,) = [(p, i) for n, _, p, i in kernel.dev.log if n == "pd_commit_851"]
        assert commit[0][4] == sh.b["lkeys_a"].data_ptr() and commit[0][12] == sh.b["ea_codes"].data_ptr()
        assert commit[1] == (32, 1024, 0, sh.mp)
    for index, (w, x) in enumerate(calls):
        a = torch.zeros((32, 1024), dtype=torch.float32)
        a[:x.shape[0]] = x.float()
        want = commit_rows(activation_binding(salt, index), index, rows(a), h, a_salts[index]).commitment.root
        assert PPDR._bytes(run.roots_buf[nb * index:nb * index + nb]) == want, index
    count = run.cur.tiles
    assert count == 4 + 2
    zero = C.leaf(C.Checked([[0] * 64] * 32, [[0] * 64] * 32), "h3")
    assert PPDR._bytes(run.leaves[:nb * count]) == zero * count
    leaves = rng.integers(0, 256, nb * count, dtype=np.uint8).tobytes()      # no GEMM ran: every tile's leaf is the same
    run.leaves[:nb * count] = torch.frombuffer(bytearray(leaves), dtype=torch.uint8)
    kernel.dev.log.clear()
    drawn, fresh = [], ENDS.fresh_into

    def recorded(dst, n, raw=None, *, salts):
        fresh(dst, n, raw, salts=salts)
        drawn.append(PPDR._bytes(dst[:HM96_SALT * n]))
    monkeypatch.setattr(ENDS, "fresh_into", recorded)
    out = run.end()
    assert [n for n, *_ in kernel.dev.log][-2] == "s512_frame_leaves"
    (salts_t,) = drawn
    dt = transcript_domain(Epoch(Workload(tuple(run.cur.matmuls)), roots, salt), count, h)
    tree = MerkleTree(dt, {i: leaves[nb * i:nb * i + nb] + salts_t[HM96_SALT * i:HM96_SALT * (i + 1)] for i in range(count)},
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
PROMOTED = sorted(s for s in kernel_names("pearl-c") if "unpromoted" not in s)


@pytest.fixture(params=PROMOTED)
def honest_pass(tmp_path, request):
    return proved_pass(tmp_path, request.param)


def seeded_salts(seed: int):
    """192-byte salts drawn from `seed`, a fixture's stand-in for fresh ones, the same on every run."""
    rng = random.Random(seed)
    return lambda n: [rng.randbytes(HM96_SALT) for _ in range(n)]


def pinned_salts(key: bytes, nonces: list[int]):
    """A stand-in for the reference prover's `audit.fresh_salts` that hands it, as data and in the order it draws them
    (`nonces`, one a tree), the salts `proved_pass`'s passes are pinned under: `audit.hm96_salt`'s stream under `key`.
    The facts these tests hold a pass to (which credited rows are over their caps, which blocks a row's words flag) follow
    A's salted roots, so they are the pinned salts' facts."""
    left = iter(nonces)

    def draw(n: int) -> list[bytes]:
        nonce = next(left)
        return [hm96_salt(key, nonce, i) for i in range(n)]
    return draw


def split_salts(kept: bytes) -> list[bytes]:
    """A salts file's salts, `HM96_SALT` bytes each in rank order."""
    return [kept[i:i + HM96_SALT] for i in range(0, len(kept), HM96_SALT)]


def kept_salts(tree: MerkleTree) -> bytes:
    """The salts of an hm96 tree's values in rank order, each value's last `HM96_SALT` bytes: a pass's salts file."""
    return b"".join(tree.open(i).value[-HM96_SALT:] for i in range(tree.domain.count))


def kept_leaves(d, tree: MerkleTree, schema: str = ROW_SCHEMA) -> list[bytes]:
    return [d.leaf(i, i, schema, tree.open(i).value) for i in range(tree.domain.count)]


def proved_pass(tmp_path: Path, scheme: str, round_: int = 0, key: bytes | None = None,
                live: int | None = None, seed: int = 11, spike: float | None = None, bad_b: int | None = None,
                tile: tuple[int, int] = (16, 16), zero_a: int | None = None,
                load: tuple[int, int] | None = None, k: int = 128, filler: int | None = None):
    """Two matmuls of a test instance (k = 128, 16 x 16 tiles), one with 12 filler rows and one liveness-failing row, proved by
    the scheme's reference prover and written in the device run's layout, for each promoted device scheme (hm96 trees, their
    salts the prover's, kept in the pass's salts files beside the rows; the unpromoted record's debit isn't stated, so
    clear_work can't audit it).  The prover's salts are the pinned ones under `key` (`pinned_salts`), fixed per scheme unless
    `key` names one: -h3's row seeds follow A's salted roots, and under some keys (about 2 in 15) a credited row of a
    sampled tile is over its own cap, which the sampling design and verify_run both reject.  On the h100 record (Hopper's
    14-bit atom) 2 of the first 8 keys are, the zero key among them: at k = 128 a row's cap admits one flagged atom-word
    (at k = 2,048 and 6,144 the worst of 96 Gaussian rows each is at 0.27 and 0.11 of its cap).
    The pass is round `round_` of the execution phase, its two calls at their places (`protocol.call_place`), and every
    pass built here is of the one execution phase (the same weights, salts and salt coin), as passes of one install are.  With `live` the salt coin is a live session's, opened after salt execution phase
    `live`'s root over the weights' root (`live_coin`), and the manifest names that execution phase (``pass_index``).  `seed` samples
    the rows: under another, the weights are others, and so is the salt the one coin gives them.  With `spike`, credited
    row 9 of the first call holds one entry `spike` times its every-8th entries' RMS (ρ's positions), off them.  With
    `bad_b`, that row of the second call's weight fails liveness as the first call's row 5 does, and is excluded.  `tile`
    is the instance's tile: at 64 x 64 each weight has 64 rows and each matmul is one tile, partial down its rows, k_min 32
    keeping it a test instance.  With `zero_a`, that real row of the first call is all +0: it fails liveness and is
    excluded, its U is +0 and so is every word of its served y.  With `load` (rows, committed rows), both calls are of
    that many real rows in that many committed ones, as a round of a class's fixed load is (`verify_run.held_to_load`),
    the first call's failing row its row 1.  `k` is the calls' depth.  With `filler` (an FP32 word of a BF16 value), the first
    call's first filler row holds that word at its column 0, committed and proved as every row is: a prover whose filler
    row is not zero."""
    if key is None:
        key = bytes([1]) * 32 if SchemeId.parse(scheme).device == "h100" else bytes(32)
    s = pouw_schemes.get(scheme, k_min=32, tile=tile)
    rng = np.random.default_rng(seed)
    n0, n1 = max(16, tile[1]), max(32, tile[1])
    wrows = {0: bf16_words(rng, n0, k), 1: bf16_words(rng, n1, k)}
    (real, m0), m1, bad = (load, load[1], 1) if load else ((20, 32), 16, 5)
    a0 = bf16_words(rng, m0, k)
    a0[real:] = 0
    a0[bad, [1, 2, 3]] = np.float32(1e4).view(np.uint32) & 0xFFFF0000
    if spike is not None:
        a0[9, 1] = np.float32(spike * np.sqrt(np.mean(a0[9, ::8].view(np.float32) ** 2))).view(np.uint32) & 0xFFFF0000
    if bad_b is not None:
        wrows[1][bad_b, [1, 2, 3]] = np.float32(1e4).view(np.uint32) & 0xFFFF0000
    if zero_a is not None:
        a0[zero_a] = 0
    if filler is not None:
        a0[real, 0] = filler
    i0, i1 = call_place(round_, 0), call_place(round_, 1)
    acts = {i0: a0, i1: bf16_words(rng, m1, k)}
    if load:
        acts[i1][real:] = 0
    def rows(x):
        return [r.tobytes() for r in x]
    work = Workload((Matmul(i0, Shape(m0, k, n0), 0), Matmul(i1, Shape(m1, k, n1), 1)))
    drawn = pinned_salts(key, [HM96_WEIGHTS + w for w in wrows] + [u.index for u in work.matmuls] + [HM96_TRANSCRIPT])
    with patch.object(sys.modules["clear_audit"], "fresh_salts", drawn):
        prover = Prover(s, {w: rows(x) for w, x in wrows.items()})
    if live is None:
        coins = salt_coin(prover.weight_roots)
        epoch = Epoch.start(work, prover.weight_roots, *coins)
    else:
        coins = live_coin(PS.epoch_open_root(weights_root(prover.weight_roots), live))
        epoch = PPD.epoch_of(work, prover.weight_roots, *coins, live)
    with patch.object(sys.modules["clear_audit"], "fresh_salts", drawn):
        tr = prover.prove(epoch, {i: rows(x) for i, x in acts.items()})
    splits = {w: PW.Split(k) for w in wrows}
    records = CW.declare(s, epoch, {i: rows(x) for i, x in acts.items()}, {w: rows(x) for w, x in wrows.items()}, splits,
                         real_rows={i0: real, **({i1: real} if load else {})})
    assert records[0].excluded_a == tuple(sorted({bad} | ({zero_a} - {None})))
    assert bad_b is None or records[1].excluded_b == (bad_b,)
    (tmp_path / "calls").mkdir(parents=True)
    (tmp_path / "weights").mkdir()
    weights = {}
    for w, x in wrows.items():
        d = rows_domain(weight_binding(w), -2, len(x), s.commitment_hash)
        weights[str(w)] = {"n": len(x), "k": k, "root": prover.weight_roots[w].hex(), "excluded_b": list(records[0 if w == 0 else 1].excluded_b),
                           "rows": PPDR._write(tmp_path / "weights" / f"{w}.rows.bf16", bf16_file(rows(x)), digest=False),
                           "tree": PPDR._write(tmp_path / "weights" / f"{w}.tree", laid_out(d, kept_leaves(d, prover.weight_trees[w]))),
                           "salts": PPDR._write(tmp_path / "weights" / f"{w}.salts", kept_salts(prover.weight_trees[w]))}
    matmuls = []
    for u, rec, root, at in zip(work.matmuls, records, tr.activation_roots, prover._act_trees):
        x = rows(acts[u.index])
        d = rows_domain(activation_binding(epoch.salt, u.index), u.index, u.shape.m, s.commitment_hash)
        y = reference_y(s, Context(epoch.salt, u, root, epoch.weight_roots[u.weight]), x, rows(wrows[u.weight]))
        y = y[:2 * rec.real_rows * u.shape.n]
        matmuls.append({"index": u.index, "m": u.shape.m, "k": k, "n": u.shape.n, "weight": u.weight, "real_rows": rec.real_rows,
                        "k_kept": rec.k_kept, "excluded_a": list(rec.excluded_a), "root_a": root.hex(),
                        "rows": PPDR._write(tmp_path / "calls" / f"{u.index}.rows.bf16", bf16_file(x), digest=False),
                        "tree": PPDR._write(tmp_path / "calls" / f"{u.index}.tree", laid_out(d, kept_leaves(d, at))),
                        "salts": PPDR._write(tmp_path / "calls" / f"{u.index}.salts", kept_salts(at)),
                        "y": SY.retain_y(tmp_path, u.index, y, u.shape.n, epoch.salt)})
    count = tr.commitment.count
    leaves = [prover.digest(t) for t in range(count)]
    dt = transcript_domain(epoch, count, s.commitment_hash)
    head = {"schema": CC.MANIFEST_SCHEMA, "run": RUN, "entry": f"pouw:{scheme}", "scheme": scheme,
            "instance": {"k_min": 32, "tile": list(tile)},
            "coins": coins[0].to_json(), "salt_opening": coins[1].to_json(), "salt": epoch.salt.hex(),
            **({} if live is None else {"pass_index": live}),
            "records_root": PW.records_root(epoch.salt, epoch.weights_root, splits, records).hex(),
            "commitment": {"root": tr.commitment.root.hex(), "count": count},
            "leaves": PPDR._write(tmp_path / "leaves", b"".join(leaves)),
            "tree_t": PPDR._write(tmp_path / "tree_t", laid_out(dt, kept_leaves(dt, prover._tree, TILE_SCHEMA))),
            "salts_t": PPDR._write(tmp_path / "salts_t", kept_salts(prover._tree))}
    path = PPDR.write_manifest(tmp_path, head, weights, matmuls)
    (tmp_path / "coins.json").write_text(json.dumps(coins[0].to_json()))
    return path, s, epoch, prover, tr, splits, records


@pytest.mark.slow
def test_verify_run_accepts_an_honest_pass(honest_pass):
    path, s = honest_pass[:2]
    source = b"verifier" * 4
    got = VR.verify(path, 3, 1, "test", source, coins=published(path))
    assert got["accepted"], got.get("line") or got.get("error")
    assert got["exclusion_tiles"] == 1
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
    """``served_passes.py verify`` on serve dir `d` at 3 samples, a test instance's (`proved_pass`'s): its exit, its document
    and its core answer."""
    SP = load("pearl_c_vllm_served_passes", POUW / "pearl_c_vllm" / "served_passes.py")
    rc = SP.main(["verify", "--passes", str(d), "--out", str(out), "--jobs", "1", "--draws", "3", "--test-instance", *more])
    answer = json.loads((out.parent / "answer.json").read_text())
    from verity.protocols.accounting.compute.pouw.verifier import core_check as CC
    CC.answer_of(answer)
    return rc, json.loads(out.read_text()), answer


def test_answer_names_its_schemes_assumption():
    """An accepted Pearl+ answer's ``public.assumption`` (`verify_run.assumption_of`) is the named Prop its scheme's credit,
    (1 − ρ)·`creditDevRev1`, is at: TT_OUT(1/400) per audit tile, rev1, at the scheme's device record. The served scheme,
    ``pearl-c-h100-v1-h3``, is at `TTOutTilePearlCH100Rev1`, every other scheme the device path runs at
    `TTOutTilePearlCDevRev1` at its record, a def of `TTOutRev1.lean`, which the H100's is at `devH100` (a def of the
    records Proofs/Pouw/PearlC/Records.lean keeps since 12b47061986); the Lean port's `ASSUMPTION` is the H100's. Neither
    Pearl's own Assumption 1 (`pearl_kw`, which names no gap or constant) nor a v0 scheme, which has no TT_OUT accounting,
    is ever named."""
    root = Path(__file__).resolve().parents[3]
    lean = (root / "security/Definitions/Pouw/Assumptions/PearlC/TTOutRev1.lean").read_text()
    records = (root / "security/Proofs/Pouw/PearlC/Records.lean").read_text()
    dev, h100 = (name.rsplit(".", 1)[1] for name in (VR.ASSUMPTION_DEV, VR.ASSUMPTION_H100))
    assert all(name.startswith("Pouw.PearlC.Assumptions.") for name in (VR.ASSUMPTION_H100, VR.ASSUMPTION_DEV))
    assert f"def {dev} " in lean, dev
    assert re.search(rf"^def {h100} .*\n(?:.*\n)*?  {dev} CM devH100 sem ρ$", records, re.M), h100
    assert VR.assumption_of(pouw_schemes.get("pearl-c-h100-v1-h3")) == VR.ASSUMPTION_H100
    for scheme in map(pouw_schemes.get, kernel_names("pearl-c")):
        got = VR.assumption_of(scheme)
        assert got == (VR.ASSUMPTION_H100 if scheme.device.name == "h100" else f"{VR.ASSUMPTION_DEV} at {scheme.device.name}")
        assert "pearl_kw" not in got
    run = (root / "verity/protocols/accounting/compute/pouw/verifier/lean/pearlplus/PouwPearlPlus/Run.lean").read_text()
    assert f'def ASSUMPTION : String := "{VR.ASSUMPTION_H100}"' in run


@pytest.mark.slow
def test_served_passes_verify_accepts_only_when_every_pass_with_calls_is_accepted(honest_pass, tmp_path):
    """``served_passes.py verify`` samples over the passes with calls that the serve's served.json names, skipping the empty
    one the run left open: it accepts an honest pass and credits its MACs at its scheme's named assumption in a core
    answer (`test_answer_names_its_schemes_assumption`), and the
    same pass with a weight row flipped (opened on the excluded row's tile, whatever the sample) fails the whole run.  A
    test instance's verdict credits nothing.  The line names the credit and the sample, and neither the exclusion tiles'
    count nor the MACs the pass declares, which its given-up rows decide."""
    path, *_ = honest_pass
    served = {"run": RUN, "passes": [{"pass": 0, "units": 2, "tiles": 4, "manifest": str(path)},
                                     {"pass": 1, "units": 0, "tiles": 0}]}
    (path.parent / "served.json").write_text(json.dumps(served))
    rc, doc, answer = served_verify(path.parent, tmp_path / "verify.json")
    assert rc == 0 and answer["verdict"] == "accept" and doc["line"].startswith("ACCEPT verity/pouw/pearl-plus/v1"), doc["line"]
    work = answer["public"]["work"]
    assert work["credited"] == 0 < work["checked_macs"] and work["credited_tiles"] > 0 and answer["profile"]["delta"] == "1/100"
    assert answer["profile"]["test_instance"] and "a test instance's, credited nothing" in doc["line"]
    named = VR.assumption_of(pouw_schemes.get(answer["public"]["scheme"]))
    assert answer["public"]["assumption"] == named and f"({named})" in doc["line"]
    assert len(answer["profile"]["tiles"]) == 4 and {i for i, _ in answer["profile"]["tiles"]} == {0}
    man = json.loads(path.read_text())
    assert answer["public"]["passes"][0]["y"] == [u["y"]["root"] for u in man["matmuls"]]
    assert "y committed and checked on 2 of 2 calls" in doc["line"]
    assert answer["profile"]["exclusion_tiles"] > 0 and " drawn tiles and the exclusion tiles over 1 passes, " in doc["line"]
    assert "; 0 MACs credited at ε " in doc["line"] and f"{work['checked_macs']:,}" not in doc["line"]
    assert not re.search(r"\d[\d,]* (exclusion tiles|of [\d,]+ MACs)", doc["line"])
    rows = path.parent / man["weights"]["0"]["rows"]["path"]
    bad = bytearray(rows.read_bytes())
    bad[1] ^= 0x01
    rows.write_bytes(bytes(bad))
    rc, doc, answer = served_verify(path.parent, tmp_path / "verify.json")
    assert rc == 1 and answer["verdict"] == "reject" and "weight opening" in answer["reason"] and doc["line"].startswith("REJECT")


@pytest.mark.slow
def test_a_credited_call_that_commits_no_y_root_is_refused_at_load(tmp_path):
    """A call with a credited cell commits its y: without one no tile of it would be held to y, so the pass is refused
    before the coin, by name."""
    path = proved_pass(tmp_path, PROMOTED[0])[0]
    man = json.loads(path.read_text())
    del man["matmuls"][0]["y"]
    path.write_text(json.dumps(man))
    with pytest.raises(ValueError, match=f"^matmul {man['matmuls'][0]['index']}: a credited call commits no y root$"):
        VR.load(path, published(path))
    got = VR.verify_passes([path], 3, 1, "test", published(path), b"verifier" * 4, test_instance=True)
    assert got["answer"]["verdict"] == "reject" and "a credited call commits no y root" in got["answer"]["reason"]


@pytest.mark.slow
def test_a_served_y_that_is_not_the_unscale_of_its_u_rejects_the_run(tmp_path):
    """A pass whose y is not what its U unscales to fails the run, whichever tile is sampled: committed as it is (a new root,
    so the run's commitment changes too) each sampled tile names a word that isn't its U's unscale, and left under the root
    the pass published, the rows don't open."""
    path = proved_pass(tmp_path, PROMOTED[0])[0]
    man = json.loads(path.read_text())
    good = VR.verify_passes([path], 3, 1, "test", published(path), b"verifier" * 4, test_instance=True)
    assert good["answer"]["verdict"] == "accept", good["line"]
    for u in man["matmuls"]:
        y = bytearray((path.parent / u["y"]["rows"]["path"]).read_bytes())
        y[0::2] = bytes(b ^ 0x01 for b in y[0::2])
        u["y"] = SY.retain_y(path.parent, u["index"], bytes(y), u["n"], bytes.fromhex(man["salt"]))
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
        u["y"] = {**u["y"], "root": poison[-64:].hex(), "tree": PPDR._write(file, poison)}
    path.write_text(json.dumps(man))
    one = VR.verify(path, 3, 1, "test", source, coins=published(path))
    assert not one["accepted"] and "y opening" in one["line"], one["line"]
    passes = VR.verify_passes([path], 3, 1, "test", published(path), source, test_instance=True)
    assert passes["answer"]["verdict"] == "reject" and "y opening" in passes["answer"]["reason"], passes["line"]


def short_y(path: Path, pos: int, drop: int) -> None:
    """Call `pos`'s y rows file less its last `drop` bytes, committed as the verifier's opener reads it (`VR.PO.f32_row`: row
    i is the words the file holds of its bytes [2ni, 2n(i + 1)), a ragged row's whole ones here) under a new root, which the
    manifest then names: a prover's own y whose last real row is short, and opens."""
    man = json.loads(path.read_text())
    u = man["matmuls"][pos]
    n, real = u["n"], u["real_rows"]
    ys = (path.parent / u["y"]["salts"]["path"]).read_bytes()
    file = path.parent / u["y"]["rows"]["path"]
    data = file.read_bytes()
    data = data[:len(data) - drop]
    rows = [data[2 * n * i:2 * n * (i + 1)] for i in range(real)]
    d = SY.y_domain(bytes.fromhex(man["salt"]), u["index"], real)
    tree = laid_out(d, [d.leaf(i, i, ROW_SCHEMA, VR.PO.f32_row(r[:len(r) & ~1], len(r), 0) + ys[HM96_SALT * i:HM96_SALT * (i + 1)])
                        for i, r in enumerate(rows)])
    u["y"] = {"root": tree[-64:].hex(), "rows": PPDR._write(file, data, digest=False),
              "tree": PPDR._write(path.parent / u["y"]["tree"]["path"], tree), "salts": u["y"]["salts"]}
    path.write_text(json.dumps(man))


@pytest.mark.slow
def test_a_served_y_row_that_ends_before_its_tiles_columns_rejects_the_run(tmp_path):
    """A y row shorter than its tile is refused by its length, and never read past its end.  The opening binds a row's
    value and not how many words it has, so a prover can commit one: here the last real row of the first call, all +0
    (its A row is zero, so it is excluded and its tile is opened whatever the sample), less its last 4 words, the tree and
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
    real, index, salt = 8, 3, bytes(64)
    salts = fresh_salts(real)
    d = SY.y_domain(salt, index, real)
    tile = types.SimpleNamespace(rows=(7,), cols=(2, 3))
    side = lambda n: types.SimpleNamespace(alpha=[PCU.ONE] * n)  # noqa: E731
    v = types.SimpleNamespace(checked=types.SimpleNamespace(u=[[0, 0]]), ops=types.SimpleNamespace(a=side(1), b=side(2)))
    assert PCU.y(0, PCU.ONE, PCU.ONE) == 0

    def reason(row: bytes) -> str | None:
        values = [(row if i == 7 else bytes(16)) + salts[i] for i in range(real)]
        tree = laid_out(d, [d.leaf(i, i, ROW_SCHEMA, x) for i, x in enumerate(values)])
        p = types.SimpleNamespace(scheme=types.SimpleNamespace(commitment_hash=HM96, forming="v1", name="stand-in"),
                                  epoch=types.SimpleNamespace(salt=salt, workload=types.SimpleNamespace(
                                      matmuls=[types.SimpleNamespace(index=index)])),
                                  man={"matmuls": [{"real_rows": real, "y": {"root": tree[-64:].hex()}}]})
        return VR.check_y(p, 0, tile, v, (VR.PO.opening(tree, real, 7, values[7], 64),))

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


def test_served_passes_verify_refuses_a_pass_whose_calls_are_not_at_its_place(tmp_path):
    """Two honest passes of one execution phase, both proved as round 0 (calls 0 and 1): the second is the run's round 1,
    whose calls' places are 2^16 and 2^16 + 1, so its call 0's seeds, noise and tickets are the first's and the run is
    refused (``call-place``, `verify_run.passes_placed`) before any pass is replayed, whether served.json or
    ``--manifest`` names them."""
    a, b = (proved_pass(tmp_path / name, PROMOTED[0])[0] for name in ("a", "b"))
    assert json.loads(a.read_text())["salt"] == json.loads(b.read_text())["salt"]
    for d, more in ((_serve_dir(tmp_path, [a, b]), ()), (a.parent, ("--manifest", str(a), "--manifest", str(b)))):
        rc, doc, answer = served_verify(d, tmp_path / d.name / "verify.json", *more)
        assert rc == 1 and answer["verdict"] == "reject" and "verdicts" not in doc and doc["refused"][0] == VR.CALL_PLACE
        assert "pass 1's call 0 holds index 0, not its place 65536 (round 1, position 0)" in doc["refused"][1]
        assert [x.split(" sha256:")[0].split(":", 1)[1] for x in doc["passes"]] == [str(a), str(b)]


@pytest.mark.slow
def test_served_passes_verify_accepts_an_epochs_passes_numbered_on(tmp_path):
    """Two honest passes of one execution phase at their places, rounds 0 and 1, as the device numbers a serve's passes: one sample over
    both, each pass's exclusion tile opened, accepted, its MACs over both (a test instance's, credited nothing)."""
    a, b = (proved_pass(tmp_path / name, PROMOTED[0], s)[0] for name, s in (("a", 0), ("b", 1)))
    rc, doc, answer = served_verify(_serve_dir(tmp_path, [a, b]), tmp_path / "verify.json")
    assert rc == 0 and answer["verdict"] == "accept" and "refused" not in doc
    assert answer["profile"]["exclusion_tiles"] == 2 and len(answer["public"]["passes"]) == 2
    one = VR.credit_at_assumption([VR.load(a, published(a))], 3, Fraction(1, 100))
    assert answer["public"]["work"]["checked_macs"] == 2 * one["checked_macs"]


@pytest.mark.slow
def test_verify_passes_reads_a_test_instance_only_as_one_and_a_runs_passes_of_one_salt(tmp_path):
    """A test instance's pass (k_min 32, at 16 x 16 tiles or at the served 64 x 64) is refused as a served run's, whose
    instance is k_min 1,024 at 64 x 64, and accepted as a test instance's, crediting nothing.  Passes of two salts (other
    weights under the one coin) are refused, though each holds its calls at its place."""
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
        assert test["answer"]["verdict"] == "accept" and work["credited"] == 0 < work["checked_macs"], test["line"]
    coins, source = published(a), b"verifier" * 4
    b = proved_pass(tmp_path / "b", PROMOTED[0], 1, seed=12)[0]
    assert json.loads(a.read_text())["salt"] != json.loads(b.read_text())["salt"]
    for kw in ({}, {"test_instance": True}):
        two = VR.verify_passes([a, b], 3, 1, "test", coins, source, **kw)
        assert two["answer"]["reason"] == "ValueError: the passes are not all of one salt" and "refused" not in two


def test_a_runs_credit_at_pearls_assumption_is_its_credited_macs_less_epsilon(tmp_path):
    """`verify_run.epsilon` is the least ε (on its 10^-9 grid) with (1 − ε)^t ≤ δ, and, with X_O samples over the
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
    assert got["checked_macs"] == ((20 - 1) * 16 + 16 * 32) * 128 and got["epsilon"] == str(eps)
    assert 0 < got["credited"] <= (1 - p.scheme.cap) * (1 - eps) * got["checked_macs"]


def test_served_passes_verify_credits_the_timed_call_its_passes_less_the_runs_whole_deduction(tmp_path):
    """Where served.json says which passes the timed call made, an accepted verdict also carries the credit over those
    alone (`served_passes.timed_credit`): the run's credited MACs less the warm-up pass's credited cells' MACs at (1 − ρ),
    rounded up, so the run's deduction for ε falls wholly on the timed pass."""
    SP = load("pearl_c_vllm_served_passes", POUW / "pearl_c_vllm" / "served_passes.py")
    a, b = (proved_pass(tmp_path / name, PROMOTED[0], s)[0] for name, s in (("a", 0), ("b", 1)))
    served = {"run": RUN, "passes": [{"pass": 0, "timed": False, "units": 2, "tiles": 4, "manifest": str(a)},
                                     {"pass": 1, "timed": True, "units": 2, "tiles": 4, "manifest": str(b)}]}
    work = VR.credit_at_assumption([VR.load(x, published(x)) for x in (a, b)], SP.DRAWS, Fraction(1, 100))
    warm = VR.pass_macs(VR.load(a, published(a)))
    got = SP.timed_credit(served, work, published(a))
    assert warm == work["checked_macs"] // 2 and got["passes"] == 1 and got["macs"] == work["checked_macs"] - warm
    assert got["credited"] == work["credited"] - math.ceil((1 - Fraction(work["rho"])) * warm) > 0
    few = VR.credit_at_assumption([VR.load(x, published(x)) for x in (a, b)], 3, Fraction(1, 100))
    assert few["credited"] < (1 - Fraction(few["rho"])) * warm and SP.timed_credit(served, few, published(a))["credited"] == 0
    assert SP.timed_credit({**served, "passes": [{**p, "timed": True} for p in served["passes"]]}, work, published(a)) == {
        "passes": 2, "macs": work["checked_macs"], "credited": work["credited"]}
    # in the grid form the warm-up pass's MACs are its whole records', so the timed credit is at most the exact one
    passes = [VR.load(x, published(x)) for x in (a, b)]
    grid = VR.grid_work(VR.structural_credit(passes, SP.DRAWS, Fraction(1, 100)), work)
    whole = VR.pass_macs(VR.whole_pass(passes[0]))
    timed = SP.timed_credit(served, grid, published(a))
    assert whole > warm and VR.pass_macs(passes[0]) == warm and timed["macs"] == grid["checked_macs"] - whole
    assert timed["credited"] == grid["credited"] - math.ceil((1 - Fraction(grid["rho"])) * whole) <= got["credited"]
    assert 0 < timed["credited"] and grid["credited"] <= work["credited"] and grid["give_up"] > 0


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


def test_a_seeded_serve_draws_the_same_coins_and_no_salt_key_and_takes_no_retention_or_live_coins(tmp_path, monkeypatch):
    """`serve --seeded`'s streams (`served_passes.seeded`) give one label's bytes however they are sampled, so two seeded
    installs (`seed_install`) give the attached scheme's in-process challenger (`serving.in_process_coins`) one coin
    commitment, and seed nothing else: every leaf's salt is fresh, so there is no salt key to seed; ``--seeded`` is
    refused beside retention and beside the challenger gate, before anything is built."""
    from verity.protocols.accounting.compute.pouw.prover import serving as SV
    from verity.protocols.accounting.compute.pouw.verifier import verifier

    SP = load("pearl_c_vllm_served_passes", POUW / "pearl_c_vllm" / "served_passes.py")
    a, b = SP.seeded("coins"), SP.seeded("coins")
    assert a(3) + a(70) + a(1) == b(74) and SP.seeded("salts")(32) != SP.seeded("coins")(32)
    monkeypatch.setattr(verifier, "coins", verifier.coins)
    SP.seed_install()
    one = SV.in_process_coins(0).commitment.to_json()
    SP.seed_install()
    assert SV.in_process_coins(0).commitment.to_json() == one
    assert not hasattr(PPD, "new_salt_key") and not hasattr(PPD, "no_salt_key")
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
    SP = load("pearl_c_vllm_served_passes", POUW / "pearl_c_vllm" / "served_passes.py")
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
    from verity.protocols.accounting.compute.pouw.prover.serving.pearl_c.window import macs

    path = proved_pass(tmp_path, PROMOTED[0])[0]
    man = json.loads(path.read_text())
    p = VR.load(path, published(path))
    assert macs(str(path)) == sum(u["real_rows"] * u["n"] * u["k"] for u in man["matmuls"]) and macs(None) is None
    assert any(u["excluded_a"] for u in man["matmuls"]) and any(u["real_rows"] < u["m"] for u in man["matmuls"])
    assert VR.pass_macs(p) == VR.credit_at_assumption([p], 459, Fraction(1, 100))["checked_macs"] < macs(str(path))
    r = p.records[0]
    blocks = {j // PW.BAND for j in (*r.excluded_b, *r.voluntary_b)}
    b = min(j for j in range(r.matmul.shape.n) if j // PW.BAND not in blocks)
    block = sum(1 for j in range(r.matmul.shape.n) if j // PW.BAND == b // PW.BAND)
    more = dataclasses.replace(p, records=[dataclasses.replace(r, excluded_b=(*r.excluded_b, b)), *p.records[1:]])
    assert not r.voluntary_a and block > 1
    assert VR.pass_macs(more) == VR.pass_macs(p) - (r.real_rows - len(r.excluded_a)) * block * r.k_kept


def _split(got: dict) -> None:
    """Each call's real MACs are its checked MACs and its causes, and its checked MACs its three terms and its credit,
    exactly; the totals are the calls' sums, ``credited`` floored."""
    sums: dict = {}
    for c in got["calls"]:
        terms = [Fraction(c[x]) for x in ("epsilon_macs", "over_bound_macs", "rho_macs", "credited_macs")]
        assert c["macs"] == c["checked_macs"] + sum(c["uncredited_macs"].values()) and c["checked_macs"] == sum(terms), c
        assert set(c["uncredited_macs"]) == set(VR.UNCREDITED_ROWS) and min(terms) >= 0
        for x, v in (("macs", c["macs"]), ("checked_macs", c["checked_macs"]), *c["uncredited_macs"].items(),
                     *((x, Fraction(c[x])) for x in ("epsilon_macs", "over_bound_macs", "rho_macs", "credited_macs"))):
            sums[x] = sums.get(x, 0) + v
    assert got["totals"] == {**{x: str(v) if isinstance(v, Fraction) else v for x, v in sums.items()},
                             "credited": math.floor(sums["credited_macs"])}


def test_uncredited_calls_split_each_calls_macs_by_cause_and_sum_to_the_runs_credit(tmp_path):
    """`verify_run.uncredited_calls` splits each call's real MACs into its credited cells' (at |K|) and those the band rule
    leaves out, by cause: the fixture's excluded A row (its filler rows are no call's real MACs); an excluded B row's block
    of 64 columns, which takes the excluded A row's cells in it too; a voluntary A row's band, likewise.  Its terms sum to
    `credit_at_assumption`'s credit exactly, with or without samples over the inert-work bound, which only move part of the
    deduction to ``over_bound_macs``."""
    path = proved_pass(tmp_path, PROMOTED[0])[0]
    p = VR.load(path, published(path))
    for over in (0, 3):
        got = VR.uncredited_calls([p], 459, Fraction(1, 100), over)
        credit = VR.credit_at_assumption([p], 459, Fraction(1, 100), over)
        _split(got)
        assert (got["totals"]["checked_macs"], got["totals"]["credited"]) == (credit["checked_macs"], credit["credited"])
        assert (Fraction(got["totals"]["over_bound_macs"]) > 0) == (over > 0) and got["over_cap_draws"] == over
    calls = got["calls"]
    assert [c["call"] for c in calls] == [r.matmul.index for r in p.records]
    at = next(j for j, r in enumerate(p.records) if r.excluded_a)
    r, n = p.records[at], p.records[at].matmul.shape.n

    def changed(**kw) -> dict:
        got = VR.uncredited_calls([dataclasses.replace(p, records=[dataclasses.replace(x, **kw) if j == at else x
                                                                   for j, x in enumerate(p.records)])], 459, Fraction(1, 100))
        _split(got)
        return got["calls"][at]

    assert calls[at]["rows"]["excluded_a"] == len(r.excluded_a) and calls[at]["macs"] == r.real_rows * n * r.k_kept
    assert calls[at]["uncredited_macs"] == {"excluded_a": len(r.excluded_a) * n * r.k_kept, "excluded_b": 0,
                                            "voluntary": 0, "kept_width": 0}
    block = min(PW.BAND, n)
    assert changed(excluded_b=(0,))["uncredited_macs"] == {"excluded_a": len(r.excluded_a) * (n - block) * r.k_kept,
                                                           "excluded_b": r.real_rows * block * r.k_kept, "voluntary": 0,
                                                           "kept_width": 0}
    other = next(x for x in range(r.real_rows) if x not in r.excluded_a)
    band = sum(1 for x in range(r.real_rows) if x // PW.BAND == other // PW.BAND)
    inside = sum(1 for x in r.excluded_a if x // PW.BAND == other // PW.BAND)
    assert changed(voluntary_a=(other,))["uncredited_macs"] == {"excluded_a": (len(r.excluded_a) - inside) * n * r.k_kept,
                                                                "excluded_b": 0, "voluntary": band * n * r.k_kept,
                                                                "kept_width": 0}


def _uncredited_case(tmp_path: Path, monkeypatch, capsys):
    """An accepted verdict over one proved pass in ``tmp_path/w0``, a test instance's verdict over it, and ``run(doc,
    passes)``: ``served_passes.py uncredited`` on that document, its exit code and what it printed."""
    SP = load("pearl_c_vllm_served_passes", POUW / "pearl_c_vllm" / "served_passes.py")
    path = proved_pass(tmp_path / "w0" / "pass-00000", PROMOTED[0])[0]
    shutil.copy(path.parent / "coins.json", tmp_path / "w0" / "coins.json")
    coins, source = published(path), b"verifier" * 4
    test = VR.verify_passes([path], 3, 1, "test", coins, source, test_instance=True)
    monkeypatch.setattr(VR, "K_MIN", 32)
    monkeypatch.setattr(VR, "TILE", 16)
    doc = VR.verify_passes([path], 3, 1, "test", coins, source)
    verify, out = tmp_path / "verify.json", tmp_path / "uncredited.json"

    def run(d: dict, passes: Path = tmp_path / "w0") -> tuple[int, str]:
        verify.write_text(json.dumps(d))
        rc = SP.main(["uncredited", "--passes", str(passes), "--verify", str(verify), "--out", str(out)])
        return rc, capsys.readouterr().out

    return doc, test, VR.load(path, coins), out, run


@pytest.mark.slow
def test_an_accepted_run_verdict_says_where_its_macs_went_uncredited_and_served_passes_recomputes_it(tmp_path, monkeypatch, capsys):
    """An accepted verdict that credits carries `uncredited_calls` in its document (not in its answer); a test instance's
    credits nothing and carries none.  ``served_passes.py uncredited`` recomputes it from the passes in DIR, found by the
    document's SHA-256s, without sampling, and refuses a document that didn't accept, a DIR without its passes, passes of
    another commitment and a credit the calls don't sum to."""
    doc, test, held, out, run = _uncredited_case(tmp_path, monkeypatch, capsys)
    assert test["answer"]["verdict"] == "accept" and "uncredited" not in test
    work = doc["answer"]["public"]["work"]
    assert doc["answer"]["verdict"] == "accept" and work["credited"] > 0, doc["line"]
    assert doc["uncredited"] == VR.uncredited_calls([held], 3, VR.DELTA, work["over_cap_draws"])
    assert "uncredited" not in doc["answer"] and "calls" not in json.dumps(doc["answer"])
    _split(doc["uncredited"])
    rc, said = run(doc)
    got = json.loads(out.read_text())
    assert rc == 0 and got["uncredited"] == doc["uncredited"] and got["commitment"] == doc["commitment"], said
    assert got["records"] == [p["records"] for p in doc["answer"]["public"]["passes"]] and "the verdict's credit exactly" in said
    renamed = tmp_path / "wK"
    (tmp_path / "w0").rename(renamed)
    assert run(doc, renamed)[0] == 0
    for d, why in (({**doc, "answer": {**doc["answer"], "verdict": "reject"}}, "no accepted verdict"),
                   ({**doc, "test_instance": True}, "no accepted verdict"),
                   ({**doc, "commitment": "00" * 64}, "commitment is not the verify document's"),
                   ({**doc, "answer": {**doc["answer"], "public": {**doc["answer"]["public"],
                                                                     "work": {**work, "credited": work["credited"] + 1}}}},
                    "the verdict's credit is")):
        rc, said = run(d, renamed)
        assert rc == 1 and why in said and not out.exists(), (why, said)
    rc, said = run(doc, tmp_path)
    assert rc == 1 and "no pass in" in said and not out.exists()


@pytest.mark.slow
def test_served_passes_uncredited_reads_the_grid_form_of_a_fixed_load_run(tmp_path, monkeypatch, capsys):
    """A run held to its class's fixed load (`verify_run.grid_work`) publishes the passes' whole MACs (the checked MACs
    and the excluded and voluntary cells') and their credit less the give-up padded up to the execution phase grid.
    ``uncredited`` accepts it where the calls' credit lies between the published credit and that plus the give-up, the
    padding past the calls' credit being the cause ``grid``, and refuses calls whose credit is past the give-up's room
    or whose whole MACs are not the published ones."""
    doc, _, _, out, run = _uncredited_case(tmp_path, monkeypatch, capsys)
    t = doc["uncredited"]["totals"]
    whole = t["checked_macs"] + t["excluded_a"] + t["excluded_b"] + t["voluntary"]
    assert whole > t["checked_macs"] and t["credited"] > 3
    credited = t["credited"] - 3

    def held(give_up: int = 8, macs: int = whole) -> dict:
        work = {**doc["answer"]["public"]["work"], "checked_macs": macs, "credited": credited, "give_up": give_up}
        return {**doc, "answer": {**doc["answer"], "public": {**doc["answer"]["public"], "work": work}}}

    rc, said = run(held())
    assert rc == 0 and json.loads(out.read_text())["uncredited"] == doc["uncredited"], said
    assert "the verdict's grid form, 3 past its credit (grid)" in said
    assert run(held(give_up=3))[0] == 0
    for give_up, macs in ((2, whole), (8, whole + 1), (8, t["checked_macs"])):
        rc, said = run(held(give_up, macs))
        assert rc == 1 and not out.exists(), said
        assert (f"the calls credit {t['credited']:,} of {whole:,} whole MACs, and the verdict's grid form "
                f"{credited:,} of {macs:,} less a give-up of {give_up:,}") in said


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


@pytest.mark.slow
def test_a_salt_dead_element_puts_its_row_over_the_inert_bound(tmp_path, monkeypatch):
    """P3's forming term in the tile rule (`pearl_c_inert_work.inert_elements`): credited row 9 holds one entry 300 times its
    RMS, which keeps the per-row rules and is salt-dead, inert work of fs·|cols|/n = 64 over the row's bound of 41 at
    k = 128 with no flagged atom, so its tile, and it alone, fails; without the term every tile passes."""
    path = proved_pass(tmp_path, "pearl-c-h100-v1-h3", spike=300.0)[0]
    p = VR.load(path, published(path))
    assert _replay_every_tile(p) == {0: "A row 9's replayed debit 64 exceeds its cap 41 (the row cap: ρ·its credited "
                                        "cells' credit)"}
    monkeypatch.setattr(CW.D, "inert_elements", lambda flags, dead, device=None: 0)
    assert _replay_every_tile(p) == {}


@pytest.mark.slow
def test_volunteer_pass_gives_up_the_rows_over_their_inert_bounds_and_every_tile_then_passes(tmp_path):
    """V-EX's remedy on a retained pass before its sampling coin: under the h100 record's zero key a credited row is over its
    own cap (`proved_pass`), so the verifier rejects its tile.  `volunteer_pass` finds the same tiles from the words' counts,
    gives the row up, and rewrites the records and their root alone: the commitment, A's roots and the files are the pass's,
    every tile then passes the verifier's check, and a second run changes nothing."""
    VP = load("pearl_c_vllm_volunteer_pass", POUW / "pearl_c_vllm" / "volunteer_pass.py")
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


@pytest.mark.slow
def test_the_give_up_runs_exactly_while_the_verdict_rejects_a_drawn_credited_row_over_its_inert_bound(tmp_path, capsys):
    """The pipelines give the rows over a debit cap up before the coin exactly where ``served_passes.py volunteer-needed``
    says yes, so it says yes exactly where the verdict they call rejects a run on a sampled tile with a credited row over its
    cap: on a pass with such rows (`proved_pass` under the h100 record's zero key) and a coin that samples them, ``verify``
    (the run's verdict, `served_pearlplus.sh`'s) and `verify_run.verify` (a pass at a time, `glm53_passes.py`'s) reject iff
    it says yes, under their own rule and each ``--over-cap`` they take.  A verifier whose over-cap rule changes moves the
    give-up onto or off the prover's path with it; one that takes no ``over_cap`` refuses any."""
    SP = load("pearl_c_vllm_served_passes", POUW / "pearl_c_vllm" / "served_passes.py")
    path = proved_pass(tmp_path, "pearl-c-h100-v1-h3", key=bytes(32))[0]
    coins = published(path)
    p = VR.load(path, coins)
    over = _replay_every_tile(p)
    assert over and all("replayed debit" in r for r in over.values()), over
    exclusion = len(p.tile_ids(CW.exclusion_tiles(p.scheme, p.records)))
    d, source = _serve_dir(tmp_path, [path]), b"verifier" * 4
    (d / SP.COIN).write_text(json.dumps({"commitment": VR.run_commitment([p]).hex(), "source": source.hex()}))
    takes = {}
    for verdict, fn in (("run", VR.verify_passes), ("pass", VR.verify)):
        takes[verdict] = "over_cap" in inspect.signature(fn).parameters
        for rule in (None, *(SP.OVER_INERT_BOUND if takes[verdict] else ())):
            flag = () if rule is None else ("--over-cap", rule)
            capsys.readouterr()
            assert SP.main(["volunteer-needed", "--verdict", verdict, *flag]) == 0
            yes, took = capsys.readouterr().out.split()
            assert yes in ("yes", "no") and took in SP.OVER_INERT_BOUND and took == (rule or took)
            if verdict == "run":
                _, doc, answer = served_verify(d, tmp_path / "verify.json", "--draws", "32", *flag)
                drawn = {v["tile"] for v in doc["verdicts"][:len(doc["verdicts"]) - exclusion]}
                rejects = answer["verdict"] == "reject" and "replayed debit" in answer["reason"]
            else:
                r = VR.verify(path, 32, 1, "test", source, coins=coins, **({} if rule is None else {"over_cap": rule}))
                drawn, rejects = set(r["tiles"][:r["drawn"]]), not r["accepted"] and "replayed debit" in r["line"]
            assert drawn & set(over), (verdict, rule, sorted(drawn), sorted(over))
            assert (yes == "yes") == rejects, (
                f"volunteer-needed --verdict {verdict} says {yes} under over-cap rule {took}, but the verdict "
                f"{'rejects' if rejects else 'does not reject'} a run on a drawn row over its cap: served_passes.over_inert_bound_rule "
                "must name the rule the verdict takes")
    for verdict in (v for v, t in takes.items() if not t):
        capsys.readouterr()
        assert SP.main(["volunteer-needed", "--verdict", verdict, "--over-cap", "reject"]) == 2
        assert "takes no over_cap" in capsys.readouterr().err


class _StandInInertWork:
    """`debit.InertWork` without a card: `fast_tile.words_inert_work` (module `ft`) on the whole of A′ by B̃, with the words `bad`
    names in the first call marked non-finite and their counts wrong, and one added to the first call's atoms at `off`."""

    def __init__(self, ft, device, bad=(), off=None):
        self.ft, self.device, self.bad, self.off, self.calls = ft, device, set(bad), off, []

    def words(self, a, b):
        self.calls.append((a.shape, b.shape))
        atoms, zeros = (x.astype(np.uint16) for x in self.ft.words_inert_work(a.tolist(), b.tolist(), self.device))
        bad = np.zeros(atoms.shape, bool)
        if len(self.calls) == 1:
            for i, j in self.bad:
                bad[i, j], atoms[i, j] = True, atoms[i, j] + 7
            if self.off is not None:
                atoms[self.off] += 1
        return atoms, zeros, bad


def test_volunteer_pass_on_the_device_counts_every_tile_as_the_replay_and_gives_up_the_same_rows(tmp_path):
    """`volunteer_pass.words_device` on a stand-in for the card (`_StandInInertWork`): each call's A′ (its real rows) and each
    weight's B̃ formed once as `encode` forms a tile's, every word's counts cut into the tiles in `scheme.tiles`' order with
    filler rows zero, give every tile's counts as the CPU replay (`words`) gives them, so the pass gives up the same rows to
    the same records root.  A tile with a word the device marks non-finite takes the replay's counts, and a count unlike
    the replay's on a cross-checked tile stops the step before the records change."""
    VP = load("pearl_c_vllm_volunteer_pass", POUW / "pearl_c_vllm" / "volunteer_pass.py")
    path = proved_pass(tmp_path, "pearl-c-h100-v1-h3", key=bytes(32))[0]
    coins, before = published(path), path.read_text()
    p = VP.VR.load(path, coins)
    want = VP.words(p, 1)
    cubin = tmp_path / "debit_sm90.cubin"
    cubin.write_bytes(b"stand-in")
    # call 0's row 17 is in its second row tile (tile id 1), which no cross-check at every 2^20th tile replays
    stand = _StandInInertWork(VP.FT, p.scheme.device, bad=[(17, 3)])
    VP._DEBIT[str(cubin.resolve())] = stand
    VP._B_CODES.clear()
    try:
        got, on = VP.words_device(VP.VR.load(path, coins), 1, cubin, cross_check=1 << 20)
        assert got.keys() == want.keys()
        assert all(np.array_equal(got[t][0], want[t][0]) and np.array_equal(got[t][1], want[t][1]) for t in want)
        assert [a for a, _ in stand.calls] == [(r.real_rows, r.matmul.shape.k) for r in p.records]
        assert on["bad_tiles"] == 1 and on["replayed"] == 2 and on["cross_check"] == 1 << 20
        VP._DEBIT[str(cubin.resolve())] = _StandInInertWork(VP.FT, p.scheme.device, off=(0, 0))
        with pytest.raises(RuntimeError, match="differ from the CPU replay's on 1 of 1 tiles"):
            VP.volunteer(path, coins, 1, cubin, 1 << 20)
        assert path.read_text() == before
        VP._DEBIT[str(cubin.resolve())] = _StandInInertWork(VP.FT, p.scheme.device)
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


@pytest.mark.slow
def test_the_run_verdict_credits_no_given_up_row_and_refuses_a_give_up_past_v_ex_or_its_published_root(tmp_path):
    """A pass whose rows over a cap are given up (`volunteer_pass`) is accepted by the run's verdict, which credits no cell
    of a given-up A row's band of 64 rows or a given-up B row's block of 64 columns.  More given-up rows than V-EX allows (⌈m/64⌉ of a call's A), or records changed after the
    pass published their root, the verdict refuses before it takes its coin."""
    VP = load("pearl_c_vllm_volunteer_pass", POUW / "pearl_c_vllm" / "volunteer_pass.py")
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
    assert good["answer"]["public"]["work"]["checked_macs"] == kept < whole
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
    """Pearl+'s execution phase record W over passes `paths`, pass s its round s, as the tie writes it: a v2 record at W's
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
    return {"format": DM.REGISTRATION, "window": {"kind": "served", **({"version": 2} if version == 2 else {})},
            "registered": [*reg, *(weights[w] for w in sorted(weights) if version == 2),
                           {"value": "input-ids/0", "root": "00", "leaves": 1, "words": 1, "domain": "00"}]}


@pytest.mark.slow
def test_the_run_verdict_holds_its_window_record_to_each_calls_a_and_y_and_each_weights_b_root_and_domain(tmp_path):
    """``served_passes.py verify --tie W`` accepts a run whose execution phase record registers each call's A rows and y at the roots
    the verdict credits, under the domains it recomputes from the salt and the call's index, and each weight's B rows at
    the root the execution phase commits it under (PoUW's weights tree), and names W's digest in its answer.  A W whose root, domain
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
    assert rc == 0 and answer["public"]["tie"] == {"record": DD.digest(w), "tables": 6}, doc["line"]
    assert f"tied to W {DD.digest(w)[:16]} on its 6 tables" in doc["line"] and doc["tie_record"] == str(wfile)
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
             ({**w, "format": DM.REGISTRATION_V1}, f"W is no {DM.REGISTRATION} record")]
    for bad, why in cases:
        r = VR.verify_passes([path], 3, 1, "test", coins, b"verifier" * 4, window=bad, test_instance=True)
        assert r["answer"]["verdict"] == "reject" and why in r["answer"]["reason"] and "source" not in r, (why, r["line"])
    (lfile := tmp_path / "legacy.json").write_text(json.dumps(legacy))
    rc, doc, answer = served_verify(path.parent, tmp_path / "legacy-verify.json", "--manifest", str(path), "--tie", str(lfile))
    assert rc != 0 and answer["verdict"] == "reject" and VR.UNTIED_WEIGHTS in answer["reason"], doc["line"]
    untied = {"record": DD.digest(old), "tables": 4, "weights": VR.WEIGHTS_UNTIED}
    r = VR.verify_passes([path], 3, 1, "test", coins, b"verifier" * 4, window=old, test_instance=True)
    assert r["answer"]["verdict"] == "accept" and r["answer"]["public"]["tie"] == untied, r["line"]
    (ofile := tmp_path / "v1.json").write_text(json.dumps(old))
    rc, doc, answer = served_verify(path.parent, tmp_path / "v1-verify.json", "--manifest", str(path), "--tie", str(ofile))
    assert rc == 0 and answer["public"]["tie"] == untied, doc["line"]
    assert f"on its 4 tables, {VR.WEIGHTS_UNTIED}" in doc["line"]


@pytest.mark.slow
def test_retain_writes_w_over_the_retained_timed_rounds_and_the_tied_verdict_credits_them(tmp_path):
    """``retain --window-record W`` on a serve of a warm-up pass, a pass without calls and two timed passes: it drops the
    warm-up's round 0, retains rounds 1 and 2 (places among served.json's passes with calls), and writes W over those two
    under their numbers, so ``verify --tie W`` accepts the run tied to W's digest on its 8 tables and credits the timed
    passes alone.  The warm-up retained, W can't tie it (W registers none of round 0, and `pearl_c.window.write` refuses
    to); once the coin is flipped ``retain`` refuses to write W again.  With ``--previous-record FILE`` the W it writes is
    the same record chained after that one (its ``previous`` the record's digest, as a serve chains W after its first
    registration record, the weights'), which the tie holds to the same passes under W's own digest; the flag without
    ``--window-record`` is a usage error, and a record that can't be read leaves no W.  With ``--day-declaration FILE``
    (a serve's declaration.json) W's header names the file's day and declaration digest and is otherwise that record, tied
    to the same passes; the flag without ``--window-record`` is a usage error, and a file that states no day leaves no W."""
    SP = load("pearl_c_vllm_served_passes", POUW / "pearl_c_vllm" / "served_passes.py")
    warm, a, b = (proved_pass(tmp_path / name, PROMOTED[0], s)[0] for name, s in (("w", 0), ("a", 1), ("b", 2)))
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
    first, chained = tmp_path / "first.json", tmp_path / "w-chained.json"
    first.write_text(json.dumps({"format": DM.REGISTRATION, "protocol": DM.ONE_STAGE_PROTOCOL, "registered": [], "previous": None}))
    assert SP.main(["retain", "--passes", str(d), "--window-record", str(chained), "--previous-record", str(first)]) == 0
    wc = json.loads(chained.read_text())
    assert w["previous"] is None and wc == {**w, "previous": DD.digest(json.loads(first.read_text()))}
    assert VR.tie(wc, [VR.load(x, published(x)) for x in (a, b)], [1, 2], (0,)) == {"record": DD.digest(wc), "tables": 10}
    assert DD.digest(wc) != DD.digest(w)
    with pytest.raises(SystemExit):
        SP.main(["retain", "--passes", str(d), "--previous-record", str(first)])
    assert SP.main(["retain", "--passes", str(d), "--window-record", str(tmp_path / "none.json"),
                    "--previous-record", str(tmp_path / "absent.json")]) == 1 and not (tmp_path / "none.json").exists()
    declared, named = tmp_path / "declaration.json", tmp_path / "w-named.json"
    declared.write_text(json.dumps({"day": 3, "declaration": "5d" * 64, "program": "ab" * 64, "blocks": [], "head": "cd" * 64}))
    assert SP.main(["retain", "--passes", str(d), "--window-record", str(named), "--previous-record", str(first),
                    "--day-declaration", str(declared)]) == 0
    wn = json.loads(named.read_text())
    assert wn == {**wc, "window": {**wc["window"], "day": 3, "declaration": "5d" * 64}}
    assert VR.tie(wn, [VR.load(x, published(x)) for x in (a, b)], [1, 2], (0,)) == {"record": DD.digest(wn), "tables": 10}
    with pytest.raises(SystemExit):
        SP.main(["retain", "--passes", str(d), "--day-declaration", str(declared)])
    declared.write_text(json.dumps({"day": 3}))
    assert SP.main(["retain", "--passes", str(d), "--window-record", str(tmp_path / "none.json"),
                    "--day-declaration", str(declared)]) == 1 and not (tmp_path / "none.json").exists()
    rc, doc, answer = served_verify(d, tmp_path / "verify.json", "--tie", str(wfile))
    assert rc == 0 and answer["public"]["tie"] == {"record": DD.digest(w), "tables": 10}, doc["line"]
    served = json.loads((d / "served.json").read_text())
    timed = SP.timed_credit(served, answer["public"]["work"], published(a), answer["public"]["rounds"])
    assert (answer["public"]["rounds"], answer["public"]["dropped"], timed["passes"]) == ([1, 2], [0], 2)
    before = wfile.read_text()
    assert SP.main(["retain", "--passes", str(d), "--window-record", str(wfile)]) == 1 and wfile.read_text() == before


@pytest.mark.slow
def test_served_passes_volunteer_gives_up_each_retained_passs_rows_over_an_inert_bound_and_refuses_an_unretained_serve(tmp_path):
    """``served_passes.py volunteer``, the prover's step between the serve and the verifier's coin, runs `volunteer_pass` on
    each pass with calls that served.json names: a pass with a credited row over its cap (`proved_pass` under the h100
    record's zero key) is left with none and its records' root is rewritten, so every tile then passes the verifier's check;
    its document names the node it ran on; a serve that retained no pass is refused."""
    SP = load("pearl_c_vllm_served_passes", POUW / "pearl_c_vllm" / "served_passes.py")
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
    """``served_passes.py volunteer --device`` (a stand-in for the card, `_StandInInertWork`) gives up the rows the CPU replay
    gives up, to the same records root, and its document names the debit cubin, the cross-check rate (``--device-check``,
    else `volunteer_pass.CROSS_CHECK`) and the tiles it replayed on the CPU too; the CPU's document has no ``device``."""
    SP = load("pearl_c_vllm_served_passes", POUW / "pearl_c_vllm" / "served_passes.py")
    VP = load("volunteer_pass", POUW / "pearl_c_vllm" / "volunteer_pass.py")
    path = proved_pass(tmp_path, "pearl-c-h100-v1-h3", key=bytes(32))[0]
    d, out, before = _serve_dir(tmp_path, [path]), tmp_path / "volunteer.json", path.read_text()
    cubin = tmp_path / "debit_sm90.cubin"
    cubin.write_bytes(b"stand-in")
    device = VP.VR.load(path, published(path)).scheme.device
    got = {}
    try:
        for check in ("2", None):
            path.write_text(before)
            VP._DEBIT[str(cubin.resolve())] = _StandInInertWork(VP.FT, device)
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


@pytest.mark.slow
def test_served_passes_verify_credits_only_the_passes_retained_before_its_coin_and_refuses_a_retention_after_it(tmp_path):
    """``retain`` drops passes before the coin (one ``--drop``, and with ``--volunteer`` each pass that step didn't finish or
    left over a cap): ``verify`` then reads the retained passes alone, each at its round, credits only them, and records
    its coin against their commitment (``coin.json``).  The same retention again takes the same source, never another;
    ``retain`` and ``volunteer`` refuse once the coin is flipped, and a retention changed behind them is refused
    (``after-coin``) before any replay, whichever way it changed."""
    SP = load("pearl_c_vllm_served_passes", POUW / "pearl_c_vllm" / "served_passes.py")
    a, b = (proved_pass(tmp_path / name, PROMOTED[0], s)[0] for name, s in (("a", 0), ("b", 1)))
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
    served = json.loads((d / "served.json").read_text())
    assert SP.timed_credit(served, work, published(b), [1]) == {"passes": 1, "macs": one["checked_macs"], "credited": 0}
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


@pytest.mark.slow
def test_the_run_verdict_refuses_a_give_up_made_after_its_coin(tmp_path):
    """V-EX's give-up is the prover's before the coin: a pass whose records change after the verifier recorded its coin
    (`volunteer_pass` run past ``served_passes.py volunteer``'s refusal) is another run, and the verdict refuses it
    (``after-coin``) rather than sample again over it."""
    VP = load("pearl_c_vllm_volunteer_pass", POUW / "pearl_c_vllm" / "volunteer_pass.py")
    path = proved_pass(tmp_path, "pearl-c-h100-v1-h3", key=bytes(32))[0]
    coins, record = published(path), tmp_path / "coin.json"
    first = VR.verify_passes([path], 3, 1, "test", coins, record=record, test_instance=True)
    assert "source" in first and json.loads(record.read_text())["commitment"] == first["commitment"]
    assert VP.volunteer(path, coins, 1)["ok"]
    late = VR.verify_passes([path], 3, 1, "test", coins, record=record, test_instance=True)
    assert late["answer"]["verdict"] == "reject" and late["refused"][0] == "after-coin" and "source" not in late, late["line"]
    assert VR.verify_passes([path], 3, 1, "test", coins, test_instance=True)["answer"]["verdict"] == "accept"


@pytest.mark.slow
def test_the_run_verdict_takes_its_source_from_its_draw_once_and_never_asks_a_draw_again(tmp_path):
    """``verify_passes``' ``draw(C, asking)``: its source is the sample's coin on the run's commitment C, recorded (v2) with
    the sample before any replay and reused by a later verdict without sampling; a sample that fails before it asks rejects
    and may be asked again, while one that fails after ``asking`` marked the record is refused (``asked-once``) and
    never redrawn."""
    VP = load("pearl_c_vllm_volunteer_pass", POUW / "pearl_c_vllm" / "volunteer_pass.py")
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
    lane's open, it reserves a sample session withheld (``reserve --peer P --scheme S --families [["pouw/draw",1]]
    --withhold``), marks the record with the reserve before its open, and returns the session's ``pouw/draw`` coin 0
    after C with the record's sample.  It refuses before reserving while the lane's period is open, and before its open
    when the gate reserves it plain or into a period; `live_draw` is None with neither setting and refused with one."""
    from verity.protocols.accounting.compute.pouw.prover.serving.coins import challenger as PHC
    from verity.protocols.accounting.compute.pouw.prover.serving.coins import window as PHW

    BA = load("pearl_c_vllm_bench_arm", POUW / "pearl_c_vllm" / "bench_arm.py")
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
    """Given the Pearl-C period P the run's salt session joined (``served.json``'s ``coins.reserved.period``), the sample
    is lane vllm's first reserve after P's close, the one a compose takes (`pouw_h1_window.after_close`): it is refused
    before reserving while the ledger doesn't close P or the lane (``for``, else ``by``) reserved since, and before its
    coin is asked when another reserve of the lane reached the gate first; another lane's reserve doesn't count."""
    from verity.protocols.accounting.compute.pouw.prover.serving.coins import window as PHW

    BA = load("pearl_c_vllm_bench_arm", POUW / "pearl_c_vllm" / "bench_arm.py")
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


@pytest.mark.slow
def test_a_tie_over_a_retention_holds_each_retained_pass_at_its_round_and_passes_over_the_dropped_rounds(tmp_path):
    """With round 0 dropped before the coin, W over both rounds ties the retained pass at its round 1, its dropped round's
    entries crediting nothing; without the drop those entries are a call the run doesn't hold, and at round 0 the pass's
    calls are not at their places (`passes_placed`)."""
    a, b = (proved_pass(tmp_path / name, PROMOTED[0], s)[0] for name, s in (("a", 0), ("b", 1)))
    coins, w = published(b), window_record(a, b)
    r = VR.verify_passes([b], 3, 1, "test", coins, b"verifier" * 4, window=w, rounds=[1], dropped=(0,), test_instance=True)
    assert r["answer"]["verdict"] == "accept" and r["answer"]["public"]["tie"]["tables"] == 6, r["line"]
    for rounds, dropped, why in (([1], (), "W registers pearl-a/0/0, no call of the run's"),
                                 ([0], (1,), "call-place: pass 0's call 0 holds index 65536, not its place 0")):
        r = VR.verify_passes([b], 3, 1, "test", coins, b"verifier" * 4, window=w, rounds=rounds, dropped=dropped, test_instance=True)
        assert r["answer"]["verdict"] == "reject" and why in r["answer"]["reason"] and "source" not in r, (why, r["line"])


def test_a_window_class_that_is_no_three_positive_integers_is_refused_by_name():
    """``--class`` (`served_passes.window_class`) reads SLOTS,PROMPT,OUTPUTS, and refuses a word that is no integer, a
    count other than three and a zero alike, each naming the text it read and raising from what failed to parse."""
    import argparse
    SP = load("pearl_c_vllm_served_passes", POUW / "pearl_c_vllm" / "served_passes.py")
    assert SP.window_class("4,32,4") == (4, 32, 4)
    for text in ("4,x,4", "4,32", "4,0,4", ""):
        with pytest.raises(argparse.ArgumentTypeError, match="a window class is SLOTS,PROMPT,OUTPUTS") as e:
            SP.window_class(text)
        assert repr(text) in str(e.value)
        assert isinstance(e.value.__cause__, ValueError) == (text in ("4,x,4", ""))


def test_the_load_check_takes_the_run_of_records_shape():
    """The class of record's run (class 4, 32, 4 over Llama 3.2 1B's 64 cut linears) is its fixed
    load, so the check accepts its shape: 64 calls a pass, round 0's of 4 x 32 = 128 rows in 128, each later round's of 4
    in 16 (`window_class.PearlTables`' ``rows`` and ``a_rows``), also with a round dropped; and no other."""
    NS = types.SimpleNamespace

    def passes(rows):
        return [NS(records=[NS(real_rows=real, matmul=NS(shape=NS(m=m))) for _ in range(64)]) for real, m in rows]
    cls, shape = (4, 32, 4), [(128, 128), (4, 16), (4, 16), (4, 16)]
    assert [VR.load_rows(cls, s) for s in range(4)] == shape
    assert VR.held_to_load(cls, passes(shape)) and VR.held_to_load(cls, passes(shape), [0, 1, 2, 3])
    assert VR.held_to_load(cls, passes(shape[:3]), [0, 1, 2], (3,))
    assert VR.held_to_load(cls, passes(shape[1:]), [1, 2, 3], (0,))
    for rows, rounds, dropped in ((shape[:3], None, ()), (shape + [(4, 16)], None, ()), (shape[::-1], None, ()),
                                  (shape, [1, 2, 3, 4], (0,)), (shape, [0, 1, 2, 2], ()), (shape, [0, 1, 2], ()),
                                  ([(128, 128), (4, 16), (3, 16), (4, 16)], None, ()),
                                  ([(128, 128), (4, 16), (4, 64), (4, 16)], None, ()),
                                  ([(127, 128), (4, 16), (4, 16), (4, 16)], None, ())):
        assert not VR.held_to_load(cls, passes(rows), rounds, dropped), (rows, rounds, dropped)
    odd = passes(shape)
    odd[2].records[40].real_rows = 5
    assert not VR.held_to_load(cls, odd)


def class_run(excluded: tuple[int, int, int, int] = (0, 0, 0, 0)) -> list:
    """The records of class (4, 32, 4)'s fixed load on Llama 3.2 1B, as passes `credit_at_assumption` reads: sixteen
    layers of four linears (k, n), q, k and v as one (2048, 3072), o (2048, 2048), gate and up as one (2048, 16384) and
    down (8192, 2048), round 0's calls of 128 rows and each later round's of 4 in 16, on the served scheme.  Round 0's
    first call of each of the four shapes excludes ``excluded``'s rows of that shape."""
    shapes, scheme = [(2048, 3072), (2048, 2048), (2048, 16384), (8192, 2048)], pouw_schemes.get("pearl-c-h100-v1-h3")

    def records(real: int, m: int, ex: tuple[int, ...]) -> list:
        return [PW.UnitRecord(Matmul(c, Shape(m, *shapes[c % 4]), c), real, shapes[c % 4][0],
                              tuple(range(ex[c] if c < 4 else 0))) for c in range(64)]
    return [types.SimpleNamespace(scheme=scheme, records=records(real, m, ex))
            for (real, m), ex in zip([(128, 128), (4, 16), (4, 16), (4, 16)], [excluded, *[(0, 0, 0, 0)] * 3])]


def test_a_class_held_runs_work_is_its_structural_credit_less_the_give_up_on_the_grid():
    """A run held to its class's fixed load publishes its work in the grid form (`grid_work`): the structural credit, MACs
    and credited tiles (`structural_credit`: the credit with no row excluded or given up, the class's and the model's),
    the credit less the give-up (structural less exact) padded up to H1's execution phase grid (`window.pad`).  So it is at most
    the exact credit, whatever the passes exclude, and two give-ups of one grid level publish one work.

    The run of record's shape and credit (class 4, 32, 4 on Llama 3.2 1B, 459 samples, none over
    the inert-work bound): its published exact credit, 134,194,399,455 MACs of 135,895,449,600 declared, is this load's
    with 19 rows excluded, 4 of an o, 4 of a gate-up and 11 of a down, and its grid form is 4,196,341 MACs less.  The
    Lean holds the same numbers (`PouwPearlPlus/Grid.lean`'s guard)."""
    t, d = 459, Fraction(1, 100)
    of_record = class_run((0, 4, 4, 11))
    assert VR.held_to_load((4, 32, 4), of_record)
    exact, whole = VR.credit_at_assumption(of_record, t, d), VR.structural_credit(of_record, t, d)
    assert (exact["credited"], exact["checked_macs"]) == (134_194_399_455, 135_895_449_600)
    assert whole == VR.credit_at_assumption(class_run(), t, d)
    assert (whole["credited"], whole["checked_macs"]) == (134_525_747_434, 136_230_993_920)
    assert [r.excluded_a for r in of_record[0].records[:4]] == [(), (0, 1, 2, 3), (0, 1, 2, 3), tuple(range(11))]
    work = VR.grid_work(whole, exact)
    assert whole["credited"] - exact["credited"] == 331_347_979 and work["give_up"] == Wd.pad(331_347_979) == 335_544_320
    assert work == {**whole, "credited": 134_190_203_114, "give_up": 335_544_320}
    assert work["credited"] == whole["credited"] - work["give_up"] and exact["credited"] - work["credited"] == 4_196_341
    assert set(work) - set(exact) == {"give_up"} and set(VR.EXACT_WORK) <= set(exact)

    for ex in ((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1), (3, 5, 0, 2), (0, 4, 4, 10), (0, 4, 4, 12), (47, 47, 47, 47),
               (128, 128, 128, 128)):
        e = VR.credit_at_assumption(class_run(ex), t, d)
        g, gave = VR.grid_work(whole, e), whole["credited"] - e["credited"]
        assert gave >= 0 and g["give_up"] == (Wd.pad(gave) if gave else 0) and (not gave or Wd.on_grid(g["give_up"]))
        assert g["credited"] == whole["credited"] - g["give_up"] <= e["credited"], ex
        assert {k: g[k] for k in e if k != "credited"} == {k: whole[k] for k in e if k != "credited"}
        assert VR.structural_credit(class_run(ex), t, d) == whole
    assert VR.grid_work(whole, whole) == {**whole, "give_up": 0}
    assert VR.grid_work(whole, VR.credit_at_assumption(class_run((0, 4, 4, 10)), t, d)) == work
    assert VR.grid_work(whole, VR.credit_at_assumption(class_run((1, 0, 0, 0)), t, d)) != work

    over = VR.structural_credit(of_record, t, d, 2)
    assert over["epsilon"] == VR.credit_at_assumption(of_record, t, d, 2)["epsilon"] != whole["epsilon"]
    assert over["over_cap_draws"] == 2 and over["credited"] < whole["credited"] and over["checked_macs"] == whole["checked_macs"]
    # the form's own floor: a give-up padded past the structural credit credits nothing, never below zero
    assert VR.grid_work({"credited": 17}, {"credited": 0}) == {"credited": 0, "give_up": 18}
    assert VR.grid_work({"credited": 5}, {"credited": 7}) == {"credited": 5, "give_up": 0}
    whole_record = VR.whole(PW.UnitRecord(Matmul(0, Shape(128, 2048, 2048), 0), 100, 2048, (3,), (5,), (70,), (9,)))
    assert whole_record == PW.UnitRecord(Matmul(0, Shape(128, 2048, 2048), 0), 100, 2048)


@pytest.mark.slow
def test_a_tie_refuses_two_passes_that_commit_one_weight_under_two_roots(tmp_path):
    """W registers each weight's B rows once, so a tie over passes whose execution phases commit weight 0 under two roots (other
    weights, seed 12) has no one entry to hold them to.  `verify_passes` never gets here (one salt is one weights root,
    checked before the tie), so the tie is called directly."""
    a, b = proved_pass(tmp_path / "a", PROMOTED[0])[0], proved_pass(tmp_path / "b", PROMOTED[0], 1, seed=12)[0]
    passes = [VR.load(x, published(x)) for x in (a, b)]
    assert passes[0].epoch.weight_roots[0] != passes[1].epoch.weight_roots[0]
    with pytest.raises(ValueError, match="pass 1 commits weight 0's B rows under another root or shape than an earlier"):
        VR.tie(window_record(a, b), passes)


#: Llama-3.2-1B's cut linears as vLLM serves them: its layers, and a layer's four in its forward's order, each (n, k):
#: qkv_proj, o_proj, gate_up_proj and down_proj (hidden 2,048, intermediate 8,192, q's 2,048 rows with k's and v's 512 each)
LLAMA32_1B = (16, ((3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192)))
#: the execution phase class of the run of record: its slots, prompt ids and outputs
#: (`window_class.WindowClass`), so 4 rounds under its fixed load
RECORD_CLASS = (4, 32, 4)


def timed(f):
    """``f()`` and the seconds it took, for a test that reports a time and asserts none."""
    start = time.perf_counter()
    out = f()
    return out, time.perf_counter() - start


def record_size_passes(tmp_path: Path, seed: int = 1) -> list[Path]:
    """The manifests of a served run of the run of record's size, one a round of class `RECORD_CLASS`'s fixed load
    (`window_class.PearlTables`: round 0 every slot's prompt, 128 rows, and each later round a row a slot, 4 rows of the 16
    its A commits), each calling `LLAMA32_1B`'s 64 cut linears at their real (n, k), weight 4·layer + linear and call
    index its place (`protocol.call_place`, round and call): 4 passes, 256 calls, 64 weights, the served instance of the run of record's scheme (`H100_SCHEME`, 64 x 64
    tiles), one execution phase under one salt coin, no row excluded.

    Each root is an hm96 root over seeded fake row digests and salts, not over rows: the tree of the real number of leaves
    (a weight's n, a call's m16 or real rows) under the real domain, each leaf `Hm96Domain`'s of a 64-byte digest no row
    was hashed to and 192 seeded bytes in place of its salt. The weights' real rows are 3.9 GB of FP32 words, and their
    salts 192 bytes a row more. The transcript root is 64 seeded bytes, and no row, tree, salts or leaf file is written
    (the manifests name them as a device run does), so the passes load and tie, and nothing of them opens."""
    layers, linears = LLAMA32_1B
    slots, prompt, outputs = RECORD_CLASS
    s, rng = pouw_schemes.get(H100_SCHEME), np.random.default_rng(seed)
    shapes = {len(linears) * layer + j: nk for layer in range(layers) for j, nk in enumerate(linears)}

    def root(binding: bytes, owner: int, count: int) -> str:
        d, blob = rows_domain(binding, owner, count, HM96, digests=True), rng.bytes(256 * count)
        return tree_levels(d, [d.leaf(i, i, ROW_SCHEMA, blob[256 * i:256 * i + 256]) for i in range(count)])[-1][0].hex()

    def named(path: str, size: int) -> dict:
        return {"path": path, "bytes": size}

    def tree(path: str, count: int) -> dict:
        return named(path, 64 * (2 * VR.PO.pow2(count) - 1))

    roots = {w: root(weight_binding(w), -2, n) for w, (n, _) in shapes.items()}
    weights = {str(w): {"n": n, "k": k, "root": roots[w], "excluded_b": [],
                        "rows": named(f"weights/{w}.rows.bf16", 2 * n * k), "tree": tree(f"weights/{w}.tree", n),
                        "salts": named(f"weights/{w}.salts", HM96_SALT * n)}
               for w, (n, k) in shapes.items()}
    committed = {w: bytes.fromhex(r) for w, r in roots.items()}
    coins, paths = salt_coin(committed), []
    for r in range(outputs):
        real = slots * prompt if r == 0 else slots
        m = -(-real // PPD.ROW_ALIGN) * PPD.ROW_ALIGN
        work = Workload(tuple(Matmul(call_place(r, c), Shape(m, k, n), c) for c, (n, k) in shapes.items()))
        epoch = Epoch.start(work, committed, *coins)
        splits = {w: PW.Split(k) for w, (_, k) in shapes.items()}
        records = [PW.UnitRecord(u, real, u.shape.k) for u in work.matmuls]
        matmuls = [{"index": u.index, "m": m, "k": u.shape.k, "n": u.shape.n, "weight": u.weight, "real_rows": real,
                    "k_kept": u.shape.k, "excluded_a": [],
                    "root_a": root(activation_binding(epoch.salt, u.index), u.index, m),
                    "rows": named(f"calls/{u.index}.rows.bf16", 2 * m * u.shape.k), "tree": tree(f"calls/{u.index}.tree", m),
                    "salts": named(f"calls/{u.index}.salts", HM96_SALT * m),
                    "y": {"root": root(SY.y_binding(epoch.salt, u.index), u.index, real),
                          "rows": named(f"y/{u.index}.rows.bf16", 2 * real * u.shape.n), "tree": tree(f"y/{u.index}.tree", real),
                          "salts": named(f"y/{u.index}.salts", HM96_SALT * real)}}
                   for u in work.matmuls]
        count = sum(len(s.tiles(u.shape)) for u in work.matmuls)
        head = {"schema": CC.MANIFEST_SCHEMA, "run": RUN, "entry": f"pouw:{s.name}", "scheme": s.name,
                "coins": coins[0].to_json(), "salt_opening": coins[1].to_json(), "salt": epoch.salt.hex(),
                "records_root": PW.records_root(epoch.salt, epoch.weights_root, splits, records).hex(),
                "commitment": {"root": rng.bytes(64).hex(), "count": count},
                "leaves": named("leaves", 64 * count), "tree_t": tree("tree_t", count),
                "salts_t": named("salts_t", HM96_SALT * count)}
        (d := tmp_path / f"pass-{r:05d}").mkdir(parents=True)
        paths.append(PPDR.write_manifest(d, head, weights, matmuls))
        (d / "coins.json").write_text(json.dumps(coins[0].to_json()))
    return paths


@pytest.mark.slow
def test_the_version_2_tie_holds_a_run_of_the_run_of_records_size_to_its_64_weights(tmp_path, capsys):
    """At the run of record's size (`record_size_passes`: class (4, 32, 4), 4 passes of Llama-3.2-1B's 64 cut linears,
    256 calls, each weight at its real (n, k)), the producer's W at version 2 (`pearl_c.window.record`) registers 576
    values: the 512 the run of record's version-1 W did, then each of the 64 weights' B rows once, at its n rows of K
    words. `tie` accepts it on its 576 tables with every weight tied, each ``pearl-b/<w>`` the root its passes' execution phases
    commit it under, and the run verdict with no sample (the passes retain no file to open) accepts tied to it. Refused, each
    by name, through the tie and the run verdict before its coin: a W whose ``pearl-b/37`` is at another root than the
    passes' rootB of weight 37, one that registers no ``pearl-b/37``, and one that registers no weight at all
    (`UNTIED_WEIGHTS`); and through the tie alone (the run verdict refuses passes of two salts first), a pass whose rootB
    of weight 37 is another than the other passes'.

    Prints W's size and the tie's seconds at this size, and asserts neither. Slow on its trees alone (about 400,000 hm96
    leaves in Python): the tie's own cost is what a cheaper test can't show at the size the verdict of record reads."""
    paths = record_size_passes(tmp_path)
    mans, coins = [json.loads(p.read_text()) for p in paths], published(paths[0])
    w = PWR.record(mans, version=2)
    names = [e["value"] for e in w["registered"]]
    assert PWR.version_of(w) == 2 and len(names) == 576 and len(set(names)) == 576
    assert names[:512] == [f"pearl-{x}/{s}/{c}" for s in range(4) for c in range(64) for x in "ay"]
    assert names[512:] == [f"pearl-b/{x}" for x in range(64)]
    assert PWR.record(mans, version=1)["registered"] == w["registered"][:512]
    layers, linears = LLAMA32_1B
    assert [(e["leaves"], e["words"], e["word_bits"]) for e in w["registered"][512:]] == [(n, k, 32) for n, k in linears] * layers
    assert [(e["leaves"], e["words"]) for e in w["registered"][:8]] == [x for n, k in linears for x in ((128, k), (128, n))]
    assert [(e["leaves"], e["words"]) for e in w["registered"][128:136]] == [x for n, k in linears for x in ((16, k), (4, n))]

    passes = [VR.load(p, coins) for p in paths]
    assert sum(len(p.records) for p in passes) == 256 and all(len(p.epoch.workload.weights()) == 64 for p in passes)
    tied, seconds = timed(lambda: VR.tie(w, passes))
    assert tied == {"record": DD.digest(w), "tables": 576}
    want = VR.tie_entries(passes)
    assert list(want) == names
    for x in range(64):
        e = next(e for e in w["registered"] if e["value"] == f"pearl-b/{x}")
        assert all(p.epoch.weight_roots[x].hex() == e["root"] == mans[0]["weights"][str(x)]["root"] for p in passes)
        assert want[f"pearl-b/{x}"] == {k: e[k] for k in ("root", "leaves", "words", "word_bits", "domain")}
    assert len({e["root"] for e in w["registered"]}) == 576
    r = VR.verify_passes(paths, 0, 1, "test", coins, b"verifier" * 4, window=w)
    assert r["answer"]["verdict"] == "accept" and r["answer"]["public"]["tie"] == tied, r["line"]
    assert "256 of 256 calls, tied to W" in r["line"] and "on its 576 tables;" in r["line"]

    other = next(e["root"] for e in w["registered"] if e["value"] == "pearl-b/38")
    cases = [({**w, "registered": [{**e, "root": other} if e["value"] == "pearl-b/37" else e for e in w["registered"]]},
              "W's pearl-b/37: its root not the weight's"),
             ({**w, "registered": [e for e in w["registered"] if e["value"] != "pearl-b/37"]}, "W registers no pearl-b/37"),
             ({**w, "registered": w["registered"][:512]}, f"{VR.UNTIED_WEIGHTS}, so the run's rootB is no W's (pearl-b/0 first)")]
    for bad, why in cases:
        with pytest.raises(ValueError) as refused:
            VR.tie(bad, passes)
        assert str(refused.value) == why
        r = VR.verify_passes(paths, 0, 1, "test", coins, b"verifier" * 4, window=bad)
        assert (r["answer"]["verdict"], r["answer"]["reason"]) == ("reject", f"ValueError: {why}") and "source" not in r
    p = passes[2]
    another = dataclasses.replace(p, epoch=dataclasses.replace(p.epoch, weight_roots={**p.epoch.weight_roots,
                                                                                    37: bytes.fromhex(other)}))
    with pytest.raises(ValueError) as refused:
        VR.tie(w, [*passes[:2], another, passes[3]])
    assert str(refused.value) == ("pass 2 commits weight 37's B rows under another root or shape than an earlier pass, and W "
                                  "registers one")

    written = len(json.dumps(w, indent=1)) + 1
    with capsys.disabled():
        print(f"\nW v2 at the run of record's size: {len(names)} values (512 of 256 calls, 64 weights' B rows), {written:,} bytes as "
              f"`window.write` writes it ({len(json.dumps(w, separators=(',', ':'))):,} compact), "
              f"{written - len(json.dumps(PWR.record(mans, version=1), indent=1)) - 1:,} more than its v1; verify_run.tie on "
              f"4 passes {seconds * 1e3:.1f} ms")


def test_volunteer_words_gives_up_one_a_row_a_band_and_a_b_row_once_the_a_count_is_spent():
    """Of two credited rows of one band over their row caps, the one further over is given up as an A row, and the band
    rule leaves the other no credit, so it needs no B row.  With V-EX's ⌈m/64⌉ = 2 A rows of a 128-row unit already
    spent, the row furthest over gives up the B row holding its flagged atom-word, whose block of 64 columns is the
    weight's every column, and no tile is left over."""
    s = pouw_schemes.get("pearl-c-h100-v1-h3", k_min=32, tile=(16, 16))
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
        for _, r, _, row_cell, tiles in units if CW.row_inert_bound(scheme) else ():
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
        for _, r, _, row_cell, tiles in units if CW.row_inert_bound(scheme) else ():
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


def test_volunteer_words_ranks_a_units_rows_over_their_inert_bounds_in_one_scan():
    """`volunteer_words` gives up a unit's A rows over their caps from one scan of its tiles, one a band, since an A row
    given up takes its band out and leaves every other band's rows' sums and caps as they were: on seeded counts with more rows over their caps than ⌈m/64⌉ allows, filler
    rows, and B rows over theirs, across three units, it gives up the A and B rows the one-row-at-a-time statement does."""
    s = pouw_schemes.get("pearl-c-h100-v1-h3", k_min=32, tile=(16, 16))
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


def test_verify_run_refuses_a_v1_manifest(honest_pass):
    """A v1 manifest (a pass recorded before the execution phase's coins) is refused, as the Lean `PouwVerifier.PearlPlus`
    refuses it, and so is a manifest of another schema; there is no ``--recorded`` reading of one."""
    path, *_ = honest_pass
    man = json.loads(path.read_text())
    v1 = path.parent / "manifest-v1.json"
    for schema, why in ((CC.MANIFEST_RECORDED, "recorded before epoch coins"),
                        ("verity-vllm/pouw-pearl-c-run/v0", f"not {CC.MANIFEST_SCHEMA}")):
        v1.write_text(json.dumps({**man, "schema": schema}))
        r = VR.verify(v1, 3, 1, "test", b"verifier" * 4, coins=published(path))
        assert not r["accepted"] and r["line"].startswith("REJECT") and why in r["line"], (schema, r["line"])
    v1.unlink()
    with pytest.raises(SystemExit):
        VR.main([str(path), *coins_args(path), "--recorded"])


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
    `after`, from a seeded in-process live challenger (the auditor device's session as `pouw_h1_window.open_session`
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
        return PPDR.PearlCDeviceRun(pouw_schemes.get(SCHEME), f"pouw:{SCHEME}", kernel, weights, roots, c, o,
                                   capacity=1 << 12, weight_salts=salts_b(weights), pass_index=k)
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


@pytest.mark.slow
def test_a_runs_coins_file_is_a_live_sessions_commitment_or_an_in_process_challengers(tmp_path):
    """`verify_run.epoch_coins`, the coins file every step of a served run reads (``served_passes.py volunteer`` and
    ``verify``, ``verify_run.py --coins``, ``volunteer_pass.py --coins``): a live session's commitment when its scheme is
    `window.LIVE_SCHEME`, under which a pass salted by that session restarts its execution phase, else an in-process challenger's;
    a pass restarts under its own execution phase's commitment only."""
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
    """A served run whose execution phase's salt coin is a live session's (``serve`` with the challenger gate): ``verify`` reads its
    coins file as that session's commitment and accepts its honest pass."""
    path = proved_pass(tmp_path, PROMOTED[0], live=0)[0]
    (path.parent / "served.json").write_text(json.dumps({"run": RUN, "passes": [{"pass": 0, "units": 2, "tiles": 4,
                                                                                 "manifest": str(path)}]}))
    rc, doc, answer = served_verify(path.parent, tmp_path / "verify.json")
    assert rc == 0 and answer["verdict"] == "accept" and doc["line"].startswith("ACCEPT"), doc["line"]


@pytest.mark.slow
def test_served_inert_work_records_the_tile_rules_own_inert_work_and_bound(honest_pass, tmp_path):
    path, s, epoch, prover, tr, splits, records = honest_pass
    out = tmp_path / "out"
    assert SD.main([str(path), *coins_args(path), "--tiles", "3", "--jobs", "1", "--out", str(out)]) == 0
    assert CW._inert_work_and_bound is SD._INERT_WORK_AND_BOUND and CW.D.tile_inert_work is SD._TILE_INERT_WORK
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
        debit, cap = CW._inert_work_and_bound(s, rec, ref.tile, ops, rows, cols)
        credit = CW.cell_credit(s, rec) * len(rows) * len(cols)
        assert (t["debit_exact"], t["ratio"], t["credit_exact"]) == (str(debit), float(debit / cap), str(credit))
        assert {key: t[key] for key in t if key.startswith("share")} == {"share": float(debit / credit)}
        pos_a, pos_b = {i: p for p, i in enumerate(ref.tile.rows)}, {j: p for p, j in enumerate(ref.tile.cols)}
        a_codes, b_codes = [ops.a.codes[pos_a[i]] for i in rows], [ops.b.codes[pos_b[j]] for j in cols]
        formed = [(ops.a.alpha[pos_a[i]], ops.a.x[pos_a[i]], ops.a.beta[pos_a[i]]) for i in rows]
        formed_b = [(ops.b.alpha[pos_b[j]], ops.b.x[pos_b[j]], ops.b.beta[pos_b[j]]) for j in cols]
        assert (t["atoms"], t["later_zero_promotions"], t["forming_elements"], t["mends"]) == CW.D.tile_inert_work(
            a_codes, b_codes, formed, formed_b, s.device)
        assert t["ok"] == (CW.check_opened(s, rec, ref.tile, a, b, ops) is None)
    assert got["later_zero_promotions"] == sum(t.get("later_zero_promotions", 0) for t in tiles)


@pytest.mark.slow
def test_served_inert_work_per_input_bounds_each_input_on_its_own(honest_pass, tmp_path):
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


ST = load("pearl_c_vllm_served_table", POUW / "pearl_c_vllm" / "served_table.py")


def test_the_served_table_cites_each_models_worst_input_and_refuses_what_it_cannot_cite(tmp_path):
    """Every input is re-bounded at δ = 0.05/m, m every input the table measures: both models' and a model's fresh trials."""
    def row(run, i, nt, fails=0):
        return {"pass": "decode", "run": run, "manifest_sha256": "ab", "matmul": i, "k": 4096, "n": 4096,
                "NT": nt, "W": nt * 4096, "w_max": 4096, "method": "census", "drawn": nt,
                "openings_failed": 0, "failures": fails, "share_max": 5e-5}
    def doc(run, nts):
        return {"m": len(nts), "inputs": [row(run, i, nt) for i, nt in enumerate(nts)]}
    def verify(run, label, ok):
        return {"manifest": f"/p/{run}/{label}/manifest.json", "accepted": ok, "transcript": f"{run}:/p sha256:00",
                "transcript_sha256": "00", "tiles": [1, 2], "line": f"{'ACCEPT' if ok else 'REJECT'} x", "replay": "batch",
                "cross_checked": [1]}
    def trial(i, fails):
        return row("rq", i, 1) | {"method": "fresh", "drawn": 320, "failures": fails}
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
    def audit(n, cap, ok=True):
        return {
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


FH = load("pearl_c_vllm_forming_headroom", POUW / "pearl_c_vllm" / "forming_headroom.py")


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


FD = load("pearl_c_vllm_forming_domain", POUW / "pearl_c_vllm" / "forming_domain.py")


def test_forming_domain_on_a_synthetic_pass(tmp_path, monkeypatch):
    """The domain check's numpy rows agree with s5_stats and an all-Fraction count on every credited row, skip filler and
    excluded rows, and find a row with α·s > 444 (no stride energy: ρ = 0, α·s = 448, so every element of it is loud and its
    unit fails liveness too), a row through the subnormal fallback, and a unit whose loud elements break liveness at 8ρ_max.
    Its pool forks: under forkserver (Python 3.14's default) a worker can't import this file's module to unpickle a task."""
    rng = np.random.default_rng(7)
    k = 256
    def bf16(a):
        return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype("<u2")
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
    def p_at(p):
        return sum(math.comb(100, i) * p ** i * (1 - p) ** (100 - i) for i in range(4))
    assert p_at(up) == pytest.approx(0.05, abs=1e-9) and 0.03 < up < 0.1
    assert SD.clopper_pearson_upper(5, 5, 0.05) == 1.0


def test_served_inert_work_input_bounds():
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


def test_served_inert_work_rebound_recomputes_an_inputs_json(tmp_path):
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


def _inert_work_codes(rng, rows: int, k: int, zero_first: bool) -> list[list[int]]:
    """E4M3 code rows whose words flag atoms: small codes with some large 32-code slices (the small ones' atoms then round
    away), and the first promotion group zero when `zero_first` (a later promotion into a +0 total)."""
    c = rng.integers(1, 0x10, (rows, k))
    for t in range(0, k, 32):
        if rng.random() < 0.3:
            c[:, t:t + 32] = rng.integers(0x50, 0x7E, (rows, 32))
    if zero_first:
        c[:, :128] = 0
    return [[int(x) for x in row] for row in c | rng.integers(0, 2, (rows, k)) << 7]


def test_the_batch_inert_work_is_the_scalar_inert_work_and_a_nan_code_is_the_scalars_rejection():
    FT, s = SD.FT, pouw_schemes.get(SCHEME)
    rng, seen = np.random.default_rng(3), set()
    for trial in range(8):
        a, b = _inert_work_codes(rng, 3, 384, trial % 2 == 0), _inert_work_codes(rng, 4, 384, trial % 2 == 0)
        want = CW.D.chain_inert_work(a, b, s.device)
        assert FT.chain_inert_work(a, b, s.device) == want
        seen.add(want[1] > 0)
        assert want[0] > 0
    assert seen == {True, False}
    a[0][5] = 0x7F
    with pytest.raises(Exception) as want:
        CW.D.chain_inert_work(a, b, s.device)
    with pytest.raises(type(want.value), match=str(want.value)):
        FT.chain_inert_work(a, b, s.device)


def test_the_batch_tile_inert_work_is_the_scalar_tile_inert_work():
    """`fast_tile.tile_inert_work` is `pearl_c_inert_work.tile_inert_work`, forming elements and mends included: on rows with salt-dead
    elements (accumulators past both ends' saturation, and at 0), a zero row whose every block every word flags, B rows
    whose ends make dead pairs and forced zeros with them, and a row whose accumulator overflows, where both reject
    alike."""
    FT, s = SD.FT, pouw_schemes.get(SCHEME)
    rng, seen = np.random.default_rng(7), set()
    one = 0x3F800000
    for trial in range(6):
        a, b = _inert_work_codes(rng, 3, 384, trial % 2 == 0), _inert_work_codes(rng, 4, 384, trial % 2 == 0)
        a[1] = [0] * 384
        mags = 10.0 ** rng.uniform(-3, 5, (3, 384)) * rng.choice((-1, 1), (3, 384))
        x = [[int(w) for w in row.astype(np.float32).view(np.uint32)] for row in mags]
        x[0][:7] = [0, 0x80000000, 1, 0x00800000, 0x3F000000, 0x45000000, 0xC5000000]
        beta = [int(np.float32(v).view(np.uint32)) for v in rng.uniform(0.01, 2.0, 3)]
        formed = [(one, x[i], beta[i]) for i in range(3)]
        formed_b = [(one, x[i % 3][::-1] if i else x[0], beta[i % 3]) for i in range(4)]
        want = CW.D.tile_inert_work(a, b, formed, formed_b, s.device)
        assert FT.tile_inert_work(a, b, formed, formed_b, s.device) == want
        flags = [[CW.D.word_flags(r, c, s.device) for c in b] for r in a]
        dead = [CW.D.salt_dead(*f, s.device) for f in formed]
        assert want[:2] == CW.D.chain_inert_work(a, b, s.device) and want[2] == sum(
            1 for i in range(3) for l in range(384) if dead[i][l] or all(f[l // 32] for f in flags[i]))
        ends_a, ends_b = [CW.D.row_ends(*f, s.device) for f in formed], [CW.D.row_ends(*f, s.device) for f in formed_b]
        assert want[3] == sum(CW.D.word_mend(ea, eb, s.device) for ea in ends_a for eb in ends_b)
        seen.add(0 < sum(dead[0]) < 384)
        seen.add(0 < want[3] < 3 * 4 * 384)
    assert seen == {True}
    formed[0] = (0x7F000000, x[0], beta[0])
    with pytest.raises(Exception) as want:
        CW.D.tile_inert_work(a, b, formed, formed_b, s.device)
    with pytest.raises(type(want.value), match=str(want.value)):
        FT.tile_inert_work(a, b, formed, formed_b, s.device)


def test_the_batch_inert_work_word_by_word_is_each_words_scalar_inert_work():
    """`fast_tile.words_inert_work` is `pearl_c_inert_work.word_inert_work` on every word, and its sums `chain_inert_work`'s on any rows and
    columns, so the remedy's row and column debits are the verifier's."""
    FT, s = SD.FT, pouw_schemes.get(SCHEME)
    rng = np.random.default_rng(3)
    for trial in range(4):
        a, b = _inert_work_codes(rng, 3, 384, trial % 2 == 0), _inert_work_codes(rng, 4, 384, trial % 2 == 0)
        atoms, zeros = FT.words_inert_work(a, b, s.device)
        assert atoms.shape == zeros.shape == (3, 4) and atoms.sum() > 0
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                flags, z = CW.D.word_inert_work(x, y, s.device)
                assert (atoms[i, j], zeros[i, j]) == (sum(flags), z)
        assert (atoms[[0, 2]][:, [1, 3]].sum(), zeros[[0, 2]][:, [1, 3]].sum()) == CW.D.chain_inert_work([a[0], a[2]], [b[1], b[3]], s.device)


def test_the_batch_forms_encode_and_check_a_tile_as_the_scalar_ones():
    """`fast_tile.install`'s forms (forming, the peel factors, G, C̃ and U, the debit, native BLAKE3) give a tile's encoding,
    check and debit field for field as the scalar reference, and its undo puts every scalar function back."""
    FT, s = SD.FT, pouw_schemes.get(SCHEME)
    rng = np.random.default_rng(5)
    shape = Shape(8, 256, 64)
    mm, tile = Matmul(0, shape, 0), s.tiles(shape)[0]
    a = [x.tobytes() for x in bf16_words(rng, len(tile.rows), shape.k)]
    b = [x.tobytes() for x in bf16_words(rng, len(tile.cols), shape.k, 0.02)]
    ctx = Context(bytes(range(32)), mm, bytes(32), bytes(range(1, 33)))

    def run():
        ops = s.encode(ctx, s.noise(ctx), a, b, tile)
        formed, formed_b = list(zip(ops.a.alpha, ops.a.x, ops.a.beta)), list(zip(ops.b.alpha, ops.b.x, ops.b.beta))
        return (ops, s.checked(ctx, ops), CW.D.chain_inert_work(ops.a.codes, ops.b.codes, s.device),
                CW.D.tile_inert_work(ops.a.codes, ops.b.codes, formed, formed_b, s.device))

    debits = (CW.D.chain_inert_work, CW.D.tile_inert_work)
    before = (C.form_v1, C.gram, C.peel_factors_a, C.peel_factors_b, C._basis, *debits, type(s).checked, C.blake3)
    undo = FT.install(s)
    assert C.form_v1 is FT.form_v1 and (CW.D.chain_inert_work, CW.D.tile_inert_work) == (FT.chain_inert_work, FT.tile_inert_work)
    fast = run()
    undo()
    debits = (CW.D.chain_inert_work, CW.D.tile_inert_work)
    assert (C.form_v1, C.gram, C.peel_factors_a, C.peel_factors_b, C._basis, *debits, type(s).checked, C.blake3) == before
    ref = run()
    for side in ("a", "b"):
        for field in ("codes", "alpha", "beta", "s", "rho", "ep", "x"):
            assert getattr(getattr(fast[0], side), field) == getattr(getattr(ref[0], side), field), (side, field)
    assert (fast[0].pa, fast[0].pb) == (ref[0].pa, ref[0].pb)
    assert (fast[1].c, fast[1].u) == (ref[1].c, ref[1].u) and fast[2:] == ref[2:]
    with pytest.raises(ValueError):
        FT.install(pouw_schemes.get("pearl-c-sm120-unpromoted-v1-h3"))


@pytest.mark.slow
def test_served_inert_work_batch_replay_and_resume_record_the_scalar_lines(honest_pass, tmp_path):
    """The batch replay writes the scalar replay's tile lines (all of them cross-checked here), and `--resume` on them
    measures nothing again and writes the same summary."""
    path, *_ = honest_pass
    lines = {}
    for name, extra in (("scalar", ["--scalar"]), ("batch", ["--cross-check", "1"])):
        out = tmp_path / name
        assert SD.main([str(path), *coins_args(path), "--tiles", "3", "--jobs", "1", "--out", str(out), *extra]) == 0
        lines[name] = [json.loads(x) for x in (out / f"{path.parent.name}.tiles.jsonl").read_text().splitlines()]
    drop = ("seconds", "replay", "cross_checked", "resumed_from")
    def strip(ts):
        return sorted(({k: v for k, v in t.items() if k not in drop} for t in ts), key=lambda t: t["tile"])
    assert strip(lines["scalar"]) == strip(lines["batch"]) and all(t["cross_checked"] for t in lines["batch"])
    assert {t["replay"] for t in lines["batch"]} == {"batch"} and {t["replay"] for t in lines["scalar"]} == {"scalar"}
    out = tmp_path / "resumed"
    src = tmp_path / "batch" / f"{path.parent.name}.tiles.jsonl"
    assert SD.main([str(path), *coins_args(path), "--tiles", "3", "--jobs", "1", "--out", str(out), "--resume", str(src)]) == 0
    again = [json.loads(x) for x in (out / f"{path.parent.name}.tiles.jsonl").read_text().splitlines()]
    assert strip(again) == strip(lines["batch"]) and all(t["resumed_from"] == str(src) for t in again)
    def summary(d):
        return {k: v for k, v in json.loads((d / "summary.json").read_text())["passes"][0].items() if k != "cpu_seconds"}
    assert summary(out) == summary(tmp_path / "batch")
    unstamped = tmp_path / "unstamped.jsonl"
    unstamped.write_text("".join(json.dumps({k: v for k, v in t.items() if k != "manifest_sha256"}) + "\n" for t in lines["batch"]))
    assert SD.resumed([str(unstamped)], lines["batch"][0]["manifest_sha256"]) == {}
    assert len(SD.resumed([f"{unstamped}={lines['batch'][0]['manifest_sha256']}"], lines["batch"][0]["manifest_sha256"])) == len(lines["batch"])


def test_served_inert_work_share_is_the_tile_rules_inert_work_over_its_rev1_credit():
    """served_inert_work's credit is the tile rule's cap over ρ, `creditDevRev1`'s share of the credited cells, with the first
    promotion add of every word out once: every share it records is `tileInertWorkRev1` over that credit, none with the add
    taken out a second time."""
    s = pouw_schemes.get(SCHEME)
    rng, shape, rows, cols = np.random.default_rng(11), Shape(64, 512, 48), 3, 2
    rec = PW.UnitRecord(Matmul(0, shape, 0), shape.m, shape.k, (), ())
    def codes(r):
        return [[0] * shape.k] + [[int(c) for c in rng.integers(0, 0x70, shape.k) | (rng.integers(0, 2, shape.k) << 7)]
                                  for _ in range(r - 1)]
    debit, cap = CW.tile_inert_bound(s, rec, codes(rows), codes(cols), [(0x3F800000, [0] * shape.k, 0x3F800000)] * rows,
                             [(0x3F800000, [0] * shape.k, 0x3F800000)] * cols)
    credit = Fraction(s.credit_of(Shape(1, shape.k, shape.n)), shape.n) * rows * cols
    assert s.first_adds(Shape(1, shape.k, shape.n)) == s.prices.fadd * shape.n > 0 and debit > 0
    got = SD.tile_numbers(s, debit, cap)
    assert (got["debit_exact"], got["credit_exact"], got["ratio"]) == (str(debit), str(credit), float(debit / cap))
    assert {key: got[key] for key in got if key.startswith("share")} == {"share": float(debit / credit)}


#: pearl_c_sm90's pipeline (pearl-c-h100-v1-h3's), whose calls split A's commitment (`run.Pipeline.FUSED`'s tree_seed_a,
#: past pd's rows `FUSED_GENERIC`'s)
R90 = load("pearl_c_sm90_run", KERNEL.parent / "pearl_c_sm90" / "run.py")
H100_SCHEME = "pearl-c-h100-v1-h3"


class H100Dev(RecordingDev):
    """`RecordingDev` with an H100's 132 SMs and opt-in shared memory."""

    sms = 132

    def get(self, f, *_):
        return {"cuDeviceGetAttribute": 232448, "cuFuncGetAttribute": 0}[f]


class H100TwinDev(TwinDev):
    """`TwinDev` with an H100's 132 SMs and opt-in shared memory, for pearl_c_sm90's pipeline."""

    sms = 132

    def get(self, f, *_):
        return {"cuDeviceGetAttribute": 232448, "cuFuncGetAttribute": 0}[f]
