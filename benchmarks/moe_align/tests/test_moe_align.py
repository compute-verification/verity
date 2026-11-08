"""The canonical `moe_align_block_size` (`moe_align.reference`): vLLM's buffers and padding, each expert's pairs in ascending
pair index, on hand-made cases (vLLM's docstring example, empty experts, whole blocks, fewer pairs than experts, ids no
expert takes, an expert map); the same schedule the align Definitions give when every pair's key is its index; any output
vLLM's atomic order can give canonicalizes to it; the op wrapper replaces the op once and the plugin is off unless asked;
and, where Triton is importable, the kernels (in Triton's interpreter) write the reference on both paths."""
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Triton builds its own library's kernels (tl.sum, tl.cumsum) for the interpreter only when this is set at its first import
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import moe_align
from moe_align import reference

# DeepSeek-V4-Pro's routed experts and slots
E, TOPK = 384, 6


def test_vllms_docstring_example():
    # fused_moe/moe_align_block_size.py's example, with experts 1..4 of 5: each expert's pairs ascend, padded to blocks of 4
    ids = [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3]
    s, b, n = reference(ids, 5, 4)
    assert s[:16] == [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12]
    assert len(s) == 12 + 5 * 3 and set(s[16:]) == {12}
    assert b == [1, 2, 3, 4, -1, -1, -1] and n == 16


def test_empty_experts_whole_blocks_and_few_pairs():
    # experts 0 and 2 empty; expert 1 exactly two blocks of 2 (no pad slot); expert 3 one pair
    s, b, n = reference([1, 3, 1, 1, 1], 4, 2)
    assert s == [0, 2, 3, 4, 1, 5, 5, 5, 5] and b == [1, 1, 3, -1, -1] and n == 6
    # fewer pairs than experts: vLLM allocates P * BM slots
    s, b, n = reference([5, 0], 8, 4)
    assert len(s) == 8 and s == [1, 2, 2, 2, 0, 2, 2, 2] and b == [0, 5] and n == 8
    assert reference([], 8, 4) == ([], [], 0)


def test_ids_no_expert_takes_and_the_expert_map():
    # -1, E and -2**31 are no expert's; 2**32 + 1 reads as 1, as vLLM's int conversion reads an int64 id
    s, b, n = reference([-1, 2, 3, 1 << 32 | 1, -(1 << 31), 1], 3, 2)
    assert s[:4] == [3, 5, 1, 6] and b[:2] == [1, 2] and n == 4 and set(s[4:]) == {6} and set(b[2:]) == {-1}
    # the map sends global 1 to local 0 and global 0 nowhere
    s, b, n = reference([0, 1, 2, 1], 3, 2, expert_map=[-1, 0, 2])
    assert s[:4] == [1, 3, 2, 4] and b[:2] == [0, 2] and n == 4


def test_block_of_one_and_dsv4_decode_row():
    s, b, n = reference([2, 0, 2, 2, 0], 3, 1)
    assert s == [1, 4, 0, 2, 3] and b == [0, 0, 2, 2, 2] and n == 5
    # one decode row of DeepSeek-V4-Pro: six experts, one pair each, padded to the moe block of 8
    s, b, n = reference([300, 7, 42, 383, 0, 128], E, 8)
    assert len(s) == 48 and n == 48 and b == [0, 7, 42, 128, 300, 383]
    assert [s[8 * k] for k in range(6)] == [4, 1, 2, 5, 0, 3] and s.count(6) == 42


def routing(rng: random.Random, rows: int, how: str) -> list[int]:
    if how == "same6":
        six = rng.sample(range(E), TOPK)
        return [e for _ in range(rows) for e in six]
    if how == "skew":
        hot = rng.sample(range(E), 12)
        return [e for _ in range(rows) for e in rng.sample(hot if rng.random() < 0.7 else range(E), TOPK)]
    return [e for _ in range(rows) for e in rng.sample(range(E), TOPK)]


@pytest.mark.parametrize("how", ["uniform", "skew", "same6"])
def test_reference_is_the_definitions_schedule_keyed_by_pair_index(how):
    # MoeAlignRanks_v1 over key = ix = the pair index, then MoeAlignBlocks_v1: every pair at blk * BM + rank % BM
    np = pytest.importorskip("numpy")
    from verity.ml.definitions.numerics.moe_align_rows import moe_align_blocks, moe_align_ranks

    rng = random.Random(how)
    for rows in (1, 3, 16, 50):
        ids = routing(rng, rows, how)
        P, BM = len(ids), 8
        s, b, n = reference(ids, E, BM)
        ix = np.arange(P)
        st = dict(E=E, P=P, BM=BM)
        r = moe_align_ranks(st, [np.array(ids), ix, ix])
        rank, start = r["0"].view(np.int32), r["1"].view(np.int32)
        blk, par = (v.view(np.int32) for v in moe_align_blocks(dict(P=P, BM=BM), [np.array(ids), rank, start]).values())
        assert [s[k * BM + q % BM] for k, q in zip(blk.tolist(), rank.tolist())] == list(range(P))
        assert int(par[0]) * BM == n and all(b[k] == ids[p] for p, k in enumerate(blk.tolist()))


def atomic_order(ids, num_experts, block, rng):
    """An output vLLM's kernel can give: the reference with each expert's pairs in a random order."""
    s, b, n = reference(ids, num_experts, block)
    out, at = list(s), 0
    for e in range(num_experts):
        c = sum(1 for v in ids if v == e)
        mine = out[at:at + c]
        rng.shuffle(mine)
        out[at:at + c] = mine
        at += moe_align.cdiv(c, block) * block
    return out, b, n


def test_any_atomic_order_canonicalizes_to_the_reference():
    rng = random.Random(0)
    for rows, how in ((1, "uniform"), (64, "skew"), (200, "same6")):
        ids = routing(rng, rows, how)
        s, b, n = atomic_order(ids, E, 16, rng)
        assert moe_align.canonical(s, b, n, 16, len(ids)) == reference(ids, E, 16)[0]


def test_sorted_len_is_vllms():
    assert moe_align.sorted_len(6, E, 8) == 48 and moe_align.sorted_len(6 * 64, E, 8) == 384 + E * 7
    assert moe_align.sorted_len(10, 3, 4) == 10 + 9 and moe_align.sorted_len(10, 3, 4, pad_sorted_ids=True) == 20


def stand_in():
    log = []

    def real(*args):
        log.append(args)
    return SimpleNamespace(moe_align_block_size=real), log


class Recorder(moe_align.Align):
    def __init__(self):
        super().__init__()
        self.args = []

    def __call__(self, *args):
        self.args.append(args)


def test_install_replaces_once_and_keeps_the_real_op():
    ns, log = stand_in()
    rec = Recorder()
    assert moe_align.install(ns, rec) is rec
    op = ns.moe_align_block_size
    assert moe_align.install(ns) is rec and ns.moe_align_block_size is op and op.real.__name__ == "real"
    op("ids", 384, 8, "s", "b", "n", None)
    op("ids", 384, 8, "s", "b", "n")
    assert rec.args == [("ids", 384, 8, "s", "b", "n", None)] * 2 and log == []


def test_register_is_off_unless_asked(monkeypatch):
    calls = []
    monkeypatch.setattr(moe_align, "install", lambda *a, **k: calls.append(a))
    for v in (None, "", "0"):
        if v is None:
            monkeypatch.delenv(moe_align.ENV, raising=False)
        else:
            monkeypatch.setenv(moe_align.ENV, v)
        moe_align.register()
    assert calls == []
    monkeypatch.setenv(moe_align.ENV, "sorted")
    with pytest.raises(ValueError, match="canonical"):
        moe_align.register()
    monkeypatch.setenv(moe_align.ENV, "canonical")
    monkeypatch.delenv(moe_align.LOG_ENV, raising=False)
    moe_align.register()
    assert calls == [()]


def test_log_rewrites_its_record(monkeypatch, tmp_path):
    monkeypatch.setattr(moe_align, "_capturing", lambda: False)
    monkeypatch.setattr(moe_align, "FLUSH_EVERY", 2)
    align = moe_align.Align()
    moe_align.log_to(align, str(tmp_path))
    align.calls["one"] = 2
    align.on_flush()
    rec = json.loads((tmp_path / f"moe_align.{os.getpid()}.json").read_text())
    assert rec["calls"] == {"one": 2, "chunked": 0} and rec["small_max"] == moe_align.SMALL_MAX


def test_site_declares_the_plugin(tmp_path):
    d = moe_align.site(tmp_path)
    eps = (d / "verity_moe_align-0.0.dist-info" / "entry_points.txt").read_text()
    assert "[vllm.general_plugins]" in eps and "verity_moe_align = moe_align:register" in eps
    for name in ("moe_align.py", "moe_align_kernels.py"):
        assert (d / name).read_text() == Path(moe_align.__file__).with_name(name).read_text()


def test_refuses_what_vllms_op_cannot_take():
    torch = pytest.importorskip("torch")
    i32 = torch.int32
    ids, s, b, n = torch.zeros(2, 6, dtype=i32), torch.empty(48, dtype=i32), torch.empty(6, dtype=i32), torch.empty(1, dtype=i32)
    with pytest.raises(ValueError, match="num_experts"):
        moe_align.Align()(ids, 993, 8, s, b, n)
    with pytest.raises(TypeError, match="integer"):
        moe_align.Align()(ids.float(), E, 8, s, b, n)
    with pytest.raises(TypeError, match="int32"):
        moe_align.Align()(ids, E, 8, s.long(), b, n)
    with pytest.raises(ValueError, match="contiguous"):
        moe_align.Align()(torch.zeros(6, 2, dtype=i32).t(), E, 8, s, b, n)
    with pytest.raises(ValueError, match="blocks"):
        moe_align.Align()(ids, E, 8, s, b[:5], n)


def run_kernels(torch, align, ids, num_experts, block, expert_map=None, dtype=None):
    t = torch.tensor(ids, dtype=dtype or torch.int32).reshape(-1, 1)
    L = moe_align.sorted_len(len(ids), num_experts, block)
    s = torch.full((L,), -7, dtype=torch.int32)
    b = torch.full((moe_align.cdiv(L, block),), -7, dtype=torch.int32)
    n = torch.full((1,), -7, dtype=torch.int32)
    m = None if expert_map is None else torch.tensor(expert_map, dtype=torch.int32)
    align(t, num_experts, block, s, b, n, m)
    return s.tolist(), b.tolist(), int(n[0])


def interpreted(monkeypatch):
    pytest.importorskip("triton")
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(moe_align, "_KERNELS", {})
    return torch


@pytest.mark.parametrize("path", ["one", "chunked"])
def test_kernels_write_the_reference_in_interpreter(monkeypatch, path):
    torch = interpreted(monkeypatch)
    align = moe_align.Align(small_max=1 << 20, min_pb=16) if path == "one" else moe_align.Align(small_max=0, chunk=16)
    rng = random.Random(path)
    cases = [([2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3], 5, 4, None), ([1, 3, 1, 1, 1], 4, 2, None), ([5, 0], 8, 4, None),
             ([-1, 2, 3, 7, -9, 1], 3, 2, None), ([0, 1, 2, 1], 3, 2, [-1, 0, 2]), ([2, 0, 2, 2, 0], 3, 1, None)]
    cases += [([rng.randrange(-1, 9) for _ in range(rng.randrange(1, 70))], 8, bm, None) for bm in (1, 2, 3, 8, 16)]
    cases += [(routing(rng, rows, how), E, 8, None) for rows, how in ((1, "uniform"), (7, "skew"), (11, "same6"))]
    for ids, ne, bm, mp in cases:
        assert run_kernels(torch, align, ids, ne, bm, mp) == reference(ids, ne, bm, mp), (ids, ne, bm, mp)
    ids = [1 << 32 | 1, 2, -1, 1, 0, 1 << 33]
    assert run_kernels(torch, align, ids, 3, 2, dtype=torch.int64) == reference(ids, 3, 2)
    assert align.calls[path] == len(cases) + 1
