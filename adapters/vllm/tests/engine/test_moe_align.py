"""`engine.moe_align`: the canonical `moe_align_block_size` (`moe_align.reference`), vLLM's buffers and padding with each
expert's pairs in ascending pair index, on hand-made cases (vLLM's docstring example, empty experts, whole blocks, fewer pairs
than experts, ids no expert takes, an expert map); the same schedule the align Definitions give when every pair's key is its
index; any output vLLM's atomic order can give canonicalizes to it; the Marlin experts' wrapper allocates as vLLM's does, is what
the experts' function reaches, is refused where that function would bypass it, and is installed once, armed before vLLM's
module is imported, by the driver, every rank and the distribution's vLLM plugin; every call launches the program that
writes the total, 0 pairs included; and, where Triton is installed, the kernels write the reference on both paths and at 0
pairs in Triton's interpreter, in a pytest of their own (`moe_align_interpreter_cases.py`)."""
from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
import random
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest

from verity_vllm.engine import hooks, moe_align
from verity_vllm.engine.moe_align import reference

ROOT = Path(__file__).resolve().parents[2]
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
    from verity.definitions.numerics.moe_align_rows import (
        moe_align_blocks,
        moe_align_ranks,
    )

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


class RefAlign(moe_align.Align):
    """`Align` with `reference` for its kernels, on CPU tensors: the calls it gets and the buffers' lengths."""

    def __init__(self) -> None:
        super().__init__()
        self.args: list[tuple] = []

    def __call__(self, topk_ids, num_experts, block_size, sorted_token_ids, experts_ids, num_tokens_post_pad, expert_map=None):
        import torch

        self.args.append((sorted_token_ids.numel(), experts_ids.numel(), num_tokens_post_pad.numel(), expert_map is not None))
        s, b, n = reference(topk_ids.flatten().tolist(), num_experts, block_size,
                            None if expert_map is None else expert_map.tolist(), sorted_token_ids.numel())
        sorted_token_ids.copy_(torch.tensor(s, dtype=torch.int32))
        experts_ids[:len(b)] = torch.tensor(b, dtype=torch.int32)
        num_tokens_post_pad[0] = n


#: vLLM's Marlin experts module as far as this goes: its module-global wrapper, which must not run once installed, and the
#: experts' function, which calls it as fused_marlin_moe does (marlin_moe.py, `moe_align_block_size(..., ignore_invalid_experts=True)`)
MARLIN = '''
def moe_align_block_size(*args, **kwargs):
    raise AssertionError("vLLM's wrapper ran")


def fused_marlin_moe(topk_ids, block_size_m, global_num_experts, expert_map=None):
    return moe_align_block_size(topk_ids, block_size_m, global_num_experts, expert_map, ignore_invalid_experts=True)
'''
#: the same module with the experts reaching vLLM's aligner by another name, which an install on the module global would miss
BYPASSED = MARLIN.replace("    return moe_align_block_size(", "    import vllm_ops as ops\n    return ops.moe_align_block_size(")


def marlin_module(where: Path, name: str = "fake_marlin_moe", source: str = MARLIN) -> types.ModuleType:
    path = where / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def clean():
    moe_align.uninstall()
    yield
    moe_align.uninstall()


def test_install_replaces_marlins_wrapper_once_and_uninstall_restores_it(clean, tmp_path):
    m = marlin_module(tmp_path)
    vllms = m.moe_align_block_size
    rec = RefAlign()
    assert moe_align.install(m, rec) is rec and moe_align.installed() is rec
    ours = m.moe_align_block_size
    assert ours is not vllms and moe_align.install(m) is rec and m.moe_align_block_size is ours
    moe_align.uninstall()
    assert m.moe_align_block_size is vllms and moe_align.installed() is None


def test_the_wrapper_allocates_as_vllms_and_applies_the_expert_map(clean):
    torch = pytest.importorskip("torch")
    rec = RefAlign()
    call = moe_align.wrapper(rec)
    ids = [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3]
    t = torch.tensor(ids, dtype=torch.int32).reshape(4, 3)
    for pad in (False, True):
        s, b, n = call(t, 4, 5, None, pad_sorted_ids=pad)
        L = moe_align.sorted_len(12, 5, 4, pad)
        assert (s.numel(), b.numel(), n.numel()) == (L, moe_align.cdiv(L, 4), 1) == rec.args[-1][:3]
        assert (s.tolist(), b.tolist(), int(n[0])) == reference(ids, 5, 4, length=L)
    emap = torch.tensor([-1, 0, 2, 1, -1], dtype=torch.int32)
    # Marlin's call: invalid experts ignored, so the op takes the map and counts no pair it sends to -1
    s, b, n = call(t, 4, 5, emap, ignore_invalid_experts=True)
    assert rec.args[-1][3] and (s.tolist(), b.tolist(), int(n[0])) == reference(ids, 5, 4, emap.tolist())
    # otherwise every expert counts and the map renames the blocks' experts after the op, as vLLM's wrapper does
    s, b, n = call(t, 4, 5, emap)
    gs, gb, gn = reference(ids, 5, 4)
    assert not rec.args[-1][3] and s.tolist() == gs and int(n[0]) == gn and b.tolist() == emap[torch.tensor(gb)].tolist()


def test_marlins_experts_reach_the_canonical_op_and_an_install_they_would_bypass_is_refused(clean, tmp_path):
    torch = pytest.importorskip("torch")
    m = marlin_module(tmp_path)
    moe_align.install(m, RefAlign())
    ids = [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3]
    s, b, n = m.fused_marlin_moe(torch.tensor(ids, dtype=torch.int32).reshape(4, 3), 4, 5)
    assert (s.tolist(), b.tolist(), int(n[0])) == reference(ids, 5, 4)
    moe_align.uninstall()
    bypassing = marlin_module(tmp_path, "bypassing_marlin_moe", BYPASSED)
    for bypassed in (bypassing, types.ModuleType("no_experts")):
        with pytest.raises(RuntimeError, match="bypass"):
            moe_align.install(bypassed, RefAlign())
        assert moe_align.installed() is None
    assert moe_align.loads_global(m.fused_marlin_moe, moe_align.NAME)
    assert not moe_align.loads_global(bypassing.fused_marlin_moe, moe_align.NAME)


def test_install_is_refused_under_a_live_wrap_of_the_name(clean, tmp_path):
    m = marlin_module(tmp_path)
    on = hooks.wrap(m, moe_align.NAME, lambda orig: lambda *a, **k: orig(*a, **k))
    try:
        with pytest.raises(RuntimeError, match="already on"):
            moe_align.install(m, RefAlign())
        assert moe_align.installed() is None
    finally:
        on.uninstall()


def test_arm_installs_when_vllms_module_is_imported_and_imports_nothing_itself(clean, tmp_path, monkeypatch):
    pkg = tmp_path / "fake_vllm_experts"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "marlin_moe.py").write_text(MARLIN)
    monkeypatch.syspath_prepend(str(tmp_path))
    site = "fake_vllm_experts.marlin_moe"
    try:
        moe_align.arm(site)
        moe_align.arm(site)
        assert site not in sys.modules and moe_align.installed() is None
        assert sum(isinstance(f, moe_align._OnImport) and f.site == site for f in sys.meta_path) == 1
        mod = importlib.import_module(site)
        assert moe_align.installed() is not None and mod.moe_align_block_size.__module__ == moe_align.__name__
    finally:
        sys.modules.pop(site, None)
        sys.modules.pop("fake_vllm_experts", None)


def test_register_arms_and_the_distribution_declares_it_as_a_vllm_plugin(monkeypatch):
    armed = []
    monkeypatch.setattr(moe_align, "arm", lambda *a: armed.append(a))
    moe_align.register()
    assert armed == [()]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    target = project["entry-points"]["vllm.general_plugins"]["verity_moe_align"]
    module, _, attr = target.partition(":")
    assert importlib.import_module(module) is moe_align and attr == "register"


def test_it_is_armed_at_import_by_every_rank_and_by_the_driver_before_it_builds():
    from verity_vllm.engine import build
    from verity_vllm.pipeline.tp import rank_worker

    top = ast.parse(inspect.getsource(moe_align)).body
    assert any(isinstance(n, ast.Expr) and ast.unparse(n) == "arm()" for n in top)
    assert any(isinstance(n, ast.ImportFrom) and n.module == "verity_vllm.engine" and "moe_align" in {a.name for a in n.names}
               for n in ast.parse(inspect.getsource(rank_worker)).body)
    src = inspect.getsource(build.build_engine)
    assert src.index("moe_align.arm()") < src.index("llm = LLM(**kw)")


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


class Grids:
    """The Triton kernels' stand-ins: each launch's kernel and grid."""

    def __init__(self) -> None:
        self.launched: list[tuple[str, tuple]] = []

    def kernel(self, name: str):
        grids = self

        class Kernel:
            def __getitem__(self, grid):
                return lambda *a, **k: grids.launched.append((name, grid))
        return Kernel()


def test_every_call_launches_the_program_that_writes_the_total_even_at_zero_pairs(monkeypatch):
    # program 0 of either path writes num_tokens_post_pad (the kernels' `_tail`); align_small's grid of cdiv(P, T) programs
    # launched none at P = 0 and left it unwritten
    torch = pytest.importorskip("torch")
    grids = Grids()
    monkeypatch.setattr(moe_align, "_KERNELS", {k: grids.kernel(k) for k in ("align_small", "count_chunks", "scan_chunks", "place_chunks")})
    align = moe_align.Align(fast_launch=False)
    for tokens in (0, 1, moe_align.SMALL_MAX // TOPK, moe_align.SMALL_MAX // TOPK + 1):
        L = moe_align.sorted_len(tokens * TOPK, E, 8)
        s, b, n = (torch.empty(k, dtype=torch.int32) for k in (L, moe_align.cdiv(L, 8), 1))
        grids.launched.clear()
        align(torch.zeros((tokens, TOPK), dtype=torch.int32), E, 8, s, b, n)
        assert grids.launched and all(min(g) >= 1 for _k, g in grids.launched), (tokens, grids.launched)
        if tokens == 0:
            assert grids.launched == [("align_small", (1,))]
    assert align.calls == {"one": 3, "chunked": 1}


INTERPRETER_CASES = "tests/engine/moe_align_interpreter_cases.py"


@pytest.fixture(scope="module")
def interpreter(tmp_path_factory) -> subprocess.CompletedProcess[str]:
    """`moe_align_interpreter_cases.py` in a pytest of their own, once for this module, wherever Triton is installed.  Triton
    takes `TRITON_INTERPRET` at its first import (the cases' docstring), and this process made that import before any test:
    `conftest.py` imports `torch._inductor.config`, which loads Triton.  So the variable is in the child's environment from its
    start, and in no other test's."""
    absent = [name for name in ("triton", "torch") if importlib.util.find_spec(name) is None]
    if absent:
        pytest.skip(f"not installed: {', '.join(absent)}")
    return subprocess.run([sys.executable, "-m", "pytest", INTERPRETER_CASES, "-rA", "-p", "no:cacheprovider",
                           f"--basetemp={tmp_path_factory.mktemp('interpreter') / 'b'}"], cwd=ROOT,
                          env={**os.environ, "TRITON_INTERPRET": "1"}, capture_output=True, text=True)


def passed_in_interpreter(run: subprocess.CompletedProcess[str], case: str) -> None:
    """The child ran `case` and passed it, and nothing else of the child failed."""
    assert run.returncode == 0 and f"PASSED {INTERPRETER_CASES}::{case}" in run.stdout.splitlines(), run.stdout + run.stderr


@pytest.mark.parametrize("path", ["one", "chunked"])
def test_kernels_write_the_reference_in_interpreter(interpreter, path):
    passed_in_interpreter(interpreter, f"test_kernels_write_the_reference_in_interpreter[{path}]")


def test_kernels_write_the_total_at_zero_pairs_in_interpreter(interpreter):
    passed_in_interpreter(interpreter, "test_kernels_write_the_total_at_zero_pairs_in_interpreter")
