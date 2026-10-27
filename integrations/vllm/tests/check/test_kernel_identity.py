"""kernel_identity: a file the scan can't read is named in the record, and no identity is claimed over the partial sets.

A complete scan's record is pinned byte for byte, so naming unreadable files changes nothing for a cache the scan reads whole.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from verity_vllm.engine import kernel_identity as KI

_WRAPPER = '''# AOT ID: ['0_inference']
from torch._inductor.async_compile import AsyncCompile
async_compile = AsyncCompile()
# kernel path: /tmp/torchinductor_x/ab/cabc.py
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', \'\'\'
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x0 = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    tl.store(out_ptr0 + x0, tl.load(in_ptr0 + x0) * 2, x0 < xnumel)
\'\'\', device_str='cuda')

def call(args):
    arg0_1, = args
    buf0 = empty_strided_cuda((8, ), (1, ), torch.float32)
    triton_poi_fused_0.run(arg0_1, buf0, 8, stream=stream0)
    return (buf0, )

def benchmark_compiled_module(times=10, repeat=10):
    return 0
'''
_KERNEL = '''@triton_heuristics.pointwise(size_hints={'x': 8}, filename=__file__)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    pass
'''
_PTX = '''.version 8.4
.file 1 "/tmp/torchinductor_x/cd/kdef.py"
.visible .entry triton_poi_fused_0()
{
.loc 1 3 0
ret;
}
'''
#: sha256 of a complete scan's record of `_tree` (cache dirs left out), as `collect` wrote it before it named unreadable files
COMPLETE_RECORD_SHA256 = "46ac2a4dad7e2ab2404116431034c88b5c6321c7fb606511ce8a1e585b2b73a9"


def _tree(base: Path) -> tuple[Path, Path]:
    ind, tri = base / "inductor", base / "triton"
    for d in (ind / "ab", ind / "cd", tri / "x"):
        d.mkdir(parents=True)
    (ind / "ab" / "cabc.py").write_text(_WRAPPER)
    (ind / "cd" / "kdef.py").write_text(_KERNEL)
    (ind / "cd" / "kdef.best_config").write_text(json.dumps({"XBLOCK": 512, "num_warps": 8, "num_stages": 1, "configs_hash": "c0ffee",
                                                             "found_by_coordesc": False}))
    (tri / "x" / "triton_poi_fused_0.cubin").write_bytes(b"\x7fELF cubin bytes")
    (tri / "x" / "triton_poi_fused_0.ptx").write_text(_PTX)
    return ind, tri


def _unreadable(ind: Path, tri: Path) -> None:
    """A directory named like a module or a PTX file can't be read as text; a best_config that isn't JSON can't be parsed."""
    (ind / "ab" / "zz.py").mkdir()
    (ind / "cd" / "kdef.best_config").write_text("{not json")
    (tri / "x" / "triton_poi_fused_0.ptx").unlink()
    (tri / "x" / "triton_poi_fused_0.ptx").mkdir()


def _record_sha256(doc: dict) -> str:
    kept = {k: v for k, v in doc.items() if k not in ("inductor_cache_dir", "triton_cache_dir")}
    return hashlib.sha256(json.dumps(kept, sort_keys=True).encode()).hexdigest()


def test_a_complete_scan_records_exactly_what_it_did(tmp_path):
    doc = KI.collect(*map(str, _tree(tmp_path)))
    assert "unreadable" not in doc and "error" not in doc["summary"]
    assert _record_sha256(doc) == COMPLETE_RECORD_SHA256


def test_unreadable_files_are_named_and_mark_the_record_partial(tmp_path):
    ind, tri = _tree(tmp_path)
    _unreadable(ind, tri)
    doc = KI.collect(str(ind), str(tri))
    assert [u["path"] for u in doc["unreadable"]] == ["ab/zz.py", "cd/kdef.best_config", "x/triton_poi_fused_0.ptx"]
    errors = {u["path"]: u["error"] for u in doc["unreadable"]}
    assert errors["ab/zz.py"].startswith("IsADirectoryError") and errors["cd/kdef.best_config"].startswith("JSONDecodeError")
    assert doc["summary"]["error"].startswith("3 file(s) unreadable, so these sets are partial: ab/zz.py: IsADirectoryError")
    assert doc["autotune_configs"] == [] and doc["cubins"][0]["ptx_code_sha256"] is None
    assert doc["summary"]["wrappers"] == 1


def test_compare_claims_no_identity_over_a_partial_scan(tmp_path):
    whole = [KI.collect(*map(str, _tree(tmp_path / side))) for side in ("a", "b")]
    same = KI.compare(*whole)
    assert same["kernel_sources_identical"] and same["wrappers_identical_modulo_cache_path"] and same["call_programs_identical"]
    assert same["ptx_code_identical"] and same["autotune"]["chosen_configs_identical"] and "unreadable" not in same
    partial = []
    for side in ("a", "b"):
        ind, tri = tmp_path / side / "inductor", tmp_path / side / "triton"
        (ind / "ab" / "zz.py").mkdir()
        partial.append(KI.collect(str(ind), str(tri)))
    cmp = KI.compare(*partial)
    # both sides left the same file out, so their sets agree, and that agreement shows nothing
    assert cmp["kernel_sources"]["common"] == 1 and not cmp["kernel_sources"]["only_a"] and not cmp["kernel_sources"]["only_b"]
    for claim in ("kernel_sources_identical", "wrappers_identical", "wrappers_identical_modulo_cache_path", "call_programs_identical",
                  "cubins_identical", "ptx_code_identical", "b_kernel_sources_subset_of_a"):
        assert cmp[claim] is False, claim
    assert cmp["autotune"]["chosen_configs_identical"] is False and cmp["autotune"]["candidates_identical"] is False
    assert {side: [u["path"] for u in us] for side, us in cmp["unreadable"].items()} == {"a": ["ab/zz.py"], "b": ["ab/zz.py"]}


def test_the_run_record_nulls_its_kernel_pin_when_a_file_was_unreadable(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from verity_vllm.engine import code_identity as CI

    ind, _ = _tree(tmp_path)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(ind))
    monkeypatch.setenv(CI._VERITY_INDUCTOR_DIR_MARK, "1")
    assert len(CI.generated_kernels_of_this_process()["kernel_set_sha256"]) == 64
    (ind / "ab" / "zz.py").mkdir()
    gk = CI.generated_kernels_of_this_process()
    assert gk["kernel_set_sha256"] is None and gk["call_set_sha256"] is None
    assert gk["error"].startswith("1 file(s) unreadable, so these sets are partial: ab/zz.py: IsADirectoryError")


def test_no_cache_dir_raises_rather_than_scanning_nothing(monkeypatch):
    import sys

    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.setitem(sys.modules, "torch._inductor.runtime.runtime_utils", None)
    with pytest.raises(ImportError):
        KI.inductor_cache_dir()
