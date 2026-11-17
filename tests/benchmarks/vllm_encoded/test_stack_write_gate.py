"""The stack write gate's ptxas check (`benchmarks/vllm_encoded/stack_write_gate.py --ptxas`): a launched kernel with a
stack frame and no spill passes, its frame's bytes in the output; one that spills fails, named."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "vllm_encoded"))

import stack_write_gate as SWG

ENCODE = "_ZN2kv16pg_encode_kernelEPKmiiiiiii"
LEAVES = "_ZN2rb13leaves_kernelEPKmmmPmPh"


def report(*kernels: tuple[str, int, int, int]) -> str:
    """ptxas -v's report of `kernels`, each a mangled name, its stack frame and its spill stores and loads."""
    return "".join(f"ptxas info    : Compiling entry function '{m}' for 'sm_90a'\n"
                   f"ptxas info    : Function properties for {m}\n"
                   f"    {f} bytes stack frame, {s} bytes spill stores, {l} bytes spill loads\n"
                   f"ptxas info    : Used 202 registers, used 0 barriers, {f} bytes cumulative stack size\n"
                   for m, f, s, l in kernels)


def test_a_stack_frame_with_no_spill_passes_and_its_bytes_are_in_the_output(tmp_path):
    (tmp_path / "paged.ptxas").write_text(report((ENCODE, 576, 0, 0), (LEAVES, 0, 0, 0)))
    res = SWG.ptxas([str(tmp_path / "paged.ptxas")], 3)
    assert res["ok"] and res["spills"] == [] and len(res["kernels"]) == 2
    assert res["frames"] == {"kv::pg_encode_kernel(unsigned long const*, int, int, int, int, int, int, int)": 576}


def test_a_spill_fails_and_names_its_kernel(tmp_path):
    (tmp_path / "paged.ptxas").write_text(report((ENCODE, 576, 0, 0)))
    (tmp_path / "reader.ptxas").write_text(report((LEAVES, 144, 144, 144)))
    res = SWG.ptxas([str(tmp_path / "paged.ptxas"), str(tmp_path / "reader.ptxas")], 3)
    assert not res["ok"]
    assert res["spills"] == ["rb::leaves_kernel(unsigned long const*, unsigned long, unsigned long, unsigned long*, "
                             "unsigned char*)"]
