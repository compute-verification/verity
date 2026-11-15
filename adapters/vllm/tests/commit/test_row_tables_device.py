"""`row_tables_device`: `row_tables.step_roots` on the GPU (`row_tables_device.step_roots`) and its CPU twin, each held bit for bit to the CPU reference
(`row_tables.step_roots`, over `verity.commitments.row_table`) on random steps.  The steps cover a row's segment cases (K + 4
a multiple of 1024 and two past one, codes whose count is not a multiple of 8), one-row and non-power-of-two tables, a one-table
step, rows out of rank order, tables out of step order, and an engine step past 2^32 (the nonce's second word).  The twin needs a
C++ compiler; the device tests need CUDA and nvcc, and skip elsewhere.  What a build hands its compiler (`built`) is held with a
compiler that only records it, on every machine."""
from __future__ import annotations

import ctypes
import re
import tempfile
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from verity.commitments import row_table as RT

from verity_vllm.commit import row_tables as RTB
from verity_vllm.commit import row_tables_device as D

CTX, KEY = bytes(range(64)), bytes(range(100, 132))
STEPS = [
    {"model.layers.0.mlp.down_proj": (1, 2)},
    {"model.layers.0.self_attn.qkv_proj": (3, 1020), "model.layers.0.mlp.down_proj": (5, 1022), "lm_head": (2, 2046)},
    {"model.layers.1.self_attn.o_proj": (17, 4096), "model.layers.0.self_attn.o_proj": (9, 4100),
     "model.layers.0.mlp.down_proj": (33, 14336)},
]


def outputs(shapes: dict[str, tuple[int, int]], seed: int) -> dict:
    """A step's quantize outputs: random codes and scales, each table's (request, token row) ids shuffled."""
    rng = np.random.default_rng(seed)
    out = {}
    for op, (b, k) in shapes.items():
        ids = [(f"req{i % 3}", i) for i in range(b)]
        rng.shuffle(ids)
        out[op] = (ids, rng.integers(-128, 128, (b, k), dtype=np.int8), rng.standard_normal(b).astype(np.float32))
    return out


def test_the_table_domain_ids_are_row_tables():
    for step, index, leaves in [(0, 0, 1), (9, 3, 32), (2**33 + 1, 127, 300), (5, 1000, 2**16 + 1)]:
        assert D.table_domain_id(CTX, step, index, leaves) == RT.table_domain(CTX, step, index, leaves).domain_id


@pytest.mark.parametrize("engine_step", [7, 2**33 + 5])
@pytest.mark.parametrize("shapes", STEPS, ids=["one-row", "segment-edges", "llama-shapes"])
def test_the_cpu_twin_is_the_cpu_reference(shapes, engine_step):
    outs = outputs(shapes, engine_step)
    assert D.host_step(CTX, engine_step, outs, KEY) == RTB.step_roots(CTX, engine_step, outs, KEY)


def test_tables_that_share_one_rows_list_are_ranked_alike():
    outs = outputs({"model.layers.0.self_attn.qkv_proj": (6, 4096), "model.layers.0.mlp.down_proj": (6, 14336)}, 5)
    ids = outs["model.layers.0.mlp.down_proj"][0]
    outs = {op: (ids, c, s) for op, (_ids, c, s) in outs.items()}
    assert D.host_step(CTX, 9, outs, KEY) == RTB.step_roots(CTX, 9, outs, KEY)


def test_the_plan_refuses_what_step_roots_refuses():
    for bad in [{"x": ([("r", 0)], 1, 3)}, {"x": ([("r", 0), ("r", 0)], 2, 4)}, {"x": ([("r", 0)], 2, 4)}, {}]:
        with pytest.raises(ValueError):
            D.plan(CTX, 1, bad, KEY)
    with pytest.raises(ValueError):
        D.plan(bytes(32), 1, {"x": ([("r", 0)], 1, 4)}, KEY)


#: a compiler that records what it is given (its arguments, its directory and the files there) and writes its -o
RECORDER = """#!/bin/sh
if [ "$1" = --version ]; then echo "a compiler that records"; exit 0; fi
printf '%s\\n' "$@" > "@REC@/argv"; pwd > "@REC@/cwd"; ls > "@REC@/files"
while [ $# -gt 0 ]; do if [ "$1" = -o ]; then out=$2; fi; shift; done
cat ./* > "$out"
"""


def _recorder(tmp_path: Path, monkeypatch) -> tuple[str, Path]:
    """The recording compiler, named as one, and where it records; builds go under tmp_path."""
    tool, rec = tmp_path / "bin" / "nvcc", tmp_path / "rec"
    tool.parent.mkdir()
    rec.mkdir()
    tool.write_text(RECORDER.replace("@REC@", str(rec)))
    tool.chmod(0o755)
    monkeypatch.setattr(D, "BUILD_DIR", tmp_path / "built")
    return str(tool), rec


def test_a_build_hands_its_compiler_no_path_in_the_tree(tmp_path, monkeypatch):
    """The compiler runs in a directory of the build's own that holds a copy of each source and header under its base name,
    and is given the last source by that name: of its arguments only -o's is a path, and it is the build's output."""
    tool, rec = _recorder(tmp_path, monkeypatch)
    flags = (*D.NVCC_FLAGS, "-gencode=arch=compute_90,code=sm_90")
    sources = (str(D.SHA512), *D.DEVICE_SOURCES)                # a header named among the sources and the headers: one file
    lib = D.built(tool, sources, flags, "librow_table_cuda.so", D.SHARED)
    argv = (rec / "argv").read_text().splitlines()
    out = argv[argv.index("-o") + 1]
    assert argv == [*flags, "-o", out, "row_table.cu"] and not any("/" in a for a in argv if a != out)
    assert Path(out).parent == lib.parent and lib.parent.parent == tmp_path / "built" and lib.stat().st_size > 0
    stage = Path((rec / "cwd").read_text().strip())
    assert stage.name.startswith("verity-build.") and stage.parent.resolve() == Path(tempfile.gettempdir()).resolve()
    assert not stage.exists()
    assert set((rec / "files").read_text().split()) == {Path(s).name for s in sources} | {p.name for p in D.SHARED}
    (rec / "argv").unlink()
    assert D.built(tool, sources, flags, "librow_table_cuda.so", D.SHARED) == lib and not (rec / "argv").exists()


def test_two_files_with_one_name_are_refused_before_the_compiler_runs(tmp_path, monkeypatch):
    tool, rec = _recorder(tmp_path, monkeypatch)
    other = tmp_path / "elsewhere" / "ncp2.cuh"
    other.parent.mkdir()
    other.write_text("// another file of that name\n")
    with pytest.raises(ValueError) as refused:
        D.built(tool, D.DEVICE_SOURCES, D.NVCC_FLAGS, "librow_table_cuda.so", (*D.SHARED, other))
    assert f"two files named ncp2.cuh among a build's sources and headers, {D.CUDA_DIR / 'ncp2.cuh'} and {other}" in str(refused.value)
    assert not list(rec.iterdir()) and not (tmp_path / "built").exists()


def test_every_build_takes_its_sources_from_the_one_directory_of_pouws_prover():
    """`CUDA_DIR` and `pouw.device`'s directory are one directory, found from the package that holds it and not from where the
    adapter's own files are, so it is the same in a checkout and in an installed adapter; and every source a build names is a
    file there, the device builds' too, which only nvcc compiles and no test without it reads."""
    pytest.importorskip("torch")
    from verity.protocols.accounting.pouw import prover

    from verity_vllm.protocols.pouw import device as PD
    from verity_vllm.protocols.pouw.h1 import receipts
    from verity_vllm.protocols.pouw.h1 import registered_device as RD

    assert D.CUDA_DIR == PD.HERE == Path(prover.__file__).resolve().parent / "kernels" / "pouw_cuda"
    named = {*D.DEVICE_SOURCES, *D.HOST_SOURCES, *PD.SOURCES, *PD.CUDA_SOURCES, *PD.FUSED_SOURCES, *RD.DEVICE_SOURCES,
             *RD.HOST_SOURCES, *RD.ZY_DEVICE_SOURCES, *RD.ZY_HOST_SOURCES, receipts.NOISE_CU}
    assert sorted(s for s in named if not (D.CUDA_DIR / s).is_file()) == []


def _declared(name: str) -> list[Any]:
    """The ctypes of ``name``'s parameters as `row_table.cu` declares them: `Args`' pointer for a ``const Args*``, a void pointer
    for any other pointer, and the integer of the declared width."""
    (parameters,) = re.findall(rf'extern "C" int {name}\(([^)]*)\)', (D.CUDA_DIR / "row_table.cu").read_text())
    kinds = [" ".join(p.split()).rsplit(" ", 1)[0] for p in parameters.split(",")]
    integers = {"uint64_t": ctypes.c_uint64, "uint32_t": ctypes.c_uint32}
    return [ctypes.POINTER(D.Args) if k == "const Args*" else ctypes.c_void_p if k.endswith("*") else integers[k] for k in kinds]


def test_the_device_build_is_called_as_its_source_declares_it(monkeypatch):
    """`device_lib` builds `DEVICE_SOURCES` with nvcc for the architecture it is given, and gives ``rt_midstates`` and ``rt_step``
    the arguments `row_table.cu` declares for them and an ``int`` to return.  A stand-in is the compiler and the library it
    builds, so nothing is compiled and the test runs without CUDA."""
    handed: list[tuple[Any, ...]] = []
    lib: Any = SimpleNamespace(rt_midstates=SimpleNamespace(), rt_step=SimpleNamespace())

    def build(*given: Any) -> Any:
        handed.append(given)
        return lib

    monkeypatch.setattr(D, "nvcc", lambda: "an-nvcc")
    monkeypatch.setattr(D, "_build", build)
    assert D.device_lib.__wrapped__("90") is lib
    assert handed == [("an-nvcc", D.DEVICE_SOURCES, (*D.NVCC_FLAGS, "-gencode=arch=compute_90,code=sm_90"), "librow_table_cuda.so")]
    for name in ("rt_midstates", "rt_step"):
        kernel = getattr(lib, name)
        assert list(kernel.argtypes) == _declared(name) and kernel.restype is ctypes.c_int


def _cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or D.nvcc() is None:
        pytest.skip("the device path needs CUDA and nvcc")
    return torch


def _on(torch, outs: dict) -> dict:
    return {op: (ids, torch.from_numpy(c).cuda(), torch.from_numpy(s).cuda()) for op, (ids, c, s) in outs.items()}


@pytest.mark.parametrize("shapes", STEPS, ids=["one-row", "segment-edges", "llama-shapes"])
def test_the_device_path_is_the_cpu_reference(shapes):
    torch = _cuda()
    outs = outputs(shapes, 11)
    pending = D.step_roots(CTX, 2**33 + 5, _on(torch, outs), KEY)
    assert isinstance(pending, D.Pending)
    assert pending.result() == RTB.step_roots(CTX, 2**33 + 5, outs, KEY)
    timed = D.step_roots(CTX, 2**33 + 5, _on(torch, outs), KEY, timing=True)
    assert timed.kernel_ms() > 0 and timed.result() == pending.result()


def test_the_side_stream_reads_the_codes_the_serving_stream_wrote_before_the_launch():
    torch = _cuda()
    outs = outputs(STEPS[2], 12)
    dev = {op: (ids, torch.zeros_like(c), torch.zeros_like(s)) for op, (ids, c, s) in _on(torch, outs).items()}
    big = torch.ones(4096, 4096, device="cuda")
    for _ in range(8):
        big = big @ big / 4096
    for op, (_ids, c, s) in dev.items():
        c.copy_(torch.from_numpy(outs[op][1]), non_blocking=True)
        s.copy_(torch.from_numpy(outs[op][2]), non_blocking=True)
    assert D.step_roots(CTX, 3, dev, KEY).result() == RTB.step_roots(CTX, 3, outs, KEY)


def test_record_ready_hands_over_the_ready_steps_in_launch_order():
    torch = _cuda()
    steps = {s: outputs(STEPS[s % 3], 20 + s) for s in range(4)}
    queue = deque(D.step_roots(CTX, s, _on(torch, o), KEY) for s, o in steps.items())
    queue[-1].done.synchronize()
    got: list[RTB.StepRoots] = []
    assert D.record_ready(queue, got.append) == 4 and not queue
    assert got == [RTB.step_roots(CTX, s, o, KEY) for s, o in steps.items()]
    assert D.record_ready(queue, got.append) == 0
