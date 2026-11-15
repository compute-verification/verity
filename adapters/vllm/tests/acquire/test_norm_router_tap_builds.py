"""The builds of the norm-scale and router-softmax taps without a compiler: `verity-vllm norm-tap-build`
(`verity_vllm.acquire.norm_tap_src.build_ext`) and `router-tap-build` (`verity_vllm.acquire.router_tap_src.build_ext`).

A build's `main` is run whole on a stand-in vLLM tree, with torch's `load`, its device queries, `nvcc --version` and
the module's clock replaced by recorders: what it hands the compiler, the one program it starts, the header and the
record it writes, and what its `--check` says of an extension. The pins (`PINNED_SOURCES`, `BLOCK_SHA256`) are set to
the stand-in tree's digests here; `test_norm_tap_src.py` and `test_router_tap_src.py` hold the pinned texts and the
edits.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import verity_vllm.acquire.norm_tap_src.build_ext as NORM
import verity_vllm.acquire.router_tap_src.build_ext as ROUTER
from verity_vllm.config import UsageError

NVCC = "nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) NVIDIA\nCuda compilation tools, release 0.0\nBuild a.b\n"
VLLM = "d9105ea8001e0a6d77a96327d17515bb5791fb36"
FLAGS = ["-O3", "-DNDEBUG", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda"]
USAGE = r"give --vllm-src and --build-dir \(build\) or --check SO"


@dataclass(frozen=True)
class Tap:
    """One build: its module, and what it is to name, written out here and not read from the module."""

    module: Any
    name: str
    kernel_file: str
    others: tuple[str, ...]
    includes: tuple[str, ...]
    header: str
    launcher: str
    version_flag: str
    ops: tuple[str, ...]
    word: str
    block: str

    def text(self) -> str:
        return "// a stand-in for the pinned file\n" + self.block + "void after_the_block() {}\n"


NORM_TAP = Tap(
    module=NORM, name="verity_norm_tap", kernel_file="csrc/libtorch_stable/layernorm_kernels.cu",
    others=("csrc/libtorch_stable/cub_helpers.h", "csrc/libtorch_stable/type_convert.cuh"),
    includes=("csrc/libtorch_stable",), header="norm_tap_kernel.cuh", launcher="norm_tap.cu",
    version_flag="-DVERITY_NORM_TAP_VERSION=1", ops=("fused_add_rms_norm", "fused_add_rms_norm_scale", "tap_version"),
    word="NORM-TAP", block=NORM.BLOCK_START + " */\n" + "".join(old for old, _ in NORM.EDITS) + "}\n")
ROUTER_TAP = Tap(
    module=ROUTER, name="verity_router_tap", kernel_file="csrc/libtorch_stable/moe/topk_softmax_kernels.cu",
    others=("csrc/cuda_compat.h", "csrc/libtorch_stable/cub_helpers.h"), includes=("csrc", "csrc/libtorch_stable"),
    header="router_tap_kernel.cuh", launcher="router_tap.cu", version_flag="-DVERITY_ROUTER_TAP_VERSION=1",
    ops=("topk_softmax", "topk_softmax_tap", "tap_version"), word="ROUTER-TAP",
    block=ROUTER.BLOCK_START + "".join(anchor + "\n" for anchor, _, _ in ROUTER.EDITS) + ROUTER.BLOCK_END)
TAPS = pytest.mark.parametrize("tap", [NORM_TAP, ROUTER_TAP], ids=["norm", "router"])


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Toolkit:
    """What a build asked of torch and of the program it started, the extension `load` leaves behind, and the clock a
    build reads: `load` takes 2.54 seconds of it and the program eight."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.now = 100.0
        self.loads: list[dict[str, Any]] = []
        self.programs: list[tuple[list[str], dict[str, Any]]] = []
        self.devices: list[tuple[str, int]] = []
        self.libraries: list[str] = []

    def load(self, **kwargs: Any) -> None:
        self.loads.append(kwargs)
        self.now += 2.54
        Path(kwargs["build_directory"], kwargs["name"] + ".so").write_bytes(b"an extension")

    def run(self, argv: list[str], **kwargs: Any) -> types.SimpleNamespace:
        self.programs.append((argv, kwargs))
        self.now += 8.0
        return types.SimpleNamespace(stdout=NVCC, stderr="")

    def time(self) -> float:
        return self.now

    def capability(self, index: int) -> tuple[int, int]:
        self.devices.append(("capability", index))
        return (12, 3)

    def device_name(self, index: int) -> str:
        self.devices.append(("name", index))
        return "a device"


@pytest.fixture
def toolkit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Toolkit:
    """The recorders in place of torch's `load`, its toolkit's home, its two device queries, `load_library`,
    `subprocess.run` and the `time` each build reads; `monkeypatch` puts each back after the test."""
    torch = pytest.importorskip("torch")
    cpp_extension = pytest.importorskip("torch.utils.cpp_extension")
    made = Toolkit(tmp_path / "cuda")
    monkeypatch.setattr(cpp_extension, "load", made.load)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", str(made.home))
    monkeypatch.setattr(torch.cuda, "get_device_capability", made.capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", made.device_name)
    monkeypatch.setattr(torch.ops, "load_library", made.libraries.append)
    monkeypatch.setattr(subprocess, "run", made.run)
    for module in (NORM, ROUTER):
        monkeypatch.setattr(module, "time", made)
    return made


def _pinned_tree(tap: Tap, root: Path, monkeypatch: pytest.MonkeyPatch, text: str | None = None) -> dict[str, bytes]:
    """A stand-in vLLM tree under `root` and the module's pins set to it: the kernel file holds `text` (the stand-in
    block between two other lines, by default) and each other pinned file its own name. The files' bytes, by path."""
    files = {tap.kernel_file: (tap.text() if text is None else text).encode()}
    files.update({rel: rel.encode() for rel in tap.others})
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    monkeypatch.setattr(tap.module, "PINNED_SOURCES", {rel: _sha256(data) for rel, data in files.items()})
    monkeypatch.setattr(tap.module, "BLOCK_SHA256", _sha256(tap.block.encode()))
    return files


@TAPS
def test_a_build_writes_the_tapped_header_compiles_its_launcher_alone_and_records_what_it_built(
        tap: Tap, toolkit: Toolkit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """The header is the kernel file's block with the tap's edits; `load` gets this directory's launcher as its one
    source, the vLLM tree's include paths and then the build directory, where the header is; `nvcc --version` is the
    one program started, from torch's toolkit; the record names every input, keeps the version's last line and
    gives `load`'s seconds alone, to a tenth."""
    torch = pytest.importorskip("torch")
    tree, build = tmp_path / "vllm", tmp_path / "build"
    files = _pinned_tree(tap, tree, monkeypatch)
    launcher = Path(tap.module.__file__).resolve().parent / tap.launcher
    flags, so = [*FLAGS, tap.version_flag], str(build / (tap.name + ".so"))

    assert tap.module.main(tap.module.Options(vllm_src=str(tree), build_dir=str(build))) == 0

    assert toolkit.loads == [{
        "name": tap.name, "sources": [str(launcher)], "extra_cuda_cflags": flags, "extra_cflags": ["-O3", "-DNDEBUG"],
        "extra_include_paths": [*(str(tree / rel) for rel in tap.includes), str(build)], "build_directory": str(build),
        "verbose": True, "is_python_module": False, "with_cuda": True,
    }]
    captured = {"capture_output": True, "text": True}
    assert toolkit.programs == [([str(toolkit.home / "bin" / "nvcc"), "--version"], captured)]
    assert toolkit.devices == [("capability", 0), ("name", 0)] and toolkit.libraries == []
    header = (build / tap.header).read_text()
    assert header == tap.module.tapped_kernel(tap.text()) and "after_the_block" not in header
    info = json.loads((build / "build_info.json").read_text())
    assert info == {
        "name": tap.name, "tap_version": 1, "vllm": VLLM,
        "sources": {rel: _sha256(data) for rel, data in files.items()},
        "launcher": {"file": str(launcher), "sha256": _sha256(launcher.read_bytes())},
        "kernel_header_sha256": _sha256(header.encode()), "nvcc_flags": flags, "arch": "12.3", "device": "a device",
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_home": str(toolkit.home),
        "nvcc_version": ["Build a.b"], "compile_seconds": 2.5, "so": so, "so_sha256": _sha256(b"an extension"),
    }
    assert list(info["sources"]) == [tap.kernel_file, *tap.others]
    assert capsys.readouterr().out == f"BUILD-OK {so} {_sha256(b'an extension')} arch=12.3 (2.5 s)\n"


def test_the_router_header_is_the_renamed_namespace_under_its_own_includes_and_the_files_two_macros(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """`tapped_kernel` of the router build, whole: the generated-file comment, the includes the pinned file had above
    its namespace, the two macros, then the edited block. (`test_norm_tap_src.py` holds the norm build's.)"""
    monkeypatch.setattr(ROUTER, "BLOCK_SHA256", _sha256(ROUTER_TAP.block.encode()))
    assert ROUTER.tapped_kernel(ROUTER_TAP.text()) == (
        f"// Generated by verity_vllm.acquire.router_tap_src.build_ext from vLLM {VLLM} "
        "csrc/libtorch_stable/moe/topk_softmax_kernels.cu: its vllm::moe\n"
        "// namespace, renamed, with the router-softmax tap (topkGating<..., Tap = true> stores the softmax per row)."
        "  Do not edit.\n"
        "#pragma once\n#include <cassert>\n#include <cfloat>\n#include <cstdint>\n#include <type_traits>\n\n"
        "#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n\n"
        '#include "cuda_compat.h"\n#include "cub_helpers.h"\n\n'
        "#define MAX(a, b) ((a) > (b) ? (a) : (b))\n#define MIN(a, b) ((a) < (b) ? (a) : (b))\n\n"
        + ROUTER.edit_block(ROUTER_TAP.block))


@pytest.mark.parametrize(("opening", "closing"), [("", ROUTER.BLOCK_END), (ROUTER.BLOCK_START, "")],
                         ids=["no-opening", "no-closing"])
def test_the_router_edits_refuse_lines_that_do_not_open_and_close_the_namespace(opening: str, closing: str) -> None:
    """Every edit's line once is not enough: what `edit_block` renames is the namespace's two ends."""
    lines = "".join(anchor + "\n" for anchor, _, _ in ROUTER.EDITS)
    with pytest.raises(ValueError, match="the block does not open and close the vllm::moe namespace"):
        ROUTER.edit_block(opening + lines + closing)


@TAPS
def test_a_build_without_a_toolkit_home_starts_nvcc_by_its_relative_path_and_records_no_home(
        tap: Tap, toolkit: Toolkit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cpp_extension = pytest.importorskip("torch.utils.cpp_extension")
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", None)
    _pinned_tree(tap, tmp_path / "vllm", monkeypatch)

    tap.module.main(tap.module.Options(vllm_src=str(tmp_path / "vllm"), build_dir=str(tmp_path / "build")))

    assert [argv for argv, _ in toolkit.programs] == [["bin/nvcc", "--version"]]
    assert json.loads((tmp_path / "build" / "build_info.json").read_text())["cuda_home"] is None


@pytest.mark.parametrize("fault", ["another-kernel-file", "another-last-file", "another-block", "an-edits-line-twice"])
@TAPS
def test_a_build_refuses_a_tree_that_is_not_the_pinned_one_before_it_asks_torch_for_anything(
        tap: Tap, fault: str, toolkit: Toolkit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pinned file with another byte is refused by its path before the build directory is made. A kernel file whose
    block is not the pinned one, or holds an edit's line twice, is refused before a header or a record is written. No
    device is queried, no program started and nothing handed to `load`."""
    tree, build = tmp_path / "vllm", tmp_path / "build"
    start = tap.module.BLOCK_START
    another = tap.block.replace(start, start + "// another line\n")
    twice = tap.block.replace("\n}\n", "\n  __syncthreads();\n}\n") if tap is NORM_TAP else tap.block.replace(
        ROUTER.BLOCK_END, "        scale /= denom;\n" + ROUTER.BLOCK_END)
    assert another != tap.block != twice
    text = {"another-block": tap.text().replace(tap.block, another),
            "an-edits-line-twice": tap.text().replace(tap.block, twice)}.get(fault)
    _pinned_tree(tap, tree, monkeypatch, text)
    if fault == "an-edits-line-twice":
        monkeypatch.setattr(tap.module, "BLOCK_SHA256", _sha256(twice.encode()))
        refusal = "occurs 2 times in the .*, not once"
    elif fault == "another-block":
        refusal = r"is not the pinned one \(sha256 [0-9a-f]{64}, want [0-9a-f]{64}\)"
    else:
        changed = tap.kernel_file if fault == "another-kernel-file" else tap.others[-1]
        (tree / changed).write_bytes((tree / changed).read_bytes() + b"\n")
        refusal = re.escape(f"{tree}/{changed}: sha256 ") + "[0-9a-f]{64}, not the pinned [0-9a-f]{64}"

    with pytest.raises(ValueError, match=refusal):
        tap.module.main(tap.module.Options(vllm_src=str(tree), build_dir=str(build)))

    assert build.exists() == (fault in ("another-block", "an-edits-line-twice"))
    assert not (build / tap.header).exists() and not (build / "build_info.json").exists()
    assert toolkit.loads == [] and toolkit.programs == [] and toolkit.devices == []


@pytest.mark.parametrize(("ops", "version", "ok"), [((0, 1, 2), 1, True), ((0, 2), 1, False), ((0, 1), 1, False),
                                                    ((0, 1, 2), 2, False)],
                         ids=["its-three-ops", "no-tap-op", "no-tap-version", "another-tap-version"])
@TAPS
def test_a_check_loads_the_extension_and_passes_only_on_its_three_ops_at_its_tap_version(
        tap: Tap, ops: tuple[int, ...], version: int, ok: bool, toolkit: Toolkit, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """`--check SO` loads the library, reads the namespace of the build's name and prints one line: `-OK` and exit 0
    with the three ops at tap version 1, `-FAIL` and exit 1 otherwise. With an op missing the version is not read."""
    torch = pytest.importorskip("torch")
    so = tmp_path / "an_extension.so"
    so.write_bytes(b"an extension")
    held = {tap.ops[i]: (lambda: version) for i in ops}
    monkeypatch.setattr(torch.ops, tap.name, types.SimpleNamespace(**held), raising=False)
    missing = [op for i, op in enumerate(tap.ops) if i not in ops]

    assert tap.module.main(tap.module.Options(check=str(so))) == (0 if ok else 1)

    said = {"so": str(so), "sha256": _sha256(b"an extension"), "missing_ops": missing,
            "tap_version": None if missing else version, "ok": ok}
    assert capsys.readouterr().out == f"{tap.word}-{'OK' if ok else 'FAIL'} {json.dumps(said)}\n"
    assert toolkit.libraries == [str(so)] and toolkit.loads == [] and toolkit.programs == []


@pytest.mark.parametrize("given", [{}, {"vllm_src": "a tree"}, {"build_dir": "a directory"}],
                         ids=["neither", "no-build-dir", "no-vllm-src"])
@TAPS
def test_a_build_without_both_of_its_directories_is_a_usage_error(
        tap: Tap, given: dict[str, str], toolkit: Toolkit) -> None:
    with pytest.raises(UsageError, match=USAGE):
        tap.module.main(tap.module.Options(**given))
    assert toolkit.loads == [] and toolkit.programs == [] and toolkit.devices == [] and toolkit.libraries == []
