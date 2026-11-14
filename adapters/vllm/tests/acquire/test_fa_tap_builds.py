"""The two builds of the FlashAttention taps without a compiler: `verity-vllm fa2-tap-build`
(`verity_vllm.acquire.fa2_tap_src.build_ext`) and `fa3-tap-build` (`verity_vllm.acquire.fa3_tap_src.build_fa3_ext`).

A build's `main` is run whole on a tree of empty files, with torch's `load` and the toolkit's two programs replaced by
recorders: what it hands the compiler, which programs it starts, and what it writes beside the extension.
"""

from __future__ import annotations

import json
import os
import subprocess
import types
from pathlib import Path
from typing import Any

import pytest

import verity_vllm.acquire.fa2_tap_src.build_ext as FA2
import verity_vllm.acquire.fa3_tap_src.build_fa3_ext as FA3
from verity_vllm import settings

NAME = "verity_fa_tap_builds_test"
NVCC = "nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) NVIDIA\nCuda compilation tools, release 0.0\nBuild a.b\n"
SET_BY_A_BUILD = ("CUDA_HOME", "MAX_JOBS", "TORCH_CUDA_ARCH_LIST")


class Toolkit:
    """What a build asked of torch's `load` and of the programs it started, and the extension `load` leaves behind."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.loads: list[dict[str, Any]] = []
        self.programs: list[tuple[list[str], dict[str, Any]]] = []
        self.extension: bytes | None = b"an extension"

    def load(self, **kwargs: Any) -> None:
        self.loads.append(kwargs)
        if self.extension is not None:
            Path(kwargs["build_directory"], kwargs["name"] + ".so").write_bytes(self.extension)

    def run(self, argv: list[str], **kwargs: Any) -> types.SimpleNamespace:
        self.programs.append((argv, kwargs))
        if argv[1:] == ["--version"]:
            return types.SimpleNamespace(stdout=NVCC, stderr="")
        return types.SimpleNamespace(stdout="Fatbin elf code:\n", stderr="a warning\n")


@pytest.fixture
def toolkit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Toolkit:
    """The recorders in place, the extension's op namespace as one public name and one private, and the three settings a
    build writes set through `monkeypatch`, which puts each back after the test."""
    torch = pytest.importorskip("torch")
    cpp_extension = pytest.importorskip("torch.utils.cpp_extension")
    made = Toolkit(tmp_path / "cuda")
    for name, value in zip(SET_BY_A_BUILD, (str(made.home), "7", "0.0"), strict=True):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(cpp_extension, "load", made.load)
    monkeypatch.setattr(subprocess, "run", made.run)
    monkeypatch.setattr(torch.ops, NAME, types.SimpleNamespace(verity_tap_mode=None, _private=None), raising=False)
    return made


def _files(root: Path, *names: str) -> list[str]:
    """Each named file made under `root` with its own name as its bytes; their paths, in the order given."""
    for name in names:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(name)
    return [str(root / name) for name in names]


def _fa2_tree(tmp_path: Path, *kernels: str) -> tuple[Path, list[str]]:
    csrc = tmp_path / "csrc"
    _files(csrc / "flash_attn" / "src", "b.h", "a.h", "not_a_header.txt")
    return csrc, _files(csrc / "flash_attn", "flash_api.cpp") + _files(csrc / "flash_attn" / "src", *kernels)


def _fa3_tree(tmp_path: Path, *kernels: str) -> tuple[Path, list[str], list[str]]:
    tree = tmp_path / "tree"
    headers = sorted(_files(tree / "hopper", "b.h", "a.h")) + _files(tree / "hopper", "c.hpp")
    _files(tree / "hopper", "not_a_header.txt")
    sources = _files(tree / "hopper", "flash_api.cpp", "fa3_torch_lib_verity.cpp", "flash_prepare_scheduler.cu")
    return tree, sources, headers + _files(tree / "hopper" / "instantiations", *kernels)


def _record(build_dir: Path) -> dict[str, Any]:
    info: dict[str, Any] = json.loads((build_dir / "build_info.json").read_text())
    return info


def _flags(defines: list[str]) -> list[str]:
    return ["-D" + d for d in defines]


def test_fa2_build_compiles_the_split_kernels_of_each_head_dim_and_records_what_it_built(
        toolkit: Toolkit, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Two head dims and one define, the rest by default: the sources are the tree's `flash_api.cpp`, this directory's
    registration and stubs, then four split kernels a head dim; no `-U` flag reaches nvcc; the record names every input."""
    torch = pytest.importorskip("torch")
    kernels = [f"flash_fwd_split{align}_hdim{hd}_bf16{causal}_sm80.cu"
               for hd in (64, 128) for align, causal in (("", "_causal"), ("_align", ""), ("", ""), ("_align", "_causal"))]
    csrc, tree_sources = _fa2_tree(tmp_path, *kernels)
    src, build = csrc / "flash_attn" / "src", tmp_path / "build"
    mine = Path(FA2.__file__).parent
    sources = [tree_sources[0], str(mine / "fa2_torch_lib_verity.cpp"), str(mine / "fa2_stubs.cu"), *tree_sources[1:]]

    FA2.main(FA2.Options(tree=str(csrc), name=NAME, build_dir=str(build), define=["VERITY_TAP=3"], hdims="128,64",
                         arch="8.0", tree_commit="a commit"))

    defines = ["FLASHATTENTION_DISABLE_BACKWARD", "FLASHATTENTION_DISABLE_DROPOUT", "FLASHATTENTION_DISABLE_PYBIND",
               "USE_CUDA", "TORCH_TARGET_VERSION=0x020A000000000000ULL", "CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1",
               "VERITY_TAP=3", "VERITY_HAVE_SPLIT_HDIM64_BF16", "VERITY_HAVE_SPLIT_HDIM128_BF16"]
    nvcc_flags = ["-O3", "-DNDEBUG", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda",
                  "--use_fast_math", *_flags(defines), "-Xptxas", "-v"]
    so = str(build / (NAME + ".so"))
    assert toolkit.loads == [{
        "name": NAME, "sources": sources, "extra_cflags": ["-O3", "-DNDEBUG", "-std=c++20", *_flags(defines)],
        "extra_cuda_cflags": nvcc_flags, "build_directory": str(build), "verbose": True, "is_python_module": False,
        "with_cuda": True,
        "extra_include_paths": [str(csrc / "flash_attn"), str(src), str(csrc / "common"), str(csrc / "cutlass" / "include")],
    }]
    captured = {"capture_output": True, "text": True}
    assert toolkit.programs == [([str(toolkit.home / "bin" / "nvcc"), "--version"], captured),
                                ([str(toolkit.home / "bin" / "cuobjdump"), "--dump-resource-usage", so], captured)]
    written = {name: settings.raw(name) for name in SET_BY_A_BUILD}
    assert written == dict(zip(SET_BY_A_BUILD, (str(toolkit.home), "7", "8.0"), strict=True))

    info = _record(build)
    seconds = info.pop("compile_seconds")
    assert isinstance(seconds, float) and seconds >= 0
    assert info == {
        "name": NAME, "tree": str(csrc), "tree_commit": "a commit", "sources": sources,
        "source_sha256": {s: FA2.sha256(s) for s in sources},
        "header_sha256": {h: FA2.sha256(h) for h in (str(src / "a.h"), str(src / "b.h"))},
        "defines": defines, "cxx_flags": ["-O3", "-DNDEBUG", "-std=c++20", *_flags(defines)], "nvcc_flags": nvcc_flags,
        "arch": "8.0", "hdims": [64, 128], "nvcc_version": ["Cuda compilation tools, release 0.0", "Build a.b"],
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_home": str(toolkit.home), "max_jobs": "7",
        "so": so, "so_sha256": FA2.sha256(so), "so_bytes": len(b"an extension"), "ops": ["verity_tap_mode"],
    }
    assert list(info["header_sha256"]) == [str(src / "a.h"), str(src / "b.h")]
    assert (build / "resource_usage.txt").read_text() == "Fatbin elf code:\na warning\n"
    assert capsys.readouterr().out == f"BUILD-OK {NAME} {seconds} s {so} ['verity_tap_mode']\n"


def test_fa3_build_compiles_the_paged_kernel_of_each_head_dim_links_the_driver_stub_and_records_what_it_built(
        toolkit: Toolkit, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """One head dim and no define: the sources are the patched tree's three, this directory's stubs, then one paged kernel
    a head dim; the link takes the toolkit's `libcuda` stub; the record names every input, `.h` and `.hpp` headers both."""
    torch = pytest.importorskip("torch")
    tree, tree_sources, rest = _fa3_tree(tmp_path, "flash_fwd_hdim128_bf16_paged_sm90.cu")
    headers, kernel = rest[:3], rest[3]
    hop, build = tree / "hopper", tmp_path / "build"
    sources = [*tree_sources, str(Path(FA3.__file__).parent / "fa3_stubs.cu"), kernel]

    FA3.main(FA3.Options(tree=str(tree), name=NAME, build_dir=str(build), hdims="128", tree_commit="a commit"))

    defines = [
        "FLASHATTENTION_DISABLE_BACKWARD", "FLASHATTENTION_DISABLE_DROPOUT", "FLASHATTENTION_DISABLE_UNEVEN_K",
        "FLASHATTENTION_DISABLE_PYBIND", "FLASHATTENTION_VARLEN_ONLY", "FLASHATTENTION_PACKGQA_ONLY",
        "FLASHATTENTION_DISABLE_CLUSTER", "FLASHATTENTION_DISABLE_SM8x", "FLASHATTENTION_DISABLE_APPENDKV",
        "CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "CUTLASS_ENABLE_GDC_FOR_SM90", "TORCH_TARGET_VERSION=0x020A000000000000ULL",
        "USE_CUDA", "CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1", "FLASHATTENTION_DISABLE_FP16", "FLASHATTENTION_DISABLE_FP8",
        "FLASHATTENTION_DISABLE_SOFTCAP", "FLASHATTENTION_DISABLE_LOCAL", "FLASHATTENTION_DISABLE_SPLIT",
        "FLASHATTENTION_DISABLE_HDIM96", "FLASHATTENTION_DISABLE_HDIM192", "FLASHATTENTION_DISABLE_HDIM256",
        "FLASHATTENTION_DISABLE_HDIMDIFF64", "FLASHATTENTION_DISABLE_HDIMDIFF192", "VERITY_HAVE_FA3_HDIM128_BF16"]
    nvcc_flags = ["-O3", "-DNDEBUG", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda",
                  "--use_fast_math", *_flags(defines), "-Xptxas", "-v"]
    ldflags = [f"-L{toolkit.home / 'lib64' / 'stubs'}", "-lcuda"]
    so = str(build / (NAME + ".so"))
    assert toolkit.loads == [{
        "name": NAME, "sources": sources, "extra_cflags": ["-O3", "-DNDEBUG", "-std=c++20", *_flags(defines)],
        "extra_cuda_cflags": nvcc_flags, "extra_ldflags": ldflags, "build_directory": str(build), "verbose": True,
        "is_python_module": False, "with_cuda": True,
        "extra_include_paths": [str(hop), str(tree / "csrc" / "common"), str(tree / "csrc" / "cutlass" / "include")],
    }]
    captured = {"capture_output": True, "text": True}
    assert toolkit.programs == [([str(toolkit.home / "bin" / "nvcc"), "--version"], captured),
                                ([str(toolkit.home / "bin" / "cuobjdump"), "--dump-resource-usage", so], captured)]
    written = {name: settings.raw(name) for name in SET_BY_A_BUILD}
    assert written == dict(zip(SET_BY_A_BUILD, (str(toolkit.home), "7", "9.0a"), strict=True))

    info = _record(build)
    seconds = info.pop("compile_seconds")
    assert isinstance(seconds, float) and seconds >= 0
    assert info == {
        "name": NAME, "tree": str(tree), "tree_commit": "a commit", "sources": sources,
        "source_sha256": {s: FA3.sha256(s) for s in sources}, "header_sha256": {h: FA3.sha256(h) for h in headers},
        "defines": defines, "cxx_flags": ["-O3", "-DNDEBUG", "-std=c++20", *_flags(defines)], "nvcc_flags": nvcc_flags,
        "arch": "9.0a", "hdims": [128], "nvcc_version": ["Cuda compilation tools, release 0.0", "Build a.b"],
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_home": str(toolkit.home), "max_jobs": "7",
        "ldflags": ldflags, "so": so, "so_sha256": FA3.sha256(so), "so_bytes": len(b"an extension"),
        "ops": ["verity_tap_mode"],
    }
    assert list(info["header_sha256"]) == headers
    assert (build / "resource_usage.txt").read_text() == "Fatbin elf code:\na warning\n"
    assert capsys.readouterr().out == f"BUILD-OK {NAME} {seconds} s {so} ['verity_tap_mode']\n"


@pytest.mark.parametrize(("build", "home"), [("fa2", "/usr/local/cuda-12.9"), ("fa3", "/usr/local/cuda")], ids=["fa2", "fa3"])
def test_a_build_names_its_own_toolkit_and_job_count_where_the_environment_names_none(
        build: str, home: str, toolkit: Toolkit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without `CUDA_HOME` and `MAX_JOBS` a build sets its own, starts that toolkit's programs and records both; with
    `--no-ptxas-v` nvcc gets no `-Xptxas -v`; and where `load` leaves no extension the record says so with two nulls."""
    monkeypatch.delenv("CUDA_HOME")
    monkeypatch.delenv("MAX_JOBS")
    toolkit.extension = None
    out = tmp_path / "build"
    if build == "fa2":
        csrc, _ = _fa2_tree(tmp_path, "flash_fwd_hdim64_bf16_causal_sm80.cu")
        FA2.main(FA2.Options(tree=str(csrc), name=NAME, build_dir=str(out), files="fwd_causal", no_ptxas_v=True))
    else:
        tree, _, _ = _fa3_tree(tmp_path, "flash_fwd_hdim64_bf16_paged_sm90.cu", "flash_fwd_hdim128_bf16_paged_sm90.cu")
        FA3.main(FA3.Options(tree=str(tree), name=NAME, build_dir=str(out), no_ptxas_v=True))

    assert [argv[0] for argv, _ in toolkit.programs] == [os.path.join(home, "bin", "nvcc"), os.path.join(home, "bin", "cuobjdump")]
    written = {name: settings.raw(name) for name in SET_BY_A_BUILD[:2]}
    assert written == {"CUDA_HOME": home, "MAX_JOBS": "32"}
    info = _record(out)
    assert (info["cuda_home"], info["max_jobs"], info["so_sha256"], info["so_bytes"]) == (home, "32", None, None)
    (load,) = toolkit.loads
    assert "-Xptxas" not in load["extra_cuda_cflags"] and load["extra_cuda_cflags"] == info["nvcc_flags"]
    if build == "fa2":
        assert info["defines"][-1] == "VERITY_HAVE_FWD_HDIM64_BF16" and info["hdims"] == [64]
    else:
        assert info["defines"][-2:] == ["VERITY_HAVE_FA3_HDIM64_BF16", "VERITY_HAVE_FA3_HDIM128_BF16"]
        assert load["extra_ldflags"] == [f"-L{os.path.join(home, 'lib64', 'stubs')}", "-lcuda"]


@pytest.mark.parametrize(("build", "hdims", "files", "refusal"), [
    ("fa2", "32", "split_causal", r"\[32\]"),
    ("fa2", "", "split_causal", r"\[\]"),
    ("fa2", "64,96", "split_causal", r"flash_fwd_split_hdim96_bf16_causal_sm80\.cu"),
    ("fa2", "128", "fwd_causal", "non-split fwd kernels are only wired for hdim 64"),
    ("fa3", "96", "", r"\[96\]"),
    ("fa3", "64,128", "", r"flash_fwd_hdim128_bf16_paged_sm90\.cu"),
], ids=["fa2-head-dim-32", "fa2-no-head-dim", "fa2-no-kernel-for-96", "fa2-unsplit-128", "fa3-head-dim-96", "fa3-no-kernel-for-128"])
def test_a_build_refuses_a_head_dim_it_does_not_know_or_a_kernel_the_tree_lacks_before_it_starts_a_program(
        build: str, hdims: str, files: str, refusal: str, toolkit: Toolkit, tmp_path: Path) -> None:
    """Each refusal is an `AssertionError` that names the head dims or the missing file, raised before `nvcc --version`
    and before `load`; no build directory is made. (`files` is the FA2 build's `--files`; the FA3 build has none.)"""
    out = tmp_path / "build"
    with pytest.raises(AssertionError, match=refusal):
        if build == "fa2":
            csrc, _ = _fa2_tree(tmp_path, "flash_fwd_split_hdim64_bf16_causal_sm80.cu", "flash_fwd_hdim128_bf16_causal_sm80.cu")
            FA2.main(FA2.Options(tree=str(csrc), name=NAME, build_dir=str(out), hdims=hdims, files=files))
        else:
            tree, _, _ = _fa3_tree(tmp_path, "flash_fwd_hdim64_bf16_paged_sm90.cu")
            FA3.main(FA3.Options(tree=str(tree), name=NAME, build_dir=str(out), hdims=hdims))
    assert toolkit.programs == [] and toolkit.loads == [] and not out.exists()
