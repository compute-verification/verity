"""A ship build's cubin doesn't follow where its sources sit, with CUDA's own nvcc and no GPU: this directory's kernel
(s8_wgmma.cu) compiled as build.sh compiles it (ship.sh's frame, its cubin flags, ship_nvcc.sh) from two copies of the tree
placed differently is one cubin, byte for byte. In the second copy the tree's root has another length, the kernel's
directory another name and depth, and its header sits in a directory of its own, with the #include respelled as a move
respells it. The kernel alone could pass by following no path at all, so a probe shows the comparison can fail: nvcc
names a kernel of internal linkage by a hash of its source's path, so nvcc itself gives the probe two cubins from the two
copies and ship_nvcc.sh one. And a header that uses __FILE__ is refused by name before the device compiler runs.

These need what a ship build needs: a kernel that lets a process turn address randomization off, and the toolkit (nvcc
12.8 or later at $CUDA, /usr/local/cuda-13.0 or /usr/local/cuda, found by ship.sh's own rule). Where either is missing
they skip, with ship.sh's reason.
test_ship_nvcc.py holds what ship_nvcc.sh hands the device compiler, and its refusals, on every machine. The frame builds
under a directory of its own in /tmp, whose path has one length, and removes it."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

KERNEL = Path(__file__).resolve().parents[1]
FRAME = KERNEL.parent
#: the frame's files, which keep their place under a copy's root (ship.sh finds the root from its own)
FRAME_FILES = ("ship.sh", "ship_nvcc.sh", "ship_clock.c")
TOOLKITS = ("/usr/local/cuda-13.0", "/usr/local/cuda")

#: the toolkit a ship build would compile with, by ship.sh's own rule (ship.sh, then the toolkits to try): its path, or
#: nothing; ship.sh's frame refuses first where it can't build
FIND = """set -euo pipefail
. "$1"
shift
ship_find_cuda "$@"
ship_nvcc_ok "$CUDA" && echo "$CUDA"
"""

#: build.sh's cubin step for one source: ROOT, the source's directory from it, `ship` (ship_nvcc.sh) or `nvcc` (the
#: toolkit's own), the source's name, the cubin and the toolkit
DRIVER = """set -euo pipefail
. "$1/benchmarks/torch_gpu/ship.sh"
CUDA=$6
ship_cuda
ship_cubin_flags sm_90a
ship_stage "$2"
case $3 in ship) cc=$SHIP_NVCC ;; nvcc) cc=$CUDA/bin/nvcc ;; esac
"$cc" "${FLAGS[@]}" -o "$5" "$SRC/$2/$4"
"""

#: where a copy puts the tree's root, the kernel's directory and the header beside the kernel
AS_IS = ("as_is", "benchmarks/torch_gpu/s8_linear", "")
MOVED = ("moved_to_a_root_of_another_length", "kernels/exact_linear_at_another_place/and_depth/s8", "headers/of_its_own/")

PROBE_CU = """#include "probe.cuh"
namespace {
__global__ void probe(const unsigned *in, unsigned *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = mix(in[i]);
}
}  // namespace
void launch(const unsigned *in, unsigned *out, int n) { probe<<<(n + 31) / 32, 32>>>(in, out, n); }
"""
PROBE_CUH = "__device__ inline unsigned mix(unsigned x) { return x * 2654435761u + 1u; }\n"
USES_FILE = 'inline const char *where() { return __FILE__; }\n'


def toolkit(tmp: Path) -> tuple[str, str]:
    """The toolkit a ship build finds here, or "" and why it finds none."""
    (tmp / "find.sh").write_text(FIND)
    r = subprocess.run(["bash", str(tmp / "find.sh"), str(FRAME / "ship.sh"), *TOOLKITS], capture_output=True, text=True,
                       check=False)
    found = r.stdout.strip()
    return found, "" if found else r.stderr.strip() or f"no nvcc 12.8 or later at $CUDA or {' or '.join(TOOLKITS)}"


@pytest.fixture(scope="session")
def cuda(tmp_path_factory: pytest.TempPathFactory) -> str:
    found, why_not = toolkit(tmp_path_factory.mktemp("toolkit"))
    if not found:
        pytest.skip(f"{why_not}: a ship build's placements are compiled where a ship builds")
    return found


def copy(tmp: Path, where: tuple[str, str, str], source: str, text: str, header: str, header_text: str) -> tuple[Path, str]:
    """A copy of the tree under tmp placed as `where` says, holding the frame and one source with its header, the source's
    #include spelled for the header's place; the copy's root and the source's directory from it."""
    root, own, beside = tmp / where[0], where[1], where[2]
    (root / "benchmarks/torch_gpu").mkdir(parents=True)
    for f in FRAME_FILES:
        shutil.copy(FRAME / f, root / "benchmarks/torch_gpu" / f)
    (root / own / beside).mkdir(parents=True)
    assert text.count(f'#include "{header}"') == 1
    (root / own / source).write_text(text.replace(f'#include "{header}"', f'#include "{beside}{header}"'))
    (root / own / beside / header).write_text(header_text)
    (root / "build.sh").write_text(DRIVER)
    return root, own


def build(cuda: str, root: Path, own: str, source: str, cc: str = "ship") -> tuple[subprocess.CompletedProcess, Path]:
    out = root / f"{cc}.cubin"
    r = subprocess.run(["bash", str(root / "build.sh"), str(root), own, cc, source, str(out), cuda], capture_output=True,
                       text=True, check=False)
    return r, out


def cubins(cuda: str, tmp: Path, source: str, text: str, header: str, header_text: str, cc: str = "ship") -> list[bytes]:
    """The source's cubin from each of the two copies."""
    out = []
    for where in (AS_IS, MOVED):
        root, own = copy(tmp / cc, where, source, text, header, header_text)
        r, cubin = build(cuda, root, own, source, cc)
        assert r.returncode == 0, r.stderr[-4000:]
        out.append(cubin.read_bytes())
        assert out[-1][:4] == b"\x7fELF"
    return out


@pytest.mark.slow
def test_the_kernel_is_one_cubin_from_two_places_of_the_tree(cuda: str, tmp_path: Path) -> None:
    a, b = cubins(cuda, tmp_path, "s8_wgmma.cu", (KERNEL / "s8_wgmma.cu").read_text(), "s8_acc.h",
                  (KERNEL / "s8_acc.h").read_text())
    assert a == b


def test_a_probe_nvcc_builds_differently_at_the_two_places_is_one_cubin_too(cuda: str, tmp_path: Path) -> None:
    a, b = cubins(cuda, tmp_path, "probe.cu", PROBE_CU, "probe.cuh", PROBE_CUH, cc="nvcc")
    assert a != b, ("nvcc itself gives this probe one cubin at both places, so the comparisons here show nothing: give "
                    "the probe something nvcc makes follow its path")
    a, b = cubins(cuda, tmp_path, "probe.cu", PROBE_CU, "probe.cuh", PROBE_CUH)
    assert a == b


def test_a_header_that_uses_file_is_refused(cuda: str, tmp_path: Path) -> None:
    root, own = copy(tmp_path, MOVED, "probe.cu", PROBE_CU, "probe.cuh", PROBE_CUH + USES_FILE)
    r, cubin = build(cuda, root, own, "probe.cu")
    assert r.returncode != 0 and not cubin.exists()
    assert "the text cicc compiles names a file" in r.stderr
    assert re.search(r"take it out of \S*headers/of_its_own/probe\.cuh line 2: ", r.stderr), r.stderr[-2000:]
