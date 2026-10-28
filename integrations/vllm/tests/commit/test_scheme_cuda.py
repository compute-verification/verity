"""The CUDA copies of the ``vllm-v1`` rules hash the conformance vectors' bytes (``verity.ml.commitments.vllm_v1``).

Three copies, each checked on every vector its rule covers:

* ``native_tree.cu`` -- the production ``verity_native_collect`` extension, through its extern "C" launchers: chunk leaves of a whole
  step and of a ``ci_base`` window, node/lift levels, trees and paths, the run and weights folds; and through ``StepHasher``, whole trees
  and window subtrees on both sides of the bound under which a tree's narrow top is one launch;
* ``hidden_gpu_tree.cu`` -- ``hidden_gpu.ext()``: chunk leaves, node/lift levels, trees and paths, the run and weights folds;
* ``verity_tap.h`` at ``VERITY_TAP=4`` -- the FA2 thread leaf, compiled into a one-thread harness against the pod's FA2 tree.

The chunk kernels hard-wire header words 8/9 to 0, so they only reproduce legacy-tile chunk vectors; the device has no empty tree and
no root binding (those are :mod:`verity_vllm.commit.scheme` on the host, which binds the device folds here).
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.pod
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the CUDA copies need a GPU", allow_module_level=True)

from verity.ml.commitments.vllm_v1 import vectors as VEC  # noqa: E402
from verity_vllm.commit import scheme  # noqa: E402

V = VEC.load()
H = bytes.fromhex
PKG = Path(__file__).resolve().parents[2] / "verity_vllm"
FA2_CSRC = Path(os.environ.get("VERITY_FA2_CSRC", "/workspace/cp/fa2/trees/tap8/csrc"))
LEGACY_CHUNKS = [c for c in V["chunk_leaf"] if (c["fields"]["BN"], c["fields"]["D"]) == (0, 0)]
TREES = [t for t in V["trees"] if t["leaf_count"] > 0]
NODE = next(h for h in V["hash"] if h["tag"] == "verity-vllm/node/v1")
LIFT = next(h for h in V["hash"] if h["tag"] == "verity-vllm/lift/v1")
RUNS = [r for r in V["roots"] if r["kind"] == "run" and r["inputs"]["step_roots"]]
WEIGHTS = next(r for r in V["roots"] if r["kind"] == "weights")


def _rows(ds: list[bytes]) -> torch.Tensor:
    return torch.from_numpy(np.frombuffer(b"".join(ds), dtype=np.uint32).reshape(-1, 8).view(np.int32).copy()).cuda()


def _digests(t: torch.Tensor) -> list[bytes]:
    b = t.cpu().numpy().astype(np.uint32).tobytes()
    return [b[32 * i:32 * i + 32] for i in range(len(b) // 32)]


def _stream(c: dict) -> tuple[np.ndarray, int]:
    """The vector's chunk at its index in an otherwise zero stream, and that index."""
    f = c["fields"]
    chunk = np.frombuffer(H(c["chunk"]), dtype="<u4")
    return np.concatenate([np.zeros(f["chunk_index"] * f["chunk_words"], dtype=np.uint32), chunk]), f["chunk_index"]


def _paths(levels: list[list[bytes]], n: int) -> list[list[str | None]]:
    return [[None if p is None else p.hex() for p in scheme.path_in_levels(levels, i)] for i in range(n)]


def _run_root(r: dict, fold: bytes) -> bytes:
    i = r["inputs"]
    assert fold == scheme.fold([H(s) for s in i["step_roots"]])
    return scheme.run_root(H(i["program_digest"]), H(i["geo_digest"]), [H(s) for s in i["step_roots"]])


def _check_tree_copy(chunk_leaves, levels, one_level) -> None:
    for c in LEGACY_CHUNKS:
        stream, ci = _stream(c)
        assert chunk_leaves(stream, c["fields"], None)[ci].hex() == c["leaf"]
    for t in TREES:
        lv = levels([H(x) for x in t["leaves"]])
        assert lv[-1][0].hex() == t["root"]
        assert _paths(lv, t["leaf_count"]) == t["paths"]
    assert one_level([H(p) for p in NODE["parts"]])[0].hex() == NODE["digest"]
    assert one_level([H(p) for p in LIFT["parts"]])[0].hex() == LIFT["digest"]
    for r in RUNS:
        assert _run_root(r, levels([H(s) for s in r["inputs"]["step_roots"]])[-1][0]).hex() == r["root"]
    i = WEIGHTS["inputs"]
    fold = levels([H(x) for x in i["tensor_roots"]])[-1][0]
    assert scheme.weights_root(H(i["geo_digest"]), H(i["names_digest"]), [H(x) for x in i["tensor_roots"]]).hex() == WEIGHTS["root"]
    assert fold == scheme.fold([H(x) for x in i["tensor_roots"]])


class _NativeTree:
    """ctypes over ``native_tree.cu``'s extern "C" launchers (device pointers of torch tensors, the default stream)."""

    def __init__(self, so: str) -> None:
        lib = ctypes.CDLL(so)
        u32, cl, ci, vp = ctypes.c_uint32, ctypes.c_long, ctypes.c_int, ctypes.c_void_p
        self.base, self.level = lib.verity_chunk_leaves_launch_base, lib.verity_tree_level_launch
        self.base.restype = self.level.restype = ci
        self.base.argtypes = [vp, vp, cl, ci, u32, u32, u32, u32, u32, cl, vp]
        self.level.argtypes = [vp, vp, cl, vp]

    def chunk_leaves(self, stream: np.ndarray, f: dict, ci_base: int | None) -> list[bytes]:
        s = torch.from_numpy(stream.view(np.int32).copy()).cuda()
        n = stream.size // f["chunk_words"]
        out = torch.zeros((n, 8), dtype=torch.int32, device="cuda")
        rc = self.base(s.data_ptr(), out.data_ptr(), n, f["chunk_words"], f["launch_tag"], f["src_mask"], f["HB"], f["M"], f["NB"],
                       ci_base or 0, None)
        torch.cuda.synchronize()
        assert rc == 0
        return _digests(out)

    def one_level(self, ds: list[bytes]) -> list[bytes]:
        out = torch.zeros(((len(ds) + 1) // 2, 8), dtype=torch.int32, device="cuda")
        rc = self.level(_rows(ds).data_ptr(), out.data_ptr(), len(ds), None)
        torch.cuda.synchronize()
        assert rc == 0
        return _digests(out)

    def levels(self, leaves: list[bytes]) -> list[list[bytes]]:
        lv = [leaves]
        while len(lv[-1]) > 1:
            lv.append(self.one_level(lv[-1]))
        return lv


def test_native_tree() -> None:
    from verity_vllm.commit.committer import native_collect as NC

    nt = _NativeTree(NC.ext().__file__)
    _check_tree_copy(nt.chunk_leaves, nt.levels, nt.one_level)
    for c in LEGACY_CHUNKS:
        f = c["fields"]
        window = np.frombuffer(H(c["chunk"]), dtype="<u4").copy()
        assert nt.chunk_leaves(window, f, f["chunk_index"])[0].hex() == c["leaf"]


@pytest.mark.parametrize("n", [1, 2, 3, 1023, 1024, 1025, 2049, 3589, 4097, 65537, 131075])
def test_step_hasher_trees_are_fold_levels(n: int) -> None:
    """``StepHasher.tree``, ``tree_window`` and ``tree_window_top`` (a level of more than 65536 nodes a launch, the narrower ones ten a
    launch, each block folding an aligned slice of 1024 nodes) give every level of ``scheme.fold_levels``: odd widths, a partial last
    slice of one node or of an odd count, a second launch over the first's tops, one and two wide levels of odd width before the
    narrow ones, and windows of 2048 leaves whose partial last one climbs by lifts."""
    from verity_vllm.commit.committer import native_collect as NC

    hasher, st = NC.ext().StepHasher(), torch.cuda.current_stream().cuda_stream
    rng = np.random.default_rng(n)
    leaves = [rng.bytes(32) for _ in range(n)]
    want = scheme.fold_levels(leaves)
    levels, root, _ = hasher.tree(_rows(leaves), st, True, torch.empty(0, dtype=torch.uint8))
    assert [_digests(t) for t in levels] == want and root == want[-1][0]
    j = 11
    n_levels = min(j, len(want) - 1) + 1
    tops = []
    for k in range(-(-n // (1 << j))):
        part = leaves[k << j:(k + 1) << j]
        dsts = [torch.empty((-(-len(part) // (1 << L)), 8), dtype=torch.int32, pin_memory=True) for L in range(n_levels)]
        tops.append(hasher.tree_window(_rows(part), n_levels, st, dsts))
        assert [_digests(d) for d in dsts] == [want[L][(k << j) >> L:][:d.shape[0]] for L, d in enumerate(dsts)], k
        assert _digests(tops[-1]) == want[n_levels - 1][k:k + 1]
        assert torch.equal(hasher.tree_window_top(_rows(part), n_levels, st), tops[-1])
    assert hasher.tree(torch.cat(tops), st, False, torch.empty(0, dtype=torch.uint8))[1] == want[-1][0]


def test_hidden_gpu_tree() -> None:
    sys.path.insert(0, str(PKG / "commit" / "committer" / "hidden_gpu_src"))
    import hidden_gpu as HG

    e = HG.ext()

    def chunk_leaves(stream: np.ndarray, f: dict, _ci_base: None) -> list[bytes]:
        s = torch.from_numpy(stream.view(np.int32).copy()).cuda()
        return _digests(e.chunk_leaves(s, f["chunk_words"], f["launch_tag"], f["src_mask"], f["HB"], f["M"], f["NB"]))

    def levels(leaves: list[bytes]) -> list[list[bytes]]:
        return [_digests(t) for t in e.tree_levels(_rows(leaves))]

    def one_level(ds: list[bytes]) -> list[bytes]:
        if len(ds) == 1:                          # tree_level refuses one child: a lone child's lift is level 1's tail of [x, x, child]
            x = hashlib.sha256(b"x").digest()
            return [levels([x, x, ds[0]])[1][1]]
        return _digests(e.tree_level(_rows(ds)))

    _check_tree_copy(chunk_leaves, levels, one_level)


HARNESS = r"""
#include <cuda_runtime.h>
#include <cstdint>
#include <cute/tensor.hpp>
using namespace cute;   // FA2's kernel_traits.h does this at global scope before utils.h, which relies on it
#include "namespace_config.h"
namespace FLASH_NAMESPACE {
struct Verity_tap { void *acc_ptr; long acc_cap_words; float *diag_ptr; int mode; int H, M, N, NB; int tag; int BN, D; };
}
#define VERITY_TAP 4
#define VERITY_SHA_SRC @MASK@
#include "@TAP@"
using namespace FLASH_NAMESPACE;

__global__ void h1_leaf_kernel(const uint32_t *words, int n_words, int slab, int m_block, int n_block, int tidx, int seqlen_q, int seqlen_k,
                               int tag, int nbmax, int ok0, int ok1, uint32_t *acc, int NB) {
    Verity_tap vt{};
    vt.acc_ptr = acc; vt.NB = NB; vt.tag = tag;
    Verity_sha_ctx c(vt, 0, 1, slab, m_block, tidx, 4, seqlen_q, seqlen_k, nbmax);
    c.ok0 = ok0 != 0; c.ok1 = ok1 != 0;
    c.init();
    c.header(n_block);
    for (int b = 0; b < n_words / 16; ++b) {
        for (int i = 0; i < 16; ++i) c.m[i] = vsha_bswap(words[b * 16 + i]);
        c.compress_m();
    }
    c.finish_store(n_block);
}

extern "C" int h1_leaf(const uint32_t *words_h, int n_words, int slab, int m_block, int n_block, int tidx, int seqlen_q, int seqlen_k, int tag,
                       int nbmax, int ok0, int ok1, uint8_t *leaf_out) {
    const int NB = n_block + 1;
    const long idx = ((((long)slab * 1 + m_block) * NB + n_block) * 128 + tidx) * 8;
    uint32_t *dw, *dacc;
    cudaMalloc(&dw, sizeof(uint32_t) * (n_words > 0 ? n_words : 1));
    cudaMalloc(&dacc, sizeof(uint32_t) * (idx + 8));
    cudaMemset(dacc, 0, sizeof(uint32_t) * (idx + 8));
    cudaMemcpy(dw, words_h, sizeof(uint32_t) * n_words, cudaMemcpyHostToDevice);
    h1_leaf_kernel<<<1, 1>>>(dw, n_words, slab, m_block, n_block, tidx, seqlen_q, seqlen_k, tag, nbmax, ok0, ok1, dacc, NB);
    int rc = (int)cudaDeviceSynchronize();
    cudaMemcpy(leaf_out, dacc + idx, 32, cudaMemcpyDeviceToHost);
    cudaFree(dw); cudaFree(dacc);
    return rc;
}
"""


def test_fa2_tap_thread_leaf(tmp_path: Path) -> None:
    if not (FA2_CSRC / "flash_attn" / "src").is_dir():
        pytest.skip(f"no FA2 source tree at {FA2_CSRC} (the tap tree of pod_fa2_tap.sh; VERITY_FA2_CSRC names another)")
    from torch.utils.cpp_extension import CUDA_HOME, include_paths

    tap = PKG / "acquire" / "fa2_tap_src" / "verity_tap.h"
    nvcc = os.path.join(CUDA_HOME or "/usr/local/cuda", "bin", "nvcc")
    arch = "sm_%d%d" % torch.cuda.get_device_capability(0)
    incs = [FA2_CSRC / "flash_attn" / "src", FA2_CSRC / "flash_attn", FA2_CSRC / "common", FA2_CSRC / "cutlass" / "include", *include_paths()]
    libs: dict[int, ctypes.CDLL] = {}
    for tv in V["thread_leaf"]:
        f = tv["fields"]
        mask = f["src_mask"]
        if mask not in libs:
            cu, so = tmp_path / f"h1_{mask}.cu", tmp_path / f"h1_{mask}.so"
            cu.write_text(HARNESS.replace("@MASK@", str(mask)).replace("@TAP@", str(tap)))
            cmd = [nvcc, "-O3", "-DNDEBUG", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math", f"-arch={arch}",
                   "-Xcompiler", "-fPIC", "-shared", *[f"-I{i}" for i in incs], str(cu), "-o", str(so)]
            p = subprocess.run(cmd, capture_output=True, text=True)
            assert p.returncode == 0, p.stderr[-3000:]
            libs[mask] = ctypes.CDLL(str(so))
        words = np.frombuffer(H(tv["words"]), dtype="<u4")
        assert words.size % 16 == 0
        leaf = (ctypes.c_uint8 * 32)()
        tag = f["launch_tag"] - (1 << 32) if f["launch_tag"] >= (1 << 31) else f["launch_tag"]
        rc = libs[mask].h1_leaf((ctypes.c_uint32 * words.size)(*map(int, words)), int(words.size), f["slab"], f["m_block"], f["n_block"],
                                f["tidx"], f["seqlen_q"], f["seqlen_k"], ctypes.c_int(tag), f["n_block_max"], int(f["ok0"]), int(f["ok1"]), leaf)
        assert rc == 0
        assert bytes(leaf).hex() == tv["leaf"]
        assert scheme.thread_leaf(scheme.thread_header(f["src_mask"], f["slab"], f["m_block"], f["n_block"], f["tidx"], f["seqlen_q"],
                                                       f["seqlen_k"], f["launch_tag"], f["n_block_max"], f["ok0"], f["ok1"])
                                  + words.tobytes()).hex() == tv["leaf"]
