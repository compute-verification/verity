"""The CUDA copy of the ``vllm-v1`` thread leaf hashes the conformance vectors' bytes (``verity.ml.commitments.vllm_v1``).

One copy, checked on every thread-leaf vector: ``verity_tap.h`` at ``VERITY_TAP=4`` -- the FA2 thread leaf, compiled into a one-thread
harness against the pod's FA2 tree, beside :mod:`verity_vllm.commit.scheme`'s host rule for the same vector.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
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
