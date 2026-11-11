"""The device witness's host-side code in ``cuda/prove_circuit.cuh``, compiled on the host with the CUDA calls stood in for:
``fc_sha_runs``'s runs and ``fc_rows_words``'s word assembly against ``fc_sha_rows``'s one bit a row, and ``host_copies``'s 1D, 2D
and 3D copies of the host slots against ``fc_host_slots``'s scatter. Both must give the same words of z, a, b; the device's
own run (``prover_is_deterministic`` pinned to main's transcripts) covers one statement's layout, these every layout and
plan the generators draw, and that a slot range's slots in every block take one copy a word kind."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "cuda" / "prove_circuit.cuh"
CXX = shutil.which("g++") or shutil.which("c++")

pytestmark = pytest.mark.skipif(CXX is None, reason="no host C++ compiler (g++ or c++)")


def _extract(pattern: str) -> str:
    m = re.search(pattern, SOURCE.read_text(), re.S | re.M)
    assert m, f"{SOURCE.name} no longer has what {pattern!r} matches: update this test's extraction with the code"
    return m.group(0)


def _run(tmp_path: Path, program: str) -> str:
    src = tmp_path / "t.cpp"
    src.write_text(program)
    exe = tmp_path / "t"
    r = subprocess.run([CXX, "-O1", "-std=c++17", "-Wall", "-Wno-unused-variable", "-o", str(exe), str(src)], capture_output=True,
                       text=True, check=False)
    assert r.returncode == 0, f"the host build failed:\n{r.stderr[-4000:]}"
    r = subprocess.run([str(exe)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout[-4000:]
    return r.stdout


def _device_free(code: str) -> str:
    """A kernel or lambda body as host code: no qualifiers, launches as calls, one block of one thread."""
    code = re.sub(r"<<<.*?>>>", "", code).replace("#pragma unroll", "")
    code = code.replace("__global__ ", "").replace("__device__ __forceinline__ ", "static ").replace("__restrict__ ", "")
    for k, v in (("blockIdx.x", "0"), ("gridDim.x", "1"), ("threadIdx.x", "0"), ("blockDim.x", "1")):
        code = code.replace(k, v)
    return code


def test_sha_rows_runs_match_per_bit(tmp_path: Path) -> None:
    runs = _extract(r"^struct FcRuns \{.*?^\};\n") + _extract(r"^static void fc_sha_runs\(.*?^}\n")
    words = _extract(r"^__device__ __forceinline__ void fc_rows_words\(.*?^}\n")
    program = r"""
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <unordered_map>
#include <vector>
struct uint2 { uint32_t x, y; };
struct uint4 { uint32_t x, y, z, w; };
static uint2 make_uint2(uint32_t x, uint32_t y) { return {x, y}; }
static uint4 make_uint4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { return {a, b, c, d}; }
""" + runs + _device_free(words) + r"""
// fc_sha_rows's words: one plan entry and one tape bit a row
static void words_bits(const uint64_t* tape, long long n, long long s, const uint32_t* plan, int useful, int g, uint32_t out[2][8]) {
    for (int h = 0; h < 2; h++) for (int j = 0; j < 8; j++) {
        uint32_t x = 0; const int r0 = 32 * (8 * g + j);
        for (int i = 0; i < 32 && r0 + i < useful; i++) {
            const uint32_t q = plan[2 * (r0 + i) + h];
            x |= (q < 2 ? q : (uint32_t)(tape[(long long)((q - 2) >> 6) * n + s] >> ((q - 2) & 63)) & 1u) << i;
        }
        out[h][j] = x;
    }
}
int main() {
    std::mt19937_64 rng(7);
    long long words = 0;
    for (int trial = 0; trial < 400; trial++) {
        const int tape_words = 1 + (int)(rng() % 300), n = 1 + (int)(rng() % 5), useful = 1 + (int)(rng() % 3000);
        std::vector<uint32_t> plan(2 * (size_t)useful);
        const int mode = trial % 3;  // 0: rows drawn alone; 1: runs from any bit; 2: runs from a word's bit 0, as an addition's
        for (int h = 0; h < 2; h++)
            for (int i = 0; i < useful;) {
                const int kind = (int)(rng() % (mode == 0 ? 3 : 10));
                if (mode == 0 || kind >= 8) { plan[2 * i + h] = kind < 2 ? kind : 2 + (uint32_t)(rng() % (64ull * tape_words)); i++; continue; }
                const int len = 1 + (int)(rng() % 64);
                const uint32_t w = (uint32_t)(rng() % tape_words), b0 = mode == 2 ? 0 : (uint32_t)(rng() % 64);
                for (int t = 0; t < len && i < useful; t++, i++) plan[2 * i + h] = kind == 7 ? 1 : 2 + 64 * w + (b0 + t) % 64;
            }
        std::vector<uint64_t> tape((size_t)tape_words * n);
        for (auto& v : tape) v = rng();
        const int nrg = (useful + 255) / 256;
        FcRuns R;
        fc_sha_runs(plan.data(), useful, nrg, R);
        if (R.poff.size() != 16 * (size_t)nrg + 1 || R.goff.size() != (size_t)nrg + 1) { printf("trial %d: %zu offsets\n", trial, R.poff.size()); return 1; }
        for (int g = 0; g < nrg; g++) {
            const uint32_t nw = R.goff[g + 1] - R.goff[g];
            if ((int)nw > R.nwc) { printf("trial %d group %d: %u words, nwc %d\n", trial, g, nw, R.nwc); return 1; }
            for (uint32_t k = R.poff[16 * g]; k < R.poff[16 * g + 16]; k++)
                for (uint32_t c : {R.rq[k].y >> 16, R.rq[k].w >> 16})
                    if (c >= nw && c != (uint32_t)R.nwc) { printf("trial %d group %d: column %u of %u\n", trial, g, c, nw); return 1; }
        }
        for (int q = 0; q < 16 * nrg; q++)
            if ((R.poff[q + 1] - R.poff[q]) % 2) { printf("trial %d: word and operand %d has an odd count\n", trial, q); return 1; }
        // a block's 8 groups, its runs from base on and each group's columns as the kernel's shared memory holds them; the
        // columns past a group's own words hold other values, which no run may read
        for (int g0 = 0; g0 < nrg; g0 += 8) {
            const uint32_t base = R.poff[16 * g0], end = R.poff[16 * std::min(g0 + 8, nrg)];
            if (end - base > R.block_max) { printf("trial %d block %d: %u runs, block_max %u\n", trial, g0 / 8, end - base, R.block_max); return 1; }
            const std::vector<uint4> rs(R.rq.begin() + base, R.rq.begin() + end);
            for (int g = g0; g < std::min(g0 + 8, nrg); g++)
                for (int s = 0; s < n; s++) {
                    std::vector<uint64_t> col(R.nwc + 1);
                    for (auto& v : col) v = rng();
                    for (uint32_t i = R.goff[g]; i < R.goff[g + 1]; i++) col[i - R.goff[g]] = tape[(size_t)R.gw[i] * n + s];
                    col[R.nwc] = ~0ull;
                    uint32_t a[2][8] = {}, b[2][8];
                    fc_rows_words(col.data(), 1, R.poff.data(), rs.data(), base, g, a);
                    words_bits(tape.data(), n, s, plan.data(), useful, g, b);
                    for (int h = 0; h < 2; h++) for (int j = 0; j < 8; j++, words++)
                        if (a[h][j] != b[h][j]) { printf("trial %d slot %d word %d operand %d: %08x, per bit %08x\n", trial, s, 8 * g + j, h, a[h][j], b[h][j]); return 1; }
                }
        }
    }
    printf("%lld\n", words);
}
"""
    assert int(_run(tmp_path, program)) > 100_000


def test_host_copies_match_scatter(tmp_path: Path) -> None:
    lam = _extract(r"(?<=auto host_copies = \[&\]\(\) -> int \{\n).*?(?=\n            \};\n)")
    kernel = _extract(r"^__global__ void fc_host_slots_and\(.*?^}\n")
    program = r"""
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
struct uint4 { uint32_t x, y, z, w; };
static uint4 make_uint4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { return {a, b, c, d}; }
typedef uint4 F128;
enum { cudaMemcpyHostToDevice = 1 };
typedef int cudaStream_t;
static long long n_copies;
static int cudaMemcpyAsync(void* d, const void* s, size_t n, int, cudaStream_t) { memcpy(d, s, n); n_copies++; return 0; }
static int cudaMemcpy2DAsync(void* d, size_t dp, const void* s, size_t sp, size_t w, size_t h, int, cudaStream_t) {
    if (dp < w || sp < w) return 1;
    for (size_t r = 0; r < h; r++) memcpy((char*)d + r * dp, (const char*)s + r * sp, w);
    n_copies++;
    return 0;
}
struct cudaPitchedPtr { void* ptr; size_t pitch, xsize, ysize; };
struct cudaExtent { size_t width, height, depth; };
struct cudaMemcpy3DParms { cudaPitchedPtr srcPtr, dstPtr; cudaExtent extent; int kind; };
static cudaPitchedPtr make_cudaPitchedPtr(void* d, size_t p, size_t x, size_t y) { return {d, p, x, y}; }
static cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) { return {w, h, d}; }
static int cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t) {
    const cudaExtent& e = p->extent;
    if (e.width > p->srcPtr.pitch || e.width > p->dstPtr.pitch || e.height > p->srcPtr.ysize || e.height > p->dstPtr.ysize) return 1;
    for (size_t z = 0; z < e.depth; z++)
        for (size_t y = 0; y < e.height; y++)
            memcpy((char*)p->dstPtr.ptr + (z * p->dstPtr.ysize + y) * p->dstPtr.pitch,
                   (const char*)p->srcPtr.ptr + (z * p->srcPtr.ysize + y) * p->srcPtr.pitch, e.width);
    n_copies++;
    return 0;
}
static int cudaGetLastError() { return 0; }
static int cudaEventRecord(int, cudaStream_t) { return 0; }
#define CK(x) do { if ((x) != 0) { printf("failed: %s\n", #x); return 1; } } while (0)
""" + _device_free(kernel) + r"""
struct Params { long long n_host0; int host0_ab; const long long* host0_at; const int* host0_words; const uint64_t* host0_zab; };
static int copies(const Params* N, long long hs_max_pitch, F128* df, F128* d_a, F128* d_b, long long* d_hat, int* d_hw) {
    const bool hs_copy = true;
    int n_sm = 1, hs_stream = 0, hs_done = 0;
""" + _device_free(lam) + r"""
}
// fc_host_slots's scatter, the reference; then host_copies into zeroed words, which must equal it
static int check(const std::vector<long long>& at, const std::vector<int>& words, int ab, long long pitch, std::mt19937_64& rng) {
    const long long n = (long long)at.size(), per = ab ? 4 : 6;
    long long len = 8;
    for (long long i = 0; i < n; i++) len = std::max(len, at[i] + words[i] + 8);
    std::vector<uint64_t> zab;
    for (long long i = 0; i < n; i++) for (long long k = 0; k < per * words[i]; k++) zab.push_back(rng());
    std::vector<F128> z1(len), a1(len), b1(len), z2(len), a2(len), b2(len);
    for (auto* v : {&z1, &a1, &b1, &z2, &a2, &b2}) memset(v->data(), 0, len * sizeof(F128));
    for (long long i = 0, o = 0; i < n; o += per * words[i], i++) {
        const int w = words[i];
        const uint4* s = reinterpret_cast<const uint4*>(zab.data() + o);
        for (int k = 0; k < w; k++) {
            if (ab) {
                const uint4 x = s[k], y = s[w + k];
                a1[at[i] + k] = x; b1[at[i] + k] = y; z1[at[i] + k] = make_uint4(x.x & y.x, x.y & y.y, x.z & y.z, x.w & y.w);
            } else {
                z1[at[i] + k] = s[k]; a1[at[i] + k] = s[w + k]; b1[at[i] + k] = s[2 * w + k];
            }
        }
    }
    Params P{n, ab, at.data(), words.data(), zab.data()};
    n_copies = 0;
    if (copies(&P, pitch, z2.data(), a2.data(), b2.data(), const_cast<long long*>(at.data()), const_cast<int*>(words.data()))) return -1;
    if (memcmp(z1.data(), z2.data(), len * 16) || memcmp(a1.data(), a2.data(), len * 16) || memcmp(b1.data(), b2.data(), len * 16)) return -2;
    return (int)n_copies;
}
int main() {
    std::mt19937_64 rng(11);
    const long long max_pitch = (1LL << 31) - 1;
    // the moe-w13 statement's shape, scaled down: 8 consecutive slots a block, then a unit a block, over 64 blocks
    {
        std::vector<long long> at; std::vector<int> words;
        for (int b = 0; b < 64; b++) for (int j = 0; j < 8; j++) { at.push_back(4096LL * b + 32 * j); words.push_back(32); }
        for (int b = 0; b < 64; b++) { at.push_back(4096LL * b + 1024); words.push_back(256); }
        const int c = check(at, words, 1, max_pitch, rng);
        if (c != 4) { printf("block-regular slots: %d copies, want 4 (one a group and word kind)\n", c); return 1; }
    }
    for (int trial = 0; trial < 3000; trial++) {
        std::vector<long long> at; std::vector<int> words;
        long long pos = 0;
        for (int g = 0, groups = 1 + (int)(rng() % 6); g < groups; g++) {
            const int kind = (int)(rng() % 4), cnt = 1 + (int)(rng() % 40), w = 1 + (int)(rng() % 9);
            const long long stride = kind == 0 ? w : w + (long long)(rng() % 20);
            pos += (long long)(rng() % 7);
            if (kind == 3) {  // runs of L slots at one stride, each one outer stride from the last (at times not a multiple of it)
                const int L = 1 + (int)(rng() % 6), D = 1 + (int)(rng() % 12);
                const long long outer = stride * (L + (long long)(rng() % 4)) + (rng() % 3 == 0 ? 1 : 0);
                for (int d = 0; d < D; d++) for (int l = 0; l < L; l++) { at.push_back(pos + d * outer + l * stride); words.push_back(w); }
                pos += D * outer + L * stride;
                continue;
            }
            for (int c = 0; c < cnt; c++) {  // 0: contiguous; 1: one stride; 2: irregular widths and gaps
                const int ww = kind == 2 ? 1 + (int)(rng() % 9) : w;
                at.push_back(pos); words.push_back(ww);
                pos += kind == 2 ? ww + (long long)(rng() % 5) : stride;
            }
        }
        const long long pitch = trial % 5 == 0 ? 16 * (1 + (long long)(rng() % 12)) : max_pitch;
        const int c = check(at, words, trial % 2, pitch, rng);
        if (c < 0) { printf("trial %d (ab %d, max pitch %lld): %s\n", trial, trial % 2, pitch, c == -1 ? "a copy failed" : "words differ"); return 1; }
    }
    printf("ok\n");
}
"""
    assert _run(tmp_path, program).strip() == "ok"
