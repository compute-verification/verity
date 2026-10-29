"""Pearl-C's sm_90a GPU check fixture, built off the card (needs verity and numpy): inputs at a small shape and every buffer
pearl_c_sm90.cu's kernels write, in its row-major layout (every operand in natural order along k), for forming v1 on
`pearl_c_device.H100` (G = 4: `pearl-c-h100-v1-h3`), hashing format -h3 (SHA-512 throughout); the unpromoted plain
baseline (gemm_fast's y_fast) on the H100 record without promotion (`H100_UNPROMOTED`, no scheme of its own).  Forming,
lines and hashes come from `verity.core.protocols.pouw.schemes.pearl_c` on the device record; the products over every
cell are batched in numpy (the chains window by window, the peel with `verity.ml.kernels.tc.group_sum_total_batch` on the
record's peel step), and one whole 64 x 64 tile is recomputed by the scheme (`encode`, `checked`, `digest`) and must
agree.  run.py compares the device's buffers with these bytes before it times anything.

  uv run python benchmarks/pouw/pearl_c_sm90/fixture.py OUT_DIR [m n k]     (default 256 384 1024; m, n multiples of 128)

The instance is a one-unit epoch as `verity.core.protocols.pouw.audit` starts it: B's `audit.HM96` row tree under a
fixed prover salt key (`SALT_KEY`), the salt from the salt coin of the gate's test challenger (`test_challenger`,
`Epoch.start`) opened after B's root, A's HM96 row tree, seed_A from A's committed root (so E_A's padding line key is an
output the device derives, `line_a`) and each row's seed and line key from its SHA-512 row digest.  Writes
OUT_DIR/check.json (the shape, keys, seeds, the coin commitment and the salt coin's opening, and per buffer its file, dtype
and SHA-256; `key_files` holds `audit.hm96_kernel_constants`, too large for the keys' hex) and the buffers as raw
little-endian files: `common` (both trees and row digests, seed_A and E_A's line keys, the row seeds, lines, inputs,
forming, the plain and adopted baselines) and per variant (`h100`) the chains and everything after them, through the tile
leaves and the tile tree.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import multiprocessing
import sys
from dataclasses import replace
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "verity" / "core" / "protocols" / "pouw" / "tests"))


import pearl_c_vectors as CV  # noqa: E402
from verity.ml.commitments.merkle import tree_levels  # noqa: E402
from verity.ml.circuits.silicon import fp32  # noqa: E402
from verity.ml.kernels.tc import group_sum_total_batch  # noqa: E402
from verity.core.protocols.pouw.audit import (HM96, HM96_TRANSCRIPT, HM96_WEIGHTS, ROW_SCHEMA, TILE_SCHEMA,  # noqa: E402
                                                         Epoch, Workload, activation_binding, commit_rows, hm96_kernel_constants,
                                                         hm96_salt, rows_domain, transcript_domain, weight_binding, weights_root)
from verity.core.protocols.pouw.coins import Challenger, CoinOpening  # noqa: E402
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, Tile  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_c as C  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_kw as P  # noqa: E402
from verity.core.protocols.pouw.schemes.pearl_c_device import (  # noqa: E402
    DEVICES,
    H100,
    Device,
)

VARIANTS = (H100,)
#: the plain baseline unpromoted (gemm_fast): one accumulator over all of k on the H100's atom
H100_UNPROMOTED = replace(H100, name="h100-unpromoted", g=None)
CASE, INDEX, WEIGHT = "gpu-check", 5, 1
COINS = b"verity-pouw/pearl-c-sm90-gpu-check/coins"
#: the check's prover hm96 salt key (a served run draws its own)
SALT_KEY = hashlib.sha256(b"verity-pouw/pearl-c-sm90-gpu-check/salt-key").digest()


def test_challenger() -> Challenger:
    """The gate's test challenger: epoch 0's coins flipped from SHA-512 in counter mode over `COINS`, the same at every build,
    so a replay starts the fixture's epoch again from B's root."""
    blocks, buf = (hashlib.sha512(COINS + i.to_bytes(8, "little")).digest() for i in itertools.count()), bytearray()

    def entropy(n: int) -> bytes:
        while len(buf) < n:
            buf.extend(next(blocks))
        out = bytes(buf[:n])
        del buf[:n]
        return out

    return Challenger(0, entropy)


def unit_block(salt: bytes, root_b: bytes, index: int) -> bytes:
    """s512_seeds' p3: salt ‖ root_B (64 bytes) ‖ u64 index."""
    return salt + root_b + index.to_bytes(8, "little")


def tree_bytes(domain, values: list[bytes], schema: str, salt_key: bytes, nonce: int) -> bytes:
    """An `audit.HM96` tree's levels as the kernels lay them out: level 0 (each value under its salt, then pads), then each
    level above."""
    leaves = [domain.leaf(i, i, schema, v + hm96_salt(salt_key, nonce, i)) for i, v in enumerate(values)]
    return b"".join(b"".join(level) for level in tree_levels(domain, leaves))


def epoch_of(matmul: Matmul, b_rows: list[bytes], coins: Challenger, salt_key: bytes
             ) -> tuple[Epoch, CoinOpening, bytes, bytes]:
    """A one-unit epoch on B's rows as `verity.core.protocols.pouw.audit` starts it, from `coins`' salt coin
    opened after B's root: (epoch, the salt coin's opening, B's tree bytes, its root)."""
    dom_b = rows_domain(weight_binding(matmul.weight), -2, len(b_rows), HM96)
    tree_b = tree_bytes(dom_b, b_rows, ROW_SCHEMA, salt_key, HM96_WEIGHTS + matmul.weight)
    roots = {matmul.weight: tree_b[-64:]}
    opening = coins.open("salt", weights_root(roots))
    return Epoch.start(Workload((matmul,)), roots, coins.commitment, opening), opening, tree_b, tree_b[-64:]


def keys_of(scheme: C.PearlC, epoch: Epoch, matmul: Matmul, seed_b: bytes) -> dict[str, bytes]:
    """The pipeline's key inputs for a unit of `epoch` (run.Pipeline's `keys`) but the prover's salt key and the hm96
    constants, which the caller adds: E_B's line key and the domains in the scheme's tree hash."""
    s, salt, h = matmul.shape, epoch.salt, scheme.commitment_hash
    return {"line_b": C.line_key(seed_b),
            "dom_a": rows_domain(activation_binding(salt, matmul.index), matmul.index, s.m, h).domain_id,
            "dom_b": rows_domain(weight_binding(matmul.weight), -2, s.n, h).domain_id,
            "dom_t": transcript_domain(epoch, len(scheme.tiles(s)), h).domain_id,
            "unit": unit_block(salt, epoch.weight_roots[matmul.weight], matmul.index)}


# ---- layouts and batched pieces -------------------------------------------------------------------------------------

def padded(x: np.ndarray, rows: int) -> np.ndarray:
    out = np.zeros((rows, x.shape[1]), np.uint8)
    out[:x.shape[0]] = x
    return out


def bf16_bytes(words: np.ndarray) -> np.ndarray:
    """(rows, e) BF16 words as the (rows, 2e) byte matrix of their little-endian halves."""
    return np.ascontiguousarray(words, "<u2").view(np.uint8).reshape(words.shape[0], -1)


def rne_bf16(w: np.ndarray) -> np.ndarray:
    w = w.astype(np.uint64)
    return ((w + 0x7FFF + ((w >> 16) & 1)) >> 16).astype(np.uint16)


def fmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.view(np.float32) * b.view(np.float32)).view(np.uint32)


def grid(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return ii.ravel(), jj.ravel()


def pa_rows(be: np.ndarray, af: np.ndarray) -> np.ndarray:
    """C.peel_factors_a from the batched A′·F_B words: [be ‖ RNE_bf16(AF)]."""
    return np.concatenate([be, rne_bf16(af)], axis=1)


def pb_rows(be: np.ndarray, bf: np.ndarray, g: np.ndarray) -> np.ndarray:
    """C.peel_factors_b from the batched B̃·F_A and G words, the FMA chain in the scalar reference's order."""
    mid = np.zeros(bf.shape, np.uint16)
    for j in range(be.shape[0]):
        for q in range(C.R):
            acc = 0
            for p in range(C.R):
                acc = fp32.fma(int(be[j, p]) << 16, int(g[p, q]), acc)
            mid[j, q] = P.f32_to_bf16(P.f32_of_bits(fp32.add(acc, fp32.neg(int(bf[j, q])))))
    return np.concatenate([mid, be ^ 0x8000], axis=1)


def plain(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The baseline's quantize (plainq): s = 448/amax, codes = fp8(x·s), glue amax/448, per row of FP32 values."""
    codes, glue = np.zeros(x.shape, np.uint8), np.zeros(x.shape[0], np.uint32)
    for i, row in enumerate(x):
        amax = P.f32_bits(float(np.abs(row).max()))
        s = fp32.div(P.f32_bits(448.0), amax) if amax else P.f32_bits(1.0)
        glue[i] = fp32.div(amax, P.f32_bits(448.0)) if amax else P.f32_bits(1.0)
        codes[i] = [P.f32_to_fp8(P.f32_of_bits(fp32.mul(P.f32_bits(float(v)), s))) for v in row]
    return codes, glue


def adopted(x: np.ndarray, alpha: list[int]) -> np.ndarray:
    """The adopted quantizer (quant_s5): e4m3(α·x) per element, α from v1's statistics."""
    return np.array([[P.f32_to_fp8(P.f32_of_bits(fp32.mul(a, P.f32_bits(float(v))))) for v in row] for row, a in zip(x, alpha)], np.uint8)


def glue_y(words: np.ndarray, rs: np.ndarray, cs: np.ndarray) -> np.ndarray:
    return rne_bf16(fmul(fmul(words, np.broadcast_to(rs[:, None], words.shape).copy()), np.broadcast_to(cs[None, :], words.shape).copy()))


def recip(alpha: np.ndarray) -> np.ndarray:
    return (np.float32(1.0) / alpha.view(np.float32)).view(np.uint32)


def tile_leaves(c: np.ndarray, u: np.ndarray, m: int, n: int) -> list[bytes]:
    """The tile leaves, row-major (`pearl_c.leaf` under -h3 over each tile's C~ and U); the last row tile is partial when
    64 ∤ m."""
    return [C.leaf(C.Checked(c[r:r + 64, q:q + 64].tolist(), u[r:r + 64, q:q + 64].tolist()), "h3")
            for r in range(0, m, 64) for q in range(0, n, 64)]


class Out:
    def __init__(self, path: Path):
        self.path, self.files = path, {}

    def put(self, name: str, data, dtype: str) -> None:
        raw = data if isinstance(data, bytes) else b"".join(data) if isinstance(data, list) else np.ascontiguousarray(data).tobytes()
        (self.path / f"{name}.bin").write_bytes(raw)
        self.files[name] = {"file": f"{name}.bin", "dtype": dtype, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def rowmajor(x: np.ndarray) -> bytes:
    """pearl_c_sm90.cu's layout of a (rows, depth) byte matrix: row-major, depth contiguous, natural order along k."""
    return np.ascontiguousarray(x, np.uint8).tobytes()


def e4m3_values(codes: np.ndarray) -> np.ndarray:
    """E4M3 codes' values as FP32 (exact: every finite E4M3 value is one); no NaN code (0x7F, 0xFF) is passed."""
    c = codes.astype(np.int64)
    e, m = (c >> 3) & 0xF, (c & 7).astype(np.float64)
    v = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * np.exp2(e - 7.0))
    return np.where(c & 0x80, -v, v).astype(np.float32)


def widen_case(out: Out, rows: int, k: int) -> dict:
    """B's widening (run.py's WIDEN) on a stored B of `rows` rows (a partial last 128-row block when 128 doesn't divide
    them): random BF16 words, widened exactly; random E4M3 codes (no NaN) with an FP32 scale per 128 x 128 block, exponents
    from 2^-135 (FP32 subnormals, so a flush would show) to 2^0, each product RN in FP32."""
    rng = np.random.default_rng(20261023)
    words = rng.integers(0, 1 << 16, (rows, k), dtype=np.uint16)
    codes = rng.integers(0, 256, (rows, k), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] ^= 1
    sb = (-(-rows // 128), -(-k // 128))
    exps = rng.integers(-135, 1, sb)
    exps.flat[:3] = (-135, -122, 0)       # a subnormal scale, a normal one with subnormal products, and 1
    scales = (np.exp2(exps.astype(np.float64)) * (1 + rng.random(sb))).astype(np.float32)
    xb_fp8 = e4m3_values(codes) * np.repeat(np.repeat(scales, 128, 0), 128, 1)[:rows, :k]
    out.put("widen_src_bf16", words.astype("<u2"), "u16")
    out.put("widen_src_fp8_block", codes, "u8")
    out.put("widen_scales", scales.astype("<f4"), "f32")
    out.put("widen_xb_bf16", (words.astype(np.uint32) << 16).astype("<u4"), "u32")
    out.put("widen_xb_fp8_block", xb_fp8.astype("<f4"), "f32")
    names = {"src_bf16": "widen_src_bf16", "src_fp8_block": "widen_src_fp8_block", "scales": "widen_scales",
             "xb_bf16": "widen_xb_bf16", "xb_fp8_block": "widen_xb_fp8_block"}
    return {"shape": [rows, k], "files": {key: out.files[name] for key, name in names.items()}}


def chain(a: np.ndarray, b: np.ndarray, ii: np.ndarray, jj: np.ndarray, device: Device, block: int = 1 << 16) -> np.ndarray:
    """`pearl_c.chain` on `device` for the cells (ii, jj) of code matrices a and b: +0 per window of G atoms, one FP32 add
    per window; unpromoted, one accumulator over all of k."""
    out = np.empty(len(ii), np.uint32)
    k = a.shape[1]
    window = device.window if device.g else k
    for lo in range(0, len(ii), block):
        ci, cj = ii[lo:lo + block], jj[lo:lo + block]
        total = np.zeros(len(ci), np.uint32)
        for ws in range(0, k, window):
            local = np.zeros(len(ci), np.uint32)
            for s in range(ws, min(ws + window, k), C.SLICE):
                local = group_sum_total_batch(device.atom, local, a[ci, s:s + C.SLICE], b[cj, s:s + C.SLICE])[0]
            total = (total.view(np.float32) + local.view(np.float32)).view(np.uint32) if device.g else local
        if device.g:
            total[total == 0x80000000] = 0
        out[lo:lo + block] = total
    return out


def product(device: Device, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """C.chain on `device` for every cell of codes a (rows x k) and b, batched."""
    ii, jj = grid(a.shape[0], b.shape[0])
    return chain(a, b, ii, jj, device).reshape(a.shape[0], b.shape[0])


def peel_np(device: Device, c: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    ii, jj = grid(*c.shape)
    acc = c.ravel().copy()
    for s in range(0, C.PEEL_K, 16):
        acc = group_sum_total_batch(device.peel, acc, pa[ii, s:s + 16], pb[jj, s:s + 16])[0]
    return acc.reshape(c.shape)


def _side(job):
    device, rows, idx, side, seed_e, seed_b, k, e = job
    s = C.PearlC(k_min=32, tile=(16, 16), forming="v1", device=DEVICES[device])
    return s._side(rows, idx, side, seed_e, seed_b, k, e)


def form(pool: Pool, device: Device, rows: list[bytes], side: int, seed_e: bytes, seed_b: bytes, k: int,
         e: list[list[int]] | None = None, chunk: int = 16):
    """One side's forming in chunks of rows; `e` gives each row's E line (A's, each from its own seed), else seed_e's."""
    jobs = [(device.name, rows[i:i + chunk], list(range(i, min(i + chunk, len(rows)))), side, seed_e, seed_b, k,
             None if e is None else e[i:i + chunk]) for i in range(0, len(rows), chunk)]
    parts = pool.map(_side, jobs)
    built = parts[0][0]
    for p, _ in parts[1:]:
        for field in built.__slots__:
            getattr(built, field).extend(getattr(p, field))
    return built, [x for _, ep in parts for x in ep]


def build(out_dir: Path, m: int = 256, n: int = 384, k: int = 1024) -> dict:
    assert m % 128 == 0 and n % 128 == 0 and k % 1024 == 0, (m, n, k)
    out_dir.mkdir(parents=True, exist_ok=True)
    ra, rb = CV.rows_f32(CASE, "A", m, k), CV.rows_f32(CASE, "B", n, k)
    scheme = C.PearlC(forming="v1", device=H100, hashing="h3")
    h = scheme.commitment_hash
    mm = Matmul(INDEX, Shape(m, k, n), WEIGHT)
    coins = test_challenger()
    epoch, salt_opening, tree_b, root_b = epoch_of(mm, rb, coins, SALT_KEY)
    assert root_b == commit_rows(weight_binding(WEIGHT), -2, rb, h, SALT_KEY, HM96_WEIGHTS + WEIGHT).commitment.root
    dom_a = rows_domain(activation_binding(epoch.salt, INDEX), INDEX, m, h)
    tree_a = tree_bytes(dom_a, ra, ROW_SCHEMA, SALT_KEY, INDEX)
    root_a = tree_a[-64:]
    assert root_a == commit_rows(activation_binding(epoch.salt, INDEX), INDEX, ra, h, SALT_KEY, INDEX).commitment.root
    ctx = Context(epoch.salt, mm, root_a, root_b)
    noise = scheme.noise(ctx)
    seeds = scheme.row_seeds(ctx, ra, range(m))
    keys = keys_of(scheme, epoch, mm, noise.seed_b)
    keys["salt_key"] = SALT_KEY
    dom_t = transcript_domain(epoch, len(scheme.tiles(mm.shape)), h)
    manifest = {"schema": "verity-pouw/pearl-c-sm90-gpu-check/v1", "shape": [m, n, k], "case": CASE, "layout": "rowmajor",
                "hashing": scheme.hashing, "weight": WEIGHT, "coins": coins.commitment.to_json(), "salt_opening": salt_opening.to_json(),
                "keys": {name: x.hex() for name, x in keys.items()},
                "seeds": {"seed_a": noise.seed_a.hex(), "seed_b": noise.seed_b.hex(), "salt": epoch.salt.hex(),
                          "root_a": root_a.hex(), "root_b": root_b.hex()}, "variants": {}}
    kf = Out(out_dir)
    kf.put("hm", hm96_kernel_constants(), "u8")
    manifest["key_files"] = kf.files
    o = Out(out_dir)
    o.put("tree_a", tree_a, "u8")
    o.put("tree_b", tree_b, "u8")
    o.put("rowdig_a", [C.row_digest(r) for r in ra], "u8")
    o.put("rowdig_b", [C.row_digest(r) for r in rb], "u8")
    o.put("seed_a", noise.seed_a, "u8")
    o.put("line_a", C.line_key(noise.seed_a), "u8")
    o.put("seeds_a", seeds, "u8")
    o.put("lkeys_a", [C.line_key(z) for z in seeds], "u8")
    e_rows = [C.sample_line(z, 0, 0, 0) for z in seeds]
    lines = {"ea": np.array(e_rows, np.uint8), "eb": np.array([C.sample_line(noise.seed_b, 1, 0, t) for t in range(n)], np.uint8)}
    for name, side in (("fa", 0), ("fb", 1)):
        lines[name] = np.array(C._basis(noise.seed_b, side, k, C.LINE_NORM_V1), np.uint8)
    for name, x in lines.items():
        o.put(f"{name}_codes", x, "u8")
    faT, fbT = lines["fa"].T.copy(), lines["fb"].T.copy()
    fbT128 = padded(fbT, 128)
    o.put("faT32", rowmajor(faT), "u8")
    o.put("fbT32", rowmajor(fbT), "u8")
    o.put("fbT128", rowmajor(fbT128), "u8")
    o.put("xa", b"".join(ra), "u8")
    o.put("xb", b"".join(rb), "u8")
    # fork: the workers unpickle `_side` from this module, which callers load from its path, not by an importable name
    with multiprocessing.get_context("fork").Pool() as pool:
        a, e_a = form(pool, H100, ra, 0, noise.seed_a, noise.seed_b, k, e_rows)
        b, e_b = form(pool, H100, rb, 1, noise.seed_b, noise.seed_b, k)
    inv = {}
    for tag, side in (("a", a), ("b", b)):
        o.put(f"alpha_{tag}", np.array(side.alpha, "<u4"), "u32")
        o.put(f"beta_{tag}", np.array(side.beta, "<u4"), "u32")
        o.put(f"aux_{tag}", np.array([[s, r] for s, r in zip(side.s, side.rho)], "<u4"), "u32")
        o.put(f"ep{tag}_codes", np.array(side.ep, np.uint8), "u8")
        inv[tag] = recip(np.array(side.alpha, np.uint32))
        o.put(f"ia_{tag}", inv[tag], "u32")
    at, bt = np.array(a.codes, np.uint8), np.array(b.codes, np.uint8)
    o.put("at", rowmajor(at), "u8")
    o.put("bt", rowmajor(bt), "u8")
    fa_vals = np.frombuffer(b"".join(ra), "<f4").reshape(m, k)
    fb_vals = np.frombuffer(b"".join(rb), "<f4").reshape(n, k)
    qa, sa = plain(fa_vals)
    qb, sb = plain(fb_vals)
    o.put("qa", rowmajor(qa), "u8")
    o.put("qb", rowmajor(qb), "u8")
    o.put("sa", sa, "u32")
    o.put("sb", sb, "u32")
    o.put("y_plain", glue_y(product(H100, qa, qb), sa, sb), "u16")
    o.put("y_fast", glue_y(product(H100_UNPROMOTED, qa, qb), sa, sb), "u16")
    qa5, qb5 = adopted(fa_vals, a.alpha), adopted(fb_vals, b.alpha)
    o.put("qa_s5", rowmajor(qa5), "u8")
    o.put("qb_s5", rowmajor(qb5), "u8")
    o.put("y_adopt", glue_y(product(H100, qa5, qb5), inv["a"], inv["b"]), "u16")
    widen_files = Out(out_dir)
    manifest["widen"] = widen_case(widen_files, max(n - 64, 64), k)
    manifest["common"] = o.files

    be_a, be_b = np.array(C.scaled_noise(a, e_a), np.uint16), np.array(C.scaled_noise(b, e_b), np.uint16)
    for device in VARIANTS:
        vdir = out_dir / device.name
        vdir.mkdir(exist_ok=True)
        v = Out(vdir)
        g = product(device, fbT128, faT)
        assert all(int(g[p, q]) == C.gram(noise.seed_b, k, C.LINE_NORM_V1, device)[p][q] for p, q in ((0, 0), (3, 17), (31, 31)))
        v.put("g", g, "u32")
        af, bf = product(device, at, fbT), product(device, bt, faT)
        v.put("af", af, "u32")
        v.put("bf", bf, "u32")
        pa, pb = pa_rows(be_a, af), pb_rows(be_b, bf, g)
        v.put("pa_t", rowmajor(bf16_bytes(pa)), "u8")
        v.put("pb_t", rowmajor(bf16_bytes(pb)), "u8")
        c = product(device, at, bt)
        u = peel_np(device, c, pa, pb)
        v.put("c", c, "u32")
        v.put("u", u, "u32")
        v.put("y", glue_y(u, inv["a"], inv["b"]), "u16")
        lv = tile_leaves(c, u, m, n)
        v.put("leaves", lv, "u8")
        v.put("tree_t", tree_bytes(dom_t, lv, TILE_SCHEMA, SALT_KEY, HM96_TRANSCRIPT), "u8")
        agree = crosscheck(device, ctx, ra, rb, at, bt, pa, pb, c, u, lv, n)
        manifest["variants"][device.name] = {"scheme": C.PearlC(forming="v1", device=device, hashing="h3").name, "g": device.g,
                                             "files": v.files, "crosscheck": agree}
    (out_dir / "check.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def crosscheck(device, ctx, ra, rb, at, bt, pa, pb, c, u, lv, n) -> dict:
    """One 64 x 64 tile (rows 64-127, the last 64 columns) through the scheme on `device` at -h3: encode, checked, digest."""
    s = C.PearlC(forming="v1", device=device, hashing="h3")
    rows, cols = range(64, 128), range(n - 64, n)
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, [ra[i] for i in rows], [rb[j] for j in cols], Tile(tuple(rows), tuple(cols)))
    assert ops is not None
    checked = s.checked(ctx, ops)
    sub = lambda x: [[int(w) for w in row] for row in x[64:128, n - 64:n]]  # noqa: E731
    assert [list(r) for r in ops.a.codes] == at[64:128].tolist() and [list(r) for r in ops.b.codes] == bt[n - 64:].tolist()
    assert ops.pa == pa[64:128].tolist() and ops.pb == pb[n - 64:].tolist()
    assert checked.c == sub(c) and checked.u == sub(u)
    assert s.admissible(ctx, ops)
    assert s.digest(ctx, noise, checked) == lv[(n // 64) + n // 64 - 1]
    return {"tile": [64, n - 64], "cells": 64 * 64, "agree": True}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dims = [int(x) for x in sys.argv[2:5]] or [256, 384, 1024]
    man = build(Path(sys.argv[1]), *dims)
    total = sum(f["bytes"] for part in [man["common"], *(v["files"] for v in man["variants"].values())] for f in part.values())
    print(f"fixture {man['shape']} -> {sys.argv[1]}: {total / 1e6:.1f} MB, cross-check "
          + ", ".join(f"{k} {v['crosscheck']['cells']} cells" for k, v in man["variants"].items()))
