"""Pearl-C's GPU check fixture, built off the pod (needs verity and numpy): inputs at a small shape and every buffer the
kernels write, for both formings and the plain baseline.  Forming, lines and hashes come from
`verity.core.protocols.pouw.schemes.pearl_c` itself; the products over every cell are batched in numpy with
`verity.ml.kernels.tc.group_sum_total_batch`, and one whole 64 x 64 tile is recomputed by the scheme (`encode`,
`checked`, `digest`, `tickets`) and must agree.  The hashing is format -h1, the verifier's: BLAKE3 messages and leaves,
and A's rows and the tile leaves committed as `verity.core.protocols.pouw.audit` commits them (frame-b3), whose roots must agree too.  Each
forming's seed_A is its unit's over its own A rows' root (`pearl_c.unit_seed_a`), as the device derives it, so E_A's lines,
A's forming and the tickets follow A's commitment.
run.py compares the device's buffers with these bytes before it times anything.

  uv run python benchmarks/pouw/pearl_c/fixture.py OUT_DIR [m n k]     (default 256 384 1024; m, n multiples of 128)

Writes OUT_DIR/check.json (the shape, keys, and per buffer its file, dtype and SHA-256) and the buffers as raw
little-endian files.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import multiprocessing
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "verity" / "core" / "protocols" / "pouw" / "tests"))

import pearl_c_vectors as CV  # noqa: E402
from verity.ml.crypto.blake3 import blake3, keyed  # noqa: E402
from verity.ml.kernels.tc import group_sum_total_batch  # noqa: E402
from verity.ml.circuits.silicon import fp32  # noqa: E402
from verity.ml.circuits.silicon.models import HOPPER_BF16_WGMMA_K16, HOPPER_E4M3_K32  # noqa: E402
from verity.ml.commitments import MerkleTree  # noqa: E402
from verity.ml.commitments.merkle import tree_levels  # noqa: E402
from verity.core.protocols.pouw.audit import (ROW_SCHEMA, TILE_SCHEMA, Epoch, Workload, activation_binding, commit_rows,  # noqa: E402
                               rows_domain, transcript_domain)
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, Tile  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_c as C  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_kw as P  # noqa: E402

CASE, INDEX, WEIGHT = "gpu-check", 5, 1
HASHING = "h1"                     # the kernels' hashing format: BLAKE3 messages, leaves and frame-b3 trees


# ---- layouts --------------------------------------------------------------------------------------------------------

def tiled(x: np.ndarray, h: int) -> bytes:
    """pearl_c.cu's tiled layout of a (rows, depth) byte matrix in blocks of h rows."""
    r, d = x.shape
    assert r % h == 0 and d % 32 == 0, (r, d, h)
    return np.ascontiguousarray(x, np.uint8).reshape(r // h, h // 8, 8, d // 32, 2, 16).transpose(0, 3, 1, 4, 2, 5).tobytes()


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


# ---- batched products ---------------------------------------------------------------------------------------------

def chain_np(a: np.ndarray, b: np.ndarray, ii: np.ndarray, jj: np.ndarray, window: int = C.WINDOW, raw: bool = False) -> np.ndarray:
    """C.chain for the cells (ii[c], jj[c]) of codes a (rows x k) and b; window = k with raw leaves the one accumulator
    unpromoted (the FAST kernel's words)."""
    k, total = a.shape[1], np.zeros(len(ii), np.uint32)
    for ws in range(0, k, window):
        local = np.zeros(len(ii), np.uint32)
        for s in range(ws, min(ws + window, k), C.SLICE):
            local = group_sum_total_batch(HOPPER_E4M3_K32, local, a[ii, s:s + C.SLICE], b[jj, s:s + C.SLICE])[0]
        if raw:
            return local
        total = (total.view(np.float32) + local.view(np.float32)).view(np.uint32)
        total[total == 0x80000000] = 0
    return total


def grid(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return ii.ravel(), jj.ravel()


def product(a: np.ndarray, b: np.ndarray, **kw) -> np.ndarray:
    ii, jj = grid(a.shape[0], b.shape[0])
    return chain_np(a, b, ii, jj, **kw).reshape(a.shape[0], b.shape[0])


def peel_np(c: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    ii, jj = grid(*c.shape)
    acc = c.ravel().copy()
    for s in range(0, C.PEEL_K, 16):
        acc = group_sum_total_batch(HOPPER_BF16_WGMMA_K16, acc, pa[ii, s:s + 16], pb[jj, s:s + 16])[0]
    return acc.reshape(c.shape)


# ---- the scheme's pieces ------------------------------------------------------------------------------------------

def _side(job):
    forming, rows, idx, side, seed_e, seed_b, k = job
    s = C.PearlC(k_min=32, tile=(16, 16), forming=forming)
    built, e = s._side(rows, idx, side, seed_e, seed_b, k)
    norms = [P.exact_norms(*P.parse_row(r, k)) for r in rows] if forming == "v0" else None
    return built, e, norms


def form(pool: Pool, forming: str, rows: list[bytes], side: int, seed_e: bytes, seed_b: bytes, k: int, chunk: int = 16):
    jobs = [(forming, rows[i:i + chunk], list(range(i, min(i + chunk, len(rows)))), side, seed_e, seed_b, k)
            for i in range(0, len(rows), chunk)]
    parts = pool.map(_side, jobs)
    built = parts[0][0]
    for p, _, _ in parts[1:]:
        for field in built.__slots__:
            getattr(built, field).extend(getattr(p, field))
    e = [x for _, ep, _ in parts for x in ep]
    norms = [x for _, _, n in parts for x in n] if forming == "v0" else None
    return built, e, norms


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


def digests(c: np.ndarray, u: np.ndarray) -> list[bytes]:
    out = []
    for i in range(c.shape[0]):
        for h in range(0, c.shape[1], C.CHUNK):
            out.append(C.message_digest(c[i, h:h + C.CHUNK].astype("<u4").tobytes() + u[i, h:h + C.CHUNK].astype("<u4").tobytes(),
                                        HASHING))
    return out


def leaves(dg: list[bytes], m: int, n: int) -> list[bytes]:
    """The tile leaves, row-major; the last row tile is partial when 64 ∤ m (decode's 32 x 64: 64 digests)."""
    per = n // C.CHUNK
    return [keyed(C.H1_KEYS["leaf"], b"".join(dg[r * per + 2 * tj + h] for r in range(64 * ti, min(64 * ti + 64, m))
                                              for h in range(2)))
            for ti in range(-(-m // 64)) for tj in range(n // 64)]


def tickets(c: np.ndarray, seed_a: bytes) -> list[bytes]:
    return [P.hash_labelled(P.extract([[P.f32_of_bits(int(w)) for w in row] for row in c[i:i + 16, j:j + 16]]), "jackpot", seed_a)
            for i in range(0, c.shape[0], 16) for j in range(0, c.shape[1], 16)]


def tree_bytes(domain, values: list[bytes], schema: str) -> bytes:
    """A frame-b3 tree's levels as the kernels lay them out: level 0 (leaves, then pads), then each level above."""
    return b"".join(b"".join(level) for level in tree_levels(domain, [domain.leaf(i, i, schema, v) for i, v in enumerate(values)]))


def a_domain(ctx: Context):
    """The domain A's rows are committed under (`verity.core.protocols.pouw.audit`): the activation binding, frame-b3."""
    return rows_domain(activation_binding(ctx.salt, ctx.matmul.index), ctx.matmul.index, ctx.matmul.shape.m, "blake3")


def t_domain(ctx: Context, tiles: int):
    """The tile tree's domain for a one-unit epoch whose weight root is the context's."""
    epoch = Epoch(Workload((ctx.matmul,)), {ctx.matmul.weight: ctx.root_b}, ctx.salt)
    return transcript_domain(epoch, tiles, "blake3")


# ---- the fixture -------------------------------------------------------------------------------------------------

class Out:
    def __init__(self, path: Path):
        self.path, self.files = path, {}

    def put(self, name: str, data, dtype: str) -> None:
        raw = data if isinstance(data, bytes) else b"".join(data) if isinstance(data, list) else np.ascontiguousarray(data).tobytes()
        (self.path / f"{name}.bin").write_bytes(raw)
        self.files[name] = {"file": f"{name}.bin", "dtype": dtype, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def context(m: int, n: int, k: int) -> Context:
    h = lambda s: hashlib.sha256(f"{s}/{CASE}".encode()).digest()  # noqa: E731
    return Context(h("salt"), Matmul(INDEX, Shape(m, k, n), WEIGHT), h("root_a"), h("root_b"))


def build(out_dir: Path, m: int = 256, n: int = 384, k: int = 1024) -> dict:
    assert m % 128 == 0 and n % 128 == 0 and k % 1024 == 0, (m, n, k)
    out_dir.mkdir(parents=True, exist_ok=True)
    job = context(m, n, k)                      # the salt, the unit and root_B; each forming's root_A is its A rows' root
    seed_b = C.PearlC().noise(job).seed_b
    manifest = {"schema": "verity-pouw/pearl-c-gpu-check/v2", "shape": [m, n, k], "case": CASE,
                "keys": {"salt": job.salt.hex(), "root_b": job.root_b.hex(), "seed_key": blake3(C.LABELS["seed-A"]).hex(),
                         "line_b": P._line_key(seed_b).hex(), "dom_a": a_domain(job).domain_id.hex(),
                         "dom_t": t_domain(job, (m // 64) * (n // 64)).domain_id.hex(), "k_msg": C.H1_KEYS["message"].hex(),
                         "k_leaf": C.H1_KEYS["leaf"].hex()},
                "unit": {"index": job.matmul.index}, "hashing": HASHING, "seeds": {"seed_b": seed_b.hex()}, "formings": {}}
    lines = {"eb": np.array([P.sample_line(seed_b, 1, 0, t) for t in range(n)], np.uint8)}
    common = Out(out_dir)
    common.put("eb_codes", lines["eb"], "u8")
    common.put("eb_tiled", tiled(lines["eb"], 128), "u8")
    manifest["common"] = common.files

    # fork: the workers unpickle their function from this module, which callers load from its path, not by an importable name
    with multiprocessing.get_context("fork").Pool() as pool:
        for forming in ("v0", "v1"):
            fdir = out_dir / forming
            fdir.mkdir(exist_ok=True)
            o = Out(fdir)
            norm = C.PearlC(forming=forming).line_norm
            make = CV.rows if forming == "v0" else CV.rows_f32
            ra, rb = make(CASE, "A", m, k), make(CASE, "B", n, k)
            tree_a = tree_bytes(a_domain(job), ra, ROW_SCHEMA)
            ctx = Context(job.salt, job.matmul, tree_a[-32:], job.root_b)
            noise = C.PearlC().noise(ctx)
            assert noise.seed_a == C.unit_seed_a(ctx.salt, ctx.root_a, ctx.root_b, ctx.matmul.index) and noise.seed_b == seed_b
            o.put("seed_a", noise.seed_a, "u8")
            o.put("a_keys", P._line_key(noise.seed_a) + P.subkey("jackpot", noise.seed_a), "u8")
            lines["ea"] = np.array([P.sample_line(noise.seed_a, 0, 0, t) for t in range(m)], np.uint8)
            o.put("ea_codes", lines["ea"], "u8")
            o.put("ea_tiled", tiled(lines["ea"], 128), "u8")
            for name, side in (("fa", 0), ("fb", 1)):
                lines[name] = np.array(C._basis(noise.seed_b, side, k, norm), np.uint8)
                o.put(f"{name}_codes", lines[name], "u8")
                o.put(f"{name}_tiled", tiled(lines[name], 128), "u8")
            o.put("faT32", tiled(lines["fa"].T, 32), "u8")
            o.put("fbT32", tiled(lines["fb"].T, 32), "u8")
            o.put("fbT128", tiled(padded(lines["fb"].T, 128), 128), "u8")
            g = product(padded(lines["fb"].T, 128), lines["fa"].T.copy())
            o.put("g", g, "u32")
            assert all(int(g[p, q]) == C.gram(noise.seed_b, k, norm)[p][q] for p, q in ((0, 0), (3, 17), (31, 31)))
            o.put("xa", b"".join(ra), "u8")
            o.put("xb", b"".join(rb), "u8")
            a, e_a, na = form(pool, forming, ra, 0, noise.seed_a, noise.seed_b, k)
            b, e_b, nb = form(pool, forming, rb, 1, noise.seed_b, noise.seed_b, k)
            inv = {}
            for tag, side, norms in (("a", a, na), ("b", b, nb)):
                if forming == "v0":
                    o.put(f"alpha_{tag}", np.array(side.alpha, "<u2"), "u16")
                    o.put(f"beta_{tag}", np.array(side.beta, "<u2"), "u16")
                    o.put(f"aux_{tag}", np.array([l2 | linf << 16 for l2, linf in norms], "<u4"), "u32")
                    alpha32 = (np.array(side.alpha, np.uint32) << 16)
                else:
                    o.put(f"alpha_{tag}", np.array(side.alpha, "<u4"), "u32")
                    o.put(f"beta_{tag}", np.array(side.beta, "<u4"), "u32")
                    o.put(f"aux_{tag}", np.array([[s, r] for s, r in zip(side.s, side.rho)], "<u4"), "u32")
                    ep = np.array(side.ep, np.uint8)
                    o.put(f"ep{tag}_codes", ep, "u8")
                    o.put(f"ep{tag}_tiled", tiled(ep, 128), "u8")
                    alpha32 = np.array(side.alpha, np.uint32)
                inv[tag] = recip(alpha32)
                o.put(f"ia_{tag}", inv[tag], "u32")
            at, bt = np.array(a.codes, np.uint8), np.array(b.codes, np.uint8)
            o.put("at", tiled(at, 128), "u8")
            o.put("bt", tiled(bt, 128), "u8")
            af, bf = product(at, lines["fb"].T.copy()), product(bt, lines["fa"].T.copy())
            o.put("af", af, "u32")
            o.put("bf", bf, "u32")
            be_a = np.array(C.scaled_noise(a, e_a), np.uint16)
            be_b = np.array(C.scaled_noise(b, e_b), np.uint16)
            pa, pb = pa_rows(be_a, af), pb_rows(be_b, bf, g)
            o.put("pa_t", tiled(bf16_bytes(pa), 128), "u8")
            o.put("pb_t", tiled(bf16_bytes(pb), 128), "u8")
            c = product(at, bt)
            u = peel_np(c, pa, pb)
            o.put("c", c, "u32")
            o.put("u", u, "u32")
            o.put("y", glue_y(u, inv["a"], inv["b"]), "u16")
            dg = digests(c, u)
            o.put("digests", dg, "u8")
            lv = leaves(dg, m, n)
            o.put("leaves", lv, "u8")
            o.put("tickets", tickets(c, noise.seed_a), "u8")
            o.put("tree_a", tree_a, "u8")
            o.put("tree_t", tree_bytes(t_domain(ctx, len(lv)), lv, TILE_SCHEMA), "u8")

            if forming == "v0":
                xa = np.array([P.open_row(*P.parse_row(r, k)) for r in ra], "<u2")
                xb = np.array([P.open_row(*P.parse_row(r, k)) for r in rb], "<u2")
                o.put("xa_bf16", xa, "u16")
                o.put("xb_bf16", xb, "u16")
                fa_vals, fb_vals = (xa.astype(np.uint32) << 16).view(np.float32), (xb.astype(np.uint32) << 16).view(np.float32)
            else:
                fa_vals = np.frombuffer(b"".join(ra), "<f4").reshape(m, k)
                fb_vals = np.frombuffer(b"".join(rb), "<f4").reshape(n, k)
            qa, sa = plain(fa_vals)
            qb, sb = plain(fb_vals)
            o.put("qa", tiled(qa, 128), "u8")
            o.put("qb", tiled(qb, 128), "u8")
            o.put("sa", sa, "u32")
            o.put("sb", sb, "u32")
            o.put("y_plain", glue_y(product(qa, qb), sa, sb), "u16")
            o.put("y_fast", glue_y(product(qa, qb, window=k, raw=True), sa, sb), "u16")
            if forming == "v1":
                qa5, qb5 = adopted(fa_vals, a.alpha), adopted(fb_vals, b.alpha)
                o.put("qa_s5", tiled(qa5, 128), "u8")
                o.put("qb_s5", tiled(qb5, 128), "u8")
                o.put("y_adopt", glue_y(product(qa5, qb5), inv["a"], inv["b"]), "u16")

            agree = crosscheck(forming, ctx, ra, rb, at, bt, pa, pb, c, u, dg, lv, m, n)
            manifest["formings"][forming] = {"scheme": C.PearlC(forming=forming, hashing=HASHING).name, "files": o.files,
                                             "seed_a": noise.seed_a.hex(), "crosscheck": agree}
    x, y = ieee_inputs()
    ops, q = ieee_reference(x, y)
    ieee = Out(out_dir)
    for name, w in (("ieee_x", x), ("ieee_y", y), ("ieee_ops", ops), ("ieee_q", q)):
        ieee.put(name, w.astype("<u4"), "u32")
    manifest["ieee"] = {"count": int(x.size), "files": ieee.files, "coverage": ieee_coverage(x, y, ops)}
    (out_dir / "check.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


# ---- the IEEE test kernels' words (ieee_ops, ieee_q: the sequences ftz_gate.py admits) -------------------------------------

F448, ONE = 0x43E00000, 0x3F800000


def ieee_inputs(seed: int = 20260930) -> tuple[np.ndarray, np.ndarray]:
    """x > 0 and y ≠ 0, finite FP32 words: subnormal x (and the smallest normals), quotients x / y and x / 448 that are
    subnormal, x at or above 2^126 (a subnormal reciprocal), subnormal y, and random normals, with y of both signs.  bf16(x)
    is finite and bf16(y) nonzero, so every reference is defined."""
    rng = np.random.default_rng(seed)
    sub = np.concatenate([[1, 2, 3, 0x7FFFFF, 0x400000, 0x7FFFFE], rng.integers(1, 0x800000, 250)])
    tiny = rng.integers(0x00800000, 0x0D800000, 250)                 # 2^-126 .. 2^-100: subnormal x / y, x / 448
    big = rng.integers(0x7E800000, 0x7F7F0000, 150)                  # 2^126 .. : a subnormal 1 / x
    normal = rng.integers(0x00800000, 0x7F7F0000, 800)
    edges = np.array([0x00800000, 0x00800001, ONE, F448, 0x7F7F0000, 0x3E800000, 0x01000000])
    x = np.concatenate([sub, tiny, big, normal, edges]).astype(np.uint32)
    ymag = np.concatenate([rng.integers(0x3F000000, 0x40800000, sub.size),        # 0.5 .. 4
                           rng.integers(0x43800000, 0x54000000, tiny.size),       # 2^8 .. 2^41
                           rng.integers(0x00800000, 0x7F7F0000, big.size),
                           np.where(rng.random(normal.size) < 0.1, rng.integers(0x8001, 0x800000, normal.size),
                                    rng.integers(0x00800000, 0x7F7F0000, normal.size)),
                           [ONE, 0x7F7F0000, 0x8001, F448, 0x00800000, 0x3F800001, 0x4B000000]])
    y = (ymag | (rng.integers(0, 2, ymag.size) << 31)).astype(np.uint32)
    return x, y


def ieee_reference(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ieee_ops' six words per element (x / y, 448 / x, x / 448, 1 / x, sqrt x, bf16(x) / bf16(y)) and ieee_q's 448 / x, each
    correctly rounded with subnormals kept (`verity.ml.circuits.silicon.fp32`)."""
    bx, by = rne_bf16(x).astype(np.uint32) << 16, rne_bf16(y).astype(np.uint32) << 16
    ops = np.array([[fp32.div(int(a), int(b)), fp32.div(F448, int(a)), fp32.div(int(a), F448), fp32.div(ONE, int(a)),
                     fp32.sqrt(int(a)), int(rne_bf16(np.array([fp32.div(int(p), int(q))], np.uint32))[0])]
                    for a, b, p, q in zip(x, y, bx, by)], np.uint32)
    return ops, ops[:, 1].copy()


def ieee_coverage(x: np.ndarray, y: np.ndarray, ops: np.ndarray) -> dict[str, int]:
    """How many subnormal inputs and outputs the words exercise, per operand and op."""
    sub = lambda w: int((((w & 0x7F800000) == 0) & ((w & 0x7FFFFF) != 0)).sum())  # noqa: E731
    names = ("x / y", "448 / x", "x / 448", "1 / x", "sqrt x")
    return {"subnormal x": sub(x), "subnormal y": sub(y), **{f"subnormal {n}": sub(ops[:, j]) for j, n in enumerate(names)},
            "subnormal bf16 quotient": int((((ops[:, 5] & 0x7F80) == 0) & ((ops[:, 5] & 0x7F) != 0)).sum())}


def crosscheck(forming, ctx, ra, rb, at, bt, pa, pb, c, u, dg, lv, m, n) -> dict:
    """One 64 x 64 tile (rows 64-127, the last 64 columns) through the scheme itself: encode, checked, digest, tickets; and the
    trees' roots as `verity.core.protocols.pouw.audit` commits A's rows and the tile leaves."""
    s = C.PearlC(forming=forming, hashing=HASHING)
    root_a = commit_rows(activation_binding(ctx.salt, ctx.matmul.index), ctx.matmul.index, ra, s.commitment_hash).commitment.root
    assert tree_bytes(a_domain(ctx), ra, ROW_SCHEMA)[-32:] == root_a == ctx.root_a
    tree_t = MerkleTree(t_domain(ctx, len(lv)), dict(enumerate(lv)), lambda _: TILE_SCHEMA)
    assert tree_bytes(t_domain(ctx, len(lv)), lv, TILE_SCHEMA)[-32:] == tree_t.commitment.root
    rows, cols = range(64, 128), range(n - 64, n)
    tile = Tile(tuple(rows), tuple(cols))
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, [ra[i] for i in rows], [rb[j] for j in cols], tile)
    assert ops is not None
    checked = s.checked(ctx, ops)
    sub = lambda x: [[int(w) for w in row] for row in x[64:128, n - 64:n]]  # noqa: E731
    assert [list(r) for r in ops.a.codes] == at[64:128].tolist() and [list(r) for r in ops.b.codes] == bt[n - 64:].tolist()
    assert ops.pa == pa[64:128].tolist() and ops.pb == pb[n - 64:].tolist()
    assert checked.c == sub(c) and checked.u == sub(u)
    per = n // C.CHUNK
    assert s.admissible(ctx, ops)
    mine = keyed(C.H1_KEYS["leaf"], b"".join(dg[r * per + per - 2 + h] for r in rows for h in range(2)))
    assert s.digest(ctx, noise, checked) == mine
    assert s.tickets(noise, checked) == tickets(np.array(checked.c, np.uint32), noise.seed_a)
    return {"tile": [64, n - 64], "cells": 64 * 64, "agree": True}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dims = [int(x) for x in sys.argv[2:5]] or [256, 384, 1024]
    man = build(Path(sys.argv[1]), *dims)
    total = sum(f["bytes"] for part in [man["common"], *(v["files"] for v in man["formings"].values())] for f in part.values())
    print(f"fixture {man['shape']} -> {sys.argv[1]}: {total / 1e6:.1f} MB, cross-check "
          + ", ".join(f"{k} {v['crosscheck']['cells']} cells" for k, v in man["formings"].items()))
