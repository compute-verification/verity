"""Pearl-C's sm_90a GPU check fixture, built off the card (needs verity and numpy): inputs at a small shape and every buffer
pearl_c_sm90.cu's kernels write, in its row-major layout (every operand in natural order along k), for forming v1 on
`pearl_c_device.H100` (G = 4: `pearl-c-h100-v1-h3`), hashing format -h3 (SHA-512 throughout); the unpromoted plain
baseline (gemm_fast's y_fast) on the H100 record without promotion (`H100_UNPROMOTED`, no scheme of its own).  Forming,
lines and hashes come from `verity.protocols.accounting.compute.pouw.prover.schemes.pearl_c` on the device record; the products over every
cell are batched in numpy (the chains window by window, the peel with `verity.circuits.kernels.tc.group_sum_total_batch` on the
record's peel step), and one whole 64 x 64 tile is recomputed by the scheme (`encode`, `checked`, `digest`) and must
agree.  run.py compares the device's buffers with these bytes before it times anything.

  uv run python benchmarks/pouw/pearl_c_sm90/fixture.py OUT_DIR [m n k]     (default 256 384 1024; m, n multiples of 128)

The instance is a one-unit epoch as `verity.protocols.accounting.compute.pouw.verifier.audit` starts it: B's `audit.HM96` row tree, each
leaf under its salt in the fixture's fixed salts (`salts_of`; check.json's `salts` names each tree's file), the salt from the salt coin of the gate's test challenger (`test_challenger`,
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
import importlib.util
import json
import multiprocessing
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tests" / "verity" / "protocols" / "accounting" / "compute" / "pouw" / "verifier"))


def _sm120_fixture():
    """pearl_c_sm120/fixture.py, whose formation, chains, peel and tile leaves this fixture shares, under one module
    name, so the fork pool's workers find the functions it maps."""
    name = "pearl_c_sm120_fixture"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE.parent / "pearl_c_sm120" / "fixture.py")
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


F120 = _sm120_fixture()
from pearl_c_sm120_fixture import (Out, adopted, bf16_bytes, crosscheck, epoch_of, form, glue_y,  # noqa: E402
                                   padded, pa_rows, pb_rows, peel_np, plain, product, recip, tile_leaves, tree_bytes)
import pearl_c_vectors as CV  # noqa: E402
from verity.protocols.accounting.compute.pouw.verifier.audit import (HM96_SALT, ROW_SCHEMA, TILE_SCHEMA,  # noqa: E402
                                                         activation_binding, hm96_kernel_constants,
                                                         rows_domain, transcript_domain)
from verity.protocols.accounting.compute.pouw.verifier.coins import Challenger  # noqa: E402
from verity.protocols.accounting.compute.pouw.verifier.protocol import Context, Matmul, Shape  # noqa: E402
from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c as C  # noqa: E402
from verity.protocols.accounting.compute.pouw.prover.schemes.pearl_c import keys_of, unit_block  # noqa: E402, F401 - the scheme's unit keys, re-exported
from verity.protocols.accounting.compute.pouw.prover.schemes.pearl_c_device import H100  # noqa: E402

VARIANTS = (H100,)
#: the plain baseline unpromoted (gemm_fast): one accumulator over all of k on the H100's atom
H100_UNPROMOTED = replace(H100, name="h100-unpromoted", g=None)
CASE, INDEX, WEIGHT = "gpu-check", 5, 1
COINS = b"verity-pouw/pearl-c-sm90-gpu-check/coins"


def test_challenger() -> Challenger:
    return F120.test_challenger(COINS)


def salts_of(tree: str, count: int) -> bytes:
    """The fixture's hm96 salts of tree `tree` ("a": A's rows, "b": B's rows, "t": the tiles), `HM96_SALT` bytes a leaf in
    leaf order, as the kernels read them from ``salts_<tree>``: fixed bytes, so the fixture is.  A serving caller's are
    fresh bytes from the operating system."""
    return hashlib.shake_256(f"verity-pouw/pearl-c-sm90-gpu-check/salts_{tree}".encode()).digest(HM96_SALT * count)


# ---- layouts and batched pieces -------------------------------------------------------------------------------------


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


def build(out_dir: Path, m: int = 256, n: int = 384, k: int = 1024) -> dict:
    assert m % 128 == 0 and n % 128 == 0 and k % 1024 == 0, (m, n, k)
    out_dir.mkdir(parents=True, exist_ok=True)
    ra, rb = CV.rows_f32(CASE, "A", m, k), CV.rows_f32(CASE, "B", n, k)
    scheme = C.PearlC(forming="v1", device=H100, hashing="h3")
    h = scheme.commitment_hash
    mm = Matmul(INDEX, Shape(m, k, n), WEIGHT)
    coins = test_challenger()
    tiles = len(scheme.tiles(mm.shape))
    salts = {"a": salts_of("a", m), "b": salts_of("b", n), "t": salts_of("t", tiles)}
    epoch, salt_opening, tree_b, root_b = epoch_of(mm, rb, coins, salts["b"])
    dom_a = rows_domain(activation_binding(epoch.salt, INDEX), INDEX, m, h)
    tree_a = tree_bytes(dom_a, ra, ROW_SCHEMA, salts["a"])
    root_a = tree_a[-64:]
    ctx = Context(epoch.salt, mm, root_a, root_b)
    noise = scheme.noise(ctx)
    seeds = scheme.row_seeds(ctx, ra, range(m))
    keys = keys_of(scheme, epoch, mm, noise.seed_b)
    dom_t = transcript_domain(epoch, tiles, h)
    manifest = {"schema": "verity-pouw/pearl-c-sm90-gpu-check/v2", "shape": [m, n, k], "case": CASE, "layout": "rowmajor",
                "hashing": scheme.hashing, "weight": WEIGHT, "coins": coins.commitment.to_json(), "salt_opening": salt_opening.to_json(),
                "keys": {name: x.hex() for name, x in keys.items()},
                "seeds": {"seed_a": noise.seed_a.hex(), "seed_b": noise.seed_b.hex(), "salt": epoch.salt.hex(),
                          "root_a": root_a.hex(), "root_b": root_b.hex()}, "variants": {}}
    kf = Out(out_dir)
    kf.put("hm", hm96_kernel_constants(), "u8")
    manifest["key_files"] = kf.files
    sf = Out(out_dir)
    for tree, x in salts.items():
        sf.put(f"salts_{tree}", x, "u8")
    manifest["salts"] = {name.removeprefix("salts_"): f for name, f in sf.files.items()}
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
        v.put("tree_t", tree_bytes(dom_t, lv, TILE_SCHEMA, salts["t"]), "u8")
        agree = crosscheck(device, ctx, ra, rb, at, bt, pa, pb, c, u, lv, n)
        manifest["variants"][device.name] = {"scheme": C.PearlC(forming="v1", device=device, hashing="h3").name, "g": device.g,
                                             "files": v.files, "crosscheck": agree}
    (out_dir / "check.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dims = [int(x) for x in sys.argv[2:5]] or [256, 384, 1024]
    man = build(Path(sys.argv[1]), *dims)
    total = sum(f["bytes"] for part in [man["common"], *(v["files"] for v in man["variants"].values())] for f in part.values())
    print(f"fixture {man['shape']} -> {sys.argv[1]}: {total / 1e6:.1f} MB, cross-check "
          + ", ".join(f"{k} {v['crosscheck']['cells']} cells" for k, v in man["variants"].items()))
