"""``pearl-c-h100-v0`` and ``-v1``: the chain against Pearl's own Rust (H100) and against `pearl_kw`'s replay, the clean-up's
pinning and algebra, the seeds' keying, the leaf and tickets rebuilt from their bytes, the pinned vectors, and the
lifecycle on a test instance."""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
import sys
from fractions import Fraction
from pathlib import Path

import pytest

import pearl_c_vectors as CV
import pearl_vectors as PV
from verity.ml.crypto.turboshake import turboshake128
from verity.ml.circuits.ops.fp32 import F32DivV2, F32FmaV2, F32MulV2
from verity.ml.circuits.ops.scalar import F32Fabs, F32Fmaxf, I32Le, SelectI32
from verity.ml.circuits.silicon.models import HOPPER_BF16_WGMMA_K16
from verity.core.protocols.pouw.protocol import Context, DomainError, Matmul, PoUWScheme, Shape, Tile
from verity.ml.protocols.pouw.schemes import get
from verity.ml.protocols.pouw.schemes import pearl_c as C
from verity.ml.protocols.pouw.schemes import pearl_c_debit as D
from verity.ml.protocols.pouw.schemes import pearl_kw as P

# the clear audit is a reference tool (benchmarks/pouw/clear), imported by name
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "benchmarks" / "pouw" / "clear"))

PINNED = json.loads((Path(__file__).with_name("vectors") / "pearl_c.json").read_text())
RUST = json.loads(PV.PINNED.read_text())["outputs"]
IN = PV.inputs()


def hexwords(ws) -> str:
    return struct.pack(f"<{len(ws)}I", *ws).hex()


def codes(rng: random.Random, k: int) -> list[int]:
    """E4M3 codes over the whole finite range, with zeros and subnormals over-represented."""
    out = []
    for _ in range(k):
        r = rng.random()
        c = 0 if r < 0.1 else rng.randrange(1, 8) if r < 0.2 else rng.randrange(0, 0x7F)
        out.append(c | (0x80 if rng.random() < 0.5 else 0))
    return out


def test_chain_matches_pearl_rust_h100_matmul():
    """Every cell of the pinned matmuls against Pearl's Rust `H100::matmul_fp8` at 284b147b."""
    checked = 0
    for c, want in zip(IN["matmul"], RUST["matmul"]):
        if want["h100"] is None:
            continue
        a, b, k = bytes.fromhex(c["a"]), bytes.fromhex(c["b"]), c["k"]
        got = [C.chain(a[i * k:(i + 1) * k], b[j * k:(j + 1) * k]) for i in range(c["m"]) for j in range(c["n"])]
        assert hexwords(got) == want["h100"]
        checked += 1
    assert checked


@pytest.mark.parametrize("t", [0, 1])
def test_chain_matches_pearl_rust_h100_tile(t):
    """The full 16 x 16 H100 tile of Pearl's Rust: its noisy operands through `pearl_kw`, then C̃ by `chain`."""
    import test_pouw_pearl_kw as T

    c, want = IN["tiles"][t], RUST["tiles_h100"][t]
    (a, *_), (b, *_) = T.side(c["a"], P.H100), T.side(c["b"], P.H100)
    got = [C.chain(ra, rb) for ra in a.codes for rb in b.codes]
    assert hexwords(got) == want["acc"]


def test_chain_is_pearl_kw_replay_on_random_codes():
    """`chain` (HOPPER_E4M3_K32 atoms and fp32.add) equals `pearl_kw.replay` (Pearl's GFloat port) bit for bit, including
    partial windows (k not a multiple of 128)."""
    rng = random.Random("pearl-c-chain")
    for k in (32, 96, 128, 160, 256, 512):
        for _ in range(40):
            a, b = codes(rng, k), codes(rng, k)
            assert C.chain(a, b) == P.f32_bits(P.replay(P.H100, a, b)[0])


def test_chain_resets_every_four_atoms():
    """Pearl's own unit tests: promotion every 128 products (32,896, not the continuous 32,768), partial windows, +0."""
    k = 1024
    a = [0x78] * 128 + [0x38] * 128 + [0] * (k - 256)
    assert C.chain(a, [0x38] * k) == P.f32_bits(32896.0)
    for k, expected in ((1056, 32.0), (1088, 64.0), (1120, 96.0), (1152, 128.0)):
        assert C.chain([0] * 1024 + [0x38] * (k - 1024), [0x38] * k) == P.f32_bits(expected)
    a = [0x78, 0xF8] + [0] * 1022
    assert C.chain(a, [0x38] * 1024) == 0


def test_peel_is_four_bf16_atoms_into_c():
    rng = random.Random("pearl-c-peel")
    for _ in range(50):
        c = P.f32_bits(rng.uniform(-1e4, 1e4))
        pa = [P.f32_to_bf16(rng.uniform(-8, 8)) for _ in range(C.PEEL_K)]
        pb = [P.f32_to_bf16(rng.uniform(-8, 8)) for _ in range(C.PEEL_K)]
        acc = c
        for s in range(0, C.PEEL_K, 16):
            acc = HOPPER_BF16_WGMMA_K16.step(acc, pa[s:s + 16], pb[s:s + 16])
        assert C.peel(c, pa, pb) == acc


def instance(case: str, shape=(16, 128, 16), index=3, weight=9, salt=b"salt"):
    s = C.PearlC(k_min=32, tile=(16, 16))
    ctx = Context(hashlib.sha256(salt).digest(), Matmul(index, Shape(*shape), weight), hashlib.sha256(b"root_a").digest(),
                  hashlib.sha256(b"root_b").digest())
    a, b = CV.rows(case, "A", shape[0], shape[1]), CV.rows(case, "B", shape[2], shape[1])
    tile = Tile(tuple(range(shape[0])), tuple(range(shape[2])))
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, a, b, tile)
    return s, ctx, noise, ops, s.checked(ctx, ops), a, b


def clean(row: bytes, k: int) -> list[float]:
    return [P.bf16_to_f32(x) for x in P.open_row(*P.parse_row(row, k))]


def test_useful_output_removes_the_noise():
    """U/(α_A·α_B) tracks A·Bᵀ at FP8 accuracy, while the noisy C̃/(α_A·α_B) carries δ = 1 noise, about as large as the
    signal: the clean-up works."""
    s, ctx, noise, ops, checked, a, b = instance("useful")
    k = ctx.matmul.shape.k
    fa, fb = [clean(r, k) for r in a], [clean(r, k) for r in b]
    useful = s.useful(ctx, noise, ops, checked)

    def rel_rms(got):
        exact = [[sum(x * y for x, y in zip(ra, rb)) for rb in fb] for ra in fa]
        num = sum((g - e) ** 2 for gr, er in zip(got, exact) for g, e in zip(gr, er))
        return (num / sum(e * e for er in exact for e in er)) ** 0.5

    noisy = [[P.f32_of_bits(w) / (P.bf16_to_f32(ops.a.alpha[i]) * P.bf16_to_f32(ops.b.alpha[j])) for j, w in enumerate(row)]
             for i, row in enumerate(checked.c)]
    assert rel_rms(useful) < 0.08 and rel_rms(noisy) > 3 * rel_rms(useful)


def test_peel_factors_are_pinned():
    """P_A's second half is the fused chain A′·F_B; P_B's second half is the exact negation of bf16(β_B·E_B); G is the
    FP8 chain of the F bases."""
    s, ctx, noise, ops, checked, a, b = instance("factors")
    k = ctx.matmul.shape.k
    fb = C._basis(noise.seed_b, 1, k)
    for i in range(2):
        assert ops.pa[i][:C.R] == [C._bf16_scaled(ops.a.beta[i], e) for e in ops.e_a[i]]
        assert ops.pa[i][C.R:] == [P.f32_to_bf16(P.f32_of_bits(C.chain(ops.a.codes[i], [line[p] for line in fb])))
                                   for p in range(C.R)]
        assert ops.pb[i][C.R:] == [C._bf16_scaled(ops.b.beta[i], e) ^ 0x8000 for e in ops.e_b[i]]
    g = C.gram(noise.seed_b, k)
    fa = C._basis(noise.seed_b, 0, k)
    assert g[1][2] == C.chain([line[1] for line in fb], [line[2] for line in fa])


def test_seeds_are_keyed_as_the_design_says():
    """E_A reads (salt, root_A, root_B, index); F_A, E_B and F_B read (salt, root_B) only: per job.  A duplicated unit
    under another index gets fresh E_A and fresh C̃ (X-PC-5)."""
    s = C.PearlC(k_min=32, tile=(16, 16))

    def ctx(index=3, salt=b"s", ra=b"a", rb=b"b"):
        return Context(salt * 32, Matmul(index, Shape(16, 128, 16), 9), ra * 32, rb * 32)

    base = s.noise(ctx())
    assert s.noise(ctx(index=4)).seed_b == base.seed_b != s.noise(ctx(rb=b"c")).seed_b
    for other in (ctx(index=4), ctx(salt=b"t"), ctx(ra=b"x"), ctx(rb=b"c")):
        assert s.noise(other).seed_a != base.seed_a
    assert s.noise(ctx(ra=b"x")).seed_b == base.seed_b
    one = instance("dup", index=3)[4]
    two = instance("dup", index=4)[4]
    assert one.c != two.c and one.u != two.u


def h3_row(k: int) -> bytes:
    """The pins' FP32 row of k words: byte i is 7i + k mod 256."""
    return bytes((7 * i + k) % 256 for i in range(4 * k))


#: -h3's row digests (`row_digest`, ``sha512/row-seg/v1``) of `h3_row(k)`: one segment, two with a 176-byte last, and the
#: 16 and 56 of Llama's 4096- and 14336-wide rows; `sha512_rows.cuh`'s `s512_rows`, built for the host, writes the same
H3_DIGESTS = {
    128: "ee92fa29ad7ace8732aa9a138b3d816ce140018e9e9e9d820e3ac4a64d974c2e9410831984599a36a1e58d29d1e59d330b845fe0a98d4cc753d5093c4c70ac99",
    300: "0986088e50ea36f19742c6af42083c59323d69f0db773bc9a49c12bf86999be75a7d615614bf137fe495fe32bac4e978a41b6f5781d1a8e2034411fccbe4a3a5",
    4096: "823a21352763ecee3ba7b06da44a04525577197dbee48ef82b12927e6eb6a847360933bed78d80f85b798b77096a8746ce81dc3f48e6891d95d80d0b9432788f",
    14336: "9743c48588f788041e55fc2ef5cf44aa01225029edcf0539230053fedfcf240032ee695fd36f194c832834df8da839899cdb251e74c0090c150affc90aecadb8"}
#: -h3's row seeds over the digest of `h3_row(4096)`, at salt bytes 0 … 31 and a 64-byte root_B, bytes 32 … 95, as
#: `s512_seeds` writes them
H3_SEEDS = {(7, 3): "48f3f41d75a3d208007bdf3f7ffd1f4649de6ac8cbeae2bc03bdc306b3358896",
             (7, 200): "f1ccc13e0ae11eb9e61f36ef1a1ccfc2c8d059f5f9bd7811fd166982d1763900",
             (2 ** 32 - 1, 3): "e7c2a2ca4c4c207c3ed6075fd0b4a9055d8aacb725a930ce3d8d1f6e7d4b6a4a",
             (2 ** 32 - 1, 200): "a8f6f7a7e2f8b1672167196ad066e4857954a064f44aa010c95fe8b95dcb3b2d"}


def s512_label(role: str) -> bytes:
    """-h3's label block; its ``h3s`` is the domain tag the format was pinned under (`pearl_c.S512_LABELS`)."""
    return f"pearl-c/h3s/{role}\0".encode().ljust(128, b"\0")


def test_h3_row_digests_are_row_seg_v1_over_the_rows_bytes():
    """-h3 keys a row's seed on ``sha512/row-seg/v1`` of its bytes, rebuilt here from hashlib: SHA-512 of each 1 KiB segment
    under its 128-byte header S(j, s, m_j) (m_j in bits), then SHA-512 of the row header Q(n_bits, s, 1024) and the segment digests.  The
    seed is the first 32 bytes of SHA-512 over seed-A's label block, salt ‖ digest ‖ root_B ‖ (unit · 2^32 + row) u64 LE, so
    two rows, two units or two digests never share one; a unit or row of 2^32 or more is refused."""
    def header(tag: bytes, *fields: tuple[int, int]) -> bytes:
        return (tag + b"".join(v.to_bytes(n, "big") for v, n in fields)).ljust(128, b"\0")

    for k, pin in H3_DIGESTS.items():
        row = h3_row(k)
        segs = [row[i:i + 1024] for i in range(0, len(row), 1024)]
        s = len(segs)
        inner = b"".join(hashlib.sha512(header(b"verity/sha512-row-seg/v1/segment\0", (j, 8), (s, 8), (8 * len(g), 8)) + g).digest()
                         for j, g in enumerate(segs))
        outer = hashlib.sha512(header(b"verity/sha512-row-seg/v1\0", (8 * len(row), 8), (s, 8), (1024, 4)) + inner).digest()
        assert C.row_digest(row) == outer and outer.hex() == pin
    salt, root_b, d = bytes(range(32)), bytes(range(32, 96)), C.row_digest(h3_row(4096))
    for (unit, row), seed in H3_SEEDS.items():
        want = hashlib.sha512(s512_label("seed-A") + salt + d + root_b + (unit << 32 | row).to_bytes(8, "little")).digest()[:32]
        got = C.row_seed(salt, d, root_b, unit, row)
        assert got.hex() == seed == want.hex() and isinstance(got, C.S512Seed)
    other = bytearray(h3_row(4096))
    other[4095] ^= 1
    assert C.row_seed(salt, C.row_digest(bytes(other)), root_b, 7, 3).hex() != H3_SEEDS[7, 3]
    for unit, row in ((2 ** 32, 0), (0, 2 ** 32), (-1, 0), (0, -1)):
        with pytest.raises(ValueError, match="must each be below 2"):
            C.row_seed(salt, d, root_b, unit, row)


def test_leaf_and_tickets_rebuilt_from_bytes():
    s, ctx, noise, ops, checked, a, b = instance("leaf")
    digests = []
    for crow, urow in zip(checked.c, checked.u):
        for h in range(0, len(crow), 32):
            msg = b"".join(w.to_bytes(4, "little") for w in crow[h:h + 32]) + b"".join(w.to_bytes(4, "little") for w in urow[h:h + 32])
            digests.append(turboshake128(msg, 0x21, 32))
    assert s.digest(ctx, noise, checked) == turboshake128(b"".join(digests), 0x22, 32)
    tile = [[P.f32_of_bits(w) for w in row] for row in checked.c]
    assert s.tickets(noise, checked) == [P.hash_labelled(P.extract(tile), "jackpot", noise.seed_a)]


def test_registry_and_domain():
    s = get("pearl-c-h100-v0")
    assert isinstance(s, PoUWScheme) and s.name == "pearl-c-h100-v0" and s.certificate.conjecture == "tt-out"
    s.check(Matmul(0, Shape(64, 1024, 64), 1))
    s.check(Matmul(0, Shape(32, 1024, 64), 1))
    s.check(Matmul(0, Shape(112, 1024, 64), 1))
    for bad in (Shape(64, 992, 64), Shape(64, 1 << 17, 64), Shape(40, 1024, 64), Shape(64, 1040, 64), Shape(64, 1024, 96)):
        with pytest.raises(DomainError):
            s.check(Matmul(0, bad, 1))
    sh = Shape(128, 2048, 256)
    assert s.wref(sh) == 128 * 256 * (2048 + 32 * 15) + 128 * 2048 * 32 + 2 * 64 * 128 * 256 + 320 * 128 * 2048
    assert s.first_adds(sh) == 32 * 128 * 256


def test_the_census_k_is_in_v1s_domain_only():
    """The H100 record's census k = 256 (GLM-5.3's down projections per TP8 rank) is in -h3 v1's domain, where a row's cap
    is 4ρ (card a6872606), and no other k under 1,024 is; v0, h0 and sm_120's record have no census k."""
    h3 = get("pearl-c-h100-v1-h3")
    params = dict(h3.certificate.parameters)
    assert h3.census_ks == (256,) and params["census_ks"] == (256,) and params["census_row_cap"] == "1/100"
    assert any("TT_OUT per tile at ρr = 1/100" in h for h in h3.certificate.hypotheses)
    h3.check(Matmul(0, Shape(64, 256, 6144), 1))
    h3.check(Matmul(0, Shape(8192, 256, 6144), 1))
    for bad in (Shape(64, 128, 6144), Shape(64, 512, 6144), Shape(64, 256, 96)):
        with pytest.raises(DomainError, match="or k in 256"):
            h3.check(Matmul(0, bad, 1))
    for name in ("pearl-c-h100-v0", "pearl-c-h100-v1", "pearl-c-sm120-v1", "pearl-c-sm120-v1-h3"):
        assert get(name).census_ks == () and dict(get(name).certificate.parameters)["census_ks"] == ()
        assert "census_row_cap" not in dict(get(name).certificate.parameters)
        with pytest.raises(DomainError, match="1024 <= k"):
            get(name).check(Matmul(0, Shape(64, 256, 6144), 1))
    assert C.PearlC(k_min=32, tile=(16, 16), forming="v1").census_ks == ()


@pytest.mark.parametrize("case", [c["case"] for c in PINNED["cases"]])
def test_vectors_are_pinned(case):
    spec = next(c for c in CV.CASES if c[0] == case)
    assert CV.case_vector(*spec) == next(c for c in PINNED["cases"] if c["case"] == case)


def test_the_h100_h3_vectors_are_the_registered_schemes():
    """`pearl-c-h100-v1-h3` (GLM-5.3's FP8 form on the H200) is the scheme its pinned production case was made under, and
    the case is in its domain."""
    s, case = get("pearl-c-h100-v1-h3"), next(c for c in PINNED["cases"] if c["case"] == "decode-32-v1-h3")
    v = CV.scheme(case["test_instance"], "v1", "h3")
    assert s.name == case["scheme"] == v.name and s.commitment_hash == "hm96-sha512" and s.device is C.H100
    assert (s.k_min, s.tile, s.forming, s.hashing, s.prices) == (v.k_min, v.tile, v.forming, v.hashing, v.prices)
    s.check(Matmul(case["index"], Shape(*case["shape"]), case["weight"]))


def test_lifecycle_on_a_test_instance():
    from verity.core.protocols.pouw.audit import Epoch, Workload
    from clear_audit import Prover, Sampled, Verifier

    s, k = C.PearlC(k_min=32, tile=(16, 16)), 128
    rows_a, rows_b = CV.rows("life", "A", 32, k), CV.rows("life", "B", 16, k)
    work = Workload((Matmul(3, Shape(32, k, 16), 9),))
    prover = Prover(s, {9: rows_b})
    epoch = Epoch.unverified(work, prover.weight_roots, b"beacon")
    tr = prover.prove(epoch, {3: rows_a})
    assert Verifier(s, epoch).audit(prover, tr, Sampled(2), b"verifier").accepted


# ---- forming v1 (§5) ------------------------------------------------------------------------------------------------

def f32w(x: float) -> int:
    return P.f32_bits(x)


def test_v1_stats_pinned_on_a_known_row():
    """All ones at k = 1,024: 128 positions, S = 128, ρ = 1, α = RN(448/5), β_row = RN(RN(ρ·α)·f32(√32/(256·16)))."""
    s, rho, alpha, beta = C.s5_stats([f32w(1.0)] * 1024)
    assert (s, rho) == (f32w(1.0), f32w(1.0))
    assert alpha == f32w(P.f32(448.0 / 5.0))
    assert beta == f32w(P.f32(P.f32(P.f32_of_bits(alpha)) * P.f32_of_bits(C.S5_LINE)))
    assert C.S5_LINE == f32w(P.f32(32 ** 0.5 / 4096))
    zero = C.s5_stats([0] * 1024)
    assert zero[:2] == (0, 0) and zero[2] == f32w(448.0 * 2.0 ** 32) and zero[3] == 0


def test_v1_rho_reads_every_eighth_position_only():
    rng = random.Random("pearl-c-rho")
    x = [f32w(rng.uniform(-4, 4)) for _ in range(512)]
    y = list(x)
    for p in range(512):
        if p % C.S5_STRIDE:
            y[p] = f32w(rng.uniform(-1, 1))
    assert C.s5_stats(x)[1] == C.s5_stats(y)[1]
    exact = sum((P.f32_of_bits(x[p]) ** 2 for p in range(0, 512, 8))) / 64
    assert abs(P.f32_of_bits(C.s5_stats(x)[1]) / exact ** 0.5 - 1) < 1e-6


def v1_instance(case: str, shape=(16, 128, 16), index=3, weight=9):
    s = C.PearlC(k_min=32, tile=(16, 16), forming="v1")
    ctx = Context(hashlib.sha256(b"salt").digest(), Matmul(index, Shape(*shape), weight), hashlib.sha256(b"root_a").digest(),
                  hashlib.sha256(b"root_b").digest())
    a, b = CV.rows_f32(case, "A", shape[0], shape[1]), CV.rows_f32(case, "B", shape[2], shape[1])
    tile = Tile(tuple(range(shape[0])), tuple(range(shape[2])))
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, a, b, tile)
    return s, ctx, noise, ops, s.checked(ctx, ops), a, b


def test_v1_atom_is_alpha_x_plus_the_noise():
    """Each code is the cast of one atom with α·x as its accumulator: within the atom's 14-bit window of α·x + E′·F,
    and the noise E′·F has an RMS near σ·α."""
    s, ctx, noise, ops, checked, a, b = v1_instance("atom")
    k = ctx.matmul.shape.k
    fa = C._basis(noise.seed_b, 0, k, C.LINE_NORM_V1)
    assert fa == tuple(tuple(P.sample_line(noise.seed_b, 0, 1, t, 16.0)) for t in range(k))
    ratios = []
    for i in range(4):
        x = C.parse_f32_row(a[i], k)
        alpha, rho, ep = P.f32_of_bits(ops.a.alpha[i]), P.f32_of_bits(ops.a.rho[i]), ops.a.ep[i]
        assert ep == C.eprime(ops.a.beta[i], ops.e_a[i])
        noise_t = [sum(P.fp8_to_f32(e) * P.fp8_to_f32(f) for e, f in zip(ep, fa[t])) for t in range(k)]
        for t in range(k):
            ax = P.f32_of_bits(P.f32_bits(P.f32(alpha * P.f32_of_bits(x[t]))))
            d = P.f32_of_bits(s.device.atom.step(P.f32_bits(ax), ep, fa[t]))
            assert ops.a.codes[i][t] == P.f32_to_fp8(d)
            assert abs(d - (ax + noise_t[t])) <= 2.0 ** -12 * max(abs(ax), max(abs(n) for n in noise_t)) + 2.0 ** -20
        ratios.append((sum(n * n for n in noise_t) / k) ** 0.5 / (rho * alpha))
    assert all(0.5 < r < 2.0 for r in ratios), ratios


def test_v1_useful_output_removes_the_noise():
    s, ctx, noise, ops, checked, a, b = v1_instance("useful")
    k = ctx.matmul.shape.k
    fa = [[P.f32_of_bits(w) for w in C.parse_f32_row(r, k)] for r in a]
    fb = [[P.f32_of_bits(w) for w in C.parse_f32_row(r, k)] for r in b]
    exact = [[sum(x * y for x, y in zip(ra, rb)) for rb in fb] for ra in fa]

    def rel_rms(got):
        num = sum((g - e) ** 2 for gr, er in zip(got, exact) for g, e in zip(gr, er))
        return (num / sum(e * e for er in exact for e in er)) ** 0.5

    useful = s.useful(ctx, noise, ops, checked)
    noisy = [[P.f32_of_bits(w) / (P.f32_of_bits(ops.a.alpha[i]) * P.f32_of_bits(ops.b.alpha[j])) for j, w in enumerate(row)]
             for i, row in enumerate(checked.c)]
    assert rel_rms(useful) < 0.08 and rel_rms(noisy) > 3 * rel_rms(useful)


def test_v1_peel_uses_e_prime_exactly():
    s, ctx, noise, ops, checked, a, b = v1_instance("factors")
    for i in range(2):
        assert ops.pa[i][:C.R] == [P.f32_to_bf16(P.fp8_to_f32(c)) for c in ops.a.ep[i]]
        assert ops.pb[i][C.R:] == [P.f32_to_bf16(P.fp8_to_f32(c)) ^ 0x8000 for c in ops.b.ep[i]]


def test_bf16_of_code_keeps_the_sign_of_minus_zero():
    """`bf16(E′)` widens `−0` (0x80) to `0x8000`, with its sign. The Lean's `codeBf16` pins the same words
    (`WholeCircuit.codeBf16_examples`)."""
    assert [C._bf16_of_code(c) for c in (0x80, 0x00, 0x38, 0xFE)] == [0x8000, 0x0000, 0x3F80, 0xC3E0]


def test_eprime_keeps_the_sign_of_a_zero_noise_code():
    """`E′ = e4m3(fp32.mul(β, e))` on a noise code: `fp32.mul(β, −0)` is `−0` for `β > 0`, so the `−0` code (0x80)
    gives 0x80, and every zero product takes the XOR of the signs, `β = +0` and `β < 0` included. The NaN codes are
    refused. The Lean's `lineCodes` pins the same codes (`WholeCircuit.lineCodes_examples`)."""
    betas = (0x3F800000, 0x3AB504F3, 0xBF800000, 0x00000000, 0x40400000)
    codes = (0x00, 0x80, 0x01, 0x81, 0x38, 0xB8, 0x7E, 0xFE, 0x08, 0x88)
    assert [C.eprime(b, codes) for b in betas] == [
        [0x00, 0x80, 0x01, 0x81, 0x38, 0xB8, 0x7E, 0xFE, 0x08, 0x88],
        [0x00, 0x80, 0x00, 0x80, 0x01, 0x81, 0x32, 0xB2, 0x00, 0x80],
        [0x80, 0x00, 0x81, 0x01, 0xB8, 0x38, 0xFE, 0x7E, 0x88, 0x08],
        [0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80],
        [0x00, 0x80, 0x03, 0x83, 0x44, 0xC4, 0x7E, 0xFE, 0x14, 0x94]]
    for c in (0x7F, 0xFF):
        with pytest.raises(P.Reject):
            C.eprime(0x3F800000, [c])


def test_f32_mul_signs_a_zero_product():
    """`F32Mul_v2` writes a zero product of opposite signs as `−0` and of equal signs as `+0`, as the Lean's
    `Pc8OpsAgree.mulZero` and `.mul` state."""
    pairs = ((0, 0xBF800000), (0x80000000, 0x3F800000), (1, 0x80000000), (0x80000000, 0xBF800000), (0, 0x3F800000))
    assert [F32MulV2.evaluate(a, b) for a, b in pairs] == [0x80000000] * 3 + [0, 0]


def test_a_row_words_sign_of_zero_reaches_neither_its_magnitude_nor_its_square():
    """`F32Fabs_v1` clears the sign bit, and `F32Fma_v2(−0, −0, c)` is `F32Fma_v2(+0, +0, c)` on every word `c`, the
    infinities and NaNs included, as the Lean's `Pc8OpsAgree.fabs` and `.fmaSq0` state: a row's `−0` words give its
    `+0` words' `s` and `ρ`."""
    rng = random.Random(7)
    specials = [0, 0x80000000, 1, 0x80000001, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00001,
                0x7F800001, 0x3F800000, 0xBF800000]
    words = specials + [rng.getrandbits(32) for _ in range(4000)]
    assert all(F32Fabs.evaluate(w) == w & 0x7FFFFFFF for w in words)
    assert all(F32FmaV2.evaluate(0x80000000, 0x80000000, c) == F32FmaV2.evaluate(0, 0, c) for c in words)


def f32_word(q: Fraction) -> int:
    """The Lean's `f32Word`: one nearest-even rounding of `q` to FP32, with gradual underflow and overflow to `±inf`;
    an exact zero is `+0`."""
    if q == 0:
        return 0
    s, a = (0x80000000 if q < 0 else 0), abs(q)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    e += (Fraction(2) ** (e + 1) <= a) - (Fraction(2) ** e > a)
    big_e = max(e, -126)
    n = round(a / Fraction(2) ** (big_e - 23))
    if n == 2 ** 24:
        big_e, n = big_e + 1, 2 ** 23
    if big_e > 127:
        return s | 0x7F800000
    return s | n if n < 2 ** 23 else s | (big_e + 127) << 23 | (n - 2 ** 23)


def test_f32_div_fmaxf_and_select_as_the_lean_states():
    """`F32Div_v2` rounds finite words' exact quotient by a nonzero word as `f32Word` (a zero quotient of equal signs
    `+0`) and writes a finite word over `+inf` as the zero of its sign; `F32Mul_v2` of a non-finite first factor by a
    finite word is not finite; `F32Fmaxf` against a finite nonzero word is the first operand where it is `+inf` or
    larger, else the second (a NaN, `−inf` or a smaller value); and `SelectI32` selects on its condition's low bit. These
    are the Lean's `Pc8OpsAgree.div`, `.divInf`, `.mulNonfin`, `.fmaxfAny` and `.sel`."""
    rng = random.Random(11)
    specials = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000,
                0x7FC00000, 0xFFFFFFFF, 0x7F800001, 0x3F800000, 0xBF800000, 0x2F800000, 0x43E00000, 0x45280000]
    words = specials + [rng.getrandbits(32) for _ in range(1500)]
    finite = {w for w in words if (w >> 23) & 0xFF != 0xFF}
    nonfinite = set(words) - finite

    def val(w: int) -> Fraction:
        return Fraction(P.f32_of_bits(w))

    pairs = [(a, b) for a in specials for b in specials] + [(rng.choice(words), rng.choice(words)) for _ in range(4000)]
    divs = [(a, b) for a, b in pairs if a in finite and b in finite and val(b) != 0 and (val(a) != 0 or a >> 31 == b >> 31)]
    assert len(divs) > 2000
    assert all(F32DivV2.evaluate(a, b) == f32_word(val(a) / val(b)) for a, b in divs)
    assert all(F32DivV2.evaluate(a, 0x7F800000) == a & 0x80000000 for a in finite)
    assert all((F32MulV2.evaluate(a, b) >> 23) & 0xFF == 0xFF for a in nonfinite for b in specials + words[-200:]
               if b in finite)
    maxes = [(a, b) for a, b in pairs if b in finite and val(b) != 0]
    assert all(F32Fmaxf.evaluate(a, b) == (a if a == 0x7F800000 or (a in finite and val(b) < val(a)) else b)
               for a, b in maxes)
    assert all(SelectI32.evaluate(c, a, b) == (a if c % 2 == 1 else b) for c in range(4) for a, b in pairs[:500])


def test_i32_le_of_words_with_bit_31_clear_is_their_order_as_the_lean_states():
    """`I32Le_v1` (signed 32-bit `a ≤ b`) of two words below `2³¹` is their order, as the Lean's `Pc4OpsAgree.le`
    states: `pc4`'s running max `s` compares `F32Fabs_v1` words, whose sign bit is clear."""
    rng = random.Random(13)
    specials = [0, 1, 0x007FFFFF, 0x00800000, 0x3F800000, 0x3F800001, 0x7F7FFFFF, 0x7F800000, 0x7FC00000, 0x7FFFFFFF]
    words = specials + [rng.getrandbits(31) for _ in range(1500)]
    pairs = [(a, b) for a in specials for b in specials] + [(rng.choice(words), rng.choice(words)) for _ in range(4000)]
    assert all(I32Le.evaluate(a, b) == (1 if a <= b else 0) for a, b in pairs)


def test_f32_fma_overflow_and_div_of_a_nonfinite_word_as_the_lean_states():
    """`F32Fma_v2` of finite words whose exact `x·y + z` overflows writes `f32Word` of it (`±inf`), as the squares-overflow
    row's `fma(2¹²⁶, 2¹²⁶, +0)` does; and `F32Div_v2` of a non-finite word by a finite one (zero included) is not finite.
    These are the Lean's `Pc8OpsAgree.fmaOvf` and `.divNonfin`."""
    rng = random.Random(12)

    def val(w: int) -> Fraction:
        return Fraction(P.f32_of_bits(w))

    def big() -> int:
        return rng.getrandbits(1) << 31 | rng.randrange(0xC0, 0xFF) << 23 | rng.getrandbits(23)

    def finite_word() -> int:
        w = rng.getrandbits(32)
        return w if (w >> 23) & 0xFF != 0xFF else w & 0xFF7FFFFF

    triples = [(0x7E800000, 0x7E800000, 0), (0x7F7FFFFF, 0x7F7FFFFF, 0xFF7FFFFF), (0x7F7FFFFF, 0x3F800000, 0x73800000),
               (0xFF7FFFFF, 0x3F800000, 0xF3800000)]
    triples += [(big(), big(), rng.choice([finite_word(), big(), 0, 0x80000000])) for _ in range(6000)]
    ovf = [(a, b, c) for a, b, c in triples
           if (f32_word(val(a) * val(b) + val(c)) >> 23) & 0xFF == 0xFF]
    assert len(ovf) > 1000 and (0x7E800000, 0x7E800000, 0) in ovf
    assert all(F32FmaV2.evaluate(a, b, c) == f32_word(val(a) * val(b) + val(c)) for a, b, c in ovf)
    nonfinite = [0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFFFFFFF, 0x7FFFFFFF]
    nonfinite += [rng.getrandbits(1) << 31 | 0xFF << 23 | rng.getrandbits(23) for _ in range(200)]
    finite = [0, 0x80000000, 1, 0x3F800000, 0x41800000, 0x7F7FFFFF, 0xFF7FFFFF] + [finite_word() for _ in range(200)]
    assert all((F32DivV2.evaluate(a, b) >> 23) & 0xFF == 0xFF for a in nonfinite for b in finite)


def test_v1_registry_and_cost():
    s = get("pearl-c-h100-v1")
    assert s.name == "pearl-c-h100-v1" and dict(s.certificate.parameters)["forming"] == "v1"
    assert dict(get("pearl-c-h100-v0").certificate.parameters)["forming"] == "v0"
    sh = Shape(128, 2048, 256)
    v0, v1 = get("pearl-c-h100-v0").wref(sh), s.wref(sh)
    assert v0 - v1 == (C.FORMING_V0_UNITS - C.H100_PRICES.forming) * 128 * 2048
    with pytest.raises(ValueError):
        C.PearlC(forming="v2")
    with pytest.raises(ValueError):
        get("pearl-c-h100-v0").credit_of(sh)


def wref_of(m: int, k: int, n: int, fs: int = 64, qa: int = 4, add: int = 32, bf16: int = 2) -> Fraction:
    """`DeviceRev1.lean`'s `wrefOfRev1 Params.pi Costs.adopted` (r = 32, G = 4), `creditOfRev1` + qa·m·k; at other prices
    the device statement's `wrefDevRev1`: mkn + mkn·add/(32·G) − add·m·n + m·k·r + 2·bf16·r·m·n + fs·m·k + qa·m·k, the
    promotion adds less the first of every word (`firstAddDev`)."""
    mkn = Fraction(m * k * n)
    return mkn + mkn * add / 128 - add * m * n + m * k * 32 + 2 * bf16 * 32 * m * n + fs * m * k + qa * m * k


#: sm_120's measured W1 rates (node 2, locked 2,100 MHz; `r20260930-063213-c94a`, `r20260930-061717-3e84`), per SM per
#: clock: the unit, a dense E4M3 `mma.sync` MAC with FP32 accumulation; FADD in the measured 64-add loop, and alone
#: (issue-bound, from the two-loop fit); FFMA alone; and the BF16 MAC's and the E4M3 cast's prices (alone, and in the
#: packing loop)
SM120_MEASURED = {"fp8_mac": 1023.98, "fadd_loop": 122.26, "fadd_alone": 128, "ffma_alone": 128, "bf16_mac": 2.000,
                  "cast_alone": 8.0, "cast_loop": 12.31}


def test_prices_are_per_device_and_sm120s_are_the_measured_ones():
    """sm_90's prices are the H100's (`Prices.h100`); sm_120's are the device statement's `Prices.sm120`, each step at its
    cheapest bit-exact form on the card, and none above what the card measures in the loop.  Crediting sm_120 at the
    H100's FADD of 32 (and forming of 64 + 4) overstates the credit; at sm_120's prices the scheme is `wrefDevRev1`, and
    the debit per flagged atom-word, element and later +0 promotion follows the same record."""
    h, s, m = C.PRICES["sm_90"], C.PRICES["sm_120"], SM120_MEASURED
    assert (h, s) == (C.H100_PRICES, C.SM120_PRICES)
    assert (h.fadd, h.bf16_mac, h.forming_credited, h.forming_a_only) == (32, 2, 64, 4)
    assert (s.fadd, s.bf16_mac, s.forming_credited, s.forming_a_only) == (8, 2, 40, 1)
    assert s.fadd == round(m["fp8_mac"] / m["fadd_alone"]) and s.fadd <= m["fp8_mac"] / m["fadd_loop"] < 8.38
    assert s.bf16_mac == m["bf16_mac"]
    assert s.forming_credited == 32 + m["cast_alone"] <= 32 + m["cast_loop"]
    assert s.forming_a_only == round(m["fp8_mac"] / m["ffma_alone"]) / 8
    sm120 = C.PearlC(forming="v1", hashing="h3", prices=s)
    assert sm120.name == "pearl-c-h100-v1-h3@sm_120" and dict(sm120.certificate.parameters)["prices"] == "sm_120"
    assert dict(C.PearlC(forming="v1", hashing="h3").certificate.parameters)["prices"] == "sm_90"
    for sh in (Shape(32, 1024, 64), Shape(2048, 3584, 18944), Shape(8192, 8192, 8192)):
        assert sm120.wref(sh) == wref_of(sh.m, sh.k, sh.n, fs=40, qa=1, add=8, bf16=2)
        assert sm120.credit_of(sh) == sm120.wref(sh) - sh.m * sh.k < get("pearl-c-h100-v1").credit_of(sh)
    assert D.debit_units(1, 0, s, zeros=0) == 34 and D.debit_units(0, 1, s, zeros=0) == 40
    assert D.debit_units(1, 1, zeros=0) == 40 + 64 and D.debit_units(0, 0, s, zeros=1) == 8
    assert D.debit_units(0, 0, zeros=1) == 32
    with pytest.raises(ValueError):
        C.PearlC(forming="v0", prices=s)


@pytest.mark.parametrize("m", [1, 32, 128, 2048])
@pytest.mark.parametrize("k,n", [(128, 128), (1024, 64), (3584, 4608), (3584, 37888), (18944, 3584), (65536, 16)])
def test_wref_is_leans_wref_of(m, k, n):
    """Red team F2: W_ref is Lean's `wrefOfRev1` on every unit shape, and the credit its `creditOfRev1`; the weight side
    is not in either, at any m."""
    s = get("pearl-c-h100-v1")
    sh = Shape(m, k, n)
    assert s.wref(sh) == wref_of(m, k, n) and s.credit_of(sh) == wref_of(m, k, n) - 4 * m * k
    assert s.wref(Shape(2 * m, k, n)) == 2 * s.wref(sh)


def test_lifecycle_work_has_no_weight_side():
    """Red team F2: two calls on one weight give W = the two units' `wrefOfRev1`; the weight side is `job_cost`, apart."""
    from verity.core.protocols.pouw.audit import Epoch, Workload
    from clear_audit import Prover, Verifier

    s, k = C.PearlC(k_min=32, tile=(16, 16), forming="v1"), 128
    work = Workload((Matmul(0, Shape(32, k, 16), 9), Matmul(1, Shape(16, k, 16), 9)))
    epoch = Epoch.unverified(work, Prover(s, {9: CV.rows_f32("job", "B", 16, k)}).weight_roots, b"beacon")
    assert Verifier(s, epoch).work == s.wref(Shape(48, k, 16)) == wref_of(48, k, 16)
    assert s.job_cost(16, k) == 68 * 16 * k + 16 * 32 * (k + 32) + 32 * 32 * (k + 32) + 32 * 16 * 32 * 32


def test_v1_lifecycle_on_a_test_instance():
    from verity.core.protocols.pouw.audit import Epoch, Workload
    from clear_audit import Prover, Sampled, Verifier

    s, k = C.PearlC(k_min=32, tile=(16, 16), forming="v1"), 128
    rows_a, rows_b = CV.rows_f32("life", "A", 32, k), CV.rows_f32("life", "B", 16, k)
    work = Workload((Matmul(3, Shape(32, k, 16), 9),))
    prover = Prover(s, {9: rows_b})
    epoch = Epoch.unverified(work, prover.weight_roots, b"beacon")
    tr = prover.prove(epoch, {3: rows_a})
    assert Verifier(s, epoch).audit(prover, tr, Sampled(2), b"verifier").accepted


def test_each_hashing_format_is_on_the_scheme():
    """Each instance carries its format's tree hash, so a device path reads it from the scheme; h0's messages are
    TurboSHAKE128, and a format outside `HASHINGS` is refused."""
    msg = bytes(range(64))
    assert C.HASHINGS == ("h0", "h3")
    for hashing in C.HASHINGS:
        assert C.PearlC(forming="v1", hashing=hashing).commitment_hash == C.TREE_HASH[hashing]
    assert C.message_digest(msg) == C.turboshake128(msg, C.D_MESSAGE, 32)
    for bad in ("h1", "h2", "h3s", "h4"):
        with pytest.raises(ValueError):
            C.PearlC(hashing=bad)


def test_h3_is_sha512_throughout():
    """Hashing format -h3, SHA-512 throughout, each rebuilt here from hashlib: seed_B, seed_A and
    row i's seed (keyed on its `row_digest`) are H_ℓ, the first 32 bytes of SHA-512 after ℓ's 128-byte block; a line's 32
    bytes are H_"line"(H_"noise-line"(seed) ‖ side ‖ factor ‖ u32 LE line) under Pearl's map to codes; the tile leaf is
    ``sha512/row-seg/v1`` over the tile's messages; a ticket is H_"jackpot"(seed_A ‖ fold).  E_A moves with a row's bytes
    and position alone; no BLAKE3 claim is left."""
    from verity.ml.commitments.rowleaf import sha512_row_seg_digest
    from verity.core.protocols.pouw.serving import HASH_FORMATS, SchemeId, hash_format

    s, h0 = (C.PearlC(k_min=32, tile=(16, 16), forming="v1", hashing=h) for h in ("h3", "h0"))
    assert s.name == "pearl-c-h100-v1-h3" and s.commitment_hash == "hm96-sha512"
    assert "cr/sha-512" in s.certificate.claims and not any("blake3" in c for c in s.certificate.claims)
    assert dict(s.certificate.parameters)["hashing"] == "h3" and C.ROW_SEEDED == ("h3",)
    assert SchemeId.parse("pearl-c-sm120-v1-h3").hashing == "h3" and hash_format(s) is HASH_FORMATS["h3"]
    assert HASH_FORMATS["h3"].row_digest == "sha512/row-seg/v1"
    ctx, k = CV.ctx_of("h3-rows", (16, 384, 16), 3, 9, wide=True), 384
    a, b = CV.rows_f32("h3-rows", "A", 16, k), CV.rows_f32("h3-rows", "B", 16, k)
    tile, noise = Tile(tuple(range(16)), tuple(range(16))), s.noise(ctx)
    H = lambda role, msg: hashlib.sha512(s512_label(role) + msg).digest()[:32]  # noqa: E731
    assert noise.seed_b == H("seed-B", ctx.salt + ctx.root_b) and isinstance(noise.seed_b, C.S512Seed)
    assert noise.seed_a == H("seed-A", ctx.salt + ctx.root_a + ctx.root_b + (3).to_bytes(8, "little"))
    seeds = s.row_seeds(ctx, a, tile.rows)
    assert seeds == [H("seed-A", ctx.salt + C.row_digest(a[i]) + ctx.root_b + (3 << 32 | i).to_bytes(8, "little")) for i in range(16)]
    assert len(set(seeds)) == 16
    for seed, side, factor, t in ((seeds[0], 0, 0, 0), (noise.seed_b, 1, 1, 17)):
        raw = H("line", H("noise-line", seed) + bytes([side, factor]) + t.to_bytes(4, "little"))
        xs = [(-1 if x >> 7 else 1) * ((x & 0x7F) + 1) for x in raw]
        norm = P.NOISE_NORM if factor == 0 else C.LINE_NORM_V1
        scale = P.bf16_div(P.f32_to_bf16(norm * 32), P.f32_to_bf16(float(math.isqrt(sum(x * x for x in xs) * 1024))))
        want = [P.f32_to_fp8(P.bf16_to_f32(P.bf16_mul(P.f32_to_bf16(float(x)), scale))) for x in xs]
        assert C.sample_line(seed, side, factor, t, norm) == want != P.sample_line(bytes(seed), side, factor, t, norm)
    assert C._basis(noise.seed_b, 1, k, C.LINE_NORM_V1) != C._basis(bytes(noise.seed_b), 1, k, C.LINE_NORM_V1)
    ops = s.encode(ctx, noise, a, b, tile)
    assert ops.e_a == [C.sample_line(x, 0, 0, 0) for x in seeds]
    assert ops.e_b == [C.sample_line(noise.seed_b, 1, 0, j) for j in range(16)]
    checked = s.checked(ctx, ops)
    msgs = b"".join(struct.pack("<16I", *cr) + struct.pack("<16I", *ur) for cr, ur in zip(checked.c, checked.u))
    assert s.digest(ctx, noise, checked) == sha512_row_seg_digest(msgs, 8 * len(msgs)) and len(msgs) == 2048
    folds = [P.extract([[P.f32_of_bits(w) for w in row] for row in checked.c])]
    assert s.tickets(noise, checked) == [H("jackpot", noise.seed_a + f) for f in folds]
    other = list(a)
    other[5] = a[6]
    e = s.encode(ctx, noise, other, b, tile).e_a
    assert [i for i in range(16) if e[i] != ops.e_a[i]] == [5] and e[5] != ops.e_a[6]
    moved = Context(ctx.salt, ctx.matmul, bytes(64), ctx.root_b)
    assert s.encode(moved, s.noise(moved), a, b, tile).e_a == ops.e_a and s.noise(moved).seed_a != noise.seed_a
    assert ops.e_a != h0.encode(ctx, h0.noise(ctx), a, b, tile).e_a


def test_hm96_salts_are_chacha20s_stream():
    """The prover's hm96 salts are ChaCha20 blocks under its key (RFC 8439 §2.3.2's block, its 96-bit nonce and 32-bit counter
    as this layout's 64-bit counter and nonce), leaf i's three blocks 3i … 3i + 2."""
    from verity.core.protocols.pouw.audit import chacha20_block, hm96_salt

    block = chacha20_block(bytes(range(32)), 1 | 0x09000000 << 32, 0x4A000000)
    assert block.hex() == ("10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e"
                           "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e")
    key = bytes(range(32, 64))
    assert hm96_salt(key, 5, 2) == b"".join(chacha20_block(key, c, 5) for c in (6, 7, 8))


def test_h3_lifecycle_commits_hm96_sha512_trees():
    """An -h3 transcript is accepted by an -h3 verifier.  Every tree is ``frame-v3-sha512`` with hm96-sha512 leaves under
    the pinned key and the prover's salts, rebuilt here from core's pieces: A's and B's rows over their ``sha512/row-seg/v1``
    digests, the tiles over their 64-byte leaves.  A wrong salt, or h0's verifier (SHA-256 trees), refuses the opening."""
    from dataclasses import replace

    from verity.ml.commitments import CommitmentDomain, hm96
    from verity.ml.commitments.merkle import tree_levels
    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG, row_leaf
    from verity.core.protocols.pouw.audit import (HM96_TRANSCRIPT, HM96_WEIGHTS, Epoch, Workload, activation_binding, hm96_salt,
                                                  rows_domain, transcript_domain, weight_binding)
    from clear_audit import Prover, Sampled, Verifier

    s, k = C.PearlC(k_min=32, tile=(16, 16), forming="v1", hashing="h3"), 128
    rows_a, rows_b = CV.rows_f32("life", "A", 32, k), CV.rows_f32("life", "B", 16, k)
    work = Workload((Matmul(3, Shape(32, k, 16), 9),))
    key = bytes(range(100, 132))
    prover = Prover(s, {9: rows_b}, salt_key=key)
    epoch = Epoch.unverified(work, prover.weight_roots, b"beacon")
    tr = prover.prove(epoch, {3: rows_a})
    assert Verifier(s, epoch).audit(prover, tr, Sampled(2), b"verifier").accepted
    K = hm96.SHA512.default_key

    def root(hm, xs, nonce):
        dom = CommitmentDomain(hm.binding, hm.owner, hm.positions, hash="sha512")
        hls = [hm96.SHA512.tree_leaf(K, hm96.SHA512.commit_string(K, x, hm96_salt(key, nonce, i))) for i, x in enumerate(xs)]
        return tree_levels(dom, [row_leaf(dom, i, hl, SCHEMA_HM96_SHA512_ROW_SEG) for i, hl in enumerate(hls)])[-1][0]

    da, db = rows_domain(activation_binding(epoch.salt, 3), 3, 32, "hm96-sha512"), rows_domain(weight_binding(9), -2, 16, "hm96-sha512")
    assert da.hash == db.hash == "sha512" and len(tr.activation_roots[0]) == 64
    assert tr.activation_roots[0] == root(da, [C.row_digest(r) for r in rows_a], 3)
    assert prover.weight_roots[9] == root(db, [C.row_digest(r) for r in rows_b], HM96_WEIGHTS + 9)
    digests = [prover.digest(t) for t in range(2)]
    assert tr.commitment.root == root(transcript_domain(epoch, 2, "hm96-sha512"), digests, HM96_TRANSCRIPT)
    opening = prover.open(epoch, 0)
    bad = replace(opening, a=(replace(opening.a[0], value=opening.a[0].value[:-1] + b"\0"),) + opening.a[1:])
    assert Verifier(s, epoch).check_tile(tr, bad).reason == "activation opening"
    h0 = Verifier(C.PearlC(k_min=32, tile=(16, 16), forming="v1"), epoch).check_tile(tr, opening)
    assert not h0.ok and h0.reason == "activation opening"


def test_v1_domain_rules():
    """Pearl's row rules with σ: a zero row fails the noise floor (and stays finite), a side with more than 1/64 of its
    entries at |x| >= 8σ fails liveness; the generated rows pass."""
    s, ctx, noise, ops, checked, a, b = v1_instance("domain")
    assert s.admissible(ctx, ops)
    k = ctx.matmul.shape.k
    zero = [C.f32_row([0] * k)] + a[1:]
    tile = Tile(tuple(range(16)), tuple(range(16)))
    ops0 = s.encode(ctx, noise, zero, b, tile)
    assert ops0 is not None and all(c == 0 for c in ops0.a.codes[0]) and not s.admissible(ctx, ops0)
    spiky = []
    for r in a:
        w = list(C.parse_f32_row(r, k))
        for p in range(1, k, 8)[:4]:
            w[p] = f32w(1e4)
        spiky.append(C.f32_row(w))
    assert not s.admissible(ctx, s.encode(ctx, noise, spiky, b, tile))


def test_v1_liveness_is_per_row_on_the_exact_statistic():
    """One row with 3 entries at 8ρ among 128 fails (64·3 > 128), though its side has far fewer than 1/64 dead entries in
    all; 2 such entries pass.  An entry exactly at x² = 64ρ² counts as dead (≥), with ρ² the exact every-8th mean."""
    k = 128
    base = [f32w(1.0)] * k
    assert C.live_row(base)
    two, three = list(base), list(base)
    for p in (1, 2):
        two[p] = f32w(8.0)
    for p in (1, 2, 3):
        three[p] = f32w(8.0)
    assert C.live_row(two) and not C.live_row(three)
    just_below = list(three)
    just_below[3] = f32w(P.f32_of_bits(f32w(8.0)) - 2.0 ** -20)
    assert C.live_row(just_below)
    s = C.PearlC(k_min=32, tile=(16, 16), forming="v1")
    ctx = CV.ctx_of("liveness", (16, k, 16), 3, 9)
    noise = s.noise(ctx)
    rows_a = [C.f32_row(base)] * 15 + [C.f32_row(three)]
    rows_b = CV.rows_f32("liveness", "B", 16, k)
    ops = s.encode(ctx, noise, rows_a, rows_b, Tile(tuple(range(16)), tuple(range(16))))
    assert sum(1 for row in ops.a.x for w in row if P.f32_of_bits(w) >= 8.0) * 64 <= 16 * k
    assert not s.admissible(ctx, ops)


def tolerant_kernel(scheme, ctx, noise, a, b, tiles):
    """`reference_kernel`, with a zero digest for a tile the scheme does not encode, so a weight with a refused row
    still commits."""
    out = []
    for t in tiles:
        ops = scheme.encode(ctx, noise, [a[i] for i in t.rows], [b[j] for j in t.cols], t)
        if ops is None:
            out.append((bytes(32), [[0.0] * len(t.cols) for _ in t.rows]))
            continue
        c = scheme.checked(ctx, ops)
        out.append((scheme.digest(ctx, noise, c), scheme.useful(ctx, noise, ops, c)))
    return out


@pytest.mark.parametrize("clause", [None, "finite", "live", "floor"])
def test_check_tile_refuses_a_weight_row_outside_the_rules(clause):
    """`check_tile` refuses a tile one of whose weight rows fails a clause of `OperandOK`, as Lean's `TileCheck.Accepts`
    asks each of the tile's weight rows to pass the protocol's `weightOK` (`PearlCSem.RowOK`). One row per clause: a
    non-finite word (Lean's `rowOK8_nonFinite`), three of 128 words `8.0` among `1.0`s (`rowOK8_notLive`), and every 8th
    word `2¹²⁶`, so ρ is not finite and the floor is 0 (`pc8Sem_squaresOverflow`). The tile beside it, whose weight rows
    pass, is accepted, and so are both on the honest weight."""
    from verity.core.protocols.pouw.audit import Epoch, Workload
    from clear_audit import Prover, Verifier

    s, k = C.PearlC(k_min=32, tile=(16, 16), forming="v1"), 128
    rows_a, rows_b = CV.rows_f32("weight-rule", "A", 16, k), CV.rows_f32("weight-rule", "B", 32, k)
    row = list(C.parse_f32_row(rows_b[0], k))
    if clause == "finite":
        row[1] = 0x7F800000
    elif clause == "live":
        row = [f32w(8.0) if t in (1, 2, 3) else f32w(1.0) for t in range(k)]
    elif clause == "floor":
        row = [0x7E800000 if t % 8 == 0 else w for t, w in enumerate(row)]
    rows_b[0] = C.f32_row(row)
    work = Workload((Matmul(3, Shape(16, k, 32), 9),))
    prover = Prover(s, {9: rows_b}, tolerant_kernel)
    epoch = Epoch.unverified(work, prover.weight_roots, b"beacon")
    tr = prover.prove(epoch, {3: rows_a})
    v = Verifier(s, epoch)
    assert [0 in r.tile.cols for r in v.refs] == [True, False]
    verdicts = [v.check_tile(tr, prover.open(epoch, r.id)) for r in v.refs]
    assert [(x.ok, x.reason) for x in verdicts] == [(clause is None, "" if clause is None else "inadmissible inputs"),
                                                    (True, "")]


def test_domain_needs_whole_promotion_groups():
    s = get("pearl-c-h100-v1")
    s.check(Matmul(0, Shape(64, 1152, 64), 1))
    with pytest.raises(DomainError):
        s.check(Matmul(0, Shape(64, 1056, 64), 1))


@pytest.mark.parametrize("name", ["pearl-c-h100-v0", "pearl-c-h100-v1", "pearl-c-h100-v1-h3", "pearl-c-sm120-v1",
                                  "pearl-c-sm120-v1-h3"])
def test_the_domain_has_lean_s_n_floor(name):
    """Lean's `pearlCDomainDev` takes `64 <= n` and `64 | n` (a 1 x (n mod 64) edge tile's words can be guessed
    whole); the 64-column tile is that floor, and it keeps GLM-5.3's n = 512 w13 and gate_up and n = 256 router at TP8."""
    s = get(name)
    assert s.tile[1] == 64
    for n in (16, 32, 63, 96, 160, 1000):
        with pytest.raises(DomainError):
            s.check(Matmul(0, Shape(32, 6144, n), 1))
    for n in (64, 256, 512, 1024, 4096):
        s.check(Matmul(0, Shape(32, 6144, n), 1))
