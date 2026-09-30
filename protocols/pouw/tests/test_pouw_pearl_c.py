"""``pearl-c-h100-v0`` and ``-v1``: the chain against Pearl's own Rust (H100) and against `pearl_kw`'s replay, the clean-up's
pinning and algebra, the seeds' keying, the leaf and tickets rebuilt from their bytes, the pinned vectors, and the
lifecycle on a test instance."""

from __future__ import annotations

import hashlib
import json
import random
import struct
from fractions import Fraction
from pathlib import Path

import pytest

import pearl_c_vectors as CV
import pearl_vectors as PV
from verity.commitments.turboshake import turboshake128
from verity.ml.tc.models import HOPPER_BF16_WGMMA_K16
from verity_pouw.protocol import Context, DomainError, Matmul, PoUWScheme, Shape, Tile
from verity_pouw.schemes import get
from verity_pouw.schemes import pearl_c as C
from verity_pouw.schemes import pearl_kw as P

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
    for bad in (Shape(64, 992, 64), Shape(64, 1 << 17, 64), Shape(48, 1024, 64), Shape(64, 1040, 64)):
        with pytest.raises(DomainError):
            s.check(Matmul(0, bad, 1))
    sh = Shape(128, 2048, 256)
    assert s.wref(sh) == 128 * 256 * (2048 + 32 * 16) + 128 * 2048 * 32 + 2 * 64 * 128 * 256 + 320 * 128 * 2048


@pytest.mark.parametrize("case", [c["case"] for c in PINNED["cases"]])
def test_vectors_are_pinned(case):
    spec = next(c for c in CV.CASES if c[0] == case)
    assert CV.case_vector(*spec) == next(c for c in PINNED["cases"] if c["case"] == case)


def test_lifecycle_on_a_test_instance():
    from verity_pouw.audit import Epoch, Prover, Sampled, Verifier, Workload

    s, k = C.PearlC(k_min=32, tile=(16, 16)), 128
    rows_a, rows_b = CV.rows("life", "A", 32, k), CV.rows("life", "B", 16, k)
    work = Workload((Matmul(3, Shape(32, k, 16), 9),))
    prover = Prover(s, {9: rows_b})
    epoch = Epoch.start(work, prover.weight_roots, b"beacon")
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


def test_v1_registry_and_cost():
    s = get("pearl-c-h100-v1")
    assert s.name == "pearl-c-h100-v1" and dict(s.certificate.parameters)["forming"] == "v1"
    assert dict(get("pearl-c-h100-v0").certificate.parameters)["forming"] == "v0"
    sh = Shape(128, 2048, 256)
    v0, v1 = get("pearl-c-h100-v0").wref(sh), s.wref(sh)
    assert v0 - v1 == (C.FORMING_V0_UNITS - C.FORMING_V1_UNITS) * 128 * 2048
    with pytest.raises(ValueError):
        C.PearlC(forming="v2")
    with pytest.raises(ValueError):
        get("pearl-c-h100-v0").credit_of(sh)


def wref_of(m: int, k: int, n: int, fs: int = 64, qa: int = 4) -> Fraction:
    """`Game.lean`'s `wrefOf Params.pi Costs.adopted` (r = 32, G = 4): `creditOf` + qa·m·k."""
    mkn = Fraction(m * k * n)
    return mkn + mkn / 4 + m * k * 32 + 4 * 32 * m * n + fs * m * k + qa * m * k


@pytest.mark.parametrize("m", [1, 32, 128, 2048])
@pytest.mark.parametrize("k,n", [(128, 128), (1024, 64), (3584, 4608), (3584, 37888), (18944, 3584), (65536, 16)])
def test_wref_is_leans_wref_of(m, k, n):
    """Red team F2: W_ref is Lean's `wrefOf` on every unit shape, and the credit its `creditOf`; the weight side is not in
    either, at any m."""
    s = get("pearl-c-h100-v1")
    sh = Shape(m, k, n)
    assert s.wref(sh) == wref_of(m, k, n) and s.credit_of(sh) == wref_of(m, k, n) - 4 * m * k
    assert s.wref(Shape(2 * m, k, n)) == 2 * s.wref(sh)


def test_lifecycle_work_has_no_weight_side():
    """Red team F2: two calls on one weight give W = the two units' `wrefOf`; the weight side is `job_cost`, apart."""
    from verity_pouw.audit import Epoch, Prover, Verifier, Workload

    s, k = C.PearlC(k_min=32, tile=(16, 16), forming="v1"), 128
    work = Workload((Matmul(0, Shape(32, k, 16), 9), Matmul(1, Shape(16, k, 16), 9)))
    epoch = Epoch.start(work, Prover(s, {9: CV.rows_f32("job", "B", 16, k)}).weight_roots, b"beacon")
    assert Verifier(s, epoch).work == s.wref(Shape(48, k, 16)) == wref_of(48, k, 16)
    assert s.job_cost(16, k) == 68 * 16 * k + 16 * 32 * (k + 32) + 32 * 32 * (k + 32) + 32 * 16 * 32 * 32


def test_v1_lifecycle_on_a_test_instance():
    from verity_pouw.audit import Epoch, Prover, Sampled, Verifier, Workload

    s, k = C.PearlC(k_min=32, tile=(16, 16), forming="v1"), 128
    rows_a, rows_b = CV.rows_f32("life", "A", 32, k), CV.rows_f32("life", "B", 16, k)
    work = Workload((Matmul(3, Shape(32, k, 16), 9),))
    prover = Prover(s, {9: rows_b})
    epoch = Epoch.start(work, prover.weight_roots, b"beacon")
    tr = prover.prove(epoch, {3: rows_a})
    assert Verifier(s, epoch).audit(prover, tr, Sampled(2), b"verifier").accepted


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


def test_domain_needs_whole_promotion_groups():
    s = get("pearl-c-h100-v1")
    s.check(Matmul(0, Shape(64, 1152, 64), 1))
    with pytest.raises(DomainError):
        s.check(Matmul(0, Shape(64, 1056, 64), 1))
