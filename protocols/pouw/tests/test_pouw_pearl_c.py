"""``pearl-c-h100-v0``: the chain against Pearl's own Rust (H100) and against `pearl_kw`'s replay, the clean-up's
pinning and algebra, the seeds' keying, the leaf and tickets rebuilt from their bytes, the pinned vectors, and the
lifecycle on a test instance."""

from __future__ import annotations

import hashlib
import json
import random
import struct
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
    windows = 2048 // 128
    assert s.wref(sh) == (128 * 256 + 128 * 32 + 256 * 32) * (2048 + 32 * windows) + 2 * 64 * 128 * 256 + 320 * (128 + 256) * 2048


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
