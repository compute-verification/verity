"""``pearl-c-sm120-v1``: Pearl-C's device seam.  The H100's record is every other file's default, so their pinned vectors are
its outputs; here the sm_120 record carries the card's captured atom and names its stand-ins, `pearl_c_u` restates its steps
(and a multi-group atom group by group), the scheme and its debit run on the record's atom, the accounting reads the
record's G and prices, and the unpromoted record chains one accumulator over all of k."""

from __future__ import annotations

import dataclasses
import random
from fractions import Fraction

import pytest

import quicknet as Q

import pearl_c_u_vectors as UV
import pearl_c_vectors as CV
import test_pouw_pearl_c_debit as TD
import test_pouw_pearl_c_u as TU
from verity.ml.tc.models import (ADA_E4M3_M16N8K32, BLACKWELL_SM120_E4M3_M16N8K32, HOPPER_BF16_M16N8K16,
                                 HOPPER_E4M3_K32)
from verity_pouw.protocol import Matmul, Shape, Tile
from verity_pouw.schemes import get
from verity_pouw.schemes import pearl_c as C
from verity_pouw.schemes import pearl_c_debit as D
from verity_pouw.schemes import pearl_c_u as U
from verity_pouw.schemes import pearl_c_work as W
from verity_pouw.schemes import pearl_kw as P
from verity_pouw.schemes.pearl_c_device import DEVICES, H100, SM120, SM120_UNPROMOTED, Device, Prices

SM120_ATOM = BLACKWELL_SM120_E4M3_M16N8K32


def test_records_are_leans_pipelines():
    assert DEVICES == {"h100": H100, "sm120": SM120, "sm120-unpromoted": SM120_UNPROMOTED}
    assert H100.pipelines == {"atom": ((32,), 14, -139), "peel": ((16,), 26, -133)} and H100.window == 128
    assert SM120.pipelines == {"atom": ((32,), 26, -133), "peel": ((16,), 26, -133)} and SM120.window == 128
    assert H100.standins == () and [s.split(":")[0] for s in SM120.standins] == ["peel", "g", "k_max"]
    assert SM120_UNPROMOTED.pipelines == SM120.pipelines and SM120_UNPROMOTED.window == 32
    assert [s.split(":")[0] for s in SM120_UNPROMOTED.standins] == ["peel", "k_max", "g"]
    assert (H100.promotions(8192), SM120.promotions(8192), SM120_UNPROMOTED.promotions(8192)) == (64, 64, 0)
    with pytest.raises(ValueError):
        dataclasses.replace(SM120, atom=HOPPER_BF16_M16N8K16)
    with pytest.raises(ValueError):
        dataclasses.replace(SM120, g=0)


def test_registry_names_the_device_and_its_standins():
    s = get("pearl-c-sm120-v1")
    assert s.name == "pearl-c-sm120-v1" and s.device is SM120 and s.forming == "v1"
    params = dict(s.certificate.parameters)
    assert (params["device"], params["atom"], params["peel"], params["g"], params["window"]) == (
        "sm120", "blackwell_sm120_e4m3_m16n8k32", "hopper_bf16_m16n8k16", 4, 128)
    assert all(f"stand-in: {x}" in s.certificate.hypotheses for x in SM120.standins)
    assert dict(get("pearl-c-h100-v1").certificate.parameters)["device"] == "h100"
    assert not any(h.startswith("stand-in") for h in get("pearl-c-h100-v1").certificate.hypotheses)
    with pytest.raises(ValueError):
        C.PearlC(forming="v0", device=SM120)


def test_u_restates_the_sm120_steps():
    """`pearl_c_u`'s integer statement against the sm120 record's steps, and group by group against Ada's two groups of 16
    (a multi-group atom the seam still takes)."""
    rng = random.Random("pearl-c-sm120/grouped")
    ada = dataclasses.replace(SM120, atom=ADA_E4M3_M16N8K32)
    for _ in range(1500):
        acc = TU.acc_word(rng)
        x = [UV.bf16(rng, 60, 190, 0.1) for _ in range(16)]
        y = [UV.bf16(rng, 60, 190, 0.1) for _ in range(16)]
        assert U.k16(acc, x, y, SM120) == HOPPER_BF16_M16N8K16.step(acc, x, y)
    for _ in range(1500):
        acc = TU.acc_word(rng) & 0x87FFFFFF if rng.random() < 0.5 else TU.acc_word(rng)
        a, b = TU.e4m3_codes(rng, 32), TU.e4m3_codes(rng, 32)
        assert U.atom(acc, a, b, SM120) == SM120_ATOM.step(acc, a, b)
        assert U.atom(acc, a, b, ada) == ADA_E4M3_M16N8K32.step(acc, a, b)


def test_chain_runs_on_the_devices_atom():
    """C̃ on sm120 is the pin's chain on sm120 and differs from the H100's on some words (the adders' widths differ)."""
    rng = random.Random("pearl-c-sm120/chain")
    differ = 0
    for k in (32, 128, 160, 512):
        for _ in range(30):
            a, b = TU.e4m3_codes(rng, k), TU.e4m3_codes(rng, k)
            got = C.chain(a, b, SM120)
            assert got == U.chain(a, b, device=SM120)
            differ += got != C.chain(a, b)
    assert differ


def sm120_instance(case: str, shape=(16, 128, 16), device=SM120):
    s = C.PearlC(k_min=32, tile=(16, 16), forming="v1", device=device)
    ctx = CV.ctx_of(case, shape, 3, 9)
    a, b = CV.rows_f32(case, "A", shape[0], shape[1]), CV.rows_f32(case, "B", shape[2], shape[1])
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, a, b, Tile(tuple(range(shape[0])), tuple(range(shape[2]))))
    return s, ctx, noise, ops, s.checked(ctx, ops), a, b


def test_sm120_instance_is_the_pin_on_its_record():
    """Forming, C̃, the peel factors and U on the sm120 record: each A′ code is the cast of its atom over α·x, and C̃ and U
    are `pearl_c_u`'s at sm120; the noise still comes out."""
    s, ctx, noise, ops, checked, a, b = sm120_instance("sm120")
    k = ctx.matmul.shape.k
    fa = C._basis(noise.seed_b, 0, k, C.LINE_NORM_V1)
    fb = C._basis(noise.seed_b, 1, k, C.LINE_NORM_V1)
    for i in range(2):
        x = C.parse_f32_row(a[i], k)
        for t in range(k):
            d = SM120_ATOM.step(C.fp32.mul(ops.a.alpha[i], x[t]), ops.a.ep[i], fa[t])
            assert ops.a.codes[i][t] == P.f32_to_fp8(P.f32_of_bits(d))
    af = [[U.chain(row, [line[p] for line in fb], device=SM120) for p in range(C.R)] for row in ops.a.codes]
    assert ops.pa == [U.pa_row(be, w) for be, w in zip(C.scaled_noise(ops.a, ops.e_a), af)]
    for i, j in ((0, 0), (3, 7), (15, 15)):
        assert checked.c[i][j] == U.chain(ops.a.codes[i], ops.b.codes[j], device=SM120)
        assert checked.u[i][j] == U.u(checked.c[i][j], ops.pa[i], ops.pb[j], device=SM120)
    assert s.admissible(ctx, ops)
    exact = [[sum(P.f32_of_bits(x) * P.f32_of_bits(y) for x, y in zip(C.parse_f32_row(ra, k), C.parse_f32_row(rb, k)))
              for rb in b] for ra in a]
    useful = s.useful(ctx, noise, ops, checked)
    num = sum((g - e) ** 2 for gr, er in zip(useful, exact) for g, e in zip(gr, er))
    assert (num / sum(e * e for er in exact for e in er)) ** 0.5 < 0.08


def test_sm120_lifecycle_and_work_audit():
    from verity_pouw.audit import Epoch, Prover, Sampled, Verifier, Workload

    s, k = C.PearlC(k_min=32, tile=(16, 16), forming="v1", device=SM120), 128
    rows_a, rows_b = CV.rows_f32("life", "A", 32, k), CV.rows_f32("life", "B", 16, k)
    work = Workload((Matmul(3, Shape(32, k, 16), 9),))
    prover = Prover(s, {9: rows_b})
    epoch = Epoch.unverified(work, prover.weight_roots, b"beacon")
    tr = prover.prove(epoch, {3: rows_a})
    assert Verifier(s, epoch).audit(prover, tr, Sampled(2), b"verifier").accepted
    splits = {9: W.Split(k)}
    records = W.declare(s, epoch, {3: rows_a}, {9: rows_b}, splits)
    res = W.audit(s, epoch, prover, tr, splits, records, Q.recorded_before(Q.LATEST), Q.LATEST, 3)
    assert res.audit.accepted and res.audit.profile.scheme == "pearl-c-sm120-v1"


def test_debit_replays_the_devices_atom():
    """sm120's flags are joint skips of its own atom replayed from scratch, and the debit prices a flagged atom-word at
    atom.k + fadd/G."""
    def replay(a, b, skip):
        total = 0
        for g in range(0, len(a) // 32, SM120.g):
            word = 0
            for t in range(g, g + SM120.g):
                if t not in skip:
                    word = SM120_ATOM.step(word, a[32 * t:32 * t + 32], b[32 * t:32 * t + 32])
            total = D._promote(total, word)
        return D._key(total)

    for _, k, a, b, _ in [v for v in TD.vectors() if "spikes" in v[0] and v[1] == 512][:4]:
        flags, out = D.word_flags(a, b, SM120), replay(a, b, set())
        for g in range(0, k // 32, SM120.g):
            keeps = set()
            for mask in range(1, 1 << SM120.g):
                skip = {g + j for j in range(SM120.g) if mask >> j & 1}
                if replay(a, b, skip) == out:
                    keeps |= skip
            assert {t for t in range(g, g + SM120.g) if flags[t]} == keeps
    assert D.atom_units(H100) == D.ATOM_UNITS == 40 and D.atom_units(SM120) == 34
    assert D.debit_units(3, 2, device=SM120) == D.debit_units(3, 2, C.SM120_PRICES) == 3 * 34 + 2 * 40
    other = dataclasses.replace(SM120, g=8, prices=dataclasses.replace(SM120.prices, fadd=16, forming_credited=60))
    assert D.atom_units(other) == 34 and D.debit_units(3, 2, device=other) == 3 * 34 + 2 * 60


@pytest.mark.parametrize("m,k,n", [(1, 128, 128), (128, 3584, 4608), (2048, 18944, 3584)])
def test_accounting_reads_the_record(m, k, n):
    """sm120's W_ref, credit and weight side are the H100's atom's at SM120_PRICES (the same G = 4 window), below the
    H100's own; another record's G and prices move each term by its own price."""
    sh = Shape(m, k, n)
    h, s, hs = get("pearl-c-h100-v1"), get("pearl-c-sm120-v1"), C.PearlC(forming="v1", prices=C.SM120_PRICES)
    assert s.prices == C.SM120_PRICES and hs.name == "pearl-c-h100-v1@sm_120"
    assert (s.wref(sh), s.credit_of(sh), s.job_cost(n, k)) == (hs.wref(sh), hs.credit_of(sh), hs.job_cost(n, k))
    assert s.credit_of(sh) < h.credit_of(sh)
    pr = Prices("test", fadd=16, bf16_mac=3, forming_credited=60, forming_a_only=5)
    o = C.PearlC(forming="v1", device=dataclasses.replace(SM120, g=2, prices=pr, standins=()))
    chain = m * n * (k + 16 * (k // 64))
    assert o.wref(sh) == chain + m * k * 32 + 3 * 64 * m * n + 65 * m * k
    assert o.credit_of(sh) == o.wref(sh) - 5 * m * k
    assert o.job_cost(n, k) == 65 * n * k + n * 32 * (k + 16 * (k // 64)) + 32 * 32 * (k + 16 * (k // 64)) + 16 * n * 32 * 32
    assert Fraction(o.wref(sh)) != Fraction(h.wref(sh))


def test_unpromoted_chain_is_one_accumulator():
    """G None: every atom chains one accumulator from +0 and the word is the last atom's (no promotion add); it
    differs from G = 4's promoted chain on some words, and `pearl_c_u` restates it."""
    rng = random.Random("pearl-c-sm120/unpromoted")
    differ = 0
    for k in (32, 96, 256, 512):
        for _ in range(30):
            a, b = TU.e4m3_codes(rng, k), TU.e4m3_codes(rng, k)
            word = 0
            for s in range(0, k, 32):
                word = SM120_ATOM.step(word, a[s:s + 32], b[s:s + 32])
            trace = {}
            assert C.chain(a, b, SM120_UNPROMOTED) == U.chain(a, b, trace, SM120_UNPROMOTED) == word
            assert "totals" not in trace and len(trace["atoms"]) == k // 32
            differ += word != C.chain(a, b, SM120)
    assert differ


def test_unpromoted_instance_and_accounting():
    """The unpromoted record forms, chains and peels a k = 96 instance (32 | k), counts no promotion adds, registers no
    scheme and has no debit rule."""
    s, ctx, noise, ops, checked, a, b = sm120_instance("unpromoted", (16, 96, 16), SM120_UNPROMOTED)
    for i, j in ((0, 0), (5, 11), (15, 15)):
        assert checked.c[i][j] == U.chain(ops.a.codes[i], ops.b.codes[j], device=SM120_UNPROMOTED)
        assert checked.u[i][j] == U.u(checked.c[i][j], ops.pa[i], ops.pb[j], device=SM120_UNPROMOTED)
    assert s.admissible(ctx, ops) and s.name == "pearl-c-sm120-unpromoted-v1"
    assert "no promotion" in s.certificate.hypotheses[0]
    sh = Shape(128, 4096, 256)
    promoted = C.PearlC(forming="v1", device=SM120)
    assert promoted.wref(sh) - s.wref(sh) == 128 * 256 * C.SM120_PRICES.fadd * (4096 // 128)
    with pytest.raises(KeyError):
        get("pearl-c-sm120-unpromoted-v1")
    with pytest.raises(NotImplementedError):
        D.word_flags(ops.a.codes[0], ops.b.codes[0], SM120_UNPROMOTED)
    with pytest.raises(NotImplementedError):
        D.atom_units(SM120_UNPROMOTED)
