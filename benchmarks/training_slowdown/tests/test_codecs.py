import pytest
import torch

from training_slowdown.codecs import (Compose, Identity, LowRank, Plan, QuantDelta, ScaledQuant, SeedReplay, SparseMask,
                            Unencodable, derive_seed, from_spec)
from training_slowdown.codecs.wire import pack_bits, unpack_bits


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(-1).view(torch.uint8), b.view(-1).view(torch.uint8))


@pytest.fixture
def base():
    return torch.randn(8, 12, generator=torch.Generator().manual_seed(1))


def test_bits_roundtrip():
    idx = torch.randint(0, 8, (101,), generator=torch.Generator().manual_seed(0))
    assert torch.equal(unpack_bits(pack_bits(idx, 3), 3, 101), idx)
    assert pack_bits(idx, 3).numel() == (101 * 3 + 7) // 8


def test_identity_is_the_full_update(base):
    target = base + 1
    c = Identity()
    enc = c.encode(target, base)
    assert same(c.decode(enc, base), target)
    assert c.size_bits(enc) >= 32 * base.numel()
    assert c.cost_macs(enc, base) == 0


def test_sparse_is_lossless_and_smaller_when_sparse(base):
    target = base.clone()
    target.view(-1)[[3, 50, 77]] += 0.5
    c = SparseMask()
    enc = c.encode(target, base)
    assert same(c.decode(enc, base), target)
    assert c.size_bits(enc) < Identity().size_bits(Identity().encode(target, base))
    dense = base + 1  # every entry differs: the bitmask form
    assert same(c.decode(c.encode(dense, base), base), dense)


def test_lowrank_factors_rebuild_bit_for_bit(base):
    g = torch.Generator().manual_seed(2)
    a, b = torch.randn(2, 12, generator=g), torch.randn(8, 2, generator=g)
    c = LowRank(2, scale=0.5)
    target = LowRank.apply(base, a, b, 0.5)
    enc = LowRank.encode_factors(a, b)
    assert same(c.decode(enc, base), target)
    assert c.cost_macs(enc, base) == 2 * 96 + 2 * 96
    approx = c.decode(c.encode(target, base), base)
    assert torch.allclose(approx, target, atol=1e-5)
    with pytest.raises(Unencodable):
        c.encode(torch.ones(5), torch.zeros(5))


def test_quant_delta_rebuilds_a_quantized_parameterization(base):
    c = QuantDelta(2)
    codebook = torch.tensor([-0.3, -0.1, 0.1, 0.3])
    index = torch.arange(base.numel()) % 4
    target = QuantDelta.apply(base, codebook, index)
    assert same(c.decode(c.encode_index(codebook, index), base), target)
    enc = c.encode(target, base)  # from a target: right to within rounding, not bit for bit
    assert torch.allclose(c.decode(enc, base), target, atol=1e-6)
    assert c.size_bits(enc) < 2 * base.numel() + 8 * 200
    assert c.cost_macs(enc, base) == base.numel()


def test_scaled_quant_rounds_each_entry_to_its_groups_nearest_level():
    """Each entry's delta is coded to the nearest of the levels across its group's largest magnitude, so it comes
    back within half a level, and f16's rounding of the codebook and the scale; a group that didn't move comes back
    as the base's bits; the codes take at most `bits` bits an entry beside the codebook and the scales."""
    gen = torch.Generator().manual_seed(2)
    base = torch.randn(6, 128, generator=gen)
    delta = 1e-3 * torch.randn(6, 128, generator=gen) * torch.tensor([1.0, 4.0, 0.0, 0.25, 16.0, 1.0])[:, None]
    target = base + delta
    moved = (target - base).view(6, 128)
    top = moved.abs().amax(1, keepdim=True)
    for bits in (8, 4, 2):
        c = ScaledQuant(bits)
        enc = c.encode(target, base)
        out = c.decode(enc, base)
        assert out.dtype == base.dtype and same(out[2], base[2]) and not same(out[0], base[0])
        assert ((out - base) - moved).abs().le(top * (0.5 / (2 ** (bits - 1) - 1) + 2 ** -10) + 3e-7).all(), bits
        assert c.size_bits(enc) <= bits * base.numel() + 16 * (2 ** bits + 6) + 8 * 200
        assert c.cost_macs(enc, base) == 2 * base.numel()
    out = ScaledQuant(8).decode(ScaledQuant(8).encode(target, base), base)
    assert (out - target).norm() / moved.norm() < 0.01
    assert from_spec(c.spec()).spec() == c.spec() == {"kind": "scaledquant", "bits": 2, "group": 128}


def test_scaled_quant_groups_divide_the_tensor_and_keep_the_bases_dtype():
    base = torch.randn(5, 12, generator=torch.Generator().manual_seed(3)).bfloat16()
    codebook = ((torch.arange(16) - 8) / 7 * 0.5).half()
    scales = torch.tensor([0.25, 0.5, 1.0, 0.0, 2.0]).half()
    target = ScaledQuant.apply(base, codebook, scales, (torch.arange(60) * 7) % 16)
    assert target.dtype == torch.bfloat16 and same(target[3], base[3]) and not same(target[4], base[4])
    c = ScaledQuant(4, group=12)
    out = c.decode(c.encode(target, base), base)
    assert out.dtype == torch.bfloat16 and same(out[3], base[3])
    assert (out.float() - target.float()).abs().max() <= 0.5 * (2.0 * 0.5 / 7) + 2 ** -6
    odd = torch.zeros(7, 9)   # 63 entries: groups of one, so each entry is its own scale times the top level
    out = ScaledQuant(8).decode(ScaledQuant(8).encode(odd + 0.5, odd), odd)
    assert (out - 0.5).abs().max() <= 0.5 * 2 ** -10
    with pytest.raises(ValueError, match="for a weight of 60"):
        c.decode(ScaledQuant(4).encode(odd + 0.5, odd), base)
    with pytest.raises(Unencodable):
        c.encode(torch.zeros(0), torch.zeros(0))
    for bits in (1, 9):
        with pytest.raises(ValueError, match="2 to 8 bits"):
            ScaledQuant(bits)


def test_seed_replay_rebuilds_a_mezo_log(base):
    seeds, coeffs = [derive_seed(s, "w") for s in range(3)], [1e-3, -2e-3, 5e-4]
    w = base.float().clone()
    for s, k in zip(seeds, coeffs):
        SeedReplay.apply(w, s, k)
    c = SeedReplay()
    enc = SeedReplay.encode_log(seeds, coeffs)
    assert same(c.decode(enc, base), w)
    assert c.cost_macs(enc, base) == 3 * base.numel()
    assert c.size_bits(enc) < 8 * 120
    with pytest.raises(Unencodable):
        c.encode(w, base)


def test_compose_lowrank_then_sparse_is_lossless(base):
    target = base + 0.01 * torch.randn(8, 1) @ torch.randn(1, 12)
    target.view(-1)[5] = 9.0
    c = Compose([LowRank(1), SparseMask()])
    enc = c.encode(target, base)
    assert same(c.decode(enc, base), target)
    assert from_spec(c.spec()).spec() == c.spec()


def test_sparse_values_compose_with_quant(base):
    c = from_spec({"kind": "sparse", "values": {"kind": "quant", "bits": 1}})
    target = base.clone()
    target.view(-1)[[1, 2]] = base.view(-1)[[1, 2]] + torch.tensor([-0.5, 0.5])
    out = c.decode(c.encode(target, base), base)
    assert torch.allclose(out, target, atol=1e-6)
    keep = torch.ones(base.numel(), dtype=torch.bool)
    keep[[1, 2]] = False
    assert same(out.view(-1)[keep], base.view(-1)[keep])


def test_plan_routes_keys_and_round_trips(base):
    plan = Plan([("*.bias", Identity()), ("w*", SparseMask())])
    targets = {"w1": base + (torch.arange(96).reshape(8, 12) == 7), "x.bias": torch.ones(3)}
    bases = {"w1": base, "x.bias": torch.zeros(3)}
    enc = plan.encode(targets, bases)
    out = plan.decode(enc, bases)
    assert all(same(out[k], targets[k].to(bases[k].dtype)) for k in targets)
    assert sorted(plan.keys(enc)) == ["w1", "x.bias"]
    assert Plan.from_spec(plan.spec()).spec() == plan.spec()
    with pytest.raises(Unencodable):
        plan.encode({"emb": base}, {"emb": base})
