import pytest
import torch

from te_core.codecs import (Compose, Identity, LowRank, Plan, QuantDelta, SeedReplay, SparseMask, Unencodable,
                            derive_seed, from_spec)
from te_core.codecs.wire import pack_bits, unpack_bits


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
