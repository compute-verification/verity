"""The chain test's field term (``chain_term``; red-team-bligero-real-k finding A1): the red team's per-shape table from the
real-K statements' linked rows, the booked term (coins of maximum probability 3 / 2^32), and -- with torch -- the compiled
systems' own count, ``protocol.soundness`` booking it and ``config_for`` sizing ``t`` with it."""
from __future__ import annotations

import math

import pytest

from . import chain_term as CT

LOG2_P = math.log2(2013265921)

# red-team-bligero-real-k A1: (statements, linked rows, degree, (degree / p)^6 in log2), and the booked (3 degree / 2^32)^6
SHAPES = [
    (["bf16-ampere-x4-k2048+sha256", "fp8-ada-x4-k8192+sha256", "bf16-hopper-x4-k8192+sha256"], 69, 23, -158.30, -155.35),
    (["fp8-ada-x4-k2048+blake3-xob", "fp8-hopper-x4-k2048+blake3-xob", "bf16-ampere-x4+blake3-xob"], 133, 45, -152.49, -149.54),
    (["bf16-ampere-x4-k2048+blake3-xob", "bf16-hopper-x4-k2048+blake3-xob"], 165, 55, -150.75, -147.80),
    (["fp8-ada-x4-k8192+blake3-xob", "fp8-hopper-x4-k8192+blake3-xob"], 293, 98, -145.75, -142.80),
    (["bf16-ampere-x4-k8192+blake3-xob", "bf16-hopper-x4-k8192+blake3-xob"], 549, 183, -140.35, -137.40),
]


@pytest.mark.parametrize("names,links,deg,uniform,booked", SHAPES)
def test_red_team_per_shape_table(names, links, deg, uniform, booked) -> None:
    for name in names:
        e = CT.hashed_extras(name)
        assert e == CT.extra_constraints(links) == 2 * (links - 3), name
        assert CT.degree(e) == deg, name
        assert abs(6 * (math.log2(deg) - LOG2_P) - uniform) < 0.005, name
        assert abs(math.log2(CT.field_term(e, 6)) - booked) < 0.005, name
        # the booked term is the table's with the coins' maximum probability 3 / 2^32 for 1 / p: 6 x 0.49 bits
        assert abs(math.log2(CT.field_term(e, 6)) - 6 * (math.log2(deg) - LOG2_P) - 6 * math.log2(3 * 2013265921 / 2**32)) < 1e-9


def test_degree_and_no_term_without_extras() -> None:
    assert CT.degree(0) == 1 and CT.field_term(0, 6) == 0 and CT.extra_constraints(3) == 0
    assert [CT.degree(e) for e in (1, 6, 7, 12, 13)] == [2, 2, 3, 3, 4]      # the last extra's monomial: (e - 1) div 6 + 2
    assert CT.extra_constraints(19, n_fp=16) == 48
    with pytest.raises(ValueError):
        CT.extra_constraints(2)


def test_red_team_rebound_of_the_tightest_cell() -> None:
    """art:b1d710da (A100 BF16 K = 8192 xob): 2^-128.104 over 60 sub-batches is 2^-128.086 with the table's term and
    2^-127.97 with the booked one -- above 2^-128."""
    pp, n = -134.01044941441342, 60
    e = CT.hashed_extras("bf16-ampere-x4-k8192+blake3-xob")

    def total(term: float) -> float:
        return math.log2(2.0 ** pp + 2.0 ** term) + math.log2(n)

    assert abs(total(6 * (math.log2(CT.degree(e)) - LOG2_P)) - (-128.086)) < 0.001
    assert -128.0 < total(math.log2(CT.field_term(e, 6))) < -127.96


def test_compiled_systems_count_the_same_extras() -> None:
    pytest.importorskip("torch")
    from . import chain, hashchain
    from .relations import RELATIONS
    for name in ("fp8-ada-x4-k2048+sha256", "fp8-ada-x4-k2048+blake3-xob", "bf16-ampere-x4-k2048+blake3-xob"):
        base, _, leaf = name.partition("+")
        hr = hashchain.compose(RELATIONS[base], leaf)
        assert hr.census["linked_rows"] == len(hr.sys.chain["c"])
        assert chain.extra_constraints(hr.sys.chain) == CT.hashed_extras(name), name


def test_soundness_books_the_term_and_config_sizes_t_with_it() -> None:
    pytest.importorskip("torch")
    from . import protocol as pm
    e = CT.hashed_extras("bf16-ampere-x4-k8192+blake3-xob")
    target, n_proofs, l = -128.0, 60, 4096
    old = pm.config_for(l, target, 2, n_proofs, zk=True)
    new = pm.config_for(l, target, 2, n_proofs, zk=True, chain_extras=e)
    assert old.t == 203 and new.t > old.t and new.D == old.D == 6
    s_old = pm.soundness(old, 10, 10, n_proofs, chain_extras=e)
    s_new = pm.soundness(new, 10, 10, n_proofs, chain_extras=e)
    assert abs(s_old["terms_log2"]["chain_field"] - math.log2(CT.field_term(e, 6))) < 1e-9
    assert s_old["chain_extras"] == e and s_old["chain_degree"] == 183
    assert s_old["total_log2"] > target >= s_new["total_log2"]            # the old t misses 2^-128 once the term is booked
    assert "chain_field" not in pm.soundness(old, 10, 10, n_proofs)["terms_log2"]   # three linked rows: unchanged
    # Fiat-Shamir: x 2^60, the term alone exceeds the target at D = 7 whatever t is -- config_for raises D instead
    fs0 = pm.config_for(l, target, 2, n_proofs, zk=True, mode=pm.MODE_FIAT_SHAMIR)
    fs = pm.config_for(l, target, 2, n_proofs, zk=True, mode=pm.MODE_FIAT_SHAMIR, chain_extras=e)
    assert fs0.D == 7 and pm.soundness(fs0, 10, 10, n_proofs, chain_extras=e)["total_log2"] > target
    assert fs.D > 7 and pm.soundness(fs, 10, 10, n_proofs, chain_extras=e)["total_log2"] <= target
