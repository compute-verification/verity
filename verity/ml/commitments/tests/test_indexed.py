import pytest

from verity.ml.commitments.identity import identity_digest
from verity.ml.commitments.indexed import (ExplicitIndexedDomain, IndexedDomain, RangeIndexedDomain, RangeU64Domain, domains_equal,
                                           iter_domain, iter_members)
from verity.errors import InvalidArtifact


def test_explicit_domain_has_stable_rank_unrank_and_identity():
    domain = ExplicitIndexedDomain((7, 2, 11))

    assert domain.count == 3
    assert tuple(domain) == (7, 2, 11)
    assert domain.rank(2) == 1
    assert domain.unrank(2) == 11
    assert domain.at_rank(0) == 7
    assert domain.contains(7)
    assert 11 in domain
    assert 3 not in domain
    assert domain.identity_digest == ExplicitIndexedDomain((7, 2, 11)).digest
    assert domain.identity_digest != ExplicitIndexedDomain((2, 7, 11)).digest
    assert isinstance(domain, IndexedDomain)


def test_explicit_domain_rejects_duplicates_and_unhashable_items():
    with pytest.raises(InvalidArtifact, match="repeats"):
        ExplicitIndexedDomain((1, 1))
    with pytest.raises(InvalidArtifact, match="hashable"):
        ExplicitIndexedDomain(([1],))


def test_explicit_domain_reports_absent_items_and_bad_ranks_precisely():
    domain = ExplicitIndexedDomain(("a", "b"))

    with pytest.raises(KeyError):
        domain.rank("missing")
    with pytest.raises(IndexError):
        domain.unrank(-1)
    with pytest.raises(IndexError):
        domain.unrank(2)
    with pytest.raises(TypeError):
        domain.unrank(True)


def test_range_domain_is_lazy_and_supports_steps():
    domain = RangeIndexedDomain(10, 21, 3)

    assert domain.count == 4
    assert tuple(iter_domain(domain)) == (10, 13, 16, 19)
    assert domain.rank(16) == 2
    assert domain.unrank(3) == 19
    assert domain.contains(13)
    assert not domain.contains(14)
    assert not domain.contains(True)
    with pytest.raises(KeyError):
        domain.rank(14)


def test_range_domain_supports_empty_and_huge_counts_without_materialization():
    empty = RangeIndexedDomain(0)
    huge = RangeIndexedDomain(0, 10**30)

    assert empty.count == 0
    assert tuple(empty) == ()
    assert huge.count == 10**30
    assert huge.unrank(10**30 - 1) == 10**30 - 1
    with pytest.raises(OverflowError):
        len(huge)


@pytest.mark.parametrize(
    "args",
    [
        (True,),
        (-1,),
        (2, 1),
        (0, 2, 0),
        (0, 2, -1),
    ],
)
def test_range_domain_rejects_invalid_bounds(args):
    with pytest.raises(InvalidArtifact):
        RangeIndexedDomain(*args)


def test_range_u64_domain_ranks_from_its_start_with_step_one():
    domain = RangeU64Domain(2**32 - 1, 3)

    assert isinstance(domain, IndexedDomain)
    assert domain.count == len(domain) == 3
    assert tuple(domain) == tuple(iter_domain(domain)) == tuple(iter_members(domain)) == (2**32 - 1, 2**32, 2**32 + 1)
    assert [domain.rank(p) for p in domain] == [0, 1, 2]
    assert domain.unrank(2) == domain.at_rank(2) == 2**32 + 1
    assert 2**32 in domain and 2**32 - 2 not in domain and 2**32 + 2 not in domain and not domain.contains(True)
    with pytest.raises(KeyError):
        domain.rank(2**32 + 2)
    with pytest.raises(IndexError):
        domain.unrank(3)
    assert tuple(RangeU64Domain(5, 0)) == ()


def test_range_u64_domain_reaches_2_64_without_materialization():
    whole, top = RangeU64Domain(0, 2**64 - 1), RangeU64Domain(2**64 - 2, 1)

    assert whole.unrank(2**64 - 2) == 2**64 - 2 and whole.rank(2**64 - 2) == 2**64 - 2
    assert tuple(top) == (2**64 - 2,) and top.rank(2**64 - 2) == 0
    with pytest.raises(OverflowError):
        len(whole)


@pytest.mark.parametrize("args", [(True, 1), (0, True), (-1, 1), (0, -1), (0, 2**64), (1, 2**64 - 1), (2**64 - 1, 1), (2**64, 0), (0.0, 1)])
def test_range_u64_domain_refuses_what_is_not_below_2_64(args):
    with pytest.raises(InvalidArtifact):
        RangeU64Domain(*args)


def test_range_u64_domain_has_no_positions_identity():
    with pytest.raises(InvalidArtifact, match="no positions identity"):
        _ = RangeU64Domain(0, 4).identity_digest
    assert not hasattr(RangeU64Domain(0, 4), "identity_sha512")


def test_domains_equal_compares_semantics_across_implementations():
    explicit = ExplicitIndexedDomain((0, 1, 2, 3))
    ranged = RangeIndexedDomain(4)

    assert explicit.identity_digest != ranged.identity_digest
    assert domains_equal(explicit, ranged)
    assert not domains_equal(explicit, RangeIndexedDomain(1, 5))


def test_lazy_domain_contract_requires_rank_unrank_without_materialization():
    class EvenPositions:
        count = 10**20
        identity_digest = identity_digest(
            "tests/even-positions",
            {"count": count},
        )

        @staticmethod
        def contains(item):
            return type(item) is int and 0 <= item < 2 * 10**20 and item % 2 == 0

        @staticmethod
        def rank(item):
            if not EvenPositions.contains(item):
                raise KeyError(item)
            return item // 2

        @staticmethod
        def unrank(rank):
            if type(rank) is not int or not 0 <= rank < 10**20:
                raise IndexError(rank)
            return 2 * rank

    domain = EvenPositions()

    assert isinstance(domain, IndexedDomain)
    assert domain.count == 10**20
    assert domain.unrank(10**20 - 1) == 2 * (10**20 - 1)
