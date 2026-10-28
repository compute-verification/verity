"""Deterministic finite indexed domains: the position sets a commitment covers.

Promoted from veritor ``core/indexed.py`` (eb865ce2): the protocol, the explicit and range domains and the iteration
helpers.  The ``identity_digest`` tags are wire constants (they enter every ``CommitmentDomain.domain_id`` but a
:class:`RangeU64Domain`'s, which frames its start and count fixed-width instead).  The session-specific interval domains
were left behind.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from sys import maxsize
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

from verity.errors import InvalidArtifact

from .identity import Digest, JSONValue, identity_digest, identity_digest_sha512

Position = int


@runtime_checkable
class IndexedDomain[T](Protocol):
    """A finite domain with stable zero-based rank and unrank operations."""

    @property
    def count(self) -> int: ...

    @property
    def identity_digest(self) -> Digest: ...

    def contains(self, item: T) -> bool: ...

    def rank(self, item: T) -> int: ...

    def unrank(self, rank: int) -> T: ...


@runtime_checkable
class FiniteIndexedDomain[T](IndexedDomain[T], Protocol):
    """Descriptive spelling for any finite indexed domain."""


@runtime_checkable
class LazyIndexedDomain[T](FiniteIndexedDomain[T], Protocol):
    """Contract for domains that may compute members without materializing all."""


def _checked_rank(rank: object, count: int) -> int:
    if type(rank) is not int:
        raise TypeError("rank must be an integer")
    if rank < 0 or rank >= count:
        raise IndexError(f"rank {rank} is outside domain of size {count}")
    return rank


@dataclass(frozen=True, slots=True, init=False)
class ExplicitIndexedDomain[T]:
    """An immutable ordered domain backed by an explicit tuple."""

    _items: tuple[T, ...]
    _rank_by_item: Mapping[T, int] = field(repr=False, compare=False, hash=False)
    identity_digest: Digest

    def __init__(self, items: Iterable[T]) -> None:
        values = tuple(items)
        ranks: dict[T, int] = {}
        for index, item in enumerate(values):
            try:
                if item in ranks:
                    raise InvalidArtifact(
                        f"explicit indexed domain repeats item {item!r}"
                    )
                ranks[item] = index
            except TypeError as error:
                raise InvalidArtifact(
                    "explicit indexed domain items must be hashable"
                ) from error
        digest = identity_digest(
            "veritor/indexed-domain/explicit/v1",
            {"items": cast(JSONValue, list(values))},
        )
        object.__setattr__(self, "_items", values)
        object.__setattr__(self, "_rank_by_item", MappingProxyType(ranks))
        object.__setattr__(self, "identity_digest", digest)

    @property
    def items(self) -> tuple[T, ...]:
        return self._items

    def identity_sha512(self) -> bytes:
        """The same identity manifest under :func:`identity_digest_sha512`: what a SHA-512 frame's domain id absorbs."""
        return identity_digest_sha512("veritor/indexed-domain/explicit/v1", {"items": cast(JSONValue, list(self._items))})

    @property
    def count(self) -> int:
        return len(self._items)

    @property
    def digest(self) -> Digest:
        return self.identity_digest

    def contains(self, item: T) -> bool:
        try:
            return item in self._rank_by_item
        except TypeError:
            return False

    def __contains__(self, item: object) -> bool:
        return self.contains(item)  # type: ignore[arg-type]

    def rank(self, item: T) -> int:
        try:
            return self._rank_by_item[item]
        except (KeyError, TypeError) as error:
            raise KeyError(item) from error

    def unrank(self, rank: int) -> T:
        return self._items[_checked_rank(rank, self.count)]

    def at_rank(self, rank: int) -> T:
        return self.unrank(rank)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __len__(self) -> int:
        return self.count


@dataclass(frozen=True, slots=True, init=False)
class RangeIndexedDomain:
    """A lazy nonnegative integer domain equivalent to ``range``."""

    start: int
    stop: int
    step: int
    count: int
    identity_digest: Digest

    def __init__(self, start: int, stop: int | None = None, step: int = 1) -> None:
        if stop is None:
            stop = start
            start = 0
        for value, name in ((start, "start"), (stop, "stop"), (step, "step")):
            if type(value) is not int:
                raise InvalidArtifact(f"range domain {name} must be an integer")
        if start < 0 or stop < 0:
            raise InvalidArtifact("range domain bounds must be nonnegative")
        if stop < start:
            raise InvalidArtifact("range domain stop must not precede start")
        if step <= 0:
            raise InvalidArtifact("range domain step must be positive")
        count = 0 if start == stop else (stop - start + step - 1) // step
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "count", count)
        object.__setattr__(
            self,
            "identity_digest",
            identity_digest(
                "veritor/indexed-domain/range/v1",
                {"start": start, "step": step, "stop": stop},
            ),
        )

    @property
    def digest(self) -> Digest:
        return self.identity_digest

    def identity_sha512(self) -> bytes:
        """The same identity manifest under :func:`identity_digest_sha512`: what a SHA-512 frame's domain id absorbs."""
        return identity_digest_sha512("veritor/indexed-domain/range/v1", {"start": self.start, "step": self.step, "stop": self.stop})

    def contains(self, item: int) -> bool:
        return (
            type(item) is int
            and self.start <= item < self.stop
            and (item - self.start) % self.step == 0
        )

    def __contains__(self, item: object) -> bool:
        return self.contains(item)  # type: ignore[arg-type]

    def rank(self, item: int) -> int:
        if not self.contains(item):
            raise KeyError(item)
        return (item - self.start) // self.step

    def unrank(self, rank: int) -> Position:
        checked = _checked_rank(rank, self.count)
        return Position(self.start + checked * self.step)

    def at_rank(self, rank: int) -> Position:
        return self.unrank(rank)

    def __iter__(self) -> Iterator[Position]:
        for rank in range(self.count):
            yield self.unrank(rank)

    def __len__(self) -> int:
        if self.count > maxsize:
            raise OverflowError("domain is too large for len(); use .count")
        return self.count


#: A ``range-u64`` domain's bound: ``start + count`` is below it, so every position and the count are u64 words.
U64_BOUND = 1 << 64


@dataclass(frozen=True, slots=True, init=False)
class RangeU64Domain:
    """The positions ``start, start + 1, ..., start + count - 1``, rank ``position - start``: the ``range-u64`` kind.

    A ``frame-v3-sha512`` domain over it frames ``start`` and ``count`` as u64be words in its domain id
    (``merkle.range_u64_domain_message``, ``frame_v3/PROTOCOL.md`` §6), so it has no positions identity:
    ``identity_digest`` refuses, and so does every frame but SHA-512's.
    """

    start: int
    count: int

    def __init__(self, start: int, count: int) -> None:
        for value, name in ((start, "start"), (count, "count")):
            if type(value) is not int:
                raise InvalidArtifact(f"range-u64 domain {name} must be an integer")
            if value < 0:
                raise InvalidArtifact(f"range-u64 domain {name} must be nonnegative")
        if start + count >= U64_BOUND:
            raise InvalidArtifact("range-u64 domain needs start + count < 2^64")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "count", count)

    @property
    def identity_digest(self) -> Digest:
        raise InvalidArtifact("a range-u64 domain has no positions identity: its domain id frames start and count")

    def contains(self, item: int) -> bool:
        return type(item) is int and self.start <= item < self.start + self.count

    def __contains__(self, item: object) -> bool:
        return self.contains(item)  # type: ignore[arg-type]

    def rank(self, item: int) -> int:
        if not self.contains(item):
            raise KeyError(item)
        return item - self.start

    def unrank(self, rank: int) -> Position:
        return Position(self.start + _checked_rank(rank, self.count))

    def at_rank(self, rank: int) -> Position:
        return self.unrank(rank)

    def __iter__(self) -> Iterator[Position]:
        return iter(range(self.start, self.start + self.count))

    def __len__(self) -> int:
        if self.count > maxsize:
            raise OverflowError("domain is too large for len(); use .count")
        return self.count


def iter_domain[T](domain: IndexedDomain[T]) -> Iterator[T]:
    """Iterate a domain in canonical rank order, one ``unrank`` per member (the reference enumeration)."""

    for rank in range(domain.count):
        yield domain.unrank(rank)


def iter_members[T](domain: IndexedDomain[T]) -> Iterator[T]:
    """Iterate a domain in rank order by its own ``__iter__`` when it has one, else by :func:`iter_domain`.

    Every domain's iteration order is its rank order (``tests/veritor``
    holds each to :func:`iter_domain`); the lazy domains walk their members in
    linear time where ``unrank`` costs a descent per member, so hot loops
    (committing an interior) go through this.
    """

    if isinstance(domain, Iterable):
        return iter(domain)
    return iter_domain(domain)


def domains_equal[T](
    left: IndexedDomain[T],
    right: IndexedDomain[T],
) -> bool:
    """Compare represented members and order, independent of implementation."""

    if left.count != right.count:
        return False
    if left.identity_digest == right.identity_digest:
        return True
    return all(left.unrank(rank) == right.unrank(rank) for rank in range(left.count))



ExplicitDomain = ExplicitIndexedDomain
RangeDomain = RangeIndexedDomain
