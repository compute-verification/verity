"""P3's primitives: a key derivation `H` and a tweakable permutation `P_T` of labels (`docs/p3-instantiation.md` §1).

Integers and byte strings are little-endian; a label of `m` bits is `m / 8` bytes.

**Default backend (`reference`).**

~~~text
Feistel10 = (L, R) -> (R, L xor F_i(R)) for i = 1..10;  inverse (L, R) -> (R xor F_i(L), L) for i = 10..1
            (L is the first half of the bytes, R the second)
Pi2       = Feistel10 on 2m + 64 B,  F_i(x)    = SHAKE256(0x48 || u8(i) || x)[0 : half]
P_T       = Feistel10 on m,          F_i(T, x) = SHAKE256(0x50 || u8(i) || T || x)[0 : half]
            T = tag_s || u8(layer) || u16(v)                                                35 B
H(tag_s, layer, v, parents), the overwrite chain:
  h_0 = "pous-p3-v1-H" || tag_s || u8(layer) || u16(v) || u8(npar), zero-filled to m + 64 B    (no call)
  for B in (parents newest first, then B_pad = 0x01 || 0..0 || 0x80):  (r, h) <- Pi2(B || h)
      r = the first m B, dropped; h = the last m + 64 B, chained
  key = r of the B_pad call
~~~

A node with `j` parents costs `j + 1` calls of `Π₂`.  A chaining value `h_j` finishes its own row's key but reveals no
parent, since walking back needs the dropped `r_j`; anything that reveals a parent and also finishes the row is a whole
call state of `2m + 512` bits, more than the two labels it yields.

**Broken modes, kept to replay their attacks** (`tests/test_pous_two_for_one.py`): `XorSponge`, the sponge of the
15:50Z instantiation (`s_j = Π(s_{j−1} ⊕ (B_j ‖ 0))`, rate `m`, capacity 512, a header block first), from whose states
`Π⁻¹` recovers an absorbed parent (`lean/submissions/sponge-dense`, `dense_two_labels_from_one_state`); and
`MiyaguchiPreneel` (`s_j = Π(x_j) ⊕ x_j`, `x_j = s_{j−1} ⊕ (B_j ‖ 0)`), whose pre-image `x_j` is a checkpoint that is
also a masked parent.  Neither is a default anywhere.

**Heuristic backend (`heuristic_arx`)**, fast on GPUs and not analysed: the overwrite chain over an ARX `Π₂`, and `P_T`
the 5-round iterated Even–Mansour cipher `k ⊕ π₅(k ⊕ … π₁(x ⊕ k))` over five butterflies, `k = SHAKE256(0x4B || T)[0 : m]`.
A butterfly stage `s` pairs word `i` with `j = i + 2^b`, `b = s mod ⌈log₂ N⌉`, for every `i` with bit `b` clear and
`j < N`, and maps `(a, c)` to `a' = rotl((a + c) mod 2^32, ra_s) ⊕ k_{s,i}`, `c' = rotl(c, rc_s) ⊕ a'`, so both words
are rotated (the §2c butterfly never rotated word 0: `docs/efficient-crypto`, tc-mixer).  It runs twice the stages
bit-level diffusion needs (`butterfly_stages`), with rotations and constants from SHA-256 in counter mode over its name.

**Ideal backend (`ideal`)**: the overwrite chain over a lazily sampled ideal `Π₂`, and an ideal tweakable `P`.
`RandomOracleKey` is a monolithic `H` (Lean's `p3Model'`), for tests against the Lean model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

from verity.ml.randomness import Key, derive

from .labels import mask
from .oracle import HQuery, KeyQuery, PiInvQuery, PiQuery, Query, Tweak
from .params import TAG_BITS, Params

CAPACITY_BITS = 512
HEADER = b"pous-p3-v1-H"
PI_DOMAIN = 0x48
P_DOMAIN = 0x50
EM_KEY_DOMAIN = 0x4B

FEISTEL_ROUNDS = 10
ROUND_PREFIX_BYTES = 2
"""A round function's input before its half: the domain byte and `u8(i)`."""
SHAKE256_RATE = 136
"""SHAKE256's rate in bytes: one Keccak-f[1600] per 136 bytes absorbed or squeezed."""
KECCAK_F_NS = 50
"""The assumed floor of one sequential Keccak-f[1600], 50 ns (no ASIC): the conversion of a deadline into rounds rests
on it, outside Lean."""


def pi2_keccak_f(m: int) -> int:
    """The sequential Keccak-f per call of the Feistel-SHAKE256 `Π₂` (`2m + 512` bits) in a long overwrite chain:
    `FEISTEL_ROUNDS` times the blocks one round function absorbs, the prefix and a half of `m/8 + 32` bytes (the pad
    takes at least one byte).  At `m = 524288` (64 KiB labels) that is 10 × 483 = 4,830.

    The rounds stream: round `i + 1` absorbs `L ⊕ F_i(R)`, and its absorb block `k` needs only round `i`'s squeeze
    block `k`, so a round boundary costs one round's absorb.  A call boundary is a round boundary: the next call's
    right half is this call's last round output.  Counting each round's absorb and then the rest of its squeeze
    (`⌈half / 136⌉ − 1` more) strictly in sequence gives 10 × (483 + 482) = 9,650 at 64 KiB, about twice this.

    This counts a whole call, which is what the ideal-permutation model charges.  It does not bound an adversary at
    this `Π₂`: each round function is a long SHAKE256 squeeze from a 200-byte post-absorb state, so a prover that keeps
    the five odd rounds' states of a block's last call (1000 B, 1.53% of a 64 KiB label) rebuilds the whole label in
    481 sequential Keccak-f, with no call and no chain walk (the red team's state attack)."""
    half = (2 * m + CAPACITY_BITS) // 16
    return FEISTEL_ROUNDS * ((ROUND_PREFIX_BYTES + half) // SHAKE256_RATE + 1)


def pi2_call_ns(m: int) -> int:
    """The fastest whole `Π₂` call on `m`-bit labels: `pi2_keccak_f(m)` Keccak-f of at least `KECCAK_F_NS`.  The
    ideal-permutation model's unit of time; at the reference Feistel-SHAKE256 `Π₂` it is not the adversary's floor,
    since a label is rebuilt from 1.53% of `C` in 481 Keccak-f, about a tenth of this at 64 KiB (`pi2_keccak_f`)."""
    return pi2_keccak_f(m) * KECCAK_F_NS


def key_query(params: Params, tag: int, layer: int, node: int, labels: Mapping[int, int] | Sequence[int]) -> KeyQuery:
    """Node `node`'s key query: Lean `parentsOf L v = fun u => if u ∈ G.par v then some (L u) else none`."""
    return KeyQuery(tag, layer, node, tuple((u, labels[u]) for u in params.parents(node)))


def check_key_query(params: Params, q: KeyQuery) -> None:
    if not 0 <= q.tag < 1 << TAG_BITS or q.layer not in (0, 1) or not 0 <= q.node < params.n:
        raise ValueError("key query outside KQ'")
    last = -1
    for u, label in q.parents:
        if not last < u < params.n or not 0 <= label < 1 << params.m:
            raise ValueError("key query parents must be increasing nodes with m-bit labels")
        last = u


class KeyStepper(Protocol):
    """One key derivation in progress: `query(labels)` builds the next query (reading only the parents `needs`
    names), `feed(answer)` takes its answer, and `key` is set once `done`."""

    done: bool
    key: int | None

    def query(self, labels: Mapping[int, int]) -> Query: ...

    def feed(self, answer: int) -> None: ...


class KeyDerivation(Protocol):
    """`H`, as the queries it makes."""

    name: str

    def stepper(self, tag: int, layer: int, node: int, parents: Sequence[int]) -> KeyStepper: ...

    def steps(self, parents: Sequence[int]) -> list[tuple[int, ...]]:
        """The parents each successive query reads: the key's round structure."""
        ...

    def answer(self, q: Query) -> int: ...


def key_rounds(h: KeyDerivation, parents: Sequence[int], ready: Mapping[int, int]) -> int:
    """The round after which the key is known when parent `u` is known after round `ready[u]` (0 when held)."""
    r = 0
    for needs in h.steps(parents):
        r = max([r] + [ready[u] for u in needs]) + 1
    return r


def key_program(h: KeyDerivation, q: KeyQuery):
    """The key of `q` as an oracle program, every parent known from the start."""
    labels = dict(q.parents)
    st = h.stepper(q.tag, q.layer, q.node, [u for u, _ in q.parents])
    while not st.done:
        [a] = yield [st.query(labels)]
        st.feed(a)
    return st.key


class Permutation(Protocol):
    name: str
    width: int

    def forward(self, x: int) -> int: ...

    def inverse(self, y: int) -> int: ...


class TweakablePermutation(Protocol):
    name: str
    width: int

    def forward(self, t: Tweak, x: int) -> int: ...

    def inverse(self, t: Tweak, y: int) -> int: ...


@dataclass(frozen=True, slots=True)
class Primitives:
    h: KeyDerivation
    p: TweakablePermutation


def header(tag: int, layer: int, node: int, npar: int) -> bytes:
    """`"pous-p3-v1-H" || tag_s || u8(layer) || u16(v) || u8(npar)`, 48 B."""
    if not 0 <= npar < 256:
        raise ValueError("at most 255 parents")
    return HEADER + tag.to_bytes(32, "little") + bytes([layer]) + node.to_bytes(2, "little") + bytes([npar])


def pad_block(m: int) -> int:
    """pad10*1 as a whole block of `m` bits, its first and last bit set (Lean `padBlk`): bytes 0x01, zeros, 0x80."""
    return 1 | (1 << (m - 1))


@dataclass(slots=True)
class OverwriteStepper:
    m: int
    h: int
    order: list[int]
    i: int = 0
    r: int = 0
    done: bool = False
    key: int | None = None

    def block(self, labels: Mapping[int, int]) -> int:
        return labels[self.order[self.i]] if self.i < len(self.order) else pad_block(self.m)

    def query(self, labels: Mapping[int, int]) -> Query:
        return PiQuery(self.block(labels) | (self.h << self.m))

    def feed(self, answer: int) -> None:
        self.r, self.h = answer & mask(self.m), answer >> self.m
        self.i += 1
        if self.i == len(self.order) + 1:
            self.done, self.key = True, self.r


@dataclass(frozen=True, slots=True)
class OverwriteChain:
    """`H`: the overwrite chain of `docs/p3-instantiation.md` §1 over `pi2`, a permutation of `2m + 512` bits."""

    params: Params
    pi2: Permutation
    name: str = "pous-p3-v1-H/overwrite"

    def __post_init__(self) -> None:
        if self.pi2.width != 2 * self.params.m + CAPACITY_BITS:
            raise ValueError(f"the overwrite chain needs a permutation of 2m + {CAPACITY_BITS} bits")
        if self.params.m % 8:
            raise ValueError("labels must be whole bytes")

    def initial(self, tag: int, layer: int, node: int, npar: int) -> int:
        """`h_0`: the header, zero-filled to `m + 64` B."""
        return int.from_bytes(header(tag, layer, node, npar), "little")

    def stepper(self, tag: int, layer: int, node: int, parents: Sequence[int]) -> OverwriteStepper:
        return OverwriteStepper(self.params.m, self.initial(tag, layer, node, len(parents)),
                                 sorted(parents, reverse=True))

    def steps(self, parents: Sequence[int]) -> list[tuple[int, ...]]:
        return [(u,) for u in sorted(parents, reverse=True)] + [()]

    def answer(self, q: Query) -> int:
        if isinstance(q, PiQuery):
            return self.pi2.forward(q.x)
        if isinstance(q, PiInvQuery):
            return self.pi2.inverse(q.y)
        raise TypeError(f"the overwrite chain answers Π₂ and Π₂⁻¹ queries, not {type(q).__name__}")


@dataclass(slots=True)
class _RateStepper:
    """The broken rate-`m` modes: a header block, the parents newest first, the pad; `x` is the last pre-image."""

    blocks: list[tuple[int, int | None]]
    m: int
    feed_forward: bool
    i: int = 0
    state: int = 0
    x: int = 0
    done: bool = False
    key: int | None = None

    def block(self, labels: Mapping[int, int]) -> int:
        constant, parent = self.blocks[self.i]
        return constant if parent is None else labels[parent]

    def query(self, labels: Mapping[int, int]) -> Query:
        self.x = self.state ^ self.block(labels)
        return PiQuery(self.x)

    def feed(self, answer: int) -> None:
        self.state = answer ^ self.x if self.feed_forward else answer
        self.i += 1
        if self.i == len(self.blocks):
            self.done, self.key = True, self.state & mask(self.m)


@dataclass(frozen=True, slots=True)
class XorSponge:
    """**Broken.** The 15:50Z sponge: `s_j = Π(s_{j−1} ⊕ (B_j ‖ 0))` over `pi` of `m + 512` bits, zero IV, blocks the
    header (zero-filled to `m`), the parents newest first and the pad; the key is the rate.  One state regenerates two
    labels through `Π⁻¹`."""

    params: Params
    pi: Permutation
    name: str = "pous-p3-v1-H/xor-sponge-broken"
    feed_forward = False

    def __post_init__(self) -> None:
        if self.pi.width != self.params.m + CAPACITY_BITS:
            raise ValueError(f"the sponge needs a permutation of m + {CAPACITY_BITS} bits")
        if self.params.m % 8 or self.params.m // 8 < len(HEADER) + 36:
            raise ValueError("the sponge's rate must be whole bytes and hold its 48-byte header")

    def stepper(self, tag: int, layer: int, node: int, parents: Sequence[int]) -> _RateStepper:
        blocks = ([(int.from_bytes(header(tag, layer, node, len(parents)), "little"), None)]
                  + [(0, u) for u in sorted(parents, reverse=True)] + [(pad_block(self.params.m), None)])
        return _RateStepper(blocks, self.params.m, self.feed_forward)

    def steps(self, parents: Sequence[int]) -> list[tuple[int, ...]]:
        return [()] + [(u,) for u in sorted(parents, reverse=True)] + [()]

    def answer(self, q: Query) -> int:
        if isinstance(q, PiQuery):
            return self.pi.forward(q.x)
        if isinstance(q, PiInvQuery):
            return self.pi.inverse(q.y)
        raise TypeError(f"the sponge answers Π and Π⁻¹ queries, not {type(q).__name__}")


@dataclass(frozen=True, slots=True)
class MiyaguchiPreneel(XorSponge):
    """**Broken on band rows.** As `XorSponge`, but `s_j = Π(x_j) ⊕ x_j`: the pre-image `x_j` is a checkpoint and a
    masked parent."""

    name: str = "pous-p3-v1-H/miyaguchi-preneel-broken"
    feed_forward = True


@dataclass(slots=True)
class _OneQuery:
    tag: int
    layer: int
    node: int
    parents: tuple[int, ...]
    done: bool = False
    key: int | None = None

    def query(self, labels: Mapping[int, int]) -> Query:
        return HQuery(KeyQuery(self.tag, self.layer, self.node, tuple((u, labels[u]) for u in self.parents)))

    def feed(self, answer: int) -> None:
        self.done, self.key = True, answer


@dataclass(frozen=True, slots=True)
class RandomOracleKey:
    """`H` as one query to `fn : KQ' → Lab`: Lean's monolithic random oracle."""

    fn: Callable[[KeyQuery], int]
    name: str = "random-oracle-key"

    def stepper(self, tag: int, layer: int, node: int, parents: Sequence[int]) -> _OneQuery:
        return _OneQuery(tag, layer, node, tuple(sorted(parents)))

    def steps(self, parents: Sequence[int]) -> list[tuple[int, ...]]:
        return [tuple(sorted(parents))]

    def answer(self, q: Query) -> int:
        if not isinstance(q, HQuery):
            raise TypeError(f"a random-oracle key answers H queries, not {type(q).__name__}")
        return self.fn(q.q)


def _xor(a: bytes, b: bytes) -> bytes:
    return (int.from_bytes(a, "little") ^ int.from_bytes(b, "little")).to_bytes(len(a), "little")


@dataclass(frozen=True, slots=True)
class Feistel10:
    """Ten balanced Feistel rounds on `nbytes` bytes with `F_i(T, x) = SHAKE256(domain || u8(i) || T || x)[0 : half]`."""

    nbytes: int
    domain: int

    def __post_init__(self) -> None:
        if self.nbytes < 2 or self.nbytes % 2:
            raise ValueError("a balanced Feistel network needs an even number of bytes")

    def f(self, i: int, t: bytes, x: bytes) -> bytes:
        return hashlib.shake_256(bytes([self.domain, i]) + t + x).digest(self.nbytes // 2)

    def forward(self, x: int, t: bytes = b"") -> int:
        b = x.to_bytes(self.nbytes, "little")
        left, right = b[:self.nbytes // 2], b[self.nbytes // 2:]
        for i in range(1, FEISTEL_ROUNDS + 1):
            left, right = right, _xor(left, self.f(i, t, right))
        return int.from_bytes(left + right, "little")

    def inverse(self, y: int, t: bytes = b"") -> int:
        b = y.to_bytes(self.nbytes, "little")
        left, right = b[:self.nbytes // 2], b[self.nbytes // 2:]
        for i in range(FEISTEL_ROUNDS, 0, -1):
            left, right = _xor(right, self.f(i, t, left)), left
        return int.from_bytes(left + right, "little")


def _check_width(width: int, x: int) -> None:
    if not 0 <= x < 1 << width:
        raise ValueError(f"input outside {width} bits")


@dataclass(frozen=True, slots=True)
class FeistelPi:
    """`Π₂` (or the broken modes' `Π`): Feistel10 on `width` bits, untweaked, domain byte 0x48."""

    width: int
    name: str = "feistel10-shake256-pi/v1"
    core: Feistel10 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.width % 8:
            raise ValueError("Π works on whole bytes")
        object.__setattr__(self, "core", Feistel10(self.width // 8, PI_DOMAIN))

    def forward(self, x: int) -> int:
        _check_width(self.width, x)
        return self.core.forward(x)

    def inverse(self, y: int) -> int:
        _check_width(self.width, y)
        return self.core.inverse(y)


@dataclass(frozen=True, slots=True)
class FeistelP:
    """`P_T`: Feistel10 on labels with the 35-byte tweak in every round input, domain byte 0x50."""

    width: int
    name: str = "feistel10-shake256-p-tweakable/v1"
    core: Feistel10 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.width % 8:
            raise ValueError("P works on whole bytes")
        object.__setattr__(self, "core", Feistel10(self.width // 8, P_DOMAIN))

    def forward(self, t: Tweak, x: int) -> int:
        _check_width(self.width, x)
        return self.core.forward(x, t.to_bytes())

    def inverse(self, t: Tweak, y: int) -> int:
        _check_width(self.width, y)
        return self.core.inverse(y, t.to_bytes())


def reference(params: Params) -> Primitives:
    """The default backend: the overwrite chain over Feistel-SHAKE256 `Π₂`, and the tweakable Feistel-SHAKE256 `P_T`."""
    return Primitives(OverwriteChain(params, FeistelPi(2 * params.m + CAPACITY_BITS)), FeistelP(params.m))


def _expand(label: bytes, n: int) -> bytes:
    out = bytearray()
    c = 0
    while len(out) < n:
        out += hashlib.sha256(label + c.to_bytes(8, "big")).digest()
        c += 1
    return bytes(out[:n])


@lru_cache(maxsize=None)
def diffusion_stages(words: int) -> int:
    """The fewest butterfly stages after which every output word depends on every input word."""
    if words < 2:
        raise ValueError("a butterfly needs at least two words")
    levels = (words - 1).bit_length()
    deps = [1 << i for i in range(words)]
    full = (1 << words) - 1
    s = 0
    while not all(d == full for d in deps):
        step = 1 << (s % levels)
        for i in range(words):
            if not i & step and i + step < words:
                deps[i] = deps[i + step] = deps[i] | deps[i + step]
        s += 1
    return s


def _lanes(words: int, members: Sequence[int], lane: int) -> int:
    """The mask with `lane` (a 32-bit value) in every word of `members`."""
    buf = bytearray(4 * words)
    chunk = lane.to_bytes(4, "little")
    for i in members:
        buf[4 * i:4 * i + 4] = chunk
    return int.from_bytes(buf, "little")


@dataclass(frozen=True, slots=True)
class _Stage:
    shift: int
    rot_a: int
    rot_c: int
    a_words: tuple[int, ...]
    lanes: int
    low: int
    high: int
    keep: int
    rot: dict[int, tuple[int, int]]
    carry: tuple[tuple[int, int], ...]
    constants: int

    def rotl(self, x: int, r: int) -> int:
        """Rotate every paired lane of `x` left by `r`."""
        hi, lo = self.rot[r]
        return ((x << r) & hi) | ((x >> (32 - r)) & lo)


def _rotations(seed: bytes) -> tuple[int, int]:
    b = _expand(seed + b"/rot", 2)
    return 1 + b[0] % 31, 1 + b[1] % 31


@lru_cache(maxsize=16)
def _schedule(name: str, words: int, stages: int) -> tuple[_Stage, ...]:
    levels = (words - 1).bit_length()
    full = (1 << (32 * words)) - 1
    out = []
    for s in range(stages):
        step = 1 << (s % levels)
        a_words = tuple(i for i in range(words) if not i & step and i + step < words)
        seed = f"verity-pous/arx/{name}/{words}/{s}".encode()
        rot_a, rot_c = _rotations(seed)
        lanes = _lanes(words, a_words, 0xFFFFFFFF)
        rot = {r: (_lanes(words, a_words, (0xFFFFFFFF << r) & 0xFFFFFFFF), _lanes(words, a_words, (1 << r) - 1))
               for r in {rot_a, rot_c, 32 - rot_a, 32 - rot_c}}
        carry = tuple((j, _lanes(words, a_words, (0xFFFFFFFF << j) & 0xFFFFFFFF)) for j in (1, 2, 4, 8, 16))
        rc = int.from_bytes(_expand(seed + b"/rc", 4 * words), "little") & lanes
        out.append(_Stage(32 * step, rot_a, rot_c, a_words, lanes, _lanes(words, a_words, 0x7FFFFFFF),
                          _lanes(words, a_words, 0x80000000), full ^ lanes ^ (lanes << (32 * step)), rot, carry, rc))
    return tuple(out)


def _bit_reach(name: str, words: int, bit: int, limit: int) -> int:
    """The stages after which every bit depends on input bit `bit`: an added bit reaches every higher bit of its word
    (the carries), and rotations and XORs move dependencies exactly."""
    full = (1 << (32 * words)) - 1
    d = 1 << bit
    for s, st in enumerate(_schedule(name, words, limit)):
        a = d & st.lanes
        c = (d >> st.shift) & st.lanes
        t = a | c
        for j, m in st.carry:
            t |= (t << j) & m
        a2 = st.rotl(t, st.rot_a)
        c2 = st.rotl(c, st.rot_c) | a2
        d = (d & st.keep) | a2 | (c2 << st.shift)
        if d == full:
            return s + 1
    raise AssertionError(f"input bit {bit} does not reach every output bit in {limit} stages")


def bit_diffusion_stages(name: str, words: int, bits: Sequence[int] | None = None) -> int:
    """The fewest stages after which every output bit depends on each input bit in `bits` (default: every bit of
    words 0, 1, the middle, the top power of two and the last).  Dependency is a lower bound on statistical
    diffusion, which `docs/efficient-crypto` measured at 16–20 stages for 256 words."""
    if bits is None:
        top = 1 << ((words - 1).bit_length() - 1)
        bits = [32 * w + j for w in sorted({0, 1, words // 2, top - 1, top, words - 1}) for j in range(32)]
    limit = 8 * (words - 1).bit_length() + 64
    return max(_bit_reach(name, words, b, limit) for b in bits)


@lru_cache(maxsize=None)
def butterfly_stages(name: str, words: int) -> int:
    """Twice the bit-level count, and at least twice word-level diffusion plus 12 (20 stages at 256 words, the top of
    the measured 16–20)."""
    return 2 * max(bit_diffusion_stages(name, words), diffusion_stages(words) + 12)


@dataclass(frozen=True, slots=True)
class ArxButterfly:
    """The wide ARX butterfly on `width` bits (a multiple of 32, at least 64).  Each pair `(a, c)` maps to
    `a' = rotl((a + c) mod 2^32, ra_s) ⊕ k_{s,i}` and `c' = rotl(c, rc_s) ⊕ a'`, so every word is rotated in every stage
    it is paired.  `forward_words` is the per-word specification; `forward` computes the same map on all pairs of a
    stage at once."""

    width: int
    name: str = "arx-butterfly/heuristic-v2"
    stages: int = field(init=False)

    def __post_init__(self) -> None:
        if self.width % 32 or self.width < 64:
            raise ValueError("the ARX butterfly works on at least two 32-bit words")
        object.__setattr__(self, "stages", butterfly_stages(self.name, self.width // 32))

    @property
    def words(self) -> int:
        return self.width // 32

    def forward(self, x: int) -> int:
        _check_width(self.width, x)
        for st in _schedule(self.name, self.words, self.stages):
            a = x & st.lanes
            c = (x >> st.shift) & st.lanes
            s = ((a & st.low) + (c & st.low)) ^ ((a ^ c) & st.high)
            a2 = st.rotl(s, st.rot_a) ^ st.constants
            c2 = st.rotl(c, st.rot_c) ^ a2
            x = (x & st.keep) | a2 | (c2 << st.shift)
        return x

    def inverse(self, y: int) -> int:
        _check_width(self.width, y)
        for st in reversed(_schedule(self.name, self.words, self.stages)):
            a2 = y & st.lanes
            c2 = (y >> st.shift) & st.lanes
            c = st.rotl(c2 ^ a2, 32 - st.rot_c)
            s = st.rotl(a2 ^ st.constants, 32 - st.rot_a)
            a = ((s | st.high) - (c & st.low)) ^ ((s ^ c ^ st.high) & st.high)
            y = (y & st.keep) | a | (c << st.shift)
        return y

    def forward_words(self, x: int) -> int:
        """The specification, one pair at a time."""
        w = [(x >> (32 * i)) & 0xFFFFFFFF for i in range(self.words)]

        def rotl(v: int, r: int) -> int:
            return ((v << r) | (v >> (32 - r))) & 0xFFFFFFFF

        for st in _schedule(self.name, self.words, self.stages):
            step = st.shift // 32
            for i in st.a_words:
                a, c = w[i], w[i + step]
                a = rotl((a + c) & 0xFFFFFFFF, st.rot_a) ^ ((st.constants >> (32 * i)) & 0xFFFFFFFF)
                c = rotl(c, st.rot_c) ^ a
                w[i], w[i + step] = a, c
        return sum(v << (32 * i) for i, v in enumerate(w))


@dataclass(frozen=True, slots=True)
class EvenMansourArx:
    """Heuristic `P_T`: `k ⊕ π₅(k ⊕ π₄(k ⊕ π₃(k ⊕ π₂(k ⊕ π₁(x ⊕ k)))))` with `k = SHAKE256(0x4B || T)[0 : m]`."""

    width: int
    name: str = "even-mansour5-arx/heuristic-v2"
    rounds: tuple[ArxButterfly, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rounds", tuple(ArxButterfly(self.width, f"arx-em-pi{j}/heuristic-v2")
                                                 for j in range(1, 6)))

    def _key(self, t: Tweak) -> int:
        return int.from_bytes(hashlib.shake_256(bytes([EM_KEY_DOMAIN]) + t.to_bytes()).digest(self.width // 8), "little")

    def forward(self, t: Tweak, x: int) -> int:
        _check_width(self.width, x)
        k = self._key(t)
        x ^= k
        for pi in self.rounds:
            x = pi.forward(x) ^ k
        return x

    def inverse(self, t: Tweak, y: int) -> int:
        _check_width(self.width, y)
        k = self._key(t)
        for pi in reversed(self.rounds):
            y = pi.inverse(y ^ k)
        return y ^ k


def heuristic_arx(params: Params) -> Primitives:
    """The heuristic fast option (GPU-friendly, not analysed): the overwrite chain over an ARX `Π₂`, Even–Mansour ARX
    `P_T`."""
    return Primitives(OverwriteChain(params, ArxButterfly(2 * params.m + CAPACITY_BITS, "arx-pi2/heuristic-v2")),
                      EvenMansourArx(params.m))


IDEAL_DOMAIN = "verity/pous/ideal-model/v1"


@dataclass(slots=True)
class IdealPermutation:
    """A lazily sampled uniform permutation of `width` bits, consistent in both directions, drawn from
    `verity.ml.randomness` under `seed`."""

    width: int
    seed: bytes
    label: str = "pi"
    name: str = "ideal-permutation"
    _key: Key = field(init=False, repr=False)
    _fwd: dict[int, int] = field(init=False, repr=False, default_factory=dict)
    _inv: dict[int, int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._key = derive(self.seed, IDEAL_DOMAIN, {"primitive": self.label, "width": self.width})

    def _fresh(self, direction: str, x: int, used: dict[int, int]) -> int:
        attempt = 0
        while True:
            y = self._key.uniform(1 << self.width, (direction, x, attempt))
            if y not in used:
                return y
            attempt += 1

    def forward(self, x: int) -> int:
        _check_width(self.width, x)
        if x not in self._fwd:
            y = self._fresh("f", x, self._inv)
            self._fwd[x], self._inv[y] = y, x
        return self._fwd[x]

    def inverse(self, y: int) -> int:
        _check_width(self.width, y)
        if y not in self._inv:
            x = self._fresh("i", y, self._fwd)
            self._inv[y], self._fwd[x] = x, y
        return self._inv[y]


@dataclass(slots=True)
class IdealTweakablePermutation:
    """An independent lazily sampled permutation for every tweak."""

    width: int
    seed: bytes
    name: str = "ideal-tweakable-permutation"
    _perms: dict[Tweak, IdealPermutation] = field(init=False, repr=False, default_factory=dict)

    def _perm(self, t: Tweak) -> IdealPermutation:
        if t not in self._perms:
            self._perms[t] = IdealPermutation(self.width, self.seed, f"p/{t.to_bytes().hex()}")
        return self._perms[t]

    def forward(self, t: Tweak, x: int) -> int:
        return self._perm(t).forward(x)

    def inverse(self, t: Tweak, y: int) -> int:
        return self._perm(t).inverse(y)


def ideal(params: Params, seed: bytes) -> Primitives:
    """The two-permutation model: the overwrite chain over an ideal `Π₂`, and an ideal tweakable `P`."""
    return Primitives(OverwriteChain(params, IdealPermutation(2 * params.m + CAPACITY_BITS, seed, "pi2")),
                      IdealTweakablePermutation(params.m, seed))
