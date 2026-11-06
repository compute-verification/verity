"""The recipe of ``bench-instances/v1``: how every instance of the frozen benchmark set is derived.

Everything here is deterministic from string labels (:class:`Rng`), the 24 seed captures and the pinned Hugging Face
revision of ``Qwen/Qwen2.5-1.5B``; :mod:`verity_numerical.bench.instances` drives it and owns the byte layout.  The
relation is never restated here: expected words come from ``verity.ml.circuits.silicon.silicon`` (the target's executable
definition) and are cross-checked against ``verity_numerical.reference.hawkeye_close`` by the driver.

Contents, in order: the hash-stream RNG; BF16/FP32 word helpers; the seeds (the 24 real K=1536 GEMM coordinates of
capture ``B1-b3eba618b1b5-20260907T1731Z``); the zero-dependency safetensors row fetcher and the fused vLLM row order;
the real-instance plan (one activation row x many weight rows of the same layer/op); the eleven valid corner families
of the VU tier; the twelve designed corner families of the transition-unit tier; the deliberately wrong models
(``MISTAKES``) that the negatives tier and the sensitivity table are built from, including an exact IEEE-754 FP32
sequential fma.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import time
import urllib.error
import urllib.request
from array import array
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from verity.ml.circuits.silicon import silicon as S
from verity.ml.circuits.silicon.cast import f32_to_bf16_word
from verity.ml.circuits.silicon.errors import InvalidArtifact

from verity_numerical.reference import hawkeye_close as H

__all__ = [
    "BF16_INF",
    "BF16_ONE",
    "CORNER_FAMILIES",
    "DATASET",
    "EXPERIMENT_ID",
    "FAMILY_N",
    "HF_FILE",
    "HF_REPO",
    "HF_REVISION",
    "HF_URL",
    "MISTAKES",
    "PIPELINE",
    "REPO_FIXTURE_DIR",
    "ROWS_PER_X",
    "SEED_NAMES",
    "STEP",
    "STEPS",
    "TU_FAMILIES",
    "K",
    "Mistake",
    "Rng",
    "Safetensors",
    "Seed",
    "bf16",
    "bf16_pow2",
    "chain_accs",
    "epilogue",
    "generate_corner",
    "generate_tu_corner",
    "hawkeye_step",
    "ieee_fma_chain",
    "is_finite_bf16",
    "is_finite_f32",
    "load_seeds",
    "non_sticky_tc_dot",
    "plan_real",
    "tensor_of",
    "variant_chain",
]

DATASET = "bench-instances/v1"
K = 1536
STEP = 16
STEPS = K // STEP
PIPELINE = S.AMPERE_BF16_M16N8K16

BF16_ONE = 0x3F80
BF16_MAX = 0x7F7F
BF16_INF = 0x7F80
F32_INF = 0x7F800000


# --- deterministic randomness -----------------------------------------------------------------------------------------------


class Rng:
    """A SHA-256 counter stream keyed by labels: ``key = sha256("bench-instances/v1|" + "|".join(labels))``, block ``n`` is
    ``sha256(key || n as u64 LE)``.  Stable across Python versions and trivially re-implementable in any language; every
    sample below consumes the stream in a documented way (``bits`` takes ceil(n/8) bytes little-endian and masks; ``below``
    rejection-samples ``bits(bit_length(n-1))``)."""

    def __init__(self, *labels: object) -> None:
        self._key = hashlib.sha256((DATASET + "|" + "|".join(str(x) for x in labels)).encode()).digest()
        self._ctr = 0
        self._buf = b""
        self._pos = 32

    def bytes(self, n: int) -> bytes:
        out = bytearray()
        while n:
            if self._pos >= 32:
                self._buf = hashlib.sha256(self._key + self._ctr.to_bytes(8, "little")).digest()
                self._ctr += 1
                self._pos = 0
            take = min(n, 32 - self._pos)
            out += self._buf[self._pos : self._pos + take]
            self._pos += take
            n -= take
        return bytes(out)

    def bits(self, n: int) -> int:
        if n <= 0:
            return 0
        return int.from_bytes(self.bytes((n + 7) // 8), "little") & ((1 << n) - 1)

    def below(self, n: int) -> int:
        """Uniform in ``[0, n)``."""

        if n <= 0:
            raise ValueError("below() needs n >= 1")
        k = (n - 1).bit_length()
        while True:
            r = self.bits(k)
            if r < n:
                return r

    def range(self, lo: int, hi: int) -> int:
        """Uniform in ``[lo, hi)``."""

        return lo + self.below(hi - lo)

    def choice(self, seq: Sequence):
        return seq[self.below(len(seq))]

    def sample(self, n: int, k: int, exclude: frozenset[int] | set[int] = frozenset()) -> list[int]:
        """``k`` distinct values of ``range(n)`` outside ``exclude``, in draw order."""

        if k > n - len(exclude):
            raise ValueError("not enough values to sample")
        seen: set[int] = set(exclude)
        out: list[int] = []
        while len(out) < k:
            r = self.below(n)
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    def bf16(self, e_lo: int, e_hi: int, sign: int | None = None, mantissa: int | None = None) -> int:
        """A BF16 word with exponent field uniform in ``[e_lo, e_hi]`` (inclusive), random sign and mantissa unless given."""

        e = self.range(e_lo, e_hi + 1)
        m = self.bits(7) if mantissa is None else mantissa
        s = self.bits(1) if sign is None else sign
        return bf16(s, e, m)

    def f32(self, e_lo: int, e_hi: int, sign: int | None = None) -> int:
        e = self.range(e_lo, e_hi + 1)
        s = self.bits(1) if sign is None else sign
        return (s << 31) | (e << 23) | self.bits(23)


# --- words -------------------------------------------------------------------------------------------------------------------


def bf16(sign: int, e: int, m: int) -> int:
    return ((sign & 1) << 15) | ((e & 0xFF) << 7) | (m & 0x7F)


def bf16_pow2(e: int, sign: int = 0) -> int:
    """The BF16 word of ``+-2^e`` for ``-126 <= e <= 127``."""

    if not -126 <= e <= 127:
        raise ValueError(f"2^{e} is not a normal BF16")
    return bf16(sign, e + 127, 0)


def is_finite_bf16(w: int) -> bool:
    return ((w >> 7) & 0xFF) != 0xFF


def is_finite_f32(u: int) -> bool:
    return ((u >> 23) & 0xFF) != 0xFF


def is_inf_f32(u: int) -> bool:
    return (u & 0x7FFFFFFF) == F32_INF


def _pair_pow2(rng: Rng, e: int, sign: int) -> tuple[int, int]:
    """Two BF16 powers of two whose product is ``+-2^e`` (``-252 <= e <= 254``), the split sampled from ``rng``."""

    lo, hi = max(-126, e - 127), min(127, e + 126)
    e1 = rng.range(lo, hi + 1)
    return bf16_pow2(e1, sign), bf16_pow2(e - e1)


def _pair_with_exponent(rng: Rng, e: int, sign: int) -> tuple[int, int]:
    """Two BF16 words with random mantissas whose exact product has :func:`silicon.bf16_product` exponent ``e``
    (``ea + eb - 254 = e``, both normal)."""

    lo, hi = max(1, e), min(254, e + 253)
    ea = rng.range(lo, hi + 1)
    eb = e + 254 - ea
    return bf16(sign, ea, rng.bits(7)), bf16(0, eb, rng.bits(7))


# --- the relation, as the generator uses it -------------------------------------------------------------------------------------


def chain_accs(x: Sequence[int], w: Sequence[int], acc0: int = 0, pipeline: S.Pipeline = PIPELINE) -> list[int]:
    """The ``K/16`` accumulator words of the chain (``acc_0 .. acc_{K/16-1}``), raising ``InvalidArtifact`` outside the domain."""

    acc = acc0
    out = []
    for s in range(0, len(x), STEP):
        acc = S.tc_dot(pipeline, acc, x[s : s + STEP], w[s : s + STEP])
        out.append(acc)
    return out


def epilogue(acc: int) -> int:
    return f32_to_bf16_word(acc)


def hawkeye_step(c: int, a: Sequence[int], b: Sequence[int], silicon_word: int) -> int | None:
    """The source-close word for one transition, or ``None`` where D8 says the simulator is not the reference (the
    silicon result ``silicon_word`` saturates).  Callers compare the returned word to ``silicon_word``."""

    if is_inf_f32(silicon_word):
        return None
    return H.tc_dot16(c, list(a), list(b))


# --- seeds ----------------------------------------------------------------------------------------------------------------------

EXPERIMENT_ID = "B1-b3eba618b1b5-20260907T1731Z"
REPO_FIXTURE_DIR = "verity/ml/circuits/ops/tests/fixtures/gemm-B1-b3eba618b1b5-20260907T1731Z"
SEED_FAMILIES = ("qkv_proj", "o_proj", "gate_up_proj", "lm_head")
PHASES = ("decode", "prefill")
#: Seed order: id 0 of ``vu-k1536`` is ``qkv_proj_decode_0`` (the target's K=1536 conformance vector).
SEED_NAMES = tuple(f"{fam}_{ph}_{n}" for fam in SEED_FAMILIES for ph in PHASES for n in range(3))
FAMILY_N = {"qkv_proj": 2048, "o_proj": 1536, "gate_up_proj": 17920, "lm_head": 151936}


@dataclass(frozen=True)
class Seed:
    name: str
    family: str
    layer: str
    op: str
    phase: str
    j: int
    N: int
    x: tuple[int, ...]
    w: tuple[int, ...]
    committed_word: int
    sha256: dict[str, str]


def read_u16(path: Path) -> list[int]:
    data = path.read_bytes()
    return list(struct.unpack(f"<{len(data) // 2}H", data))


def load_seeds(seeds_dir: Path) -> list[Seed]:
    """The 24 seeds from ``seeds/<name>/{x.u16, w_row.u16, fixture.json}`` in :data:`SEED_NAMES` order."""

    out = []
    for name in SEED_NAMES:
        d = seeds_dir / name
        fx = json.loads((d / "fixture.json").read_text())
        x, w = read_u16(d / "x.u16"), read_u16(d / "w_row.u16")
        if fx["K"] != K or len(x) != K or len(w) != K:
            raise ValueError(f"seed {name}: not a K={K} capture")
        out.append(Seed(name, fx["family"], str(fx["layer"]), fx["op"], fx["phase"], int(fx["j"]), int(fx["N"]), tuple(x), tuple(w),
                        int(fx["committed_word"]), {k: v["sha256"] for k, v in fx["files"].items()}))
    return out


# --- the model's weights: safetensors over HTTP range requests --------------------------------------------------------------------

HF_REPO = "Qwen/Qwen2.5-1.5B"
HF_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
HF_FILE = "model.safetensors"
HF_URL = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{HF_FILE}"
Q_ROWS, KV_ROWS, INTERMEDIATE = 1536, 256, 8960


def tensor_of(family: str, layer: str, j: int) -> tuple[str, int]:
    """The safetensors tensor and row holding fused vLLM row ``j`` of ``family`` at ``layer`` (TP=1: plain concatenation
    ``q | k | v`` and ``gate | up``; ``lm_head`` is tied to ``model.embed_tokens.weight``, the checkpoint has no
    ``lm_head.weight``)."""

    if family == "qkv_proj":
        if j < Q_ROWS:
            return f"model.layers.{layer}.self_attn.q_proj.weight", j
        if j < Q_ROWS + KV_ROWS:
            return f"model.layers.{layer}.self_attn.k_proj.weight", j - Q_ROWS
        return f"model.layers.{layer}.self_attn.v_proj.weight", j - Q_ROWS - KV_ROWS
    if family == "o_proj":
        return f"model.layers.{layer}.self_attn.o_proj.weight", j
    if family == "gate_up_proj":
        if j < INTERMEDIATE:
            return f"model.layers.{layer}.mlp.gate_proj.weight", j
        return f"model.layers.{layer}.mlp.up_proj.weight", j - INTERMEDIATE
    if family == "lm_head":
        return "model.embed_tokens.weight", j
    raise ValueError(f"unknown family {family!r}")


class Safetensors:
    """Rows of BF16 ``[N, K]`` tensors of one safetensors file over HTTP range requests (``urllib`` only): the 8-byte
    little-endian header length, the JSON header with ``data_offsets`` relative to the byte after it, then raw data.
    Fetched rows are cached under ``cache`` as ``<tensor>/<row>.u16``."""

    def __init__(self, url: str = HF_URL, cache: Path | None = None, timeout: float = 60.0, retries: int = 6, workers: int = 16) -> None:
        self.url = url
        self.cache = cache
        self.timeout = timeout
        self.retries = retries
        self.workers = workers
        self._header: dict | None = None
        self._header_sha256 = ""
        self._header_len = 0
        self._base = 0
        self.requests = 0

    def _get(self, lo: int, hi: int) -> bytes:
        """Bytes ``[lo, hi)``, retried on transient failures."""

        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi - 1}", "User-Agent": "verity-bench-instances/1"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    self.requests += 1
                    if r.status != 206:
                        raise OSError(f"expected 206 Partial Content, got {r.status}")
                    data = r.read()
                if len(data) != hi - lo:
                    raise OSError(f"range {lo}-{hi} returned {len(data)} bytes")
                return data
            except (urllib.error.URLError, OSError, ConnectionError) as e:
                last = e
                time.sleep(min(30.0, 0.5 * 2**attempt))
        raise OSError(f"range request {lo}-{hi} failed after {self.retries} attempts: {last}")

    @property
    def header(self) -> dict:
        if self._header is None:
            n = struct.unpack("<Q", self._get(0, 8))[0]
            raw = self._get(8, 8 + n)
            self._header = json.loads(raw)
            self._header_sha256 = hashlib.sha256(raw).hexdigest()
            self._header_len = n
            self._base = 8 + n
        return self._header

    def header_info(self) -> dict:
        _ = self.header
        return {"length": self._header_len, "sha256": self._header_sha256}

    def tensor(self, name: str) -> dict:
        t = self.header.get(name)
        if t is None:
            raise KeyError(f"{name} is not in {self.url}")
        return t

    def row(self, name: str, r: int) -> bytes:
        t = self.tensor(name)
        if t["dtype"] != "BF16" or len(t["shape"]) != 2:
            raise ValueError(f"{name}: expected a 2-D BF16 tensor, got {t}")
        n_rows, k = t["shape"]
        if not 0 <= r < n_rows:
            raise IndexError(f"{name}: row {r} of {n_rows}")
        path = self.cache / name / f"{r}.u16" if self.cache else None
        if path and path.is_file() and path.stat().st_size == k * 2:
            return path.read_bytes()
        off = self._base + t["data_offsets"][0] + r * k * 2
        data = self._get(off, off + k * 2)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return data

    def rows(self, items: Sequence[tuple[str, int]]) -> dict[tuple[str, int], bytes]:
        _ = self.header  # fetch once, before the threads start
        with ThreadPoolExecutor(self.workers) as ex:
            got = list(ex.map(lambda it: self.row(*it), items))
        return dict(zip(items, got, strict=True))


# --- the real-instance plan -------------------------------------------------------------------------------------------------------

#: Weight rows drawn per activation row beyond the seed's own: 24 x (1 + ROWS_PER_X) = 3,456 real instances.
ROWS_PER_X = 143


@dataclass(frozen=True)
class RealSpec:
    seed: int  # index into SEED_NAMES: the activation row x
    j: int  # the weight row of the same layer/op (== the seed's own j for the 24 seed instances)
    tensor: str
    row: int


def plan_real(seeds: Sequence[Seed]) -> tuple[list[RealSpec], dict[tuple[str, str], list[int]]]:
    """Round 0: the 24 seeds (x, W[j]); rounds 1..ROWS_PER_X: for every seed in order, (x, W[j'_r]) with ``j'`` the
    r-th row drawn for the seed's (family, layer) group -- seeds sharing a layer/op share the same drawn rows, so the
    set also contains "same weight row, different activation" pairs.  Rows are sampled by ``Rng("rows", family, layer)``
    from ``[0, N)`` excluding the group's seed rows.  Returns the specs in id order and the drawn rows per group."""

    groups: dict[tuple[str, str], list[int]] = {}
    for i, s in enumerate(seeds):
        groups.setdefault((s.family, s.layer), []).append(i)
    rows: dict[tuple[str, str], list[int]] = {}
    for (family, layer), members in groups.items():
        exclude = {seeds[i].j for i in members}
        rows[(family, layer)] = Rng("rows", family, layer).sample(FAMILY_N[family], ROWS_PER_X, exclude)
    specs = []
    for i, s in enumerate(seeds):
        specs.append(RealSpec(i, s.j, *tensor_of(s.family, s.layer, s.j)))
    for r in range(ROWS_PER_X):
        for i, s in enumerate(seeds):
            j = rows[(s.family, s.layer)][r]
            specs.append(RealSpec(i, j, *tensor_of(s.family, s.layer, j)))
    return specs, rows


# --- corner families of the VU tier -----------------------------------------------------------------------------------------------


@dataclass
class Corner:
    x: list[int]
    w: list[int]
    tags: list[str]
    params: dict = field(default_factory=dict)


def _body(rng: Rng, seeds: Sequence[Seed]) -> tuple[list[int], list[int], str]:
    s = rng.choice(seeds)
    return list(s.x), list(s.w), s.name


def _gen_sat_final(rng: Rng, i: int, seeds: Sequence[Seed]) -> Corner:
    """Saturation in the last transition only: y = +-inf through the epilogue (D8: a valid VU output).  ``g1``: group 1
    of step 95 overflows; ``g2``: group 2 does; ``g1-cancel-g2``: group 1 overflows and group 2 carries the exact
    negatives (a non-sticky implementation would cancel back to +0)."""

    x, w, body = _body(rng, seeds)
    sign = i & 1
    variant = ("g1", "g2", "g1-cancel-g2")[(i // 2) % 3]
    last = K - STEP
    if variant == "g2":
        for k in range(last + 8, K):
            x[k], w[k] = bf16(sign, 254, rng.bits(7)), bf16(0, 254, rng.bits(7))
    else:
        for k in range(last, last + 8):
            x[k], w[k] = bf16(sign, 254, rng.bits(7)), bf16(0, 254, rng.bits(7))
        if variant == "g1-cancel-g2":
            for k in range(last, last + 8):
                x[k + 8], w[k + 8] = x[k] ^ 0x8000, w[k]
    accs = chain_accs(x, w)
    if not is_inf_f32(accs[-1]) or (accs[-1] >> 31) != sign:
        raise AssertionError("sat-final did not saturate as designed")
    return Corner(x, w, ["corner", "sat-final", variant, "-" if sign else "+"], {"body": body})


def _gen_near_sat(rng: Rng, i: int, seeds: Sequence[Seed]) -> Corner:
    """A finite final accumulator ``+-0x7F7F0000 | L`` (the largest BF16 magnitude plus ``L`` low mantissa bits): ``L <
    0x8000`` rounds to the largest finite BF16, ``L >= 0x8000`` rounds to +-inf in the epilogue (a finite chain whose only
    infinity is the cast's)."""

    x, w, body = _body(rng, seeds)
    sign = i & 1
    up = (i // 2) & 1
    positions: set[int] = set()
    n_extra = rng.range(0, 7)
    while len(positions) < n_extra:
        positions.add(rng.below(15))
    if up:
        positions.add(15)
    L = sum(1 << t for t in positions)
    last = K - STEP
    x[last], w[last] = bf16(sign, 254, 0x7F), BF16_ONE
    for n, t in enumerate(sorted(positions)):
        x[last + 1 + n], w[last + 1 + n] = _pair_pow2(rng, 104 + t, sign)
    accs = chain_accs(x, w)
    want = (sign << 31) | 0x7F7F0000 | L
    if accs[-1] != want:
        raise AssertionError(f"near-sat: accumulator {accs[-1]:#010x}, designed {want:#010x}")
    y = epilogue(accs[-1])
    if up != (y & 0x7FFF == BF16_INF):
        raise AssertionError("near-sat: the epilogue did not round as designed")
    return Corner(x, w, ["corner", "near-sat", "rounds-to-inf" if up else "max-finite", "-" if sign else "+"], {"body": body, "L": L})


def _gen_cancel(rng: Rng, i: int, seeds: Sequence[Seed]) -> Corner:
    """Heavy cancellation.  ``zero``: every term has its exact negative (within a group, across the two groups, or across
    two steps) so the chain returns to exactly +0 and y = 0x0000.  ``tiny``: three cancelling pairs per group plus two
    small survivors ``d`` binades below them (partially truncated by the alignment), so the accumulator is a tiny residue
    -- subnormal for the low exponents."""

    x, w = [0] * K, [0] * K
    if i % 8 < 3:
        mode = ("within-group", "across-groups", "across-steps")[i % 8]
        if mode == "within-group":
            # four (p, -p) pairs per group: equal magnitudes truncate identically, so every group sums to exactly zero
            for g in range(K // 8):
                for q in range(4):
                    a, b = rng.bf16(60, 190), rng.bf16(60, 190)
                    k = g * 8 + 2 * q
                    x[k], w[k], x[k + 1], w[k + 1] = a, b, a, b ^ 0x8000
        elif mode == "across-groups":
            # one product p in group 1 (exact: the only nonzero term over a zero accumulator), -p in group 2 of the same step
            for s in range(STEPS):
                a, b = rng.bf16(60, 190), rng.bf16(60, 190)
                k1, k2 = s * STEP + rng.below(8), s * STEP + 8 + rng.below(8)
                x[k1], w[k1], x[k2], w[k2] = a, b, a, b ^ 0x8000
        else:
            # step 2m holds one product p (the accumulator becomes exactly p); step 2m+1 holds -p
            for m in range(STEPS // 2):
                a, b = rng.bf16(60, 190), rng.bf16(60, 190)
                k1, k2 = 2 * m * STEP + rng.below(STEP), (2 * m + 1) * STEP + rng.below(STEP)
                x[k1], w[k1], x[k2], w[k2] = a, b, a, b ^ 0x8000
        accs = chain_accs(x, w)
        if accs[-1] != 0:
            raise AssertionError("cancel-zero did not cancel")
        return Corner(x, w, ["corner", "cancel", "zero", mode], {})
    E = rng.choice((-135, -120, -100, -60, -20, 0, 20, 60, 100))
    d = rng.range(14, 25)
    for g in range(K // 8):
        for q in range(3):
            a, b = _pair_with_exponent(rng, E, rng.bits(1))
            k = g * 8 + 2 * q
            x[k], w[k], x[k + 1], w[k + 1] = a, b, a, b ^ 0x8000
        for q in (6, 7):
            x[g * 8 + q], w[g * 8 + q] = _pair_with_exponent(rng, E - d, rng.bits(1))
    accs = chain_accs(x, w)
    sub = (accs[-1] & 0x7F800000) == 0 and accs[-1] & 0x7FFFFFFF
    return Corner(x, w, ["corner", "cancel", "tiny", "subnormal-result" if sub else "normal-result"], {"E": E, "d": d})


NO_FLOOR = S.Pipeline(name="mistake:no-exponent-floor", arch="mistake", dtype="bf16", operand_bits=16, groups=(8, 8), width=25,
                      zero_exponent=-(1 << 20), validated="not hardware: a deliberately wrong model (no -132 floor)")


def _tiny_group(rng: Rng, e_lo: int, e_hi: int, signed: bool = True) -> tuple[list[int], list[int]]:
    """Eight products with silicon exponents in ``[e_lo, e_hi]`` (a subnormal ``a`` -- exponent field 0 counts as 1 -- times
    a normal ``b``), random mantissas, random signs unless ``signed`` is false."""

    a, b = [], []
    for _ in range(8):
        e = rng.range(e_lo, e_hi + 1)
        a.append(bf16(rng.bits(1) if signed else 0, 0, 1 + rng.bits(7) % 127))
        b.append(bf16(0, max(1, min(254, e + 253)), rng.bits(7)))
    return a, b


def floor_sensitive_group(rng: Rng, attempts: int = 8000, need_zero: bool = True) -> tuple[list[int], list[int], int, int] | None:
    """Eight tiny products over a zero accumulator whose group result differs between the -132 floor and no floor: the
    floor's alignment window (LSB 2^-156) truncates the terms more coarsely than a window at their own maximum, and the
    difference survives the denormalising truncation to the 2^-149 grid only near a grid boundary (about 1% of random
    groups in the ranges used).  With ``need_zero`` the floor's word must be +0 (the no-floor word then is a nonzero
    subnormal): positive terms with exponents in [-160, -150]; otherwise signed terms in [-157, -146].  Returns
    ``(a, b, floor_word, no_floor_word)`` or ``None``."""

    zeros = [0] * 8
    for _ in range(attempts):
        a, b = _tiny_group(rng, -160, -150, signed=False) if need_zero else _tiny_group(rng, -157, -146)
        with_floor = S.tc_dot(PIPELINE, 0, a + zeros, b + zeros)
        without = S.tc_dot(NO_FLOOR, 0, a + zeros, b + zeros)
        if with_floor != without and (not need_zero or with_floor == 0):
            return a, b, with_floor, without
    return None


def _gen_floor_flush(rng: Rng, i: int, seeds: Sequence[Seed]) -> Corner:
    """The -132 exponent floor.  It only acts when the accumulator is exactly zero (a nonzero subnormal accumulator carries
    exponent -126 and sets the maximum itself), and a difference of a few 2^-149 quanta cannot reach a BF16 word except
    through a rounding tie at the BF16 subnormal boundary -- so two designs:

    ``y-visible``: step 0, group 1: a floor-sensitive group (floor: +0; no floor: a nonzero subnormal); group 2: one
    product exactly 2^-134, half the smallest BF16 subnormal.  The final accumulator is 0x00008000 (tie, rounds to even:
    y = 0x0000) under the floor and 0x00008000 + r (rounds up: y = 0x0001) without it.  The remaining steps carry exactly
    cancelling pairs at exponents in [-140, -130], which keep the accumulator's low quanta inside the alignment window.

    ``acc-visible``: 48 two-step cycles: step 2m, group 1: a floor-sensitive group over a zero accumulator, group 2:
    zeros (so acc_{2m} = r is exposed); step 2m+1: +2^40 in group 1 (r is lost against it) and -2^40 in group 2 (back to
    exactly zero).  The intermediate accumulators differ between the models; the output word (+0) does not."""

    x, w = [0] * K, [0] * K
    if i & 1 == 0:
        found = floor_sensitive_group(rng)
        if found is None:
            raise AssertionError("no floor-sensitive group found")
        a, b, _, without = found
        x[0:8], w[0:8] = a, b
        x[8], w[8] = _pair_pow2(rng, -134, 0)
        for s in range(1, STEPS):
            for q in range(4):
                wa, wb = _pair_with_exponent(rng, rng.range(-140, -129), rng.bits(1))
                k = s * STEP + 2 * q
                x[k], w[k], x[k + 1], w[k + 1] = wa, wb, wa, wb ^ 0x8000
        accs = chain_accs(x, w)
        alt = chain_accs(x, w, pipeline=NO_FLOOR)
        if not (accs[-1] == 0x00008000 and epilogue(accs[-1]) == 0 and alt[-1] == 0x00008000 + without and epilogue(alt[-1]) == 1):
            raise AssertionError("floor-flush y-visible did not come out as designed")
        return Corner(x, w, ["corner", "floor-flush", "y-visible"], {"no_floor_r": without})
    for m in range(STEPS // 2):
        found = floor_sensitive_group(rng, need_zero=False)
        if found is None:
            raise AssertionError("no floor-sensitive group found")
        a, b, _, _ = found
        k = 2 * m * STEP
        x[k : k + 8], w[k : k + 8] = a, b
        k = (2 * m + 1) * STEP
        x[k], w[k] = _pair_pow2(rng, 40, 0)
        x[k + 8], w[k + 8] = _pair_pow2(rng, 40, 1)
    accs = chain_accs(x, w)
    alt = chain_accs(x, w, pipeline=NO_FLOOR)
    if accs[-1] != 0 or alt == accs:
        raise AssertionError("floor-flush acc-visible did not come out as designed")
    return Corner(x, w, ["corner", "floor-flush", "acc-visible"], {"differing_steps": sum(1 for p, q in zip(accs, alt, strict=True) if p != q)})


def _gen_subnormal(rng: Rng, i: int, seeds: Sequence[Seed]) -> Corner:
    """BF16 subnormal operands (exponent field 0, no implicit bit: value m x 2^-133) against large normals so the products
    are ordinary normal terms, mixed with normal x normal terms of similar size and 10% subnormal x subnormal (flushed).
    Odd ``i`` swaps the roles of x and w."""

    x, w = [0] * K, [0] * K
    for k in range(K):
        r = rng.below(10)
        if r < 5:
            a, b = bf16(rng.bits(1), 0, 1 + rng.bits(7) % 127), rng.bf16(150, 250)
        elif r < 9:
            a, b = rng.bf16(100, 130), rng.bf16(100, 130)
        else:
            a, b = bf16(rng.bits(1), 0, 1 + rng.bits(7) % 127), bf16(rng.bits(1), 0, 1 + rng.bits(7) % 127)
        if i & 1:
            a, b = b, a
        x[k], w[k] = a, b
    chain_accs(x, w)
    return Corner(x, w, ["corner", "subnormal-operands", "w-subnormal" if i & 1 else "x-subnormal"], {})


def _gen_signed_zero(rng: Rng, i: int, seeds: Sequence[Seed]) -> Corner:
    """+-0 operands: ``all-zero-w`` (every product is a signed zero, y = +0), ``half-zero-x`` (a real body with half of x
    zeroed by sign), ``zero-groups`` (whole groups of a real body zeroed: identity transitions), ``all-zero-x``."""

    variant = ("all-zero-w", "half-zero-x", "zero-groups", "all-zero-x")[i % 4]
    x, w, body = _body(rng, seeds)
    if variant == "all-zero-w":
        w = [rng.bits(1) << 15 for _ in range(K)]
    elif variant == "half-zero-x":
        x = [rng.bits(1) << 15 if rng.bits(1) else v for v in x]
    elif variant == "zero-groups":
        for g in range(K // 8):
            if rng.below(3) == 0:
                for k in range(g * 8, g * 8 + 8):
                    w[k] = rng.bits(1) << 15
    else:
        x = [rng.bits(1) << 15 for _ in range(K)]
    accs = chain_accs(x, w)
    if variant in ("all-zero-w", "all-zero-x") and accs[-1] != 0:
        raise AssertionError("all-zero products did not give +0")
    return Corner(x, w, ["corner", "signed-zero", variant], {"body": body})


def _gen_align_max(rng: Rng, i: int, seeds: Sequence[Seed]) -> Corner:
    """Who sets the alignment maximum.  ``acc-max``: step 0 puts the accumulator at 2^E and every later product lies 15..30
    binades below it (partially or wholly truncated against the accumulator).  ``prod-max``: a real body's accumulator
    (a few binades around 2^0) meets one product +-2^E, E in [4, 24], in group 1 of every step, with its exact negative in
    group 2 -- the accumulator and the body's products are truncated against 2^E by 4..24 binades, then 2^E leaves."""

    if i & 1 == 0:
        x, w = [0] * K, [0] * K
        E = rng.range(30, 91)
        x[0], w[0] = _pair_pow2(rng, E, 0)
        for k in range(1, K):
            x[k], w[k] = _pair_with_exponent(rng, E - rng.range(15, 31), rng.bits(1))
        chain_accs(x, w)
        return Corner(x, w, ["corner", "align-max", "acc-max"], {"E": E})
    x, w, body = _body(rng, seeds)
    E = rng.range(4, 25)
    for s in range(0, K, STEP):
        a, b = _pair_pow2(rng, E, rng.bits(1))
        x[s], w[s], x[s + 8], w[s + 8] = a, b, a ^ 0x8000, b
    chain_accs(x, w)
    return Corner(x, w, ["corner", "align-max", "prod-max"], {"body": body, "E": E})


def _gen_exp_gap(rng: Rng, i: int, seeds: Sequence[Seed]) -> Corner:
    """Exponent gaps inside a group at the 25-bit boundary: one anchor product at exponent E per group and seven others
    23..40 binades below it (a gap of 25 or more truncates a term entirely)."""

    x, w = [0] * K, [0] * K
    gaps = (23, 24, 25, 26, 27, 30, 40)
    for g in range(K // 8):
        E = rng.range(-60, 61)
        k0 = g * 8 + rng.below(8)
        for k in range(g * 8, g * 8 + 8):
            if k == k0:
                x[k], w[k] = _pair_with_exponent(rng, E, rng.bits(1))
            else:
                x[k], w[k] = _pair_with_exponent(rng, E - rng.choice(gaps), rng.bits(1))
    chain_accs(x, w)
    return Corner(x, w, ["corner", "exp-gap"], {})


def _gen_acc_monotone(rng: Rng, i: int, seeds: Sequence[Seed]) -> Corner:
    """``growth``: same-sign products whose exponent rises 1.25 binades per step (the accumulator climbs from ~2^-30 to
    ~2^95 monotonically).  ``decay``: step 0 sets +-2^100; every later step subtracts exactly half the accumulator (a power
    of two), with small same-sign texture products, so the exponent falls by one per step."""

    x, w = [0] * K, [0] * K
    sign = i & 1
    if (i // 2) & 1 == 0:
        for s in range(STEPS):
            e = -30 + (s * 5) // 4
            for k in range(s * STEP, (s + 1) * STEP):
                x[k], w[k] = _pair_with_exponent(rng, e - rng.below(3), sign)
        accs = chain_accs(x, w)
        mags = [a & 0x7FFFFFFF for a in accs]
        if any(b < a for a, b in pairwise(mags)):
            raise AssertionError("growth chain not monotone")
        return Corner(x, w, ["corner", "acc-monotone", "growth", "-" if sign else "+"], {})
    x[0], w[0] = _pair_pow2(rng, 100, sign)
    for s in range(1, STEPS):
        e = 100 - s  # acc is 2^(101-s) at the start of step s; subtract 2^(100-s)
        x[s * STEP], w[s * STEP] = _pair_pow2(rng, e, 1 - sign)
        for k in range(s * STEP + 1, (s + 1) * STEP):
            if rng.below(3) == 0:
                x[k], w[k] = _pair_with_exponent(rng, e - rng.range(26, 40), rng.bits(1))
    accs = chain_accs(x, w)
    mags = [a & 0x7FFFFFFF for a in accs]
    if any(b > a for a, b in pairwise(mags)):
        raise AssertionError("decay chain not monotone")
    return Corner(x, w, ["corner", "acc-monotone", "decay", "-" if sign else "+"], {})


def _gen_acc_swing(rng: Rng, i: int, seeds: Sequence[Seed]) -> Corner:
    """The accumulator's exponent swings: on a real body, step 2m adds +-2^E_m and step 2m+1 subtracts it, E_m following a
    triangle wave between -40 and 90 -- each excursion truncates the accumulator's low bits against 2^E_m."""

    x, w, body = _body(rng, seeds)
    period = rng.choice((16, 24, 32))
    for m in range(STEPS // 2):
        phase = (m % period) / period
        E = int(-40 + 130 * (2 * phase if phase < 0.5 else 2 - 2 * phase))
        sign = rng.bits(1)
        a, b = _pair_pow2(rng, E, sign)
        k = 2 * m * STEP + rng.below(STEP)
        x[k], w[k] = a, b
        k2 = (2 * m + 1) * STEP + rng.below(STEP)
        x[k2], w[k2] = a ^ 0x8000, b
    chain_accs(x, w)
    return Corner(x, w, ["corner", "acc-swing"], {"body": body, "period": period})


def _gen_uniform_bits(rng: Rng, i: int, seeds: Sequence[Seed]) -> Corner:
    """Uniform random finite BF16 words (random sign and mantissa, exponent field uniform in [0, 180] so the chain cannot
    leave the FP32 range: every product is below 2^108)."""

    x = [rng.bf16(0, 180) for _ in range(K)]
    w = [rng.bf16(0, 180) for _ in range(K)]
    chain_accs(x, w)
    return Corner(x, w, ["corner", "uniform-bits"], {})


#: family name -> (generator, count); 640 corners = 15.6% of the VU tier.
CORNER_FAMILIES: dict[str, tuple[Callable[[Rng, int, Sequence[Seed]], Corner], int]] = {
    "sat-final": (_gen_sat_final, 64),
    "near-sat": (_gen_near_sat, 64),
    "cancel": (_gen_cancel, 64),
    "floor-flush": (_gen_floor_flush, 64),
    "subnormal-operands": (_gen_subnormal, 64),
    "signed-zero": (_gen_signed_zero, 48),
    "align-max": (_gen_align_max, 64),
    "exp-gap": (_gen_exp_gap, 48),
    "acc-monotone": (_gen_acc_monotone, 64),
    "acc-swing": (_gen_acc_swing, 32),
    "uniform-bits": (_gen_uniform_bits, 64),
}


def generate_corner(family: str, i: int, seeds: Sequence[Seed]) -> Corner:
    """Corner ``i`` of ``family``: deterministic from ``Rng("corner", family, i)`` and the seeds; validated by ``silicon``."""

    gen, n = CORNER_FAMILIES[family]
    if not 0 <= i < n:
        raise IndexError(f"{family} has {n} corners")
    c = gen(Rng("corner", family, i), i, seeds)
    c.params["family"] = family
    c.params["i"] = i
    return c


# --- designed corner families of the transition-unit tier ------------------------------------------------------------------------------


@dataclass
class TuCorner:
    c: int
    a: list[int]
    b: list[int]
    tags: list[str]
    params: dict = field(default_factory=dict)


def _real_step(rng: Rng, seeds: Sequence[Seed]) -> tuple[list[int], list[int], str, int]:
    s = rng.choice(seeds)
    step = rng.below(STEPS)
    return list(s.x[step * STEP : (step + 1) * STEP]), list(s.w[step * STEP : (step + 1) * STEP]), s.name, step


def _tu_c_zero(rng: Rng, i: int, seeds: Sequence[Seed]) -> TuCorner:
    a, b, name, step = _real_step(rng, seeds)
    return TuCorner((i & 1) << 31, a, b, ["c-zero", "-0" if i & 1 else "+0"], {"seed": name, "step": step})


def _tu_c_near_sat(rng: Rng, i: int, seeds: Sequence[Seed]) -> TuCorner:
    sign = i & 1
    c = (sign << 31) | (rng.range(0x7F00, 0x7F80) << 16) | rng.bits(16)
    if (i // 2) & 1:
        a = [bf16(sign, 254, rng.bits(7)) for _ in range(STEP)]
        b = [bf16(0, rng.range(200, 255), rng.bits(7)) for _ in range(STEP)]
    else:
        a, b = [], []
        for _ in range(STEP):
            wa, wb = _pair_with_exponent(rng, rng.range(90, 121), rng.bits(1))
            a.append(wa)
            b.append(wb)
    out = S.tc_dot(PIPELINE, c, a, b)
    return TuCorner(c, a, b, ["c-near-sat", "overflows" if is_inf_f32(out) else "stays-finite"], {})


def _tu_c_dominant(rng: Rng, i: int, seeds: Sequence[Seed]) -> TuCorner:
    E = rng.range(-100, 101)
    c = rng.f32(E + 127, E + 127)
    lost = i & 1 == 0
    a, b = [], []
    for _ in range(STEP):
        wa, wb = _pair_with_exponent(rng, E - (rng.range(26, 61) if lost else rng.range(15, 26)), rng.bits(1))
        a.append(wa)
        b.append(wb)
    return TuCorner(c, a, b, ["c-dominant", "products-lost" if lost else "products-truncated"], {"E": E})


def _tu_c_tiny(rng: Rng, i: int, seeds: Sequence[Seed]) -> TuCorner:
    E = rng.range(-120, 41)
    c = rng.f32(E + 127, E + 127)
    lost = i & 1 == 0
    a, b = [], []
    for _ in range(STEP):
        wa, wb = _pair_with_exponent(rng, E + (rng.range(26, 81) if lost else rng.range(12, 26)), rng.bits(1))
        a.append(wa)
        b.append(wb)
    return TuCorner(c, a, b, ["c-tiny", "c-lost" if lost else "c-truncated"], {"E": E})


def _tu_align(rng: Rng, i: int, seeds: Sequence[Seed], c_is_max: bool) -> TuCorner:
    E = rng.range(-100, 101)
    c = rng.f32(E + 127, E + 127)
    top = E - 1 if c_is_max else E + 1
    a, b = [], []
    for k in range(STEP):
        e = top if k == 0 else top - rng.range(0, 12)
        wa, wb = _pair_with_exponent(rng, e, rng.bits(1))
        a.append(wa)
        b.append(wb)
    return TuCorner(c, a, b, ["c-max-align" if c_is_max else "prod-max-align"], {"E": E})


def _tu_c_max_align(rng: Rng, i: int, seeds: Sequence[Seed]) -> TuCorner:
    return _tu_align(rng, i, seeds, True)


def _tu_prod_max_align(rng: Rng, i: int, seeds: Sequence[Seed]) -> TuCorner:
    return _tu_align(rng, i, seeds, False)


def _tu_c_subnormal(rng: Rng, i: int, seeds: Sequence[Seed]) -> TuCorner:
    c = (rng.bits(1) << 31) | (1 + rng.bits(23) % 0x7FFFFF)
    a, b = [], []
    for _ in range(STEP):
        e = rng.range(-150, -125)
        wa, wb = bf16(rng.bits(1), 0, 1 + rng.bits(7) % 127), bf16(0, max(1, min(254, e + 253)), rng.bits(7))
        a.append(wa)
        b.append(wb)
    return TuCorner(c, a, b, ["c-subnormal"], {})


def _tu_zero_acc_tiny_cancel(rng: Rng, i: int, seeds: Sequence[Seed]) -> TuCorner:
    """tc_probe's ``zero_acc_tiny_cancellation`` (log entry D9): +-0 accumulators, subnormal or zero a, b in [2^-27, 2^-5]."""

    c = (i & 1) << 31
    a = [0 if rng.below(10) < 7 else bf16(rng.bits(1), 0, 1 + rng.bits(7) % 127) for _ in range(STEP)]
    b = [bf16(rng.bits(1), rng.range(100, 123), rng.bits(7)) for _ in range(STEP)]
    return TuCorner(c, a, b, ["zero-acc-tiny-cancel", "-0" if i & 1 else "+0"], {})


def _tu_sat_g1_sticky(rng: Rng, i: int, seeds: Sequence[Seed]) -> TuCorner:
    sign = i & 1
    c = rng.f32(1, 254)
    a = [bf16(sign, 254, rng.bits(7)) for _ in range(8)]
    b = [bf16(0, 254, rng.bits(7)) for _ in range(8)]
    a2 = [v ^ 0x8000 for v in a]
    return TuCorner(c, a + a2, b + b, ["sat-g1-sticky", "-" if sign else "+"], {})


def _tu_cancel_exact(rng: Rng, i: int, seeds: Sequence[Seed]) -> TuCorner:
    """``c`` is the exact negative of the first product (every BF16 x BF16 product is an exact FP32) and the rest cancel or
    are zero: the result is a zero magnitude, encoded +0 (D4).  Odd variants: c = -0 with all products +-0."""

    if i % 4 == 3:
        c = 0x80000000
        a = [rng.bits(1) << 15 for _ in range(STEP)]
        b = [rng.bf16(1, 254) for _ in range(STEP)]
        return TuCorner(c, a, b, ["cancel-exact", "neg-zero-c-zero-products"], {})
    a0, b0 = rng.bf16(70, 190), rng.bf16(70, 190)  # the product's exponent lies in [-114, 126]: a normal, exact FP32
    p = S.tc_dot(PIPELINE, 0, [a0] + [0] * 15, [b0] + [0] * 15)
    c = p ^ 0x80000000
    a, b = [a0] + [0] * 15, [b0] + [0] * 15
    if i % 4 == 1:
        for q in range(1, 8):
            wa, wb = rng.bf16(40, 220), rng.bf16(40, 220)
            a[2 * q - 1], b[2 * q - 1], a[2 * q], b[2 * q] = wa, wb, wa, wb ^ 0x8000
    return TuCorner(c, a, b, ["cancel-exact", "c-negates-p0", "with-cancelling-pairs" if i % 4 == 1 else "rest-zero"], {})


def _tu_floor_window(rng: Rng, i: int, seeds: Sequence[Seed]) -> TuCorner:
    """The -132 floor at the transition level.  Even ``i``: a zero accumulator and a floor-sensitive group 1 (the no-floor
    model gives another word), group 2 tiny products too.  Odd ``i``: a subnormal accumulator (exponent -126 sets the
    maximum: the floor is inert) with products across [-160, -128]."""

    if i & 1 == 0:
        found = floor_sensitive_group(rng, need_zero=False)
        if found is None:
            raise AssertionError("no floor-sensitive group found")
        a, b, _, _ = found
        a2, b2 = _tiny_group(rng, -160, -128)
        return TuCorner((i // 2 & 1) << 31, a + a2, b + b2, ["floor-window", "c-zero", "floor-sensitive"], {})
    c = (rng.bits(1) << 31) | (1 + rng.bits(23) % 0x7FFFFF)
    a, b = _tiny_group(rng, -160, -128)
    a2, b2 = _tiny_group(rng, -160, -128)
    return TuCorner(c, a + a2, b + b2, ["floor-window", "c-subnormal"], {})


def _tu_uniform(rng: Rng, i: int, seeds: Sequence[Seed]) -> TuCorner:
    c = rng.f32(0, 250)
    a = [rng.bf16(0, 180) for _ in range(STEP)]
    b = [rng.bf16(0, 180) for _ in range(STEP)]
    return TuCorner(c, a, b, ["uniform"], {})


TU_FAMILIES: dict[str, Callable[[Rng, int, Sequence[Seed]], TuCorner]] = {
    "c-zero": _tu_c_zero,
    "c-near-sat": _tu_c_near_sat,
    "c-dominant": _tu_c_dominant,
    "c-tiny": _tu_c_tiny,
    "c-max-align": _tu_c_max_align,
    "prod-max-align": _tu_prod_max_align,
    "c-subnormal": _tu_c_subnormal,
    "zero-acc-tiny-cancel": _tu_zero_acc_tiny_cancel,
    "sat-g1-sticky": _tu_sat_g1_sticky,
    "cancel-exact": _tu_cancel_exact,
    "floor-window": _tu_floor_window,
    "uniform": _tu_uniform,
}
TU_PER_FAMILY = 64


def generate_tu_corner(family: str, i: int, seeds: Sequence[Seed]) -> TuCorner:
    t = TU_FAMILIES[family](Rng("tu", family, i), i, seeds)
    t.tags = ["designed", *t.tags]
    t.params["family"] = family
    t.params["i"] = i
    if not is_finite_f32(t.c) or any(not is_finite_bf16(v) for v in (*t.a, *t.b)):
        raise AssertionError(f"{family}[{i}] is outside the domain")
    return t


# --- deliberately wrong models ------------------------------------------------------------------------------------------------------

SINGLE_GROUP = S.Pipeline(name="mistake:single-group-16", arch="mistake", dtype="bf16", operand_bits=16, groups=(16,), width=25,
                          zero_exponent=-132, validated="not hardware: a deliberately wrong model (one group of sixteen)")
ADDER_24 = S.Pipeline(name="mistake:adder-24-bit", arch="mistake", dtype="bf16", operand_bits=16, groups=(8, 8), width=24,
                      zero_exponent=-132, validated="not hardware: a deliberately wrong model (24-bit adder)")


def non_sticky_tc_dot(acc_bits: int, a: Sequence[int], b: Sequence[int]) -> int:
    """``silicon.tc_dot`` without the sticky saturation: a group whose result leaves the FP32 range continues into the
    next group as a finite term (as the pinned simulator does) and only the final packing saturates."""

    acc = S.fp32_term(acc_bits)
    start = 0
    for size in PIPELINE.groups:
        products = [S.bf16_product(a[k], b[k]) for k in range(start, start + size)]
        acc = S.group_sum([acc, *products], width=PIPELINE.width, zero_exponent=PIPELINE.zero_exponent)
        start += size
    return S.pack_fp32(acc)


def _chain(step: Callable[[int, Sequence[int], Sequence[int]], int], x: Sequence[int], w: Sequence[int], acc0: int = 0) -> list[int]:
    acc = acc0
    out = []
    for s in range(0, len(x), STEP):
        acc = step(acc, x[s : s + STEP], w[s : s + STEP])
        out.append(acc)
    return out


# --- exact IEEE-754 binary32 sequential fma (what a naive CPU reference computes) -----------------------------------------------------------

_SCALE = 300  # every value below is an integer multiple of 2^-300 (BF16 products reach 2^-266, FP32 subnormals 2^-149)


def _bf16_parts(word: int) -> tuple[int, int, int]:
    e = (word >> 7) & 0xFF
    if e == 0xFF:
        raise InvalidArtifact("non-finite BF16 operand")
    m = word & 0x7F
    return (word >> 15) & 1, m | (0x80 if e else 0), e or 1


def _f32_scaled(bits: int) -> tuple[int, int]:
    """``(sign, |value| x 2^SCALE)`` of a finite FP32 word."""

    e = (bits >> 23) & 0xFF
    if e == 0xFF:
        raise InvalidArtifact("non-finite FP32 accumulator")
    m = bits & 0x7FFFFF
    sig = m | ((1 << 23) if e else 0)
    return (bits >> 31) & 1, sig << (_SCALE + (e or 1) - 150)


def _round_f32(sign: int, mag: int) -> int:
    """Round-to-nearest-even of ``mag x 2^-SCALE > 0`` to a binary32 word (subnormals and overflow to infinity included)."""

    bl = mag.bit_length()
    e_msb = bl - 1 - _SCALE
    shift = bl - 24 if e_msb >= -126 else _SCALE - 149
    if shift > 0:
        q = mag >> shift
        r = mag - (q << shift)
        half = 1 << (shift - 1)
        if r > half or (r == half and q & 1):
            q += 1
    else:
        q = mag << -shift
    if e_msb >= -126:
        if q == 1 << 24:
            q >>= 1
            e_msb += 1
        if e_msb > 127:
            return (sign << 31) | F32_INF
        return (sign << 31) | ((e_msb + 127) << 23) | (q & 0x7FFFFF)
    return (sign << 31) | q  # q < 2^23 is a subnormal; q == 2^23 is the smallest normal


def ieee_fma_chain(x: Sequence[int], w: Sequence[int], acc: int = 0) -> int:
    """``acc = fma(x[k], w[k], acc)`` in IEEE-754 binary32 round-to-nearest-even, ``k`` ascending, exactly (integer
    arithmetic).  Once the accumulator is infinite it stays so (finite products cannot change it)."""

    for k in range(len(x)):
        if not is_finite_f32(acc):
            break
        sa, ma, ea = _bf16_parts(x[k])
        sb, mb, eb = _bf16_parts(w[k])
        sp, mp = sa ^ sb, (ma * mb) << (_SCALE + ea + eb - 268)
        sc, mc = _f32_scaled(acc)
        total = (-mc if sc else mc) + (-mp if sp else mp)
        if total == 0:
            acc = 0x80000000 if (mc == 0 and mp == 0 and sc and sp) else 0
        else:
            acc = _round_f32(int(total < 0), abs(total))
    return acc


@dataclass(frozen=True)
class Mistake:
    """A plausible implementation mistake, as a function of a positive instance.  ``chain(x, w, ctx)`` returns the mistaken
    accumulator words (``None`` where the mistaken model cannot be evaluated), ``word(acc, y, ctx)`` the mistaken output
    word from the correct final accumulator for epilogue-level mistakes."""

    name: str
    description: str
    level: str  # "epilogue" (a function of the correct final word) or "chain" (a different accumulation)
    chain: Callable[[Sequence[int], Sequence[int], dict], list[int] | None] | None = None
    word: Callable[[int, int, dict], int | None] | None = None


def _ulp(delta: int) -> Callable[[int, int, dict], int | None]:
    def f(acc: int, y: int, ctx: dict) -> int | None:
        if not is_finite_bf16(y):
            return None
        mag = (y & 0x7FFF) + delta
        if mag < 0 or mag >= BF16_INF:
            return None
        return (y & 0x8000) | mag

    return f


def _pipeline_chain(pipeline: S.Pipeline) -> Callable[[Sequence[int], Sequence[int], dict], list[int] | None]:
    def f(x: Sequence[int], w: Sequence[int], ctx: dict) -> list[int] | None:
        try:
            return chain_accs(x, w, pipeline=pipeline)
        except InvalidArtifact:
            return None

    return f


def _non_sticky_chain(x: Sequence[int], w: Sequence[int], ctx: dict) -> list[int] | None:
    try:
        return _chain(non_sticky_tc_dot, x, w)
    except InvalidArtifact:
        return None


def _ieee_chain(x: Sequence[int], w: Sequence[int], ctx: dict) -> list[int] | None:
    out = []
    acc = 0
    for s in range(0, len(x), STEP):
        acc = ieee_fma_chain(x[s : s + STEP], w[s : s + STEP], acc)
        out.append(acc)
    return out


def _stale_chain(x: Sequence[int], w: Sequence[int], ctx: dict) -> list[int] | None:
    c0 = ctx.get("previous_final")
    if c0 is None or not is_finite_f32(c0):
        return None
    try:
        return chain_accs(x, w, acc0=c0)
    except InvalidArtifact:
        return None


def _reversed_chain(x: Sequence[int], w: Sequence[int], ctx: dict) -> list[int] | None:
    try:
        return chain_accs(list(x)[::-1], list(w)[::-1])
    except InvalidArtifact:
        return None


MISTAKES: tuple[Mistake, ...] = (
    Mistake("ulp-plus", "the output word one BF16 ulp above the correct magnitude (finite results only)", "epilogue", word=_ulp(+1)),
    Mistake("ulp-minus", "the output word one BF16 ulp below the correct magnitude (finite, nonzero results only)", "epilogue", word=_ulp(-1)),
    Mistake("sign-flip", "the correct magnitude with the sign bit flipped (+0 becomes -0)", "epilogue", word=lambda acc, y, ctx: y ^ 0x8000),
    Mistake("epilogue-rtz", "FP32 -> BF16 by truncation (round toward zero) instead of round-to-nearest-even", "epilogue",
            word=lambda acc, y, ctx: (acc >> 16) & 0xFFFF),
    Mistake("ieee-fp32-fma", "IEEE-754 binary32 sequential fma from +0 in ascending k (a naive CPU reference), then the RNE epilogue", "chain",
            chain=_ieee_chain),
    Mistake("single-group-16", "one grouped sum of all sixteen products (Hopper's structure) instead of (8, 8)", "chain", chain=_pipeline_chain(SINGLE_GROUP)),
    Mistake("adder-24-bit", "a 24-bit truncating adder instead of 25-bit", "chain", chain=_pipeline_chain(ADDER_24)),
    Mistake("no-exponent-floor", "the alignment maximum taken over the nonzero terms alone, without the -132 floor", "chain", chain=_pipeline_chain(NO_FLOOR)),
    Mistake("non-sticky-saturation", "a group result beyond the FP32 range continues into the next group instead of holding +-inf", "chain",
            chain=_non_sticky_chain),
    Mistake("c0-stale", "the accumulator not reset between coordinates: c_0 = the previous instance's final FP32 accumulator (id - 1, cyclic)", "chain",
            chain=_stale_chain),
    Mistake("reversed-k", "the chain evaluated in descending k (the same products, the other order)", "chain", chain=_reversed_chain),
)
MISTAKE_BY_NAME = {m.name: m for m in MISTAKES}
#: Negative classes whose instances are outside the supported domain: the checker must REJECT, not compute.
REJECT_CLASSES = ("nonfinite-operand", "sat-intermediate")


def variant_chain(name: str, x: Sequence[int], w: Sequence[int], ctx: dict) -> list[int] | None:
    m = MISTAKE_BY_NAME[name]
    if m.chain is None:
        raise ValueError(f"{name} is an epilogue-level mistake")
    return m.chain(x, w, ctx)


def evaluate_mistakes(x: Sequence[int], w: Sequence[int], accs: Sequence[int], y: int, ctx: dict) -> dict[str, dict]:
    """For every mistake: ``{"y": mistaken word or None, "acc_differs": bool | None}`` on one positive instance."""

    out: dict[str, dict] = {}
    for m in MISTAKES:
        if m.level == "epilogue":
            out[m.name] = {"y": m.word(accs[-1], y, ctx), "acc_differs": None}
        else:
            alt = m.chain(x, w, ctx)
            if alt is None:
                out[m.name] = {"y": None, "acc_differs": None}
            else:
                out[m.name] = {"y": epilogue(alt[-1]), "acc_differs": list(alt) != list(accs)}
    return out


def pack_u16(words: Sequence[int]) -> bytes:
    a = array("H", words)
    if sys.byteorder == "big":
        a.byteswap()
    return a.tobytes()
