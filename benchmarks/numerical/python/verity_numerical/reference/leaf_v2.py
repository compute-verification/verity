"""Reference implementation of the proposed ``verity-vllm/leaf/v2`` algebraic leaf (auth-integration lane, 2026-09-22).

**Status: proposal.**  The live protocol is ``verity.ml.commitments.leaves`` (``leaf/v1``: SHA-256 over the whole tensor's
bytes) and is not touched.  The specification this module implements is
``verity/ml/commitments/LEAF_V2.md``; the costing that motivates it is
``backends/numerical/reports/auth_integration.md`` (option (d), "algebraic binding").  Stdlib + ``verity`` only.

What a v2 leaf is
=================

A tensor ``T`` (``canonical_id``, ``dtype``, ``shape``, little-endian words) is encoded as one or more *Ligero tableaus*
over BabyBear: every tensor row is split into ``ceil(cols / l)`` payload rows of ``l`` field elements (one 16-bit word
per element, zero padded), each payload row gets ``k - l`` uniformly random *randomizer* elements from the committer's
PRG and is Reed-Solomon encoded (degree ``< k`` polynomial interpolating the ``k`` symbols on the order-``k``
subgroup ``H_k``, evaluated on the **coset** ``s * H_n`` of the order-``n`` subgroup, ``s = 31`` a generator of
``F_p^*``, so ``s * H_n`` is disjoint from ``H_k``: the encoding is NOT systematic and no codeword symbol is a
message symbol).  Consecutive
tableau rows are grouped into *chunks* of at most ``R`` rows; a chunk is committed by a Merkle tree over its ``n``
**columns** (column leaf = ``H("verity-vllm/col/v2", chunk, column, bytes)``), and the tensor's *leaf digest* --
the 32 bytes that take the place of the v1 leaf hash in the unchanged ids/node/lift tree -- is

    H("verity-vllm/leaf/v2", canonical_id, dtype, shape_text, code_text, u32be(chunks), chunk_root_0, ...).

Everything above the leaf digest (``levels``, ``node_hash``, ``lift_hash``, ``ids_digest``, ``LeafTree``, the path
shape, the sampling protocol's ``(tree, rank, word index)`` addressing) is ``leaves.py`` verbatim and is reused here.

What the prover shows
=====================

The numerical proof commits its witness (the sampled VU's ``x`` row, ``w`` row, ``y`` word) in its *own* Ligero tableau
``U_P`` over the same code.  Binding the witness to the Verity roots is then two Ligero tests run **across** the two
tableaus on one verifier-chosen column set ``Q``:

* **IRS (proximity) test** -- ``v = b_irs + sum_i r'_i U_i`` over every tensor row and witness row involved: the
  prover sends ``v`` as a degree ``< k`` polynomial; the verifier checks it against the opened columns.  (A tensor
  committed by a dishonest operator with rows far from the code is caught here; the leaf digest is binding only
  through this test -- see LEAF_V2.md "binding".)
* **Linear (equality) test** -- one constraint per claimed word equality ``U_V[row][j] - U_P[rho][j'] = 0``; with the
  verifier's coefficient ``r_c`` per constraint the prover sends ``q = b_lin + sum_i r_i(X) p_i(X)`` (degree
  ``< 2k - 1``; ``r_i`` interpolates the per-row coefficient vector on the order-``k`` subgroup, zero on the
  randomizer positions) and the verifier checks ``sum_{zeta in k-subgroup} q(zeta) = 0`` and ``q`` against the
  opened columns.  Whole-row equalities (``x``, ``w``) and single-word ones (``y``) are the same test.

``b_irs`` / ``b_lin`` are the proof tableau's two blinding rows (a random codeword; a random degree ``< 2k - 1``
polynomial summing to zero on the subgroup): the sent polynomials are uniform subject to the checked relation, and
``t <= k - l`` opened columns of rows with fresh randomizers are uniform -- the openings reveal nothing about the
payload (AHIV22 Lemma 4.15).  **The lemma needs the opened positions to be disjoint from the message positions**:
the first version of this reference evaluated on the order-``n`` subgroup itself (systematic: codeword symbol
``(n/k) j`` *was* message symbol ``j``) and the red team read payload words straight off the openings
(``redteam_vu.md`` F3, ``redteam.leaf_v2_exhaustion``).  Since 2026-09-22 the codeword lives on the coset
``31 * H_n``, which contains no point of ``H_k``; ``test_leaf_v2.py::test_no_opened_column_is_a_payload_word``
opens every column of every chunk and checks that.  The toy draws its challenge coefficients from BabyBear itself
(soundness ~ 2^-31 per term); the campaign's protocols draw them from ``F_{p^6}``.

This module is a *reference*: readable, slow (pure Python NTTs), and tested on the real captured rows of
``fixtures/bench-instances/v1/seeds`` at reduced code parameters (``n = 4096, k = 1024, l = 768``: a 1536-word row
is two tableau rows) so the suite stays under 20 s.  Nothing here is a performance number.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from verity.ml.commitments.leaves import (
    DIGEST_BYTES,
    H,
    LeafTree,
    fold_path,
    levels,
    open_path,
    shape_text,
)
from verity.errors import InvalidArtifact

__all__ = [
    "BABYBEAR",
    "COL_TAG",
    "COSET_SHIFT",
    "LEAF_V2_TAG",
    "Chunk",
    "Claim",
    "Code",
    "ColumnOpening",
    "CommittedTensor",
    "Position",
    "ProofTableau",
    "Session",
    "Verdict",
    "commit_tensor",
    "coset_intt",
    "coset_ntt",
    "decode_row",
    "encode_row",
    "intt",
    "ntt",
    "position_of_word",
    "prove",
    "verify",
    "verify_transparent_word",
]

LEAF_V2_TAG = "verity-vllm/leaf/v2"
COL_TAG = "verity-vllm/col/v2"

BABYBEAR = 15 * 2**27 + 1
#: Plonky3's two-adic generator of order 2^27 (``31^15 mod p``; ``0x1a427a41``).
TWO_ADIC_ROOT = pow(31, 15, BABYBEAR)
TWO_ADICITY = 27
ELEMENT_BYTES = 4
#: The codeword domain is the coset ``COSET_SHIFT * H_n``.  31 generates ``F_p^*`` (order ``p - 1 = 15 * 2^27``), so it
#: lies in no proper two-adic subgroup: the coset is disjoint from every ``H_m`` (``m <= 2^27``), in particular from the
#: interpolation domain ``H_k``.  (Plonky3's ``generator()`` for BabyBear is the same 31.)
COSET_SHIFT = 31


# ---------------------------------------------------------------------------------------------------------------------
# Field and NTT
# ---------------------------------------------------------------------------------------------------------------------


def root_of_unity(order: int) -> int:
    if order & (order - 1) or order > 2**TWO_ADICITY or order < 1:
        raise ValueError(f"order must be a power of two <= 2^{TWO_ADICITY}, got {order}")
    return pow(TWO_ADIC_ROOT, 2**TWO_ADICITY // order, BABYBEAR)


def ntt(values: Sequence[int], inverse: bool = False) -> list[int]:
    """Evaluations on the order-``len`` subgroup (``w^i`` ascending) of the polynomial with the given coefficients;
    iterative radix-2, in-place bit reversal.  ``inverse`` interpolates."""
    p = BABYBEAR
    n = len(values)
    a = list(values)
    if n == 1:
        return a
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    w_n = root_of_unity(n)
    if inverse:
        w_n = pow(w_n, p - 2, p)
    length = 2
    while length <= n:
        w_len = pow(w_n, n // length, p)
        half = length >> 1
        tw = [1] * half
        for i in range(1, half):
            tw[i] = tw[i - 1] * w_len % p
        for start in range(0, n, length):
            for i in range(half):
                u = a[start + i]
                v = a[start + i + half] * tw[i] % p
                a[start + i] = u + v if u + v < p else u + v - p
                a[start + i + half] = u - v if u >= v else u - v + p
        length <<= 1
    if inverse:
        inv_n = pow(n, p - 2, p)
        a = [x * inv_n % p for x in a]
    return a


def intt(values: Sequence[int]) -> list[int]:
    return ntt(values, inverse=True)


def coset_ntt(coeffs: Sequence[int], shift: int = COSET_SHIFT) -> list[int]:
    """Evaluations of ``sum_i c_i X^i`` on the coset ``shift * H_len`` (``shift * w^i`` ascending): ``P(shift X)`` has
    coefficients ``c_i shift^i``, so scale then NTT."""
    p = BABYBEAR
    scaled = []
    si = 1
    for c in coeffs:
        scaled.append(c * si % p)
        si = si * shift % p
    return ntt(scaled)


def coset_intt(values: Sequence[int], shift: int = COSET_SHIFT) -> list[int]:
    """Coefficients of the degree ``< len`` polynomial with the given evaluations on ``shift * H_len``."""
    p = BABYBEAR
    inv = pow(shift, p - 2, p)
    d = intt(values)
    out = []
    si = 1
    for c in d:
        out.append(c * si % p)
        si = si * inv % p
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Code and layout
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Code:
    """Reed-Solomon ``[n, k]`` over BabyBear with ``l`` payload symbols per row and chunks of ``R`` rows."""

    n: int = 16384
    k: int = 4096
    l: int = 3840
    R: int = 64
    word_bits: int = 16

    def __post_init__(self) -> None:
        if not (0 < self.l < self.k < self.n) or self.n & (self.n - 1) or self.k & (self.k - 1):
            raise ValueError("need powers of two n > k > l > 0")
        if self.n % self.k:
            raise ValueError("k must divide n")
        if self.R < 1:
            raise ValueError("R >= 1")
        if 2**self.word_bits >= BABYBEAR:
            raise ValueError("a word must fit one BabyBear element")

    @property
    def shift(self) -> int:
        """The codeword domain is ``shift * H_n``, disjoint from the message domain ``H_k`` (no systematic positions)."""
        return COSET_SHIFT

    @property
    def randomizers(self) -> int:
        return self.k - self.l

    @property
    def text(self) -> str:
        return f"ligero-rs/babybear/n={self.n}/k={self.k}/l={self.l}/R={self.R}/w={self.word_bits}/coset={self.shift}"

    def domain(self) -> list[int]:
        """The ``n`` evaluation points ``shift * w_n^i``."""
        w = root_of_unity(self.n)
        out = []
        x = self.shift % BABYBEAR
        for _ in range(self.n):
            out.append(x)
            x = x * w % BABYBEAR
        return out

    def message_domain(self) -> list[int]:
        """The ``k`` interpolation points ``w_k^j`` (message symbol ``j`` is the value at ``w_k^j``)."""
        w = root_of_unity(self.k)
        out = []
        x = 1
        for _ in range(self.k):
            out.append(x)
            x = x * w % BABYBEAR
        return out

    def rows_per_tensor_row(self, cols: int) -> int:
        return -(-cols // self.l)


@dataclass(frozen=True)
class Position:
    """Where one word of a tensor lives: tableau row (global over the tensor), payload index, chunk, row in chunk."""

    tableau_row: int
    payload_index: int
    chunk: int
    row_in_chunk: int


def position_of_word(code: Code, shape: Sequence[int], word_index: int) -> Position:
    """The ``leaf/v1`` word index (row-major over the tensor) mapped to its v2 position."""
    dims = tuple(int(d) for d in shape)
    total = 1
    for d in dims:
        total *= d
    if not 0 <= word_index < total:
        raise InvalidArtifact(f"word index {word_index} outside a tensor of {total} words")
    cols = dims[-1] if dims else 1
    tensor_row, col = divmod(word_index, cols)
    per = code.rows_per_tensor_row(cols)
    sub, payload_index = divmod(col, code.l)
    tableau_row = tensor_row * per + sub
    chunk, row_in_chunk = divmod(tableau_row, code.R)
    return Position(tableau_row, payload_index, chunk, row_in_chunk)


def encode_row(code: Code, payload: Sequence[int], randomizers: Sequence[int]) -> list[int]:
    """Non-systematic RS encoding of ``payload || randomizers`` (``k`` symbols on ``H_k``) to ``n`` symbols on the coset
    ``shift * H_n``.  One INTT (interpolate) + one pointwise scaling + one NTT: the same cost as the systematic form,
    and no codeword symbol equals a message symbol (``shift * H_n`` and ``H_k`` are disjoint)."""
    if len(payload) != code.l or len(randomizers) != code.randomizers:
        raise ValueError("payload must have l symbols and randomizers k - l")
    message = list(payload) + list(randomizers)
    coeffs = intt(message)
    return coset_ntt(coeffs + [0] * (code.n - code.k), code.shift)


def decode_row(code: Code, codeword: Sequence[int]) -> list[int] | None:
    """The ``k`` message symbols of a codeword (``None`` if the ``n`` values are not a degree ``< k`` polynomial on the
    coset): what a transparent checker or the committer does; a verifier of the private form never decodes."""
    if len(codeword) != code.n:
        return None
    coeffs = coset_intt(codeword, code.shift)
    if any(coeffs[code.k:]):
        return None
    return ntt(coeffs[:code.k])


def _column_bytes(column: Sequence[int]) -> bytes:
    return b"".join(int(x).to_bytes(ELEMENT_BYTES, "little") for x in column)


def column_leaf(chunk: int, column: int, values: Sequence[int]) -> bytes:
    return H(COL_TAG, chunk.to_bytes(4, "big"), column.to_bytes(4, "big"), _column_bytes(values))


def leaf_v2_digest(canonical_id: str, dtype: str, shape: Sequence[int] | str, code: Code, chunk_roots: Sequence[bytes]) -> bytes:
    text = shape if isinstance(shape, str) else shape_text(shape)
    for root in chunk_roots:
        if len(root) != DIGEST_BYTES:
            raise InvalidArtifact("chunk root must be 32 bytes")
    return H(LEAF_V2_TAG, canonical_id.encode(), dtype.encode(), text.encode(), code.text.encode(),
             len(chunk_roots).to_bytes(4, "big"), *chunk_roots)


# ---------------------------------------------------------------------------------------------------------------------
# Committer (the vLLM side at capture time)
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Chunk:
    index: int
    codewords: list[list[int]]          # rows x n
    levels: list[list[bytes]] = field(default_factory=list)

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def column(self, c: int) -> list[int]:
        return [row[c] for row in self.codewords]

    def build(self) -> None:
        n = len(self.codewords[0])
        self.levels = levels([column_leaf(self.index, c, self.column(c)) for c in range(n)])

    def open_column(self, c: int) -> ColumnOpening:
        return ColumnOpening(self.index, c, self.column(c), open_path(self.levels, c))


@dataclass(frozen=True)
class ColumnOpening:
    chunk: int
    column: int
    values: tuple[int, ...]
    path: tuple[bytes | None, ...]

    def fold(self) -> bytes:
        return fold_path(column_leaf(self.chunk, self.column, self.values), self.column, self.path)


@dataclass
class CommittedTensor:
    canonical_id: str
    dtype: str
    shape: tuple[int, ...]
    code: Code
    words: list[int]
    payload_rows: list[list[int]]       # tableau rows (payload only)
    randomizer_rows: list[list[int]]    # the committer's secret (opening key)
    chunks: list[Chunk]

    @property
    def chunk_roots(self) -> list[bytes]:
        return [c.root for c in self.chunks]

    @property
    def digest(self) -> bytes:
        return leaf_v2_digest(self.canonical_id, self.dtype, self.shape, self.code, self.chunk_roots)

    def tableau_row(self, r: int) -> list[int]:
        chunk, i = divmod(r, self.code.R)
        return self.chunks[chunk].codewords[i]

    def position(self, word_index: int) -> Position:
        return position_of_word(self.code, self.shape, word_index)

    def descriptor(self) -> dict:
        return {"canonical_id": self.canonical_id, "dtype": self.dtype, "shape": list(self.shape), "code": self.code.text,
                "chunk_roots": [r.hex() for r in self.chunk_roots], "digest": self.digest.hex()}


def _prg_elements(seed: bytes, label: bytes, count: int) -> list[int]:
    """Deterministic randomizers from a seed: SHA-256 counter stream, rejection-sampled below p."""
    out: list[int] = []
    counter = 0
    while len(out) < count:
        block = hashlib.sha256(seed + label + counter.to_bytes(8, "little")).digest()
        counter += 1
        for i in range(0, 32, 4):
            v = int.from_bytes(block[i:i + 4], "little") & 0x7FFFFFFF
            if v < BABYBEAR:
                out.append(v)
                if len(out) == count:
                    break
    return out


def commit_tensor(canonical_id: str, dtype: str, shape: Sequence[int], words: Sequence[int], code: Code,
                  seed: bytes | None = None) -> CommittedTensor:
    """Encode and commit one tensor (little-endian words already decoded to ints)."""
    dims = tuple(int(d) for d in shape)
    total = 1
    for d in dims:
        total *= d
    if len(words) != total:
        raise InvalidArtifact(f"{canonical_id}: {len(words)} words for shape {dims}")
    if any(not 0 <= w < 2**code.word_bits for w in words):
        raise InvalidArtifact(f"{canonical_id}: a word exceeds {code.word_bits} bits")
    seed = secrets.token_bytes(32) if seed is None else seed
    cols = dims[-1] if dims else 1
    rows = total // cols if cols else 0
    per = code.rows_per_tensor_row(cols)
    payload_rows: list[list[int]] = []
    for r in range(rows):
        row = list(words[r * cols:(r + 1) * cols])
        for s in range(per):
            piece = row[s * code.l:(s + 1) * code.l]
            payload_rows.append(piece + [0] * (code.l - len(piece)))
    randomizer_rows = [_prg_elements(seed, canonical_id.encode() + b"|" + i.to_bytes(4, "big"), code.randomizers)
                       for i in range(len(payload_rows))]
    codewords = [encode_row(code, pl, rz) for pl, rz in zip(payload_rows, randomizer_rows, strict=True)]
    chunks = []
    for ci in range(0, len(codewords), code.R):
        chunk = Chunk(ci // code.R, codewords[ci:ci + code.R])
        chunk.build()
        chunks.append(chunk)
    return CommittedTensor(canonical_id, dtype, dims, code, list(words), payload_rows, randomizer_rows, chunks)


# ---------------------------------------------------------------------------------------------------------------------
# Transparent opening of one word (the reference-checker path: not zero-knowledge, like packed.py today)
# ---------------------------------------------------------------------------------------------------------------------


def verify_transparent_word(tensor_descriptor: dict, code: Code, word_index: int, payload_row: Sequence[int],
                            randomizers: Sequence[int], chunk_levels_root: bytes, column_paths: dict[int, tuple[bytes | None, ...]],
                            chunk_rows: Sequence[Sequence[int]] | None = None) -> int | None:
    """Re-encode a disclosed tableau row and check it against the chunk root through *all* n column leaves (needs
    the chunk's other rows' column values -- ``chunk_rows`` -- because a column leaf hashes the whole column).  Returns
    the word or ``None``.  This is what a transparent checker (``verity_sp1.proofs.transparent``) would do; it is
    exactly as non-private as ``leaf/v1``'s whole-leaf opening and exists for conformance, not for proofs."""
    pos = position_of_word(code, tensor_descriptor["shape"], word_index)
    cw = encode_row(code, payload_row, randomizers)
    if chunk_rows is None:
        return None
    rows = [list(r) for r in chunk_rows]
    if rows[pos.row_in_chunk] != cw:
        return None
    for c in range(code.n):
        leaf = column_leaf(pos.chunk, c, [r[c] for r in rows])
        if fold_path(leaf, c, column_paths[c]) != chunk_levels_root:
            return None
    roots = [bytes.fromhex(h) for h in tensor_descriptor["chunk_roots"]]
    if roots[pos.chunk] != chunk_levels_root:
        return None
    if leaf_v2_digest(tensor_descriptor["canonical_id"], tensor_descriptor["dtype"], tensor_descriptor["shape"], code, roots).hex() \
            != tensor_descriptor["digest"]:
        return None
    return int(payload_row[pos.payload_index])


# ---------------------------------------------------------------------------------------------------------------------
# The proof side: its own tableau, the cross-tableau tests
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class ProofTableau:
    """The numerical proof's Ligero commitment restricted to what this reference needs: the witness rows (payload +
    fresh randomizers, encoded) and the two blinding rows.  Row ``len(rows) - 2`` is ``b_irs`` (a random codeword),
    row ``len(rows) - 1`` is ``b_lin`` (random degree < 2k-1, summing to 0 on the order-k subgroup)."""

    code: Code
    payloads: list[list[int]]
    codewords: list[list[int]]
    levels: list[list[bytes]]

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    @property
    def n_witness_rows(self) -> int:
        return len(self.codewords) - 2

    def column(self, c: int) -> list[int]:
        return [row[c] for row in self.codewords]

    def open_column(self, c: int) -> ColumnOpening:
        return ColumnOpening(0, c, tuple(self.column(c)), open_path(self.levels, c))

    @classmethod
    def commit(cls, code: Code, witness_rows: Sequence[Sequence[int]], rng) -> ProofTableau:
        p = BABYBEAR
        payloads = []
        codewords = []
        for w in witness_rows:
            row = list(w) + [0] * (code.l - len(w))
            payloads.append(row)
            codewords.append(encode_row(code, row, [rng.randrange(p) for _ in range(code.randomizers)]))
        b_irs = coset_ntt([rng.randrange(p) for _ in range(code.k)] + [0] * (code.n - code.k), code.shift)
        # b_lin: degree < 2k-1 with sum over the order-k subgroup = 0.  For a polynomial sum_c a_c X^c the subgroup sum
        # is k * sum_{c = 0 mod k} a_c, so pick a_0 = -a_k and everything else free.
        coeffs = [rng.randrange(p) for _ in range(2 * code.k - 1)]
        coeffs[0] = (-coeffs[code.k]) % p
        b_lin = coset_ntt(coeffs + [0] * (code.n - len(coeffs)), code.shift)
        codewords += [b_irs, b_lin]
        lv = levels([column_leaf(0, c, [row[c] for row in codewords]) for c in range(code.n)])
        return cls(code, payloads, codewords, lv)


@dataclass(frozen=True)
class Claim:
    """``U_V[tensor][tableau_row][payload positions] == U_P[proof_row][proof positions]`` (same length)."""

    tensor: str
    tableau_row: int
    tensor_positions: tuple[int, ...]
    proof_row: int
    proof_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.tensor_positions) != len(self.proof_positions):
            raise InvalidArtifact("a claim pairs equally many positions")


@dataclass
class Session:
    """One interactive run: the verifier's coins and the prover's messages, as a plain record."""

    code: Code
    claims: list[Claim]
    Q: list[int]
    irs_coeffs: dict[tuple[str, int], int]      # (tensor id or "" for the proof, row) -> r'
    lin_coeffs: list[int]                        # one per constraint (claim x position)
    v_coeffs: list[int] = field(default_factory=list)          # degree < k
    q_coeffs: list[int] = field(default_factory=list)          # degree < 2k - 1
    tensor_openings: dict[str, dict[int, list[ColumnOpening]]] = field(default_factory=dict)   # tensor -> chunk -> per Q
    proof_openings: list[ColumnOpening] = field(default_factory=list)

    @property
    def constraints(self) -> list[tuple[tuple[str, int], int, tuple[str, int], int]]:
        out = []
        for cl in self.claims:
            for a, b in zip(cl.tensor_positions, cl.proof_positions, strict=True):
                out.append(((cl.tensor, cl.tableau_row), a, ("", cl.proof_row), b))
        return out


def _rows_involved(claims: Sequence[Claim]) -> list[tuple[str, int]]:
    seen: dict[tuple[str, int], None] = {}
    for cl in claims:
        seen.setdefault((cl.tensor, cl.tableau_row), None)
        seen.setdefault(("", cl.proof_row), None)
    return list(seen)


def verifier_coins(code: Code, claims: Sequence[Claim], t: int, rng) -> Session:
    """Step 0 material: ``Q`` (``t`` distinct columns), the IRS coefficients, one coefficient per equality constraint.
    In the campaign's protocols these are committed (HM96) before the prover's first message."""
    if t > code.randomizers:
        raise InvalidArtifact(f"t = {t} opened columns exceed the k - l = {code.randomizers} randomizers: not hiding")
    Q = sorted(rng.sample(range(code.n), t))
    irs = {key: rng.randrange(1, BABYBEAR) for key in _rows_involved(claims)}
    n_constraints = sum(len(cl.tensor_positions) for cl in claims)
    lin = [rng.randrange(1, BABYBEAR) for _ in range(n_constraints)]
    return Session(code, list(claims), Q, irs, lin)


def _coefficient_rows(session: Session) -> dict[tuple[str, int], list[int]]:
    """Per involved row the length-``k`` coefficient vector ``a_i`` (payload positions; zero on randomizers)."""
    code = session.code
    a: dict[tuple[str, int], list[int]] = {key: [0] * code.k for key in _rows_involved(session.claims)}
    for (row_v, jv, row_p, jp), r in zip(session.constraints, session.lin_coeffs, strict=True):
        a[row_v][jv] = (a[row_v][jv] + r) % BABYBEAR
        a[row_p][jp] = (a[row_p][jp] - r) % BABYBEAR
    return a


def prove(session: Session, tensors: dict[str, CommittedTensor], proof: ProofTableau) -> Session:
    """The prover's messages: ``v`` (IRS combination, degree < k), ``q`` (linear test, degree < 2k-1), and the column
    openings of every touched chunk and of the proof tableau at ``Q``."""
    p = BABYBEAR
    code = session.code
    n, k = code.n, code.k

    def codeword(key: tuple[str, int]) -> list[int]:
        return proof.codewords[key[1]] if key[0] == "" else tensors[key[0]].tableau_row(key[1])

    # IRS: v = b_irs + sum r'_i U_i  (all codewords of degree < k: combine the coset evaluations, interpolate on the
    # coset; an honest v has coefficients k.. = 0 -- there is no systematic quarter to read the message off any more)
    v_eval = list(proof.codewords[-2])
    for key, r in session.irs_coeffs.items():
        cw = codeword(key)
        v_eval = [(x + r * y) % p for x, y in zip(v_eval, cw, strict=True)]
    session.v_coeffs = coset_intt(v_eval, code.shift)[:k]
    # linear: q = b_lin + sum_i r_i(X) p_i(X), evaluated on the coset then interpolated (degree < 2k-1 <= n)
    a = _coefficient_rows(session)
    q_eval = list(proof.codewords[-1])
    for key, a_i in a.items():
        r_i = coset_ntt(intt(a_i) + [0] * (n - k), code.shift)
        cw = codeword(key)
        q_eval = [(x + ri * y) % p for x, ri, y in zip(q_eval, r_i, cw, strict=True)]
    # an honest run has coeffs[2k-1:] == 0; a prover holding a non-codeword row can only truncate (and is then caught
    # by the IRS test or at Q), so the truncation is silent here, as a cheater's would be
    session.q_coeffs = coset_intt(q_eval, code.shift)[:2 * k - 1]
    # openings
    session.tensor_openings = {}
    for tid, tensor in tensors.items():
        chunks = sorted({cl.tableau_row // code.R for cl in session.claims if cl.tensor == tid})
        session.tensor_openings[tid] = {ci: [tensor.chunks[ci].open_column(c) for c in session.Q] for ci in chunks}
    session.proof_openings = [proof.open_column(c) for c in session.Q]
    return session


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = ""


def verify(session: Session, descriptors: dict[str, dict], tree: LeafTree, tree_openings: dict[str, tuple[bytes | None, ...]],
           proof_root: bytes, n_proof_rows: int) -> Verdict:
    """The verifier: leaf digests fold to the Verity root; every opened column folds to its chunk root; ``v`` and
    ``q`` are consistent with the openings at ``Q``; ``q`` sums to zero on the order-``k`` subgroup."""
    p = BABYBEAR
    code = session.code
    n, k = code.n, code.k
    try:
        # 1. tensor digests under the (unchanged) ids tree
        for tid, desc in descriptors.items():
            roots = [bytes.fromhex(h) for h in desc["chunk_roots"]]
            digest = leaf_v2_digest(desc["canonical_id"], desc["dtype"], desc["shape"], code, roots)
            if digest.hex() != desc["digest"] or desc["code"] != code.text:
                return Verdict(False, f"{tid}: descriptor digest mismatch")
            if fold_path(digest, tree.index_of(tid), tree_openings[tid]) != tree.root:
                return Verdict(False, f"{tid}: leaf digest does not fold to the Verity root")
        # 2. column openings
        for tid, by_chunk in session.tensor_openings.items():
            roots = [bytes.fromhex(h) for h in descriptors[tid]["chunk_roots"]]
            for ci, ops in by_chunk.items():
                if [o.column for o in ops] != session.Q:
                    return Verdict(False, f"{tid}/chunk {ci}: openings not at Q")
                for o in ops:
                    if o.chunk != ci or o.fold() != roots[ci]:
                        return Verdict(False, f"{tid}/chunk {ci}/col {o.column}: column does not fold to the chunk root")
        if [o.column for o in session.proof_openings] != session.Q:
            return Verdict(False, "proof openings not at Q")
        for o in session.proof_openings:
            if len(o.values) != n_proof_rows or o.fold() != proof_root:
                return Verdict(False, f"proof col {o.column}: does not fold to the proof root")
        if len(session.v_coeffs) != k or len(session.q_coeffs) != 2 * k - 1:
            return Verdict(False, "polynomial degree bounds violated")

        def opened(key: tuple[str, int], qi: int) -> int:
            if key[0] == "":
                return session.proof_openings[qi].values[key[1]]
            pos_chunk, row_in_chunk = divmod(key[1], code.R)
            return session.tensor_openings[key[0]][pos_chunk][qi].values[row_in_chunk]

        # 3. IRS test at Q (v evaluated on the coset: the opened columns are coset points)
        v_full = coset_ntt(session.v_coeffs + [0] * (n - k), code.shift)
        for qi, c in enumerate(session.Q):
            lhs = session.proof_openings[qi].values[n_proof_rows - 2]
            for key, r in session.irs_coeffs.items():
                lhs = (lhs + r * opened(key, qi)) % p
            if lhs != v_full[c]:
                return Verdict(False, f"IRS test fails at column {c}")
        # 4. linear test: q at Q and the subgroup sum
        q_full = coset_ntt(session.q_coeffs + [0] * (n - (2 * k - 1)), code.shift)
        a = _coefficient_rows(session)
        r_at_q = {key: None for key in a}
        for key, a_i in a.items():
            r_at_q[key] = coset_ntt(intt(a_i) + [0] * (n - k), code.shift)
        for qi, c in enumerate(session.Q):
            lhs = session.proof_openings[qi].values[n_proof_rows - 1]
            for key in a:
                lhs = (lhs + r_at_q[key][c] * opened(key, qi)) % p
            if lhs != q_full[c]:
                return Verdict(False, f"linear test fails at column {c}")
        # sum over the order-k subgroup of a polynomial sum_c a_c X^c is k * sum_{c = 0 mod k} a_c
        total = sum(session.q_coeffs[c] for c in range(0, 2 * k - 1, k)) * k % p
        if total != 0:
            return Verdict(False, "equality constraints not satisfied (q does not sum to zero)")
        return Verdict(True)
    except (KeyError, IndexError, ValueError, TypeError, InvalidArtifact) as error:
        return Verdict(False, f"malformed: {error}")


def tensor_tree(tensors: Iterable[CommittedTensor], tree_id: str = "") -> LeafTree:
    """The ids tree over v2 leaf digests: ``leaves.LeafTree`` unchanged."""
    return LeafTree({t.canonical_id: t.digest for t in tensors}, tree_id)
