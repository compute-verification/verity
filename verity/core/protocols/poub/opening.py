"""Check 3 under zero knowledge: each receipt the remote session draws goes through the remote bandwidth audit's opening
unit (`verity.ml.commitments.gates.block_check`, PROTOCOL.md §8's checks 1-3 and 5 as one unit) in place of its opening,
so the verifier holds the drawn block, its payload, its receipt leaf, the step's plan row and every salt and path only
inside the unit, and reads the unit's verdict.

For a draw (pass i of window t at step s, position p, block j = plan_i[p]) the prover stages one `BlockOpeningCheck`:
the receipt leaf on its path to the pass's receipt root and the plan row holding j at p, the block decoding to its
payload under b_j(c_a) (under the seed's key for a registered block), the payload on its path to its append's root (the
registration's payload root for a registered block) and, appended, to the canonical payload commitment at row j - NR (the
row the memory-map unit binds block j's ``pay`` row to, `requirement_units.map_value_leaves`), and the block's sketch
receipt the leaf's entry p mod 16. The verifier opens only the unit's public rows, which it derives itself (`public`: the
codec seed, j, the append's coin, the pass's chaining value, sign rows and p, the three roots with their domain ids, ranks
and depths), and its verdict row; it holds the one witness row it doesn't open, the plan row, to the canonical plan row
of step s by ``b || c`` (`plan_link`). No block, payload, receipt leaf or plan is opened.

`Session` is `audit.Session` with check 3's outcome in place of `verify`'s openings (`drawn`, `decide`), and `Store` is
`audit.Store` keeping the payloads it appends (the prover's witness, `witness`). `check` decides one draw by the unit's
reference on the prover's arguments and the plan link, and `reference` a session's windows by it: the stand-in for the
units' proofs where a session isn't proved."""
from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

from verity.ml.commitments import Commitment, CommitmentDomain, hm96
from verity.ml.commitments.gates import block_check as BC
from verity.ml.commitments.merkle import merkle_depth, tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2, row_leaf, sha512_row_v2_digest

from . import audit, codec, receipts
from .receipts import PER_LEAF, Plan

ROWS = receipts.SKETCH_ROWS


def statics(c: codec.Codec, leaf: int, P: int, DR: int, DP: int, DV: int) -> SimpleNamespace:
    """The unit's statics at codec ``c`` for a receipt leaf of ``leaf`` receipts, a plan row of P entries and trees of at
    most depths DR (receipts), DP (payloads) and DV (values)."""
    return SimpleNamespace(N=c.field.n, C=c.field.c, K=c.rounds, MASKED=int(c.masked), BYTES=codec.BLOCK_BYTES, ROWS=ROWS,
                           LEAF=leaf, P=P, DR=DR, DP=DP, DV=DV)


def depth(n: int) -> int:
    """A unit's D for trees of at most n leaves: their depth, at least 1."""
    return max(1, merkle_depth(n))


@dataclass(frozen=True)
class Values:
    """The value commitment an appended block's payload is a row of: the canonical payload commitment, its root and domain
    (the verifier's, from the registration), its rows' published ``b || c`` in order, and the store index of row 0."""

    root: bytes
    domain: CommitmentDomain
    base: int
    bcs: tuple[bytes, ...]

    def path(self, row: int) -> tuple[bytes, ...]:
        leaves = [row_leaf(self.domain, i, receipts.HM96.tree_leaf(receipts.KEY, bc), V2) for i, bc in enumerate(self.bcs)]
        out = []
        for level in tree_levels(self.domain, leaves)[:-1]:
            out.append(level[row ^ 1])
            row >>= 1
        return tuple(out)


@dataclass(frozen=True)
class Draw:
    """What the verifier knows of one draw: window t, pass i at step s, position p, block j = plan_i[p], the pass's seed,
    plan and root, and the commitment block j's payload is a leaf of: the registration's payload tree (``coin`` None) or
    its append's, under the append's coin."""

    window: int
    index: int
    position: int
    block: int
    step: int
    seed: bytes
    plan: Plan
    root: bytes
    domain: CommitmentDomain
    commitment: Commitment
    coin: bytes | None

    @property
    def leaf(self) -> int:
        return self.position // PER_LEAF

    @property
    def leaf_size(self) -> int:
        return min(PER_LEAF, self.plan.length - PER_LEAF * self.leaf)


class Session(audit.Session):
    """`audit.Session` with check 3 decided by proofs: `drawn` is what the verifier knows of window t's draws, and
    `decide` takes check 3's outcome on each in place of `verify`'s openings."""

    def drawn(self, t: int) -> tuple[Draw, ...]:
        w = self.windows[t] if 0 <= t < len(self.windows) else None
        if w is None or w.draws is None:
            raise audit.SessionError(f"window {t} has no draws")
        N, out = self.params.blocks, []
        for i, p in w.draws:
            plan = w.plans[i]
            j = plan.block(p)
            if j < N:
                domain, commitment, coin = self._payload_domain, Commitment(self.registration.payload_root, N), None
            else:
                _f, domain, commitment, coin = self._appends[bisect.bisect_right(self._appends, j, key=lambda a: a[0]) - 1]
            out.append(Draw(t, i, p, j, w.steps[i], receipts.seed(w.challenge, i, self._previous(t, i)), plan, w.roots[i],
                            domain, commitment, coin))
        return tuple(out)

    def decide(self, t: int, failures: Sequence[str | None]) -> audit.Verdict:
        """Window t's verdict from check 3's outcome on each of its draws, in the order drawn: the first failure (one of
        `audit.FAILURES`) rejects it, and a window with no draws is `malformed`, as `verify` has it."""
        self._live()
        w = self.windows[t] if 0 <= t < len(self.windows) else None
        if w is None or w.draws is None or w.verdict is not None:
            raise audit.SessionError(f"window {t} has no draws awaiting check 3")
        if not w.draws or len(failures) != len(w.draws):
            return self._fail(t, "malformed")
        for at, f in zip(w.draws, failures):
            if f is not None:
                return self._fail(t, f if f in audit.FAILURES else "malformed", at)
        w.verdict = audit.Verdict(t, True)
        return w.verdict


class Store(audit.Store):
    """`audit.Store` keeping each appended block's payload, which check 3's unit takes as its witness."""

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self.kept: dict[int, bytes] = {}

    def append(self, payloads: Sequence[bytes], salts: Sequence[bytes]) -> audit.Append:
        a = super().append(payloads, salts)
        self.kept.update((a.first + u, m) for u, m in enumerate(payloads))
        return a

    def payload(self, j: int) -> bytes:
        return self.payloads[j] if j < self.registration.blocks else self.kept[j]


def public(d: Draw, codec_seed: bytes, S, values: Values) -> dict[str, list[int]]:
    """The verifier's own copy of draw ``d``'s public rows (`BC.opening_public`): block j's codec key, bound under its
    append's coin if appended; the pass's chaining value, sign rows and p; the pass's receipt root under its seed and plan
    (`receipts.domain`); the payload's tree, at rank j - first; and, appended, the canonical payload commitment at row
    j - base."""
    rdom = receipts.domain(d.seed, d.plan)
    first = d.domain.positions.start
    vals = None if d.coin is None else (values.root, values.domain.domain_id, d.block - values.base, merkle_depth(len(values.bcs)))
    return BC.opening_public(S, codec_seed, d.block, d.coin, receipts.sketch_prefix(d.seed), receipts.sign_bits(d.seed, ROWS),
                             d.position, (d.root, rdom.domain_id, merkle_depth(rdom.count)),
                             (d.commitment.root, d.domain.domain_id, d.block - first, merkle_depth(d.domain.count)), vals)


def witness(d: Draw, ps: audit.Pass, store: Store, plan_row: Sequence[int], S, values: Values) -> dict[str, list[int]]:
    """The honest prover's arguments for draw ``d`` of pass ``ps``: its block and payload from its store, the receipt leaf
    under its salt and path in the pass's tree, the payload under its salt and path in its append's tree (or the
    registration's), the step's certified plan row, and, appended, the payload's path in the canonical payload commitment."""
    j, c = d.block, d.leaf
    pay_salt, pay_path, _d = store.committed(j)
    leaf = b"".join(ps.receipts[PER_LEAF * c:PER_LEAF * (c + 1)])
    vpath = values.path(j - values.base) if d.coin is not None else ()
    return BC.opening_args(S, public(d, store.registration.codec_seed, S, values), store.blocks[j], store.payload(j), leaf,
                           list(plan_row), ps.salts[c], pay_salt, tuple(ps.tree.open(c).path), tuple(pay_path), vpath)


def commit(bits: int, words: Sequence[int], salt: bytes) -> bytes:
    """A bit row's ``b || c`` under ``salt``: hm96-sha512 over its sha512/row/v2 digest, the row of ``bits`` bits held in
    ``words`` (u16, low bit first), as the proof system commits a unit's witness row."""
    if len(words) != -(-bits // 16):
        raise ValueError(f"a row of {bits} bits is {-(-bits // 16)} u16 words, not {len(words)}")
    row = b"".join(int(w).to_bytes(2, "little") for w in words)[:-(-bits // 8)]
    return hm96.SHA512.commit_string(hm96.SHA512.default_key, sha512_row_v2_digest(row, bits), salt)


def plan_link(plan_bc: bytes, certified_bc: bytes) -> str | None:
    """The plan row's ``b || c`` against the canonical plan row of the pass's step: the unit holds plan[p] = j of the row it
    was given, and this holds that row to the certified one. Returns "plan" or None."""
    return None if plan_bc == certified_bc else "plan"


def check(d: Draw, args: dict, codec_seed: bytes, S, values: Values, plan_salt: bytes, certified_bc: bytes) -> str | None:
    """Check 3 on one draw by the unit's reference (`BC.opening_check`) on the prover's arguments, its public rows the
    verifier's own, then `plan_link` on the plan row's ``b || c`` under ``plan_salt``: None, or the failure (one of
    `audit.FAILURES`)."""
    if any(args[k] != v for k, v in public(d, codec_seed, S, values).items()):
        return "malformed"
    v = BC.verdict(BC.opening_check(S, args))
    if v is not None:
        return v
    return plan_link(commit(16 * S.P, args["plan"], plan_salt), certified_bc)


def reference(session: Session, store: Store, passes: dict, c3) -> list[audit.Verdict]:
    """Every window of a session whose draws are drawn, decided by `check` on the honest prover's witness: the stand-in
    for the units' proofs. Stops at the first rejected window. ``c3`` holds the codec, the plan row's P and the depths DR,
    DP and DV, each step's plan row (``plan_rows``), its salt (``plan_salts``) and certified ``b || c`` (``plan_bcs``), and
    the value commitment (``values``)."""
    out = []
    for t in range(len(session.windows)):
        fails = []
        for d in session.drawn(t):
            S = statics(c3.codec, d.leaf_size, c3.P, c3.DR, c3.DP, c3.DV)
            a = witness(d, passes[d.window, d.index], store, c3.plan_rows[d.step], S, c3.values)
            fails.append(check(d, a, store.registration.codec_seed, S, c3.values, c3.plan_salts[d.step], c3.plan_bcs[d.step]))
        out.append(session.decide(t, fails))
        if not out[-1].accepted:
            break
    return out
