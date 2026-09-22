"""CK-SELECT: the token-selection checker -- ``TokenSelect_v1|whole-scan`` (one VU per decode step, 18-bit index unit).

**The prescribed function** (pinned, never chosen here).  ``verity_vllm/registry/b1.py:471-476`` ``TokenSelect``::

    init = (logits[0], Const32[0], Const32[1])          # (best, best_index, i)
    step = scan(ArgmaxStep, init, xs=(logits[1:V],))
    return step[0][1]                                    # best_index (I32)

``verity_vllm/registry/b1.py:458-468`` ``ArgmaxStep(carry=(best, best_i, i), x)``::

    gt  = Bf16GtStrict(x, best)        # prims.py:538-542: a > b on bf16 widened to f32; "NaN compares false"
    nb  = SelectBf16(gt, x, best)      # prims.py:528-530: c ? a : b
    ni  = SelectI32(gt, i, best_i)     # prims.py:533-535: c ? a : b
    nxt = I32Add(i, Const32[1])        # prims.py:511-514: wrapping 32-bit add
    return ((nb, ni, nxt), ni)

Exact semantics (a consequence of the text above, stated, not selected):

* **equality** -- ``x == best`` is NOT greater: the earlier index is kept (``Bf16GtStrict`` is strict).  ``+0.0`` and
  ``-0.0`` compare equal (f32 ``>``).  Hence *lowest index wins ties*.
* **NaN** -- ``x > best`` is false when EITHER is NaN.  A NaN candidate at ``i >= 1`` never replaces the carry.  A NaN at
  index 0 is the initial carry and nothing ever exceeds it: the result is 0.  (So "NaN never wins" holds for ``i >= 1``
  only; index 0 wins by initialisation.  This differs from ``torch.argmax`` -- the docstring's phrase -- which propagates
  NaN; the gate-level text is what is proved.)
* **-inf / all-equal** -- ``-inf > -inf`` is false; an all-equal row (any value, ``-inf`` included) selects 0; a row of
  ``-inf`` with one finite entry selects that entry.
* **the index** -- the I32 carry ``i`` starts at the literal ``Const32[1]`` and steps by the literal ``Const32[1]`` each
  step, so ``best_index`` is the vocabulary position of the winning word; the declared unit is the 18-bit index
  (``population_v3.TOKEN_SELECT_INDEX_BITS``) in the I32 container.

**The Const32 literals are compiled into this consumer** (E5 decision, ``out/eod/e5/const32-consequence.md``): the checker
reads the three literals (initial ``best_index``, initial ``i``, the per-step increment) FROM THE PROGRAM TEXT
(:func:`pin_function` walks the located ``TokenSelect`` call gate by gate) and puts them in the statement; the guest applies
the literals it is given; the verifier's expectation (:func:`expectation`) is derived from the program alone, so a
statement whose literal differs from the program's is refused BY NAME (``literal init_i: statement 2 != program 1``,
:func:`check_expected`).  No ``Const32`` VU remains in the population for this consumer.

**What is proved** (family ``veritor.select-argmax@1``, guest ``zk/sp1/programs/select/``): given the committed RU tree
root, the rank + canonical id of the step's ``lm_head.gemm/port=0`` logits row (bf16 ``[V]``) and of its
``sampler.select/port=0`` word (int32 scalar), the guest authenticates both leaves under the serving side's framing
(``verity-vllm/leaf/v1``; ``veritor_zk_common::packed`` -- the SAME hashing as the approved guest), evaluates the
prescribed function over the row with the statement's literals, and accepts iff the result equals the committed int32 and
is below ``2^index_bits``.  Public values: ``sha256(statement) || verdict`` (33 bytes, the approved guest's layout).

**Host mirror**: :func:`verify_transparent` is the guest's Python mirror, byte for byte on the statement/witness codecs and
bit for bit on the relation (pure integer arithmetic, no floats).  :func:`argmax_oracle` is the registry's own evaluator
(``verity.ir.evaluate``) over a ``TokenSelect_v1{V}`` program -- the independent reference the vectors are generated from.

The record (prover side): a ``ck_elem.RecordHandle`` (pinned program, ``ProgramLocator``, ``commitments.json``, capture
reader).  The verifier side derives the expectation from the program + the committed id lists only (no reader).
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO), str(_REPO / "vllm-poc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = [
    "FAMILY",
    "KIND",
    "STEP_KIND",
    "STATEMENT_MAGIC",
    "INDEX_BITS",
    "LOGITS_DTYPE",
    "SELECT_DTYPE",
    "Literals",
    "PinnedFunction",
    "SelectStatement",
    "SelectWitness",
    "ResolvedSelect",
    "Built",
    "Refusal",
    "REJECT_CODES",
    "bf16_gt_strict",
    "argmax_pinned",
    "argmax_oracle",
    "pin_function",
    "resolve",
    "expectation",
    "check_expected",
    "build",
    "verify_transparent",
    "public_values",
    "leaf_hash",
    "fold_path",
]

FAMILY = "veritor.select-argmax@1"
KIND = "TokenSelect_v1|whole-scan"  # population_v3.TOKEN_SELECT_KIND (Amendment A2 decision 4)
STEP_KIND = "ArgmaxStep_v1|scan"  # a located scan step resolves to its enclosing whole-scan unit (regrouped)
STATEMENT_MAGIC = b"VTSELCT1"
INDEX_BITS = 18  # population_v3.TOKEN_SELECT_INDEX_BITS
LOGITS_DTYPE = "bfloat16"
SELECT_DTYPE = "int32"
LEAF_TAG = "verity-vllm/leaf/v1"
NODE_TAG = "verity-vllm/node/v1"
LIFT_TAG = "verity-vllm/lift/v1"
PUBLIC_VALUES_LEN = 33
VERDICT_ACCEPT, VERDICT_REJECT = 1, 0
MAX_V = 1 << 24
MAX_ID = 1 << 16

#: The guest's rejection codes (``zk/sp1/programs/select/core/src/lib.rs`` ``Reject``), in the order the guest checks them.
REJECT_CODES = (
    "STATEMENT_MALFORMED",  # magic / lengths / V < 2 / index_bits not in 1..=32 / V > 2^24 / leaf_count 0 / rank >= leaf_count
    "WITNESS_MALFORMED",  # witness does not parse to exactly (logits bytes, path, select bytes, path)
    "LOGITS_LENGTH",  # logits leaf is not 2*V bytes
    "SELECT_LENGTH",  # select leaf is not 4 bytes
    "LOGITS_AUTH",  # logits leaf hash + path do not fold to the statement root at logits_rank
    "SELECT_AUTH",  # select leaf hash + path do not fold to the statement root at select_rank
    "INDEX_RANGE",  # the prescribed function's result is not below 2^index_bits
    "SELECT_MISMATCH",  # the prescribed function's result != the committed int32
)


class Refusal(ValueError):
    """The builder / expectation refuses: the located unit is not a whole-scan TokenSelect over committed words, or the
    program text at the located call is not the registered ``TokenSelect`` shape.  Never a statement over invented words."""


# --- the pinned semantics, bit for bit ---------------------------------------------------------------------------------


def bf16_gt_strict(a: int, b: int) -> bool:
    """``Bf16GtStrict`` on two 16-bit words in pure integer arithmetic (what the guest computes): false when either is NaN
    (exponent all ones and mantissa nonzero), else signed-magnitude order; ``+0 == -0``."""

    ma, mb = a & 0x7FFF, b & 0x7FFF
    if ma > 0x7F80 or mb > 0x7F80:
        return False
    ka = -ma if a & 0x8000 else ma
    kb = -mb if b & 0x8000 else mb
    return ka > kb


@dataclass(frozen=True, slots=True)
class Literals:
    """The three ``Const32`` literals of the consumer, read from the program text."""

    init_index: int  # TokenSelect: Const32[0] -- initial best_index
    init_i: int  # TokenSelect: Const32[1] -- initial i
    step: int  # ArgmaxStep: Const32[1] -- the increment of i (one literal per step; all must agree)

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.init_index, self.init_i, self.step)


REGISTRY_LITERALS = Literals(0, 1, 1)


def argmax_pinned(words: Sequence[int], lits: Literals = REGISTRY_LITERALS) -> int:
    """The prescribed function over bf16 words with the given literals -- the guest's relation, bit for bit."""

    if len(words) < 2:
        raise ValueError("TokenSelect needs V >= 2")
    best, best_index, i = int(words[0]), lits.init_index & 0xFFFFFFFF, lits.init_i & 0xFFFFFFFF
    for x in words[1:]:
        x = int(x)
        if bf16_gt_strict(x, best):
            best, best_index = x, i
        i = (i + lits.step) & 0xFFFFFFFF
    return best_index


def argmax_oracle(words: Sequence[int]) -> int:
    """The registry's own evaluator over a ``TokenSelect_v1{V=len(words)}`` program (``verity.ir.evaluate``): the reference
    :func:`argmax_pinned` and the guest are checked against.  Slow (V gates * 5); for vectors and tests."""

    from verity.ir import Program, bind
    from verity.ir.evaluate import evaluate_region
    from verity_vllm.registry import b1

    V = len(words)
    p = Program(bind(b1.TokenSelect, V=V))
    c = p.circuit
    t = {g: int(w) for g, w in zip(range(V), words, strict=True)}
    evaluate_region(c, c.region(0, c.size), t)
    (out,) = c.root_outputs()
    return int(t[out])


# --- the program text at the located call -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PinnedFunction:
    """What the program says at ``[lo, hi)``: a ``TokenSelect`` over ``V`` logits gates with these literals, returning ``out``."""

    lo: int
    hi: int
    V: int
    literals: Literals
    logits: tuple[int, ...]  # the V operand gates, vocabulary order
    out: int  # the best_index gate (the call's one return)

    @property
    def steps(self) -> int:
        return self.V - 1


def _prim_name(gate: Any) -> str:
    return str(getattr(gate.prim, "name", None) or gate.prim.id)


def _const_value(gate: Any, where: str) -> int:
    pid = str(gate.prim.id)
    if not pid.startswith("Const32["):
        raise Refusal(f"{where}: gate {gate.index} is {pid}, not a Const32 literal")
    if gate.operands:
        raise Refusal(f"{where}: Const32 gate {gate.index} has operands")
    return int(gate.prim.evaluate()) & 0xFFFFFFFF


def pin_function(circuit: Any, lo: int, hi: int, args: Sequence[int], returns: Sequence[int]) -> PinnedFunction:
    """Walk the gates ``[lo, hi)`` and refuse unless they are exactly the registered ``TokenSelect`` body over ``args``:
    two init literals, then ``V-1`` ``ArgmaxStep`` bodies of five gates each wired as ``b1.py:458-468`` wires them, the
    call returning the last step's ``ni``.  The literals are READ here (never assumed)."""

    V = len(args)
    if V < 2:
        raise Refusal(f"TokenSelect call [{lo},{hi}) has {V} logits operands (needs >= 2)")
    if hi - lo != 2 + 5 * (V - 1):
        raise Refusal(f"TokenSelect call [{lo},{hi}) has {hi - lo} gates, the registered body over V={V} has {2 + 5 * (V - 1)}")
    if len(returns) != 1:
        raise Refusal(f"TokenSelect call [{lo},{hi}) returns {len(returns)} gates, the registered body returns one (best_index)")
    g = circuit.gate
    init_index = _const_value(g(lo), "init best_index")
    init_i = _const_value(g(lo + 1), "init i")
    step: int | None = None
    best, best_i, i = int(args[0]), lo, lo + 1
    for k in range(V - 1):
        base = lo + 2 + 5 * k
        x = int(args[k + 1])
        where = f"ArgmaxStep#{k} (gates [{base},{base + 5}))"
        gt, nb, ni, c, nxt = (g(base + j) for j in range(5))
        exp = (
            ("Bf16GtStrict", gt, (x, best)),
            ("SelectBf16", nb, (base, x, best)),
            ("SelectI32", ni, (base, i, best_i)),
            ("I32Add", nxt, (i, base + 3)),
        )
        for name, gate, ops in exp:
            if _prim_name(gate) != name:
                raise Refusal(f"{where}: gate {gate.index} is {gate.prim.id}, the registered body has {name}")
            if tuple(int(o) for o in gate.operands) != ops:
                raise Refusal(f"{where}: {name}@{gate.index} reads {tuple(gate.operands)}, the registered body reads {ops}")
        cv = _const_value(c, where)
        if step is None:
            step = cv
        elif cv != step:
            raise Refusal(f"{where}: increment literal Const32[{cv:#x}] differs from step 0's Const32[{step:#x}] (one literal per consumer)")
        best, best_i, i = base + 1, base + 2, base + 4
    if int(returns[0]) != best_i:
        raise Refusal(f"TokenSelect call [{lo},{hi}) returns gate {returns[0]}, the registered body returns the last ni = {best_i}")
    assert step is not None
    return PinnedFunction(lo, hi, V, Literals(init_index, init_i, step), tuple(int(a) for a in args), best_i)


# --- statement / witness codecs -------------------------------------------------------------------------------------------
#
# Statement (``veritor.select-argmax@1``, big-endian integers, no padding):
#   magic[8] | V u32 | index_bits u32 | lit_init_index u32 | lit_init_i u32 | lit_step u32 | root[32] | leaf_count u64 |
#   logits_rank u64 | select_rank u64 | logits_id (u16 len, utf8) | select_id (u16 len, utf8)
# Witness: logits_len u32 | logits bytes | path (one byte per level: 0 = lift, 1 = sibling + 32 bytes) |
#          select_len u32 | select bytes | path.  The number of levels is a function of leaf_count (tree depth); the
#          guest parses exactly that many and refuses trailing bytes.


def _u16(n: int) -> bytes:
    return struct.pack(">H", n)


def _text(s: str) -> bytes:
    b = s.encode("utf-8")
    if len(b) >= MAX_ID:
        raise ValueError(f"text too long: {len(b)} bytes")
    return _u16(len(b)) + b


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data, self.pos = bytes(data), 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise ValueError("truncated")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def u8(self) -> int:
        return self.take(1)[0]

    def u32(self) -> int:
        return struct.unpack(">I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack(">Q", self.take(8))[0]

    def text(self) -> str:
        n = struct.unpack(">H", self.take(2))[0]
        return self.take(n).decode("utf-8")

    def done(self) -> bool:
        return self.pos == len(self.data)


@dataclass(frozen=True, slots=True)
class SelectStatement:
    """The public statement: everything the verifier derives from the program and the committed id lists."""

    V: int
    index_bits: int
    literals: Literals
    tree: str  # the RU tree id (``ru/<request>/<step>``) -- NOT in the bytes (the root is); carried for the verifier's naming
    root: bytes
    leaf_count: int
    logits_rank: int
    logits_id: str
    select_rank: int
    select_id: str

    @property
    def logits_shape_text(self) -> str:
        return f"[{self.V}]"

    def encode(self) -> bytes:
        if len(self.root) != 32:
            raise ValueError("root must be 32 bytes")
        head = struct.pack(">IIIII", self.V, self.index_bits, *(x & 0xFFFFFFFF for x in self.literals.as_tuple()))
        tail = struct.pack(">QQQ", self.leaf_count, self.logits_rank, self.select_rank)
        return STATEMENT_MAGIC + head + self.root + tail + _text(self.logits_id) + _text(self.select_id)

    @classmethod
    def decode(cls, data: bytes, *, tree: str = "") -> SelectStatement:
        """Strict: refuses (``ValueError``) a wrong magic, a truncated / trailing byte, or a field outside the guest's bounds."""

        r = _Reader(data)
        if r.take(8) != STATEMENT_MAGIC:
            raise ValueError("bad magic")
        V, index_bits, a, b, c = struct.unpack(">IIIII", r.take(20))
        root = r.take(32)
        leaf_count, logits_rank, select_rank = struct.unpack(">QQQ", r.take(24))
        logits_id = r.text()
        select_id = r.text()
        if not r.done():
            raise ValueError("trailing bytes")
        st = cls(V, index_bits, Literals(a, b, c), tree, root, leaf_count, logits_rank, logits_id, select_rank, select_id)
        st.check_bounds()
        return st

    def check_bounds(self) -> None:
        if not 2 <= self.V <= MAX_V:
            raise ValueError(f"V={self.V} outside [2, 2^24]")
        if not 1 <= self.index_bits <= 32:
            raise ValueError(f"index_bits={self.index_bits} outside [1, 32]")
        if self.leaf_count < 1:
            raise ValueError("leaf_count 0")
        if self.logits_rank >= self.leaf_count or self.select_rank >= self.leaf_count:
            raise ValueError("rank >= leaf_count")
        if self.logits_rank == self.select_rank:
            raise ValueError("logits and select at the same rank")

    @property
    def digest(self) -> bytes:
        return hashlib.sha256(self.encode()).digest()

    def to_json(self) -> dict[str, Any]:
        return {
            "family": FAMILY, "kind": KIND, "V": self.V, "index_bits": self.index_bits,
            "literals": {"init_index": self.literals.init_index, "init_i": self.literals.init_i, "step": self.literals.step},
            "tree": self.tree, "root": self.root.hex(), "leaf_count": self.leaf_count,
            "logits": {"rank": self.logits_rank, "canonical_id": self.logits_id, "dtype": LOGITS_DTYPE, "shape": self.logits_shape_text},
            "select": {"rank": self.select_rank, "canonical_id": self.select_id, "dtype": SELECT_DTYPE, "shape": "[]"},
            "statement_hex": self.encode().hex(), "statement_digest": self.digest.hex(),
        }


@dataclass(frozen=True, slots=True)
class SelectWitness:
    logits: bytes
    logits_path: tuple[bytes | None, ...]
    select: bytes
    select_path: tuple[bytes | None, ...]

    @staticmethod
    def _path_bytes(path: Sequence[bytes | None]) -> bytes:
        out = bytearray()
        for p in path:
            if p is None:
                out.append(0)
            else:
                if len(p) != 32:
                    raise ValueError("sibling must be 32 bytes")
                out.append(1)
                out += p
        return bytes(out)

    def encode(self) -> bytes:
        return (struct.pack(">I", len(self.logits)) + self.logits + self._path_bytes(self.logits_path)
                + struct.pack(">I", len(self.select)) + self.select + self._path_bytes(self.select_path))

    @classmethod
    def decode(cls, data: bytes, depth: int) -> SelectWitness:
        r = _Reader(data)

        def path() -> tuple[bytes | None, ...]:
            out: list[bytes | None] = []
            for _ in range(depth):
                flag = r.u8()
                if flag == 0:
                    out.append(None)
                elif flag == 1:
                    out.append(r.take(32))
                else:
                    raise ValueError("bad path flag")
            return tuple(out)

        logits = r.take(r.u32())
        lp = path()
        select = r.take(r.u32())
        sp = path()
        if not r.done():
            raise ValueError("trailing bytes")
        return cls(logits, lp, select, sp)


# --- hashing (mirror of veritor_zk_common::packed / verity_vllm.hashing) -----------------------------------------------------


def _H(tag: str, *parts: bytes) -> bytes:
    h = hashlib.sha256()
    t = tag.encode("utf-8")
    h.update(len(t).to_bytes(4, "big"))
    h.update(t)
    for p in parts:
        h.update(len(p).to_bytes(8, "big"))
        h.update(p)
    return h.digest()


def leaf_hash(canonical_id: str, dtype: str, shape_text: str, value: bytes) -> bytes:
    return _H(LEAF_TAG, canonical_id.encode("utf-8"), dtype.encode("utf-8"), shape_text.encode("utf-8"), value)


def tree_depth(count: int) -> int:
    d, w = 0, count
    while w > 1:
        w = (w + 1) // 2
        d += 1
    return d


def fold_path(leaf: bytes, count: int, index: int, path: Sequence[bytes | None]) -> bytes | None:
    """``veritor_zk_common::packed::vllm_fold_path``: None when the path has the wrong length or shape for (count, index)."""

    if count == 0 or index >= count or len(path) != tree_depth(count):
        return None
    digest, width, cursor = leaf, count, index
    for step in path:
        lifted = cursor == width - 1 and width % 2 == 1
        if step is None:
            if not lifted:
                return None
            digest = _H(LIFT_TAG, digest)
        else:
            if lifted:
                return None
            digest = _H(NODE_TAG, digest, step) if cursor % 2 == 0 else _H(NODE_TAG, step, digest)
        width = (width + 1) // 2
        cursor //= 2
    return digest


def public_values(statement_bytes: bytes, verdict: int) -> bytes:
    return hashlib.sha256(statement_bytes).digest() + bytes([verdict])


# --- the guest's mirror ----------------------------------------------------------------------------------------------------


def verify_transparent(statement_bytes: bytes, witness_bytes: bytes) -> str:
    """The guest, in Python: ``"accept"`` or ``"reject:<CODE>"`` (:data:`REJECT_CODES`), the checks in the guest's order.
    LABEL: a local mirror of the relation and the openings -- NOT the approved checker's proof (that is
    ``veritor-select-host prove`` on the ELF)."""

    try:
        st = SelectStatement.decode(statement_bytes)
    except (ValueError, UnicodeDecodeError):
        return "reject:STATEMENT_MALFORMED"
    depth = tree_depth(st.leaf_count)
    try:
        w = SelectWitness.decode(witness_bytes, depth)
    except ValueError:
        return "reject:WITNESS_MALFORMED"
    if len(w.logits) != 2 * st.V:
        return "reject:LOGITS_LENGTH"
    if len(w.select) != 4:
        return "reject:SELECT_LENGTH"
    if fold_path(leaf_hash(st.logits_id, LOGITS_DTYPE, st.logits_shape_text, w.logits), st.leaf_count, st.logits_rank, w.logits_path) != st.root:
        return "reject:LOGITS_AUTH"
    if fold_path(leaf_hash(st.select_id, SELECT_DTYPE, "[]", w.select), st.leaf_count, st.select_rank, w.select_path) != st.root:
        return "reject:SELECT_AUTH"
    words = struct.unpack(f"<{st.V}H", w.logits)
    selected = argmax_pinned(words, st.literals)
    if st.index_bits < 32 and selected >= (1 << st.index_bits):
        return "reject:INDEX_RANGE"
    if selected != struct.unpack("<I", w.select)[0]:
        return "reject:SELECT_MISMATCH"
    return "accept"


# --- resolving a located unit against a record --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedSelect:
    vu_index: int
    kind: str
    request_id: str
    t: int
    root_name: str  # ``t=<t>/final``
    pinned: PinnedFunction
    tree: str
    logits_id: str
    logits_rank: int
    select_id: str
    select_rank: int
    regrouped_step: int | None = None  # set when the located unit was one ArgmaxStep of this scan (Amendment A2 decision 4)

    @property
    def target(self) -> str:
        return f"{self.root_name}/TokenSelect@[{self.pinned.lo},{self.pinned.hi})"


def _t_of(root_name: str) -> int:
    head, _, rest = root_name.partition("/")
    if not head.startswith("t=") or rest != "final":
        raise Refusal(f"root call {root_name!r} is not a t=<t>/final call (TokenSelect lives only there)")
    return int(head[2:])


def _identity_fields(cid: str) -> dict[str, str] | None:
    parts = cid.split("/")
    d: dict[str, str] = {}
    for p in parts:
        k, _, v = p.partition("=")
        if not _ or k in d:
            return None
        d[k] = v
    return d if {"p", "r", "t", "ph", "l", "op", "port"} <= set(d) else None


def _find_leaves(record: Any, request_id: str, t: int) -> tuple[str, str, int, str, int]:
    """(tree, logits id, logits rank, select id, select rank) of step ``t`` of the request, from the committed id lists."""

    hits: list[tuple[str, str, int, str, int]] = []
    for tree in record.ru_tree_ids(request_id):
        ids = record.ids_of(tree)
        lg = [(k, cid) for k, cid in enumerate(ids) if (f := _identity_fields(cid)) and f["t"] == str(t) and f["l"] == "lm_head" and f["op"] == "lm_head.gemm" and f["port"] == "0"]
        ss = [(k, cid) for k, cid in enumerate(ids) if (f := _identity_fields(cid)) and f["t"] == str(t) and f["l"] == "sampler" and f["op"] == "sampler.select" and f["port"] == "0"]
        if not lg and not ss:
            continue
        if len(lg) != 1 or len(ss) != 1:
            raise Refusal(f"{tree}: t={t} has {len(lg)} lm_head.gemm rows and {len(ss)} sampler.select words committed (need one each)")
        hits.append((tree, lg[0][1], lg[0][0], ss[0][1], ss[0][0]))
    if len(hits) != 1:
        raise Refusal(f"request {request_id} t={t}: the logits row / selected token are committed in {len(hits)} RU trees (need exactly one)")
    return hits[0]


def _circuit_map(record: Any):
    """``verity_vllm.correspondence.core.CircuitMap`` over the locator's named root calls (what ``ProgramLocator._x_candidates`` does):
    the verifier's own identity table, without walking every root call of a lazy program."""

    from verity_vllm.correspondence.core import CircuitMap

    cm = CircuitMap.__new__(CircuitMap)
    cm.program, cm.circuit, cm.names, cm.named = record.program, record.program.circuit, {}, record.locator.cmap.named
    return cm


_PIN_CACHE: dict[tuple[int, int, int], PinnedFunction] = {}


def resolve(located: Mapping[str, Any], record: Any) -> ResolvedSelect:
    """The located VU as one whole-scan TokenSelect over committed words, or a :class:`Refusal` naming what is missing."""

    from verity_vllm.correspondence.core import CaptureIdentity

    kind = str(located["kind"])
    if kind != KIND and kind != STEP_KIND and not kind.startswith("TokenSelect_v1"):
        raise Refusal(f"{kind}: not a token-selection kind ({KIND} / {STEP_KIND})")
    lo, hi = (int(x) for x in located["gate_interval"])
    request_id = str(located["request_id"])
    loc = record.locator
    name, root = loc.root_of(lo)
    if name is None:
        raise Refusal(f"{kind}: gate {lo} lies in no root call")
    t = _t_of(name)
    calls = [c for c in root.children() if c.name == "TokenSelect"]
    if len(calls) != 1:
        raise Refusal(f"{name}: {len(calls)} TokenSelect calls (need one)")
    ts = calls[0]
    regrouped = None
    if kind == STEP_KIND:
        if not (ts.lo + 2 <= lo and hi <= ts.hi and hi - lo == 5 and (lo - ts.lo - 2) % 5 == 0):
            raise Refusal(f"{kind}: [{lo},{hi}) is not one ArgmaxStep body of {name}/TokenSelect [{ts.lo},{ts.hi})")
        regrouped = (lo - ts.lo - 2) // 5
    elif (lo, hi) != (ts.lo, ts.hi):
        raise Refusal(f"{kind}: [{lo},{hi}) is not the whole TokenSelect call [{ts.lo},{ts.hi}) of {name}")
    key = (id(record.program), ts.lo, ts.hi)
    pinned = _PIN_CACHE.get(key)
    if pinned is None:
        pinned = pin_function(record.program.circuit, ts.lo, ts.hi, tuple(ts.args), tuple(ts.returns))
        _PIN_CACHE[key] = pinned
    tree, logits_id, logits_rank, select_id, select_rank = _find_leaves(record, request_id, t)
    # the verifier's own identity table must put the committed row exactly on the call's operands and the word on its return
    cm = _circuit_map(record)
    _, lg_gates = cm.locate(CaptureIdentity.parse(logits_id))
    if tuple(int(g) for g in lg_gates) != pinned.logits:
        raise Refusal(f"{logits_id}: its {len(lg_gates)} program gates are not the TokenSelect operands (CircuitMap vs call args)")
    _, sel_gates = cm.locate(CaptureIdentity.parse(select_id))
    if tuple(int(g) for g in sel_gates) != (pinned.out,):
        raise Refusal(f"{select_id}: its program gates {tuple(sel_gates)} are not the TokenSelect return {pinned.out}")
    return ResolvedSelect(int(located.get("vu_index", -1)), kind, request_id, t, name, pinned, tree, logits_id, logits_rank, select_id, select_rank, regrouped)


def expectation(located: Mapping[str, Any], record: Any, *, index_bits: int = INDEX_BITS) -> SelectStatement:
    """The verifier's expected statement for a located unit: every field from the program text (V, literals, the call's
    wiring), ``commitments.json`` (root, leaf_count) and the committed id lists (ranks, canonical ids).  Never from the
    prover's object.  The proof's public digest must equal ``sha256(expectation(...).encode())``."""

    u = resolve(located, record)
    rec = record.commitments["trees"][u.tree]
    return SelectStatement(u.pinned.V, index_bits, u.pinned.literals, u.tree, bytes.fromhex(rec["root"]), int(rec["leaf_count"]),
                           u.logits_rank, u.logits_id, u.select_rank, u.select_id)


def check_expected(statement: SelectStatement | bytes, expected: SelectStatement) -> list[str]:
    """Field-by-field comparison of a presented statement with the verifier's expectation; every difference is named
    (``literal init_i: statement 2 != program 1``).  Empty = the statement is exactly the expected one."""

    if isinstance(statement, (bytes, bytearray)):
        try:
            statement = SelectStatement.decode(bytes(statement), tree=expected.tree)
        except (ValueError, UnicodeDecodeError) as e:
            return [f"statement: malformed ({e})"]
    problems: list[str] = []
    if statement.V != expected.V:
        problems.append(f"V: statement {statement.V} != program {expected.V}")
    if statement.index_bits != expected.index_bits:
        problems.append(f"index_bits: statement {statement.index_bits} != declared {expected.index_bits}")
    for nm in ("init_index", "init_i", "step"):
        a, b = getattr(statement.literals, nm), getattr(expected.literals, nm)
        if a != b:
            problems.append(f"literal {nm}: statement {a} != program {b}")
    if statement.root != expected.root:
        problems.append(f"root: statement {statement.root.hex()[:16]} != committed {expected.root.hex()[:16]} ({expected.tree})")
    if statement.leaf_count != expected.leaf_count:
        problems.append(f"leaf_count: statement {statement.leaf_count} != committed {expected.leaf_count}")
    if (statement.logits_rank, statement.logits_id) != (expected.logits_rank, expected.logits_id):
        problems.append(f"logits leaf: statement ({statement.logits_rank}, {statement.logits_id}) != committed ({expected.logits_rank}, {expected.logits_id})")
    if (statement.select_rank, statement.select_id) != (expected.select_rank, expected.select_id):
        problems.append(f"select leaf: statement ({statement.select_rank}, {statement.select_id}) != committed ({expected.select_rank}, {expected.select_id})")
    if not problems and statement.encode() != expected.encode():
        problems.append("statement bytes differ from the expectation's canonical encoding")
    return problems


# --- the prover's statement ------------------------------------------------------------------------------------------------------


@dataclass
class Built:
    """One select statement with its witness (the two openings), the verifier's expectation and the local verdict."""

    unit: ResolvedSelect
    statement: SelectStatement
    statement_bytes: bytes
    witness: SelectWitness
    witness_bytes: bytes
    openings: dict[str, Any]
    verdict: str
    record: dict[str, Any] = field(default_factory=dict)
    family: str = FAMILY

    @property
    def expectations(self) -> list[SelectStatement]:
        return [self.statement]


def _opening_checks(o: Any, tree: str, rank: int, cid: str, dtype: str, shape: tuple[int, ...], nbytes: int) -> None:
    if o.tree != tree or int(o.index) != rank or o.canonical_id != cid:
        raise Refusal(f"{cid}: opened at ({o.tree}, {o.index}), located at ({tree}, {rank})")
    if o.dtype != dtype or tuple(int(x) for x in o.shape) != shape:
        raise Refusal(f"{cid}: committed as {o.dtype} {tuple(o.shape)}, the prescribed function reads {dtype} {shape}")
    if len(o.value) != nbytes:
        raise Refusal(f"{cid}: {len(o.value)} bytes, expected {nbytes}")


def build(located: Mapping[str, Any], record: Any, *, name: str = "ck-select", index_bits: int = INDEX_BITS, authenticate: bool = True) -> Built:
    """``build(located_unit, record) -> Built``: open the two committed leaves, encode statement + witness, run the mirror."""

    u = resolve(located, record)
    st = expectation(located, record, index_bits=index_bits)
    lo = record.open_leaf(u.logits_id)
    so = record.open_leaf(u.select_id)
    _opening_checks(lo, u.tree, u.logits_rank, u.logits_id, LOGITS_DTYPE, (u.pinned.V,), 2 * u.pinned.V)
    _opening_checks(so, u.tree, u.select_rank, u.select_id, SELECT_DTYPE, (), 4)
    if authenticate:
        from verity_vllm.commit import verify_opening_in

        for o in (lo, so):
            if not verify_opening_in(record.commitments, o):
                raise Refusal(f"{o.canonical_id}: opening does not authenticate against commitments.json")
    w = SelectWitness(bytes(lo.value), tuple(lo.path), bytes(so.value), tuple(so.path))
    sb, wb = st.encode(), w.encode()
    verdict = verify_transparent(sb, wb)
    rec = {
        "name": name,
        "family": FAMILY,
        "kind": KIND,
        "description": f"CK-SELECT: {u.target} -- argmax over the committed lm_head logits row equals the committed sampler.select word",
        "statement": st.to_json(),
        "statement_hex": sb.hex(),
        "witness_hex": wb.hex(),
        "public_values_hex": public_values(sb, VERDICT_ACCEPT if verdict == "accept" else VERDICT_REJECT).hex(),
        "verdict_transparent": verdict,
        "provenance": {
            "builder": "e2e.coverage_v1.ck_select",
            "unit": {"vu_index": u.vu_index, "kind": u.kind, "request_id": u.request_id, "t": u.t, "target": u.target,
                     "call": [u.pinned.lo, u.pinned.hi], "regrouped_step": u.regrouped_step,
                     "literals": u.pinned.literals.as_tuple(), "V": u.pinned.V},
            "openings": {o.canonical_id: {"tree": o.tree, "index": int(o.index), "dtype": o.dtype, "shape": list(o.shape), "bytes": len(o.value)} for o in (lo, so)},
        },
        "prove": "zk/sp1/programs/select: veritor-select-host prove --statement <statement bytes> --witness <witness bytes> --out <proof>",
    }
    return Built(u, st, sb, w, wb, {lo.canonical_id: lo, so.canonical_id: so}, verdict, rec)


def dump_vector(built: Built, path: Path) -> None:
    path.write_text(json.dumps(built.record, indent=1, sort_keys=True))
