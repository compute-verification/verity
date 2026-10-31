"""The ⊥-lifted, fully padded Definitions — **lifted-padded**.

Two rules hold for them:
(a) total relation behaviour for noncanonical `1‖nonzero` words per operation class (PAD-01 CLOSED: `INV_w`, never normalized
before commitment) and (b) TRUE widened widths (PAD-02 CLOSED: no activity/payload split, real widths in admissibility and summaries);
and DP-pad-02 holds (PAD-05 CLOSED: `LAG` a declared specialization parameter, one rule with moe/GP-01, never inferred from a trace).
Nothing here edits a frozen Definition: every lifted Definition is registered under a **new** id (`Lifted[<id>]_v1`,
`LSelect<w>_v1`, …) with its own spec string and digest; the originals are untouched and are what the lifted
evaluators call for the payload arithmetic.  Every lifted result carries `LIFTED_RULES_VERSION`.

Encodings (AGREED A3): a `w`-bit domain is represented in `w+1` bits, the tag is bit `w`:

    enc(Active(v)) = v          (tag 0, payload v)        v ∈ [0, 2^w)
    enc(⊥)         = 2^w        (tag 1, payload 0)
    valid_w(x)     = tag(x) = 0 ∨ payload(x) = 0          the 2^w + 1 canonical patterns
    INV_w          = 2^w + 1    (tag 1, payload 0…01)     AGREED: the one noncanonical word every lifted
                                                           primitive maps any noncanonical operand to; never an honest value

Three classes of lifted primitive:

* **strict** (§3.1, the default; AGREED A5 "ordinary arithmetic"): `lifted_prim(F)` wraps the frozen evaluator mechanically —
  any invalid operand → `INV`, else any ⊥ operand → ⊥, else `0‖F(payloads)` bit for bit.  No hand-written semantics per primitive.
* **specified** (§3.2, the closed list; AGREED A5 "selection, indexing, batching, state"): `LSelect<w>`, `LIsActive<w>`,
  `LBot<w>`, `LGatherRow<w>x<N>`, `LStepLive<E>`, the Continuation_v1 members `LIsEOS<E>` / `LEmitNext[H]`  and
  `LFixedSelect<E>x<w>` (FixedSelect among computed slot outputs) — hand-authored relations below, each with
  the reason it is not strict.  Nothing else is lifted non-strictly (`test_lifted_tiny.py::test_specified_list_is_closed`).
* **constants / inputs** (§2): `Const<w>[bits]` lifts to the existing `Const<w+1>[0‖bits]` (`b1.const(w+1, bits)`, tag 0 by
  construction); `Input<w>` lifts to `Input<w+1>` (the Program lowers it from the lifted parameter type); prescribed inputs are
  checked `tag = 0 ∧ valid` at the boundary (`check_boundary`).

Composites (§3.3) are lifted by **node replay** (`lift_def`): the lifted body is the original body with every callee replaced by
its lifted Definition and every type lifted — same node indices, same operand runs, same gate count, so canonical references
are unchanged modulo spec ids (address stability holds by construction, and is tested).  The evaluator never
shortcuts: it evaluates the denoted circuit gate by gate, so the §3.3 strictness/payload-agreement theorems are *measured* on
each lifted composite of the fixture rather than assumed.

`LServe_v1` (§4) is the lifted padded serving Program on the `Serve_v2` skeleton: no live array; the activity guard sits on each
step's **input** token (`GUARD = "in"`: `tok_in_s = LSelect(LStepLive(out[s−1], eos_set), out[s−1], LBot<32>)`, `out[s] = y_s`) —
DP-pad-01 option (a), a labelled candidate: the spec's letter (§4.2, output guard) leaves step `a_r` half-Active and contradicts its
own §4.3/§6/§7; the cache append is unguarded (an inactive step's new K/V row is ⊥ by strictness).  The
guard is two root-body nodes per step (`s=<s>/live`, `s=<s>/tok_in`) plus one `LBot32` and `|EOS|` constants — extra nodes the
unlifted skeleton does not have; how address stability reads across them is PAD-03, an open question, not a silent fix.
`activity_prefix` / `served_count` / `check_served_prefix` / `assign_inactive` are §7's rule as pure functions on committed token
arrays (AGREED of record).  Widths: `lifted_width_summary` / `lifted_width_admissibility` report
the TRUE lifted width `Σ (w+1)` and run the untouched `partition.validate_width` at the lifted parameter `W_MAX_LIFTED`; the earlier
`validate_width_lifted` (payload rule + `n_max`) is WITHDRAWN.  `LAG`: `lag_from_header` (the attestation) and
`declared_lag_of_row` (the Build's declaration) both go through GP-01's one rule (`_lag_rule`); the lifted tool's R0 needs both, equal.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Sequence

from verity.ml.circuits.defs import (REGISTRY, Builder, CompositeDefinition, FunctionDefinition, Node, PrimitiveDefinition, SpecializedDefinition, bind,
                    composite)
from verity.ml.circuits.refs import Affine, Coll, Concat, Explicit, Refs, Strided, coll_concat, concat, tuple_of
from verity.ml.circuits.types import Array, Record, Tuple, Type, Value
from verity.ml.definitions.registry import b1, operand_widths

LIFTING_VERSION = "v1"  # the `lifting: v1` the derive report names
LABEL = "lifted-padded"  # every result of this lane is labelled so, never a panel / active-reference row

# ---------------------------------------------------------------------------------------------------------
# encodings (§2) and the PAD-01 candidate (§9a)
# ---------------------------------------------------------------------------------------------------------


def enc_bot(w: int) -> int:
    """`enc(⊥)` at payload width `w`: `1‖0^w` = `2^w` (AGREED A3)."""
    return 1 << w


def inv(w: int) -> int:
    """`INV_w = 1‖0^{w−1}‖1` = `2^w + 1` — CANDIDATE PAD-01: the single fixed total extension for invalid operands.  A name
    for "error", never an honest encoding of anything."""
    return (1 << w) | 1


def tag(w: int, x: int) -> int:
    return (x >> w) & 1


def payload(w: int, x: int) -> int:
    return x & ((1 << w) - 1)


def in_width(w: int, x: int) -> bool:
    """`x` is a `w+1`-bit word at all."""
    return 0 <= x < (1 << (w + 1))


def valid(w: int, x: int) -> bool:
    """`valid_w(x) := tag(x) = 0 ∨ payload(x) = 0` — the `2^w + 1` canonical patterns."""
    return in_width(w, x) and (tag(w, x) == 0 or payload(w, x) == 0)


def is_bot(w: int, x: int) -> bool:
    return x == (1 << w)


def is_active(w: int, x: int) -> bool:
    return in_width(w, x) and tag(w, x) == 0


def enc_active(w: int, v: int) -> int:
    if not (0 <= v < (1 << w)):
        raise ValueError(f"payload {v:#x} is not a {w}-bit word")
    return v


class BoundaryFailure(ValueError):
    """A named rejection at an observable boundary (PAD-01 rule 1): `(what, w, x, reason)`."""

    def __init__(self, what: str, w: int, x: int, reason: str) -> None:
        super().__init__(f"{what}: {w + 1}-bit word {x:#x} rejected: {reason}")
        self.what, self.w, self.x, self.reason = what, w, x, reason


def check_boundary(w: int, x: int, *, what: str = "value", prescribed: bool = False, index_range: int | None = None) -> int:
    """The validity check every observable boundary performs (PAD-01 rule 1): a `1‖nonzero` word is rejected by
    name; a **prescribed** input must additionally be Active; an index-valued input is range-checked here
    (range is a boundary fact).  Returns `x` unchanged when it passes."""
    if not in_width(w, x):
        raise BoundaryFailure(what, w, x, f"not a {w + 1}-bit word")
    if not valid(w, x):
        raise BoundaryFailure(what, w, x, "invalid representation (tag 1, payload != 0): neither Active nor ⊥")
    if prescribed and tag(w, x) != 0:
        raise BoundaryFailure(what, w, x, "prescribed input must be Active (tag 0)")
    if index_range is not None and tag(w, x) == 0:
        t = payload(w, x)
        t = t - (1 << w) if t & (1 << (w - 1)) else t
        if not (0 <= t < index_range):
            raise BoundaryFailure(what, w, x, f"index {t} out of range [0, {index_range})")
    return x


# ---------------------------------------------------------------------------------------------------------
# types (§2)
# ---------------------------------------------------------------------------------------------------------


def lift_type(t: Type) -> Type:
    """`Lift(Value<w>) = Value<w+1>`; structure over references is unchanged."""
    if isinstance(t, Value):
        return Value(t.w + 1)
    if isinstance(t, Array):
        return Array(t.n, lift_type(t.elem))
    if isinstance(t, Tuple):
        return Tuple(*[lift_type(f) for f in t.fields])
    if isinstance(t, Record):
        return Record(*[(k, lift_type(ft)) for k, ft in t.fields])
    raise TypeError(t)


def unlift_type(t: Type) -> Type:
    if isinstance(t, Value):
        if t.w < 2:
            raise TypeError(f"{t!r} is not a lifted type")
        return Value(t.w - 1)
    if isinstance(t, Array):
        return Array(t.n, unlift_type(t.elem))
    if isinstance(t, Tuple):
        return Tuple(*[unlift_type(f) for f in t.fields])
    if isinstance(t, Record):
        return Record(*[(k, unlift_type(ft)) for k, ft in t.fields])
    raise TypeError(t)


def lift_words(t: Type, words: Sequence[int]) -> list[int]:
    """Encode a flat leaf list of an unlifted type as all-Active lifted words (`0‖v`; the identity on the ints)."""
    ws = t.leaf_widths()
    if len(ws) != len(words):
        raise ValueError(f"{t!r}: {len(ws)} leaves, {len(words)} words")
    return [enc_active(w, v) for w, v in zip(ws, words)]


def bot_words(t: Type) -> list[int]:
    """The all-⊥ assignment of an unlifted type, as lifted words."""
    return [enc_bot(w) for w in t.leaf_widths()]


# ---------------------------------------------------------------------------------------------------------
# the strict class (§3.1): one generic wrapper, generated mechanically from the frozen evaluator
# ---------------------------------------------------------------------------------------------------------

_LIFTED_PRIMS: dict[str, PrimitiveDefinition] = {}
_SPECIFIED_IDS: set[str] = set()  # ids of the specified-class primitives (the closed list, §3.2)
#: the indexing FAMILY, not one name — every registered `Gather<T>x<N>_v1` with the shape
# `(token: Value<32>, column: Array<N, Value<w>>) -> Value<w>` is a gather; its out-of-range extension is read from ITS original.
_GATHER_RE = re.compile(r"^Gather([A-Za-z0-9]+?)x(\d+)_v1$")
_OOR_PROBES = (None, 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF)  # N (filled in), N+1, −1, i32 min, i32 max — all out of range for N ≤ 2^31−1


def _gather_shape(p: PrimitiveDefinition) -> tuple[int, int] | None:
    """`(w, N)` if `p` has a gather's shape, else None."""
    if not _GATHER_RE.match(p.id) or len(p.params) != 2:
        return None
    (_, ti), (_, tc) = p.params
    if not (isinstance(ti, Value) and ti.w == 32 and isinstance(tc, Array) and isinstance(tc.elem, Value) and p.ret == tc.elem):
        return None
    return tc.elem.w, tc.n


def registered_oor_extension(p: PrimitiveDefinition) -> int:
    """The original gather's **registered total extension** on an Active out-of-range index, read from the original itself
    (its `evaluate_lazy` on five out-of-range probes: `N`, `N+1`, `−1`, i32 min, i32 max — no column word is touched): one
    constant, else `TypeError` by name.  Nothing is invented and nothing is keyed on a payload width."""
    shape = _gather_shape(p)
    if shape is None:
        raise TypeError(f"{p.id} is not a registered gather family (shape (Value<32>, Array<N, Value<w>>) -> Value<w>)")
    w, N = shape
    lazy = getattr(p, "evaluate_lazy", None)
    seen = set()
    for probe in _OOR_PROBES:
        tok = N if probe is None else (probe + N if probe == 1 else probe)
        if lazy is not None:
            seen.add(int(lazy(lambda i, _t=tok: _t if i == 0 else 0)))
        else:
            seen.add(int(p.evaluate(tok, *([0] * N))))
    if len(seen) != 1:
        raise TypeError(f"{p.id}: no single total out-of-range extension (probes gave {sorted(seen)})")
    (oor,) = seen
    if not (0 <= oor < (1 << w)):
        raise TypeError(f"{p.id}: out-of-range extension {oor:#x} is not a {w}-bit word")
    return oor


def _gather_families() -> dict[int, Callable[[int], PrimitiveDefinition]]:
    """The registered original gather families by payload width — for the id-only path (`l_gather_row(w, N)` from the codec's
    lazy resolver); the extension itself is always read from the constructed original, never from this table."""
    from verity.ml.definitions.registry import prims as P
    from verity.ml.definitions.registry import ref_prims as R
    return {16: P.gather_bf16, 32: R.gather_f32}


def _classify(ops: Sequence[int], widths: Iterable[int]) -> int:
    """PAD-01 / §3.1 operand scan: 2 = some operand invalid (→ INV), 1 = all valid and some ⊥ (→ ⊥), 0 = all Active."""
    saw_bot = False
    for x, w in zip(ops, widths):
        t = x >> w
        if t:
            if t != 1 or (x & ((1 << w) - 1)):
                return 2
            saw_bot = True
    return 1 if saw_bot else 0


def _strict_wrapper(p: PrimitiveDefinition) -> PrimitiveDefinition:
    widths = operand_widths.leaf_widths(p.params)
    wout = p.ret.w
    bot_out, inv_out = enc_bot(wout), inv(wout)
    f = p.evaluate

    def ev(*ops: int, _f=f, _widths=widths, _bot=bot_out, _inv=inv_out) -> int:
        c = _classify(ops, _widths)
        if c == 2:
            return _inv
        if c == 1:
            return _bot
        return _f(*ops)  # tag 0 everywhere: the payloads ARE the words; the original semantics bit for bit

    lp = PrimitiveDefinition(f"Lifted[{p.id}]", 1, tuple((n, lift_type(t)) for n, t in p.params), Value(wout + 1), ev,
                             doc=f"strict lift of {p.id} (SPEC §3.1): INV on any invalid operand, ⊥ on any ⊥ operand, else 0‖{p.id}",
                             conformance="protocol-defined", tables=p.tables)
    lp.lifted_from = p  # type: ignore[attr-defined]
    lp.lift_class = "strict"  # type: ignore[attr-defined]
    return lp


def lifted_prim_v1(p: PrimitiveDefinition) -> PrimitiveDefinition:
    """LEGACY `lifted-v1` (propagation semantics): the lifted Definition of a registered primitive WITHOUT an
    enable operand — constants → `Const<w+1>[0‖bits]`; inputs → `Input<w+1>`; the selects and gathers → their specified-class
    member; everything else → the strict wrapper.  Kept so a `lifted-v1` descriptor (`Lifted[<id>]_v1`) still cold-decodes; new
    lifts use `lifted_prim` (v2 explicit enablement).  Cached by id."""
    lp = _LIFTED_PRIMS.get(p.id)
    if lp is not None:
        return lp
    fam = getattr(p, "family", None)
    if p.id in _SPECIFIED_IDS or p.id.startswith("Lifted["):
        raise TypeError(f"{p.id} is already a lifted definition")
    if fam == "Const":
        from verity.ml.definitions.registry.b1 import const
        lp = const(p.ret.w + 1, p.evaluate())
    elif fam == "Input":
        from verity.ml.circuits.program import RUN, input_prim
        lp = input_prim(p.ret.w + 1, getattr(p, "binding", RUN))
    elif p.id in ("SelectI32_v1", "SelectF32_v1"):
        lp = l_select(32)
    elif p.id == "SelectBf16_v1":
        lp = l_select(16)
    elif _gather_shape(p) is not None:
        lp = l_gather_row_of(p)  # any registered gather family, extension read from this original
    else:
        lp = _strict_wrapper(p)
    _LIFTED_PRIMS[p.id] = lp
    return lp


# ---------------------------------------------------------------------------------------------------------
# explicit enablement: the enable is the LAST operand of every lifted computed gate
# ---------------------------------------------------------------------------------------------------------

LIFT_VERSION = 2  # `Lifted[<id>]_v2`: the enable-threaded lift (padded-v1 rules); `_v1` = the propagation lift (lifted-v1, legacy)
ENABLE_PARAM = "enable"  # the name of the enable parameter, last in every lifted signature (B7; last so `scan` shared-args work)
ENABLE_T = Value(2)  # a Boolean word: Active(0) = 0b00, Active(1) = 0b01; ⊥ = 0b10 and INV_1 = 0b11 are not enables (→ INV)
_ENABLED_PRIMS: dict[str, PrimitiveDefinition] = {}


def enable_word(a: int) -> int:
    """`a ∈ {0, 1}` as the enable word (`Active(a)`)."""
    if a not in (0, 1):
        raise ValueError(f"an enable is 0 or 1, got {a}")
    return a


def _enable_wrap(lp: PrimitiveDefinition, original_id: str, *, reason: str) -> PrimitiveDefinition:
    """`Lifted[<original id>]_v2(<lp's params…>, enable: Value<2>)`: `enable = Active(1)` → `lp`'s total relation on the other operands
    (the registered enabled result: v0.5.2 A3's rows, now FACTS — A9.4); `Active(0)` → `⊥` WHATEVER the operands (weights, preserved
    state and a constant's own bits are all ignored: the value is produced inside a disabled computation, A9.3); any other enable word
    (`⊥`, `INV_1`) → `INV_w` (A9.2: an enable is a Boolean word owned by the parent)."""
    wout = lp.ret.w - 1
    bot_out, inv_out = enc_bot(wout), inv(wout)
    f = lp.evaluate
    n = sum(t.leaves for _, t in lp.params)  # the evaluator receives FLAT leaf words; the enable is the one after the last data leaf

    def ev(*ops: int, _f=f, _n=n, _bot=bot_out, _inv=inv_out) -> int:
        a = ops[_n]
        if a == 1:
            return _f(*ops[:_n])
        if a == 0:
            return _bot
        return _inv

    ep = PrimitiveDefinition(f"Lifted[{original_id}]", LIFT_VERSION, tuple(lp.params) + ((ENABLE_PARAM, ENABLE_T),), Value(wout + 1), ev,
                             doc=f"enabled lift of {original_id} (LIFTING-SPEC v0.6 A9.3): enable=1 → {lp.id}'s relation; enable=0 → ⊥ whatever "
                                 f"the operands; other enable words → INV. {reason}", conformance="protocol-defined", tables=lp.tables)
    ep.lifted_from = getattr(lp, "lifted_from", None) or lp  # type: ignore[attr-defined]
    ep.lift_class = getattr(lp, "lift_class", "strict")  # type: ignore[attr-defined]
    ep.lift_enabled = True  # type: ignore[attr-defined]
    ep.lifted_v1 = lp  # type: ignore[attr-defined]  # the same relation at enable = 1
    return ep


def lifted_prim(p: PrimitiveDefinition) -> PrimitiveDefinition:
    """The lifted Definition OF RECORD of a registered primitive (padded-v1): `Lifted[<id>]_v2` = the lifted-v1
    relation (`lifted_prim_v1`: strict wrapper / specified member / lifted constant) with the enable as its LAST operand, so that a
    constant produced inside a disabled body is ⊥ (`Lifted[Const<w>[k]_v1]_v2(enable)`) and a select/gather inside a disabled body is ⊥
    whatever it would have selected.  `Input<w>` lifts to `Input<w+1>` unchanged — an input is the Program's boundary, produced by no
    body, so it owns no enable (A9.3: "a prescribed constant or weight supplied from outside that body remains unchanged")."""
    ep = _ENABLED_PRIMS.get(p.id)
    if ep is not None:
        return ep
    if p.id.startswith("Lifted["):
        raise TypeError(f"{p.id} is already a lifted definition")
    if p.id in _SPECIFIED_IDS:
        # a SPECIFIED lifted-v1 member (LSelect, LFixedSelect, LGatherRow, LStepLive, LIsEOS, ...) used INSIDE a conditional body: its enabled
        # form `Lifted[<member>_v1]_v2(..., enable)` -- the same relation at enable = 1, ⊥ at enable = 0 (A12's slot combine, moe's B7 ask)
        ep = _enable_wrap(p, p.id, reason="specified lifted member under an enable")
    else:
        lp = lifted_prim_v1(p)
        if getattr(p, "family", None) == "Input":
            ep = lp
        else:
            ep = _enable_wrap(lp, p.id, reason=f"class {getattr(lp, 'lift_class', 'constant')}")
    _ENABLED_PRIMS[p.id] = ep
    return ep


def has_enable(fn: Any) -> bool:
    """Does this lifted callee take the enable as its last parameter?"""
    params = fn.params if isinstance(fn, PrimitiveDefinition) else fn.signature()[0]
    return bool(params) and params[-1][0] == ENABLE_PARAM and params[-1][1] == ENABLE_T


# ---------------------------------------------------------------------------------------------------------
# the specified class (§3.2): the closed list, hand-authored, each with its reason
# ---------------------------------------------------------------------------------------------------------

_SELECTS: dict[int, PrimitiveDefinition] = {}
_IS_ACTIVE: dict[int, PrimitiveDefinition] = {}
_BOTS: dict[int, PrimitiveDefinition] = {}
_GATHERS: dict[tuple[int, ...], PrimitiveDefinition] = {}
_STEP_LIVE: dict[int, PrimitiveDefinition] = {}


def _register_specified(p: PrimitiveDefinition, reason: str) -> PrimitiveDefinition:
    p.lift_class = "specified"  # type: ignore[attr-defined]
    p.lift_reason = reason  # type: ignore[attr-defined]
    _SPECIFIED_IDS.add(p.id)
    return p


def l_select(w: int) -> PrimitiveDefinition:
    """`LSelect<w>_v1(c: Value<2>, a, b: Value<w+1>) -> Value<w+1>`: `c = Active(1) → a`; `c = Active(0) → b`; `c = ⊥ → ⊥`;
    any invalid operand (including the **untaken** branch) → `INV_w` (PAD-01 rule 2, "untaken branch poisons" —
    conservative).  Not strict because `Select(true, 7, ⊥) = 7` is the intended relation (AGREED A5)."""
    p = _SELECTS.get(w)
    if p is None:
        bot_w, inv_w = enc_bot(w), inv(w)

        def ev(c: int, a: int, b: int, _w=w, _bot=bot_w, _inv=inv_w) -> int:
            if _classify((c, a, b), (1, _w, _w)) == 2:
                return _inv
            if c == 2:  # ⊥ condition
                return _bot
            return a if c & 1 else b

        p = PrimitiveDefinition(f"LSelect{w}", 1, (("c", Value(2)), ("a", Value(w + 1)), ("b", Value(w + 1))), Value(w + 1), ev,
                                doc=l_select.__doc__ or "", conformance="protocol-defined")
        _SELECTS[w] = _register_specified(p, "selection: the untaken branch may be inactive (Select(true, 7, ⊥) = 7)")
    return p


def l_is_active(w: int) -> PrimitiveDefinition:
    """`LIsActive<w>_v1(x: Value<w+1>) -> Value<2>`: `Active(1)` iff `tag(x) = 0`, `Active(0)` for ⊥; never ⊥; `INV → INV_1`
    (= `0b11`).  The only primitive that reads the tag into a payload — how activity becomes a value (the
    S-migration hook `n_live_t = Σ_r LIsActive(out_r[t−1])`)."""
    p = _IS_ACTIVE.get(w)
    if p is None:
        def ev(x: int, _w=w) -> int:
            if _classify((x,), (_w,)) == 2:
                return 0b11
            return 1 if (x >> _w) == 0 else 0

        p = PrimitiveDefinition(f"LIsActive{w}", 1, (("x", Value(w + 1)),), Value(2), ev, doc=l_is_active.__doc__ or "",
                                conformance="protocol-defined")
        _IS_ACTIVE[w] = _register_specified(p, "reads the tag into a payload: activity as a value")
    return p


_ENABLE_PRIM: list[PrimitiveDefinition] = []


def l_enable() -> PrimitiveDefinition:
    """`LEnable_v1(parent: Value<2>, local: Value<2>) -> Value<2>` -- the enable of a CONDITIONAL call:
    `a_child = a_parent ∧ a_local`, TOTAL and never ⊥ (an enable is `Active(0)` or `Active(1)`, whatever its operands):
    `parent = Active(0)` → `Active(0)` unconditionally (a disabled parent disables every child, whatever the local condition -- the local
    condition may be ⊥ because its own gate is disabled); `parent = Active(1)` → `local` when `local ∈ {Active(0), Active(1)}`; any other
    word (`⊥`/`INV_1` in the parent, or in the local under an enabled parent) → `INV_1` (a noncanonical/ill-formed enable NAMED, so that
    the gates it enables produce `INV_w` -- never a silent Active(0)).  The step enable of `Continuation_v1` (`run_t`) and the expert-slot
    enable of A12 (`a_parent ∧ [r_k = e]`) are its instances."""
    if not _ENABLE_PRIM:
        def ev(parent: int, local: int) -> int:
            if parent == 0:
                return 0
            if parent == 1 and local in (0, 1):
                return local
            return 0b11

        p = PrimitiveDefinition("LEnable", 1, (("parent", Value(2)), ("local", Value(2))), Value(2), ev, doc=l_enable.__doc__ or "",
                                conformance="protocol-defined")
        _ENABLE_PRIM.append(_register_specified(p, "the conditional enable: parent ∧ local, total, never ⊥"))
    return _ENABLE_PRIM[0]


def l_bot(w: int) -> PrimitiveDefinition:
    """`LBot<w>_v1() -> Value<w+1>`: the constant `enc(⊥) = 2^w` — the one named inactive constant per width (never an ad-hoc
    bit pattern)."""
    p = _BOTS.get(w)
    if p is None:
        p = PrimitiveDefinition(f"LBot{w}", 1, (), Value(w + 1), (lambda _b=enc_bot(w): _b), doc=l_bot.__doc__ or "",
                                conformance="protocol-defined")
        p.family = "Const"  # type: ignore[attr-defined]
        _BOTS[w] = _register_specified(p, "the named inactive constant")
    return p


def _gather_key(w: int, N: int, oor_payload: int) -> tuple[int, ...]:
    """`(w, N)` at the extension of the width's id-only family (`_gather_families`), else `(w, N, extension)`, named in the id
    (`LGatherRow32x16offffffff`): originals of one width and size never depend on which was lifted first."""
    fam = _gather_families().get(w)
    return (w, N) if fam is not None and registered_oor_extension(fam(N)) == oor_payload else (w, N, oor_payload)


def l_gather_row_of(original: PrimitiveDefinition) -> PrimitiveDefinition:
    """The lifted gather of a registered original gather family: `(w, N)` from its shape, the out-of-range extension read from
    the original itself (`registered_oor_extension`), part of the id unless it is the id-only family's (`_gather_key`)."""
    shape = _gather_shape(original)
    if shape is None:
        raise TypeError(f"{original.id} is not a registered gather family")
    w, N = shape
    return _l_gather_row(w, N, registered_oor_extension(original), original.id)


def l_gather_row(w: int, N: int, oor_payload: int | None = None) -> PrimitiveDefinition:
    """`LGatherRow<w>x<N>_v1(idx: Value<33>, rows: Array<N, Value<w+1>>) -> Value<w+1>`: `idx` ⊥ → ⊥; `idx`
    Active (in range) → `rows[idx]` — which may itself be ⊥ (a read of an inactive row is that row's ⊥, not an error, and a ⊥
    row elsewhere does **not** poison the read: this is why it is not strict); any invalid operand anywhere (index or any
    row, PAD-01 rule 2 taken literally) → `INV_w`.  Total extension on an Active **out-of-range** index: `0‖oor` — the
    original gather's registered total extension, **read from the original** (`GatherBf16x{N}`: 0x7FFF; `GatherF32x{N}`:
    0x7FC00000), lifted — never `INV` (range is a boundary fact, checked by `check_boundary(index_range=N)`; no
    valid operand maps to INV).  This id-only entry (the codec's lazy resolver, tests) constructs the registered original
    family of payload width `w` and defers to `l_gather_row_of`; a width with no registered family is refused (`TypeError`),
    and an explicit `oor_payload` may only *confirm* the original's extension (`ValueError` otherwise) — no constant can be
    invented through this API."""
    fam = _gather_families().get(w)
    if fam is None:
        raise TypeError(f"LGatherRow{w}: no registered original gather family for payload width {w} — nothing is invented")
    p = l_gather_row_of(fam(N))
    if oor_payload is not None and oor_payload != p.oor_payload:  # type: ignore[attr-defined]
        raise ValueError(f"LGatherRow{w}: out-of-range extension {oor_payload:#x} != the original {p.lifted_from_id}'s "  # type: ignore[attr-defined]
                         f"{p.oor_payload:#x}")  # type: ignore[attr-defined]
    return p


def _l_gather_row(w: int, N: int, oor_payload: int, original_id: str) -> PrimitiveDefinition:
    key = _gather_key(w, N, oor_payload)
    name = f"LGatherRow{w}x{N}" if len(key) == 2 else f"LGatherRow{w}x{N}o{oor_payload:x}"
    p = _GATHERS.get(key)
    if p is None:
        bot_w, inv_w = enc_bot(w), inv(w)
        mask32 = (1 << 32) - 1

        def _index(idx: int) -> int | None:
            """None for ⊥; the signed i32 payload otherwise (validity already checked)."""
            if idx >> 32:
                return None
            t = idx & mask32
            return t - (1 << 32) if t & 0x80000000 else t

        def ev(idx: int, *rows: int, _N=N, _w=w, _bot=bot_w, _inv=inv_w, _oor=oor_payload) -> int:
            if _classify((idx,) + tuple(rows), (32,) + (_w,) * _N) == 2:
                return _inv
            t = _index(idx)
            if t is None:
                return _bot
            return rows[t] if 0 <= t < _N else _oor

        def ev_lazy(get, _N=N, _w=w, _bot=bot_w, _inv=inv_w, _oor=oor_payload) -> int:
            idx = get(0)
            if _classify((idx,), (32,)) == 2:
                return _inv
            for j in range(_N):  # PAD-01 rule 2, literally: an INV anywhere among the operands is INV (cost noted in PAD-01)
                if _classify((get(1 + j),), (_w,)) == 2:
                    return _inv
            t = _index(idx)
            if t is None:
                return _bot
            return get(1 + t) if 0 <= t < _N else _oor

        p = PrimitiveDefinition(name, 1, (("idx", Value(33)), ("rows", Array(N, Value(w + 1)))), Value(w + 1), ev,
                                doc=l_gather_row.__doc__ or "", conformance="protocol-defined")
        p.evaluate_lazy = ev_lazy  # type: ignore[attr-defined]
        p.oor_payload = oor_payload  # type: ignore[attr-defined]
        p.lifted_from_id = original_id  # type: ignore[attr-defined]
        _GATHERS[key] = _register_specified(p, "indexing: an inactive index yields ⊥; an inactive unselected row does not poison the read")
    return p


_FIXED_SELECTS: dict[tuple[int, int], PrimitiveDefinition] = {}


def l_fixed_select(E: int, w: int) -> PrimitiveDefinition:
    """`LFixedSelect<E>x<w>_v1(r: Value<33>, z: Array<E, Value<w+1>>) -> Value<w+1>` — PADDING-DECISION §3 `FixedSelect(r_j, z_0..z_{E-1})`
    : the fixed selection among **computed** lifted values (the E expert slots' outputs) by a
    **computed** route id.  `r` ⊥ → ⊥ (an inactive token selects nothing); `r` Active with `0 ≤ r < E` → `z_r` bit for bit, INCLUDING `z_r = ⊥`
    (selecting an inactive slot yields that slot's ⊥ — the check that then fails is the slot's, not a wrap); an unselected ⊥ row has no
    effect (why it is not strict); `r` Active OUT OF RANGE → `INV_w`, a NAMED relation failure (never a silent wrap, never a default slot — unlike `LGatherRow`, which reads the ORIGINAL gather family's registered total extension
    (`GatherBf16xN`: 0x7FFF) because a weight-bank / vocabulary read has one; a selection among computed outputs has no original to inherit
    from, so its extension is authored here, as a failure); any noncanonical operand anywhere → `INV_w` (A3: INV beats ⊥).  One object per
    `(E, w)`: the same Definition, identities and interfaces for every route."""
    key = (E, w)
    p = _FIXED_SELECTS.get(key)
    if p is None:
        if E < 1 or w < 1:
            raise ValueError(f"LFixedSelect{E}x{w}: E ≥ 1 and w ≥ 1 required")
        bot_w, inv_w = enc_bot(w), inv(w)
        mask32 = (1 << 32) - 1

        def _route(r: int) -> int | None:
            if r >> 32:
                return None
            t = r & mask32
            return t - (1 << 32) if t & 0x80000000 else t

        def ev(r: int, *z: int, _E=E, _w=w, _bot=bot_w, _inv=inv_w) -> int:
            if _classify((r,) + tuple(z), (32,) + (_w,) * _E) == 2:
                return _inv
            t = _route(r)
            if t is None:
                return _bot
            return z[t] if 0 <= t < _E else _inv  # out of range: NAMED (INV), no wrap, no default slot

        def ev_lazy(get, _E=E, _w=w, _bot=bot_w, _inv=inv_w) -> int:
            r = get(0)
            if _classify((r,), (32,)) == 2:
                return _inv
            for j in range(_E):  # PAD-01 rule 2: an INV anywhere among the operands is INV
                if _classify((get(1 + j),), (_w,)) == 2:
                    return _inv
            t = _route(r)
            if t is None:
                return _bot
            return get(1 + t) if 0 <= t < _E else _inv

        p = PrimitiveDefinition(f"LFixedSelect{E}x{w}", 1, (("r", Value(33)), ("z", Array(E, Value(w + 1)))), Value(w + 1), ev,
                                doc=l_fixed_select.__doc__ or "", conformance="protocol-defined")
        p.evaluate_lazy = ev_lazy  # type: ignore[attr-defined]
        p.oor_named = "INV"  # type: ignore[attr-defined]
        _FIXED_SELECTS[key] = _register_specified(
            p, "fixed indexed selection among COMPUTED values: one object for every route; out of range is a named failure (INV), never a "
               "default slot; an unselected inactive slot does not poison the read")
    return p


def l_step_live(E: int) -> PrimitiveDefinition:
    """`LStepLive<E>_v1(prev_tok: Value<33>, eos_set: Array<E, Value<33>>) -> Value<2>`: `Active(1)` iff
    `tag(prev_tok) = 0 ∧ payload(prev_tok) ∉ eos_set`; `Active(0)` otherwise (including `prev_tok = ⊥`); never ⊥; any
    invalid operand → `INV_1`.  The continuation relation: activity of step `s+1` is a function of the **value** sampled at
    step `s`.  `eos_set` is a structural Const array (always Active in any Program built here); the total extension for a ⊥
    member — it never matches — is a totality corner no honest Program reaches, recorded in LIFTING-STATUS §2 (note N1)."""
    p = _STEP_LIVE.get(E)
    if p is None:
        def ev(prev: int, *eos: int, _E=E) -> int:
            if _classify((prev,) + tuple(eos), (32,) * (1 + _E)) == 2:
                return 0b11
            if prev >> 32:
                return 0
            return 0 if any(e == prev for e in eos) else 1

        p = PrimitiveDefinition(f"LStepLive{E}", 1, (("prev_tok", Value(33)), ("eos_set", Array(E, Value(33)))), Value(2), ev,
                                doc=l_step_live.__doc__ or "", conformance="protocol-defined")
        _STEP_LIVE[E] = _register_specified(p, "state transition: the continuation relation reads the previous committed token")
    return p


# ---------------------------------------------------------------------------------------------------------
# Continuation_v1: the controller family -- `b_t`, `e_t`, `o_t` as REGISTERED gates
# beside the unit's `r_{t+1}` (`s=<s>/live` = LStepLive) and `u_{t+1}` (`s=<s>/tok_in` = LSelect), which compute exactly `run` / `in`
# of the rule (`verity_vllm/program/lifting/continuation.py` `chain_bits`, exhaustively tested).  Slot `t` <-> event `s = t - 1`.
# ---------------------------------------------------------------------------------------------------------

CONTINUATION_VERSION = "continuation-v1"
CONTINUATION_FAMILIES = ("is_eos", "emit", "out")  # the gates this chain ADDS per slot (`live` / `tok_in` are `run` / `in`)
_IS_EOS: dict[int, PrimitiveDefinition] = {}
_EMIT_NEXT: dict[int, PrimitiveDefinition] = {}


def l_is_eos(E: int) -> PrimitiveDefinition:
    """`LIsEOS<E>_v1(tok: Value<33>, eos_set: Array<E, Value<33>>) -> Value<2>`: `b_t = [tag(tok) = 0] AND [payload(tok) in
    EOS_pinned]` -- TOTAL over the lifted encoding: `Active(1)` iff the token is an Active member of the pinned set, `Active(0)` otherwise
    (`IsEOS(bot) = 0` by the tag GATE, not a convention; a noncanonical `1||nonzero` token is invalid -> `INV_1`).  Never bot.  The EOS set is
    the Program's pinned constant array, never a tokenizer read at check time."""
    p = _IS_EOS.get(E)
    if p is None:
        def ev(tok: int, *eos: int, _E=E) -> int:
            if _classify((tok,) + tuple(eos), (32,) * (1 + _E)) == 2:
                return 0b11
            if tok >> 32:
                return 0
            return 1 if any(e == tok for e in eos) else 0

        p = PrimitiveDefinition(f"LIsEOS{E}", 1, (("tok", Value(33)), ("eos_set", Array(E, Value(33)))), Value(2), ev,
                                doc=l_is_eos.__doc__ or "", conformance="protocol-defined")
        _IS_EOS[E] = _register_specified(p, "Continuation_v1 stop predicate: total over the encoding (the tag test is a gate)")
    return p


def l_emit_next(H: int = 1) -> PrimitiveDefinition:
    """`LEmitNext[H=<h>]_v1(emit: Value<2>, is_eos: Value<2>) -> Value<2>`: `e_{t+1} = h_t AND e_t AND NOT b_t` with the cap term
    `h_t = [t < C]` a construction-time STATIC of the gate (built with `H = 1` at every slot that has a successor; the `H = 0` member exists so
    the final transition can be replayed -- both next bits false -- and is never fitted).  Never bot; an invalid bit -> `INV_1`."""
    if H not in (0, 1):
        raise ValueError(f"H={H!r}: the cap term is a bit")
    p = _EMIT_NEXT.get(H)
    if p is None:
        def ev(e: int, b: int, _H=H) -> int:
            if _classify((e, b), (1, 1)) == 2:
                return 0b11
            if e == 2 or b == 2:  # a bot controller bit is not honest computation: the controller bits are always Active
                return 0b11
            return 1 if (_H and e == 1 and b == 0) else 0

        p = PrimitiveDefinition(f"LEmitNext[H={H}]", 1, (("emit", Value(2)), ("is_eos", Value(2))), Value(2), ev,
                                doc=l_emit_next.__doc__ or "", conformance="protocol-defined")
        _EMIT_NEXT[H] = _register_specified(p, "Continuation_v1 emit transition: e' = h AND e AND NOT b")
    return p


def continuation_gate_count(cap: int, n_eos: int, lag: int) -> int:
    """Gates the Continuation_v1 chain adds to the unlifted Program: the guard (`guard_gate_count`, which includes the root enable
    constant `enable/one` = `enable_gate_count`) + per slot `t = 1..cap` one `is_eos` and one `out`, per slot `t = 1..cap-1` one `emit`
    transition, plus the `emit_1` constant: `3·cap` more.  `state_select_count(...)` = 0 for this chain."""
    return guard_gate_count(cap, n_eos, lag) + 3 * cap


def enable_gate_count(cap: int, n_eos: int, lag: int) -> int:
    """The NEW enable gates of a Continuation_v1 request = 1, the root's constant `enable/one` (`Const2[0x1]`).
    The step enables ARE the guard's `s=<s>/live` gates (counted in `guard_gate_count`, one value in two roles: `run_t` and the enable of
    step t's body) and the enable operand of every lifted gate is WIRING (an operand, not a gate) -- so no `LEnable` gate is added under a
    root whose own enable is the constant 1.  Conditional calls under a conditional parent (A12's expert slots) add one `LEnable` each."""
    return 1


def state_select_count(cap: int, n_eos: int, lag: int) -> int:
    """Explicit state-update sites the padded dense request adds = 0.  Every K/V row `t` of the unrolled Program is produced
    ONCE, by step t's own gates, whose enable is `run_t`: a disabled step's rows are ⊥ by A9.3 (their identities exist and are committed,
    A14) with no separate `LSelect` site -- the enable is the selection.  A preallocated cache with rows rewritten in place (A10.1 `cap`
    rows initialised ⊥) would add one `LSelect` per (row, writer); the Continuation_v1 dense chain has none."""
    return 0


class ContinuationChain:
    """The builder-side Continuation_v1 controller of ONE request (one versioned, builder-owned component): call `slot(t, y)` once per
    token-generation slot in order with the slot's RAW sampled token `y` (`Value<33>`); it appends `slot=<t>/is_eos = LIsEOS(y, eos_set)`,
    `slot=<t>/out = LSelect32(emit_t, y, bot)` (the DESIGNATED output: emit, never run) and, when `t < cap`, `slot=<t+1>/emit = LEmitNext(emit_t,
    is_eos_t)`; `slot=1/emit` is the constant `Active(1)`.  `run` / `in` are the guard's `live` / `tok_in` (the caller builds them; they
    are the same values as the rule's, `continuation.chain_bits`).  `outputs()` is `o[1..cap]` in slot order -- the request's return.
    Nothing here reads a served length: `cap` slots, always."""

    def __init__(self, B: Builder, *, cap: int, eos_set: Coll, bot: Coll, w: int = 32) -> None:
        if cap < 1:
            raise ValueError(f"cap = {cap}: a request has at least one token-generation slot")
        self.B, self.cap, self.eos_set, self.bot = B, int(cap), eos_set, bot
        eos_ord = 0b10 if eos_set.type.elem == Value(w) else 0  # G1 (A11): the EOS set is prescribed structure -- ordinary Value<w> consts
        self.is_eos_prim, self.emit_next_prim, self.sel = with_ordinary(l_is_eos(eos_set.type.n), eos_ord), l_emit_next(1), l_select(w)
        self.emit = B.call(b1.const(2, 1), name="slot=1/emit")  # e_1 = 1 (r_1 = 1 is the unconditionally live prefill event)
        self.outs: list[Coll] = []
        self.is_eos: list[Coll] = []
        self.emits: list[Coll] = [self.emit]
        self.node_names: list[str] = ["slot=1/emit"]

    def slot(self, t: int, y: Coll) -> Coll:
        if t != len(self.outs) + 1 or t > self.cap:
            raise ValueError(f"slot {t} out of order or past the cap {self.cap} (built {len(self.outs)})")
        b = self.B.call(self.is_eos_prim, y, self.eos_set, name=f"slot={t}/is_eos")
        o = self.B.call(self.sel, self.emit, y, self.bot, name=f"slot={t}/out")
        self.is_eos.append(b)
        self.outs.append(o)
        self.node_names += [f"slot={t}/is_eos", f"slot={t}/out"]
        if t < self.cap:
            self.emit = self.B.call(self.emit_next_prim, self.emit, b, name=f"slot={t + 1}/emit")
            self.emits.append(self.emit)
            self.node_names.append(f"slot={t + 1}/emit")
        return o

    def outputs(self) -> Coll:
        if len(self.outs) != self.cap:
            raise ValueError(f"{len(self.outs)} of {self.cap} slots built: the designated outputs are the WHOLE fixed population")
        return Coll(Array(self.cap, Value(33)), concat([o.refs for o in self.outs]))


# --- CANDIDATE PAD-04: the S-migration operand -----------------------------------------

SPLITS_MAX_BATCH = 64  # `topp_geometry.SPLIT_MAX_BATCH`: above it vLLM serves the sampler batch by the sort path (no `_apply_topp_split`)
N_LIVE_BITS = 7  # the unlifted count `n_live_t ∈ [0, 64]` as a `Value<7>` (a CANDIDATE width — PAD-04)
SPLITS_BITS = 6  # stoch's `SPLITS_FOR_V1_WIDTH`: S ∈ {1, 2, 4, 8, 16, 32}
_SPLITS_FOR: dict[int, PrimitiveDefinition] = {}
_L_SPLITS_FOR: dict[int, PrimitiveDefinition] = {}


def _splits_for_v1(n_live: int, num_sms: int) -> int:
    """stoch's `SplitsFor_v1` (lane/r16-stoch c1ec0aea4 `topp_geometry.splits_for_v1`) when the tree has it; otherwise the same
    domain rule (verbatim) over the base tree's `splits_for` loop.  Refuses by name outside `1 ≤ n_live ≤ 64`, `num_SMs ≥ 1`."""
    from verity.ml.definitions.registry import topp_geometry
    f = getattr(topp_geometry, "splits_for_v1", None)
    if f is not None:
        return int(f(n_live, num_sms))
    n, m = int(n_live), int(num_sms)
    if n < 1 or n > topp_geometry.SPLIT_MAX_BATCH:
        raise ValueError(f"SplitsFor_v1: n_live={n} outside the domain 1..{topp_geometry.SPLIT_MAX_BATCH} (0 = no sampling event; > {topp_geometry.SPLIT_MAX_BATCH} = not the split pipeline)")
    if m < 1:
        raise ValueError(f"SplitsFor_v1: num_SMs={m} is not a target profile")
    s = topp_geometry.splits_for(n, m)
    assert 1 <= s <= topp_geometry.SPLIT_MAX_SPLITS and s & (s - 1) == 0 and s < (1 << SPLITS_BITS)
    return s


def splits_for_prim(num_sms: int) -> PrimitiveDefinition:
    """`SplitsForSMS<SMS>_v1(n_live: Value<7>) -> Value<6>` — the UNLIFTED primitive form of `SplitsFor_v1`
    (`S_t = SplitsFor_v1(n_live_t, num_SMs)`, `num_SMs` a target-profile static).  Total on `1 ≤ n_live ≤ 64`; `n_live = 0`
    and `n_live > 64` are refused by name (no launch exists there).  Built here so the lifted form has an
    original to be measured against; registering it as the operand of `TopPMask_v1` is a Program-side change."""
    if int(num_sms) < 1:
        raise ValueError(f"SplitsFor_v1: num_SMs={num_sms} is not a target profile")
    p = _SPLITS_FOR.get(int(num_sms))
    if p is None:
        p = PrimitiveDefinition(f"SplitsForSMS{int(num_sms)}", 1, (("n_live", Value(N_LIVE_BITS)),), Value(SPLITS_BITS),
                                lambda n, _m=int(num_sms): _splits_for_v1(n, _m), doc=splits_for_prim.__doc__ or "",
                                conformance="protocol-defined")
        _SPLITS_FOR[int(num_sms)] = p
    return p


def l_splits_for(num_sms: int) -> PrimitiveDefinition:
    """CANDIDATE PAD-04 — `LSplitsForSMS<SMS>_v1(n_live: Value<8>) -> Value<7>`, proposed SIXTH member of the closed specified
    class.  Rule: `INV` operand → `INV_6`; `⊥` operand → `⊥` (strictness; an honest Program never
    produces a ⊥ count — `n_live_t = Σ_r LIsActive(tok_in_{r,t})` is Active by construction); **`Active(0)` → `⊥`** — the
    specified exception and a new KIND of rule (an Active operand mapped to ⊥): no live row ⇒ no sampler launch ⇒ no sampling
    event ⇒ the split count is not a value of any execution (state/batching); `Active(n)`,
    `1 ≤ n ≤ 64` → `0‖SplitsFor_v1(n, SMS)` (payload agreement with the original, measured); `Active(n > 64)` is **refused by
    name** — the total behaviour is a STRUCTURAL refusal: a lifted Program binding this primitive must have at most 64
    requests (`n_live_t ≤ #requests` by construction), asserted where the Program is built (`assert_splits_domain`), never
    INV and never an invented S.  Hook: the operand is the sum of the steps' INPUT-token activity
    `LIsActive(tok_in_{r,t})`, never of `out_r[t−1]` (which over-counts the EOS-stopped request at `a_r`)."""
    orig = splits_for_prim(num_sms)
    p = _L_SPLITS_FOR.get(int(num_sms))
    if p is None:
        bot6, inv6 = enc_bot(SPLITS_BITS), inv(SPLITS_BITS)

        def ev(n: int, _o=orig, _bot=bot6, _inv=inv6) -> int:
            c = _classify((n,), (N_LIVE_BITS,))
            if c == 2:
                return _inv
            if c == 1:
                return _bot
            if n == 0:
                return _bot  # the specified rule: no live row, no sampling event
            if n > SPLITS_MAX_BATCH:
                raise ValueError(f"LSplitsFor_v1: n_live={n} > {SPLITS_MAX_BATCH}: not the split pipeline — a structural bound "
                                 f"(≤ {SPLITS_MAX_BATCH} requests) the Program builder asserts; no S is invented")
            return _o.evaluate(n)

        p = PrimitiveDefinition(f"LSplitsForSMS{int(num_sms)}", 1, (("n_live", Value(N_LIVE_BITS + 1)),), Value(SPLITS_BITS + 1), ev,
                                doc=l_splits_for.__doc__ or "", conformance="protocol-defined")
        p.lifted_from = orig  # type: ignore[attr-defined]
        p.candidate = "PAD-04"  # type: ignore[attr-defined]
        _L_SPLITS_FOR[int(num_sms)] = _register_specified(p, "state/batching: the empty sampler batch has no event (CANDIDATE PAD-04)")
    return p


def assert_splits_domain(n_requests: int) -> int:
    """The structural bound `LSplitsFor_v1` relies on: a lifted Program with `n_requests` requests has `n_live_t ≤ n_requests` at
    every step; more than 64 requests is not the split pipeline — refused at build, by name."""
    if not (1 <= int(n_requests) <= SPLITS_MAX_BATCH):
        raise ValueError(f"{n_requests} requests: the split pipeline serves at most {SPLITS_MAX_BATCH} live rows per step "
                         f"(topp_split.SPLIT_MAX_BATCH); a Program binding LSplitsFor_v1 must be built with 1..{SPLITS_MAX_BATCH} requests")
    return int(n_requests)


def specified_ids() -> frozenset[str]:
    """The closed list of specified-class primitive ids materialised so far (`LSplitsForSMS<SMS>_v1` is the PAD-04 candidate)."""
    return frozenset(_SPECIFIED_IDS)


# ---------------------------------------------------------------------------------------------------------
# G1 — the ORDINARY class (SCOPE-NOTE §6: "weights and other values that never become inactive keep
# their ordinary representations"): a PRESCRIBED input of the Program (a weight, the prompt, the EOS set, an `Input<w>` gate -- a value
# produced by no body) keeps `Value<w>`; every COMPUTED node of a padded Program is lifted (`Value<w+1>`; its tag is identically 0 when the
# producing call is always enabled).  A lifted gate reads an ordinary operand through the SAME relation with that operand declared at its
# ordinary type (`with_ordinary`: an ordinary word `v < 2^w` IS the Active word `0‖v` numerically -- the evaluators are class-agnostic).
# One operand is one Type: an ordinary parameter whose leaves share an operand (or the return) with lifted leaves is itself lifted --
# the fixpoint `accept_classes` decides the class vector of a body from the classes its caller can offer (B11 of v0.7).
# ---------------------------------------------------------------------------------------------------------

ORDINARY, LIFTED = "ordinary", "lifted"
ORD_STATIC = "ORD"  # the class vector of a lifted composite as a static: the bit mask of its ORDINARY parameters (0 = every parameter lifted)
_ORD_PRIMS: dict[tuple[str, int], PrimitiveDefinition] = {}


def class_type(t: Type, cls: str) -> Type:
    """The representation type of an unlifted `t` in class `cls`."""
    if cls == LIFTED:
        return lift_type(t)
    if cls == ORDINARY:
        return t
    raise ValueError(cls)


def ord_mask(classes: Sequence[str]) -> int:
    return sum(1 << i for i, c in enumerate(classes) if c == ORDINARY)


def classes_of_mask(mask: int, n: int) -> tuple[str, ...]:
    if mask >> n:
        raise ValueError(f"ORD mask {mask:#x} names a parameter beyond the {n} of the signature")
    return tuple(ORDINARY if (mask >> i) & 1 else LIFTED for i in range(n))


class _OrdinaryOperandsPrimitive(PrimitiveDefinition):
    """`<lifted id>{ORD=<mask>}`: the lifted primitive with the parameters in `mask` declared at their ORDINARY type.  `name` is the
    lifted primitive's id (so a query case written for `Lifted[F_v1]_v2` matches it by name -- gateset NameSel.matches), the id carries
    the mask as a record key (so every id reader that splits at `{` sees the lifted family, and `unlifted_id` strips it)."""
    ord: int = 0

    @property
    def id(self) -> str:  # type: ignore[override]
        return f"{self.name}{{{ORD_STATIC}={self.ord}}}"


def with_ordinary(p: PrimitiveDefinition, mask: int) -> PrimitiveDefinition:
    """`p` with the parameters named by `mask` (bit i = parameter i) at their ordinary type `Value<w>` instead of `Value<w+1>`: the SAME
    evaluator (an ordinary word is the Active word of the same value), the same return.  `mask = 0` is `p` itself.  The enable parameter
    is not a data operand (refused); a Boolean lifted parameter (`Value<2>`) is refused by `unlift_type`'s width floor."""
    mask = int(mask)
    if mask == 0:
        return p
    key = (p.id, mask)
    op = _ORD_PRIMS.get(key)
    if op is not None:
        return op
    if isinstance(p, _OrdinaryOperandsPrimitive):
        raise TypeError(f"{p.id} already declares ordinary operands")
    n = len(p.params)
    if mask >> n:
        raise ValueError(f"{p.id}: ORD mask {mask:#x} names a parameter beyond its {n}")
    params = []
    for i, (nm, t) in enumerate(p.params):
        if (mask >> i) & 1:
            if nm == ENABLE_PARAM and t == ENABLE_T:
                raise TypeError(f"{p.id}: the enable is not a data operand -- it has no ordinary class")
            params.append((nm, unlift_type(t)))
        else:
            params.append((nm, t))
    op = _OrdinaryOperandsPrimitive(p.id, p.version, tuple(params), p.ret, p.evaluate,
                                    doc=f"{p.id} with ordinary (unwidened) operands {mask:#x} (LIFTING-SPEC v0.7 A11 / G1): the same relation",
                                    conformance=p.conformance, register=False, tables=p.tables)
    op.ord = mask
    for attr in ("lifted_from", "lift_class", "lift_enabled", "lifted_v1", "lift_reason", "family"):
        if hasattr(p, attr):
            setattr(op, attr, getattr(p, attr))
    op.ordinary_of = p  # type: ignore[attr-defined]
    REGISTRY.add(op)
    _ORD_PRIMS[key] = op
    return op


_ORD_KEY_RE = re.compile(r",?ORD=\d+")


def unlifted_id(spec: str) -> str:
    """The unlifted spec id of a padded-v1 id: `Lifted[X_vN]_v2{K=1,ORD=2}` -> `X_vN{K=1}`, `Lifted[F_v1]_v2{ORD=1}` -> `F_v1`,
    `LStepLive1_v1{ORD=2}` -> `LStepLive1_v1`; anything else unchanged (the one rule for `unlift`, PAD-03 and Lift(Q))."""
    m = re.match(r"^Lifted\[(.+)\]_v\d+(\{.*)?$", spec, re.S)
    inner, rest = (m.group(1), m.group(2) or "") if m else (spec.split("{", 1)[0], ("{" + spec.split("{", 1)[1]) if "{" in spec else "")
    rest = _ORD_KEY_RE.sub("", rest)
    if rest in ("{", "{}"):
        rest = ""
    return inner + rest


def member_id(p: Any) -> str:
    """The relation a lifted gate computes AT ENABLE = 1, by id: for a `Lifted[X]_v2` whose enabled relation is a specified member
    (`LSelect…`, `LGatherRow…`, `LFixedSelect…`, `LStepLive…`) that member's id, else the gate's own id (an `{ORD=…}` suffix stripped).
    The specified class is reached THROUGH the v2 wrapper under B7 (`Lifted[GatherBf16x4_v1]_v2` computes `LGatherRow16x4_v1` when
    enabled), so a closure test asks this, not `fn.id`."""
    inner = getattr(p, "lifted_v1", None)
    pid = (getattr(inner, "id", None) if inner is not None else None) or getattr(p, "id", "") or ""
    return pid.split("{ORD=", 1)[0]


def is_lifted_prim(p: Any) -> bool:
    """Does this gate's definition produce a LIFTED word (`Value<w+1>` with a tag): a `Lifted[...]` primitive (with or without ordinary
    operands) or a specified member?  False for an ordinary gate -- `Input<w>`, an unwidened constant (the EOS set, `enable/one`) --
    whose word has no tag (G1, A11)."""
    pid = getattr(p, "id", "") or ""
    return pid.startswith("Lifted[") or pid.split("{", 1)[0] in _SPECIFIED_IDS


def _sources(refs: Refs) -> set[tuple[str, int]]:
    """The `(space, idx)` sources of an operand's leaves (`("p", i)` parameter i, `("n", k)` node k) -- read STRUCTURALLY (an `Affine` /
    `Strided` run is one source, a `Concat` the union of its parts), never by enumerating leaves: a weights operand has thousands."""
    if isinstance(refs, (Affine, Strided)):
        return {(refs.space, refs.idx)}
    if isinstance(refs, Concat):
        out: set[tuple[str, int]] = set()
        for part in refs.parts:
            out |= _sources(part)
        return out
    return {(space, idx) for space, idx, _ in refs}


def _is_ordinary_node(node: Node) -> bool:
    """A node whose output is ordinary in every padded Program: an `Input<w>` gate (a prescribed value, produced by no body)."""
    return getattr(node.fn, "family", None) == "Input"


def _operand_class(refs: Refs, pcls: Sequence[str], body, ncls: dict[int, str] | None = None) -> str:
    """ORDINARY iff every leaf of the operand comes from an ordinary parameter or an ordinary node (an `Input<w>` gate not forced lifted)."""
    for space, idx in _sources(refs):
        if space == "p":
            if pcls[idx] != ORDINARY:
                return LIFTED
        elif space == "n":
            if not _is_ordinary_node(body.nodes[idx]) or (ncls is not None and ncls.get(idx) == LIFTED):
                return LIFTED
        else:
            return LIFTED
    return ORDINARY


def _force(refs: Refs, pcls: list[str], ncls: dict[int, str]) -> bool:
    """Every ORDINARY parameter / ordinary node read by `refs` becomes LIFTED; True iff something changed."""
    changed = False
    for space, idx in _sources(refs):
        if space == "p" and pcls[idx] == ORDINARY:
            pcls[idx] = LIFTED
            changed = True
        elif space == "n" and ncls.get(idx) == ORDINARY:
            ncls[idx] = LIFTED
            changed = True
    return changed


def infer_classes(body, requested: Sequence[str]) -> tuple[tuple[str, ...], dict[int, str]]:
    """The class vector a body ACCEPTS given the classes its caller offers (LIFTED for a parameter is always accepted; ORDINARY is kept
    unless the body needs the parameter lifted), and the class of each of its ordinary-by-nature nodes (`Input<w>` gates: ordinary unless
    forced): (i) one operand is one Type -- a parameter / input gate whose leaves share an operand with lifted leaves is lifted; (ii) the
    return is lifted -- a parameter / input gate it reads directly is lifted; (iii) a scan's carry is lifted (fed back from the step's output)
    -- a parameter / input gate in the init is lifted; (iv) a composite callee accepts its own classes (a hand lift declares them): a
    parameter / input gate feeding an argument the callee needs lifted is lifted.  Monotone (ORDINARY -> LIFTED only): the fixpoint is
    reached in at most |params| + |inputs| rounds."""
    pcls = list(requested)
    if len(pcls) != len(body.params):
        raise ValueError(f"{len(pcls)} classes for {len(body.params)} parameters")
    ck = (id(body), tuple(pcls))
    hit = _INFER_CACHE.get(ck)
    if hit is not None and hit[0] is body:
        return hit[1], dict(hit[2])
    ncls = {k: ORDINARY for k, n in enumerate(body.nodes) if _is_ordinary_node(n)}
    changed = True
    while changed:
        changed = False
        for node in body.nodes:
            acls = [_operand_class(a.refs, pcls, body, ncls) for a in node.args]
            for a, c in zip(node.args, acls):
                if c == LIFTED and _force(a.refs, pcls, ncls):
                    changed = True
            if node.form == "scan" and node.args and _force(node.args[0].refs, pcls, ncls):
                changed = True
            if isinstance(node.fn, SpecializedDefinition):
                acls = [_operand_class(a.refs, pcls, body, ncls) for a in node.args]
                want = list(acls)
                if node.form == "scan":
                    want[0] = LIFTED
                acc = lifted_param_classes(lift_def(node.fn, want))
                for a, c_have, c_acc in zip(node.args, acls, acc):
                    if c_acc == LIFTED and c_have == ORDINARY and _force(a.refs, pcls, ncls):
                        changed = True
        if _force(body.ret, pcls, ncls):
            changed = True
    _INFER_CACHE[ck] = (body, tuple(pcls), dict(ncls))  # the body is held so its id stays its own
    return tuple(pcls), ncls


_INFER_CACHE: dict[tuple, tuple] = {}


def accept_classes(body, requested: Sequence[str]) -> tuple[str, ...]:
    """The parameter classes a body accepts given the offered ones (`infer_classes`)."""
    return infer_classes(body, requested)[0]


def lifted_param_classes(lsp: FunctionDefinition) -> tuple[str, ...]:
    """The class vector of a lifted callee's DATA parameters (the enable excluded): from its `ORD` binding (a replay lift), or by comparing
    its signature with its twin's (a hand lift: a parameter typed as the twin's is ordinary, as `lift_type` of it lifted)."""
    if isinstance(lsp, PrimitiveDefinition):
        n = len(lsp.params) - (1 if has_enable(lsp) else 0)
        return classes_of_mask(getattr(lsp, "ord", 0), n)
    params_l, _ = lsp.signature()
    n = len(params_l) - (1 if has_enable(lsp) else 0)
    b = getattr(lsp, "bindings", {})
    if ORD_STATIC in b:
        return classes_of_mask(int(b[ORD_STATIC]), n)
    twin = getattr(lsp.defn, "lifted_from", None)
    if twin is None:
        raise TypeError(f"{lsp.id}: not a lifted callee (no ORD binding, no twin)")
    tw = bind(twin, **{k: v for k, v in lsp.bindings.items() if k in twin.statics})
    params_u, _ = tw.signature()
    out = []
    for (nm, tl), (_, tu) in zip(params_l[:n], params_u):
        if tl == tu:
            out.append(ORDINARY)
        elif tl == lift_type(tu):
            out.append(LIFTED)
        else:
            raise TypeError(f"{lsp.id}: parameter {nm!r} typed {tl!r} is neither the twin's {tu!r} nor its lift")
    return tuple(out)


# ---------------------------------------------------------------------------------------------------------
# composites (§3.3): lifting by node replay — identical node indices and operand runs
# ---------------------------------------------------------------------------------------------------------

_LIFTED_COMPOSITES: dict[str, CompositeDefinition] = {}
_UNLIFTED_OF: dict[str, CompositeDefinition] = {}
_HAND_LIFTS: dict[str, CompositeDefinition] = {}  # unlifted composite id (Name_vN) -> its HAND-AUTHORED padded-v1 lift


def register_hand_lift(original: CompositeDefinition, lifted: CompositeDefinition) -> CompositeDefinition:
    """MOE-PAD route A: declare `lifted` as THE padded-v1 lift of the composite `original` — honoured on BOTH paths,
    `lift_composite`/`lift_callee` (registry composites, the python-body path) and `_lift_decoded`/`lift_def` (descriptor-decoded
    specializations: GP-01 `request_component`, `codec.decode_program`) — instead of the node replay `Lifted[original]_v2`.  Use: a fixed-slot
    twin `MoeFfnFixed_v1{E,TOPK,VPT,H,I}` (router -> ALL E expert bodies -> select by the router ids -> MoeSum; value-equal to MOE-01 on Active
    inputs) whose strict replay would be an all-Active Program, and whose padded meaning (the unselected slots DISABLED, bot by A9.3) is the
    hand-authored `LMoeFfn_v1`.  SOUNDNESS, by name, at registration: same statics (names; a binding of the twin is a binding of the lift);
    neither is a `Lifted[...]` object; one lift per id (a second registration with a different object is refused).  Per BINDING, at
    `lift_def`/`lift_composite`-bind time (`_check_hand_lift`): the lift's signature == `lift_type` of the twin's params + `("enable",
    Value<2>)` LAST (B7) and `lift_type` of the twin's return — a lift whose interface is not the twin's lifted interface is refused by name
    (it could not stand where `Lifted[twin]_v2` stands: PAD-03 correspondence, `Lift(Q)` id matching modulo the wrapper).  The lift's BODY
    is the author's (the padded MoE's construction, with its N-pad negatives) — this hook does not check it."""
    if not isinstance(original, CompositeDefinition) or not isinstance(lifted, CompositeDefinition):
        raise TypeError(f"register_hand_lift: composites only ({type(original).__name__}, {type(lifted).__name__})")
    oid, lid = f"{original.name}_v{original.version}", f"{lifted.name}_v{lifted.version}"
    if oid.startswith("Lifted[") or lid.startswith("Lifted["):
        raise TypeError(f"register_hand_lift({oid}, {lid}): a hand lift maps an UNLIFTED composite to a hand-authored one")
    if tuple(original.statics) != tuple(lifted.statics):
        raise TypeError(f"register_hand_lift({oid}, {lid}): statics differ {original.statics} != {lifted.statics} (a binding of the twin must be a "
                        f"binding of its lift)")
    prev = _HAND_LIFTS.get(oid)
    if prev is not None and prev is not lifted:
        raise TypeError(f"register_hand_lift({oid}): already registered to {prev.name}_v{prev.version}; one lift per id")
    if oid in _LIFTED_COMPOSITES and _LIFTED_COMPOSITES[oid] is not lifted:
        raise TypeError(f"register_hand_lift({oid}): its replay lift Lifted[{oid}]_v2 was already built; register before the first lift")
    _HAND_LIFTS[oid] = lifted
    _LIFTED_COMPOSITES[oid] = lifted
    _UNLIFTED_OF[lid] = original
    lifted.lifted_from = original  # type: ignore[attr-defined]
    lifted.hand_lift = True  # type: ignore[attr-defined]
    return lifted


def hand_lift_of(defn_id: str) -> CompositeDefinition | None:
    """The registered hand lift of `Name_vN`, or None."""
    return _HAND_LIFTS.get(defn_id)


def _check_hand_lift(lsp: SpecializedDefinition, params_u: Sequence[tuple[str, Type]], ret_u: Type, oid: str,
                     requested: Sequence[str] | None = None) -> SpecializedDefinition:
    """Per binding: the hand lift's signature must be the twin's signature with every parameter at its lifted type OR (G1, A11) at its
    ordinary type -- the author declares the classes --, the enable (B7) LAST, and the lifted return; else refused by name.  A parameter
    the hand lift declares ORDINARY where the caller can only offer a LIFTED value is refused too (a lifted word cannot be narrowed)."""
    params_l, ret_l = lsp.defn.signature(lsp.S())
    params_l = tuple((n, t) for n, t in params_l)
    ok = len(params_l) == len(params_u) + 1 and params_l[-1] == (ENABLE_PARAM, ENABLE_T) and ret_l == lift_type(ret_u)
    classes = []
    if ok:
        for (nl, tl), (nu, tu) in zip(params_l[:-1], params_u):
            if nl != nu or tl not in (tu, lift_type(tu)):
                ok = False
                break
            classes.append(ORDINARY if tl == tu else LIFTED)
    if not ok:
        want = tuple((n, lift_type(t)) for n, t in params_u) + ((ENABLE_PARAM, ENABLE_T),)
        raise TypeError(f"hand lift {lsp.id} of {oid}: signature {params_l} -> {ret_l} is not the twin's lifted interface {want} -> "
                        f"{lift_type(ret_u)} (LIFTING-SPEC v0.7 B7/A11: each param lifted or ordinary, ('enable', Value<2>) last)")
    if requested is not None:
        for (nm, _), c_hand, c_req in zip(params_u, classes, requested):
            if c_hand == ORDINARY and c_req == LIFTED:
                raise TypeError(f"hand lift {lsp.id} of {oid}: parameter {nm!r} is declared ORDINARY but the call site offers a lifted value "
                                f"(a lifted word cannot be narrowed; declare it lifted)")
    return lsp


def lift_callee(fn: FunctionDefinition) -> FunctionDefinition:
    if isinstance(fn, PrimitiveDefinition):
        return lifted_prim(fn)
    if isinstance(fn, SpecializedDefinition):
        return lift_def(fn)
    if isinstance(fn, CompositeDefinition):
        return lift_composite(fn)
    raise TypeError(fn)


def lift_composite(defn: CompositeDefinition) -> CompositeDefinition:
    """`Lifted[<Name_vN>]_v1`: the same statics (the binding record of the lifted composite IS the original's — `Lift` is
    applied to the fully bound original), the lifted signature, and a body that replays the original's nodes with lifted
    callees and lifted types."""
    ld = _LIFTED_COMPOSITES.get(defn.id)
    if ld is not None:
        return ld
    if defn.id.startswith("Lifted["):
        raise TypeError(f"{defn.id} is already lifted")
    if ORD_STATIC in defn.statics:
        raise TypeError(f"{defn.id} has a static named {ORD_STATIC!r}: the name is the lifted class vector's (G1)")

    def _twin(S, _d=defn):
        return bind(_d, **{k: v for k, v in S.__dict__.items() if k != ORD_STATIC})

    def sig(S, _d=defn):
        params, ret = _twin(S).signature()
        params = tuple(params)
        cls = classes_of_mask(int(S.__dict__.get(ORD_STATIC, 0)), len(params))
        return tuple((n, class_type(t, c)) for (n, t), c in zip(params, cls)) + ((ENABLE_PARAM, ENABLE_T),), lift_type(ret)

    def body(B: Builder, S, *args: Coll, _d=defn) -> Coll:
        tw = _twin(S)
        cls = classes_of_mask(int(S.__dict__.get(ORD_STATIC, 0)), len(tw.body.params))
        return replay_lifted(B, tw.body, args[:-1], enable=args[-1], arg_classes=cls)

    ld = CompositeDefinition(f"Lifted[{defn.id}]", LIFT_VERSION, tuple(defn.statics) + (ORD_STATIC,), sig, body,
                             doc=f"enabled ⊥-lift of {defn.id} (LIFTING-SPEC v0.7 A9/B7/A11): same body, every callee lifted, the call's enable "
                                 f"threaded to every node it owns; ORD = the mask of its ordinary (unwidened) parameters", conformance="protocol-defined")
    ld.lifted_from = defn  # type: ignore[attr-defined]
    _LIFTED_COMPOSITES[defn.id] = ld
    _UNLIFTED_OF[ld.id] = defn
    return ld


_LIFTED_DECODED: dict[tuple, SpecializedDefinition] = {}  # (id(decoded spec), ORD mask) → its lift (a decoded object is one per decode/canon)
_LIFTED_DECODED_DEFS: dict[str, CompositeDefinition] = {}


def _is_decoded(sp: SpecializedDefinition) -> bool:
    """A specialization materialised by `codec.decode_program` (or GP-01's `request_component`): its definition has no python
    signature/body (`lambda S: ((), None)`), everything is in the set body."""
    try:
        _, ret = sp.defn.signature(sp.S())
    except Exception:  # noqa: BLE001
        return sp._body is not None
    return ret is None and sp._body is not None


def _lift_decoded(sp: SpecializedDefinition, classes: Sequence[str] | None = None) -> SpecializedDefinition:
    """The lift of a decoded (body-only) specialization: `Lifted[<id>]` with the same bindings (+ `ORD`, G1) and key, the body replayed
    with lifted callees/types (the same replay as `lift_composite`; the registry's python body is not needed — the descriptor's body IS
    the definition of record)."""
    body = sp.body
    cls = tuple(classes) if classes is not None else (LIFTED,) * len(body.params)
    mask = ord_mask(cls)
    cached = _LIFTED_DECODED.get((id(sp), mask))
    if cached is not None:
        return cached
    ld = _LIFTED_DECODED_DEFS.get(sp.defn.id)
    if ld is None:
        ld = CompositeDefinition(f"Lifted[{sp.defn.id}]", LIFT_VERSION, tuple(sp.defn.statics) + (ORD_STATIC,), lambda S: ((), None), lambda *_: None,
                                 doc=f"enabled ⊥-lift of the decoded {sp.defn.id} (LIFTING-SPEC v0.7 A9/B7/A11): the descriptor's body replayed, "
                                     f"every callee lifted, the call's enable threaded; ORD = its ordinary parameters", conformance="protocol-defined",
                                 register=False)
        ld.lifted_from = sp.defn  # type: ignore[attr-defined]
        _LIFTED_DECODED_DEFS[sp.defn.id] = ld
    lsp = SpecializedDefinition(ld, {**sp.bindings, ORD_STATIC: mask}, tuple(sp.key) + ((ORD_STATIC, mask),))
    params_l = tuple((n, class_type(t, c)) for (n, t), c in zip(body.params, cls)) + ((ENABLE_PARAM, ENABLE_T),)
    B = Builder(params_l, lift_type(body.ret_type))
    args = [Coll(class_type(t, c), Affine("p", i, 0, t.leaves)) for i, ((_, t), c) in enumerate(zip(body.params, cls))]
    enable = Coll(ENABLE_T, Affine("p", len(body.params), 0, 1))
    lsp.set_body(B.finish(replay_lifted(B, body, args, enable=enable, arg_classes=cls)))
    lsp.lifted_from = sp  # type: ignore[attr-defined]
    _LIFTED_DECODED[(id(sp), mask)] = lsp
    _LIFTED_DECODED.setdefault(("keep", id(lsp)), sp)  # keep the original alive as long as the lift is cached
    return lsp


def lift_def(sp: SpecializedDefinition, param_classes: Sequence[str] | None = None) -> SpecializedDefinition:
    """The lifted specialization of a complete specialization: `bind(Lifted[defn], **same bindings, ORD=mask)` for a registry composite;
    for a DECODED specialization (a descriptor's definition, body only) the body-replay lift `_lift_decoded`.  `param_classes` (G1, A11):
    the classes the CALLER offers per data parameter (`ORDINARY` for a prescribed input it holds unwidened, `LIFTED` otherwise; None =
    every parameter lifted -- the pre-G1 interface); the result's classes are what the body ACCEPTS (`accept_classes`: never more ordinary
    than offered; read them back with `lifted_param_classes`)."""
    if not sp.complete:
        raise TypeError(f"{sp.id} is not fully bound")
    hand = _HAND_LIFTS.get(sp.defn.id)
    if hand is not None:
        # the hand-authored padded lift (register_hand_lift), on BOTH paths: the twin's interface comes from its python signature (registry
        # composite) or from the descriptor's body (decoded specialization) -- checked per binding
        if _is_decoded(sp):
            params_u, ret_u = tuple(sp.body.params), sp.body.ret_type
        else:
            params_u, ret_u = sp.defn.signature(sp.S())
            params_u = tuple(params_u)
        return _check_hand_lift(bind(hand, **sp.bindings), params_u, ret_u, sp.defn.id, requested=param_classes)
    n = len(sp.body.params) if _is_decoded(sp) else len(tuple(sp.defn.signature(sp.S())[0]))
    req = tuple(param_classes) if param_classes is not None else (LIFTED,) * n
    if len(req) != n:
        raise ValueError(f"{sp.id}: {len(req)} classes for {n} parameters")
    acc = accept_classes(sp.body, req) if ORDINARY in req else req
    if _is_decoded(sp):
        return _lift_decoded(sp, acc)
    return bind(lift_composite(sp.defn), **sp.bindings, **{ORD_STATIC: ord_mask(acc)})


def _remap(r: Refs, args: Sequence[Coll], k0: int) -> Refs:
    """`body`-scoped refs → refs in the replaying builder: `("p", i, leaf)` becomes leaf `leaf` of the *argument* `args[i]`
    (the replay is inlined wherever the original's python body would have been — at a Program root the arguments are the
    Input collections, inside a caller they are that body's parameter refs), and `("n", k, leaf)` becomes `("n", k0 + k,
    leaf)` (the replay's node `k` is builder node `k0 + k`).  Structural forms are kept (an `Affine`/`Strided` run of a
    parameter whose argument is itself one `Affine` run stays one run), so descriptor sizes — hence operand runs — match."""
    if isinstance(r, Affine):
        if r.space == "n":
            return Affine("n", r.idx + k0, r.base, r.count)
        if r.space == "p":
            return args[r.idx].refs.slice(r.base, r.base + r.count)
        return r
    if isinstance(r, Strided):
        if r.space == "n":
            return Strided("n", r.idx + k0, r.base, r.inner, r.outer_stride, r.inner_stride, r.count)
        if r.space == "p":
            a = args[r.idx].refs
            if isinstance(a, Affine):
                return Strided(a.space, a.idx, a.base + r.base, r.inner, r.outer_stride, r.inner_stride, r.count)
            return Explicit(tuple(a[leaf] for _, _, leaf in r))
        return r
    if isinstance(r, Concat):
        return concat([_remap(p, args, k0) for p in r.parts])
    items = []
    for space, idx, leaf in r:
        if space == "n":
            items.append(("n", idx + k0, leaf))
        elif space == "p":
            items.append(args[idx].refs[leaf])
        else:
            items.append((space, idx, leaf))
    return Explicit(tuple(items))


def emit_lifted_node(B: Builder, node: Node, lf: Any, largs: Sequence[Coll], enable: Coll | None) -> Coll:
    """Emit ONE lifted node with `enable` as the callee's last operand (B7): a `call`/`primitive` appends it; a `batch` shares it across
    the members (axis `None`); a `scan` passes it as the last `shared` argument (the lifted step's last parameter).  A callee without an
    enable parameter (`Input<w+1>`, the root-owned controller members) takes none."""
    en = [enable] if (enable is not None and has_enable(lf)) else []
    if node.form in ("primitive", "call"):
        return B.call(lf, *largs, *en, name=node.name)
    if node.form == "batch":
        return B.batch(lf, *largs, *en, axes=tuple(node.axes) + tuple(None for _ in en), n=node.n, name=node.name)
    if node.form == "scan":
        return B.scan(lf, largs[0], xs=largs[1:1 + node.n_xs], shared=list(largs[1 + node.n_xs:]) + en, n=node.n, name=node.name)
    raise ValueError(node.form)


def lifted_callee_for(node: Node, acls: Sequence[str], node_class: str = ORDINARY) -> FunctionDefinition:
    """The lifted callee of `node` reading its arguments at the classes `acls` (G1): a primitive -> `with_ordinary(lifted_prim(fn), mask)`;
    a specialization -> `lift_def(fn, acls)` (a scan's carry is always lifted); an `Input<w>` gate -> itself (ordinary, A11) unless the
    body forced it lifted (`node_class`), then `Input<w+1>`."""
    fn = node.fn
    if isinstance(fn, PrimitiveDefinition):
        if _is_ordinary_node(node):
            return fn if node_class == ORDINARY else lifted_prim(fn)
        return with_ordinary(lifted_prim(fn), ord_mask(acls))
    if isinstance(fn, SpecializedDefinition):
        want = list(acls)
        if node.form == "scan" and want:
            want[0] = LIFTED
        lsp = lift_def(fn, want)
        acc = lifted_param_classes(lsp)
        for i, (c_have, c_acc) in enumerate(zip(acls if node.form != "scan" else want, acc)):
            if c_acc != c_have:
                raise TypeError(f"{fn.id}: argument {i} offered {c_have} but the lift accepts only {c_acc} (accept_classes did not converge)")
        return lsp
    return lift_callee(fn)


def replay_lifted(B: Builder, body, args: Sequence[Coll], enable: Coll | Callable[[int], Coll] | None = None, *,
                  arg_classes: Sequence[str] | None = None) -> Coll:
    """Emit `body`'s nodes into `B` with lifted callees/types.  Node `k` of the replay is node `k` of the original with the
    same gate count (one lifted gate per gate, by induction), so offsets, node indices and operand runs are identical.
    `enable` (B7): the enable every replayed node receives as its last operand — one `Coll` (the call's enable: every node the call
    owns inherits it, `a_local = 1`), or a function of the ORIGINAL node index (a flat body whose segments are owned by different
    enables: `lift_request_component`), or None (legacy lifted-v1 replay; refused if any callee takes an enable).
    `arg_classes` (G1, A11): the class of each parameter as `args` holds it (None = every parameter lifted); MUST be a vector the body
    accepts (`accept_classes`) -- an operand mixing ordinary and lifted leaves is refused by the callee's type check."""
    k0 = len(B.nodes)
    pcls = tuple(arg_classes) if arg_classes is not None else (LIFTED,) * len(body.params)
    acc, ncls = infer_classes(body, pcls)
    if acc != pcls:
        raise TypeError(f"replay: the body needs parameters {[i for i, (a, b) in enumerate(zip(acc, pcls)) if a != b]} lifted but the caller "
                        f"offers them ordinary (accept_classes first)")
    for k, node in enumerate(body.nodes):
        acls = [_operand_class(a.refs, pcls, body, ncls) for a in node.args]
        lf = lifted_callee_for(node, acls, ncls.get(k, LIFTED))
        largs = [Coll(class_type(a.type, c), _remap(a.refs, args, k0)) for a, c in zip(node.args, acls)]
        en = enable(k) if callable(enable) else enable
        if en is None and has_enable(lf):
            raise TypeError(f"replay of {lf.id} needs an enable (LIFTING-SPEC v0.6 B7): every lifted computed gate is owned by an enable")
        emit_lifted_node(B, node, lf, largs, en)
    return Coll(lift_type(body.ret_type), _remap(body.ret, args, k0))


# ---------------------------------------------------------------------------------------------------------
# the lifted padded serving Program (§4) on the Serve_v2 skeleton
# ---------------------------------------------------------------------------------------------------------


@composite("Prefill", 1, ["C", "LP", "GEMM", "ATTENTION", "NORM"],
           lambda S: ((("prompt", Array(S.LP, Value(32))), ("weights", b1.weights_type(S.C))),
                      Tuple(Value(32), Array(S.C["NL"], b1.kv_type(S.C, S.LP)), Array(S.C["NL"], b1.kv_type(S.C, S.LP)))),
           conformance="protocol-defined")
def Prefill(B, S, prompt, weights):
    """The prefill of `Serve_v2` as a registered composite (unlifted; gates verbatim `_forward_positions_v2` + `Final_v2`):
    `(y_0, kc[NL][LP], vc[NL][LP])`.  Registered so the lifted Program can call `Lifted[Prefill_v1]`; `Serve_v1/v2/v3` untouched."""
    C, LP = S.C, S.LP
    fin = bind(b1.FinalV2, C=C, GEMM=S.GEMM, NORM=S.NORM)
    caches, finals = b1._forward_positions_v2(B, S, prompt, weights, 0, LP, None)
    y0 = B.call(fin, finals[-1][0], finals[-1][1], weights["norm"], weights["embed"], name=f"t={LP - 1}/final")
    kc = Coll(Array(C["NL"], b1.kv_type(C, LP)), concat([caches[L][0].refs for L in range(C["NL"])]))
    vc = Coll(Array(C["NL"], b1.kv_type(C, LP)), concat([caches[L][1].refs for L in range(C["NL"])]))
    return tuple_of(y0, kc, vc)


def _row_type(C: dict) -> Array:
    return Array(C["NL"], Array(C["KVH"], Array(C["D"], Value(16))))


@composite("StepBody", 1, ["C", "T", "GEMM", "ATTENTION", "NORM"],
           lambda S: ((("tok", Value(32)),
                       ("kc", Array(S.C["NL"], b1.kv_type(S.C, S.T - 1))),
                       ("vc", Array(S.C["NL"], b1.kv_type(S.C, S.T - 1))),
                       ("weights", b1.weights_type(S.C))),
                      Tuple(Value(32), _row_type(S.C), _row_type(S.C))),
           conformance="protocol-defined")
def StepBody(B, S, tok, kc, vc, weights):
    """One decode step of `Serve_v2` at attention length `T` (position `T−1`) as a registered composite, **unguarded**:
    exactly `PadDecodeStep_v1`'s part (a) — `(y, k_new[NL][KVH][D], v_new[NL][KVH][D])`.  The lifted Program guards the
    *token* with `LStepLive`/`LSelect` and appends the rows as they are ."""
    C, T = S.C, S.T
    NL = C["NL"]
    caches = [(kc[L], vc[L]) for L in range(NL)]
    new_kv, last = b1._forward_positions_v2(B, S, Coll(Array(1, Value(32)), tok.refs), weights, T - 1, 1, caches)
    fin = bind(b1.FinalV2, C=C, GEMM=S.GEMM, NORM=S.NORM)
    y = B.call(fin, last[0][0], last[0][1], weights["norm"], weights["embed"], name=f"t={T - 1}/final")
    k_new = Coll(_row_type(C), concat([new_kv[L][0].refs for L in range(NL)]))
    v_new = Coll(_row_type(C), concat([new_kv[L][1].refs for L in range(NL)]))
    return tuple_of(y, k_new, v_new)


def _lifted_weights_type(C: dict) -> Type:
    return lift_type(b1.weights_type(C))


GUARD_IN = "in"  # DP-pad-01 option (a): the activity guard sits on the step's INPUT token — the implemented candidate
GUARDS_IMPLEMENTED = (GUARD_IN,)
LAGS_IMPLEMENTED = (0, 1)  # DP-pad-02: the engine's post-EOS placeholder rows (vLLM async scheduling: 1); a DECLARED static, no default


def check_lag(lag: int) -> int:
    """DP-pad-02: the serving profile's post-EOS lag is a declared static of the padded Program —
    `LAG = 0` (the step after the EOS is never executed) or `LAG = 1` (vLLM async scheduling: the EOS-stopped request is fed forward
    once more on its EOS token, its sample discarded).  Neither is a default; anything else is refused by name."""
    if lag not in LAGS_IMPLEMENTED:
        raise ValueError(f"LAG={lag!r}: the profile's post-EOS lag must be declared as one of {LAGS_IMPLEMENTED} (DP-pad-02); nothing is assumed")
    return int(lag)


DP_PAD_02 = "DP-pad-02(b), RULED"  # LAG is a declared specialization parameter, no default


def guard_gate_count(cap: int, n_eos: int, lag: int) -> int:
    """Gates the guard adds to the unlifted Program: `|E|` EOS constants + `bot` + the root's enable constant `enable/one`  +
    `(live, tok_in)` per guarded step; steps `1..lag` are unguarded (unconditionally live): `2·(cap − 1 − lag) + |E| + 2`."""
    return 2 * max(cap - 1 - check_lag(lag), 0) + n_eos + 2


def enable_one() -> PrimitiveDefinition:
    """The root's enable, `Const2[0x1]`: the constant Active(1) Boolean word ("the root's enable is the constant 1"); an
    ORDINARY constant of the Program (owned by the always-enabled root), so it is `b1.const`, not a lifted constant."""
    return b1.const(2, 1)


def lserve_validate(C: dict, LP: int, STEPS: int, eos: Sequence[int], guard: str, lag: int) -> tuple[int, ...]:
    """LServe_v1's validity, by name, callable BEFORE the (lazy) specialization is built (`lserve()` calls it eagerly;
    the body calls it again so a direct `bind(LServe, …)` is validated at `Program()` at the latest).  Returns the EOS tuple."""
    if guard not in GUARDS_IMPLEMENTED:
        raise ValueError(f"LServe_v1: GUARD={guard!r} is not implemented — only {GUARD_IN!r} (DP-pad-01 option (a)) is; "
                         "the output guard (option (b)) is a semantic choice recorded in DECISIONS-PENDING DP-pad-01, not code")
    eos_ids = tuple(int(e) for e in eos)
    if not eos_ids:
        raise ValueError("LServe_v1 requires a non-empty declared EOS set")
    for e in eos_ids:
        check_boundary(32, e, what="eos id", prescribed=True)
    if not (1 <= STEPS and LP + STEPS <= C["MAX_POS"]):
        raise ValueError(f"LServe_v1 validity: 1 <= STEPS and LP + STEPS <= MAX_POS (got LP={LP}, STEPS={STEPS}, MAX_POS={C['MAX_POS']})")
    check_lag(lag)
    return eos_ids


@composite("LServe", 2, ["C", "LP", "STEPS", "EOS", "GUARD", "LAG", "GEMM", "ATTENTION", "NORM"],  # v2 = padded-v1 (the enables are in the body)
           lambda S: ((("prompt", Array(S.LP, Value(32))), ("weights", b1.weights_type(S.C))), Array(S.STEPS + 1, Value(33))),
           conformance="protocol-defined")
def LServe(B, S, prompt, weights):
    """The ⊥-lifted padded serving Program (§5) for one request, **guard on the input** (`GUARD = "in"`, DP-pad-01
    option (a) — the spec's §4.2 letter puts the guard on the event's output and is inconsistent with its own §4.3/§6/§7 at
    step `a_r`; the choice is a labelled candidate, not a settled rule):
    `Lifted[Prefill_v1]` emits `out[0]` unguarded (event 0 unconditionally live); for each of the `STEPS`
    structurally present decode steps `s ≥ 1` (`STEPS = cap − 1` under the Build's convention, as `Serve_v2{STEPS}`; the event
    population is `STEPS + 1 = cap = D_MAX_r`: the static is named for what it counts)

        live_s   = LStepLive(g_{s}, eos_set)                -- Active(1) iff the gating token is an Active non-EOS token
        tok_in_s = LSelect(live_s, out[s−1], LBot<32>)      -- the step's input token, ⊥ once the request has stopped
        (y_s, k_s, v_s) = Lifted[StepBody_v1]{T = LP + s}(tok_in_s, kc, vc, weights)
        out[s]   = y_s                                       -- ⊥ by strictness when tok_in_s = ⊥ (no output guard)

    where the gating token `g_s` is the profile's post-EOS lag (DP-pad-02, static `LAG`): `LAG = 0`: `g_s = out[s−1]`; `LAG = 1`
    (vLLM async scheduling: the step after the EOS is executed on the EOS token and its sample discarded): steps `1..LAG` are
    unguarded (`tok_in_1 = out[0]`) and `g_s = tok_in_{s−1}` for `s > LAG` — "the previous step ran on an Active non-EOS token" —
    so step `a_r` runs Active on the EOS (its `out[a_r]` is the committed, discarded sample) and step `a_r + 1` is the first ⊥
    step; a guard reading `out[s−2]` alone would revive the request at `a_r + 2` (the discarded sample is an ordinary token).

    so step `a_r` (the first step after the committed EOS) is ⊥ from its embedding on, "every value of a ⊥ step is ⊥" and
    Commit's `enc(⊥) iff s ≥ a_r` are theorems, and the K/V append is unguarded — an inactive step's rows are ⊥ .
    `EOS` is the declared EOS **set**.  `GUARD = "out"` (option (b)) is refused by name: not implemented, not
    chosen.  Output: `out[0..STEPS]`; its Active prefix is the transcript, its first ⊥ is step `a_r`."""
    C, LP, D_MAX, LAG = S.C, S.LP, S.STEPS, S.LAG
    eos_ids = lserve_validate(C, LP, D_MAX, S.EOS, S.GUARD, LAG)
    NL = C["NL"]
    one = B.call(enable_one(), name="enable/one")  # the root's enable: the constant Active(1) (A9.1: the root is always enabled)
    # G1: the prompt and the weights are PRESCRIBED inputs -- ordinary `Value<w>`; the prefill reads them unwidened
    pre = B.call(lift_def(bind(Prefill, C=C, LP=LP, GEMM=S.GEMM, ATTENTION=S.ATTENTION, NORM=S.NORM), (ORDINARY, ORDINARY)), prompt, weights, one,
                 name="prefill")
    prev, kc, vc = pre[0], pre[1], pre[2]
    eos_consts = [B.call(b1.const(32, e), name=f"eos[{i}]") for i, e in enumerate(eos_ids)]  # the EOS set: ordinary structure (A8/A11)
    eos_set = Coll(Array(len(eos_ids), Value(32)), concat([c.refs for c in eos_consts]))
    step_live = with_ordinary(l_step_live(len(eos_ids)), 0b10)
    sel, bot = l_select(32), B.call(l_bot(32), name="bot32")
    # Continuation_v1: the controller family beside the guard -- `slot=<t>/is_eos`, `slot=<t>/emit`, `slot=<t>/out` per slot t = s + 1; the
    # request's RETURN is the designated outputs o_t = Select(emit_t, y_t, bot) (the discarded async token never leaves the output interface)
    cont = ContinuationChain(B, cap=D_MAX + 1, eos_set=eos_set, bot=bot)
    cont.slot(1, prev)
    gate = prev  # the gating token of the next step: out[s-1] under LAG=0, tok_in_{s-1} under LAG=1
    for s in range(1, D_MAX + 1):
        T = LP + s
        body = lift_def(bind(StepBody, C=C, T=T, GEMM=S.GEMM, ATTENTION=S.ATTENTION, NORM=S.NORM), (LIFTED, LIFTED, LIFTED, ORDINARY))
        if s <= LAG:
            tok_in, en = prev, one  # unconditionally live: no guard nodes (the prefill's event is always executed, and so is its placeholder)
        else:
            live = B.call(step_live, gate, eos_set, name=f"s={s}/live")
            tok_in = B.call(sel, live, prev, bot, name=f"s={s}/tok_in")
            en = live  # `run_{s+1} = live_s` IS the step's enable; its body is ⊥ by A9.3 whatever it reads
        r = B.call(body, tok_in, kc, vc, weights, en, name=f"s={s}/step")
        gate = tok_in if LAG else r[0]
        prev = r[0]
        # structural, unguarded append : every step, every layer
        kc = Coll(Array(NL, lift_type(b1.kv_type(C, T))), concat([concat([kc[L].refs, r[1][L].refs]) for L in range(NL)]))
        vc = Coll(Array(NL, lift_type(b1.kv_type(C, T))), concat([concat([vc[L].refs, r[2][L].refs]) for L in range(NL)]))
        cont.slot(s + 1, prev)
    return cont.outputs()


def lserve(C: dict, LP: int, STEPS: int, eos: Sequence[int], *, lag: int, gemm: dict | None = None, attention: dict | None = None,
           norm: dict | None = None, guard: str = GUARD_IN) -> SpecializedDefinition:
    """`LServe_v1` bound for one profile (default: the Ampere eager profile, as `b1.serve_v2`), the input guard and the DECLARED
    post-EOS lag; validated EAGERLY: every refusal fires here, by name, before the lazy specialization exists."""
    eos_t = lserve_validate(C, LP, STEPS, eos, guard, lag)
    # LAG is bound as a LABELLED candidate — `DP-pad-02(b), pending` — never rendered as an agreed baseline
    return bind(LServe, C=C, LP=LP, STEPS=STEPS, EOS=eos_t, GUARD=guard, LAG=int(lag), GEMM=gemm if gemm is not None else b1.GEMM_AMPERE,
                ATTENTION=attention if attention is not None else b1.ATTENTION_FA2, NORM=norm if norm is not None else b1.norm_bundle(C))


# ---------------------------------------------------------------------------------------------------------
# §7 — the served-prefix rule (one definition for Match and Commit) and the pre-challenge ⊥ assignment
# ---------------------------------------------------------------------------------------------------------


class ServedPrefixFailure(ValueError):
    def __init__(self, request: Any, n_r: int, a_r: int, name: str) -> None:
        super().__init__(f"request {request!r}: n_r={n_r} a_r={a_r}: {name}")
        self.request, self.n_r, self.a_r, self.name = request, n_r, a_r, name


def activity_prefix(tokens: Sequence[int], eos_set: Sequence[int], n_events: int, *, w: int = 32) -> int:
    """`a_r`: the least event `s ∈ [1, n_events)` with `tokens[s−1]` an Active member of `eos_set`, else `n_events`
    (the cap: every event Active).  `tokens` are lifted words (`0‖id` or ⊥) or plain ids; `tokens[0]` is the prefill's event,
    which is unconditionally live  — the scan therefore reads `tokens[s−1]` for `s ≥ 1` only.  A function of the
    committed values, never of a flag or of `n_r`."""
    eos = set(int(e) for e in eos_set)
    for s in range(1, n_events):
        if s - 1 >= len(tokens):
            break
        x = int(tokens[s - 1])
        if is_active(w, x) and payload(w, x) in eos:
            return s
    return n_events


def served_count(out: Sequence[int], *, w: int = 32) -> int:
    """`n_r`: committed Active output events before the first ⊥ (Commit's reading of the served prefix)."""
    n = 0
    for x in out:
        if is_active(w, x):
            n += 1
        else:
            break
    return n


def expected_served(a_r: int, n_events: int, lag: int) -> int:
    """The number of executed (committed, Active) events the profile predicts: `a_r` under `LAG = 0`; `a_r + 1` under `LAG = 1` when the
    request stopped on an EOS before the cap (the placeholder step's discarded sample is an executed event), `cap` otherwise."""
    check_lag(lag)
    return min(a_r + lag, n_events) if a_r < n_events else n_events


def check_served_prefix(request: Any, n_r: int, a_r: int, *, lag: int, n_events: int | None = None) -> None:
    """The two-directional rule `n_r = a_r + LAG` (under DP-pad-02): (i) `n_r >` expected → FAIL "execution past EOS";
    (ii) `n_r <` expected → FAIL "declared activity not executed" (under `LAG = 1` the placeholder step is declared activity)."""
    want = expected_served(a_r, n_events, lag) if n_events is not None else a_r + check_lag(lag)
    if n_r > want:
        raise ServedPrefixFailure(request, n_r, a_r, f"execution past EOS (expected {want} executed events under LAG={lag})")
    if n_r < want:
        raise ServedPrefixFailure(request, n_r, a_r, f"declared activity not executed (expected {want} executed events under LAG={lag})")


def assign_inactive(request: Any, served: Sequence[int], eos_set: Sequence[int], n_events: int, *, lag: int, w: int = 32) -> list[int]:
    """The pre-challenge ⊥ assignment rule (AGREED A7): from the **served** Active tokens `served[0..n_r−1]` (the
    prefill's event plus `n_r − 1` decode events), the full fixed `out[0..n_events−1]` is `0‖served[s]` for `s < n_r` and
    `enc(⊥)` for `s ≥ n_r` — admissible only if `n_r = a_r` with `a_r` derived from the served tokens themselves (the event
    after the committed EOS, or the cap).  A shortfall or excess is a named failure; nothing is filled with ⊥ silently."""
    n_r = len(served)
    if n_r < 1 or n_r > n_events:
        raise ServedPrefixFailure(request, n_r, n_events, f"served count outside [1, {n_events}]")
    # every served token is a committed Active word — checked by name at this boundary BEFORE the prefix scan
    # (a ⊥ / INV / over-width word inside the served prefix is a BoundaryFailure, not a bare ValueError, and never an EOS).
    served = [check_boundary(w, int(t), what=f"request {request!r}: served token {i}", prescribed=True) for i, t in enumerate(served)]
    a_r = activity_prefix(served, eos_set, n_events, w=w)
    check_served_prefix(request, n_r, a_r, lag=lag, n_events=n_events)
    return [enc_active(w, int(t)) for t in served] + [enc_bot(w)] * (n_events - n_r)


# ---------------------------------------------------------------------------------------------------------
# TRUE widened widths (PAD-02 CLOSED): the width of a lifted proof-block set is Σ (w+1) over its escaping
# leaves, nothing is split to fit 32 bits, and admissibility is the UNCHANGED validator at the lifted protocol parameter
# ---------------------------------------------------------------------------------------------------------

LIFTED_RULES_VERSION = "lifted-v1"  # every lifted result carries its own rules version, never the active-reference one
PADDED_RULES_VERSION = "padded-v1"  # a Program built with B7's enables (Lifted[.]_v2, LIFT_VERSION 2) and a record read by R4' (B9)
USE_KIND_LIFTED = "vu-partition-lifted"  # the use.kind of a lifted admissibility document (G8 refuses it by name)
W_MAX = 32  # the ACTIVE-REFERENCE protocol parameter of `partition.validate_width` (unchanged; not the lifted one)
W_MAX_LIFTED = 2 * W_MAX  # (PROPOSED value, AGREED that the parameter exists — A4.3): the tight universal admitting value
W_MAX_LIFTED_STATUS = ("accepted-revb-M-0042 (LIFTING-SPEC v0.5.1 B1: w_max_lifted = 2*w_max = 64, the declared constant of lifted-v1; the "
                       "parameter's existence is AGREED, A4.3; B1's condition met on the first real-row lifted use document -- OLMoE B2 "
                       "r17_1653 r0, out/gen/sweep/evidence/pad/olmoe-1b-7b__…b2…/r17_1653_lane9c1e1f79f_mech_lifted/use_document_r0: "
                       "w_max_lifted 64, max_seen 33, n_out histogram (exact walk 50,000 sets = 1; structural bound complete over "
                       "7,004,686,199 sets <= 1); the constant's value itself is proposed, its confirmation pending)")


def lifted_width_summary(out_widths_payload: Sequence[int], *, w_max_lifted: int = W_MAX_LIFTED) -> dict:
    """For a proof-block set whose escaping leaves have **payload** widths `out_widths_payload`, the width of the
    lifted set is the TRUE lifted width `w_out_lifted = Σ (w+1)` and admissibility is `w_out_lifted <= w_max_lifted` — the unlifted
    rule at the lifted parameter, no activity/payload split, no `n_max`, no re-tiling.  Reported beside it, for information only
    (A4.4 "summaries report real widths"): the payload width, `n_out` (how many bits the set grew by), and the verdict of the
    active-reference rule `w_out_lifted <= W_MAX`, which a lifted set holding a 32-bit leaf fails BY DESIGN (named, never concealed,
    never the lifted verdict).  The result carries `rules_version = lifted-v1` and `kind = vu-partition-lifted` (A5)."""
    ws = [int(w) for w in out_widths_payload]
    if any(w < 1 for w in ws):
        raise ValueError(f"payload widths must be >= 1: {ws}")
    n_out = len(ws)
    w_lifted = sum(w + 1 for w in ws)
    return {
        "kind": USE_KIND_LIFTED, "rules_version": LIFTED_RULES_VERSION, "lifted": True,
        "w_max_lifted": int(w_max_lifted), "w_max_lifted_status": W_MAX_LIFTED_STATUS, "w_max_active_reference": W_MAX,
        "n_out": n_out, "w_out_lifted": w_lifted, "w_out_payload": sum(ws), "grown_by_bits": n_out,
        "ok": w_lifted <= int(w_max_lifted),
        "active_reference_rule_on_lifted_width_ok": w_lifted <= W_MAX,  # information: the 32-bit rule on the true width
        "violations": [f"w_out_lifted {w_lifted} > w_max_lifted {int(w_max_lifted)}"] if w_lifted > int(w_max_lifted) else [],
        "rule": "w_out_lifted = sum(w+1) over escaping leaves <= w_max_lifted (no activity/payload split; no n_max)",
    }


N_OUT_HISTOGRAM_MAX_SETS = 50_000  # the exact per-set walk is bounded; the summary says whether it covered the whole family


def n_out_histogram(fam: Any, *, max_sets: int = N_OUT_HISTOGRAM_MAX_SETS, program: Any = None) -> dict:
    """The histogram of `n_out(S) = |out(S)|` — escaping leaves per proof-block set — over
    the family, computed by the SAME boundary module the validator uses (`boundary.out_gates`, exact; `Unsupported` sets counted apart),
    and with `program` the per-set TRUE lifted width `w_out^⊥(S) = Σ_{g ∈ out(S)} width(g)` (`boundary.gate_width`): `w_out_lifted_max`,
    `n_out_max`, `growth_max` (the tag bits of a set: one per escaping LIFTED leaf -- under G1  an escaping ORDINARY leaf, a
    weight or the prompt read across the cut, carries no tag and grows nothing; `n_out_lifted_max` is that count) and, for information
    only, `w_out_payload_max` (= max over sets of `w_out^⊥ − tag bits`, the unlifted width; never the rule).  Bounded by `max_sets`;
    `complete` says whether every set was visited."""
    from verity.ml.circuits import boundary as BD
    hist: dict[int, int] = {}
    n = unsupported = multipart = 0
    complete = True
    w_lift_max = n_out_max = payload_max = growth_max = None
    for gs in fam:  # a gateset.GateSet: a union of Parts; structural queries (default / instances / members) give one Part per set
        if n >= max_sets:
            complete = False
            break
        n += 1
        parts = getattr(gs, "parts", None)
        if parts is None or len(parts) != 1:  # an algebra set (several Parts): its exact out() needs the cross-part consumers -- counted apart, never guessed
            multipart += 1
            continue
        og = BD.out_gates(parts[0])
        if isinstance(og, frozenset):
            k = len(og)
            hist[k] = hist.get(k, 0) + 1
            n_out_max = k if n_out_max is None else max(n_out_max, k)
            if program is not None:
                prims = [BD.gate_prim(program, g) for g in og]
                w = sum(p.ret.w for p in prims)
                kl = sum(1 for p in prims if is_lifted_prim(p))  # the tag bits: one per escaping LIFTED leaf (G1: an ordinary leaf has none)
                w_lift_max = w if w_lift_max is None else max(w_lift_max, w)
                growth_max = kl if growth_max is None else max(growth_max, kl)
                payload_max = (w - kl) if payload_max is None else max(payload_max, w - kl)
            else:
                growth_max = k if growth_max is None else max(growth_max, k)
        else:
            unsupported += 1
    return {"n_out_histogram": {str(k): v for k, v in sorted(hist.items())}, "sets_visited": n, "sets_unsupported": unsupported,
            "sets_multipart_uncounted": multipart, "max_n_out": n_out_max, "n_out_max": n_out_max, "growth_max": growth_max,
            "n_out_lifted_max": growth_max, "w_out_lifted_max": w_lift_max, "w_out_payload_max": payload_max, "complete": complete,
            "max_sets": int(max_sets)}


def lifted_width_admissibility(program: Any, fam: Any, *, w_max_lifted: int = W_MAX_LIFTED, limits: Any = None,
                               histogram_max_sets: int = N_OUT_HISTOGRAM_MAX_SETS) -> dict:
    """Admissibility of a LIFTED Program's proof-block family: `partition.validate_width` — the one validator, UNCHANGED — run at the
    lifted parameter `w_max_lifted` on the lifted (w+1) widths the Program's types carry.  Returns the verdict's JSON with the lifted
    labels (A5) and the A4.4 width summary (`max_seen` from the validator, the `n_out` histogram from the boundary module); beside it,
    for information, the same validator's verdict at the active-reference `W_MAX` (fails on any set holding a 32-bit leaf, by design)."""
    from verity.ml.circuits import partition as PT
    lim = limits if limits is not None else PT.Limits()
    v = PT.validate_width(program, fam, int(w_max_lifted), lim)
    v_ref = PT.validate_width(program, fam, W_MAX, lim)
    return {"kind": USE_KIND_LIFTED, "rules_version": LIFTED_RULES_VERSION, "lifted": True,
            "w_max_lifted": int(w_max_lifted), "w_max_lifted_status": W_MAX_LIFTED_STATUS,
            "verdict": v.to_json(), "ok": v.status == "ok",
            "width_summary": width_summary_a44(program, fam, v, w_max_lifted=w_max_lifted, histogram_max_sets=histogram_max_sets),
            "active_reference_rule_verdict": v_ref.to_json(),
            "note": "the unchanged validator at the lifted parameter; widths are the true lifted Σ(w+1) — nothing split, nothing re-tiled"}


WIDTH_SUMMARY_FIELDS = ("w_max_lifted", "max_seen", "w_out_lifted_max", "n_out_max", "growth_max", "n_out_histogram", "w_out_payload_max", "readiness_at_32")


def width_summary_a44(program: Any, fam: Any, width_verdict: Any, *, w_max_lifted: int = W_MAX_LIFTED, histogram_max_sets: int = N_OUT_HISTOGRAM_MAX_SETS) -> dict:
    """The `width_summary` EVERY lifted use document and admissibility report carries: `w_max_lifted`, the validator's `max_seen` (a true lifted width), `w_out_lifted_max`
    (the boundary module's per-set Σ(w+1), equal to `max_seen` when the validator certified every set), `n_out_max`, `growth_max`
    (= `n_out_max`), the `n_out` histogram, `w_out_payload_max` (information, never the rule), and `readiness_at_32` (the 32-bit gate
    readiness of the lifted Program, information beside the verdict at `w_max_lifted`)."""
    from verity.ml.circuits import partition as PT
    h = n_out_histogram(fam, max_sets=histogram_max_sets, program=program)
    if not h["complete"]:  # the exact walk did not cover the family (real bodies: 10^9 sets): the STRUCTURAL bound per class covers it all
        h = {**h, "n_out_structural_bound": n_out_structural_bound(program, fam)}
    max_seen = getattr(width_verdict, "max_seen", None)
    try:
        r32 = PT.gate_width_readiness(program, W_MAX)
        r64 = PT.gate_width_readiness(program, int(w_max_lifted))
        ready = {"readiness_at_32": {"ok": bool(r32.ok), "gates_over": int(r32.gates_over), "max_w": int(r32.max_w)},
                 "readiness_at_w_max_lifted": {"ok": bool(r64.ok), "gates_over": int(r64.gates_over), "max_w": int(r64.max_w)}}
    except Exception as e:  # noqa: BLE001  -- readiness is information; its absence is named, never silent
        ready = {"readiness_at_32": {"error": f"{type(e).__name__}: {e}"}, "readiness_at_w_max_lifted": None}
    return {"w_max_lifted": int(w_max_lifted), "w_max_lifted_status": W_MAX_LIFTED_STATUS, "max_seen": max_seen,
            "w_out_lifted_max": h["w_out_lifted_max"], "n_out_max": h["n_out_max"], "max_n_out": h["n_out_max"], "growth_max": h["growth_max"],
            "n_out_histogram": h["n_out_histogram"], "w_out_payload_max": h["w_out_payload_max"],
            "max_seen_equals_w_out_lifted_max": (max_seen == h["w_out_lifted_max"]) if (max_seen is not None and h["w_out_lifted_max"] is not None) else None,
            "sets_visited": h["sets_visited"], "sets_unsupported": h["sets_unsupported"], "sets_multipart_uncounted": h["sets_multipart_uncounted"],
            "complete": h["complete"], "n_out_structural_bound": h.get("n_out_structural_bound"),
            "w_max_active_reference": W_MAX, **ready, "rules_version": LIFTED_RULES_VERSION}


def n_out_structural_bound(program: Any, fam: Any) -> dict:
    """A COMPLETE, sound bound on `n_out(S)` over EVERY set of a tiled family, by member class (the validator's own `CheckedTiling`; one class =
    the sets one plan entry induces, structurally identical across activations): `n_out(S) ≤ Σ_members distinct returned leaves(member)`
    -- a gate of S read outside S is a leaf the member RETURNS (a primitive: its one `ret`; a composite activation: its body's distinct
    returned leaves, the same count `gateset.member_width` sums widths over; internal gates are scoped, nothing else escapes).  Returns the
    histogram of the bound (sets per bound value, every set counted), `n_out_bound_max`, `growth_max_bound`.  Exact for a one-leaf member
    that is read (the default tiling's atomic sets: bound 1, exact walk 1).  No usable tiling -> unavailable by name; nothing is guessed."""
    from verity.ml.circuits import partition as PT
    tiling = PT.family_tiling(fam)
    if tiling is None:
        return {"available": False, "why": "no tiling (an algebra family): the exact walk is the only source"}
    try:
        t = PT.CheckedTiling(program, tiling)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "why": f"{type(e).__name__}: {e}"}
    if t.problems:
        return {"available": False, "why": f"malformed tiling: {t.problems[:3]}"}
    memo: dict[int, tuple[Any, int]] = {}

    def gates_of(fn) -> int:  # the leaves one activation RETURNS: a primitive one (`ret`); a composite its body's distinct returned refs
        hit = memo.get(id(fn))
        if hit is not None and hit[0] is fn:
            return hit[1]
        from verity.ml.circuits.defs import PrimitiveDefinition
        g = 1 if isinstance(fn, PrimitiveDefinition) else len({ref for ref in fn.body.ret})
        memo[id(fn)] = (fn, g)
        return g

    hist: dict[int, int] = {}
    n_sets = 0
    for cl in t.classes:
        insts = t.instances.get(cl.key, 0)
        if insts == 0:
            continue
        body = t.spec[cl.key].body
        for shape, (count, members) in cl.shapes().items():
            b = sum(gates_of(body.nodes[j].fn) for j in cl.group) if cl.rule == "zip" else members * gates_of(body.nodes[cl.node].fn)
            hist[b] = hist.get(b, 0) + count * insts
            n_sets += count * insts
    bmax = max(hist) if hist else None
    return {"available": True, "complete": True, "sets": n_sets, "classes": len(t.classes), "n_out_bound_histogram": {str(k): v for k, v in sorted(hist.items())},
            "n_out_bound_max": bmax, "growth_max_bound": bmax,
            "rule": "n_out(S) <= sum over members of the member's distinct returned leaves, per member class (the validator's tiling); a bound, exact for one-leaf members that are read"}


def word_class(w: int, x: int) -> str:
    """The name of a `w+1`-bit word: `"active"` (tag 0), `"bot"` (`1‖0^w`), `"noncanonical"` (`1‖nonzero`, `INV_w`
    included).  A word outside the width is refused.  Naming, never rewriting."""
    if not in_width(w, x):
        raise ValueError(f"not a {w + 1}-bit word: {x:#x}")
    if tag(w, x) == 0:
        return "active"
    return "bot" if payload(w, x) == 0 else "noncanonical"


def commit_leaf(w: int, x: int) -> dict:
    """The COMMITMENT layer's treatment of one lifted leaf: the raw `w+1`-bit word `x` as held, laid out in
    `leaf_words(w)` machine words (`pack_leaf`, total on the representation) and NAMED (`word_class`).  A noncanonical word is
    committed as the distinct raw bitstring it is and reported by name; it is never normalized to ⊥, to `INV_w`, to `0‖payload`, or
    dropped.  `unpack_leaf(w, words) == x` for every word, canonical or not."""
    return {"w": int(w), "lifted_width": int(w) + 1, "word": int(x), "class": word_class(w, x), "words": pack_leaf(w, x),
            "normalized": False}


WORD_BITS = 32  # the machine word of the statement / leaf layout (PAD-02: a > 32-bit lifted leaf occupies ⌈(w+1)/32⌉ words)


class LiftedLayoutError(ValueError):
    """A committed leaf layout that cannot hold the lifted word (the tag would be dropped) — refused, never read as Active."""


def leaf_words(w: int) -> int:
    """Machine words of one lifted leaf of payload width `w` (PAD-02 layout rule): `⌈(w+1)/32⌉` — 1 for `w ≤ 31`
    (bf16 → 17 bits, the activity bit → 2 bits), 2 for `w = 32` (an i32/f32 leaf needs its 33rd bit)."""
    if w < 1:
        raise ValueError(w)
    return -(-(w + 1) // WORD_BITS)


def pack_leaf(w: int, x: int) -> list[int]:
    """The lifted word `x` (`w+1` bits, valid or INV — the layout is total on the representation) as `leaf_words(w)` little-endian
    32-bit words.  Nothing is dropped: `unpack_leaf(w, pack_leaf(w, x)) == x` for every `x < 2^{w+1}`."""
    if not (0 <= x < (1 << (w + 1))):
        raise ValueError(f"not a {w + 1}-bit word: {x:#x}")
    n = leaf_words(w)
    return [(x >> (WORD_BITS * i)) & ((1 << WORD_BITS) - 1) for i in range(n)]


def unpack_leaf(w: int, words: Sequence[int]) -> int:
    """Inverse of `pack_leaf`.  A layout with fewer words than `leaf_words(w)` is refused by name (attack A5: a 33-bit leaf stored
    in one 32-bit word has no tag — it would read as Active whatever was committed); bits above `w+1` must be zero."""
    n = leaf_words(w)
    if len(words) != n:
        raise LiftedLayoutError(f"lifted leaf of payload width {w} needs {n} word(s) (w+1 = {w + 1} bits), layout has {len(words)}: "
                                "the tag bit is not representable — refused, not read as Active")
    x = 0
    for i, wd in enumerate(words):
        if not (0 <= wd < (1 << WORD_BITS)):
            raise ValueError(f"not a 32-bit word: {wd:#x}")
        x |= wd << (WORD_BITS * i)
    if x >> (w + 1):
        raise LiftedLayoutError(f"bits above the lifted width {w + 1} are set: {x:#x}")
    return x


# ---------------------------------------------------------------------------------------------------------
# DENSE B1, the Program OF RECORD: the Build's derived flat request component, lifted by node replay + the guard
# ---------------------------------------------------------------------------------------------------------

def replay_body(B: Builder, body, args: Sequence[Coll]) -> Coll:
    """Emit `body`'s nodes into `B` verbatim (identity replay; the lifting replay is `replay_lifted`).  Used as the python body of
    body-only definitions (`LiftedRequest_v1`) so `Program(...)` can lower them at a root."""
    k0 = len(B.nodes)
    for node in body.nodes:
        rargs = [Coll(a.type, _remap(a.refs, args, k0)) for a in node.args]
        if node.form in ("primitive", "call"):
            B.call(node.fn, *rargs, name=node.name)
        elif node.form == "batch":
            B.batch(node.fn, *rargs, axes=node.axes, n=node.n, name=node.name)
        elif node.form == "scan":
            B.scan(node.fn, rargs[0], xs=rargs[1:1 + node.n_xs], shared=rargs[1 + node.n_xs:], n=node.n, name=node.name)
        else:
            raise ValueError(node.form)
    return Coll(body.ret_type, _remap(body.ret, args, k0))


def as_program(fn: SpecializedDefinition, **kw) -> "Program":
    """`Program(fn)` for any complete specialization, INCLUDING body-only ones (a decoded `WorkloadRequest_v1` component or a
    `Lifted[decoded]`): those get a same-id definition whose python body is the identity replay of the set body, so the root lowers,
    evaluates and digests exactly as the body says (the spec id — hence `program_fn` in the digest — is unchanged)."""
    from verity.ml.circuits.program import Program
    if not _is_decoded(fn):
        return Program(fn, **kw)
    body = fn.body
    d = CompositeDefinition(fn.defn.name, fn.defn.version, fn.defn.statics, lambda S: (body.params, body.ret_type),
                            lambda B, S, *a: replay_body(B, body, a), conformance=fn.defn.conformance, register=False)
    sp = SpecializedDefinition(d, fn.bindings, fn.key)
    return Program(sp, **kw)


_LREQUEST_BODIES: dict[tuple, Any] = {}  # LiftedRequest key → its Body (the definition of record; the python body replays it)


def _lrequest_key_of(S) -> tuple:
    return (("program_digest", S.program_digest), ("program_fn", S.program_fn), ("EOS", tuple(S.EOS)), ("GUARD", S.GUARD), ("LAG", int(S.LAG)))


def _lrequest_sig(S):
    b = _LREQUEST_BODIES[_lrequest_key_of(S)]
    return b.params, b.ret_type


def _lrequest_body(B, S, *args):
    return replay_body(B, _LREQUEST_BODIES[_lrequest_key_of(S)], args)


_LREQUEST_DEF = CompositeDefinition("LiftedRequest", 2, ("program_digest", "program_fn", "EOS", "GUARD", "LAG"), _lrequest_sig, _lrequest_body,  # v2 = padded-v1 (B8)
                                    doc="the ⊥-lift of a derived request Program component (GP-01 `WorkloadRequest_v1{program_digest, program_fn}`): "
                                        "its flat body replayed node for node with lifted callees, plus the activity guard (`LStepLive` + "
                                        "`LSelect` on the token entering each decode step) spliced in before each step's first reader; identity = "
                                        "the unlifted component's digest + program function + the declared EOS set + the guard placement",
                                    conformance="protocol-defined", register=False)
_LREQUEST_CACHE: dict[tuple, SpecializedDefinition] = {}
BOUNDARY_PREFIX = "TokenSelect_v1"  # the sampling event of the derived flat body (the step boundary; GP-01's X-09 segmentation)


def _reads(r: Refs) -> set[int]:
    """Body node indices a refs object reads — at RUN level, never per leaf (a derived SmolLM2 body has 19k nodes whose weight
    operands are 10^7-leaf runs: the per-leaf form took > 9 min on the real fixture; this takes 2 s)."""
    if isinstance(r, (Affine, Strided)):
        return {r.idx} if r.space == "n" else set()
    if isinstance(r, Concat):
        out: set[int] = set()
        for p in r.parts:
            out |= _reads(p)
        return out
    return {idx for space, idx, _ in r if space == "n"}  # Explicit: enumerated leaves, small by construction


def _remap_nodes(r: Refs, args: Sequence[Coll], nidx: Sequence[int], redirect: dict[int, int]) -> Refs:
    """`_remap` with a per-node index map (`nidx[k]` = builder node of original node `k`; guard nodes may sit between) and a
    redirect (`original node b` → builder node of its `tok_in`: a one-leaf token read is redirected to the guarded token)."""
    if isinstance(r, Affine):
        if r.space == "n":
            if r.idx in redirect:
                if not (r.base == 0 and r.count == 1):
                    raise ValueError(f"a guarded token is one leaf; node {r.idx} is read as [{r.base}, {r.base + r.count})")
                return Affine("n", redirect[r.idx], 0, 1)
            return Affine("n", nidx[r.idx], r.base, r.count)
        if r.space == "p":
            return args[r.idx].refs.slice(r.base, r.base + r.count)
        return r
    if isinstance(r, Strided):
        if r.space == "n":
            if r.idx in redirect:
                raise ValueError(f"a guarded token is one leaf; node {r.idx} is read strided")
            return Strided("n", nidx[r.idx], r.base, r.inner, r.outer_stride, r.inner_stride, r.count)
        if r.space == "p":
            a = args[r.idx].refs
            if isinstance(a, Affine):
                return Strided(a.space, a.idx, a.base + r.base, r.inner, r.outer_stride, r.inner_stride, r.count)
            return Explicit(tuple(a[leaf] for _, _, leaf in r))
        return r
    if isinstance(r, Concat):
        return concat([_remap_nodes(p, args, nidx, redirect) for p in r.parts])
    items = []
    for space, idx, leaf in r:
        if space == "n":
            if idx in redirect:
                if leaf != 0:
                    raise ValueError(f"a guarded token is one leaf; node {idx} leaf {leaf} read")
                items.append(("n", redirect[idx], 0))
            else:
                items.append(("n", nidx[idx], leaf))
        elif space == "p":
            items.append(args[idx].refs[leaf])
        else:
            items.append((space, idx, leaf))
    return Explicit(tuple(items))


def request_boundaries(body, boundary_prefix: str = BOUNDARY_PREFIX) -> list[int]:
    """The sampling-event nodes of a flat request body: the nodes whose spec id starts with `boundary_prefix`, each a one-leaf
    `Value<32>` (the committed token of event `s`), in body order.  `cap = len(...)`."""
    from verity.ml.circuits.codec import _spec_id
    b = [k for k, n in enumerate(body.nodes) if _spec_id(n.fn).startswith(boundary_prefix)]
    for k in b:
        if body.nodes[k].out != Value(32):
            raise ValueError(f"boundary node {k} ({_spec_id(body.nodes[k].fn)}) returns {body.nodes[k].out!r}, not one Value<32> token")
    return b


def lift_request_component(fn: SpecializedDefinition, eos: Sequence[int], *, max_tokens: int, lag: int, guard: str = GUARD_IN,
                           boundary_prefix: str = BOUNDARY_PREFIX) -> SpecializedDefinition:
    """The lifted fixed Program of ONE derived request: `fn` is GP-01's request component (`workload.request_component(
    descriptor).fn` — the Build's derived flat body with the prescribed inputs as parameters, built from the manifest at `T = cap−1`),
    `eos` the declared EOS set.  Result: `LiftedRequest_v2{program_digest, program_fn, EOS, GUARD, LAG}` (v2 = padded-v1, B8) whose body is

        [eos[0..|E|), bot32] ++ replay of the original nodes with lifted callees, where immediately before the FIRST reader of the
        token of event `s−1` (for every `s ∈ 1..cap−1`) the guard `s=<s>/live = LStepLive(tok_{s−1}, eos_set)`,
        `s=<s>/tok_in = LSelect32(live, tok_{s−1}, bot32)` is spliced in and every node reader of `tok_{s−1}` is redirected to
        `tok_in` (DP-pad-01 option (a), the INPUT guard); the returned `out[]` keeps reading the committed tokens themselves.

    PAD-03 (address stability modulo the guard nodes) holds by construction: original node `k` ↦ node `nidx[k]` with the same
    gate count and operand runs; the other nodes are exactly the named guard nodes.  Nothing about a served length enters: the
    guards are placed by dataflow (who reads the token), and the component's boundary count MUST equal the manifest's declared
    `max_tokens`: a component derived at a SERVED shape (`build_request_LP<lp>_T<served>`, the MOE-05
    eos-truncated derivation) has fewer events than the declaration and is refused by name — it is the post-hoc deletion (A3)
    disguised as a Program of record, never the fixed population."""
    if guard != GUARD_IN:
        raise ValueError(f"GUARD={guard!r}: only the input guard is implemented ({GUARDS_IMPLEMENTED}); the output guard (DP-pad-01 (b)) is refused by name")
    eos_t = tuple(int(e) for e in eos)
    if not eos_t:
        raise ValueError("the declared EOS set is empty: the request could never stop before its cap — declare it, do not default it")
    for e in eos_t:
        check_boundary(32, e, what="eos id", prescribed=True)
    bindings = dict(fn.bindings)
    if fn.defn.name != "WorkloadRequest":
        raise TypeError(f"{fn.id} is not a GP-01 request component (WorkloadRequest_v1{{program_digest, program_fn}})")
    lag = check_lag(lag)
    key = (("program_digest", bindings["program_digest"]), ("program_fn", bindings["program_fn"]), ("EOS", eos_t), ("GUARD", guard), ("LAG", lag))
    body = fn.body
    bounds = request_boundaries(body, boundary_prefix)
    cap = len(bounds)
    if cap < 1:
        raise ValueError(f"no {boundary_prefix} node in the body: not a request Program")
    if int(max_tokens) != cap:
        raise ValueError(f"the component has {cap} sampling events but the manifest declares max_tokens={max_tokens}: a Program derived at a served "
                         "shape is not the fixed population (refuse, never lift the served length)")
    cached = _LREQUEST_CACHE.get(key)
    if cached is not None:
        if tuple(cached.boundaries) != tuple(bounds):  # type: ignore[attr-defined]  # one identity, one segmentation
            raise ValueError(f"{cached.id} was lifted with boundaries {cached.boundaries}, now asked with {tuple(bounds)} ({boundary_prefix!r})")  # type: ignore[attr-defined]
        return cached
    ret_reads = _reads(body.ret)
    if not set(bounds) <= ret_reads:
        raise ValueError("the request's return does not read every sampling event's token")
    # the first node reader of each token (event s−1 → guard of step s); no reader may precede the token (dataflow order)
    first_reader: dict[int, int] = {}
    for k, node in enumerate(body.nodes):
        for a in node.args:
            for b in _reads(a.refs) & set(bounds):
                first_reader.setdefault(b, k)
    guard_at: dict[int, int] = {}
    for s in range(1, cap):
        b = bounds[s - 1]
        if b not in first_reader:
            raise ValueError(f"the token of event {s - 1} (node {b}) is read by no node: step {s} does not consume it — not a request Program")
        if first_reader[b] > bounds[s]:
            raise ValueError(f"the token of event {s - 1} is first read after event {s}'s node: the steps are not in body order")
        if s > lag:  # steps 1..LAG are unconditionally live: no guard, their token is read as is
            # the guard is spliced in at the START of step s's segment (right after event s−1's node and its controller), so
            # `s=<s>/live` precedes EVERY node the step owns — it is their enable operand, not only the token's selector
            guard_at[b + 1] = s
    if bounds[-1] in first_reader:
        raise ValueError("the last event's token is read by a node: the body has steps beyond its last boundary")
    # G1: the component's parameters are the Build's PRESCRIBED inputs (weights, prompt tokens,...) -- ordinary `Value<w>`, unless
    # the body itself needs one lifted (accept_classes: a parameter sharing an operand or the return with lifted leaves)
    pcls, ncls = infer_classes(body, (ORDINARY,) * len(body.params))
    params_l = tuple((n, class_type(t, c)) for (n, t), c in zip(body.params, pcls))
    B = Builder(params_l, lift_type(body.ret_type))
    args = [Coll(class_type(t, c), Affine("p", i, 0, t.leaves)) for i, ((_, t), c) in enumerate(zip(body.params, pcls))]
    eos_cols = [B.call(b1.const(32, e), name=f"eos[{i}]") for i, e in enumerate(eos_t)]  # the EOS set: ordinary structure (A8/A11)
    eos_set = Coll(Array(len(eos_t), Value(32)), concat([c.refs for c in eos_cols]))
    bot = B.call(l_bot(32), name="bot32")
    one = B.call(enable_one(), name="enable/one")  # the root's enable (A9.1): prefill and the unguarded steps 1..LAG are owned by it
    # Continuation_v1: the controller family -- `slot=1/emit` here, then `slot=<t>/is_eos`, `slot=<t>/out`, `slot=<t+1>/emit` right after the
    # sampling-event node of slot t (y_t = the committed token of event t-1); the request's return reads the designated outputs `slot=<t>/out`
    cont = ContinuationChain(B, cap=cap, eos_set=eos_set, bot=bot)
    slot_of_bound = {b: t for t, b in enumerate(bounds, start=1)}
    out_node: dict[int, int] = {}  # boundary node -> its designated-output node (the return's redirect)
    nidx: list[int] = []
    redirect: dict[int, int] = {}
    gate_node: int | None = None  # LAG=1: the previous step's tok_in (or out[0] for step 2)
    enable = one  # B7: the enable every replayed node of the CURRENT segment receives — `one` until the first guarded step
    for k, node in enumerate(body.nodes):
        s = guard_at.get(k)
        if s is not None:
            b = bounds[s - 1]
            prev = Coll(Value(33), Affine("n", nidx[b], 0, 1))
            if lag == 0:
                gate = prev
            else:
                gate = Coll(Value(33), Affine("n", nidx[bounds[s - 2]] if gate_node is None else gate_node, 0, 1))
            live = B.call(with_ordinary(l_step_live(len(eos_t)), 0b10), gate, eos_set, name=f"s={s}/live")
            B.call(l_select(32), live, prev, bot, name=f"s={s}/tok_in")
            redirect[b] = len(B.nodes) - 1
            gate_node = redirect[b]
            enable = live  # A13.1: `run_{s+1} = live_s` is the enable of every gate step s owns (the flat body's segment s)
        acls = [_operand_class(a.refs, pcls, body, ncls) for a in node.args]
        lf = lifted_callee_for(node, acls, ncls.get(k, LIFTED))
        largs = [Coll(class_type(a.type, c), _remap_nodes(a.refs, args, nidx, redirect)) for a, c in zip(node.args, acls)]
        nidx.append(len(B.nodes))
        emit_lifted_node(B, node, lf, largs, enable)
        t = slot_of_bound.get(k)
        if t is not None:  # the slot's controller gates, immediately after its sampling event (they read y_t and emit_t only)
            cont.slot(t, Coll(Value(33), Affine("n", nidx[k], 0, 1)))
            out_node[k] = len(B.nodes) - (2 if t < cap else 1)  # the `slot=<t>/out` node: before the `slot=<t+1>/emit` transition when there is one
    if [B.nodes[v].name for v in out_node.values()] != [f"slot={t}/out" for t in range(1, cap + 1)]:
        raise AssertionError("the designated-output nodes are not where the chain put them")  # a construction invariant, never silent
    # the return: every reader of a committed token in the ORIGINAL return reads the slot's designated output o_t = Select(emit_t, y_t, bot)
    # instead (C: the output interface uses emit, never run); the raw tokens stay committed body values (required values)
    ret = Coll(lift_type(body.ret_type), _remap_nodes(body.ret, args, nidx, out_node))
    sp = SpecializedDefinition(_LREQUEST_DEF, {**bindings, "EOS": eos_t, "GUARD": guard, "LAG": lag}, key)
    sp.set_body(B.finish(ret))
    _LREQUEST_BODIES[key] = sp.body
    sp.node_map = tuple(nidx)  # type: ignore[attr-defined]  # PAD-03: original node k ↦ lifted node nidx[k]
    sp.param_classes = tuple(pcls)  # type: ignore[attr-defined]  # G1: the class of each prescribed input (ordinary unless the body needs it lifted)
    sp.guard_nodes = tuple(sorted(set(range(len(B.nodes))) - set(nidx)))  # type: ignore[attr-defined]
    sp.boundaries = tuple(bounds)  # type: ignore[attr-defined]
    sp.cap = cap  # type: ignore[attr-defined]
    sp.lag = lag  # type: ignore[attr-defined]
    sp.continuation = CONTINUATION_VERSION  # type: ignore[attr-defined]  # the controller family is part of the Program
    sp.output_nodes = tuple(out_node[b] for b in bounds)  # type: ignore[attr-defined]  # slot=<t>/out per slot, in order
    sp.candidate = DP_PAD_02  # type: ignore[attr-defined]  # bound as a labelled candidate, not an agreed baseline
    _LREQUEST_CACHE[key] = sp
    return sp


def address_map_modulo_guards(lifted: SpecializedDefinition) -> list[tuple[str, int, int]]:
    """PAD-03's deletion rule as a function: the lifted body's `(spec id, lo, hi)` rows with the guard rows deleted and the gates
    renumbered — equal to the unlifted component's rows modulo the `Lifted[…]` spec ids (tested)."""
    from verity.ml.circuits.codec import _spec_id
    b = lifted.body
    guards = set(lifted.guard_nodes)  # type: ignore[attr-defined]
    rows, g = [], 0
    for k, n in enumerate(b.nodes):
        if k in guards:
            continue
        w = b.offsets[k + 1] - b.offsets[k]
        rows.append((_spec_id(n.fn), g, g + w))
        g += w
    return rows


# ---------------------------------------------------------------------------------------------------------
# DENSE B1 — the lifted fixed Program of one declared request (built from the manifest only)
# ---------------------------------------------------------------------------------------------------------


def lifted_request(C: dict, prompt_len: int, cap: int, eos: Sequence[int], *, lag: int, T: int | None = None, gemm: dict | None = None,
                   attention: dict | None = None, norm: dict | None = None) -> SpecializedDefinition:
    """The lifted padded serving Program of ONE declared request `(prompt_len, max_tokens = cap, eos_set)` under the Build's
    `T = cap − 1` decode-step convention (REL-01/R-16: prefill emits `out[0]`, decode steps `1..cap−1` emit
    `out[1..cap−1]`, so `STEPS = cap − 1` and the event population is exactly `cap = D_MAX_r`).  Inputs: the manifest's declared numbers
    and the registered Definitions — nothing from an execution.  `T`, when a caller passes one, must equal `cap − 1`: a builder
    fed a served length is refused by name (the unlifted `WorkloadProgram` rule, kept verbatim)."""
    if cap < 1:
        raise ValueError(f"max_tokens = {cap}: a request declares at least one sampling event")
    if T is not None and T != cap - 1:
        raise ValueError(f"T={T} is not cap-1={cap - 1}: the Program must be built from the declared cap, never from a served length")
    return lserve(C, prompt_len, cap - 1, eos, lag=lag, gemm=gemm, attention=attention, norm=norm) if cap > 1 else \
        _prefill_only(C, prompt_len, eos, gemm=gemm, attention=attention, norm=norm)


def _prefill_only(C, LP, eos, *, gemm, attention, norm):
    raise NotImplementedError("max_tokens = 1 (prefill-only request): LServe_v1 needs STEPS >= 1; not built tonight (declare it, do not fake it)")


def expected_bot_return_leaves(C: dict, cap: int, a_r: int, *, lag: int) -> int:
    """T-3(ii) arithmetic under the implemented rule (GUARD=in, declared LAG): every decode step `s ∈ [a_r + LAG, cap−1]` returns
    `1 + 2·NL·KVH·D` leaves, all `enc(⊥)`; `a_r ∈ [1, cap]` (`a_r = cap` ⇔ no EOS: zero ⊥ leaves)."""
    if not (1 <= a_r <= cap):
        raise ValueError(f"a_r={a_r} outside [1, cap={cap}]")
    return (cap - expected_served(a_r, cap, lag)) * (1 + 2 * C["NL"] * C["KVH"] * C["D"])


# ---------------------------------------------------------------------------------------------------------
# MIXED BATCH — the lifted fixed WORKLOAD Program: GP-01's compose, verbatim, over lifted components
# ---------------------------------------------------------------------------------------------------------


_LIVE_COUNT: dict[int, PrimitiveDefinition] = {}


def live_count_prim(R: int) -> PrimitiveDefinition:
    """`LiveCount<R>_v1(live: Array<R, Value<1>>) → Value<7>`: the number of live rows among the `R` requests programmed for an engine step
    (a popcount).  The UNLIFTED count; its strict lift `Lifted[LiveCount<R>_v1]` (`Array<R, Value<2>>` → `Value<8>`) is the operand of
    `LSplitsForSMS` (PAD-04).  `R ≤ 64`: the split pipeline's batch bound (`assert_splits_domain`)."""
    R = assert_splits_domain(R)
    p = _LIVE_COUNT.get(R)
    if p is None:
        p = PrimitiveDefinition(f"LiveCount{R}", 1, (("live", Array(R, Value(1))),), Value(N_LIVE_BITS), lambda *bits: sum(int(b) & 1 for b in bits),
                                doc=live_count_prim.__doc__ or "", conformance="protocol-defined")
        _LIVE_COUNT[R] = p
    return p


def live_count_hook(B: Builder, outs: Sequence[Coll], arrivals: Sequence[int], cap: int | Sequence[int], eos_set: Coll, bot: Coll, lag: int, *,
                    num_sms: int | None = None, name: str = "S") -> tuple[list[Coll], list[Coll]]:
    """P9's hook as Program nodes at the WORKLOAD root (CANDIDATE PAD-04 wiring): for every engine step
    `t` of the programmed horizon, `n_live_t = LiveCount(live_{r,t} : r programmed for t)` where `live_{r,t}` is recomputed from request
    `r`'s committed tokens exactly as its own guard computes it — `LAG = 0`: `1` at `t = a_r`, `LStepLive(out_r[s−1], E)` for `s = t − a_r ≥ 1`;
    `LAG = 1`: `1` at `s ∈ {0, 1}`, `LStepLive(tok_in_{r,s−1}, E)` for `s ≥ 2` with `tok_in_{r,1} = out_r[0]`, `tok_in_{r,s} = LSelect(live_{r,s},
    out_r[s−1], ⊥)` — so `n_live_t = Σ_r LIsActive(tok_in_{r,t})` (LIFTING-STATUS P9) as a value of the Program, never of an observation.
    With `num_sms` the split count `S_t = LSplitsForSMS<SMS>(n_live_t)` is appended (⊥ when no request is live: no sampler launch).
    Returns `(n_live[t], S[t])` per engine step (`S` empty without `num_sms`).  Rows outside a request's window are not members (structural).
    `cap` is per request (GP-01's populations have UNEQUAL caps — its standard B2 fixture is cap 3 / cap 2): one int
    for all or one per request; request `r` is a member of engine steps `a_r <= t < a_r + cap_r`, and `len(out_r) != cap_r` is refused by
    name (a served-length `out_r` is not the fixed population)."""
    check_lag(lag)
    R = len(outs)
    if R != len(arrivals):
        raise ValueError(f"{R} request outputs but {len(arrivals)} arrivals")
    caps = [int(cap)] * R if isinstance(cap, int) else [int(c) for c in cap]
    if len(caps) != R:
        raise ValueError(f"{R} request outputs but {len(caps)} caps: one cap per request (or one int for all)")
    for r, (out, c) in enumerate(zip(outs, caps)):
        if c < 1:
            raise ValueError(f"request {r}: cap {c} < 1")
        if out.type.leaves != c:
            raise ValueError(f"request {r}: out has {out.type.leaves} events but its declared cap is {c}: the hook takes the FIXED population "
                             "(cap_r events per request), never a served length")
    assert_splits_domain(R)
    horizon = max(a + c for a, c in zip(arrivals, caps))
    live: list[list[Coll]] = [[] for _ in range(R)]
    one = B.call(b1.const(2, enc_active(1, 1)), name=f"{name}/one")
    for r, (out, a) in enumerate(zip(outs, arrivals)):
        tok_in: Coll | None = None
        for s in range(caps[r]):
            if s <= lag:
                lv = one
            else:
                gate = out[s - 1] if lag == 0 else tok_in
                lv = B.call(l_step_live(len(eos_set)), gate, eos_set, name=f"{name}/live[r{r},s{s}]")
            live[r].append(lv)
            if lag == 1 and s >= 1:
                tok_in = out[0] if s == 1 else B.call(l_select(32), lv, out[s - 1], bot, name=f"{name}/tok_in[r{r},s{s}]")
    n_live, S = [], []
    for t in range(horizon):
        members = [live[r][t - a] for r, a in enumerate(arrivals) if 0 <= t - a < caps[r]]
        if not members:
            raise ValueError(f"engine step {t}: no request is programmed for it (a gap in the declared chronology)")
        # the hook is ROOT-owned (always enabled, A9.1): the propagation lift `Lifted[LiveCount]_v1` is its total relation, no enable operand
        n = B.call(lifted_prim_v1(live_count_prim(len(members))), coll_concat(Array(len(members), Value(2)), members), name=f"{name}/n_live[t{t}]")
        n_live.append(n)
        if num_sms is not None:
            S.append(B.call(l_splits_for(num_sms), n, name=f"{name}/S[t{t}]"))
    return n_live, S


__all__ = ["LIFTING_VERSION", "LABEL", "enc_bot", "inv", "tag", "payload", "valid", "is_bot", "is_active", "enc_active", "in_width",
           "WORD_BITS", "LiftedLayoutError", "leaf_words", "pack_leaf", "unpack_leaf", "lifted_request", "expected_bot_return_leaves",
           "lift_request_component", "request_boundaries", "address_map_modulo_guards", "BOUNDARY_PREFIX",
           "BoundaryFailure", "check_boundary", "lift_type", "unlift_type", "lift_words", "bot_words", "lifted_prim", "lifted_prim_v1", "LIFT_VERSION",
           "ENABLE_PARAM", "ENABLE_T", "has_enable", "enable_one", "emit_lifted_node", "l_enable", "enable_gate_count", "state_select_count",
           "register_hand_lift", "hand_lift_of", "l_select",
           "l_is_active", "l_bot", "l_gather_row", "l_gather_row_of", "registered_oor_extension", "l_step_live", "splits_for_prim", "l_splits_for", "assert_splits_domain", "SPLITS_MAX_BATCH", "N_LIVE_BITS", "SPLITS_BITS", "specified_ids", "lift_callee", "lift_composite", "lift_def",
           "replay_lifted", "replay_body", "as_program", "Prefill", "StepBody", "LServe", "lserve", "lserve_validate", "GUARD_IN", "LAGS_IMPLEMENTED", "check_lag", "guard_gate_count", "expected_served", "GUARDS_IMPLEMENTED", "ServedPrefixFailure",
           "activity_prefix", "served_count",
           "check_served_prefix", "assign_inactive", "W_MAX", "W_MAX_LIFTED", "W_MAX_LIFTED_STATUS", "LIFTED_RULES_VERSION", "PADDED_RULES_VERSION", "USE_KIND_LIFTED",
           "lifted_width_summary", "lifted_width_admissibility", "n_out_structural_bound", "n_out_histogram", "N_OUT_HISTOGRAM_MAX_SETS", "width_summary_a44", "WIDTH_SUMMARY_FIELDS",
           "word_class", "commit_leaf",
           "live_count_prim", "live_count_hook", "DP_PAD_02"]
