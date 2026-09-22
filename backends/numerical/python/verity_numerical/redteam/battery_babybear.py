"""The BabyBear forgery battery: concrete witnesses every BabyBear prover of the WP2 relation must reject.

A case is a unit (``level = "step"``: ``c, a[k], b[k]`` -> ``y``) or a chain (``level = "vu"``: ``a, b`` -> ``y16``) with
the WP2 hint tree (the layout of ``checker.step.gen_step`` plus the limb hints ``acc.f_lo/f_hi``, ``pack.low_lo/low_hi``),
one or more hints (or operands) mutated, and the output word a checker that only reads congruences mod p would be led
to; or a gadget (``level = "gadget"``) with the inputs/hints/claimed outputs of a z3 model from ``z3_babybear``.
Every case's expected verdict is **reject**.  The battery has teeth: the ``naive`` verdict recorded per case is what
a checker that asserts every WP2 predicate mod p *without* the ``babybear.py`` rules says -- ``accept`` on the cases
that matter.  The WP2 integer checker and the embedded BabyBear checker (all rules) reject every case.

Mutation classes (the autoproof review's forgery classes, applied to every hint of the relation):

* ``wrap``       -- the +p shadow of a truncation as it exists in a field: a pair of hints ``(u, w)`` of one gadget
                    re-split so that ``u' 2^s + w' = u 2^s + w + j p`` (two canonical elements, same congruence).
* ``fractional`` -- a hint ``x`` that is only checked through ``x * 2^s`` (a scaled range): ``x' = (x 2^s + j p) / 2^s``
                    is an integer with ``x' 2^s = x 2^s (mod p)``.
* ``zero_swap``  -- ``+0 <-> -0`` in an operand (or the accumulator) with the honest hints kept.
* ``hint_swap``  -- the hints of another instance (cross-pairing).
* ``chain``      -- the accumulator handed between two units as ``c + p`` (a field-element chain instead of a word).
* ``epilogue``   -- +-p on the RN-even epilogue's hints (``hi``, ``lo``, ``lsb``, ``carry``), incl. the tie boundary.
* ``z3``         -- SAT models of ``z3_babybear`` with one embedding rule dropped: what a prover missing that rule accepts.

    python -m verity_numerical.redteam.battery_babybear [--z3 z3_babybear.json] [--out fixtures/redteam/babybear-battery/v1]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from verity_numerical.checker import REAL, Params, Reject
from verity_numerical.checker.ctx import IntCtx
from verity_numerical.checker.step import (
    accepts_step,
    accepts_vu,
    check_step_expr,
    check_vu_expr,
)
from verity_numerical.redteam.checker_target import LaxCtx

from .babybear import (
    DEFAULT,
    BabyBearCtx,
    Embedding,
    P,
    accepts_step_bb,
    accepts_vu_bb,
    gen_step_bb,
    gen_vu_bb,
    limbs,
)

__all__ = ["BatteryReport", "Case", "build_battery", "load_battery", "main", "run_battery", "write_battery"]

VERSION = "v1"
FIXTURE_DIR = Path("fixtures/redteam/babybear-battery") / VERSION
CLASSES = ("wrap", "fractional", "zero_swap", "hint_swap", "chain", "epilogue", "z3")


@dataclass
class Case:
    id: str
    cls: str
    level: str  # "step" | "vu" | "gadget"
    params: str  # "REAL"
    inputs: dict  # step: {c, a, b}; vu: {a, b}; gadget: the gadget's inputs
    hints: dict  # the mutated hint tree (step/vu) or the gadget's hints
    claimed: dict  # step: {y}; vu: {y16}; gadget: {outputs: [...]}: what the mutated witness "proves"
    honest: dict  # the honest output(s) for the same inputs
    mutation: str  # human-readable description
    expected: str = "reject"
    verdicts: dict = field(default_factory=dict)  # int / naive / embedded: "accept" | "reject: <predicate>"
    gadget: str | None = None
    dropped_rule: str | None = None
    source: str = ""


@dataclass
class BatteryReport:
    total: int
    accepted: list[str]  # ids the prover accepted (each one is a forgery it lets through)
    by_class: dict
    ok: bool
    seconds: float

    def render(self) -> str:
        lines = [f"battery: {self.total} cases, {len(self.accepted)} accepted -> {'PASS' if self.ok else 'FAIL'}"]
        for cls, d in sorted(self.by_class.items()):
            lines.append(f"  {cls:11} {d['rejected']:4}/{d['total']:<4} rejected")
        for cid in self.accepted[:20]:
            lines.append(f"  ACCEPTED {cid}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------------
# Verdicts: the three reference checkers a case is scored against
# ---------------------------------------------------------------------------------------------------------------------


def _verdict(fn: Callable[[], tuple[bool, str]]) -> str:
    try:
        ok, why = fn()
    except Reject as e:
        return f"reject: {e}"
    except Exception as e:  # noqa: BLE001 -- a forged hint may break the arithmetic itself: a rejection
        return f"reject: raised {type(e).__name__}"
    return "accept" if ok else f"reject: {why}"


def _int_accepts(case: Case, p: Params) -> tuple[bool, str]:
    x, h = case.inputs, case.hints
    if case.level == "step":
        return accepts_step(p, x["c"], x["a"], x["b"], case.claimed["y"], h), "integer checker"
    if case.level == "vu":
        return accepts_vu(p, x["a"], x["b"], case.claimed["y16"], h), "integer checker"
    raise Reject("gadget cases are scored by z3_babybear's reference")


class NaiveCtx(BabyBearCtx):
    """Every WP2 predicate asserted mod p as written: products read mod p, no limbs, and a range wider than p
    (a 32-bit word column) left unchecked -- there is no in-field way to check it without limbs."""

    def rng(self, name, x, lo, hi):
        if hi - lo > self.p:
            return
        super().rng(name, x, lo, hi)


def _naive_accepts(case: Case, p: Params) -> tuple[bool, str]:
    ctx = NaiveCtx()
    x, h = case.inputs, case.hints
    try:
        if case.level == "step":
            ctx.eq("y", check_step_expr(ctx, p, x["c"], list(x["a"]), list(x["b"]), h), case.claimed["y"])
        elif case.level == "vu":
            ctx.eq("y16", check_vu_expr(ctx, p, list(x["a"]), list(x["b"]), h), case.claimed["y16"])
        else:
            raise Reject("gadget cases are scored by z3_babybear's reference")
    except Reject as e:
        return False, str(e)
    return True, ""


def _embedded_accepts(case: Case, p: Params, emb: Embedding = DEFAULT) -> tuple[bool, str]:
    x, h = case.inputs, case.hints
    if case.level == "step":
        return accepts_step_bb(p, x["c"], x["a"], x["b"], case.claimed["y"], h, emb)
    if case.level == "vu":
        return accepts_vu_bb(p, x["a"], x["b"], case.claimed["y16"], h, emb)
    raise Reject("gadget cases are scored by z3_babybear's reference")


def _gadget_accepts(case: Case, p: Params, emb: Embedding) -> tuple[bool, str]:
    """One embedded gadget on ints mod p: the case's inputs/hints must reproduce its claimed outputs."""
    from .z3_babybear import SPECS_BB

    spec = SPECS_BB[case.gadget]
    ctx = BabyBearCtx()
    try:
        contract = spec.contract(emb, p)
        for name, (lo, hi) in contract.inputs.items():  # the wires' guarantees from upstream gadgets (premises)
            if name in case.inputs and not lo <= case.inputs[name] < hi:
                raise Reject(f"premise {name} in [{lo}, {hi}) violated: {case.inputs[name]}")
        outs = spec.call(emb, ctx, p, dict(case.inputs), dict(case.hints))
        for i, (got, want) in enumerate(zip(outs, case.claimed["outputs"])):
            ctx.eq(f"out[{i}]", got, want)
    except Reject as e:
        return False, str(e)
    return True, ""


def score(case: Case, p: Params = REAL) -> Case:
    if case.level != "gadget":
        case.verdicts = {
            "int": _verdict(lambda: _int_accepts(case, p)),
            "naive": _verdict(lambda: _naive_accepts(case, p)),
            "embedded": _verdict(lambda: _embedded_accepts(case, p)),
        }
    else:
        from .babybear import ALL_RULES

        without = Embedding(ALL_RULES - {case.dropped_rule}) if case.dropped_rule else DEFAULT
        case.verdicts = {
            "int": "reject: integer semantics differ (z3 reference)",
            "embedded": _verdict(lambda: _gadget_accepts(case, p, DEFAULT)),
            f"without_{case.dropped_rule}": _verdict(lambda: _gadget_accepts(case, p, without)),
        }
    return case


# ---------------------------------------------------------------------------------------------------------------------
# Honest instances and the mutation engine
# ---------------------------------------------------------------------------------------------------------------------


def _lax_output(p: Params, level: str, x: dict, h: dict) -> int:
    """The output word a checker that verifies nothing forms from ``h``: the forger's claim, canonical mod p."""
    if level == "step":
        got = check_step_expr(LaxCtx(), p, x["c"], list(x["a"]), list(x["b"]), h)
    else:
        got = check_vu_expr(LaxCtx(), p, list(x["a"]), list(x["b"]), h)
    return int(got) % P


def _leaves(tree, path: str = "") -> Iterator[tuple[str, int]]:
    """``(path, value)`` for every integer leaf of a hint tree."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from _leaves(v, f"{path}[{i}]")
    elif isinstance(tree, int):
        yield path, tree


def _leaf_dicts(tree, path: str = "") -> Iterator[tuple[str, dict]]:
    """Every dict of the hint tree that holds at least two integer leaves (one gadget's hints)."""
    if isinstance(tree, dict):
        if sum(isinstance(v, int) for v in tree.values()) >= 2:
            yield path, tree
        for k, v in tree.items():
            yield from _leaf_dicts(v, f"{path}.{k}" if path else k)
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from _leaf_dicts(v, f"{path}[{i}]")


def _set_leaf(tree, path: str, value: int) -> None:
    node = tree
    parts = [p for p in path.replace("]", "").replace("[", ".").split(".") if p]
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def honest_instances(p: Params = REAL, *, limit: int = 12) -> list[tuple[str, dict]]:
    """Finite Ada golden records as ``(name, {c, a, b})``; the first ``limit`` whose hints the WP2 checker forms."""
    from .cases import golden_cases

    out: list[tuple[str, dict]] = []
    cases, _ = golden_cases()
    for name, (c, a, b) in cases:
        try:
            gen_step_bb(p, c, a, b)
        except Reject:
            continue
        out.append((name, {"c": c, "a": list(a), "b": list(b)}))
        if len(out) >= limit:
            break
    return out


def gen_step_from(p: Params, c: int, a: Sequence[int], b: Sequence[int], overrides: dict[str, int]) -> dict:
    """``gen_step`` with some hints imposed: the forger fixes ``overrides`` (path -> value) and derives every
    downstream hint honestly from the values the imposed hints produce, so the claimed output is what the mutated
    prefix actually leads to.  Nothing is checked (``LaxCtx``)."""
    from verity_numerical.checker.gadgets import (
        AlignProduct,
        AlignState,
        Bf16Decode,
        Fp32Decode,
        GroupMax,
        LeadingBit,
        Normalise,
        Pack,
        Product,
        SignedSum,
        StateOut,
    )

    def imp(path: str, hd: dict) -> dict:
        for k in list(hd):
            key = f"{path}.{k}"
            if key in overrides:
                hd[k] = overrides[key]
        return hd

    ctx = LaxCtx()
    h: dict = {"acc": imp("acc", Fp32Decode.gen(p, c)), "a": [], "b": [], "prod": [], "groups": []}
    h["acc"]["f_lo"], h["acc"]["f_hi"] = limbs(h["acc"]["f"] % 2**23)
    imp("acc", h["acc"])
    state = Fp32Decode.check(ctx, p, c, h["acc"])
    dec_a, dec_b = [], []
    for i, (wa, wb) in enumerate(zip(a, b)):
        h["a"].append(imp(f"a[{i}]", Bf16Decode.gen(p, wa)))
        h["b"].append(imp(f"b[{i}]", Bf16Decode.gen(p, wb)))
        dec_a.append(Bf16Decode.check(ctx, p, wa, h["a"][i]))
        dec_b.append(Bf16Decode.check(ctx, p, wb, h["b"][i]))
    prods = []
    for i, (da, db) in enumerate(zip(dec_a, dec_b)):
        h["prod"].append(imp(f"prod[{i}]", Product.gen(p, da[1], db[1])))
        prods.append(Product.check(ctx, p, da, db, h["prod"][i]))
    start = 0
    for g, size in enumerate(p.groups):
        group = prods[start : start + size]
        start += size
        t = f"groups[{g}]"
        hg: dict = {}
        cands = [state[1]] + [e for _, _, e in group]
        hg["max"] = imp(f"{t}.max", GroupMax.gen(p, cands))
        E, ds = GroupMax.check(ctx, p, cands, hg["max"])
        hg["acc_align"] = imp(f"{t}.acc_align", AlignState.gen(p, state[2], ds[0]))
        q_acc = AlignState.check(ctx, p, state[2], ds[0], hg["acc_align"])
        hg["prod_align"] = [
            imp(f"{t}.prod_align[{i}]", AlignProduct.gen(p, u, d))
            for i, ((u, _, _), d) in enumerate(zip(group, ds[1:]))
        ]
        q_prods = [AlignProduct.check(ctx, p, u, d, hq) for (u, _, _), d, hq in zip(group, ds[1:], hg["prod_align"])]
        terms = [(state[0], q_acc)] + [(sg, q) for (_, sg, _), q in zip(group, q_prods)]
        hg["sum"] = imp(f"{t}.sum", SignedSum.gen(p, terms))
        sgn, v = SignedSum.check(ctx, p, terms, hg["sum"])
        hg["lead"] = imp(f"{t}.lead", LeadingBit.gen(p, v))
        l = LeadingBit.check(ctx, p, v, hg["lead"])
        hg["norm"] = imp(f"{t}.norm", Normalise.gen(p, E, l, v))
        e1, M = Normalise.check(ctx, p, E, l, v, hg["norm"])
        hg["out"] = imp(f"{t}.out", StateOut.gen(p, e1, M))
        s_out, e_out, M_out, _ovf, _z = StateOut.check(ctx, p, sgn, e1, M, hg["out"])
        state = (s_out, e_out, M_out)
        h["groups"].append(hg)
    h["pack"] = imp("pack", Pack.gen(p, state[2]))
    h["pack"]["low_lo"], h["pack"]["low_hi"] = limbs(h["pack"]["low"] % 2**23)
    imp("pack", h["pack"])
    return h


def _cid(prefix: str, *parts) -> str:
    h = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:10]
    return f"{prefix}-{h}"


def _canon(tree):
    if isinstance(tree, dict):
        return {k: _canon(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_canon(v) for v in tree]
    return tree % P if isinstance(tree, int) else tree


def _mk(cls: str, level: str, x: dict, h: dict, honest_y: int, mutation: str, p: Params, source: str) -> Case:
    h = _canon(h)
    claimed = _lax_output(p, level, x, h)
    key = "y" if level == "step" else "y16"
    return Case(
        id=_cid(cls, level, source, mutation),
        cls=cls,
        level=level,
        params="REAL",
        inputs=x,
        hints=h,
        claimed={key: claimed, "as_word": claimed if claimed < 2 ** (32 if level == "step" else 16) else None},
        honest={key: honest_y},
        mutation=mutation,
        source=source,
    )


def _impose(p: Params, level: str, x: dict, h0: dict, overrides: dict[str, int]) -> dict:
    """The mutated hint tree with every downstream hint re-derived (step level); a plain overwrite otherwise."""
    if level == "step":
        return gen_step_from(p, x["c"], x["a"], x["b"], overrides)
    h = copy.deepcopy(h0)
    for path, v in overrides.items():
        _set_leaf(h, path, v)
    return h


def _wrap_cases(cls: str, level: str, x: dict, h0: dict, y0: int, name: str, p: Params, keep, dicts) -> Iterator[Case]:
    """A (u, w) pair of hints of one gadget re-split so that u' 2^s + w' = u 2^s + w + p: the +p shadow of a
    truncation as it exists in a field (two canonical elements that satisfy the same congruence)."""
    for dpath, d in dicts:
        keys = [k for k, v in d.items() if isinstance(v, int)]
        for ku in keys:
            for kw in keys:
                if ku == kw:
                    continue
                for s in (1, 8, 16, 23, 24):
                    u, w = d[ku], d[kw]
                    w2 = (w + P) % 2**s
                    u2 = (u + (w + P - w2) // 2**s) % P
                    h = _impose(p, level, x, h0, {f"{dpath}.{ku}": u2, f"{dpath}.{kw}": w2})
                    c = _mk(cls, level, x, h, y0, f"{dpath}.{ku},{kw}: ({u}, {w}) -> ({u2}, {w2}) [2^{s}, +p]", p, name)
                    if keep(c):
                        yield c


def _step_cases(p: Params, name: str, x: dict, *, keep: Callable[[Case], bool]) -> Iterator[Case]:
    h0 = gen_step_bb(p, x["c"], x["a"], x["b"])
    y0 = int(check_step_expr(IntCtx(), p, x["c"], list(x["a"]), list(x["b"]), h0))
    leaves = list(_leaves(h0))
    yield from _wrap_cases("wrap", "step", x, h0, y0, name, p, keep, _leaf_dicts(h0))
    # fractional: x -> (x 2^s + j p) / 2^s for the shifts the relation applies to that hint
    for path, v in leaves:
        for s in (1, 4, 8, 16, 23, 24, 31, 40, 48):
            ipow = 2**s
            j = (-(v * ipow) * pow(P, -1, ipow)) % ipow
            if j == 0:
                j = ipow
            xprime = ((v * ipow + j * P) // ipow) % P
            if (v * ipow + j * P) % ipow or xprime == v:
                continue
            h = _impose(p, "step", x, h0, {path: xprime})
            c = _mk("fractional", "step", x, h, y0, f"{path}: {v} -> ({v}*2^{s} + {j}p)/2^{s} = {xprime}", p, name)
            if keep(c):
                yield c
    # zero swap: +0 <-> -0 in an operand / the accumulator, honest hints kept
    for i, w in enumerate(x["a"]):
        if w in (0x0000, 0x8000):
            xz = copy.deepcopy(x)
            xz["a"][i] = w ^ 0x8000
            yield _mk("zero_swap", "step", xz, copy.deepcopy(h0), y0, f"a[{i}]: {w:#06x} -> {w ^ 0x8000:#06x}", p, name)
    for i, w in enumerate(x["b"]):
        if w in (0x0000, 0x8000):
            xz = copy.deepcopy(x)
            xz["b"][i] = w ^ 0x8000
            yield _mk("zero_swap", "step", xz, copy.deepcopy(h0), y0, f"b[{i}]: {w:#06x} -> {w ^ 0x8000:#06x}", p, name)
    if x["c"] in (0, 0x80000000):
        xz = copy.deepcopy(x)
        xz["c"] = x["c"] ^ 0x80000000
        yield _mk("zero_swap", "step", xz, copy.deepcopy(h0), y0, f"c: {x['c']:#010x} -> {xz['c']:#010x}", p, name)


def _hint_swap_cases(p: Params, inst: list[tuple[str, dict]]) -> Iterator[Case]:
    for (na, xa), (nb, xb) in itertools.pairwise(inst):
        ha, hb = gen_step_bb(p, xa["c"], xa["a"], xa["b"]), gen_step_bb(p, xb["c"], xb["a"], xb["b"])
        ya = int(check_step_expr(IntCtx(), p, xa["c"], list(xa["a"]), list(xa["b"]), ha))
        # whole tree of B under the operands of A
        yield _mk(
            "hint_swap",
            "step",
            copy.deepcopy(xa),
            copy.deepcopy(hb),
            ya,
            f"hints of {nb} under operands of {na}",
            p,
            na,
        )
        # one operand's decode + product hints from B
        for i in range(len(xa["a"])):
            h = copy.deepcopy(ha)
            h["a"][i], h["b"][i], h["prod"][i] = hb["a"][i], hb["b"][i], hb["prod"][i]
            yield _mk("hint_swap", "step", copy.deepcopy(xa), h, ya, f"a[{i}]/b[{i}]/prod[{i}] hints from {nb}", p, na)
        # the accumulator decode from B
        h = copy.deepcopy(ha)
        h["acc"] = copy.deepcopy(hb["acc"])
        yield _mk("hint_swap", "step", copy.deepcopy(xa), h, ya, f"acc decode hints from {nb}", p, na)


def _chain_cases(p: Params, inst: list[tuple[str, dict]], *, keep: Callable[[Case], bool]) -> Iterator[Case]:
    from verity_numerical.checker.gadgets import Epilogue

    for (na, xa), (nb, xb) in itertools.pairwise(inst):
        x = {"a": list(xa["a"]) + list(xb["a"]), "b": list(xa["b"]) + list(xb["b"])}
        try:
            h0 = gen_vu_bb(p, x["a"], x["b"])
        except Reject:
            continue
        y0 = int(check_vu_expr(IntCtx(), p, x["a"], x["b"], h0))
        # the word handed from unit 1 to unit 2, as an integer
        c1 = int(check_step_expr(IntCtx(), p, 0, x["a"][: len(xa["a"])], x["b"][: len(xa["b"])], h0["steps"][0]))
        for j in (1, 2):
            c_shadow = c1 + j * P
            if c_shadow >= 2**32:
                continue
            try:
                h2 = gen_step_bb(p, c_shadow, xb["a"], xb["b"])  # unit 2 decodes the shadow as a different FP32 word
                c2 = int(check_step_expr(IntCtx(), p, c_shadow, list(xb["a"]), list(xb["b"]), h2))
            except Reject:
                continue
            h = copy.deepcopy(h0)
            h["steps"][1] = h2
            h["epilogue"] = Epilogue.gen(p, c2)
            c = _mk(
                "chain",
                "vu",
                x,
                h,
                y0,
                f"unit-2 accumulator {c1:#010x} -> {c1} + {j}p = {c_shadow:#010x}",
                p,
                f"{na}+{nb}",
            )
            if keep(c):
                yield c
        # epilogue: re-split pairs of the RN-even epilogue's hints
        yield from _wrap_cases("epilogue", "vu", x, h0, y0, f"{na}+{nb}", p, keep, [("epilogue", h0["epilogue"])])


def _tie_case(p: Params, *, keep: Callable[[Case], bool]) -> Iterator[Case]:
    """The RN-even tie boundary: a chain whose final accumulator has low half exactly 0x8000, then +-p on ``lo``."""
    k = sum(p.groups)
    # 1.0 * (1 + 2^-7) accumulated once: c = 0x3F810000 -> low half 0; we want a tie: pick a*b with bit 15 set
    for w in (0x3F81, 0x3F83, 0x3F87, 0x3F8F, 0x3FFF):
        a = [w] + [0] * (k - 1)
        b = [0x3F81] + [0] * (k - 1)
        try:
            h0 = gen_vu_bb(p, a, b)
        except Reject:
            continue
        c = int(check_step_expr(IntCtx(), p, 0, a, b, h0["steps"][0]))
        if c & 0xFFFF != 0x8000:
            continue
        x = {"a": a, "b": b}
        y0 = int(check_vu_expr(IntCtx(), p, a, b, h0))
        yield from _wrap_cases("epilogue", "vu", x, h0, y0, f"tie {c:#010x}", p, keep, [("epilogue", h0["epilogue"])])
        # the tie flipped: lsb / carry hints toggled with the rest honest (the rounding direction)
        for key in ("lsb", "carry"):
            if key in h0["epilogue"]:
                h = copy.deepcopy(h0)
                h["epilogue"][key] ^= 1
                cs = _mk("epilogue", "vu", x, h, y0, f"tie {c:#010x}; epilogue.{key} toggled", p, "tie")
                yield score(cs, p)


def z3_cases(z3_json: dict) -> list[Case]:
    """Gadget-level cases from ``z3_babybear --json``: the first SAT model of each (gadget, dropped rule)."""
    out: list[Case] = []
    for v in z3_json.get("verdicts", []):
        if str(v.get("status", "")).lower() != "sat" or not v.get("models"):
            continue
        m = v["models"][0]
        out.append(
            score(
                Case(
                    id=_cid("z3", v["gadget"], v.get("dropped") or "all", json.dumps(m, sort_keys=True)),
                    cls="z3",
                    level="gadget",
                    params="REAL",
                    inputs=m["inputs"],
                    hints=m["hints"],
                    claimed={"outputs": m["outputs"]},
                    honest={"outputs": m["reference"]},
                    mutation=f"z3 model with rule {v.get('dropped') or '(none)'} dropped; pins {m.get('pins')}",
                    gadget=v["gadget"],
                    dropped_rule=v.get("dropped") or None,
                    source="z3_babybear",
                )
            )
        )
    return out


def build_battery(p: Params = REAL, *, z3_json: dict | None = None, teeth_only_shadow: bool = True) -> list[Case]:
    """Every case, scored.  ``shadow``/``fractional``/``chain``/``epilogue`` cases are kept when the naive mod-p checker
    accepts them (teeth); one rejected specimen per (class, hint path) is kept too so the coverage is visible."""
    inst = honest_instances(p)
    seen_paths: set[tuple[str, str]] = set()

    def keep(c: Case) -> bool:
        if c.claimed.get("y", c.claimed.get("y16")) == c.honest.get("y", c.honest.get("y16")) and c.cls not in (
            "zero_swap",
        ):
            return False  # the mutated witness proves the honest word: not a forgery
        score(c, p)
        if c.verdicts["naive"] == "accept" or c.cls in ("hint_swap", "zero_swap", "chain"):
            return True
        if not teeth_only_shadow:
            return True
        path = c.mutation.split(":")[0].split(",")[0].rsplit(".", 1)[0]  # the gadget's hint dict
        key = (c.cls, path)
        if key in seen_paths:
            return False
        seen_paths.add(key)
        return True

    cases: list[Case] = []
    for name, x in inst[:4]:
        cases.extend(_step_cases(p, name, x, keep=keep))
    cases.extend(c for c in _hint_swap_cases(p, inst[:4]) if keep(c))
    cases.extend(_chain_cases(p, inst[:5], keep=keep))
    cases.extend(_tie_case(p, keep=keep))
    if z3_json:
        cases.extend(z3_cases(z3_json))
    # dedupe by id
    uniq: dict[str, Case] = {}
    for c in cases:
        uniq.setdefault(c.id, c)
    return list(uniq.values())


# ---------------------------------------------------------------------------------------------------------------------
# Serialisation and the lanes' entry point
# ---------------------------------------------------------------------------------------------------------------------


def write_battery(cases: Sequence[Case], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.json"):
        f.unlink()
    entries = []
    for c in cases:
        path = out / f"{c.id}.json"
        path.write_text(json.dumps(asdict(c), indent=1, sort_keys=True) + "\n")
        entries.append(
            {
                "id": c.id,
                "class": c.cls,
                "level": c.level,
                "gadget": c.gadget,
                "dropped_rule": c.dropped_rule,
                "naive": c.verdicts.get("naive", ""),
                "file": path.name,
            }
        )
    by_class = {cls: sum(1 for c in cases if c.cls == cls) for cls in CLASSES}
    teeth = sum(1 for c in cases if c.verdicts.get("naive") == "accept")
    manifest = {
        "version": VERSION,
        "field": {"name": "BabyBear", "p": P},
        "relation": "WP2 reference checker (verity_numerical.checker), params REAL, hint layout of gen_step + limb hints",
        "expected": "reject (every case)",
        "total": len(cases),
        "by_class": by_class,
        "naive_modp_accepts": teeth,
        "int_checker_accepts": sum(1 for c in cases if c.verdicts.get("int") == "accept"),
        "embedded_accepts": sum(1 for c in cases if c.verdicts.get("embedded") == "accept"),
        "cases": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def load_battery(root: Path | None = None) -> list[Case]:
    root = root or _repo_root() / FIXTURE_DIR
    manifest = json.loads((root / "manifest.json").read_text())
    return [Case(**json.loads((root / e["file"]).read_text())) for e in manifest["cases"]]


def run_battery(prover_accepts: Callable[[Case], bool], cases: Sequence[Case] | None = None) -> BatteryReport:
    """Run every case through ``prover_accepts(case) -> bool``; a BabyBear prover passes iff it accepts none."""
    cases = list(cases) if cases is not None else load_battery()
    t0 = time.time()
    accepted: list[str] = []
    by_class: dict[str, dict] = {}
    for c in cases:
        d = by_class.setdefault(c.cls, {"total": 0, "rejected": 0})
        d["total"] += 1
        try:
            ok = bool(prover_accepts(c))
        except Exception:  # noqa: BLE001 -- the prover choking on a forged witness is a rejection
            ok = False
        if ok:
            accepted.append(c.id)
        else:
            d["rejected"] += 1
    return BatteryReport(len(cases), accepted, by_class, not accepted, time.time() - t0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z3", type=Path, default=None, help="z3_babybear --json output: SAT models become gadget cases")
    ap.add_argument("--out", type=Path, default=_repo_root() / FIXTURE_DIR)
    args = ap.parse_args(argv)
    z3_json = json.loads(args.z3.read_text()) if args.z3 else None
    t0 = time.time()
    cases = build_battery(REAL, z3_json=z3_json)
    manifest = write_battery(cases, args.out)
    print(f"wrote {manifest['total']} cases to {args.out} in {time.time() - t0:.1f}s")
    print(f"  by class: {manifest['by_class']}")
    print(
        f"  naive mod-p accepts {manifest['naive_modp_accepts']}; int checker accepts {manifest['int_checker_accepts']}; "
        f"embedded accepts {manifest['embedded_accepts']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
