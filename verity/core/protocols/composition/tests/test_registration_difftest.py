"""The registration log's Lean (`core-registration serve`: `Core.Registration.protocol`,
`verity/registration/log/v1`, in the core interface's package) against its Python (`verity.core.protocols.registration`). Hypothesis generates a log for
either composite, from the honest example site and its information (`EXAMPLE`, the Lean's `Core.Registration.Example`)
with a few changes, or from small random entries, chained or with a broken `previous`; it checks it under the
example's auditor or a changed one, and sometimes breaks the JSON of the log, the params or the evidence. The two must
agree on every field of the outcome, and on the reason when the check refuses a log (`entry I: REFUSAL`, a composite
the params don't name, or a log or params outside ASCII or holding DEL): the public view of an accepted log among them."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.core import site as S
from verity.core.protocols import registration as G
from verity_difftest import protocol as P

INTERFACE = Path(__file__).resolve().parents[2] / "interface"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

SHEET = S.Sheet(S.DeviceFigures("sim-a", 2_000_000_000, (10**15, 2 * 10**15, 2 * 10**15, 15 * 10**14, 4 * 10**15)), 3200,
                1 << 25, 64 << 30)
SHEET_B = S.Sheet(S.DeviceFigures("sim-b", 1_000_000_000, (5 * 10**14, 10**15, 10**15, 75 * 10**13, 2 * 10**15)), 6400,
                  1 << 26, 128 << 30)
AUDITOR = G.Auditor((SHEET,), Fraction(1, 20), (("cert0", "k0"), ("chal0", "k1")))


def stated(k: S.Sheet, **figures: object) -> S.Sheet:
    """`k` with some of its figures restated."""
    return dataclasses.replace(k, figures=dataclasses.replace(k.figures, **figures))


#: Sheets stated against `SHEET`'s at a twentieth's tolerance: within it at either edge, and just past it slow, fast or
#: in one peak rate.
RATES = SHEET.figures.peak_rate
WITHIN = (stated(SHEET, clock=1_900_000_000), stated(SHEET, clock=2_100_000_000, peak_rate=(*RATES[:4], 38 * 10**14)))
PAST = (stated(SHEET, clock=1_899_999_999), stated(SHEET, clock=2_100_000_001),
        stated(SHEET, peak_rate=(RATES[0], 21 * 10**14 + 1, *RATES[2:])))
#: A sheet with a zero peak rate, refused at any tolerance.
ZERO = stated(SHEET, peak_rate=(*RATES[:3], 0, RATES[4]))
LOCS = (S.Loc("site0", "site"), S.Loc("node0", "host", "site0"), S.Loc("gpu0", "device", "node0", sheet=SHEET.digest()),
        S.Loc("hbm0", "memory", "gpu0", tier="hbm", bytes=1 << 30),
        S.Loc("cert0", "certifier", "site0", owner="auditor", key="k0"),
        S.Loc("chal0", "challenger", "site0", owner="auditor", key="k1"))
LINKS = (S.Link("up0", ("node0", "cert0"), 100, 1), S.Link("up1", ("cert0", S.OUTSIDE), 100, 1),
         S.Link("mc0", ("chal0", "node0"), 10, 1))
WEIGHTS = {"checkpoint_sha512": "c" * 128, "domain": "b" * 128, "kind": "theta", "leaves": 4, "root": "a" * 128,
           "value": "weights", "words": 64}
ITEMS = (WEIGHTS, {"domain": "e" * 128, "kind": "theta", "leaves": 1, "root": "d" * 128, "value": "state",
                   "word_bits": 32, "words": 8},
         {"advice": "verity/advice/v1", "cap": 32, "digest": "f" * 128, "kind": "builder"})
EXAMPLE = (G.site((SHEET,), LOCS, LINKS), G.Information(ITEMS))


def _wrap(f):
    def decode(x):
        try:
            return f(x)
        except G.Malformed as e:
            raise D.Reject(str(e)) from None
    return decode


def _evidence(b: bytes) -> None:
    if P.canonical_json(b) != {}:
        raise D.Reject("the evidence: the keys must be []")


def reference(case: Any) -> dict:
    return P.answer(G.PROTOCOL, case, params=_wrap(G.params_of), commitment=_wrap(lambda b: G.log_of(P.canonical_json(b))),
                    evidence=_evidence, check=lambda p, c, _: G.check_log(p, c),
                    digest=lambda c: bytes.fromhex(G.head(*c)))


def _digest(e: G.Entry) -> str:
    """An entry's digest through `D.canonical`, the same bytes on ASCII, which also writes an entry past ASCII or with
    DEL (`wire.canonical` raises on one), so a log can hold one anywhere for the check to refuse."""
    j = e.to_json()
    return hashlib.sha512(j["format"].encode() + b"\0" + D.canonical(j).encode()).hexdigest()


def chain(composite: str, entries: Any) -> list[G.Entry]:
    """`G.chain`, through `_digest`."""
    log: list[G.Entry] = []
    for e in entries:
        log.append(dataclasses.replace(e, previous=_digest(log[-1]) if log else S.genesis(composite)))
    return log


def case(composite: str, log: list[G.Entry], auditor: G.Auditor = AUDITOR, composed: str | None = None) -> dict:
    commitment = D.canonical({"composite": composed or composite, "entries": [e.to_json() for e in log]})
    return {"params": G.params(composite, auditor), "commitment": commitment.encode().hex(), "evidence": b"{}".hex()}


@dataclasses.dataclass(frozen=True)
class RegistrationPort(D.ProtocolPort):
    """A `ProtocolPort` that also compares the check's own reasons: the refused entry and its refusal."""

    def key(self, outcome: D.Outcome) -> Any:
        if outcome[0] == "error":
            return outcome
        out = json.loads(outcome[1])
        reason = out.get("reason") or ""
        if not reason.startswith(("entry ", "composite: ", "commitment: not ASCII")):
            out.pop("reason")
        return out


IDS = ["site0", "node0", "node1", "gpu0", "hbm0", "cert0", "chal0", S.OUTSIDE, "up0"]
HEX = st.sampled_from(["a" * 128, "b" * 128, "e" * 128])
locs = st.builds(S.Loc, st.sampled_from(IDS), st.sampled_from(S.KINDS), st.none() | st.sampled_from(IDS),
                 st.sampled_from(S.OWNERS), st.sampled_from(["", SHEET.digest(), SHEET_B.digest()]),
                 st.none() | st.sampled_from(S.TIERS), st.sampled_from([0, 1, 1 << 30, (64 << 30) + 1]),
                 st.sampled_from(["", "k0", "k1"]))
links = st.builds(S.Link, st.sampled_from(["up0", "up1", "mc0", "l3", "gpu0"]),
                  st.tuples(st.sampled_from(IDS), st.sampled_from(IDS)), st.sampled_from([0, 1, 100]),
                  st.sampled_from([0, 1]))
sheets = st.sampled_from([SHEET, SHEET_B, dataclasses.replace(SHEET, mem=1), *WITHIN, *PAST, ZERO])
#: Names a θ value takes: the example's, a new one, one outside ASCII and one with DEL.
VALUES = ["weights", "state", "tables", "poids\u00e9", "poids\x7f"]


def _theta(value: str, root: str, leaves: int, wide: bool, pin: str | None) -> dict:
    return {"domain": "b" * 128, "kind": "theta", "leaves": leaves, "root": root, "value": value, "words": 64,
            **({"word_bits": 32} if wide else {}), **({"checkpoint_sha512": pin} if pin else {})}


items = st.one_of(
    st.builds(_theta, st.sampled_from(VALUES), HEX, st.sampled_from([0, 4]), st.booleans(), st.none() | HEX),
    st.builds(lambda d, a, c: {"advice": a, "cap": c, "digest": d, "kind": "builder"}, HEX,
              st.sampled_from(["verity/advice/v1", "verity/advice/v2"]), st.sampled_from([0, 32])))
site = st.builds(G.site, st.lists(sheets, max_size=2).map(tuple), st.lists(locs, max_size=4).map(tuple),
                     st.lists(links, max_size=3).map(tuple))
information = st.builds(G.Information, st.lists(items, max_size=3).map(tuple))


def pick(draw, xs):
    """One of `xs`, uniformly: Hypothesis's integers lean to their low end."""
    return draw(st.sampled_from(list(xs)))


#: Values a location's or link's field takes when a change sets it.
FIELDS = {"kind": S.KINDS, "parent": [None, *IDS], "owner": S.OWNERS, "sheet": ["", SHEET.digest(), SHEET_B.digest()],
          "tier": [None, *S.TIERS], "bytes": [0, 1, 64 << 30, (64 << 30) + 1], "key": ["", "k0", "k1"],
          "ends": [("node0", S.OUTSIDE), ("node0", "node0"), ("chal0", "hbm0"), ("cert0", "node0"), ("up0", "node0")],
          "beta": [0, 1], "id": IDS}


#: Site entries that, after the example's, each fail one check of their own: an uplink no certifier splits
#: (`uncertified`, onsite), a host's memory no challenger reaches (`challenger-reach`, onsite), a device's memory over its
#: sheet's (`capacity`), a sheet the auditor has no data sheet for (`sheet-digest`), a certifier that splits no uplink
#: (`kind`).
TWISTS = (G.site(links=(S.Link("l3", ("node0", S.OUTSIDE), 1, 0),)),
          G.site(locs=(S.Loc("node1", "host", "site0"), S.Loc("dram1", "memory", "node1", tier="dram", bytes=1))),
          G.site(locs=(S.Loc("hbm1", "memory", "gpu0", tier="hbm", bytes=64 << 30),)),
          G.site((SHEET_B,), (S.Loc("gpu1", "device", "node0", sheet=SHEET_B.digest()),)),
          G.site(locs=(S.Loc("cert1", "certifier", "site0", owner="auditor", key="k0"),)))


@st.composite
def changed(draw) -> list:
    """The example, sometimes with a twist after it, and with up to three changes: a location, a link, a sheet or an item
    dropped, replaced, added or with one field set, a site entry split in two, or entries swapped."""
    bodies = list(EXAMPLE) + ([pick(draw, TWISTS)] if pick(draw, [True, False]) else [])
    for _ in range(pick(draw, [0, 0, 0, 1, 1, 2, 3])):
        i = pick(draw, range(len(bodies)))
        b = bodies[i]
        op = pick(draw, ["drop", "drop", "set", "set", "set", "replace", "add", "split", "swap"])
        if op == "swap":
            j = pick(draw, range(len(bodies)))
            bodies[i], bodies[j] = bodies[j], bodies[i]
        elif isinstance(b, G.Information):
            xs = list(b.items)
            if op in ("drop", "replace", "set") and xs:
                k = pick(draw, range(len(xs)))
                xs[k:k + 1] = [] if op == "drop" else [draw(items)]
            else:
                xs.append(draw(items))
            bodies[i] = G.Information(tuple(xs))
        else:
            field = pick(draw, ["sheets", "locs", "locs", "links", "links"])
            xs = list(getattr(b, field))
            new = {"sheets": sheets, "locs": locs, "links": links}[field]
            if op == "split" and len(xs) > 1:
                k = pick(draw, range(1, len(xs)))
                bodies[i:i + 1] = [dataclasses.replace(b, **{field: tuple(xs[:k])}),
                                   G.site(**{field: tuple(xs[k:])})]
                continue
            if op == "set" and xs and field != "sheets":
                k = pick(draw, range(len(xs)))
                names = [f.name for f in dataclasses.fields(xs[k]) if f.name in FIELDS]
                name = pick(draw, names)
                xs[k] = dataclasses.replace(xs[k], **{name: pick(draw, FIELDS[name])})
            elif op in ("drop", "replace", "set") and xs:
                k = pick(draw, range(len(xs)))
                xs[k:k + 1] = [] if op == "drop" else [draw(new)]
            else:
                xs.append(draw(new))
            bodies[i] = dataclasses.replace(b, **{field: tuple(xs)})
    return bodies


def _break(draw, j: Any) -> Any:
    """`j` with one value somewhere replaced by a JSON value of another shape, or a key dropped."""
    if isinstance(j, dict) and j:
        k = pick(draw, sorted(j))
        if draw(st.booleans()):
            return {x: v for x, v in j.items() if x != k}
        return {**j, k: _break(draw, j[k])}
    if isinstance(j, list) and j:
        k = pick(draw, range(len(j)))
        return [*j[:k], _break(draw, j[k]), *j[k + 1:]]
    return pick(draw, [None, -1, True, "A" * 128, "x", [], {}])


@st.composite
def cases(draw) -> dict:
    composite = pick(draw, S.COMPOSITES)
    source = pick(draw, ["changed"] * 5 + ["twist"] * 3 + ["random"] * 2)
    if source == "twist":
        return case(composite, chain(composite, [*EXAMPLE, pick(draw, TWISTS)]))
    bodies = draw(changed()) if source == "changed" else draw(st.lists(st.one_of(site, information), max_size=3))
    log = chain(composite, bodies)
    if log and not pick(draw, range(8)):
        k = pick(draw, range(len(log)))
        log[k] = dataclasses.replace(log[k], previous=pick(draw, ["0" * 128, S.genesis("remote")]))
    auditor = pick(draw, [AUDITOR] * 6 + [G.Auditor(), dataclasses.replace(AUDITOR, sheets=(SHEET_B,)),
                                         dataclasses.replace(AUDITOR, tolerance=0),
                                         dataclasses.replace(AUDITOR, tolerance=1),
                                         dataclasses.replace(AUDITOR, keys=(("cert0", "k1"), ("chal0", "k1"))),
                                         dataclasses.replace(AUDITOR, keys=(("cert0", "k\u00e9"), ("chal0", "k1"))),
                                         dataclasses.replace(AUDITOR, keys=(("cert0", "k\x7f"), ("chal0", "k1")))])
    composed = composite if pick(draw, range(16)) else pick(draw, S.COMPOSITES)
    c = case(composite, log, auditor, composed)
    broken = pick(draw, ["none"] * 12 + ["commitment", "params", "evidence"])
    if broken == "commitment":
        j = json.loads(bytes.fromhex(c["commitment"]))
        c["commitment"] = D.canonical(_break(draw, j)).encode().hex()
    elif broken == "params":
        c["params"] = _break(draw, c["params"])
    elif broken == "evidence":
        c["evidence"] = pick(draw, [b"", b"[]", b'{"a":1}', b"{ }"]).hex()
    return c


PORT = RegistrationPort(exe="core-registration", package=INTERFACE, reference=reference, inputs=cases())


def test_the_example_and_its_refusals_agree_with_the_lean_guards() -> None:
    """The Lean's `#guard`s on `Core.Registration.Example`, in Python, and each replayed against the Lean."""
    bare = [G.site((SHEET,), LOCS[:4], (S.Link("up0", ("node0", S.OUTSIDE), 100, 1),))]
    ascii = "commitment: not ASCII without DEL (a byte at or above 0x7f)"
    expect = [(case("onsite", chain("onsite", EXAMPLE)), None),
              (case("onsite", chain("remote", EXAMPLE)), "entry 0: previous"),
              (case("onsite", chain("onsite", bare)), "entry 0: uncertified"),
              (case("remote", chain("remote", bare)), None),
              (case("onsite", chain("onsite", [G.site((SHEET,), LOCS[:5], LINKS[:2])])), "entry 0: challenger-reach"),
              (case("onsite", chain("onsite", [*EXAMPLE, G.Information(ITEMS[:1])])), "entry 2: duplicate-information"),
              (case("onsite", chain("onsite", [*EXAMPLE, G.Information(ITEMS[2:])])), "entry 2: duplicate-information"),
              (case("onsite", chain("onsite", [*EXAMPLE, G.Information(({**WEIGHTS, "value": "tables"},))])), None),
              (case("onsite", chain("remote", EXAMPLE), composed="remote"), "composite: the log is remote's, not onsite's"),
              *((case("remote", chain("remote", [G.site((k,))])), None) for k in WITHIN),
              *((case("remote", chain("remote", [G.site((k,))])), "entry 0: figures") for k in PAST),
              (case("remote", chain("remote", [G.site((dataclasses.replace(SHEET, mem=1),))])), "entry 0: figures"),
              (case("remote", chain("remote", [G.site((stated(SHEET, sku="sim-b"),))])), "entry 0: sheet-digest"),
              (case("remote", chain("remote", [G.site((WITHIN[0],))]), dataclasses.replace(AUDITOR, tolerance=0)),
               "entry 0: figures"),
              (case("remote", chain("remote", [G.site((stated(SHEET, clock=0),))]),
                    dataclasses.replace(AUDITOR, tolerance=1)), None),
              (case("remote", chain("remote", [G.site((ZERO,))]), dataclasses.replace(AUDITOR, tolerance=1)),
               "entry 0: figures"),
              (case("onsite", chain("onsite", [G.Information(({**WEIGHTS, "value": "poids\u00e9"},))])), ascii),
              (case("onsite", chain("onsite", [G.Information(({**WEIGHTS, "value": "poids\x7f"},))])), ascii),
              (case("onsite", chain("onsite", [G.Information(({**WEIGHTS, "value": "poids\x7f"},)), *EXAMPLE])), ascii)]
    with D.Server(PORT.argv()) as server:
        for c, reason in expect:
            out = reference(c)
            assert out["reason"] == reason, (out["reason"], reason)
            assert out["verdict"] == ("accept" if reason is None else "reject")
            D.replay(PORT, c, server)
    view = G.check(AUDITOR, "onsite", chain("onsite", EXAMPLE)).view()
    assert view.value("weights") == WEIGHTS and view.theta == ITEMS[:2] and view.builder == ITEMS[2]
    assert view.site.locs == LOCS


def test_registration_difftest() -> None:
    stats = D.run(PORT, examples=1000, seed=0)
    assert stats.counts.get("accept", 0) >= 50 and stats.counts.get("reject", 0) >= 200, stats.counts


def test_a_reference_without_the_challengers_reach_is_caught() -> None:
    """A Python that skips onsite's `challenger-reach` disagrees with the Lean on some drawn log."""
    real = G.site_refusal

    def lax(auditor, composite, s, head, e):
        r = real(auditor, composite, s, head, e)
        return None if r == "challenger-reach" else r

    G.site_refusal = lax
    try:
        with pytest.raises(D.Mismatch):
            D.run(PORT, examples=1000, seed=0)
    finally:
        G.site_refusal = real
