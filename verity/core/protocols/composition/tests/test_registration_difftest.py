"""The registration log's Lean (`core-registration serve`: `Core.Registration.protocol`,
`verity/registration/log/v1`, in the core interface's package) against its Python (`verity.core.protocols.registration`). Hypothesis generates a log for
either composite, from the honest example site and its information (`EXAMPLE`, the Lean's `Core.Registration.Example`)
with a few changes, or from small random entries, chained or with a broken `previous`; it checks it under the
example's auditor or a changed one, and sometimes breaks the JSON of the log, the params or the evidence. The two must
agree on every field of the outcome, and on the reason when the check refuses a log (`entry I: REFUSAL`, a composite
the params don't name, or a log or params outside ASCII): the public view of an accepted log among them."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.core.protocols import registration as G
from verity_difftest import protocol as P

INTERFACE = Path(__file__).resolve().parents[2] / "interface"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

CARD = G.Card("sim-a", 2_000_000_000, (10**15, 2 * 10**15, 2 * 10**15, 15 * 10**14, 4 * 10**15), 3200, 1 << 25,
              64 << 30)
CARD_B = G.Card("sim-b", 1_000_000_000, (5 * 10**14, 10**15, 10**15, 75 * 10**13, 2 * 10**15), 6400, 1 << 26,
                128 << 30)
AUDITOR = G.Auditor((CARD,), (("cert0", "k0"), ("chal0", "k1")), (1, 20))
#: Cards stated against `CARD`'s sheet at a twentieth's tolerance: within it at either edge, and just past it slow,
#: fast or in one peak rate.
WITHIN = (dataclasses.replace(CARD, clock=1_900_000_000),
          dataclasses.replace(CARD, clock=2_100_000_000, peak_rate=(*CARD.peak_rate[:4], 38 * 10**14)))
PAST = (dataclasses.replace(CARD, clock=1_899_999_999), dataclasses.replace(CARD, clock=2_100_000_001),
        dataclasses.replace(CARD, peak_rate=(10**15, 21 * 10**14 + 1, *CARD.peak_rate[2:])))
#: A card with a zero peak rate, refused at any tolerance.
ZERO = dataclasses.replace(CARD, peak_rate=(*CARD.peak_rate[:3], 0, CARD.peak_rate[4]))
LOCS = (G.Loc("site0", "site"), G.Loc("node0", "host", "site0"), G.Loc("gpu0", "device", "node0", card=CARD.digest()),
        G.Loc("hbm0", "memory", "gpu0", tier="hbm", bytes=1 << 30),
        G.Loc("cert0", "certifier", "site0", owner="auditor", key="k0"),
        G.Loc("chal0", "challenger", "site0", owner="auditor", key="k1"))
LINKS = (G.Link("up0", ("node0", "cert0"), 100, 1), G.Link("up1", ("cert0", G.OUTSIDE), 100, 1),
         G.Link("mc0", ("chal0", "node0"), 10, 1))
WEIGHTS = {"checkpoint_sha512": "c" * 128, "domain": "b" * 128, "kind": "theta", "leaves": 4, "root": "a" * 128,
           "value": "weights", "words": 64}
ITEMS = (WEIGHTS, {"domain": "e" * 128, "kind": "theta", "leaves": 1, "root": "d" * 128, "value": "state",
                   "word_bits": 32, "words": 8},
         {"advice": "verity/advice/v1", "cap": 32, "digest": "f" * 128, "kind": "builder"})
EXAMPLE = (G.Hardware((CARD,), LOCS, LINKS), G.Information(ITEMS))


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


def case(composite: str, log: list[G.Entry], auditor: G.Auditor = AUDITOR, composed: str | None = None) -> dict:
    return {"params": G.params(composite, auditor), "commitment": G.committed(composed or composite, log).hex(),
            "evidence": b"{}".hex()}


@dataclasses.dataclass(frozen=True)
class RegistrationPort(D.ProtocolPort):
    """A `ProtocolPort` that also compares the check's own reasons: the refused entry and its refusal."""

    def key(self, outcome: D.Outcome) -> Any:
        if outcome[0] == "error":
            return outcome
        out = json.loads(outcome[1])
        reason = out.get("reason") or ""
        if not reason.startswith(("entry ", "composite: ", "commitment: ascii", "params: ascii")):
            out.pop("reason")
        return out


IDS = ["site0", "node0", "node1", "gpu0", "hbm0", "cert0", "chal0", G.OUTSIDE, "up0"]
HEX = st.sampled_from(["a" * 128, "b" * 128, "e" * 128])
locs = st.builds(G.Loc, st.sampled_from(IDS), st.sampled_from(G.KINDS), st.none() | st.sampled_from(IDS),
                 st.sampled_from(G.OWNERS), st.sampled_from(["", CARD.digest(), CARD_B.digest()]),
                 st.none() | st.sampled_from(G.TIERS), st.sampled_from([0, 1, 1 << 30, (64 << 30) + 1]),
                 st.sampled_from(["", "k0", "k1"]))
links = st.builds(G.Link, st.sampled_from(["up0", "up1", "mc0", "l3", "gpu0"]),
                  st.tuples(st.sampled_from(IDS), st.sampled_from(IDS)), st.sampled_from([0, 1, 100]),
                  st.sampled_from([0, 1]))
cards = st.sampled_from([CARD, CARD_B, dataclasses.replace(CARD, mem=1), *WITHIN, *PAST, ZERO])
#: Names a θ value takes: the example's, a new one, and one outside ASCII.
VALUES = ["weights", "state", "tables", "poids\u00e9"]


def _theta(value: str, root: str, leaves: int, wide: bool, pin: str | None) -> dict:
    return {"domain": "b" * 128, "kind": "theta", "leaves": leaves, "root": root, "value": value, "words": 64,
            **({"word_bits": 32} if wide else {}), **({"checkpoint_sha512": pin} if pin else {})}


items = st.one_of(
    st.builds(_theta, st.sampled_from(VALUES), HEX, st.sampled_from([0, 4]), st.booleans(), st.none() | HEX),
    st.builds(lambda d, a, c: {"advice": a, "cap": c, "digest": d, "kind": "builder"}, HEX,
              st.sampled_from(["verity/advice/v1", "verity/advice/v2"]), st.sampled_from([0, 32])))
hardware = st.builds(G.Hardware, st.lists(cards, max_size=2).map(tuple), st.lists(locs, max_size=4).map(tuple),
                     st.lists(links, max_size=3).map(tuple))
information = st.builds(G.Information, st.lists(items, max_size=3).map(tuple))


def pick(draw, xs):
    """One of `xs`, uniformly: Hypothesis's integers lean to their low end."""
    return draw(st.sampled_from(list(xs)))


#: Values a location's or link's field takes when a change sets it.
FIELDS = {"kind": G.KINDS, "parent": [None, *IDS], "owner": G.OWNERS, "card": ["", CARD.digest(), CARD_B.digest()],
          "tier": [None, *G.TIERS], "bytes": [0, 1, 64 << 30, (64 << 30) + 1], "key": ["", "k0", "k1"],
          "ends": [("node0", G.OUTSIDE), ("node0", "node0"), ("chal0", "hbm0"), ("cert0", "node0"), ("up0", "node0")],
          "beta": [0, 1], "id": IDS}


#: Hardware entries that, after the example's, each fail one check of their own: an uplink no certifier splits
#: (`uncertified`, onsite), a host's memory no challenger reaches (`challenger-reach`, onsite), a device's memory over its
#: card's (`capacity`), a card the auditor has no sheet for (`card-digest`), a certifier that splits no uplink (`kind`).
TWISTS = (G.Hardware(links=(G.Link("l3", ("node0", G.OUTSIDE), 1, 0),)),
          G.Hardware(locs=(G.Loc("node1", "host", "site0"), G.Loc("dram1", "memory", "node1", tier="dram", bytes=1))),
          G.Hardware(locs=(G.Loc("hbm1", "memory", "gpu0", tier="hbm", bytes=64 << 30),)),
          G.Hardware((CARD_B,), (G.Loc("gpu1", "device", "node0", card=CARD_B.digest()),)),
          G.Hardware(locs=(G.Loc("cert1", "certifier", "site0", owner="auditor", key="k0"),)))


@st.composite
def changed(draw) -> list:
    """The example, sometimes with a twist after it, and with up to three changes: a location, a link, a card or an item dropped, replaced, added or with
    one field set, a hardware entry split in two, or entries swapped."""
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
            field = pick(draw, ["cards", "locs", "locs", "links", "links"])
            xs = list(getattr(b, field))
            new = {"cards": cards, "locs": locs, "links": links}[field]
            if op == "split" and len(xs) > 1:
                k = pick(draw, range(1, len(xs)))
                bodies[i:i + 1] = [dataclasses.replace(b, **{field: tuple(xs[:k])}),
                                   G.Hardware(**{field: tuple(xs[k:])})]
                continue
            if op == "set" and xs and field != "cards":
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
    composite = pick(draw, G.COMPOSITES)
    source = pick(draw, ["changed"] * 5 + ["twist"] * 3 + ["random"] * 2)
    if source == "twist":
        return case(composite, G.chain(composite, [*EXAMPLE, pick(draw, TWISTS)]))
    bodies = draw(changed()) if source == "changed" else draw(st.lists(st.one_of(hardware, information), max_size=3))
    log = G.chain(composite, bodies)
    if log and not pick(draw, range(8)):
        k = pick(draw, range(len(log)))
        log[k] = dataclasses.replace(log[k], previous=pick(draw, ["0" * 128, G.genesis("remote")]))
    auditor = pick(draw, [AUDITOR] * 6 + [G.Auditor(), G.Auditor((CARD_B,), AUDITOR.keys),
                                         G.Auditor(AUDITOR.sheets, AUDITOR.keys, (0, 7)),
                                         G.Auditor(AUDITOR.sheets, AUDITOR.keys, (1, 1)),
                                         G.Auditor(AUDITOR.sheets, (("cert0", "k1"), ("chal0", "k1"))),
                                         G.Auditor(AUDITOR.sheets, (("cert0", "k\u00e9"), ("chal0", "k1")))])
    composed = composite if pick(draw, range(16)) else pick(draw, G.COMPOSITES)
    c = case(composite, log, auditor, composed)
    broken = pick(draw, ["none"] * 12 + ["commitment", "params", "evidence"])
    if broken == "commitment":
        j = P.canonical_json(bytes.fromhex(c["commitment"]))
        c["commitment"] = D.canonical(_break(draw, j)).encode().hex()
    elif broken == "params":
        c["params"] = _break(draw, c["params"])
    elif broken == "evidence":
        c["evidence"] = pick(draw, [b"", b"[]", b'{"a":1}', b"{ }"]).hex()
    return c


PORT = RegistrationPort(exe="core-registration", package=INTERFACE, reference=reference, inputs=cases())


def test_the_example_and_its_refusals_agree_with_the_lean_guards() -> None:
    """The Lean's `#guard`s on `Core.Registration.Example`, in Python, and each replayed against the Lean."""
    bare = [G.Hardware((CARD,), LOCS[:4], (G.Link("up0", ("node0", G.OUTSIDE), 100, 1),))]
    expect = [(case("onsite", G.chain("onsite", EXAMPLE)), None),
              (case("onsite", G.chain("remote", EXAMPLE)), "entry 0: previous"),
              (case("onsite", G.chain("onsite", bare)), "entry 0: uncertified"),
              (case("remote", G.chain("remote", bare)), None),
              (case("onsite", G.chain("onsite", [G.Hardware((CARD,), LOCS[:5], LINKS[:2])])), "entry 0: challenger-reach"),
              (case("onsite", G.chain("onsite", [*EXAMPLE, G.Information(ITEMS[:1])])), "entry 2: duplicate-information"),
              (case("onsite", G.chain("onsite", [*EXAMPLE, G.Information(ITEMS[2:])])), "entry 2: duplicate-information"),
              (case("onsite", G.chain("onsite", [*EXAMPLE, G.Information(({**WEIGHTS, "value": "tables"},))])), None),
              (case("onsite", G.chain("remote", EXAMPLE), composed="remote"), "composite: the log is remote's, not onsite's"),
              *((case("remote", G.chain("remote", [G.Hardware((k,))])), None) for k in WITHIN),
              *((case("remote", G.chain("remote", [G.Hardware((k,))])), "entry 0: figures") for k in PAST),
              (case("remote", G.chain("remote", [G.Hardware((dataclasses.replace(CARD, mem=1),))])), "entry 0: figures"),
              (case("remote", G.chain("remote", [G.Hardware((dataclasses.replace(CARD, sku="sim-b"),))])),
               "entry 0: card-digest"),
              (case("remote", G.chain("remote", [G.Hardware((WITHIN[0],))]),
                    dataclasses.replace(AUDITOR, tolerance=(0, 7))), "entry 0: figures"),
              (case("remote", G.chain("remote", [G.Hardware((dataclasses.replace(CARD, clock=0),))]),
                    dataclasses.replace(AUDITOR, tolerance=(1, 1))), None),
              (case("remote", G.chain("remote", [G.Hardware((ZERO,))]), dataclasses.replace(AUDITOR, tolerance=(1, 1))),
               "entry 0: figures"),
              (case("onsite", G.chain("onsite", [G.Information(({**WEIGHTS, "value": "poids\u00e9"},))])),
               "commitment: ascii: the log holds a character outside ASCII")]
    with D.Server(PORT.argv()) as server:
        for c, reason in expect:
            out = reference(c)
            assert out["reason"] == reason, (out["reason"], reason)
            assert out["verdict"] == ("accept" if reason is None else "reject")
            D.replay(PORT, c, server)
    view = G.check(AUDITOR, "onsite", G.chain("onsite", EXAMPLE)).view()
    assert view.value("weights") == WEIGHTS and view.theta == ITEMS[:2] and view.builder == ITEMS[2]
    assert view.site.locs == LOCS


def test_registration_difftest() -> None:
    stats = D.run(PORT, examples=1000, seed=0)
    assert stats.counts.get("accept", 0) >= 50 and stats.counts.get("reject", 0) >= 200, stats.counts


def test_a_reference_without_the_challengers_reach_is_caught() -> None:
    """A Python that skips onsite's `challenger-reach` disagrees with the Lean on some drawn log."""
    real = G.hardware_refusal

    def lax(auditor, composite, site, h):
        r = real(auditor, composite, site, h)
        return None if r == "challenger-reach" else r

    G.hardware_refusal = lax
    try:
        with pytest.raises(D.Mismatch):
            D.run(PORT, examples=1000, seed=0)
    finally:
        G.hardware_refusal = real
