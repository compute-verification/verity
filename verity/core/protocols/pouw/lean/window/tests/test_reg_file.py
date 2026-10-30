"""`PouwWindow.RegFile`, a drawn H1 leaf's ``--registered`` files from the accepted registration and produced record, held
to the Python on the shared harness (`verity_difftest`): `leaf_lanes.registered_file` at `leaf_lanes.reads` and at
`leaf_lanes.verifier_entries`' entries and roots and at the unit's placement (each value its own table from row 0, or
placed in a window table from a base row), the files h1_lanes' statements read (`benchmarks/pouw/h1_lanes/run.py`),
and `pouw-leaf-own`'s check (`leafOwn?`), which accepts each such file and refuses one whose data port is not the
layout's read at the records' root and the unit's placement or one of whose ports has two reads, and its sample policy
(`checkFull`), which accepts a session record that samples every unit and refuses one that doesn't.

The drawn inputs are registrations naming the leaf's four registered tables (one sometimes missing, one sometimes named
twice, the partials sometimes among them), produced records with and without the partials, leaf tiles of 2 to 64 lanes,
every lane, the combine and a lane past the last, known ports that do and don't collide with a read, and placements
that do and don't move each value.
"""
from __future__ import annotations

import copy
import json
import random
import shutil
from pathlib import Path

import pytest
import verity_difftest as D
from hypothesis import strategies as st

from verity.core.protocols.pouw import leaf_lanes as LL
from verity.core.protocols.sampled_proofs.one_stage import registered as G

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

LEAN = Path(__file__).resolve().parents[1]
EXE = "pouw-window-difftest"
TILES = ((32, 32), (8, 16), (16, 8), (64, 32), (4, 16))
VALUES = (LL.A, LL.B, LL.ZY, LL.RECEIPT)


def _hex(rng: random.Random) -> str:
    return rng.randbytes(64).hex()


def _entry(rng: random.Random, value: str) -> dict:
    e = {"value": value, "root": _hex(rng), "leaves": rng.randrange(1, 65), "words": rng.randrange(1, 300),
         "domain": _hex(rng)}
    if rng.random() < 0.2:
        e["word_bits"] = 32
    return e


def _known(rng: random.Random, name: str) -> dict:
    return {"value": name, "root": _hex(rng), "leaves": 1, "words": 1, "word_bits": 16, "domain": _hex(rng),
            G.SCHEMA_KEY: G.SCHEMA_V2, "reads": [0]}


def placement(rng: random.Random) -> dict[str, tuple[str, int]] | None:
    """A unit's placement of the leaf's registered values, each in a window table from a base row (a call's A table of
    its step, the weights' B, its z and receipt tables), or None: each value its own table from row 0."""
    if rng.random() < 0.4:
        return None
    return {v: (f"h1/w-{rng.randrange(3)}/{v}@{rng.randrange(4)}", rng.randrange(1 << 12)) for v in VALUES}


def records(rng: random.Random, lanes: int, *, whole: bool = False,
            place: dict[str, tuple[str, int]] | None = None) -> tuple[dict, dict]:
    """A registration record of the leaf's tables (each value's, at ``place``) and a produced record of its partials
    (``whole``: each table once)."""
    tables = [(place or {}).get(v, (v, 0))[0] for v in VALUES]
    reg = [_entry(rng, v) for v in tables if whole or rng.random() > 0.05]
    if not whole and rng.random() < 0.2:
        reg.append(_entry(rng, rng.choice((*tables, *VALUES, LL.PARTIALS))))
    rng.shuffle(reg)
    prod = [] if not whole and rng.random() < 0.05 else [
        {"value": LL.PARTIALS, "root": _hex(rng), "leaves": lanes, "words": 8, "domain": _hex(rng)}]
    if rng.random() < 0.2:
        prod.insert(0, {"value": "other", "root": _hex(rng), "leaves": 1, "words": 1, "domain": _hex(rng)})
    return ({"format": "verity/registration/v2", "program": _hex(rng), "registered": reg},
            {"format": "verity/registration/produced/v1", "produced": prod})


def file_in(rng: random.Random) -> dict:
    r, c = rng.choice(TILES)
    n = LL.lanes(r, c)
    place = placement(rng)
    registration, produced = records(rng, n, place=place)
    names = rng.sample((LL.PUB, LL.ADMIT, LL.OK), rng.randrange(4))
    if rng.random() < 0.1:
        names.append(rng.choice(("a0", "p1", LL.PARTIALS)))
    return {"registration": registration, "produced": produced, "r": r, "c": c, "k": rng.choice((64, 448, 4096)),
            "lane": rng.choice((None, None, None, 0, n - 1, rng.randrange(n), rng.randrange(n), n)),
            "known": {nm: _known(rng, nm) for nm in names}, "complete": rng.random() < 0.8,
            "place": None if place is None else {v: list(tb) for v, tb in place.items()}}


def registered_file(x: dict) -> dict:
    """The statement's file as h1_lanes writes it: `registered_file` at `verifier_entries`, every value's rows ``row/v2``."""
    ln = x["lane"]
    kind = LL.COMBINE if ln is None else LL.LANE
    try:
        reads = LL.reads(x["r"], x["c"], kind, ln)
        ent, roots = LL.verifier_entries(x["registration"], x["produced"], LL.lanes(x["r"], x["c"]), 2 * LL.words(x["k"]))
        place = {v: (t, b) for v, (t, b) in (x.get("place") or {}).items()}
        return LL.registered_file(reads, ent, roots, {v: G.SCHEMA_V2 for v in ent}, x["known"], x["complete"], place)
    except (ValueError, KeyError) as e:
        raise D.Reject(str(e)) from e


FILE = D.Port(EXE, "registered_file", reference=registered_file, inputs=st.randoms(use_true_random=False).map(file_in),
              package=LEAN, of_record="lean")
#: `pouw-leaf-own`'s check alone: a file's verdict at the records (no Python counterpart)
OWN = D.Port(EXE, "leaf_own", reference=lambda x: None, inputs=st.nothing(), package=LEAN)

server = D.server_fixture(FILE)


def test_the_python_is_the_lean(server) -> None:
    D.run(FILE, examples=300, server=server)


#: a window unit's placement: its call's A table of its step, the weights' B, its call's z and receipt tables, each from
#: the unit's base row
WINDOW = {LL.A: ("h1/w-0/a/448@1", 64), LL.B: ("h1/w-0/b/448", 96), LL.ZY: ("h1/w-0/zy@3", 5),
          LL.RECEIPT: ("h1/w-0/receipt@3", 5)}


def _served(seed: int, ln: int | None, place: dict[str, tuple[str, int]] | None = None) -> tuple[dict, dict]:
    """A served statement's input (each table registered once, the known ports h1_lanes registers, each value at
    ``place``) and its Python file."""
    rng = random.Random(seed)
    registration, produced = records(rng, LL.lanes(32, 32), whole=True, place=place)
    known = (LL.PUB, LL.ADMIT, LL.OK) if ln is not None else (LL.OK,)
    x = {"registration": registration, "produced": produced, "r": 32, "c": 32, "k": 448, "lane": ln,
         "known": {nm: _known(rng, nm) for nm in known}, "complete": True,
         "place": None if place is None else {v: list(tb) for v, tb in place.items()}}
    return x, registered_file(x)


def _own(server, x: dict, file: dict) -> tuple[str, str]:
    return OWN.lean(server, {k: x[k] for k in ("registration", "produced", "r", "c", "k", "lane", "place")}
                    | {"file": file})


@pytest.mark.parametrize("place", [None, WINDOW], ids=["own-tables", "window"])
@pytest.mark.parametrize("ln", [0, 5, 31, None])
def test_the_served_file_is_the_lean_and_passes_its_check(ln: int | None, place: dict | None, server) -> None:
    """At a served statement the Python's file is the Lean's, and `pouw-leaf-own` accepts it: each value its own table
    from row 0, and each placed in a window table from the unit's base row, where every data port reads its table at
    the base plus its leaf row and the ports of A, B and partials rows stay marked read in full."""
    x, file = _served(ln or 0, ln, place)
    D.replay(FILE, x, server)
    assert _own(server, x, file) == ("output", "true")
    for port, (value, row) in LL.reads(32, 32, LL.COMBINE if ln is None else LL.LANE, ln).items():
        table, base = (place or {}).get(value, (value, 0))
        e = file["ports"][port]
        assert (e["value"], e["reads"], e.get(LL.READ_IN_FULL, False)) == (table, [base + row], value in LL.FULL)


@pytest.mark.parametrize("ln", [3, None])
def test_the_check_refuses_a_file_at_another_placement(ln: int | None, server) -> None:
    """A file at the unit's placement is refused by name when it is checked at another: with each value its own table
    from row 0 (tables the registration doesn't register), or at another unit's base rows."""
    x, file = _served(30 + (ln or 0), ln, WINDOW)
    kind, why = _own(server, x | {"place": None}, file)
    assert kind == "reject" and "whose entry and root the verifier does not hold" in why
    shifted = {v: [t, b + 1] for v, (t, b) in WINDOW.items()}
    kind, why = _own(server, x | {"place": shifted}, file)
    assert kind == "reject" and "is not the layout's read" in why


@pytest.mark.parametrize("ln", [3, None])
@pytest.mark.parametrize("field", ["root", "domain", "leaves", "reads"])
def test_the_check_refuses_a_data_port_off_the_records(ln: int | None, field: str, server) -> None:
    """A file one of whose data ports is not the layout's read at the records' root (another root, domain, leaf count or
    row) is refused by name; a file built from another registration is refused too."""
    x, file = _served(10 + (ln or 0), ln)
    port = next(iter(LL.reads(32, 32, LL.COMBINE if ln is None else LL.LANE, ln)))
    bad = copy.deepcopy(file)
    e = bad["ports"][port]
    e[field] = {"root": "00" * 64, "domain": "11" * 64, "leaves": e["leaves"] + 1, "reads": [e["reads"][0] + 1]}[field]
    kind, why = _own(server, x, bad)
    assert kind == "reject" and f"port {port} is not the layout's read" in why
    other, _ = _served(99, ln)
    kind, why = _own(server, other, file)
    assert kind == "reject" and "is not the layout's read" in why


@pytest.mark.parametrize("ln", [3, None])
@pytest.mark.parametrize("known", [True, False])
def test_the_check_refuses_a_port_of_two_reads(ln: int | None, known: bool, server) -> None:
    """A file one of whose ports, a known port or a data port, states two reads is refused by name: a statement is one
    unit, which reads each port once."""
    x, file = _served(20 + (ln or 0), ln)
    port = LL.OK if known else next(iter(LL.reads(32, 32, LL.COMBINE if ln is None else LL.LANE, ln)))
    bad = copy.deepcopy(file)
    bad["ports"][port]["reads"] = bad["ports"][port]["reads"] * 2
    kind, why = _own(server, x, bad)
    assert kind == "reject" and f"port {port} has 2 reads" in why


#: `pouw-leaf-own`'s draw policy alone: a session record's verdict (no Python counterpart)
FULL = D.Port(EXE, "full_draw", reference=lambda x: None, inputs=st.nothing(), package=LEAN)


def _record(draw: dict | None) -> str:
    return json.dumps({"format": "flock-session/v1"} | ({} if draw is None else {"unit_draw": draw}))


@pytest.mark.parametrize("n", [1, 3])
def test_the_full_draw_passes(n: int, server) -> None:
    """A session that samples every unit of its population, `subset` of all of them, passes the sample policy."""
    draw = {"law": "subset", "population": n, "k": n, "units": list(range(n))}
    assert FULL.lean(server, {"record": _record(draw)}) == ("output", "true")


@pytest.mark.parametrize(("draw", "why"), [
    (None, "has no draw"),
    ({"law": "subset", "population": 3, "k": 2, "units": [0, 2]}, "not every unit"),
    ({"law": "subset", "population": 0, "k": 0, "units": []}, "not every unit"),
    ({"law": "bernoulli", "population": 2, "p": "1/1", "units": [0, 1]}, "law is not subset"),
    ({"law": "subset", "population": 2, "k": 2, "units": [1, 0]}, "the session's draw: U2"),
])
def test_the_draw_policy_refuses_a_partial_draw(draw: dict | None, why: str, server) -> None:
    """A session record with no sample, a sample that leaves a unit out, an empty population, another law, or a sample
    flock-verify's format refuses, is refused by name."""
    kind, got = FULL.lean(server, {"record": _record(draw)})
    assert kind == "reject" and why in got
