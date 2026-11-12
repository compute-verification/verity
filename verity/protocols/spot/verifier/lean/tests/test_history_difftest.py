"""History tables (`Spot.HistoryTable`) against `verity.commitments.history`: a table's domain, placeholder and root
at a capacity (`history_root`), the registered root held to it (`history_check`), and each position's source from the
program's runs (`history_order`), on valid and broken inputs at capacities up to 64, and once at the pinned capacity,
`2^17`. The named cases hold what the guarantee reads off the check: a rank past the table's length opens as the public
placeholder, and the root binds the rows' order and the capacity. The strings come from the writer's members' public
bodies as the circuit prover lays them out (`history_strings`), and each table's root from bodies the leaf layer binds
(`history_roots`), each held to its Python mirror (`history.table_strings`, `history.roots_of`). An interior's tables
are held to the boundary registration's ports (`tables_served`, `service.tables_served`)."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tomllib
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.protocols.spot.verifier import service as S
from verity.commitments import history as H

PACKAGE = Path(__file__).resolve().parents[1]
PIECES = next(e["name"] for e in tomllib.loads((PACKAGE / "lakefile.toml").read_text())["lean_exe"]
              if e["root"] == "Spot.DifftestMain")
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

#: `Flock.unhex`'s domain: pairs of lowercase hex digits
HEX = re.compile(r"(?:[0-9a-f]{2})*")
NAMES = st.sampled_from(["", "k_hist", "layer.3/keys", "é"])
CAPS = st.sampled_from([2, 4, 8, 16, 32, 64])


def _bc(seed: int, i: int) -> bytes:
    return b"".join(hashlib.sha512(f"{seed}/{i}/{h}".encode()).digest() for h in range(2))


def _nat(x, what: str) -> int:
    if not (type(x) is int and x >= 0):
        raise D.Reject(f"{what} is not a natural number")
    return x


def _str(x, what: str) -> str:
    if type(x) is not str:
        raise D.Reject(f"{what} is not a string")
    return x


def _hexes(xs, what: str) -> list[bytes]:
    if not isinstance(xs, list):
        raise D.Reject(f"{what} is not a list")
    if not all(type(x) is str and HEX.fullmatch(x) for x in xs):
        raise D.Reject(f"{what}: not lowercase hex")
    return [bytes.fromhex(x) for x in xs]


def _table(j) -> tuple[int, str, str, int, list[bytes]]:
    if not isinstance(j, dict):
        raise D.Reject("not an object")
    return (_nat(j.get("capacity"), "capacity"), _str(j.get("program"), "program"), _str(j.get("table"), "table"),
            _nat(j.get("words"), "words"), _hexes(j.get("bcs"), "bcs"))


def reference_root(j):
    cap, program, table, words, bcs = _table(j)
    if not H.cap_ok(cap) or not H.words_ok(words) or cap < len(bcs) or any(len(b) != H.BC_BYTES for b in bcs):
        raise D.Reject("checkAt's refusals")
    dom, ph = H.domain(program, table, words, cap), H.placeholder(words)
    return {"domain": dom.domain_id.hex(), "placeholder": ph.hex(), "root": H.root(dom, bcs, ph).hex()}


def reference_check(j):
    if not isinstance(j, dict) or type(j.get("registered")) is not str:
        raise D.Reject("registered is not a hex string")
    (registered,) = _hexes([j["registered"]], "registered")
    cap, program, table, words, bcs = _table(j)
    try:
        H.check(program, table, words, bcs, registered, cap)
    except ValueError as e:
        raise D.Reject(str(e)) from None
    return {"ok": True}


def reference_order(j):
    if not isinstance(j, dict) or not isinstance(j.get("runs"), list):
        raise D.Reject("runs is not a list")
    held = j.get("held")
    if not isinstance(held, dict) or not all(type(n) is int and n >= 0 for n in held.values()):
        raise D.Reject("held is not an object of row counts")
    length, writer, out, units = (_nat(j.get("length"), "length"), _str(j.get("writer"), "writer"),
                                  _str(j.get("out"), "out"), _nat(j.get("units"), "units"))
    try:
        srcs = H.order(j["runs"], length, writer, out, units, held)
    except ValueError as e:
        raise D.Reject(str(e)) from None
    return {"sources": [{"unit": s[1]} if s[0] == "unit" else {"held": s[1], "row": s[2]} for s in srcs]}


def _hex_of(bcs: list[bytes]) -> list[str]:
    return [b.hex() for b in bcs]


@st.composite
def tables(draw, broken: bool = False):
    cap = draw(CAPS)
    seed, n = draw(st.integers(0, 2 ** 32)), draw(st.integers(0, cap))
    j = {"capacity": cap, "program": draw(NAMES), "table": draw(NAMES), "words": draw(st.integers(1, 1200)),
         "bcs": _hex_of([_bc(seed, i) for i in range(n)])}
    if not broken:
        return j
    how = draw(st.integers(0, 10))
    if how == 0:
        j["capacity"] = draw(st.sampled_from([0, 1, 3, 6, 12, 48, 100, -4, True, "8"]))
    elif how == 1:
        j["words"] = draw(st.sampled_from([0, -1, 1 << 60, "2", None]))
    elif how == 2:
        j["bcs"] = _hex_of([_bc(seed, i) for i in range(cap + draw(st.integers(1, 3)))])
    elif how == 3:
        j["bcs"] = [*j["bcs"], _bc(seed, n)[:draw(st.sampled_from([0, 64, 127]))].hex()]
    elif how == 4:
        j["bcs"] = [*j["bcs"], (_bc(seed, n) + b"\0").hex()]
    elif how == 5:
        j["bcs"] = [*j["bcs"], _bc(seed, n).hex().upper()]
    elif how == 6:
        j["bcs"] = [*j["bcs"], _bc(seed, n).hex()[1:]]
    elif how == 7:
        j["bcs"] = [*j["bcs"], 7]
    elif how == 8:
        j["bcs"] = {"0": _bc(seed, 0).hex()}
    elif how == 9:
        j[draw(st.sampled_from(["program", "table"]))] = draw(st.sampled_from([None, 3, ["x"]]))
    else:
        del j[draw(st.sampled_from(sorted(j)))]
    return j


@st.composite
def checks(draw, broken: bool = False):
    j = draw(tables(broken=broken and draw(st.booleans())))
    try:
        root = reference_root(j)["root"]
    except D.Reject:
        root = "00" * H.DIGEST_BYTES
    how = draw(st.integers(0, 6)) if broken else 0
    if how == 1:
        i = draw(st.integers(0, len(root) - 1))
        root = root[:i] + ("1" if root[i] == "0" else "0") + root[i + 1:]
    elif how == 2:
        root = root[:-2]
    elif how == 3:
        root = root.upper()
    elif how == 4:
        j = dict(j, capacity=2 * j["capacity"]) if type(j.get("capacity")) is int else j
    elif how == 5 and isinstance(j.get("bcs"), list) and j["bcs"]:
        j = dict(j, bcs=j["bcs"][:-1])
    elif how == 6:
        root = draw(st.sampled_from([None, 5, ["00"]]))
    return {**j, "registered": root}


def _span(radix, step, units, draw) -> int:
    low = sum(min(0, s * (r - 1)) for r, s in zip(radix, step))
    high = sum(max(0, s * (r - 1)) for r, s in zip(radix, step))
    if units - 1 - high < -low:
        return draw(st.integers(-2, units + 2))
    return draw(st.integers(-low, units - 1 - high))


@st.composite
def orders(draw, broken: bool = False):
    units = draw(st.integers(1, 40))
    held = {v: draw(st.integers(0, 30)) for v in draw(st.lists(st.sampled_from(["E", "F"]), unique=True, max_size=2))}
    runs, at = [], 0
    for _ in range(draw(st.integers(0, 5))):
        if any(held.values()) and draw(st.booleans()):
            v = draw(st.sampled_from(sorted(k for k, n in held.items() if n)))
            row = draw(st.integers(0, held[v] - 1))
            count = draw(st.integers(1, held[v] - row))
            runs.append({"start": at, "count": count, "held": v, "row": row})
        else:
            k = draw(st.integers(0, 3))
            radix = [draw(st.integers(1, 4)) for _ in range(k)]
            step = [draw(st.integers(-3, 3)) for _ in range(k)]
            count = math.prod(radix)
            runs.append({"start": at, "count": count, "writer": "W", "out": "o", "radix": radix,
                         "base": _span(radix, step, units, draw), "step": step})
        at += count
    runs = draw(st.permutations(runs))
    j = {"runs": list(runs), "length": at, "writer": "W", "out": "o", "units": units, "held": held}
    if not broken:
        return j
    how = draw(st.integers(0, 14))
    unit_runs = [r for r in j["runs"] if "writer" in r]
    held_runs = [r for r in j["runs"] if "held" in r]
    if how == 0:
        j["length"] += draw(st.sampled_from([-1, 1]))
    elif how == 1 and j["runs"]:
        r = draw(st.sampled_from(j["runs"]))
        r["start"] += draw(st.sampled_from([-1, 1]))
    elif how == 2:
        j["runs"].append({"start": at, "count": 0, "held": "E", "row": 0})
    elif how == 3 and unit_runs:
        draw(st.sampled_from(unit_runs))[draw(st.sampled_from(["writer", "out"]))] = "X"
    elif how == 4 and unit_runs:
        draw(st.sampled_from(unit_runs))["base"] += draw(st.sampled_from([-units, units]))
    elif how == 5 and unit_runs:
        r = draw(st.sampled_from(unit_runs))
        r["radix"] = [*r["radix"], 0]
        r["step"] = [*r["step"], 1]
    elif how == 6 and unit_runs:
        r = draw(st.sampled_from(unit_runs))
        r["step"] = [*r["step"], 1]
    elif how == 7 and held_runs:
        r = draw(st.sampled_from(held_runs))
        r["row"] = held[r["held"]]
    elif how == 8 and held_runs:
        draw(st.sampled_from(held_runs))["held"] = "G"
    elif how == 9 and j["runs"]:
        draw(st.sampled_from(j["runs"]))["extra"] = 1
    elif how == 10 and j["runs"]:
        draw(st.sampled_from(j["runs"]))["count"] = draw(st.sampled_from([True, -1, "1"]))
    elif how == 11:
        j["held"] = draw(st.sampled_from([[], {"E": -1}, {"E": "3"}]))
    elif how == 12:
        j["runs"] = draw(st.sampled_from([{}, None, [3]]))
    elif how == 13 and unit_runs:
        r = draw(st.sampled_from(unit_runs))
        r["radix"] = [-1, *r["radix"]]
        r["step"] = [0, *r["step"]]
    else:
        j["units"] = draw(st.sampled_from([-1, None, 0]))
    return j


ROOT = D.Port(PIECES, "history_root", reference_root, st.one_of(*[tables() for _ in range(4)], tables(broken=True)),
              package=PACKAGE, of_record="lean")
CHECK = D.Port(PIECES, "history_check", reference_check,
               st.one_of(*[checks() for _ in range(2)], checks(broken=True)), package=PACKAGE, of_record="lean")
ORDER = D.Port(PIECES, "history_order", reference_order, st.one_of(*[orders() for _ in range(3)], orders(broken=True)),
               package=PACKAGE, of_record="lean")


def reference_served(j):
    """`Run.tablesServed` (`service.tables_served`): whether each ``[name, root, domain]`` of ``tables`` is a port of
    ``record`` at that root and domain."""
    ts = j.get("tables") if isinstance(j, dict) else None
    if not isinstance(ts, list):
        raise D.Reject("tables is not a list")
    tables = []
    for t in ts:
        if not (isinstance(t, list) and len(t) == 3 and all(type(x) is str for x in t) and HEX.fullmatch(t[1])
                and HEX.fullmatch(t[2])):
            raise D.Reject("a table is [name, root, domain], its root and domain hex strings")
        tables.append((t[0], bytes.fromhex(t[1]), bytes.fromhex(t[2])))
    return {"served": S.tables_served(j.get("record"), tables)}


@st.composite
def served(draw, broken: bool = False):
    """A boundary registration's record, its ``roots`` its own port and one to three tables', and tables an interior
    reads: some of the record's at their roots and domains, or one with its root, domain or name changed, one the record
    has no port of, the same name twice, a port the record lists again under another root before or after its own, or
    the record's ``roots`` in another form; broken, a table that isn't ``[name, root, domain]`` in lowercase hex."""
    digest = st.binary(min_size=64, max_size=64).map(bytes.hex)
    names = draw(st.lists(NAMES, min_size=1, max_size=3, unique=True))
    entries = [{"port": n, "root": draw(digest), "leaves": 2 ** 17, "domain": draw(digest)} for n in names]
    roots = [{"port": "b.ingress", "root": draw(digest), "leaves": 4, "domain": draw(digest)}, *entries]
    tables = [[e["port"], e["root"], e["domain"]] for e in draw(st.lists(st.sampled_from(entries), max_size=3))]
    how = draw(st.integers(0, 10))
    if how == 1 and tables:
        tables[0][1] = draw(digest)
    elif how == 2 and tables:
        tables[-1][2] = draw(digest)
    elif how == 3 and tables:
        tables[0][0] += "/x"
    elif how == 4:
        tables.append(["layer.9/keys", draw(digest), draw(digest)])
    elif how == 5 and tables:
        tables.append([tables[0][0], draw(digest), tables[0][2]])
    elif how == 6:
        roots = draw(st.sampled_from([None, {}, "roots", [1, None], [{"port": n} for n in names]]))
    elif how == 7 and tables:
        roots[1:] = [{**e, "root": e["root"].upper()} for e in entries]
    elif how == 9:
        again = {**draw(st.sampled_from(entries)), "root": draw(digest)}
        roots = [roots[0], again, *roots[1:]] if draw(st.booleans()) else [*roots, again]
    record = draw(st.sampled_from([{"roots": roots}, {"roots": roots, "population": 3}])) if how != 8 else roots
    if broken:
        bad = draw(st.sampled_from([[1, "aa", "bb"], ["n", "AA", "bb"], ["n", "a", "bb"], ["n", "aa"], "n", None]))
        tables = [*tables, bad] if draw(st.booleans()) else [bad, *tables]
        return {"record": record, "tables": tables} if draw(st.integers(0, 3)) else {"record": record, "tables": bad}
    return {"record": record, "tables": tables}


SERVED = D.Port(PIECES, "tables_served", reference_served, st.one_of(*[served() for _ in range(3)], served(broken=True)),
                package=PACKAGE, of_record="lean")

pieces = D.server_fixture(ROOT)


@pytest.mark.parametrize("port, examples", [(ROOT, 300), (CHECK, 300), (ORDER, 1500), (SERVED, 300)],
                         ids=["history_root", "history_check", "history_order", "tables_served"])
def test_a_history_piece_agrees(pieces, port, examples):
    stats = D.run(port, examples=examples, server=pieces)
    assert stats.outputs > examples // 2 and stats.rejects > 0, stats


def test_a_table_is_served_only_at_its_ports_root_and_domain(pieces):
    """An interior's table is round 1's only where round 1's record has a port of its name at its root and domain
    (`Run.tablesServed`): another root, another domain, another name or no port is refused; a port the record lists twice
    is held at its last root, the one round 1's checks hold (`Record.byPort`); and an interior with no tables passes."""
    port = {"port": "layer.3/keys", "root": "aa" * 64, "leaves": 2 ** 17, "domain": "bb" * 64}
    record = {"roots": [{"port": "b.ingress", "root": "cc" * 64, "leaves": 4, "domain": "dd" * 64}, port]}
    for tables, want in [([["layer.3/keys", "aa" * 64, "bb" * 64]], True), ([], True),
                         ([["layer.3/keys", "ab" * 64, "bb" * 64]], False), ([["layer.3/keys", "aa" * 64, "ba" * 64]], False),
                         ([["layer.4/keys", "aa" * 64, "bb" * 64]], False), ([["b.ingress", "aa" * 64, "bb" * 64]], False),
                         ([["layer.3/keys", "aa" * 64, "bb" * 64], ["layer.3/keys", "ab" * 64, "bb" * 64]], False)]:
        j = {"record": record, "tables": tables}
        assert _call(pieces, "tables_served", j) == reference_served(j) == {"served": want}, tables
    again = {**port, "root": "ab" * 64}
    for roots, want in [([port, again], False), ([again, port], True)]:
        j = {"record": {"roots": roots}, "tables": [["layer.3/keys", "aa" * 64, "bb" * 64]]}
        assert _call(pieces, "tables_served", j) == reference_served(j) == {"served": want}, roots


def _call(pieces, fn: str, j):
    kind, out = pieces.call(fn, j)
    assert kind == "output", (fn, kind, out)
    return json.loads(out)


def _base(n: int = 5, cap: int = 16, words: int = 64) -> dict:
    return {"capacity": cap, "program": "p", "table": "k_hist", "words": words,
            "bcs": _hex_of([_bc(1, i) for i in range(n)])}


def test_a_rank_past_the_length_opens_as_the_placeholder(pieces):
    """The root over `n` rows is the root over those rows and the placeholder at every rank up to the capacity, so a read
    past the length opens only as the zero row under the zero salt, which anyone computes."""
    j = _base()
    got = _call(pieces, "history_root", j)
    assert got["placeholder"] == H.placeholder(j["words"]).hex()
    padded = dict(j, bcs=[*j["bcs"], *[got["placeholder"]] * (j["capacity"] - len(j["bcs"]))])
    assert _call(pieces, "history_root", padded) == got == reference_root(j)


def test_the_root_binds_the_rows_order(pieces):
    j = _base()
    root = _call(pieces, "history_root", j)["root"]
    assert _call(pieces, "history_check", dict(j, registered=root)) == {"ok": True}
    swapped = dict(j, bcs=[j["bcs"][1], j["bcs"][0], *j["bcs"][2:]], registered=root)
    assert pieces.call("history_check", swapped)[0] == "reject"
    with pytest.raises(D.Reject):
        reference_check(swapped)


def test_the_root_binds_the_capacity_and_the_names(pieces):
    j = _base()
    root = _call(pieces, "history_root", j)["root"]
    for change in ({"capacity": 32}, {"table": "v_hist"}, {"program": "q"}, {"words": 32}):
        k = dict(j, **change, registered=root)
        assert pieces.call("history_check", k)[0] == "reject", change
        with pytest.raises(D.Reject):
            reference_check(k)


def test_the_pinned_capacity(pieces):
    """One table at `CAPACITY`, the capacity every table's tree has, whatever its length."""
    j = _base(n=3, cap=H.CAPACITY)
    got = _call(pieces, "history_root", j)
    assert got == reference_root(j)
    assert _call(pieces, "history_check", dict(j, registered=got["root"])) == {"ok": True}


def _gather(**change) -> dict:
    """Six positions from writer `W`'s units `2 + 3 d_0 - d_1`, `d_0` the slow digit of radix 2 and `d_1` of radix 3."""
    run = {"start": 0, "count": 6, "writer": "W", "out": "o", "radix": [2, 3], "base": 2, "step": [3, -1]}
    return {"runs": [dict(run, **change)], "length": 6, "writer": "W", "out": "o", "units": 8, "held": {}}


@pytest.mark.parametrize("name, j, want", [
    ("a gather of two digits", _gather(), [2, 1, 0, 5, 4, 3]),
    ("an earlier window's rows, then the writer's", {
        "runs": [{"start": 2, "count": 2, "writer": "W", "out": "o", "radix": [2], "base": 5, "step": [1]},
                 {"start": 0, "count": 2, "held": "E", "row": 3}],
        "length": 4, "writer": "W", "out": "o", "units": 8, "held": {"E": 5}},
     [("E", 3), ("E", 4), 5, 6]),
], ids=lambda x: x if isinstance(x, str) else "")
def test_named_orders(pieces, name, j, want):
    want_srcs = [{"unit": w} if isinstance(w, int) else {"held": w[0], "row": w[1]} for w in want]
    assert _call(pieces, "history_order", j) == reference_order(j) == {"sources": want_srcs}


@pytest.mark.parametrize("name, j", [
    ("a writer unit below 0", _gather(base=1)),
    ("a writer unit past the template", _gather(base=5)),
    ("another writer", _gather(writer="X")),
    ("a radix that covers other positions", _gather(radix=[2, 2], step=[2, -1])),
    ("rows past the held value", {"runs": [{"start": 0, "count": 3, "held": "E", "row": 3}], "length": 3,
                                  "writer": "W", "out": "o", "units": 8, "held": {"E": 5}}),
], ids=lambda x: x if isinstance(x, str) else "")
def test_refused_orders(pieces, name, j):
    assert pieces.call("history_order", j)[0] == "reject"
    with pytest.raises(D.Reject):
        reference_order(j)


def _body(rows: list[list[bytes]]) -> str:
    """A member's public body of one row a unit a port, as `rows.Committed.public_body` writes it: unit-major."""
    return b"".join(b"".join(r) for r in rows).hex()


def _shared(tables: list[list[bytes]], refs: list[list[int]]) -> str:
    """A shared-row member's public body: each port's table rows, port-major, then each unit's refs (u32 LE, unit-major)."""
    return (b"".join(b"".join(t) for t in tables) + b"".join(r.to_bytes(4, "little") for u in refs for r in u)).hex()


def _strings(**change) -> dict:
    """Writer `W`'s units 0-4 in two members, `x` then `o` a unit: positions 2-4 its units 4, 3, 2's `o`, after an earlier
    window's rows 1 and 2 of `E`."""
    a = [[_bc(2, 10 * u), _bc(3, u)] for u in range(3)]
    b = [[_bc(2, 10 * u), _bc(3, u)] for u in range(3, 5)]
    j = {"runs": [{"start": 2, "count": 3, "writer": "W", "out": "o", "radix": [3], "base": 4, "step": [-1]},
                  {"start": 0, "count": 2, "held": "E", "row": 1}],
         "length": 5, "writer": "W", "out": "o", "first": 7, "units": 5,
         "members": [{"base": 10, "n": 2, "ports": ["x", "o"], "body": _body(b)},
                     {"base": 7, "n": 3, "ports": ["x", "o"], "body": _body(a)}],
         "held": {"E": _hex_of([_bc(4, r) for r in range(3)])}}
    return {**j, **change}


def _py_strings(j: dict) -> dict:
    """The Python mirror's `history_strings`: `history.table_strings` over the piece's input."""
    held = {v: [bytes.fromhex(x) for x in ss] for v, ss in j["held"].items()}
    return {"bcs": _hex_of(H.table_strings(j["runs"], j["length"], j["writer"], j["out"], j["first"], j["units"],
                                           [H.member_of(m) for m in j["members"]], held))}


def test_strings_from_the_writers_public_bodies(pieces):
    """Each position's string is its writer unit's `o` string as its member's body holds it, the members in unit order,
    and a held run's the verifier's own of the earlier window."""
    want = [_bc(4, 1), _bc(4, 2), _bc(3, 4), _bc(3, 3), _bc(3, 2)]
    assert _call(pieces, "history_strings", _strings()) == {"bcs": _hex_of(want)} == _py_strings(_strings())


def test_strings_from_a_shared_row_body(pieces):
    """A shared-row body: unit `u`'s string is the row its ref names in its port's table."""
    xs, os_ = [_bc(5, 0)], [_bc(6, 0), _bc(6, 1)]
    refs = [[0, 1], [0, 0], [0, 1]]
    j = {"runs": [{"start": 0, "count": 3, "writer": "W", "out": "o", "radix": [3], "base": 0, "step": [1]}],
         "length": 3, "writer": "W", "out": "o", "first": 0, "units": 3, "held": {},
         "members": [{"base": 0, "n": 3, "ports": ["x", "o"], "tables": {"x": 1, "o": 2}, "body": _shared([xs, os_], refs)}]}
    assert _call(pieces, "history_strings", j) == {"bcs": _hex_of([os_[1], os_[0], os_[1]])} == _py_strings(j)
    past = dict(j, members=[dict(j["members"][0], body=_shared([xs, os_], [[0, 2], [0, 0], [0, 1]]))])
    assert pieces.call("history_strings", past)[0] == "reject"
    with pytest.raises(ValueError):
        _py_strings(past)


@pytest.mark.parametrize("name, j", [
    ("a unit no member's body holds", _strings(members=_strings()["members"][:1])),
    ("members that leave a gap", _strings(first=6)),
    ("a body that is not its units' strings", _strings(members=[dict(m, n=m["n"] + 1) for m in _strings()["members"]])),
    ("a member without the table's row", _strings(out="y", runs=[{"start": 0, "count": 5, "writer": "W", "out": "y",
                                                                 "radix": [5], "base": 0, "step": [1]}])),
    ("a shared-row member's null tables", _strings(members=[dict(m, tables=None) for m in _strings()["members"]])),
    ("a body in uppercase hex", _strings(members=[dict(m, body=m["body"].upper()) for m in _strings()["members"]])),
], ids=lambda x: x if isinstance(x, str) else "")
def test_refused_strings(pieces, name, j):
    assert pieces.call("history_strings", j)[0] == "reject"
    with pytest.raises(ValueError):
        _py_strings(j)


def _roots(unbound: str | None = None, **change) -> dict:
    """`_strings`' table `k_hist` at capacity 16, its writer's two members and a reader's body under the leaf layer
    `rows.leaf_layer` makes of all three (of all but `unbound`), the window registering the root over its rows."""
    from verity.commitments.windows import rows as CS
    s = _strings()
    members = [dict(m, key=f"W/{i}") for i, m in enumerate(s["members"])]
    publics = {m["key"]: bytes.fromhex(m["body"]) for m in members if m["key"] != unbound} | {"R/0": b"the reader's body"}
    dom, ph = H.domain("p", "k_hist", 64, 16), H.placeholder(64)
    bcs = [_bc(4, 1), _bc(4, 2), _bc(3, 4), _bc(3, 3), _bc(3, 2)]
    j = {"leaf_layer": CS.leaf_layer(publics),
         "digests": {k: hashlib.sha512(b).hexdigest() for k, b in publics.items()},
         "tables": [{"name": "k_hist", "program": "p", "writer": "W", "out": "o", "words": 64, "runs": s["runs"],
                     "capacity": 16, "first": s["first"], "units": s["units"], "members": members}],
         "held": {"k_hist": s["held"]}, "registered": {"k_hist": H.root(dom, bcs, ph).hex()}}
    return {**j, **change}


def test_roots_from_the_bodies_the_leaf_layer_binds(pieces):
    """Each table's root is recomputed from its writer's bodies, which the leaf layer over every member's body binds, and
    the root and domain its reads open against are the registered root, held to it, and the table's domain."""
    from verity.commitments.windows import rows as CS
    j = _roots()
    dom = H.domain("p", "k_hist", 64, 16).domain_id.hex()
    want = {"roots": {"k_hist": {"domain": dom, "root": j["registered"]["k_hist"]}}}
    assert _call(pieces, "history_roots", j) == want == _py_roots(j)
    publics = {m["key"]: bytes.fromhex(m["body"]) for m in j["tables"][0]["members"]} | {"R/0": b"the reader's body"}
    assert H.leaf_layer(j["digests"]) == CS.leaf_layer(publics) == j["leaf_layer"]


def _py_roots(j: dict) -> dict:
    """The Python mirror's `history_roots`: `history.roots_of` over the piece's input."""
    return {"roots": {n: {"domain": d.hex(), "root": r.hex()} for n, (r, d) in H.roots_of(j["leaf_layer"], j).items()}}


def _tampered_body() -> dict:
    j = _roots()
    m = j["tables"][0]["members"][0]
    body = bytearray.fromhex(m["body"])
    body[-1] ^= 1
    j["tables"][0]["members"][0] = dict(m, body=body.hex())
    return j


@pytest.mark.parametrize("name, j", [
    ("a body that is not the one the leaf layer binds", _tampered_body()),
    ("a writer member the leaf layer doesn't name", _roots(unbound="W/1")),
    ("another leaf layer", _roots(leaf_layer="00" * 64)),
    ("a registered root over other rows", _roots(registered={"k_hist": _roots()["registered"]["k_hist"][::-1]})),
    ("no registered root", _roots(registered={})),
    ("a table named twice", _roots(tables=_roots()["tables"] * 2)),
], ids=lambda x: x if isinstance(x, str) else "")
def test_refused_roots(pieces, name, j):
    assert pieces.call("history_roots", j)[0] == "reject"
    with pytest.raises(ValueError):
        _py_roots(j)


def _window(table: H.HistoryTable, seed: int = 0, n: int = 4) -> tuple:
    """``n`` units of ``Gemm_v2{K=16, N=8}``, two a member, their outputs appended to ``table`` as history table
    ``latent``'s next rows after those it holds of earlier windows, committed as `ml.windows` commits a window: -> the
    circuit record, the verifier's window and the members' public bodies."""
    import numpy as np

    from verity.circuits.codec import descriptor_id
    from verity.circuits.defs import bind
    from verity.circuits.evaluation import evaluate_batch
    from verity.definitions.ops import gemm as G
    from verity.commitments.windows import interface as I

    g8 = bind(G.Gemm, K=16, N=8, **G.GEMM_AMPERE)
    gid, at, rng = descriptor_id(g8), len(table.rows), np.random.default_rng(seed)

    def bf16(a):
        return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)

    x, w = bf16(rng.standard_normal((n, 16))), bf16(rng.standard_normal((n, 8, 16)) / 4)
    y = np.asarray(evaluate_batch(g8, x, w, kernel="reference"), dtype=np.uint16).reshape(n, 8)
    table.append(y)
    runs = ([{"start": 0, "count": at, "held": "latent", "row": 0}] if at else []) + [
        {"start": at, "count": n, "writer": gid, "out": "o0", "radix": [n], "base": 0, "step": [1]}]
    history = I.History("latent", "program", gid, "o0", 8, tuple(runs), table.leaves)
    win = I.Window(I.Circuit("history", ((g8, n),), registered=I.Registered({}, histories=(history,))), "ab" * 64,
                   {"law": "stratified", "k": 2}, 2, registered={"latent": table})
    win.record(g8, [x, w], [y])
    c = win.close(salt_key=bytes(32))
    doc = json.loads(I.circuit_bytes(win.circuit_record()))
    return doc, win.window, {m.key: c.public_body(k) for k, m in enumerate(win.window.members)}


def _window_roots(window: tuple, root: str, held: dict | None = None) -> dict:
    """The piece's input for a window's history table as the served audit hands it to the verifier of record
    (`ml.windows.history.params`), under the window's leaf layer, the window registering ``root``."""
    from verity.commitments.windows import history as HW, rows as CS
    doc, mine, publics = window
    return {"leaf_layer": CS.leaf_layer(publics), **HW.params(doc, mine, publics, HW.pins(doc, {"latent": root}), held)}


def test_roots_of_a_windows_history_table(pieces):
    """A served window's history table (`ml.windows.history.params`): the first window's root from its writer's bodies,
    and a later window's from its own and the rows it holds of the first, are the table's, in the Lean and its mirror
    alike."""
    table = H.HistoryTable("program", "latent", 8, 16)
    j = _window_roots(_window(table), table.root.hex())
    want = {"roots": {"latent": {"domain": table.domain.domain_id.hex(), "root": table.root.hex()}}}
    assert _call(pieces, "history_roots", j) == want == _py_roots(j)
    first = list(table.bcs)
    j = _window_roots(_window(table, seed=1), table.root.hex(), {"latent": first})
    assert j["held"] == {"latent": {"latent": _hex_of(first)}}
    want = {"roots": {"latent": {"domain": table.domain.domain_id.hex(), "root": table.root.hex()}}}
    assert _call(pieces, "history_roots", j) == want == _py_roots(j)


def test_a_wrong_root_of_a_windows_history_table_is_refused(pieces):
    """The root a served window registers for its history table is refused, by the Lean and its mirror alike, where it
    isn't the one over the table's strings: the root over the same rows in another order, the root with a bit flipped,
    a later window registering the earlier one's root, and a later window's root over held rows in another order."""
    import numpy as np
    table = H.HistoryTable("program", "latent", 8, 16)
    one = _window(table)
    first, root = list(table.bcs), table.root.hex()
    other = H.HistoryTable("program", "latent", 8, 16)
    other.append(np.stack(table.rows[::-1]), table.salts[::-1])
    two = _window(table, seed=1)
    for j in (_window_roots(one, other.root.hex()), _window_roots(one, root[:-1] + format(int(root[-1], 16) ^ 1, "x")),
              _window_roots(two, root, {"latent": first}), _window_roots(two, table.root.hex(), {"latent": first[::-1]})):
        assert pieces.call("history_roots", j)[0] == "reject"
        with pytest.raises(ValueError, match="not the table's root"):
            _py_roots(j)
