"""The three conditions on a two-stage design's wires (`Spot.SecondWires`: `rowsUnder`, `namesOnce`, `wiresWide`, and
`decided`, the three together) held to their Python (`second_wires`): the ``second_wires`` piece of the difftest's
executable answers each decider, the three together and the failures' codes on an instance, its parts, wires, rows and
each template's circuit file's layout, and the reference below reads the same instance into `second_wires`' types.
Named cases accept and refuse each condition alone; generated ones are small designs, mostly well formed, some with one
field replaced by arbitrary JSON, and must reach both answers of every decider.
"""

from __future__ import annotations

import shutil

import pytest
from hypothesis import strategies as st

from test_wires_difftest import PACKAGE, PIECES

import verity_difftest as D
from verity.protocols.spot.verifier import second_stage, second_wires

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

LAYOUT_KEYS = ["const_pos", "in_words", "leaves_out", "out_col", "out_ports", "out_row_ports", "ports", "upv"]
WIRE_KEYS = ["port", "template", "wport", "writer"]


def _nat(x) -> bool:
    return type(x) is int and x >= 0


def _nats(x) -> bool:
    return isinstance(x, list) and all(map(_nat, x))


def _named(x) -> bool:
    """A list of ``[name, natural]``."""
    return isinstance(x, list) and all(isinstance(e, list) and len(e) == 2 and isinstance(e[0], str) and _nat(e[1])
                                       for e in x)


def _layout(j) -> second_wires.Head:
    """`SecondWires.circuitOf?`: a circuit file's layout from its JSON, exactly its eight fields."""
    if not (isinstance(j, dict) and sorted(j) == LAYOUT_KEYS):
        raise D.Reject("a layout is its eight fields")
    if not (_named(j["ports"]) and _named(j["out_ports"]) and _nats(j["out_row_ports"])
            and isinstance(j["leaves_out"], list) and all(map(_nats, j["leaves_out"]))
            and all(_nat(j[k]) for k in ("upv", "in_words", "const_pos", "out_col"))):
        raise D.Reject("a layout's fields are naturals and lists of them")
    return second_wires.Head(tuple((n, b) for n, b in j["ports"]), tuple((n, w) for n, w in j["out_ports"]),
                               tuple(j["out_row_ports"]), j["upv"], tuple(map(tuple, j["leaves_out"])), j["in_words"],
                               j["const_pos"], j["out_col"])


def reference(j, seen: dict | None = None):
    """`SecondWires.piece` on `second_wires`: each decider's answer, the three together and the failures' codes; `seen`
    counts each decider's answers."""
    get = j.get if isinstance(j, dict) else lambda _k: None
    ps = get("parts")
    if not (isinstance(ps, list) and all(isinstance(p, list) and len(p) == 3 and isinstance(p[0], str) and _nat(p[1])
                                         and _nat(p[2]) for p in ps)):
        raise D.Reject("parts is a list of [template, lo, n]")
    parts = tuple(second_stage.Part(*p) for p in ps)
    ws = get("wires")
    if ws is not None:
        if not (isinstance(ws, list) and all(isinstance(w, dict) and sorted(w) == WIRE_KEYS
                                             and all(isinstance(v, str) for v in w.values()) for w in ws)):
            raise D.Reject("wires is null or a list of {template, port, writer, wport}")
        ws = tuple(second_stage.Wire(w["template"], w["port"], w["writer"], w["wport"]) for w in ws)
    rows = second_stage.rows_of(get("rows"))
    if rows is None:
        raise D.Reject("rows is a list of {template, port, per, runs}")
    cs = get("circuits")
    if not (isinstance(cs, list) and all(isinstance(c, list) and len(c) == 2 and isinstance(c[0], str) for c in cs)):
        raise D.Reject("circuits is a list of [template, layout]")
    circuits = tuple((t, _layout(c)) for t, c in cs)
    own = second_stage.Own((), parts, None, None, (), wires=ws, rows=rows)
    out = {"rows_under": second_wires.rows_under(own), "names_once": second_wires.names_once(parts),
           "wires_wide": second_wires.wires_wide(circuits, own), "decided": second_wires.decided(parts, circuits, own),
           "codes": [code for code, _why in second_wires.failures(parts, circuits, own)]}
    if seen is not None:
        for k in ("rows_under", "names_once", "wires_wide", "decided"):
            seen[k, out[k]] = seen.get((k, out[k]), 0) + 1
    return out


# --- named cases -----------------------------------------------------------------------------------------------------

def _writer(words: int = 2, **change) -> dict:
    """A writer's circuit file: one input row `x` of a word, the unit net's inputs, and one output row `o0` of `words`
    output ports of a word each, one instance returning each word as a leaf on the columns after the inputs."""
    return {"ports": [["x", 16], ["o0", 16 * words]], "out_ports": [[f"y{i}", 1] for i in range(words)],
            "out_row_ports": [words], "upv": 1, "leaves_out": [list(range(words))], "in_words": 1,
            "const_pos": 128 + 16 * words, "out_col": 1} | change


def _reader(bits: int = 32) -> dict:
    """A reader's circuit file: its input row `k` of `bits` bits and an output row of a word."""
    return {"ports": [["k", bits], ["o0", 16]], "out_ports": [["y0", 1]], "out_row_ports": [1], "upv": 1,
            "leaves_out": [[0]], "in_words": 1, "const_pos": 144, "out_col": 1}


def _design(**change) -> dict:
    """Reader `A`'s port `k` wired to writer `G`'s output row `o0`: 6 units of `G` a replay block and 2 of `A`, whose
    units read rows 0 and 3."""
    j = {"parts": [["G", 0, 6], ["A", 6, 2]],
         "wires": [{"template": "A", "port": "k", "writer": "G", "wport": "o0"}],
         "rows": [{"template": "A", "port": "k", "per": 6, "runs": [[0, 2, 0, 3]]}],
         "circuits": [["G", _writer()], ["A", _reader()]]}
    return j | change


def _answer(*codes: str) -> dict:
    return {"rows_under": "second-wire-rows" not in codes, "names_once": "second-part-names" not in codes,
            "wires_wide": "second-wire-widths" not in codes, "decided": not codes, "codes": list(codes)}


def _rows(per: int, *runs: list[int]) -> list[dict]:
    return [{"template": "A", "port": "k", "per": per, "runs": list(runs)}]


CASES = [
    ("the design", _design(), _answer()),
    # rowsUnder
    ("a run whose last unit reads the last row", _design(rows=_rows(6, [0, 2, 2, 3])), _answer()),
    ("a run whose last unit reads a row past the block's", _design(rows=_rows(6, [0, 2, 3, 3])),
     _answer("second-wire-rows")),
    ("a later run past the block's rows", _design(rows=_rows(6, [0, 1, 0, 0], [1, 1, 6, 0])),
     _answer("second-wire-rows")),
    ("a run of no unit, its first row under", _design(rows=_rows(6, [0, 0, 5, 9])), _answer()),
    ("a run of no unit, its first row past", _design(rows=_rows(6, [0, 0, 6, 0])), _answer("second-wire-rows")),
    ("rows of no row a block", _design(rows=_rows(0, [0, 1, 0, 0])), _answer("second-wire-rows")),
    ("a wire whose port has no rows", _design(rows=[]), _answer()),
    ("rows of a port no wire names", _design(rows=[{"template": "A", "port": "q", "per": 1, "runs": [[0, 2, 0, 3]]}]),
     _answer()),
    ("the first rows of a port are its rows", _design(rows=_rows(6, [0, 2, 0, 3]) + _rows(1, [0, 2, 0, 3])), _answer()),
    # namesOnce
    ("two parts of one template name", _design(parts=[["G", 0, 6], ["A", 6, 2], ["G", 8, 6]]),
     _answer("second-part-names")),
    ("no part", _design(parts=[]), _answer()),
    # wiresWide
    ("a reader port a word narrower than its writer's row", _design(circuits=[["G", _writer()], ["A", _reader(16)]]),
     _answer()),
    ("a reader port a bit wider than its writer's row", _design(circuits=[["G", _writer()], ["A", _reader(33)]]),
     _answer("second-wire-widths")),
    ("a writer that returns one word of its row's two",
     _design(circuits=[["G", _writer(leaves_out=[[0]])], ["A", _reader()]]), _answer("second-wire-widths")),
    ("a writer whose second instance returns the second word",
     _design(circuits=[["G", _writer(upv=2, leaves_out=[[0], [1]], out_col=0, in_words=0, const_pos=16)],
                       ["A", _reader()]]), _answer()),
    ("an instance past the writer's instances a unit",
     _design(circuits=[["G", _writer(upv=1, leaves_out=[[0], [1]], out_col=0, in_words=0, const_pos=16)],
                       ["A", _reader()]]), _answer("second-wire-widths")),
    ("a leaf's last bit on the constant's column",
     _design(circuits=[["G", _writer(const_pos=128 + 31)], ["A", _reader()]]), _answer("second-wire-widths")),
    ("a leaf's bits among the unit net's inputs", _design(circuits=[["G", _writer(out_col=0)], ["A", _reader()]]),
     _answer("second-wire-widths")),
    ("a unit net of fewer columns than its input words", _design(circuits=[["G", _writer(const_pos=100)],
                                                                             ["A", _reader()]]),
     _answer("second-wire-widths")),
    ("a wire of another output row", _design(wires=[{"template": "A", "port": "k", "writer": "G", "wport": "o1"}]),
     _answer("second-wire-widths")),
    ("a writer of two output rows, the wire's the second",
     _design(wires=[{"template": "A", "port": "k", "writer": "G", "wport": "o1"}],
             circuits=[["G", _writer(ports=[["x", 16], ["o0", 16], ["o1", 32]], out_ports=[["y0", 1], ["y1", 1], ["y2", 1]],
                                     out_row_ports=[1, 2], leaves_out=[[0, 1, 2]], const_pos=128 + 48)],
                       ["A", _reader()]]), _answer()),
    ("no circuit file for the writer", _design(circuits=[["A", _reader()]]), _answer("second-wire-widths")),
    ("no circuit file for the reader", _design(circuits=[["G", _writer()]]), _answer("second-wire-widths")),
    ("the first file of a name is its file", _design(circuits=[["G", _writer()], ["A", _reader()], ["A", _reader(48)]]),
     _answer()),
    ("a reader with no port of the wire's name",
     _design(wires=[{"template": "A", "port": "q", "writer": "G", "wport": "o0"}]), _answer()),
    # the three
    ("no wire", _design(wires=[], rows=[], circuits=[]), _answer()),
    ("no derived wires", _design(wires=None, rows=_rows(0, [0, 1, 0, 0]), circuits=[]), _answer()),
    ("all three refused", _design(parts=[["A", 0, 1], ["A", 1, 1]], rows=_rows(1, [0, 2, 0, 3]), circuits=[]),
     _answer("second-wire-rows", "second-part-names", "second-wire-widths")),
]


# --- strategies ------------------------------------------------------------------------------------------------------

JSON = st.recursive(st.none() | st.booleans() | st.integers(-3, 3) | st.text("ab/1", max_size=3),
                    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text("abk", max_size=2), inner, max_size=3),
                    max_leaves=6)
NAMES = ["A", "B", "G"]


@st.composite
def layouts(pick):
    """A circuit file's layout: one or two input rows, one or two output rows of one or two ports of one or two words,
    one or two instances returning a few output words, mostly on computed columns."""
    rows = pick(st.integers(1, 2))
    counts = [pick(st.integers(1, 2)) for _ in range(rows)]
    out_ports = [[f"y{i}", pick(st.integers(1, 2))] for i in range(sum(counts))]
    words = sum(w for _, w in out_ports)
    ins = [[p, pick(st.sampled_from([0, 16, 17, 32, 48]))] for p in pick(st.sampled_from([["k"], ["x", "k"], ["k", "k"]]))]
    outs = [[f"o{i}", 16] for i in range(rows - (pick(st.integers(0, 9)) == 0))]
    upv = pick(st.integers(0, 2))
    listed = pick(st.sampled_from([upv, upv, upv, upv + 1, max(upv - 1, 0)]))
    leaves = [pick(st.lists(st.integers(0, words), max_size=3)) if pick(st.integers(0, 3)) else list(range(words))
              for _ in range(listed)]
    in_words = pick(st.integers(0, 1))
    most = max([len(ls) for ls in leaves] + [0])
    return {"ports": ins + outs, "out_ports": out_ports, "out_row_ports": counts, "upv": upv, "leaves_out": leaves,
            "in_words": in_words,
            "const_pos": 128 * in_words + 16 * most + pick(st.sampled_from([0, 0, 0, 0, -1, -17, 5])) if most else
            pick(st.integers(0, 130)),
            "out_col": pick(st.sampled_from([in_words, in_words, in_words, 0, 1]))}


@st.composite
def instances(pick, broken: bool = False):
    """Instances mostly well formed: two or three parts, sometimes two of one name, a circuit file for most, up to two
    wires between them and rows for most wired ports, their runs mostly under the rows a block; at `broken` one field
    of one part of the instance replaced by arbitrary JSON."""
    names = pick(st.lists(st.sampled_from(NAMES), min_size=0, max_size=3, unique=pick(st.integers(0, 4)) != 0))
    parts = [[t, 4 * i, pick(st.integers(0, 4))] for i, t in enumerate(names)]
    circuits = [[t, pick(layouts())] for t in NAMES if pick(st.integers(0, 7))]
    wires, rows = [], []
    for _ in range(pick(st.integers(0, 2))):
        w = {"template": pick(st.sampled_from(NAMES)), "port": pick(st.sampled_from(["k", "k", "x", "q"])),
             "writer": pick(st.sampled_from(NAMES)), "wport": pick(st.sampled_from(["o0", "o0", "o1", "q"]))}
        wires.append(w)
        if pick(st.integers(0, 5)):
            per = pick(st.integers(0, 6))
            runs = [[pick(st.integers(0, 3)), pick(st.integers(0, 3)), pick(st.integers(0, 3)), pick(st.integers(0, 2))]
                    for _ in range(pick(st.integers(0, 3)))]
            rows.append({"template": w["template"], "port": w["port"], "per": per, "runs": runs})
    j = {"parts": parts, "wires": wires if pick(st.integers(0, 9)) else None, "rows": rows, "circuits": circuits}
    if broken:
        part = pick(st.sampled_from(["parts", "wires", "rows", "circuits", "wire", "row", "layout", "part", "missing"]))
        if part == "wire" and wires:
            wires[0][pick(st.sampled_from([*WIRE_KEYS, "extra"]))] = pick(JSON)
        elif part == "row" and rows:
            rows[0][pick(st.sampled_from(["template", "port", "per", "runs", "extra"]))] = pick(JSON)
        elif part == "layout" and circuits:
            circuits[0][1][pick(st.sampled_from([*LAYOUT_KEYS, "extra"]))] = pick(JSON)
        elif part == "part" and parts:
            parts[0][pick(st.integers(0, 2))] = pick(JSON)
        elif part == "missing":
            del j[pick(st.sampled_from(["parts", "rows", "circuits"]))]
        else:
            j[pick(st.sampled_from(["parts", "wires", "rows", "circuits"]))] = pick(JSON)
    return j


@st.composite
def wired(pick):
    """A named case with a few of its numbers moved: the cases' designs are the ones whose widths mostly agree."""
    j = pick(st.sampled_from([c for _name, c, _want in CASES]))
    j = {"parts": [list(p) for p in j["parts"]], "wires": j["wires"],
         "rows": [dict(r, runs=[list(x) for x in r["runs"]]) for r in j["rows"]],
         "circuits": [[t, dict(c, ports=[list(p) for p in c["ports"]])] for t, c in j["circuits"]]}
    for _t, c in j["circuits"]:
        if pick(st.integers(0, 2)) == 0:
            k = pick(st.sampled_from(["const_pos", "in_words", "out_col", "upv"]))
            c[k] = max(0, c[k] + pick(st.sampled_from([-1, 1])))
        if pick(st.integers(0, 3)) == 0:
            p = pick(st.sampled_from(c["ports"]))
            p[1] = max(0, p[1] + pick(st.sampled_from([-16, -1, 1, 16])))
    for r in j["rows"]:
        if pick(st.integers(0, 2)) == 0:
            r["per"] = max(0, r["per"] + pick(st.integers(-2, 1)))
    return j


INPUTS = st.one_of(*[instances() for _ in range(4)], wired(), wired(), instances(broken=True), JSON)
SECOND_WIRES = D.Port(PIECES, "second_wires", reference, INPUTS, package=PACKAGE, of_record="lean")

pieces = D.server_fixture(SECOND_WIRES)


@pytest.mark.parametrize("name, j, want", CASES, ids=lambda x: x if isinstance(x, str) else "")
def test_named_cases(pieces, name, j, want):
    assert reference(j) == want
    assert pieces.call("second_wires", j) == ("output", D.canonical(want))


def test_the_deciders_agree_and_each_gives_both_answers(pieces):
    seen: dict = {}
    port = D.Port(PIECES, "second_wires", lambda j: reference(j, seen), INPUTS, package=PACKAGE, of_record="lean")
    stats = D.run(port, examples=1500, server=pieces)
    assert stats.outputs > 1000 and stats.rejects > 0, stats
    few = {k: seen.get(k, 0) for k in [(d, a) for d in ("rows_under", "names_once", "wires_wide", "decided")
                                       for a in (True, False)] if seen.get(k, 0) < 20}
    assert not few, f"answers the generated instances reach fewer than 20 times: {few}"


@pytest.mark.parametrize("j", [
    None, [], {"parts": [], "rows": []}, {"parts": [], "circuits": []}, {"rows": [], "circuits": []},
    _design(parts=[["G", 0]]), _design(parts=[["G", 0, -1]]), _design(parts=[["G", 0, True]]),
    _design(wires=[{"template": "A", "port": "k", "writer": "G"}]), _design(wires={}),
    _design(rows=[{"template": "A", "port": "k", "per": 6, "runs": [[0, 2, 0]]}]),
    _design(circuits=[["G", _writer() | {"extra": 0}]]), _design(circuits=[["G", _writer(upv=-1)]]),
    _design(circuits=[["G", _writer(leaves_out=[0])]]), _design(circuits=[["G", _writer(ports=[["x"]])]]),
    _design(circuits=[["G"]]), _design(circuits=[[0, _writer()]]),
], ids=lambda j: D.canonical(j)[:60])
def test_a_malformed_instance_is_rejected_by_both(pieces, j):
    with pytest.raises(D.Reject):
        reference(j)
    kind, _why = pieces.call("second_wires", j)
    assert kind == "reject"
