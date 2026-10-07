"""V*'s circuits in Lean (`VStar`, served by `vstar-difftest`) against `verity_flock`'s builders, part by part on drawn
parameters: the SHA-512 gadget, the climb and the GF(2^128) residuals as laid-out netlists, each statement's unit
(`rec_outer.lowering`, `rec_residuals.lowering`) as its lowering text, and its whole `flock-circuit` file (`circuit.compose`)
under drawn bindings. Big outputs are compared as digest and size; `test_*_file_is_byte_equal` compares whole files.

The bindings (`partition`, `program`, `program_digests`) are core's descriptor codec's and are inputs here: the reference's
subcircuit hands `compose` the drawn `program_digests` as they are (`_given`)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.ml.circuits.boolean.forms import Circuit
from verity_flock import circuit as FC
from verity_flock import ir_lower as IL
from verity_flock import rec_open as R
from verity_flock import rec_outer as O
from verity_flock import rec_residuals as RR

LEAN = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

HEX = "0123456789abcdef"
SEED = 20261007


# --- the references: Lean's readers, then the Python builders -----------------------------------------------------------------

def _obj(v, keys: tuple[str, ...]) -> dict:
    if not isinstance(v, dict) or set(v) != set(keys):
        raise D.Reject(f"an object of the keys {keys}")
    return v


def _nat(x) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        raise D.Reject("not a natural number")
    return x


def _str(x) -> str:
    if not isinstance(x, str):
        raise D.Reject("not a string")
    return x


def _list(x) -> list:
    if not isinstance(x, list):
        raise D.Reject("not a list")
    return x


def _term(t) -> list:
    t = _list(t)
    if not t:
        raise D.Reject("term: not w, v or wt")
    tag = _str(t[0])
    if tag in ("w", "v") and len(t) == 2:
        return [tag, _nat(t[1])]
    if tag == "wt" and len(t) == 3:
        return [tag, _nat(t[1]), _nat(t[2])]
    raise D.Reject("term: not w, v or wt")


def _pair(ab) -> list:
    ab = _list(ab)
    if len(ab) != 2:
        raise D.Reject("a product is [a, b]")
    return [_nat(ab[0]), _nat(ab[1])]


def _gf(j) -> dict:
    """`VStar.GfS.ofJson`: GfResiduals' S, read strictly."""
    o = _obj(j, ("nw", "nv", "ops", "res"))
    ops = [[_term(t) for t in _list(op)] for op in _list(o["ops"])]
    res = []
    for r in _list(o["res"]):
        ro = _obj(r, ("mul", "add"))
        res.append({"mul": [_pair(ab) for ab in _list(ro["mul"])], "add": [_term(t) for t in _list(ro["add"])]})
    return {"nw": _nat(o["nw"]), "nv": _nat(o["nv"]), "ops": ops, "res": res}


def _reps(j) -> dict:
    """`VStar.RepS.ofJson`: InnerRepCheck's S."""
    o = _obj(j, ("gf", "ports"))
    gf = _gf(o["gf"])
    ports = []
    for p in _list(o["ports"]):
        p = _list(p)
        if len(p) != 2:
            raise D.Reject("a row port is [name, elements]")
        ports.append([_str(p[0]), _nat(p[1])])
    return {"gf": gf, "ports": ports}


def _part(C: Circuit, ins: list[str], out: list) -> dict:
    """A part laid out as `VStar.DT.partNet` lays it out: its input ports as one group, its outputs as one port."""
    C.output("y", out)
    lay = IL.layout(C, ins, ["y"])
    text = IL.netlist(lay, "part")
    return {"sha256": IL.digest(text), "bytes": len(text.encode()), "rows": lay.useful}


def sha(v) -> dict:
    o = _obj(v, ("cv", "bits", "pre"))
    cv = [_nat(x) for x in _list(o["cv"])]
    n, pre = _nat(o["bits"]), _nat(o["pre"])
    if len(cv) != 8 or any(x >= 1 << 64 for x in cv):
        raise D.Reject("cv: eight 64-bit words")
    if n % 8 or pre % 128:
        raise D.Reject("a message of whole bytes after whole blocks")
    C = Circuit()
    x = C.input("x", n)
    return _part(C, ["x"], R._sha(R._Forms(C), cv, x, pre))


def climb(v) -> dict:
    o = _obj(v, ("H", "row"))
    H, n = _nat(o["H"]), _nat(o["row"])
    if H > 16 or n % 8:
        raise D.Reject("at most 16 siblings, a row of whole bytes")
    C = Circuit()
    row, aux, d = C.input("row", n), C.input("aux", 512 * H), C.input("d", 16)
    return _part(C, ["row", "aux", "d"], R._climb(R._Forms(C), H, row, aux, d))


def residuals(v) -> dict:
    S = _gf(v)
    C = Circuit()
    w, vv = C.input("w", 128 * S["nw"]), C.input("v", 128 * S["nv"])
    return _part(C, ["w", "v"], RR.residual_forms(C, S, w, vv))


def _lowering(low: IL.Lowering) -> dict:
    return {"sha256": low.sha256, "bytes": len(low.text.encode()), "rows": low.layout.useful}


def _given(**_):
    raise TypeError("the bindings are given")


def _sub(id_: str, template: str, inputs: list[str], outputs: list[str], digests) -> SimpleNamespace:
    """`rec_outer.subcircuit`'s and `rec_residuals.subcircuit`'s shape, its `program_digests` the given ones."""
    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")
    return SimpleNamespace(id=id_, template=SimpleNamespace(id=template), inputs=[port(nm) for nm in inputs],
                           outputs=[port(nm) for nm in outputs], definition=_given, program_digests=lambda statics: digests)


def _file(text: str) -> dict:
    return {"sha512": FC.digest(text), "bytes": len(text.encode())}


def rec_open_unit(v) -> dict:
    o = _obj(v, ("LANES", "H"))
    return _lowering(O.lowering(_nat(o["LANES"]), _nat(o["H"])))


def _rec_open_text(o: dict) -> str:
    L, H = _nat(o["LANES"]), _nat(o["H"])
    partition, program = _str(o["partition"]), _str(o["program"])
    sub = _sub(f"rec-open/l{L}-h{H}", "rec-open", list(R.PORTS), ["out", "acc_out"], o["program_digests"])
    return FC.compose(O.lowering(L, H), sub, partition=partition, program=program)


def rec_open_circuit(v) -> dict:
    return _file(_rec_open_text(_obj(v, ("LANES", "H", "partition", "program", "program_digests"))))


def _inner(S) -> dict:
    """InnerRepCheck's S, refused where `rec_residuals.lowering` has no input word to lay out (`ir_lower.layout` then fails
    on its missing input group)."""
    S = _reps(S)
    if sum(RR.port_words(n) for _, n in S["ports"]) + RR.port_words(S["gf"]["nv"]) == 0:
        raise D.Reject("a unit of no input words")
    return S


def inner_unit(v) -> dict:
    o = _obj(v, ("S", "name"))
    return _lowering(RR.lowering(_inner(o["S"]), _str(o["name"])))


def _inner_text(o: dict) -> str:
    S, name = _inner(o["S"]), _str(o["name"])
    partition, program = _str(o["partition"]), _str(o["program"])
    sub = _sub(name, "rec-residuals", [nm for nm, _ in S["ports"]] + ["v"], ["out"], o["program_digests"])
    return FC.compose(RR.lowering(S, name), sub, partition=partition, program=program)


def inner_circuit(v) -> dict:
    return _file(_inner_text(_obj(v, ("S", "name", "partition", "program", "program_digests"))))


# --- the inputs: valid ones and the readers' and builders' refusals -------------------------------------------------------------

JSON = st.recursive(st.none() | st.booleans() | st.integers(-3, 3) | st.text(max_size=3),
                    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=3), inner, max_size=3),
                    max_leaves=6)
WORD = st.integers(0, (1 << 64) - 1) | st.sampled_from([0, (1 << 64) - 1])


def _near(lo: int, hi: int):
    """Naturals in [lo, hi] and a few just outside, negatives and booleans included."""
    return st.integers(lo, hi) | st.sampled_from([hi + 1, -1, True])


SHA = st.one_of(
    st.fixed_dictionaries({"cv": st.just(list(R.IV)) | st.lists(WORD, min_size=8, max_size=8),
                           "bits": st.integers(0, 239).map(lambda b: 8 * b), "pre": st.integers(0, 3).map(lambda b: 128 * b)}),
    st.fixed_dictionaries({"cv": st.lists(WORD, min_size=7, max_size=9) | st.just([1 << 64] * 8) | st.just([-1] * 8),
                           "bits": st.integers(0, 64) | st.just(-8), "pre": st.integers(0, 300)}),
    st.dictionaries(st.sampled_from(["cv", "bits", "pre", "x"]), st.integers(0, 8), max_size=3),
    JSON)
CLIMB = st.one_of(
    st.fixed_dictionaries({"H": st.integers(0, 1), "row": st.integers(0, 111).map(lambda b: 8 * b)}),
    st.fixed_dictionaries({"H": st.sampled_from([17, 40, -1, True]), "row": st.just(64)}),
    st.fixed_dictionaries({"H": st.just(1), "row": st.integers(1, 7) | st.just(-8)}),
    st.dictionaries(st.sampled_from(["H", "row", "d"]), st.integers(0, 2), max_size=2),
    JSON)


@st.composite
def gf_s(draw, big: bool = False, strict: bool = False) -> dict:
    """A GfResiduals S, its indices mostly in range (`check_structure` refuses the rest), or all of them (`strict`): ports of a
    few elements, or (`big`) a w of at least 128 elements, for `wt` terms."""
    nw = draw(st.integers(128, 130) if big else st.integers(1 if strict else 0, 3))
    nv = draw(st.integers(0, 3))
    tags = [t for t, ok in (("w", nw > 0 or not strict), ("v", nv > 0 or not strict), ("wt", big)) if ok]
    out = (lambda: st.just(False)) if strict else (lambda: st.integers(0, 9).map(lambda x: x == 0))

    @st.composite
    def term(d):
        tag = d(st.sampled_from(tags))
        if tag == "wt":
            return ["wt", d(st.integers(0, nw - 128 + d(out()))), d(st.integers(0, 127 + d(out())))]
        size = nw if tag == "w" else nv
        return [tag, d(st.integers(0, size if d(out()) else max(size - 1, 0)))]

    ops = draw(st.lists(st.lists(term(), max_size=3), max_size=3))
    k = len(ops)
    pair = st.lists(st.integers(0, k if draw(out()) else max(k - 1, 0)), min_size=2, max_size=2)
    res = draw(st.lists(st.fixed_dictionaries({"mul": st.lists(pair, max_size=2) if k else st.just([]),
                                               "add": st.lists(term(), max_size=3)}), max_size=3))
    return {"nw": nw, "nv": nv, "ops": ops, "res": res}


def _broken(S: dict):
    """S with one field of the wrong kind: a key missing or extra, a term or product misshapen."""
    return st.sampled_from([
        {k: v for k, v in S.items() if k != "res"}, {**S, "x": 0}, {**S, "nw": True}, {**S, "nv": -1},
        {**S, "ops": [[["u", 0]]]}, {**S, "ops": [[["w"]]]}, {**S, "ops": [[[]]]}, {**S, "ops": [[["wt", 0]]]},
        {**S, "res": [{"mul": [[0]], "add": []}]}, {**S, "res": [{"mul": [], "add": [], "x": []}]}])


GF = st.one_of(gf_s(), gf_s(strict=True), gf_s(big=True), gf_s().flatmap(_broken), JSON)
NAME = st.text(max_size=4) | st.sampled_from(["rec-residuals/l0-p0", "row0"])


@st.composite
def rep_s(draw, valid: bool = True) -> dict:
    """InnerRepCheck's S: a GfResiduals S with w split into named row ports (`valid`), or ports that refuse."""
    gf = draw(gf_s(strict=valid))
    if valid:
        cuts = sorted(draw(st.lists(st.integers(1, max(gf["nw"] - 1, 1)), max_size=2, unique=True)))
        sizes = [b - a for a, b in zip([0, *cuts], [*cuts, gf["nw"]]) if b > a]
        names = draw(st.lists(NAME.filter(lambda s: s != "v"), min_size=len(sizes), max_size=len(sizes), unique=True))
        return {"gf": gf, "ports": [[nm, n] for nm, n in zip(names, sizes)]}
    return {"gf": gf, "ports": draw(st.lists(st.tuples(st.sampled_from(["a", "a", "v"]), st.integers(0, 4)).map(list),
                                             max_size=3))}


REPS = st.one_of(rep_s(), rep_s(valid=False), JSON)
DIGEST = st.text(HEX, min_size=128, max_size=128)
BINDING = st.one_of(DIGEST, st.sampled_from(["a" * 128, "A" * 128, "a" * 127, "a" * 129, "g" * 128]))
DIGESTS = st.dictionaries(st.text(max_size=12), st.text(HEX, max_size=64), max_size=2) | JSON


def port(fn: str, reference, inputs) -> D.Port:
    return D.Port("vstar-difftest", fn, reference, inputs, package=LEAN, rejects=(D.Reject, ValueError))


server = D.server_fixture(port("sha", sha, SHA))


# --- the parts ------------------------------------------------------------------------------------------------------------------

def test_residuals_agree(server):
    stats = D.run(port("residuals", residuals, GF), examples=60, seed=SEED, server=server)
    assert stats.outputs >= 10 and stats.rejects >= 10, stats


@pytest.mark.slow
def test_sha_agrees(server):
    stats = D.run(port("sha", sha, SHA), examples=20, seed=SEED, server=server)
    assert stats.outputs >= 3 and stats.rejects >= 3, stats


@pytest.mark.slow
def test_climb_agrees(server):
    stats = D.run(port("climb", climb, CLIMB), examples=16, seed=SEED, server=server)
    assert stats.outputs >= 3 and stats.rejects >= 3, stats


@pytest.mark.slow
def test_rec_open_residuals_agree(server):
    """`RecOpen{LANES, ·}`'s running sums (`rec_open.cons_structure`), the structure the level units hold."""
    for L in (8, 16):
        assert D.replay(port("residuals", residuals, GF), R.cons_structure(L), server)[0] == "output"


# --- the units ------------------------------------------------------------------------------------------------------------------

REC_OPEN = st.one_of(st.just({"LANES": 8, "H": 1}),
                     st.fixed_dictionaries({"LANES": st.sampled_from([0, 4, 7, 9, 12, -8, True]), "H": st.just(1)}),
                     st.fixed_dictionaries({"LANES": st.just(8), "H": st.sampled_from([0, 17, -1, True])}),
                     st.dictionaries(st.sampled_from(["LANES", "H", "x"]), st.just(7), max_size=3), JSON)


@pytest.mark.slow
def test_inner_unit_agrees(server):
    stats = D.run(port("inner_unit", inner_unit, st.fixed_dictionaries({"S": REPS, "name": NAME}) | JSON), examples=40,
                  seed=SEED, server=server)
    assert stats.outputs >= 5 and stats.rejects >= 5, stats


@pytest.mark.slow
def test_rec_open_unit_agrees(server):
    stats = D.run(port("rec_open_unit", rec_open_unit, REC_OPEN), examples=30, seed=SEED, server=server)
    assert stats.outputs >= 1 and stats.rejects >= 10, stats


# --- the circuit files ----------------------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_inner_circuit_agrees(server):
    """`compose` on small InnerRepCheck units under drawn bindings, invalid ones included: META, the zero instance's
    outputs and `b ‖ c`, the slot ranges and the block size."""
    inputs = st.fixed_dictionaries({"S": rep_s(), "name": NAME, "partition": BINDING, "program": BINDING,
                                    "program_digests": DIGESTS})
    stats = D.run(port("inner_circuit", inner_circuit, inputs), examples=5, seed=SEED, server=server)
    assert stats.outputs >= 2 and stats.rejects >= 1, stats


@pytest.mark.slow
def test_rec_open_circuit_refuses_as_compose_does(server):
    """RecOpen's file is refused where `compose` refuses it (its unit's parameters, a key missing, a binding that is not a
    SHA-512 digest); `test_rec_open_file_is_byte_equal` compares a file it writes."""
    p = port("rec_open_circuit", rec_open_circuit, st.nothing())
    good = {"LANES": 8, "H": 1, "partition": "a" * 128, "program": "b" * 128, "program_digests": {}}
    for bad in ({**good, "LANES": 7}, {**good, "LANES": True}, {**good, "H": 0}, {**good, "H": 17},
                {k: v for k, v in good.items() if k != "program"}, {**good, "x": 0}, {**good, "partition": "A" * 128}):
        assert D.replay(p, bad, server)[0] == "reject", bad


# --- whole files, byte for byte -------------------------------------------------------------------------------------------------

def _vstar(tmp_path: Path, *args) -> bytes:
    exe = D.build(LEAN, "vstar")
    out = tmp_path / "circuit.txt"
    p = subprocess.run([str(exe), *args, str(out)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    data, said = out.read_bytes(), json.loads(p.stdout)
    assert (said["sha512"], said["bytes"]) == (hashlib.sha512(data).hexdigest(), len(data))
    return data


#: `rec_outer.subcircuit(8, 1)`'s program digest (core's codec), and the circuit file `compose` writes under it with the
#: bindings `a…a` and `b…b`.
REC_OPEN_8_1 = {"RecOpen_v4{LANES=8,H=1}": "cb8c446680677966b08cdf47d09426438abcf47e3ac7cff5f76cd0225ef157ce"}
REC_OPEN_8_1_FILE = ("53f94750ba21b8d16f489eb419cce68160af910f414df000dcc6f0f348a08eb4"
                     "21599bc8e20d7cc3b6985b011c3b303acd31bff8c609c8fd6622522ae2bcff46", 92449954)


@pytest.mark.slow
def test_rec_open_file_is_byte_equal(tmp_path):
    """RecOpen{8, 1}'s file from `vstar rec-open`, against `compose`'s text and its pinned digest."""
    b = {"partition": "a" * 128, "program": "b" * 128, "program_digests": REC_OPEN_8_1}
    (tmp_path / "b.json").write_text(json.dumps(b))
    lean = _vstar(tmp_path, "rec-open", "8", "1", str(tmp_path / "b.json"))
    assert (hashlib.sha512(lean).hexdigest(), len(lean)) == REC_OPEN_8_1_FILE
    assert lean == _rec_open_text({"LANES": 8, "H": 1, **b}).encode()


@pytest.mark.slow
def test_inner_file_is_byte_equal(tmp_path):
    """A small InnerRepCheck's file from `vstar inner`, names outside ASCII included, against `compose`'s text."""
    S = {"gf": {"nw": 3, "nv": 2, "ops": [[["w", 0], ["v", 1]], [["w", 2]]],
                "res": [{"mul": [[0, 1], [1, 1]], "add": [["v", 0]]}, {"mul": [], "add": [["w", 1]]}]},
         "ports": [["é0", 2], ["row\t1", 1]]}
    b = {"partition": "c" * 128, "program": "d" * 128, "program_digests": {"InnerRepCheck_v2{S=…}": "e" * 64}}
    (tmp_path / "s.json").write_text(json.dumps(S))
    (tmp_path / "b.json").write_text(json.dumps(b))
    lean = _vstar(tmp_path, "inner", str(tmp_path / "s.json"), "rec-residuals/ü", str(tmp_path / "b.json"))
    assert lean == _inner_text({"S": S, "name": "rec-residuals/ü", **b}).encode()

