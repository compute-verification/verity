"""V*'s circuits in Lean (`VStar`, served by `vstar-difftest`) against `verity_flock`'s builders, part by part on drawn
parameters: the SHA-512 gadget, the climb and the GF(2^128) residuals as laid-out netlists, the algebra's structure and its
parts (`rec_algebra.structure`, `rec_residuals.spec`, `rec_residuals.parts`), V*'s statements over an inner session's shape
(`rec_vstage.stage`'s), each statement's unit (`rec_outer.lowering`, `rec_residuals.lowering`) as its lowering text, and its
whole `flock-circuit` file (`circuit.compose`) under drawn bindings. Big outputs are compared as digest and size;
`test_*_file_is_byte_equal` compares whole files.

The bindings (`partition`, `program`, `program_digests`) are core's descriptor codec's and are inputs here: the reference's
subcircuit hands `compose` the drawn `program_digests` as they are (`_given`)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.ml.circuits.boolean.forms import Circuit
from verity.ml.circuits.ops.boolean import gf2k as G
from verity_flock import circuit as FC
from verity_flock import ir_lower as IL
from verity_flock import rec_algebra as A
from verity_flock import rec_open as R
from verity_flock import rec_outer as O
from verity_flock import rec_residuals as RR

LEAN = Path(__file__).resolve().parents[1]
#: inner sessions' shapes, ``M,K_LOG,CLAIMS`` separated by ``;``, whose whole stage `test_stage_is_byte_equal` compares
STAGE = os.environ.get("VERITY_VSTAR_STAGE", "")
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


def _sha256(x) -> str:
    """A JSON value's SHA-256 as `json.dumps(sort_keys=True, separators=(",", ":"))` writes it (`VStar.DT.jsonSha256`)."""
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _gf_summary(S: dict) -> dict:
    return {"sha256": _sha256(S), "nw": S["nw"], "nv": S["nv"], "ops": len(S["ops"]), "res": len(S["res"]),
            "products": sum(len(r["mul"]) for r in S["res"])}


def _shape(v, holo: bool = False) -> A.Shape:
    o = _obj(v, ("m", "k_log", "claims"))
    return A.Shape(_nat(o["m"]), _nat(o["k_log"]), _nat(o["claims"]), holo=holo)


def _algebra(f, sh: A.Shape):
    """``f(sh)``; below ``k_log = 6`` `rec_algebra.run` reads past the first claim's point (an IndexError), a refusal."""
    try:
        return f(sh)
    except IndexError:
        if sh.k_log < 6:
            raise D.Reject("k_log below 6") from None
        raise


def structure(v) -> dict:
    return _gf_summary(_algebra(lambda sh: A.structure(sh)[0], _shape(v)))


def spec(v, holo: bool = False) -> dict:
    S = _algebra(RR.spec, _shape(v, holo))
    return {"S": _sha256(S), "ports": S["ports"], **_gf_summary(S["gf"])}


def parts(v) -> list:
    """`rec_residuals.parts` on a checked S: it reads S's terms through their ports unchecked, so past `check_structure`
    its lookups fail."""
    o = _obj(v, ("S", "max_regions", "max_slots", "max_lone_slots"))
    S = _reps(o["S"])
    G.check_structure(S["gf"])
    return [{"S": p.S, "res": p.res, "v": p.v}
            for p in RR.parts(S, _nat(o["max_regions"]), _nat(o["max_slots"]), _nat(o["max_lone_slots"]))]


def statements(v, holo: bool = False) -> list:
    """`rec_vstage.stage`'s statements: per Ligerito level, `rec_outer.levels` of the session's openings (`rec_vstar.decode`
    gives level l's rows ``2^lanes`` elements and ``d - cap`` siblings, each rep ``q`` openings), then the algebra's parts;
    ``holo``: over a holo inner session (`rec_algebra.Shape.holo`)."""
    sh = _shape(v, holo)
    ps = _algebra(lambda sh: RR.parts(RR.spec(sh)), sh)
    out = [{"name": f"rec-open/l{1 << lv.lanes}-h{lv.d - lv.cap}", "dir": f"L{l}", "instances": 2 * lv.q, "level": l,
            "LANES": 1 << lv.lanes, "H": lv.d - lv.cap} for l, lv in enumerate(A.fast100(sh.m).levels)]
    return out + [{"name": f"rec-residuals/m{sh.m}-k{sh.k_log}-c{sh.claims}-p{j}", "dir": f"alg/p{j}", "instances": 2,
                   "S": _sha256(p.S), "ports": len(p.S["ports"]), "res_of": p.res, "v_of": p.v, **_gf_summary(p.S["gf"])}
                  for j, p in enumerate(ps)]


def verifier_parts(v, holo: bool = False) -> dict:
    """Every algebra statement's unit is the Lean verifier's own `rec-residuals/…` net, and it refuses the part past the
    last."""
    sh = _shape(v, holo)
    ps = _algebra(lambda sh: RR.parts(RR.spec(sh)), sh)
    return {"parts": [{"name": f"rec-residuals/m{sh.m}-k{sh.k_log}-c{sh.claims}-p{j}", "equal": True} for j in range(len(ps))],
            "past_refused": True}


def _lowering(low: IL.Lowering) -> dict:
    return {"sha256": low.sha256, "bytes": len(low.text.encode()), "rows": low.layout.useful}


def verifier_algebra(v) -> dict:
    """`rec_residuals.spec` and `parts` at ``Shape(m, k_log, claims, holo=holo)``: the whole S's summary and each part's S,
    residuals and v elements, with ``units`` each part's lowering under its `rec-residuals/…` name."""
    o = _obj(v, ("m", "k_log", "claims", "holo", "units"))
    if not isinstance(o["holo"], bool) or not isinstance(o["units"], bool):
        raise D.Reject("holo and units are booleans")
    sh = A.Shape(_nat(o["m"]), _nat(o["k_log"]), _nat(o["claims"]), holo=o["holo"])
    S = _algebra(RR.spec, sh)
    out = []
    for j, p in enumerate(RR.parts(S)):
        name = f"rec-residuals/m{sh.m}-k{sh.k_log}-c{sh.claims}-p{j}"
        out.append({"name": name, "S": _sha256(p.S), "res": p.res, "v": p.v,
                    "unit": _lowering(RR.lowering(p.S, name)) if o["units"] else None})
    return {"parts": out, **_gf_summary(S["gf"])}


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


def _inner_text(o: dict, holo: bool = False) -> str:
    """`compose` of InnerRepCheck{S} as `rec_residuals.statement` composes it, ``holo`` its META's `HOLO_KEY`."""
    S, name = _inner(o["S"]), _str(o["name"])
    partition, program = _str(o["partition"]), _str(o["program"])
    sub = _sub(name, "rec-residuals", [nm for nm, _ in S["ports"]] + ["v"], ["out"], o["program_digests"])
    return FC.compose(RR.lowering(S, name), sub, partition=partition, program=program,
                      extra_meta={RR.HOLO_KEY: True} if holo else None)


def inner_circuit(v, holo: bool = False) -> dict:
    return _file(_inner_text(_obj(v, ("S", "name", "partition", "program", "program_digests")), holo))


def spec_holo(v) -> dict:
    return spec(v, holo=True)


def statements_holo(v) -> list:
    return statements(v, holo=True)


def verifier_parts_holo(v) -> dict:
    return verifier_parts(v, holo=True)


def inner_circuit_holo(v) -> dict:
    return inner_circuit(v, holo=True)


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
def rep_s(draw, valid: bool = True, big: bool = False) -> dict:
    """InnerRepCheck's S: a GfResiduals S (`big`: with `wt` terms) with w split into named row ports (`valid`), or ports
    that refuse."""
    gf = draw(gf_s(strict=valid, big=big))
    if valid:
        cuts = sorted(draw(st.lists(st.integers(1, max(gf["nw"] - 1, 1)), max_size=2, unique=True)))
        sizes = [b - a for a, b in zip([0, *cuts], [*cuts, gf["nw"]]) if b > a]
        names = draw(st.lists(NAME.filter(lambda s: s != "v"), min_size=len(sizes), max_size=len(sizes), unique=True))
        return {"gf": gf, "ports": [[nm, n] for nm, n in zip(names, sizes)]}
    return {"gf": gf, "ports": draw(st.lists(st.tuples(st.sampled_from(["a", "a", "v"]), st.integers(0, 4)).map(list),
                                             max_size=3))}



def _one(*strategies):
    """One branch of a `one_of` (which flattens nested ones and drops repeated ones), so valid inputs can outweigh it."""
    return st.one_of(*strategies).map(lambda v: v)


def _shapes(claims: int):
    """Inner sessions' shapes, three times in four: fast100's m (each of three ranges a branch), k_log from 6, 2 to
    ``claims`` claims; else m, k_log or claims just outside, or not a shape."""
    valid = lambda lo, hi: st.fixed_dictionaries({"m": st.integers(lo, hi), "k_log": st.integers(6, 40),
                                                  "claims": st.integers(2, claims)})
    return st.one_of(valid(22, 26), valid(27, 30), valid(31, 35), _one(
        st.fixed_dictionaries({"m": st.sampled_from([0, 21, 36, -1, True]), "k_log": st.just(20), "claims": st.just(2)}),
        st.fixed_dictionaries({"m": st.just(22), "k_log": st.sampled_from([0, 5, -1]), "claims": st.just(2)}),
        st.fixed_dictionaries({"m": st.just(22), "k_log": st.just(6), "claims": st.sampled_from([0, 1, claims + 1, -1])}),
        st.dictionaries(st.sampled_from(["m", "k_log", "claims", "x"]), st.integers(20, 24), max_size=3), JSON))


@st.composite
def _many(draw) -> dict:
    """A valid S of more residuals than one statement holds (a whole algebra's, as at 12 claims), each one of S's."""
    S = draw(rep_s())
    res = draw(st.lists(st.sampled_from(S["gf"]["res"] or [{"mul": [], "add": []}]), min_size=17, max_size=20))
    return {**S, "gf": {**S["gf"], "res": res}}


#: S cut under limits it fits (64 regions, 160 slots, 223 alone), limits few residuals fit (3 regions: one port a part; 1 or
#: 2 slots, alone 1 to 3 or more than packed), S of more residuals than a part holds, and S or limits that refuse
MAX = {"max_regions": st.just(RR.MAX_REGIONS), "max_slots": st.just(RR.MAX_SLOTS), "max_lone_slots": st.just(RR.MAX_LONE_SLOTS)}
PARTS = st.one_of(
    st.fixed_dictionaries({"S": rep_s(), "max_regions": st.sampled_from([3, 4, RR.MAX_REGIONS]),
                           "max_slots": st.sampled_from([1, 2, RR.MAX_SLOTS]),
                           "max_lone_slots": st.sampled_from([0, 1, 2, 3, RR.MAX_LONE_SLOTS])}),
    st.fixed_dictionaries({"S": rep_s(big=True), "max_regions": st.sampled_from([3, RR.MAX_REGIONS]),
                           "max_slots": st.sampled_from([2, RR.MAX_SLOTS]), "max_lone_slots": st.sampled_from([2, 3, RR.MAX_LONE_SLOTS])}),
    _one(st.fixed_dictionaries({"S": _many(), **MAX})),
    _one(st.fixed_dictionaries({"S": rep_s(valid=False) | rep_s(), "max_regions": _near(0, 3), "max_slots": _near(0, 3),
                                "max_lone_slots": _near(0, 3)}),
         st.fixed_dictionaries({"S": rep_s(valid=False), **MAX}), JSON))

REPS = st.one_of(rep_s(), rep_s(valid=False), JSON)
DIGEST = st.text(HEX, min_size=128, max_size=128)
DIGESTS = st.dictionaries(st.text(max_size=12), st.text(HEX, max_size=64), max_size=2) | JSON
#: `compose`'s inputs under bindings that are digests (`test_inner_circuit_agrees` replays ones that are not)
INNER = st.fixed_dictionaries({"S": rep_s(), "name": NAME, "partition": DIGEST, "program": DIGEST,
                               "program_digests": DIGESTS})


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


# --- the algebra's structure and V*'s statements ----------------------------------------------------------------------------------

def test_statements_agree_at_the_smallest_shape(server):
    assert D.replay(port("statements", statements, st.nothing()), {"m": 22, "k_log": 6, "claims": 2}, server)[0] == "output"


@pytest.mark.slow
def test_structure_agrees(server):
    stats = D.run(port("structure", structure, _shapes(14)), examples=32, seed=SEED, server=server)
    assert stats.outputs >= 8 and stats.rejects >= 4, stats


@pytest.mark.slow
def test_spec_agrees(server):
    stats = D.run(port("spec", spec, _shapes(14)), examples=32, seed=SEED, server=server)
    assert stats.outputs >= 8 and stats.rejects >= 4, stats


@pytest.mark.slow
def test_parts_agree(server):
    stats = D.run(port("parts", parts, PARTS), examples=80, seed=SEED, server=server)
    assert stats.outputs >= 20 and stats.rejects >= 10, stats


@pytest.mark.slow
def test_statements_agree(server):
    """Drawn shapes, the universal unit's inner shape (17 residuals, two parts), llama32-1b/b1/i256's 2^29 bucket (16 claims:
    its batched T alone, 172 slots), and the largest whose widest residual stands alone within MAX_LONE_SLOTS (20 claims:
    215 slots; 21 claims' 226 refuse)."""
    p = port("statements", statements, _shapes(14))
    for sh in ({"m": 26, "k_log": 23, "claims": 12}, {"m": 35, "k_log": 27, "claims": 16}, {"m": 35, "k_log": 32, "claims": 20}):
        assert D.replay(p, sh, server)[0] == "output"
    assert D.replay(p, {"m": 35, "k_log": 32, "claims": 21}, server)[0] == "reject"
    stats = D.run(p, examples=32, seed=SEED, server=server)
    assert stats.outputs >= 8 and stats.rejects >= 4, stats


@pytest.mark.slow
def test_the_verifiers_part_nets_are_vstars_units(server):
    """The Lean verifier refuses a `rec-residuals/…` unit unless it is the verifier's own net for that part
    (`Flock.RecResiduals`, PROTOCOL.md §16.10), which it computes from the shape alone; V*'s units must be those nets, or
    the verifier refuses honest recursive sessions. At a shape of one part and one of two (about two minutes: each unit's
    net, built twice, is a few million rows); no cheaper test builds the real parts' nets on both sides."""
    p = port("verifier_parts", verifier_parts, st.nothing())
    for shape in ({"m": 22, "k_log": 6, "claims": 2}, {"m": 28, "k_log": 20, "claims": 4}):
        assert D.replay(p, shape, server)[0] == "output", shape


#: the smallest shape a holo part was refused at (m22-k10-c2), and candidate 3's inner shape, Shape(25, 22, 8)
ALGEBRA = ({"m": 22, "k_log": 10, "claims": 2}, {"m": 25, "k_log": 22, "claims": 8})


@pytest.mark.parametrize("holo", [False, True], ids=["plain", "holo"])
@pytest.mark.parametrize("shape", ALGEBRA, ids=lambda s: f"m{s['m']}-k{s['k_log']}-c{s['claims']}")
def test_the_verifiers_algebra_is_rec_algebras(server, shape, holo):
    """The Lean verifier's own algebra (`Flock.RecAlgebra.structureOf`, `Flock.RecResiduals.part`), from which it builds
    the net a `rec-residuals/…` unit must be, is `rec_residuals.spec`'s and `parts`' byte for byte: the whole S, and each
    part's S (row ports by name), residuals and v elements. A holo shape's h round is a row port after z_partial's, and the
    comb's 64 operands read it (`[v comb_l, w h_l]`)."""
    p = port("verifier_algebra", verifier_algebra, st.nothing())
    assert D.replay(p, {**shape, "holo": holo, "units": False}, server)[0] == "output"


@pytest.mark.slow
@pytest.mark.parametrize("shape", ALGEBRA, ids=lambda s: f"m{s['m']}-k{s['k_log']}-c{s['claims']}")
def test_the_verifiers_holo_part_nets_are_rec_residuals_units(server, shape):
    """Each holo part's net, as `RecResiduals.check` builds it, is `rec_residuals.lowering`'s unit for that part: its text's
    SHA-256, bytes and rows; `test_the_verifiers_holo_part_nets_are_vstars_units` holds V*'s holo units to the same nets.
    No cheaper test builds a holo part's net on both sides."""
    p = port("verifier_algebra", verifier_algebra, st.nothing())
    assert D.replay(p, {**shape, "holo": True, "units": True}, server)[0] == "output"


# --- holo inner sessions: V*'s statements over an inner session whose circuits are private ---------------------------------------

@pytest.mark.parametrize("shape", ALGEBRA, ids=lambda s: f"m{s['m']}-k{s['k_log']}-c{s['claims']}")
def test_holo_statements_agree(server, shape):
    """V*'s algebra and statements over a holo inner session (`VStar.Algebra.specHolo`, `VStar.statementsHolo`) are
    `rec_residuals.spec`'s and `rec_vstage.stage`'s at ``Shape(..., holo=True)``: h's row port after z_partial's, every later
    w element 64 places on, the comb's operands reading h (`[v comb_l, w h_l]`), and each part's S, residuals and v."""
    assert D.replay(port("spec_holo", spec_holo, st.nothing()), shape, server)[0] == "output"
    assert D.replay(port("statements_holo", statements_holo, st.nothing()), shape, server)[0] == "output"


@pytest.mark.slow
def test_holo_statements_agree_on_drawn_shapes(server):
    for fn, ref in (("spec_holo", spec_holo), ("statements_holo", statements_holo)):
        stats = D.run(port(fn, ref, _shapes(14)), examples=24, seed=SEED, server=server)
        assert stats.outputs >= 6 and stats.rejects >= 3, (fn, stats)


@pytest.mark.slow
def test_inner_holo_circuit_agrees(server):
    """`compose` with META `holo` (`rec_residuals.statement(..., holo=True)`) on small units under drawn bindings, and
    replayed with names outside ASCII, past the BMP included: `VStar.withMeta` reads META back, so a name it could not
    write again as it was would be refused."""
    p = port("inner_circuit_holo", inner_circuit_holo, INNER)
    good = {"S": {"gf": {"nv": 1, "nw": 1, "ops": [], "res": []}, "ports": [["row0", 1]]}, "name": "row0",
            "partition": "a" * 128, "program": "b" * 128, "program_digests": {}}
    for name in ("row0", "rec-residuals/ü", "\U0001f600\t\"\\"):
        assert D.replay(p, {**good, "name": name, "program_digests": {name: "e" * 64}}, server)[0] == "output", name
    stats = D.run(p, examples=5, seed=SEED, server=server)
    assert stats.outputs >= 2, stats


@pytest.mark.slow
@pytest.mark.parametrize("shape", ALGEBRA, ids=lambda s: f"m{s['m']}-k{s['k_log']}-c{s['claims']}")
def test_the_verifiers_holo_part_nets_are_vstars_units(server, shape):
    """V*'s units over a holo inner session are the nets the Lean verifier holds a `rec-residuals/…` unit to when its
    statement's META says `holo` (`Flock.RecResiduals.check`); no cheaper test builds both."""
    assert D.replay(port("verifier_parts_holo", verifier_parts_holo, st.nothing()), shape, server)[0] == "output"


# --- the units ------------------------------------------------------------------------------------------------------------------

#: H = 0 and 17 take a minute to lower, so `test_rec_open_unit_agrees` replays them once rather than drawing them.
REC_OPEN = st.one_of(st.just({"LANES": 8, "H": 1}),
                     st.fixed_dictionaries({"LANES": st.sampled_from([0, 4, 7, 9, 12, -8, True]), "H": st.just(1)}),
                     st.fixed_dictionaries({"LANES": st.just(8), "H": st.sampled_from([33, -1, True])}),
                     st.dictionaries(st.sampled_from(["LANES", "H", "x"]), st.just(7), max_size=3), JSON)


@pytest.mark.slow
def test_inner_unit_agrees(server):
    """Drawn, and replayed on an S whose GfResiduals S is in range but whose row ports repeat a name (`checkRepS`'s refusal)
    and on one whose ports are right but whose add term is past w (`checkS`'s): Lean's `residualForms` reads such a term as
    zero, so without its own check `innerUnit` would emit a unit where `unit_circuit` refuses. A drawn invalid S is
    usually refused by `checkRepS` first, so the draws alone rarely reach `checkS`."""
    p = port("inner_unit", inner_unit, st.fixed_dictionaries({"S": REPS, "name": NAME}) | JSON)
    gf = lambda t: {"nw": 2, "nv": 0, "ops": [], "res": [{"mul": [], "add": [t]}]}
    good = {"S": {"gf": gf(["w", 1]), "ports": [["a", 2]]}, "name": "row0"}
    assert D.replay(p, good, server)[0] == "output"
    for S in ({"gf": gf(["w", 1]), "ports": [["a", 1], ["a", 1]]}, {"gf": gf(["w", 2]), "ports": [["a", 2]]}):
        assert D.replay(p, {"S": S, "name": "row0"}, server)[0] == "reject", S
    stats = D.run(p, examples=40, seed=SEED, server=server)
    assert stats.outputs >= 5 and stats.rejects >= 5, stats


@pytest.mark.slow
def test_rec_open_unit_agrees(server):
    """`RecOpen{LANES, H}`'s lowering, drawn at H = 1 and at parameters both refuse, and replayed at the climb's edges: H = 0
    (`aux` one row the unit does not read) and 17 (the first height whose side bits are `dirs`' word 1). Lowering H = 17
    takes about a minute; no cheaper test compares V*'s builder with `rec_open` at those heights."""
    p = port("rec_open_unit", rec_open_unit, REC_OPEN)
    for h in (0, 17):
        assert D.replay(p, {"LANES": 8, "H": h}, server)[0] == "output", h
    stats = D.run(p, examples=30, seed=SEED, server=server)
    assert stats.outputs >= 1 and stats.rejects >= 10, stats


# --- the circuit files ----------------------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_inner_circuit_agrees(server):
    """`compose` on small InnerRepCheck units under drawn bindings: META, the zero instance's outputs and `b ‖ c`, the slot
    ranges and the block size. A binding that is not a digest is replayed in each place, not drawn: drawn, it refused
    most of the five units, and which ones moved with the source of modules the reference loads (Hypothesis draws their
    constants too)."""
    p = port("inner_circuit", inner_circuit, INNER)
    good = {"S": {"gf": {"nv": 1, "nw": 1, "ops": [], "res": []}, "ports": [["row0", 1]]}, "name": "row0",
            "partition": "a" * 128, "program": "b" * 128, "program_digests": {}}
    assert D.replay(p, good, server)[0] == "output"
    for key in ("partition", "program"):
        for bad in ("A" * 128, "a" * 127, "a" * 129, "g" * 128):
            assert D.replay(p, {**good, key: bad}, server)[0] == "reject", (key, bad)
    stats = D.run(p, examples=5, seed=SEED, server=server)
    assert stats.outputs >= 2 and stats.rejects >= 1, stats


@pytest.mark.slow
def test_rec_open_circuit_refuses_as_compose_does(server):
    """RecOpen's file is refused where `compose` refuses it (its unit's parameters, a key missing, a binding that is not a
    SHA-512 digest); `test_rec_open_file_is_byte_equal` compares a file it writes."""
    p = port("rec_open_circuit", rec_open_circuit, st.nothing())
    good = {"LANES": 8, "H": 1, "partition": "a" * 128, "program": "b" * 128, "program_digests": {}}
    for bad in ({**good, "LANES": 7}, {**good, "LANES": True}, {**good, "H": -1}, {**good, "H": 33},
                {k: v for k, v in good.items() if k != "program"}, {**good, "x": 0}, {**good, "partition": "A" * 128}):
        assert D.replay(p, bad, server)[0] == "reject", bad


# --- whole files, byte for byte -------------------------------------------------------------------------------------------------

def _vstar(tmp_path: Path, *args, flags: tuple[str, ...] = ()) -> bytes:
    exe = D.build(LEAN, "vstar")
    out = tmp_path / "circuit.txt"
    p = subprocess.run([str(exe), *args, str(out), *flags], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    data, said = out.read_bytes(), json.loads(p.stdout)
    assert (said["sha512"], said["bytes"]) == (hashlib.sha512(data).hexdigest(), len(data))
    return data


#: `rec_outer.subcircuit(8, 1)`'s program digest (core's codec), and the circuit file `compose` writes under it with the
#: bindings `a…a` and `b…b`.
REC_OPEN_8_1 = {"RecOpen_v4{LANES=8,H=1}": "cb8c446680677966b08cdf47d09426438abcf47e3ac7cff5f76cd0225ef157ce"}
REC_OPEN_8_1_FILE = ("1087318c40a3a8da1994bd1eae6c836ea7025bc17878cf187cf7aad290ad4bfe"
                     "afb71f00d0e0802270e76a47c5ef50213a02a6f5252151f654fbdc5c4ee661ce", 92449957)


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


@pytest.mark.slow
@pytest.mark.parametrize("shape", ALGEBRA, ids=lambda s: f"m{s['m']}-k{s['k_log']}-c{s['claims']}")
def test_holo_part_files_are_byte_equal(tmp_path, shape):
    """Each algebra part's circuit file over a holo inner session, from `vstar inner S NAME BINDINGS OUT --holo` on the S
    `vstar statements --holo` gives, against `rec_residuals.statement(p.S, name, 2, holo=True)`'s text under that
    statement's bindings, byte for byte: the file the verifier compares with V*'s own (`RecVStar.checkFile`) once it takes
    a private inner session. A few minutes a shape (Python lowers each part, a few million rows); no cheaper test writes a
    real holo part's whole file on both sides."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P

    sh = A.Shape(shape["m"], shape["k_log"], shape["claims"], holo=True)
    p = subprocess.run([str(D.build(LEAN, "vstar")), "statements", *map(str, (sh.m, sh.k_log, sh.claims)), "--holo"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    lean = [s for s in json.loads(p.stdout) if s["template"] == "rec-residuals"]
    ps = RR.parts(RR.spec(sh))
    assert [(s["S"], s["res"], s["v"]) for s in lean] == [(q.S, q.res, q.v) for q in ps]
    for s in lean:
        st_ = RR.statement(s["S"], s["name"], 2, holo=True)
        b = {"partition": P.digest(st_.part), "program": st_.prog, "program_digests": FC.program_digests(st_.sub)}
        (tmp_path / "s.json").write_text(json.dumps(s["S"]))
        (tmp_path / "b.json").write_text(json.dumps(b))
        data = _vstar(tmp_path, "inner", str(tmp_path / "s.json"), s["name"], str(tmp_path / "b.json"), flags=("--holo",))
        assert json.loads(data.split(b"\n", 2)[1].removeprefix(b"META "))[RR.HOLO_KEY] is True
        assert data == st_.text.encode(), s["name"]
        del data, st_


def _stage(sh: A.Shape) -> list[tuple[str, dict, object]]:
    """`rec_vstage.stage`'s statements over an inner session of shape ``sh``: each one's directory, its bindings, and a
    function writing its circuit as `stage` does. A level's bindings and text are `rec_outer.stage`'s (its Program over the
    level's ``2 q`` openings, the template-instance partition of it, `compose` of the level's lowering); a part's text is
    `rec_residuals.statement`'s, which computes its bindings again (the test checks they are these)."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P

    def bindings(defn, n: int, sub) -> dict:
        program = P.population_program(defn, n)
        return {"partition": P.digest(P.partition(program, P.template_instance_query(defn))),
                "program": P.program_sha512(program), "program_digests": FC.program_digests(sub)}

    out = []
    for level, lv in enumerate(A.fast100(sh.m).levels):
        L, H, n = 1 << lv.lanes, lv.d - lv.cap, 2 * lv.q
        b = bindings(R.definition(L, H), n, O.subcircuit(L, H))
        out.append((f"L{level}", b, lambda L=L, H=H, b=b: FC.compose(O.lowering(L, H), O.subcircuit(L, H),
                                                                       partition=b["partition"], program=b["program"])))
    for j, p in enumerate(RR.parts(RR.spec(sh))):
        name = f"rec-residuals/m{sh.m}-k{sh.k_log}-c{sh.claims}-p{j}"
        b = bindings(RR.definition(p.S), 2, RR.subcircuit(p.S, name))

        def text(S=p.S, name=name, b=b) -> str:
            st = RR.statement(S, name, 2)
            assert (P.digest(st.part), st.prog) == (b["partition"], b["program"])
            return st.text
        out.append((f"alg/p{j}", b, text))
    return out


@pytest.mark.slow
@pytest.mark.skipif(not STAGE, reason="V*'s whole stage, byte for byte (VERITY_VSTAR_STAGE=M,K_LOG,CLAIMS[;...]): Python's "
                                      "builders take tens of minutes and gigabytes a shape, a CPU pod's job")
@pytest.mark.parametrize("shape", [tuple(map(int, x.split(","))) for x in STAGE.split(";") if x] or [None],
                         ids=lambda s: "m{}-k{}-c{}".format(*s) if s else "none")
def test_stage_is_byte_equal(tmp_path, shape):
    """Every statement of V* over an inner session of ``shape``, from `vstar stage`, against the file `rec_vstage.stage`
    writes for it (`_stage`), byte for byte. With ``VERITY_VSTAR_STAGE_OUT`` set, each statement's digests go to
    ``stage-M-K-C.json`` there."""
    sh = A.Shape(*shape)
    stage = _stage(sh)
    (tmp_path / "b.json").write_text(json.dumps([b for _, b, _ in stage]))
    p = subprocess.run([str(D.build(LEAN, "vstar")), "stage", *map(str, shape), str(tmp_path / "b.json"), str(tmp_path / "st")],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    lean = json.loads(p.stdout)
    assert [s["dir"] for s in lean] == [d for d, _, _ in stage]
    record = []
    for s, (d, b, text) in zip(lean, stage, strict=True):
        data = (tmp_path / "st" / d / "circuit.txt").read_bytes()
        assert (s["sha512"], s["bytes"]) == (hashlib.sha512(data).hexdigest(), len(data))
        python = text().encode()
        record.append({**s, "python_sha512": hashlib.sha512(python).hexdigest(), "python_bytes": len(python),
                       "equal": data == python, "partition": b["partition"], "program": b["program"]})
        del data, python
    if out := os.environ.get("VERITY_VSTAR_STAGE_OUT"):
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / f"stage-{'-'.join(map(str, shape))}.json").write_text(json.dumps(record, indent=1))
    assert all(r["equal"] for r in record), [r["dir"] for r in record if not r["equal"]]
