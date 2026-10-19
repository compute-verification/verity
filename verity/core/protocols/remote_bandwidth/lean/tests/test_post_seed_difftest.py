"""The post-seed verdict in Lean (`remote-audit-verdict`, `RemoteAudit.PostSeed`) against the Python that mirrors it:
property/v1 (`property_check`: `verity.ml.circuits.stacking.check` on the Program's own census), the record's windows and
lifetime cap read in census steps (`property_reading`: `audit.property_reading`), the whole verdict with its bindings
(`post_seed`: `verity.ml.circuits.post_seed.verdict` after that reading), and the table (`class_of`: `stacking.class_of`,
`stacking.gather` and the table's digest).

The Lean reads each Program as its descriptor (`codec.encode_program`), the Python as the `Program` that descriptor encodes,
found by its digest. The corpus is every Program `verity/ml/circuits/tests/test_stacking.py` builds (its exact int8 Program
the remote audit's writers among them) and every property red-team case (`benchmarks/property_redteam`), each under
windows, c, R_in, L and B that pass it and that fail it, and the inputs both sides must refuse. A form the Lean doesn't
cover yet it refuses by name where the Python judges it (`REFUSED_FORMS`).

`audit.post_seed`, the verifier's call, runs the Lean on a recorded session (`test_appends.honest_session`), and `ledger`
holds what it returns to the record (binding 4)."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.core.protocols.remote_bandwidth import audit, lean_verdict
from verity.ml.circuits import defs, post_seed
from verity.ml.circuits import ports as ir_ports
from verity.ml.circuits import stacking as P
from verity.ml.circuits.census import census
from verity.ml.circuits.codec import encode_program, program_digest
from verity.ml.circuits.ops import prims
from verity.ml.circuits.ports import In, Out, Ports
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import array_of, tuple_of
from verity.ml.circuits.types import Array, Value

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
ROOT = HERE.parents[5]
EXE = lean_verdict.EXE
#: What the Python raises where the Lean refuses: a decoder's `Reject`, and the references' `ValueError` and `IndexError`
#: (`map_value_leaves` on a public row whose `pos` is short)
REFUSALS = (D.Reject, ValueError, IndexError)


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: `test_stacking.py`'s Programs and stand-ins
S = _module("stacking_programs", ROOT / "verity/ml/circuits/tests/test_stacking.py")


@functools.cache
def redteam() -> Any:
    sys.path.insert(0, str(ROOT / "benchmarks/property_redteam"))
    import cases
    return cases


@contextlib.contextmanager
def unchecked() -> Any:
    """Programs built past the IR's build-time refusals of a word read at another width (`E_REINTERPRET`) and of a run
    input's declared bound (`E_PORT_BOUND`), where the IR has them: the Lean reads whatever descriptor it is sent, so it is
    held to the Python on those Programs too, as `test_stacking` holds the property past the first."""
    with pytest.MonkeyPatch.context() as m:
        if hasattr(defs, "_Widths"):
            m.setattr(defs._Widths, "first", lambda self, c: None)
        if hasattr(ir_ports, "_is_harness"):
            m.setattr(ir_ports, "_is_harness", lambda fn: True)
        yield


@functools.cache
def appends() -> Any:
    sys.path.append(str(ROOT / "verity/core/protocols/remote_bandwidth/tests"))
    import test_appends
    return test_appends


# --- the Programs, by descriptor --------------------------------------------------------------------------------------

_ENCODED: dict[int, tuple[Any, dict, str]] = {}
#: each encoded Program by its descriptor's digest, which the references read the Lean's input by
PROGRAMS: dict[str, Any] = {}


def encoded(program: Any) -> tuple[dict, str]:
    """`program`'s descriptor and digest, kept so the references find `program` by its descriptor."""
    if id(program) not in _ENCODED:
        desc = encode_program(program)
        digest = program_digest(desc)
        _ENCODED[id(program)] = (program, desc, digest)
        PROGRAMS.setdefault(digest, program)
    return _ENCODED[id(program)][1:]


# --- decoding, as `RemoteAudit.PostSeed` decodes ----------------------------------------------------------------------

def _get(j: Any, k: str) -> Any:
    if not isinstance(j, dict) or k not in j:
        raise D.Reject(f"{k} missing")
    return j[k]


def _list(v: Any, k: str) -> list:
    if not isinstance(v, list):
        raise D.Reject(f"{k}: not a list")
    return v


def _int(v: Any, k: str) -> int:
    if type(v) is not int:
        raise D.Reject(f"{k}: not an integer")
    return v


def _nat(v: Any, k: str) -> int:
    if _int(v, k) < 0:
        raise D.Reject(f"{k}: not a natural number")
    return v


def _bool(v: Any, k: str) -> bool:
    if type(v) is not bool:
        raise D.Reject(f"{k}: not a boolean")
    return v


def _str(v: Any, k: str) -> str:
    if not isinstance(v, str):
        raise D.Reject(f"{k}: not a string")
    return v


def _hex(v: Any, k: str) -> bytes:
    if not re.fullmatch(r"(?:[0-9a-f]{2})*", _str(v, k)):
        raise D.Reject(f"{k}: not lowercase hex")
    return bytes.fromhex(v)


def _frac(v: Any, k: str) -> Fraction:
    """`Q.ofJson`: a JSON integer, or a string `str(Fraction)` writes (an integer, or n/d with d > 0)."""
    if type(v) is int:
        return Fraction(v)
    if isinstance(v, str):
        parts = v.split("/")
        if len(parts) <= 2 and all(re.fullmatch(r"-?[0-9]+", p) for p in parts):
            n, d = int(parts[0]), int(parts[1]) if len(parts) == 2 else 1
            if d > 0:
                return Fraction(n, d)
    raise D.Reject(f"{k}: not an integer or a fraction")


def _c(v: Any) -> dict[str, Fraction]:
    if not isinstance(v, dict) or len(v) != 4:
        raise D.Reject("c gives units per certified byte for exactly the four budgeted classes")
    return {k: _frac(_get(v, k), f"c.{k}") for k in P.BUDGETED}


def _program(v: Any) -> tuple[Any, str]:
    desc = _get(v, "program")
    if not isinstance(desc, dict):
        raise D.Reject("program: not a descriptor")
    digest = program_digest(desc)
    if digest not in PROGRAMS:
        raise D.Reject("program: a descriptor no Program here encodes")
    return PROGRAMS[digest], digest


def _ends(v: Any) -> tuple[int, ...]:
    return tuple(_nat(e, "ends") for e in _list(_get(v, "ends"), "ends"))


def _record(v: Any) -> SimpleNamespace:
    """What `audit.property_reading` reads of a `Record`: its windows' steps, receipts and spans, its block size and cap."""
    wins = [SimpleNamespace(steps=tuple(_int(s, "window.steps") for s in _list(_get(w, "steps"), "steps")),
                            receipts=_nat(_get(w, "receipts"), "receipts"), span_ns=_nat(_get(w, "span_ns"), "span_ns"))
            for w in _list(_get(v, "windows"), "windows")]
    return SimpleNamespace(windows=wins, block_bytes=_nat(_get(v, "block_bytes"), "block_bytes"),
                           lifetime=_nat(_get(v, "lifetime"), "lifetime"))


def _row(row: Any) -> dict[str, bytes]:
    if not isinstance(row, dict):
        raise D.Reject("map_rows: a unit's rows are an object")
    return {r: _hex(x, f"map_rows.{r}") for r, x in row.items()}


def _windows_json(windows: Any) -> list:
    return [[w.start, w.stop, w.reads, str(w.seconds)] for w in windows]


# --- the references --------------------------------------------------------------------------------------------------

@functools.cache
def analysis(digest: str, ends: tuple[int, ...]) -> P.Analysis:
    program = PROGRAMS[digest]
    return P.analyze(program, census(program, list(ends), digest))


def property_check(v: Any) -> dict:
    """`stacking.check` (`judge` of `analyze`) on the Program's census at `ends`."""
    _program_, digest = _program(v)
    a = analysis(digest, _ends(v))
    windows = []
    for w in _list(_get(v, "windows"), "windows"):
        if not isinstance(w, list) or len(w) != 4:
            raise D.Reject("a window is [start, stop, reads, seconds]")
        windows.append(P.Window(range(_int(w[0], "window"), _int(w[1], "window")), _nat(w[2], "window"),
                                _frac(w[3], "window")))
    L = _get(v, "L")
    return P.judge(a, c=_c(_get(v, "c")), windows=windows, R_in=_frac(_get(v, "R_in"), "R_in"),
                   L=None if L is None else _nat(L, "L"), B=_nat(_get(v, "B"), "B")).to_json()


def property_reading(v: Any) -> dict:
    windows, L = audit.property_reading(_record(v), _nat(_get(v, "census_steps"), "census_steps"),
                                        paced=_bool(_get(v, "paced"), "paced"))
    return {"windows": _windows_json(windows), "L": L}


def verdict(v: Any) -> dict:
    """`audit.property_reading` of the record, then `post_seed.verdict` at what it reads."""
    program, _digest = _program(v)
    ends = _ends(v)
    windows, L = audit.property_reading(_record(v), len(ends), paced=_bool(_get(v, "paced"), "paced"))
    mp = _get(v, "map")
    statics = SimpleNamespace(**{k: _nat(_get(mp, k), f"map.{k}") for k in "CKV"})
    pubs = [{"first": p.get("first", 0), "count": _nat(_get(p, "count"), "count"),
             "present": _nat(_get(p, "present"), "present"),
             "pos": [_nat(x, "pos") for x in _list(_get(p, "pos"), "pos")]}
            for p in _list(_get(v, "map_public"), "map_public")]
    rows = [_row(row) for row in _list(_get(v, "map_rows"), "map_rows")]
    out = post_seed.verdict(
        program, ends, audited=_str(_get(v, "audited"), "audited"),
        values=[_hex(x, "values") for x in _list(_get(v, "values"), "values")], map_statics=statics, map_public=pubs,
        map_rows=rows, mapped=_bool(_get(v, "mapped"), "mapped"), c=_c(_get(v, "c")),
        windows=[P.Window(range(w.start, w.stop), w.reads, w.seconds) for w in windows], L=L,
        R_in=_frac(_get(v, "R_in"), "R_in"), B=_nat(_get(v, "B"), "B"))
    return {**out, "census_steps": len(ends), "L": L, "windows": _windows_json(windows)}


def table(v: Any) -> dict:
    ids = [_str(f, "ids") for f in _list(_get(v, "ids"), "ids")]
    return {"digest": P.digest(), "classes": [P.class_of(f) for f in ids],
            "gathers": [list(g) if (g := P.gather(f)) is not None else None for f in ids]}


def port(fn: str, reference: Any, inputs: Any = st.nothing(), **kw: Any) -> D.Port:
    return D.Port(EXE, fn, reference, inputs, package=PACKAGE, rejects=REFUSALS, **kw)


CHECK = port("property_check", property_check)
READING_PORT = "property_reading"
POST_SEED = port("post_seed", verdict)

server = D.server_fixture(CHECK)


# --- the corpus ------------------------------------------------------------------------------------------------------

FREE = {k: 0 for k in P.BUDGETED}
RICH = {k: "1000000" for k in P.BUDGETED}
LG3 = S.stand_in("LGatherRow16x3", (("idx", Value(33)), ("rows", Array(3, Value(17)))), Value(17))
LG6 = S.stand_in("LGatherRow16x6", (("idx", Value(33)), ("rows", Array(6, Value(17)))), Value(17))
G1 = S.stand_in("GatherBf16x1", (("i", S.S32), ("col", Array(1, S.S16))), S.S16)
TR = S.stand_in("TableRead4x1", (("index", S.S32), ("table", Array(1, Array(64, S.S16)))), S.S16)


def _whole(fn: Any, registered: tuple[str, ...] = ()) -> tuple[Program, list[int]]:
    program = Program(fn, registered=registered)
    return program, [len(program.root.body.nodes)]


def _table() -> tuple[Program, list[int]]:
    def body(B, _S, i):
        return B.call(S.GATHER128, i, array_of([B.call(prims.const(32, (k * k + 7 * k) % 251)) for k in range(128)]))
    return _whole(S.comp("TestTable", (("i", S.S32),), S.S32, body))


def _two_kinds() -> tuple[Program, list[int]]:
    fn = S.comp("TestTwoKinds", (("i", S.S32), ("t", Array(1, Array(64, S.S16))), ("col", Array(16, S.S16))),
                S.TupleT((S.S16, S.S16, S.S16)),
                lambda B, _S, i, t, col: tuple_of(B.call(TR, i, t), B.call(S.GATHER, i, col), B.call(S.GATHER, i, col)))
    return _whole(fn, ("t", "col"))


def _gather_tree() -> tuple[Program, list[int]]:
    def body(B, _S, i, j):
        leaves = [B.call(S.GATHER, i, array_of([B.call(prims.const(16, (k * 31 + r * 7) % 65521)) for r in range(16)]))
                  for k in range(16)]
        return B.call(S.GATHER, j, array_of(leaves))
    return _whole(S.comp("TestGatherTree", (("i", S.S32), ("j", S.S32)), S.S16, body))


def _bit_at_5() -> tuple[Program, list[int]]:
    return _whole(S.comp("TestBitAt5", (("word", S.S32),), S.BIT,
                         lambda B, _S, word: B.call(S.BITAT, word, B.call(prims.const(32, 5)))))


def _lut_sum() -> tuple[Program, list[int]]:
    def body(B, _S, t):
        total = B.call(S.EXP2LUT, t[0])
        for j in range(1, 4):
            total = B.call(S.scalar.I32Add, total, B.call(S.EXP2LUT, t[j]))
        return total
    return _whole(S.comp("TestLutSum", (("t", Array(4, S.S8)),), S.S32, body))


def _from(label: str, start: Any, ports: Any = None) -> tuple[Program, list[int]]:
    """`test_stacking._from`'s Program: two s8 steps from the accumulator `start` makes of the run input c."""
    fn = S.comp(f"TestFrom{label}", (("c", Array(2, S.S32)), ("x", Array(64, S.S8)), ("w", Array(64, S.S8))), S.S32,
                lambda B, _S, c, x, w: B.call(S._step(64), start(B, c), x, w), ports=ports)
    return _whole(fn, ("w",))


@functools.cache
def corpus() -> tuple[tuple[str, Any, list[int]], ...]:
    """`test_stacking.py`'s Programs, each with its step ends."""
    out = [("int8", *S.int8_program()), ("int8-writers", *S.int8_program(S=5)),
           ("int8-epilogue", *S.int8_program(S=5, epilogue=True)), ("int8-k96", *S.int8_program(S=2, K=96)),
           ("int8-wrap", *S.int8_program(mmas=[S.IMMA, S.WRAP, S.IMMA])),
           ("int8-qmma", *S.int8_program(mmas=[S.IMMA, S.QMMA, S.IMMA])),
           ("int8-u8s8", *S.int8_program(mmas=[S.U8S8] * 3)), ("int8-acc-input", *S.int8_program(acc0="input")),
           ("int8-reread", *S.int8_program(reread=(2, 0))), ("scan-1", *S._scan_program(1)),
           ("scan-4", *S._scan_program(4)), ("carry", *S._carry_program()),
           ("compare-split", *S._compare_split_program()), ("table", *_table()), ("two-kinds", *_two_kinds()),
           ("gather-tree", *_gather_tree()), ("bit-at-5", *_bit_at_5()), ("lut-sum", *_lut_sum()),
           ("from-undeclared", *_from("Undeclared", lambda B, c: B.call(S.I32DIV, c[0], c[1])))]
    with unchecked():
        out += [(f"reinterpret-{how}", *S._reinterpreting_program(how)) for how in ("broadcast", "affine")]
    out += [(f"gather-{how}", *S._gather_program(how)) for how in ("same", "two", "mapped")]
    out += [(f"one-{prim.name}-{'-'.join(reg) or 'run'}", *S._one_gate_program(prim, reg)) for prim, reg in (
        (S.GATHER, ()), (S.GATHER, ("col",)), (S.GATHER, ("i",)), (G1, ("col",)), (LG3, ("idx",)), (LG3, ("rows",)),
        (LG6, ("rows",)), (S.BITAT, ("word",)), (S.BITAT, ("i",)), (S.NCPDOT, ("b",)), (S.SQACC, ()),
        (S.F64MUL, ()), (S.TABLE, ("table",)), (S.U32GE, ()), (S.I32MUL, ()), (S.I32SUB, ()),
        (prims.Bf16ToF32, ()), (S.scalar.F32Fmaxf, ()), (S.RZBYTE, ()))]
    out += [(f"pv-{K}-{steps}", *_whole(S._pv(K, steps), ("v",))) for K, steps in ((64, "maxmin"), (96, "max"), (64, "min"))]
    out += [(f"lut-pv-{K}", *_whole(S._lut_pv(K))) for K in (132_096, 132_128)]
    return tuple(out)


#: Programs the Lean refuses by name, a form it doesn't cover yet, with the reason it gives; the Python judges each. A root
#: Call whose callee returns its argument, or a view of it, would need the census's aliases followed through the root.
REFUSED_FORMS = {
    "from-silent": "may pass an argument through",
    "reinterpret-ret": "may pass an argument through",
    "reinterpret-mixed": "may pass an argument through",
    "from-declared": "declares ports",
    "from-transfer": "declares ports",
    "from-composite": "declares ports",
}


#: `test_stacking`'s range check on an s32 word, which a composite's declared return over a run input reads it through
RANGE = S.stand_in("TestRangeS32", (("a", S.S32),), ports=Ports(ret=Out(interval=S.BOUND)))


@functools.cache
def refused_forms() -> dict[str, tuple[Program, list[int]]]:
    with unchecked():
        ret, mixed = S._reinterpreting_program("ret"), S._reinterpreting_program("mixed")
    return {"from-silent": _from("Silent", lambda B, c: B.call(S.SILENT, c)),
            "reinterpret-ret": ret,
            "reinterpret-mixed": mixed,
            "from-declared": _from("Declared", lambda B, c: B.call(S.DIV, c[0], c[1])),
            "from-transfer": _from("Transfer", lambda B, c: B.call(S.REM, c[0], B.call(prims.const(32, 1000)))),
            "from-composite": _from("Composite", lambda B, c: B.call(S.FIRST, B.batch(RANGE, c, axes=(0,))))}


def checks(program: Any, ends: list[int]) -> list[dict]:
    """property/v1's inputs on `program`: free classes under one window of plenty; plenty of every class under a window a
    step at L = 0; a rounding per 32 bytes with the last step uncovered at L = 3; and one window of 1 ms at B = 1."""
    desc, _digest = encoded(program)
    n = len(ends)
    per = [[s, s + 1, 64, 1] for s in range(n)]
    base = {"program": desc, "ends": list(ends), "B": P.OPERAND_BITS, "R_in": "1000000000", "L": None}
    return [{**base, "c": FREE, "windows": [[0, n, 10**9, 1]]},
            {**base, "c": RICH, "windows": per, "L": 0},
            {**base, "c": {**FREE, "roundings": "1/32"}, "windows": per[:-1] or per, "R_in": "100000", "L": 3},
            {**base, "c": RICH, "windows": [[0, n, 64, "1/1000"]], "R_in": "100000", "B": 1}]


# --- the table -------------------------------------------------------------------------------------------------------

LISTED = sorted(P.CLASSES)
ODD_IDS = [
    "GatherBf16x16_v1", "GatherF32x128_v1", "GatherI32x64_v1", "GatherE4m3x7_v1", "GatherE2m1x1_v1", "GatherE8m0x3_v1",
    "GatherBf16x16_v2", "GatherBf16x0_v1", "GatherBf16x128256_v1", "TableRead16x4_v1", "TableRead8x0_v1",
    "LGatherRow16x5_v1", "LGatherRow32x8o7fc00000_v1", "LGatherRow16x5oXY_v1", "Exp2Lut127_v1", "Exp2Lut127_v2",
    "BitAtx32_v1", "BitAtx0_v1", "And2_v1", "And3_v1", "And1_v1", "And10_v1", "Xor17_v1", "Xor01_v1", "And_v1",
    "TopPMaskWordx151936_v1", "LSelect32_v1", "LSelect_v1", "LFixedSelect3x32_v1", "LIsActive4_v1", "LIsEOS2_v1",
    "LStepLive9_v1", "LEnable_v1", "LEmitNext[H=4]_v1", "LEmitNext[H=0]_v1", "SplitsForSMS4_v1", "LSplitsForSMS4_v1",
    "LiveCount8_v1", "LBot32_v1", "LBot_v1", "Const8[0x7f]_v1", "Const32[0x00000000]_v1", "Const0[0x0]_v1",
    "Const8[0x7F]_v1", "Const8[0x]_v1", "Lifted[F32Add_v1]_v2", "Lifted[I32Add_v1]_v2", "Lifted[I32Add_v1]_v1",
    "Lifted[Const32[0x00000000]_v1]_v2", "Lifted[BlackwellS8ImmaDot32_v1]_v2", "Lifted[LEmitNext[H=4]_v1]_v2{ORD=1}",
    "Lifted[LGatherRow16x5_v1]_v2", "Lifted[GatherBf16x16_v1]_v2", "Lifted[NoSuchPrimitive_v1]_v2",
    "Lifted[Lifted[Bf16ToF32_v1]_v2]_v1", "F32Add_v1{ORD=3}", "F32Add_v1{ORD=}", "Lifted[]_v2", "Lifted[F32Add_v1]_v3",
    "I32Add_v2", "Zero_v1", "NoSuchPrimitive_v1", "", "Input_v1", "RegisteredInput_v1", "PouwDuoAbsorb_v1",
]


def test_the_table_is_the_python_s_id_for_id(server) -> None:
    """Every id the table lists, the families' instances and their edges, the lifts (nested, of a free primitive, of an
    unlisted one) and ids no class takes, each with its gather reading, and the table's digest."""
    D.replay(port("class_of", table), {"ids": LISTED + ODD_IDS}, server)
    ids = [f for _label, program, ends in corpus() for f in census(program, ends)["classes"]]
    D.replay(port("class_of", table), {"ids": sorted(set(ids))}, server)


_FRAGMENTS = st.sampled_from(["Gather", "Bf16", "F32", "x", "TableRead", "LGatherRow", "o", "7f", "BitAtx", "And", "Xor",
                              "LSelect", "LiveCount", "LBot", "Const", "[0x", "]", "{ORD=", "}", "Lifted[", "]_v2", "_v1",
                              "_v2", "1", "0", "16", "33", "Exp2Lut127", "I32Add", "F32Fma", "LEmitNext[H=", "=", ""])


def test_the_table_agrees_on_drawn_ids(server) -> None:
    """Ids spliced from the patterns' pieces, most of them near misses of a family, a gather or a lift."""
    ids = st.lists(st.one_of(st.sampled_from(LISTED + ODD_IDS), st.lists(_FRAGMENTS, max_size=6).map("".join)), max_size=8)
    D.run(port("class_of", table, st.fixed_dictionaries({"ids": ids}), need_rejects=False), examples=300, server=server)


# --- property/v1 -----------------------------------------------------------------------------------------------------

def test_property_v1_agrees_on_test_stacking_s_programs(server) -> None:
    """Each Program under the windows, c, L and B of `checks`, which pass some and fail others for each reason."""
    seen = set()
    for label, program, ends in corpus():
        for v in checks(program, ends):
            out = D.replay(CHECK, v, server)
            seen.add((label, out[0]))
    assert {kind for _label, kind in seen} == {"output"}


def test_property_v1_agrees_on_the_red_team_s_cases(server) -> None:
    """Each red-team case that builds here, at its own size, under `checks`, at one step."""
    cases = redteam()
    judged = 0
    for case in cases.CASES:
        if not case.available:
            continue
        with unchecked():
            program = case.program()
        for v in checks(program, [len(program.root.body.nodes)]):
            D.replay(CHECK, v, server)
        judged += 1
    assert judged >= 10


def test_forms_the_lean_doesn_t_cover_are_refused_by_name(server) -> None:
    """A port declaration below the root (a declared interval, a transfer, a composite's declared return) and a root Call
    that may pass an argument through: the Python judges each, the Lean refuses it, saying which form."""
    for label, (program, ends) in refused_forms().items():
        v = checks(program, ends)[0]
        assert CHECK.python(v)[0] == "output", label
        kind, reason = server.call("property_check", v)
        assert kind == "reject" and REFUSED_FORMS[label] in reason, (label, kind, reason)


def test_a_root_s_own_port_declaration_is_read_as_the_python_reads_it(server) -> None:
    """A run input's declared bound on the program function itself is not taken on trust (18 Oct), here as in Python: the
    accumulator it feeds is unproven. Where the IR refuses the bound outright (`E_PORT_BOUND`), the Python refuses the Program
    before judging it, and the Lean still judges that accumulator unproven."""
    with unchecked():
        program, ends = S.int8_program(acc0="input", ports=Ports(params={"c": In(bound=S.BOUND)}))
    for v in checks(program, ends):
        kind, why = CHECK.python(v)
        if kind == "reject" and "E_PORT_BOUND" in why:
            kind, out = server.call("property_check", v)
            assert kind == "output", (kind, out)
            verdict = json.loads(out)
            assert not verdict["ok"] and "s8-window" in [r["code"] for r in verdict["reasons"]], verdict
        else:
            D.replay(CHECK, v, server)


def test_property_v1_refuses_what_the_python_refuses(server) -> None:
    """Ends that aren't the Program's steps, windows outside its steps or of no time, a c short of a class or with one
    more, a rational that isn't one, a B below 1, an L below 0, and a descriptor that isn't one."""
    program, ends = S.int8_program()
    good = checks(program, ends)[1]
    bad = [{**good, "ends": e} for e in ([], ends[:-1], ends[::-1], [ends[0] - 2, *ends[1:]], ends + [ends[-1]],
                                         [True, *ends[1:]], [*ends[:-1], ends[-1] + 1])]
    bad += [{**good, "windows": [w]} for w in ([0, 0, 1, 1], [0, 4, 1, 1], [-1, 1, 1, 1], [0, 3, -1, 1], [0, 3, 1, 0],
                                              [0, 3, 1, "-1/2"], [0, 3, 1, "1/0"], [0, 3, 1, "0.5"], [0, 3, 1],
                                              [0, 3, 1, True], [0, "3", 1, 1])]
    bad += [{**good, "c": c} for c in ({"roundings": 0}, {**FREE, "products": 0}, {**FREE, "roundings": "x"},
                                       {**FREE, "roundings": "1/-2"}, {**FREE, "roundings": ""}, [0, 0, 0, 0])]
    bad += [{**good, "B": B} for B in (0, -1, True, "128")] + [{**good, "L": L} for L in (-1, "1", True)]
    bad += [{**good, "R_in": r} for r in ("1e5", "", None)] + [{**good, "program": "not a descriptor"}]
    for v in bad:
        assert D.replay(CHECK, v, server)[0] == "reject", v
    ok = [{**good, "c": {**RICH, "folds": "-3/6"}}, {**good, "R_in": -5}, {**good, "windows": [[0, 3, 0, "7/3"]]}]
    for v in ok:
        assert D.replay(CHECK, v, server)[0] == "output", v


@st.composite
def _judged(draw: Any) -> dict:
    label, program, ends = draw(st.sampled_from([e for e in corpus() if e[0].startswith(("int8", "scan", "gather"))]))
    n = len(ends)
    cuts = sorted(draw(st.sets(st.integers(0, n), max_size=4)))
    windows = [[a, b, draw(st.integers(0, 400)), draw(st.sampled_from([1, "1/1000", "3/2", 0]))]
               for a, b in zip([0] + cuts, cuts + [n])]
    windows = draw(st.permutations(windows)) if draw(st.booleans()) else windows
    c = {k: draw(st.sampled_from([0, "1/32", "1/64", 1, "1000000"])) for k in P.BUDGETED}
    return {"program": encoded(program)[0], "ends": list(ends), "c": c, "windows": windows,
            "R_in": draw(st.sampled_from(["100000", "1000000000", 1, "64"])), "L": draw(st.sampled_from([None, 0, 1, 2, 3])),
            "B": draw(st.sampled_from([1, 128, 544, 1000]))}


def test_property_v1_agrees_on_drawn_windows_and_parameters(server) -> None:
    """Windows that cut the steps anywhere (empty, overlapping, out of order, of no time), any c, R_in, L and B."""
    D.run(port("property_check", property_check, _judged()), examples=200, server=server)


# --- the reading (binding 4) -----------------------------------------------------------------------------------------

@st.composite
def _record_json(draw: Any) -> dict:
    wins, s = [], draw(st.integers(-2, 0))
    for _ in range(draw(st.integers(0, 4))):
        steps = []
        for _ in range(draw(st.integers(0, 3))):
            s += draw(st.integers(0, 3))
            steps.append(s)
        wins.append({"steps": steps, "receipts": draw(st.integers(-1, 40)),
                     "span_ns": draw(st.sampled_from([0, 1, 10**9, 3 * 10**9 + 7, -1]))})
    return {"windows": wins, "census_steps": draw(st.integers(-1, s + 3)), "paced": draw(st.booleans()),
            "block_bytes": draw(st.sampled_from([4096, 512, 0, -1])), "lifetime": draw(st.integers(-1, 5))}


def test_the_reading_agrees_on_drawn_records(server) -> None:
    """Records of up to 4 windows of up to 3 passes, steps repeated, out of the census or short of its end, paced or not."""
    D.run(port(READING_PORT, property_reading, _record_json()), examples=500, server=server)


def test_the_reading_is_mem_38_s(server) -> None:
    """Pass i covers (s_{i−1}, s_i]: windows of passes at steps (0, 1, 2) and (3, 4) cover [0, 3) and [3, 5); a paced
    session's closing window takes the steps after its last pass; L_steps is the most steps any L_w + 1 consecutive
    windows cover."""
    rec = {"windows": [{"steps": [0, 1, 2], "receipts": 30, "span_ns": 2 * 10**9},
                       {"steps": [3, 4], "receipts": 24, "span_ns": 10**9}],
           "census_steps": 5, "paced": False, "block_bytes": 4096, "lifetime": 0}
    reading = port(READING_PORT, property_reading)
    out = D.replay(reading, rec, server)
    assert out == ("output", D.canonical({"windows": [[0, 3, 30 * 4096, "2"], [3, 5, 24 * 4096, "1"]], "L": 3}))
    assert D.replay(reading, {**rec, "lifetime": 1}, server)[1] == D.canonical(
        {"windows": [[0, 3, 30 * 4096, "2"], [3, 5, 24 * 4096, "1"]], "L": 5})
    assert D.replay(reading, {**rec, "census_steps": 7}, server)[0] == "reject"
    assert D.replay(reading, {**rec, "census_steps": 7, "paced": True}, server)[1] == D.canonical(
        {"windows": [[0, 3, 30 * 4096, "2"], [3, 7, 24 * 4096, "1"]], "L": 4})
    assert D.replay(reading, {**rec, "census_steps": 4}, server)[0] == "reject"


# --- the whole verdict and its bindings ------------------------------------------------------------------------------

#: the one-stage value commitments' leaves (opaque commit strings here), and two memory-map units' public rows and rows
VALUES = [hashlib.sha512(b"value leaf %d" % i).digest() for i in range(4)]
MAP = {"C": 2, "K": 2, "V": 4}
MAP_PUBLIC = [{"first": 8, "count": 2, "present": 0b11, "pos": [0, 1]},
              {"first": 10, "count": 1, "present": 0b01, "pos": [3, 0]}]
MAP_ROWS = [{"val0": VALUES[0], "val1": VALUES[1], "pay0": b"page 8", "pay1": b"page 9"},
            {"val0": VALUES[3], "pay0": b"page 10"}]
LIFETIME = 4
R_IN = 10**6
WINDOWS = [{"steps": [0, 1, 2], "receipts": 30, "span_ns": 2 * 10**9}, {"steps": [3, 4], "receipts": 24, "span_ns": 10**9}]


def request(program: Any, ends: list[int], **kw: Any) -> dict:
    """`audit.post_seed`'s input to the Lean on `program`: the honest session's shape, every binding holding."""
    desc, digest = encoded(program)
    return {"program": desc, "ends": list(ends), "audited": digest, "values": [v.hex() for v in VALUES], "map": MAP,
            "map_public": MAP_PUBLIC, "map_rows": [{r: v.hex() for r, v in rows.items()} for rows in MAP_ROWS],
            "mapped": True, "c": FREE, "R_in": str(R_IN), "B": P.OPERAND_BITS, "block_bytes": 4096,
            "lifetime": LIFETIME, "paced": False, "windows": WINDOWS, **kw}


def test_the_verdict_and_its_bindings_agree(server) -> None:
    """The writers pass with every binding; another Program's digest, a value row that isn't the leaf at its `pos`, a
    `pos` past the leaves, appends the map didn't make, and a class over its budget are each refused by name, the same
    names in Lean and Python; an unpaced record short of the census, a malformed public row, rows for another number of
    units and an L the record can't give are refused by both."""
    writers, ends = S.int8_program(S=5)
    epilogue, epi_ends = S.int8_program(S=5, epilogue=True)
    honest = request(writers, ends)
    out = D.replay(POST_SEED, honest, server)
    assert out[0] == "output" and '"refused":[]' in out[1]
    rows = [dict(r) for r in honest["map_rows"]]
    rows[1]["val0"] = VALUES[2].hex()
    named = {"program": {**honest, "audited": encoded(epilogue)[1]}, "values": {**honest, "map_rows": rows},
             "mapped": {**honest, "mapped": False},
             "property": request(epilogue, epi_ends, c={**FREE, "roundings": "5/122880"})}
    named["values-short"] = {**honest, "values": honest["values"][:3]}
    for name, v in named.items():
        kind, body = D.replay(POST_SEED, v, server)
        assert kind == "output" and f'"refused":["{name.split("-")[0]}"]' in body, (name, body)
    assert '"refused":[]' in D.replay(POST_SEED, request(epilogue, epi_ends, c={**FREE, "roundings": "6/122880"}),
                                      server)[1]
    six, six_ends = S.int8_program(S=6)
    paced = D.replay(POST_SEED, request(six, six_ends, paced=True), server)
    assert paced[0] == "output" and '"windows":[[0,3,122880,"2"],[3,6,98304,"1"]]' in paced[1]
    refused = [request(six, six_ends), {**honest, "map_public": [{**MAP_PUBLIC[0], "count": 3}, MAP_PUBLIC[1]]},
               {**honest, "map_public": [MAP_PUBLIC[0], {**MAP_PUBLIC[1], "present": 0b11, "pos": [3]}]},
               {**honest, "map_rows": honest["map_rows"][:1]}, {**honest, "values": ["ABCD"]},
               {**honest, "windows": []}, {**honest, "ends": ends[:-1]}, {**honest, "lifetime": -1}]
    for v in refused:
        assert D.replay(POST_SEED, v, server)[0] == "reject", v


# --- the verifier's call ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def record() -> audit.Record:
    r = appends().honest_session()
    assert all(v.accepted for v in r.finish())
    rec = r.s.record()
    assert [w.steps for w in rec.windows] == [(0, 1, 2), (3, 4)]
    return rec


def _post_seed(rec: audit.Record, program: Any, ends: list[int], **kw: Any) -> audit.PostSeed:
    desc, digest = encoded(program)
    args = dict(audited=digest, values=VALUES, map_statics=MAP, map_public=MAP_PUBLIC, map_rows=MAP_ROWS,
                link=appends().Link(True, 8 * 4096, {}), c=S.FREE, R_in=R_IN)
    return audit.post_seed(rec, desc, ends, **{**args, **kw})


def test_the_verifier_s_call_is_the_lean_s_verdict_on_the_record(server, record, monkeypatch) -> None:
    """`audit.post_seed` sends the record's windows and cap with the bindings' inputs, and its `PostSeed` is the Lean's
    answer, which the mirror gives too: W's digest, the census's steps, the record's reading and property/v1's verdict."""
    writers, ends = S.int8_program(S=5)
    sent = []
    call = lean_verdict.call
    monkeypatch.setattr(lean_verdict, "call", lambda fn, v: (sent.append((fn, v)), call(fn, v))[1])
    link = appends().Link(True, 8 * 4096, {})
    v = _post_seed(record, writers, ends, link=link)
    assert [fn for fn, _v in sent] == ["post_seed"] and link.asked == [record]
    D.replay(POST_SEED, sent[0][1], server)
    windows, L = audit.property_reading(record, 5)
    assert v.holds and v.refused == () and v.program == encoded(writers)[1] and v.census_steps == 5
    assert (v.windows, v.lifetime) == (windows, L) and v.checked["ok"]
    assert v.checked["numbers"]["max_lifetime"] == 1 and v.checked["numbers"]["macs"] == 10 * 32
    assert [w["input_bytes"] for w in v.checked["numbers"]["windows"]] == ["192", "128"]


def test_the_verifier_refuses_what_the_lean_refuses(record) -> None:
    """Ends that aren't W's steps, and a census longer than the record's last pass in an unpaced session; and, before the
    Lean, a float c or R_in, which property/v1 refuses since it would be read at its binary value."""
    writers, ends = S.int8_program(S=5)
    with pytest.raises(ValueError, match="step ends"):
        _post_seed(record, writers, ends[:-1] + [ends[-1] + 1])
    for kw in (dict(R_in=1e5), dict(c={**S.FREE, "roundings": 0.1})):
        with pytest.raises(ValueError, match="is exact"):
            _post_seed(record, writers, ends, **kw)
    six, six_ends = S.int8_program(S=6)
    with pytest.raises(ValueError, match="uncovered"):
        _post_seed(record, six, six_ends)
    assert _post_seed(record, six, six_ends, paced=True).windows[-1].stop == 6


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason=audit.OPEN_CARRIER)
def test_the_ledger_credits_the_verifier_s_verdict(record) -> None:
    """The honest verdict passes binding 4 (`tests/test_post_seed.py` has the ledger's refusals) and credits the
    post-seed receipts once the carriers' bytes read L in its units."""
    v = _post_seed(record, *S.int8_program(S=5))
    audit.ledger(record, capacity_bytes_per_s=4096 * 10**9 / 10, chip_bytes=0,
                 rebuild=audit.Rebuild(appends().P512, 0.0), credit_post_seed=v)
