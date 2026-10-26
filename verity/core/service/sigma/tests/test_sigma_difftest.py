"""`Flock.Sigma.check`, the Lean verifier's check that a Boolean Program is σ of its word Program, held to the Python that
mirrors it, `verity.ml.definitions.boolean.sigma_check`.

The reference decodes both descriptors (`codec.decode_program`), checks the Boolean descriptor's `word` link, σ's table and the
input streams as the Lean does, then runs `sigma_check` itself on the two roots seen as Definitions: each root's Calls in
order, the input nodes one parameter (the word side a Tuple of their outputs, the Boolean side that Tuple on bits), and σ the
table (`boolean_version` patched to read it). Its first mismatch is the refusal the Lean must give, by code and place.

The corpus is circuits' σ composites (a derived function with every kind of reference sequence, and lifted registered
composites) and the negatives the check exists for: a moved operand run, a wrong arity, σ not a function, and one of each other
refusal. `sigma_check` walks a strided part's rows as `refs.runs` does, which drops a short last row: no case has one.
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from hypothesis import strategies as st

import verity_difftest as D
import verity.ml.definitions.boolean as BB
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.codec import decode_program, descriptor_id, encode_program, program_digest
from verity.ml.circuits.defs import Builder, CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops import prims as MP
from verity.ml.circuits.program import Program, input_id, is_input
from verity.ml.circuits.refs import Affine, Coll, Concat, Explicit, PartLog, Strided, coll_concat
from verity.ml.circuits.types import Array, Record, Tuple as TupleT, Type, Value
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry.catalog import load as load_registry
from verity.ml.definitions.replay import replayer

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

LEAN = Path(__file__).resolve().parents[1]
REGISTRY = load_registry()[0]
F32, BF16 = Value(32), Value(16)

# ---- the reference -----------------------------------------------------------------------------------------------------------


def _decode(desc: dict):
    try:
        return decode_program(desc, REGISTRY)
    except (ValueError, KeyError, TypeError, IndexError) as e:
        raise D.Reject(f"{type(e).__name__}: {e}") from e


def _no(code: str, call: int | None = None, operand: int | None = None, definition: str | None = None) -> dict:
    out: dict = {"ok": False, "code": code}
    for k, v in (("call", call), ("operand", operand), ("definition", definition)):
        if v is not None:
            out[k] = v
    return out


def _types(fn) -> list[Type]:
    params, ret = fn.signature()
    return [t for _, t in params] + [ret]


def _stream(body) -> tuple[list[list], int | None, int]:
    """The input stream's (bits, binding) runs, the first input node that isn't bits, and the stream's bits."""
    runs: list[list] = []
    wide, total = None, 0
    for k, n in enumerate(body.nodes):
        if not is_input(n.fn):
            continue
        if wide is None and BL.bit_type(n.out) != n.out:
            wide = k
        b, how = BB._bits(n.out), input_id(n.fn.id)[0]
        total += b
        if b and runs and runs[-1][1] == how:
            runs[-1][0] += b
        elif b:
            runs.append([b, how])
    return runs, wide, total


def _view(body, param: Type):
    """A root as the Definition `sigma_check` reads: its Calls, its input nodes' leaves leaves of one parameter."""
    calls = [k for k, n in enumerate(body.nodes) if not is_input(n.fn)]
    index = {k: c for c, k in enumerate(calls)}
    start, pos = {}, 0
    for k, n in enumerate(body.nodes):
        if is_input(n.fn):
            start[k], pos = pos, pos + n.out.leaves

    def at(space, k, leaf):
        assert space == "n", "a root has no parameters"
        return ("p", 0, start[k] + leaf) if k in start else ("n", index[k], leaf)

    def moved(r):
        if isinstance(r, Affine):
            return Affine(*at(r.space, r.idx, r.base), r.count)
        if isinstance(r, Strided):
            return Strided(*at(r.space, r.idx, r.base), r.inner, r.outer_stride, r.inner_stride, r.count)
        if isinstance(r, Concat):
            return Concat([moved(p) for p in r.parts])
        return Explicit(tuple(at(*x) for x in r))

    nodes = [SimpleNamespace(fn=n.fn, form=n.form, n=n.n, axes=n.axes, name=n.name, n_xs=n.n_xs, out=n.out,
                             args=[Coll(a.type, moved(a.refs)) for a in n.args]) for n in (body.nodes[k] for k in calls)]
    view = SimpleNamespace(body=SimpleNamespace(nodes=nodes, ret=moved(body.ret)))
    view.signature = lambda: ((("inputs", param),), body.ret_type)
    return view, calls


_NO_ENTRY = PrimitiveDefinition("NoSigmaEntry", 1, (), Value(1), lambda: 0, register=False)


def reference(case: dict) -> dict:
    wd, bd = case["word"], case["bits"]
    PW, PB = _decode(wd), _decode(bd)
    link = bd.get("word")
    if (not isinstance(link, dict) or sorted(link) != ["program_sha512", "sigma"] or link["program_sha512"] != program_digest(wd)
            or not isinstance(link["sigma"], dict)):
        return _no("word-link")
    image = {}
    for w in sorted(link["sigma"]):
        b = link["sigma"][w]
        if not isinstance(b, str) or w not in PW.defs or b not in PB.defs:
            return _no("sigma-table", definition=w)
        if [BL.bit_type(t) for t in _types(PW.defs[w])] != _types(PB.defs[b]):
            return _no("sigma-signature", definition=w)
        image[id(PW.defs[w])] = PB.defs[b]
    (rw, _, _), (rb, wide, bits) = _stream(PW.root.body), _stream(PB.root.body)
    if wide is not None or rw != rb:
        return _no("inputs")
    inputs = TupleT(*[n.out for n in PW.root.body.nodes if is_input(n.fn)])
    vw, calls = _view(PW.root.body, inputs)
    vb, _ = _view(PB.root.body, BL.bit_type(inputs))
    with mock.patch.object(BB, "boolean_version", lambda fn: image.get(id(fn), _NO_ENTRY)):
        sc = BB.sigma_check(vw, vb)
    if sc["ok"]:
        return {"ok": True, "calls": len(calls), "definitions": sc["definitions"], "input_bits": bits}
    first = sc["mismatches"][0]
    word_fn = lambda c: descriptor_id(vw.body.nodes[c].fn)  # noqa: E731
    if first.startswith("signature"):
        return _no("signature")
    if re.fullmatch(r"\d+ nodes, the word body \d+", first):
        return _no("node-count")
    if m := re.match(r"node (\d+): Calls ", first):
        c = int(m[1])
        earlier = any(word_fn(c0) == word_fn(c) for c0 in range(c))
        return _no("sigma-not-function" if earlier else "sigma-table", call=c, definition=word_fn(c))
    if m := re.match(r"node (\d+): form ", first):
        return _no("node-form", call=int(m[1]))
    if m := re.match(r"node (\d+) operand (\d+): ", first):
        c, g = int(m[1]), int(m[2])
        wa, ba = vw.body.nodes[c].args[g], vb.body.nodes[c].args[g]
        return _no("operand-type" if ba.type != BL.bit_type(wa.type) else "operand-runs", call=c, operand=g)
    assert first.startswith("return"), first
    return _no("return-runs")


# ---- the corpus --------------------------------------------------------------------------------------------------------------

PARAMS = (("x", Array(4, F32)), ("y", Array(2, Array(4, F32))), ("weights", Record(("s", F32), ("t", Array(3, BF16)), ("u", Array(2, F32)))))
FNS = {"add": W.F32AddFtz, "mul": W.F32MulFtz, "cast": MP.F2fpBf16, "max": W.F32Max, "guard": W.GuardNegInfZero}


def _derived(name: str, params, build) -> object:
    B = Builder(params, TupleT())
    out = build(B, *(Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params)))
    B.ret_type = out.type
    body = B.finish(out)
    defn = CompositeDefinition(f"Derived[{name}]", 1, (), lambda S: (params, out.type), replayer(body), register=False)
    sd = bind(defn)
    sd.set_body(body)
    return sd


def tiny(last=W.F32AddFtz, ret: str = "full", extra: bool = False):
    """`test_boolean_switch._derived`'s function: every kind of reference sequence (affine, strided, a concat with explicit
    references, a `PartLog` cut), with node 5's Definition, the return and an extra node as the variants need."""
    def build(B, x, y, w):
        n0 = B.batch(FNS["add"], x, y[1], axes=(0, 0))
        y_t = Coll(Array(8, F32), Strided("p", 1, 0, 2, 1, 4, 8))
        x_twice = Coll(Array(8, F32), Concat([Affine("p", 0, 0, 4), Explicit(tuple(("p", 0, j) for j in (3, 2, 1, 0)))]))
        n1 = B.batch(FNS["mul"], y_t, x_twice, axes=(0, 0))
        n2 = B.batch(FNS["cast"], n1, axes=(0,))
        n3 = B.call(FNS["max"], w.field("s"), n0[2])
        n4 = B.batch(FNS["guard"], w.field("u"), axes=(0,))
        cut = PartLog([Affine("n", 0, 3, 1), Affine("n", 0, 0, 1), Affine("n", 3, 0, 1)]).cut(3)
        n5 = B.batch(last, Coll(Array(3, F32), cut), Coll(Array(3, F32), Affine("p", 0, 1, 3)), axes=(0, 0))
        if extra:
            B.call(FNS["max"], n3, n3)
        t = w.field("t") if ret != "moved" else Coll(Array(3, BF16), Affine("n", 2, 4, 3))
        parts = [n2, n3, n4, t, n5] if ret != "short" else [n2, n3, n4, t]
        return coll_concat(TupleT(*(p.type for p in parts)), parts)
    return _derived("Tiny", PARAMS, build)


def one_add(row: int):
    return _derived("OneAdd", PARAMS[:2], lambda B, x, y: B.batch(W.F32AddFtz, x, y[row], axes=(0, 0)))


def scalar_add(form: str):
    """One F32 add of x's first two leaves, as a Call or as a one-member batch, returning x."""
    def build(B, x):
        if form == "call":
            B.call(W.F32AddFtz, x[0], x[1])
        else:
            B.batch(W.F32AddFtz, Coll(Array(1, F32), Affine("p", 0, 0, 1)), Coll(Array(1, F32), Affine("p", 0, 1, 1)), axes=(0, 0))
        return x
    return _derived("ScalarAdd", PARAMS[:1], build)


def sigma(fn) -> dict:
    out = {}
    for n in fn.body.nodes:
        v = BB.boolean_version(n.fn)
        out[descriptor_id(n.fn)] = descriptor_id(bind(v) if isinstance(v, CompositeDefinition) else v)
    return out


def pair(word, bits=None, *, table=None, registered=(), link=None) -> dict:
    wd = encode_program(Program(word))
    bd = encode_program(Program(bits if bits is not None else BB.lift(word), registered=registered))
    bd["word"] = link if link is not None else {"program_sha512": program_digest(wd), "sigma": table if table is not None else sigma(word)}
    return {"word": wd, "bits": bd}


def _registered() -> list:
    return [bind(b1.ResidualAdd, N=4), bind(b1.BiasAdd, N=4), bind(b1.SiluMul, I=2), bind(b1.RoPEHead, D=4),
            bind(b1.Bf16MulScalarTensor, N=4)]


def _cases() -> list[dict]:
    word = tiny()
    lifted = [pair(word), pair(one_add(0)), pair(one_add(1)), pair(scalar_add("call")), *map(pair, _registered())]
    add, mul, guard = (descriptor_id(BB.boolean_version(f)) for f in (W.F32AddFtz, W.F32MulFtz, W.GuardNegInfZero))
    t = sigma(word)
    shifted = pair(word)
    root = shifted["bits"]["definitions"][shifted["bits"]["root"]]["body"]
    last = root["nodes"][-1]["args"][1]["refs"]
    assert "affine" in last and last["affine"][2] > 0, last
    last["affine"][2] -= 1
    negatives = [
        pair(one_add(1), BB.lift(one_add(0))),                                         # a moved operand run
        shifted,                                                                        # the same, in the descriptor
        pair(word, table={**t, descriptor_id(W.F32AddFtz): guard}),                     # σ(add) of another arity
        pair(word, BB.lift(tiny(last=W.F32MulFtz))),                                      # σ not a function
        pair(word, link={}),
        pair(word, link={"program_sha512": "0" * 128, "sigma": t}),
        pair(word, link={"program_sha512": program_digest(encode_program(Program(word))), "sigma": t, "extra": 1}),
        pair(word, table={k: v for k, v in t.items() if k != descriptor_id(MP.F2fpBf16)}),
        pair(word, table={**t, "Nothing_v1": add}),
        pair(word, table={**t, descriptor_id(W.F32AddFtz): "Nothing_v1"}),
        pair(word, table={**t, descriptor_id(W.F32AddFtz): mul}),
        pair(word, registered=("x",)),                                                  # an input bound otherwise
        pair(word, word, table={}),                                                     # inputs that aren't bits
        pair(word, BB.lift(tiny(ret="short"))),                                         # signature
        pair(word, BB.lift(tiny(extra=True))),                                          # node count
        pair(scalar_add("call"), BB.lift(scalar_add("batch"))),                         # node form
        pair(word, BB.lift(tiny(ret="moved"))),                                         # return
    ]
    bad_schema = copy.deepcopy(lifted[0])
    bad_schema["bits"]["schema"] = "verity-ir/descriptor/v0-not"
    bad_word = copy.deepcopy(lifted[0])
    del bad_word["word"]["root"]
    return lifted + negatives + [bad_schema, bad_word]


CASES = _cases()


def test_the_corpus_reaches_every_refusal():
    """Each refusal the check can give on decodable descriptors is some case's, and the composites pass."""
    got = [reference(c) for c in CASES[:-2]]
    assert all(r["ok"] for r in got[:9])
    codes = {r.get("code") for r in got}
    assert codes >= {"word-link", "sigma-table", "sigma-signature", "sigma-not-function", "inputs", "signature", "node-count",
                     "node-form", "operand-runs", "return-runs"}, codes
    assert got[9] == {"ok": False, "code": "operand-runs", "call": 0, "operand": 1}
    assert got[12]["code"] == "sigma-not-function" and got[12]["call"] == 5


CHECK = D.Port("sigma-difftest", "check", reference=reference, inputs=st.sampled_from(CASES), package=LEAN)
server = D.server_fixture(CHECK)


def test_the_lean_check_agrees_with_sigma_check(server):
    D.run(CHECK, examples=120, server=server)


def test_flock_verify_sigma_check_prints_the_verdict_and_the_fix(tmp_path):
    """`flock-verify sigma-check WORD BITS`, the verifier's command: one `SIGMACHECK` line with the verdict the difftest holds,
    exit 0 when the lift holds and 1 when it is refused, and each refusal's reason followed by its fix."""
    exe = D.build(LEAN.parent / "flock", "flock-verify")
    for k, case in enumerate(CASES):
        files = [tmp_path / f"{k}.{side}.json" for side in ("word", "bits")]
        for f, side in zip(files, ("word", "bits")):
            f.write_text(D.canonical(case[side]))
        p = subprocess.run([str(exe), "sigma-check", *map(str, files)], capture_output=True, text=True)
        [line] = p.stdout.splitlines()
        tag, out = line.split(" ", 1)
        got = json.loads(out)
        try:
            want = reference(case)
        except D.Reject:
            want = {"ok": False, "code": "decode"}
        assert tag == "SIGMACHECK" and p.returncode == (0 if want["ok"] else 1), (k, p.returncode, line)
        if want["ok"]:
            assert got == want, k
            continue
        why = got.pop("why")
        assert ": " in why, f"case {k}: {why!r} gives no fix"
        assert got == want or want["code"] == got["code"] == "decode", (k, got, want)


# ---- bit offsets: the prefix sums of the leaf widths ---------------------------------------------------------------------------

_TYPES = st.recursive(
    st.integers(1, 40).map(Value),
    lambda t: st.one_of(st.tuples(st.integers(0, 3), t).map(lambda a: Array(*a)),
                        st.lists(t, max_size=3).map(lambda fs: TupleT(*fs)),
                        st.lists(t, max_size=3).map(lambda fs: Record(*((f"f{i}", f) for i, f in enumerate(fs))))),
    max_leaves=8)


@st.composite
def _offsets_input(draw):
    t = draw(_TYPES)
    return {"type": t.to_json(), "js": draw(st.lists(st.integers(0, t.leaves), min_size=1, max_size=6))}


def offsets_reference(x: dict) -> list[int]:
    from verity.ml.circuits.codec import _type
    widths = _type(x["type"]).leaf_widths()
    return [sum(widths[:j]) for j in x["js"]]


OFFSETS = D.Port("sigma-difftest", "bit-offsets", reference=offsets_reference, inputs=_offsets_input(), package=LEAN,
                 need_rejects=False)


def test_bit_offsets_are_the_prefix_sums_of_the_leaf_widths(server):
    D.run(OFFSETS, examples=300, server=server)
