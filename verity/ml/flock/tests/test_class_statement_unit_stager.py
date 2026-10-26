"""`class_statement.UnitStager`: a unit Definition's circuit built once and staged a round at a time, the IR reference checked
on every instance by one `evaluate_batch` call.  `stage_unit`, one round of a fresh stager, writes the bytes and returns the
values `_stage_unit_at_base` does (`stage_unit`'s code before the stager, less its per-instance reference loop, which writes
nothing) on `test_pouw_rows`' statements and a Boolean unit; a stager's rounds share its pin and are each the statement of
their own rows; a lowering output that is not the reference's is refused, naming the first such instance, on both of the
check's paths (bit-sliced for a Boolean Definition, the reference for the rest); and the stager's circuit renders the unit's
netlist once, the slot nets not again, and is the circuit composed with every netlist rendered afresh; and under
FLOCK_PLAIN_LEAVES=1 it names the plain leaf scheme."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_pouw_rows import POUW_TESTS, STAGED_TILES, V
from verity.ml.circuits import evaluation as EV
from verity.ml.circuits.boolean import BIT
from verity.ml.circuits.boolean.forms import add
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.types import Array, Value
from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X, SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import ir_lower as IL
from verity.core.protocols.pouw.circuit import leaves as L, pc4

#: a Boolean unit (a + b over 20 bits with its carry, a & c, and w[0] & w[2047] ^ w[1000]), its parameters' leaves off word
#: boundaries in two rows that split c's, its 34-bit output padded to three words; w's row spans SHA-512 blocks, as a composed
#: circuit needs one row to
TOY = bind(definition("UnitStagerToy", 1, lambda C, a, b, c, w: add(C, a, b, width=20, carry_out=True) + [C.AND(x, y) for x, y in zip(a, c)]
                      + [C.AND(w[0], w[2047]) ^ w[1000]],
                      (("a", Array(12, BIT)), ("b", Array(20, BIT)), ("c", Array(16, BIT)), ("w", Array(2048, BIT))), Array(34, BIT),
                      register=False))
TOY_ROWS = [("ac", V2, 16, ["a", *(f"c[{r}]" for r in range(4))]), ("bc", V2, 32, ["b", *(f"c[{r}]" for r in range(4, 16))]),
            ("w", V2, 2048, ["w"])]


def _stage_unit_at_base(definition, name, in_rows, out_rows, rows, out_dir, salt_key, salts=None, public_inputs=None) -> dict:
    public_inputs = dict(public_inputs or {})
    low, out_words = CS.unit_lowering(definition, name, [(r, ps) for r, _s, _b, ps in in_rows], tuple(public_inputs))
    oname = lambda ps: tuple(f"o{p}" for p in ps)  # noqa: E731
    seg_out = tuple(oname(ps) for s, ps in out_rows if s == SCHEMA_HM96_SHA512_ROW_SEG)
    v2_out = tuple(oname(ps) for s, ps in out_rows if s != SCHEMA_HM96_SHA512_ROW_SEG)
    CS.cache_slot_circuits()
    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")  # noqa: E731
    content = json.dumps([definition.id, [list(r[:3]) + [list(r[3])] for r in in_rows], [[s, list(ps)] for s, ps in out_rows]]
                         + ([public_inputs] if public_inputs else [])).encode()
    sub = SimpleNamespace(id=name, template=SimpleNamespace(id="unit-class"), inputs=[port(r) for r in [*public_inputs, *(r for r, *_ in in_rows)]],
                          outputs=[port(f"o{j}") for j in range(len(out_words))],
                          program_digests=lambda statics: {name: hashlib.sha256(content).hexdigest()})
    iset = SimpleNamespace(subcircuit=sub, name=name.replace("/", "-"), content_digest=hashlib.sha256(content).hexdigest())
    text = FC.compose(low, sub, port_bits=[None] * len(public_inputs) + [b for _r, _s, b, _ps in in_rows],
                      seg_ports=frozenset(r for r, s, _b, _ps in in_rows if s == SCHEMA_HM96_SHA512_ROW_SEG),
                      seg_out_rows=seg_out, v2_out_rows=v2_out, public_inputs=public_inputs or None)
    n = len(rows[in_rows[0][0]])
    flat = np.concatenate([np.asarray(rows[r], dtype=np.uint16) for r in [*public_inputs, *(r for r, *_ in in_rows)]], axis=1)
    got, _cut, ok = low.evaluate(flat.astype(np.uint64))
    assert int(ok) == (1 << n) - 1
    got = got.astype(np.uint16)
    at = np.cumsum([0] + out_words)
    outs = {f"o{j}": got[:, at[j]:at[j + 1]] for j in range(len(out_words))}
    meta = json.loads(text.split("\n", 2)[1][5:])
    salts = {**FC.stage_salts(iset, 0, n, FC._meta_ports(meta), salt_key), **(salts or {})}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "circuit.txt").write_text(text)
    FC.write(text, iset, 0, n, out_dir / "inst.bin", out_dir / "pub.bin", salt_key, rows={r: np.asarray(rows[r], dtype=np.uint16) for r, *_ in in_rows},
             outs=outs, salts=salts, public_words={p: np.asarray(rows[p], dtype=np.uint16) for p in public_inputs})
    return {"meta": meta, "outs": outs, "salts": salts, "ands": low.layout.counts["ands"],
            "unit_nnz": low.layout.counts["nnz_a"] + low.layout.counts["nnz_b"], "circuit_sha512": FC.digest(text)}


def _tile(m: str) -> tuple:
    """`test_a_real_pouw_unit_is_staged_and_composed_on_its_rows`' statement of `STAGED_TILES[m]`: (stager arguments, rows, keywords)."""
    template, k, row, _want = STAGED_TILES[m]
    d = template()
    names = [p for p, _t in d.signature()[0]]
    bits = 16 * L.ROW_WORDS[row](k)
    in_rows = [("a", L.ROW_SCHEMA[row, ROLE_X], bits, [f"{p}[0]" for p in names if p in ("ac", "as", "pa")]),
               ("b", L.ROW_SCHEMA[row, ROLE_W], bits, [f"{p}[0]" for p in names if p in ("bc", "bs", "pb")])]
    rows = {r: np.array([V.words(row, V.inputs(f"stage-unit-{m}-{r}-{i}", row, k)) for i in range(4)], dtype=np.uint16) for r in ("a", "b")}
    return (d, f"unit/pouw-{m}-tile", in_rows, [(V2, (0, 1))]), rows, {}


def _anchor() -> tuple:
    """`test_pouw_anchors_are_public_input_ports_held_to_the_verifiers_recomputation`' statement: three public input ports."""
    from verity.core.protocols.pouw.circuit.anchors import public_ports

    spec = importlib.util.spec_from_file_location("pouw_regions_vectors", POUW_TESTS / "pouw_regions_vectors.py")
    RV = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RV)
    t = RV.anchors("pc4-64", "pc4", 64, None)
    types = dict(pc4.templates(64)["row_a"].signature()[0])
    X = Array(128, Value(16))
    d = bind(CompositeDefinition("PouwAnchorStandIn", 1, (), lambda S: ((("x", X), ("e", types["e"]), ("fa", types["fa"]), ("fb", types["fb"])), X),
                                 lambda B, S, x, e, fa, fb: x, register=False))
    regions = [t.regions("row_a", i) for i in (1, 0)]
    rows = {"x": np.arange(256, dtype=np.uint16).reshape(2, 128), **{p: np.array([r[p] for r in regions], dtype=np.uint16) for p in ("e", "fa", "fb")}}
    return (d, "unit/pouw-anchor-stand-in", [("x", V2, 2048, ["x"])], [(V2, (0,))]), rows, {"public_inputs": public_ports(t.shape, "row_a")}


def _toy(seed: int = 0) -> tuple:
    rows = {r: np.random.default_rng([seed, i]).integers(0, 1 << 16, (5, bits // 16), dtype=np.uint16) for i, (r, _s, bits, _ps) in enumerate(TOY_ROWS)}
    return (TOY, "unit/stager-toy", TOY_ROWS, [(V2, (0,))]), rows, {}


STATEMENTS = {"pc4": lambda: _tile("pc4"), "pc8": lambda: _tile("pc8"), "anchor": _anchor, "toy": _toy}


def _same(base: Path, base_got: dict, new: Path, got: dict, circuit: Path | None = None) -> None:
    for f, p in (("circuit.txt", circuit or new / "circuit.txt"), ("inst.bin", new / "inst.bin"), ("pub.bin", new / "pub.bin")):
        assert p.read_bytes() == (base / f).read_bytes(), f
    assert {k: v for k, v in got.items() if k != "outs"} == {k: v for k, v in base_got.items() if k != "outs"}
    assert set(got["outs"]) == set(base_got["outs"])
    for o, a in base_got["outs"].items():
        assert got["outs"][o].dtype == a.dtype and np.array_equal(got["outs"][o], a), o


@pytest.mark.slow
@pytest.mark.parametrize("which", list(STATEMENTS))
def test_stage_unit_writes_the_bytes_and_returns_the_values_it_did_before_the_stager(which, tmp_path):
    """On each of `test_pouw_rows`' statements (the two tile units and the anchors' stand-in, with public input ports) and a
    Boolean unit, `stage_unit`'s circuit.txt, inst.bin and pub.bin are byte for byte, and its META, outputs, salts, ANDs,
    matrix entries and pin are, those of its code before `UnitStager`."""
    args, rows, kw = STATEMENTS[which]()
    base = _stage_unit_at_base(*args, rows, tmp_path / "base", b"k" * 32, **kw)
    _same(tmp_path / "base", base, tmp_path / "new", CS.stage_unit(*args, rows, tmp_path / "new", b"k" * 32, **kw))


@pytest.mark.slow
@pytest.mark.parametrize("once", [False, True], ids=["each-round", "once"])
def test_a_stagers_rounds_share_its_pin_and_are_each_the_statement_of_their_own_rows(once, tmp_path):
    """One stager, two rounds on different rows (the second with given salts on one row): one pin and one circuit.txt, beside
    each round's files or written once where the stager's ``circuit`` says, and the second round's files and values are those
    `stage_unit`'s code before the stager gives that round alone."""
    args, rows, _kw = _toy(0)
    _args, rows2, _kw = _toy(1)
    salts = {"ac": [bytes([7]) * FC.SALT_BYTES] * len(rows2["ac"])}
    circuit = tmp_path / "once" / "circuit.txt" if once else None
    st = CS.UnitStager(*args, circuit=circuit)
    first = st.stage(rows, tmp_path / "r1", b"k" * 32)
    second = st.stage(rows2, tmp_path / "r2", b"k" * 32, salts)
    assert first["circuit_sha512"] == second["circuit_sha512"] == st.pin == FC.digest(st.text)
    assert st.class_sha512 == FC.class_digest(st.text)
    written = [circuit] if once else [tmp_path / r / "circuit.txt" for r in ("r1", "r2")]
    assert [p.read_text() for p in written] == [st.text] * len(written)
    assert (tmp_path / "r1" / "circuit.txt").exists() != once
    assert (tmp_path / "r1" / "inst.bin").read_bytes() != (tmp_path / "r2" / "inst.bin").read_bytes()
    assert second["salts"]["ac"] == salts["ac"]
    _same(tmp_path / "base", _stage_unit_at_base(*args, rows2, tmp_path / "base", b"k" * 32, salts), tmp_path / "r2", second, circuit)


@pytest.mark.slow
def test_a_bound_stager_names_its_definition_and_the_partition_and_its_rounds_their_global_units(tmp_path):
    """Bound to a template query's partition of a two-member program, the stager's META names the toy as the circuit's one
    template (core's digest of its Program), the partition and the program; a round's header names the units it is given; the
    class is the unbound stager's; a round staged as another input set, on the one circuit, binds other frame domains; and
    only a bound stager's round takes indices."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P
    from verity.ml.circuits.codec import descriptor_id, program_digest
    from verity.ml.circuits.program import Program

    args, rows, _kw = _toy(0)
    program = P.mixed_population_program([(TOY, 3), (TOY, 5)])
    query = P.template_instances_query([TOY])
    obj = P.partition(program, query)
    bind = {"partition": P.digest(obj), "program": obj["program"]}
    st, plain = CS.UnitStager(*args, bind=bind), CS.UnitStager(*args)
    meta = st.meta
    assert meta["program_digests"] == {descriptor_id(TOY): program_digest(Program(TOY))}
    assert meta["partition"] == {"rule": FC.PARTITION_V1, "digest": bind["partition"]} and meta["program_sha512"] == bind["program"]
    assert st.class_sha512 == plain.class_sha512 and st.pin != plain.pin
    st.stage(rows, tmp_path / "bound", b"k" * 32, indices=range(3, 8))
    header = json.loads((tmp_path / "bound" / "pub.bin").read_bytes().split(b"\n", 1)[0])
    assert header["units"]["indices"] == [3, 4, 5, 6, 7] and header["set"] == st.iset.name
    st.stage(rows, tmp_path / "other-set", b"k" * 32, indices=range(3, 8), set_name="toy-c1")
    h2 = json.loads((tmp_path / "other-set" / "pub.bin").read_bytes().split(b"\n", 1)[0])
    assert h2["set"] == "toy-c1" and h2["circuit_sha512"] == header["circuit_sha512"] == st.pin
    doms, doms2 = header["frame_v3"]["domain_ids"], h2["frame_v3"]["domain_ids"]
    assert doms.keys() == doms2.keys() and all(doms[p] != doms2[p] for p in doms)
    with pytest.raises(ValueError, match="global indices"):
        st.stage(rows, tmp_path / "no-indices", b"k" * 32)
    with pytest.raises(ValueError, match="global indices"):
        plain.stage(rows, tmp_path / "plain", b"k" * 32, indices=range(5))
    with pytest.raises(ValueError, match="exactly partition and program"):
        CS.UnitStager(*args, bind={"partition": bind["partition"]})


@pytest.mark.slow
@pytest.mark.parametrize("which", ["toy", "anchor"])
def test_a_lowering_output_that_is_not_the_references_is_refused_naming_the_first_such_instance(which, tmp_path, monkeypatch):
    """With bit 0 of the lowering's first output word wrong on every odd instance, the round is refused by one `evaluate_batch`
    call, naming instance 1, and writes nothing: on a Boolean unit, bit-sliced (the per-instance reference never runs), and on
    the anchors' stand-in, which has no kernel (the reference)."""
    args, rows, kw = STATEMENTS[which]()
    st = CS.UnitStager(*args, **kw)
    evaluate, batch, calls = st.low.evaluate, EV.evaluate_batch, []

    def wrong(flat):
        got, cut, ok = evaluate(flat)
        got = got.copy()
        got[1::2, 0] ^= np.uint64(1)
        return got, cut, ok

    monkeypatch.setattr(st.low, "evaluate", wrong)
    monkeypatch.setattr(EV, "evaluate_batch", lambda *a, **k: calls.append(1) or batch(*a, **k))
    if which == "toy":
        monkeypatch.setattr(EV.batch, "evaluate", lambda *a: pytest.fail("a Boolean unit's check ran the per-instance reference"))
    with pytest.raises(AssertionError, match=f"^{args[1]}: instance 1's outputs are not the IR reference's$"):
        st.stage(rows, tmp_path / "out", b"k" * 32)
    assert calls == [1] and not (tmp_path / "out").exists()


@pytest.mark.slow
def test_the_stagers_circuit_renders_the_units_netlist_once_and_is_the_uncached_renders(monkeypatch):
    """Composing a stager's circuit renders the unit's netlist once (its CIRCUIT section, which ``unit_sha256`` hashes with the
    LEAVES line) and the slot nets `cache_slot_circuits` holds not at all; `Lowering.sha256` renders on its first read only;
    and the circuit is, byte for byte, the one composed with every netlist rendered afresh and ``unit_sha256`` the digest of
    `Lowering.text`, as `circuit.compose` did before it rendered once."""
    args, _rows, _kw = _toy(0)
    CS.cache_slot_circuits()
    netlist, renders = IL.netlist, []
    monkeypatch.setattr(IL, "netlist", lambda lay, name, *a: renders.append(name) or netlist(lay, name, *a))
    st = CS.UnitStager(*args)
    assert renders == [st.low.name]
    renders.clear()
    assert st.low.sha256 == st.low.sha256 == IL.digest(st.low.text) and renders == [st.low.name] * 2
    assert st.meta["unit_sha256"] == st.low.sha256
    monkeypatch.setattr(IL, "netlist", netlist)
    monkeypatch.setattr(FC, "_expanded", lambda lay, name: IL.netlist(lay, name, None, None))
    monkeypatch.setattr(IL.Lowering, "sha256_of", lambda self, _expanded: IL.digest(self.text))
    assert CS.UnitStager(*args).text == st.text


def test_a_unit_lowering_drops_its_circuit_once_laid_out(monkeypatch):
    """`unit_lowering` keeps no circuit (a GLM-5.3 norm unit's is 51 GiB, which nothing reads after the layout), and its compact
    circuit (`_UnitCircuit`) lays out a plain `gf2.Circuit`'s netlist; `Lowering.domain` then reads the layout's assertions."""
    from verity_flock import gf2

    rows = [(r, ps) for r, _s, _b, ps in TOY_ROWS]
    low, words = CS.unit_lowering(TOY, "unit/stager-toy", rows)
    monkeypatch.setattr(CS, "_UnitCircuit", gf2.Circuit)
    plain, plain_words = CS.unit_lowering(TOY, "unit/stager-toy", rows)
    assert low.circuit is None and words == plain_words and low.layout.counts == plain.layout.counts
    assert low.text == plain.text and low.domain == "total"
    for n, want in ((0, "total"), (2, "finite-only")):
        assert IL.Lowering.domain.fget(SimpleNamespace(circuit=None, layout=SimpleNamespace(counts={"asserts": n}))) == want


@pytest.mark.parametrize("chunk", [1, 5, 1 << 24])
def test_a_lowerings_sha256_is_its_texts_in_pieces(chunk, monkeypatch):
    """`Lowering.sha256_of` (``unit_sha256``) is SHA-256 of `Lowering.text`, its netlist encoded :data:`ir_lower.HASH_CHUNK`
    characters at a time."""
    class Whole(str):
        def encode(self, *a):
            raise AssertionError("encoded whole")

    low, _words = CS.unit_lowering(TOY, "unit/stager-toy", [(r, ps) for r, _s, _b, ps in TOY_ROWS])
    monkeypatch.setattr(IL, "HASH_CHUNK", chunk)
    assert low.sha256_of(Whole(IL.netlist(low.layout, low.name))) == hashlib.sha256(low.text.encode()).hexdigest() == low.sha256


def test_a_stagers_circuit_takes_flock_plain_leaves_leaf_scheme(monkeypatch):
    """Under ``FLOCK_PLAIN_LEAVES=1`` (`class_statement.LEAF_SCHEME`, plain SHA-512) a stager's circuit names the plain leaf
    scheme, a recursive session's inner statement, as `stage`'s circuit does; its META is the default circuit's but for
    ``leaf_scheme``, and its pin another statement's."""
    args, _rows, _kw = _toy(0)
    st = CS.UnitStager(*args)
    assert st.meta["leaf_scheme"] == FC.LEAF_SCHEME
    monkeypatch.setattr(CS, "LEAF_SCHEME", FC.LEAF_SCHEME_PLAIN)
    plain = CS.UnitStager(*args)
    assert plain.meta["leaf_scheme"] == FC.LEAF_SCHEME_PLAIN
    assert {k: v for k, v in plain.meta.items() if k != "leaf_scheme"} == {k: v for k, v in st.meta.items() if k != "leaf_scheme"}
    assert plain.pin != st.pin
