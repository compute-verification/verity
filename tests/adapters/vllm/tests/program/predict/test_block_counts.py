"""verity_vllm.program.predict.block_counts: a day's (c, k)-inference blocks from its served windows' circuit records,
judged against its declared c, k and n. On a small circuit (two catalog templates over two slots) its count, a block's
gates, its gate table and its verdict agree with `InferenceOnlyCheck.Count`'s Python reference (`inference_only_reference.check_count_detail`
and `check_count`, which `inference-only-count-difftest` holds the Lean to) on the session Program, m activations of the block
Definition; the public judgment holds only its record's fields and no count, so days of different counts under n publish
the same bytes, and a refusal past n names n; the exact counts are in the private file alone (0600); the wiring between
leaf widths prices at no gate; a primitive with no Boolean body refuses the circuit, and a block reaching one a table lacks
leaves the gates undecided; `main` counts a record only beside the registration that binds it (its SHA-512,
program, partition and run) and refuses one with none; on `test_window_circuit`'s TINY window a block's fresh bits are B's k
(`inference_blocks.budgets`); and the module's import alone loads no registry module, which `resolve` loads on its first
call."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import verity_vllm.definitions  # noqa: F401  (registers the vLLM registry with the Boolean-versions lookup)
from verity_vllm.program.predict import block_counts as BC
from verity_vllm.records import block_counts as RBC
from verity_vllm.records.fields import RecordError, UnknownField

from verity.circuits import codec
from verity.circuits import count as K
from verity.circuits import partition_v2 as PV2
from verity.circuits import program as PG
from verity.circuits.defs import (
    CompositeDefinition,
    PrimitiveDefinition,
    bind,
)
from verity.circuits.ops import gemm as GEMM_OPS
from verity.circuits.ops.boolean import versions as V
from verity.circuits.population import BitsWord, WordBits, wired_population_program
from verity.circuits.program import NoBooleanBody, Program
from verity.circuits.types import Value
from verity.commitments.periods import interface as IF

REFERENCE = Path(__file__).resolve().parents[6] / "verity/protocols/policy/training_detection/inference_only/verifier/lean/scripts/inference_only_reference.py"
GEMM = 'Gemm_v2{K=64,N=8,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}'
ROPE = "RoPEHead_v1{D=16}"
DAY = "2026-11-01"
ON_DEMAND = "GatherBf16x64_v1"
ALONE = """
import json, sys
from verity_vllm.program.predict import block_counts as BC
from verity.circuits import codec
from verity.circuits.defs import REGISTRY


def asked(fid):
    try:
        return codec.lookup_primitive(REGISTRY, fid).id
    except KeyError:
        return None


loaded = sorted(m for m in sys.modules if m.startswith("verity_vllm.definitions.registry"))
before = asked(sys.argv[1])
BC.resolve(sys.argv[2])
print(json.dumps({"loaded": loaded, "before": before, "after": asked(sys.argv[1])}))
"""


def _reference():
    spec = importlib.util.spec_from_file_location("inference_only_reference", REFERENCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def small() -> dict:
    """A window circuit record of two slots and three steps: four `GEMM` units, a slot's two in one entry, and six `ROPE`
    units, a slot's three across the steps."""
    entries = (IF.InferenceEntry(GEMM, 0, (2, 2), ("slot", "-"), 0), IF.InferenceEntry(ROPE, 0, (3, 2), ("step", "slot"), 0))
    circuit = IF.Circuit("test/blocks", ((BC.resolve(GEMM), 4), (BC.resolve(ROPE), 6)),
                         blocks=IF.Blocks(slots=2, steps=3, inference=entries))
    return IF.descriptor_record(circuit, 1)


def test_the_count_is_ncicheck_counts_on_the_session_program(small):
    """The session Definition activated m times is the block Definition `InferenceOnlyCheck.Count` counts: its row's gates, by the
    judgment's own table, are the judgment's gates a block, its blocks the private judgment's, and its verdict against the
    same c, k and n the judgment's."""
    pub, private = BC.judgment(DAY, [(small, None)], {"c": 10 ** 12, "k": 10 ** 12, "n": 2})
    (cls,) = pub.classes
    session, m, steps = BC.session(small)
    fn = session.fn
    assert (m, steps, cls.slots, cls.steps, private["total"], private["windows"]) == (2, 3, 2, 3, 2, 1)
    assert cls.definition == PV2.id_digest(fn) and not cls.unpriced and private["blocks"] == {cls.definition: 2}
    key = codec.descriptor_id(fn)
    program, _ = wired_population_program([(fn, m)], {"templates": [[key, m]]})
    text = codec.canonical_json(codec.encode_program(program)).decode()
    for n, verdict in ((2, "accept"), (1, "refuse")):
        inst = {"program": text, "theta": [], "blocks": [key], "table": dict(pub.table), "c": cls.gates, "k": 10 ** 12,
                "n": n}
        got = _reference().check_count_detail(inst)
        assert got["blocks"][0][:2] == [key, cls.gates] and got["blocks"][0][3] == m == got["total"]
        assert _reference().check_count(inst) == {"complies": got["complies"]}
        mine, _ = BC.judgment(DAY, [(small, None)], {"c": cls.gates, "k": 10 ** 12, "n": n})
        assert mine.verdict == verdict == ("accept" if got["complies"] else "refuse")
    by = {t: BC.gates(BC.resolve(t), pub.table)[0] for t in (GEMM, ROPE)}
    assert cls.gates == 2 * by[GEMM] + 3 * by[ROPE] > 0
    assert cls.fresh == IF.input_count(small)["free"] // 2


def _impostor(tid: str):
    """A Definition under ``tid`` whose body is another: the registered one's with its two parameters swapped."""
    real = BC.resolve(tid)
    d = real.defn
    other = CompositeDefinition(d.name, d.version, d.statics, d.signature, lambda B, S, a, b: d.body(B, S, b, a),
                                register=False, ports=d.ports)
    fn = bind(other, **real.bindings)
    assert codec.descriptor_id(fn) == tid and fn is not real
    return fn


def test_a_record_its_ids_do_not_rebuild_is_refused_by_name(small, monkeypatch):
    """The counts refuse a record whose template ids rebuild another program than the record's digest names: its program
    field tampered, or one id bound to another body than the record's (two bodies under one id); and one that names no
    program digest. The honest record is counted."""
    assert BC.session(small)[1:] == (2, 3) and BC.fresh(small, 2, None, None) > 0
    honest = codec.program_digest(IF.program_of_record(small, BC.resolve)[0])
    assert honest == small["program"]
    bare = {k: v for k, v in small.items() if k != "program"}
    for count in (lambda d: BC.session(d), lambda d: BC.fresh(d, 2, None, None)):
        with pytest.raises(ValueError, match="test/blocks: the circuit record names no program digest, so the Definitions its "
                                             "template ids name can't be held to the declared circuit, and it is not counted"):
            count(bare)
        with pytest.raises(ValueError, match=f"test/blocks: its template ids rebuild the program {honest}, not the record's "
                                             f"{'0' * 128}: a template's body under its id is not the declared circuit's"):
            count({**small, "program": "0" * 128})
    swapped = {ROPE: _impostor(ROPE)}
    real = BC.resolve
    monkeypatch.setattr(BC, "resolve", lambda tid: swapped.get(tid) or real(tid))
    for count in (lambda d: BC.session(d), lambda d: BC.fresh(d, 2, None, None),
                  lambda d: BC.judgment(DAY, [(d, None)], {"c": 10 ** 12, "k": 10 ** 12, "n": 2})):
        with pytest.raises(ValueError, match=f"not the record's {small['program']}: a template's body under its id is not "
                                             "the declared circuit's, and the record is not counted"):
            count(small)


def _registration(body: bytes, record: dict, **change) -> dict:
    """The registration that binds the circuit record ``record`` whose file's bytes are ``body``, but for ``change``."""
    return {"program": record.get("program"), "partition": record.get("partition"),
            "window": {"circuit": hashlib.sha512(body).hexdigest(), "run": record.get("run")}, **change}


def _main(tmp_path: Path, record: dict, windows: int, decl: dict, tag: str, registration: dict | None = None) -> int:
    body = json.dumps(record).encode()
    (tmp_path / "circuit.json").write_bytes(body)
    reg = tmp_path / "registration.json"
    reg.write_text(json.dumps(_registration(body, record) if registration is None else registration))
    (tmp_path / "declaration.json").write_text(json.dumps(decl))
    a = BC.Options(day=DAY, declaration=tmp_path / "declaration.json", private=tmp_path / f"private-{tag}.json",
                   out=tmp_path / f"public-{tag}.json", window=[[str(tmp_path / "circuit.json"), str(reg)]] * windows)
    return BC.main(a)


def _run(tmp_path: Path, record: dict, windows: int, decl: dict, tag: str) -> tuple[bytes, Path]:
    """`main` on ``windows`` served windows of ``record``, each with the registration that binds it: its public file's
    bytes and its private file."""
    assert _main(tmp_path, record, windows, decl, tag) == 0
    return (tmp_path / f"public-{tag}.json").read_bytes(), tmp_path / f"private-{tag}.json"


@pytest.mark.parametrize("change, refusal", [
    ({"program": "ab" * 20}, "the circuit record names another program, partition or run than its registration"),
    ({"partition": "ab" * 20}, "the circuit record names another program, partition or run than its registration"),
    ({"window": {"circuit": "ab" * 64}}, r"circuit.json is not the circuit record the registration binds \(abab"),
])
def test_a_record_its_registration_doesnt_bind_is_not_counted(small, tmp_path, change, refusal):
    """A registration that names another program or partition than the record, or binds other bytes, refuses the count
    before any window is counted, and nothing is written."""
    body = json.dumps(small).encode()
    with pytest.raises(ValueError, match=refusal):
        _main(tmp_path, small, 1, {"c": 10 ** 12, "k": 10 ** 12, "n": 2}, "x", _registration(body, small, **change))
    assert not (tmp_path / "public-x.json").exists() and not (tmp_path / "private-x.json").exists()


def test_a_window_with_no_registration_is_not_counted(small, tmp_path):
    (tmp_path / "circuit.json").write_text(json.dumps(small))
    (tmp_path / "declaration.json").write_text(json.dumps({"c": 10 ** 12, "k": 10 ** 12, "n": 2}))
    a = BC.Options(day=DAY, declaration=tmp_path / "declaration.json", private=tmp_path / "private.json",
                   window=[[str(tmp_path / "circuit.json"), "-"]])
    with pytest.raises(ValueError, match="circuit.json: no registration is given, so nothing holds the circuit record to "
                                         "the registered circuit, and it is not counted"):
        BC.main(a)
    assert not (tmp_path / "private.json").exists()


def test_days_of_different_counts_under_n_publish_the_same_bytes(small, tmp_path, capsys):
    """One, three and seventeen windows of two slots (2, 6 and 34 blocks) under n = 40 publish byte-identical public
    judgments, which name no count and which the record reads back; only the private file (0600) holds the exact counts."""
    decl = {"c": 10 ** 12, "k": 10 ** 12, "n": 40, "note": "anything else the declaration holds"}
    runs = {w: _run(tmp_path, small, w, decl, str(w)) for w in (1, 3, 17)}
    assert len({pub for pub, _ in runs.values()}) == 1
    doc = json.loads(runs[1][0])
    assert RBC.BlockJudgment.of(doc, "public").dump() == doc and set(doc) == set(RBC.FIELDS)
    assert (doc["verdict"], doc["reason"], doc["checks"]) == ("accept", None, dict.fromkeys(RBC.RULES, "accept"))
    assert all(set(c) == set(RBC.CLASS) for c in doc["classes"])
    for w, (_, private) in runs.items():
        held = json.loads(private.read_text())
        assert (held["total"], held["windows"], private.stat().st_mode & 0o777) == (2 * w, w, 0o600)
    assert "blocks" not in capsys.readouterr().err


def test_a_refusal_past_n_names_n_and_not_the_count(small, tmp_path, capsys):
    """Seventeen windows (34 blocks) against n = 5 refuse on the count rule alone, the reason naming n and not the count,
    padded or exact; the stderr line says the same."""
    pub, private = _run(tmp_path, small, 17, {"c": 10 ** 12, "k": 10 ** 12, "n": 5}, "17")
    doc = json.loads(pub)
    assert doc["checks"] == {"gates": "accept", "fresh": "accept", "count": "refuse"} and doc["verdict"] == "refuse"
    assert doc["reason"] == "the blocks' exact count is more than n = 5"
    err = capsys.readouterr().err
    for count in ("34", str(PV2.grid(34)), "17"):
        assert count not in doc["reason"] and count not in err
    assert json.loads(private.read_text())["total"] == 34


def test_the_judgment_holds_its_fields_alone(small):
    """The public record refuses a count (a class's blocks, the day's total or windows), a window list, and checks or a
    verdict that aren't its classes'; it names no window, run or root."""
    pub, private = BC.judgment(DAY, [(small, {"weights/x@4": "ab" * 64})] * 3, {"c": 10 ** 12, "k": 10 ** 12, "n": 6})
    doc = pub.dump()
    for extra in ({"total": 6}, {"windows": 3}, {"blocks": private["blocks"]}):
        with pytest.raises(UnknownField):
            RBC.BlockJudgment.of({**doc, **extra}, "judgment")
    with pytest.raises(UnknownField):
        RBC.BlockJudgment.of({**doc, "classes": [{**doc["classes"][0], "blocks": 6}]}, "judgment")
    with pytest.raises(RecordError, match="checks.gates"):
        RBC.BlockJudgment.of({**doc, "c": 0}, "judgment")
    with pytest.raises(RecordError, match="verdict"):
        RBC.BlockJudgment.of({**doc, "checks": {**doc["checks"], "count": "refuse"}}, "judgment")
    text = json.dumps(doc)
    assert set(re.findall(r"[0-9a-f]{128}", text)) == {doc["classes"][0]["definition"], doc["declaration"]}
    assert "ab" * 64 not in text and "test/blocks" not in text


def _calling(*fns) -> Program:
    """A circuit calling each of ``fns`` once on parameters of its own."""
    params = tuple((f"{i}.{n}", t) for i, fn in enumerate(fns) for n, t in fn.signature()[0])
    ret = fns[0].signature()[1]

    def body(B, S, *a):
        out, at = None, 0
        for fn in fns:
            k = len(fn.signature()[0])
            got = B.call(fn, *a[at:at + k])
            out, at = got if out is None else out, at + k
        return out
    return Program(bind(CompositeDefinition("BlockCountsCalling", 1, (), lambda S: (params, ret), body, register=False)))


def test_the_gate_table_prices_bit_at_by_its_body():
    """`TokenFeed_v1` reads `BitAtx32_v1`, which the gate table (`count.table`) prices by its body in the circuit,
    `BitAtx32_v2`: 147 gates, every primitive priced, so a feed's gates are its primitives' sum."""
    feed = BC.resolve("TokenFeed_v1{NP=4,FIRST=false}")
    table = K.table(_calling(feed))
    assert table["BitAtx32_v1"] == 147
    g, miss = BC.gates(feed, table)
    assert not miss and g == sum(table.values()) > 147


def test_a_primitive_with_no_boolean_body_is_refused_and_a_table_lacking_one_leaves_it_unpriced(monkeypatch):
    """A word primitive no Boolean Definition names as its word is refused by name when a circuit reaches it (under the
    catalog's lookup alone: the suite's conftest gives a test's own primitive a stand-in body), and a block counted by a table
    that lacks it lists its activations apart, its gates not guessed."""
    monkeypatch.setattr(PG, "_BODY_LOOKUP", [V.boolean_version])
    lonely = PrimitiveDefinition("BlockCountsLonelyWord", 1, (("x", Value(32)),), Value(32), lambda x: x, register=False)
    with pytest.raises(NoBooleanBody, match="BlockCountsLonelyWord_v1"):
        _calling(lonely)
    assert BC.gates(lonely, {}) == (0, {"BlockCountsLonelyWord_v1": 1})


def test_the_wiring_between_leaf_widths_is_priced_at_no_gate():
    """`WordBits{16}` and `BitsWord{32}` only rename wires (`BitOf{W}x{I}`, `WordOf{W}`, whose bodies hold no gate), so the
    table prices each at 0 and an int8 product reading its weights as u16 words prices as its coordinates."""
    assert K.table(_calling(bind(WordBits, W=16), bind(BitsWord, W=32))) == \
        {**{f"BitOf16x{i}_v1": 0 for i in range(16)}, "WordOf32_v1": 0}
    mm = bind(GEMM_OPS.ScaledMmS8V3, K=64, N=8, **GEMM_OPS.GEMM_SM120_S8)
    coordinate = bind(GEMM_OPS.ScaledMmS8Coordinate, K=64, **GEMM_OPS.GEMM_SM120_S8)
    table = K.table(_calling(mm))
    assert BC.gates(mm, table) == (8 * BC.gates(coordinate, table)[0], {}) and mm.gates > 8 * coordinate.gates


def test_a_block_with_an_unpriced_primitive_leaves_the_gates_undecided():
    """Its gates are a lower bound: under c the gates rule is undecided and the day refused, naming the primitive and c; a
    lower bound already over c refuses on the gates."""
    cls = RBC.BlockClass(definition="ab" * 64, slots=2, steps=3, gates=100, unpriced={"BitAtx32_v1": 1}, fresh=7)
    checks, reason = RBC.judge([cls], 2, 10 ** 6, 10, 2)
    assert checks == {"gates": "undecided", "fresh": "accept", "count": "accept"}
    assert reason == ("undecided: a block reaches BitAtx32_v1, which the gate table can't price, so its gates are a lower "
                      "bound against c = 1000000")
    assert RBC.judge([cls], 2, 99, 10, 2) == ({"gates": "refuse", "fresh": "accept", "count": "accept"},
                                             "a block of 100 gates, more than c = 99")


@pytest.mark.slow
def test_on_the_tiny_window_a_blocks_fresh_bits_are_bs_k():
    """TINY's fixed load at its class (2 slots, the class circuit): two blocks a window, each of the session's gates, and fresh bits B's k with the
    weights held (the scan state its lanes' units read)."""
    from verity_vllm.program.predict import inference_blocks as IB
    from verity_vllm.program.predict.window_builder import WEIGHTS
    from verity_vllm.program.predict.window_circuit import class_circuit

    from tests.program.predict import test_steps as S
    from tests.program.predict.test_window_circuit import CLASS, GRAIN

    circuit = class_circuit(S._registered(), CLASS, None, GRAIN)
    doc = IF.descriptor_record(circuit, 64)
    roots = {p: "ab" * 64 for p in circuit.registered.params if p.split("/", 1)[0] == WEIGHTS}
    pub, private = BC.judgment(DAY, [(doc, roots)], {"c": 0, "k": 0, "n": 0}, pins=roots)
    (cls,) = pub.classes
    assert (cls.slots, private["total"], cls.fresh) == (2, 2, IB.budgets(circuit)[1]) and cls.fresh > 0
    assert not cls.unpriced and cls.gates > IB.budgets(circuit)[0]


def test_the_import_alone_loads_no_registry_module():
    """A process that has imported the module and called nothing has loaded no module of `verity_vllm.definitions.registry`, so
    `codec` holds none of the catalog's on-demand families and refuses an id of one (`ON_DEMAND`); `resolve` loads the
    catalog on its first call, and the same id then answers."""
    r = subprocess.run([sys.executable, "-c", ALONE, ON_DEMAND, ROPE], capture_output=True, text=True, check=False,
                       env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-4000:]
    assert json.loads(r.stdout.splitlines()[-1]) == {"loaded": [], "before": None, "after": ON_DEMAND}


def _json_values(rng, depth: int = 0):
    """JSON values of every kind a declaration file can hold, as `json.loads` gives them."""
    leaves = [None, True, False, 0, -1, 10 ** 12, 2 ** 63 + 5, -0.0, 1.5, 1e-7, 1e21, 0.1 + 0.2, "", "note", "é",
              "\u2028", "\U0001f600", "\x00\x1f\x7f", '"\\/']
    k = rng.randrange(8 if depth < 3 else 6)
    if k < 6:
        return rng.choice(leaves)
    n = rng.randrange(4)
    if k == 6:
        return [_json_values(rng, depth + 1) for _ in range(n)]
    return {rng.choice(["c", "k", "n", "note", "Z", "é", "\U0001f600", "a b", ""]) + str(i): _json_values(rng, depth + 1)
            for i in range(n)}


def test_a_declaration_s_digest_is_its_bytes_as_compact_sorted_ascii_json():
    import hashlib
    import random

    rng = random.Random(0)
    docs = [{"c": 10 ** 12, "k": 10 ** 12, "n": 40, "note": "anything else the declaration holds"}, {"c": 0, "k": 0, "n": 0}]
    docs += [{"c": rng.randrange(10 ** 15), "k": rng.randrange(10 ** 15), "n": rng.randrange(10 ** 6),
              **{f"x{i}": _json_values(rng) for i in range(rng.randrange(6))}} for _ in range(500)]
    for doc in docs:
        doc = json.loads(json.dumps(doc))
        want = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
        assert codec.canonical_json(doc) == want
        assert BC.declaration(doc)[0] == hashlib.sha512(want).hexdigest()


@pytest.mark.slow
def test_a_closed_epochs_fresh_bits_are_its_free_inputs_with_every_registered_read_held(tmp_path, monkeypatch):
    """The epoch of class 1,1,1 on the small model, closed and written as the serve's close writes it (`Window.write`,
    which records ``registered_roots`` under the message's ``window``): `main` counts a block's fresh bits with every
    registered read held at the verifier's pins, so they are the declared circuit's free inputs, never those plus the
    registered bits; a message that records no registered roots is refused by name. What no cheaper test reaches: the
    roots where `Window.write` puts them, read by `main`. Slow (minutes) for the epoch's close."""
    from types import SimpleNamespace

    from verity.commitments.periods import interface as IF
    from verity.protocols.accounting.network.prover import scan_state_rows as SR
    from verity.protocols.proof.spot.prover import docket as DK
    from verity_vllm.program.predict import inference_blocks as IB
    from verity_vllm.program.predict import scan_state as SS
    from verity_vllm.program.predict import window_circuit as WC
    from verity_vllm.program.predict import window_rows as WR

    from tests.program.predict.test_window_rows import DESIGN, RUN, _window

    cls, lanes = WC.WindowClass(1, 1, 1), [(1, 1, 1, [17])]
    low, _f, values, _w, slots = _window(cls=cls, lanes=lanes)
    heads = [([occ, lp, *ids], m) for occ, lp, m, ids in lanes]
    state = WR.state_value(IB.program_digest(low.circuit), cls, SR.standin_rows(cls, heads), window=RUN)
    vals = {**values, SS.SCAN_STATE: state}
    win = IF.Window(low.circuit, RUN, DESIGN, 64, registered=vals, forms=WR.FORMS)
    win.supply()
    WR.Rows(low, slots, vals).record(win)
    win.write(win.close(), tmp_path, {}, registrar=DK)
    message = json.loads((tmp_path / "registration.json").read_text())
    doc = json.loads((tmp_path / "circuit.json").read_text())
    roots = IF.registration_roots(message)
    assert roots == message["window"]["registered_roots"] and "registered_roots" not in message
    held = IF.input_count(doc, BC.resolve, pins=roots, roots=roots)
    loose = IF.input_count(doc, BC.resolve, pins=roots, roots={})
    assert held["registered"] > 0 == loose["registered"]
    assert held["free"] + held["ingress"] == loose["free"] + loose["ingress"] - held["registered"]
    (tmp_path / "pins.json").write_text("{}")
    monkeypatch.setattr(BC, "WeightsRegistration", SimpleNamespace(of=lambda doc, name: SimpleNamespace(roots=roots)))
    (tmp_path / "declaration.json").write_text(json.dumps({"c": 10 ** 15, "k": 10 ** 15, "n": 1}))

    def counted(reg: dict) -> dict:
        (tmp_path / "message.json").write_text(json.dumps(reg))
        BC.main(BC.Options(day=DAY, declaration=tmp_path / "declaration.json", private=tmp_path / "private.json",
                           out=tmp_path / "public.json", pins=tmp_path / "pins.json",
                           window=[[str(tmp_path / "circuit.json"), str(tmp_path / "message.json")]]))
        return json.loads((tmp_path / "private.json").read_text())

    private = counted(message)
    assert BC.fresh(doc, 1, roots, roots) == held["free"] + held["ingress"], private
    assert str(held["free"] + held["ingress"]) in json.dumps(private) and str(loose["free"] + loose["ingress"]) not in json.dumps(private)
    bare = {**message, "window": {k: v for k, v in message["window"].items() if k != "registered_roots"}}
    with pytest.raises(ValueError, match=re.escape(IF.NO_REGISTERED_ROOTS)):
        counted(bare)
    print(f"FRESH-COUNT held {held} loose {loose}")
