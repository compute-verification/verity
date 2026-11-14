"""difftest.py (protected): the one-call evaluator agrees with a numpy reference on BiasAdd (bf16 add, RNE), is
deterministic and refuses operands that are not its parameters; the word generators cover the required edge kinds in
bf16 and in f32, give integers below one bound, and refuse what they have no generator for; `check` fails when an edge
kind is missing, takes an input that names no kind as random, passes one case that passes and fails a directory of none,
reports a case it cannot compare as an error, and seals its report as the property `difftest`'s record; `produce` takes a
case's explicit words and a saved array's, carries a word of each dtype it names to the operation and back, and
writes its manifest and each case into directories it makes; and what `main` hands `produce`, writes for a `check`, prints
and exits with."""
from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from verity.circuits.defs import bind
from verity.definitions.registry import b1

from verity_vllm.pipeline import cli
from verity_vllm.properties import admission as difftest
from verity_vllm.records.property_record import PropertyRecord


def _bf16_add_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    fa = (a.astype(np.uint32) << 16).view(np.float32)
    fb = (b.astype(np.uint32) << 16).view(np.float32)
    s = (fa + fb).astype(np.float32)
    # round-to-nearest-even to bf16
    u = s.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


def test_evaluate_spec_bias_add_matches_numpy_reference():
    rng = np.random.default_rng(0)
    N = 8
    y = difftest.words_for("random", [N], "bfloat16", rng)
    b = difftest.words_for("random", [N], "bfloat16", rng)
    spec = bind(b1.BiasAdd, N=N)
    got = difftest.evaluate_spec(spec, [y, b])
    exp = _bf16_add_ref(y, b)
    finite = ~np.isin(exp & 0x7F80, [0x7F80])
    assert np.array_equal(np.array(got, np.uint16)[finite], exp[finite])
    assert got == difftest.evaluate_spec(spec, [y, b])


def test_evaluate_spec_refuses_operands_that_are_not_its_parameters() -> None:
    spec = bind(b1.BiasAdd, N=4)
    y = difftest.words_for("random", [4], "bfloat16", np.random.default_rng(0))
    for operands, why in (([y], "2 parameters, 1 operands"), ([y, y, y], "2 parameters, 3 operands"),
                          ([y, y[:3]], "parameter b has 4 leaves, operand has 3 words"),
                          ([np.append(y, y[:1]), y], "parameter y has 4 leaves, operand has 5 words")):
        with pytest.raises(ValueError) as refused:
            difftest.evaluate_spec(spec, operands)
        assert str(refused.value) == f"{spec.id}: {why}"
    assert difftest.evaluate_spec(spec, [y.reshape(2, 2), y]) == difftest.evaluate_spec(spec, [y, y])      # leaves are counted, not shapes


@pytest.mark.parametrize("kind", difftest.REQUIRED_EDGE_KINDS + ("random",))
def test_word_generators(kind):
    w = difftest.words_for(kind, [3, 5], "bfloat16", np.random.default_rng(1))
    assert w.dtype == np.uint16 and w.shape == (3, 5)
    if kind == "subnormal":
        assert np.all((w & 0x7F80) == 0) and np.all((w & 0x007F) != 0)
    if kind == "zeros":
        assert np.all((w & 0x7FFF) == 0)


def test_the_bf16_zeros_are_of_both_signs() -> None:
    w = difftest.words_for("zeros", [3, 5], "bfloat16", np.random.default_rng(1))
    assert w.reshape(-1).tolist() == [0x8000 if i % 7 == 0 else 0 for i in range(15)]                      # -0 at every seventh word, +0 between


class _Largest:
    """In a generator's place: every integer it gives is the largest its bounds allow."""

    def integers(self, low: int, high: int, n: int) -> np.ndarray:
        return np.full(n, high - 1)


_LARGEST = cast(np.random.Generator, _Largest())                                                           # a `Generator` to mypy; the calls below reach its `integers` alone


@pytest.mark.parametrize("kind", difftest.REQUIRED_EDGE_KINDS + ("random",))
def test_word_generators_of_f32(kind: str) -> None:
    w = difftest.words_for(kind, [8, 8], "float32", np.random.default_rng(1))
    assert w.dtype == np.uint32 and w.shape == (8, 8)
    assert np.array_equal(w, difftest.words_for(kind, [8, 8], "float32", np.random.default_rng(1)))
    f = w.reshape(-1).view(np.float32)
    assert np.all(np.isfinite(f))
    if kind == "random":
        assert f.min() < 0 < f.max()
    if kind == "zeros":
        assert not w.any()
    if kind == "max":
        assert set(w.reshape(-1).tolist()) == {0x7F7FFFFF, 0xFF7FFFFF, 0x43000000}                        # ±max finite, 128
    if kind == "subnormal":
        assert np.all(w != 0) and np.all(w < 0x00800000)
        assert np.all(difftest.words_for(kind, [3], "float32", _LARGEST) == 0x007FFFFF)
    if kind == "alternating":
        assert np.all((f[0::2] <= -1) & (f[0::2] >= -2)) and np.all((f[1::2] >= 1) & (f[1::2] <= 2))
        bf16 = difftest.words_for(kind, [8, 8], "bfloat16", np.random.default_rng(1))                      # the same values, cut to bf16
        assert np.array_equal(w >> 16, bf16)


@pytest.mark.parametrize("dtype, words, largest", [("int32", np.uint32, 2**31 - 2), ("int64", np.uint64, 2**62 - 1)])
def test_word_generators_of_integers_stay_below_one_bound_whatever_the_kind_but_zeros(dtype: str, words: type, largest: int) -> None:
    for kind in ("random", "max", "subnormal", "alternating"):
        w = difftest.words_for(kind, [2, 3], dtype, _LARGEST)
        assert w.dtype == words and w.shape == (2, 3) and np.all(w == largest)
    z = difftest.words_for("zeros", [2, 3], dtype, _LARGEST)
    assert z.dtype == words and z.shape == (2, 3) and not z.any()
    one = difftest.words_for("random", [], dtype, np.random.default_rng(1))                                # no shape: one word
    assert one.dtype == words and one.shape == () and one <= largest


@pytest.mark.parametrize("kind, dtype", [("random", "float16"), ("random", "uint8"), ("tiny", "bfloat16"), ("tiny", "float32")])
def test_word_generators_refuse_a_kind_or_a_dtype_they_have_no_generator_for(kind: str, dtype: str) -> None:
    with pytest.raises(ValueError) as refused:
        difftest.words_for(kind, [2], dtype, np.random.default_rng(1))
    assert str(refused.value) == f"no generator for kind={kind} dtype={dtype}"


def test_check_requires_edge_kinds(tmp_path, monkeypatch):
    # a fake adapter for BiasAdd whose cases are random-only -> admission fails for missing edge kinds
    mod = types.ModuleType("fake_adapter")
    vars(mod).update(
        SEMANTIC_ID="test/BiasAdd@v1", spec=lambda st: bind(b1.BiasAdd, N=st["N"]),
        cases=lambda rng, i: {"statics": {"N": 4}, "inputs": [{"shape": [4], "dtype": "bfloat16", "kind": "random"}] * 2})
    monkeypatch.setitem(sys.modules, "fake_adapter", mod)
    d = tmp_path / "dt"
    (d / "case_0").mkdir(parents=True)
    rng = np.random.default_rng(3)
    y, b = difftest.words_for("random", [4], "bfloat16", rng), difftest.words_for("random", [4], "bfloat16", rng)
    np.save(d / "case_0" / "in_0.npy", y)
    np.save(d / "case_0" / "in_1.npy", b)
    np.save(d / "case_0" / "out_0.npy", _bf16_add_ref(y, b))
    rec = {"i": 0, "statics": {"N": 4}, "inputs": [{"shape": [4], "dtype": "bfloat16", "kind": "random"}] * 2,
           "outputs": [{"shape": [4], "dtype": "bfloat16"}], "edge": False}
    (d / "manifest.json").write_text(json.dumps({"seed": 3, "n": 1, "cases": [rec]}))
    rep = difftest.check("fake_adapter", d)
    assert rep["differential"] == {"passed": 1, "total": 1}
    assert rep["status"] == "fail" and set(rep["edge_kinds_missing"]) == set(difftest.REQUIRED_EDGE_KINDS)


def test_produce_writes_bf16_outputs_as_words(tmp_path, monkeypatch):
    # a production op returning bfloat16 must cross to numpy as uint16 words (numpy has no bf16;
    # `Tensor.numpy()` raised "Got unsupported ScalarType BFloat16" and every bf16-output Definition was un-difftestable)
    pytest.importorskip("torch")
    _whole_adapter(monkeypatch, "fake_bf16_adapter")
    man = difftest.produce("fake_bf16_adapter", 1, 3, tmp_path / "dt", device="cpu")
    out = np.load(tmp_path / "dt" / "case_0" / "out_0.npy")
    assert man["cases"][0]["outputs"] == [{"shape": [4], "dtype": "bfloat16"}] and out.dtype == np.uint16 and out.shape == (4,)
    y, b = (np.load(tmp_path / "dt" / "case_0" / f"in_{i}.npy") for i in (0, 1))
    assert np.array_equal(out, _bf16_add_ref(y, b))


def _adapter(monkeypatch: pytest.MonkeyPatch, name: str, **names: object) -> None:
    """A BiasAdd adapter importable as `name`: its `spec`, and the `names` given."""
    mod = types.ModuleType(name)
    vars(mod).update(spec=lambda st: bind(b1.BiasAdd, N=st["N"]), **names)
    monkeypatch.setitem(sys.modules, name, mod)


def _whole_adapter(monkeypatch: pytest.MonkeyPatch, name: str, inputs: list[list[dict[str, object]]] | None = None, *,
                   edge: dict[int, bool] | None = None, op: Callable[[list[Any]], list[Any]] = lambda ins: [ins[0] + ins[1]]) -> None:
    """The BiasAdd adapter whole, importable as `name`: `op` as the pinned operation, torch's own sum where none is given, on
    the operands `inputs` gives each case, two random bf16 ones where it gives none.  A case in `edge` says itself whether
    it is an edge case."""
    two = [{"shape": [4], "dtype": "bfloat16", "kind": "random"}] * 2
    _adapter(monkeypatch, name, SEMANTIC_ID="test/BiasAdd@v1", production=lambda st, ins: op(ins),
             cases=lambda rng, i: {"statics": {"N": 4}, "inputs": inputs[i] if inputs else two, **({"edge": edge[i]} if edge and i in edge else {})})


def _produced(d: Path, kinds: tuple[str, ...], wrong: tuple[int, ...] = ()) -> Path:
    """A directory as `produce` leaves it for a BiasAdd adapter, one case a kind.  A case's output is the evaluator's own words,
    with the lowest bit of the first turned in the cases `wrong`."""
    cases = []
    for i, kind in enumerate(kinds):
        rng = np.random.default_rng(i)
        y, b = difftest.words_for(kind, [4], "bfloat16", rng), difftest.words_for(kind, [4], "bfloat16", rng)
        out = np.array(difftest.evaluate_spec(bind(b1.BiasAdd, N=4), [y, b]), np.uint16)
        if i in wrong:
            out[0] ^= 1
        (d / f"case_{i}").mkdir(parents=True)
        for name, words in (("in_0", y), ("in_1", b), ("out_0", out)):
            np.save(d / f"case_{i}" / f"{name}.npy", words)
        cases.append({"i": i, "statics": {"N": 4}, "inputs": [{"shape": [4], "dtype": "bfloat16", "kind": kind}] * 2,
                      "outputs": [{"shape": [4], "dtype": "bfloat16"}]})
    (d / "manifest.json").write_text(json.dumps({"seed": 3, "n": len(cases), "cases": cases}))
    return d


def _check(adapter: str, d: Path, out: Path, *more: str) -> int:
    return difftest.main(cli.parse("properties-admission", ["check", "--adapter", adapter, "--dir", str(d), "--out", str(out), *more], {}))


def test_produce_takes_a_cases_explicit_words_and_the_words_of_a_saved_array(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    rows = np.array([[1.5, -2.0, 0.25, 8.0], [3.0, 0.5, -0.125, 1.0]], np.float32)
    np.save(tmp_path / "rows.npy", rows)
    words = [0x3F800000, 0xBF000000, 0x40400000, 0x00000000]                                               # 1, -0.5, 3, 0
    given: dict[str, object] = {"shape": [4], "dtype": "float32", "words": words}
    saved: dict[str, object] = {"shape": [2, 4], "dtype": "float32", "npy": str(tmp_path / "rows.npy")}
    inputs: list[list[dict[str, object]]] = [
        [given, {**saved, "shape": [4], "row": 1}], [saved, {"shape": [2, 4], "dtype": "float32", "words": words * 2}],
        [{"shape": [4], "dtype": "float32", "kind": "zeros"}, given]]
    _whole_adapter(monkeypatch, "given_words_adapter", inputs)
    man = difftest.produce("given_words_adapter", 3, 3, tmp_path / "dt", device="cpu")

    def case(i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y, b, out = (np.load(tmp_path / "dt" / f"case_{i}" / f"{name}.npy") for name in ("in_0", "in_1", "out_0"))
        return y, b, out

    as_given = np.array(words, np.uint32)
    for i, (y, b) in enumerate([(as_given, rows[1].view(np.uint32)), (rows.view(np.uint32), np.array(words * 2, np.uint32).reshape(2, 4)),
                                (np.zeros(4, np.uint32), as_given)]):
        got = case(i)
        assert [a.dtype for a in got] == [np.uint32] * 3 and [a.shape for a in got] == [y.shape] * 3
        assert np.array_equal(got[0], y) and np.array_equal(got[1], b)
        assert np.array_equal(got[2], (y.view(np.float32) + b.view(np.float32)).view(np.uint32))
    assert [(c["i"], c["inputs"], c["outputs"], c["edge"]) for c in man["cases"]] == [
        (0, inputs[0], [{"shape": [4], "dtype": "float32"}], False), (1, inputs[1], [{"shape": [2, 4], "dtype": "float32"}], False),
        (2, inputs[2], [{"shape": [4], "dtype": "float32"}], True)]                                        # an edge case: a kind that is not random
    assert json.loads((tmp_path / "dt" / "manifest.json").read_text())["cases"] == man["cases"]
    assert [json.loads((tmp_path / "dt" / f"case_{i}" / "case.json").read_text()) for i in range(3)] == man["cases"]


def test_produce_writes_its_manifest_and_each_case_into_directories_it_makes_and_takes_no_kind_as_random(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    bare: dict[str, object] = {"shape": [4], "dtype": "bfloat16"}                                          # names no kind
    named: dict[str, object] = {**bare, "kind": "random"}
    zeros: dict[str, object] = {**bare, "kind": "zeros"}
    _whole_adapter(monkeypatch, "bare_adapter", [[bare, bare], [bare, bare], [zeros, bare]], edge={1: True, 2: False})
    _whole_adapter(monkeypatch, "named_adapter", [[named, named], [named, named], [zeros, named]])
    readings = iter(range(64))
    monkeypatch.setattr(difftest, "time", types.SimpleNamespace(time=lambda: next(readings) / 4))          # a quarter second a reading
    d = tmp_path / "not" / "there" / "yet"
    man = difftest.produce("bare_adapter", 3, 3, d, device="cpu")
    assert list(man) == ["schema", "adapter", "semantic_id", "seed", "n", "torch", "device", "cases"]
    assert [man[k] for k in ("schema", "adapter", "semantic_id", "seed", "n", "device")] == [
        "verity-gen/difftest-produce/v1", "bare_adapter", "test/BiasAdd@v1", 3, 3, "cpu"]
    assert [list(c) for c in man["cases"]] == [["i", "statics", "inputs", "outputs", "seconds", "edge"]] * 3
    assert [(c["i"], c["statics"], c["seconds"], c["edge"]) for c in man["cases"]] == [
        (0, {"N": 4}, 0.25, False), (1, {"N": 4}, 0.25, True), (2, {"N": 4}, 0.25, False)]                 # a case's own word, over its kinds'
    by_name = difftest.produce("named_adapter", 3, 3, tmp_path / "named", device="cpu")
    assert [c["edge"] for c in by_name["cases"]] == [False, False, True]
    for i in range(3):
        for name in ("in_0", "in_1", "out_0"):
            assert np.array_equal(np.load(d / f"case_{i}" / f"{name}.npy"), np.load(tmp_path / "named" / f"case_{i}" / f"{name}.npy"))
    assert difftest.produce("bare_adapter", 3, 3, d, device="cpu") == man                                  # into the directories that are there now


def test_produce_carries_a_word_of_each_dtype_it_names_to_the_operation_and_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    words = {"float16": [0x3C00, 0xC000], "int32": [7, 2**32 - 1], "int64": [7, 2**64 - 1], "uint8": [0, 255], "int8": [1, 0x80], "bool": [0, 1]}
    inputs: list[list[dict[str, object]]] = [[{"shape": [2], "dtype": dtype, "words": w} for dtype, w in words.items()]]
    handed: list[list[str]] = []

    def same(ins: list[Any]) -> list[Any]:
        handed.append([str(t.dtype) for t in ins])
        return list(ins)

    _whole_adapter(monkeypatch, "each_dtype_adapter", inputs, op=same)
    man = difftest.produce("each_dtype_adapter", 1, 3, tmp_path / "dt", device="cpu")
    assert handed == [[f"torch.{dtype}" for dtype in words]]
    assert man["cases"][0]["outputs"] == [{"shape": [2], "dtype": dtype} for dtype in words]
    for j, (dtype, w) in enumerate(words.items()):
        for name in (f"in_{j}", f"out_{j}"):
            got = np.load(tmp_path / "dt" / "case_0" / f"{name}.npy")
            assert got.dtype == np.dtype(f"uint{difftest.DTYPE_BITS[dtype]}") and got.tolist() == w


def test_check_takes_an_input_that_names_no_kind_as_random(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _adapter(monkeypatch, "kindless_adapter", SEMANTIC_ID="test/BiasAdd@v1")
    d = _produced(tmp_path / "dt", ("zeros", "random"))
    man = json.loads((d / "manifest.json").read_text())
    man["cases"][1]["inputs"] = [{"shape": [4], "dtype": "bfloat16"}] * 2
    (d / "manifest.json").write_text(json.dumps(man))
    rep = difftest.check("kindless_adapter", d)
    assert [c["kinds"] for c in rep["cases"]] == [["zeros", "zeros"], ["random", "random"]]
    assert rep["differential"] == {"passed": 2, "total": 2} and rep["edge_kinds_missing"] == ["max", "subnormal", "alternating"]
    assert (rep["status"], rep["reason"]) == ("fail", "edge-kind inputs missing: ['max', 'subnormal', 'alternating']")


def test_check_passes_one_case_that_passes_and_fails_a_directory_of_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _adapter(monkeypatch, "lone_adapter", SEMANTIC_ID="test/BiasAdd@v1")
    monkeypatch.setattr(difftest, "REQUIRED_EDGE_KINDS", ())                                               # no kind is asked for: the count alone decides
    (tmp_path / "none").mkdir()
    one = difftest.check("lone_adapter", _produced(tmp_path / "one", ("random",)))
    none = difftest.check("lone_adapter", _produced(tmp_path / "none", ()))
    assert (one["differential"], one["status"], one["ok"]) == ({"passed": 1, "total": 1}, "pass", True)
    assert (none["differential"], none["status"], none["ok"], none["cases"]) == ({"passed": 0, "total": 0}, "fail", False, [])
    assert "reason" not in one and "reason" not in none


def test_check_reports_a_case_it_cannot_compare_as_an_error_and_goes_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _adapter(monkeypatch, "erring_adapter", SEMANTIC_ID="test/BiasAdd@v1")
    d = _produced(tmp_path / "dt", difftest.REQUIRED_EDGE_KINDS)
    np.save(d / "case_1" / "out_0.npy", np.load(d / "case_1" / "out_0.npy")[:3])                           # an output one word short
    (d / "case_2" / "in_1.npy").unlink()                                                                   # an operand's file gone
    man = json.loads((d / "manifest.json").read_text())
    man["cases"][3]["inputs"] = man["cases"][3]["inputs"][:1]                                              # one operand for two parameters
    (d / "manifest.json").write_text(json.dumps(man))
    rep = difftest.check("erring_adapter", d)
    first, short, gone, lone = rep["cases"]
    assert (first["status"], first["n_words"], first["n_diff"]) == ("pass", 4, 0)
    assert short == {"i": 1, "statics": {"N": 4}, "kinds": ["max", "max"], "status": "error", "reason": short["reason"]}
    assert short["reason"].startswith("ValueError: 4 return encodings vs 3 ") and short["reason"].endswith(" words")
    assert gone == {"i": 2, "statics": {"N": 4}, "kinds": ["subnormal", "subnormal"], "status": "error", "reason": gone["reason"]}
    assert gone["reason"].startswith("FileNotFoundError: ") and "in_1.npy" in gone["reason"]
    assert lone == {"i": 3, "statics": {"N": 4}, "kinds": ["alternating"], "status": "error",
                    "reason": f"ValueError: {bind(b1.BiasAdd, N=4).id}: 2 parameters, 1 operands"}
    assert rep["differential"] == {"passed": 1, "total": 4} and rep["edge_kinds_missing"] == [] and "reason" not in rep
    assert (rep["status"], rep["ok"]) == ("fail", False)


def test_check_seals_its_report_as_the_record_of_the_property_difftest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _adapter(monkeypatch, "sealed_adapter", SEMANTIC_ID="test/BiasAdd@v1")
    held = PropertyRecord(difftest.check("sealed_adapter", _produced(tmp_path / "held", difftest.REQUIRED_EDGE_KINDS)))
    broke = PropertyRecord(difftest.check("sealed_adapter", _produced(tmp_path / "broke", difftest.REQUIRED_EDGE_KINDS, wrong=(2,))))
    assert (held.name, held.ok, held.problems()) == ("difftest", True, [])
    assert (broke.name, broke.ok, broke.problems()) == ("difftest", False, [])
    assert list(held.doc) == list(broke.doc) == [
        "schema", "adapter", "semantic_id", "produce", "cases", "status", "differential", "edge_kinds_missing", "ok", "digest"]
    assert (held.doc["adapter"], held.doc["semantic_id"], held.doc["produce"]) == (
        "sealed_adapter", "test/BiasAdd@v1", {"seed": 3, "n": 4, "torch": None, "device": None})
    assert [(c["status"], c["n_diff"], c["first_diff"]) for c in broke.doc["cases"]] == [
        ("pass", 0, []), ("pass", 0, []), ("fail", 1, [0]), ("pass", 0, [])]


def test_main_check_writes_the_admission_and_exits_0_where_its_report_passes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _adapter(monkeypatch, "passing_adapter", SEMANTIC_ID="test/BiasAdd@v1")
    d = _produced(tmp_path / "dt", difftest.REQUIRED_EDGE_KINDS)
    checked: list[tuple[str, Path, dict[str, object]]] = []
    check = difftest.check

    def recorded(adapter: str, d: Path) -> dict[str, object]:
        checked.append((adapter, d, check(adapter, d)))
        return checked[-1][2]

    monkeypatch.setattr(difftest, "check", recorded)
    out = tmp_path / "run" / "gates" / "quarantine.json"
    assert _check("passing_adapter", d, out) == 0
    (adapter, read, rep), = checked
    assert (adapter, read) == ("passing_adapter", d) and (rep["status"], rep["ok"]) == ("pass", True)
    doc = json.loads(out.read_text())
    assert list(doc) == ["schema", "definitions", "differential", "status"]
    assert doc == {"schema": "verity-gen/quarantine/v1", "definitions": {"test/BiasAdd@v1": rep},
                   "differential": {"passed": 4, "total": 4}, "status": "pass"}
    assert out.read_text() == json.dumps(doc, indent=1) + "\n"
    printed = capsys.readouterr()
    assert printed.out == "" and printed.err == "[difftest] pass: 4/4 cases bit-exact for test/BiasAdd@v1\n"


def test_main_check_appends_to_an_admission_only_where_asked_and_exits_by_its_own_report(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _adapter(monkeypatch, "first_adapter", SEMANTIC_ID="test/First@v1")
    _adapter(monkeypatch, "second_adapter", SEMANTIC_ID="test/Second@v1")
    passing = _produced(tmp_path / "passing", difftest.REQUIRED_EDGE_KINDS)
    failing = _produced(tmp_path / "failing", (*difftest.REQUIRED_EDGE_KINDS, "random"), wrong=(4,))
    out = tmp_path / "quarantine.json"

    def written() -> tuple[list[str], dict[str, int], str]:
        doc = json.loads(out.read_text())
        return list(doc["definitions"]), doc["differential"], doc["status"]

    assert _check("first_adapter", passing, out, "--append") == 0       # nothing to append to: a new document
    assert written() == (["test/First@v1"], {"passed": 4, "total": 4}, "pass")
    assert _check("second_adapter", failing, out, "--append") == 1
    assert written() == (["test/First@v1", "test/Second@v1"], {"passed": 8, "total": 9}, "fail")
    assert _check("first_adapter", passing, out, "--append") == 0       # its own report passes; the document still fails
    assert written() == (["test/First@v1", "test/Second@v1"], {"passed": 8, "total": 9}, "fail")
    assert _check("second_adapter", failing, out) == 1                  # without --append the file is written over
    assert written() == (["test/Second@v1"], {"passed": 4, "total": 5}, "fail")
    printed = capsys.readouterr()
    assert printed.out == "" and printed.err.split("\n") == [
        "[difftest] pass: 4/4 cases bit-exact for test/First@v1", "[difftest] fail: 4/5 cases bit-exact for test/Second@v1",
        "[difftest] pass: 4/4 cases bit-exact for test/First@v1", "[difftest] fail: 4/5 cases bit-exact for test/Second@v1", ""]


def test_main_check_files_an_adapter_with_no_semantic_id_under_its_module_and_prints_the_reports_reason(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _adapter(monkeypatch, "nameless_adapter")
    out = tmp_path / "quarantine.json"
    assert _check("nameless_adapter", _produced(tmp_path / "dt", ("random",)), out) == 1
    doc = json.loads(out.read_text())
    assert (list(doc["definitions"]), doc["differential"], doc["status"]) == (["nameless_adapter"], {"passed": 1, "total": 1}, "fail")
    assert doc["definitions"]["nameless_adapter"]["semantic_id"] is None
    printed = capsys.readouterr()
    assert printed.out == "" and printed.err == (
        "[difftest] fail: 1/1 cases bit-exact for None; edge-kind inputs missing: ['zeros', 'max', 'subnormal', 'alternating']\n")


def test_main_produce_hands_produce_the_command_lines_values_and_its_defaults(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    produced: list[tuple[object, ...]] = []

    def produce(*args: object) -> dict[str, object]:
        produced.append(args)
        return {"semantic_id": "test/BiasAdd@v1", "cases": [{}] * len(produced)}

    monkeypatch.setattr(difftest, "produce", produce)
    argv = ["produce", "--adapter", "an_adapter", "--out", "DIR"]
    assert difftest.main(cli.parse("properties-admission", argv, {})) == 0
    assert difftest.main(cli.parse("properties-admission", [*argv, "--n", "2", "--seed", "3", "--device", "cpu"], {})) == 0
    assert produced == [("an_adapter", 32, 7, Path("DIR"), "cuda"), ("an_adapter", 2, 3, Path("DIR"), "cpu")]
    printed = capsys.readouterr()
    assert printed.out == "" and printed.err.split("\n") == [
        "[difftest] produced 1 cases for test/BiasAdd@v1 -> DIR", "[difftest] produced 2 cases for test/BiasAdd@v1 -> DIR", ""]


def test_main_produce_writes_the_cases_that_main_check_then_reads(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("torch")
    _whole_adapter(monkeypatch, "whole_adapter")
    d, out = tmp_path / "dt", tmp_path / "quarantine.json"
    argv = ["produce", "--adapter", "whole_adapter", "--n", "2", "--seed", "3", "--out", str(d), "--device", "cpu"]
    assert difftest.main(cli.parse("properties-admission", argv, {})) == 0
    man = json.loads((d / "manifest.json").read_text())
    assert (man["adapter"], man["seed"], man["n"], man["device"], len(man["cases"])) == ("whole_adapter", 3, 2, "cpu", 2)
    assert _check("whole_adapter", d, out) == 1
    rep = json.loads(out.read_text())["definitions"]["test/BiasAdd@v1"]
    assert rep["produce"] == {"seed": 3, "n": 2, "torch": man["torch"], "device": "cpu"}
    assert [(c["status"], c["n_diff"]) for c in rep["cases"]] == [("pass", 0), ("pass", 0)]
    printed = capsys.readouterr()
    assert printed.out == "" and printed.err.split("\n") == [
        f"[difftest] produced 2 cases for test/BiasAdd@v1 -> {d}",
        "[difftest] fail: 2/2 cases bit-exact for test/BiasAdd@v1; edge-kind inputs missing: ['zeros', 'max', 'subnormal', 'alternating']",
        ""]
