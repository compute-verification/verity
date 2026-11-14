"""difftest.py (protected): the one-call evaluator agrees with a numpy reference on BiasAdd (bf16 add, RNE) and is
deterministic; the word generators cover the required edge kinds; `check` fails when an edge kind is missing; and what
`main` hands `produce`, writes for a `check`, prints and exits with."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from verity.circuits.defs import bind
from verity.definitions.registry import b1

from verity_vllm.pipeline import cli
from verity_vllm.properties import admission as difftest


def _bf16_add_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    fa = (a.astype(np.uint32) << 16).view(np.float32)
    fb = (b.astype(np.uint32) << 16).view(np.float32)
    s = (fa + fb).astype(np.float32)
    # round-to-nearest-even to bf16
    u = s.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) >> 16
    return u.astype(np.uint16)


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


@pytest.mark.parametrize("kind", difftest.REQUIRED_EDGE_KINDS + ("random",))
def test_word_generators(kind):
    w = difftest.words_for(kind, [3, 5], "bfloat16", np.random.default_rng(1))
    assert w.dtype == np.uint16 and w.shape == (3, 5)
    if kind == "subnormal":
        assert np.all((w & 0x7F80) == 0) and np.all((w & 0x007F) != 0)
    if kind == "zeros":
        assert np.all((w & 0x7FFF) == 0)


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
    mod = types.ModuleType("fake_bf16_adapter")
    vars(mod).update(
        SEMANTIC_ID="test/BiasAdd@v1", spec=lambda st: bind(b1.BiasAdd, N=st["N"]), production=lambda st, ins: [ins[0] + ins[1]],
        cases=lambda rng, i: {"statics": {"N": 4}, "inputs": [{"shape": [4], "dtype": "bfloat16", "kind": "random"}] * 2})
    monkeypatch.setitem(sys.modules, "fake_bf16_adapter", mod)
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
    _adapter(monkeypatch, "whole_adapter", SEMANTIC_ID="test/BiasAdd@v1", production=lambda st, ins: [ins[0] + ins[1]],
             cases=lambda rng, i: {"statics": {"N": 4}, "inputs": [{"shape": [4], "dtype": "bfloat16", "kind": "random"}] * 2})
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
