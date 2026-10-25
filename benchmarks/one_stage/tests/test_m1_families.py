"""`m1_families.py --estimate`: each family's split units (its unit templates past the row itself) sized once however many
families share them, each listed with its count per row; the ``--also`` Definitions (and ``--also-file``'s non-blank lines)
sized as ``candidates``; and `sized`'s verdict on a Definition from its estimate and block, or the reason it has none.
Nothing is lowered: `sized`, `estimate`, `block` and the serving rows' `_bind` are stand-ins."""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import m1_families as M  # noqa: E402

ROW_A, ROW_B, WHOLE = "RowA_v1{N=4}", "RowB_v1{N=2}", "Whole_v1{K=8}"
UNIT, OWN, CAND = "Unit_v1{N=1}", "OwnB_v1{H=1}", "Cand_v1{K=1}"


@pytest.fixture
def calls(monkeypatch):
    """`sized` as a stand-in: each spec's ANDs are its length, and each call is recorded with the memo it was given."""
    seen: list[tuple[str, dict]] = []

    def fake(spec, memo):
        seen.append((spec, memo))
        return {"ands": len(spec)}

    monkeypatch.setattr(M, "sized", fake)
    return seen


def run_sizes(tmp_path, also=()):
    census = {"requests": {"r0": {"families": {
        ROW_A: {"rows": 3, "unit_templates": [[ROW_A, 1], [UNIT, 4]]},
        ROW_B: {"rows": 5, "unit_templates": [[ROW_B, 1], [UNIT, 2], [OWN, 2]]},
        WHOLE: {"rows": 1, "unit_templates": [[WHOLE, 1]]}}}}}
    src, out = tmp_path / "census.json", tmp_path / "sized.json"
    src.write_text(json.dumps(census))
    assert M.sizes(str(src), str(out), list(also)) == 0
    return json.loads(out.read_text())


def test_each_familys_units_are_listed_past_the_row_itself_with_their_count(tmp_path, calls):
    fams = run_sizes(tmp_path)["requests"]["r0"]["families"]
    assert fams[ROW_A]["estimate"] == {"ands": len(ROW_A), "units": [{"spec": UNIT, "n": 4, "ands": len(UNIT)}]}
    assert fams[ROW_B]["estimate"]["units"] == [{"spec": UNIT, "n": 2, "ands": len(UNIT)},
                                                {"spec": OWN, "n": 2, "ands": len(OWN)}]
    assert fams[WHOLE]["estimate"] == {"ands": len(WHOLE), "units": []}


def test_a_unit_two_families_share_is_sized_once_on_one_memo(tmp_path, calls):
    run_sizes(tmp_path)
    specs = [s for s, _m in calls]
    assert sorted(specs) == sorted([ROW_A, ROW_B, WHOLE, UNIT, OWN])
    assert all(m is calls[0][1] for _s, m in calls)


def test_also_definitions_are_sized_as_candidates(tmp_path, calls):
    assert run_sizes(tmp_path)["candidates"] == {}
    assert run_sizes(tmp_path, [CAND])["candidates"] == {CAND: {"ands": len(CAND)}}


@pytest.mark.parametrize("file, also", [(None, ["A_v1{K=1}"]),
                                        ("B_v1{K=2}\n\n   \nC_v1{K=3}\n", ["A_v1{K=1}", "B_v1{K=2}", "C_v1{K=3}"])])
def test_main_sizes_also_then_also_files_non_blank_lines(tmp_path, monkeypatch, file, also):
    got = {}
    monkeypatch.setattr(M, "sizes", lambda census, out, also=(): got.update(census=census, out=out, also=list(also)) or 0)
    argv = ["m1_families.py", "--estimate", "c.json", "--out", "o.json", "--also", "A_v1{K=1}"]
    if file is not None:
        (tmp_path / "specs.txt").write_text(file)
        argv += ["--also-file", str(tmp_path / "specs.txt")]
    monkeypatch.setattr(sys, "argv", argv)
    assert M.main() == 0
    assert got == {"census": "c.json", "out": "o.json", "also": also}


@pytest.fixture
def bound(monkeypatch):
    """`serving_rows._bind` as a table (spec to its Definition, None, or the exception it raises), with `estimate` and
    `block` stand-ins that refuse any Definition but the table's."""
    table: dict = {}

    def bind(spec):
        if isinstance(table[spec], Exception):
            raise table[spec]
        return table[spec]

    srw = types.ModuleType("verity_vllm.pipeline.serving_rows")
    srw._bind = bind
    pipeline = types.ModuleType("verity_vllm.pipeline")
    pipeline.serving_rows = srw
    pkg = types.ModuleType("verity_vllm")
    pkg.pipeline = pipeline
    for name, mod in (("verity_vllm", pkg), ("verity_vllm.pipeline", pipeline), ("verity_vllm.pipeline.serving_rows", srw)):
        monkeypatch.setitem(sys.modules, name, mod)

    def known(fn):
        assert fn is not None and any(fn is d for d in table.values()), "sized a Definition _bind didn't return"

    def estimate(fn, memo, leaves):
        known(fn)
        return {"ands": table["ands"]}

    def block(fn, ands):
        known(fn)
        return {"ands": ands, "est_log": table["est_log"], "rows_log": table["rows_log"]}

    monkeypatch.setattr(M, "estimate", estimate)
    monkeypatch.setattr(M, "block", block)
    return table


def test_sized_says_a_spec_that_does_not_bind(bound):
    bound["X_v1{}"] = None
    e = M.sized("X_v1{}", {})
    assert e["error"] == "does not bind" and "ands" not in e and "block" not in e


def test_sized_reports_a_failure_cut_to_300_characters(bound):
    bound["X_v1{}"] = ValueError("x" * 400)
    e = M.sized("X_v1{}", {})
    assert e["error"] == ("ValueError: " + "x" * 400)[:300] and "ands" not in e


@pytest.mark.parametrize("ands, log2, under", [(2**26 - 1, round(math.log2(2**26 - 1), 2), True), (2**26, 26.0, False),
                                               (0, None, True)])
@pytest.mark.parametrize("est_log, fits", [(27, True), (28, False)])
@pytest.mark.parametrize("rows_log, refused", [(27, False), (28, True)])
def test_sized_verdicts(bound, ands, log2, under, est_log, fits, rows_log, refused):
    bound.update({"X_v1{}": object(), "ands": ands, "est_log": est_log, "rows_log": rows_log})
    e = M.sized("X_v1{}", {})
    assert {k: e[k] for k in ("ands", "log2_ands", "under_2^26", "block", "fits_block", "refused_block")} == {
        "ands": ands, "log2_ands": log2, "under_2^26": under,
        "block": {"ands": ands, "est_log": est_log, "rows_log": rows_log}, "fits_block": fits, "refused_block": refused}
