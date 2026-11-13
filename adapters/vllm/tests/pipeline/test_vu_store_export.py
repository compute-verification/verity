"""What an export writes (`verity_vllm.pipeline.vu_store.export_vus`) on the replay's honest test world, with the export's clock the
test's own: each sampled unit's line in its Program's units and in the list, the Program's file, the store's run file, the values and
the summary; and what it writes of a unit it does not store."""
from __future__ import annotations

import dataclasses
import json
import types
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from tests.check.test_sampled_replay import _programs, _weights
from tests.pipeline.test_vu_export import ROOT, _world
from verity_vllm.pipeline import vu_export as VX
from verity_vllm.pipeline import vu_store as VS

SAMPLED = ("i", "family", "spec", "stratum", "phase")
REPLAYED = ("engine_step", "step", "op_path", "row", "result", "why", "inputs")
UNIT = ["vu", "request", "i", "family", "spec", "stratum", "engine_step", "step", "op_path", "row", "phase", "result", "why", "inputs"]
LISTED = [*(k for k in UNIT if k != "step"), "stored"]
BUILT, EACH, EXTRACTED = 8.0625, 0.03125, 2.0625


@pytest.fixture(scope="module")
def world():
    return _world(attn=True)


@dataclasses.dataclass
class Exported:
    summary: dict[str, Any]
    pool: list[dict[str, Any]]
    picked: list[dict[str, Any]]
    replayed: list[dict[str, Any]]
    printed: list[str]
    out: Path

    @property
    def units(self) -> list[dict[str, Any]]:
        return [json.loads(x) for p in sorted((self.out / "store" / "programs").glob("*/units.jsonl")) for x in p.read_text().splitlines()]

    @property
    def listed(self) -> list[dict[str, Any]]:
        return [json.loads(x) for x in (self.out / "vus.jsonl").read_text().splitlines()]

    def values_named(self) -> set[str]:
        named = [a.get("sample", a) for u in self.units for ev in u.get("evaluations", ()) for a in [*ev["args"], *ev["outs"].values()]]
        return {f"{a['value']}.{a['dtype']}" for a in named}


def _export(world, out: Path, monkeypatch, capsys, *, limits: VX.Limits, replay=None, **kw) -> Exported:
    """An export of the honest world under a clock that stands still but where the test moves it: the population is built at `BUILT`,
    each unit's replay takes `EACH` and the extraction `EXTRACTED`.  `replay(k, record)`: the record unit `k`'s replay ends with."""
    now = [0.0]
    seen: dict[str, Any] = {"replayed": []}
    select, capture, extract = VX.select, VX.capture, VS.extract

    def selecting(pool, lim, key):
        seen["pool"], seen["picked"], now[0] = list(pool), select(pool, lim, key), BUILT
        return seen["picked"]

    def capturing(*a, **k):
        rec, evals = capture(*a, **k)
        if replay is not None:
            rec = replay(len(seen["replayed"]), rec)
        seen["replayed"].append(rec)
        now[0] += EACH
        return rec, evals

    def extracting(*a, **k):
        got = extract(*a, **k)
        now[0] += EXTRACTED
        return got

    monkeypatch.setattr(VS, "time", types.SimpleNamespace(perf_counter=lambda: now[0]))
    monkeypatch.setattr(VX, "select", selecting)
    monkeypatch.setattr(VX, "capture", capturing)
    monkeypatch.setattr(VS, "extract", extracting)
    bmap, committed = world.bmap_and_reader()
    capsys.readouterr()
    s = VS.export_vus(_programs(world), bmap, committed, _weights(world), out_dir=out, run_root=ROOT, limits=limits, **kw)
    return Exported(s, seen["pool"], seen["picked"], seen["replayed"], capsys.readouterr().out.splitlines(), out)


def _json(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


def test_an_export_writes_each_sampled_unit_once_in_its_programs_units_and_once_in_the_list(world, tmp_path, monkeypatch, capsys):
    limits = VX.Limits(per_family=2, workers=1, gemm_coords_per_row=3)
    e = _export(world, tmp_path, monkeypatch, capsys, limits=limits, provenance={"row": "test"}, program_dirs={"r0": "/builds/r0"},
                executed_prefix={"r1": {"n_exec": 1, "cap": 3}})
    n, digest = len(e.picked), world.inst["digest"]
    assert n == 14 and e.printed == [f"[vu-export] population: 2/2 request Programs, {e.summary['population']['vus']} VUs, 14 sampled, 8 s",
                                     "[vu-export] 0/14 VUs, 0 s"]
    assert [p.name for p in (tmp_path / "store" / "programs").iterdir()] == [digest]
    assert json.loads((tmp_path / "store" / "programs" / digest / "program.json").read_text()) == {
        "program_digest": digest, "requests": ["r0", "r1"], "build_dirs": {"r0": "/builds/r0", "r1": None}}
    assert len(e.units) == len(e.listed) == len(e.replayed) == n
    for k, (unit, listed, vu, rec) in enumerate(zip(e.units, e.listed, e.picked, e.replayed, strict=True)):
        assert list(unit) == [*UNIT, "evaluations", "stored"] and list(listed) == LISTED
        assert unit["vu"] == k and unit["request"] == vu["request_id"] and unit["stored"] is True and unit["result"] is True
        assert {x: unit[x] for x in SAMPLED} == {x: vu[x] for x in SAMPLED}
        assert {x: unit[x] for x in REPLAYED} == _json({x: rec.get(x) for x in REPLAYED})
        assert listed == {x: unit[x] for x in LISTED}
        assert unit["evaluations"] and all(list(ev) == ["family", "statics", "args", "outs"] for ev in unit["evaluations"])
    assert all(u["step"] == 0 for u in e.units if u["request"] == "r1") and any(u["step"] > 0 for u in e.units)
    key, sampled_weights = VX.selection_key(ROOT), 0
    for unit in e.units:
        for j, ev in enumerate(unit["evaluations"]):
            for a in ev["args"]:
                if "sample" in a:
                    index = ("row", unit["request"], unit["i"], j)
                    assert a["sample"]["index"] == list(index) and len(a["sample"]["rows"]) == 3
                    assert a["sample"]["rows"] == VX._Picker(key, index)(a["weight"]["shape"][0], 3)
                    sampled_weights += 1
    assert sampled_weights > 0
    values = {p.name for p in (tmp_path / "store" / "values").iterdir()}
    assert values == e.values_named() and VS.verify_store(tmp_path / "store") == {
        "ok": True, "bad_values": [], "missing_values": 0, "values": len(values)}


def test_an_exports_summary_counts_its_population_its_units_by_family_its_store_and_its_seconds(world, tmp_path, monkeypatch, capsys):
    limits = VX.Limits(per_family=2, workers=1)
    e = _export(world, tmp_path, monkeypatch, capsys, limits=limits, provenance={"row": "test"}, executed_prefix={"r1": {"n_exec": 1, "cap": 3}})
    s, n = e.summary, len(e.picked)
    assert list(s) == ["schema", "provenance", "limits", "population", "by_family", "store", "sets", "wall_s", "timing"]
    assert json.loads((tmp_path / "export.json").read_text()) == _json(s)
    run = json.loads((tmp_path / "store" / "run.json").read_text())
    assert run == {"schema": "vllm-vu-store/v1", "provenance": _json(s["provenance"]), "limits": _json(dataclasses.asdict(limits))}
    assert s["schema"] == VX.EXPORT_SCHEMA and s["limits"] == dataclasses.asdict(limits)
    selection = s["provenance"].pop("selection")
    assert repr(VX.SELECTION_DOMAIN) in selection and s["provenance"] == {
        "row": "test", "run_root": ROOT.hex(), "weights": {"sample": VS.WEIGHT_SAMPLE_RULE, "rows_per_gemm": limits.gemm_coords_per_row}}
    left_out = s["population"]["not_executed"]
    assert left_out > 0 and all(v["request_id"] != "r1" or v["step"] == 0 for v in e.pool)
    assert s["population"] == {"vus": len(e.pool) + left_out, "pool": len(e.pool), "not_executed": left_out, "programs": 2, "programs_built": 2,
                               "programs_past_budget": [], "source": "built here",
                               "pool_by_family": dict(Counter(v["family"] for v in e.pool)), "not_exported_families": VX.NOT_EXPORTED}
    sets, instances = VS.extract(tmp_path / "store", tmp_path / "again")
    assert s["sets"] == _json(sets) and sum(x["n"] for x in sets) == sum(instances.values()) > n
    of_family = Counter(v["family"] for v in e.picked)
    assert list(s["by_family"]) == sorted(of_family) and set(instances) == set(of_family)
    for family, c in s["by_family"].items():
        assert list(c)[1:] == ["replayed_equal", "stored", "instances", "seconds"]
        assert list(c.values()) == [of_family[family], of_family[family], of_family[family], instances[family], round(of_family[family] * EACH, 3)]
    files = [p for p in (tmp_path / "store").rglob("*") if p.is_file()]
    assert s["store"] == {"values": len(list((tmp_path / "store" / "values").iterdir())), "bytes": sum(p.stat().st_size for p in files)}
    assert len(files) == s["store"]["values"] + 3
    population_s, sample_s, extract_s, rule = s["timing"].items()
    assert population_s == ("population_s", round(BUILT, 3)) and extract_s == ("extract_s", round(EXTRACTED, 3)) and rule[0] == "rule"
    assert sample_s[0].endswith("_s") and sample_s[1] == round(n * EACH, 3) and s["wall_s"] == round(BUILT + n * EACH + EXTRACTED, 3)


def test_a_unit_with_a_registered_operand_past_the_word_limit_is_listed_and_counted_and_not_stored(world, tmp_path, monkeypatch, capsys):
    e = _export(world, tmp_path, monkeypatch, capsys, limits=VX.Limits(per_family=2, workers=1, max_row_words=8))
    unstored = {"RMSNormFusedCuda_v2", "RMSNormTriton_v1", "RoPE_v1"}
    assert {u["family"] for u in e.units if not u["stored"]} == unstored
    for unit, listed in zip(e.units, e.listed, strict=True):
        kept = unit["family"] not in unstored
        assert unit["result"] is True and unit["stored"] is kept and listed["stored"] is kept and ("evaluations" in unit) is kept
    for family, c in e.summary["by_family"].items():
        if family in unstored:
            assert list(c)[1:] == ["replayed_equal", "not_stored_operand_too_large", "seconds"]
        else:
            assert list(c)[1:3] == ["replayed_equal", "stored"]
        assert list(c.values())[:3] == [2, 2, 2]
    assert {p.name for p in (tmp_path / "store" / "values").iterdir()} == e.values_named()


def test_a_unit_whose_replay_ends_unequal_or_unevaluated_is_listed_with_its_reason_and_never_stored(world, tmp_path, monkeypatch, capsys):
    ends: dict[int, dict[str, Any]] = {0: {"result": False, "why": "word 3 differs"}, 1: {"result": None, "why": "an operand is not committed"}}
    e = _export(world, tmp_path, monkeypatch, capsys, limits=VX.Limits(per_family=2, workers=1), replay=lambda k, rec: {**rec, **ends.get(k, {})})
    for k, (unit, listed) in enumerate(zip(e.units, e.listed, strict=True)):
        if k in ends:
            assert list(unit) == UNIT and listed["stored"] is None
            assert (unit["result"], unit["why"]) == (listed["result"], listed["why"]) == (ends[k]["result"], ends[k]["why"])
        else:
            assert unit["stored"] is True and unit["why"] is None
    unequal, unevaluated = (e.summary["by_family"][e.picked[k]["family"]] for k in ends)
    assert unequal is not unevaluated
    assert list(unequal)[1:] == ["not_exported_mismatch", "replayed_equal", "stored", "instances", "seconds"]
    assert list(unevaluated)[1:] == ["not_exported_not_evaluated", "replayed_equal", "stored", "instances", "seconds"]
    assert list(unequal.values())[:4] == list(unevaluated.values())[:4] == [2, 1, 1, 1]
    assert {p.name for p in (tmp_path / "store" / "values").iterdir()} == e.values_named()


def test_an_export_into_a_directory_that_holds_one_writes_the_same_sets(world, tmp_path, monkeypatch, capsys):
    limits = VX.Limits(per_family=2, workers=1)
    first = _export(world, tmp_path, monkeypatch, capsys, limits=limits).summary
    again = _export(world, tmp_path, monkeypatch, capsys, limits=limits).summary
    assert first["sets"] and again["sets"] == first["sets"] and again["store"] == first["store"]


def test_a_program_that_states_no_digest_has_its_units_under_program(world, tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(world.inst, "digest", None)
    e = _export(world, tmp_path, monkeypatch, capsys, limits=VX.Limits(per_family=1, workers=1))
    assert [p.name for p in (tmp_path / "store" / "programs").iterdir()] == ["program"] and len(e.units) == len(e.picked) > 0
    assert json.loads((tmp_path / "store" / "programs" / "program" / "program.json").read_text()) == {
        "program_digest": "program", "requests": ["r0", "r1"], "build_dirs": {"r0": None, "r1": None}}
