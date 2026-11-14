"""`topp_split_probe.check` and the command's `check` and `produce` branches, on a fixture the model made.

The fixture of record (`tests/program/test_topp_split_geometry_of_record.py`, fetched, slow) holds a GPU's outputs and shows the
model equal to them.  Here the model's own outputs stand in for the kernel's, at small vocabularies, so no case below says the
model is right.  Each shows what the checker does with what a fixture holds: what it counts, which difference it reports under
which name, and the command's exit code.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from verity.definitions.registry import topp_geometry
from verity.definitions.registry import topp_split as TS

from verity_vllm.pipeline import cli
from verity_vllm.pipeline import topp_split_probe as tp

f32 = np.float32
NUM_SM = 8
RECORD_ID = 6
DRAIN_ID = 7


def _cases() -> list[dict]:
    return [
        {"id": 0, "kind": "random", "S": 2, "V": 1024, "seed": tp.MANIFEST_SEED + 12240, "p": 0.95},   # its two Z values differ
        {"id": 1, "kind": "random", "S": 2, "V": 3000, "seed": tp.MANIFEST_SEED + 32001, "p": 0.9},    # so do this one's; 3000 % 16 != 0
        {"id": 2, "kind": "ties", "S": 2, "V": 1024, "seed": 7, "k": 12, "keep_frac": 0.25, "group_bias": None, "plateau": 3.5},
        {"id": 3, "kind": "ties", "S": 1, "V": 1024, "seed": 9, "k": 3, "keep_frac": 0.5, "group_bias": 0, "plateau": 3.5},
        {"id": 4, "kind": "ties", "S": 4, "V": 1024, "seed": 13, "k": 12, "keep_frac": 0.75, "group_bias": 1, "plateau": 6.5},
        {"id": 5, "kind": "ties", "S": 8, "V": 1024, "seed": 17, "k": 24, "keep_frac": 0.5, "group_bias": None, "plateau": 3.5},
    ]


def _keep(out: np.ndarray) -> np.ndarray:
    return out.view(np.uint32) != tp.NEG_INF_BITS


def _row(arrays: dict[str, np.ndarray], c: dict, x_in: np.ndarray, p: float) -> None:
    """What `produce` files for one case, from the model's answer in the kernel's place."""
    S, V = c["S"], c["V"]
    cfg = TS.Cfg.for_vocab(V, S)
    out, stats, _parts, tr = TS.topp_split_row(x_in, p, S, cfg=cfg)
    pre = f"c{c['id']}_"
    arrays[pre + "keep"] = np.packbits(_keep(out), bitorder="little")
    arrays[pre + "stats"] = stats
    zw = np.asarray(TS.row_stats(stats, S, cfg)[1], f32).view(np.uint32)
    zu = np.unique(zw)
    arrays[pre + "zvals"] = zu
    if len(zu) == 2:
        lane = np.array([tp.thread_lane(v, V, S, cfg.spt_x) for v in range(V)])
        arrays[pre + "zbit"] = np.packbits(zw[topp_geometry.z_group(lane, S)] == zu[1], bitorder="little")
    c.update(p=float(p), kept=int(_keep(out).sum()), ties=bool(tr["numkeep"] < tr["numdup"]),
             numkeep=float(tr["numkeep"]), numdup=float(tr["numdup"]))


def _record(d: Path) -> tuple[str, tuple]:
    """A record of one row: its bf16 logits as the upper halves of their fp32 words, and the device's facts beside it."""
    bits = (tp.gen_random(11, 1024).view(np.uint32) >> 16).astype(np.uint16)
    path = str(d / "rows.npz")
    np.savez(path, r0_bf16_logits_bits=bits)
    meta = {"r0": {"device": {"top_p": 0.9, "temperature": 0.7}}}
    Path(path + ".json").write_text(json.dumps(meta))
    return path, (np.load(path), meta)


def _save(d: Path, arrays: dict[str, Any]) -> None:
    np.savez_compressed(d / "fixture.npz", **arrays)


def _drain(arrays: dict[str, np.ndarray]) -> dict:
    dc: dict = {"id": DRAIN_ID, "kind": "drain", "V": 1024, "seed": 100, "k": 24, "keep_frac": 0.5, "p": None, "num_sm": NUM_SM, "batches": []}
    rows = [tp.gen_ties(dc["seed"] + b, dc["V"], 16 if b % 2 == 0 else 32, dc["k"], dc["keep_frac"]) for b in range(3)]
    dc["rows"] = [{"seed": dc["seed"] + b, "S_gen": 16 if b % 2 == 0 else 32, "p": float(p)} for b, (_x, p) in enumerate(rows)]
    for B in (3, 2, 1):
        S = topp_geometry.splits_for(B, NUM_SM)
        keep = np.stack([_keep(TS.topp_split_row(x, p, S)[0]) for x, p in rows[:B]])
        arrays[f"drain_B{B}_keep"] = np.packbits(keep, axis=1, bitorder="little")
        dc["batches"].append({"B": B, "S": S, "kept": [int(k.sum()) for k in keep]})
    return dc


@pytest.fixture(scope="module")
def made(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("topp_fixture")
    arrays: dict[str, np.ndarray] = {}
    cases = _cases()
    for c in cases:
        _row(arrays, c, *tp.case_row(c))
    _path, record = _record(d)
    rc: dict[str, Any] = {"id": RECORD_ID, "kind": "record", "S": 2, "V": 1024, "row": 0}
    bf, top_p = tp.case_row(rc, record)
    rc["temperature"] = tp.record_temperature(rc, record)
    x_in = TS.tl_div(bf, f32(rc["temperature"]), TS.Cfg())
    rc["x_in_sha"] = hashlib.sha256(x_in.view(np.uint32).tobytes()).hexdigest()[:16]
    _row(arrays, rc, x_in, top_p)
    cases += [rc, _drain(arrays)]
    _save(d, arrays)
    (d / "manifest.json").write_text(json.dumps({"cases": cases, "seed": tp.MANIFEST_SEED, "num_sm": NUM_SM}, indent=1))
    return d


@pytest.fixture
def fx(made: Path, tmp_path: Path) -> Path:
    """A copy of the fixture a case may change."""
    return Path(shutil.copytree(made, tmp_path / "fx"))


def _alter(d: Path, key: str, change) -> None:
    with np.load(d / "fixture.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays[key] = change(arrays[key].copy())
    _save(d, arrays)


def _alter_manifest(d: Path, change) -> None:
    man = json.loads((d / "manifest.json").read_text())
    change({c["id"]: c for c in man["cases"]})
    (d / "manifest.json").write_text(json.dumps(man))


def _flip(row: int | None, lane: int):
    def change(a: np.ndarray) -> np.ndarray:
        (a if row is None else a[row])[lane // 8] ^= 1 << (lane % 8)
        return a
    return change


def _stats_word(a: np.ndarray) -> np.ndarray:
    a.view(np.uint32)[0, 0] ^= 1
    return a


def _report(d: Path, record: bool = True) -> dict:
    return tp.check(str(d), str(d / "rows.npz") if record else None, verbose=False)


def test_a_fixture_equal_to_the_model_passes_and_is_counted(fx):
    rep = _report(fx)
    assert rep == {"n": 7, "mask_ok": 7, "stats_ok": 7, "zgroup_ok": 1, "zgroup_checked": 1, "ties_cases": 6, "boundary_tie_cases": 3,
                   "fail": [], "ok": True,
                   "drain": [{"B": 3, "S": 2, "rows_ok": 3}, {"B": 2, "S": 4, "rows_ok": 2}, {"B": 1, "S": 8, "rows_ok": 1}]}


def test_the_fixture_holds_two_rows_with_two_z_values(made):
    """Without them the Z-group cases below would show nothing."""
    with np.load(made / "fixture.npz") as z:
        assert [k for k in z.files if k.endswith("_zbit")] == ["c0_zbit", "c1_zbit"]
        assert all(len(z[f"c{i}_zvals"]) == 2 for i in (0, 1))


@pytest.mark.parametrize("record", [None, "absent.npz"])
def test_without_the_record_its_rows_are_passed_over(fx, record):
    rep = tp.check(str(fx), record and str(fx / record), verbose=False)
    assert rep["ok"] and (rep["n"], rep["mask_ok"], rep["stats_ok"]) == (6, 6, 6)


def test_a_kept_lane_the_model_masks_is_a_mask_failure(fx):
    _alter(fx, "c2_keep", _flip(None, 5))
    rep = _report(fx)
    assert not rep["ok"] and (rep["n"], rep["mask_ok"], rep["stats_ok"]) == (7, 6, 7)
    [f] = rep["fail"]
    assert {k: f[k] for k in ("id", "kind", "S", "V", "mask_ok", "stats_ok", "zgroup_ok", "lanes_differ", "first")} == {
        "id": 2, "kind": "ties", "S": 2, "V": 1024, "mask_ok": False, "stats_ok": True, "zgroup_ok": None, "lanes_differ": 1, "first": [5]}
    man = {c["id"]: c for c in json.loads((fx / "manifest.json").read_text())["cases"]}
    assert (f["p"], f["numkeep_model"], f["numdup_model"], f["numkeep_kernel"], f["numdup_kernel"]) == (
        man[2]["p"], man[2]["numkeep"], man[2]["numdup"], man[2]["numkeep"], man[2]["numdup"])


def test_a_failure_lists_the_first_eight_lanes_that_differ(fx):
    _alter(fx, "c3_keep", lambda a: a ^ np.uint8(0xFF))
    [f] = _report(fx)["fail"]
    assert (f["id"], f["lanes_differ"], f["first"]) == (3, 1024, list(range(8)))


def test_a_stats_word_that_differs_is_a_stats_failure(fx):
    _alter(fx, "c1_stats", _stats_word)
    rep = _report(fx)
    assert not rep["ok"] and (rep["mask_ok"], rep["stats_ok"]) == (7, 6)
    [f] = rep["fail"]
    assert (f["id"], f["mask_ok"], f["stats_ok"], f["lanes_differ"], f["first"]) == (1, True, False, 0, [])


def test_a_lane_in_the_other_z_group_is_a_z_group_failure(fx):
    _alter(fx, "c0_zbit", lambda a: a ^ np.uint8(0xFF))
    rep = _report(fx)
    assert not rep["ok"] and (rep["mask_ok"], rep["stats_ok"], rep["zgroup_ok"], rep["zgroup_checked"]) == (7, 7, 0, 1)
    [f] = rep["fail"]
    assert (f["id"], f["mask_ok"], f["stats_ok"], f["zgroup_ok"]) == (0, True, True, False)


def test_a_z_bit_is_not_read_where_the_vocabulary_is_no_multiple_of_16(fx):
    _alter(fx, "c1_zbit", lambda a: a ^ np.uint8(0xFF))
    rep = _report(fx)
    assert rep["ok"] and (rep["zgroup_ok"], rep["zgroup_checked"]) == (1, 1)


def test_a_record_row_whose_tempered_logits_differ_from_the_recorded_digest_fails(fx):
    _alter_manifest(fx, lambda by_id: by_id[RECORD_ID].update(x_in_sha="0" * 16))
    rep = _report(fx)
    assert not rep["ok"] and (rep["n"], rep["mask_ok"], rep["stats_ok"]) == (7, 7, 7)
    assert rep["fail"] == [{"id": RECORD_ID, "kind": "record", "S": 2, "x_in": "TemperatureScale model != pod apply_temperature"}]


def test_a_record_row_with_no_digest_is_checked_by_its_mask_alone(fx):
    _alter_manifest(fx, lambda by_id: by_id[RECORD_ID].pop("x_in_sha"))
    rep = _report(fx)
    assert rep["ok"] and rep["n"] == 7


def test_a_record_row_is_tempered_at_the_manifest_s_temperature(fx):
    _alter_manifest(fx, lambda by_id: by_id[RECORD_ID].update(temperature=0.5))
    rep = _report(fx)
    assert not rep["ok"] and rep["fail"][0] == {"id": RECORD_ID, "kind": "record", "S": 2, "x_in": "TemperatureScale model != pod apply_temperature"}


def test_a_drain_row_that_differs_is_counted_out_of_its_batch(fx):
    _alter(fx, "drain_B3_keep", _flip(1, 0))
    rep = _report(fx)
    assert not rep["ok"] and rep["fail"] == [{"drain_B": 3, "S": 2, "rows_ok": 2}]
    assert rep["drain"] == [{"B": 3, "S": 2, "rows_ok": 2}, {"B": 2, "S": 4, "rows_ok": 2}, {"B": 1, "S": 8, "rows_ok": 1}]
    assert (rep["n"], rep["mask_ok"], rep["stats_ok"]) == (7, 7, 7)


def test_a_drain_batch_at_another_split_count_than_the_geometry_s_is_refused(fx):
    _alter_manifest(fx, lambda by_id: by_id[DRAIN_ID]["batches"][0].update(S=4))
    with pytest.raises(AssertionError, match=r"\(3, 4\)"):
        _report(fx)


def test_a_fixture_with_no_drain_is_checked_by_its_rows(fx):
    man = json.loads((fx / "manifest.json").read_text())
    man["cases"] = [c for c in man["cases"] if c["kind"] != "drain"]
    (fx / "manifest.json").write_text(json.dumps(man))
    rep = _report(fx)
    assert rep["ok"] and rep["n"] == 7 and rep["drain"] == []


def test_check_prints_a_line_a_case_a_line_a_drain_batch_and_the_totals(fx, capsys):
    rep = tp.check(str(fx), str(fx / "rows.npz"))
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 7 + 3 + 1
    assert [json.loads(s)["id"] for s in lines[:7]] == [0, 1, 2, 3, 4, 5, RECORD_ID]
    kept = json.loads((fx / "manifest.json").read_text())["cases"][0]["kept"]
    assert json.loads(lines[0]) == {"id": 0, "kind": "random", "S": 2, "V": 1024, "mask_ok": True, "stats_ok": True, "zgroup_ok": True,
                                    "ties": True, "kept": kept}
    assert json.loads(lines[2])["zgroup_ok"] is None and json.loads(lines[6])["ties"] is False
    assert [json.loads(s) for s in lines[7:10]] == [{"drain_B": 3, "S": 2, "rows_ok": "3/3"}, {"drain_B": 2, "S": 4, "rows_ok": "2/2"},
                                                   {"drain_B": 1, "S": 8, "rows_ok": "1/1"}]
    totals = {k: v for k, v in rep.items() if k not in ("fail", "drain")}
    assert lines[10] == f"TOTALS {json.dumps(totals)} fail: []"
    tp.check(str(fx), verbose=False)
    assert capsys.readouterr().out == ""


def test_the_totals_line_names_the_first_five_failures(fx, capsys):
    for i in range(6):
        _alter(fx, f"c{i}_keep", _flip(None, 5))
    _alter(fx, f"c{RECORD_ID}_keep", _flip(None, 5))
    rep = tp.check(str(fx), str(fx / "rows.npz"))
    assert [f["id"] for f in rep["fail"]] == [0, 1, 2, 3, 4, 5, RECORD_ID] and rep["mask_ok"] == 0
    assert capsys.readouterr().out.splitlines()[-1].endswith(f"fail: {rep['fail'][:5]}")


def test_case_row_reads_each_kind_and_refuses_another():
    random, _r, ties, biased, *_more = _cases()
    x, p = tp.case_row(random)
    assert np.array_equal(x, tp.gen_random(random["seed"], 1024)) and p == 0.95
    for c in (ties, biased):
        x, p = tp.case_row(c)
        want = tp.gen_ties(c["seed"], c["V"], c["S"], c["k"], c["keep_frac"], group_bias=c["group_bias"], plateau=c["plateau"])
        assert np.array_equal(x, want[0]) and p == want[1]
    old = {k: v for k, v in ties.items() if k not in ("group_bias", "plateau")}      # a case that names neither reads as unbiased, at 3.5
    assert np.array_equal(tp.case_row(old)[0], tp.case_row(ties)[0])
    assert not np.array_equal(tp.case_row({**ties, "plateau": 6.5})[0], tp.case_row(ties)[0])
    with pytest.raises(ValueError, match="other"):
        tp.case_row({"kind": "other"})


def test_a_record_case_is_the_row_s_bf16_logits_and_the_device_s_facts(tmp_path):
    path, record = _record(tmp_path)
    c = {"kind": "record", "row": 0}
    bf, top_p = tp.case_row(c, record)
    assert bf.dtype == f32 and np.array_equal(bf, tp.gen_random(11, 1024)) and top_p == 0.9     # `gen_random` rows are bf16 values
    assert tp.record_temperature(c, record) == 0.7


def _main(argv: list[str]) -> int | str | None:
    with pytest.raises(SystemExit) as e:
        tp.main(cli.parse("topp-split-probe", argv, {}))
    return e.value.code


def test_the_check_command_exits_0_on_a_passing_fixture_and_writes_its_report(fx, capsys):
    out = fx / "report.json"
    assert _main(["check", "--dir", str(fx), "--record", str(fx / "rows.npz"), "--out", str(out)]) == 0
    assert out.read_text() == json.dumps(_report(fx), indent=1)
    assert capsys.readouterr().out.splitlines()[-1].startswith("TOTALS ")


def test_the_check_command_without_out_writes_nothing(fx, capsys):
    before = sorted(p.name for p in fx.iterdir())
    assert _main(["check", "--dir", str(fx)]) == 0
    assert sorted(p.name for p in fx.iterdir()) == before
    assert json.loads(capsys.readouterr().out.splitlines()[0])["id"] == 0


@pytest.mark.parametrize(("key", "change", "failed"), [
    ("c2_keep", _flip(None, 5), {"id": 2, "mask_ok": False}),
    ("drain_B3_keep", _flip(1, 0), {"drain_B": 3, "S": 2, "rows_ok": 2}),
])
def test_the_check_command_exits_1_on_a_failing_fixture_and_still_writes_its_report(fx, capsys, key, change, failed):
    _alter(fx, key, change)
    out = fx / "report.json"
    assert _main(["check", "--dir", str(fx), "--out", str(out)]) == 1
    rep = json.loads(out.read_text())
    assert rep == _report(fx, record=False) and rep["ok"] is False
    [f] = rep["fail"]
    assert {k: f[k] for k in failed} == failed
    assert capsys.readouterr().out.splitlines()[-1].endswith(f"fail: {rep['fail']}")


@pytest.mark.parametrize(("argv", "passed"), [
    (["--out", "DIR"], ("DIR", None, tp.MANIFEST_SEED)),
    (["--out", "DIR", "--record", "rows.npz", "--seed", "7"], ("DIR", "rows.npz", 7)),
])
def test_the_produce_command_hands_produce_its_directory_record_and_seed(monkeypatch, argv, passed):
    calls: list[tuple] = []
    monkeypatch.setattr(tp, "produce", lambda *a: calls.append(a))
    tp.main(cli.parse("topp-split-probe", ["produce", *argv], {}))      # returns: this branch does not exit
    assert calls == [passed]
