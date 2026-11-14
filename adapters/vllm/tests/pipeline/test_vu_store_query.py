"""The VU store's own rules (`verity_vllm.pipeline.vu_store`), on stores written here by hand from one Gemm's evaluations: how a
value and an evaluation are stored, which evaluations a selector and a module keep, what an extraction writes of them and what it
counts, what the store's check lists, the free share of the memory limit, and what the Commit's hook hands the export."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from verity.circuits.query_ast import NameSel

from verity_vllm.check.replay import evaluate as EV
from verity_vllm.pipeline import vu_export as VX
from verity_vllm.pipeline import vu_store as VS

ROOT = bytes(range(32))
K, N = 32, 4
ST = {"K": K, "N": N}
MLP = "model.layers.0.mlp"


def _bf16(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (np.asarray(rng.uniform(-1, 1, n), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _gemm(seed: int) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """One Gemm row's operands (x of K words, the N x K weight) and what the row kernel returns for them."""
    x, w = _bf16(seed, K), _bf16(seed + 100, N * K)
    return [x, w], EV.evaluate("Gemm_v1", ST, [[x], []], [None, w])


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<")).tobytes()).hexdigest()


def _store(tmp_path: Path, units: list[tuple[str, str, list[tuple[dict[str, np.ndarray], list]]]], limits: VX.Limits,
           refs: tuple[dict[str, Any] | None, ...] = ()) -> Path:
    """A store of `units`, each (module path, the unit's family, its Gemm evaluations as (outputs, operands)), interior ones first.
    `refs`: each operand's reference into the checkpoint, where it has one."""
    root = tmp_path / "store"
    blobs: dict[str, tuple[str, bytes]] = {}
    (root / "values").mkdir(parents=True)
    (root / "programs" / "p").mkdir(parents=True)
    lines = []
    for vu, (op_path, family, evaluations) in enumerate(units):
        stored = [VS._evaluation((VX.selection_key(ROOT), ("row", "r0", vu, j)), "Gemm_v1", ST, ops, out, list(refs), blobs, limits)
                  for j, (out, ops) in enumerate(evaluations)]
        lines.append({"vu": vu, "request": "r0", "i": vu, "engine_step": 3, "op_path": op_path, "row": 7, "phase": "decode",
                      "family": family, "evaluations": stored})
    for h, (dt, b) in blobs.items():
        (root / "values" / f"{h}.{dt}").write_bytes(b)
    (root / "programs" / "p" / "units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in lines))
    (root / "run.json").write_text(json.dumps({"schema": VS.STORE_SCHEMA, "provenance": {"run_root": ROOT.hex(), "row": "hand"},
                                               "limits": dataclasses.asdict(limits)}))
    return root


def _index(sets: Path, name: str) -> list[list[Any]]:
    return json.loads((sets / name / "index.json").read_text())["rows"]


def test_a_value_is_stored_once_under_the_sha256_of_its_little_endian_words():
    blobs: dict[str, tuple[str, bytes]] = {}
    signed: np.ndarray = np.asarray([[1, -2], [3, 4]], dtype=np.int16)
    words: np.ndarray = signed.reshape(-1).view(np.uint16)
    raw = words.astype("<u2").tobytes()
    assert VS._blob(blobs, signed) == {"value": hashlib.sha256(raw).hexdigest(), "dtype": "u16", "words": 4}
    assert VS._blob(blobs, words) == VS._blob(blobs, signed) and blobs == {hashlib.sha256(raw).hexdigest(): ("u16", raw)}
    assert VS._blob(blobs, np.asarray([7], dtype=np.uint64))["dtype"] == "u64" and len(blobs) == 2


def test_an_evaluation_keeps_each_operand_whole_and_names_the_weight_of_one_that_has_a_reference():
    """A family whose weight no coordinate reads: every operand is kept whole.  One past the references' end has none."""
    blobs: dict[str, tuple[str, bytes]] = {}
    a, b = _bf16(1, 5), _bf16(2, 3)
    ref = {"field": "f", "range": [0, 5]}
    out = {"y": _bf16(3, 2)}
    ev = VS._evaluation((VX.selection_key(ROOT), ("row", "r0", 0, 0)), "Row_v1", {"S": 1}, [a, b], out, [ref], blobs, VX.Limits())
    assert ev == {"family": "Row_v1", "statics": {"S": 1},
                  "args": [{"value": _sha(a), "dtype": "u16", "words": 5, "weight": ref}, {"value": _sha(b), "dtype": "u16", "words": 3}],
                  "outs": {"y": {"value": _sha(out["y"]), "dtype": "u16", "words": 2}}}
    assert set(blobs) == {_sha(a), _sha(b), _sha(out["y"])}


def test_an_operand_with_a_reference_that_is_too_large_leaves_the_evaluation_unstored_and_one_without_is_kept():
    key, small = (VX.selection_key(ROOT), ("row", "r0", 0, 0)), VX.Limits(max_row_words=4)
    a, out = _bf16(1, 5), {"y": _bf16(3, 2)}
    assert VS._evaluation(key, "Row_v1", {}, [a], out, [{"field": "f", "range": [0, 5]}], {}, small) is None
    kept = VS._evaluation(key, "Row_v1", {}, [a], out, [], {}, small)
    assert kept is not None and kept["args"] == [{"value": _sha(a), "dtype": "u16", "words": 5}]
    at_the_limit = VS._evaluation(key, "Row_v1", {}, [a[:4]], out, [{"field": "f", "range": [0, 4]}], {}, small)
    assert at_the_limit is not None and at_the_limit["args"][0]["words"] == 4


def test_a_gemms_weight_is_stored_as_its_sampled_rows_with_the_whole_slice_by_reference_however_large():
    ops, out = _gemm(5)
    w = ops[1].reshape(N, K)
    ref = {"field": "mlp.w", "range": [0, N * K]}
    index = ("row", "r0", 9, 1)
    blobs: dict[str, tuple[str, bytes]] = {}
    ev = VS._evaluation((VX.selection_key(ROOT), index), "Gemm_v1", ST, ops, out, [None, ref], blobs,
                        VX.Limits(gemm_coords_per_row=2, max_row_words=1))
    assert ev is not None
    rows = VX._Picker(VX.selection_key(ROOT), index)(N, 2)
    assert len(rows) == 2 and ev["args"][0] == {"value": _sha(ops[0]), "dtype": "u16", "words": K}
    assert ev["args"][1] == {"weight": {"field": "mlp.w", "range": [0, N * K], "shape": [N, K], "dtype": "u16", "sha256": _sha(w)},
                             "sample": {"rows": rows, "index": ["row", "r0", 9, 1], "value": _sha(w[rows]), "dtype": "u16", "words": 2 * K}}
    assert list(ev["outs"]) == ["0"] and ev["outs"]["0"]["words"] == N


@pytest.mark.parametrize("where, family, statics, kept", [
    (None, "Gemm_v1", {}, True),
    ("Gemm_v1", "Gemm_v1", {}, True),
    ("Gemm", "Gemm_v1", {}, True),
    ("Gemm_v2", "Gemm_v1", {}, False),
    ("Gem", "Gemm_v1", {}, False),
    ("Scaled_vector", "Scaled_vector_v12", {}, True),
    ("Scaled", "Scaled_vector_v12", {}, False),
    ("Gemm_v1{K=2048}", "Gemm_v1", {"K": 2048, "N": 8}, True),
    ("Gemm{K=2048,N=8}", "Gemm_v1", {"K": 2048, "N": 8}, True),
    ("Gemm_v1{K=2048}", "Gemm_v1", {"K": 1024}, False),
    ("Gemm_v1{K=2048,N=8}", "Gemm_v1", {"K": 2048, "N": 4}, False),
    ("Gemm_v1{K=2048}", "Gemm_v1", {"N": 8}, False),
    ("Gemm_v1{K=*}", "Gemm_v1", {"K": 7}, True),
    ("Gemm_v1{K=*}", "Gemm_v1", {"N": 8}, False),
    ("Gemm_v1{*}", "Gemm_v1", {}, True),
])
def test_a_selector_keeps_a_definition_by_its_id_or_its_name_when_every_static_it_lists_is_bound_so(where, family, statics, kept):
    assert VS._matches(NameSel.parse(where) if where else None, family, statics) is kept


@pytest.mark.parametrize("op_path, module, kept", [
    ("model.layers.0.mlp", None, True),
    (None, "", True),
    ("model.layers.0.mlp", "model.layers.0.mlp", True),
    ("model.layers.0.mlp.up", "model.layers.0.mlp", True),
    ("model.layers.0.mlpx", "model.layers.0.mlp", False),
    ("model.layers.0", "model.layers.0.mlp", False),
    ("layers.0.mlp.up", "model.layers.0.mlp", True),
    ("layers.0.mlp", "model.layers.0.mlp", True),
    ("layers", "model.layers.0.mlp", False),
    ("0.mlp", "model.layers.0.mlp", False),
    ("model.embed", "model", True),
    ("model", "model", True),
    ("modelx", "model", False),
    (None, "model", False),
])
def test_a_module_keeps_the_units_at_it_or_below_it_with_or_without_the_paths_first_name(op_path, module, kept):
    assert VS._in_module(op_path, module) is kept


def test_an_extraction_marks_a_units_last_evaluation_committed_and_the_earlier_ones_interior(tmp_path):
    limits = VX.Limits(gemm_coords_per_row=2)
    (a_ops, a_out), (b_ops, b_out) = _gemm(1), _gemm(2)
    root = _store(tmp_path, [(MLP, "Outer_v1", [(a_out, a_ops), (b_out, b_ops)])], limits)
    sets, by_family = VS.extract(root, tmp_path / "sets")
    name = f"gemm-coordinate-ampere-bf16-k{K}"
    assert [(s["set"], s["n"]) for s in sets] == [(name, 4)] and by_family == {"Outer_v1": 4}
    rows = _index(tmp_path / "sets", name)
    assert [r[:8] for r in rows] == [[i, 0, "r0", 3, MLP, 7, "decode", "interior" if i < 2 else "committed"] for i in range(4)]
    manifest = json.loads((tmp_path / "sets" / name / "manifest.json").read_text())
    assert manifest["provenance"] == {"run_root": ROOT.hex(), "row": "hand",
                                      "extraction": {"where": None, "module": None, "weights": "stored sample"}}
    VS.extract(root, tmp_path / "asked", where="Gemm_v1", module="model.layers.0", weights=lambda _field, _lo, _hi: None)
    asked = json.loads((tmp_path / "asked" / name / "manifest.json").read_text())
    assert asked["provenance"]["extraction"] == {"where": "Gemm_v1", "module": "model.layers.0", "weights": "checkpoint"}
    assert asked["content_digest"] == manifest["content_digest"] and VS.verify_store(root)["ok"]


def test_a_selector_and_a_module_each_leave_out_what_they_do_not_keep(tmp_path):
    (a_ops, a_out), (b_ops, b_out) = _gemm(1), _gemm(2)
    root = _store(tmp_path, [(MLP, "Gemm_v1", [(a_out, a_ops)]), ("model.layers.1.mlp", "Gemm_v1", [(b_out, b_ops)])],
                  VX.Limits(gemm_coords_per_row=1))
    name = f"gemm-coordinate-ampere-bf16-k{K}"
    assert [s["n"] for s in VS.extract(root, tmp_path / "all")[0]] == [2]
    assert VS.extract(root, tmp_path / "none", where=f"Gemm_v1{{K={K + 1}}}") == ([], {})
    sets, by_family = VS.extract(root, tmp_path / "one", where=f"Gemm{{K={K},*}}", module="model.layers.1")
    assert [(s["set"], s["n"]) for s in sets] == [(name, 1)] and by_family == {"Gemm_v1": 1}
    assert [r[1:5] for r in _index(tmp_path / "one", name)] == [[1, "r0", 3, "model.layers.1.mlp"]]


def test_an_extraction_cuts_a_gemms_stored_rows_or_any_row_of_the_checkpoints_weight(tmp_path):
    ops, out = _gemm(4)
    root = _store(tmp_path, [(MLP, "Gemm_v1", [(out, ops)])], VX.Limits(gemm_coords_per_row=2),
                  refs=(None, {"field": "mlp.w", "range": [8, 8 + N * K]}))
    every_row = VX.Limits(gemm_coords_per_row=N)
    asked: list[tuple[str, int, int]] = []

    def checkpoint(field: str, lo: int, hi: int) -> list[int]:
        asked.append((field, lo, hi))
        return ops[1].tolist()

    stored, _ = VS.extract(root, tmp_path / "stored", limits=every_row)
    whole, _ = VS.extract(root, tmp_path / "whole", limits=every_row, weights=checkpoint)
    assert [s["n"] for s in stored] == [2] and [s["n"] for s in whole] == [N] and asked == [("mlp.w", 8, 8 + N * K)]
    units = root / "programs" / "p" / "units.jsonl"
    (unit,) = map(json.loads, units.read_text().splitlines())
    assert unit["evaluations"][0]["args"][1]["weight"].pop("dtype") == "u16"
    units.write_text(json.dumps(unit) + "\n")
    older, _ = VS.extract(root, tmp_path / "older", limits=every_row, weights=checkpoint)
    assert [s["content_digest"] for s in older] == [s["content_digest"] for s in whole]


def test_a_set_takes_no_instance_past_its_limit(tmp_path):
    (a_ops, a_out), (b_ops, b_out) = _gemm(1), _gemm(2)
    root = _store(tmp_path, [(MLP, "Gemm_v1", [(a_out, a_ops), (b_out, b_ops)])], VX.Limits(gemm_coords_per_row=2))
    sets, by_family = VS.extract(root, tmp_path / "sets", limits=VX.Limits(gemm_coords_per_row=2, max_instances_per_set=3))
    assert [s["n"] for s in sets] == [3] and by_family == {"Gemm_v1": 3}
    assert [r[7] for r in _index(tmp_path / "sets", sets[0]["set"])] == ["interior", "interior", "committed"]


def test_an_evaluation_whose_words_its_cut_does_not_reproduce_is_printed_and_counted_under_its_units_family(tmp_path, capsys):
    (a_ops, a_out), (b_ops, b_out) = _gemm(1), _gemm(2)
    wrong = {"0": b_out["0"] ^ 1}
    root = _store(tmp_path, [(MLP, "Outer_v1", [(a_out, a_ops), (wrong, b_ops)])], VX.Limits(gemm_coords_per_row=2))
    sets, by_family = VS.extract(root, tmp_path / "sets")
    assert [s["n"] for s in sets] == [2] and by_family == {"Outer_v1": 2, "decomposition_error:Outer_v1": 1}
    assert [r[7] for r in _index(tmp_path / "sets", sets[0]["set"])] == ["interior", "interior"]
    (line,) = capsys.readouterr().out.splitlines()
    assert line.startswith("[vu-export] unit 0 evaluation 1: Gemm_v1 coordinate ") and " != row word 0x" in line


def test_the_stores_check_lists_eight_of_the_values_that_do_not_hash_to_their_names(tmp_path):
    (tmp_path / "values").mkdir()
    (tmp_path / "programs").mkdir()
    good = b"\x01\x02"
    (tmp_path / "values" / f"{hashlib.sha256(good).hexdigest()}.u8").write_bytes(good)
    wrong = {f"{k:064x}.u8" for k in range(9)}
    for name in wrong:
        (tmp_path / "values" / name).write_bytes(b"\x00")
    res = VS.verify_store(tmp_path)
    assert len(res["bad_values"]) == 8 and set(res["bad_values"]) < wrong
    assert {k: v for k, v in res.items() if k != "bad_values"} == {"ok": False, "missing_values": 0, "values": 10}


V2 = ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current")
V1 = ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")
MEMINFO = "/proc/meminfo"


@pytest.mark.parametrize("files, free", [
    ({V2[0]: "1000\n", V2[1]: "250\n"}, 0.75),
    ({V2[0]: "1000", V2[1]: "1500"}, 0.0),
    ({V2[0]: "max\n", V2[1]: "250\n", V1[0]: "2000\n", V1[1]: "500\n"}, 0.75),
    ({V2[0]: "1000", V1[0]: "400", V1[1]: "100"}, 0.75),
    ({V2[0]: "1000", V2[1]: "a few", V1[0]: "400", V1[1]: "100"}, 0.75),
    ({V1[0]: str(1 << 60), V1[1]: "100", MEMINFO: "MemTotal:       1000 kB\nMemFree: 5 kB\nMemAvailable:    250 kB\n"}, 0.25),
    ({V2[0]: str((1 << 60) - 1), V2[1]: str(1 << 59), MEMINFO: "MemTotal: 8 kB\nMemAvailable: 8 kB\n"}, 1.0 - (1 << 59) / ((1 << 60) - 1)),
    ({V2[0]: "max", V2[1]: "1", MEMINFO: "Blank:\nMemTotal: 400\nHugePages_Total:       0\nMemAvailable: 100 kB\n"}, 0.25),
    ({MEMINFO: "MemTotal: 400 kB\n"}, None),
    ({MEMINFO: "MemTotal: 0 kB\nMemAvailable: 0 kB\n"}, None),
    ({MEMINFO: "MemTotal: many kB\nMemAvailable: 1 kB\n"}, None),
    ({}, None),
])
def test_the_free_share_is_the_cgroups_under_a_finite_limit_then_the_hosts_and_none_when_neither_is_readable(monkeypatch, files, free):
    class Named:
        def __init__(self, name: str) -> None:
            self.name = name

        def read_text(self) -> str:
            if self.name not in files:
                raise FileNotFoundError(self.name)
            return files[self.name]

    monkeypatch.setattr(VS, "Path", Named)
    got = VS.memory_headroom()
    assert got == free and (free is None) == (got is None)


def _reader(*_a: Any) -> None:
    return None


def _weights(*_a: Any) -> None:
    return None


class _Engine:
    cp = {"repo": "tiny/model", "revision": "abc"}


def _hook(tmp_path, monkeypatch, export, **given):
    """`after_replay` on a passing pair 0 with `export` in the export's place: what it returns."""
    kept: dict[str, Any] = {"indices": {}, "vus": []}
    monkeypatch.setattr(VS, "export_vus", export)
    monkeypatch.setattr(VS, "memory_headroom", lambda: 1.0)
    monkeypatch.setattr(VS.SR, "take_population", lambda: kept)
    a: dict[str, Any] = {"sr": {"result": True}, "manifest": None, "replay_kw": {}, "loaded": {}, "eng": _Engine(), **given}
    got = VS.after_replay(str(tmp_path), a["sr"], 0, ROOT, {"r0": {"digest": "d"}}, {"b": 1}, _reader, _weights, {"r0": 2}, a["manifest"],
                          a["replay_kw"], a["loaded"], {"r0": "d" * 64}, {"git_head": "f00", "n": 3, "skipped": ["a", "list"]}, a["eng"])
    return got, kept


def test_the_hook_hands_the_export_the_replays_own_record_and_population(tmp_path, monkeypatch, capsys):
    seen: list[tuple[tuple, dict]] = []
    summary = {"sets": [{"n": 2, "bytes": 10}, {"n": 3, "bytes": 20}], "wall_s": 1.5}

    def export(*a, **kw):
        seen.append((a, kw))
        return summary

    (tmp_path / "provenance.json").write_text(json.dumps({"row": 101, "model": "replaced"}))
    (tmp_path / "limits.json").write_text(json.dumps({"per_family": 3}))
    got, kept = _hook(tmp_path, monkeypatch, export, sr={"result": True, "weights": {"committed_root": "ab" * 32}},
                      manifest={"manifest_digest": "m" * 64}, replay_kw={"executed_prefix": {"r0": {"steps": 1}}},
                      loaded={"r0": {"dir": "/builds/r0"}, "r1": {"dir": ""}, "r2": {}})
    ((a, kw),) = seen
    assert got is summary and a == ({"r0": {"digest": "d"}}, {"b": 1}, _reader, _weights)
    assert kw.pop("population_of_record") is kept and kw.pop("limits") == VX.Limits(per_family=3)
    checkpoint = {"repo": "tiny/model", "revision": "abc", "weights_of_record_root": "ab" * 32}
    assert kw == {"out_dir": tmp_path, "run_root": ROOT, "arrivals": {"r0": 2}, "manifest": {"manifest_digest": "m" * 64},
                  "executed_prefix": {"r0": {"steps": 1}}, "program_dirs": {"r0": "/builds/r0"},
                  "provenance": {"row": 101, "model": "tiny/model", "revision": "abc", "program_digests": {"r0": "d" * 64},
                                 "manifest_digest": "m" * 64, "pair": 0, "versions": {"git_head": "f00", "n": 3},
                                 "weights": {"checkpoint": checkpoint}}}
    assert capsys.readouterr().out == f"[vu-export] 2 sets, 5 instances, 30 bytes in 1.5 s -> {tmp_path}\n"


def test_the_hook_without_a_provenance_file_a_manifest_or_a_checkpoint_states_none_of_them(tmp_path, monkeypatch):
    seen: list[dict] = []

    def export(*_a, **kw):
        seen.append(kw)
        return {"sets": [], "wall_s": 0}

    _hook(tmp_path, monkeypatch, export, manifest="a path, not the manifest", eng=object())
    (kw,) = seen
    assert kw["manifest"] is None and kw["executed_prefix"] is None and kw["program_dirs"] == {}
    assert kw["provenance"] == {"model": None, "revision": None, "program_digests": {"r0": "d" * 64}, "manifest_digest": None, "pair": 0,
                                "versions": {"git_head": "f00", "n": 3},
                                "weights": {"checkpoint": {"repo": None, "revision": None, "weights_of_record_root": None}}}


def test_an_export_that_raises_is_printed_and_returned_by_name_in_its_first_400_characters(tmp_path, monkeypatch, capsys):
    def export(*_a, **_kw):
        raise ValueError("x" * 500)

    got, _kept = _hook(tmp_path, monkeypatch, export)
    why = repr(ValueError("x" * 500))
    assert got == {"ok": False, "why": why[:400]} and len(got["why"]) == 400
    assert capsys.readouterr().out == f"[vu-export] FAILED: {why}"[:400] + "\n"
