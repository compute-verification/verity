"""Layout 1.1 (`verity_vllm.build_paths`): a `vllm-build/v1` tree built at one path resolves its components after being moved.

r20 fetched Builds had to be symlinked back to `/workspace/cp/<sweep>/<row>/...` because `workload_program.json` carried only the build
machine's absolute paths.  GP-01 now writes a `layout` block with artifact-root-relative paths beside them; readers resolve through
`build_paths.component_dir` / `manifest_path` (relative under the record dir, else the legacy absolute, else its basename under the
record dir).  A layout-1.0 record (no block) keeps working through the last two rules."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from verity_capture.experimental.cb_a import global_program as gp
from verity_capture.experimental.cb_a.tests.test_global_program_regress import _derived_dir, _manifest
from verity_vllm import build_paths as BP


def _synthetic_build(tmp_path: Path) -> tuple[Path, dict]:
    """A row dir laid out as the sweep does: <row>/workload.json, <row>/build_request_LP*_T*/, <row>/build_workload/workload_program.json."""
    row = tmp_path / "cp" / "sweep-x" / "row-1"
    row.mkdir(parents=True)
    d1, d2 = _derived_dir(row, 3, 2), _derived_dir(row, 5, 0)
    for d, name in ((d1, "build_request_LP3_T2"), (d2, "build_request_LP5_T0")):
        os.rename(d, row / name)
    m = _manifest(row, [{"LP": 3, "cap": 3}, {"LP": 5, "cap": 1, "arrive": 1}], eos_token_ids=[7])
    art = gp.build(m, [str(row / "build_request_LP3_T2"), str(row / "build_request_LP5_T0")], str(row / "build_workload"))
    assert art["ok"], art
    return row, art


def test_writer_records_root_relative_paths_beside_the_absolute_ones(tmp_path):
    row, art = _synthetic_build(tmp_path)
    lay = art["layout"]
    assert lay["schema"] == BP.LAYOUT_SCHEMA and lay["version"] == "1.1" == BP.layout_version(art)
    assert Path(lay["artifact_root"]) == row.resolve() or Path(lay["artifact_root"]) == row
    assert lay["manifest"] == "workload.json"
    assert lay["derived"] == {"3,2": "build_request_LP3_T2", "5,0": "build_request_LP5_T0"}
    # the legacy keys are still the absolute paths (layout-1.0 readers: global_match, tp2 bench, commit/*)
    assert all(os.path.isabs(v) for v in art["inputs"]["derived"].values()) and os.path.isabs(art["inputs"]["manifest"])
    on_disk = json.loads((row / "build_workload" / "workload_program.json").read_text())
    assert on_disk["layout"] == lay
    # the digest-bearing parts do not move
    assert on_disk["schema"] == "verity/workload-program/v1" and on_disk["workload_digest"] == art["workload_digest"]


def test_a_relocated_build_resolves_through_the_layout_block(tmp_path):
    row, art = _synthetic_build(tmp_path)
    moved = tmp_path / "fetched" / "art-e2feeb59"
    shutil.move(str(row), str(moved))
    assert not row.exists()
    wp = json.loads((moved / "build_workload" / "workload_program.json").read_text())
    rd = BP.record_dir_of(moved / "build_workload" / "workload_program.json")
    assert rd == moved.resolve()
    c = BP.component_dir(wp, 3, 2, rd)
    assert c == rd / "build_request_LP3_T2" and (c / "descriptor.json.gz").exists()
    assert BP.component_dir(wp, 5, 0, rd) == rd / "build_request_LP5_T0"
    assert BP.manifest_path(wp, rd) == rd / "workload.json"
    assert BP.component_dirs(wp, rd) == {"3,2": rd / "build_request_LP3_T2", "5,0": rd / "build_request_LP5_T0"}
    assert BP.component_dir(wp, 9, 9, rd) is None and BP.declared_component(wp, 9, 9) is None      # never composed: nothing inferred
    assert BP.declared_component(wp, 3, 2) == art["inputs"]["derived"]["3,2"]


def test_a_layout_1_0_record_relocated_whole_resolves_by_basename_and_a_stale_absolute_path_loses_to_the_relative_one(tmp_path):
    row, art = _synthetic_build(tmp_path)
    wp_path = row / "build_workload" / "workload_program.json"
    wp = json.loads(wp_path.read_text())
    legacy = {k: v for k, v in wp.items() if k != "layout"}          # what a pre-1.1 GP-01 wrote
    assert BP.layout_version(legacy) == "1.0"
    moved = tmp_path / "elsewhere" / "row-1"
    shutil.move(str(row), str(moved))
    rd = moved
    # 1.0: the absolute path dangles -> its basename under the record dir
    assert BP.component_dir(legacy, 3, 2, rd) == rd / "build_request_LP3_T2"
    assert BP.manifest_path(legacy, rd) == rd / "workload.json"
    # 1.0 with nothing at either place: None (the reader refuses by name)
    assert BP.component_dir(legacy, 3, 2, tmp_path / "empty") is None
    # 1.1: another Build later occupies the ORIGINAL absolute path -- the relocated record still reads its own component (relative first)
    row.mkdir(parents=True)
    (row / "build_request_LP3_T2").mkdir()
    assert Path(wp["inputs"]["derived"]["3,2"]).exists()
    assert BP.component_dir(wp, 3, 2, rd) == rd / "build_request_LP3_T2"
    # ... and the legacy record at the same original path resolves to the absolute one first (the build machine's view)
    assert BP.component_dir(legacy, 3, 2, None) == Path(wp["inputs"]["derived"]["3,2"])


def test_relative_to_root_refuses_paths_outside_the_root():
    assert BP.relative_to_root("/a/b/c", "/a/b") == "c"
    assert BP.relative_to_root("/a/b/c/d", "/a/b") == "c/d"
    assert BP.relative_to_root("/a/x", "/a/b") is None
    assert BP.relative_to_root("/a/b", "/a/b") is None
    assert BP.relative_to_root(None, "/a/b") is None
    blk = BP.layout_block("/a/b", "/tmp/w.json", {"3,2": "/a/b/build_request_LP3_T2", "5,0": "/elsewhere/x"})
    assert blk["manifest"] is None and blk["derived"] == {"3,2": "build_request_LP3_T2", "5,0": None}


@pytest.mark.parametrize("via", ["relative", "legacy"])
def test_lifted_record_cli_resolution_matches_build_paths(tmp_path, via):
    """The `--build-workload` request list (lifting/record.py) resolves each component through build_paths: after a move, a 1.1 record
    finds its dirs by the layout block and a 1.0 record by basename -- the same dir either way."""
    row, art = _synthetic_build(tmp_path)
    moved = tmp_path / "moved-row"
    shutil.move(str(row), str(moved))
    wp = json.loads((moved / "build_workload" / "workload_program.json").read_text())
    if via == "legacy":
        wp.pop("layout")
    for r in wp["requests"]:
        lp, cap = int(r["prompt_len"]), int(r["cap"])
        src = BP.declared_component(wp, lp, cap - 1)
        comp = BP.component_dir(wp, lp, cap - 1, moved) or moved / Path(src).name          # record.py's expression
        assert comp == moved / f"build_request_LP{lp}_T{cap - 1}" and (comp / "descriptor.json.gz").exists()
