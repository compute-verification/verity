"""lift_expected.py takes a row's class from the newest `class` label by daniel on its record of record, else from the index, and
fixtures.toml states which (`class_source`).  A stand-in vault index against a tmp store root with no remote: nothing reads ~/.research."""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from tests.regression import lift_expected as LE
from tests.regression import store_io

ROW = "smollm2-135m__bf16__l40s__tp1__b16__i1024__o128__mixed__greedy__bi-eager"
INDEX = {"rows": {ROW: {"row": 4, "class": "FAIL", "dir": "/o/r4", "files": {"stages.txt": {"bytes": 5, "sha256": "c" * 64, "extracted": True}}}}}


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for k in ("RESEARCH_RUN_DIR", "RESEARCH_RUN_ID", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    (tmp_path / "home").mkdir()
    if not store_io.reachable():
        pytest.skip("the research store package is not importable here")
    return tmp_path


def _record(tmp_path: Path, row: str = ROW) -> str:
    tree = tmp_path / f"record-{row[:8]}"
    (tree / "record" / row).mkdir(parents=True)
    (tree / "record" / row / "verdict.json").write_text('{"schema": "verity-vllm/regression-result/v1"}\n')
    return store_io.put_fixture(tree, {"name": f"epoch-record/{row}", "role": "record", "row_id": row, "row": 4})


def _label(art: str, value: str, by: str) -> None:
    store_io._cli("label", art, "class", value, "--by", by, "--ref", "r20260929-194627-4b5f")


def _class(tmp_path: Path) -> tuple[str, str]:
    text, _ = LE.build_fixtures(INDEX, tmp_path, {"record"}, class_labels=LE.class_labels_for(INDEX["rows"]))
    spec = tomllib.loads(text)["rows"][ROW]
    return spec["class"], spec["class_source"]


def test_without_a_label_the_class_is_the_index_s(store: Path) -> None:
    _record(store)
    assert _class(store) == ("FAIL", "INDEX.json")


def test_a_class_label_by_daniel_on_the_record_of_record_wins(store: Path) -> None:
    art = _record(store)
    _label(art, "GREEN", "daniel")
    cls, source = _class(store)
    assert cls == "GREEN"
    assert source.startswith(f"label {art} by daniel ") and source.endswith("ref r20260929-194627-4b5f")


def test_a_class_label_by_anyone_else_is_not_read(store: Path) -> None:
    _label(_record(store), "GREEN", "vllm-epoch-run")
    assert _class(store) == ("FAIL", "INDEX.json")
