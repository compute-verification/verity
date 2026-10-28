from __future__ import annotations

import ast
import dataclasses
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from verity_config import Admission, ConfigError, dumps, gpus, load, sha256


@dataclass(frozen=True)
class Child:
    enabled: bool


@dataclass(frozen=True)
class Example:
    name: str
    values: tuple[int, ...]
    child: Child
    admission: Admission = Admission()


def test_toml_loads_into_nested_frozen_dataclasses_and_refuses_unknown_keys(tmp_path):
    good = tmp_path / "good.toml"
    good.write_text('name = "α"\nvalues = [1, 2]\n[child]\nenabled = true\n')
    got = load(good, Example)
    assert got == Example("α", (1, 2), Child(True))
    with pytest.raises(dataclasses.FrozenInstanceError):
        got.name = "changed"  # type: ignore[misc]

    bad = tmp_path / "bad.toml"
    bad.write_text('name = "x"\nvalues = []\nsurprise = 1\n[child]\nenabled = true\n')
    with pytest.raises(ConfigError, match=r"unknown key\(s\): surprise"):
        load(bad, Example)


def test_canonical_json_is_utf8_sorted_compact_and_its_sha256_is_the_identity():
    doc = {"z": [1, True, None], "é": "snowman ☃", "a": -0.0}
    encoded = dumps(doc)
    assert encoded == '{"a":0,"z":[1,true,null],"é":"snowman ☃"}'.encode()
    assert sha256(doc) == hashlib.sha256(encoded).hexdigest()


def test_admission_cost_deadline_and_duration_do_not_change_identity():
    base = Example("same", (1,), Child(True), Admission(1.5, "2026-10-14T13:00:00Z", 60))
    other = Example("same", (1,), Child(True), Admission(20.0, "2027-01-01T00:00:00Z", 240))
    assert dumps(base) != dumps(other)
    assert dumps(base, identity=True) == dumps(other, identity=True)
    assert sha256(base, identity=True) == sha256(other, identity=True)


def test_gpu_registry_keeps_namespaces_distinct():
    gpu = gpus()["rtx-pro-6000"]
    assert gpu.id != gpu.census_id != gpu.nvidia_smi_name
    assert gpus()["h200"].nebius_platform == "gpu-h200-sxm"


def test_package_imports_only_the_standard_library():
    package = Path(__file__).parents[1] / "verity_config"
    imports = set()
    for source in package.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                imports.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module.partition(".")[0])
    assert not {name for name in imports if name not in sys.stdlib_module_names and name != "__future__"}
