"""The Build's identity stamps: `construction_version` hashes the builder's own sources as they are in the integration (an edited source
is another value, a missing one raises), and `model_pin.dtype` is the dtype the engine config runs the checkpoint in."""
from __future__ import annotations

import hashlib
import os
import shutil
from types import SimpleNamespace

import pytest
import torch

import verity_vllm
from verity_vllm.harness import derive_step as ds

INTEGRATION = os.path.dirname(os.path.dirname(os.path.abspath(verity_vllm.__file__)))


def test_construction_version_hashes_the_integration_sources():
    cv = ds.construction_version([])
    assert [f["file"] for f in cv["files"]] == list(ds._CONSTRUCTION_SOURCES)
    for f in cv["files"]:
        with open(os.path.join(INTEGRATION, f["file"]), "rb") as fh:
            assert f["sha256"] == hashlib.sha256(fh.read()).hexdigest(), f["file"]


def test_construction_version_changes_with_a_listed_source_and_refuses_a_missing_one(tmp_path):
    for rel in ds._CONSTRUCTION_SOURCES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(os.path.join(INTEGRATION, rel), tmp_path / rel)
    a = ds.construction_version([], root=str(tmp_path))
    assert a == ds.construction_version([])
    edited = tmp_path / ds._CONSTRUCTION_SOURCES[-1]
    edited.write_bytes(edited.read_bytes() + b"\n# edited\n")
    assert ds.construction_version([], root=str(tmp_path))["sources_sha256"] != a["sources_sha256"]
    edited.unlink()
    with pytest.raises(FileNotFoundError, match=ds._CONSTRUCTION_SOURCES[-1]):
        ds.construction_version([], root=str(tmp_path))


@pytest.mark.parametrize("quantization,dtype,expected", [(None, torch.bfloat16, "bfloat16"), ("fp8", torch.bfloat16, "fp8")])
def test_engine_dtype_is_the_dtype_the_engine_config_runs(quantization, dtype, expected):
    assert ds.engine_dtype(SimpleNamespace(quantization=quantization, dtype=dtype)) == expected
