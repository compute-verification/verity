"""training_slowdown.starts: an installed start is read only when its files are the registry's, and the registry's
shapes are the model's."""
import dataclasses
import json

import pytest
import torch
from synthetic import tiny_start, write_safetensors

from training_slowdown import starts
from training_slowdown.model import LM


def test_an_installed_start_loads_when_its_files_are_the_registrys(tmp_path):
    s, theta = tiny_start(tmp_path)
    got = starts.load(s, tmp_path)
    assert sorted(got) == sorted(theta) and all(torch.equal(got[k], theta[k]) for k in theta)
    LM.of(s.shape).load_state_dict(got)


def test_a_start_whose_files_or_record_differ_is_refused(tmp_path):
    s, _ = tiny_start(tmp_path)
    with pytest.raises(SystemExit, match="sha256"):
        starts.load(dataclasses.replace(s, theta_file_sha256="0" * 64), tmp_path)
    with pytest.raises(SystemExit, match="digest, C0 and loss"):
        starts.load(dataclasses.replace(s, c0_macs=1.0), tmp_path)
    (tmp_path / s.name / "start.json").unlink()
    with pytest.raises(SystemExit, match=f"research data fetch {s.art} --path starts/{s.name}/start.json"):
        starts.load(s, tmp_path)


def test_a_start_names_the_parts_its_own_training_read_in_its_start_json(tmp_path):
    """`provenance.data.parts`, as the file states them; none of a start.json that says nothing of what it read; and
    a start.json that isn't the registry's is refused before anything is read from it."""
    parts = [{"index": 3, "sha256": "a" * 64, "tokens": 7}, {"index": 0, "sha256": "b" * 64}]
    named, _ = tiny_start(tmp_path / "named", read=parts)
    assert starts.parts_read(named, tmp_path / "named") == parts
    assert starts.record(named, tmp_path / "named")["provenance"]["data"] == {"parts": parts}
    silent, _ = tiny_start(tmp_path / "silent")
    assert starts.parts_read(silent, tmp_path / "silent") == [] and "data" not in starts.record(
        silent, tmp_path / "silent")["provenance"]
    assert starts.installed(silent, tmp_path / "silent") == tmp_path / "silent" / "tiny"
    with pytest.raises(SystemExit, match=f"start.json has sha256 {silent.start_json_sha256}, the registry's tiny"):
        starts.parts_read(dataclasses.replace(silent, start_json_sha256="0" * 64), tmp_path / "silent")
    with pytest.raises(SystemExit, match="digest, C0 and loss"):
        starts.parts_read(dataclasses.replace(silent, l0_bpb=1.0), tmp_path / "silent")


def test_a_safetensors_file_reads_back_its_tensors_misaligned_too(tmp_path):
    state = {"a": torch.arange(3, dtype=torch.bfloat16), "b": torch.randn(2, 3), "c": torch.zeros(0)}
    write_safetensors(tmp_path / "t.safetensors", state)  # b starts 6 bytes in: not a float32's alignment
    got = starts.tensors(starts.read_file(tmp_path / "t.safetensors"))
    assert all(torch.equal(got[k], v) and got[k].dtype == v.dtype for k, v in state.items())
    raw = (tmp_path / "t.safetensors").read_bytes()
    assert json.loads(raw[8: 8 + int.from_bytes(raw[:8], "little")])["b"]["data_offsets"][0] == 6


def test_each_registered_shape_is_the_models_parameters():
    for name in starts.registry()["starts"]:
        shape = starts.start(name).shape
        with torch.device("meta"):
            m = LM.of(shape)
        assert sum(p.numel() for p in m.parameters()) == shape["parameters"], name
