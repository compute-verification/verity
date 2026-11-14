"""verity_vllm.program.predict.compiled: the two commands and `pin_of` open each file by the name they are given, as `open` takes
it. A name with a slash after it names no file, the empty name names none, and a file's name in bytes is read."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from verity.definitions.registry.bf16_gemm_check import device_pin

from verity_vllm.config import UsageError
from verity_vllm.program.predict import compiled as PC

RTX = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
BUILD = {"program_digest": "0" * 128, "manifest_digest": "1" * 64, "artifact": {"digest": "2" * 16}, "identities": [], "complete": True}


def _written(path: Path, doc: Any) -> str:
    path.write_text(json.dumps(doc))
    return str(path)


def test_the_command_opens_its_workload_by_the_name_given(tmp_path, monkeypatch):
    """`predict compiled` reads ``--workload`` through `open`: a workload that states no steps is read and then refused for
    them, by its name in a str or in bytes, and the name with a slash after it and the empty name are `open`'s errors."""
    name = _written(tmp_path / "workload.json", {"requests": []})
    a = PC.Options(role="LLAMA32_1B", device=RTX, workload=name, out=str(tmp_path / "build.json"))
    in_bytes: Any = os.fsencode(name)
    for given in (name, in_bytes):
        with pytest.raises(UsageError, match="--steps"):
            PC.main(replace(a, workload=given))
    with pytest.raises(NotADirectoryError):
        PC.main(replace(a, workload=name + "/"))
    monkeypatch.chdir(tmp_path)  # the suites' read guard takes an open of the empty name for one of the working directory
    with pytest.raises(FileNotFoundError):
        PC.main(replace(a, workload=""))


def test_the_command_opens_the_build_it_compares_with_by_the_name_given(tmp_path, monkeypatch):
    """`predict compiled --compare` reads the traced Build through `open`: its name in bytes is read and compared, and the name
    with a slash after it is `open`'s error. `predict` is stood in: the comparison reads only what it returns."""
    monkeypatch.setattr(PC, "predict", lambda role, device, schedule, steps: dict(BUILD))
    seen: list[Any] = []

    def differences(predicted: dict, traced: Any) -> list[str]:
        seen.append(traced)
        return []

    monkeypatch.setattr(PC, "differences", differences)
    name = _written(tmp_path / "traced.json", {"traced": True})
    a = PC.Options(role="LLAMA32_1B", device=RTX, workload=_written(tmp_path / "workload.json", {"requests": []}), steps=1,
                   out=str(tmp_path / "build.json"))
    in_bytes: Any = os.fsencode(name)
    assert PC.main(replace(a, compare=in_bytes)) == 0 and seen == [{"traced": True}]
    with pytest.raises(NotADirectoryError):
        PC.main(replace(a, compare=name + "/"))


def _pin_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[PC.PinOptions, list[Any]]:
    """`predict compiled-pin`'s options over a Build and its ``routes.json`` in ``tmp_path``, and the list that takes the
    fused-MoE record each call hands `pin_of`, which is stood in: it reads the Inductor dir of a serving process."""
    seen: list[Any] = []

    def pin_of(build: dict, inductor_dir: str, moe: Any, **kw: Any) -> dict[str, Any]:
        seen.append(moe)
        return {"kernels": {}, "artifact": {"winners": []}}

    monkeypatch.setattr(PC, "pin_of", pin_of)
    build = _written(tmp_path / "build.json", {"program": {}})
    _written(tmp_path / "routes.json", {"device": RTX})
    return PC.PinOptions(build=build, inductor_dir="inductor", role="R", evidence="r0", out=str(tmp_path / "pin.json")), seen


def test_the_pin_command_opens_its_build_by_the_name_given(tmp_path, monkeypatch):
    """`predict compiled-pin` reads ``--build`` through `open`: the empty name is `open`'s error, and no pin is made."""
    a, seen = _pin_command(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)  # as in the workload's test
    with pytest.raises(FileNotFoundError):
        PC.main(replace(a, build=""))
    assert PC.main(a) == 0 and seen == [None]


def test_the_pin_command_opens_its_moe_record_by_the_name_given(tmp_path, monkeypatch):
    """`predict compiled-pin` reads ``--moe`` through `open`: the fused-MoE record's name in bytes is read, and that name with a
    slash after it is `open`'s error before any pin is made."""
    a, seen = _pin_command(tmp_path, monkeypatch)
    moe = _written(tmp_path / "moe.json", {"moe": 1})
    in_bytes: Any = os.fsencode(moe)
    assert PC.main(replace(a, moe=in_bytes)) == 0 and seen == [{"moe": 1}]
    with pytest.raises(NotADirectoryError):
        PC.main(replace(a, moe=moe + "/"))
    assert seen == [{"moe": 1}]


def test_pin_of_opens_a_piece_wrapper_by_the_name_its_build_gives(tmp_path):
    """`pin_of` reads each of the Build's ``piece_wrappers`` through `open`, joined to the Inductor dir, and pins the kernels the
    piece launches by their sources' digests; a wrapper the Build names with a slash after it is `open`'s error, not the file
    of that name."""
    source = "\nimport triton\n"
    (tmp_path / "piece0.py").write_text(f"k0 = async_compile.triton('k0', '''{source}''', device_str='cuda')\n")
    dev = device_pin(RTX)
    program = {"piece_wrappers": ["piece0.py"], "calls": [{"piece": 0, "op": "k0"}, {"piece": 0, "op": "not_in_the_wrapper"}],
               "dispatch": {"sms": dev["sms"], "cc": dev["compute_capability"], "max_num_seqs": 4}}
    build = {"program": program, "artifact": {"kernels": ["k"], "winners": ["w"], "digest": "d"}}
    assert PC.pin_of(build, str(tmp_path), None, role="R", device=RTX, evidence="r0") == {
        "role": "R", "device": RTX, "evidence": "r0", "dispatch": {"max_num_seqs": 4},
        "kernels": {"first": {"k0": hashlib.sha256(source.encode()).hexdigest()}, "middle": {}, "last": {}},
        "artifact": {"kernels": ["k"], "winners": ["w"]}}
    with pytest.raises(NotADirectoryError):
        PC.pin_of({**build, "program": {**program, "piece_wrappers": ["piece0.py/"]}}, str(tmp_path), None, role="R", device=RTX, evidence="r0")
