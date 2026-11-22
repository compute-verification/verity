"""The adapter's one statement (`verity_vllm.inputs.serves.SERVES`) and its door, on processors: the remote applications' input
dataset passes it whole, and each record or application it can't take is refused by name before anything is sent."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import input_datasets
import pytest

from verity_vllm.inputs import serves
from verity_vllm.pipeline.advice_windows import CAP_BITS

ROOT = Path(__file__).resolve().parents[5]
INPUTS = str(ROOT / "adapters/vllm/tools/served_run/inputs.py")
S = serves.SERVES
DATASET = serves.application(str(ROOT / "applications/remote.toml"))[0]
BODY = {"model": S.model, "user": "u.0", "prompt": [128000] * S.prompt, "max_tokens": S.outputs, "temperature": 0.0,
        "stream": True, "ignore_eos": True}


def test_the_remote_dataset_passes_the_door_whole():
    ds = input_datasets.open(DATASET)
    assert serves.door(ds.kind, ds) == list(input_datasets.open(DATASET))


@pytest.mark.parametrize("kind, records, name, said", [
    ("chat-requests", [BODY], "kind", "the vLLM adapter plays input datasets of kind completion-requests, not chat-requests"),
    ("completion-requests", [{**BODY, "model": "other/model"}], "model",
     "record 0 (u.0) asks for model 'other/model', and the adapter serves unsloth/Llama-3.2-1B"),
    ("completion-requests", [{**BODY, "prompt": "hello"}], "prompt",
     "record 0 (u.0): its prompt is not a non-empty list of token ids below the vocabulary's 128256"),
    ("completion-requests", [{**BODY, "prompt": [128256]}], "prompt",
     "record 0 (u.0): its prompt is not a non-empty list of token ids below the vocabulary's 128256"),
    ("completion-requests", [BODY, {**BODY, "user": "u.1", "prompt": [1] * (S.prompt + 1)}], "positions",
     f"record 1 (u.1): its prompt is {S.prompt + 1} ids, past the class {S.window_class}'s {S.prompt}"),
    ("completion-requests", [{**BODY, "max_tokens": S.outputs + 1}], "steps",
     f"record 0 (u.0): max_tokens {S.outputs + 1}, not 1 to the class {S.window_class}'s {S.outputs} outputs"),
    ("completion-requests", [{**BODY, "temperature": 0.7}], "sampling",
     "record 0 (u.0): temperature 0.7 and n 1, and the class serves one greedy completion a request"),
    ("completion-requests", [BODY] * (S.slots + 1), "slots",
     f"the dataset has {S.slots + 1} records, past the class {S.window_class}'s {S.slots} slots"),
], ids=["kind", "model", "prompt text", "prompt id", "positions", "steps", "sampling", "slots"])
def test_the_door_refuses_the_first_record_it_cannot_take_by_name(kind, records, name, said):
    with pytest.raises(serves.Refused) as e:
        serves.door(kind, records)
    assert serves.refusal(e.value) == (name, said)


def test_an_application_the_statement_is_not_and_a_timing_it_cannot_read_are_refused_by_name():
    s = serves.SERVES
    assert serves.served(s.model, s.revision, s.precision, s.slots, s.outputs) is s
    with pytest.raises(serves.Refused) as e:
        serves.served(s.model, s.revision, s.precision, s.slots, s.outputs + 14)
    assert e.value.what == "not-served" and "not unsloth/Llama-3.2-1B" in str(e.value) and f"{s.outputs + 14} window steps" in str(e.value)
    assert serves.timing("at-once") == ("at-once", {}) and serves.timing("rate:2.5") == ("rate", {"rate": 2.5})
    for bad in ("rate:0", "rate:x", "poisson", ""):
        with pytest.raises(serves.Refused) as e:
            serves.timing(bad)
        assert e.value.what == "timing"


def test_the_command_prints_the_statement_and_the_doors_answer():
    run = lambda *a: subprocess.run([sys.executable, INPUTS, *a], capture_output=True, text=True)  # noqa: E731
    p = run("statement")
    assert p.returncode == 0 and p.stdout.split() == [S.model, S.revision, "bfloat16", S.window_class, str(CAP_BITS)]
    p = run("door", DATASET, "at-once")
    identity = input_datasets.open(DATASET).timed("at-once").identity
    assert p.returncode == 0 and p.stdout == f"DOOR {DATASET}: {len(input_datasets.open(DATASET))} records the class {S.window_class} takes; identity {identity}\n"
    p = run("door", "no_such_dataset")
    assert p.returncode == 3 and p.stdout.startswith("REFUSED unknown-dataset: no input dataset 'no_such_dataset'")


def test_an_application_gives_its_dataset_and_timing_once_the_statement_takes_it(tmp_path):
    good = tmp_path / "good.toml"
    good.write_text('[workload.params]\nmodel = "unsloth/Llama-3.2-1B"\nrevision = "9535bd9b1d1dea6acafbdc4813b728796aeb28da"\n'
                    f'precision = "bfloat16"\nslots = {S.slots}\nwindow-steps = {S.outputs}\ndataset = "{DATASET}"\ntiming = "rate:2"\n')
    assert serves.application(str(good)) == (DATASET, "rate:2")
    p = subprocess.run([sys.executable, INPUTS, "application", str(good)], capture_output=True, text=True)
    first, door, _ = p.stdout.split("\n")
    assert p.returncode == 0 and first.split()[3:] == [S.window_class, DATASET, "rate:2", str(CAP_BITS)] and door.startswith(f"DOOR {DATASET}: ")
    varied = tmp_path / "varied.toml"
    varied.write_text(good.read_text().replace(f'dataset = "{DATASET}"\ntiming = "rate:2"\n', 'load = "varied"\n'))
    with pytest.raises(serves.Refused) as e:
        serves.application(str(varied))
    assert e.value.what == "application" and "names no model, window, dataset and timing" in str(e.value)
