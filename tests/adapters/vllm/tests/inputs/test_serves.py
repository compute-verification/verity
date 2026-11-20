"""The adapter's one statement (`verity_vllm.inputs.serves.SERVES`) and its door, on processors: the remote applications' input
dataset passes it whole, and each record or application it can't take is refused by name before anything is sent."""
from __future__ import annotations

import subprocess
import sys

import input_datasets
import pytest

from verity_vllm.inputs import serves

DATASET = "llama32_1b_fixed_4_27_2"
BODY = {"model": serves.SERVES.model, "user": "u.0", "prompt": [128000, 5, 6], "max_tokens": 2, "temperature": 0.0,
        "stream": True, "ignore_eos": True}


def test_the_remote_dataset_passes_the_door_whole():
    ds = input_datasets.open(DATASET)
    assert serves.door(ds.kind, ds) == list(input_datasets.open(DATASET))
    assert serves.SERVES.window_class == "4,27,2" and (serves.SERVES.slots, serves.SERVES.prompt, serves.SERVES.outputs) == (4, 27, 2)


@pytest.mark.parametrize("kind, records, name, said", [
    ("chat-requests", [BODY], "kind", "the vLLM adapter plays input datasets of kind completion-requests, not chat-requests"),
    ("completion-requests", [{**BODY, "model": "other/model"}], "model",
     "record 0 (u.0) asks for model 'other/model', and the adapter serves unsloth/Llama-3.2-1B"),
    ("completion-requests", [{**BODY, "prompt": "hello"}], "prompt",
     "record 0 (u.0): its prompt is not a non-empty list of token ids below the vocabulary's 128256"),
    ("completion-requests", [{**BODY, "prompt": [128256]}], "prompt",
     "record 0 (u.0): its prompt is not a non-empty list of token ids below the vocabulary's 128256"),
    ("completion-requests", [BODY, {**BODY, "user": "u.1", "prompt": [1] * 28}], "positions",
     "record 1 (u.1): its prompt is 28 ids, past the class 4,27,2's 27"),
    ("completion-requests", [{**BODY, "max_tokens": 3}], "steps",
     "record 0 (u.0): max_tokens 3, not 1 to the class 4,27,2's 2 outputs"),
    ("completion-requests", [{**BODY, "temperature": 0.7}], "sampling",
     "record 0 (u.0): temperature 0.7 and n 1, and the class serves one greedy completion a request"),
    ("completion-requests", [BODY] * 5, "slots", "the dataset has 5 records, past the class 4,27,2's 4 slots"),
], ids=["kind", "model", "prompt text", "prompt id", "positions", "steps", "sampling", "slots"])
def test_the_door_refuses_the_first_record_it_cannot_take_by_name(kind, records, name, said):
    with pytest.raises(serves.Refused) as e:
        serves.door(kind, records)
    assert (e.value.name, str(e.value)) == (name, said)


def test_an_application_the_statement_is_not_and_a_timing_it_cannot_read_are_refused_by_name():
    s = serves.SERVES
    assert serves.served(s.model, s.revision, s.precision, 4, 2) is s
    with pytest.raises(serves.Refused) as e:
        serves.served(s.model, s.revision, s.precision, 4, 16)
    assert e.value.name == "not-served" and "not unsloth/Llama-3.2-1B" in str(e.value) and "16 window steps" in str(e.value)
    assert serves.timing("at-once") == ("at-once", {}) and serves.timing("rate:2.5") == ("rate", {"rate": 2.5})
    for bad in ("rate:0", "rate:x", "poisson", ""):
        with pytest.raises(serves.Refused) as e:
            serves.timing(bad)
        assert e.value.name == "timing"


def test_the_command_prints_the_statement_and_the_doors_answer():
    run = lambda *a: subprocess.run([sys.executable, "-m", "verity_vllm.inputs.serves", *a], capture_output=True, text=True)  # noqa: E731
    p = run()
    assert p.returncode == 0 and p.stdout.split() == [serves.SERVES.model, serves.SERVES.revision, "bfloat16", "4,27,2"]
    p = run("door", DATASET, "at-once")
    identity = input_datasets.open(DATASET).timed("at-once").identity
    assert p.returncode == 0 and p.stdout == f"DOOR {DATASET}: 4 records the class 4,27,2 takes; identity {identity}\n"
    p = run("door", "no_such_dataset")
    assert p.returncode == 3 and p.stdout.startswith("REFUSED unknown-dataset: no input dataset 'no_such_dataset'")


def test_an_application_gives_its_dataset_and_timing_once_the_statement_takes_it(tmp_path):
    good = tmp_path / "good.toml"
    good.write_text('[workload.params]\nmodel = "unsloth/Llama-3.2-1B"\nrevision = "9535bd9b1d1dea6acafbdc4813b728796aeb28da"\n'
                    f'precision = "bfloat16"\nslots = 4\nwindow-steps = 2\ndataset = "{DATASET}"\ntiming = "rate:2"\n')
    assert serves.application(str(good)) == (DATASET, "rate:2")
    p = subprocess.run([sys.executable, "-m", "verity_vllm.inputs.serves", "application", str(good)], capture_output=True,
                       text=True)
    first, door, _ = p.stdout.split("\n")
    assert p.returncode == 0 and first.split()[3:] == ["4,27,2", DATASET, "rate:2"] and door.startswith(f"DOOR {DATASET}: ")
    varied = tmp_path / "varied.toml"
    varied.write_text(good.read_text().replace(f'dataset = "{DATASET}"\ntiming = "rate:2"\n', 'load = "varied"\n'))
    with pytest.raises(serves.Refused) as e:
        serves.application(str(varied))
    assert e.value.name == "application" and "names no model, window, dataset and timing" in str(e.value)
