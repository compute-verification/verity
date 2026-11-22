"""An application's load: a ``load``, or an input ``dataset`` at its ``timing`` that the adapter's door reads through when the
configuration is loaded, so `verity run --dry-run` refuses a dataset the served class can't take before any node is touched."""
from __future__ import annotations

import copy
import dataclasses
import re
from pathlib import Path

import pytest
from verity import config as K
from verity.channel import Refused
from verity_cli import run as R
from verity_cli import served_table as T
from verity_vllm.inputs import serves

ROOT = Path(__file__).resolve().parents[4]
REMOTE = ROOT / "applications" / "remote.toml"
REG = R.registry()


def loaded(**params) -> dict:
    cfg = copy.deepcopy(K.parse(REMOTE.read_text()))
    for k, v in params.items():
        if v is None:
            cfg["workload"]["params"].pop(k, None)
        else:
            cfg["workload"]["params"][k] = v
    return K.check(cfg, REG.components, REG.workloads)


def test_the_remote_application_names_an_input_dataset_and_its_timing():
    params = K.parse(REMOTE.read_text())["workload"]["params"]
    assert params["timing"] == "at-once" and "load" not in params
    assert loaded()["workload"]["params"]["dataset"] == params["dataset"] == serves.application(str(REMOTE))[0]


@pytest.mark.parametrize("params, said", [
    ({"load": "varied"}, "dataset, load, timing"),
    ({"dataset": None, "timing": None}, "neither"),
    ({"timing": None}, "dataset"),
    ({"dataset": None, "load": "varied"}, "load, timing"),
], ids=["both", "neither", "dataset-alone", "load-and-timing"])
def test_a_load_named_both_ways_or_neither_is_refused_by_name(params, said):
    with pytest.raises(Refused) as e:
        loaded(**params)
    assert e.value.name == "load" and e.value.detail.endswith(f"one of the two whole, not {said}"), e.value.detail


def test_the_retired_fixed_load_is_refused_with_the_dataset_to_name():
    with pytest.raises(Refused) as e:
        loaded(dataset=None, timing=None, load="fixed")
    assert e.value.name == "load" and "not 'fixed'" in e.value.detail and "dataset and timing" in e.value.detail


@pytest.mark.parametrize("params, name", [
    ({"dataset": "nothing_by_this_name"}, "unknown-dataset"),
    ({"timing": "soon"}, "timing"),
    ({"slots": serves.SERVES.slots + 4}, "not-served"),
], ids=["unknown-dataset", "timing", "not-served"])
def test_a_dataset_the_adapter_does_not_take_is_refused_by_the_doors_name(params, name):
    with pytest.raises(Refused) as e:
        loaded(**params)
    assert e.value.name == name and "[workload.params] dataset" in e.value.detail, e.value.detail


def test_the_dry_run_refuses_a_dataset_past_the_class_before_any_node(monkeypatch, capsys, tmp_path):
    """The four-request fixed load against a class of 16 prompt ids, under an application of four slots and two steps."""
    text = re.sub(r'^(slots|window-steps|dataset) = .*$', lambda m: {"slots": "slots = 4", "window-steps": "window-steps = 2"}.get(
        m[1], 'dataset = "llama32_1b_fixed_4_27_2"'), REMOTE.read_text(), flags=re.M)
    (app := tmp_path / "app.toml").write_text(text)
    monkeypatch.setattr(serves, "SERVES", dataclasses.replace(serves.SERVES, window_class="4,16,2"))
    rc = R.run_main([str(app), "--dry-run"], REG)
    err = capsys.readouterr().err
    assert rc == 2 and "refused positions:" in err and "past the class 4,16,2's 16" in err, err


def test_a_dataset_application_runs_its_class_from_the_adapters_statement():
    cfg = loaded()
    hooks = T.SERVE.variables(cfg, {})
    assert {k: hooks[k] for k in ("LOAD", "DATASET", "TIMING", "V3C_CLASS")} == {
        "LOAD": "dataset", "DATASET": serves.application(str(REMOTE))[0], "TIMING": "at-once",
        "V3C_CLASS": serves.SERVES.window_class}
