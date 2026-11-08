"""spot_proved's match gate over every call (`stage.py --every-call`) on the fake 8-rank capture of
`test_spot_stage_allreduce`: one instance of each symm-mem allreduce call is gated, the NCCL-route call is reported as not
modelled, and the recorded ops no type gates, the calls a population refused among them, are listed as not gated."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_call_gates_one_instance_of_each_call(tmp_path, monkeypatch):
    fake = _load("spot_fake", Path(__file__).with_name("test_spot_stage_allreduce.py"))
    S = _load("spot_stage_every", ROOT / "benchmarks" / "spot_proved" / "stage.py")
    monkeypatch.setitem(sys.modules, "spot_stage_every", S)  # the pool pickles its task by module name
    cap, out = tmp_path / "cap", tmp_path / "out"
    seqs = fake.fake_capture(cap, [(6144, "symm_mem"), (6144, "nccl"), (6144, "symm_mem")])
    monkeypatch.setattr(sys, "argv", ["stage.py", "--capture", str(cap), "--step", "0", "--out", str(out),
                                      "--every-call", "1", "--jobs", "2"])
    assert S.main() == 0
    rep = json.loads((out / "gate.json").read_text())
    t = rep["types"][f"AllReduceMultimem_W{fake.W}"]
    assert (t["calls"], t["calls_gated"], t["agree"], t["disagree"], t["errors"]) == (2, 2, 2, 0, 0)
    assert sorted(f["key"][0] for f in t["failures"]) == []
    assert rep["unstaged"][f"AllReduceNccl_W{fake.W}"]["calls"][0]["seq"] == seqs[1]
    assert rep["gated_ops"] == {"vllm::all_reduce": 2 * fake.W}
    assert rep["unstaged"]["AtenAdd_other"]["population"] == fake.W
    assert rep["not_gated_ops"] == {"aten::add": fake.W, "vllm::all_reduce": fake.W}
    assert rep["whole"] is False and rep["layers"] == [fake.LAYER]
