"""Script mode: a covert_witness-shaped train.py runs unmodified under the harness (cw.py), and the counter recognizes θ's
weights by value after a recast."""
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.overrides import TorchFunctionMode

from te_core import run as te_run
from te_core.flops import Counter, plain

FAKE = Path(__file__).parent / "cw_fake"


def fake_root(root: Path, monkeypatch) -> Path:
    """A held-out set under `root`. A run's audit hook outlives it, so two runs in one process need two roots."""
    monkeypatch.setenv("CW_FAKE_ROOT", str(root / "root"))
    for split in ("search", "final"):
        d = root / "root" / "eval" / split
        d.mkdir(parents=True)
        torch.save(torch.randint(0, 32, (4, 9), generator=torch.Generator().manual_seed(7)), d / "rows.pt")
    return root


@pytest.fixture
def cw_root(tmp_path, monkeypatch):
    return fake_root(tmp_path, monkeypatch)


def run_island(tmp_path, *island_args) -> tuple[int, dict]:
    out = tmp_path / "out"
    rc = te_run.main(["--script", str(FAKE / "islands" / "sgd" / "train.py"), "--out", str(out), "--",
                      "--start", "fake", "--out", str(tmp_path / "sub"), *island_args])
    return rc, json.loads((out / "run.json").read_text())


def test_an_island_runs_unmodified_and_is_scored_at_its_encodings(cw_root):
    rc, rec = run_island(cw_root, "--steps", "8")
    assert rc == 0 and rec["invalid"] == []
    assert len(rec["steps"]) == 8 and all(s["tokens"] == 4 * 8 for s in rec["steps"])
    assert rec["setup"]["tokens"] == 0 and rec["tail"]["tokens"] == 0
    # the fp32 recast of θ is θ: step 0's forward reads θ's values, every later one the updated weights
    assert "fwd/updated" not in rec["steps"][0]["by"] and rec["steps"][0]["by"]["fwd/theta"] > 0
    assert rec["steps"][1]["by"]["fwd/updated"] > 0
    # the buffer multiplies like θ: it is the architecture's, not a weight
    assert not any(k.startswith("buffer") for k in rec["new_weights"])
    cps = rec["checkpoints"]
    assert [cp["step"] for cp in cps] == [1, 3, 5, 7]
    assert all(cp["source"] == "encode" and math.isfinite(cp["val_bpb"]) for cp in cps)
    assert all(cp["format"] == "full" and cp["rebuild_macs"] == 0 for cp in cps)
    assert cps[-1]["submitted"] and rec["submitted"]["model_is_decode"]
    assert cps[0]["macs"] < cps[-1]["macs"] == rec["totals"]["macs"]
    assert rec["start"]["val_bpb"] > cps[-1]["val_bpb"]
    assert rec["c"] == 4096 * rec["forward_macs_per_token"]


def test_the_held_out_set_is_unreadable_to_the_island(cw_root):
    rc, rec = run_island(cw_root, "--steps", "2", "--peek")
    assert rc == 1 and "PermissionError" in rec["error"]
    assert any("PermissionError" in p for p in rec["invalid"])
    assert rec["start"]["val_bpb"] is not None


def test_a_recast_copy_of_theta_is_theta_and_a_lossy_one_is_not():
    torch.manual_seed(0)
    theta = {"w": torch.randn(8, 8)}
    c = Counter(theta, digests=False)
    w = torch.nn.Parameter(theta["w"].clone())
    c.register({"w": w})
    idx = torch.tensor([[1, 2, 3]])
    emb = torch.randn(10, 8)
    with c:
        c.begin_step()
        w.data = w.data.double()
        torch.nn.functional.embedding(idx, emb).double() @ w.T
        lossy = theta["w"].bfloat16()  # a cast of θ inside a step is computed from θ; carried, it is a weight
        c.end_step()
        torch.nn.functional.embedding(idx, emb).bfloat16() @ lossy.T
        c.end_step()
        with torch.no_grad():
            w.add_(1e-12)
        torch.nn.functional.embedding(idx, emb).double() @ w.T
        c.end_step()
    by = [s.by for s in c.steps]
    assert "fwd/updated" not in by[0] and by[0]["fwd/theta"] == 3 * 8 * 8
    assert by[1]["fwd/updated"] == 3 * 8 * 8
    assert by[2]["fwd/updated"] == 3 * 8 * 8


def test_the_screen_reads_names_not_substrings():
    assert te_run.screen("t0 = time.perf_counter()\nfwd = torch.compile(net)\nfrom te_core import hooks\n") == []
    for bad in ("te.counter.paused = 1", "from te_core import flops", "import te_core.run", "torch._dynamo.reset()",
                "with uncounted():", "sys.modules['x']"):
        assert te_run.screen(bad), bad


def run_probe(tmp_path, monkeypatch, variant: str) -> dict:
    out = fake_root(tmp_path / variant, monkeypatch) / "out"
    te_run.main(["--script", str(FAKE / "islands" / "probe" / "train.py"), "--out", str(out), "--",
                 "--start", "fake", "--out", str(out / "sub"), "--variant", variant])
    return json.loads((out / "run.json").read_text())


def test_the_programs_code_in_what_it_hands_the_encoder_is_counted(tmp_path, monkeypatch):
    base, hidden = run_probe(tmp_path, monkeypatch, "baseline"), run_probe(tmp_path, monkeypatch, "dict_encoder")
    assert base["invalid"] == [] and hidden["invalid"] == []
    (b,), (h,) = base["steps"], hidden["steps"]
    assert h["tokens"] == b["tokens"] == 4 * 8
    assert h["by"]["fwd/updated"] - b["by"]["fwd/updated"] == 50 * 4 * 8 * 16 * 16


def test_the_harness_scores_with_its_own_prepare(tmp_path, monkeypatch):
    """A program that rebinds prepare.decode changes its own decoding, not the checkpoints' or the bit-equality check's."""
    rec = run_probe(tmp_path, monkeypatch, "rebind_decode")
    (cp,) = rec["checkpoints"]
    assert cp["val_bpb"] == rec["start"]["val_bpb"] and not rec["submitted"]["model_is_decode"]


def test_rebinding_what_the_harness_calls_makes_the_run_invalid(tmp_path, monkeypatch):
    rec = run_probe(tmp_path, monkeypatch, "rebind_torch")
    assert any("torch.nn.functional.cross_entropy" in p for p in rec["invalid"]), rec["invalid"]


def test_writing_the_theta_load_start_returned_does_not_move_the_harness_theta(tmp_path, monkeypatch):
    base, moved = run_probe(tmp_path, monkeypatch, "baseline"), run_probe(tmp_path, monkeypatch, "theta_write")
    assert moved["invalid"] == [] and moved["steps"][0]["by"] == base["steps"][0]["by"]
    assert moved["theta_sha256"] == base["theta_sha256"]


def test_the_stream_batch_is_token_data_without_an_embedding(tmp_path, monkeypatch):
    rec = run_probe(tmp_path, monkeypatch, "index_lookup")
    assert rec["invalid"] == [], rec["invalid"]
    (s,) = rec["steps"]
    assert s["tokens"] == 4 * 8 and s["by"]["fwd/updated"] == 4 * 8 * 16 * 16


def test_plain_copies_containers_and_refuses_what_could_run_code():
    class Sub(dict):
        pass

    class T(torch.Tensor):
        pass

    t = torch.randn(3)
    out = plain(Sub(a=t, b=[1, (2.0, "x")], c=np.float32(1.5), d=Path("/x"), p=torch.nn.Parameter(t), e=bytearray(b"1")))
    assert type(out) is dict and type(out["a"]) is torch.Tensor and type(out["p"]) is torch.Tensor
    assert out["b"] == [1, (2.0, "x")] and type(out["c"]) is float and out["d"] == Path("/x") and out["e"] == b"1"
    for bad in (t.as_subclass(T), object(), {"k": lambda: 0}):
        with pytest.raises(TypeError):
            plain(bad)


def test_the_harness_work_runs_no_mode_hook_or_profiler_of_the_programs(tmp_path):
    te = te_run.Session({}, tmp_path)
    seen = []

    class Mode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            seen.append("mode")
            return func(*args, **(kwargs or {}))

    def profiler(*args):
        return None

    lin = torch.nn.Linear(2, 2)
    hook = torch.nn.modules.module.register_module_forward_hook(lambda m, i, o: seen.append("hook"))
    sys.setprofile(profiler)
    try:
        with Mode():
            with te.uncounted():
                n, collecting, profiling = len(seen), gc.isenabled(), sys.getprofile()
                lin(torch.ones(2))
                inside = seen[n:]
            lin(torch.ones(2))
        restored = sys.getprofile() is profiler
    finally:
        sys.setprofile(None)
        hook.remove()
    assert inside == [] and not collecting and profiling is None and restored and gc.isenabled()
    assert "mode" in seen and "hook" in seen



def run_mezo(root, *island_args) -> tuple[int, Path, dict]:
    out = root / "out"
    rc = te_run.main(["--script", str(FAKE / "islands" / "mezo" / "train.py"), "--out", str(out), "--",
                      "--start", "fake", "--out", str(root / "sub"), *island_args])
    return rc, out, json.loads((out / "run.json").read_text())


def test_declared_steps_are_a_family_rebuilt_by_prepares_decoder(cw_root):
    """Each step's two perturbed reads are credited through `prepare.declare_step` (cw.CWPlan)."""
    rc, _, rec = run_mezo(cw_root)
    assert rc == 0 and rec["invalid"] == [], rec["invalid"]
    assert rec["digest_key"] and all(len(ds) == 2 for ds in rec["steps"][2]["weights"].values())
    (fam,) = rec["families"]
    assert fam["family"] == "E" and fam["failed"] == {} and fam["covers"] == list(range(6))
    assert all(b > 0 for b in fam["beta"].values())


def test_a_step_that_reads_what_it_did_not_declare_fails_its_rebuild(cw_root):
    import sys

    from te_core.counterexample import check
    from te_core.cw import CWPlan

    rc, out, rec = run_mezo(cw_root, "--bad-step", "3")
    prepare = sys.modules["prepare"]
    theta = prepare.theta_of()
    v = check(out / "families" / "E", rec, theta, rec["c"], n_sample=6, coin=b"x", plan=CWPlan(prepare, theta, "cpu"))
    assert list(v.failed) == [3] and v.covers == [0, 1, 2, 4, 5]
