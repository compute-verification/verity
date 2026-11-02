"""Script mode: a covert_witness-shaped train.py runs unmodified under the harness (cw.py), and the counter recognizes θ's
weights by value after a recast."""
import json
import math
from pathlib import Path

import pytest
import torch

from te_core import run as te_run
from te_core.flops import Counter

FAKE = Path(__file__).parent / "cw_fake"


@pytest.fixture
def cw_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CW_FAKE_ROOT", str(tmp_path / "root"))
    for split in ("search", "final"):
        d = tmp_path / "root" / "eval" / split
        d.mkdir(parents=True)
        torch.save(torch.randint(0, 32, (4, 9), generator=torch.Generator().manual_seed(7)), d / "rows.pt")
    return tmp_path


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
