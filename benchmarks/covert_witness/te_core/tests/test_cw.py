"""Script mode: a covert_witness-shaped train.py runs unmodified under the harness (cw.py), and the counter recognizes θ's
weights by value after a recast."""
import json
import math
import subprocess
import sys
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


def run_island(tmp_path, *island_args, harness=()) -> tuple[int, dict]:
    out = tmp_path / "out"
    rc = te_run.main(["--script", str(FAKE / "islands" / "sgd" / "train.py"), "--out", str(out), *harness, "--",
                      "--start", "fake", "--out", str(tmp_path / "sub"), *island_args])
    return rc, json.loads((out / "run.json").read_text())


SCORE = """import json, sys
sys.path[:0] = sys.argv[1:3]
import prepare
from te_core.cw import score_run
out = sys.argv[3]
rec = json.load(open(out + "/run.json"))
start = {"name": "fake", "digest": "fake", "config": {"vocab": prepare.VOCAB, "ctx": prepare.EVAL_CTX}}
rec = score_run(prepare, rec, out, prepare.theta_of(), start, "cpu")
print(json.dumps({"rec": rec, "sealed": prepare.SEALED_KEYS}))
"""


def score(out: Path) -> tuple[dict, list[str]]:
    """The evaluation job, a process of its own (the training job's audit hook isn't in it)."""
    r = subprocess.run([sys.executable, "-c", SCORE, str(FAKE), str(Path(te_run.__file__).parents[1]), str(out)],
                       capture_output=True, text=True, check=True)
    got = json.loads(r.stdout.strip().splitlines()[-1])
    return got["rec"], got["sealed"]


def test_an_island_runs_unmodified_and_its_encodings_are_scored_by_the_evaluation_job(cw_root):
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
    # the training job scores nothing: each checkpoint is its E, written for the evaluation job
    assert all(cp["source"] == "encode" and cp["val_bpb"] is None and "val_bpb" not in rec["start"] for cp in cps)
    assert all((cw_root / "out" / cp["path"]).is_file() for cp in cps)
    assert cps[-1]["submitted"] and rec["submitted"]["e_sha256"] == cps[-1]["e_sha256"]
    assert cps[0]["macs"] < cps[-1]["macs"] == rec["totals"]["macs"]
    assert rec["c"] == 4096 * rec["forward_macs_per_token"]
    rec, sealed = score(cw_root / "out")
    cps = rec["checkpoints"]
    assert rec["scored"]["problems"] == [] and all(math.isfinite(cp["val_bpb"]) for cp in cps)
    assert rec["start"]["val_bpb"] > cps[-1]["val_bpb"]
    # each scored model's value is sealed under its output digest, and its score names that digest
    assert {cp["output_sha256"] for cp in cps} <= set(sealed)
    assert all(cp["score"]["slice"]["weights_digest"] == cp["output_sha256"] for cp in cps)


def test_an_e_file_that_is_not_the_recorded_one_is_not_scored(cw_root):
    rc, rec = run_island(cw_root, "--steps", "4")
    assert rc == 0
    (cw_root / "out" / rec["checkpoints"][0]["path"]).write_bytes(b"CWFULL1\nnot the encoding")
    rec, _ = score(cw_root / "out")
    assert rec["checkpoints"][0]["val_bpb"] is None and len(rec["scored"]["problems"]) == 1
    assert all(cp["val_bpb"] is not None for cp in rec["checkpoints"][1:])


def test_a_sealed_split_is_never_returned(cw_root):
    rc, rec = run_island(cw_root, "--steps", "4", harness=("--tier", "2"))
    assert rc == 0 and rec["invalid"] == []
    assert rec["checkpoints"] and all("test_bpb" not in cp for cp in rec["checkpoints"])


@pytest.mark.parametrize("flag", ["--peek", "--peek-sealed"])
def test_the_held_out_and_sealed_sets_are_unreadable_to_the_island(cw_root, flag):
    (cw_root / "root" / "sealed").mkdir()
    rc, rec = run_island(cw_root, "--steps", "2", flag)
    assert rc == 1 and "PermissionError" in rec["error"]
    assert any("PermissionError" in p for p in rec["invalid"])


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
