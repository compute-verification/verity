import importlib.util
import json

import pytest
import torch

from training_slowdown import run
from training_slowdown.counterexample import check

MODEL = '''
import torch
import torch.nn.functional as F
from torch import nn

V, D = 32, 16

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(V, D) * 0.1)
        self.w1 = nn.Linear(D, D, bias=False)

    def forward(self, idx):
        x = F.embedding(idx, self.emb)
        return F.linear(x + self.w1(F.silu(x)), self.emb)

def loss_of(m, idx):
    return F.cross_entropy(m(idx).reshape(-1, V), idx.reshape(-1))

def batch(step):
    return torch.randint(0, V, (3, 8), generator=torch.Generator().manual_seed(100 + step))
'''

SGD = MODEL + '''
from training_slowdown.codecs import Identity, Plan

def train(te):
    m = te.load(Tiny())
    te.plan(6)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    fam = te.family("full", Plan([("*", Identity())]))
    for t in range(6):
        with te.step():
            fam.put(t, fam.plan.encode(dict(m.named_parameters()), te.theta))
            loss_of(m, batch(t)).backward()
            opt.step()
            opt.zero_grad()
'''

MEZO = MODEL + '''
from training_slowdown.codecs import Plan, SeedReplay, derive_seed

EPS, LR = 1e-3, 1e-2

def f32(x):
    return float(torch.tensor(x, dtype=torch.float32))

def train(te):
    m = te.load(Tiny())
    te.plan(6)
    fam = te.family("mezo", Plan([("*", SeedReplay())]))
    logs = {k: [] for k, _ in m.named_parameters()}
    for t in range(6):
        with te.step(), torch.no_grad():
            seeds = {k: derive_seed(t, k) for k in logs}
            def shift(c):
                for k, p in m.named_parameters():
                    SeedReplay.apply(p, seeds[k], c)
                    logs[k].append((seeds[k], c))
            shift(EPS)
            enc = {f"{k}@0": SeedReplay.encode_log(*zip(*logs[k])) for k in logs}
            l1 = loss_of(m, batch(t)).item()
            shift(-2 * EPS)
            enc |= {f"{k}@1": SeedReplay.encode_log(*zip(*logs[k])) for k in logs}
            l2 = loss_of(m, batch(t)).item()
            shift(EPS)
            shift(f32(-LR * (l1 - l2) / (2 * EPS)))
            fam.put(t, fam.plan.encode_parts(enc))
'''


@pytest.fixture
def theta_file(tmp_path):
    (tmp_path / "tiny_model.py").write_text(MODEL)
    spec = importlib.util.spec_from_file_location("tiny_model", tmp_path / "tiny_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(0)
    path = tmp_path / "theta.pt"
    torch.save({"state": mod.Tiny().state_dict(), "meta": {}}, path)
    return path


def launch(tmp_path, theta_file, source: str, name: str, *extra):
    prog = tmp_path / f"{name}.py"
    prog.write_text(source)
    out = tmp_path / f"out-{name}"
    rc = run.main(["--program", str(prog), "--theta", str(theta_file), "--out", str(out), "--device", "cpu",
                      "--meta", json.dumps({"agent": "t"}), *extra])
    return rc, out, json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else None


def test_sgd_run_records_every_step_and_verifies_an_identity_family(tmp_path, theta_file):
    rc, out, rec = launch(tmp_path, theta_file, SGD, "sgd")
    assert rc == 0, rec["invalid"]
    assert len(rec["steps"]) == 6
    assert [cp["step"] for cp in rec["checkpoints"]] == [0, 1, 2, 3, 4, 5]
    assert rec["c"] == 4096 * rec["forward_macs_per_token"]
    (fam,) = rec["families"]
    assert fam["failed"] == {}
    assert len(fam["sampled"]) == 5
    assert fam["covers"] == [1, 2, 3, 4, 5]  # step 0 reads θ itself: nothing to cover
    # a full update costs at least p bits per updated value (V·D + D·D of them, V, D = 32, 16); per token the updated MACs
    # are the tied table's lookup (58 multiplexers per word), w1 and the head
    floor = 32 * (32 * 16 + 16 * 16) / (58 * 16 + 16 * 16 + 32 * 16)
    assert all(b > floor for b in fam["beta"].values()), fam["beta"]
    assert (out / "checkpoints").exists()


def test_mezo_run_is_rebuilt_from_its_seed_log(tmp_path, theta_file):
    rc, out, rec = launch(tmp_path, theta_file, MEZO, "mezo")
    assert rc == 0, rec["invalid"]
    assert all(len(ds) == 2 for ds in rec["steps"][3]["weights"].values())  # each weight read at θ ± εz
    (fam,) = rec["families"]
    assert fam["failed"] == {}, fam["failed"]
    assert fam["covers"] == [0, 1, 2, 3, 4, 5]
    assert fam["beta"]["0"] < fam["beta"]["5"]  # the log grows with the step
    theta = run.load_theta(theta_file)[0]
    assert check(out / "families" / "mezo", rec, theta, c=10, coin=b"x").failed  # a rebuild over c is refused
    bad = out / "families" / "mezo" / "3.enc"
    bad.write_bytes(bad.read_bytes()[:-1] + bytes([bad.read_bytes()[-1] ^ 1]))
    v = check(out / "families" / "mezo", rec, theta, c=rec["c"], n_sample=6, coin=b"x")
    assert list(v.failed) == [3]
    assert v.covers == [0, 1, 2, 4, 5]  # a uniform family with a failed step covers only what it verified


TOKENS_OUTSIDE = MODEL + '''
def train(te):
    m = te.load(Tiny())
    te.plan(1)
    m(batch(0))
    with te.step():
        pass
'''


def test_tokens_outside_a_step_make_the_run_invalid(tmp_path, theta_file):
    rc, _, rec = launch(tmp_path, theta_file, TOKENS_OUTSIDE, "outside")
    assert rc == 1
    assert any("outside a step" in p for p in rec["invalid"])


def test_a_program_reaching_into_the_harness_is_refused(tmp_path, theta_file):
    with pytest.raises(SystemExit, match="refused"):
        launch(tmp_path, theta_file, MODEL + "\ndef train(te):\n    te.counter.pause()\n", "sneaky")
