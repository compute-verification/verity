import json

import pytest
import torch
import torch.nn.functional as F
from synthetic import tiny_data, tiny_registry, tiny_start, write_config
from torch import nn

from training_slowdown import config, run, starts
from training_slowdown.codecs import Identity, Plan, QuantDelta, SeedReplay, derive_seed
from training_slowdown.counterexample import check
from training_slowdown.data import Stream
from training_slowdown.evaluate import Evaluator
from training_slowdown.model import LM

V, D = 32, 16


def test_c_is_4096_forward_passes():
    assert run.block_c(10) == 10 * run.C_PER_FORWARD_TOKEN == 40_960


class Small(nn.Module):
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


@pytest.fixture
def theta():
    torch.manual_seed(0)
    return {k: v.detach().clone() for k, v in Small().state_dict().items()}


def counted(tmp_path, theta, train):
    """`train(te)` run inside a harness of θ, and the run's record."""
    te = run.Harness(theta, tmp_path / "out", clock=lambda: 0.0)
    with te.counter:
        train(te)
    return te.finish({})


def sgd(te):
    m = te.load(Small())
    te.plan(6)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    fam = te.family("full", Plan([("*", Identity())]))
    for t in range(6):
        with te.step():
            fam.put(t, fam.plan.encode(dict(m.named_parameters()), te.theta))
            loss_of(m, batch(t)).backward()
            opt.step()
            opt.zero_grad()


def test_sgd_run_records_every_step_and_verifies_an_identity_family(tmp_path, theta):
    rec = counted(tmp_path, theta, sgd)
    assert rec["invalid"] == []
    assert len(rec["steps"]) == 6
    assert [cp["step"] for cp in rec["checkpoints"]] == [0, 1, 2, 3, 4, 5]
    assert rec["c"] == 4096 * rec["forward_macs_per_token"]
    (fam,) = rec["families"]
    assert fam["failed"] == {}
    assert len(fam["sampled"]) == 5
    assert fam["covers"] == [1, 2, 3, 4, 5]  # step 0 reads θ itself: nothing to cover
    # a full update costs at least p bits per updated value (V·D + D·D of them); per token the updated MACs are the
    # tied table's lookup (58 multiplexers per word), w1 and the head
    floor = 32 * (V * D + D * D) / (58 * D + D * D + V * D)
    assert all(b > floor for b in fam["beta"].values()), fam["beta"]
    assert (tmp_path / "out" / "checkpoints").exists()


EPS, LR = 1e-3, 1e-2


def mezo(te):
    m = te.load(Small())
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
            shift(float(torch.tensor(-LR * (l1 - l2) / (2 * EPS), dtype=torch.float32)))
            fam.put(t, fam.plan.encode_parts(enc))


def test_mezo_run_is_rebuilt_from_its_seed_log(tmp_path, theta):
    rec = counted(tmp_path, theta, mezo)
    assert rec["invalid"] == []
    assert all(len(ds) == 2 for ds in rec["steps"][3]["weights"].values())  # each weight read at θ ± εz
    (fam,) = rec["families"]
    assert fam["failed"] == {}, fam["failed"]
    assert fam["covers"] == [0, 1, 2, 3, 4, 5]
    assert fam["beta"]["0"] < fam["beta"]["5"]  # the log grows with the step
    families = tmp_path / "out" / "families" / "mezo"
    assert check(families, rec, theta, c=10, coin=b"x").failed  # a rebuild over c is refused
    bad = families / "3.enc"
    bad.write_bytes(bad.read_bytes()[:-1] + bytes([bad.read_bytes()[-1] ^ 1]))
    v = check(families, rec, theta, c=rec["c"], n_sample=6, coin=b"x")
    assert list(v.failed) == [3]
    assert v.covers == [0, 1, 2, 4, 5]  # a uniform family with a failed step covers only what it verified


def test_tokens_outside_a_step_make_the_run_invalid(tmp_path, theta):
    def outside(te):
        m = te.load(Small())
        te.plan(1)
        m(batch(0))
        with te.step():
            pass
    rec = counted(tmp_path, theta, outside)
    assert any("outside a step" in p for p in rec["invalid"])


def one_step(plan):
    def train(te):
        m = te.load(Small())
        te.encode_with(plan)
        te.plan(1)
        opt = torch.optim.SGD([m.w1.weight], lr=0.1)
        with te.step():
            loss_of(m, batch(0)).backward()
            opt.step()
    return train


def test_each_checkpoint_carries_its_e_and_one_that_doesnt_rebuild_is_invalid(tmp_path, theta):
    (cp,) = counted(tmp_path / "full", theta, one_step(Plan([("w1.*", Identity())])))["checkpoints"]
    assert (cp["format"], cp["changed"], cp["rebuild_macs"]) == ("full", 1, 0)
    assert cp["e_bytes"] >= 4 * D * D and len(cp["e_sha256"]) == 64
    rec = counted(tmp_path / "lossy", theta, one_step(Plan([("w1.*", QuantDelta(bits=1))])))
    assert rec["checkpoints"][0]["format"] == "codecs"
    assert any("doesn't rebuild its output's ['w1.weight']" in p for p in rec["invalid"])
    rec = counted(tmp_path / "none", theta, one_step(Plan([("emb", Identity())])))
    assert rec["checkpoints"][0]["e_bytes"] is None
    assert any("has no E of its output" in p for p in rec["invalid"])


def test_a_route_reaching_past_the_codecs_and_the_model_is_refused(tmp_path, monkeypatch):
    assert run.screen("from training_slowdown.codecs import Plan\nfrom training_slowdown import model\n") == []
    assert run.screen("from training_slowdown import flops\n")
    assert run.screen("import training_slowdown.run\n")
    (tmp_path / "sneaky.py").write_text("def build(shape, dials):\n    model.counter.pause()\n")
    monkeypatch.setattr(config, "ROUTES", tmp_path)
    with pytest.raises(SystemExit, match="sneaky.py: refused, it names"):
        config.route("sneaky")


def test_the_route_loop_trains_scores_and_encodes_from_the_config(tmp_path):
    reg = tiny_registry(tmp_path / "reg")
    path = write_config(tmp_path / "a.toml")
    cfg = config.load(path, starts.registry(reg))
    out = tmp_path / "out"
    rc = run.main(["--config", str(path), "--seed", "0", "--split", "search", "--out", str(out), "--starts", str(reg),
                   "--device", "cpu"])
    rec = json.loads((out / "run.json").read_text())
    assert rc == 0, rec["invalid"]
    assert rec["label"] == f"training-slowdown={cfg.digest}/0/search"
    assert rec["config"] == json.loads(json.dumps(cfg.record())) and rec["route"]["name"] == "subset"
    assert len(rec["steps"]) == 3 and [cp["step"] for cp in rec["checkpoints"]] == [0, 1, 2]
    for cp in rec["checkpoints"]:
        assert cp["score"]["split"] == "search" and cp["bpb"] == cp["score"]["bpb"]
        assert (cp["format"], cp["changed"]) == ("full", 3) and cp["e_bytes"] > 0


def test_each_checkpoint_of_a_registered_start_is_scored_on_the_jobs_split(tmp_path):
    data = tiny_data(tmp_path / "data")
    s, _ = tiny_start(tmp_path / "starts")
    theta = starts.load(s, tmp_path / "starts")
    te = run.Harness(theta, tmp_path / "out", start=s.entry(), split="search", clock=lambda: 0.0,
                     evaluator=Evaluator(data, "search", theta, s.shape, "cpu"))
    stream = Stream(data, seed=0, ctx=16)
    with te.counter:
        m = te.load(LM.of(s.shape))
        te.plan(2)
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        for _ in range(2):
            with te.step():
                rows = te.batch(stream.next(4))
                logits = m(rows[:, :-1])
                F.cross_entropy(logits.reshape(-1, logits.shape[-1]), rows[:, 1:].reshape(-1)).backward()
                opt.step()
                opt.zero_grad()
    rec = te.finish({})
    assert rec["split"] == "search" and rec["start"]["name"] == s.name and rec["start"]["shape"] == s.shape
    assert [cp["step"] for cp in rec["checkpoints"]] == [0, 1]
    for cp in rec["checkpoints"]:
        sc = cp["score"]
        assert sc["split"] == "search" and cp["bpb"] == sc["bpb"]
        assert sc["slice"]["weights_digest"] == cp["output_sha256"]
    assert rec["invalid"] == []
