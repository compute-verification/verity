import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from synthetic import run_job, tiny_data, tiny_registry, tiny_start, write_config
from torch import nn

from training_slowdown import VERSION, config, formalism, policy, run, starts
from training_slowdown.capture import products
from training_slowdown.codecs import Identity, Plan, QuantDelta, SeedReplay, SparseTernary, derive_seed
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


def test_each_checkpoint_carries_its_e_and_its_output_is_what_e_rebuilds(tmp_path, theta):
    """Under a plan that keeps the trained weights whole the output model is the trained state, as it was hashed and
    scored before any plan could keep less. Under one that keeps less, the output is the model E rebuilds from θ:
    its digest, the state the evaluator gets and the kept weights are that model's, and `trained_sha256` is the
    trained state's. A plan with no E of the output makes the run invalid."""
    scored = {}

    def run_one(name, plan):
        def evaluator(state, sha):
            scored[name] = (sha, {k: v.clone() for k, v in state.items()})
            return {"bpb": 1.0}
        te = run.Harness(theta, tmp_path / name, clock=lambda: 0.0, evaluator=evaluator, keep_weights=True)
        with te.counter:
            one_step(plan)(te)
        return te.finish({})

    (cp,) = run_one("full", Plan([("w1.*", Identity())]))["checkpoints"]
    assert (cp["format"], cp["changed"], cp["rebuild_macs"]) == ("full", 1, 0)
    assert cp["e_bytes"] >= 4 * D * D and len(cp["e_sha256"]) == 64 and "trained_sha256" not in cp
    sha, trained = scored["full"]
    assert sha == cp["output_sha256"] == run.theta_sha256(trained) != run.theta_sha256(theta)
    assert run.theta_sha256(torch.load(tmp_path / "full" / cp["path"], weights_only=True)) == sha

    rec = run_one("lossy", Plan([("w1.*", QuantDelta(bits=1))]))
    (lossy,) = rec["checkpoints"]
    assert rec["invalid"] == [] and lossy["format"] == "codecs" and lossy["trained_sha256"] == cp["output_sha256"]
    sha, output = scored["lossy"]
    enc = (tmp_path / "lossy" / lossy["e_path"]).read_bytes()
    rebuilt = theta | Plan([("w1.*", QuantDelta(bits=1))]).decode(enc, {"w1.weight": theta["w1.weight"]})
    assert sha == lossy["output_sha256"] == run.theta_sha256(rebuilt) == run.theta_sha256(output) != cp["output_sha256"]
    assert not torch.equal(output["w1.weight"], trained["w1.weight"]) and torch.equal(output["emb"], trained["emb"])
    assert run.theta_sha256(torch.load(tmp_path / "lossy" / lossy["path"], weights_only=True)) == sha

    rec = counted(tmp_path / "none", theta, one_step(Plan([("emb", Identity())])))
    assert rec["checkpoints"][0]["e_bytes"] is None
    assert any("has no E of its output" in p for p in rec["invalid"])


def test_a_weight_handed_over_as_its_trained_parts_is_what_the_plans_codec_rebuilds_from_them(tmp_path, theta):
    """An output may give a weight of θ's as the parts the run trained of it. E is the plan's codec's packing of
    those parts, and the output model's weight is what that codec decodes from E on θ: here codes the run's own
    weight was never made from, so the model scored is E's, not the one the run holds. The checkpoint says how many
    weights came as parts and has no digest of a trained state. Parts with no E leave no model to score, and the run
    is invalid: parts of a weight θ lacks, parts that are not of the weight's size, and parts with no plan."""
    codes = SparseTernary.round(torch.randn(D, D, generator=torch.Generator().manual_seed(3)), 0.25)
    scored, ternary = [], Plan([("w1.*", SparseTernary(0.25))])

    def handing(parts, plan=ternary):
        def train(te):
            m = te.load(Small())
            if plan:
                te.encode_with(plan)
            te.output(lambda: m.state_dict() | parts)
            te.plan(1)
            with te.step():
                loss_of(m, batch(0)).backward()
        return train

    def run_one(name, *a):
        te = run.Harness(theta, tmp_path / name, clock=lambda: 0.0, keep_weights=True,
                         evaluator=lambda state, sha: scored.append((sha, dict(state))) or {"bpb": 1.0})
        with te.counter:
            handing(*a)(te)
        return te.finish({})

    rec = run_one("parts", {"w1.weight": codes})
    (cp,) = rec["checkpoints"]
    assert rec["invalid"] == [] and "trained_sha256" not in cp
    assert (cp["format"], cp["changed"], cp["parts"], cp["rebuild_macs"]) == ("codecs", 1, 1, 2 * D * D)
    (sha, output), = scored
    want = SparseTernary.apply(theta["w1.weight"], *codes)
    assert torch.equal(output["w1.weight"], want) and not torch.equal(want, theta["w1.weight"])
    assert torch.equal(output["emb"], theta["emb"]) and set(output) == set(theta)
    assert sha == cp["output_sha256"] == run.theta_sha256(theta | {"w1.weight": want})
    enc = (tmp_path / "parts" / cp["e_path"]).read_bytes()
    assert len(enc) == cp["e_bytes"] and hashlib.sha256(enc).hexdigest() == cp["e_sha256"]
    assert torch.equal(ternary.decode(enc, theta)["w1.weight"], want)
    assert run.theta_sha256(torch.load(tmp_path / "parts" / cp["path"], weights_only=True)) == sha

    small = SparseTernary.round(torch.ones(D), 0.25)
    for name, a, why in (("lacks", ({"w2.weight": codes},), "which θ lacks"),
                         ("size", ({"w1.weight": small},), f"for a weight of {D * D}"),
                         ("whole", ({"w1.weight": codes}, Plan([("w1.*", Identity())])), "packs no trained parts"),
                         ("planless", ({"w1.weight": codes}, None), "no plan to pack them")):
        rec = run_one(name, *a)
        (cp,) = rec["checkpoints"]
        assert any(why in p for p in rec["invalid"]), (name, rec["invalid"])
        assert cp["output_sha256"] is None and cp["bpb"] is None and "path" not in cp and "e_path" not in cp
    assert len(scored) == 1


def test_a_route_reaching_past_the_codecs_and_the_model_is_refused(tmp_path, monkeypatch):
    assert run.screen("from training_slowdown.codecs import Plan\nfrom training_slowdown import model\n") == []
    assert run.screen("from training_slowdown import flops\n")
    assert run.screen("import training_slowdown.run\n")
    (tmp_path / "sneaky.py").write_text("def build(shape, dials):\n    model.counter.pause()\n")
    monkeypatch.setattr(config, "ROUTES", tmp_path)
    with pytest.raises(SystemExit, match="sneaky.py: refused, it names"):
        config.route("sneaky")


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """One search job of the tiny start's config, as a node runs it."""
    return run_job(tmp_path_factory.mktemp("job"))


def test_the_route_loop_trains_scores_and_encodes_from_the_config(ran):
    rec, cfg, reg, _ = ran
    final = starts.registry(reg)["data"]["tiny"]["final"]
    assert rec["invalid"] == [] and rec["sandbox"] == {"interfaces": {"lo": False}, "final_sets": {final: "empty"}}
    assert rec["label"] == f"training-slowdown={cfg.digest}/0/search"
    assert rec["config"] == json.loads(json.dumps(cfg.record())) and rec["route"]["name"] == "subset"
    assert len(rec["steps"]) == 3 and [cp["step"] for cp in rec["checkpoints"]] == [0, 1, 2]
    for cp in rec["checkpoints"]:
        assert cp["score"]["split"] == "search" and cp["bpb"] == cp["score"]["bpb"]
        assert cp["score"]["slice"]["weights_digest"] == cp["output_sha256"]
        assert (cp["format"], cp["changed"]) == ("full", 3) and cp["e_bytes"] > 0


def test_a_record_whose_checkpoints_arent_the_states_scored_is_invalid():
    def unscored():
        return {"invalid": [], "checkpoints": [{"step": t, "bpb": None, "score": None, "output_sha256": sha}
                                               for t, sha in enumerate("ab")]}
    rec = run.with_scores(unscored(), [("a", {"bpb": 1.5}), ("b", {"bpb": 1.25})])
    assert rec["invalid"] == [] and [(cp["bpb"], cp["score"]) for cp in rec["checkpoints"]] == [
        (1.5, {"bpb": 1.5}), (1.25, {"bpb": 1.25})]
    for scores in ([("a", {"bpb": 1.5})], [("b", {"bpb": 1.25}), ("a", {"bpb": 1.5})]):
        rec = run.with_scores(unscored(), scores)
        assert all(cp["score"] is None for cp in rec["checkpoints"])
        assert rec["invalid"] == [f"the {len(scores)} states the evaluator scored aren't the record's 2 checkpoints' "
                                  f"outputs, one for one"]


def test_a_job_keeps_no_weights_but_its_last_e(ran, tmp_path, theta, monkeypatch):
    """By default a job's output is its record and its last checkpoint's E, which rebuilds that checkpoint's output
    from θ bit for bit; an E at or over the size a job keeps stays out too."""
    rec, cfg, reg, out = ran
    *before, last = rec["checkpoints"]
    assert sorted(p.name for p in out.iterdir()) == ["e", "run.json"] and [p.name for p in (out / "e").iterdir()] == [
        "step0000002.bin"] == [Path(last["e_path"]).name]
    assert not any("path" in cp or "e_path" in cp for cp in before) and "path" not in last
    enc = (out / last["e_path"]).read_bytes()
    assert (len(enc), hashlib.sha256(enc).hexdigest()) == (last["e_bytes"], last["e_sha256"])
    start = starts.load(cfg.start, reg.parent / "starts")
    plan = config.route(cfg.route).plan(cfg.dials)
    rebuilt = start | plan.decode(enc, {k: start[k] for k in plan.keys(enc)})
    assert run.theta_sha256(rebuilt) == last["output_sha256"] != run.theta_sha256(start)

    full = Plan([("w1.*", Identity())])
    (cp,) = counted(tmp_path / "small", theta, one_step(full))["checkpoints"]
    assert (tmp_path / "small" / "out" / cp["e_path"]).stat().st_size == cp["e_bytes"]
    monkeypatch.setattr(run, "E_KEEP_BYTES", cp["e_bytes"])
    (cp,) = counted(tmp_path / "large", theta, one_step(full))["checkpoints"]
    assert "e_path" not in cp and [p.name for p in (tmp_path / "large" / "out").iterdir()] == ["run.json"]


@pytest.mark.slow
def test_a_job_keeps_its_last_checkpoints_whole_state_only_when_asked(tmp_path):
    """`--keep-weights` adds the last checkpoint's whole state, and no other's, to what a job keeps. It is a job of
    its own, so slow; the command's test shows the flag reaches the job, not what the job then writes."""
    rec, _, _, out = run_job(tmp_path / "kept", "--keep-weights")
    last = rec["checkpoints"][-1]
    state = torch.load(out / last["path"], weights_only=True)
    assert last["path"] == "checkpoints/step0000002.pt" and run.theta_sha256(state) == last["output_sha256"]
    assert not any("path" in cp for cp in rec["checkpoints"][:-1])


RAW = {"count", "phase", "op", "dims", "token_dims", "dtypes", "affected", "widths", "self", "gathered", "rates"}


def test_the_job_traces_its_routes_step(ran):
    """The record's `trace` is the capture's measurement of the middle one of the route's last three steps at the
    job's batch and the start's context, the tokens the counter counted a step, with the version of the code that
    took it, the rows unpriced and nothing the formalism derives; the rows price where they are read."""
    rec, cfg = ran[0], ran[1]
    tr = rec["trace"]
    assert set(tr) == {"capture", "capture_code", "torch", "step", "axes", "tokens", "order", "updated", "rows", "k",
                       "p_match", "at", "elementwise_products", "data_dependent", "carried_scalars", "circuit"}
    assert (tr["capture"], tr["capture_code"], tr["torch"]) == (products.CAPTURE, VERSION, torch.__version__)
    assert tr["step"] == 1 and len(rec["steps"]) == run.TRACED_STEPS
    assert tr["axes"] == {"batch": 2, "seq": 1024} and {s["tokens"] for s in rec["steps"]} == {tr["tokens"]} == {2048}
    assert tr["data_dependent"] == tr["carried_scalars"] == [] and tr["k"] == policy.CONSOLE[1]
    assert sorted(map(int, tr["at"])) == sorted(set(products.DIAL_K) - {int(tr["k"])})
    for block in (tr, *tr["at"].values()):
        assert block["rows"] and all(set(r) == RAW for r in block["rows"])
    # the three trained matrices of block 0's MLP: each one's forward product reads it, and their MACs a token add up
    trained = [u for u in tr["updated"] if u[2:] == [u[2], tr["tokens"], 32]]
    assert len(tr["updated"]) == len(trained) >= 3
    priced = [products.priced(r) for r in tr["rows"]]
    assert sum(r["count"] * r["macs"] for r in priced if r["phase"] == "forward" and all(r["affected"])) > 0
    assert all(r["gates_per_mac"] > 0 and "dtypes" not in r for r in priced)
    # the step's circuit, raw: the gradient's cut and a token group's partial are the trained matrices' dW at the
    # accumulator's 32 bits, the carried state no token op reads AdamW's two moments, and a window's tokens are read
    # at its inputs and its last target
    weights = sum(p.numel() for p in config.route(cfg.route).build(cfg.start.shape, cfg.dials)[0].parameters()
                  if p.requires_grad)
    c = tr["circuit"]
    assert {*formalism.TRACED_STRUCTURE, "tau_ids", "tau_other_bits", "a_bits", "u_in", "u1_bits", "tail_share"} < set(c)
    assert not {"parts", "labels", "u", "trace", *formalism.READ_BACKS} & set(c)
    assert c["grad_cut_bits"] == c["partial_cut_bits"] == c["u1_bits"] == 32 * weights > 0
    assert c["state_bits"] == 2 * 32 * weights and c["delta_grad_bits"] == 3 * 32 * weights
    assert (c["region"], c["tokens"], c["token_factor"], c["tau_ids"]) == (1, tr["tokens"], 1, 1025 / 1024)
    assert 0 < c["affected_share"] < c["sigma"] < 1


@pytest.mark.parametrize("steps, traced", [(1, 0), (2, 1), (5, 3)])
def test_the_traced_step_has_a_step_before_it_where_the_run_has_one(tmp_path, steps, traced):
    """The traced step is the middle one of the run's last three, the second of a run of two, and the only step of a
    run of one, which reads θ alone: its circuit shows no update."""
    cfg = config.load(write_config(tmp_path / "a.toml", budget={"steps": steps, "batch": 2}),
                      starts.registry(tiny_registry(tmp_path / "reg")))
    route = config.route(cfg.route)
    tr = run.trace_step(cfg, route, *route.build(cfg.start.shape, cfg.dials), "cpu")
    assert tr["step"] == traced
    assert (tr["circuit"]["affected_share"] > 0) == (tr["circuit"]["u1_bits"] > 0) == (steps > 1)


def test_what_a_routes_output_computes_is_traced_and_charged_with_its_step(tmp_path):
    """A route's `output` is no free place to compute: a product it takes of two trained weights is a row of the
    trace, a forward product of the measured step with both operands the update's, and the step's circuit holds its
    gates. Untraced, a route's `output` would be work no step is charged for."""
    cfg = config.load(write_config(tmp_path / "a.toml", budget={"steps": 5, "batch": 2}),
                      starts.registry(tiny_registry(tmp_path / "reg")))
    route = config.route(cfg.route)

    def output(model, dials):
        b = model.blocks[0]
        return model.state_dict() | {"blocks.0.wo.weight": b.w_gate.weight.T @ b.w_up.weight}

    def trace(r):
        tr = run.trace_step(cfg, r, *route.build(cfg.start.shape, cfg.dials), "cpu")
        return tr, [x for x in tr["rows"] if x["phase"] == "forward" and not x["token_dims"]]

    (bare, no_rows), (traced, rows) = trace(route), trace(SimpleNamespace(step=route.step, output=output))
    d, ff = cfg.start.shape["d_model"], cfg.start.shape["d_ff"]
    assert not no_rows and len(rows) == 1 and len(traced["rows"]) == len(bare["rows"]) + 1
    assert (rows[0]["op"], rows[0]["count"], rows[0]["affected"]) == ("mm", 1, [True, True])
    assert [rows[0]["dims"][x] for x in "mkn"] == [d, ff, d]
    assert traced["circuit"]["gates"] > bare["circuit"]["gates"]


@pytest.mark.parametrize("steps, width", [(3, 16), (5, 32)])
def test_a_weight_is_passed_at_the_updates_width_only_among_the_runs_first_steps(tmp_path, steps, width):
    """Muon casts its update to 16 bits on its way into a 32-bit weight. A run of three steps is traced at its
    second, whose weight is θ plus that update: a block makes it from the update alone, so the operand is passed at
    16 bits. A longer run is traced mid-run, where the weight the update is added to is the run's own: the operand is
    passed at its 32 bits."""
    cfg = config.load(write_config(tmp_path / "a.toml", budget={"steps": steps, "batch": 2},
                                   dials={"trainable": "blocks.0.w_*", "lr": 1e-2, "opt": "muon"}),
                      starts.registry(tiny_registry(tmp_path / "reg")))
    route = config.route(cfg.route)
    tr = run.trace_step(cfg, route, *route.build(cfg.start.shape, cfg.dials), "cpu")
    read = [r["widths"] for r in tr["rows"] if r["phase"] == "forward" and r["affected"][1]]   # each reads a weight
    assert read and all(w == [32, width] for w in read)


def test_a_retrace_writes_the_same_record_and_refuses_another_digest(ran, tmp_path):
    """`--retrace` takes a record's trace again and changes nothing else, and refuses a record whose config this tree
    reads as another digest. Here `retrace` runs in the test's process; the job's own path to it is the slow test."""
    rec, _, reg, _ = ran
    stale = tmp_path / "stale.json"

    def retrace(out):
        return run.retrace(run.parse(["--retrace", str(stale), "--out", str(tmp_path / out), "--starts", str(reg),
                                      "--device", "cpu", "--inside", str(tmp_path)]))

    stale.write_text(json.dumps(rec | {"trace": {"capture": "0"}}))
    assert retrace("again") == 0 and [p.name for p in (tmp_path / "again").iterdir()] == ["run.json"]
    assert json.loads((tmp_path / "again" / "run.json").read_text()) == rec
    stale.write_text(json.dumps(rec | {"config": rec["config"] | {"digest": "0" * 64}}))
    with pytest.raises(SystemExit, match="retrace in a clone at the record's commit"):
        retrace("other")
    assert not (tmp_path / "other" / "run.json").exists()


@pytest.mark.slow
def test_the_job_retraces_where_its_route_runs(ran, tmp_path):
    """Through the job's command a retrace runs in the process a route runs in, started through the sandbox's entry,
    and writes the record alone. It is a process of its own, so slow."""
    rec, _, reg, _ = ran
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(rec | {"trace": {"capture": "0"}}))
    assert run.main(["--retrace", stale, "--out", tmp_path / "again", "--starts", reg, "--device", "cpu"]) == 0
    assert [p.name for p in (tmp_path / "again").iterdir()] == ["run.json"]
    assert json.loads((tmp_path / "again" / "run.json").read_text()) == rec


def test_a_step_the_capture_refuses_is_recorded_as_refused(tmp_path):
    """A step that reads a tensor's value into Python is no fixed graph: the trace says so, with the capture's reason
    and op, and holds nothing else."""
    cfg = config.load(write_config(tmp_path / "a.toml"), starts.registry(tiny_registry(tmp_path / "reg")))
    route = config.route(cfg.route)
    model, optimizers = route.build(cfg.start.shape, cfg.dials)

    class Branching:
        @staticmethod
        def step(model, rows, optimizers, dials, t, n):
            if float(model(rows[:, :-1]).sum()) > 0:
                route.step(model, rows, optimizers, dials, t, n)

    tr = run.trace_step(cfg, Branching, model, optimizers, "cpu")
    assert set(tr) == {"refused"} and set(tr["refused"]) == {"reason", "op", "message"}
    assert tr["refused"]["reason"] == "dynamic_control_flow" and "reads a tensor's value" in tr["refused"]["message"]


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
