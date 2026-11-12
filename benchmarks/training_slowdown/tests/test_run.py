import contextlib
import copy
import hashlib
import json
import math
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from synthetic import JOBS, LORA, job_counted, run_job, tiny_data, tiny_registry, tiny_start, write_config
from torch import nn

from training_slowdown import VERSION, config, flops, formalism, gates, grade, policy, run, starts, work
from training_slowdown.capture import products
from training_slowdown.codecs import Identity, Plan, QuantDelta, SeedReplay, SparseTernary, derive_seed
from training_slowdown.counterexample import check
from training_slowdown.data import Stream
from training_slowdown.evaluate import Evaluator
from training_slowdown.flops import gather_macs
from training_slowdown.model import LM
from training_slowdown.opnames import gemm_like
from training_slowdown.traced import TRACED_STEPS, trace_step, traced_step

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


def test_a_record_states_torchs_tf32_switch_as_it_stands_when_it_is_written(tmp_path, theta):
    """A record states both of torch's readings of its switch for TF32 matrix products, read when the record is
    written: off as torch starts, and on by whichever reading a job's code set. The record of a job that set it
    names its float32 products float32, as torch does, which `gates.exact_under_tf32` lists and the grade refuses."""
    allow, precision = torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()

    def switched(te):
        torch.backends.cuda.matmul.allow_tf32 = True
        sgd(te)

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        off = counted(tmp_path / "off", theta, sgd)
        assert off["tf32"] == {"cuda_matmul_allow_tf32": False, "float32_matmul_precision": "highest"} == run.tf32()
        assert gates.tf32_on(off) is False and gates.exact_under_tf32(off) == []
        on = counted(tmp_path / "on", theta, switched)
        assert on["tf32"]["cuda_matmul_allow_tf32"] is True and gates.tf32_on(on) is True
        assert gates.exact_under_tf32(on) == ["bwd mm", "fwd mm"] and on["invalid"] == []
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("medium")
        assert run.tf32()["float32_matmul_precision"] == "medium" and gates.tf32_on({"tf32": run.tf32()}) is True
        assert gates.tf32_on({}) is None and gates.exact_under_tf32(off | {"tf32": None}) == []
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = allow


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


def test_an_output_that_computes_on_tokens_with_an_updated_weight_makes_the_unbroken_job_invalid(tmp_path, theta):
    """A route's `output` at a job's last checkpoint is counted after the last step, in the tail. So an output that
    takes a forward product of token data with an updated weight puts updated MACs outside a step in every job of
    its route, stopped or not: a job continued after a stop, which holds its stopped checkpoint's output outside its
    steps too (`run.joined`), is not invalid for that where the unbroken job is valid."""
    def train(te):
        m = te.load(Small())
        te.plan(2)
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        te.output(lambda: m.state_dict() | {"w1.weight": m.w1.weight + 0 * m(batch(0)).sum()})
        for t in range(2):
            with te.step():
                loss_of(m, batch(t)).backward()
                opt.step()
                opt.zero_grad()
    rec = counted(tmp_path, theta, train)
    assert len(rec["steps"]) == 2 and rec["tail"]["by"]["fwd/updated"] > 0
    assert rec["invalid"] == ["tokens or updated MACs outside a step (tail)"]


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
    weights came as parts, and its `trained_sha256` is the digest of the model the run trains, which took no step
    here and is θ. Parts with no E leave no model to score, and the run is invalid: parts of a weight θ lacks, parts
    that are not of the weight's size, and parts with no plan."""
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
    assert rec["invalid"] == [] and cp["trained_sha256"] == run.theta_sha256(theta) != cp["output_sha256"]
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
    # a config that doesn't ask for fresh windows is served every window, and its record says nothing of them
    assert "windows" not in rec and "windows" not in rec["config"]["run"]
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


RAW = {"count", "phase", "op", "dims", "token_dims", "dtypes", "attention", "affected", "widths", "self", "gathered",
       "rates"}


def test_the_job_traces_its_routes_step(ran):
    """The record's `trace` is the capture's measurement of the middle one of the route's last three steps at the
    job's batch and the start's context, the tokens the counter counted a step, with the version of the code that
    took it, the rows unpriced and nothing the formalism derives; the rows price where they are read."""
    rec, cfg = ran[0], ran[1]
    tr = rec["trace"]
    assert set(tr) == {"capture", "capture_code", "torch", "step", "axes", "tokens", "order", "updated", "rows", "k",
                       "p_match", "at", "elementwise_products", "data_dependent", "carried_scalars", "circuit"}
    assert (tr["capture"], tr["capture_code"], tr["torch"]) == (products.CAPTURE, VERSION, torch.__version__)
    assert tr["step"] == 1 and len(rec["steps"]) == TRACED_STEPS
    assert tr["axes"] == {"batch": 2, "seq": 1024} and {s["tokens"] for s in rec["steps"]} == {tr["tokens"]} == {2048}
    assert tr["data_dependent"] == tr["carried_scalars"] == [] and tr["k"] == policy.CONSOLE[1]
    assert sorted(map(int, tr["at"])) == sorted(set(products.DIAL_K) - {int(tr["k"])})
    for block in (tr, *tr["at"].values()):
        assert block["rows"] and all(set(r) == RAW for r in block["rows"])
    # the forward's ops that read the update, each over the step's tokens: block 0's MLP reads its three trained
    # matrices in products at the 16 bits the route's autocast runs them in, on this CPU as on a GPU, and the norms
    # above it, which autocast runs in fp32, read what it wrote at 32. Attention's products (`bmm`) are traced in
    # fp32, as torch's math path computes them from upcast operands, and every other product in bf16; the rows state
    # the bf16 the calls were made in, which is their price
    assert all(u[2:4] == [u[2], tr["tokens"]] for u in tr["updated"])
    assert sorted({u[4] for u in tr["updated"]}) == [16, 32] and len([u for u in tr["updated"] if u[4] == 16]) >= 3
    assert {(r["op"] == "bmm", tuple(r["dtypes"]), r["attention"]) for r in tr["rows"]} == {
        (False, ("bfloat16", "bfloat16"), None), (True, ("float32", "float32"), "bfloat16")}
    priced = [products.priced(r, tr["capture"]) for r in tr["rows"]]
    assert len({r["gates_per_mac"] for r in priced if r["phase"] != "optimizer"}) == 1
    assert sum(r["count"] * r["macs"] for r in priced if r["phase"] == "forward" and all(r["affected"])) > 0
    assert all(r["gates_per_mac"] > 0 and "dtypes" not in r for r in priced)
    # the step's circuit, raw: the gradient's cut and a token group's partial are the trained matrices' dW at the
    # accumulator's 32 bits, the update the forward reads those matrices at the 16 bits autocast casts them to, the
    # carried state no token op reads AdamW's two moments, and a window's tokens are read at its inputs and its last
    # target
    weights = sum(p.numel() for p in config.route(cfg.route).build(cfg.start.shape, cfg.dials)[0].parameters()
                  if p.requires_grad)
    c = tr["circuit"]
    assert {*formalism.TRACED_STRUCTURE, "tau_ids", "tau_other_bits", "a_bits", "u_in", "u1_bits", "tail_share"} < set(c)
    assert not {"parts", "labels", "u", "trace", *formalism.READ_BACKS} & set(c)
    assert c["grad_cut_bits"] == c["partial_cut_bits"] == 32 * weights > 0 and c["u1_bits"] == 16 * weights
    assert c["state_bits"] == 2 * 32 * weights and c["delta_grad_bits"] == 3 * 32 * weights
    assert (c["region"], c["tokens"], c["token_factor"], c["tau_ids"]) == (1, tr["tokens"], 1, 1025 / 1024)
    assert 0 < c["affected_share"] < c["sigma"] < 1


def test_the_counters_split_of_a_step_on_a_model_with_position_tables_is_the_traces(ran):
    """The start's model rotates its queries and keys by position tables, buffers outside its state dict. They are
    θ's to the counter as they are constants to the capture. Of the traced step's forward products of a token row
    with a weight, the counter calls updated the ones whose weight the trace says an update reaches, and θ's the
    rest: the same products by op and dimensions, as many, at the same MACs. A rotation by a table is θ's product, so
    the counter's updated MACs are those products' and nothing else's, the first step, which reads θ alone, has none,
    and no step lists a table as a weight the route made."""
    rec, cfg = ran[0], ran[1]
    tr, shape = rec["trace"], cfg.start.shape
    step = rec["steps"][tr["step"]]
    log = [row for row in rec["structures"][step["struct"]]["log"] if row[0] == "fwd"]
    counted, traced = ({"theta": {}, "updated": {}} for _ in range(2))

    def add(side: dict, kind: str, key: tuple, count: int, macs: int) -> None:
        was = side[kind].get(key, (0, 0))
        side[kind][key] = (was[0] + count, was[1] + macs)

    for _, kind, op, shapes, count, macs, *_ in log:
        if kind in counted and op in ("mm", "addmm"):
            (m, k), (_, n) = json.loads(shapes)[-2:]
            add(counted, kind, (op, m, k, n), count, macs)
    for r in tr["rows"]:
        if r["phase"] == "forward" and r["token_dims"] == ["m"]:
            d = r["dims"]
            add(traced, "updated" if r["affected"][1] else "theta", (r["op"], d["m"], d["k"], d["n"]), r["count"],
                r["count"] * products.priced(r, tr["capture"])["macs"])
    assert counted == traced and counted["updated"] and counted["theta"]
    assert step["by"]["fwd/updated"] == sum(macs for _, macs in counted["updated"].values())
    assert {row[1] for row in log if row[2] != "mm"} <= {"theta", "act"}

    table = [shape["ctx"], shape["d_model"] // shape["n_head"] // 2]
    rotations = [row for row in log if json.loads(row[3])[-1] == table]
    assert rotations and {(row[1], row[2]) for row in rotations} == {("theta", "mul")}
    assert "fwd/updated" not in rec["steps"][0]["by"] and rec["steps"][0]["weights"] == {}
    assert not [key for s in rec["steps"] for key in s["weights"] if key.startswith("new/")]


@pytest.mark.parametrize("steps, traced", [(1, 0), (2, 1), (5, 3)])
def test_the_traced_step_has_a_step_before_it_where_the_run_has_one(tmp_path, steps, traced):
    """The traced step is the middle one of the run's last three, the second of a run of two, and the only step of a
    run of one, which reads θ alone: its circuit shows no update."""
    cfg = config.load(write_config(tmp_path / "a.toml", budget={"steps": steps, "batch": 2}),
                      starts.registry(tiny_registry(tmp_path / "reg")))
    route = config.route(cfg.route)
    tr = trace_step(cfg, route, *route.build(cfg.start.shape, cfg.dials), "cpu")
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
        tr = trace_step(cfg, r, *route.build(cfg.start.shape, cfg.dials), "cpu")
        return tr, [x for x in tr["rows"] if x["phase"] == "forward" and not x["token_dims"]]

    (bare, no_rows), (traced, rows) = trace(route), trace(SimpleNamespace(step=route.step, output=output))
    d, ff = cfg.start.shape["d_model"], cfg.start.shape["d_ff"]
    assert not no_rows and len(rows) == 1 and len(traced["rows"]) == len(bare["rows"]) + 1
    assert (rows[0]["op"], rows[0]["count"], rows[0]["affected"]) == ("mm", 1, [True, True])
    assert [rows[0]["dims"][x] for x in "mkn"] == [d, ff, d]
    assert traced["circuit"]["gates"] > bare["circuit"]["gates"]


def in_float32(step):
    """A route's `step` with its forward in float32: torch's autocast is a no-op while it runs."""
    def stepped(*args):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch, "autocast", lambda *a, **k: contextlib.nullcontext())
            return step(*args)

    return stepped


@pytest.mark.parametrize("steps, cast, widths", [(3, False, [32, 16]), (5, False, [32, 32]),
                                                 (3, True, [16, 16]), (5, True, [16, 16])])
def test_a_weight_is_passed_at_the_updates_width_only_among_the_runs_first_steps(tmp_path, steps, cast, widths):
    """Muon casts its update to 16 bits on its way into a 32-bit weight. With the forward in float32 (the route's
    autocast off), a run of three steps is traced at its second, whose weight is θ plus that update: a block makes
    it from the update alone, so the operand is passed at 16 bits; a longer run is traced mid-run, where the weight
    the update is added to is the run's own, and the operand is passed at its 32 bits. Under the route's bf16
    autocast the forward reads every weight at 16 bits, so the operand is passed at 16 either way."""
    cfg = config.load(write_config(tmp_path / "a.toml", budget={"steps": steps, "batch": 2},
                                   dials={"trainable": "blocks.0.w_*", "lr": 1e-2, "opt": "muon"}),
                      starts.registry(tiny_registry(tmp_path / "reg")))
    route = config.route(cfg.route)
    stepped = SimpleNamespace(step=route.step if cast else in_float32(route.step))
    tr = trace_step(cfg, stepped, *route.build(cfg.start.shape, cfg.dials), "cpu")
    read = [r["widths"] for r in tr["rows"] if r["phase"] == "forward" and r["affected"][1]]   # each reads a weight
    assert read and all(w == widths for w in read)


COUNTS = ("tokens", "matmul_macs", "elementwise_macs", "macs")


def counted_and_traced(root: Path, steps: int, **over):
    """A job's steps as it runs them (`job_counted`), and then its trace as the job takes it: the counter's record, the
    trace, the capture's graph the trace was read from, and the model."""
    rec, (cfg, route, model, optimizers) = job_counted(root, steps, **over)
    return rec, *traced_step(cfg, route, model, optimizers, "cpu"), model


#: the dtype a job's optimizer computes its products in, and their kind: Muon casts its Newton-Schulz to bf16, and
#: Shampoo's factors and roots stay in the weights' f32
OPTIMIZER_PRODUCTS = {"muon": ("bfloat16", "gemm/bf16"), "shampoo": ("float32", "gemm/f32")}
#: attention's ops, as the counter writes them for a fused kernel (`flops.Counter._attention`), and the dtype each
#: side computes them in under the routes' bf16 autocast: a fused kernel in its operands', and torch's math kernel,
#: which a trace is taken under, from operands it casts to float32
ATTENTION = {"bmm": "its products", "mul": "its scalings"}
FUSED, MATH = "bfloat16", "float32"


@contextlib.contextmanager
def attention_rows():
    """What the counter writes for a fused attention kernel while this is entered, the rows it names attention's ops
    by: {step: {(phase, kind, op, shapes, dtype): [count, MACs]}}, what each call of `flops.Counter._attention` adds
    to its step's log."""
    added, inner = {}, flops.Counter._attention

    def spy(self, func, *rest):
        before = {key: list(n) for key, n in self.cur.log.items()}
        out = inner(self, func, *rest)
        into = added.setdefault(self.cur.index, {})
        for key, n in self.cur.log.items():
            was = before.get(key, [0, 0])
            if n != was:
                into[key] = [a + b - c for a, b, c in zip(into.get(key, [0, 0]), n, was)]
        return out

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(flops.Counter, "_attention", spy)
        yield added


def in_dtype(log: list, rows: dict, dtype: str) -> list:
    """A step's op log with the rows of `rows` (`attention_rows`' of that step) named `dtype`."""
    out: dict = {}
    for *key, count, macs, was in log:
        moved = rows.get((*key, was), [0, 0])
        for name, n in ((dtype, moved), (was, [count - moved[0], macs - moved[1]])):
            if any(n):
                out[*key, name] = [a + b for a, b in zip(out.get((*key, name), [0, 0]), n)]
    return sorted([*key, *n, name] for (*key, name), n in out.items())


@pytest.mark.parametrize("job, steps", [("whole", 1), ("whole", 3), ("block", 3), ("lora", 3), ("ema", 3), ("muon", 2),
                                        ("shampoo", 3)])
def test_a_jobs_trace_and_its_counter_agree_on_the_traced_step(tmp_path, job, steps):
    """What a job's trace says of a step (`traced.traced_step`: the capture's graph of it, from shapes alone) against
    what the job's counter recorded of that same step as it ran: the step's tokens, its GEMMs' MACs, its elementwise
    MACs and all its MACs are equal; the trace's rows, which the slowdown's products are read from, hold exactly the
    GEMMs' MACs; the forward's GEMMs are equal apart; and every shape the rows run is one the counter's op log ran
    (`formalism.other_program`). At a step that reads the step before's update, and for the whole model at a run's
    only step, which reads none: there the trained embedding is θ's and its lookup counts nothing, and from the next
    step on the counter holds it at the gather Definition's multiplexers, which |C| counts and no product is, so the
    counter's forward is the trace's forward GEMMs, that lookup and the forward's elementwise work.

    In gates, attention's ops excepted by name (`ATTENTION`): the counted step's, each op at the price of the dtype
    its row names with no assumption made (`gates.of_log`), against the traced circuit's, whose nodes are priced by
    the same rule. A job runs attention under a fused kernel, in its operands' bf16, and the trace under the math
    kernel, which casts them to float32, so the rows the counter writes for the kernel name bf16 where the circuit
    holds float32: the counted step is under the circuit by those rows' MACs at the two dtypes' prices and by nothing
    else, and with those rows named float32 it is the circuit, to the last digit: both are compared as they are,
    with no tolerance, since at this shape every sum of gates is a whole number of eighths a float holds exactly. An
    optimizer's products are in the dtype it ran them in, whatever the forward's: Muon's in bf16, Shampoo's in f32,
    with its roots.

    As a job runs: under torch's fused attention kernel, which the counter counts as the ops the math kernel computes
    attention by (`flops.Counter._attention`), the ones the trace holds.

    The step's `updated` rows are not compared: the trace's are by update dependence and the counter's by use. No
    cheaper test holds the trace to the counter: each of the two has its own tests, on its own count."""
    with attention_rows() as kernel:
        rec, tr, g, model = counted_and_traced(tmp_path, steps, **JOBS[job])
    assert rec["invalid"] == [] and tr["step"] == min(1, steps - 1)
    step = rec["steps"][tr["step"]]
    assert [step[k] for k in COUNTS] == [g.tokens, g.macs, g.elementwise_macs, g.c_macs]
    log, attention = rec["structures"][step["struct"]]["log"], kernel[step["step"]]
    ops = {gates.op_key(r): r[4:6] for r in log}
    counted = gates.of_log({"by": step["by"], "ops": ops})
    assert {(key[2], key[4]) for key in attention} == {(op, FUSED) for op in ATTENTION}
    assert all(ops[gates.op_key([*key[:4], *n, key[4]])] == n for key, n in attention.items())
    as_traced = gates.of_log({"by": step["by"], "ops": {gates.op_key(r): r[4:6] for r in in_dtype(log, attention, MATH)}})
    assert counted["assumed"] == as_traced["assumed"] == {}
    assert as_traced["gates"] == tr["circuit"]["gates"] and counted["macs"] == as_traced["macs"]
    under = sum(n[1] * (gates.per_mac(gates.kind_of(*key[:3], MATH)[0]) - gates.per_mac(gates.kind_of(*key[:3], FUSED)[0]))
                for key, n in attention.items())
    assert under > 0 and tr["circuit"]["gates"] - counted["gates"] == under
    if job in OPTIMIZER_PRODUCTS:
        dtype, kind = OPTIMIZER_PRODUCTS[job]
        products = {k: v for k, v in ops.items() if k.startswith("other param mm ")}
        macs = sum(m for _, m in products.values())
        assert macs and {gates.split_key(k)[4] for k in products} == {dtype}
        assert gates.of_log({"ops": products})["kinds"] == {kind: {"macs": macs, "gates": macs * gates.per_mac(kind)}}
        assert (step["approx_macs"] > 0) == (job == "shampoo")
    rows = work.priced(tr)["rows"]
    assert tr["tokens"] == step["tokens"] and sum(r["count"] * r["macs"] for r in rows) == step["matmul_macs"]
    log = [r for r in rec["structures"][step["struct"]]["log"] if r[0] == "fwd"]
    forward = sum(r["count"] * r["macs"] for r in rows if r["phase"] == "forward" and r["token_dims"])
    assert sum(r[5] for r in log if gemm_like(r[2])) == forward > 0
    lookup = gather_macs("embedding", tuple(model.embed.shape), None, step["tokens"] * model.embed.shape[1])
    assert [r[5] for r in log if r[2] == "embedding"] == ([lookup] if job == "whole" and tr["step"] else [])
    assert sum(v for k, v in step["by"].items() if k.startswith("fwd/")) == forward + sum(
        r[5] for r in log if not gemm_like(r[2]))
    assert formalism.other_program(rows, grade.macs_log(rec, None)) is None


@pytest.mark.parametrize("job", JOBS)
def test_a_jobs_record_is_one_record_under_either_attention_kernel(tmp_path, job):
    """A job's steps under the fused attention kernel it runs against the same job's under the math kernel, the one
    its trace is taken under: every step's counts, its op log, its updated nodes and the weights it read, the setup's
    and the tail's too, but for the dtype the log names for attention's ops (`ATTENTION`). The fused kernel computes
    them in its operands' bf16, and the math kernel casts those operands to float32 first, so each record names what
    its kernel ran: the rows the counter writes for the fused kernel, named float32, are the math kernel's rows. The
    fused kernel does run: the model's attention reaches the counter as one op."""
    seen = []
    inner = flops.Counter._attention

    def spy(self, func, *rest):
        seen.append(str(func.overloadpacket))
        return inner(self, func, *rest)

    def steps_of(rec, rows=None):
        keep = ("step", "tokens", "macs", "matmul_macs", "elementwise_macs", "approx_macs", "by", "shapes", "device_macs")
        out = [{k: s.get(k) for k in keep} | rec["structures"][s["struct"]] | {"weights": sorted(s["weights"])}
               for s in (rec["setup"], *rec["steps"], rec["tail"])]
        return [s | {"log": in_dtype(s["log"], (rows or {}).get(s["step"], {}), MATH)} for s in out]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(flops.Counter, "_attention", spy)
        with attention_rows() as kernel:
            ran = job_counted(tmp_path / "fused", 3, **JOBS[job])[0]
        kernels, seen[:] = set(seen), []
        math = steps_of(job_counted(tmp_path / "math", 3, math=True, **JOBS[job])[0])
    assert kernels == {"aten._scaled_dot_product_flash_attention_for_cpu",
                       "aten._scaled_dot_product_flash_attention_for_cpu_backward"} and not seen
    fused = steps_of(ran)
    assert steps_of(ran, kernel) == math != fused and fused[2]["matmul_macs"] > 0 < fused[2]["elementwise_macs"]
    named = [{(row[2], row[6]) for row in s["log"] if (*row[:4], row[6]) in kernel.get(s["step"], {})} for s in fused]
    assert named[2] == {(op, FUSED) for op in ATTENTION} and all(n in (set(), named[2]) for n in named)


@pytest.mark.parametrize("job", ["whole", "lora"])
def test_a_jobs_count_and_its_trace_agree_in_gates_at_attention(tmp_path, job):
    """On the path a job runs, a fused attention kernel under the route's bf16 autocast, the counter counts each
    call as the math kernel's ops in the query's dtype (`flops.Counter._attention`, on stand-ins `_shaped` makes in
    it), and the trace, which is taken under the math kernel and holds those ops in float32, counts them in the
    query's dtype too (`capture.structure._counted_dtype`). So at attention the traced step's gates are the counted
    step's: the ops the counter writes for the kernel in that step, each at its dtype's price, against the gates the
    traced circuit holds at the nodes of its calls. At float32's price the same ops are more, which is what the
    circuit held before it read the call's dtype."""
    import numpy as np

    from training_slowdown import gates
    from training_slowdown.capture.structure import _attention

    added, dtypes = {}, {}
    inner, shaped = flops.Counter._attention, flops.Counter._shaped

    def attention(self, func, *rest):
        before = {key: list(n) for key, n in self.cur.log.items()}
        out = inner(self, func, *rest)
        into = added.setdefault(self.cur.index, {})
        for key, n in self.cur.log.items():
            was = before.get(key, [0, 0])
            if n != was:
                into[key] = [a + b - c for a, b, c in zip(into.get(key, [0, 0]), n, was)]
        return out

    def stand_in(self, like, shape):
        dtypes.setdefault(self.cur.index, set()).add(str(like.dtype).removeprefix("torch."))
        return shaped(self, like, shape)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(flops.Counter, "_attention", attention)
        patch.setattr(flops.Counter, "_shaped", stand_in)
        rec, tr, g, _ = counted_and_traced(tmp_path, 3, **JOBS[job])
    assert rec["invalid"] == []
    step = rec["steps"][tr["step"]]["step"]
    kernel, calls = added[step], _attention(g.graph)
    assert dtypes[step] == set(calls.values()) == {"bfloat16"} and {key[2] for key in kernel} == {"bmm", "mul"}

    def counted(dtype: str) -> float:
        return sum(macs * gates.per_mac(gates.kind_of(*key[:3], dtype)[0]) for key, (_, macs) in kernel.items())

    at = [i for i, n in enumerate(g.graph.graph.nodes) if n in calls]
    traced = float(g.circuit.gates[np.isin(g.circuit.node, at)].sum())
    assert traced == counted("bfloat16") > 0 and counted("float32") > traced
    assert {r["attention"] for r in tr["rows"] if r["op"] == "bmm"} == {"bfloat16"}
    assert sum(r["count"] * math.prod(r["dims"].values()) for r in tr["rows"] if r["attention"]) == sum(
        macs for key, (_, macs) in kernel.items() if key[2] == "bmm")


def test_a_routes_output_is_the_traces_step_and_the_counters_next_step_or_tail(tmp_path):
    """The one place the two differ. A route's `output` (LoRA's: each adapted matrix with its
    B·A merged in) is traced as the end of the measured step and counted where the harness calls it, after the
    step's checkpoint: in the step after it, or in the tail after the last. So a run of one step traces the merge in
    its step and counts it in its tail, and the step's counts differ by exactly the tail's; a longer run's traced
    step holds the merge of the checkpoint before it, and the two agree (the test above)."""
    rec, _, g, _ = counted_and_traced(tmp_path, 1, **JOBS["lora"])
    step, tail = rec["steps"][0], rec["tail"]
    assert tail["matmul_macs"] > 0 and tail["elementwise_macs"] > 0
    assert [a - step[k] for a, k in zip((g.tokens, g.macs, g.elementwise_macs, g.c_macs), COUNTS)] == [
        tail[k] for k in COUNTS]


def test_a_trace_at_another_rank_is_not_of_the_jobs_program(tmp_path):
    """`formalism.other_program` on a job's op log and a trace the job didn't take: LoRA at rank 3 runs shapes the
    rank-2 job never ran (each adapter's factors and their gradients), so the grade doesn't price its rows as that
    job's, where the job's own trace it does (the test above)."""
    rec, *_ = counted_and_traced(tmp_path, 3, **JOBS["lora"])
    cfg = config.load(write_config(tmp_path / "b.toml", budget={"steps": 3, "batch": 4}, route="lora",
                                   dials=LORA | {"rank": 3}), starts.registry(tmp_path / "reg" / "starts.toml"))
    route = config.route(cfg.route)
    tr = run.trace_step(cfg, route, *route.build(cfg.start.shape, cfg.dials), "cpu")
    o = formalism.other_program(work.priced(tr)["rows"], grade.macs_log(rec, None))
    assert o is not None and o["unmatched"] > 0 and all(3 in s for s in o["largest"]), o


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

    tr = trace_step(cfg, Branching, model, optimizers, "cpu")
    assert set(tr) == {"refused"} and set(tr["refused"]) == {"reason", "op", "message"}
    assert tr["refused"]["reason"] == "dynamic_control_flow" and "reads a tensor's value" in tr["refused"]["message"]


def test_a_step_that_deep_copies_the_model_is_recorded_as_refused_with_what_to_write_instead(tmp_path):
    """A route's step that deep-copies its model can't be traced, and the copy would end the process that tried:
    the trace says so before any tensor is copied, with the line and the state dict's copy the step is to take."""
    cfg = config.load(write_config(tmp_path / "a.toml"), starts.registry(tiny_registry(tmp_path / "reg")))
    route = config.route(cfg.route)

    class Copying:
        @staticmethod
        def step(model, rows, optimizers, dials, t, n):
            was = copy.deepcopy(model)
            route.step(model, rows, optimizers, dials, t, n)
            return was

    tr = trace_step(cfg, Copying, *route.build(cfg.start.shape, cfg.dials), "cpu")
    assert set(tr) == {"refused"} and (tr["refused"]["reason"], tr["refused"]["op"]) == ("deep_copy", None)
    assert "copy the state dict instead, {k: v.detach().clone() for k, v in model.state_dict().items()}" in \
        tr["refused"]["message"] and "in step: was = copy.deepcopy(model)" in tr["refused"]["message"]


def test_a_processs_status_says_how_it_ended():
    assert run.ended(-signal.SIGKILL) == "was ended by SIGKILL" and run.ended(-signal.SIGSEGV) == "was ended by SIGSEGV"
    assert run.ended(245) == "exited with status 245" and run.ended(-200) == "was ended by signal 200"


#: what a job's record holds of clocks: no two runs of one job agree on them
CLOCKS = {"cpu_s", "wall_s", "gpu_s", "process_cpu_s", "process_wall_s", "children_cpu_s", "self_cpu_s"}


def unclocked(x):
    if isinstance(x, dict):
        return {k: unclocked(v) for k, v in x.items() if k not in CLOCKS}
    return [unclocked(v) for v in x] if isinstance(x, list) else x


def test_the_routes_process_writes_its_record_before_it_traces_and_a_trace_that_raises_is_the_traces_reason(
        tmp_path, monkeypatch):
    """The route's process has its record on disk, whole, when the trace begins, with why there is no trace where
    the trace goes; a trace that raises leaves that record with its error as the reason, and the process ends well.
    A module the trace imported is still screened as the job's. In this process, the sandbox's check stood in for."""
    reg, scratch, sneaky = tiny_registry(tmp_path / "reg"), tmp_path / "scratch", tmp_path / "written_as_it_ran.py"
    scratch.mkdir()
    argv = ["--config", write_config(tmp_path / "a.toml"), "--seed", 0, "--split", "search", "--out",
            tmp_path / "out", "--starts", reg, "--device", "cpu", "--inside", scratch]
    monkeypatch.setattr(run.sandbox, "refusals", lambda hidden: [])
    monkeypatch.syspath_prepend(tmp_path)
    seen = []

    def trace(*a):
        seen.append(json.loads((scratch / "run.json").read_text()))
        sneaky.write_text("")
        import written_as_it_ran   # noqa: F401
        raise RuntimeError("no trace today")

    monkeypatch.setattr(run, "trace_step", trace)
    assert run.main(argv) == 0 and [p.name for p in scratch.iterdir()] == ["run.json"]
    rec = json.loads((scratch / "run.json").read_text())
    (before,), screened = seen, rec.pop("invalid")
    assert before.pop("trace") == {"failed": run.UNTRACED} and rec.pop("trace") == {"failed": "RuntimeError: no trace today"}
    assert before.pop("invalid") == [] and len(screened) == 1 and screened[0].startswith(
        "modules the job imported break the route contract: ['written_as_it_ran (") and "written during the run" in screened[0]
    assert unclocked(before) == unclocked(rec) and len(rec["steps"]) == 3 and rec["totals"]["macs"] > 0
    assert [cp["step"] for cp in rec["checkpoints"]] == [0, 1, 2] and all(cp["output_sha256"] for cp in rec["checkpoints"])


STANDIN = """#!{python}
import os, signal, sys
if sys.argv[1:3] != ["-m", "training_slowdown.run"]:
    os.execv({python!r}, [{python!r}, *sys.argv[1:]])
import training_slowdown.capture as capture
from training_slowdown import run

def ends(*a, **k):
    {how}

capture.capture_steps = ends
sys.exit(run.main(sys.argv[3:]))
"""


@pytest.mark.slow
@pytest.mark.parametrize("how, ended", [("os.kill(os.getpid(), signal.SIGKILL)", "was ended by SIGKILL"),
                                        ("os._exit(245)", "exited with status 245")])
def test_a_job_whose_routes_process_ends_as_it_takes_the_trace_writes_its_record_without_one(
        ran, tmp_path, monkeypatch, capfd, how, ended):
    """A route's process that is killed while it takes the trace, or leaves there with a status, leaves a job with
    its record: the count, the outputs' digests and the scores of the job whose trace was taken, valid, with how the
    process ended where the trace would be, and the job says so and exits as that job does. The route's process is
    the job's own but for the capture, which ends the process it runs in. Slow: a job as a node runs it, and only
    that shows the job's own process reading its child's end."""
    standin = tmp_path / "bin" / "python"
    standin.parent.mkdir()
    standin.write_text(STANDIN.format(python=sys.executable, how=how))
    standin.chmod(0o755)
    monkeypatch.setattr(run.sys, "executable", str(standin))
    rec, _, _, out = run_job(tmp_path / "job")
    why = f"the route's process {ended} after it wrote the job's record"
    assert rec["trace"] == {"failed": why} and rec["invalid"] == [] and "stopped" not in rec
    assert f"training_slowdown: {why}: the record stands, without a trace" in capfd.readouterr().err
    whole = ran[0]
    assert "capture" in whole["trace"] and rec["training_slowdown"] == whole["training_slowdown"] == VERSION

    def count(r):
        return (r["totals"]["macs"], r["theta_sha256"], r["config"], r["c"], [(s["step"], s["tokens"], s["macs"], s["by"])
                                                                             for s in r["steps"]])

    def outputs(r):
        return [(cp["step"], cp["macs"], cp["output_sha256"], cp["bpb"], cp["e_bytes"], cp.get("e_sha256"))
                for cp in r["checkpoints"]]

    assert count(rec) == count(whole) and outputs(rec) == outputs(whole) and all(cp["bpb"] for cp in rec["checkpoints"])
    assert sorted(p.name for p in out.iterdir()) == ["e", "run.json"]


def test_a_job_on_a_cpu_limits_onednn_in_each_of_its_processes_before_inside_or_outside_is_entered(
        ran, tmp_path, monkeypatch):
    """The record of a job on a CPU names the limit its route's process set. Each of the job's processes sets it
    itself, over whatever it was started with, before `inside` or `outside` is entered, and changes nothing else of
    its environment; on a GPU neither changes anything, and the record's field is null."""
    name, limit = run.CPU_ISA
    assert (name, limit) == ("ONEDNN_MAX_CPU_ISA", "AVX512_CORE_BF16") and ran[0]["cpu_isa"] == {name: limit}
    entered = []
    monkeypatch.setattr(run, "inside", lambda a: entered.append((a.cpu_isa, dict(os.environ))) or 0)
    monkeypatch.setattr(run, "outside", lambda a, argv: entered.append((a.cpu_isa, dict(os.environ))) or 0)
    job = ["--config", tmp_path / "a.toml", "--seed", 0, "--split", "search", "--out", tmp_path / "out"]
    for device, started_with in [("cpu", None), ("cpu", "ALL"), ("cuda", None), ("cuda:1", "ALL")]:
        for process in ([], ["--inside", tmp_path]):
            if started_with is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, started_with)
            before = dict(os.environ)
            assert run.main([*job, "--device", device, *process]) == 0
            said, held = entered.pop()
            assert (said, held) == (({name: limit}, before | {name: limit}) if device == "cpu" else (None, before))


#: the route's process's interpreter: the job's own, started without the limit and with oneDNN's log on
LOGGED = """#!{python}
import os, sys
if sys.argv[1:3] != ["-m", "training_slowdown.run"]:
    os.execv({python!r}, [{python!r}, *sys.argv[1:]])
os.environ.pop("ONEDNN_MAX_CPU_ISA", None)
os.environ["ONEDNN_VERBOSE"] = "1"
from training_slowdown import run

sys.exit(run.main(sys.argv[3:]))
"""


@pytest.mark.slow
def test_no_product_of_a_jobs_route_on_a_cpu_is_computed_by_an_amx_kernel(tmp_path, monkeypatch, capfd):
    """A job with Muon, whose Newton-Schulz multiplies in bf16, on a CPU: oneDNN's own log of the route's process,
    which is started without the limit, names no AMX, in the instruction set it says it uses or in a product's
    kernel, and the record names the limit that process set. On a CPU with AMX the products are oneDNN's at the
    limit, so it was set before the first of them. Slow: a job as a node runs it, and only oneDNN's log of that
    process shows the limit was read in time."""
    standin = tmp_path / "bin" / "python"
    standin.parent.mkdir()
    standin.write_text(LOGGED.format(python=sys.executable))
    standin.chmod(0o755)
    monkeypatch.setattr(run.sys, "executable", str(standin))
    dials = {"trainable": "blocks.0.*,norm", "opt": "muon", "lr": 1e-2, "weight_decay": 0.05}
    rec, _, _, _ = run_job(tmp_path / "job", dials=dials)
    log = [line for line in capfd.readouterr().out.splitlines() if line.startswith("onednn_verbose")]
    assert rec["invalid"] == [] and rec["cpu_isa"] == {"ONEDNN_MAX_CPU_ISA": "AVX512_CORE_BF16"}
    assert not [line for line in log if "amx" in line.lower()]
    if "amx_bf16" in Path("/proc/cpuinfo").read_text():
        bf16 = [line.split(",")[6] for line in log if ",exec,cpu,matmul," in line and "src:bf16" in line]
        assert bf16 and set(bf16) == {"brg_matmul:avx512_core_bf16"}


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
