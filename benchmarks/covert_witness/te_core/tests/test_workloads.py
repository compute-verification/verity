"""te_core.workloads models each covert workload from its dials to the one bound's kinds and rho at the console's policy,
te_core.policy's CONSOLE, one full session of the 45M reference (the dashboard's 10:23Z ruling): workloads.json (the
console's tests read it) is the module's output; rho takes only the kinds valid on their own (te_core 4.5's
formalism.KINDS: the fresh products, the token kind under reading A and decoding), Fit where it holds, max(1, the
measured terms) with its binding kind and partial where a kind is unmeasured, and null, never 1, where no kind is
measured (the coordinator's 09:46Z and 11:18Z rulings); reduction, update, reading B's forms and io_bound's chain
figures are sensitivities beside it, never in its max, and the target is formalism.published at reading B (page_rho on
the columns); the fresh-products kind is te_core.formalism.fresh_products on the model's products, which match
io_bound's count on its captured steps, summed in the reference circuit's chains (te_core.formalism.ORDER), so it is
measured: io_bound's figure without P-cell, with the whole-cell figure beside it; the token and decoding
kinds need a trace's entries and op log, so they are unmeasured on every model and record here; the shapes, u(1),
Delta, Delta_grad, a and a training token's gates are what the 45M pilot's traces and the standard-workloads island's
rows (s-t20 and m-t20) measure; on every record each kind's figure and sensitivity is near the model's at its dials;
and the readings the spec quotes hold: DiLoCo is standard training's bound until a phase fits a block, a teacher
lowers rho by the student's share and leaves the blocks, RL's counting bounds exceed the target, and a LoRA whose
update fits k has its products escape."""
import inspect
import json
import math
from dataclasses import replace

import pytest

pytest.importorskip("matplotlib")

from te_core import algorithm_policy as AP
from te_core import formalism, io_bound, policy
from te_core import workloads as W

#: the console's c before the dashboard's 10:23Z ruling, 5,120 session tokens of the 45M reference at 2N FLOPs and 740
#: gates a FLOP, attention left out: the c the island's rows in workloads_records.json were published at (te_core 4.4)
OLD_C = 5120 * 2 * 45e6 * 740
WORKLOAD_OF = {ck.name: w.name for w in W.WORKLOADS.values() for ck in w.checks}
#: the island's generation rows, untraced: each record's target is te_core's untraced stub
GENERATION = {W.ISLAND_RL_LORA, W.ISLAND_RL_FULL, W.ISLAND_INFERENCE, W.ISLAND_RL_FULL_60M, W.ISLAND_INFERENCE_60M}


def close(a, b, rel: float) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k], rel) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(close(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, (int, float)) and not isinstance(b, bool):
        return math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)
    return a == b


def finite(x):
    return None if isinstance(x, float) and math.isinf(x) else x


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads(json.dumps(W.reference(), allow_nan=False))


def points(ref: dict):
    for name, w in ref["workloads"].items():
        for pname, p in w["points"].items():
            yield name, pname, p


def rhos(ref, workload, point) -> dict:
    return ref["workloads"][workload]["points"][point]["rho"]["default"]


def every_rho(ref):
    """Every published rho: each point's, and each check's on its record and on the model."""
    for name, pname, p in points(ref):
        yield (name, pname), p["rho"]["default"]
    for name, w in ref["workloads"].items():
        for cname, ck in w["checks"].items():
            for side in ("record", "model"):
                yield (name, cname, side), ck["rho"]["default"][side]


def settings(workload: str, point: str) -> dict:
    return dict(W.WORKLOADS[workload].points)[point]


def test_reference_json_is_current(ref):
    pinned = json.loads(W.REFERENCE.read_text())
    assert close(pinned, ref, 1e-6), ("te_core/workloads.json is stale: from benchmarks/covert_witness run "
                                      "python -m te_core.workloads --json te_core/workloads.json")
    assert W.dumps(ref) + "\n" == W.REFERENCE.read_text()


def test_rho_takes_only_the_kinds_valid_on_their_own(ref):
    """Coordinator 08:26Z, 10:21Z and 11:18Z, as te_core.formalism.at gives it: Fit's c/(c - d) where Fit holds, else
    max(1, the valid kinds' measured terms), its binding kind the largest where it exceeds 1 and none where none does,
    partial where a kind is unmeasured beside it, and null only where no kind is measured."""
    assert ref["valid"] == list(W.VALID) == list(formalism.KINDS) == ["fresh_products", "token", "decoding"]
    assert ref["reading"] == "A" and set(ref["rests_on"]) == {*W.VALID, "fit", "none", "unmeasured"}
    assert list(ref["sensitivities"]) == list(dict.fromkeys([*formalism.SENSITIVITIES, "fresh_chain", "fresh_cell"]))
    assert not set(ref["sensitivities"]) & set(ref["valid"])
    for at, r in every_rho(ref):
        assert list(r["terms"]) == ref["valid"] and list(r["sensitivities"]) == list(ref["sensitivities"]), at
        priced = {kind: x for kind, x in r["terms"].items() if x is not None}
        top = max([1.0, *priced.values()]) if priced else None
        binding = max(priced, key=priced.get) if priced and top > 1 else None
        assert r["binding_kind"] == binding, at
        if r["basis"] == "fit":
            assert not r["fit_fails"] and (top is None or top <= r["rho"] * (1 + 1e-9)), at
        elif binding:
            assert r["basis"] == binding and math.isclose(r["rho"], top), at
        elif priced:
            assert r["basis"] == "none" and r["rho"] == 1.0, at
        else:
            assert r["basis"] == "unmeasured" and r["rho"] is None, at
        assert r["partial"] == (r["basis"] != "fit" and bool(priced) and len(priced) < len(r["terms"])), at
        assert r["rho"] != 1.0 or r["basis"] in ("fit", "none"), at


def test_a_row_with_no_measured_kind_is_null_never_one(ref, monkeypatch):
    """Coordinator 09:46Z and 11:18Z: only a row with no measured kind publishes null, its basis unmeasured, never
    max(1, nothing). A model's products are summed in the reference circuit's chains (te_core.formalism.ORDER), so the
    fresh-products kind is measured and no row is null; were they declared summed in a tree, which io_bound has no
    constant for, every point but the one where Fit holds would be."""
    assert not [at for at, r in every_rho(ref) if r["rho"] is None]
    monkeypatch.setattr(W, "ORDER", "tree")
    c, k = policy.CONSOLE
    for name, pname, _ in points(ref):
        r = W.rho(W.model(name, **settings(name, pname)), c, k)
        if (name, pname) == ("diloco", "Fit's corner"):
            assert r["basis"] == "fit"
            continue
        assert r["rho"] is None and r["basis"] == "unmeasured", (name, pname)
        assert r["fit_fails"] and r["bound"] is None and r["binding_kind"] is None and not r["partial"], (name, pname)
        assert set(r["why"]) == set(W.VALID) <= set(r["unmeasured"]), (name, pname)
        assert "declares its products summed in a tree" in r["why"]["fresh_products"], (name, pname)
    assert ref["rests_on"]["unmeasured"].startswith("unmeasured")


def test_a_measured_kind_publishes_its_max_with_one_beneath_it():
    """Coordinator 11:18Z, where a kind is measured: rho is max(1, the measured terms), labelled with its binding kind
    (none where no measured term exceeds 1), and partial where another kind is unmeasured. The products are summed in
    the reference circuit's chains (te_core.formalism.ORDER), whose constant io_bound has, so formalism measures the
    fresh-products kind on the model, and the term is io_bound's figure without P-cell, formalism's, with its
    whole-cell figure (formalism's fresh_cell) beside it."""
    c, k = policy.CONSOLE
    for workload, point, basis in (("standard", "default", "fresh_products"), ("lora", "rank 1 on W_q", "none"),
                                   ("lora", "rank 1 on W_q, one layer", "none"), ("diloco", "Fit's corner", "fit")):
        m = W.model(workload, **settings(workload, point))
        r = W.rho(m, c, k)
        me = replace(m.measured, products={"k": k, "order": W.ORDER, "products": W.products(m, c, k)})
        term, why, got = formalism.fresh_products(me, c, k)
        assert why is None and got["fresh_cell"] == got["io_bound"]["proved"]["rho"] == r["sensitivities"]["fresh_cell"]
        assert term == r["terms"]["fresh_products"] == got["io_bound"]["proved"]["rho_no_cell"] == r["sensitivities"][
            "fresh_chain"]
        assert r["basis"] == basis and r["partial"] == (basis != "fit"), (workload, point)
        if basis == "none":
            assert r["rho"] == 1.0 and r["terms"]["fresh_products"] <= 1 and r["binding_kind"] is None
        elif basis == "fresh_products":
            assert r["rho"] == r["terms"]["fresh_products"] > 1 and r["binding_kind"] == basis


def test_the_token_and_decoding_kinds_need_a_trace(ref):
    """Te_core 4.5's packer reads a trace's entries and its op log's matrix-product tiles: a model has neither and the
    records here carry neither, so both kinds are unmeasured (never infinite, never 1), with reading A's verdict beside
    them: unmodelled on a model, refused on a record."""
    c, k = policy.CONSOLE
    for w in W.WORKLOADS.values():
        for pname, s in w.points:
            b = formalism.bound(W.model(w.name, **s).measured, c, k)
            assert b["terms"]["token"] is None and b["terms"]["decoding"] is None, (w.name, pname)
    for at, r in every_rho(ref):
        assert r["terms"]["token"] is None and r["terms"]["decoding"] is None, at
        assert r["token_A"] == ("refused" if at[-1] == "record" else "unmodelled"), at
    assert set(ref["reading_a"]) == {"unmodelled", "refused"}
    assert W.NO_TRACE in ref["rests_on"]["token"] and W.NO_TRACE in ref["rests_on"]["decoding"]


def test_the_sensitivities_and_the_target_are_te_cores(ref):
    """One calculator: reduction, update, reading B's token form and decoding and the per-gate figure are
    formalism.bound's sensitivities on the model's inputs, the target is formalism.published at reading B, and the
    console's dial formula on a point's columns alone (page_rho) gives that target, Fit's or the token form's."""
    c, k = policy.CONSOLE
    for w in W.WORKLOADS.values():
        for pname, s in w.points:
            m = W.model(w.name, **s)
            p = ref["workloads"][w.name]["points"][pname]
            me = replace(m.measured, products={"k": k, "order": W.ORDER, "products": W.products(m, c, k)})
            r, b = p["rho"]["default"], formalism.bound(me, c, k)
            for name in formalism.SENSITIVITIES:
                assert close(r["sensitivities"][name], finite(b["sensitivities"][name]), 1e-6), (w.name, pname)
            pub = formalism.published(m.measured, c, k, "B")
            assert pub == formalism.published(m.measured, reading="B")
            assert close(r["target"], {"rho": finite(pub["rho"]), "basis": pub["binding_kind"]}, 1e-6)
            got, basis = AP.page_rho(p["inputs"], c, k)
            assert (basis == "fit") == (r["target"]["basis"] == "fit"), (w.name, pname)
            assert math.isclose(got, r["target"]["rho"], rel_tol=1e-5), (w.name, pname)


def test_the_fresh_kind_is_formalisms_on_the_models_products(ref):
    """The fresh-products kind is te_core.formalism.fresh_products on the model's products as `Measured.products`, at
    the console's k and in no declared order (`ORDER`), so in the reference circuit's chains (te_core.formalism.ORDER):
    it is measured, io_bound's figure without P-cell (which needs no premise on the partition: coordinator 06:57Z,
    obligation 1), and io_bound's figures are beside it, without P-cell and with it, never below."""
    assert ref["io_bound"] == W.IO_BOUND and W.ORDER is None and (W.FRESH_FIGURE, W.FRESH_CELL) == ("rho_no_cell", "rho")
    assert not formalism.P_CELL and formalism.ORDER == "chain"
    c, k = policy.CONSOLE
    for w in W.WORKLOADS.values():
        for pname, s in w.points:
            m = W.model(w.name, **s)
            r = rhos(ref, w.name, pname)
            me = replace(m.measured, products={"k": k, "order": None, "products": W.products(m, c, k)})
            term, why, got = formalism.fresh_products(me, c, k)
            assert why is None and "fresh_products" not in r["why"] and got["order"] == formalism.ORDER
            assert close(r["terms"]["fresh_products"], term, 1e-6) and close(r["sensitivities"]["fresh_cell"],
                                                                             got["fresh_cell"], 1e-6), (w.name, pname)
            b = got["io_bound"]["proved"]
            assert close(r["sensitivities"]["fresh_chain"], b["rho_no_cell"], 1e-6), (w.name, pname)
            assert close(r["sensitivities"]["fresh_cell"], b["rho"], 1e-6), (w.name, pname)
            assert b["rho"] >= b["rho_no_cell"], (w.name, pname)
            direct = io_bound.bound(W.products(m, c, k), c, k, m.measured.gates)["proved"]
            assert direct["rho_no_cell"] == b["rho_no_cell"], (w.name, pname)


@pytest.mark.slow
def test_the_model_is_io_bounds_captured_steps():
    """The model's fresh products against te_core.io_bound's count on its captured s-shape steps (s-t20's shape, d 512,
    6 layers, V = 32,000, 256 sequences of 1,024 tokens) at the console's policy: dense Muon training's and the
    rank-16 LoRA's counted gates the capture's to 0.1%, and io_bound's chain figures within 1.2% and 2.6%, all of the
    gap |C|'s (the model's on the capture's |C| within 0.2%). It captures two steps of model.py (about 35 s): no
    cheaper test ties the model's products to a captured step's."""
    pytest.importorskip("torch")
    n, tokens = 35_658_240, 256 * 1024
    c, k = policy.CONSOLE
    assert io_bound.CONSOLE == policy.CONSOLE
    models = {"dense": W.model("standard", n_params=n, vocab=32_000, optimizer="muon", batch_tokens=tokens,
                               tokens_per_param=tokens / n),
              "lora": W.model("lora", n_params=n, vocab=32_000, rank=16, targets="all", batch_tokens=tokens,
                              tokens=tokens)}
    for name, m, rel in (("dense", models["dense"], 0.012), ("lora", models["lora"], 0.026)):
        cap = io_bound.measure(io_bound.capture(name), {"console": (c, k)})
        want = cap["policies"]["console"]["bound"]
        ps = W.products(m, c, k)
        got = io_bound.bound(ps, c, k, m.measured.gates)
        assert math.isclose(got["G"], want["G"], rel_tol=1e-3), name
        assert math.isclose(got["proved"]["rho"], want["proved"]["rho"], rel_tol=rel), name
        assert math.isclose(got["proved"]["rho_no_cell"], want["proved"]["rho_no_cell"], rel_tol=rel), name
        same = io_bound.bound(ps, c, k, cap["C_gates"])
        assert math.isclose(same["proved"]["rho"], want["proved"]["rho"], rel_tol=2e-3), name
        assert math.isclose(same["proved"]["rho_no_cell"], want["proved"]["rho_no_cell"], rel_tol=2e-3), name


def test_the_policy_is_the_consoles(ref):
    """The rows are reported at the console's policy only (the dashboard's 10:23Z ruling), te_core.policy's CONSOLE:
    one full session of the 45M reference, attention and the LM head included, 1.5739 times the attention-free c the
    console used before. A block holds 2,294 of the model's 45M training tokens, where the old c held 1,457.5 (the
    pilot's records: 1,458 to 1,471) and TRN-40's 1,250 takes a training token as 4.1 session tokens' 2N FLOPs, 17%
    above the counted 3.5. Only Fit's Δ threshold, k − s·τ, and the target read s: 28.8 kbit."""
    assert W.POLICIES == {"default": policy.CONSOLE}
    assert policy.CONSOLE == (policy.session_gates(W.REFERENCE_PARAMS, 4096, 1024), 65_536.0)
    assert ref["policies"]["default"]["c"] == policy.CONSOLE[0] and round(policy.CONSOLE[0] / OLD_C, 4) == 1.5739
    w = W.training(W.dense(W.REFERENCE_PARAMS), "bf16", 1024).w
    session = 2 * 45e6 * 740
    assert 2293 < policy.CONSOLE[0] / w < 2295 and 1457 < OLD_C / w < 1458
    assert 1458 < OLD_C / 2.32623e11 < OLD_C / 2.31908e11 < 1471
    assert round(OLD_C / 1250 / session, 3) == 4.096 and round(w / session, 2) == 3.51
    assert 28_700 < rhos(ref, "standard", "default")["fit_delta_bits"] < 28_900


def test_dials_are_the_functions_parameters():
    for w in W.WORKLOADS.values():
        params = inspect.signature(w.fn).parameters
        assert [d.name for d in w.dials] == list(params), w.name
        for d in w.dials:
            assert d.default == params[d.name].default, (w.name, d.name)
        for _, s in w.points:
            assert set(s) <= set(params), w.name
        for ck in w.checks:
            assert set(ck.dials) <= set(params), (w.name, ck.name)


def test_shape_and_traced_quantities():
    sh = W.dense(W.REFERENCE_PARAMS)
    assert math.isclose(sh.d_model, 512, abs_tol=1e-9) and math.isclose(sh.n_layer, 6, abs_tol=1e-9)
    st20 = W.dense(35_658_240, vocab=32_000)
    assert math.isclose(st20.d_model, 512, abs_tol=1e-9) and st20.tau == 15
    assert W.dense(35_658_240, vocab=32_000, d_model=512).n_layer == 6
    # m-t20, as te_core.policy states it: d 768 and 8 layers of SwiGLU 2,048, whose matrices the model's 7.84 layers of
    # 2.75 d hold to 1e-5
    shape = policy.SHAPES[W.M_T20_PARAMS]
    assert policy.parameters(shape) == W.M_T20_PARAMS and (shape["d_model"], shape["n_layer"]) == (W.M_T20, 8)
    mt20 = W.dense(W.M_T20_PARAMS, vocab=shape["vocab"], d_model=W.M_T20)
    assert math.isclose(mt20.n_layer, W.M_T20_LAYERS) and 7.83 < mt20.n_layer < 7.84
    assert math.isclose(mt20.matrix_params, 8 * (4 * 768 ** 2 + 3 * 768 * shape["d_ff"]), rel_tol=1e-5)
    half = W.dense(26_450_304, vocab=32_000, d_model=384)
    assert math.isclose(half.matrix_params, 8 * (4 * 384 ** 2 + 3 * 384 * 1024), rel_tol=1e-5)
    # the pilot's traces: gradient cut plus state, r20261102-140347-a1c2 (AdamW) and r20261102-134356-ee8f (Muon)
    assert W.delta_grad(sh, "adamw") == 4_322_869_248
    assert W.delta_grad(sh, "muon") == 3_706_306_560
    # starts_store.json's forward_macs_per_token for s50 and s100
    assert math.isclose(sh.params + sh.attention(1024), 51_342_080, rel_tol=1e-3)
    w = W.training(sh, "bf16", 1024).w
    assert 2.319e11 * 0.99 < w < 2.341e11 * 1.01
    # a served sequence: the forward at the prompt and every generated token but the last, the head at the sampled
    assert W.served(256, 512) == (767, 512 / 767)


def island(workload: str | None = None) -> list:
    return [ck for w in W.WORKLOADS.values() for ck in w.checks
            if "standard-workloads" in ck.name and workload in (None, w.name)]


def test_the_model_is_calibrated_on_the_records():
    pilot, st20 = W.dense(W.REFERENCE_PARAMS), W.dense(35_658_240, vocab=32_000)
    student = W.dense(13_012_224, vocab=32_000, d_model=256)
    assert student.n_layer == 6 and student.tau == 15
    # u(1): the matrices in bf16, the embedding and norms in fp32, exactly the pilot's and the island's
    assert math.isclose(pilot.weight_bits("bf16"), 1_132_675_072, rel_tol=1e-12)
    assert round(st20.weight_bits("bf16")) == 832_782_336 and student.weight_bits("bf16") == 339_320_832
    # Delta: theta' in fp32, the records' code less its header (1,440,996,608 and 416,431,792 bits)
    for sh, code in ((pilot, 1_440_996_608), (student, 416_431_792)):
        assert 0 < code - W.model("standard", n_params=sh.params, vocab=sh.vocab).measured.delta_bits < 50_000
    # Delta_grad: the island's traces, 96 N for AdamW (the student), Muon's state on s-t20, every worker's in DiLoCo
    assert W.delta_grad(student, "adamw") == 1_249_173_504
    assert round(W.delta_grad(st20, "muon")) == 2_806_628_352 == 1_141_063_680 + 1_665_564_672
    (dl,) = island("diloco")
    assert dl.dials["workers"] == 4 and round(W.model("diloco", **dl.dials).measured.parts[0].delta_grad_bits) \
        == 13_692_764_160 == 4 * 96 * 35_658_240
    for ck in island():
        m, col = W.model(WORKLOAD_OF[ck.name], **ck.dials), ck.columns
        me = m.measured
        # the shape's u(1), Delta and a and the run's |C|, each the row's (m-t20's at its width)
        assert math.isclose(me.u1_bits, AP.profile(col["u_points"])[-1][1], rel_tol=2e-3), ck.name
        assert math.isclose(me.delta_bits, col["delta_bits"], rel_tol=1e-3), ck.name
        assert math.isclose(me.a_bits, col["a_bits"], rel_tol=1e-9) or ck.record.startswith(W.ISLAND_RL_LORA), ck.name
        assert math.isclose(me.gates, col["C_gates"], rel_tol=0.03), ck.name
        # the gradient cut and one block's partial: 32 bits a trained value (every worker's in DiLoCo)
        main = max(me.parts, key=lambda p: p.token_gates)
        assert math.isclose(ck.menu["grad_bits"], main.grad_bits, rel_tol=1e-9), ck.name
        assert math.isclose(ck.menu["partial_bits"], main.partial_bits, rel_tol=1e-9), ck.name
    # a LoRA's a: its lowest adapted layer's q, k and v rank outputs in bf16, 768 bits at rank 16 (both island rows)
    for ck in island("lora"):
        assert ck.columns["a_bits"] == 768 == W.model("lora", **ck.dials).measured.a_bits, ck.name


def test_the_island_rows_are_its_panels(ref):
    rows = {r["run"]: r for r in json.loads(W.RECORDS.read_text())["rows"]}
    checks = island()
    assert {ck.record.split("@")[0] for ck in checks} == set(rows) == {
        W.ISLAND_STANDARD, W.ISLAND_DILOCO, W.ISLAND_LORA_ATTN, W.ISLAND_LORA_ALL, W.ISLAND_DISTILL, W.ISLAND_RL_LORA,
        W.ISLAND_RL_FULL, W.ISLAND_INFERENCE, W.ISLAND_STANDARD_60M, W.ISLAND_DISTILL_60M, W.ISLAND_RL_FULL_60M,
        W.ISLAND_INFERENCE_60M}
    assert sum(len(w.checks) for w in W.WORKLOADS.values()) == 15 and set(W.ISLAND_DIFFERS) == set(rows)
    for ck in checks:
        run = ck.record.split("@")[0]
        row = rows[run]
        # page_rho on the row's columns is the island's own rho at the c its rows were published at
        assert math.isclose(AP.page_rho(ck.columns, OLD_C, 65_536.0)[0], row["rho_console"], rel_tol=1e-5), ck.name
        width = {"60M": {"teacher_d_model" if row["workload"] == "distill" else "d_model": W.M_T20}}.get(row["model"], {})
        rounds = {"rounds": True, "tokens": sum(p["tokens"] for p in row["parts"])} if run in W.ISLAND_ROUNDS else {}
        assert ck.dials == row["dials"] | width | rounds, ck.name
        assert ck.differs == W.ISLAND_DIFFERS[run] and ck.record == f"{run}@{row['checkpoint']}"
        got = ref["workloads"][WORKLOAD_OF[ck.name]]["checks"][ck.name]
        assert got["record_status"]["grad_bits"].startswith("measured: ")
        assert got["record_status"]["partial_bits"].startswith("measured: ")
        status = W.input_status(W.model(WORKLOAD_OF[ck.name], **ck.dials), (ck,))
        measured = {key for key, v in status.items() if v.startswith("measured: ")}
        assert {"delta_bits", "delta_t_bits", "d_gates"} <= measured, ck.name
        assert all(status[key].startswith("measured: " + ck.record) for key in measured)
    assert set(W.ISLAND_ROUNDS) == {r for r, row in rows.items() if row["workload"] == "rl"}
    mt20 = [ck for ck in checks if ck.name.startswith("60M ")]
    assert len(mt20) == 4 and all(W.M_T20 in ck.dials.values() for ck in mt20)


def test_checks_are_near_their_records(ref):
    """Each kind's figure and sensitivity on each record beside the model's at its dials: io_bound's chain figure within
    2.5% (the RL LoRA's w is 3% above its record's), update within 2% and reduction by the main part's blocks a step
    within 2%. The target, reading B's token form, goes as 1/t on the traced dense rows (the model's t, the 45M
    trace's, against each trace's) and within 2% on the island's LoRAs; the 4.1 LoRA pilot's is its a, te_core 4.1's
    residual stream (64 times the model's 128 bits), and up to 20% more by its front-loaded u_in; a generation row's is
    te_core's untraced stub's, so it isn't compared."""
    checks = {ck.name: ck for w in W.WORKLOADS.values() for ck in w.checks}
    for name, w in ref["workloads"].items():
        for cname, ck in w["checks"].items():
            x = ck["rho"]["default"]
            rec, mod = x["record"], x["model"]
            s, t = rec["sensitivities"], mod["sensitivities"]
            assert rec["basis"] == mod["basis"] == "fresh_products", cname
            assert (rec["rho"], mod["rho"]) == (s["fresh_chain"], t["fresh_chain"]), cname
            assert math.isclose(s["fresh_chain"], t["fresh_chain"], rel_tol=0.025), cname
            assert math.isclose(s["update"], t["update"], rel_tol=0.02, abs_tol=1e-12), cname
            blocks = (max(rec["b"]) - 1) / (max(mod["b"]) - 1)
            assert math.isclose(s["reduction"], blocks * t["reduction"], rel_tol=0.02, abs_tol=1e-12), cname
            if checks[cname].record.split("@")[0] in GENERATION:
                continue
            ratio = mod["target"]["rho"] / rec["target"]["rho"]
            t_rec = next(x for x, v in AP.profile(checks[cname].columns["u_points"]) if v > 0)
            if cname.startswith("45M ref-optim"):
                assert math.isclose(ratio, 1, rel_tol=0.01), cname
            elif name in ("standard", "diloco", "distill"):
                assert t_rec > W.T_DENSE and math.isclose(ratio, t_rec / W.T_DENSE, rel_tol=0.02), cname
            elif "standard-workloads" in cname:
                assert math.isclose(ratio, 1, rel_tol=0.02), cname
            else:
                assert 64 <= 1 / ratio <= 64 * 1.2, cname


def test_every_input_says_where_it_comes_from(ref):
    for name, pname, p in points(ref):
        assert set(p["input_status"]) == set(W.KEYS), (name, pname)
        assert all(v.startswith(("measured: ", "assumed: ")) for v in p["input_status"].values()), (name, pname)
    assert set(ref["menu_status"]) == {"fresh", "widths", "parts", "grad_bits", "partial_bits", "op_update_bits"}
    assert all(v.startswith("assumed: ") for v in ref["menu_status"].values())
    # the default dense point is the 45M pilot's AdamW record wherever the record measures the model's value
    st = ref["workloads"]["standard"]["points"]["default"]["input_status"]
    measured = {key for key, v in st.items() if v.startswith("measured: ")}
    assert measured == {"tau_bits", "a_bits", "u_points", "delta_bits", "delta_t_bits", "d_gates", "delta_grad_bits"}
    assert all(st[key].startswith(f"measured: {W.PILOT_ADAMW}") for key in measured)
    assert st["w_gates"].startswith("assumed: ") and W.PILOT_ADAMW in st["w_gates"]
    assert all(v.startswith("assumed: ") for v in ref["workloads"]["standard"]["points"]["7B"]["input_status"].values())


def test_every_model_input_is_tagged_an_estimate(ref):
    for name, pname, p in points(ref):
        tagged = {t.split(":")[0] for t in p["estimated"].split()}
        assert tagged == set(AP.LABELLED), (name, pname, set(AP.LABELLED) - tagged)


def test_the_fresh_kind_counts_products_of_two_fresh_operands():
    sh, k = W.dense(W.REFERENCE_PARAMS), 65_536.0
    total = {name: sum(f.total(k) for f in W.model(name, **dials).fresh) / W.model(name, **dials).measured.tokens
             for name, dials in (("standard", {}), ("inference", {"precision": "bf16"}))}
    # dense training: every weight GEMM's three passes but the lowest layer's gathered q, k and v forward and weight
    # gradient, and attention's forward and backward
    gathered = 3 * sh.d_model ** 2
    assert math.isclose(total["standard"], 3 * (sh.matrix_params + sh.vocab * sh.d_model) - 2 * gathered
                        + 3 * sh.attention(1024))
    # an unregistered model: every product of its forward but the gathered ones, the head at the sampled positions
    pos, head = W.served(4096, 1024)
    span = W.decode_span(4096, 1024)
    assert math.isclose(total["inference"], sh.matrix_params - gathered + head * sh.vocab * sh.d_model
                        + sh.attention(span))
    # a registered teacher's products all read theta
    with_teacher, alone = (W.model("distill", teacher_share=x) for x in (1.0, 0.0))
    assert with_teacher.fresh == alone.fresh
    # a LoRA's base products read theta: only the adapter's and attention's above its lowest adapted layer
    m = W.model("lora", targets="mlp", layers=1)
    assert all(f.kind == "gemm/bf16" for f in m.fresh) and len([f for f in m.fresh if f.attention]) == 2


def test_a_forward_escapes_where_its_layers_update_and_ids_fit(ref):
    f = W.Fresh("gemm/bf16", 1.0, 1.0, 1.0, 1.0, positions=1024, inputs=1024, tau=16)
    k = 65_536
    assert f.escaped(k - 16 * 1024, k) == 1.0
    assert f.escaped(k - 16 * 100, k) == 100 / 1024
    assert W.Fresh("gemm/bf16", 1.0, 1.0, 1.0, 1.0, 1024, 1024, 16, sequence=True).escaped(k - 16 * 100, k) == 0.0
    # a rank-1 W_q LoRA on one layer fits k beside a sequence's ids: every product escapes, io_bound counts none
    m = W.model("lora", rank=1, targets="q", layers=1)
    ub = W.update_bits(m.measured, policy.CONSOLE[0])
    assert ub + 16 * 1024 <= k and all(f.counted(ub, k)[0] == 0 for f in m.fresh)
    assert not W.products(m, *policy.CONSOLE)
    assert rhos(ref, "lora", "rank 1 on W_q, one layer")["sensitivities"]["fresh_chain"] == 0


def test_diloco_is_standard_training_until_a_phase_fits_a_block(ref):
    standard = rhos(ref, "standard", "default")
    for point in ("default", "1-bit syncs"):
        r = rhos(ref, "diloco", point)
        assert math.isclose(r["sensitivities"]["fresh_chain"], standard["sensitivities"]["fresh_chain"], rel_tol=2e-3)
        assert math.isclose(r["target"]["rho"], standard["target"]["rho"], rel_tol=2e-3)
    # at 16 syncs of 1,024 bits a phase's code fits a block beside its tokens' ids: Fit holds, every product escapes
    corner = rhos(ref, "diloco", "Fit's corner")
    m = W.model("diloco", **settings("diloco", "Fit's corner"))
    c, _ = policy.CONSOLE
    d = m.measured.decoder_gates
    assert corner["basis"] == "fit" and not corner["fit_fails"] and d < c
    assert math.isclose(corner["rho"], c / (c - d)) and 1.28 < corner["rho"] < 1.29
    assert corner["sensitivities"]["fresh_chain"] == 0 and not corner["partial"]


def test_a_teacher_lowers_rho_by_the_students_share_and_not_the_blocks(ref):
    pts = ref["workloads"]["distill"]["points"]
    with_teacher, alone = pts["default"], pts["no teacher"]
    share = with_teacher["derived"]["student_share"]
    fresh = [p["rho"]["default"]["sensitivities"]["fresh_chain"] for p in (with_teacher, alone)]
    assert math.isclose(fresh[0] / fresh[1], share, rel_tol=1e-3)
    assert math.isclose(*(x * p["inputs"]["C_gates"] for x, p in zip(fresh, (with_teacher, alone))), rel_tol=1e-9)


def test_rl_counting_bounds_exceed_the_target(ref):
    """Reading B's counting of tokens' worth beside the target where the token blocks differ (RL's rollouts and
    learner, Q3): each kind's own profile counts more than the pooled one, which counts more than the token form."""
    p = ref["workloads"]["rl"]["points"]["default"]
    s = p["sensitivity"]["default"]
    assert p["rho"]["default"]["target"]["rho"] < s["tokens_worth"] < s["per_kind"]


def test_a_lora_decoder_is_the_harnesss_rebuild():
    # prepare.rebuild_macs: (r + 2) fp32 multiply-adds per adapted entry, and r (d_in + d_out) to dequantize (lora2)
    ad = W.Adapter.of(W.dense(W.REFERENCE_PARAMS), 1, "q", 1)
    assert math.isclose(ad.decoder_gates(), 3 * 512 ** 2 * 12_637, rel_tol=1e-12)
    assert math.isclose(ad.decoder_gates(quantized=True), (3 * 512 ** 2 + 1024) * 12_637, rel_tol=1e-12)
