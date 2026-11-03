"""te_core.workloads models each covert workload from its dials to the one bound's kinds and rho at the console's three
policies: workloads.json (the console's tests read it) is the module's output; every row carries each kind's term,
their max, the binding kind and what it rests on, and each input's source; the console's dial formula on a point's
columns alone gives formalism.at's rho (the token kind with Fit) at every policy; the shapes, u(1), Delta, Delta_grad
and a training token's gates are what the 45M pilot's traces and the standard-workloads island's distillation measure;
the full-training records are within 1% of the model, the LoRA record above it by under 30% (the model's even u_in is
A1''s conservative form; the trace's front-loads) and the island's student below it by its measured t, every other
kind within 2% on each record; the fresh-products kind
is the amortized Loomis-Whitney cap with its entry term, over exactly the products of two fresh operands; the
reduction kind is a maximum, not 1 plus; and the readings the spec quotes hold: DiLoCo is standard training's bound
until a phase and every sync fit a block, a teacher lowers rho by the student's share and leaves the blocks, RL's
per-kind token bound is the largest, and a rank-1 LoRA whose update fits k but not Fit's reduction clause is bound by
the reduction kind."""
import inspect
import json
import math

import pytest

pytest.importorskip("matplotlib")

from te_core import algorithm_policy as AP
from te_core import panels
from te_core import workloads as W
from te_core.gates import per_mac

POLICIES = tuple(W.POLICIES)


def close(a, b, rel: float) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k], rel) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(close(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, (int, float)) and not isinstance(b, bool):
        return math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)
    return a == b


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads(json.dumps(W.reference(), allow_nan=False))


def points(ref: dict):
    for name, w in ref["workloads"].items():
        for pname, p in w["points"].items():
            yield name, pname, p


def rhos(ref, workload, point) -> dict:
    return ref["workloads"][workload]["points"][point]["rho"]


def test_reference_json_is_current(ref):
    pinned = json.loads(W.REFERENCE.read_text())
    assert close(pinned, ref, 1e-9), ("te_core/workloads.json is stale: from benchmarks/covert_witness run "
                                      "python -m te_core.workloads --json te_core/workloads.json")


def test_every_row_carries_its_kinds_their_max_and_the_binding_kind(ref):
    for name, pname, p in points(ref):
        for q in POLICIES:
            r = p["rho"][q]
            assert set(r["terms"]) <= set(W.KINDS) and "token" in r["terms"], (name, pname, q)
            terms = {kind: math.inf if x is None else x for kind, x in r["terms"].items()}
            bound = max(1.0, *terms.values())
            assert (r["bound"] is None and math.isinf(bound)) or math.isclose(r["bound"], bound), (name, pname, q)
            binding = max(terms, key=terms.get) if bound > 1 else None
            assert r["binding_kind"] == binding, (name, pname, q)
            if r["basis"] == "fit":
                assert r["fit"] and not r["fit_fails"] and r["rests_on"] == W.FIT and bound <= r["rho"] * (1 + 1e-9)
            else:
                assert r["rho"] == r["bound"] and r["basis"] == (binding or "none"), (name, pname, q)
                if binding:
                    assert r["rests_on"] == W.KINDS[binding]
                else:
                    assert "Fit fails" in r["rests_on"] and r["fit_fails"], (name, pname, q)


def test_page_rho_on_the_columns_is_at(ref):
    for name, pname, p in points(ref):
        for q in POLICIES:
            c, k = W.POLICIES[q]
            rho, basis = AP.page_rho(p["inputs"], c, k)
            want = p["rho"][q]["te_core"]
            assert basis == want["basis"], (name, pname, q)
            assert (math.isinf(rho) and want["rho"] is None) or math.isclose(rho, want["rho"], rel_tol=1e-5), \
                (name, pname, q, rho, want["rho"])
            token = p["rho"][q]["terms"]["token"]
            if want["basis"] == "target" and token is not None:
                assert math.isclose(max(1.0, token), want["rho"], rel_tol=1e-9), (name, pname, q)


def test_every_model_input_is_tagged_an_estimate(ref):
    for name, pname, p in points(ref):
        tagged = {t.split(":")[0] for t in p["estimated"].split()}
        assert tagged == set(AP.LABELLED), (name, pname, set(AP.LABELLED) - tagged)


def test_shape_and_traced_quantities():
    sh = W.dense(W.REFERENCE_PARAMS)
    assert math.isclose(sh.d_model, 512, abs_tol=1e-9) and math.isclose(sh.n_layer, 6, abs_tol=1e-9)
    st20 = W.dense(35_658_240, vocab=32_000)
    assert math.isclose(st20.d_model, 512, abs_tol=1e-9) and st20.tau == 15
    assert W.dense(35_658_240, vocab=32_000, d_model=512).n_layer == 6
    # the pilot's traces: gradient cut plus state, r20261102-140347-a1c2 (AdamW) and r20261102-134356-ee8f (Muon)
    assert W.delta_grad(sh, "adamw") == 4_322_869_248
    assert W.delta_grad(sh, "muon") == 3_706_306_560
    # starts_store.json's forward_macs_per_token for s50 and s100
    assert math.isclose(sh.params + sh.attention(1024), 51_342_080, rel_tol=1e-3)
    w = W.training(sh, "bf16", 1024).w
    assert 2.319e11 * 0.99 < w < 2.341e11 * 1.01


def test_dials_are_the_functions_parameters():
    for w in W.WORKLOADS.values():
        params = inspect.signature(w.fn).parameters
        assert [d.name for d in w.dials] == list(params), w.name
        for d in w.dials:
            assert d.default == params[d.name].default, (w.name, d.name)
        for _, settings in w.points:
            assert set(settings) <= set(params), w.name


def test_checks_are_near_their_records(ref):
    island = W.island_checks("distill")[0].name
    for name, w in ref["workloads"].items():
        for cname, ck in w["checks"].items():
            for q in POLICIES:
                record, model = ck["rho"][q]["record"], ck["rho"][q]["model"]
                if name == "standard":
                    assert math.isclose(model, record, rel_tol=0.01), (cname, q)
                elif cname == island:
                    # the model's t, the 45M trace's, against the student's measured 0.781: the token form goes as 1/t
                    assert math.isclose(model / record, 0.780834 / W.T_DENSE, rel_tol=2e-3), (cname, q)
                else:
                    assert 0.7 * record < model <= record, (cname, q)
                kinds = ck["rho"][q]["kinds"]
                assert kinds["record"]["binding_kind"] == kinds["model"]["binding_kind"] == "fresh_products"
                for kind in ("fresh_products", "reduction", "update"):
                    assert math.isclose(*(kinds[side]["terms"][kind] for side in ("record", "model")), rel_tol=0.02), \
                        (cname, q, kind)


def test_the_model_is_calibrated_on_the_records():
    pilot, student = W.dense(W.REFERENCE_PARAMS), W.dense(13_012_224, vocab=32_000, d_model=256)
    assert student.n_layer == 6 and student.tau == 15
    # u(1): the matrices in bf16, the embedding and norms in fp32, exactly the pilot's and the island's student's
    assert math.isclose(pilot.weight_bits("bf16"), 1_132_675_072, rel_tol=1e-12)
    assert student.weight_bits("bf16") == 339_320_832
    # Delta: theta' in fp32, the records' code less its header (1,440,996,608 and 416,431,792 bits)
    for sh, code in ((pilot, 1_440_996_608), (student, 416_431_792)):
        assert 0 < code - W.model("standard", n_params=sh.params, vocab=sh.vocab).measured.delta_bits < 50_000
    # Delta_grad: the island's trace, 96 N_s, and its partial gradient 32 N_s (the gradient cut less the state)
    assert W.delta_grad(student, "adamw") == 1_249_173_504
    (ck,) = W.island_checks("distill")
    assert ck.menu["delta_g_bits"] == 32 * 13_012_224 == W.model("distill", **ck.dials).steps[0].delta_g


def test_the_island_row_is_its_panels(ref):
    (ck,) = W.island_checks("distill")
    row = json.loads(W.RECORDS.read_text())["rows"][0]
    # page_rho on the row's columns gives the island's own rho at the console default, our default policy
    assert math.isclose(AP.page_rho(ck.columns, *W.CONSOLE)[0], row["rho_console"], rel_tol=1e-5)
    assert ck.dials["vocab"] == 32_000 and ck.dials["student_d_model"] == 256
    assert set(ck.dials) - {"vocab", "student_d_model"} == set(row["dials"])
    got = ref["workloads"]["distill"]["checks"][ck.name]
    assert all(v == "measured" for v in got["record_status"].values())
    measured = {key for key, v in got["model_status"].items() if v.startswith("measured: ")}
    assert measured == {"C_gates", "w_gates", "tau_bits", "a_bits", "delta_bits", "delta_t_bits", "d_gates",
                        "delta_grad_bits", "w_fit_gates", "alpha_ratio"}
    assert all(got["model_status"][key].startswith("measured: " + ck.record) for key in measured)


def test_every_input_says_where_it_comes_from(ref):
    for name, pname, p in points(ref):
        assert set(p["input_status"]) == set(W.KEYS), (name, pname)
        assert all(v.startswith(("measured: ", "assumed: ")) for v in p["input_status"].values()), (name, pname)
    assert set(ref["menu_status"]) == {"fresh", "widths", "steps", "delta_g_bits"}
    # the default dense point is the 45M pilot's AdamW record wherever the record measures the model's value
    st = ref["workloads"]["standard"]["points"]["default"]["input_status"]
    measured = {key for key, v in st.items() if v.startswith("measured: ")}
    assert measured == {"tau_bits", "a_bits", "u_points", "delta_bits", "delta_t_bits", "d_gates", "delta_grad_bits"}
    assert st["w_gates"].startswith("assumed: ") and W.PILOT_ADAMW in st["w_gates"]
    assert all(v.startswith("assumed: ") for v in ref["workloads"]["standard"]["points"]["7B"]["input_status"].values())


def test_policies_and_their_attention():
    assert W.POLICIES["default"] == (5120 * 2 * 45e6 * 740, 65_536.0)
    c, k = W.POLICIES["default_full"]
    assert k == 65_536 and round(c / W.POLICIES["default"][0], 2) == 1.57 == round(W.ATTENTION, 2)
    assert W.POLICIES["served"] == (float(panels.SERVED[0]), float(panels.SERVED[1]))
    # the console's c holds 1,457.5 training tokens of the 45M reference, which sets Fit's Δ threshold at 42.2 kbit
    s = W.CONSOLE[0] / W.training(W.dense(W.REFERENCE_PARAMS), "bf16", 1024).w
    assert 1457 < s < 1458 and 42_100 < W.CONSOLE[1] - s * 16.0156 < 42_300


def test_the_fresh_cap_is_the_amortized_loomis_whitney():
    k, bf16 = 65_536.0, W.WIDTHS["gemm/bf16"]
    assert bf16 == (16.0, 16.0, 32.0)
    cap = W.fresh_cap(k, bf16)
    assert math.isclose(cap, k ** 1.5 / (4 * math.sqrt(8192)))
    # the board's (k/3)^1.5 form, 35,700 at the default, is the per-block cap: 4/(3 sqrt 3) of the amortized one
    assert round((k / 3) ** 1.5 / math.sqrt(8192)) == 35_673
    assert math.isclose(cap / ((k / 3) ** 1.5 / math.sqrt(8192)), 3 * math.sqrt(3) / 4)
    # the Lean lane's split_attains: 32 products in 2 blocks at k = 16 and unit widths
    assert W.fresh_cap(16, (1, 1, 1)) == 16
    # each output entry is paid once: no bound below mu* b_C = 22.6 terms an entry, cap/k b_C above it
    assert math.isclose(cap / k * 32, 22.627, rel_tol=1e-4)
    assert W.fresh_blocks(22 * 1000, 1000, k, bf16) == 0.0
    assert math.isclose(W.fresh_blocks(512 * 1000, 1000, k, bf16), 512 * 1000 / cap - 32 * 1000 / k)


def test_dense_training_is_fresh_bound_at_its_charged_share(ref):
    for q in POLICIES:
        c, k = W.POLICIES[q]
        r = rhos(ref, "standard", "default")[q]
        assert r["binding_kind"] == "fresh_products" and r["rests_on"] == W.KINDS["fresh_products"]
        unit = c / (W.fresh_cap(k, W.WIDTHS["gemm/bf16"]) * per_mac("gemm/bf16"))
        assert math.isclose(r["terms"]["fresh_products"], unit * r["fresh_charged_share"], rel_tol=1e-9)
        assert 0.9 < r["fresh_charged_share"] < r["fresh_share"] < 0.99
    # 4.97e6 times the charged share at the console default
    assert 4.96e6 < W.CONSOLE[0] / (W.fresh_cap(65_536, W.WIDTHS["gemm/bf16"]) * per_mac("gemm/bf16")) < 4.98e6


def test_the_fresh_kind_counts_products_of_two_fresh_operands():
    sh = W.dense(W.REFERENCE_PARAMS)
    total = {name: sum(f.total for f in W.model(name, **dials).fresh) / W.model(name, **dials).measured.parts[0].tokens
             for name, dials in (("standard", {}), ("inference", {"precision": "bf16"}))}
    # dense training: every weight GEMM's three passes and attention's forward and backward
    assert math.isclose(total["standard"], 3 * (sh.matrix_params + sh.vocab * sh.d_model) + 3 * sh.attention(1024))
    # an unregistered model: every product of its forward
    span = W.decode_span(4096, 1024)
    assert math.isclose(total["inference"], sh.matrix_params + sh.vocab * sh.d_model + sh.attention(span))
    # a registered teacher's products all read theta
    with_teacher, alone = (W.model("distill", teacher_share=x) for x in (1.0, 0.0))
    assert with_teacher.fresh == alone.fresh
    # a LoRA on the top layer's MLP: the down projection's x.A, (x.A).B and their gradients, none of attention's
    ad = W.Adapter.of(sh, 8, "mlp", 1)
    got = sum(f.total for f in W.model("lora", targets="mlp", layers=1).fresh) / 1e8
    assert math.isclose(got, 3 * ad.rank * sh.d_model * (1 + 3 * W.MLP))


def test_a_forward_escapes_where_its_layers_update_and_ids_fit():
    f = W.Fresh("gemm/bf16", 1.0, 1.0, 1.0, 1.0, positions=1024, inputs=1024, tau=16)
    k = 65_536
    assert f.escaped(k - 16 * 1024, k) == 1.0
    assert f.escaped(k - 16 * 100, k) == 100 / 1024
    assert W.Fresh("gemm/bf16", 1.0, 1.0, 1.0, 1.0, 1024, 1024, 16, sequence=True).escaped(k - 16 * 100, k) == 0.0
    # a rank-1 W_q LoRA on six layers at the default: the update up to layers 1 and 2 fits beside a sequence's ids
    m = W.model("lora", rank=1, targets="q")
    d = m.derived["d_model"]
    (pv,) = [f for f in m.fresh if not f.sequence and f.macs == d * 1024 and f.entries == d]
    ub = W.update_bits(m.measured, W.CONSOLE[0])
    assert ub == 98_304
    assert pv.lowest == 0 and math.isclose(pv.counted(ub, 65_536)[0] / pv.total, 3 / 5)


def test_the_reduction_kind_is_a_maximum(ref):
    m = W.model("standard")
    c, k = W.CONSOLE
    (step,) = m.steps
    b = step.b(c)
    want = c / m.measured.gates * step.count * (b - 1) * step.delta_g / k
    assert math.isclose(rhos(ref, "standard", "default")["default"]["terms"]["reduction"], want, rel_tol=1e-12)
    assert math.isclose(b, step.tokens * m.derived["w_gates"] / c)


def test_diloco_is_standard_training_until_a_phase_fits_a_block(ref):
    for q in POLICIES:
        standard = rhos(ref, "standard", "default")[q]["terms"]
        for point in ("default", "1-bit syncs"):
            terms = rhos(ref, "diloco", point)[q]["terms"]
            assert math.isclose(terms["fresh_products"], standard["fresh_products"], rel_tol=2e-3)
            assert math.isclose(terms["token"], standard["token"], rel_tol=2e-3)
    corner = rhos(ref, "diloco", "Fit's corner")
    assert corner["default"]["basis"] == "fit" and 1 < corner["default"]["rho"] < 2
    # the served block can't replay a phase, so no block reads the code and the products are fresh again
    m = W.model("diloco", inner_steps=2, batch_tokens=256, syncs=32, sync_bits=1024)
    assert m.measured.decoder_gates > W.POLICIES["served"][0]
    assert corner["served"]["binding_kind"] == "fresh_products"


def test_a_teacher_lowers_rho_by_the_students_share_and_not_the_blocks(ref):
    pts = ref["workloads"]["distill"]["points"]
    with_teacher, alone = pts["default"], pts["no teacher"]
    share = with_teacher["derived"]["student_share"]
    for q in POLICIES:
        fresh = [p["rho"][q]["terms"]["fresh_products"] for p in (with_teacher, alone)]
        assert math.isclose(fresh[0] / fresh[1], share, rel_tol=1e-3)
        assert math.isclose(*(x * p["inputs"]["C_gates"] for x, p in zip(fresh, (with_teacher, alone))), rel_tol=1e-9)


def test_rl_bounds_that_count_each_kind_exceed_te_cores(ref):
    p = ref["workloads"]["rl"]["points"]["default"]
    for q in POLICIES:
        s = p["sensitivity"][q]
        assert p["rho"][q]["te_core"]["rho"] < s["tokens_worth"] < s["per_kind"]


def test_a_lora_whose_update_fits_k_but_not_fits_reduction_clause_is_reduction_bound(ref):
    served = rhos(ref, "lora", "rank 1 on W_q")["served"]
    m = W.model("lora", rank=1, targets="q")
    c, k = W.POLICIES["served"]
    assert m.measured.delta_bits + 16 * 1024 <= k < 2 * m.measured.parts[0].delta_grad_bits
    assert "fresh_products" not in served["terms"] and served["fit_fails"] == ["reduction", "update"]
    (step,) = m.steps
    want = c / m.measured.gates * step.count * (step.b(c) - 1) * step.delta_g / k
    assert served["binding_kind"] == "reduction" and math.isclose(served["rho"], want, rel_tol=1e-12) and want > 1
    assert rhos(ref, "lora", "rank 1 on W_q")["default"]["binding_kind"] == "fresh_products"
    one = rhos(ref, "lora", "rank 1 on W_q, one layer")["default"]
    assert one["rho"] == 1.0 and one["binding_kind"] is None and "reduction" in one["fit_fails"]


def test_a_lora_decoder_is_the_harnesss_rebuild():
    # prepare.rebuild_macs: (r + 2) fp32 multiply-adds per adapted entry, and r (d_in + d_out) to dequantize (lora2)
    ad = W.Adapter.of(W.dense(W.REFERENCE_PARAMS), 1, "q", 1)
    assert math.isclose(ad.decoder_gates(), 3 * 512 ** 2 * 12_637, rel_tol=1e-12)
    assert math.isclose(ad.decoder_gates(quantized=True), (3 * 512 ** 2 + 1024) * 12_637, rel_tol=1e-12)
