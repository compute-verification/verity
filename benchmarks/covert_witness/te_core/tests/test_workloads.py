"""te_core.workloads models each covert workload from its dials to rho's inputs and rho at the console's three
policies: workloads.json (the console's tests read it) is the module's output; the console's dial formula on a point's
columns alone gives formalism.at's rho and basis at every policy; the shapes, Delta_grad and a training token's gates
are what the 45M pilot's traces measure; the full-training records are within 1% of the model, and the LoRA record
above it by under 30% (the model's even u_in is A1''s conservative form; the trace's front-loads); and the readings
the spec quotes hold: DiLoCo is standard training's bound until a phase and every sync fit a block, a teacher
lowers rho by the student's share of the work and leaves the blocks the policy forces, RL's per-kind bound is the
largest, and a LoRA's rho falls to 1 once Delta <= k (Fit's c/(c - d) where Fit holds, d the harness's decoder for
the LoRA's E)."""
import inspect
import json
import math

import pytest

pytest.importorskip("matplotlib")

from te_core import algorithm_policy as AP
from te_core import workloads as W

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


def test_reference_json_is_current(ref):
    pinned = json.loads(W.REFERENCE.read_text())
    assert close(pinned, ref, 1e-9), ("te_core/workloads.json is stale: from benchmarks/covert_witness run "
                                      "python -m te_core.workloads --json te_core/workloads.json")


def test_page_rho_on_the_columns_is_at(ref):
    for name, pname, p in points(ref):
        for q in POLICIES:
            c, k = W.POLICIES[q]
            rho, basis = AP.page_rho(p["inputs"], c, k)
            want = p["rho"][q]
            assert basis == want["basis"], (name, pname, q)
            assert (math.isinf(rho) and want["rho"] is None) or math.isclose(rho, want["rho"], rel_tol=1e-5), \
                (name, pname, q, rho, want["rho"])


def test_every_model_input_is_tagged_an_estimate(ref):
    for name, pname, p in points(ref):
        tagged = {t.split(":")[0] for t in p["estimated"].split()}
        assert tagged == set(AP.LABELLED), (name, pname, set(AP.LABELLED) - tagged)


def test_shape_and_traced_quantities():
    sh = W.dense(W.REFERENCE_PARAMS)
    assert math.isclose(sh.d_model, 512, abs_tol=1e-9) and math.isclose(sh.n_layer, 6, abs_tol=1e-9)
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
    for name, w in ref["workloads"].items():
        for cname, ck in w["checks"].items():
            for q in POLICIES:
                record, model = ck["rho"][q]["record"], ck["rho"][q]["model"]
                if name == "standard":
                    assert math.isclose(model, record, rel_tol=0.01), (cname, q)
                else:
                    assert 0.7 * record < model <= record, (cname, q)


def rhos(ref, workload, point):
    return [ref["workloads"][workload]["points"][point]["rho"][q] for q in POLICIES]


def test_diloco_is_standard_training_until_a_phase_fits_a_block(ref):
    standard = rhos(ref, "standard", "default")
    assert close(rhos(ref, "diloco", "default"), standard, 1e-12)
    assert close(rhos(ref, "diloco", "1-bit syncs"), standard, 1e-12)
    corner = dict(zip(POLICIES, rhos(ref, "diloco", "Fit's corner")))
    assert corner["default"]["basis"] == corner["looser"]["basis"] == "fit"
    assert corner["tighter"]["basis"] == "target" and corner["tighter"]["rho"] > 100


def test_a_teacher_lowers_rho_by_the_students_share_and_not_the_blocks(ref):
    pts = ref["workloads"]["distill"]["points"]
    with_teacher, alone = pts["default"], pts["no teacher"]
    share = with_teacher["derived"]["student_share"]
    for q in POLICIES:
        ratio = with_teacher["rho"][q]["rho"] / alone["rho"][q]["rho"]
        assert math.isclose(ratio, share, rel_tol=1e-3)
        blocks = [p["rho"][q]["rho"] * p["inputs"]["C_gates"] for p in (with_teacher, alone)]
        assert math.isclose(*blocks, rel_tol=1e-3)


def test_rl_bounds_that_count_each_kind_exceed_te_cores(ref):
    p = ref["workloads"]["rl"]["points"]["default"]
    for q in POLICIES:
        s = p["sensitivity"][q]
        assert p["rho"][q]["rho"] < s["tokens_worth"] < s["per_kind"]


def test_lora_falls_to_one_once_delta_fits_k(ref):
    one = dict(zip(POLICIES, rhos(ref, "lora", "rank 1 on W_q")))
    assert one["default"]["rho"] > 50 and one["looser"]["rho"] == 1.0
    for r in rhos(ref, "lora", "rank 1 on W_q, one layer"):
        assert (r["rho"] == 1.0) if r["basis"] == "target" else (1.0 < r["rho"] < 1.0001)


def test_a_lora_decoder_is_the_harnesss_rebuild():
    # prepare.rebuild_macs: (r + 2) fp32 multiply-adds per adapted entry, and r (d_in + d_out) to dequantize (lora2)
    ad = W.Adapter.of(W.dense(W.REFERENCE_PARAMS), 1, "q", 1)
    assert math.isclose(ad.decoder_gates(), 3 * 512 ** 2 * 12_637, rel_tol=1e-12)
    assert math.isclose(ad.decoder_gates(quantized=True), (3 * 512 ** 2 + 1024) * 12_637, rel_tol=1e-12)


def test_policies_and_their_attention():
    c, k = W.POLICIES["default"]
    assert k == 65_536 and math.isclose(c, 5120 * W.REFERENCE_PARAMS * 1480.125)
    sh = W.dense(W.REFERENCE_PARAMS)
    with_attention = 1 + sh.attention(W.decode_span(4096, 1024)) / sh.params
    assert round(with_attention, 2) == 1.57
