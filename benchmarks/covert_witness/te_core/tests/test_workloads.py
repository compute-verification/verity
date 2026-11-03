"""te_core.workloads models each covert workload from its dials to the one bound's kinds and rho at the console's three
policies: workloads.json (the console's tests read it) is the module's output; every row carries each kind's term,
their max, the binding kind and its basis (what it rests on is the reference's `rests_on`), and each input's source;
the token, reduction, update and decoding terms are te_core.formalism.bound's (reading B's token form and decoding,
reduction and update, sensitivities of its one bound), and the console's dial formula on a point's columns alone gives
its token kind (or Fit) at every policy; reading A's tiles leave no op's term infinite, and a model has no trace; the
shapes, u(1), Delta, Delta_grad, a and a training token's gates are what the 45M pilot's traces and
the standard-workloads island's rows measure; on every record the fresh-products, reduction and update kinds are
within 2% of the model's, and the token kind within 1% on the full-training records but for t (the island's dense
rows) and a (the 4.1 LoRA pilot), within 2% on the island's LoRAs; the fresh-products kind is the amortized
Loomis-Whitney cap with its entry term, over exactly the products of two fresh operands; the reduction kind is a
maximum, not 1 plus; and the readings the spec quotes hold: DiLoCo is standard training's bound until a phase and every
sync fit a block, a teacher lowers rho by the student's share and leaves the blocks, RL's per-kind token bound is the
largest, and a rank-1 LoRA whose update fits k but not Fit's reduction clause is bound by the reduction kind."""
import inspect
import json
import math

import pytest

pytest.importorskip("matplotlib")

from te_core import algorithm_policy as AP
from te_core import formalism, panels
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
    assert W.dumps(ref) + "\n" == W.REFERENCE.read_text()


def test_every_row_carries_its_kinds_their_max_and_the_binding_kind(ref):
    assert ref["kinds"] == list(W.KINDS) == ["fresh_products", *W.TE_CORE]
    for name, pname, p in points(ref):
        for q in POLICIES:
            r = p["rho"][q]
            assert list(r["terms"]) == ref["kinds"], (name, pname, q)
            terms = {kind: math.inf if x is None else x for kind, x in r["terms"].items()}
            bound = max(1.0, *terms.values())
            assert (r["bound"] is None and math.isinf(bound)) or math.isclose(r["bound"], bound), (name, pname, q)
            binding = max(terms, key=terms.get) if bound > 1 else None
            assert r["binding_kind"] == binding, (name, pname, q)
            assert r["basis"] in ref["rests_on"]
            if r["basis"] == "fit":
                assert r["fit"] and not r["fit_fails"] and bound <= r["rho"] * (1 + 1e-9)
            else:
                assert r["rho"] == r["bound"] and r["basis"] == (binding or "none"), (name, pname, q)
                assert binding or r["fit_fails"], (name, pname, q)


def test_the_kinds_are_te_cores(ref):
    """One calculator: the token, reduction, update and decoding terms are formalism.bound's sensitivities on the
    model's inputs (`TE_CORE`), and te_core's rho is Fit's, else the max over them over 1; only the fresh products'
    term is this module's."""
    for w in W.WORKLOADS.values():
        for pname, settings in w.points:
            me = W.model(w.name, **settings).measured
            for q, (c, k) in W.POLICIES.items():
                v, b, r = formalism.at(me, c, k), formalism.bound(me, c, k), rhos(ref, w.name, pname)[q]
                for kind, name in W.TE_CORE.items():
                    want = b["sensitivities"][name]
                    assert close(r["terms"][kind], None if math.isinf(want) else want, 1e-9), (w.name, pname, q, kind)
                    assert name not in b["unmeasured"]
                want = v["rho_fit"] if v["fit"] else max(1.0, *(b["sensitivities"][n] for n in W.TE_CORE.values()))
                assert close(r["te_core"]["rho"], None if math.isinf(want) else want, 1e-9)
                assert r["unmeasured"] == []


def test_page_rho_on_the_columns_is_the_token_kind_or_fit(ref):
    for name, pname, p in points(ref):
        for q in POLICIES:
            c, k = W.POLICIES[q]
            rho, basis = AP.page_rho(p["inputs"], c, k)
            r = p["rho"][q]
            assert (basis == "fit") == (r["te_core"]["basis"] == "fit"), (name, pname, q)
            want = r["te_core"]["rho"] if basis == "fit" else r["terms"]["token"]
            assert (math.isinf(rho) and want is None) or math.isclose(rho, max(1.0, want), rel_tol=1e-5), \
                (name, pname, q, rho, want)


def test_reading_as_tiles_leave_no_full_weight_gemm_infinite(ref):
    """Reading A's default packer (te_core 4.5) splits an op into matrix-product tiles that read only the coordinates
    they use, so a full-weight workload's weight GEMMs, each reading over 100 times k update bits at every policy, leave
    its term finite; it needs a trace's entries and op log, which a model hasn't (unmodelled), and a record traced
    before 4.4 is refused."""
    full = ({("standard", p) for p, _ in W.WORKLOADS["standard"].points}
            | {("diloco", "default"), ("rl", "default"), ("inference", "bf16"), ("distill", "default")})
    for name, pname, p in points(ref):
        for q, (c, k) in W.POLICIES.items():
            got = p["rho"][q]["token_A"]
            assert got["verdict"] == "unmodelled", (name, pname, q)
            if (name, pname) in full:
                assert got["op_update_bits"] > 100 * k, (name, pname, q)
    # the head's GEMM over the tied fp32 embedding, then an MLP matrix: 824 Mbit and 11.5 Mbit at the 45M reference
    assert math.isclose(W.model("standard").op_update_bits, 32 * W.VOCAB * 512) and W.MLP * 16 * 512 ** 2 > 100 * 65_536
    assert math.isclose(W.model("lora", rank=1, targets="q").op_update_bits, 16 * 512)
    for w in ref["workloads"].values():
        for ck in w["checks"].values():
            assert all(ck["rho"][q]["record"]["token_A"] == {"verdict": "refused"} for q in POLICIES)
    assert set(ref["reading_a"]["verdicts"]) == {"unmodelled", "refused"}


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


def tail(points: str) -> float:
    """A u profile's free share, where it leaves 0."""
    return next(x for x, v in AP.profile(points) if v > 0)


def test_checks_are_near_their_records(ref):
    """Every kind's term on each record beside the model's at its dials: the fresh products, reduction (by the main
    part's blocks a step, b's ceiling) and update within 2%; the token kind within 1% on the pilot's full-training
    records, by t on the island's dense rows (the model's, the 45M trace's, against each trace's: the token form goes
    as 1/t), within 2% on the island's LoRAs (a measured), and on the 4.1 LoRA pilot by its a, te_core 4.1's
    residual stream (64 times the model's 128 bits) and up to 20% more by its front-loaded u_in."""
    checks = {ck.name: ck for w in W.WORKLOADS.values() for ck in w.checks}
    assert len(checks) == 8
    for name, w in ref["workloads"].items():
        for cname, ck in w["checks"].items():
            t = tail(checks[cname].columns["u_points"])
            for q in POLICIES:
                rec, mod = ck["rho"][q]["record"], ck["rho"][q]["model"]
                assert rec["binding_kind"] == mod["binding_kind"] == "fresh_products", (cname, q)
                for kind in ("fresh_products", "update"):
                    assert math.isclose(rec["terms"][kind], mod["terms"][kind], rel_tol=0.02), (cname, q, kind)
                blocks = (max(rec["b"]) - 1) / (max(mod["b"]) - 1)
                assert math.isclose(rec["terms"]["reduction"], blocks * mod["terms"]["reduction"], rel_tol=0.02)
                ratio = mod["terms"]["token"] / rec["terms"]["token"]
                if cname.startswith("45M ref-optim"):
                    assert math.isclose(ratio, 1, rel_tol=0.01), (cname, q)
                elif name in ("standard", "diloco", "distill"):
                    assert t > W.T_DENSE and math.isclose(ratio, t / W.T_DENSE, rel_tol=0.01), (cname, q)
                elif "standard-workloads" in cname:
                    assert math.isclose(ratio, 1, rel_tol=0.02), (cname, q)
                else:
                    assert 64 <= 1 / ratio <= 64 * 1.2, (cname, q)


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
    (dl,) = W.island_checks("diloco")
    assert dl.dials["workers"] == 4 and round(W.model("diloco", **dl.dials).measured.parts[0].delta_grad_bits) \
        == 13_692_764_160 == 4 * 96 * 35_658_240
    # the gradient cut: each trace's Delta_grad less its state, 32 bits a trained value (every worker's in DiLoCo)
    for w in ("standard", "diloco", "lora", "distill"):
        for ck in W.island_checks(w):
            (part,) = W.model(w, **ck.dials).measured.parts
            assert math.isclose(ck.menu["grad_bits"], part.grad_bits, rel_tol=1e-9), ck.name
            assert ck.menu["partial_bits"] == ck.menu["grad_bits"] / ck.dials.get("workers", 1)
            assert math.isclose(ck.menu["partial_bits"], part.partial_bits, rel_tol=1e-9), ck.name
    # a LoRA's a: its lowest adapted layer's q, k and v rank outputs in bf16, 768 bits at rank 16 (both island rows)
    for ck in W.island_checks("lora"):
        assert ck.columns["a_bits"] == 768 == W.model("lora", **ck.dials).measured.a_bits, ck.name


def test_the_island_rows_are_its_panels(ref):
    rows = {r["run"]: r for r in json.loads(W.RECORDS.read_text())["rows"]}
    checks = [ck for w in ("standard", "diloco", "lora", "distill") for ck in W.island_checks(w)]
    assert {ck.record.split("@")[0] for ck in checks} == set(rows) == {
        W.ISLAND_STANDARD, W.ISLAND_DILOCO, W.ISLAND_LORA_ATTN, W.ISLAND_LORA_ALL, W.ISLAND_DISTILL}
    for ck in checks:
        row = rows[ck.record.split("@")[0]]
        # page_rho on the row's columns gives the island's own rho at the console default, our default policy
        assert math.isclose(AP.page_rho(ck.columns, *W.CONSOLE)[0], row["rho_console"], rel_tol=1e-5), ck.name
        assert ck.dials["vocab"] == 32_000 and set(ck.dials) - {"vocab", "student_d_model", "workers"} == set(row["dials"])
        assert ck.differs and ck.record == f"{row['run']}@{row['checkpoint']}"
        got = next(c for w in ref["workloads"].values() for name, c in w["checks"].items() if name == ck.name)
        assert got["record_status"]["grad_bits"].startswith("measured: ")
        assert got["record_status"]["partial_bits"].startswith("assumed: ")
        measured = {key for key, v in got["model_status"].items() if v.startswith("measured: ")}
        assert {"tau_bits", "a_bits", "delta_bits", "delta_t_bits", "d_gates"} <= measured, ck.name
        assert all(got["model_status"][key].startswith("measured: " + ck.record) for key in measured)
    distill = next(ck for ck in checks if ck.record.startswith(W.ISLAND_DISTILL))
    assert distill.dials["student_d_model"] == 256
    got = ref["workloads"]["distill"]["checks"][distill.name]
    measured = {key for key, v in got["model_status"].items() if v.startswith("measured: ")}
    assert measured == {"C_gates", "w_gates", "tau_bits", "a_bits", "delta_bits", "delta_t_bits", "d_gates",
                        "delta_grad_bits", "w_fit_gates", "alpha_ratio"}


def test_every_input_says_where_it_comes_from(ref):
    for name, pname, p in points(ref):
        assert set(p["input_status"]) == set(W.KEYS), (name, pname)
        assert all(v.startswith(("measured: ", "assumed: ")) for v in p["input_status"].values()), (name, pname)
    assert set(ref["menu_status"]) == {"fresh", "widths", "parts", "grad_bits", "partial_bits", "op_update_bits"}
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
        assert r["binding_kind"] == r["basis"] == "fresh_products"
        assert ref["rests_on"]["fresh_products"].startswith("no A1: the amortized Loomis-Whitney cap")
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
    """te_core's reduction and update terms: (c/|C|)·steps·(b − 1)·partial/k and (c/|C|)·steps·γ/k, b a step's
    token blocks; a dense step's partial and cut are 32 N (the update kind no longer counts Δ_t)."""
    m = W.model("standard")
    c, k = W.CONSOLE
    (part,) = m.measured.parts
    b = math.ceil(part.token_gates / part.steps / c)
    assert b == 45 and math.isclose(b, 65_536 * m.derived["w_gates"] / c, rel_tol=0.01)
    assert part.partial_bits == part.grad_bits == 32 * W.REFERENCE_PARAMS
    terms = rhos(ref, "standard", "default")["default"]["terms"]
    assert math.isclose(terms["reduction"], c / m.measured.gates * part.steps * (b - 1) * part.partial_bits / k,
                        rel_tol=1e-12)
    assert math.isclose(terms["update"], c / m.measured.gates * part.steps * part.grad_bits / k, rel_tol=1e-12)
    assert math.isclose(terms["reduction"] / terms["update"], b - 1)


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
        assert p["rho"][q]["terms"]["token"] < s["tokens_worth"] < s["per_kind"]


def test_a_lora_whose_update_fits_k_but_not_fits_reduction_clause_is_reduction_bound(ref):
    served = rhos(ref, "lora", "rank 1 on W_q")["served"]
    m = W.model("lora", rank=1, targets="q")
    c, k = W.POLICIES["served"]
    (part,) = m.measured.parts
    assert m.measured.delta_bits + 16 * 1024 <= k < 2 * part.delta_grad_bits
    assert served["terms"]["fresh_products"] == 0 and served["fit_fails"] == ["reduction", "update"]
    b = math.ceil(part.token_gates / part.steps / c)
    want = c / m.measured.gates * part.steps * (b - 1) * part.partial_bits / k
    assert served["binding_kind"] == "reduction" and math.isclose(served["rho"], want, rel_tol=1e-12) and want > 1
    assert rhos(ref, "lora", "rank 1 on W_q")["default"]["binding_kind"] == "fresh_products"
    one = rhos(ref, "lora", "rank 1 on W_q, one layer")["default"]
    assert one["rho"] == 1.0 and one["binding_kind"] is None and "reduction" in one["fit_fails"]


def test_a_lora_decoder_is_the_harnesss_rebuild():
    # prepare.rebuild_macs: (r + 2) fp32 multiply-adds per adapted entry, and r (d_in + d_out) to dequantize (lora2)
    ad = W.Adapter.of(W.dense(W.REFERENCE_PARAMS), 1, "q", 1)
    assert math.isclose(ad.decoder_gates(), 3 * 512 ** 2 * 12_637, rel_tol=1e-12)
    assert math.isclose(ad.decoder_gates(quantized=True), (3 * 512 ** 2 + 1024) * 12_637, rel_tol=1e-12)
