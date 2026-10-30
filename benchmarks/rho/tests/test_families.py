import math
from dataclasses import replace

import pytest
from rho import families, instances
from rho.families import KINDS, Adapter, Nci, Partial, Seed, algorithms, family
from rho.instances import RUN_KINDS, Run
from rho.params import Registry
from rho.profile import Profile

REG = Registry.load()
MODELS = {n: instances.model(REG, n) for n in REG.names("model")}
RUN = Run("any", "kimi-k2", "pretrain", tokens=1e12)


def lacks_experts(f, model) -> bool:
    return isinstance(f, Partial) and list(f.groups) == ["experts"] and "experts" not in model.matrices


@pytest.mark.parametrize("name", REG.names("family"))
def test_every_family_gives_a_profile_at_every_model(name):
    f = family(REG, name)
    for model in MODELS.values():
        if lacks_experts(f, model):
            with pytest.raises(ValueError, match="no matrices"):
                f.profile(model, RUN)
            continue
        p = f.profile(model, RUN)
        assert 0 < p.kappa <= 1 and p.state_bits > 0 and p.legs and p.fanout


def test_every_kind_has_a_family():
    assert {REG[f"family.{n}.kind"] for n in REG.names("family")} == set(KINDS)


@pytest.mark.parametrize("name", REG.names("family"))
def test_efficiency_is_per_run_kind(name):
    keys = set(family(REG, name).efficiency)
    assert keys <= set(RUN_KINDS), f"{name}: {sorted(keys - set(RUN_KINDS))} are not run kinds"


@pytest.mark.parametrize("name", REG.names("precision"))
def test_every_precision_prices_every_run_kind(name):
    assert set(families.precision(REG, name).efficiency) == set(RUN_KINDS)


@pytest.mark.parametrize("name", REG.names("scenario"))
def test_some_family_of_every_scenario_reaches_its_target(name):
    run = instances.run(REG, REG[f"scenario.{name}.run"])
    fams = [family(REG, n) for n in REG[f"scenario.{name}.families"]]
    assert any(math.isfinite(f.profile(MODELS[run.model], run).efficiency) for f in fams)


def test_every_family_number_says_where_it_comes_from_and_how_sure_it_is():
    bad = [p.key for p in REG.params() if p.key.split(".")[0] in ("family", "precision", "nci")
           and isinstance(p.value, (int, float)) and not isinstance(p.value, bool) and not (p.source and p.status)]
    assert not bad


def adapter(form, rank, groups):
    return Adapter(name="x", form=form, rank=rank, groups=groups)


def test_lora_state_matches_the_hand_counts():
    qwen, kimi = MODELS["qwen25-1b5"], MODELS["kimi-k2"]
    # LoRA-16 (family.lora-r16) on Qwen2.5-1.5B's seven maps: 16 (2 (1536 + 1536) + 2 (1536 + 256) + 3 (1536 + 8960)) x 28
    assert adapter("lora", 16, ("attention", "dense")).sizes(qwen)[0] == 659_456 * 28 == 18_464_768
    assert adapter("xs", 5, ("attention", "dense")).sizes(qwen)[0] == 196 * 25 == 4_900
    assert adapter("xs", 16, ("attention", "dense")).sizes(qwen)[0] == 196 * 256 == 50_176
    # rows.json rl-rank1-lora-all-experts: 23,040 routed experts x 3 matrices x (7168 + 2048) = 6.37e8
    assert adapter("lora", 1, ("experts",)).sizes(kimi)[0] == 23040 * 3 * 9216 == 637_009_920
    # rows.json rl-lora-xs-all-modules: about 69,600 modules x a 2 x 2 core = 2.8e5
    assert adapter("xs", 2, ("attention", "dense", "experts", "shared")).sizes(kimi)[0] == 69_608 * 4 == 278_432
    # parameters.json update.bits_total/frontier-agentic-rl low: rank-32 LoRA on attention and shared experts, 2.7e9 bits
    p = family(REG, "lora-r32-attn").profile(kimi, RUN)
    assert math.isclose(p.state_bits, 2.7e9, rel_tol=0.06)
    assert family(REG, "tinylora-13").profile(qwen, RUN).state_bits == 272


def test_an_adapter_is_outside_update_hardness_and_its_chain_is_its_rank():
    p = family(REG, "lora-r16").profile(MODELS["kimi-k2"], RUN)
    assert p.b is None and p.bcol is None and p.legs == p.thread_d == 16
    assert p.b_w == REG["family.full.b_w"]
    # an activation into B reaches B's d_out products: the widest is the dense layer's gate and up, 18,432
    assert family(REG, "lora-r16").sizes(MODELS["kimi-k2"])[2] == 18432


def test_premises_follow_the_family_they_name():
    moved = REG.override({"family.full.b_a": 2.0, "family.full.b": 1.0})
    assert family(moved, "lora-r16").profile(MODELS["kimi-k2"], RUN).b_a == 2.0
    assert family(moved, "frozen-third").profile(MODELS["kimi-k2"], RUN).b == 1.0
    sft = family(moved, "expert-subset-sft").profile(MODELS["kimi-k2"], RUN)
    assert sft.b == REG["family.full-sft.b"] and sft.b_a == 2.0


def test_kappa_matches_kappa_py():
    kimi = MODELS["kimi-k2"]
    # benchmarks/nci/kappa.py: ESFT on Kimi K2 0.1122, full fine-tuning 0.60 (overhead 1.11 at 4k context)
    assert math.isclose(family(REG, "expert-subset-sft").profile(kimi, RUN).kappa, 0.1122, rel_tol=0.02)
    assert math.isclose(families.kappa_of(kimi, kimi.n_active, overhead=1.11), 0.60, rel_tol=0.01)
    assert family(REG, "frozen-third").profile(kimi, RUN).kappa < REG["family.full.kappa"]
    assert family(REG, "distill-online").profile(MODELS["qwen25-7b"], RUN).kappa < REG["family.distill.kappa"]


def test_seeds_are_their_description():
    assert family(REG, "es").profile(MODELS["kimi-k2"], RUN).state_bits == algorithms()["es-30"]["traits"]["description_bits"]
    p = family(REG, "mezo").profile(MODELS["kimi-k2"], RUN)
    assert p.state_bits == algorithms()["mezo"]["traits"]["description_bits"]
    assert p.b is None and p.b_w is None and p.thread_d is None


def test_a_precision_sets_beta_x_caps_activation_bits_and_prices_the_format():
    p = family(REG, "full-nvfp4").profile(MODELS["kimi-k2"], RUN)
    assert p.beta_x == REG["precision.nvfp4.beta_x"] and p.b_a <= REG["precision.nvfp4.bits"]
    assert p.efficiency == REG["precision.nvfp4.efficiency.pretrain"]
    wide = REG.override({"family.full.b_a": 16})
    assert family(wide, "full-nvfp4").profile(MODELS["kimi-k2"], RUN).b_a == REG["precision.nvfp4.bits"]


def test_a_kind_a_family_cant_reach_is_infinite():
    assert family(REG, "lora-r1").profile(MODELS["kimi-k2"], replace(RUN, kind="rl-agentic")).efficiency == math.inf


def test_sparseloco_moves_dauniels_17_gb_a_sync_of_v4_pro():
    f = family(REG, "sparseloco")
    v4 = MODELS["deepseek-v4-pro"]
    assert math.isclose(f.exchange_bits(v4) * f.sync_every, REG["exfil.sparseloco-sync.bits"], rel_tol=0.01)


def test_every_library_algorithm_gives_a_profile():
    for a in algorithms():
        p = family(REG, f"nci:{a}").profile(MODELS["kimi-k2"], RUN)
        assert p.kappa > 0 and p.state_bits > 0
    assert math.isclose(family(REG, "nci:diloco").profile(MODELS["kimi-k2"], RUN).efficiency, 1 / 0.94)
    # weight-sparsity publishes 96 bits a parameter on a quarter of the weights: the state is carried at Full's 16, the
    # rate a block pays is the library's attacker cap
    p = family(REG, "nci:weight-sparsity").profile(MODELS["kimi-k2"], RUN)
    assert math.isclose(p.state_bits, REG["nci.state_bits_cap"] * 0.25 * MODELS["kimi-k2"].n_total)
    assert p.b == p.b_w == REG["nci.rate_cap"]
    assert isinstance(family(REG, "streaming-diloco"), Nci)
    with pytest.raises(KeyError, match="no algorithm"):
        family(REG, "nci:no-such-algorithm")


def test_a_family_key_its_kind_lacks_is_refused():
    with pytest.raises(ValueError, match="not fields of Adapter"):
        families._checked(Adapter, "family.x", {"rank": 1, "workers": 4})


#: the fields the classes and attacks read beyond `thread_c`, at the values tests/test_classes.py and test_attacks.py
#: give Kimi K2 full pretraining, and how near the derivation from the matrices comes
KIMI = {"kappa_forward": (0.294, 0.005), "kappa_wgrad": (0.294, 0.005), "layers": (61, 0), "layer_weights": (4e8, 0.01),
        "layer_bits": (1.04e12 * 2.5 / 61, 0), "thread_k": (2e7, 0), "entry_width": (512, 0), "c_w": (2.5, 0),
        "c_a": (4, 0), "price": (2.8, 0), "price_max": (8, 0), "gamma": (1, 0), "beta_v": (0.63, 0),
        "map_width": (7168, 0), "map_width_min": (7168, 0), "h_eff": (17, 0), "b_ps": (1, 0), "step_tokens": (6.7e7, 0),
        "token_state_bits": (5.9e11, 0.005), "shared_state_bits": (1.18e10, 0.012), "flops_per_token": (2.07e11, 0.002),
        "cut_bits": (2 * 28672, 0), "token_bits": (20, 0), "decode_flops": (0, 0), "tile_weight_bits": (8, 0),
        "beta_x_marginal": (2.6, 0), "psum_bits": (16, 0), "tile_inputs": (7168, 0), "tile_outputs": (4096, 0)}
_FIELDS = list(Profile.__dataclass_fields__)
FILLED = ["c_w", "c_a", *_FIELDS[_FIELDS.index("thread_c") + 1:]]


def test_full_pretraining_reproduces_the_kimi_k2_values_the_classes_use():
    p = family(REG, "full").profile(MODELS["kimi-k2"], RUN)
    assert set(KIMI) | {"state_splits"} == set(FILLED)
    for name, (value, tol) in KIMI.items():
        assert math.isclose(getattr(p, name), value, rel_tol=tol, abs_tol=0 if value else 1e-12), name
    assert p.state_splits is True


@pytest.mark.parametrize("name", REG.names("family"))
def test_every_family_fills_every_field_at_every_model(name):
    f = family(REG, name)
    for model in MODELS.values():
        if lacks_experts(f, model):
            continue
        p = f.profile(model, RUN)
        unknown = {n for n in FILLED if getattr(p, n) is None}
        assert unknown == ({"kappa_wgrad"} if isinstance(f, Seed) else set()), (model.name, unknown)


def test_the_counted_sets_are_kappa_py_s():
    # benchmarks/nci/kappa.py: Kimi K2's forward products over its gates at 4k context, 0.2958; 2|M| = 1.24e11 per token
    kimi = MODELS["kimi-k2"]
    p = family(REG, "full").profile(kimi, RUN)
    assert math.isclose(p.kappa_forward, 0.2958 * REG["family.full.kappa"] / 0.6029, rel_tol=2e-3)
    assert math.isclose(p.flops_per_token * p.kappa, 1.244e11, rel_tol=2e-3)
    lora = family(REG, "lora-r16").profile(kimi, RUN)
    assert lora.kappa_forward == lora.kappa_wgrad == pytest.approx(lora.kappa / 2)


def test_a_seed_s_description_doesnt_split_and_costs_its_decode():
    kimi = MODELS["kimi-k2"]
    f = family(REG, "es")
    p = f.profile(kimi, RUN)
    assert p.state_splits is False and p.token_state_bits == p.shared_state_bits == p.state_bits
    assert p.layer_bits == p.state_bits / 61
    assert p.decode_flops == families.SEED * f.steps * f.population * kimi.n_total
    assert p.cut_bits == kimi.d_model * p.b_a and p.kappa_wgrad is None and p.kappa_forward == p.kappa
    assert family(REG, "mezo").profile(kimi, RUN).kappa_forward == pytest.approx(REG["family.mezo.kappa"] / 2)


def test_an_adapter_s_shape_is_the_adapter_s():
    kimi = MODELS["kimi-k2"]
    f = family(REG, "lora-r16")
    params, per_token, _ = f.sizes(kimi)
    p = f.profile(kimi, RUN)
    assert p.layer_bits == pytest.approx(params * p.b_w / 61)
    assert p.token_state_bits == pytest.approx(per_token * (p.b_w + REG["family.full.grad_bits"]))
    # the dense layer's MLP adapter: 16 x (2 x (7168 + 18432) + (18432 + 7168))
    assert p.layer_weights == 16 * 3 * (7168 + 18432)
    assert (p.tile_inputs, p.tile_outputs) == (7168, 16) and p.entry_width == 16
    tied = family(REG, "tinylora").profile(kimi, RUN)
    assert tied.state_splits is False and tied.decode_flops > 0 and tied.layer_weights <= REG["family.tinylora.params"]


def test_a_partial_family_spans_its_trained_layers():
    kimi = MODELS["kimi-k2"]
    f = family(REG, "frozen-third")
    p = f.profile(kimi, RUN)
    assert p.layers == round(0.67 * 61) == 41
    assert p.layer_bits == pytest.approx(f.sizes(kimi)[0] * p.b_w / 41)
    esft = family(REG, "expert-subset-sft").profile(kimi, RUN)
    assert esft.layers == 60 and esft.layer_weights == pytest.approx(0.2 * 8 * 3 * 7168 * 2048)
    assert esft.layer_bits == pytest.approx(0.1 * 23040 * 3 * 7168 * 2048 * REG["family.full-sft.b_w"] / 60)


def test_a_run_s_batch_is_its_step_and_a_teacher_s_outputs_are_data():
    kimi, qwen = MODELS["kimi-k2"], MODELS["qwen25-1b5"]
    grpo = instances.run(REG, "qwen25-1b5-grpo")
    assert family(REG, "full-rl").profile(qwen, grpo).step_tokens == REG["run.qwen25-1b5-grpo.step_tokens"]
    assert family(REG, "full").profile(kimi, RUN).step_tokens == REG["family.full.step_tokens"]
    assert family(REG, "distill").profile(kimi, RUN).token_bits == pytest.approx(20 * 69.364)
    assert family(REG, "distill-online").profile(kimi, RUN).token_bits == 20


def test_a_library_algorithm_s_shard_is_library_py_s():
    kimi = MODELS["kimi-k2"]
    t = algorithms()["es-30"]["traits"]
    p = family(REG, "nci:es-30").profile(kimi, RUN)
    assert p.state_splits is False and p.decode_flops == t["decode_flops_per_param"] * kimi.n_total
    assert p.cut_bits == kimi.d_model * 4 * 1 and p.flops_per_token == pytest.approx(2 * kimi.n_active)
    d = family(REG, "nci:diloco").profile(kimi, RUN)
    assert d.kappa_forward == pytest.approx(d.kappa / 2) and d.cut_bits == 2 * kimi.d_model * 4
