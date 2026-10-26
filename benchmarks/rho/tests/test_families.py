import math
from dataclasses import replace

import pytest
from rho import families, instances
from rho.families import KINDS, Adapter, Nci, Partial, algorithms, family
from rho.instances import RUN_KINDS, Run
from rho.params import Registry

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
    # TVI A2's LoRA-16 on Qwen2.5-1.5B's seven maps: 16 (2 (1536 + 1536) + 2 (1536 + 256) + 3 (1536 + 8960)) x 28
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
