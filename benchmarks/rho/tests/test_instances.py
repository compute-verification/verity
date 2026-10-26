import math

import pytest
from rho import instances
from rho.instances import RUN_KINDS, Run
from rho.params import Registry

REG = Registry.load()


def shaped():
    return [instances.model(REG, n) for n in REG.names("model") if "matrices" in REG.names(f"model.{n}")]


@pytest.mark.parametrize("model", shaped(), ids=lambda m: m.name)
def test_the_matrices_add_up_to_the_stated_sizes(model):
    embedding = 0 if model.tied_embeddings else model.vocab * model.d_model
    assert math.isclose(model.params() + embedding, model.n_total, rel_tol=0.03)
    assert math.isclose(model.active(), model.n_active, rel_tol=0.05)


@pytest.mark.parametrize("m", shaped(), ids=lambda m: m.name)
def test_legs_and_fanout_are_derived_from_the_matrices(m):
    assert m.legs == instances.derived_legs(m)
    assert m.fanout == instances.derived_fanout(m)


def test_the_per_layer_shape_at_hand():
    kimi = instances.model(REG, "kimi-k2")
    att = 7168 * 1536 + 1536 * 12288 + 7168 * 576 + 512 * 16384 + 8192 * 7168
    expert, router = 3 * 7168 * 2048, 7168 * 384
    assert instances.dense_layers(kimi) == 1 and instances.per_layer(kimi, "attention") == att
    assert instances.per_layer(kimi, "experts") == 8 * expert
    # the dense layer's MLP is the least (an MoE layer's is 9 experts and the router)
    assert instances.layer_weights(kimi) == 3 * 7168 * 18432 < 9 * expert + router
    assert instances.shared_run(kimi) == att + 3 * 7168 * 18432 + att + router + expert
    assert instances.tile_shape(kimi) == (7168, 4096)
    assert instances.spanned(kimi, ["experts"]) == 60 and instances.spanned(kimi, ["dense"]) == 1
    llama = instances.model(REG, "llama3-405b")
    assert instances.shared_run(llama) is None and instances.tile_shape(llama) == (16384, 2 * 53248)
    v4 = instances.model(REG, "deepseek-v4-pro")
    assert instances.dense_layers(v4) == 0 and instances.layer_weights(v4) == 7 * 3 * 7168 * 3072 + 7168 * 384


def test_the_derivation_at_hand():
    kimi = instances.model(REG, "kimi-k2")
    assert instances.derived_fanout(kimi) == 384 + 2 * 2048 * (8 + 1) == 37248
    assert instances.derived_legs(kimi) == 512
    assert instances.derived_fanout(instances.model(REG, "deepseek-v3")) == 256 + 2 * 2048 * 9
    assert instances.derived_fanout(instances.model(REG, "dense-100b")) == 4 * 12288
    v4 = instances.model(REG, "deepseek-v4-pro")
    # wq_b's 128 x 512 query outputs beat the MoE input's 384 + 2 x 3072 x 7 = 43,392; q's latent is the narrowest input
    assert instances.derived_fanout(v4) == 128 * 512 and instances.derived_legs(v4) == 1536
    assert kimi.params(["experts"]) == 23040 * 3 * 7168 * 2048


def test_tied_embeddings_add_nothing():
    q = instances.model(REG, "qwen25-1b5")
    assert q.tied_embeddings and math.isclose(q.params(), q.n_total, rel_tol=0.01)


@pytest.mark.parametrize("name", REG.names("run"))
def test_every_run_has_a_model_and_a_kind(name):
    r = instances.run(REG, name)
    assert r.kind in RUN_KINDS
    assert r.target(instances.model(REG, r.model)) > 0


def test_a_run_of_an_unknown_kind_or_no_size_is_refused():
    with pytest.raises(ValueError, match="not one of"):
        Run("x", "kimi-k2", "post-train", tokens=1)
    with pytest.raises(ValueError, match="tokens or compute"):
        Run("x", "kimi-k2", "pretrain")
    assert Run("x", "kimi-k2", "pretrain", tokens=1e12).target(instances.model(REG, "kimi-k2")) == 6 * 3.2e10 * 1e12


def test_every_instance_number_says_where_it_comes_from_and_how_sure_it_is():
    bad = [p.key for p in REG.params() if p.key.split(".")[0] in ("model", "run")
           and isinstance(p.value, (int, float)) and not isinstance(p.value, bool) and not (p.source and p.status)]
    assert not bad
