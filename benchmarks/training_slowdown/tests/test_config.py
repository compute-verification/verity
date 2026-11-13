import pytest
from synthetic import tiny_registry, write_config

from training_slowdown import calculator, config, starts
from training_slowdown.routes import ema, subset


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    return starts.registry(tiny_registry(tmp_path_factory.mktemp("reg")))


def test_a_config_fills_its_routes_dials_and_names_its_jobs(tmp_path, reg):
    c = config.load(write_config(tmp_path / "a.toml", seeds={"search": [0, 3], "final": [101]}), reg)
    assert c.name == "a" and c.start.name == "tiny" and c.route == "subset" and c.route_version == subset.VERSION
    assert c.dials == subset.DIALS | {"trainable": "blocks.0.w_*", "lr": 1e-2}
    assert [j.label for j in c.jobs()] == [f"{c.digest}/0/search", f"{c.digest}/3/search", f"{c.digest}/101/final"]
    assert c.record()["text"] == (tmp_path / "a.toml").read_text() and c.record()["run"] == c.run()


def test_the_digest_is_what_decides_the_run(tmp_path, reg):
    """The start's θ, the route and its version, every dial and the budget change it; the seeds, the target, the
    policy and the honest config don't, and a dial left at its default digests as the default written out."""
    def digest(name, **over):
        return config.load(write_config(tmp_path / f"{name}.toml", **over), reg).digest

    base = digest("base")
    assert digest("seeds", seeds={"search": [7], "final": [1, 2]}) == base
    assert digest("target", target={"gain": 2.0}) == base
    assert digest("policy", policy=calculator.policies()[0]) == base
    assert digest("default", dials={"trainable": "blocks.0.w_*", "lr": 1e-2, "clip": subset.DIALS["clip"]}) == base
    assert digest("dial", dials={"trainable": "blocks.0.w_*", "lr": 2e-2}) != base
    assert digest("budget", budget={"steps": 4, "batch": 2}) != base
    assert digest("minutes", budget={"steps": 3, "batch": 2, "minutes": 5}) != base
    other = dict(reg, starts={"tiny": dict(reg["starts"]["tiny"], theta_file_sha256="0" * 64)})
    assert config.load(tmp_path / "base.toml", other).digest != base


def test_a_later_dial_left_at_its_default_is_out_of_the_digest(tmp_path, reg):
    """A config that sets none of its route's `LATER` dials digests the dials the route had without them, so the
    digest it had before the route gained them; one it sets is in its digest like any other."""
    dials = {"trainable": "blocks.0.w_*", "lr": 1e-2}
    base = config.load(write_config(tmp_path / "base.toml", dials=dials), reg)
    assert base.undigested == subset.LATER and all(base.dials[k] == subset.DIALS[k] for k in subset.LATER)
    assert sorted(base.run()["dials"]) == [
        "clip", "decay", "lr", "micro_batches", "opt", "store", "trainable", "warmup", "weight_decay"], (
        "subset's configs digest these nine dials: a dial added to it goes in its LATER, with the step as it was for "
        "its default, or every config's digest moves and no record names its run any more")
    written = config.load(write_config(tmp_path / "written.toml", dials=dials | {"schedule": "linear"}), reg)
    assert written.digest == base.digest
    cosine = config.load(write_config(tmp_path / "cosine.toml", dials=dials | {"schedule": "cosine"}), reg)
    assert cosine.digest != base.digest and cosine.run()["dials"] == base.run()["dials"] | {"schedule": "cosine"}
    unfused = config.load(write_config(tmp_path / "unfused.toml", dials=dials | {"fused": False}), reg)
    fused = config.load(write_config(tmp_path / "fused.toml", dials=dials | {"fused": True}), reg)
    assert unfused.digest == base.digest != fused.digest, (
        "a job continues a kept state of its own config digest alone, so one kept without `fused` is never loaded "
        "into the groups of a job with it")
    average = config.load(write_config(tmp_path / "average.toml", route="ema", dials=dials), reg)
    assert sorted(set(average.run()["dials"]) - set(base.run()["dials"])) == ["ema_decay"], (
        "ema's configs digest subset's nine dials and ema_decay: a dial added to it goes in its LATER")


@pytest.mark.parametrize("over, why", [
    ({"steps": 3}, "no key 'steps' in the schema"),
    ({"route": "nonesuch"}, "no route 'nonesuch'"),
    ({"start": "nonesuch"}, "no start 'nonesuch'"),
    ({"policy": "nonesuch"}, "no policy 'nonesuch'"),
    ({"dials": {"lr": 1e-2, "nonesuch": 1}}, "has no dial 'nonesuch'"),
    ({"dials": {"lr": "fast"}}, "dial 'lr' is str"),
    ({"dials": {"micro_batches": 1.5}}, "dial 'micro_batches' is float"),
    ({"dials": {"opt": "adam"}}, "bad.toml: subset: opt 'adam' is none of adamw, sgd, lion, adafactor, muon, shampoo"),
    ({"dials": {"store": "q9"}}, "store 'q9' is neither full nor q2 to q8"),
    ({"dials": {"decay": 0}}, r"decay is a share of the steps in \(0, 1\]"),
    ({"dials": {"schedule": "step"}}, "schedule is one of linear, cosine, constant"),
    ({"dials": {"embed": "muon"}}, "embed is adamw or opt"),
    ({"dials": {"gain_lr": -1e-3}}, "gain_lr and embed_lr are rates, or 0 for lr"),
    ({"route": "ema", "dials": {"ema_horizon": 2.0}}, r"ema_horizon is a share of the steps so far, in \[0, 1\]"),
    ({"route": "ema", "dials": {"ema_horizon": 0.1, "ema_decay": 0.9}}, "dials ema_decay and ema_horizon say one thing"),
    ({"route": "ema", "dials": {"ema_horizon": 0.1, "ema_decay": ema.DIALS["ema_decay"]}}, "route ema reads one of them"),
    ({"route": "ema", "dials": {"ema_horizon": 0.0, "ema_decay": 0.9}}, r"its EITHER\): write one"),
    ({"budget": {"steps": 3}}, "budget has no batch"),
    ({"budget": {"steps": 0, "batch": 2}}, "isn't positive"),
    ({"target": {"gain": 1.0}}, "a gain over the start's own compute"),
    ({"seeds": {"search": [0, 0]}}, "not distinct non-negative integers"),
    ({"seeds": {"search": [], "final": []}}, "no seed"),
    ({"seeds": {"test": [0]}}, "no key 'test' in seeds"),
    ({"policy": None}, "missing policy"),
])
def test_a_config_is_refused_by_name(tmp_path, reg, over, why):
    with pytest.raises(SystemExit, match=why):
        config.load(write_config(tmp_path / "bad.toml", **over), reg)


def test_a_policy_is_policy_pys_rule_or_a_table_of_the_calculators_registry(tmp_path, reg):
    """The fixed policies have one reader, which reads the registry's data files and gives the names the calculator's
    own registry does; a config names one of them or policy.py's rule."""
    fixed = calculator.policies()
    assert fixed and fixed == calculator._calculator()[4].names("policy") and not set(fixed) & set(config.RULES)
    for name in (*config.RULES, *fixed):
        assert config.load(write_config(tmp_path / f"{name}.toml", policy=name), reg).policy == name


def test_the_honest_config_is_beside_it_at_the_same_start_and_gain(tmp_path, reg):
    write_config(tmp_path / "ref.toml", dials={"trainable": "*"})
    c = config.load(write_config(tmp_path / "a.toml", honest="ref"), reg)
    h = config.honest(c, reg)
    assert h.name == "ref" and h.digest != c.digest
    assert config.honest(config.load(write_config(tmp_path / "b.toml"), reg), reg) is None
    write_config(tmp_path / "far.toml", target={"gain": 2.0})
    with pytest.raises(SystemExit, match="honest config far is at start tiny"):
        config.honest(config.load(write_config(tmp_path / "c.toml", honest="far"), reg), reg)
    with pytest.raises(SystemExit, match="names the config itself"):
        config.honest(config.load(write_config(tmp_path / "d.toml", honest="d"), reg), reg)


def test_every_route_passes_the_screen_and_states_the_contract():
    for p in sorted(config.ROUTES.glob("*.py")):
        if p.stem.startswith("_"):
            continue
        mod = config.route(p.stem)
        assert isinstance(mod.VERSION, int) and mod.DIALS and set(getattr(mod, "LATER", ())) <= set(mod.DIALS)
        assert all(len(set(group)) > 1 and set(group) <= set(mod.DIALS) for group in getattr(mod, "EITHER", ())), p.stem
        assert all(callable(getattr(mod, f, None)) for f in ("build", "step", "plan")), p.stem
