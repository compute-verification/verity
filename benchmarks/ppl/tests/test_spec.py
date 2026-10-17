import tomllib

import pytest

import spec
import steps  # noqa: F401


def parse(text: str, name: str = "t"):
    return spec.parse(tomllib.loads(text), name)


def test_every_spec_parses_with_its_scope_and_plain_arm():
    want = {"int8-linears": ("linears", "int8-linears", False), "int8-attention": ("attention", "int8-attention", True),
            "int8-all": ("all", "int8-all", True), "adopted-linears": ("linears", "int8-linears", True),
            "adopted-attention": ("attention", "int8-attention", True), "adopted": ("all", "int8-all", True),
            "adopted-no-smooth": ("all", "int8-all", True)}
    got = {p.stem: spec.load(str(p)) for p in spec.SPECS.glob("*.toml")}
    assert set(got) == set(want)
    for name, (scope, plain, seeded) in want.items():
        a = got[name]
        assert (a.scope, a.plain, a.seeded) == (scope, plain, seeded), name
        assert a.description


def test_the_adopted_design_is_its_two_halves():
    a, lin, att = spec.load("adopted"), spec.load("adopted-linears"), spec.load("adopted-attention")
    assert a.linear == lin.linear and a.attention == att.attention
    kw = {s.defn.name: s.kw for s in a.linear}
    assert kw["smooth"] == {"alpha": 0.5, "kv_shared": True}
    assert kw["int8"]["act"] == "dither" and kw["int8"]["weight_top"] == 63
    akw = [(s.defn.name, s.kw) for s in a.attention]
    assert [n for n, _ in akw] == ["rotate", "center_keys", "int8", "peel", "peel"]
    i8 = akw[2][1]
    assert (i8["k"], i8["v"], i8["p_sink"], i8["q_top"]) == ("dither", "dither", True, 63)
    assert akw[3][1] == {"operand": "q", "width": 64, "offset": 0}
    assert akw[4][1] == {"operand": "p", "width": 64, "offset": 64}
    assert a.needs_calib and not att.needs_calib


def test_round_trip():
    for p in spec.SPECS.glob("*.toml"):
        a = spec.load(str(p))
        b = spec.parse(a.to_toml(), a.name)
        assert (b.linear, b.attention, b.description) == (a.linear, a.attention, a.description)


def test_defaults_are_filled_and_typed():
    a = parse('[[linear]]\nstep = "int8"\nweight_top = 63\n[[linear]]\nstep = "int8"\nlayers = [0]\nact = "dither"')
    assert a.linear[0].kw == {"act": "round", "act_top": 127, "weight_top": 63, "group": 0, "gptq": False}
    assert a.linear[1].layers == frozenset({0})
    s = parse('[[linear]]\nstep = "smooth"\nalpha = 1').linear[0]
    assert s.kw["alpha"] == 1.0 and isinstance(s.kw["alpha"], float)


@pytest.mark.parametrize("text, message", [
    ('[[linear]]\nstep = "nope"', "registered steps are"),
    ('[[linear]]\nstep = "int8"\nbits = 4', "unknown parameters ['bits']"),
    ('[[linear]]\nstep = "int8"\nweight_top = "63"', "an int"),
    ('[[linear]]\nstep = "int8"\ngptq = 1', "a bool"),
    ('[[linear]]\nstep = "smooth"', "parameter 'alpha' is required"),
    ('[[linear]]\nstep = "smooth"\nalpha = 2.0\n[[linear]]\nstep = "int8"', "alpha is in"),
    ('[[linear]]\nstep = "int8"\nact = "stochastic"', "round or dither"),
    ('[[linear]]\nstep = "int8"\nweight_top = 200', "tops within 127"),
    ('[[linear]]\nstep = "int8"\ngptq = true\ngroup = 32', "one scale per row"),
    ('[[linear]]\nstep = "int8"\nlinears = ["self_attn.qkv"]', "linears are among"),
    ('[[linear]]\nstep = "int8"\n[[linear]]\nstep = "rotate"', "steps run real, then one quantize"),
    ('[[linear]]\nstep = "int8"\n[[linear]]\nstep = "int8"', "one quantizer"),
    ('[[attention]]\nstep = "peel"\noperand = "q"\nwidth = 64', "an integer step needs a quantizer"),
    ('[[attention]]\nstep = "int8"\n[[attention]]\nstep = "peel"\noperand = "q"\nwidth = 64\noffset = 8',
     "offset serves P"),
    ('[[attention]]\nstep = "int8"\np_block = 48', "multiple of 32"),
    ('[[attention]]\nstep = "int8"\nk = "floor"', "round or dither"),
    ('[[linear]]\nstep = "int8"\n[mlp]\nx = 1', "unknown keys ['mlp']"),
])
def test_rejections_name_the_problem(text, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        parse(text)


def test_bf16_is_not_a_spec():
    import run
    with pytest.raises(ValueError, match="bf16 is the reference"):
        run.arms_for([str(spec.SPECS / "int8-linears.toml"), "bf16"])


def test_a_new_step_is_one_function(monkeypatch):
    reg = {k: dict(v) for k, v in spec.REGISTRY.items()}
    monkeypatch.setattr(spec, "REGISTRY", reg)

    @spec.linear_step("scale_x", phase="real")
    def scale_x(mm, nxt, *, by: float = 2.0):
        """x * by, W / by."""
        return nxt(mm.replace(x=mm.x * by, w=lambda w: w / by))

    a = parse('[[linear]]\nstep = "scale_x"\nby = 4\nlinears = ["mlp.down_proj"]\n[[linear]]\nstep = "int8"')
    assert a.linear[0].kw == {"by": 4.0}
    assert a.linear[0].applies(0, "linear", "mlp.down_proj") and not a.linear[0].applies(0, "linear", "mlp.up_proj")
    assert "linear.scale_x" in spec.catalogue()
    with pytest.raises(TypeError, match="keyword-only"):
        @spec.linear_step("bad")
        def bad(mm, nxt, by):
            return nxt(mm)
