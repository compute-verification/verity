import tomllib

import pytest
import torch

import spec
import steps


def parse(text: str, name: str = "t"):
    return spec.parse(tomllib.loads(text), name)


def test_every_spec_parses_with_its_scope_and_plain_arm():
    want = {"int8-linears": ("linears", "int8-linears", False), "int8-attention": ("attention", "int8-attention", True),
            "int8-all": ("all", "int8-all", True), "adopted-linears": ("linears", "int8-linears", True),
            "adopted-attention": ("attention", "int8-attention", True), "adopted": ("all", "int8-all", True),
            "adopted-no-smooth": ("all", "int8-all", True), "adopted-qkd": ("all", "int8-all", True),
            "adopted-bd": ("all", "int8-all", True), "adopted-bd-lm": ("all", "int8-all", True),
            "co": ("all", "int8-all", True), "ncp": ("all", "int8-all", True),
            "cb": ("all", "int8-all", True), "co32c": ("all", "int8-all", True)}
    for g in ("fp4c", "fp4", "int5"):
        for m in ("dither", "round"):
            want[f"{g}-{m}"] = want[f"{g}-{m}-all"] = ("all", "int8-all", True)
        want[f"{g}-dither-act"] = ("all", "int8-all", True)
    want["fp4-sr"] = want["int5-sr"] = ("all", "int8-all", True)
    want["fp4c-dither-all-kv"] = ("all", "int8-all", True)
    for name in (("nvfp4-dither", "nvfp4-sr", "nvfp4c-dither", "fp8-peel-w15", "fp8-peel-w15-gptq", "fp8-peel-w8",
                  "fp8-peel-w8-gptq", "nvfp4-int-dither", "nvfp4c-q2", "nvfp4c-q4")
                 + tuple(f"{g}-q{d}" for g in ("nvfp4", "int5", "fp8-peel-w15") for d in (2, 4, 8))):
        want[name] = ("all", "int8-all", True)
    want["nvfp4-round"] = want["nvfp4c-round"] = want["nvfp4-int-round"] = ("all", "int8-all", True)
    for name in ("pc-int8", "pc-fp8-tensor", "pc-fp8-channel", "pc-mxfp8", "pc-nvfp4", "pc-mxfp4", "pc-c3-fp8",
                 "pc-c3-fp4", *(f"pc-c1-{r}" for r in steps.ROUTES),
                 *(f"pc-c2-{g}-{s}" for g in ("int5", "e4m3i", "mx8", "nvfp4", "mxfp4") for s in ("2s", "1s")),
                 "pc-c2-e4m3i-t-2s"):
        want[name] = ("all", "int8-all", True)
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


def test_adopted_qkd_is_adopted_with_q_dithered_and_unpeeled():
    a, q = spec.load("adopted"), spec.load("adopted-qkd")
    assert q.linear == a.linear
    akw = [(s.defn.name, s.kw) for s in q.attention]
    assert [n for n, _ in akw] == ["rotate", "center_keys", "int8", "peel"]
    assert (akw[2][1]["q"], akw[2][1]["k"], akw[2][1]["v"], akw[2][1]["p_sink"]) == ("dither", "dither", "dither", True)
    assert akw[3][1] == {"operand": "p", "width": 64, "offset": 64}
    with pytest.raises(ValueError, match="int8.q is round or dither"):
        parse('[[attention]]\nstep = "int8"\nq = "peel"')


def test_adopted_bd_dithers_both_operands_of_the_linears():
    q, bd = spec.load("adopted-qkd"), spec.load("adopted-bd")
    assert bd.attention == q.attention
    kw = {s.defn.name: s.kw for s in bd.linear}
    assert (kw["int8"]["act"], kw["int8"]["weight"], kw["int8"]["weight_top"]) == ("dither", "dither", 127)
    assert not parse('[[linear]]\nstep = "int8"').seeded
    assert parse('[[linear]]\nstep = "int8"\nweight = "dither"').seeded
    for bad in ('weight = "peel"', 'weight = "dither"\ngptq = true', 'weight = "dither"\ngroup = 32',
                'weight = "dither"\nweight_top = 1'):
        with pytest.raises(ValueError, match="int8"):
            parse(f'[[linear]]\nstep = "int8"\n{bad}')


def test_lm_head_takes_only_the_steps_that_name_it():
    bd, lm = spec.load("adopted-bd"), spec.load("adopted-bd-lm")
    assert bd.attention == lm.attention
    assert [s for s in lm.linear if s.linears is None] == list(bd.linear)
    on_head = [s for s in lm.linear if s.applies(32, "linear", spec.HEAD)]
    assert [(s.defn.name, s.kw.get("act"), s.kw.get("weight")) for s in on_head] == [
        ("rotate", None, None), ("int8", "dither", "dither")]
    assert not any(s.applies(0, "linear", spec.HEAD) for s in bd.linear)
    assert not any(s.applies(0, "linear", "mlp.up_proj") for s in on_head)
    for bad in ('step = "smooth"\nalpha = 0.5', 'step = "int8"\nlayers = [0]', 'step = "int8"\ngptq = true'):
        with pytest.raises(ValueError, match="lm_head takes steps without calibration or layers"):
            parse(f'[[linear]]\n{bad}\nlinears = ["lm_head"]')


def test_co_and_ncp_serve_seven_bit_sides():
    co, ncp = spec.load("co"), spec.load("ncp")
    ckw = {s.defn.name: s.kw for s in co.linear}["int8"]
    assert (ckw["act"], ckw["act_top"], ckw["weight"], ckw["weight_top"]) == ("dither", 64, "dither", 63)
    akw = {s.defn.name: s.kw for s in co.attention}["int8"]
    assert (akw["q"], akw["q_top"], akw["k"], akw["k_top"], akw["v_top"]) == ("dither", 63, "dither", 63, 127)
    nkw = {s.defn.name: s.kw for s in ncp.linear}
    assert (nkw["int8"]["act"], nkw["int8"]["act_top"], nkw["int8"]["weight_top"]) == ("round", 63, 63)
    assert not any(s.defn.name == "peel" for s in ncp.attention)
    assert {s.defn.name: s.kw for s in ncp.attention}["int8"]["p_top"] == 63
    cb = spec.load("cb")
    assert cb.linear == ncp.linear and cb.attention == co.attention
    co32c = spec.load("co32c")
    c32 = {s.defn.name: s.kw for s in co32c.linear}["int8"]
    assert (c32["act"], c32["act_top"], c32["weight_top"]) == ("dither", 127, 63)   # 8-bit activations
    assert co32c.attention == co.attention


def test_a_dithered_operand_stays_within_its_top():
    x = torch.randn(64, 128) * 3
    for top in (127, 63):
        q, s = steps.floor_quant(x, -1, top)
        served = q + 1
        assert q.min() >= -top - 1 and served.max() <= top
        assert torch.all((x / s - q >= 0) & (x / s - q < 1))
    assert torch.equal(steps.floor_quant(x, -1)[0], steps.floor_quant(x, -1, 127)[0])


def test_round_trip():
    for p in spec.SPECS.glob("*.toml"):
        a = spec.load(str(p))
        b = spec.parse(a.to_toml(), a.name)
        assert (b.linear, b.attention, b.description) == (a.linear, a.attention, a.description)


def test_defaults_are_filled_and_typed():
    a = parse('[[linear]]\nstep = "int8"\nweight_top = 63\n[[linear]]\nstep = "int8"\nlayers = [0]\nact = "dither"')
    assert a.linear[0].kw == {"act": "round", "act_top": 127, "weight": "round", "weight_top": 63, "group": 0,
                              "gptq": False, "density": 1.0}
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
    ('[[linear]]\nstep = "int8"\nact = "dither"\nweight = "dither"\ndensity = 0.5', "a dithered weight takes density = 1"),
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
