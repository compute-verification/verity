"""`verity_vllm.protocols`: the protocol set's one spelling, the profile field, the refusals, and a composition that
installs innermost first, uninstalls to exactly what the model held, and records what each option committed to.

CPU only.  The PoUW and POUS adapters are stand-ins (`_fake_adapter`) over a toy linear layer whose ``quant_method.apply`` is
the hook site both real adapters wrap; the sampled-proofs adapter is the real one.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest

from verity_vllm import config
from verity_vllm import protocols as PO
from verity_vllm.engine import hooks
from verity_vllm.program.frontend.target_profile import ApplicabilityError, ArtifactIdentity, TargetProfile, check_reuse

DEFAULT_DIGEST = "86c255b375c6cf028a3bc3b7deeed96fd10b520956325accb8b541fbc90026b1"   # TargetProfile() on main before the field existed
ALL = ("pouw:ncp-v1", "pous:band-chain/d12/v1", "sampled-proofs:rows-sha512")


# --- the set's one spelling ---------------------------------------------------------------------------------------------

def test_a_set_is_spelled_one_way_in_install_order():
    assert config.parse_protocols(None) == config.parse_protocols("") == config.parse_protocols("none") == ()
    assert config.parse_protocols("sampled-proofs:rows-sha512, pous:band-chain/d12/v1,pouw:ncp-v1") == ALL
    assert config.parse_protocols({"pous": "band-chain/d12/v1", "pouw": "ncp-v1"}) == ALL[:2]
    assert config.parse_protocols(list(reversed(ALL))) == ALL
    assert config.PROTOCOLS == tuple(e.partition(":")[0] for e in ALL)


@pytest.mark.parametrize("value,match", [("zk:v1", "unknown protocol"), ("pous", "expected pous:<scheme>"), ("pous:", "expected"),
                                         ("pous:a b", "expected"), ("pous:x,pous:y", "enabled twice")])
def test_a_malformed_set_is_refused(value, match):
    with pytest.raises(ValueError, match=match):
        config.parse_protocols(value)


# --- the TargetProfile field --------------------------------------------------------------------------------------------

def test_an_undeclared_set_keeps_every_digest():
    p = TargetProfile()
    assert p.protocols is None and "protocols" not in p.to_json() and p.digest() == DEFAULT_DIGEST
    assert TargetProfile.from_json(p.to_json()).digest() == DEFAULT_DIGEST


def test_a_declared_set_is_in_the_profile_and_its_digest():
    p = TargetProfile(protocols=ALL[1:])
    assert p.to_json()["protocols"] == list(ALL[1:]) and p.digest() != DEFAULT_DIGEST
    assert TargetProfile.from_json(p.to_json()) == p
    art = ArtifactIdentity.build({"v": 1}, TargetProfile(), {}, "ab" * 32)
    with pytest.raises(ApplicabilityError, match="target_profile.protocols"):
        check_reuse(art, p, {})


@pytest.mark.parametrize("protocols,match", [(ALL[:2], "lacks sampled-proofs"), (ALL[2:], "is the default"),
                                             (tuple(reversed(ALL)), "must be spelled"), ("pous:x,sampled-proofs:rows-sha512", "string")])
def test_a_profile_refuses_another_spelling(protocols, match):
    with pytest.raises(ValueError, match=match):
        TargetProfile(protocols=protocols)


# --- stand-in adapters over a toy linear layer ----------------------------------------------------------------------------

class Apply:
    """A linear layer's quant method: ``apply`` is the GEMM (here a product) on the weight the layer carries."""

    def apply(self, layer, x, bias=None):
        return ("gemm", layer.weight, x)


class Linear:
    def __init__(self, weight):
        self.weight = weight
        self.quant_method = Apply()


class Model:
    def __init__(self):
        self.layers = [Linear(2), Linear(3)]

    def modules(self):
        return iter(self.layers)


def _fake_adapter(monkeypatch, name, install, *, refuse=(), schemes=("s1",)):
    mod = types.ModuleType(f"verity_vllm.protocols.fake_{name.replace('-', '_')}")
    mod.NAME = name
    mod.schemes = lambda: schemes
    mod.refusals = lambda scheme, ctx: tuple(refuse)
    mod.install = install
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setitem(PO.ADAPTERS, name, mod.__name__.rpartition(".")[2])
    return mod


def _pouw(log):
    """Replaces the GEMM, reading the weight from the layer it is called with."""
    def install(scheme, target):
        h = target.hooks(f"pouw:{scheme}")
        for m in target.model.modules():
            h.wrap(m.quant_method, "apply", lambda orig: (lambda layer, x, bias=None: ("pouw", layer.weight, x)))
        log.append("pouw")
        return PO.Option(name=f"pouw:{scheme}", hooks=h, commits={"weight_roots": "w"})
    return install


def _pous(log):
    """Stores W + 100, decodes it per call and hands the inner apply a layer holding the decoded weight; a responder
    service starts last and stops first."""
    def install(scheme, target):
        h = target.hooks(f"pous:{scheme}")
        for m in target.model.modules():
            m.stored = m.weight + 100
            h.wrap(m.quant_method, "apply",
                   lambda orig, m=m: (lambda layer, x, bias=None: orig(types.SimpleNamespace(weight=layer.stored - 100), x, bias)))
        h.service("responder", lambda: log.append("responder up") or "handle", lambda handle: log.append(f"responder down {handle}") or {"answered": 0})
        log.append("pous")
        return PO.Option(name=f"pous:{scheme}", hooks=h, commits={"storage_root": "c"}, release=lambda: log.append("pous released"))
    return install


def _serve(*enabled, **kw):
    return PO.Context(stage="serve", enabled=config.parse_protocols(enabled), **kw)


def test_the_composition_installs_innermost_first_and_unwinds_to_the_model_as_it_was(monkeypatch):
    log: list[str] = []
    _fake_adapter(monkeypatch, "pouw", _pouw(log), schemes=("ncp-v1",))
    _fake_adapter(monkeypatch, "pous", _pous(log), schemes=("band-chain/d12/v1",))
    model = Model()
    before = hooks.live()
    comp = PO.install(model, _serve("pous:band-chain/d12/v1", "pouw:ncp-v1"))
    assert comp.entries == ALL[:2] and log == ["pouw", "responder up", "pous"]
    layer = model.layers[0]
    assert layer.quant_method.apply(layer, "x") == ("pouw", 2, "x")          # PoUW's GEMM reads the weight POUS decoded
    assert [e.tag for e in hooks.live()[len(before):]] == ["pouw:ncp-v1"] * 2 + ["pous:band-chain/d12/v1"] * 3
    assert len(hooks.live("pous:band-chain/d12/v1")) == 3 and PO.undeclared(comp.entries) == ()
    assert PO.undeclared(("pous:band-chain/d12/v1",)) == ("pouw:ncp-v1",)
    comp.close()
    comp.close()
    assert log[3:] == ["responder down handle", "pous released"]
    assert all("apply" not in vars(m.quant_method) for m in model.layers) and hooks.live() == before
    assert layer.quant_method.apply(layer, "x") == ("gemm", 2, "x")


def test_the_record_says_what_each_option_committed_to(monkeypatch):
    _fake_adapter(monkeypatch, "pous", _pous([]), schemes=("band-chain/d12/v1",))
    ctx = PO.Context(stage="commit", enabled=ALL[1:])
    with PO.install(Model(), ctx) as comp:
        comp.bind("sampled-proofs", run_roots=["ab" * 32])
    rec = comp.record("d" * 64)
    assert rec["schema"] == PO.RECORD_SCHEMA and rec["stage"] == "commit" and rec["profile_digest"] == "d" * 64
    assert rec["protocols"] == [{"entry": "pous:band-chain/d12/v1", "commitment": {"storage_root": "c"}},
                                {"entry": "sampled-proofs:rows-sha512", "commitment": {"run_roots": ["ab" * 32]}}]
    assert rec["digest"] == comp.record("d" * 64)["digest"] != comp.record("e" * 64)["digest"]
    assert comp.verdict("d" * 64) == {"protocols": list(ALL[1:]), "record": rec["digest"]}
    assert comp.info()["sampled-proofs:rows-sha512"]["commitment_scheme"] == "rows-sha512"


def test_an_install_that_raises_leaves_nothing_installed(monkeypatch):
    log: list[str] = []
    _fake_adapter(monkeypatch, "pouw", _pouw(log), schemes=("ncp-v1",))

    def broken(scheme, target):
        target.hooks("pous:s1").wrap(target.model.layers[0].quant_method, "apply", lambda o: o)
        raise RuntimeError("encode failed")

    _fake_adapter(monkeypatch, "pous", broken)
    model, before = Model(), hooks.live()
    with pytest.raises(RuntimeError, match="encode failed"):
        PO.install(model, _serve("pouw:ncp-v1", "pous:s1"))
    assert hooks.live() == before and all("apply" not in vars(m.quant_method) for m in model.layers)


# --- refusals -----------------------------------------------------------------------------------------------------------

def test_refusals_name_every_reason_before_anything_is_installed(monkeypatch):
    _fake_adapter(monkeypatch, "pouw", _pouw([]), refuse=("compiled execution",), schemes=("ncp-v1",))
    monkeypatch.setitem(PO.ADAPTERS, "pous", "not_landed_here")
    ctx = _serve("pouw:ncp-v1", "pous:band-chain/d12/v1", "sampled-proofs:rows-sha512", world=2, settings={"pouw": {}, "zk": {}})
    got = PO.refusals(ctx)
    assert any("TP 2" in r for r in got) and any("['zk']" in r for r in got)
    assert "pouw:ncp-v1: compiled execution" in got
    assert any("verity_vllm.protocols.not_landed_here is not on this tree" in r for r in got)
    assert any(r.startswith("sampled-proofs:rows-sha512: sampled proofs runs in the Commit stage") for r in got)
    with pytest.raises(PO.Refused) as e:
        PO.install(Model(), ctx)
    assert e.value.reasons == got
    assert PO.refusals(_serve("pouw:ncp-v2")) == ("pouw:ncp-v2: pouw registers no scheme 'ncp-v2' (registered: ncp-v1)",)


def test_sampled_proofs_alone_at_the_commit_adds_no_hooks():
    ctx = PO.Context(stage="commit", enabled=ALL[2:])
    assert PO.refusals(ctx) == () and PO.adapter("sampled-proofs").schemes() == ("rows-sha512", "vllm-v1")
    before = hooks.live()
    with PO.install(Model(), ctx) as comp:
        assert comp.entries == ALL[2:] and hooks.live() == before
    assert comp.verdict() == {"protocols": list(ALL[2:]), "record": comp.record()["digest"]}


def test_a_commit_refuses_sampled_proofs_under_vllm_v1_by_name_and_a_recorded_target_still_parses(capsys):
    """vllm-v1 (SHA-256, 32-byte roots) is registered so a recorded run's target parses and digests as it did; a Commit
    refuses it by name, and refuses a committer whose scheme isn't the one sampled proofs is declared with."""
    assert config.DEFAULT_ROW_PROTOCOLS == ("sampled-proofs:rows-sha512",)
    with pytest.raises(PO.Refused, match=r"sampled-proofs:vllm-v1: vllm-v1 \(SHA-256, 32-byte roots\) is kept only to read "
                                         r"recorded runs: .* declare sampled-proofs:rows-sha512"):
        PO.install(Model(), PO.Context(stage="commit", enabled=("sampled-proofs:vllm-v1",)))
    with pytest.raises(PO.Refused, match="kept only to read recorded runs"):
        PO.at_row("sampled-proofs:vllm-v1", {})
    recorded = TargetProfile(protocols=("pouw:ncp-v2", "sampled-proofs:vllm-v1"))
    assert recorded.protocols == ("pouw:ncp-v2", "sampled-proofs:vllm-v1")
    assert recorded.digest() != TargetProfile(protocols=("pouw:ncp-v2", "sampled-proofs:rows-sha512")).digest()
    with pytest.raises(SystemExit):
        PO.commit_guard(recorded)
    assert "kept only to read recorded runs" in capsys.readouterr().out
    declared = TargetProfile(protocols=("pouw:ncp-v2", "sampled-proofs:rows-sha512"))
    assert PO.commit_guard(declared, commit_scheme="rows-sha512") is declared
    with pytest.raises(SystemExit):
        PO.commit_guard(declared, commit_scheme="vllm-v1")
    assert "its committer commits under vllm-v1" in capsys.readouterr().out
    assert PO.commit_guard(TargetProfile(), commit_scheme="vllm-v1") == TargetProfile()


def test_the_commit_context_is_none_for_the_default_and_checked_for_a_declared_set(monkeypatch):
    assert PO.commit_context(None) is None and PO.commit_context(TargetProfile()) is None
    versions: dict = {}
    assert PO.commit_guard(TargetProfile(), hot=True, installs=False) == TargetProfile() and PO.commit_guard(None) is None
    assert PO.at_commit(TargetProfile(), Model(), versions) is None and versions == {}
    PO.into_verdict(None, versions, "/nonexistent", TargetProfile(), program_digest=None, manifest=None)
    _fake_adapter(monkeypatch, "pouw", _pouw([]), schemes=("ncp-v1",))
    p = TargetProfile(protocols=ALL[:1] + ALL[2:], compilation="piecewise")
    ctx = PO.commit_context(p, settings={"pouw": {"draws": 7}})
    assert (ctx.stage, ctx.enabled, ctx.enforce_eager, ctx.scheme("pouw"), ctx.settings_of("pouw")) == \
        ("commit", ALL[:1] + ALL[2:], False, "ncp-v1", {"draws": 7})
    with pytest.raises(PO.Refused, match="TP 2"):
        PO.commit_context(p, world=2)


@pytest.mark.parametrize("kw,match", [({"world": 2}, "TP 2"), ({"hot": True}, "the hot worker"), ({"installs": False}, "TP"),
                                      ({"settings": {"pous": {}}}, "does not enable")])
def test_the_commit_guard_exits_3_naming_every_reason(monkeypatch, capsys, kw, match):
    _fake_adapter(monkeypatch, "pouw", _pouw([]), schemes=("ncp-v1",))
    with pytest.raises(SystemExit) as e:
        PO.commit_guard(TargetProfile(protocols=ALL[:1] + ALL[2:]), **kw)
    out = capsys.readouterr().out
    assert e.value.code == 3 and "REFUSE (fail-closed)" in out and match in out


def test_a_commit_with_pous_records_its_options_and_its_verdict_stands(monkeypatch, tmp_path):
    log: list[str] = []
    _fake_adapter(monkeypatch, "pous", _pous(log), schemes=("band-chain/d12/v1",))
    p, model, versions = TargetProfile(protocols=ALL[1:]), Model(), {}
    assert PO.commit_guard(p) is p
    comp = PO.at_commit(p, model, versions)
    assert versions["protocols"]["entries"] == list(ALL[1:]) and "apply" in vars(model.layers[0].quant_method)
    verdict = {"commit_pass": True, "run_roots": ["ab" * 32]}
    PO.into_verdict(comp, verdict, tmp_path, p, program_digest="cd" * 32, manifest={"manifest_digest": "ef" * 32})
    assert comp.closed and "apply" not in vars(model.layers[0].quant_method) and log[-1] == "pous released"
    rec = json.loads((tmp_path / "protocols.json").read_text())
    assert verdict == {"commit_pass": True, "run_roots": ["ab" * 32], "protocols": {"protocols": list(ALL[1:]), "record": rec["digest"]}}
    assert rec["profile_digest"] == p.digest()
    assert rec["protocols"][1] == {"entry": "sampled-proofs:rows-sha512",
                                   "commitment": {"run_roots": ["ab" * 32], "program_digest": "cd" * 32, "manifest_digest": "ef" * 32}}
    assert rec["protocols"][1] == {"entry": "sampled-proofs:rows-sha512",
                                   "commitment": {"run_roots": ["ab" * 32], "program_digest": "cd" * 32, "manifest_digest": "ef" * 32}}


DEFAULT_PATH = """
import json, sys
from verity_vllm import protocols as PO
from verity_vllm.program.frontend.target_profile import TargetProfile

p, model, versions = TargetProfile(), object(), {"engine_kwargs": {}}
assert p.protocols is None and PO.commit_guard(p) is p and PO.at_row(None, {}) is None
protocols = PO.at_commit(p, model, versions, None)
assert protocols is None and versions == {"engine_kwargs": {}}
assert PO.weights_view(protocols, model) is model

class Com:
    def __init__(self):
        self.model = model

    def register_weights(self):
        return {"root": "w"}

com = Com()
assert PO.register_weights(protocols, com) == {"root": "w"} and com.model is model
verdict = {"commit_pass": True, "run_roots": ["ab" * 32], "required_manifest": None}
before = json.dumps(verdict, indent=1)
PO.into_verdict(protocols, verdict, sys.argv[1], p, program_digest="cd" * 32, manifest={"manifest_digest": "ef" * 32})
assert json.dumps(verdict, indent=1) == before
loaded = sorted(m for m in sys.modules if m.startswith("verity_vllm.protocols.") and m != "verity_vllm.protocols.interface")
assert not loaded, loaded
"""


def test_with_no_protocols_the_commit_path_is_a_no_op(tmp_path):
    """With `target.protocols` unset: `at_commit` is None and imports no adapter, `into_verdict` leaves the verdict
    byte-identical and writes nothing, and `weights_view` is the model itself.  In a fresh process, so no other test's
    imports count."""
    subprocess.run([sys.executable, "-c", DEFAULT_PATH, str(tmp_path)], check=True, cwd=config.ROOT)
    assert list(tmp_path.iterdir()) == []


class Params:
    """`named_parameters` / `named_buffers` over plain values, the only reads the replay's weight readers make."""

    def __init__(self, params, buffers=()):
        self.params, self.buffers = dict(params), dict(buffers)

    def named_parameters(self):
        return iter(self.params.items())

    def named_buffers(self):
        return iter(self.buffers.items())


class Committer:
    def __init__(self, model):
        self.model = model

    def register_weights(self):
        return {"root": sorted(list(self.model.named_parameters()) + list(self.model.named_buffers()))}


def test_readers_of_live_weights_see_the_plaintext_an_option_released(monkeypatch):
    def install(scheme, target):
        target.model.params["l0.weight"] = "released"
        return PO.Option(name=f"pous:{scheme}", hooks=target.hooks(f"pous:{scheme}"), decoded=lambda: {"l0.weight": "W0"})

    _fake_adapter(monkeypatch, "pous", install)
    model = Params({"l0.weight": "W0", "l1.weight": "W1"}, {"rope.cos_sin_cache": "cs"})
    com = Committer(model)
    assert PO.weights_view(None, model) is model and PO.register_weights(None, com) == com.register_weights()
    plain = com.register_weights()
    comp = PO.install(model, _serve("pous:s1"))
    assert dict(model.named_parameters())["l0.weight"] == "released" and comp.weights() == {"l0.weight": "W0"}
    view = PO.weights_view(comp, model)
    assert dict(view.named_parameters()) == {"l0.weight": "W0", "l1.weight": "W1"} and dict(view.named_buffers()) == {"rope.cos_sin_cache": "cs"}
    assert PO.register_weights(comp, com) == plain and com.model is model
    with pytest.raises(ValueError, match="name no parameter"):
        PO.WeightsView(model, {"l9.weight": "W9"})
    comp.close()
    assert comp.weights() == {} and PO.weights_view(comp, model) is model


def test_with_pous_on_an_option_keeping_a_resident_weight_copy_is_refused(monkeypatch):
    pouw = _fake_adapter(monkeypatch, "pouw", _pouw([]), schemes=("ncp-v1",))
    pouw.keeps_weight_copy = lambda scheme, ctx: ctx.settings_of("pouw").get("weight_operands", "resident") == "resident"
    _fake_adapter(monkeypatch, "pous", _pous([]), schemes=("band-chain/d12/v1",))
    assert PO.refusals(_serve("pouw:ncp-v1")) == ()
    got = PO.refusals(_serve("pouw:ncp-v1", "pous:band-chain/d12/v1"))
    assert len(got) == 1 and got[0].startswith("pouw:ncp-v1: keeps a weight-derived copy resident across forwards")
    assert PO.refusals(_serve("pouw:ncp-v1", "pous:band-chain/d12/v1", settings={"pouw": {"weight_operands": "per-forward"}})) == ()


@pytest.mark.parametrize("knob,target,world,match", [
    ("pous:s1", {}, 1, "a row is a sampled-proofs run"),
    ("pous:s1,sampled-proofs:rows-sha512", {"protocols": ["pouw:s1", "sampled-proofs:rows-sha512"]}, 1, "contradicts"),
    ("pous:s1,sampled-proofs:rows-sha512", {}, 2, "TP 2"),
    (None, {"protocols": ["pous:s1", "sampled-proofs:rows-sha512"]}, 1, "not on this tree")])
def test_a_row_refuses_a_set_it_cannot_run_before_any_stage(monkeypatch, knob, target, world, match):
    if match != "not on this tree":
        _fake_adapter(monkeypatch, "pous", _pous([]))
    else:
        monkeypatch.setitem(PO.ADAPTERS, "pous", "not_landed_here")
    with pytest.raises(PO.Refused, match=match):
        PO.at_row(knob, {"engine": {"target": target}}, world=world)


def test_a_row_without_protocols_is_the_row_as_before(monkeypatch):
    assert PO.at_row(None, {}) is None and PO.at_row("sampled-proofs:rows-sha512", {"engine": {"target": {"flash_attn_version": 2}}}) is None
    _fake_adapter(monkeypatch, "pous", _pous([]))
    assert PO.at_row("sampled-proofs:rows-sha512,pous:s1", {"engine": {"target": {}}}) == ("pous:s1", "sampled-proofs:rows-sha512")
    with pytest.raises(PO.Refused, match="does not enable"):
        PO.at_row("sampled-proofs:rows-sha512,pous:s1", {"protocols_config": {"pouw": {}}})


def test_run_row_refuses_first_and_hands_the_set_to_the_stages(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    from verity_vllm.config import RowConfig
    from verity_vllm.pipeline import cli
    from verity_vllm.pipeline import row as ROW
    from verity_vllm.pipeline import row_records as R
    from verity_vllm.pipeline import row_stages
    row = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager"
    seen = {}

    class Single:
        def __init__(self, **kw):
            seen.update(kw)

        def run(self):
            return 0

    monkeypatch.setattr(row_stages, "SingleRow", Single)
    o = cli.parse("row", ["run", row, "LLAMA32_1B", "r", "v", "--sweep-dir", str(tmp_path)], {})
    assert ROW.run_row(o, "build") == 0 and seen["protocols"] is None
    eff = json.loads((tmp_path / row / ROW.EFFECTIVE_NAME).read_text())
    assert eff["row"] == row and eff["config"]["engine"] == RowConfig.load(row).engine.to_json() and "requests" not in eff["config"]
    assert eff["run_differs_from_config"] == ["repo", "rev"] and eff["run"]["protocols"] is None
    assert ROW.run_row(SimpleNamespace(**{**vars(o), "protocols": "pous:s1"}), "build") == 3
    assert "REFUSE (fail-closed)" in capsys.readouterr().out
    _fake_adapter(monkeypatch, "pous", _pous([]))
    assert ROW.run_row(SimpleNamespace(**{**vars(o), "protocols": "pous:s1,sampled-proofs:rows-sha512"}), "build") == 0
    assert seen["protocols"] == ("pous:s1", "sampled-proofs:rows-sha512")
    tp = SimpleNamespace(**{**vars(o), "row": row.replace("__tp1__", "__tp2__"), "protocols": "pous:s1,sampled-proofs:rows-sha512"})
    assert ROW.run_row(tp, "build") == 3
    wl = tmp_path / "wl.json"
    wl.write_text(json.dumps({"engine": {"target": {"flash_attn_version": 2}}}))
    assert R.declared_target(str(wl)) == '{"flash_attn_version": 2}'
    assert json.loads(R.declared_target(str(wl), seen["protocols"])) == {"flash_attn_version": 2, "protocols": list(seen["protocols"])}


def test_every_protocol_has_an_adapter_module_name():
    assert tuple(PO.ADAPTERS) == config.PROTOCOLS


def test_every_site_nests_in_install_order():
    for site, outermost_first in PO.SITES.items():
        rank = [config.PROTOCOLS.index(n) for n in outermost_first]
        assert rank == sorted(rank, reverse=True), site


APPLY = "LinearBase.quant_method.apply"


def test_a_site_has_one_executor(monkeypatch):
    pouw = _fake_adapter(monkeypatch, "pouw", _pouw([]), schemes=("ncp-v1", "pearl-fp8-v4"))
    pouw.EXECUTES = (APPLY,)
    pous = _fake_adapter(monkeypatch, "pous", _pous([]), schemes=("band-chain/d12/v1",))
    assert PO.refusals(_serve("pouw:ncp-v1", "pous:band-chain/d12/v1")) == ()
    pous.EXECUTES = (APPLY,)
    assert PO.refusals(_serve("pouw:ncp-v1", "pous:band-chain/d12/v1")) == \
        (f"pous:band-chain/d12/v1: {APPLY} already has an executor, pouw:ncp-v1; a site has one",)


POUW_SCHEMES = ("ncp-v1", "ncp-v1-shift24", "ncp-v2", "pearl-fp8-v4")


def _refused_beside_sampled_proofs(scheme):
    """The refusal of an executor whose scheme has no circuit, word for word."""
    return (f"pouw:{scheme}: executes {APPLY}, and the Program has no Definition for what it computes there, so sampled proofs "
            f"would trace it as the stock computation (refused until the option names one: traced_as)")


def _commit(*enabled):
    return PO.Context(stage="commit", enabled=config.parse_protocols(enabled))


def test_beside_sampled_proofs_pouw_is_admitted_for_ncp_v2_only(monkeypatch):
    """`verity.core.protocols.pouw.circuit` has circuits for `ncp-v2` and `ncp-v2-shift24`, and the adapter executes `ncp-v2` alone:
    that scheme is admitted beside sampled proofs, and every other one keeps its refusal, word for word.  Without sampled
    proofs nothing changes."""
    from verity.core.protocols.pouw import circuit
    assert set(circuit.SCHEMES) == {"ncp-v2", "ncp-v2-shift24"} and PO.CIRCUITS["pouw"] == circuit.__name__
    pouw = _fake_adapter(monkeypatch, "pouw", _pouw([]), schemes=POUW_SCHEMES)
    pouw.EXECUTES = (APPLY,)
    assert PO.refusals(_commit("pouw:ncp-v2", "sampled-proofs:rows-sha512")) == ()
    for scheme in ("ncp-v1", "ncp-v1-shift24", "pearl-fp8-v4"):
        assert PO.refusals(_commit(f"pouw:{scheme}", "sampled-proofs:rows-sha512")) == (_refused_beside_sampled_proofs(scheme),)
    for scheme in POUW_SCHEMES:
        assert PO.refusals(_serve(f"pouw:{scheme}")) == ()
    p = TargetProfile(protocols=("pouw:ncp-v2", "sampled-proofs:rows-sha512"))
    assert PO.commit_guard(p) is p and PO.at_row(None, {"engine": {"target": p.to_json()}}) == p.protocols
    with pytest.raises(SystemExit) as e:
        PO.commit_guard(TargetProfile(protocols=("pouw:ncp-v1", "sampled-proofs:rows-sha512")))
    assert e.value.code == 3
    with pytest.raises(PO.Refused) as r:
        PO.at_row("pouw:ncp-v1,sampled-proofs:rows-sha512", {})
    assert r.value.reasons == (_refused_beside_sampled_proofs("ncp-v1"),)


def test_traced_as_is_the_circuit_of_one_executed_call():
    from verity.core.protocols.pouw.protocol import DomainError
    t = PO.traced_as("pouw:ncp-v2", 16, 64, 16)
    assert t.scheme == "ncp-v2" and t.definition.name == "NcpLinear" and t.served == 16 * 16
    assert PO.traced_as("pouw:ncp-v1", 16, 64, 16) is None and PO.traced_as("pouw:pearl-fp8-v4", 16, 64, 16) is None
    assert PO.traced_as("pous:band-chain/d12/v1", 16, 64, 16) is None and PO.circuit_registry("pous") is None
    with pytest.raises(DomainError, match="64 [|] k"):
        PO.traced_as("pouw:ncp-v2", 16, 65, 16)


def test_without_its_circuit_registry_pouw_stays_refused_beside_sampled_proofs(monkeypatch):
    pouw = _fake_adapter(monkeypatch, "pouw", _pouw([]), schemes=POUW_SCHEMES)
    pouw.EXECUTES = (APPLY,)
    monkeypatch.setitem(PO.CIRCUITS, "pouw", "verity.core.protocols.pouw.not_a_circuit_registry")
    got = PO.refusals(_commit("pouw:ncp-v2", "sampled-proofs:rows-sha512"))
    assert got[-1] == _refused_beside_sampled_proofs("ncp-v2")
    assert got[0].startswith("pouw: its circuit registry verity.core.protocols.pouw.not_a_circuit_registry is not installed")
    with pytest.raises(PO.Refused, match="is not installed"):
        PO.traced_as("pouw:ncp-v2", 16, 64, 16)


def test_every_option_commits_to_the_weights_before_any_installs(monkeypatch):
    log: list[str] = []

    def adapter(name):
        def install(scheme, target):
            log.append(f"install {name} seeing {sorted(target.weight_commitments)}")
            return PO.Option(name=f"{name}:{scheme}", hooks=target.hooks(f"{name}:{scheme}"))
        mod = _fake_adapter(monkeypatch, name, install)
        mod.commit_weights = lambda scheme, target: log.append(f"commit {name}") or {"root": f"{name}-w"}

    adapter("pouw")
    adapter("pous")
    with PO.install(Model(), _serve("pouw:s1", "pous:s1")) as comp:
        pass
    assert log == ["commit pouw", "commit pous", "install pouw seeing ['pous:s1', 'pouw:s1']", "install pous seeing ['pous:s1', 'pouw:s1']"]
    assert [p.get("weights") for p in comp.record()["protocols"]] == [{"root": "pouw-w"}, {"root": "pous-w"}]
