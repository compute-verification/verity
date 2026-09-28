"""`protocol_options.pous`: POUS as a protocol option over a toy model, on the CPU.

The model's layers are a stand-in linear type (`pous.linear_types`), their weights byte strings (`pous.TENSORS`), and the
band runs at a 128-byte segment (`pous._base`), so encoding takes milliseconds.  The codec, the commitment, the salt, the
storage root and the audit are `verity.core.protocols.pous`'s own.
"""
from __future__ import annotations

import itertools
import sys
import types

import pytest

from verity_vllm import config
from verity_vllm import protocol_options as PO
from verity_vllm.engine import hooks
from verity_vllm.protocol_options import pous

BAND = "band-chain/d12/v1"
SALT_SOURCE = "00" * 32


class W:
    """A weight: its bytes, emptied when POUS releases the plaintext."""

    def __init__(self, raw: bytes) -> None:
        self.raw = raw


class Bytes:
    """`pous.TENSORS` over `W`: a tied weight is one object."""

    def key(self, w):
        return id(w)

    def meta(self, w):
        return len(w.raw)

    def to_bytes(self, w):
        return w.raw

    def from_bytes(self, raw, meta):
        assert len(raw) == meta
        return W(raw)

    def release(self, param, meta):
        param.raw = b""

    def restore(self, param, tensor):
        param.raw = tensor.raw


class Apply:
    def apply(self, layer, x, bias=None):
        return ("gemm", layer.weight.raw, x)


class Linear:
    def __init__(self, weight: W) -> None:
        self.weight = weight
        self.quant_method = Apply()


class Model:
    """Three linears; the last shares the first's weight, as a tied head does."""

    def __init__(self) -> None:
        a, b = W(bytes(range(40))), W(bytes(range(100, 200)))
        self.layers = {"q": Linear(a), "k": Linear(b), "head": Linear(a)}
        self.plain = {"q": a.raw, "k": b.raw}

    def named_modules(self):
        yield "", self
        yield from self.layers.items()

    def modules(self):
        return (m for _, m in self.named_modules())

    def named_parameters(self):
        return ((f"{n}.weight", m.weight) for n, m in self.layers.items())

    def named_buffers(self):
        return iter(())


@pytest.fixture
def toy(monkeypatch):
    from verity.core.protocols.pous import band
    monkeypatch.setattr(pous, "TENSORS", Bytes())
    monkeypatch.setattr(pous, "linear_types", lambda: (Linear,))
    monkeypatch.setattr(pous, "_base", lambda scheme: band.BandParams(16, 64, 12))
    return Model()


def _ctx(*enabled, settings=None, **kw):
    return PO.Context(stage="serve", enabled=config.parse_protocols(enabled), settings=settings or {}, **kw)


def _ticks(step: int = 1_000):
    counter = itertools.count(0, step)
    return lambda: next(counter)


def test_pous_registers_the_protocol_packages_schemes_and_the_band_is_the_default():
    assert pous.NAME in config.PROTOCOLS and pous.DEFAULT_SCHEME == BAND
    assert {BAND, "dense-chain/v1", "p3/v1", "p2-16448/v2"} <= set(pous.schemes())
    assert PO.adapter("pous") is pous


def test_pous_refuses_what_it_cannot_run():
    assert pous.refusals(BAND, _ctx(f"pous:{BAND}")) == ()
    got = pous.refusals(BAND, _ctx(f"pous:{BAND}", enforce_eager=False, settings={"pous": {"port": 1, "cache": 1}}))
    assert any("unknown pous settings ['cache']" in r for r in got) and any("compiled execution" in r for r in got)
    assert any("experimental" in r for r in pous.refusals("p2-16448/v2", _ctx("pous:p2-16448/v2")))
    assert pous.refusals("p2-16448/v2", _ctx("pous:p2-16448/v2", settings={"pous": {"experimental": True}})) == ()
    assert any("decode_group" in r for r in pous.refusals(BAND, _ctx(f"pous:{BAND}", settings={"pous": {"decode_group": "layers:2"}})))
    assert not hasattr(pous, "keeps_weight_copy")                                       # POUS keeps only C


def test_with_pous_on_the_selector_refuses_an_option_that_keeps_a_weight_copy(monkeypatch):
    _fake_pouw(monkeypatch, lambda scheme, target: None, keeps=True)
    got = PO.refusals(_ctx("pouw:ncp-v1", f"pous:{BAND}"))
    assert any(r.startswith("pouw:ncp-v1: keeps a weight-derived copy") for r in got)
    assert PO.refusals(_ctx("pouw:ncp-v1")) == ()


def test_pous_serves_decoded_weights_releases_the_plaintext_and_passes_its_own_audit(toy):
    from verity.core.protocols.pous import protocol
    before = hooks.live()
    comp = PO.install(toy, _ctx(f"pous:{BAND}", settings={"pous": {"salt_source": SALT_SOURCE}}))
    opt = comp.get("pous")
    for name, layer in toy.layers.items():
        assert layer.weight.raw == b""                                                   # only C is held
        want = toy.plain["q" if name == "head" else name]
        assert layer.quant_method.apply(layer, "x") == ("gemm", want, "x")             # decoded from C on every use
    plain = opt.weights()                                                                # tied weights: encoded once, one tensor
    assert {n: w.raw for n, w in plain.items()} == {"q.weight": toy.plain["q"], "k.weight": toy.plain["k"], "head.weight": toy.plain["q"]}
    assert plain["q.weight"] is plain["head.weight"]
    view = PO.weights_view(comp, toy)                                                    # what the Commit's replay reads
    assert {n: w.raw for n, w in view.named_parameters()} == {n: w.raw for n, w in plain.items()}
    assert [type(e).__name__ for e in hooks.live(f"pous:{BAND}")] == ["Service", "Patch", "Patch", "Patch", "Service"]

    c = opt.commitment()
    s = opt.store
    wc = next(p for p in comp.record()["protocols"] if p["entry"] == f"pous:{BAND}")["weights"]    # phase 1, before any salt
    assert wc["covers"] == "linears" and wc["tensors"] == 2 and wc["bytes"] == 140 and wc["commitment"] == c["commitment"]
    assert c["segments"] == 2
    assert bytes.fromhex(c["salt"]) == protocol.draw_salt(bytes.fromhex(SALT_SOURCE), bytes.fromhex(c["commitment"]))
    vk = protocol.verifier_key(s.scheme, bytes.fromhex(c["commitment"]), s.salt, w=s.layout.pack(toy.plain))
    assert vk.root.hex() == c["storage_root"]                                           # the verifier's own encoding agrees
    v = protocol.TimedAudit(s.scheme, vk, protocol.challenge_key(b"audit", vk), k=20, research=True, clock=_ticks())
    assert v.run(opt.answer).accepted and opt.answered == 20
    lying = protocol.TimedAudit(s.scheme, vk, protocol.challenge_key(b"audit", vk), k=20, research=True, clock=_ticks())
    assert not lying.run(lambda i: bytes(len(s.blocks[i]))).accepted
    assert comp.verdict() == {"protocols": [f"pous:{BAND}"], "record": comp.record()["digest"]}
    assert comp.info()[f"pous:{BAND}"]["salt_source"] == "settings"

    comp.close()
    assert {n: layer.weight.raw for n, layer in toy.layers.items()} == {"q": toy.plain["q"], "k": toy.plain["k"], "head": toy.plain["q"]}
    assert all("apply" not in vars(layer.quant_method) for layer in toy.layers.values()) and hooks.live() == before
    assert opt.weights() is None and PO.weights_view(comp, toy) is toy
    with pytest.raises(RuntimeError, match="closed"):
        opt.answer(0)


def _fake_pouw(monkeypatch, install, keeps=False):
    mod = types.ModuleType("verity_vllm.protocol_options.fake_pouw")
    mod.NAME, mod.schemes, mod.refusals, mod.install = "pouw", lambda: ("ncp-v1",), lambda s, c: (), install
    mod.keeps_weight_copy = lambda scheme, ctx: keeps
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setitem(PO.ADAPTERS, "pouw", "fake_pouw")


def test_pous_wraps_around_pouw_so_pouws_gemm_reads_the_decoded_weight(toy, monkeypatch):
    def pouw_install(scheme, target):
        h = target.hooks(f"pouw:{scheme}")
        for _, m in toy.layers.items():
            h.wrap(m.quant_method, "apply", lambda orig: (lambda layer, x, bias=None: ("pouw", layer.weight.raw, x)))
        return PO.Option(name=f"pouw:{scheme}", hooks=h)

    _fake_pouw(monkeypatch, pouw_install)
    before = hooks.live()
    with PO.install(toy, _ctx("pouw:ncp-v1", f"pous:{BAND}")) as comp:
        k = toy.layers["k"]
        assert comp.entries == ("pouw:ncp-v1", f"pous:{BAND}") and k.weight.raw == b""
        assert k.quant_method.apply(k, "x") == ("pouw", toy.plain["k"], "x")
        assert comp.info()[f"pous:{BAND}"]["salt_source"] == "local-random (placeholder)"
    assert hooks.live() == before and toy.layers["k"].weight.raw == toy.plain["k"]


def test_weights_that_change_after_the_commitment_are_refused_and_leave_nothing(toy, monkeypatch):
    def pouw_install(scheme, target):                                   # an inner option that alters a weight in place
        toy.layers["k"].weight.raw = bytes(100)
        return PO.Option(name=f"pouw:{scheme}", hooks=target.hooks(f"pouw:{scheme}"))

    _fake_pouw(monkeypatch, pouw_install)
    before = hooks.live()
    with pytest.raises(ValueError, match="changed after the composition committed"):
        PO.install(toy, _ctx("pouw:ncp-v1", f"pous:{BAND}"))
    assert hooks.live() == before and toy.layers["q"].weight.raw == toy.plain["q"]


def test_a_model_without_linears_is_refused_at_install_and_leaves_nothing(toy, monkeypatch):
    monkeypatch.setattr(pous, "linear_types", lambda: (Apply,))
    before = hooks.live()
    with pytest.raises(ValueError, match="no Apply layer"):
        PO.install(types.SimpleNamespace(named_modules=lambda: iter([("", object())])), _ctx(f"pous:{BAND}"))
    assert hooks.live() == before


def test_a_quantized_linear_is_refused_at_install_and_leaves_nothing(toy):
    """An FP8 (or any quantized) linear's weight is codes beside a scale that ``C`` doesn't encode, in a layout the
    restored tensor doesn't keep: POUS refuses the model rather than serve a misread weight, and changes nothing."""
    toy.layers["k"].weight_scale = b"\x01"
    before = hooks.live()
    with pytest.raises(ValueError, match=r"refuses quantized linears.*k: a quantized linear \(weight_scale\)"):
        PO.install(toy, _ctx(f"pous:{BAND}"))
    assert hooks.live() == before and toy.layers["k"].weight.raw == toy.plain["k"]
    fp8 = types.SimpleNamespace(weight=types.SimpleNamespace(dtype=types.SimpleNamespace(is_floating_point=True, itemsize=1)))
    assert "FP8 or integer codes" in pous.quantized(fp8)
    assert pous.quantized(toy.layers["q"]) is None
