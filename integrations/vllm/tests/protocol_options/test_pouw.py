"""The PoUW adapter (`protocol_options.pouw`): refusals, weights committed before any wrapper, the int7 linear served
exactly, the executor's tiles equal to `verity_pouw`'s reference bit for bit, the weight read from the layer it is handed
(so POUS's decoded weight is what the GEMM uses), and an uninstall that leaves the model as it was.

CPU only, on numpy stand-ins for a linear layer; the hook site is the same ``quant_method.apply`` vLLM's ``LinearBase``
calls.
"""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
from verity_pouw.audit import Epoch, Workload, activation_binding, commit_rows, weight_binding, weights_root
from verity_pouw.protocol import Context as UnitContext
from verity_pouw.protocol import Matmul, Shape, Tile
from verity_pouw.schemes import SCHEMES, get

from verity_vllm import protocol_options as PO
from verity_vllm.engine import hooks
from verity_vllm.protocol_options import pouw

BEACON = "5a" * 32
K, N = 1088, 1600            # inside ncp-v1's domain: 1,088 <= k, 1,600 <= n


class Apply:
    def apply(self, layer, x, bias=None):
        y = np.asarray(x, np.float32) @ np.asarray(layer.weight, np.float32).T
        return y if bias is None else y + bias


class Linear:
    def __init__(self, weight):
        self.weight = weight
        self.quant_method = Apply()


class Model:
    def __init__(self, *weights):
        self.layers = [Linear(w) for w in weights]

    def modules(self):
        return iter(self.layers)


@pytest.fixture(autouse=True)
def _linear_is_the_stand_in(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))


def _rng(seed=0):
    return np.random.default_rng(seed)


def _serve(*enabled, settings=None, **kw):
    return PO.Context(stage="serve", enabled=tuple(enabled), settings=settings or {"pouw": {"beacon": BEACON}}, **kw)


def _int7_linear(x, w, bias=None):
    a, s = pouw.quantize(x)
    b, t = pouw.quantize(w)
    z = a.astype(np.int64) @ b.astype(np.int64).T
    y = (s[:, None] * t[None, :]).astype(np.float32) * z.astype(np.float32)
    return y if bias is None else y + bias


# --- what the adapter refuses ------------------------------------------------------------------------------------------------

def test_schemes_are_the_protocol_packages_and_only_ncp_is_executed():
    assert pouw.schemes() == tuple(sorted(SCHEMES)) and "ncp-v1" in pouw.schemes()
    assert pouw.refusals("ncp-v1", _serve("pouw:ncp-v1")) == ()
    assert PO.refusals(_serve("pouw:ncp-v1")) == ()                           # through the selector, real adapter
    assert "not wired" in pouw.refusals("pearl-fp8-v4", _serve("pouw:pearl-fp8-v4"))[0]


@pytest.mark.parametrize("kw,match", [({"enforce_eager": False}, "graph-capturable"),
                                      ({"settings": {"pouw": {"beacon": "zz"}}}, "hex string"),
                                      ({"settings": {"pouw": {"beacon": "ab"}}}, "at least 32 bytes"),
                                      ({"settings": {"pouw": {"retain_steps": -1}}}, "non-negative"),
                                      ({"settings": {"pouw": {"salt": "x"}}}, "unknown pouw settings")])
def test_a_run_the_adapter_cannot_serve_is_refused(kw, match):
    assert any(match in r for r in pouw.refusals("ncp-v1", _serve("pouw:ncp-v1", **kw)))


def test_pouw_executes_the_linear_apply_and_is_refused_beside_sampled_proofs_until_traced():
    assert pouw.EXECUTES == ("LinearBase.quant_method.apply",) and "pouw" in PO.SITES[pouw.EXECUTES[0]]
    assert pouw.traced_as("ncp-v1", _serve("pouw:ncp-v1")) is None
    commit = PO.Context(stage="commit", enabled=("pouw:ncp-v1", "sampled-proofs:vllm-v1"), settings={"pouw": {"beacon": BEACON}})
    assert any("no Definition" in r for r in PO.refusals(commit))


def test_a_layer_outside_the_domain_is_refused_at_install_and_leaves_nothing():
    model = Model(_rng().standard_normal((64, 64), np.float32))
    before = model.layers[0].quant_method.apply
    with pytest.raises(ValueError, match="refused at install"):
        PO.install(model, _serve("pouw:ncp-v1"))
    assert hooks.live("pouw:ncp-v1") == () and model.layers[0].quant_method.apply == before


# --- install, serve, close ---------------------------------------------------------------------------------------------------

def test_weights_are_committed_before_the_salt_and_the_int7_linear_is_served_exactly(tmp_path):
    rng = _rng(1)
    w0, w1 = rng.standard_normal((N, K), np.float32), rng.standard_normal((N + 64, N), np.float32)
    model = Model(w0, w1)
    settings = {"pouw": {"beacon": BEACON, "transcript_dir": str(tmp_path)}}
    comp = PO.install(model, _serve("pouw:ncp-v1", settings=settings))
    opt = comp.get("pouw")
    roots = {w: commit_rows(weight_binding(w), -2, [r.tobytes() for r in pouw.quantize(m)[0]]).commitment.root
             for w, m in enumerate((w0, w1))}
    declared = Workload((Matmul(0, Shape(16, K, N), 0), Matmul(1, Shape(16, N, N + 64), 1)))
    c = opt.commitment()
    assert comp.weight_commitments["pouw:ncp-v1"]["weight_roots"] == {str(w): r.hex() for w, r in roots.items()}
    assert c["weights_root"] == weights_root(roots).hex()
    assert c["salt"] == Epoch.start(declared, roots, bytes.fromhex(BEACON)).salt.hex()
    assert comp.record()["protocols"][0]["weights"] == comp.weight_commitments["pouw:ncp-v1"]
    assert opt.outside_program and len(hooks.live("pouw:ncp-v1")) == 2

    x, bias = rng.standard_normal((5, K), np.float32), rng.standard_normal(N, np.float32)
    layer = model.layers[0]
    assert np.array_equal(layer.quant_method.apply(layer, x, bias), _int7_linear(x, w0, bias))
    info = opt.info()
    assert (info["units"], info["tiles"], info["padded_rows"]) == (1, N // 16, 11)

    comp.close()
    assert hooks.live("pouw:ncp-v1") == () and "apply" not in vars(layer.quant_method)
    c = opt.commitment()
    assert c["units"] == 1 and c["tiles"] == N // 16 and len(bytes.fromhex(c["transcript_root"])) > 0
    record = json.loads((tmp_path / "pouw-transcript.json").read_text())
    assert record["commitment"] == c and len(record["leaves"]) == N // 16 and record["units"][0]["shape"] == [16, K, N]


def test_the_executor_is_the_schemes_reference_bit_for_bit():
    """`ncp_tiles` gives the digest and output of a tile exactly as `NCP.encode`, `checked` and `digest` do."""
    rng, s = _rng(2), get("ncp-v1")
    a = rng.integers(-63, 64, (16, K)).astype(np.int8)
    b = rng.integers(-63, 64, (N, K)).astype(np.int8)
    u = Matmul(7, Shape(16, K, N), 3)
    rows = [r.tobytes() for r in a]
    ctx = UnitContext(b"\x11" * 32, u, commit_rows(activation_binding(b"\x11" * 32, 7), 7, rows).commitment.root, b"\x22" * 32)
    noise = s.noise(ctx)
    got = pouw.ncp_tiles(s, ctx, noise, a, pouw.Operands(np.ones(N, np.float32), *pouw.stack_y(s, ctx.salt, u.weight, b, noise)))
    tile = Tile(tuple(range(16)), tuple(range(48, 64)))
    ops = s.encode(ctx, noise, rows, [b[j].tobytes() for j in tile.cols], tile)
    checked = s.checked(ctx, ops)
    digest, block = got[(0, 48)]
    assert digest == s.digest(ctx, noise, checked)
    assert block.tolist() == s.useful(ctx, noise, ops, checked)


def test_resident_operands_are_prepared_at_install_and_ignore_the_handed_weight():
    rng = _rng(4)
    w = rng.standard_normal((N, K), np.float32)
    model = Model(w)
    with PO.install(model, _serve("pouw:ncp-v1")) as comp:
        assert comp.get("pouw").info()["weight_operands"] == "resident"
        x = rng.standard_normal((16, K), np.float32)
        other = types.SimpleNamespace(weight=rng.standard_normal((N, K), np.float32))
        assert np.array_equal(model.layers[0].quant_method.apply(other, x), _int7_linear(x, w))


# --- composition -------------------------------------------------------------------------------------------------------------

def _fake_pous(monkeypatch):
    """A POUS stand-in outside PoUW: it stores W + 100 and hands the inner apply a layer holding the decoded weight."""
    mod = types.ModuleType("verity_vllm.protocol_options.fake_pous")
    mod.NAME, mod.schemes, mod.refusals = "pous", lambda: ("band-chain/d12/v1",), lambda scheme, ctx: ()

    def install(scheme, target):
        h = target.hooks(f"pous:{scheme}")
        for m in target.model.modules():
            m.stored = m.weight + 100
            h.wrap(m.quant_method, "apply",
                   lambda orig, m=m: (lambda layer, x, bias=None: orig(types.SimpleNamespace(weight=m.stored - 100), x, bias)))
        return PO.Option(name=f"pous:{scheme}", hooks=h)
    mod.install = install
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setitem(PO.ADAPTERS, "pous", "fake_pous")


def test_keeps_weight_copy_follows_weight_operands():
    assert pouw.keeps_weight_copy("ncp-v1", _serve("pouw:ncp-v1"))
    assert not pouw.keeps_weight_copy("ncp-v1", _serve("pouw:ncp-v1", settings={"pouw": {"weight_operands": "per-forward"}}))
    assert "one of resident, per-forward" in pouw.refusals("ncp-v1", _serve("pouw:ncp-v1", settings={"pouw": {"weight_operands": "x"}}))[0]


def test_the_selector_refuses_pous_beside_resident_operands(monkeypatch):
    _fake_pous(monkeypatch)
    enabled = ("pouw:ncp-v1", "pous:band-chain/d12/v1")
    assert any("keeps a weight-derived copy" in r for r in PO.refusals(_serve(*enabled)))
    assert PO.refusals(_serve(*enabled, settings={"pouw": {"weight_operands": "per-forward"}})) == ()


def test_under_pous_the_gemm_reads_the_weight_it_is_handed(monkeypatch):
    _fake_pous(monkeypatch)
    rng = _rng(3)
    w = rng.standard_normal((N, K), np.float32)
    model = Model(w)
    settings = {"pouw": {"beacon": BEACON, "weight_operands": "per-forward"}}
    with PO.install(model, _serve("pouw:ncp-v1", "pous:band-chain/d12/v1", settings=settings)) as comp:
        assert comp.entries == ("pouw:ncp-v1", "pous:band-chain/d12/v1")
        layer = model.layers[0]
        layer.weight = None                                  # POUS released the plaintext: only its decode can serve
        x = rng.standard_normal((16, K), np.float32)
        assert np.array_equal(layer.quant_method.apply(layer, x), _int7_linear(x, (w + 100) - 100))
    assert hooks.live("pouw:ncp-v1") == () and hooks.live("pous:band-chain/d12/v1") == ()
