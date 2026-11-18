"""The PoUW adapter (`protocol_options.pouw`): refusals, a layer refused before any wrapper and the model left as it was,
the operands' tiles equal to `verity.protocols.resource_accounting.compute.pouw.verifier`'s reference bit for bit, and Pearl-C's
weight claim and schedule. Pearl-C's device run is a stand-in (no GPU here).

CPU only, on numpy stand-ins for a linear layer; the hook site is the same ``quant_method.apply`` vLLM's ``LinearBase``
calls.
"""
from __future__ import annotations

import hashlib
import random
import sys
import types

import numpy as np
import pytest
from verity.protocols.resource_accounting.compute.pouw.prover import circuit
from verity.protocols.resource_accounting.compute.pouw.prover.schemes import get, pearl_c
from verity.protocols.resource_accounting.compute.pouw.verifier.kernel_schemes import IDENTIFIED
from verity.protocols.resource_accounting.compute.pouw.prover.serving import h1 as pouw_h1
from verity.protocols.resource_accounting.compute.pouw.prover.serving import operands as OPS
from verity.protocols.resource_accounting.compute.pouw.prover.serving import PEARL_C_SCHEMES, pearl_c_scheme
from verity.protocols.resource_accounting.compute.pouw.prover.serving import pearl_c as PPD
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c import kernel as PPK
from verity.protocols.resource_accounting.compute.pouw.verifier import registration
from verity.protocols.resource_accounting.compute.pouw.verifier.audit import (
    HM96,
    Epoch,
    Workload,
    activation_binding,
    fresh_salts,
    commit_rows,
    weights_root,
)
from verity.protocols.resource_accounting.compute.pouw.verifier.coins import (
    Challenger,
    CoinOpening,
    coin_reason,
)
from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import Context as UnitContext
from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import (
    Matmul,
    Shape,
    Tile,
)
from verity_torch.pouw.pearl_c import run_device as PPDR

from verity_vllm import config
from verity_vllm import protocols as PO
from verity_vllm.engine import hooks
from verity_vllm.protocols import pouw

K, N = 1024, 128             # inside Pearl-C's kernels
PEARL = "pearl-c-sm120-v1-h3"
ON = {"device": "cuda", "kernel": "k"}


class Apply:
    def apply(self, layer, x, bias=None):
        y = np.asarray(x, np.float32) @ np.asarray(layer.weight, np.float32).T
        return y if bias is None else y + bias


class Linear:
    #: a scaled layer's per-row scales; a layer made plain has none
    weight_scale: np.ndarray

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


def _coins(seed=0):
    """The epoch's challenger on a seeded entropy (a served one's is the Lean verifier's, `pouw.verifier.coins`)."""
    return Challenger(0, random.Random(seed).randbytes)


def _serve(*enabled, settings=None, **kw):
    return PO.Context(stage="serve", enabled=tuple(enabled), settings=settings or {"pouw": {"coins": _coins()}}, **kw)


def _pearl(**pouw_settings):
    return _serve(f"pouw:{PEARL}", settings={"pouw": {"coins": _coins(), **ON, **pouw_settings}})


def _device_stand_in(monkeypatch, roots=None):
    """Pearl-C's kernel, its weight roots and its device run as stand-ins; the runs started, each with its epoch."""
    started = []

    class Run:
        def __init__(self, s, entry, kernel, weights, roots, coins, opening, **kw):
            started.append(Epoch.start(Workload((Matmul(0, Shape(16, K, N), 0),)), roots, coins, opening))
            self.commits = {"scheme": s.name, "salt": started[-1].salt.hex(), "coins": coins.to_json(),
                            "salt_opening": opening.to_json()}
            self.details = {}

        def gemm(self, w, x, weight, bias):
            return x

        def end_pass(self):
            return {}

        def finish(self):
            pass

    monkeypatch.setattr(PPD.Kernel, "load", staticmethod(lambda *a: None))
    monkeypatch.setattr(PPDR, "weight_roots", lambda kernel, weights, scheme, *, weight_salts: roots or {w: b"\x05" * 64 for w in weights})
    monkeypatch.setattr(pouw, "pearl_c_weight_roots", PPDR.weight_roots)
    monkeypatch.setattr(PPDR, "PearlCDeviceRun", Run)
    monkeypatch.setattr(pouw, "PearlCDeviceRun", Run)
    return started


# --- what the adapter refuses ------------------------------------------------------------------------------------------------

def test_schemes_are_the_protocol_packages_and_only_pearl_c_and_h1_are_executed():
    assert pouw.schemes() == tuple(sorted({*IDENTIFIED, *circuit.SCHEMES, *PEARL_C_SCHEMES, pouw_h1.SCHEME}))
    assert pouw.EXECUTED == (*PEARL_C_SCHEMES, pouw_h1.SCHEME)
    for scheme in ("ncp-v1", "ncp-v1-shift24", "ncp-v2", "pearl-fp8-v4"):
        assert scheme in pouw.schemes() and "executor runs" in pouw.refusals(scheme, _serve(f"pouw:{scheme}"))[0], scheme
    assert PO.refusals(_pearl()) == ()                                         # through the selector, real adapter


def test_pearl_c_sm120_runs_on_the_device_only_and_not_beside_spot():
    scheme = "pearl-c-sm120-v1-h3"
    assert scheme in pouw.EXECUTED
    assert any("device only" in r for r in pouw.refusals(scheme, _serve(f"pouw:{scheme}")))
    on = {"pouw": {"device": "cuda", "kernel": "verity/protocols/resource_accounting/compute/pouw/prover/kernels/pearl_c_sm120"}}
    assert pouw.refusals(scheme, _serve(f"pouw:{scheme}", settings=on)) == ()
    assert any("path strings" in r for r in pouw.refusals(scheme, _serve(f"pouw:{scheme}", settings={"pouw": {"kernel": 3}})))
    commit = PO.Context(stage="commit", enabled=(f"pouw:{scheme}", "sampled-proofs:rows-sha512"), settings=on)
    assert any("no Definition" in r for r in PO.refusals(commit))


def test_pearl_c_sm120_refuses_linears_outside_the_kernels():
    assert PPD.refusals("pearl-c-sm120-v1-h3", {0: (4096, 4096), 1: (6144, 4096), 2: (4096, 14336), 3: (28672, 4096)}) == []
    bad = PPD.refusals("pearl-c-sm120-v1-h3", {0: (3584, 3584), 1: (4096, 896), 2: (4160, 4096)})
    assert [r.split(":")[0] for r in bad] == ["linear 0", "linear 1", "linear 2"] and "1,024" in bad[0]


@pytest.mark.parametrize("kw,match", [({"enforce_eager": False}, "graph-capturable"),
                                      ({"settings": {"pouw": {"coins": "5a" * 32}}}, "the epoch's challenger"),
                                      ({"settings": {"pouw": {"beacon": "5a" * 32}}}, "unknown pouw settings"),
                                      ({"settings": {"pouw": {"retain_steps": -1}}}, "non-negative"),
                                      ({"settings": {"pouw": {"salt": "x"}}}, "unknown pouw settings"),
                                      ({"settings": {"pouw": {"weight_operands": "resident"}}}, "unknown pouw settings")])
def test_a_run_the_adapter_cannot_serve_is_refused(kw, match):
    assert any(match in r for r in PO.refusals(_serve(f"pouw:{PEARL}", **kw)))


def test_pouw_executes_the_linear_apply():
    assert pouw.EXECUTES == ("LinearBase.quant_method.apply",) and "pouw" in PO.SITES[pouw.EXECUTES[0]]
    assert not hasattr(pouw, "traced_as")


def test_a_layer_outside_the_domain_is_refused_at_install_and_leaves_nothing():
    model = Model(_rng().standard_normal((64, 64), np.float32))
    before = model.layers[0].quant_method.apply
    with pytest.raises(ValueError, match="refused at install"):
        PO.install(model, _pearl())
    assert hooks.live(f"pouw:{PEARL}") == () and model.layers[0].quant_method.apply == before



def test_an_install_without_the_epochs_challenger_is_refused_and_leaves_nothing(monkeypatch):
    """There is no beacon and no fallback salt: without ``coins`` install refuses before anything is patched."""
    _device_stand_in(monkeypatch)
    model = Model(_rng().standard_normal((N, K), np.float32))
    before = model.layers[0].quant_method.apply
    with pytest.raises(ValueError, match="no 'coins' setting"):
        PO.install(model, _serve(f"pouw:{PEARL}", settings={"pouw": {"retain_steps": 0, **ON}}))
    assert hooks.live(f"pouw:{PEARL}") == () and model.layers[0].quant_method.apply == before


@pytest.mark.parametrize("quantized", ["float8_e4m3fn", "int8"])
def test_a_quantized_layer_is_refused_at_install_not_misread(quantized):
    """The executor reads an unquantized weight; a layer served quantized (an FP8 checkpoint's float8, an int8
    one) is refused by name before anything is patched, never read as if it were the model's weight."""
    w = _rng(1).standard_normal((N, K), np.float32)
    if quantized == "int8":
        q = np.clip(np.rint(w * 16), -127, 127).astype(np.int8)
    else:
        torch = pytest.importorskip("torch")
        q = torch.from_numpy(w).to(getattr(torch, quantized))
    model = Model(_rng(2).standard_normal((N, K), np.float32), q)
    before = [layer.quant_method.apply for layer in model.layers]
    with pytest.raises(ValueError, match=rf"linear 1: its weight is {quantized}, .*refused, not misread"):
        PO.install(model, _pearl())
    assert hooks.live(f"pouw:{PEARL}") == () and [layer.quant_method.apply for layer in model.layers] == before


class _Packed(Linear):
    """A stand-in for compressed-tensors' packed linear: ``weight_packed`` and ``weight_scale``, and no ``weight``."""

    def __init__(self, packed, scale):
        self.weight_packed, self.weight_scale, self.quant_method = packed, scale, Apply()


def _scaled(r):
    layer = Linear(r.standard_normal((N, K), np.float32))
    layer.weight_scale = np.ones(N, np.float32)
    return layer


@pytest.mark.parametrize("layer", [_scaled, lambda r: _Packed(r.integers(0, 1 << 31, (N, K // 8)).astype(np.int32),
                                                              np.ones(N, np.float32))],
                         ids=["float-weight-with-scale", "packed-without-weight"])
def test_a_layer_carrying_quantization_parameters_is_refused_whatever_its_weight(layer):
    """A scale beside a float weight, or a packed layer with no ``weight`` to read, is refused by its parameters, at the
    commitment and at install, and the model is left as it was."""
    model = Model(_rng().standard_normal((N, K), np.float32))
    model.layers.append(layer(_rng(3)))
    before = [m.quant_method.apply for m in model.layers]
    with pytest.raises(ValueError, match=r"linear 1: it carries weight_\w+.*refused, not misread"):
        pouw.commit_weights(PEARL, types.SimpleNamespace(model=model))
    with pytest.raises(ValueError, match="refused at install"):
        PO.install(model, _pearl())
    assert hooks.live(f"pouw:{PEARL}") == () and [m.quant_method.apply for m in model.layers] == before

# --- install, serve, close ---------------------------------------------------------------------------------------------------

def test_the_executor_is_the_schemes_reference_bit_for_bit():
    """`ncp_tiles` gives the digest and output of a tile exactly as `NCP.encode`, `checked` and `digest` do."""
    rng, s = _rng(2), get("ncp-v1")
    a = rng.integers(-63, 64, (16, K)).astype(np.int8)
    b = rng.integers(-63, 64, (N, K)).astype(np.int8)
    u = Matmul(7, Shape(16, K, N), 3)
    rows = [r.tobytes() for r in a]
    ctx = UnitContext(b"\x11" * 32, u, commit_rows(activation_binding(b"\x11" * 32, 7), 7, rows, HM96, fresh_salts(len(rows))).commitment.root, b"\x22" * 32)
    noise = s.noise(ctx)
    got = OPS.ncp_tiles(s, ctx, noise, a, OPS.Operands(np.ones(N, np.float32), *OPS.stack_y(s, ctx.salt, u.weight, b, noise)))
    tile = Tile(tuple(range(16)), tuple(range(48, 64)))
    ops = s.encode(ctx, noise, rows, [b[j].tobytes() for j in tile.cols], tile)
    checked = s.checked(ctx, ops)
    digest, block = got[(0, 48)]
    assert digest == s.digest(ctx, noise, checked)
    assert block.tolist() == s.useful(ctx, noise, ops, checked)


# --- composition -------------------------------------------------------------------------------------------------------------

def _fake_pous(monkeypatch):
    """A POUS stand-in outside PoUW: it stores W + 100 and hands the inner apply a layer holding the decoded weight."""
    mod = types.ModuleType("verity_vllm.protocols.fake_pous")
    vars(mod).update(NAME="pous", schemes=lambda: ("band-chain/d12/v1",), refusals=lambda scheme, ctx: ())

    def install(scheme, target):
        h = target.hooks(f"pous:{scheme}")
        for m in target.model.modules():
            m.stored = m.weight + 100
            h.wrap(m.quant_method, "apply",
                   lambda orig, m=m: (lambda layer, x, bias=None: orig(types.SimpleNamespace(weight=m.stored - 100), x, bias)))
        return PO.Option(name=f"pous:{scheme}", hooks=h)
    vars(mod)["install"] = install
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setitem(config.PROTOCOLS, "pous", "fake_pous")


def test_a_scheme_that_keeps_weight_state_is_refused_beside_pous(monkeypatch):
    """Pearl-C forms B~ once per install: under POUS its GEMMs would read a copy made from the plaintext, not what POUS
    decodes."""
    scheme = PEARL
    _fake_pous(monkeypatch)
    settings = {"pouw": {"coins": _coins(), **ON}}
    assert pouw.keeps_weight_copy(scheme)
    assert PO.refusals(_serve(f"pouw:{scheme}", settings=settings)) == ()
    got = PO.refusals(_serve(f"pouw:{scheme}", "pous:band-chain/d12/v1", settings=settings))
    assert [r for r in got if "weight-derived copy" in r] == [
        (f"pouw:{scheme}: keeps a weight-derived copy resident across forwards, so with pous on its GEMMs would not read what "
         "POUS stores (prepare its weight operands per forward from the weight it is handed)")]


def test_kernel_directories_load_as_their_own_modules_named_by_their_sources(tmp_path, monkeypatch):
    """Two kernel directories in one process (a tuned variant beside the gated one) get their own
    run.py; the same bytes anywhere are one kernel.  Before this, the second got the first's run.py.  The loader executes
    no fixture.py, and the digest covers grouped.py where the directory has one."""
    def kernel_dir(name, mark, grouped=False):
        d = tmp_path / name
        d.mkdir()
        (d / "run.py").write_text(f"MARK = {mark!r}\n\nclass Dev:\n    def __init__(self, raw):\n        self.raw = raw\n\n"
                                  "    def open(self):\n        self.opened = True\n")
        (d / "fixture.py").write_text("raise RuntimeError('the loader executed fixture.py')\n")
        if grouped:
            (d / "grouped.py").write_text("GROUPED = 1\n")
        (d / "pearl_c_sm120.cubin").write_bytes(b"cubin " + mark.encode())
        return d

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    monkeypatch.setattr(PPK, "_KERNELS", {})
    before = set(sys.modules)
    try:
        a, b, a2 = kernel_dir("a", "gated"), kernel_dir("b", "tuned", grouped=True), kernel_dir("a2", "gated")
        ka, kb, ka2 = (PPD.Kernel.load(str(d)) for d in (a, b, a2))
        assert (ka.run.MARK, kb.run.MARK) == ("gated", "tuned") and ka.run is not kb.run and ka.dev.opened
        assert ka2 is ka
        assert ka.files == {"run.py": sha(a / "run.py")}
        assert kb.files == {"run.py": sha(b / "run.py"), "grouped.py": sha(b / "grouped.py")}
        assert ka.source_sha256 == PPD.source_digest(ka.files) != kb.source_sha256 == PPD.source_digest(kb.files)
        assert ka.cubin_sha256 == sha(a / "pearl_c_sm120.cubin")
        assert PPD.Kernel.load(str(a), str(b / "pearl_c_sm120.cubin")).cubin_sha256 == kb.cubin_sha256
    finally:
        for name in set(sys.modules) - before:
            del sys.modules[name]


def test_served_fp8_claims_registered_weights_only_under_the_shared_rule(monkeypatch):
    """Served FP8 applies no keyed rotation, so its weights register only from the curated list, through the one
    registered-weights rule Pearl-C4 reads too; refused, the claim says it is about nothing and names why."""
    s = pearl_c_scheme("pearl-c-sm120-v1-h3")
    seen: list[tuple] = []
    assert pouw.SERVED_TRANSFORMS == () and pearl_c.check_registration is registration.check_registration

    def check_registration(*a):
        seen.append(a)
        return registration.check_registration(*a)

    monkeypatch.setattr(pearl_c, "check_registration", check_registration)
    claim = pouw.weights_claim(s)
    assert seen == [((), False, False)]
    assert claim == {"transforms": [], "curated": False, "registered": False, "claims": pouw.UNREGISTERED,
                     "refused": claim["refused"]} and "must be on the curated list" in claim["refused"]
    assert pouw.weights_claim(s, curated=True) == {"transforms": [], "curated": True, "registered": True}
    folded = pouw.weights_claim(s, curated=True, needs_gamma_fold=True)
    assert not folded["registered"] and "only with γ folded" in folded["refused"]


def test_an_unregistered_served_fp8_checkpoint_is_served_and_its_commitment_claims_nothing_about_the_weights(monkeypatch):
    """Install doesn't raise on an unregistered checkpoint: the epoch's salt is still the challenger's salt coin, opened
    after the weights' root, and the commitment's registration says it claims nothing about which weights were served.
    The device run is a stand-in (no GPU here)."""
    scheme = PEARL
    roots = {0: b"\x05" * 64}
    started = _device_stand_in(monkeypatch, roots)
    model, coins = Model(_rng().standard_normal((128, 1024), np.float32)), _coins()
    settings = {"pouw": {"coins": coins, "device": "cuda", "kernel": "k"}}
    with PO.install(model, _serve(f"pouw:{scheme}", settings=settings)) as comp:
        c = comp.get("pouw").commitment()
        assert c["registration"]["registered"] is False and c["registration"]["claims"] == pouw.UNREGISTERED
        assert c["salt"] == started[0].salt.hex() and started[0].coins == coins.commitment
        assert coin_reason(coins.commitment, CoinOpening.from_json(c["salt_opening"]), "salt", weights_root(roots)) is None


def test_pearl_cs_schedule_is_a_setting_checked_against_the_scheme():
    scheme, on = "pearl-c-sm120-v1-h3", {"coins": _coins(), "device": "cuda", "kernel": "k"}
    for name in ("serial", "deferred", "deferred-caller"):
        assert pouw.refusals(scheme, _serve(f"pouw:{scheme}", settings={"pouw": {**on, "schedule": name}})) == ()
    assert any("schedule is one of" in r for r in pouw.refusals(scheme, _serve(f"pouw:{scheme}", settings={"pouw": {**on, "schedule": "h3o"}})))
    h1 = pouw_h1.SCHEME
    assert any("Pearl-C's; this executor runs serially" in r
               for r in pouw.refusals(h1, _serve(f"pouw:{h1}", settings={"pouw": {"coins": _coins(), "schedule": "deferred"}})))
