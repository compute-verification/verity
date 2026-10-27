"""The PoUW adapter (`protocol_options.pouw`): refusals, weights committed before any wrapper, the int7 linear served
exactly, the executor's tiles equal to `verity.core.protocols.pouw`'s reference bit for bit, the weight read from the layer it is handed
(so POUS's decoded weight is what the GEMM uses), and an uninstall that leaves the model as it was.

CPU only, on numpy stand-ins for a linear layer; the hook site is the same ``quant_method.apply`` vLLM's ``LinearBase``
calls.
"""
from __future__ import annotations

import json
import random
import sys
import types

import numpy as np
import pytest
from verity.core.protocols.pouw.audit import Epoch, Workload, activation_binding, commit_rows, weight_binding, weights_root
from verity.core.protocols.pouw.coins import Challenger, CoinCommitment, CoinOpening, coin_reason
from verity.core.protocols.pouw.protocol import Context as UnitContext
from verity.core.protocols.pouw.protocol import DomainError, Matmul, Shape, Tile
from verity.core.protocols.pouw.schemes import SCHEMES, get
from verity_vllm import protocols as PO
from verity_vllm.engine import hooks
from verity_vllm.protocols import pouw
from verity_vllm.protocols.pouw import pearl_c as PPD
from verity_vllm.protocols.pouw.pearl_c import kernel as PPK

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


def _coins(seed=0):
    """The epoch's challenger on a seeded entropy (a served one's is the Lean verifier of record's, `pouw.verifier.coins`)."""
    return Challenger(0, random.Random(seed).randbytes)


def _serve(*enabled, settings=None, **kw):
    return PO.Context(stage="serve", enabled=tuple(enabled), settings=settings or {"pouw": {"coins": _coins()}}, **kw)


def _int7_linear(x, w, bias=None):
    a, s = pouw.quantize(x)
    b, t = pouw.quantize(w)
    z = a.astype(np.int64) @ b.astype(np.int64).T
    y = (s[:, None] * t[None, :]).astype(np.float32) * z.astype(np.float32)
    return y if bias is None else y + bias


# --- what the adapter refuses ------------------------------------------------------------------------------------------------

def test_schemes_are_the_protocol_packages_and_only_ncp_pearl_c_sm120_and_h1_are_executed():
    from verity.core.protocols.pouw import circuit
    from verity_vllm.protocols.pouw import h1 as pouw_h1
    assert pouw.schemes() == tuple(sorted({*SCHEMES, *circuit.SCHEMES, *PPD.SCHEMES, pouw_h1.SCHEME})) and "ncp-v1" in pouw.schemes()
    assert pouw.refusals("ncp-v1", _serve("pouw:ncp-v1")) == () and pouw.refusals("ncp-v2", _serve("pouw:ncp-v2")) == ()
    assert PO.refusals(_serve("pouw:ncp-v1")) == ()                           # through the selector, real adapter
    assert "executor runs" in pouw.refusals("pearl-fp8-v4", _serve("pouw:pearl-fp8-v4"))[0]


def test_pearl_c_sm120_runs_on_the_device_only_and_not_beside_sampled_proofs():
    scheme = "pearl-c-sm120-v1-h3"
    assert scheme in pouw.EXECUTED
    assert any("device only" in r for r in pouw.refusals(scheme, _serve(f"pouw:{scheme}")))
    on = {"pouw": {"device": "cuda", "kernel": "benchmarks/pouw/pearl_c_sm120"}}
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
    assert any(match in r for r in pouw.refusals("ncp-v1", _serve("pouw:ncp-v1", **kw)))


def test_pouw_executes_the_linear_apply_and_only_ncp_v2_is_admitted_beside_sampled_proofs():
    assert pouw.EXECUTES == ("LinearBase.quant_method.apply",) and "pouw" in PO.SITES[pouw.EXECUTES[0]]
    assert not hasattr(pouw, "traced_as") and "ncp-v2" in pouw.schemes() and "ncp-v2" in pouw.EXECUTED
    for scheme in ("ncp-v1", "ncp-v1-shift24", "pearl-fp8-v4"):
        commit = PO.Context(stage="commit", enabled=(f"pouw:{scheme}", "sampled-proofs:rows-sha512"), settings={"pouw": {"coins": _coins()}})
        assert any("no Definition" in r for r in PO.refusals(commit)), scheme
    commit = PO.Context(stage="commit", enabled=("pouw:ncp-v2", "sampled-proofs:rows-sha512"), settings={"pouw": {"coins": _coins()}})
    assert PO.refusals(commit) == ()


def test_a_layer_outside_the_domain_is_refused_at_install_and_leaves_nothing():
    model = Model(_rng().standard_normal((64, 64), np.float32))
    before = model.layers[0].quant_method.apply
    with pytest.raises(ValueError, match="refused at install"):
        PO.install(model, _serve("pouw:ncp-v1"))
    assert hooks.live("pouw:ncp-v1") == () and model.layers[0].quant_method.apply == before



def test_an_install_without_the_epochs_challenger_is_refused_and_leaves_nothing():
    """There is no beacon and no fallback salt: without ``coins`` install refuses before anything is patched."""
    model = Model(_rng().standard_normal((N, K), np.float32))
    before = model.layers[0].quant_method.apply
    with pytest.raises(ValueError, match="no 'coins' setting"):
        PO.install(model, _serve("pouw:ncp-v1", settings={"pouw": {"retain_steps": 0}}))
    assert hooks.live("pouw:ncp-v1") == () and model.layers[0].quant_method.apply == before


@pytest.mark.parametrize("quantized", ["float8_e4m3fn", "int8"])
def test_a_quantized_layer_is_refused_at_install_not_misread(quantized):
    """The executor quantizes an unquantized weight itself; a layer served quantized (an FP8 checkpoint's float8, an int8
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
        PO.install(model, _serve("pouw:ncp-v1"))
    assert hooks.live("pouw:ncp-v1") == () and [layer.quant_method.apply for layer in model.layers] == before


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
        pouw.commit_weights("ncp-v1", types.SimpleNamespace(model=model))
    with pytest.raises(ValueError, match="refused at install"):
        PO.install(model, _serve("pouw:ncp-v1"))
    assert hooks.live("pouw:ncp-v1") == () and [m.quant_method.apply for m in model.layers] == before

# --- install, serve, close ---------------------------------------------------------------------------------------------------

def test_weights_are_committed_before_the_salt_and_the_int7_linear_is_served_exactly(tmp_path):
    rng = _rng(1)
    w0, w1 = rng.standard_normal((N, K), np.float32), rng.standard_normal((N + 64, N), np.float32)
    model = Model(w0, w1)
    coins = _coins()
    settings = {"pouw": {"coins": coins, "transcript_dir": str(tmp_path)}}
    comp = PO.install(model, _serve("pouw:ncp-v1", settings=settings))
    opt = comp.get("pouw")
    roots = {w: commit_rows(weight_binding(w), -2, [r.tobytes() for r in pouw.quantize(m)[0]]).commitment.root
             for w, m in enumerate((w0, w1))}
    declared = Workload((Matmul(0, Shape(16, K, N), 0), Matmul(1, Shape(16, N, N + 64), 1)))
    c = opt.commitment()
    assert comp.weight_commitments["pouw:ncp-v1"]["weight_roots"] == {str(w): r.hex() for w, r in roots.items()}
    assert c["weights_root"] == weights_root(roots).hex()
    opening = CoinOpening.from_json(c["salt_opening"])
    assert CoinCommitment.from_json(c["coins"]) == coins.commitment
    assert coin_reason(coins.commitment, opening, "salt", weights_root(roots)) is None
    assert c["salt"] == Epoch.start(declared, roots, coins.commitment, opening).salt.hex()
    with pytest.raises(DomainError):                         # install opened the salt coin, and a coin opens once
        coins.open("salt", weights_root(roots))
    assert comp.record()["protocols"][0]["weights"] == comp.weight_commitments["pouw:ncp-v1"]
    assert len(hooks.live("pouw:ncp-v1")) == 2

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


def test_weight_operands_formed_in_blocks_equal_one_block(monkeypatch):
    rng, s = _rng(5), get("ncp-v1")
    w = rng.standard_normal((N, K), np.float32)
    ctx = UnitContext(b"\x33" * 32, Matmul(0, Shape(16, K, N), 2), b"", b"\x44" * 32)
    noise = s.noise(ctx)
    whole = pouw.weight_operands(s, ctx.salt, 2, w, noise)
    monkeypatch.setattr(pouw, "FORM_BYTES", 128 * K * 7)                  # 7 rows a block, the last one short
    blocks = pouw.weight_operands(s, ctx.salt, 2, w, noise)
    for got, want in ((blocks.scales, whole.scales), (blocks.y, whole.y), (blocks.c0, whole.c0)):
        assert got.dtype == want.dtype and np.array_equal(got, want)
    b, _ = pouw.quantize(w)
    assert np.array_equal(whole.y, pouw.stack_y(s, ctx.salt, 2, b, noise)[0])


# --- composition -------------------------------------------------------------------------------------------------------------

def _fake_pous(monkeypatch):
    """A POUS stand-in outside PoUW: it stores W + 100 and hands the inner apply a layer holding the decoded weight."""
    mod = types.ModuleType("verity_vllm.protocols.fake_pous")
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


def test_ncp_v1_keeps_no_weight_copy_so_the_selector_admits_pous_beside_it(monkeypatch):
    _fake_pous(monkeypatch)
    assert not pouw.keeps_weight_copy("ncp-v1", _serve("pouw:ncp-v1"))
    assert PO.refusals(_serve("pouw:ncp-v1", "pous:band-chain/d12/v1")) == ()


@pytest.mark.parametrize("scheme", ["pearl-c-sm120-v1-h3", "ncp-v2"])
def test_a_scheme_that_keeps_weight_state_is_refused_beside_pous(monkeypatch, scheme):
    """Pearl-C forms B~ once per install, and ncp-v2 reuses each weight's side: under POUS their GEMMs would read a copy made
    from the plaintext, not what POUS decodes."""
    _fake_pous(monkeypatch)
    settings = {"pouw": {"coins": _coins(), **({"device": "cuda", "kernel": "k"} if scheme.startswith("pearl") else {})}}
    assert pouw.keeps_weight_copy(scheme, _serve(f"pouw:{scheme}", settings=settings))
    assert PO.refusals(_serve(f"pouw:{scheme}", settings=settings)) == ()
    got = PO.refusals(_serve(f"pouw:{scheme}", "pous:band-chain/d12/v1", settings=settings))
    assert [r for r in got if "weight-derived copy" in r] == [
        f"pouw:{scheme}: keeps a weight-derived copy resident across forwards, so with pous on its GEMMs would not read what "
        "POUS stores (prepare its weight operands per forward from the weight it is handed)"]


def test_kernel_directories_load_as_their_own_modules_named_by_their_sources(tmp_path, monkeypatch):
    """Two kernel directories in one process (a tuned variant beside the gated one) get their own
    run.py and fixture.py; the same bytes anywhere are one kernel.  Before this, the second got the first's run.py."""
    import hashlib

    def kernel_dir(name, mark):
        d = tmp_path / name
        d.mkdir()
        (d / "run.py").write_text(f"MARK = {mark!r}\n\nclass Dev:\n    def __init__(self, raw):\n        self.raw = raw\n\n"
                                  "    def open(self):\n        self.opened = True\n")
        (d / "fixture.py").write_text("FIX = 1\n")
        (d / "pearl_c_sm120.cubin").write_bytes(b"cubin " + mark.encode())
        return d

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    monkeypatch.setattr(PPK, "_KERNELS", {})
    before = set(sys.modules)
    try:
        a, b, a2 = kernel_dir("a", "gated"), kernel_dir("b", "tuned"), kernel_dir("a2", "gated")
        ka, kb, ka2 = (PPD.Kernel.load(str(d)) for d in (a, b, a2))
        assert (ka.run.MARK, kb.run.MARK) == ("gated", "tuned") and ka.run is not kb.run and ka.dev.opened
        assert ka2 is ka and ka.fixture is kb.fixture
        assert ka.source_sha256 == PPD.source_digest(sha(a / "run.py"), sha(a / "fixture.py")) != kb.source_sha256
        assert ka.cubin_sha256 == sha(a / "pearl_c_sm120.cubin")
        assert PPD.Kernel.load(str(a), str(b / "pearl_c_sm120.cubin")).cubin_sha256 == kb.cubin_sha256
    finally:
        for name in set(sys.modules) - before:
            del sys.modules[name]


def test_under_pous_the_gemm_reads_the_weight_it_is_handed(monkeypatch):
    _fake_pous(monkeypatch)
    rng = _rng(3)
    w = rng.standard_normal((N, K), np.float32)
    model = Model(w)
    settings = {"pouw": {"coins": _coins()}}
    with PO.install(model, _serve("pouw:ncp-v1", "pous:band-chain/d12/v1", settings=settings)) as comp:
        assert comp.entries == ("pouw:ncp-v1", "pous:band-chain/d12/v1")
        layer = model.layers[0]
        layer.weight = None                                  # POUS released the plaintext: only its decode can serve
        x = rng.standard_normal((16, K), np.float32)
        assert np.array_equal(layer.quant_method.apply(layer, x), _int7_linear(x, (w + 100) - 100))
    assert hooks.live("pouw:ncp-v1") == () and hooks.live("pous:band-chain/d12/v1") == ()


def test_served_fp8_claims_registered_weights_only_under_the_shared_rule(monkeypatch):
    """Served FP8 applies no keyed rotation, so its weights register only from the curated list, through the one
    registered-weights rule Pearl-C4 reads too; refused, the claim says it is about nothing and names why."""
    from verity.core.protocols.pouw import registration
    from verity.core.protocols.pouw.schemes import pearl_c
    s, seen = PPD.scheme_of("pearl-c-sm120-v1-h3"), []
    assert pouw.SERVED_TRANSFORMS == () and pearl_c.check_registration is registration.check_registration
    monkeypatch.setattr(pearl_c, "check_registration", lambda *a: seen.append(a) or registration.check_registration(*a))
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
    scheme = "pearl-c-sm120-v1-h3"
    roots = {0: b"\x05" * 64}
    started = []

    class Run:
        def __init__(self, s, entry, kernel, weights, roots, coins, opening, **kw):
            started.append(Epoch.start(Workload((Matmul(0, Shape(16, 1024, 128), 0),)), roots, coins, opening))
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
    monkeypatch.setattr(PPD, "weight_roots", lambda kernel, weights, scheme, salt_key=None: roots)
    monkeypatch.setattr(PPD, "PearlCDeviceRun", Run)
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
    assert any("'schedule' is one of" in r for r in pouw.refusals(scheme, _serve(f"pouw:{scheme}", settings={"pouw": {**on, "schedule": "h3o"}})))
    assert any("Pearl-C's; this executor runs serially" in r
               for r in pouw.refusals("ncp-v1", _serve("pouw:ncp-v1", settings={"pouw": {"coins": _coins(), "schedule": "deferred"}})))
