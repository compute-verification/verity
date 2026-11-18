"""The PoUW adapter (`protocol_options.pouw`): it runs H1 only and refuses every other scheme, the settings it refuses, an
install without the epoch's challenger that leaves the model as it was, and Pearl-C's kernel loader.

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
from verity.protocols.resource_accounting.compute.pouw.verifier.kernel_schemes import IDENTIFIED
from verity.protocols.resource_accounting.compute.pouw.prover.serving import h1 as pouw_h1
from verity.protocols.resource_accounting.compute.pouw.prover.serving import pearl_c as PPD
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c import kernel as PPK
from verity.protocols.resource_accounting.compute.pouw.verifier.coins import (
    Challenger,
)

from verity_vllm import protocols as PO
from verity_vllm.engine import hooks
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.protocols import pouw

K, N = 1088, 1600
H1 = pouw_h1.SCHEME


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


# --- what the adapter refuses ------------------------------------------------------------------------------------------------

def test_schemes_are_the_protocol_packages_and_only_h1_is_executed():
    assert pouw.schemes() == tuple(sorted({*IDENTIFIED, H1}))
    assert pouw.EXECUTED == (H1,)
    for scheme in ("ncp-v1", "ncp-v1-shift24", "pearl-fp8-v4", "pearl-c-sm120-v1-h3"):
        assert "executor runs" in pouw.refusals(scheme, _serve(f"pouw:{scheme}"))[0], scheme


@pytest.mark.parametrize("scheme", ["ncp-v2", "ncp-v2-shift24"])
def test_ncp_v2_is_refused_by_name_as_no_scheme_pouw_registers(scheme):
    """ncp-v2's serving is deleted: PoUW registers no ``ncp-v2`` scheme, so the selector refuses its entry by name, beside
    SPOT or not, and the Commit's context refuses it before any engine exists."""
    known = ", ".join(pouw.schemes())
    want = f"pouw:{scheme}: pouw registers no scheme {scheme!r} (registered: {known})"
    assert PO.refusals(_serve(f"pouw:{scheme}")) == (want,)
    assert want in PO.refusals(_serve(f"pouw:{scheme}", "sampled-proofs:rows-sha512"))
    with pytest.raises(PO.Refused) as r:
        PO.commit_context(TargetProfile.default().with_overrides(protocols=(f"pouw:{scheme}", "sampled-proofs:rows-sha512")))
    assert want in r.value.reasons


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
    assert any(match in r for r in PO.refusals(_serve(f"pouw:{H1}", **kw)))


def test_pouw_s_site_is_the_linear_apply_and_h1_executes_nowhere():
    assert pouw.EXECUTES == ("LinearBase.quant_method.apply",) and "pouw" in PO.SITES[pouw.EXECUTES[0]]
    assert not hasattr(pouw, "traced_as") and pouw.executes(H1) == ()


def test_an_install_without_the_epochs_challenger_is_refused_and_leaves_nothing():
    """There is no beacon and no fallback salt: without ``coins`` install refuses before anything is patched."""
    model = Model(_rng().standard_normal((N, K), np.float32))
    before = model.layers[0].quant_method.apply
    with pytest.raises(ValueError, match="no 'coins' setting"):
        pouw.install(H1, types.SimpleNamespace(context=_serve(f"pouw:{H1}", settings={"pouw": {"retain_steps": 0}}),
                                               model=model))
    assert hooks.live(f"pouw:{H1}") == () and model.layers[0].quant_method.apply == before


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
