"""Storage outside useful work at a linear layer's ``apply``, as a served engine composes them: PoUS attached by its
scheme's name on the exchange, the linear site attached after vLLM's ``load_model`` with the weights PoUS stores
released (`engine.kv_blocks`), then the first-stage hash scheme (H1) enabled through the selector (`protocols.install`).

Today the pair in one process is refused by name: H1 commits its weights from each layer's weight, which PoUS has
released (`protocols.refusals`). The case that H1's GEMM is handed the weight PoUS decoded, while the layer's own
weight is the released, empty tensor, is kept expected-to-fail for the day H1 is declared on the exchange.

The exchange alone: two kernels filed in the application's order, ``pous`` then ``pouw``; the earlier listed is the
outer (`verity.config.check`; `messages.Exchange.request`)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from verity import hooks as H
from verity.protocols.compose import schemes
from verity.protocols.resource_accounting.memory.pous.prover import weight_store as ws
from verity.protocols.resource_accounting.memory.pous.verifier import period
from verity.protocols.resource_accounting.memory.pous.verifier.schemes.p2 import P2Params

from tests.protocols import test_pouw_h1 as T1
from tests.protocols._needs_torch import torch
from verity_vllm import messages
from verity_vllm.errors import Refused
from verity_vllm import protocols as PO
from verity_vllm.engine import hooks
from verity_vllm.engine import kv_blocks as KB
from verity_vllm.protocols import pouw
from verity_vllm.protocols.linear import vllm_site

SOURCE = "a5" * 32
#: the small point's 1,024-bit blocks, 8 a segment, as test_pous_served runs P2, so encoding takes milliseconds
SMALL = P2Params(period.W, period.C0, 8)


class Linear(T1.Linear):
    """H1's int8 linear stand-in with its weight a parameter, as vLLM holds it, so PoUS can release it."""

    def __init__(self, n, seed):
        super().__init__(n, seed)
        self.weight = torch.nn.Parameter(self.weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(self.weight_scale, requires_grad=False)


class Model(T1.Model):
    def named_modules(self):
        return ((f"l{i}", m) for i, m in enumerate(self.layers))

    def named_parameters(self, remove_duplicate=True):
        return ((f"l{i}.{p}", getattr(m, p)) for i, m in enumerate(self.layers) for p in ("weight", "weight_scale"))


def _released(monkeypatch):
    """The toy H1 model with PoUS attached on the current exchange, the linear site attached and l0's weight released,
    as `kv_blocks`' ``load_model`` leaves them; its int8 arm records the weight it is handed and the layer's own."""
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))
    monkeypatch.setattr(ws, "geometry", lambda scheme: SMALL)
    model = Model(Linear(32, 1))
    layer = model.layers[0]
    plain = layer.weight.detach().clone()

    seen = {}
    arm = T1.Int8Arm.apply

    def recording(self, held, x, bias=None):
        seen["handed"] = held.weight.detach().clone()
        seen["layer.weight"] = layer.weight.detach().clone()
        return arm(self, held, x, bias)

    monkeypatch.setattr(T1.Int8Arm, "apply", recording)
    ex = messages.Exchange()
    monkeypatch.setattr(messages, "_CURRENT", ex)
    ex.attach(schemes.get("pous"), {}, salt_source=SOURCE, experimental=True)
    ex.answer(H.Declared("card", {"eager": True}, tuple(ex.attached), {n: dict(v) for n, v in ex.attached.items()}))
    sites = vllm_site.discover(model, types=(Linear,))
    ex.emit(H.RegisteredValues((("l0.weight", layer.weight),), operands={"l0.weight": 0}))
    site_hooks = hooks.Hooks(tag="linear-site")
    vllm_site.attach(sites, site_hooks)                                  # kv_blocks' load_model: the site, then release
    KB._Hook.release_values(SimpleNamespace(released=[]), model, ex)
    assert layer.weight.numel() == 0
    return model, layer, plain, seen, site_hooks


def test_storage_beside_the_first_stage_hash_scheme_in_one_process_is_refused_by_name(monkeypatch):
    model, _layer, _plain, _seen, site_hooks = _released(monkeypatch)
    try:
        with pytest.raises(Refused) as e:
            PO.install(model, PO.Context(stage="serve", enabled=(T1._entry(),), settings=T1._settings()))
    finally:
        site_hooks.uninstall()
    assert T1._entry() in str(e.value) and "pous" in str(e.value).replace(T1._entry(), "")


@pytest.mark.xfail(strict=True, raises=Refused,
                   reason="refused by name until the first-stage hash scheme is declared on the exchange: it commits "
                          "its weights from layer.weight, which PoUS released at load_model")
def test_h1s_gemm_is_handed_the_weight_pous_decoded_and_the_layer_holds_the_released_weight(monkeypatch):
    model, layer, plain, seen, site_hooks = _released(monkeypatch)
    settings = T1._settings()
    runner = settings["pouw"]["runner"]
    try:
        comp = PO.install(model, PO.Context(stage="serve", enabled=(T1._entry(),), settings=settings))
    except Refused:
        site_hooks.uninstall()
        raise
    try:
        runner.execute_model(T1._so(("a", [5], 4)))
        x = (torch.randn(1, T1.K, generator=torch.Generator().manual_seed(9)) * 3).bfloat16()
        layer.quant_method.apply(layer, x)
    finally:
        comp.close()
        site_hooks.uninstall()
    assert "handed" in seen, "H1's GEMM never ran"
    assert seen["layer.weight"].numel() == 0, "the layer's own weight is the released, empty tensor"
    assert torch.equal(seen["handed"], plain), "H1's GEMM was not handed the weight PoUS decoded"


def test_two_kernels_filed_in_list_order_put_the_earlier_listed_outside():
    ex = messages.Exchange()
    order = []

    def kernel(name):
        def compute(due, original):
            order.append(name)
            return original(None)
        return compute

    for name in ("pous", "pouw"):                                        # the application's order
        ex.request(name, H.ComputeBlockWith(kernel(name)))
    due = H.BlockDue("l0", 0, {"n": 2, "k": 2}, "s8", {"x": "x", "weight": "w"})
    ex.compute(due, lambda given: "stock")
    assert order == ["pous", "pouw"], f"outer first: {order}"
