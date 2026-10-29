"""`protocols.pouw.lead`: the lead as a PoUW scheme on the int8 plain arm. Each linear call's rows are committed
(root(A_j)) before its coins are released, the backend serves the call's y on them, its words root follows, and each
engine step's calls are one pass whose `PassCommitment` is published at the step's end; the option is refused where it
can't be what it says (no backend, other schemes' settings, beside sampled proofs or PoUS) and uninstalls to the plain
arm."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw.coins import Challenger  # noqa: E402
from verity.core.protocols.pouw.serving import PassCommitment  # noqa: E402
from verity_vllm import protocols as PO  # noqa: E402
from verity_vllm.engine import hooks  # noqa: E402
from verity_vllm.engine import int8_exact as E  # noqa: E402
from verity_vllm.engine import int8_experts as X  # noqa: E402
from verity_vllm.protocols import pouw  # noqa: E402
from verity_vllm.protocols.pouw import h1 as PH  # noqa: E402
from verity_vllm.protocols.pouw import lead as PL  # noqa: E402
from verity_vllm.protocols.pouw.lead import feed as F  # noqa: E402
from verity_vllm.protocols.pouw.lead import record as R  # noqa: E402
from verity_vllm.protocols.pouw.pearl_c import passes as PP  # noqa: E402

K = 64
ENTRY = f"pouw:{PL.SCHEME}"


class Linear:
    """A dense linear on the plain arm: int8 weight [n, K], scales [n, 1], its own `PlainGemm`."""

    def __init__(self, n: int, seed: int, method: type = X.PlainGemm):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randint(-127, 128, (n, K), generator=g, dtype=torch.int8)
        self.weight_scale = (torch.rand(n, 1, generator=g) * 0.01 + 1e-4).float()
        self.quant_method = method(f"linear{seed}")


class Model:
    def __init__(self, *layers):
        self.layers = list(layers)

    def modules(self):
        return iter(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.quant_method.apply(layer, x)[..., :K]
        return x


def _plain(layer, xq, s_a):
    return E.dequantize(E.int_mm(xq, layer.weight), s_a, layer.weight_scale)


def a_root(call, xq, key):
    return hashlib.sha512(R.header(call) + xq.numpy().tobytes() + key).digest()


class Backend:
    """The lead's kernel and committer, standing in: the plain product as y, root(A_j) a hash of the call, its rows and
    the pass's key, and a words root over the call's rows and coins."""

    variant = "stand-in"

    def __init__(self, log: list | None = None):
        self.loaded, self.log = [], [] if log is None else log

    def load(self, w, layer):
        self.loaded.append(w)

    def a_root(self, w, call, xq, key):
        self.log.append(("a_root", w, call, key))
        return a_root(call, xq, key)

    def call(self, w, layer, xq, s_a, feed):
        self.log.append(("backend", w, feed.call))
        return _plain(layer, xq, s_a), hashlib.sha512(xq.numpy().tobytes() + feed.head().tobytes()).digest()


class Link(F.InProcessCertifier):
    """The in-process certifier, logging each release with the root it was handed."""

    def __init__(self, log: list):
        super().__init__(np.random.default_rng(0).bytes)
        self.log = log

    def release(self, call, root):
        self.log.append(("release", call, root))
        return super().release(call, root)


@pytest.fixture(autouse=True)
def _linear_is_the_stand_in(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))


def _ctx(stage="serve", enabled=(ENTRY,), **settings):
    return PO.Context(stage=stage, enabled=enabled, settings={"pouw": settings})


def _x(m: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, K, generator=g).to(torch.bfloat16)


def _keys():
    n = iter(range(1 << 30))
    return lambda size: hashlib.sha512(next(n).to_bytes(8, "big")).digest()[:size]


def test_the_lead_is_a_scheme_that_executes_and_keeps_its_weights():
    assert PL.SCHEME in pouw.schemes() and PL.SCHEME in pouw.EXECUTED
    assert pouw.executes(PL.SCHEME) == pouw.EXECUTES
    assert pouw.keeps_weight_copy(PL.SCHEME, _ctx()) and not pouw.requires_build(PL.SCHEME, _ctx())


def test_the_lead_refuses_what_it_cannot_be():
    assert PO.refusals(_ctx(backend=Backend())) == ()
    assert any("'backend' is the lead's kernel and committer" in r for r in PO.refusals(_ctx()))
    taken = PO.refusals(_ctx(backend=Backend(), coins=Challenger(0, np.random.default_rng(0).bytes), retain_steps=2,
                             transcript_dir="/tmp/x"))
    assert any("pouw settings coins, transcript_dir, retain_steps are other schemes'" in r for r in taken)
    assert any("'rows' and 'steps' are H1's" in r for r in PO.refusals(_ctx(backend=Backend(), rows=32, steps=1)))
    assert any("'challenger' is for" in r for r in PO.refusals(_ctx(backend=Backend(), challenger="host:1")))
    other = PO.refusals(_ctx(enabled=("pouw:ncp-v1",), backend=Backend(),
                             coins=Challenger(0, np.random.default_rng(0).bytes)))
    assert any(f"'backend' is the lead's ({PL.SCHEME})" in r for r in other)
    with pytest.raises(ValueError, match="'backend' is the lead's kernel and committer"):
        pouw.Settings.of({"backend": object()})
    eager = PO.Context(stage="serve", enabled=(ENTRY,), settings={"pouw": {"backend": Backend()}}, enforce_eager=False)
    assert any("graph-capturable" in r for r in PO.refusals(eager))


def test_the_lead_is_refused_beside_sampled_proofs_and_pous():
    """Sampled proofs would trace the lead's call as the stock computation, and under PoUS its GEMMs would read the
    backend's loaded weights, not what PoUS stores."""
    beside = PO.refusals(_ctx(enabled=(ENTRY, "sampled-proofs:rows-sha512"), backend=Backend()))
    assert any("the Program has no Definition for what it computes there" in r for r in beside)
    pous = PO.refusals(_ctx(enabled=(ENTRY, "pous:band-chain/d12/v1"), backend=Backend()))
    assert any("keeps a weight-derived copy resident" in r for r in pous)


def test_own_auditor_coins_are_not_added_under_the_lead():
    assert "coins" not in pouw.challenged({"pouw": {}}, (ENTRY,))["pouw"]
    assert "coins" in pouw.challenged({"pouw": {}}, ("pouw:ncp-v1",))["pouw"]


def test_a_call_commits_its_rows_before_its_coins_and_its_words_after():
    log = []
    backend, layer = Backend(log), Linear(24, 1)
    run = PL.LeadRun(ENTRY, backend, Link(log), b"\x07" * 64, key=_keys())
    key = run.pass_key
    assert len(key) == PL.KEY_BYTES
    x = _x(5, 2)
    xq, s_a = E.quantize_rows(x)
    y = run.call(3, layer, xq, s_a)
    assert torch.equal(y.view(torch.int16), _plain(layer, xq, s_a).view(torch.int16))
    call = F.Call(0, 0, 3, 5, K, 24)
    root = a_root(call, xq, key)
    assert log == [("a_root", 3, call, key), ("release", call, root), ("backend", 3, call)]
    assert run.record.a_roots == [root] and len(run.record.words_roots) == 1
    run.call(0, Linear(16, 4), *E.quantize_rows(_x(1, 5)))
    assert [c.index for c in run.record.calls] == [0, 1] and run.details["calls"] == 2


def test_a_step_is_a_pass_published_at_its_end():
    log = []
    run = PL.LeadRun(ENTRY, Backend(log), Link(log), b"\x07" * 64, key=_keys(), rank=1, world=2)
    assert run.end_pass() is None
    layers = [Linear(24, 1), Linear(40, 2)]
    for w, layer in enumerate(layers):
        run.call(w, layer, *E.quantize_rows(_x(3 + w, w)))
    record = run.record
    out = run.end_pass(step=True)
    assert isinstance(out, PassCommitment) and out is record.published and run.passes == [out]
    sha = lambda parts: hashlib.sha512(b"".join(parts)).digest()  # noqa: E731
    assert (out.scheme, out.rank, out.world, out.pass_index, out.salt, out.kernel) == (PL.SCHEME, 1, 2, 0, b"", "stand-in")
    assert out.units == 2 and out.tiles == sum(R.tiles(c.shape) for c in record.calls)
    assert out.activation_roots == sha(record.a_roots) and out.transcript_root == sha(record.words_roots)
    assert run.commits["passes"] == [out.to_json()] and run.details["passes"] == 1
    assert run.end_pass() is out
    run.call(0, layers[0], *E.quantize_rows(_x(2, 9)))
    keys = [e[3] for e in log if e[0] == "a_root"]
    assert run.record.pass_index == 1 and keys[0] == keys[1] != keys[2] == run.pass_key
    with pytest.raises(R.RecordError, match="no sample before"):
        run.record.commitment()
    run.finish()
    assert [p.pass_index for p in run.passes] == [0, 1]


def test_a_failed_call_leaves_its_pass_unpublishable():
    """No retries: a call whose backend raised has its root(A_j) and no words, so its pass is never published."""
    class Broken(Backend):
        def call(self, *a):
            raise RuntimeError("kernel fault")
    run = PL.LeadRun(ENTRY, Broken(), Link([]), b"\x07" * 64, key=_keys())
    with pytest.raises(RuntimeError, match="kernel fault"):
        run.call(0, Linear(16, 1), *E.quantize_rows(_x(1, 1)))
    with pytest.raises(R.RecordError, match="has its A root and no words"):
        run.end_pass()


def test_a_call_the_link_refuses_gets_no_kernel():
    class Refusing(Link):
        def release(self, call, root):
            super().release(call, root)
            raise F.FeedError("refused")
    log = []
    run = PL.LeadRun(ENTRY, Backend(log), Refusing(log), b"\x07" * 64, key=_keys())
    with pytest.raises(F.FeedError, match="refused"):
        run.call(0, Linear(16, 1), *E.quantize_rows(_x(1, 1)))
    assert [e[0] for e in log] == ["a_root", "release"]
    with pytest.raises(R.RecordError, match="has its A root and no words"):
        run.end_pass()


class Runner:
    """A model runner class standing in for vLLM's: ``execute_model`` runs the step's forward."""

    def __init__(self, model):
        self.model = model

    def execute_model(self, scheduler_output):
        return self.model.forward(scheduler_output)


def test_install_serves_each_call_through_the_lead_and_uninstalls_to_the_plain_arm(monkeypatch):
    log = []
    monkeypatch.setattr(PP, "model_runners", lambda: (Runner,))
    monkeypatch.setattr(F, "InProcessCertifier", lambda: Link(log))
    model = Model(Linear(64, 1), Linear(64, 2), Linear(40, 3))
    x = _x(5, 7)
    plain = model.forward(x)
    backend = Backend(log)
    target = PO.Target(model, _ctx(backend=backend))
    execute_model = vars(Runner)["execute_model"]
    option = pouw.install(PL.SCHEME, target)
    assert backend.loaded == [0, 1, 2]
    assert all(m.quant_method.gemm is not None for m in model.layers)
    runner = Runner(model)
    served = runner.execute_model(x)
    assert torch.equal(served.view(torch.int16), plain.view(torch.int16))
    assert [e[0] for e in log] == ["a_root", "release", "backend"] * 3
    assert [e[2].linear for e in log if e[0] == "backend"] == [0, 1, 2]
    assert len(option.commits["passes"]) == 1 and option.commits["passes"][0]["units"] == 3
    runner.execute_model(_x(2, 8))
    assert [p["pass_index"] for p in option.commits["passes"]] == [0, 1]
    assert option.end_pass().pass_index == 1
    assert option.commits["coin_source"] == F.IN_PROCESS and option.commits["layout"] == F.LAYOUT
    assert option.info()["pass_ends"] == "each engine step, at the model runner's execute_model"
    option.close()
    assert hooks.live(ENTRY) == () and all("gemm" not in vars(m.quant_method) for m in model.layers)
    assert vars(Runner)["execute_model"] is execute_model
    assert torch.equal(model.forward(x).view(torch.int16), plain.view(torch.int16))


def test_install_refuses_a_layer_off_the_arm_or_without_gemm():
    class Observing:
        int8_arm = E.ARM

        def __init__(self, prefix):
            pass

        def apply(self, layer, x, bias=None):
            return x

        def observe(self, *a):
            pass
    target = PO.Target(Model(Linear(16, 1), Linear(16, 2, Observing)), _ctx(backend=Backend()))
    with pytest.raises(ValueError, match=rf"{ENTRY} refused at install: linear 1: Observing has no gemm"):
        pouw.install(PL.SCHEME, target)
    bf16 = Linear(16, 3)
    bf16.weight = bf16.weight.to(torch.bfloat16)
    with pytest.raises(ValueError, match=rf"{ENTRY} refused at install: linear 0: PlainGemm with a bfloat16 weight"):
        pouw.install(PL.SCHEME, PO.Target(Model(bf16), _ctx(backend=Backend())))
    with pytest.raises(ValueError, match=rf"pouw:{PH.SCHEME} refused at install"):
        pouw.commit_weights(PH.SCHEME, PO.Target(Model(bf16), _ctx(backend=Backend())))
