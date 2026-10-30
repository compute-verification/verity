"""`protocols.pouw.lead`: the lead as a PoUW scheme on the int8 plain arm.

- Each engine step's forward is one pass and one window, keyed by the committer (`lead_rows.WindowKeying`). Its salt is
  sampled after keyLeaf, the weights' root and its call count.
- Each linear call's rows are committed (root(A_j), the keyed digest of its rows) before its coins are released, and the
  backend serves the call's y on them.
- The forward's one words root, the keyed digest of the backend's committed words, comes at the step's end, and then the
  pass's `PassCommitment` is published.
- The option is refused where it can't be what it says: no backend, other schemes' settings, or beside sampled proofs or
  PoUS. It uninstalls to the plain arm."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw.coins import Challenger  # noqa: E402
from verity.core.protocols.pouw.serving import PassCommitment  # noqa: E402
from verity_vllm import protocols as PO  # noqa: E402
from verity_vllm.commit import lead_rows as L  # noqa: E402
from verity_vllm.engine import hooks  # noqa: E402
from verity_vllm.engine import int8_exact as E  # noqa: E402
from verity_vllm.engine import int8_experts as X  # noqa: E402
from verity_vllm.errors import Refused  # noqa: E402
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


def _label(what: bytes, i: int, size: int) -> bytes:
    return hashlib.shake_256(what + i.to_bytes(8, "big")).digest(size)


class Keys:
    """The run's window keyings, from fixed bytes: window i's ks_w and y0 are labels of i, so its twin keys the same
    digests."""

    def __init__(self):
        self.made = 0

    def __call__(self, calls):
        self.made += 1
        return self.twin(self.made - 1, calls)

    @staticmethod
    def twin(i, calls, salt=None):
        """Window i's keying, registered under ``W`` and opened under ``salt`` when given."""
        k = L.WindowKeying(calls, ks_w=_label(b"ks_w", i, 64), y0=_label(b"y0", i, L.KEY_SALT_BYTES))
        if salt is not None:
            k.register(W)
            k.open(salt)
        return k


class Backend:
    """The lead's kernel, standing in: the plain product as y, and each call's committed words its rows and coins."""

    variant = "stand-in"

    def __init__(self, log: list | None = None):
        self.loaded, self.log, self.streams, self.handed = [], [] if log is None else log, [], []

    def load(self, w, layer):
        self.loaded.append(w)

    def call(self, w, layer, xq, s_a, feed):
        self.log.append(("backend", w, feed.call))
        self.streams.append(xq.numpy().tobytes() + feed.head().tobytes())
        return _plain(layer, xq, s_a)

    def words(self):
        self.log.append(("words",))
        out, self.streams = self.streams, []
        self.handed.append(out)
        return out


class Link(F.InProcessCertifier):
    """The in-process certifier, logging each window it opens and each release with the root it was handed."""

    def __init__(self, log: list):
        super().__init__(np.random.default_rng(0).bytes)
        self.log = log

    def open(self, pass_index, key_leaf, weights_root, calls):
        self.log.append(("open", pass_index, key_leaf, weights_root, calls))
        return super().open(pass_index, key_leaf, weights_root, calls)

    def release(self, call, root):
        self.log.append(("release", call, root))
        return super().release(call, root)


W = b"\x07" * 64


@pytest.fixture(autouse=True)
def _linear_is_the_stand_in(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))


def _ctx(stage="serve", enabled=(ENTRY,), **settings):
    return PO.Context(stage=stage, enabled=enabled, settings={"pouw": settings})


def _x(m: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, K, generator=g).to(torch.bfloat16)


def test_the_lead_is_a_scheme_that_executes_and_keeps_its_weights():
    assert PL.SCHEME in pouw.schemes() and PL.SCHEME in pouw.EXECUTED
    assert pouw.executes(PL.SCHEME) == pouw.EXECUTES
    assert pouw.keeps_weight_copy(PL.SCHEME, _ctx()) and not pouw.requires_build(PL.SCHEME, _ctx())


def test_the_lead_refuses_what_it_cannot_be():
    assert PO.refusals(_ctx(backend=Backend())) == ()
    assert any("'backend' is the lead's kernel (`LeadBackend`: load, call, words)" in r for r in PO.refusals(_ctx()))
    taken = PO.refusals(_ctx(backend=Backend(), coins=Challenger(0, np.random.default_rng(0).bytes), retain_steps=2,
                             transcript_dir="/tmp/x"))
    assert any("pouw settings coins, transcript_dir, retain_steps are other schemes'" in r for r in taken)
    assert any("'rows' and 'steps' are H1's" in r for r in PO.refusals(_ctx(backend=Backend(), rows=32, steps=1)))
    assert any("'challenger' is for" in r for r in PO.refusals(_ctx(backend=Backend(), challenger="host:1")))
    other = PO.refusals(_ctx(enabled=("pouw:ncp-v1",), backend=Backend(),
                             coins=Challenger(0, np.random.default_rng(0).bytes)))
    assert any(f"'backend' is the lead's ({PL.SCHEME})" in r for r in other)
    with pytest.raises(ValueError, match="'backend' is the lead's kernel"):
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


def test_a_window_opens_at_its_first_call_and_each_call_commits_its_rows_before_its_coins():
    """C3's order: the window keying's registration, then the link's salt, sampled once it holds keyLeaf, the weights'
    root and the window's call count; then each call's root(A_j), the keying's digest of its rows at its position, its
    coins and its kernel."""
    log = []
    backend, layer, link = Backend(log), Linear(24, 1), Link(log)
    run = PL.LeadRun(ENTRY, backend, link, W, 4, keying=Keys())
    assert run.window is None and run.record is None
    x = _x(5, 2)
    xq, s_a = E.quantize_rows(x)
    y = run.call(3, layer, xq, s_a)
    assert torch.equal(y.view(torch.int16), _plain(layer, xq, s_a).view(torch.int16))
    window, record = run.window, run.record
    leaf = Keys.twin(0, 4).key_leaf
    assert (window.calls, window.key_leaf, record.pass_index, record.salt) == (4, leaf, 0, link.windows[0].salt)
    assert (record.weights_root, record.key_leaf, record.window_calls) == (W, leaf, 4)
    call = F.Call(0, 0, 3, 5, K, 24)
    twin = Keys.twin(0, 4, record.salt)
    root = twin.a_root(PL.FORWARD, 0, xq.numpy().tobytes())
    assert log == [("open", 0, leaf, W, 4), ("release", call, root), ("backend", 3, call)]
    assert record.a_roots == [root] and record.words_root is None
    xq1 = E.quantize_rows(_x(1, 5))[0]
    run.call(0, Linear(16, 4), xq1, torch.ones(1))
    assert record.a_roots[1] == twin.a_root(PL.FORWARD, 1, xq1.numpy().tobytes())
    assert [c.index for c in record.calls] == [0, 1] and run.details["calls"] == 2
    assert [e[0] for e in log].count("open") == 1


def test_a_step_is_a_pass_published_at_its_end_with_one_words_root():
    log = []
    link = Link(log)
    backend = Backend(log)
    run = PL.LeadRun(ENTRY, backend, link, W, 2, keying=Keys(), rank=1, world=2)
    assert run.end_pass() is None
    layers = [Linear(24, 1), Linear(40, 2)]
    for w, layer in enumerate(layers):
        run.call(w, layer, *E.quantize_rows(_x(3 + w, w)))
    record, window = run.record, run.window
    out = run.end_pass(step=True)
    assert isinstance(out, PassCommitment) and out is record.published and run.passes == [out]
    assert [e[0] for e in log][-1] == "words" and [e[0] for e in log].count("words") == 1
    twin = Keys.twin(0, 2, record.salt)
    assert record.words_root == twin.words_root(PL.FORWARD, backend.handed[0]) and len(backend.handed[0]) == 2
    sha = lambda parts: hashlib.sha512(b"".join(parts)).digest()  # noqa: E731
    assert (out.scheme, out.rank, out.world, out.pass_index, out.salt, out.kernel) == \
        (PL.SCHEME, 1, 2, 0, link.windows[0].salt, "stand-in")
    assert out.units == 2 and out.tiles == sum(R.tiles(c.shape) for c in record.calls)
    assert out.activation_roots == sha(record.a_roots) and out.transcript_root == record.words_root
    assert out.records_root == sha([window.key_leaf, (2).to_bytes(8, "big"), *map(R.header, record.calls)])
    assert run.commits["passes"] == [out.to_json()] and run.commits["calls"] == 2 and run.details["passes"] == 1
    assert run.end_pass() is out and run.window is None
    run.call(0, layers[0], *E.quantize_rows(_x(2, 9)))
    assert run.record.pass_index == 1 and run.window.key_leaf == Keys.twin(1, 2).key_leaf != window.key_leaf
    assert run.record.salt == link.windows[1].salt != link.windows[0].salt
    with pytest.raises(R.RecordError, match="no sample before"):
        run.record.commitment()
    run.finish()
    assert [p.pass_index for p in run.passes] == [0, 1]


def test_a_window_past_its_call_count_is_refused_and_never_published():
    log = []
    run = PL.LeadRun(ENTRY, Backend(log), Link(log), W, 2, keying=Keys())
    for w in range(2):
        run.call(w, Linear(16, w), *E.quantize_rows(_x(1, w)))
    with pytest.raises(Refused, match="call 2 of 2 in forward 0"):
        run.call(2, Linear(16, 2), *E.quantize_rows(_x(1, 2)))
    assert [e[0] for e in log][-1] == "backend" and [e[0] for e in log].count("release") == 2 and len(run.record.calls) == 2
    with pytest.raises(R.RecordError, match="call 2 raised before its kernel served it"):
        run.end_pass()
    assert not run.passes


def test_a_window_the_link_refuses_commits_no_call():
    class Closed(Link):
        def open(self, *a):
            super().open(*a)
            raise F.FeedError("no salt")
    log = []
    run = PL.LeadRun(ENTRY, Backend(log), Closed(log), W, 2, keying=Keys())
    with pytest.raises(F.FeedError, match="no salt"):
        run.call(0, Linear(16, 1), *E.quantize_rows(_x(1, 1)))
    assert [e[0] for e in log] == ["open"] and run.window is None
    assert run.end_pass() is None


def test_a_failed_call_leaves_its_pass_unpublishable():
    """No retries: a call whose backend raised has its root(A_j) and no served words, so its pass is never published."""
    class Broken(Backend):
        def call(self, *a):
            raise RuntimeError("kernel fault")
    run = PL.LeadRun(ENTRY, Broken(), Link([]), W, 2, keying=Keys())
    with pytest.raises(RuntimeError, match="kernel fault"):
        run.call(0, Linear(16, 1), *E.quantize_rows(_x(1, 1)))
    with pytest.raises(R.RecordError, match="call 0 raised before its kernel served it"):
        run.end_pass()


def test_a_call_the_link_refuses_gets_no_kernel():
    class Refusing(Link):
        def release(self, call, root):
            super().release(call, root)
            raise F.FeedError("refused")
    log = []
    run = PL.LeadRun(ENTRY, Backend(log), Refusing(log), W, 2, keying=Keys())
    with pytest.raises(F.FeedError, match="refused"):
        run.call(0, Linear(16, 1), *E.quantize_rows(_x(1, 1)))
    assert [e[0] for e in log] == ["open", "release"]
    with pytest.raises(R.RecordError, match="call 0 raised before its kernel served it"):
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
    assert [e[0] for e in log] == ["open", *["release", "backend"] * 3, "words"]
    assert [e[2].linear for e in log if e[0] == "backend"] == [0, 1, 2]
    assert [e[4] for e in log if e[0] == "open"] == [3] and option.commits["calls"] == 3
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
