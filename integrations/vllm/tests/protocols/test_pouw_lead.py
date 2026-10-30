"""`protocols.pouw.lead`: the lead as a PoUW scheme on the int8 plain arm.

- Each engine step's forward is one pass and one window, keyed by the committer (`lead_rows.WindowKeying`). Its salt is
  sampled after keyLeaf, the weights' root and its call count.
- Each linear call's rows are committed (root(A_j), the keyed digest of its rows) before its coins are released, and the
  backend serves the call's y on them.
- The forward's one words root comes at the step's end: the calls' shapes fix the words instance's domain, the backend
  hashes its committed words to the subtree's top under it, and the keying keys that top. Then the pass's
  `PassCommitment` is published.
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

    def __init__(self, n: int, seed: int, method: type = X.PlainGemm, k: int = K):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8)
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


def _top(dom, schema, values) -> bytes:
    """An instance's subtree top as the device hands it, computed on the host: the root's two children, or the one
    value."""
    sub = L.subtree(dom, schema, values)
    return b"".join(sub.levels[-2]) if len(sub.levels) > 1 else sub.values[0]


def top(dom, streams) -> bytes:
    """The words instance's top as `lead_rows.DeviceWords.top` hands it; refused, as the device refuses it, unless the
    streams hold the domain's leaves."""
    values = L.words_values(streams)
    if len(values) != dom.count:
        raise Refused(f"a words instance of {dom.count} leaves got {len(values)}")
    return _top(dom, L.WORDS_LEAF_TAG, values)


def a_top(dom, xq) -> bytes:
    """A_j's top as `lead_rows.device_a_top` hands it, from the rows' bytes."""
    return _top(dom, L.A_LEAF_TAG, L.a_segments(xq.contiguous().cpu().numpy().tobytes()))


class Backend:
    """The lead's kernel, standing in: each call's A top hashed on the host under the domain it was handed (``a_doms``),
    the plain product as y, each call's committed stream a hash of its rows and coins of the stream's length, and the
    words instance's top hashed on the host under the domain each call was handed (``doms``, one a call)."""

    variant = "stand-in"

    def __init__(self, log: list | None = None):
        self.loaded, self.log, self.streams, self.handed, self.doms = [], [] if log is None else log, [], [], []
        self.a_doms = []

    def load(self, w, layer):
        self.loaded.append(w)

    def a_top(self, dom, xq):
        self.log.append(("a_top", dom.count))
        self.a_doms.append(dom)
        return a_top(dom, xq)

    def call(self, w, layer, xq, s_a, feed, dom):
        self.log.append(("backend", w, feed.call))
        c = feed.call
        size = 4 * L.pad_rows(c.m) * L.pad_cols(c.n) * c.shape.S
        self.streams.append(hashlib.shake_256(xq.numpy().tobytes() + feed.head().tobytes()).digest(size))
        self.doms.append(dom)
        return _plain(layer, xq, s_a)

    def words(self):
        self.log.append(("words",))
        out, self.streams = self.streams, []
        self.handed.append(out)
        return top(self.doms[-1], out)


def _linears(*ns: int) -> list[tuple[int, int]]:
    return [(n, K) for n in ns]


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


def _x(m: int, seed: int, k: int = K) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, k, generator=g).to(torch.bfloat16)


def test_the_lead_is_a_scheme_that_executes_and_keeps_its_weights():
    assert PL.SCHEME in pouw.schemes() and PL.SCHEME in pouw.EXECUTED
    assert pouw.executes(PL.SCHEME) == pouw.EXECUTES
    assert pouw.keeps_weight_copy(PL.SCHEME, _ctx()) and not pouw.requires_build(PL.SCHEME, _ctx())


def test_the_lead_refuses_what_it_cannot_be():
    assert PO.refusals(_ctx(backend=Backend())) == ()
    assert any("'backend' is the lead's kernel (`LeadBackend`: load, a_top, call, words)" in r for r in PO.refusals(_ctx()))
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
    root and the window's call count; then the words instance's domain, of each linear's chunks at the first call's m;
    then each call's root(A_j), the keying's digest of its rows at its position from the top the backend hashed under
    its A domain, its coins and its kernel, which gets the window's one domain."""
    log = []
    backend, layer, link = Backend(log), Linear(24, 1), Link(log)
    linears = _linears(16, 8, 8, 24)
    run = PL.LeadRun(ENTRY, backend, link, W, linears, keying=Keys())
    assert run.window is None and run.record is None and run.calls == 4
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
    assert backend.a_doms == [twin.a_domain(PL.FORWARD, 0, 1)]
    root = twin.a_root(PL.FORWARD, 0, xq.numpy().tobytes())
    assert log == [("open", 0, leaf, W, 4), ("a_top", 1), ("release", call, root), ("backend", 3, call)]
    assert record.a_roots == [root] and record.words_root is None
    assert (run.m, run.leaves) == (5, 4) == (5, PL.words_leaves(5, linears))
    assert run.dom.domain_id == twin.words_domain(PL.FORWARD, 4).domain_id and backend.doms == [run.dom]
    xq1, s_a1 = E.quantize_rows(_x(5, 5))
    run.call(0, Linear(16, 4), xq1, s_a1)
    assert record.a_roots[1] == twin.a_root(PL.FORWARD, 1, xq1.numpy().tobytes())
    assert [c.index for c in record.calls] == [0, 1] and run.details["calls"] == 2
    assert [e[0] for e in log].count("open") == 1 and backend.doms == [run.dom] * 2


@pytest.mark.parametrize("check", [False, True])
@pytest.mark.parametrize("m,count", [(1, 1), (16, 1), (17, 2), (80, 5)])
def test_each_calls_root_a_keys_the_backends_top_under_its_own_a_domain(m, count, check):
    """Each call's A is hashed by the backend under its own domain at its position, of ``ceil(m K / A_SEGMENT)``
    segments (one up to the 1024-byte edge, a ragged second, several), and the root recorded from that top is the CPU's
    root(A_j) of the rows, whether ``check`` recomputes it on the host or not."""
    backend = Backend()
    run = PL.LeadRun(ENTRY, backend, Link([]), W, _linears(16, 24), keying=Keys(), check=check)
    rows = [E.quantize_rows(_x(m, w)) for w in range(2)]
    for w, (n, (xq, s_a)) in enumerate(zip((16, 24), rows)):
        run.call(w, Linear(n, w), xq, s_a)
    twin = Keys.twin(0, 2, run.record.salt)
    assert PL.a_count(rows[0][0]) == count
    assert backend.a_doms == [twin.a_domain(PL.FORWARD, j, count) for j in range(2)]
    assert run.record.a_roots == [twin.a_root(PL.FORWARD, j, xq.numpy().tobytes()) for j, (xq, _) in enumerate(rows)]


def _flip(t: bytes) -> bytes:
    return bytes([t[0] ^ 1]) + t[1:]


@pytest.mark.parametrize("m,bad,check,why", [
    (1, lambda t: b"", False, "call 0's A in forward 0: a one-leaf instance holds 1 to 1024 bytes, not 0"),
    (17, lambda t: t[:64], False, "root's two 64-byte children, not 64 bytes"),
    (1, _flip, True, "call 0's A in forward 0: the top from the device is not the CPU's"),
    (80, _flip, True, "call 0's A in forward 0: the top from the device is not the CPU's")])
def test_an_a_top_of_the_wrong_size_or_off_the_cpus_is_refused_before_its_coins(m, bad, check, why):
    """A call whose A top the keying refuses (of the wrong size, or under ``check`` not the CPU's) gets no root, coins or
    kernel, and its pass is never published."""
    class Bad(Backend):
        def a_top(self, dom, xq):
            return bad(super().a_top(dom, xq))
    log = []
    run = PL.LeadRun(ENTRY, Bad(log), Link(log), W, _linears(16), keying=Keys(), check=check)
    with pytest.raises(Refused, match=why):
        run.call(0, Linear(16, 0), *E.quantize_rows(_x(m, 0)))
    assert [e[0] for e in log] == ["open", "a_top"] and run.record.a_roots == []
    with pytest.raises(R.RecordError, match="call 0 raised before its kernel served it"):
        run.end_pass()
    assert not run.passes


def test_a_step_is_a_pass_published_at_its_end_with_one_words_root():
    log = []
    link = Link(log)
    backend = Backend(log)
    run = PL.LeadRun(ENTRY, backend, link, W, _linears(24, 40), keying=Keys(), rank=1, world=2)
    assert run.end_pass() is None
    layers = [Linear(24, 1), Linear(40, 2)]
    for w, layer in enumerate(layers):
        run.call(w, layer, *E.quantize_rows(_x(3, w)))
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
    assert run.end_pass() is out and run.window is None and run.dom is None
    for w, layer in enumerate(layers):
        run.call(w, layer, *E.quantize_rows(_x(2, 9 + w)))
    assert run.m == 2 and run.dom.domain_id != backend.doms[0].domain_id
    assert run.record.pass_index == 1 and run.window.key_leaf == Keys.twin(1, 2).key_leaf != window.key_leaf
    assert run.record.salt == link.windows[1].salt != link.windows[0].salt
    with pytest.raises(R.RecordError, match="no sample before"):
        run.record.commitment()
    run.finish()
    assert [p.pass_index for p in run.passes] == [0, 1]


@pytest.mark.parametrize("m,ns,count", [(5, (24,), 1), (64, (512,), 2), (64, (512, 40), 3), (64, (1024, 8, 512), 6)])
def test_the_words_root_keys_the_backends_top_over_its_linears_chunks_at_the_windows_m(m, ns, count):
    """The first call's m fixes the words instance's leaves, `chunk_count` of each linear at m rows (one chunk; a call of
    two, the last short; pads after a ragged call), before the backend hashes any; each call gets that one domain, and
    the top the backend hands at the end is keyed to the CPU's words root of its calls' streams."""
    backend = Backend()
    run = PL.LeadRun(ENTRY, backend, Link([]), W, _linears(*ns), keying=Keys())
    run.call(0, Linear(ns[0], 0), *E.quantize_rows(_x(m, 0)))
    dom = run.dom
    assert dom.count == count == run.leaves and not backend.handed
    for w, n in enumerate(ns[1:], 1):
        run.call(w, Linear(n, w), *E.quantize_rows(_x(m, w)))
    record = run.record
    run.end_pass()
    twin = Keys.twin(0, len(ns), record.salt)
    assert backend.doms == [dom] * len(ns) and len(L.words_values(backend.handed[0])) == count
    assert dom.domain_id == twin.words_domain(PL.FORWARD, count).domain_id
    assert record.words_root == twin.words_root(PL.FORWARD, backend.handed[0]) == run.passes[0].transcript_root


@pytest.mark.parametrize("calls,cut,why", [(1, lambda t: b"", "forward 0's words: a one-leaf instance holds 1 to"),
                                           (2, lambda t: t[:64], "root's two 64-byte children, not 64 bytes"),
                                           (2, lambda t: t + bytes(64), "not 192 bytes")])
def test_a_top_of_the_wrong_size_is_refused_and_its_pass_never_published(calls, cut, why):
    class Cut(Backend):
        def words(self):
            return cut(super().words())
    run = PL.LeadRun(ENTRY, Cut(), Link([]), W, _linears(*[16] * calls), keying=Keys())
    for w in range(calls):
        run.call(w, Linear(16, w), *E.quantize_rows(_x(1, w)))
    with pytest.raises(Refused, match=why):
        run.end_pass()
    assert not run.passes and run.record.words_root is None


@pytest.mark.parametrize("second,why", [
    ((1, 24, 4), r"pass 0: call 1 \(linear 1\) has 4 rows, and the window's words were counted at its first call's 3"),
    ((1, 32, 3), "pass 0: call 1 is 32 x 64 on linear 1, not one of the 2 linears the window's words were counted for"),
    ((2, 24, 3), "pass 0: call 1 is 24 x 64 on linear 2, not one of the 2 linears")])
def test_a_call_on_another_m_or_off_its_linear_is_refused_by_name_and_its_window_never_published(second, why):
    """A window's words are counted at its first call's m for one call of each linear's n x k: a call on another m (a MoE
    expert's ragged rows) or off its linear is refused before its root(A_j), coins or kernel, never recounted."""
    log = []
    run = PL.LeadRun(ENTRY, Backend(log), Link(log), W, _linears(16, 24), keying=Keys())
    run.call(0, Linear(16, 0), *E.quantize_rows(_x(3, 0)))
    w, n, m = second
    with pytest.raises(Refused, match=why):
        run.call(w, Linear(n, 1), *E.quantize_rows(_x(m, 1)))
    assert [e[0] for e in log] == ["open", "a_top", "release", "backend"] and len(run.record.calls) == 1
    with pytest.raises(R.RecordError, match="call 1 raised before its kernel served it"):
        run.end_pass()
    assert not run.passes


def test_a_window_short_of_its_linears_calls_is_refused_at_its_end():
    run = PL.LeadRun(ENTRY, Backend(), Link([]), W, _linears(16, 24), keying=Keys())
    run.call(0, Linear(16, 0), *E.quantize_rows(_x(3, 0)))
    with pytest.raises(Refused, match="a words instance of 2 leaves got 1"):
        run.end_pass()
    assert not run.passes and run.record.words_root is None


def test_a_window_past_its_call_count_is_refused_and_never_published():
    log = []
    run = PL.LeadRun(ENTRY, Backend(log), Link(log), W, _linears(16, 16), keying=Keys())
    for w in range(2):
        run.call(w, Linear(16, w), *E.quantize_rows(_x(1, w)))
    with pytest.raises(Refused, match="call 2 of 2 in forward 0"):
        run.call(0, Linear(16, 2), *E.quantize_rows(_x(1, 2)))
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
    run = PL.LeadRun(ENTRY, Backend(log), Closed(log), W, _linears(16, 16), keying=Keys())
    with pytest.raises(F.FeedError, match="no salt"):
        run.call(0, Linear(16, 1), *E.quantize_rows(_x(1, 1)))
    assert [e[0] for e in log] == ["open"] and run.window is None
    assert run.end_pass() is None


def test_a_failed_call_leaves_its_pass_unpublishable():
    """No retries: a call whose backend raised has its root(A_j) and no served words, so its pass is never published."""
    class Broken(Backend):
        def call(self, *a):
            raise RuntimeError("kernel fault")
    run = PL.LeadRun(ENTRY, Broken(), Link([]), W, _linears(16, 16), keying=Keys())
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
    run = PL.LeadRun(ENTRY, Backend(log), Refusing(log), W, _linears(16, 16), keying=Keys())
    with pytest.raises(F.FeedError, match="refused"):
        run.call(0, Linear(16, 1), *E.quantize_rows(_x(1, 1)))
    assert [e[0] for e in log] == ["open", "a_top", "release"]
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
    assert [e[0] for e in log] == ["open", *["a_top", "release", "backend"] * 3, "words"]
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
