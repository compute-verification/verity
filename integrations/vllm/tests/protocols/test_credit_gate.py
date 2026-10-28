"""`protocol_options.credit_gate` and its place in H1's credit (`pouw_h1_hooks.traced`, `H1Run.judge`, `H1Run.credit`): a served
call is credited only if the y the linear hands back is the credited kernel's own output, one GEMM invocation writing z and
y, no other GEMM ran for it, and its z is its committed z/y records, every proof block's y dq of its z and the served y theirs
(`pouw_h1_records.reason`).  vllm's int8 arm under H1 (``zy_rows`` set, ``verity_int8::scaled_mm_zy`` over T slots) is
credited; the arm serving ``scaled_mm``'s y with z a second ``_int_mm`` beside it is refused with its named reason, and so
is every way of handing over a z or records the served kernel didn't write for the served y."""
from __future__ import annotations

import functools
import hashlib
import itertools
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
# ruff: noqa: E402  (the imports below need torch, so they follow its importorskip)

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.circuit import live as L
from verity.core.protocols.pouw.coins import Challenger
from verity.core.protocols.sampled_proofs import challenger as LC
from verity.core.protocols.sampled_proofs import remote as R
from verity.core.protocols.sampled_proofs.challenger import LiveOpening, LiveRecord
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity_vllm import protocols as PO
from verity_vllm.commit.served_record import PORT_NAMES
from verity_vllm.engine import int8_exact as E
from verity_vllm.engine import int8_ops
from verity_vllm.protocols.pouw import credit_gate as CG
from verity_vllm.protocols import pouw
from verity_vllm.protocols.pouw import h1 as PH
from verity_vllm.protocols.pouw.h1 import challenger as PHC
from verity_vllm.protocols.pouw.h1 import commit as CT
from verity_vllm.protocols.pouw.h1 import prover as PV
from verity_vllm.protocols.pouw.h1 import records as ZR
from verity_vllm.protocols.pouw.h1 import requests as RQ
from verity_vllm.protocols.pouw.h1 import served as SV

from tests.protocols import challenger_host as CH

K = 64
#: the window's T
ROWS = 32
ENTRY = f"pouw:{PH.SCHEME}"
ZY = "verity_int8::scaled_mm_zy"
#: the name the Commit's step-input port acquires a step's input ids under
PORT = PORT_NAMES[0]
TODAY = "credit: served-output-not-credited-kernel: q y from verity_int8::scaled_mm, z from aten::_int_mm"
RECORDS = f"credit: {CG.RECORDS}: q "


def zy_linear(xq, s_a, w, s_w):
    """``scaled_mm_zy`` over the window's T slots, as (z, y): z its records."""
    y, zy = int8_ops.scaled_mm_zy(xq, s_a, w, s_w, ROWS)
    return zy, y


@torch.library.custom_op("verity_credit_test::fused_linear", mutates_args=())
def fused_linear(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A fused kernel handing over its exact s32 Y, not z/y records, with y = dequant(Y)."""
    z = E.int_mm(xq, w)
    return z, E.dequantize(z, s_a, s_w)


@torch.library.custom_op("verity_credit_test::two_gemms", mutates_args=())
def two_gemms(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One op hiding today's path: y from one ``_int_mm``, z from another."""
    return E.int_mm(xq, w), E.dequantize(E.int_mm(xq, w), s_a, s_w)


#: an opaque op like ``scaled_mm_zy`` (one GEMM the trace can't see into) whose records and y `FORGE` rewrites
_LIB = torch.library.Library("verity_credit_test", "FRAGMENT")
_LIB.define("forged_zy(Tensor xq, Tensor s_a, Tensor w, Tensor s_w, int rows) -> (Tensor y, Tensor zy)")
FORGE: list = []


def _forged_zy(xq, s_a, w, s_w, rows):
    y, zy = E.scaled_mm_zy(xq, s_a, w, s_w, rows)
    return FORGE[-1](y.clone(), zy.clone(), xq, s_a, w, s_w)


_LIB.impl("forged_zy", _forged_zy, "CPU")


def forged_linear(xq, s_a, w, s_w):
    y, zy = torch.ops.verity_credit_test.forged_zy.default(xq, s_a, w, s_w, ROWS)
    return zy, y


class TodayArm:
    """`engine.int8_vllm.Int8PlainLinearMethod` without ``zy_rows``: `int8_ops.linear` (``row_quant`` then ``scaled_mm``),
    its values handed to ``observe``, y returned."""

    int8_arm = E.ARM

    def __init__(self):
        self.calls = 0

    def apply(self, layer, x, bias=None):
        self.calls += 1
        lead, k = x.shape[:-1], x.shape[-1]
        xq, s_a, y = int8_ops.linear(x.reshape(-1, k), layer.weight, layer.weight_scale)
        self.observe(layer, xq, s_a, y)
        return y.reshape(*lead, layer.weight.shape[0])

    def observe(self, layer, xq, s_a, y, z=None):
        pass


class ZyArm(TodayArm):
    """`engine.int8_vllm.Int8PlainLinearMethod`'s apply: with ``zy_rows`` (which H1's install sets) ``int8_ops.linear_zy``
    over that many slots, its records handed to ``observe`` as z."""

    zy_rows: int | None = None

    def apply(self, layer, x, bias=None):
        self.calls += 1
        lead, k = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, k)
        rows = self.zy_rows
        if rows is not None and x2.shape[0] <= rows:
            xq, s_a, y, zy = int8_ops.linear_zy(x2, layer.weight, layer.weight_scale, rows)
            self.observe(layer, xq, s_a, y, zy)
        else:
            xq, s_a, y = int8_ops.linear(x2, layer.weight, layer.weight_scale)
            self.observe(layer, xq, s_a, y)
        return y.reshape(*lead, layer.weight.shape[0])


class FusedArm(TodayArm):
    """An arm of one kernel (`zy_linear`, or `kernel`) writing z and y after ``row_quant``: ``after`` applied to y before it
    is served, ``z_after`` to z before it is handed to ``observe``, ``side`` run beside."""

    def __init__(self, kernel=zy_linear, after=None, side=None, z_after=None):
        super().__init__()
        self.kernel, self.after, self.side, self.z_after = kernel, after, side, z_after

    def apply(self, layer, x, bias=None):
        self.calls += 1
        xq, s_a = int8_ops.row_quant(x)
        z, y = self.kernel(xq, s_a, layer.weight, layer.weight_scale)
        if self.side is not None:
            self.side(xq, layer.weight)
        if self.after is not None:
            y = self.after(y)
        if self.z_after is not None:
            z = self.z_after(z)
        self.observe(layer, xq, s_a, y, z)
        return y


class Linear:
    def __init__(self, n, seed, arm):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randint(-127, 128, (n, K), generator=g, dtype=torch.int8)
        self.weight_scale = (torch.rand(n, 1, generator=g) * 0.01 + 1e-4).float()
        self.quant_method = arm


class Model:
    def __init__(self, *layers):
        self.layers = list(layers)

    def modules(self):
        return iter(self.layers)

    def named_modules(self):
        return iter([("q", self.layers[0])] + [(f"l{i}", m) for i, m in enumerate(self.layers[1:], 1)])


@pytest.fixture(autouse=True)
def _linear_is_the_stand_in(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))


AUDITOR = "vy-challenger-1:7811"
#: the gate's reserve answer for the window's challenge phase, and the gate that opens it on the window's open root
RESERVED = {"session": CH.session_id(1)}
GATE: dict = {}


@pytest.fixture
def auditor(tmp_path, monkeypatch):
    """The auditor host's challenger, one challenge phase over a one-round live epoch, reached at `AUDITOR`, reserved as
    `RESERVED` and opened by `GATE`'s ``open``."""
    host = CH.Host(tmp_path / "auditor", Wd.live_challenger(0, 1, CH.entropy(1)), 1, gated=True)
    monkeypatch.setattr(R.RemoteChallenger, "connect",
                        classmethod(lambda cls, at, secret=None: host.client if at == AUDITOR else None))
    monkeypatch.setitem(GATE, "open", host.open)
    return host


PERIOD = "p1"
#: the fixed length `_serve`'s one request serves (its max_tokens 1, the public-cap done source), as the period's
#: open commits it (`credit.committed_length`)
LENGTH = {"max_tokens": [1], "stop": Wd.STOP_FIXED}
#: the served window's layout (`_serve`'s one call of N = 64 over T = `ROWS` slots), as the period declares it: 2 proof blocks
LAYOUT = SV.declared_layout([(64, K)], {0: "q"}, ROWS, 1)


@functools.cache
def _descriptor(shapes=((64, K),), modules=(("q"),), rows=ROWS, steps=1) -> bytes:
    """The declared window Program's descriptor bytes (`pouw_h1_served.descriptor`) on the served test's stand-in
    proof block: the period's circuit is their SHA-512, and core's credit reads the window's layout from them."""
    from tests.protocols.test_pouw_h1_served import _unit
    return SV.descriptor(SV.declared(list(shapes), dict(enumerate(modules)), rows, steps, unit=_unit()))


def _circuit() -> str:
    return hashlib.sha512(_descriptor()).hexdigest()


#: the prover's own weights key, which salted its b/<K> before the auditor's registration
WKEY = bytes(range(32, 64))
#: each K's circuit file's SHA-512 as a period-open commits them (``circuit_files``), and the live template's
FILES = {f"k{K}": hashlib.sha512(b"the k64 template's circuit file").hexdigest(),
         CR.LIVE: hashlib.sha512(b"the live template's circuit file").hexdigest()}


def _rows(edit=None):
    """`_serve`'s one linear's (``Linear(64, 1)``) int8 codes and f32 s_w, with one row's s_w or one code other with
    ``edit`` ``"s_w"`` or ``"code"``."""
    layer = Linear(64, 1, None)
    codes, scales = layer.weight.numpy().copy(), layer.weight_scale.numpy().reshape(-1).copy()
    if edit == "s_w":
        scales[3] *= 2
    elif edit == "code":
        codes[3, 5] ^= 1
    return codes, scales


def _registry(edit=None) -> CH.Registry:
    """The auditor's model registry (in process) holding `CH.MODEL`'s registration under the period's circuit:
    its checkpoint `_rows` of `edit`, committed under the auditor's own salt key (the gate's ``model-register``)."""
    reg = CH.Registry()
    codes, scales = _rows(edit)
    reg.register(CH.MODEL, _circuit(), _descriptor(), lambda name, k: CH.linear_rows(codes, scales))
    return reg


@functools.cache
def _model_key() -> SV.ModelKey:
    """The lane's ``model-key --model LLAMA31_8B_INT8 --circuit C`` of the honest registration (`_registry`)."""
    return SV.model_key(_registry().key(CH.MODEL, _circuit()))


@functools.cache
def _period_weights() -> SV.Weights:
    """The period's weights as the served path registers them: `_serve`'s one linear's b/<K> under the declared Program,
    salted with the registration's key (`_model_key`)."""
    return SV.register_weights(_circuit(), [(64, K)], {0: _rows()}, _model_key().key)


def _served(**kw) -> SV.Served:
    """The served path's `Served`, its b/<K> registered at install from the served linear under `_model_key` (or as
    ``kw`` says)."""
    from tests.protocols.test_pouw_h1_served import _unit
    return SV.Served(PH.TILE, _unit(), **(kw or {"registry": _model_key()}))


@pytest.fixture
def periodic(tmp_path, monkeypatch):
    """`auditor` with its challenge phase reserved inside lane vllm's period `PERIOD`, whose open commits `_circuit` and (unless
    None) the draw K ``draw_k`` and the fixed length `LENGTH`, opened with ``--model`` `CH.MODEL` (unless ``model`` is
    False) on ``registry`` (default `_registry`), and ``files`` its circuit files: ``periodic(draw_k)`` is the host; close the period with
    ``host.ledger.period_close`` once the challenge phase has ended."""
    def make(draw_k, model=True, files=None, registry=None):
        leaves = itertools.count()
        ledger = CH.Ledger(tmp_path / "auditor.jsonl", 1, leaves, registry=registry or _registry())
        ledger.period_open(PERIOD, circuit=_circuit(), draw_k=draw_k, rows=ROWS, circuit_files=files,
                           model=CH.MODEL if model else None, length=LENGTH)
        host = CH.Host(tmp_path / "auditor", Wd.live_challenger(0, 1, CH.entropy(1)), 1, leaves=leaves, ledger=ledger,
                       gated=True)
        monkeypatch.setattr(R.RemoteChallenger, "connect",
                            classmethod(lambda cls, at, secret=None: host.client if at == AUDITOR else None))
        monkeypatch.setitem(GATE, "open", host.open)
        return host
    return make


def _commit(tap, x, ids, n):
    """The Commit's acquires at the window's one engine step (`pouw_h1_commit.CommitTap.take`) before linear ``q``'s
    forward: the step's input ids at its step-input port, its first `n` the step's scheduled tokens, and ``q``'s
    ``row_quant`` of `x`, as the int8 source takes them at its site (`_serve` takes its y after the forward)."""
    tap.take(PORT, torch.tensor(ids), 0, n)
    xq, s_a = int8_ops.row_quant(x.reshape(-1, K))
    tap.take(f"{CT.site('q')}/0", xq, 0)
    tap.take(f"{CT.site('q')}/1", s_a, 0)


def _serve(arm, retain=1, m=3, prompt=True, remote=False, n=32, exact=True, served=None, ids=None, pad=0, graphs=()):
    """One small call (m rows of K, N = `n`) through `arm` with H1 installed (T = `ROWS`), its request admitted with an
    m-token prompt (with none: prompt embeddings, so no entry); the run closed and its y and run returned, the served y
    the plain linear's when `exact`. Its coins are the auditor host's challenger's with `remote` (the `auditor` fixture),
    else held in this process; with `served` (`pouw_h1_served.Served`) its rows are the window Program's values and the
    Commit is in its engine (`_commit`, its input ids `ids`, else the prompt's). With `pad` the call runs `pad` rows past
    the m scheduled, as a CUDA graph of the engine's capture sizes `graphs` pads it, its padded input ids 0."""
    layer = Linear(n, 1, arm)
    if served is not None:
        served.tap = CT.CommitTap(port=PORT)
    runner = SimpleNamespace(execute_model=lambda scheduler_output, *a, **kw: None, decode_query_len=1)
    if remote:
        coins = {"challenger": AUDITOR, "session": RESERVED, "gate": lambda session, root: GATE["open"](session, root)}
    else:
        coins = {"coins": Challenger(0, random.Random(0).randbytes),
                 "rounds": LiveRecord(Wd.live_challenger(0, 1, random.Random(1).randbytes), Wd.ROUND, clock=lambda: 0.0)}
    ctx = PO.Context(stage="serve", enabled=(ENTRY,), settings={"pouw": {"rows": ROWS, "steps": 1, "retain_steps": retain,
                                                                         "runner": runner, **coins, "graphs": graphs,
                                                                         **({"served": served} if served else {})}})
    comp = PO.install(Model(layer), ctx)
    try:
        req = SimpleNamespace(req_id="r", prompt_token_ids=list(range(m)) if prompt else None,
                              sampling_params=SimpleNamespace(max_tokens=1))
        runner.execute_model(SimpleNamespace(scheduled_new_reqs=[req], num_scheduled_tokens={"r": m},
                                             total_num_scheduled_tokens=m))
        x = (torch.randn(m + pad, K, generator=torch.Generator().manual_seed(9)) * 3).bfloat16()
        if served is not None:
            _commit(served.tap, x, (list(range(m)) if ids is None else ids) + [0] * pad, m)
        y = layer.quant_method.apply(layer, x)
        if served is not None:
            served.tap.take(f"{CT.out_site('q')}/0", y, 0)
        run = pouw.h1_run(ENTRY)
        run.close()
        if exact:
            assert torch.equal(y.view(torch.int16), E.plain_linear(x, layer.weight, layer.weight_scale).view(torch.int16))
        return y, run
    finally:
        comp.close()


def _refused(run, reason, m=3, n=32):
    """The window's one call refused for `reason` (a prefix of it ending in a space), and nothing credited."""
    (v,) = run.verdicts.values()
    assert not v["credit"] and (v["reason"].startswith(reason) if reason.endswith(" ") else v["reason"] == reason), v
    c = run.credit()
    assert c["gate_passed"] == {"rows": 0, "macs": 0}
    assert c["refused"] == {"calls": 1, "rows": m, "macs": m * n * K, "reasons": {v["reason"]: 1}}
    return v


def test_the_int8_arm_under_h1_serves_scaled_mm_zy_and_is_credited(auditor):
    """vllm's int8 arm with H1 installed: the install sets its ``zy_rows`` to T, so it serves ``scaled_mm_zy`` (y bit for
    bit the plain linear's) and hands over the records, which are the call's z table, one leaf a proof block; the call is
    credited and the window, its coins the auditor host's challenger's, credits its served rows; the install's close
    takes ``zy_rows`` back off."""
    arm = ZyArm()
    y, run = _serve(arm, remote=True, n=64)
    assert arm.zy_rows is None and auditor.ended() is None and run.coin_source == PH.CHALLENGER
    (v,) = run.verdicts.values()
    assert v["credit"] and "reason" not in v
    assert (v["kernel"], v["y_from"], v["z_from"]) == (ZY,) * 3 and v["gemms"] == [ZY]
    c = run.credit()
    assert c["gate_passed"] == {"rows": 3, "macs": 3 * 64 * K} and c["refused"]["calls"] == 0
    assert set(run.tables[0]) == {"a", "z"} and all(run.tables[0].values())
    _index, xq, _sa, z = run.retained[-1]
    assert np.array_equal(z, xq.astype(np.int64) @ Linear(64, 1, None).weight.numpy().astype(np.int64).T)
    assert c["requests"] == [] and run.details["requests"]["admitted"] == 1


def test_a_layer_whose_n_is_not_whole_units_is_left_on_scaled_mm_and_refused():
    """``scaled_mm_zy`` writes whole 32-column proof blocks only, so a layer of N = 40 keeps ``scaled_mm``, commits no z table and
    credits nothing."""
    arm = ZyArm()
    _y, run = _serve(arm, n=40)
    assert run.tables[0]["z"] is None
    _refused(run, TODAY, n=40)


def test_todays_served_path_is_refused_y_from_scaled_mm_z_from_a_second_int_mm():
    """The arm serving `scaled_mm`'s y with H1's z a second ``_int_mm`` beside it (J11): the call is refused with the named
    reason, both GEMMs in its trace, and the window credits nothing; the served y is untouched."""
    _y, run = _serve(TodayArm())
    v = _refused(run, TODAY)
    assert v["gemms"] == ["aten::_int_mm in verity_int8::scaled_mm", "aten::_int_mm"]


def test_todays_path_without_a_retained_z_is_refused_as_no_z():
    _y, run = _serve(TodayArm(), retain=0)
    (v,) = run.verdicts.values()
    assert v["reason"] == "credit: served-output-not-credited-kernel: q y from verity_int8::scaled_mm, z from <none>"


def test_serve_h1s_wrapper_is_refused_too():
    """The served H1 run's tree (`pouw_h1.served`): the arm's apply serves y, then the wrapper
    quantizes the same activations again and takes z from `int_mm`; the trace refuses it the same way."""
    layer = Linear(32, 1, TodayArm())
    x = (torch.randn(3, K, generator=torch.Generator().manual_seed(9)) * 3).bfloat16()
    with CG.Trace("q") as t:
        y = layer.quant_method.apply(layer, x)
        xq, _sa = E.quantize_rows(x.reshape(-1, x.shape[-1]))
        z = E.int_mm(xq, layer.weight)
    v = CG.verdict(t, y, z)
    assert v["reason"] == TODAY and len(v["gemms"]) == 2


def test_records_from_a_second_scaled_mm_zy_beside_the_served_scaled_mm_are_refused():
    """A z not from the served y's kernel: y from ``scaled_mm``, the records from a ``scaled_mm_zy`` beside it."""
    def beside(xq, s_a, w, s_w):
        return zy_linear(xq, s_a, w, s_w)[0], int8_ops.scaled_mm(xq, s_a, w, s_w)
    _y, run = _serve(FusedArm(kernel=beside))
    _refused(run, f"credit: served-output-not-credited-kernel: q y from verity_int8::scaled_mm, z from {ZY}")


def test_a_credited_call_in_a_window_whose_coins_were_held_in_process_credits_nothing():
    """The call passes the gate, but the window's coins were held in the serving process (no pouw setting
    'challenger'), so no verifier trusts them and the window credits nothing, naming why."""
    _y, run = _serve(FusedArm())
    (v,) = run.verdicts.values()
    assert v["credit"] and run.coin_source == "in-process"
    c = run.credit()
    assert c["gate_passed"] == {"rows": 0, "macs": 0} and c["coins"] == "in-process"
    assert c["refused"] == {"calls": 1, "rows": 3, "macs": 3 * 32 * K, "reasons": {PH.IN_PROCESS: 1}}


def test_in_process_coins_named_by_the_leans_claim_instance_still_credit_nothing(monkeypatch):
    """A claim instance names where coins come from, not who holds them. The harness's coins, held in the
    serving process, credit nothing (``coins-not-the-auditors``) even when the Lean verifier of record's claim instance
    names their source; a window is credited only on the auditor host's challenge phase (the int8 arm's test above)."""
    from verity.core.protocols.pouw import verifier
    held, real = [], Challenger

    def named(epoch, entropy):
        held.append(real(epoch, LC.named(verifier.CLAIM, entropy)))
        return held[-1]
    monkeypatch.setitem(globals(), "Challenger", named)
    _y, run = _serve(FusedArm())
    (v,) = run.verdicts.values()
    assert [c.claim for c in held] == [verifier.CLAIM] and v["credit"] and run.coin_source == "in-process"
    assert "coins-not-the-auditors" in PH.IN_PROCESS and run.credit()["refused"]["reasons"] == {PH.IN_PROCESS: 1}


def test_a_credited_call_in_a_window_serving_a_request_without_an_entry_credits_nothing():
    """The call passes the gate, but its rows are a request's with no entry (prompt embeddings), so
    the window credits nothing and names why."""
    _y, run = _serve(FusedArm(), prompt=False)
    (v,) = run.verdicts.values()
    assert v["credit"]
    c = run.credit()
    assert c["gate_passed"] == {"rows": 0, "macs": 0} and c["requests"] == ["step 0: 3 rows of a request with no entry"]
    assert c["refused"] == {"calls": 1, "rows": 3, "macs": 3 * 32 * K, "reasons": {RQ.UNENTERED: 1}}


@pytest.mark.parametrize("side,want", [
    (lambda xq, w: E.int_mm(xq, w), "aten::_int_mm"),
    (lambda xq, w: xq.float() @ w.float().t(), "aten::mm"),
], ids=["second-int-mm", "float-mm"])
def test_a_second_gemm_beside_the_kernel_is_refused(side, want):
    _y, run = _serve(FusedArm(side=side))
    v = _refused(run, f"credit: gemm-count: q 2 GEMMs ({ZY}, ")
    assert v["gemms"][1] == want


def test_one_op_hiding_two_gemms_is_refused():
    _y, run = _serve(FusedArm(kernel=two_gemms))
    v = _refused(run, "credit: gemm-count: q 2 GEMMs ")
    assert v["y_from"] == v["z_from"] == "verity_credit_test::two_gemms"


def test_a_fused_kernel_handing_over_its_exact_y_rather_than_records_is_refused():
    """One GEMM writing z and y passes the trace, but a z that is not the call's records commits no z table."""
    _y, run = _serve(FusedArm(kernel=fused_linear))
    v = _refused(run, f"{RECORDS}its z is not its {PH.units(K, 32, ROWS)} z/y records of {ZR.ZY_BYTES} bytes")
    assert v["kernel"] == "verity_credit_test::fused_linear" and run.tables[0]["z"] is None


@pytest.mark.parametrize("after,op", [(lambda y: y.clone(), "aten::clone"), (lambda y: y.add_(0), "aten::add_")],
                         ids=["copied", "written-after"])
def test_a_served_y_that_is_not_the_kernels_output_is_refused(after, op):
    """A y copied or written after the kernel is not the kernel's output: refused, y from the op that wrote it last."""
    _y, run = _serve(FusedArm(after=after))
    _refused(run, f"credit: served-output-not-credited-kernel: q y from {op}, z from {ZY}")


def _flip(z):
    z[0, 0] ^= 1
    return z


@pytest.mark.parametrize("z_after,op", [(lambda z: z.clone(), "aten::clone"), (_flip, "aten::copy_"),
                                        (lambda z: z.contiguous().clone()[:, :], "aten::clone")],
                         ids=["copied", "a-byte-tampered", "a-view-of-a-copy"])
def test_records_copied_or_tampered_after_the_kernel_are_refused(z_after, op):
    """The records handed over must be the tensor the kernel wrote (or a view of it): a copy, even bit for bit, or a byte
    written after the kernel is some other op's output, and credits nothing."""
    _y, run = _serve(FusedArm(z_after=z_after))
    _refused(run, f"credit: served-output-not-credited-kernel: q y from {ZY}, z from {op}")


def test_records_of_another_call_are_refused_as_untraced():
    """The records of an earlier call (written outside this call's trace) handed over with this call's y."""
    w = Linear(32, 1, None)
    xq, s_a = int8_ops.row_quant((torch.randn(3, K, generator=torch.Generator().manual_seed(9)) * 3).bfloat16())
    stale = zy_linear(xq, s_a, w.weight, w.weight_scale)[0]
    _y, run = _serve(FusedArm(z_after=lambda z: stale))
    _refused(run, f"credit: served-output-not-credited-kernel: q y from {ZY}, z from <untraced>")


def _cell(i, j, y=False):
    """Byte offset in a proof block's record of its cell (i, j)'s z word (or y half-word with `y`)."""
    return (4 * 32 * 32 + 2 * (32 * i + j)) if y else 4 * (32 * i + j)


def _xor(zy, u, off, bit):
    zy[u, off] ^= bit
    return zy


def _another_calls(y, zy, xq, s_a, w, s_w):
    return y, E.scaled_mm_zy(torch.roll(xq, 1, 0), torch.roll(s_a, 1, 0), w, s_w, ROWS)[1]


def _consistent_but_not_served(y, zy, xq, s_a, w, s_w):
    """Proof block 0's z with one served cell moved and its y redone as dq of it, so the records hold together but aren't what
    was served."""
    z, yr = E.zy_split(zy, ROWS, w.shape[0])
    z[0, 0] += 1 << 20
    sa = torch.zeros(ROWS)
    sa[:xq.shape[0]] = s_a.reshape(-1)
    return y, E.zy_records(z, E.dequantize(z, sa, s_w))


#: forged records and served y from one opaque GEMM, so the trace credits them; each must be refused by the value check
FORGERIES = {
    "y-not-dq-of-z": (lambda y, zy, *a: (y, _xor(zy, 1, _cell(1, 5, y=True), 1)),
                      "unit 1's y is not dq of its z at the call's scales"),
    "z-not-the-product": (lambda y, zy, *a: (y, _xor(zy, 0, _cell(2, 7) + 2, 0x10)),
                          "unit 0's y is not dq of its z at the call's scales"),
    "units-swapped": (lambda y, zy, *a: (y, zy[[1, 0]]), "unit 0's y is not dq of its z at the call's scales"),
    "another-calls-records": (_another_calls, "unit 0's y is not dq of its z at the call's scales"),
    "records-not-the-served-y": (_consistent_but_not_served, "the served y is not unit 0's records' y"),
    "served-y-not-the-records": (lambda y, zy, *a: (y.view(torch.int16).bitwise_xor(1).view(torch.bfloat16).clone(), zy),
                                 "the served y is not unit 0's records' y"),
    "empty-slot-z": (lambda y, zy, *a: (y, _xor(zy, 1, _cell(3, 0), 1)), "unit 1's empty slot has a z that isn't 0"),
}


@pytest.mark.parametrize("name", list(FORGERIES))
def test_a_kernel_whose_records_are_not_its_served_product_credits_nothing(name):
    """A kernel the trace credits (one opaque GEMM writing y and the records) that hands over records whose y isn't dq of
    their z at the call's scales (a flipped y bit, a moved z word, two proof blocks swapped, another call's records), whose y
    isn't the served y, or with a z on an empty slot: the call is refused as ``z-records``, naming the proof block."""
    fn, why = FORGERIES[name]
    FORGE.append(fn)
    try:
        _y, run = _serve(FusedArm(kernel=forged_linear), n=64, exact=False)
    finally:
        FORGE.pop()
    v = _refused(run, f"{RECORDS}{why}", n=64)
    assert v["kernel"] == "verity_credit_test::forged_zy" and v["gemms"] == ["verity_credit_test::forged_zy"]


def test_the_honest_forgery_stand_in_is_credited():
    """The forging op with nothing forged is credited, so each refusal above is its forgery's."""
    FORGE.append(lambda y, zy, *a: (y, zy))
    try:
        _y, run = _serve(FusedArm(kernel=forged_linear), n=64)
    finally:
        FORGE.pop()
    (v,) = run.verdicts.values()
    assert v["credit"], v


@pytest.mark.parametrize("name", [None, *FORGERIES])
def test_the_value_check_on_the_device_is_the_hosts(name):
    """`pouw_h1_records.reason` of a call's records, scales and served y left on a CUDA device, as the arm leaves them, is
    what it is of their host copies: None for honest records, and each forgery's own refusal."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    w = Linear(64, 1, None)
    xq, s_a = int8_ops.row_quant((torch.randn(3, K, generator=torch.Generator().manual_seed(9)) * 3).bfloat16())
    y, zy = E.scaled_mm_zy(xq, s_a, w.weight, w.weight_scale, ROWS)
    if name is not None:
        y, zy = FORGERIES[name][0](y.clone(), zy.clone(), xq, s_a, w.weight, w.weight_scale)
    host = ZR.reason(zy.numpy(), ROWS, 64, y, s_a, w.weight_scale)
    assert host == (None if name is None else FORGERIES[name][1])
    assert ZR.reason(zy.cuda(), ROWS, 64, y.cuda(), s_a.cuda(), w.weight_scale.cuda()) == host


def test_a_z_written_outside_the_trace_is_refused_as_untraced():
    w = torch.randint(-127, 128, (32, K), dtype=torch.int8)
    xq = torch.randint(-127, 128, (3, K), dtype=torch.int8)
    z = zy_linear(xq, torch.ones(3), w, torch.ones(32, 1))[0]
    with CG.Trace("q") as t:
        y = zy_linear(xq, torch.ones(3), w, torch.ones(32, 1))[1]
    assert CG.verdict(t, y, z)["reason"] == f"credit: served-output-not-credited-kernel: q y from {ZY}, z from <untraced>"


def test_a_call_recorded_outside_a_traced_apply_is_not_credited():
    run = PH.H1Run("t", hashlib.sha512(b"salt").digest(), {0: "q"}, [(32, K)], 32, 1)
    run.record(0, torch.zeros(2, K, dtype=torch.int8), torch.ones(2), torch.zeros(32, K, dtype=torch.int8), torch.ones(32, 1))
    run.close()
    c = run.credit()
    assert c["gate_passed"] == {"rows": 0, "macs": 0}
    assert c["refused"]["reasons"] == {"credit: untraced: q call 0 was recorded outside a traced apply": 1}


def _record(run):
    return {"commitment": run.commits, "calls": run.manifest()}


def _credit(host, rec, statements=(), k=None, descriptor=None, run=None, live=None):
    return PHC.credit_reason(rec, host.directory, [host.ledger.path], host.window, host.key,
                             _descriptor() if descriptor is None else descriptor, statements, k, run=run,
                             pins=[host.ledger.pins_path], main=CH.MAIN, live=live)


def _live(d, rec):
    """The window's live evidence (`credit.LIVE_STATEMENTS`): the live Program's descriptor at T = `ROWS`, the window's
    rounds and `SV.PREFIX`, and each round's statement, its files stand-ins in `d` (the verifier of record is
    `_verifier`, which proves round s's proof block s)."""
    steps = len(rec["commitment"]["rounds"])
    return {"descriptor": SV.descriptor(L.program_v2(ROWS, steps, SV.PREFIX)).hex(),
            "statements": [{**{x: v for x, v in _statement(d / f"live{s}", s, 1).items() if x != "call"}, "round": s}
                           for s in range(steps)]}


def _statement(d, call, n):
    """Call `call`'s statement over `n` releases, its files stand-ins in `d` (the verifier of record is `_verifier`)."""
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"s{i}").mkdir(exist_ok=True)
        (d / f"s{i}" / "session.json").write_text("{}")
    (d / "circuit.bin").write_bytes(b"")
    (d / "public.json").write_text("{}")
    return PHC.statement(call, d / "circuit.bin", d / "public.json", [d / f"s{i}" for i in range(n)])


def _verifier(units, reject=(), seen=None, opened=lambda port: True):
    """A stand-in for the verifier of record (`proofs.verify`): release i proves ``units[i]``, accepted unless in
    `reject`, its verdict naming the program it was handed and its registered finding the ports of the `--registered`
    file credit built that `opened` keeps, each at the root credit read from the window's entries. A statement on the
    live template's circuit file proves its round's proof block, the round its ``m@<s>`` port names, and opens every port."""
    from verity.core.protocols.sampled_proofs.one_stage import proofs as PF

    def run(proof, *, circuit_sha512, program, partition, registered):
        if circuit_sha512 == FILES[CR.LIVE]:
            s = int(registered["ports"]["m"]["value"].rsplit("@", 1)[1])
            found = {port: {"value": e["value"], "root": e["root"]} for port, e in registered["ports"].items()}
            return [PF.Finding(str(x), True, "", hashlib.sha512(program).hexdigest(), found, (s,))
                    for x in proof.sessions], None
        if seen is not None:
            seen.append({"circuit_sha512": circuit_sha512, "registered": registered, "partition": partition})
        found = {port: {"value": e["value"], "root": e["root"]} for port, e in registered["ports"].items()
                 if opened(port)}
        return [PF.Finding(str(s), u not in reject, "rejected" if u in reject else "", hashlib.sha512(program).hexdigest(),
                           found, (u,) if u not in reject else ()) for s, u in zip(proof.sessions, units)], None
    return run


def test_a_window_with_no_served_tables_credits_nothing(periodic):
    """The gate-passed call's manifest row carries its credit bit and its z root under calls_root, but a window
    with no served trees registers no A table and no ``b/<K>`` of its period's weights and has no Commit in its engine,
    so core's credit refuses it before the draw (`credit.window`'s ``commit``, ROOTS1) and it credits nothing."""
    auditor = periodic(1, files=FILES)
    _y, run = _serve(ZyArm(), remote=True, n=64)
    assert auditor.ended() is None
    rec = _record(run)
    assert [(r["index"], r[CR.CREDIT]) for r in rec["calls"]] == [(0, True)] and rec["calls"][0]["tables"]["z"]
    assert "layout" not in rec["commitment"]
    out = _credit(auditor, rec)
    assert out["manifest"] is None and out["k"] == 1, out
    assert out["window"] == ("commit", CR.COMMIT_FORM), out
    assert out["credit"] is None and out["eps"] is None


def test_an_honest_served_windows_weights_are_its_periods_and_its_evidence_carries_proof_files(periodic):
    """A served window whose b/<K> it registered at install under the auditor's
    registration's key (`model-key`) registers the registration's ``b/<K>`` entries, its open root over their root (the
    one the period's ``--model`` open signed), its round-0 request table and its challenge phase; its evidence
    (`window_evidence`) is core's `credit.WINDOW_KEYS`, its registered ``entries`` and its drawn calls' statements
    (proof files, never verdicts), round 0's ``requests`` the request table's root its open root binds. Core's credit
    (`credit.window`) passes its weights and refuses it only for want of its drawn calls' proof files."""
    from verity.core.protocols.pouw import salt as PS
    k = PHC.draw_k(LAYOUT)
    assert k == 2
    auditor = periodic(k, files=FILES)
    _y, run = _serve(ZyArm(), remote=True, n=64, served=_served())
    assert auditor.ended() is None and not run.served.refused
    rec, w = _record(run), _period_weights()
    c = rec["commitment"]
    (line,) = auditor.ledger.registry.lines
    opened = json.loads(auditor.ledger.lines[0])["body"]
    assert opened["weights_root"] == line["weights_root"] == _model_key().weights_root == w.root().hex()
    assert opened[CR.MODEL_FIELD] == {"name": CH.MODEL, "checkpoint_sha512": CR.PINNED_MODELS[CH.MODEL]}
    assert [e for e in c["served"]["entries"] if CR.is_held(e["value"])] == line["entries"]
    assert c["weights_root"] == w.root().hex()
    assert c["session"]["root"] == PS.window_open_root(w.root(), bytes.fromhex(c["requests_root"]),
                                                       auditor.session_id).hex()
    entries = c["served"]["entries"]
    assert [e for e in entries if CR.is_held(e["value"])] == w.entries()
    out = _credit(auditor, rec)
    assert out["manifest"] is None and out["k"] == k, out
    assert out["drawn"] == PHC.drawn_of(rec, auditor.window, k, _descriptor())[0] == [0, 1]
    assert out["window"][0] == "missing" and out["credit"] is None, out
    unpinned = PHC.credit_reason(rec, auditor.directory, [auditor.ledger.path], auditor.window, auditor.key,
                                 _descriptor(), pins=None, main=CH.MAIN)
    assert unpinned["window"] == ("coin", f"the window's coins, session or salt can't be read ({LC.NO_PINS})")
    assert unpinned["credit"] is None
    ev = PHC.window_evidence(rec, auditor.directory, auditor.window, [], _descriptor())
    assert tuple(sorted(ev)) == CR.WINDOW_KEYS and ev["entries"] == entries and ev["statements"] == []
    assert ev["rounds"][0]["requests"] == c["requests_root"]
    assert run.details["roots1"]["tied"] and run.details["roots1"]["own_trees"] == 0 and ev["commit"] == c["commit"]
    ids0 = next(e for e in entries if e["value"] == "h1/ids@0")
    r0 = ev["rounds"][0]
    assert r0["request_table"] == c["request_table"] == run._tables[0].commitment.root.hex()
    assert r0["requests"] == CR.requests_root(bytes.fromhex(r0["request_table"]), bytes.fromhex(ids0["root"])).hex()
    assert ids0["domain"] == G.domain(run.served.program_digest, ids0["value"], ids0["words"], ids0["leaves"],
                                      session=auditor.session_id).domain_id.hex()


class EditArm(ZyArm):
    """`ZyArm` handing ``observe`` (so H1's A table) the codes `edit` makes of the ones it quantized; its served y and
    records are the real ones."""

    def __init__(self, edit):
        super().__init__()
        self.edit = edit

    def observe(self, layer, xq, s_a, y, z=None):
        pass

    def apply(self, layer, x, bias=None):
        xq, s_a, y, zy = int8_ops.linear_zy(x.reshape(-1, K), layer.weight, layer.weight_scale, self.zy_rows)
        self.observe(layer, self.edit(xq.clone()), s_a, y, zy)
        return y


def _one_row(xq):
    xq[1, 0] = 1 if int(xq[1, 0]) != 1 else 2
    return xq


@pytest.mark.parametrize("arm,ids,want", [
    (lambda: EditArm(torch.zeros_like), None, "h1/a/64@0 is not the Commit's row-table/0/k64"),
    (lambda: EditArm(_one_row), None, "h1/a/64@0 is not the Commit's row-table/0/k64"),
    (lambda: ZyArm(), [0, 7, 2], "round 0's input-ids table h1/ids@0 is not the Commit's input-ids/0"),
], ids=["a-all-zero", "one-a-row-off", "ids-off-the-prompt"])
def test_a_window_whose_tables_are_not_the_commits_is_refused_by_core(periodic, monkeypatch, arm, ids, want):
    """ROOTS1, by core's credit (`credit._arow`): H1 handed every A code zero (the A = 0 probe) or one A
    row off the codes the Commit acquired at the linear's row_quant, or the Commit acquired an input id other than the
    prompt's (round 0's ids table is the prompt's, fixed before the salt), each with an honest Commit, whose record
    registers its own tree for those rows. The serving process credits nothing (`pouw_h1_commit.UNTIED`); a prover
    that skips that check and credits anyway is refused by core at ``arow``, before any proof is read."""
    auditor = periodic(PHC.draw_k(LAYOUT), files=FILES)
    own = CT.roots1
    seen = {}

    def skipped(run):
        seen.update(own(run))
        return {**seen, "reason": None}
    monkeypatch.setattr(CT, "roots1", skipped)
    _y, run = _serve(arm(), remote=True, n=64, served=_served(), ids=ids)
    assert auditor.ended() is None
    assert not seen["tied"] and seen["own_trees"] == 1, seen
    assert all(v["credit"] for v in run.gated().values())
    out = _credit(auditor, _record(run))
    assert out["manifest"] is None and out["window"][0] == "arow" and want in out["window"][1], out
    assert out["credit"] is None


@pytest.mark.parametrize("edit", [None, "a-zero"], ids=["honest", "a-all-zero"])
def test_the_gpu_harness_holds_an_in_process_window_to_cores_roots1_steps(edit):
    """`pouw_h1_gpu.core_verdict` on a window under in-process coins (the GPU run's local gate), the Commit in its
    engine: its record is one manifest under calls_root with the Commit's record (`pouw_h1_gpu.record_reason`, its salt
    from the in-process coin commitment), its A tables under their rounds' roots, and each the Commit's value; with the
    A codes all zero, core refuses it at ``arow`` alone. A calls_root off the record is refused."""
    from tests.protocols import pouw_h1_gpu as G
    from tests.protocols.test_pouw_h1_served import _unit
    arm = ZyArm() if edit is None else EditArm(torch.zeros_like)
    _y, run = _serve(arm, n=64, served=SV.Served(PH.TILE, _unit(), weights=_period_weights(), window=bytes(64)))
    assert run.commits.get("coin_source") != PH.CHALLENGER and not run.served.refused
    record = {"commitment": dict(run.commits), "calls": run.manifest()}
    v = G.core_verdict({**record, "descriptor": _descriptor().hex()})
    assert v["manifest"] is None and v["a_link"] is None and v["commit_values"] == 3, v
    assert v["tied"] if edit is None else (v["arow"][0] == "arow" and "is not the Commit's" in v["arow"][1]), v
    off = {**record["commitment"], "calls_root": "00" * 64}
    assert "calls_root is not over" in G.record_reason({**record, "commitment": off})


def _bare_table(monkeypatch):
    bare = PH.H1Run._requests0

    def outside(self, table):
        ids, self._ids0 = self._ids0, None
        try:
            return bare(self, table)
        finally:
            self._ids0 = ids
    monkeypatch.setattr(PH.H1Run, "_requests0", outside)


def _other_session(monkeypatch):
    tree = SV.Served.ids_tree

    def other(self, run, s, words):
        sid, run.session = run.session, f"{run.session}-other"
        try:
            return tree(self, run, s, words)
        finally:
            run.session = sid
    monkeypatch.setattr(SV.Served, "ids_tree", other)


def test_a_window_opened_over_round_0s_bare_request_table_is_refused_by_core(periodic, monkeypatch):
    """ROOTS1's ids@0: the window opens over `credit.requests_root` of round 0's request table's root and its input-ids
    table's root, so the prompt tokens are fixed before the salt. One opened over the bare request table's root, its
    tables otherwise the Commit's, is refused by core at ``arow`` (`credit.IDS_OUTSIDE`)."""
    auditor = periodic(PHC.draw_k(LAYOUT), files=FILES)
    _bare_table(monkeypatch)
    _y, run = _serve(ZyArm(), remote=True, n=64, served=_served())
    assert auditor.ended() is None and not run.served.refused and run.details["roots1"]["tied"]
    out = _credit(auditor, _record(run))
    assert out["window"] == ("arow", CR.IDS_OUTSIDE) and out["credit"] is None, out


@pytest.mark.parametrize("other", [False, True], ids=["its-session", "another-session"])
def test_round_0s_input_ids_bind_the_windows_session(periodic, monkeypatch, other):
    """ROOTS1's ids@0 binds the window's challenge phase in place of the open root that contains it: core's registration
    (`credit._registration`) holds the served window's entries to it, and refuses an ids@0 bound to another challenge phase."""
    auditor = periodic(PHC.draw_k(LAYOUT), files=FILES)
    if other:
        _other_session(monkeypatch)
    _y, run = _serve(ZyArm(), remote=True, n=64, served=_served())
    assert auditor.ended() is None and not run.served.refused
    c = _record(run)["commitment"]
    why = CR._registration(c["session"]["root"], run.served.program_digest, "h1/", c["served"]["entries"], [],
                           CR.layout_of(_descriptor())[0], 1, [], auditor.session_id)
    assert why == (("registration", CR.other_registration("h1/ids@0")) if other else None), why


def test_the_serving_process_credits_nothing_in_a_window_off_the_commit(auditor):
    """The same zero A with the serving process's own check (`pouw_h1_commit.roots1`): its calls' credit bits are
    false (`UNTIED`), naming the call."""
    _y, run = _serve(EditArm(torch.zeros_like), remote=True, n=64, served=_served())
    assert run.details["roots1"]["reason"].startswith("roots1: round 0's A table h1/a/64@0 of calls [0] is not the Commit's")
    assert {v["reason"] for v in run.gated().values() if not v["credit"]} == {CT.UNTIED}


@pytest.mark.parametrize("edit", ["s_w", "code", "own-key", "unsigned"])
def test_a_window_whose_b_rows_are_not_its_periods_is_refused(periodic, tmp_path, edit):
    """By core's credit (`credit.window`): a period opened with ``--model`` signs the
    auditor's registration of the model, and a window whose served b/<K> differ from that registration's is refused
    (``weights``), though the window is consistent all through (its open root over its own served weights' root) and
    every drawn release is accepted: the registration's checkpoint has one row's s_w, or one code, other than the ones
    the window serves (the window's rows salted with the registration's key, so only the rows differ), or the window
    salted the served rows with the prover's own key (as the served path did before the registry). A window in a period whose open
    names no model, so signs no weights' root, is refused too."""
    reg = _registry(None if edit in ("own-key", "unsigned") else edit)
    auditor = periodic(1, model=edit != "unsigned", files=FILES, registry=reg)
    key = WKEY if edit == "own-key" else _model_key().key
    _y, run = _serve(ZyArm(), remote=True, n=64,
                     served=_served(weights=SV.register_weights(_circuit(), [(64, K)], {0: _rows()}, key)))
    assert auditor.ended() is None and not run.served.refused
    rec = _record(run)
    drawn = _credit(auditor, rec)["drawn"]
    out = _credit(auditor, rec, [_statement(tmp_path / "c0", 0, len(drawn))], run=_verifier(drawn))
    assert out["manifest"] is None and out["credit"] is None, out
    if edit == "unsigned":
        assert out["window"] == ("weights", CR.no_weights(PERIOD)), out
        return
    signed = json.loads(auditor.ledger.lines[0])["body"]["weights_root"]
    assert signed == reg.lines[0]["weights_root"] != rec["commitment"]["weights_root"]
    assert out["window"][0] == "weights" and f"the one period {PERIOD}'s open signed" in out["window"][1], out


@pytest.mark.parametrize("edit", ["s_w", "code"])
def test_a_window_serving_other_weights_than_the_registration_refuses_itself(periodic, edit):
    """The prover's own check: the served path registers its b/<K> at install from the served linear under the
    registration's key (`Served`'s ``registry``, the lane's ``model-key``), and when they fold to another root than the
    registration's (its checkpoint has one row's s_w, or one code, other than the served one) the window registers no
    table and credits nothing, naming why."""
    reg = _registry(edit)
    auditor = periodic(1, files=FILES, registry=reg)
    _y, run = _serve(ZyArm(), remote=True, n=64, served=_served(registry=SV.model_key(reg.key(CH.MODEL, _circuit()))))
    assert auditor.ended() is None
    assert any(f"the served b/<K> are not the auditor's registration of {CH.MODEL}" in r for r in run.served.refused)
    with pytest.raises(ValueError, match="trees were refused"):
        SV.trees(run)
    assert _credit(auditor, _record(run))["credit"] is None


def test_a_period_opened_with_a_bare_weights_root_is_refused_at_the_open(tmp_path):
    """A period's weights' root is the auditor's registration, never the prover's. The gate's ``period-open``
    with ``--weights-root`` the prover's own root (its served b/<K> under its own key) is refused before
    anything is signed, by the in-process ledger and gate alike (`models.opened`); naming the registration's
    own root signs it with its model, as ``--model`` does."""
    M = CH.models()
    own = SV.register_weights(_circuit(), [(64, K)], {0: _rows()}, WKEY).root().hex()
    ledger = CH.Ledger(tmp_path / "auditor.jsonl", 1, registry=_registry())
    with pytest.raises(M.Refused, match="a period's weights' root is the auditor's registration"):
        ledger.period_open(PERIOD, circuit=_circuit(), draw_k=1, weights_root=own)
    assert ledger.lines == []
    private, key = CH.challenger_key(2)
    gate = CH.Gate(private, key, registry=_registry())
    with pytest.raises(M.Refused, match="never the prover's"):
        gate.period_open(PERIOD, circuit=_circuit(), draw_k=1, weights_root=own)
    assert gate.lines == []
    W = _model_key().weights_root
    body = gate.period_open(PERIOD, circuit=_circuit(), draw_k=1, weights_root=W)["body"]
    assert body["weights_root"] == W and body[CR.MODEL_FIELD]["name"] == CH.MODEL


def test_the_in_process_gate_registers_the_model_and_hands_the_lane_its_key():
    """`challenger_host.Gate` has the model registry's verbs. ``model-register`` commits the checkpoint's
    b/<K> under a fresh auditor key; ``model-key`` gives the lane that key, under which the served path's own
    registration of the same rows (`register_weights`) is the registration, root for root; ``period-open --model``
    signs its weights' root and model, which core's credit reads (`credit.committed_weights`). A model it has no
    registration of has no key, one registered twice under the circuit needs ``--weights-root``, and an answer whose
    key isn't its ``key_sha512``'s is refused by the served path."""
    private, key = CH.challenger_key(2)
    gate = CH.Gate(private, key)
    codes, scales = _rows()
    reg = gate.model_register(CH.MODEL, _circuit(), _descriptor(), lambda name, k: CH.linear_rows(codes, scales))
    assert (reg["model"], reg["circuit"], reg["prefix"]) == (CH.MODEL, _circuit(), SV.PREFIX)
    answer = gate.model_key(CH.MODEL, _circuit())
    mk = SV.model_key(answer)
    assert (mk.model, mk.circuit, mk.weights_root) == (CH.MODEL, _circuit(), reg["weights_root"])
    ours = SV.register_weights(_circuit(), [(64, K)], {0: (codes, scales)}, mk.key)
    assert ours.root().hex() == reg["weights_root"]
    assert {str(k): r.root.hex() for k, r in ours.registered.items()} == reg["roots"]
    body = gate.period_open(PERIOD, circuit=_circuit(), cap=1, draw_k=1, model=CH.MODEL)["body"]
    assert body["weights_root"] == reg["weights_root"]
    assert body[CR.MODEL_FIELD] == {"name": CH.MODEL, "checkpoint_sha512": CR.PINNED_MODELS[CH.MODEL]}
    assert CR.committed_weights(gate.lines, PERIOD) == (reg["weights_root"], None)
    assert "refused" in gate.model_key("LLAMA31_70B_INT8", _circuit())
    with pytest.raises(ValueError, match="model-key: refused"):
        SV.model_key(gate.model_key("LLAMA31_70B_INT8", _circuit()))
    with pytest.raises(ValueError, match="SHA-512 is the answer's key_sha512"):
        SV.model_key({**answer, "key": "00" * 32})
    again = gate.model_register(CH.MODEL, _circuit(), _descriptor(), lambda name, k: CH.linear_rows(codes, scales))
    assert again["weights_root"] != reg["weights_root"]
    assert "registered 2 times" in gate.model_key(CH.MODEL, _circuit())["refused"]
    assert SV.model_key(gate.model_key(CH.MODEL, _circuit(), reg["weights_root"])).key == mk.key


def test_a_drawn_session_that_opened_no_weights_is_refused(periodic, tmp_path):
    """Each drawn proof block's release must open its K's ``h1/b/<K>`` beside its A, since nothing else ties the B rows
    it proved to the ones the period registered. The ``--registered`` file core hands the verifier for a drawn call is
    vllm's own (`pouw_h1_served.register`'s, the one its prover's releases read), its ``b_k<K>_<j>`` ports at
    ``h1/b/<K>`` at the period's root; a release whose registered finding opened its A and record but no ``b`` port
    refuses the window (``registration``), and so does one that opened no A."""
    auditor = periodic(PHC.draw_k(LAYOUT), files=FILES)
    _y, run = _serve(ZyArm(), remote=True, n=64, served=_served())
    assert auditor.ended() is None
    rec = _record(run)
    drawn = _credit(auditor, rec)["drawn"]
    st, seen = [_statement(tmp_path / "c0", 0, len(drawn))], []
    assert _credit(auditor, rec, st, run=_verifier(drawn, seen=seen), live=_live(tmp_path, rec))["credit"] == [0]
    ports = seen[0]["registered"]["ports"]
    b = {p: e for p, e in ports.items() if p.startswith(f"b_k{K}_")}
    assert sorted(b) == sorted(f"b_k{K}_{j}" for j in range(32))
    assert {(e["value"], e["root"]) for e in b.values()} == {(SV.PREFIX + f"b/{K}", _period_weights().entries()[0]["root"])}
    own = SV.register(run).files[0]["ports"]
    assert {p: {x: e[x] for x in ("value", "root", "domain", "reads")} for p, e in ports.items()} == {
        p: {x: e[x] for x in ("value", "root", "domain", "reads")} for p, e in own.items()}
    no_b = _credit(auditor, rec, st, run=_verifier(drawn, opened=lambda p: not p.startswith("b_k")))
    assert no_b["window"][0] == "registration" and f"opened no weights {SV.PREFIX}b/{K}" in no_b["window"][1], no_b
    assert no_b["credit"] is None
    no_a = _credit(auditor, rec, st, run=_verifier(drawn, opened=lambda p: not p.startswith("a_k")))
    assert no_a["window"][0] == "registration" and "opened no A value" in no_a["window"][1], no_a


def test_a_window_is_credited_only_from_its_statements_proof_files_verified_by_credit(periodic, tmp_path):
    """Core's `credit.window` verifies each drawn call's statement itself (`proofs.verify`, here a stand-in that
    reports what it was handed) on the circuit file the period's open committed for its K, against the ``--registered``
    file credit builds from the window's entries (the call tables' domains bound to the window's open root, the
    weights' to none), and credits the gate-passed call only when every drawn proof block's release is accepted and each
    round's live@<s> proves its step's served rows: none, a rejected one, another circuit's descriptor, or no live
    evidence refuses the whole window. The closed period's credit
    (`period_credit`) is that window's, its period opened with ``--model`` (the auditor's registration, whose
    key salted the window's b/<K>), and a window whose ``b/<K>`` aren't the period's is refused there too."""
    from verity.core.protocols.sampled_proofs.one_stage import registered as G
    k = PHC.draw_k(LAYOUT)
    auditor = periodic(k, files=FILES)
    _y, run = _serve(ZyArm(), remote=True, n=64, served=_served())
    assert auditor.ended() is None
    rec = _record(run)
    out = _credit(auditor, rec)
    assert out["window"][0] == "missing" and out["credit"] is None, out
    drawn = out["drawn"]
    st, seen, live = [_statement(tmp_path / "c0", 0, len(drawn))], [], _live(tmp_path, rec)
    assert _credit(auditor, rec, st, run=_verifier(drawn))["window"] == ("live", CR.LIVE_STATEMENTS_FORM)
    out = _credit(auditor, rec, st, run=_verifier(drawn, seen=seen), live=live)
    assert out["manifest"] is None and out["window"] is None, out
    served = [row[CR.SERVED_ROWS] for row in rec["calls"]]
    eps = CR.established(LAYOUT, [0], served, k)
    assert out["calls"] == out["credit"] == [0] and out["eps"] == (None if eps is None else str(eps))
    assert 0 < served[0] < LAYOUT["m"]
    assert CR.credited_work(LAYOUT, [0], served) == served[0] * LAYOUT["linears"][0][1] * LAYOUT["linears"][0][2]
    (got,) = seen
    assert got["circuit_sha512"] == FILES[f"k{K}"]
    root, digest = auditor.window["root"], _circuit()
    for port, e in got["registered"]["ports"].items():
        held = CR.is_held(e["value"])
        assert e["domain"] == G.domain(digest, e["value"], e["words"], e["leaves"], window=None if held else root
                                       ).domain_id.hex(), port
    failed = _credit(auditor, rec, st, run=_verifier(drawn, reject=drawn[-1:]))
    assert failed["window"][0] == "failed" and failed["credit"] is None
    other = _descriptor(((64, K), (32, K)), ("q", "k"))
    assert _credit(auditor, rec, st, descriptor=other, run=_verifier(drawn))["credit"] is None
    auditor.ledger.period_close(PERIOD)
    ledger = PHC.Ledger.load([auditor.ledger.path], auditor.key, pins=[auditor.ledger.pins_path], main=CH.MAIN)
    evidence = PHC.window_evidence(rec, auditor.directory, auditor.window, st, _descriptor(), live)
    got, why = PHC.period_credit(ledger, PERIOD, [evidence], run=_verifier(drawn))
    assert why is None and got == {"windows": [[0]], "work": [CR.credited_work(LAYOUT, [0], served)], "k": k,
                                   "delta": str(CR.DELTA), "eps": [out["eps"]]}, why
    unpinned = PHC.Ledger.load([auditor.ledger.path], auditor.key, main=CH.MAIN)
    assert PHC.period_credit(unpinned, PERIOD, [evidence], run=_verifier(drawn))[1] == (
        "circuit", f"window 0 of period {PERIOD} doesn't count toward it, so the period earns nothing: {LC.NO_PINS}")
    off_main = PHC.Ledger.load([auditor.ledger.path], auditor.key, pins=[auditor.ledger.pins_path], main=CH.MAIN[:1])
    assert PHC.period_credit(off_main, PERIOD, [evidence], run=_verifier(drawn))[1] == (
        "circuit", f"window 0 of period {PERIOD} doesn't count toward it, so the period earns nothing: "
                   f"{LC.first_parent_reason(auditor.session_id, CH.PINNED, list(CH.MAIN[:1]))}")
    assert PHC.period_credit(ledger, PERIOD, [{**evidence, "descriptor": other.hex()}], run=_verifier(drawn))[1][0] == \
        "circuit"
    b = SV.register_weights(_circuit(), [(64, K)], {0: _rows("s_w")}, _model_key().key).registered[K].entry()
    forged = {**evidence, "entries": [b if CR.is_held(e["value"]) else e for e in evidence["entries"]]}
    refused = PHC.period_credit(ledger, PERIOD, [forged], run=_verifier(drawn))[1]
    assert refused[0] == "window" and refused[1].startswith(f"window 0 of period {PERIOD} is refused, so the period "
                                                            "earns nothing: weights: "), refused


def test_a_period_open_without_draw_k_is_refused(periodic):
    """The proof system commits the draw's K in the period's open (`credit.K_FIELD`) before any draw; a period
    opened without one leaves its windows no K to be drawn with, so the window, the period's reading and its credit are
    each refused (``budget``), and a window in no period has no committed K either."""
    auditor = periodic(None, files=FILES)
    _y, run = _serve(ZyArm(), remote=True, n=64, served=_served())
    assert auditor.ended() is None
    rec = _record(run)
    out = _credit(auditor, rec)
    assert out["k"] is None and out["credit"] is None
    assert out["window"][0] == "budget" and "commits no draw K" in out["window"][1]
    auditor.ledger.period_close(PERIOD)
    ledger = PHC.Ledger.load([auditor.ledger.path], auditor.key, pins=[auditor.ledger.pins_path], main=CH.MAIN)
    reasons = PHC.period_reason(ledger, PERIOD, [auditor.session_id], [auditor.window["root"]])
    assert len(reasons) == 1 and reasons[0].startswith(f"budget: period {PERIOD}'s open commits no draw K"), reasons
    evidence = PHC.window_evidence(rec, auditor.directory, auditor.window, [], _descriptor())
    assert PHC.period_credit(ledger, PERIOD, [evidence])[1][0] == "budget"
    with pytest.raises(ValueError, match="layout"):
        PHC.draw_k(None)


def test_a_window_in_no_period_has_no_committed_k(auditor):
    """A window whose challenge phase the auditor's ledger holds in no period is drawn with no K any open committed."""
    _y, run = _serve(ZyArm(), remote=True, n=64, served=_served())
    assert auditor.ended() is None
    out = _credit(auditor, _record(run))
    assert out["window"][0] == "budget" and "in no period" in out["window"][1]
    assert out["credit"] is None


def test_the_drawn_set_from_a_fixed_coin_is_cores():
    """The proof blocks vllm's verifier reads as drawn (`drawn_of`) are core's (`credit.drawn`, and
    `core_check.work_units` over `credit.strata` under the work law at K), over the layout core reads from the declared
    Program's descriptor (`credit.layout_of`, which `pouw_h1_served.declared_layout` is without building it): the
    population is the proof blocks of ``h1window.program(linears, calls, m, tile)`` in its order, one stratum per K, on the
    stream of the window's root, the registration of its draw root and the receipt of its last round's receipts."""
    shapes, modules = ((64, 64), (32, 128)), ("q", "o")
    desc = _descriptor(shapes, modules, 64, 2)
    layout, why = CR.layout_of(desc)
    assert why is None and layout == SV.declared_layout(list(shapes), dict(enumerate(modules)), 64, 2)
    coin, root, reg, receipts = (hashlib.sha512(x).hexdigest() for x in (b"coin", b"window", b"draw", b"receipts"))
    opening = LiveOpening(0, Wd.DRAW, 0, bytes(64), bytes.fromhex(coin), bytes(16), ()).to_json()
    record = {"commitment": {"draw": {"coin": opening, "root": reg, "receipts": receipts}}}
    ss, why = CR.strata(layout)
    assert why is None and [(s["template"], s["units"], s["work"]) for s in ss] == [
        ("k64", [[0, 4], [6, 10]], 32 * 32 * 64), ("k128", [[4, 6], [10, 12]], 32 * 32 * 128)]
    for k in (1, 6, 12):
        assert CR.law(ss, k) is None
        want = CC.work_units(bytes.fromhex(coin), root, reg, {"receipts": receipts}, ss)
        got, why = PHC.drawn_of(record, {"root": root}, k, desc)
        assert why is None and got == want == CR.drawn(coin, root, reg, {"receipts": receipts}, layout, k)[0]
        assert len(got) == len(set(got)) and all(0 <= u < 12 for u in got)
    assert set(PHC.drawn_of(record, {"root": root}, 12, desc)[0]) >= {4, 5, 10, 11}
    assert PHC.drawn_of(record, {"root": root}, 13, desc)[1][0] == "budget"
    assert PHC.drawn_of(record, {"root": root}, 1, None)[1] == ("layout", CR.DESCRIPTOR_HEX)
    assert PHC.draw_k(layout) == 12


def test_the_period_k_is_sized_for_the_expected_credited_share_and_the_verifier_reports_its_eps():
    """The proof system sizes the K a period's open commits (`draw_k`) for the share of a window's work it
    expects credited (`EXPECTED_CREDIT`, 8/9) at `DRAW_EPS` and δ = 2^-40, through `window.k_prime`'s own inputs, since
    that share isn't known at the open: 31,178 for the declared 8B window, which at 8/9 credited establishes at most
    0.1% (`window.eps_at`, what `credit.established` reports), and K sized as if all were credited establishes less."""
    from tests.protocols import pouw_h1_rehearsal as RH
    from verity_vllm.protocols.pouw import h1 as PH
    shapes, modules = RH.declaration()
    layout = SV.declared_layout(shapes, modules, PH.FILL.rows, PH.FILL.steps)
    k = RH.declared_draw_k()
    assert k == PHC.draw_k(layout) == 31178
    work = sum(size * w for size, w in CR.call_work(layout))
    assert Wd.eps_at(k, CR.DELTA, work * 8 // 9, work, 1) <= PHC.DRAW_EPS
    assert Wd.eps_at(PHC.draw_k(layout, credited=1), CR.DELTA, work * 8 // 9, work, 1) > PHC.DRAW_EPS


def _two_window_period(tmp_path):
    """Lane vllm's closed period `PERIOD` of two windows, each reserved, opened on its root and ended complete, as the
    auditor's ledger (its record leaves between each open's and end's), and their challenge phases and roots, in order."""
    leaves = itertools.count()
    ledger = CH.Ledger(tmp_path / "w.jsonl", 1, leaves)
    ledger.period_open(PERIOD, circuit=_circuit(), cap=2, draw_k=1)
    sessions, roots = [], []
    for i in (1, 2):
        s, root = CH.session_id(i), hashlib.sha512(bytes([i])).hexdigest()
        ledger.reserve(s)
        ledger.open(s, {"root": root})
        d = tmp_path / s
        d.mkdir()
        (d / R.RECORD).write_text(json.dumps({"signature": f"{next(leaves):08x}", "commitment": {"root": "ab" * 64}}))
        (d / R.HELLOS).write_text("")
        ledger.end(s, d, True)
        sessions.append(s)
        roots.append(root)
    ledger.period_close(PERIOD)
    return PHC.Ledger.of([ledger.lines], ledger.key, pins=[CH.PINS], main=CH.MAIN), sessions, roots


def test_a_window_presented_outside_its_periods_close_or_out_of_order_is_refused(tmp_path):
    """A period earns only as its close lists it (`credit.period`): exactly its windows, each once, in the
    close's order. Presented out of order, with one missing, with one past the close, or with a window not its own, the
    period is refused, by `period_reason` and by `period_credit` before any window is read."""
    ledger, sessions, (a, b) = _two_window_period(tmp_path)
    other = hashlib.sha512(b"another window").hexdigest()
    assert PHC.period_reason(ledger, PERIOD, sessions, [a, b]) == []

    def why(presented):
        (reason,) = PHC.period_reason(ledger, PERIOD, sessions, presented)
        return reason
    assert why([b, a]).startswith(f"order: window 0 of period {PERIOD} is {b[:16]}...")
    assert why([a]).startswith(f"missing: window 1 of period {PERIOD}")
    assert why([a, b, a]).startswith("period: the windows presented run past the 2")
    assert why([other, a, b]).startswith(f"period: window {other[:16]}... is not one of period {PERIOD}'s windows")
    stub = dict.fromkeys(CR.PERIOD_KEYS)
    got, refused = PHC.period_credit(ledger, PERIOD, [{**stub, "window": {"root": b}}, {**stub, "window": {"root": a}}])
    assert got is None and refused[0] == "order"
    assert PHC.period_credit(ledger, PERIOD, [{**stub, "window": {"root": a}}])[1][0] == "missing"


def test_a_credit_bit_changed_after_the_draw_coin_is_refused(periodic):
    """The credit bit is under calls_root and so under the draw root the draw coin followed: flipping it after
    the draw is refused, whether the draw root is left (it is no longer over the manifest) or recomputed (it is no longer
    the one root the challenger's log shows its draw coin followed); the honest record gets past them to its proofs."""
    auditor = periodic(1, files=FILES)
    _y, run = _serve(ZyArm(), remote=True, n=64, served=_served())
    assert auditor.ended() is None
    rec = _record(run)
    rows = [{**rec["calls"][0], CR.CREDIT: False}]
    c = rec["commitment"]

    def credited(commitment, calls):
        return _credit(auditor, {"commitment": commitment, "calls": calls})["window"]
    assert credited(c, rec["calls"])[0] == "missing"
    assert credited(c, rows) == ("draw", CR.DRAW_ROOT)
    salt = PHC.program_salt_of(c)
    served = SV.digest_of(c["served"]["entries"], c["served"]["refused"])
    root = CR.draw_root(CR.calls_root(salt, rows, served, CT.digest(c["commit"])), bytes.fromhex(c["draw"]["receipts"]),
                        c["rounds"])
    assert credited({**c, "draw": {**c["draw"], "root": root.hex()}}, rows) == ("draw", CR.CHANGED)
    assert credited(c, [{k: v for k, v in rec["calls"][0].items() if k != CR.CREDIT}]) == ("credit", CR.no_bit(0))
    no_z = [{**rec["calls"][0], "tables": {**rec["calls"][0]["tables"], "z": None}}]
    assert credited(c, no_z) == ("credit", CR.no_z(0))


def test_the_salt_key_and_a_calls_values_outlive_the_close_in_the_provers_directory_only(tmp_path):
    """DRAWN1: with a prover directory the run keeps its salt key and each served call's A rows, scales and z/y records
    there, owner-only, past the close; the run's published output (its transcript) holds neither the key's bytes nor
    its hex; and the call's A and z tables rebuilt from them have the roots its manifest row carries, so a drawn proof block
    opens after the close, while a kept row changed afterwards is refused."""
    import stat
    prover, out, key = tmp_path / "prover", tmp_path / "out", bytes(range(32))
    salt = hashlib.sha512(b"salt").digest()
    rounds = LiveRecord(Wd.live_challenger(0, 1, random.Random(1).randbytes), Wd.ROUND, clock=lambda: 0.0)
    run = PH.H1Run("t", salt, {0: "q"}, [(64, K)], ROWS, 1, transcript_dir=str(out), rounds=rounds, key=key,
                   prover=str(prover))
    layer = Linear(64, 1, None)
    x = (torch.randn(3, K, generator=torch.Generator().manual_seed(9)) * 3).bfloat16()
    xq, s_a = int8_ops.row_quant(x)
    _y, zy = int8_ops.scaled_mm_zy(xq, s_a, layer.weight, layer.weight_scale, ROWS)
    run.record(0, xq, s_a, layer.weight, layer.weight_scale, zy)
    run.finish()
    del run
    assert PV.key_of(prover) == key
    assert stat.S_IMODE(prover.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in prover.rglob("*") if p.is_file())
    published = [p for p in out.rglob("*") if p.is_file()]
    assert published and not any(key in p.read_bytes() or key.hex().encode() in p.read_bytes() for p in published)
    record = json.loads((out / PH.CALLS_FILE).read_text())
    assert PV.locate(record["calls"], 1) == (0, 1)
    opened = PV.open_call(prover, record, 0, salt)
    assert opened.a.commitment.root.hex() == record["calls"][0]["tables"]["a"]
    assert opened.z.commitment.root.hex() == record["calls"][0]["tables"]["z"]
    assert opened.z.open(1).position == 1
    kept = prover / PV.CALLS / "0.npz"
    with np.load(kept) as f:
        a, s, r = f["a"].copy(), f["s_a"], f["records"]
    a[0, 0] ^= 1
    PV.keep_call(prover, 0, a, s, r)
    with pytest.raises(ValueError, match="call 0's a table, rebuilt from the prover's directory, is not the one"):
        PV.open_call(prover, record, 0, salt)
