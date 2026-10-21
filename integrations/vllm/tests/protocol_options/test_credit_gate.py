"""`protocol_options.credit_gate` and its place in H1's credit (`pouw_h1_hooks.traced`, `H1Run.judge`, `H1Run.credit`): a served
call is credited only if the y the linear hands back is the credited kernel's own output, one GEMM invocation writing z and
y, no other GEMM ran for it, and its z is its committed z/y records, every unit's y dq of its z and the served y theirs
(`pouw_h1_records.reason`).  vllm's int8 arm under H1 (``zy_rows`` set, ``verity_int8::scaled_mm_zy`` over T slots) is
credited; the arm serving ``scaled_mm``'s y with z a second ``_int_mm`` beside it is refused with its named reason, and so
is every way of handing over a z or records the served kernel didn't write for the served y."""
from __future__ import annotations

import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw import credit as CR  # noqa: E402
from verity.core.protocols.pouw import window as Wd  # noqa: E402
from verity.core.protocols.pouw.coins import Challenger  # noqa: E402
from verity.core.protocols.sampled_proofs import remote as R  # noqa: E402
from verity.core.protocols.sampled_proofs.challenger import LiveRecord  # noqa: E402
from verity_vllm import protocol_options as PO  # noqa: E402
from verity_vllm.engine import int8_exact as E  # noqa: E402
from verity_vllm.engine import int8_ops  # noqa: E402
from verity_vllm.protocol_options import credit_gate as CG  # noqa: E402
from verity_vllm.protocol_options import pouw  # noqa: E402
from verity_vllm.protocol_options import pouw_h1 as PH  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_challenger as PHC  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_prover as PV  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_records as ZR  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_requests as RQ  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_served as SV  # noqa: E402

from tests.protocol_options import challenger_host as CH  # noqa: E402

K = 64
#: the window's T
ROWS = 32
ENTRY = f"pouw:{PH.SCHEME}"
ZY = "verity_int8::scaled_mm_zy"
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
#: the gate's reserve answer for the window's session, and the gate that opens it on the window's open root
RESERVED = {"session": CH.session_id(1)}
GATE: dict = {}


@pytest.fixture
def auditor(tmp_path, monkeypatch):
    """The auditor host's challenger, one session over a one-round live epoch, reached at `AUDITOR`, reserved as
    `RESERVED` and opened by `GATE`'s ``open``."""
    host = CH.Host(tmp_path / "auditor", Wd.live_challenger(0, 1, CH.entropy(1)), 1, gated=True)
    monkeypatch.setattr(R.RemoteChallenger, "connect",
                        classmethod(lambda cls, at, secret=None: host.client if at == AUDITOR else None))
    monkeypatch.setitem(GATE, "open", host.open)
    return host


def _serve(arm, retain=1, m=3, prompt=True, remote=False, n=32, exact=True, served=None):
    """One small call (m rows of K, N = `n`) through `arm` with H1 installed (T = `ROWS`), its request admitted with an
    m-token prompt (with none: prompt embeddings, so no entry); the run closed and its y and run returned, the served y
    the plain linear's when `exact`. Its coins are the auditor host's challenger's with `remote` (the `auditor` fixture),
    else held in this process; with `served` (`pouw_h1_served.Served`) its rows are the window Program's values."""
    layer = Linear(n, 1, arm)
    runner = SimpleNamespace(execute_model=lambda scheduler_output, *a, **kw: None, decode_query_len=1)
    if remote:
        coins = {"challenger": AUDITOR, "session": RESERVED, "gate": lambda session, root: GATE["open"](session, root)}
    else:
        coins = {"coins": Challenger(0, random.Random(0).randbytes),
                 "rounds": LiveRecord(Wd.live_challenger(0, 1, random.Random(1).randbytes), Wd.ROUND, clock=lambda: 0.0)}
    ctx = PO.Context(stage="serve", enabled=(ENTRY,), settings={"pouw": {"rows": ROWS, "steps": 1, "retain_steps": retain,
                                                                         "runner": runner, **coins,
                                                                         **({"served": served} if served else {})}})
    comp = PO.install(Model(layer), ctx)
    try:
        req = SimpleNamespace(req_id="r", prompt_token_ids=list(range(m)) if prompt else None,
                              sampling_params=SimpleNamespace(max_tokens=1))
        runner.execute_model(SimpleNamespace(scheduled_new_reqs=[req], num_scheduled_tokens={"r": m},
                                             total_num_scheduled_tokens=m))
        x = (torch.randn(m, K, generator=torch.Generator().manual_seed(9)) * 3).bfloat16()
        y = layer.quant_method.apply(layer, x)
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
    bit the plain linear's) and hands over the records, which are the call's z table, one leaf a unit; the call is
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
    """``scaled_mm_zy`` writes whole 32-column units only, so a layer of N = 40 keeps ``scaled_mm``, commits no z table and
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
    """SERVE-H1's run tree (cursor/serve-h1-run-ea24, `pouw_h1.served`): the arm's apply serves y, then the wrapper
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
    """RT4-INT-21: the call passes the gate, but the window's coins were held in the serving process (no pouw setting
    'challenger'), so no verifier trusts them and the window credits nothing, naming why."""
    _y, run = _serve(FusedArm())
    (v,) = run.verdicts.values()
    assert v["credit"] and run.coin_source == "in-process"
    c = run.credit()
    assert c["gate_passed"] == {"rows": 0, "macs": 0} and c["coins"] == "in-process"
    assert c["refused"] == {"calls": 1, "rows": 3, "macs": 3 * 32 * K, "reasons": {PH.IN_PROCESS: 1}}


def test_a_credited_call_in_a_window_serving_a_request_without_an_entry_credits_nothing():
    """Review item 3 on #2916: the call passes the gate, but its rows are a request's with no entry (prompt embeddings), so
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
    """Byte offset in a unit's record of its cell (i, j)'s z word (or y half-word with `y`)."""
    return (4 * 32 * 32 + 2 * (32 * i + j)) if y else 4 * (32 * i + j)


def _xor(zy, u, off, bit):
    zy[u, off] ^= bit
    return zy


def _another_calls(y, zy, xq, s_a, w, s_w):
    return y, E.scaled_mm_zy(torch.roll(xq, 1, 0), torch.roll(s_a, 1, 0), w, s_w, ROWS)[1]


def _consistent_but_not_served(y, zy, xq, s_a, w, s_w):
    """Unit 0's z with one served cell moved and its y redone as dq of it, so the records hold together but aren't what
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
    their z at the call's scales (a flipped y bit, a moved z word, two units swapped, another call's records), whose y
    isn't the served y, or with a z on an empty slot: the call is refused as ``z-records``, naming the unit."""
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


def test_the_gate_passed_call_is_credited_by_cores_rules_only_with_every_drawn_units_proof_accepted(auditor):
    """PRF-132: the gate-passed call's manifest row carries its credit bit and its z root under calls_root, so core's
    `credited` (through the auditor's ledger and the session's files) names it; the window credits nothing while a
    drawn unit's proof outcome is missing or rejected (`outcome`), and nothing until every A value is the round roots'
    tree (`a_link`): a window with no served trees has none."""
    _y, run = _serve(ZyArm(), remote=True, n=64)
    assert auditor.ended() is None
    rec = _record(run)
    assert [(r["index"], r[CR.CREDIT]) for r in rec["calls"]] == [(0, True)] and rec["calls"][0]["tables"]["z"]

    def verdict(drawn=None, proofs=None):
        return PHC.credit_reason(rec, auditor.directory, [auditor.ledger.path], auditor.window, auditor.key, drawn, proofs)
    out = verdict()
    assert out["manifest"] is None and out["credited"] is None and out["calls"] == [0], out
    assert out["outcome"] == ("outcome", CR.DRAWN_FORM) and out["credit"] is None
    missing = verdict([0, 1], [{"unit": 0, "verdict": "accept"}])
    assert missing["outcome"][0] == "missing" and "drawn unit 1 has no proof outcome" in missing["outcome"][1]
    assert missing["credit"] is None
    failed = verdict([0, 1], [{"unit": 0, "verdict": "accept"}, {"unit": 1, "verdict": "reject"}])
    assert failed["outcome"][0] == "failed" and failed["credit"] is None
    accepted = verdict([0, 1], [{"unit": 0, "verdict": "accept"}, {"unit": 1, "verdict": "accept"}])
    assert accepted["outcome"] is None and accepted["credited"] is None
    assert accepted["a_link"] == ("a-link", CR.ENTRIES_FORM) and accepted["credit"] is None


def test_an_honest_served_window_is_credited_through_cores_credited_a_link_and_outcome(auditor):
    """PRF-132: a served window (its A codes and B rows the window Program's values, each round's root over its A rows)
    whose call gate-passed is credited by core's rules once its drawn unit's proof is accepted: `credited` names the
    call, `a_link` holds the registered A tree to the round roots, and `outcome` accepts; the record's manifest holds."""
    from tests.protocol_options.test_pouw_h1_served import _unit
    _y, run = _serve(ZyArm(), remote=True, n=64, served=SV.Served(PH.TILE, _unit()))
    assert auditor.ended() is None and not run.served.refused and run.served.registered
    rec = _record(run)
    assert any(CR.is_a(e["value"]) for e in rec["commitment"]["served"]["entries"])
    out = PHC.credit_reason(rec, auditor.directory, [auditor.ledger.path], auditor.window, auditor.key, [0],
                            [{"unit": 0, "verdict": "accept"}])
    assert {k: out[k] for k in ("manifest", "credited", "a_link", "outcome")} == dict.fromkeys(
        ("manifest", "credited", "a_link", "outcome")), out
    assert out["calls"] == out["credit"] == [0]
    assert PHC.credit_reason(rec, auditor.directory, [auditor.ledger.path], auditor.window, auditor.key, [0], [])["credit"] is None


def test_a_credit_bit_changed_after_the_draw_coin_is_refused(auditor):
    """PRF-132: the credit bit is under calls_root and so under the draw root the draw coin followed: flipping it after
    the draw is refused, whether the draw root is left (it is no longer over the manifest) or recomputed (it is no longer
    the one root the challenger's log shows its draw coin followed)."""
    _y, run = _serve(ZyArm(), remote=True, n=64)
    assert auditor.ended() is None
    rec = _record(run)
    rows = [{**rec["calls"][0], CR.CREDIT: False}]
    c = rec["commitment"]

    def credited(commitment, calls):
        return PHC.credit_reason({"commitment": commitment, "calls": calls}, auditor.directory, [auditor.ledger.path],
                                 auditor.window, auditor.key)["credited"]
    assert credited(c, rec["calls"]) is None
    assert credited(c, rows) == ("draw", CR.DRAW_ROOT)
    salt = PHC.program_salt_of(c)
    root = CR.draw_root(CR.calls_root(salt, rows), bytes.fromhex(c["draw"]["receipts"]), c["rounds"])
    assert credited({**c, "draw": {**c["draw"], "root": root.hex()}}, rows) == ("draw", CR.CHANGED)
    assert credited(c, [{k: v for k, v in rec["calls"][0].items() if k != CR.CREDIT}]) == ("credit", CR.no_bit(0))
    no_z = [{**rec["calls"][0], "tables": {**rec["calls"][0]["tables"], "z": None}}]
    assert credited(c, no_z) == ("credit", CR.no_z(0))


def test_the_salt_key_and_a_calls_values_outlive_the_close_in_the_provers_directory_only(tmp_path):
    """DRAWN1: with a prover directory the run keeps its salt key and each served call's A rows, scales and z/y records
    there, owner-only, past the close; the run's published output (its transcript) holds neither the key's bytes nor
    its hex; and the call's A and z tables rebuilt from them have the roots its manifest row carries, so a drawn unit
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
