"""The PoUW bench harness's rules, on a fake device with a virtual clock (no GPU): nothing is timed before its gate passes in
the run, a failed gate times nothing, every rep is kept in a seeded random order, the negative controls stop the run, the
consistency gate rejects a hash-free call faster than the plain GEMM allows, and the report's arithmetic."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "harness"))

import arm as A  # noqa: E402
import baselines as B  # noqa: E402
import bench as H  # noqa: E402
import chain as CH  # noqa: E402
import derive as DV  # noqa: E402
import fill as F  # noqa: E402
import identity as I  # noqa: E402
import report as R  # noqa: E402
import schedule as S  # noqa: E402
import shapes as SH  # noqa: E402

UUID5 = "GPU-0a1b2c3d-0000-1111-2222-333344445555"


# ---- a fake device: virtual time, byte buffers, graphs that replay what was enqueued -----------------------------------------


class FakeGraph:
    def __init__(self, work):
        self.work = work


class FakeDevice:
    stream = 1
    l2 = 128 << 20

    def __init__(self):
        self.t, self.mem, self.events, self.log = 0.0, {}, {}, []
        self._next, self._capture = 0x10000, None

    def enqueue(self, fn):
        (self._capture.append if self._capture is not None else lambda f: f())(fn)

    def alloc(self, n):
        ptr = self._next
        self._next += n + 4096
        self.mem[ptr] = bytearray(n)
        return A.Buffer(ptr, n)

    def free(self, buf):
        self.mem.pop(buf.ptr, None)

    def flush(self, stream=None):
        self.log.append("flush")
        self.t += 0.25

    write_flush = flush

    def event(self):
        e = len(self.events) + 1
        self.events[e] = None
        return e

    def record(self, e, stream=None):
        self.enqueue(lambda: self.events.__setitem__(e, self.t))
        self.log.append("record")

    def elapsed_ms(self, a, b):
        return self.events[b] - self.events[a]

    def capture(self, body):
        self._capture = []
        try:
            body(self.stream)
        finally:
            work, self._capture = self._capture, None
        return FakeGraph(work)

    def graph_launch(self, g, stream=None):
        for fn in g.work:
            fn()

    def graph_destroy(self, g):
        pass

    def sync(self):
        pass

    def xor_byte(self, ptr, off, mask=1, stream=None):
        self.mem[ptr][off] ^= mask

    def read(self, ptr, n):
        return bytes(self.mem[ptr][:n])

    def write(self, ptr, data):
        self.mem[ptr][:len(data)] = data

    # the chain's: derive by the twin (0.1 ms), events, a side stream that runs in order on the virtual clock
    side_stream = 2

    def derive(self, src, y, m, n, kind, a, sf, k, stream=None):
        self.log.append(("derive", y, a, stream))

        def run():
            name = next(x for x, d in DV.KINDS.items() if d.id == kind)
            s = next(x for x, i in DV.SRC.items() if i == src)
            ab, scale, offs = DV.derive(name, s, bytes(self.mem[y][:m * n * CH.Y_BYTES[s]]), m, n, k)
            self.mem[a][:len(ab)] = ab
            if scale is not None:
                np.frombuffer(self.mem[sf], np.uint8)[offs] = scale
            self.t += 0.1
        self.enqueue(run)

    def wait(self, stream, e):
        self.log.append(("wait", stream, e))

    def event_destroy(self, e):
        self.events.pop(e, None)


class FakeSensors:
    def sample(self):
        return {"sm_mhz": 2400, "mem_mhz": 14000, "temp_c": 60, "power_w": 300.0, "energy_mj": 0, "throttle": "0x0"}


class FakeCall:
    """Costs `ms` of virtual time per launch and writes `pattern` into its output; `weight_ms` per weight side."""

    def __init__(self, dev, ms, pattern=b"\x5a" * 32, weight_ms=0.0):
        self.dev, self.ms, self.pattern, self.weight_ms = dev, ms, pattern, weight_ms
        self.outputs = {"root": dev.alloc(len(pattern))}
        self.freed = False

    def _run(self):
        self.dev.t += self.ms
        self.dev.mem[self.outputs["root"].ptr][:] = self.pattern

    def launch(self, stream):
        self.dev.enqueue(self._run)

    def weight_side(self, stream):
        self.dev.enqueue(lambda: setattr(self.dev, "t", self.dev.t + self.weight_ms))

    def free(self):
        self.freed = True


class FakeArm:
    """An arm whose gate checks its root in full; `lazy` accepts anything, `failing` roles fail, `bad_ok` accepts its
    known-bad variant."""

    name, stream, precision, inputs = "fake-arm", "fp8", "fp8-e4m3", "e4m3"

    def __init__(self, hash_ms=1.8, nohash_ms=1.2, weight_ms=0.5, lazy=False, failing=(), bad_ok=False, macs=None):
        self.flag_sets = {"hash": frozenset({"h"}), "nohash": frozenset()}
        self.ms = {"hash": hash_ms, "nohash": nohash_ms}
        self.weight_ms, self.lazy, self.failing, self.bad_ok, self.macs = weight_ms, lazy, set(failing), bad_ok, macs

    def build(self, ctx):
        return {}

    def domain(self, shape):
        return None

    def work(self, shape):
        return A.Work(tensor_macs={"fp8-e4m3": self.macs or shape.mkn}, w_ref=str(shape.mkn), credited=str(shape.mkn // 2),
                      unit="mac", hashed_bytes={"sha256": 1 << 20}, weight_hashed_bytes={"sha256": 1 << 10},
                      gamma={"bound": "0.005"})

    def prepare(self, dev, shape, flags, operands):
        role = "hash" if "h" in flags else "nohash"
        return FakeCall(dev, self.ms[role], weight_ms=self.weight_ms)

    def gate(self, dev, call, shape, flags):
        role = "hash" if "h" in flags else "nohash"
        ok = self.lazy or (role not in self.failing and dev.read(call.outputs["root"].ptr, 32) == b"\x5a" * 32)
        return A.Gate(ok, {"root": "all"})

    def known_bad(self, dev, shape, operands):
        return FakeCall(dev, 1.0, pattern=b"\x5a" * 32 if self.bad_ok else b"\x00" * 32)


SHAPE = A.Shape("m8192-n8192-k8192", 8192, 8192, 8192)


def opsets(dev, n=3):
    return [A.OperandSet(i, i, i, dev.alloc(64), dev.alloc(64)) for i in range(n)]


def bench(reps=S.MIN_REPS, methods=H.METHODS):
    dev = FakeDevice()
    return H.Bench(dev, FakeSensors(), reps, methods), dev


def entry(dev, key, ms, sets=3):
    return H.Entry(key, "baseline", [FakeCall(dev, ms) for _ in range(sets)])


# ---- gates before timing ---------------------------------------------------------------------------------------------------


def test_nothing_is_timed_before_its_gate_passes_in_the_run():
    b, dev = bench()
    e = entry(dev, ("baseline", "s", "fp8-e4m3", "k0"), 1.0)
    b.capture([e], 4)
    with pytest.raises(S.GateError, match="not gated"):
        b.timed([e], 4, seed=1)
    b.gate(e.key, lambda: (False, {}))
    with pytest.raises(S.GateError, match="failed"):
        b.timed([e], 4, seed=1)
    with pytest.raises(S.GateError, match="not gated"):
        b.tune([entry(dev, ("baseline", "s", "fp8-e4m3", "k1"), 1.0)], 4, 3)
    assert b.items == []


def test_a_gate_that_raises_has_failed_and_a_pass_cannot_turn_into_a_fail():
    b, _ = bench()

    def boom():
        raise RuntimeError("illegal address")
    assert not b.gate(("k",), boom)
    assert "illegal address" in b.ledger.records[("k",)]["detail"]["error"]
    assert b.gate(("j",), lambda: (True, {}))
    with pytest.raises(S.GateError, match="after one passed"):
        b.gate(("j",), lambda: (False, {}))


def test_a_failed_arm_gate_times_nothing_of_that_flag_set():
    b, dev = bench()
    arm = FakeArm(failing={"nohash"})
    got = H.arm_entries(b, dev, arm, SHAPE, opsets(dev), first=False)
    assert [e.key[-1] for e in got] == ["hash", "hash"] and [e.kind for e in got] == ["arm", "weight"]
    assert not b.ledger.passed(("arm", SHAPE.label, arm.name, "nohash"))
    b.capture(got, 4)
    b.timed(got, 4, seed=3)
    assert b.items and all(it["key"][-1] == "hash" for it in b.items)


def test_an_arm_that_cannot_prepare_fails_its_gate():
    b, dev = bench()
    arm = FakeArm()
    arm.prepare = lambda *a: (_ for _ in ()).throw(RuntimeError("out of memory"))
    assert H.arm_entries(b, dev, arm, SHAPE, opsets(dev), first=False) == []
    assert "out of memory" in b.ledger.records[("arm", SHAPE.label, arm.name, "hash")]["detail"]["error"]


def test_a_gate_must_check_one_output_in_full():
    call = FakeCall(FakeDevice(), 1.0)
    with pytest.raises(A.ArmError, match="in full"):
        A.check_gate(A.Gate(True, {"root": "sampled:32"}), call)
    with pytest.raises(A.ArmError, match="unknown outputs"):
        A.check_gate(A.Gate(True, {"y": "all"}), call)


def test_check_arm_needs_both_required_roles_and_a_known_precision():
    arm = FakeArm()
    A.check_arm(arm, set(B.FAMILIES))
    arm.flag_sets = {"hash": frozenset()}
    with pytest.raises(A.ArmError, match="roles"):
        A.check_arm(arm, set(B.FAMILIES))
    arm = FakeArm()
    arm.precision = "fp6"
    with pytest.raises(A.ArmError, match="baseline family"):
        A.check_arm(arm, set(B.FAMILIES))


# ---- negative controls -----------------------------------------------------------------------------------------------------


def test_an_honest_arm_rejects_its_flipped_output_and_its_known_bad_then_passes_again():
    b, dev = bench()
    got = H.arm_entries(b, dev, FakeArm(), SHAPE, opsets(dev), first=True)
    assert len(got) == 3
    kinds = {tuple(n["key"][-2:]) if n["key"][-1] != "known_bad" else ("known_bad",): n["rejected"] for n in b.negatives}
    assert kinds == {("hash", "flip-output-byte"): True, ("nohash", "flip-output-byte"): True, ("known_bad",): True}
    assert b.ledger.passed(("arm", SHAPE.label, "fake-arm", "hash"))


def test_a_gate_that_ignores_the_outputs_stops_the_run():
    b, dev = bench()
    with pytest.raises(H.NegativeControlFailed, match="flip-output-byte"):
        H.arm_entries(b, dev, FakeArm(lazy=True), SHAPE, opsets(dev), first=True)


def test_an_accepted_known_bad_stops_the_run():
    b, dev = bench()
    with pytest.raises(H.NegativeControlFailed, match="known_bad"):
        H.arm_entries(b, dev, FakeArm(bad_ok=True), SHAPE, opsets(dev), first=True)


def test_a_negative_control_that_cannot_run_stops_the_run():
    b, _ = bench()

    def crash():
        raise RuntimeError("no algorithm")
    with pytest.raises(H.NegativeControlFailed, match="couldn't run"):
        b.negative(("known-bad", "x"), crash)


# ---- timing ------------------------------------------------------------------------------------------------------------------


def test_every_rep_is_kept_in_a_seeded_random_order_and_timed_the_same_way():
    b, dev = bench()
    es = [entry(dev, ("baseline", "s", "f", f"k{i}"), ms) for i, ms in enumerate((1.0, 2.0, 0.5))]
    for e in es:
        b.gate(e.key, lambda: (True, {}))
    b.capture(es, 4)
    per = b.timed(es, 4, seed=11)
    assert len(b.items) == S.MIN_REPS * len(es) * len(H.METHODS)
    for e, ms in zip(es, (1.0, 2.0, 0.5)):
        for m in H.METHODS:
            assert per[e.key][m] == [pytest.approx(ms)] * S.MIN_REPS
    orders = [tuple((tuple(it["key"]), it["method"]) for it in b.items[r * 6:(r + 1) * 6]) for r in range(S.MIN_REPS)]
    assert len(set(orders)) > 1
    again, dev2 = bench()
    es2 = [entry(dev2, e.key, 1.0) for e in es]
    for e in es2:
        again.gate(e.key, lambda: (True, {}))
    again.capture(es2, 4)
    again.timed(es2, 4, seed=11)
    assert [(it["key"], it["method"]) for it in again.items] == [(it["key"], it["method"]) for it in b.items]


def test_a_flush_precedes_every_timed_burst():
    b, dev = bench(methods=("burst",))
    e = entry(dev, ("baseline", "s", "f", "k"), 1.0)
    b.gate(e.key, lambda: (True, {}))
    dev.log.clear()
    b.timed([e], 4, seed=2)
    assert dev.log == ["flush", "record", "record"] * S.MIN_REPS


def test_a_burst_cycles_the_operand_sets():
    dev = FakeDevice()
    calls = [FakeCall(dev, 1.0) for _ in range(3)]
    seen = []
    for i, c in enumerate(calls):
        c.launch = lambda s, i=i: seen.append(i)
    H.Entry(("k",), "baseline", calls).enqueue(1, 7)
    assert seen == [0, 1, 2, 0, 1, 2, 0]


def test_an_entry_that_cannot_be_captured_is_dropped_not_timed():
    b, dev = bench()
    e = entry(dev, ("arm", "s", "a", "hash"), 1.0)

    def refuse(body):
        raise RuntimeError("operation not permitted when stream is capturing")
    dev.capture = refuse
    assert b.capture([e], 4) == []
    assert "capturing" in b.dropped[0]["why"]


def test_fewer_than_twenty_reps_is_refused():
    with pytest.raises(ValueError):
        S.rep_orders([1, 2], 19, 0)


# ---- the report ----------------------------------------------------------------------------------------------------------------


def baselines_ms(fp8=1.0, rowwise=0.9, nvfp4=0.5):
    return {"fp8-e4m3": {"ms": {"graph": fp8, "burst": fp8}, "kernel": {"graph": "a", "burst": "a"}},
            "fp8-e4m3-rowwise": {"ms": {"graph": rowwise, "burst": rowwise}, "kernel": {"graph": "b", "burst": "b"}},
            "nvfp4": {"ms": {"graph": nvfp4, "burst": nvfp4}, "kernel": {"graph": "c", "burst": "c"}}}


def test_arm_row_divides_by_the_fastest_plain_gemm_of_its_precision():
    work = FakeArm().work(SHAPE)
    row = R.arm_row(SHAPE, "fp8-e4m3", work, {"hash": {"graph": 1.8}, "nohash": {"graph": 1.2}}, {"graph": 0.5},
                    baselines_ms())
    g = row["graph"]
    assert g["baseline"] == "fp8-e4m3-rowwise" and g["baseline_ms"] == 0.9
    assert g["total_x"] == pytest.approx(2.0) and g["hash_free_x"] == pytest.approx(4 / 3)
    assert g["hashing_x"] == pytest.approx(2 / 3)
    assert g["hashing_bytes_per_s"] == pytest.approx((1 << 20) / 0.6e-3)
    assert g["weight_side_x"] == pytest.approx(0.5 / 0.9) and g["reuse_1pct"] == 56
    assert g["over"]["nvfp4"]["total_x"] == pytest.approx(3.6)
    assert "burst" not in row


def test_reuse_count_is_the_least_r_that_brings_the_weight_side_under_one_percent():
    assert R.reuse_under(0.5, 1.0) == 50
    assert R.reuse_under(0.501, 1.0) == 51
    assert R.reuse_under(0.0, 1.0) == 1
    assert 0.5 / R.reuse_under(0.5, 0.9) <= 0.01 * 0.9 < 0.5 / (R.reuse_under(0.5, 0.9) - 1)


def test_baseline_rows_give_tflops_per_method():
    rows = R.baseline_rows(SHAPE, {"fp8-e4m3": {"ms": {"graph": 1.0}, "kernel": {"graph": "k"}}})
    assert rows["fp8-e4m3"]["graph"]["tflops"] == pytest.approx(2 * 8192**3 / 1e-3 / 1e12)


def test_the_consistency_gate_rejects_a_hash_free_call_faster_than_the_plain_gemm():
    assert S.consistency(0.96, {"fp8-e4m3": SHAPE.mkn}, SHAPE.mkn, {"fp8-e4m3": 1.0})["ok"]
    assert not S.consistency(0.94, {"fp8-e4m3": SHAPE.mkn}, SHAPE.mkn, {"fp8-e4m3": 1.0})["ok"]
    assert S.consistency(0.1, {"fp8-e4m3": 1}, 1 << 20, {"fp8-e4m3": 1.0}) is None


def test_shape_report_rejects_an_arm_that_skips_work_and_gives_panel_numbers_for_an_honest_one():
    b, dev = bench()
    fams = [B.FAMILIES["fp8-e4m3"], B.FAMILIES["fp8-e4m3-rowwise"]]
    for arm, rejected in ((FakeArm(), False), (FakeArm(nohash_ms=0.5), True)):
        es = [entry(dev, ("baseline", SHAPE.label, "fp8-e4m3", "k"), 1.0),
              entry(dev, ("baseline", SHAPE.label, "fp8-e4m3-rowwise", "r"), 0.9)]
        for e in es:
            b.gate(e.key, lambda: (True, {}))
        es += H.arm_entries(b, dev, arm, SHAPE, opsets(dev), first=False)
        b.capture(es, 4)
        per = b.timed(es, 4, seed=5)
        row = H.shape_report(SHAPE, fams, [arm], es, per, b)["arms"][arm.name]
        assert ("rejected" in row) == rejected
        if not rejected:
            assert row["panel"] == {"phase": "prefill", "shape": "m8192-n8192-k8192", "slowdown": 2.0,
                                    "hash_free": round(1.2 / 0.9, 4), "baseline": "fp8-e4m3-rowwise",
                                    "baseline_kernel": "r"}
            assert row["w_ref_over_mkn"] == "1" and row["credited_over_mkn"] == "1/2"
    assert len(b.rejected) == 1


# ---- identity ------------------------------------------------------------------------------------------------------------------


SERVER_MD = f"""
| GPU | UUID | Owner |
|---|---|---|
| 5 | `{UUID5}` | harness |
| 6 | GPU-ffffffff-0000-1111-2222-333344445555 | red team |
"""
ENV = {"CUDA_VISIBLE_DEVICES": "5", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def test_identity_reads_the_server_table_and_refuses_any_other_gpu():
    table = I.parse_server_md(SERVER_MD)
    assert table[5] == UUID5 and 6 in table
    assert I.check(ENV, 5, 1, UUID5, table[5])["uuid_checked"]
    with pytest.raises(I.IdentityError, match="not GPU 5"):
        I.check(ENV, 5, 1, table[6], table[5])
    with pytest.raises(I.IdentityError, match="exactly"):
        I.check({**ENV, "CUDA_VISIBLE_DEVICES": "5,6"}, 5, 1, UUID5, UUID5)
    with pytest.raises(I.IdentityError, match="devices"):
        I.check(ENV, 5, 2, UUID5, UUID5)
    with pytest.raises(I.IdentityError, match="PCI_BUS_ID"):
        I.check({"CUDA_VISIBLE_DEVICES": "5"}, 5, 1, UUID5, UUID5)
    with pytest.raises(I.IdentityError, match="no expected UUID"):
        I.check(ENV, 5, 1, UUID5, None)
    assert not I.check({**ENV, "CUDA_VISIBLE_DEVICES": "0"}, 0, 1, UUID5, None, pod=True)["uuid_checked"]


def test_a_lease_names_the_index_and_the_table_names_its_uuid():
    assert I.resolve_gpu("5", {}) == 5
    assert I.resolve_gpu("lease", {"CUDA_VISIBLE_DEVICES": "3"}) == 3
    for bad in ("0,1", "", "GPU-x"):
        with pytest.raises(I.IdentityError, match="one index"):
            I.resolve_gpu("lease", {"CUDA_VISIBLE_DEVICES": bad})
    assert I.expected_for("lease", None, SERVER_MD, {"CUDA_VISIBLE_DEVICES": "5"}) == (5, UUID5)
    assert I.expected_for("5", "GPU-given", SERVER_MD, {}) == (5, "GPU-given")
    with pytest.raises(I.IdentityError, match="no GPU 9"):
        I.expected_for("9", None, SERVER_MD, {})


def test_timed_clocks_summarize_every_items_after_sample():
    items = [{"sensors": {"after": {"sm_mhz": f, "mem_mhz": 12481}}} for f in (2092, 2085, 2100)] + [{"sensors": {}}]
    got = H.timed_clocks(items)
    assert got["sm_mhz"] == {"n": 3, "min": 2085, "median": 2092, "max": 2100}
    assert got["mem_mhz"]["min"] == got["mem_mhz"]["max"] == 12481
    assert H.timed_clocks([]) == {"sm_mhz": {"n": 0}, "mem_mhz": {"n": 0}}


def test_a_queue_job_accepts_any_listed_gpu_and_records_it():
    table = I.parse_server_md(SERVER_MD)
    node = ",".join(table.values())
    got = I.check({**ENV, "CUDA_VISIBLE_DEVICES": "0"}, 0, 1, table[6], node)
    assert got["uuid"] == table[6] and got["expected_uuids"] == list(table.values())
    with pytest.raises(I.IdentityError, match="one of"):
        I.check({**ENV, "CUDA_VISIBLE_DEVICES": "0"}, 0, 1, "GPU-00000000-0000-0000-0000-000000000000", node)
    with pytest.raises(I.IdentityError, match="malformed"):
        I.check({**ENV, "CUDA_VISIBLE_DEVICES": "0"}, 0, 1, table[6], node + ",GPU-nope")


def test_the_card_must_be_an_rtx_pro_6000_sm120_with_188_sms():
    I.check_card("NVIDIA RTX PRO 6000 Blackwell Server Edition", "12.0", 188, "595.91.07")
    for bad in (("NVIDIA GeForce RTX 5090", "12.0", 170, "575.1"), ("NVIDIA RTX PRO 6000", "12.0", 188, "570.86")):
        with pytest.raises(I.IdentityError, match="not the card"):
            I.check_card(*bad)


# ---- operands, shapes, sets -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", sorted(F.TERNARY))
def test_the_gate_operands_are_ternary(fmt):
    units = 999
    v = F.decode(F.TERNARY[fmt], F.fill(F.TERNARY[fmt], units, 3))
    assert set(np.unique(v)) == {-1.0, 0.0, 1.0}


def test_the_timing_operands_are_finite_and_seeded():
    for fmt in (F.F32, F.BF16, F.E4M3, F.E2M1, F.INT8):
        raw = F.fill(fmt, 4096, 9)
        assert raw == F.fill(fmt, 4096, 9) != F.fill(fmt, 4096, 10)
        assert np.all(np.isfinite(F.decode(fmt, raw)))
    assert F.fill(F.E4M3, 100, 1, start=50) == F.fill(F.E4M3, 150, 1)[50:]


def test_the_gate_expects_exact_ternary_products_in_bf16_and_int32():
    rng = np.random.default_rng(0)
    a, b = rng.integers(-1, 2, (4, 8192)).astype(np.float64), rng.integers(-1, 2, (5, 8192)).astype(np.float64)
    exact = a @ b.T
    assert np.array_equal(B.expected(B.FAMILIES["int8"], a, b).view(np.int32), exact.astype(np.int32))
    words = B.expected(B.FAMILIES["fp8-e4m3"], a, b)
    back = F.decode(F.BF16, words.tobytes()).reshape(exact.shape)
    assert np.all(np.abs(back - exact) <= np.abs(exact) * 2.0 ** -8)
    assert F.bf16_rne(np.array([1.0, -2.0, 257.0, 259.0])).tolist() == [0x3F80, 0xC000, 0x4380, 0x4382]


def test_no_ternary_sum_reaches_the_sentinels():
    assert B.SENTINEL["int32"][1] > 32768 and B.SENTINEL["int32"][1] == int.from_bytes(bytes([B.SENTINEL["int32"][2]]) * 4,
                                                                                         "little")
    assert np.isnan(F.decode(F.BF16, np.array([B.SENTINEL["bf16"][1]], "<u2").tobytes()))[0]


def test_every_shapes_operand_sets_pass_twice_the_l2():
    fams = list(B.FAMILIES.values())
    for s in SH.all_shapes():
        plan = H.shape_plan(s, fams, (), H.L2_SM120)
        assert plan["sets"] * plan["weight_bytes_min"] >= 2 * H.L2_SM120 or plan["sets"] == S.MAX_SETS
        assert plan["burst"] >= max(plan["sets"], S.MIN_BURST)
        assert plan["memory_bytes"] < 60 << 30
    assert H.shape_plan(SHAPE, [B.FAMILIES["nvfp4"]], (), H.L2_SM120)["sets"] == 8


def test_the_shapes_hold_both_panel_headlines_once():
    labels = [s.label for s in SH.all_shapes()]
    assert len(labels) == len(set(labels))
    assert set(SH.HEADLINE.values()) <= set(labels)
    assert [s.label for s in SH.select(["headline"])] == ["m8192-n8192-k8192", "m32-n8192-k8192"]
    assert SH.ALIASES["llama3.1-70b-o-decode"] == "m32-n8192-k8192"
    assert SH.phase_of(SH.select(["llama3.1-70b-o-decode"])[0]) == "decode"
    with pytest.raises(KeyError):
        SH.select(["m1-n1-k1"])


def test_scale_tensors_cover_every_block():
    assert B.scale_bytes("ue4m3/16", 8192, 8192) == 8192 * 512
    assert B.scale_bytes("ue8m0/32", 100, 64) == 128 * 4
    assert B.scale_bytes("f32/128", 8192, 8192) == 4 * 8192 * 64
    assert B.scale_bytes(None, 8192, 8192) == 0


def test_set_count_bounds():
    assert S.set_count(1, 128 << 20) == S.MAX_SETS
    assert S.set_count(1 << 40, 128 << 20) == S.MIN_SETS
    assert S.set_count(64 << 20, 128 << 20) == 4
    rng = random.Random(0)
    assert all(S.burst_len(n) >= S.MIN_BURST for n in (rng.randrange(1, 600) for _ in range(50)))


# ---- the dependent chain ----------------------------------------------------------------------------------------------------

DECODE = A.Shape("m32-n8192-k8192", 32, 8192, 8192)
SMALL = A.Shape("m32-n64-k64", 32, 64, 64)


def test_the_derive_twin_rounds_to_nearest_even_and_saturates():
    pos = F.E4M3_VALUES[:127]
    mids = (pos[:-1] + pos[1:]) / 2
    got = DV.e4m3_rne(mids.astype(np.float32))
    assert np.all(got % 2 == 0) and np.all((got == np.arange(126)) | (got == np.arange(1, 127)))
    assert DV.e4m3_rne(pos.astype(np.float32)).tolist() == list(range(127))
    assert DV.e4m3_rne(np.array([1e9, -1e9, -0.0, 0.0, -1e-9], np.float32)).tolist() == [0x7E, 0xFE, 0x80, 0x00, 0x80]
    vals = np.array([0.25, 0.3, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.1, 7.0, -0.1, -6.0], np.float32)
    want = [0, 1, 2, 2, 4, 4, 6, 6, 7, 7, 8, 15]
    assert DV.e2m1_rne(vals).tolist() == want
    assert np.array_equal(F.E2M1_VALUES[DV.e2m1_rne(F.E2M1_VALUES.astype(np.float32))], F.E2M1_VALUES)


@pytest.mark.parametrize("kind", sorted(DV.KINDS))
def test_every_derive_stays_bounded_reads_columns_mod_n_and_decodes_close_to_its_input(kind):
    m, n, k = 8, 96, 128
    y = DV.preflight_y("bf16", m, n, 5)
    x = DV.scaled(DV.KINDS[kind], "bf16", y, m, n, k)
    assert np.all(np.abs(x) < 2.0 ** DV.KINDS[kind].t)
    assert np.array_equal(x[:, n:], x[:, :k - n])
    a, scale, offs = DV.derive(kind, "bf16", y, m, n, k)
    assert len(a) == DV.a_bytes(DV.KINDS[kind], m, k)
    fmt = {"f32": F.F32, "bf16": F.BF16, "e4m3": F.E4M3, "mxfp8": F.E4M3, "nvfp4": F.E2M1, "mxfp4": F.E2M1, "int8": F.INT8}[kind]
    v = F.decode(fmt, a).reshape(m, k)
    if scale is not None:
        bs = DV.KINDS[kind].block
        per = F.E4M3_VALUES[scale] if kind == "nvfp4" else 2.0 ** (scale.astype(np.float64) - 127)
        v = v * np.repeat(per, bs, axis=1)
    rel = {"f32": 0, "bf16": 2**-8, "e4m3": 2**-4, "mxfp8": 2**-3, "nvfp4": 0.5, "mxfp4": 0.5, "int8": 0}[kind]
    blk = DV.KINDS[kind].block
    ref = np.max(np.abs(x.reshape(m, k // blk, blk)), axis=2).repeat(blk, axis=1) if blk > 1 else np.abs(x)
    floor = {"int8": 0.5, "e4m3": 2.0 ** -9, "nvfp4": 6 * 2.0 ** -9}.get(kind, 0.0)  # what underflows to zero
    assert np.all(np.abs(v - x) <= rel * ref + floor + 1e-30)


def test_block_scales_land_in_the_128x4_tiled_layout():
    offs = DV.sf_offsets(256, 12)
    assert offs[0, 0] == 0 and offs[0, 3] == 3 and offs[1, 0] == 16 and offs[32, 0] == 4 and offs[0, 4] == 512
    assert offs[128, 0] == 512 * 3
    assert len(set(offs.ravel().tolist())) == offs.size and offs.max() < B.scale_bytes("ue4m3/16", 256, 12 * 16)


class FakeGemm:
    """A GEMM stand-in on the fake device: y (m x n BF16) a function of A's E4M3 codes, `ms` of virtual time; with
    `phases`, before and after of their own cost."""

    def __init__(self, dev, shape, ms=1.0, phases=False, before=0.2, after=0.3):
        m, n, k = shape.m, shape.n, shape.k
        self.dev, self.shape, self.ms = dev, shape, ms
        self.ops = B.Operands(dev.alloc(m * k), dev.alloc(n * k), None, None, dev.alloc(m * n * 2))
        dev.write(self.ops.a.ptr, F.fill(F.E4M3, m * k, id(self) & 0xFFFF))
        self.outputs = {"y": self.ops.d}
        self.launches = 0
        if phases:
            self.phases = {"before": lambda s: self._cost(before), "gemm": self._gemm, "after": lambda s: self._cost(after)}
            self.launch = lambda s: [self.phases[p](s) for p in CH.PHASES]

    def _cost(self, ms):
        self.dev.enqueue(lambda: setattr(self.dev, "t", self.dev.t + ms))

    def _gemm(self, stream=None):
        def run():
            m, n, k = self.shape.m, self.shape.n, self.shape.k
            a = F.E4M3_VALUES[np.frombuffer(bytes(self.dev.mem[self.ops.a.ptr][:m * k]), np.uint8)].reshape(m, k)
            self.dev.write(self.ops.d.ptr, F.bf16_rne(a[:, np.arange(n) % k] * 3 + 1).astype("<u2").tobytes())
            self.dev.t += self.ms
            self.launches += 1
        self.dev.enqueue(run)

    def launch(self, stream):
        self._gemm(stream)

    def weight_side(self, stream):
        pass

    def free(self):
        pass


def chain_bench(sets=3, phases=False, ms=1.0):
    b, dev = bench()
    calls = [FakeGemm(dev, SMALL, ms, phases) for _ in range(sets)]
    e = H.Entry(("baseline", SMALL.label, "fp8-e4m3", "k"), "baseline", calls, meta={"source": "fake"})
    b.gate(e.key, lambda: (True, {}))
    return b, dev, e


def test_the_chain_is_the_least_multiple_of_the_sets_past_its_minimum():
    assert CH.length(8) == 64 and CH.length(3) == 66 and CH.length(100) == 100


def test_a_chain_derives_each_call_from_the_previous_calls_y_and_wraps():
    b, dev, e = chain_bench()
    links = CH.baseline_links(e, B.FAMILIES["fp8-e4m3"])
    ch = CH.Chain(dev, SMALL, links, "e4m3", "full")
    dev.log.clear()
    ch.enqueue(dev.stream, 6)
    derives = [x for x in dev.log if x[0] == "derive"]
    ys, as_ = [ln.y.ptr for ln in links], [ln.a.ptr for ln in links]
    assert [(ys.index(y), as_.index(a)) for _, y, a, _ in derives] == [(2, 0), (0, 1), (1, 2), (2, 0), (0, 1), (1, 2)]
    assert [c.launches for c in e.calls] == [2, 2, 2]
    for j in (1, 2):
        assert ch.check(j)["ok"]


def test_the_side_stream_chain_guards_set_reuse_and_joins_at_its_end():
    b, dev, e = chain_bench(phases=True)
    links = CH.baseline_links(e, B.FAMILIES["fp8-e4m3"])
    ch = CH.Chain(dev, SMALL, links, "e4m3", "side")
    dev.log.clear()
    ch.enqueue(dev.stream, 6)
    waits = [x for x in dev.log if x[0] == "wait"]
    main_waits = [ev for _, s, ev in waits if s == dev.stream]
    assert main_waits == [ch.done[0], ch.done[1], ch.done[2], ch.done[2]]
    assert [ev for _, s, ev in waits if s == dev.side_stream] == [ch.fork[j % 3] for j in range(6)]
    into_0 = [i for i, x in enumerate(dev.log) if isinstance(x, tuple) and x[0] == "derive" and x[2] == links[0].a.ptr]
    assert len(into_0) == 2 and dev.log[into_0[1] - 1] == ("wait", dev.stream, ch.done[0])
    ch.release()
    assert ch.fork == ch.done == []


def test_a_chain_is_gated_before_it_is_timed_and_a_stale_a_is_rejected():
    b, dev, e = chain_bench()
    got, unchained = H.chain_entries(b, dev, SMALL, [e], [], {}, 3, first=True)
    keys = [x.key for x in got]
    assert keys == [("chain", SMALL.label, "fp8-e4m3", "k", "full"), ("chain", SMALL.label, "derive", "e4m3", "derive-only")]
    assert all(b.ledger.passed(k) for k in keys) and unchained == {}
    stale = [n for n in b.negatives if n["key"][-1] == "chain-stale-a"]
    assert len(stale) == 1 and stale[0]["rejected"]
    assert all(x.length == CH.length(3) for x in got)
    b.capture(got, 4)
    b.timed(got, 4, seed=1)
    full = [it for it in b.items if it["key"][-1] == "full"]
    assert full and all(it["burst"] == 66 for it in full)
    assert full[0]["ms_per_call"] == pytest.approx(1.1)


def test_a_chain_whose_derive_is_wrong_is_not_timed_and_a_chain_needs_its_calls_gate():
    b, dev, e = chain_bench()
    real = dev.derive

    def wrong(src, y, m, n, kind, a, sf, k, stream=None):
        real(src, y, m, n, DV.KINDS["int8"].id if kind == DV.KINDS["e4m3"].id else kind, a, sf, k, stream)
    dev.derive = wrong
    got, _ = H.chain_entries(b, dev, SMALL, [e], [], {}, 3, first=False)
    assert got == [] and not b.ledger.passed(("chain", SMALL.label, "fp8-e4m3", "k", "full"))
    b2, dev2, e2 = chain_bench()
    b2.ledger.records.clear()
    b2.gate(e2.key, lambda: (False, {}))
    got, _ = H.chain_entries(b2, dev2, SMALL, [e2], [], {}, 3, first=False)
    assert [x.key[-1] for x in got] == ["derive-only"]
    assert "did not pass" in b2.ledger.records[("chain", SMALL.label, "fp8-e4m3", "k", "full")]["detail"]["error"]


class FakeChainArm(FakeArm):
    """A decode arm whose calls are FakeGemms on the harness's operand sets, with phases; its gate recomputes y from A."""

    def __init__(self, phases=True):
        super().__init__()
        self.phases = phases

    def prepare(self, dev, shape, flags, operands):
        c = FakeGemm(dev, shape, 1.0 if "h" in flags else 0.9, self.phases)
        c.ops = B.Operands(operands.a, operands.b, None, None, c.ops.d)
        c.outputs = {"y": c.ops.d}
        return c

    def gate(self, dev, call, shape, flags):
        m, n, k = shape.m, shape.n, shape.k
        a = F.E4M3_VALUES[np.frombuffer(dev.read(call.ops.a.ptr, m * k), np.uint8)].reshape(m, k)
        want = F.bf16_rne(a[:, np.arange(n) % k] * 3 + 1).astype("<u2").tobytes()
        return A.Gate(dev.read(call.outputs["y"].ptr, m * n * 2) == want, {"y": "all"})


def test_chain_report_gives_the_divisor_the_split_and_a_dependent_chain_panel_row(monkeypatch):
    monkeypatch.setitem(SH.HEADLINE, "decode", SMALL.label)
    b, dev, e = chain_bench(ms=0.5)
    arm = FakeChainArm()
    sets = [A.OperandSet(i, i, i, c.ops.a, c.ops.b) for i, c in enumerate(e.calls)]
    arm_es = H.arm_entries(b, dev, arm, SMALL, sets, first=False)
    got, unchained = H.chain_entries(b, dev, SMALL, [e, *arm_es], [arm], {arm.name: sets}, 3, first=False)
    variants = sorted(x.key[3:] for x in got if x.key[2] == arm.name)
    assert variants == [("hash", "before-gemm"), ("hash", "full"), ("hash", "gemm"), ("hash", "side"), ("nohash", "full")]
    assert "arm_gate" in b.ledger.records[("chain", SMALL.label, arm.name, "hash", "full")]["detail"]
    es = [e, *arm_es, *got]
    b.capture(es, 4)
    per = b.timed(es, 4, seed=2)
    rep = H.shape_report(SMALL, [B.FAMILIES["fp8-e4m3"]], [arm], es, per, b)
    base = rep["chain"]["baselines"]["fp8-e4m3"]["graph"]
    assert base["ms"] == pytest.approx(0.6) and base["independent_ms"] == pytest.approx(0.5)
    assert base["derive_ms"] == pytest.approx(0.1) and base["kernel"] == "k"
    split = rep["chain"]["arms"][arm.name]["split"]["graph"]
    assert split["derive_ms"] == pytest.approx(0.1) and split["before_ms"] == pytest.approx(0.2)
    assert split["during_ms"] == pytest.approx(1.0) and split["after_ms"] == pytest.approx(0.3)
    panel = rep["arms"][arm.name]["panel"]
    assert panel["decode_method"] == "dependent-chain" and panel["baseline_kernel"] == "k"
    assert panel["slowdown"] == pytest.approx(round(1.6 / 0.6, 4)) and panel["hash_free"] == pytest.approx(round(1.5 / 0.6, 4))
    assert panel["chain_length"] == 66 and "slowdown_side_stream" in panel


def test_an_arm_without_a_y_output_is_not_chained():
    b, dev, e = chain_bench()
    arm = FakeArm()
    arm_es = H.arm_entries(b, dev, arm, SMALL, opsets(dev), first=False)
    got, unchained = H.chain_entries(b, dev, SMALL, arm_es, [arm], {arm.name: opsets(dev)}, 3, first=False)
    assert got == [] and set(unchained) == {"fake-arm/hash", "fake-arm/nohash"}
