"""The PoUW bench harness's rules, on a fake device with a virtual clock (no GPU): nothing is timed before its gate passes in
the run, a failed gate times nothing, every rep is kept in a seeded random order, the negative controls stop the run, the
consistency gate rejects a hash-free call faster than the plain GEMM allows, and the report's arithmetic."""

from __future__ import annotations

import argparse
import random
import re
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

    def memset(self, ptr, value, n):
        self.mem[ptr][:n] = bytes([value]) * n

    def count_eq(self, buf, word_bytes, pattern):
        return sum(b == pattern for b in self.mem[buf.ptr][:buf.nbytes]) if word_bytes == 1 else 0

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
        return A.GateResult(ok, {"root": "all"})

    def known_bad(self, dev, shape, operands):
        return FakeCall(dev, 1.0, pattern=b"\x5a" * 32 if self.bad_ok else b"\x00" * 32)


SHAPE = A.Shape("m8192-n8192-k8192", 8192, 8192, 8192)


def opsets(dev, n=3):
    return [A.OperandSet(i, i, i, dev.alloc(64), dev.alloc(64)) for i in range(n)]


def bench(reps=S.MIN_REPS, methods=H.METHODS):
    dev = FakeDevice()
    return H.Bench(dev, FakeSensors(), reps, methods, clock=lambda: dev.t / 1e3), dev


def test_the_host_seconds_of_each_gate_control_and_stage_are_recorded_where_a_lease_can_sit_idle():
    b, dev = bench()

    def costs(ms, ok=True):
        def run():
            dev.t += ms
            return ok, {"x": 1}
        return run

    key = ("arm", "s", "a", "hash")
    assert b.gate(key, costs(1500.0)) and b.ledger.records[key] == {"ok": True, "detail": {"x": 1}, "wall_s": 1.5}
    assert b.gate(key, costs(500.0)) and b.ledger.records[key]["wall_s"] == pytest.approx(2.0)
    b.negative(("known-bad", "s", "a", "flip"), costs(250.0, ok=False))
    assert b.negatives[-1]["rejected"] and b.negatives[-1]["wall_s"] == pytest.approx(0.25)
    with pytest.raises(H.NegativeControlFailed):
        b.negative(("known-bad", "s", "a", "raise"), lambda: (dev.__setattr__("t", dev.t + 100.0), 1 / 0))
    assert b.negatives[-1]["rejected"] is None and b.negatives[-1]["wall_s"] == pytest.approx(0.1)
    st = H.Stages(b.clock)
    dev.t += 3000.0
    st.mark("operands")
    dev.t += 7000.0
    st.mark("arm_gates")
    dev.t += 1000.0
    st.mark("arm_gates")
    assert st.wall_s == pytest.approx({"operands": 3.0, "arm_gates": 8.0})


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


def test_a_sigterm_during_a_gate_stops_the_run_and_fails_no_gate():
    b, _ = bench()

    def term():
        H._terminated(15, None)
    with pytest.raises(H.Terminated):
        b.gate(("k",), term)
    assert ("k",) not in b.ledger.records


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
        A.check_gate(A.GateResult(True, {"root": "sampled:32"}), call)
    with pytest.raises(A.ArmError, match="unknown outputs"):
        A.check_gate(A.GateResult(True, {"y": "all"}), call)


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


class DriftSensors(FakeSensors):
    """Item i's SM clock (the timer samples three times an item) is `clocks[i]`, then the last one."""

    def __init__(self, clocks):
        self.clocks, self.n = list(clocks), 0

    def sample(self):
        s = super().sample()
        s["sm_mhz"] = self.clocks[min(self.n // 3, len(self.clocks) - 1)]
        self.n += 1
        return s


def warm_bench(clocks):
    dev = FakeDevice()
    b = H.Bench(dev, DriftSensors(clocks), S.MIN_REPS, ("graph",))
    es = [entry(dev, ("baseline", "s", "f", "k"), 1.0), entry(dev, ("arm", "s", "a", "hash"), 1.5)]
    for e in es:
        b.gate(e.key, lambda: (True, {}))
    b.capture(es, 4)
    return b, es


def test_the_warmup_runs_the_schedule_until_the_clock_is_steady_apart_from_the_reps():
    b, es = warm_bench([2600, 2600, 2500, 2500, 2400, 2400, 2395, 2395])
    w = b.warmup(es, 4, seed=3, min_ms=0.0, max_ms=1e6)
    assert w["round_sm_mhz"] == [2600, 2500, 2400, 2395] and w["steady"] and w["rounds"] == 4
    assert w["per_call"] == {es[0].key: pytest.approx(1.0), es[1].key: pytest.approx(1.5)}
    assert len(b.warmups) == 8 and {it["round"] for it in b.warmups} == {0, 1, 2, 3} and b.items == []


def test_the_warmup_lasts_its_device_time_and_stops_at_its_bound_unsteady():
    b, es = warm_bench([2400])
    assert b.warmup(es, 4, seed=3, min_ms=20.0, max_ms=1e6)["rounds"] == 2     # a round is 4 x 1.0 + 4 x 1.5 ms
    b, es = warm_bench([2400])
    assert b.warmup(es, 4, seed=3, min_ms=25.0, max_ms=1e6)["rounds"] == 3
    b, es = warm_bench([2600, 2600, 2400, 2400] * 10)
    w = b.warmup(es, 4, seed=3, min_ms=0.0, max_ms=35.0)
    assert not w["steady"] and w["rounds"] == 4 and w["device_ms"] == pytest.approx(40.0)


def test_the_warmup_needs_every_gate():
    b, dev = bench()
    e = entry(dev, ("baseline", "s", "f", "k"), 1.0)
    b.capture([e], 4)
    with pytest.raises(S.GateError, match="not gated"):
        b.warmup([e], 4, seed=1, min_ms=0.0)
    assert b.warmups == []


class FakeChain:
    def __init__(self, dev, ms):
        self.dev, self.ms = dev, ms

    def enqueue(self, stream, n):
        self.dev.enqueue(lambda: setattr(self.dev, "t", self.dev.t + self.ms * n))


def test_every_timed_item_is_sized_to_the_same_device_time():
    b, dev = bench()
    fast, slow = entry(dev, ("baseline", "s", "f", "k"), 1.0), entry(dev, ("arm", "s", "a", "hash"), 1.8)
    ch = H.Entry(("chain", "s", "f", "k", "full"), "chain", fast.calls, chain=FakeChain(dev, 0.5), length=8)
    es = [fast, slow, ch]
    for e in es:
        b.gate(e.key, lambda: (True, {}))
    b.capture(es, 4)
    per_call = {fast.key: 1.0, slow.key: 1.8, ch.key: 0.5}
    sz = b.equalize(es, per_call, 4, item_ms=40.0)
    assert sz["target_ms"] == 40.0 and [(x["calls"], x["replays"]) for x in sz["entries"]] == [(40, 1), (22, 1), (80, 10)]
    assert len(fast.graph.work) == 40 and len(slow.graph.work) == 22
    per = b.timed(es, 4, seed=1)
    for e, ms in zip(es, (1.0, 1.8, 0.5)):
        assert all(per[e.key][m] == [pytest.approx(ms)] * S.MIN_REPS for m in H.METHODS)
    item = {(tuple(it["key"]), it["method"]): it["burst_ms"] for it in b.items}
    for m in H.METHODS:
        assert [item[(e.key, m)] for e in es] == [pytest.approx(40.0), pytest.approx(39.6), pytest.approx(40.0)]
    b2, dev2 = bench()
    es2 = [entry(dev2, fast.key, 1.0), entry(dev2, slow.key, 1.8)]
    b2.capture(es2, 4)
    sz = b2.equalize(es2, per_call, 4, item_ms=5.0)
    assert sz["target_ms"] == pytest.approx(7.2) and [x["calls"] for x in sz["entries"]] == [7, 4]


def test_the_panel_rows_carry_both_sides_reps_and_tag_a_clock_gap():
    def it(key, rep, ms, mhz):
        return {"key": list(key), "method": "graph", "rep": rep, "ms_per_call": ms, "burst_ms": 30 * ms, "burst": 30,
                "sensors": {"during": {"sm_mhz": mhz}, "after": {"sm_mhz": 1}}}
    arm, base = ("arm", "s", "x", "hash"), ("baseline", "s", "f", "k")
    items = [it(arm, 1, 1.3, 2092), it(arm, 0, 1.4, 2100), it(base, 0, 1.0, 2090), it(base, 1, 1.0, 2094),
             {**it(base, 0, 9.0, 1), "method": "burst"}]
    p = H.panel_reps(items, arm, base)
    assert p["reps"]["arm"]["ms_per_call"] == [1.4, 1.3] and p["reps"]["baseline"]["calls"] == [30, 30]
    assert p["sm_clock_arm"] == "2100,2092" and p["sm_clock_base"] == "2090,2094"
    assert p["sm_clock_median_mhz"] == {"arm": 2096, "baseline": 2092} and not p["clock_gap"]
    assert p["item_ms_ratio"] == pytest.approx(1.35)
    items[0]["sensors"]["during"]["sm_mhz"] = 2200
    assert H.panel_reps(items, arm, base)["clock_gap"]
    items[0]["sensors"] = {}
    p = H.panel_reps(items, arm, base)
    assert "sm_clock_arm" not in p and "sm_clock_missing" in p


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
            p = row["panel"]
            assert {k: p[k] for k in ("phase", "shape", "slowdown", "hash_free", "baseline", "baseline_kernel")} == {
                "phase": "prefill", "shape": "m8192-n8192-k8192", "slowdown": 2.0, "hash_free": round(1.2 / 0.9, 4),
                "baseline": "fp8-e4m3-rowwise", "baseline_kernel": "r"}
            reps = p["reps"]
            assert reps["arm"]["key"] == ["arm", SHAPE.label, arm.name, "hash"]
            assert reps["baseline"]["key"] == ["baseline", SHAPE.label, "fp8-e4m3-rowwise", "r"]
            assert len(reps["arm"]["ms_per_call"]) == len(reps["baseline"]["sm_mhz"]) == S.MIN_REPS
            assert p["sm_clock_arm"] == p["sm_clock_base"] == ",".join(["2400"] * S.MIN_REPS) and not p["clock_gap"]
            assert p["item_ms_ratio"] == pytest.approx(1.8 / 0.9)
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


def test_the_enumeration_splits_k_only_where_the_output_tiles_underfill_the_card():
    assert B.lt_splitks(SHAPE, 188) == [1]
    ks = B.lt_splitks(A.Shape("m32-n8192-k8192", 32, 8192, 8192), 188)
    assert ks[0] == 1 and max(ks) == 16


def test_the_screen_keeps_the_fastest_and_ranks_the_unscreened_last():
    assert H.screened({"a": 2.0, "b": 1.0, "d": 0.5}, ["a", "b", "c", "d"], 3) == ["d", "b", "a"]
    assert H.screened({}, ["b", "a"], 1) == ["a"]


def test_an_enumerated_configuration_is_named_by_its_config_and_deduplicated_by_it():
    cfg = {"algo_id": 70, "tile": 23, "stages": 37, "splitk": 1, "reduction": 0, "swizzle": 1, "custom": 0,
           "inner_shape": 0, "cluster_shape": 2, "blk128_swap": 0, "library": "lt13"}
    name = B._enum_name("lt13", "nvfp4", 0, cfg)
    assert name == "lt13_nvfp4_0_algo70_tile23_st37_sw1_cl2"
    heur = B.Candidate("nvfp4", "cublaslt", "lt_x", cfg | {"heuristic_rank": 0})
    enum = B.Candidate("nvfp4", "cublaslt", name, cfg | {"enumerated": 5})
    assert B.lt_key(heur) == B.lt_key(enum)
    assert B.lt_key(enum) != B.lt_key(B.Candidate("nvfp4", "cublaslt", name, cfg | {"library": "lt"}))
    assert "_sk4_r1" in B._enum_name("lt", "nvfp4", 0, cfg | {"splitk": 4, "reduction": 1})


class FakeCutlassLib:
    """The registry's query and prepare (cutlass_registry.cu): a 2.x kernel refuses every order but the default, and
    `refuse` names the (kernel, swizzle, raster) a 3.x kernel can't implement."""

    def __init__(self, kernels, refuse=()):
        self.kernels, self.refuse, self.calls = kernels, set(refuse), []

    def _query(self, i, sw, r, ws):
        self.calls.append(("query", i, sw, r))
        if (sw or r) and not self.kernels[i]["name"].startswith("cutlass3x_") or (i, sw, r) in self.refuse:
            return 3000 + 11
        ws._obj.value = 4096 * (i + 1) + 16 * sw + r
        return 0

    def ph_cutlass_query(self, i, m, n, k, ws, sfa, sfb):
        return self._query(i, 0, 0, ws)

    def ph_cutlass_query_sched(self, i, m, n, k, sw, r, ws, sfa, sfb):
        return self._query(i, sw, r, ws)

    def ph_cutlass_prepare(self, i, m, n, k, *rest):
        self.calls.append(("prepare", i, m, n, k))
        return 0

    def ph_cutlass_prepare_sched(self, i, m, n, k, sw, r, *rest):
        self.calls.append(("prepare_sched", i, m, n, k, sw, r))
        return 0


def test_a_cutlass_tile_order_is_a_candidate_of_its_own_where_the_grid_tells_it_apart():
    src = (Path(__file__).parents[1] / "harness" / "native" / "cutlass_3x.cu").read_text()
    names = re.findall(r'ENTRY(?:_TS)?\("(cutlass3x_[^"]+)"', src)
    assert len(names) > 20 and all(B.cutlass_tile(n) for n in names)
    assert B.cutlass_tile("cutlass3x_nvfp4_128x32x256_coop_swap") == (128, 32)
    assert B.cutlass_tile("cutlass2x_int8_128x128x64_s4") is None
    big, decode = A.Shape("big", 8192, 8192, 8192), A.Shape("decode", 32, 8192, 8192)
    assert B.cutlass_swizzles((128, 128), big, 0) == [2, 4, 8]
    assert [B.cutlass_swizzles((128, 128), A.Shape("s", m, 8192, 8192), 0) for m in (128, 256, 384, 640, 768)] == [
        [], [2], [2, 4], [2, 4], [2, 4, 8]]
    assert B.cutlass_swizzles((128, 32), decode, 1) == []  # swapped: the kernel's N is our m, one CTA wide
    assert B.cutlass_swizzles((128, 32), A.Shape("s", 8192, 8192, 256), 1) == [2]  # and its M our n

    kernels = [{"index": 0, "name": "cutlass3x_mxfp4_128x128x128_coop", "kind": 6, "swap": 0},
               {"index": 1, "name": "cutlass3x_mxfp4_128x64x256_coop_swap", "kind": 6, "swap": 1},
               {"index": 2, "name": "cutlass2x_int8_128x128x64_s4", "kind": 7, "swap": 0}]
    lib = FakeCutlassLib(kernels, refuse={(1, 8, 2)})
    dev = argparse.Namespace(lib=lib, ck=lambda r, what: r, stream=7)
    mx = B.FAMILIES["mxfp4"]
    plain, no = B.cutlass_candidates(dev, mx, big, kernels)
    assert [c.name for c in plain] == [k["name"] for k in kernels[:2]] and not no
    assert all(c.config.keys() == {"swap", "workspace"} and not H.searched(c) for c in plain)
    assert all(call[2:] == (0, 0) for call in lib.calls)

    ok, no = B.cutlass_candidates(dev, mx, big, kernels, sched=True)
    assert len(ok) == 2 * 12 - 1 and no == [{"name": "cutlass3x_mxfp4_128x64x256_coop_swap_rasterN_sw8", "status": 3011}]
    got = {c.name: c for c in ok}
    assert len(got) == len(ok) and {"cutlass3x_mxfp4_128x128x128_coop_sw2", "cutlass3x_mxfp4_128x128x128_coop_rasterN_sw8",
                                    "cutlass3x_mxfp4_128x64x256_coop_swap_rasterM"} <= got.keys()
    v = got["cutlass3x_mxfp4_128x128x128_coop_rasterM_sw4"]
    assert v.config == {"swap": 0, "workspace": 4096 + 64 + 1, "swizzle": 4, "raster": 1} and v.cutlass == 0
    assert H.searched(v) and not H.searched(got["cutlass3x_mxfp4_128x128x128_coop"])

    ok, _ = B.cutlass_candidates(dev, mx, decode, kernels, sched=True)
    assert [c.name for c in ok if c.cutlass == 1] == ["cutlass3x_mxfp4_128x64x256_coop_swap",
                                                      "cutlass3x_mxfp4_128x64x256_coop_swap_rasterM",
                                                      "cutlass3x_mxfp4_128x64x256_coop_swap_rasterN"]
    assert [c.name for c in B.cutlass_candidates(dev, B.FAMILIES["int8"], big, kernels, sched=True)[0]] == [
        "cutlass2x_int8_128x128x64_s4"]

    buf = A.Buffer(0x1000, 64)
    ops = B.Operands(buf, buf, None, None, buf)
    B.BaselineCall(dev, got["cutlass3x_mxfp4_128x128x128_coop"], big, ops, buf)
    B.BaselineCall(dev, v, big, ops, buf)
    assert lib.calls[-2:] == [("prepare", 0, 8192, 8192, 8192), ("prepare_sched", 0, 8192, 8192, 8192, 4, 1)]

    with pytest.raises(SystemExit, match="tile orders"):
        H.gate_baselines(None, argparse.Namespace(has_cutlass_sched=False), big, [], None, [], None, False, 1, 0,
                         sched_fams=frozenset({"mxfp4"}))


def test_a_screen_only_run_screens_every_searched_candidate_and_times_nothing(monkeypatch):
    enum = B.Candidate("nvfp4", "cublaslt", "lt13_nvfp4_0_algo70_tile20_st37", {"enumerated": 0})
    order = B.Candidate("nvfp4", "cutlass", "cutlass3x_nvfp4_256x128x128_coop_sw2", {"swizzle": 2, "raster": 0})
    plain = B.Candidate("nvfp4", "cutlass", "cutlass3x_nvfp4_256x128x128_coop", {"swap": 0})
    screened_ = []

    def fake_screen(bench, dev, shape, fam, so, cands, lt_ws, burst):
        screened_.extend(c.name for c in cands)
        return {c.name: 2.0 - i for i, c in enumerate(cands)}

    def no_tune(*a, **k):
        raise AssertionError("a screen-only run tunes nothing")

    monkeypatch.setattr(H, "shape_plan", lambda *a: {"memory_bytes": 0, "sets": 2, "burst": 4})
    monkeypatch.setattr(B, "cutlass_candidates", lambda *a, **k: ([], []))
    monkeypatch.setattr(B, "allocate", lambda *a: argparse.Namespace(free=lambda dev: None))
    monkeypatch.setattr(H, "gate_baselines", lambda *a: {"nvfp4": {"gated": [enum, plain, order], "all": [], "tried": []}})
    monkeypatch.setattr(H, "screen", fake_screen)
    monkeypatch.setattr(H, "tune_baselines", no_tune)
    dev = argparse.Namespace(l2=128 << 20, mem_info=lambda: (1 << 40, 1 << 40))
    args = argparse.Namespace(screen_only=True, seed=1, lt_candidates=32)
    row = H.run_shape(bench()[0], dev, SHAPE, [B.FAMILIES["nvfp4"]], [], True, args, [], None)
    assert row["screen_only"] and screened_ == [enum.name, order.name]
    assert list(row["wall_s"]) == ["operands", "baseline_gates", "screen"]
    bc = row["baseline_candidates"]["nvfp4"]
    assert list(bc["screen_ms"]) == [order.name, enum.name] and bc["gated"] == [enum.name, plain.name, order.name]


def test_a_verity_registry_entry_is_its_own_source_in_one_tile_order_with_a_workspace_of_its_own():
    native = Path(__file__).parents[1] / "harness" / "native"
    plain = (native.parents[1] / "nvfp4_sm120" / "nvf4_plain.cu").read_text()
    entries = re.findall(r'\{"(verity_[^"]+)", (\d), 0, (query\w*),', plain)
    assert {(n, int(k)) for n, k, _ in entries} >= {("verity_nvf4_256x128_o_ew", 5), ("verity_fp8_256x128_o_ew", 1)}
    assert {q for *_, q in entries} == {"query", "query_fp8"}  # the (m, n, k) form: query_fixed
    registry = (native / "cutlass_registry.cu").read_text()
    assert re.search(r"#endif\s+phc_table_nvf4, phc_table_fp8\}", registry)  # in a no-CUTLASS build too
    assert '"$PLAIN - nvf4_plain 0"' in (native / "build.sh").read_text()

    kernels = [{"index": 0, "name": "cutlass3x_nvfp4_256x128x128_coop", "kind": 5, "swap": 0},
               {"index": 1, "name": "verity_nvf4_256x128_o_ew", "kind": 5, "swap": 0},
               {"index": 2, "name": "verity_fp8_256x128_o_ew", "kind": 1, "swap": 0}]

    class Lib(FakeCutlassLib):
        def ph_cutlass_prepare(self, i, m, n, k, a, b, sa, sb, d, ws, ws_bytes, stream, h):
            self.calls.append(("prepare", i, ws, ws_bytes))
            return 0 if i != 2 or ws_bytes < 1 << 20 else 2

    lib, mem = Lib(kernels), {}

    def alloc(n):
        buf = A.Buffer(0x100000 * (len(mem) + 1), n)
        mem[buf.ptr] = buf
        return buf

    def ck(r, what):
        if r:
            raise RuntimeError(what)

    dev = argparse.Namespace(lib=lib, ck=ck, stream=7, alloc=alloc, free=lambda b: mem.pop(b.ptr))
    big = A.Shape("big", 8192, 8192, 8192)
    ok, no = B.cutlass_candidates(dev, B.FAMILIES["nvfp4"], big, kernels, sched=True)
    got = {c.name: c for c in ok}
    assert not no and got["verity_nvf4_256x128_o_ew"].source == "verity"
    assert {c.source for c in ok if c.cutlass == 0} == {"cutlass"} and [c.name for c in ok if c.cutlass == 1] == [
        "verity_nvf4_256x128_o_ew"]
    assert not any(call[2:] != (0, 0) for call in lib.calls if call[0] == "query" and call[1] == 1)

    lt_ws = A.Buffer(0x1000, B.LT_WORKSPACE)
    ops = B.Operands(lt_ws, lt_ws, None, None, lt_ws)
    cut = B.BaselineCall(dev, got["cutlass3x_nvfp4_256x128x128_coop"], big, ops, lt_ws)
    own = B.BaselineCall(dev, got["verity_nvf4_256x128_o_ew"], big, ops, lt_ws)
    (_, _, ws_cut, _), (_, _, ws_own, own_bytes) = lib.calls[-2:]
    assert ws_cut == lt_ws.ptr and ws_own != lt_ws.ptr and own_bytes >= B.PLAIN_WORKSPACE and list(mem) == [ws_own]
    own.free()
    cut.free()
    assert not mem and own._own_ws is None

    fp8 = B.cutlass_candidates(dev, B.FAMILIES["fp8-e4m3"], big, kernels)[0]
    assert [(c.name, c.source) for c in fp8] == [("verity_fp8_256x128_o_ew", "verity")]
    fp8[0].ws = 1 << 20  # the fake refuses a workspace this large: the call frees its own before it raises
    with pytest.raises(RuntimeError, match="verity prepare verity_fp8_256x128_o_ew"):
        B.BaselineCall(dev, fp8[0], big, ops, lt_ws)
    assert not mem


def test_the_help_formats(capsys):
    with pytest.raises(SystemExit) as ex:
        H.parse(["--help"])
    out = capsys.readouterr().out
    assert ex.value.code == 0 and "--lt-enumerate-range" in out and "--cutlass-sched" in out


def test_a_slice_of_the_enumeration_takes_its_range_and_keeps_enough_to_reach_it(tmp_path):
    a = argparse.Namespace
    whole = H.lt_pick(a())
    assert whole == B.LtPick() and whole.want == B.LT_ENUMERATED_MAX and whole.takes(10**6, "x")
    p = H.lt_pick(a(lt_enumerate_range=H.lt_range("8192:12288"), lt_enumerate_max=B.LT_ENUMERATED_MAX))
    assert p.want == 12288 and [p.takes(i, "x") for i in (8191, 8192, 12287, 12288)] == [False, True, True, False]
    assert H.lt_pick(a(lt_enumerate_range=(0, 100), lt_enumerate_max=500)).want == 500
    names = tmp_path / "best.txt"
    names.write_text("lt13_fp8-e4m3_0_algo35_tile20_st15\n\n  lt_fp8-e4m3_0_algo60_tile13_st17_sw1 \n")
    q = H.lt_pick(a(lt_enumerate_names=names, lt_enumerate_max=40000))
    assert q.names == {"lt13_fp8-e4m3_0_algo35_tile20_st15", "lt_fp8-e4m3_0_algo60_tile13_st17_sw1"} and q.want == 40000
    assert q.takes(39999, "lt13_fp8-e4m3_0_algo35_tile20_st15") and not q.takes(0, "lt13_fp8-e4m3_0_algo35_tile20_st16")
    assert q.record() == {"start": 0, "stop": None, "limit": 40000, "names": 2}
    for bad in ("5:5", "7:3", "3", ":4", "a:9", "-1:4"):
        with pytest.raises(argparse.ArgumentTypeError):
            H.lt_range(bad)


class FakeLtLib:
    """lt.cu's plan, heuristic and enumeration calls: the enumeration accepts `accepted` configurations (algo 67, tile i) and
    writes the first `want` of them, as ph_lt_enumerate does."""

    def __init__(self, accepted):
        self.accepted = accepted

    def ph_lt_plan_create_ld(self, kind, m, n, k, ld, ws, swap, plan):
        plan._obj.value = 0x100 + swap
        return 0

    def ph_lt_plan_destroy(self, plan):
        pass

    def ph_lt_candidates(self, plan, sa, sb, want, arr, got):
        got._obj.value = 0
        return 0

    def ph_lt_enumerate(self, plan, sa, sb, ks, nks, want, arr, got, checked):
        for i in range(min(want, self.accepted)):
            arr[i].algo_id, arr[i].tile, arr[i].stages, arr[i].splitk = 67, i, 37, 1
        got._obj.value = checked._obj.value = self.accepted
        return 0

    def ph_last_error(self):
        return b""


def test_a_named_configuration_past_the_enumerations_cap_stops_the_run(tmp_path, monkeypatch):
    import native as N

    monkeypatch.setattr(B, "cutlass_candidates", lambda *a, **k: ([], []))
    monkeypatch.setattr(B, "BaselineCall", lambda *a: argparse.Namespace(free=lambda: None))
    monkeypatch.setattr(B, "gate", lambda *a, **k: {"ok": True})
    lt = N.LtLib(FakeLtLib(accepted=20000), "lt13", 130101, "fake")
    dev = argparse.Namespace(lts=[lt], info={"sms": 188}, has_cutlass_sched=True)
    so = argparse.Namespace(ops=lambda fam, s: argparse.Namespace(sa=None, sb=None))
    decode = A.Shape("m32-n8192-k8192", 32, 8192, 8192)
    names = tmp_path / "best.txt"
    names.write_text("lt13_fp8-e4m3_0_algo67_tile10_st37\nlt13_fp8-e4m3_0_algo67_tile15000_st37\n")
    a = argparse.Namespace

    def gate(limit):
        b, _ = bench()
        return H.gate_baselines(b, dev, decode, [B.FAMILIES["fp8-e4m3"]], so, [], None, False, 8, 0,
                                frozenset({"fp8-e4m3"}), H.lt_pick(a(lt_enumerate_names=names, lt_enumerate_max=limit)))

    with pytest.raises(H.NamesNotTaken, match=r"1 of the fp8-e4m3 .*_tile15000_st37\): .*lt13 accepted 20000: pass "
                                              r"--lt-enumerate-max 20000"):
        gate(B.LT_ENUMERATED_MAX)
    got = gate(20000)["fp8-e4m3"]
    assert [c.name for c in got["gated"]] == ["lt13_fp8-e4m3_0_algo67_tile10_st37", "lt13_fp8-e4m3_0_algo67_tile15000_st37"]
    assert got["cublaslt_enumeration"]["lt13"]["named"] == [c.name for c in got["gated"]] and not got[
        "cublaslt_enumeration_missing"]
    names.write_text("lt13_fp8-e4m3_0_algo67_tile10_st38\n")
    with pytest.raises(H.NamesNotTaken, match="no cuBLASLt loaded accepts them"):
        gate(20000)


def test_each_librarys_fastest_is_a_finalist_beside_the_fastest_few(monkeypatch):
    ms = {"cutlass3x_a": 1.0, "cutlass3x_b": 1.01, "cutlass3x_c": 1.02, "cutlass3x_d": 1.03, "lt_x": 1.05, "lt_y": 1.06,
          "lt13_z": 1.08}
    cands = [B.Candidate("fp8-e4m3", "cutlass", n, {"swap": 0}) for n in ms if n.startswith("cutlass")]
    cands += [B.Candidate("fp8-e4m3", "cublaslt", n, {"library": n.split("_")[0]}) for n in ms if n.startswith("lt")]
    b, dev = bench()
    made = []

    def call(d, c, *a):
        made.append((c.name, FakeCall(dev, ms[c.name])))
        return made[-1][1]
    monkeypatch.setattr(B, "BaselineCall", call)
    for c in cands:
        b.gate(("baseline", SHAPE.label, "fp8-e4m3", c.name), lambda: (True, {}))
    gated = {"fp8-e4m3": {"gated": cands}}
    so = argparse.Namespace(sets=3, ops=lambda fam, s: None)
    got = H.tune_baselines(b, dev, SHAPE, [B.FAMILIES["fp8-e4m3"]], so, gated, None, 2, 1, 3)
    final = ["cutlass3x_a", "cutlass3x_b", "cutlass3x_c", "lt_x", "lt13_z"]
    assert [e.key[-1] for e in got] == gated["fp8-e4m3"]["finalists"] == final
    assert [e.meta["library"] for e in got] == ["cutlass"] * 3 + ["lt", "lt13"]
    assert len(made) == 3 * len(ms) and all(fc.freed == (n not in final) for n, fc in made)
    assert [e.key[-1] for e in H.finalists_of(got, 1)] == ["cutlass3x_a", "lt_x", "lt13_z"]


def test_a_names_file_must_name_configurations_of_the_enumerated_families(tmp_path, capsys):
    names = tmp_path / "best.txt"
    base = ["--gpu", "0", "--out", str(tmp_path / "o.json"), "--lt-enumerate-names", str(names)]
    names.write_text("lt13_fp8-e4m3_0_algo67_tile10_st37\n")
    for extra, why in (([], "doesn't enumerate"), (["--lt-enumerate", "nvfp4"], "doesn't enumerate")):
        with pytest.raises(SystemExit) as ex:
            H.parse(base + extra)
        assert ex.value.code == 2 and why in capsys.readouterr().err
    assert H.parse(base + ["--lt-enumerate", "fp8-e4m3"]).lt_enumerate == ["fp8-e4m3"]
    names.write_text("\n")
    with pytest.raises(SystemExit):
        H.parse(base + ["--lt-enumerate", "fp8-e4m3"])
    assert "names no configuration" in capsys.readouterr().err


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


def test_nvfp4_rounds_without_dividing_on_the_preflight_edge_row():
    """harness.cu's k_derive rounds NVFP4's amax / 6 and x / scale from exact products (a division would bring a div.rn slow
    path the SASS gate hasn't pinned); restated here, it gives the twin's bytes on the preflight's edge row, whose blocks sit
    on those roundings' ties and turns."""
    y = DV.preflight_y("f32", 10, 384, 11)
    x = DV.scaled(DV.KINDS["nvfp4"], "f32", y, 10, 384, 384)[9]
    assert np.array_equal(x, np.frombuffer(y, "<f4").reshape(10, 384)[9])
    pos = F.E4M3_VALUES[:127].astype(np.float32)
    six_mid = np.float32(3) * (pos[:-1] + pos[1:])
    blocks = x.reshape(-1, 16)
    bmax = np.max(np.abs(blocks), axis=1)
    edge = np.stack([DV._ulps(six_mid, d) for d in (-1, 0, 1)])
    assert all(b in edge for b in bmax[1:]) and np.isin(bmax, six_mid).any()

    def reaches(b, i):
        t = six_mid[np.clip(i, 1, 126) - 1]
        return np.where(i % 2 == 0, b >= t, b > t)

    c = DV.e4m3_rne(bmax * np.float32(float.fromhex("0x1.555556p-3"))).astype(np.int32)
    hi = (c < 126) & reaches(bmax, c + 1)
    scale = c + hi - (~hi & (c > 0) & ~reaches(bmax, c))
    want = DV.e4m3_rne(bmax / np.float32(6))
    assert np.array_equal(scale, want) and (scale != c).any()
    s = pos[scale][:, None]
    th = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)
    a = np.abs(blocks)
    code = np.select([a <= th[0] * s, a < th[1] * s, a <= th[2] * s, a < th[3] * s, a <= th[4] * s, a < th[5] * s,
                      a <= th[6] * s], [0, 1, 2, 3, 4, 5, 6], 7) | np.where(np.signbit(blocks), 8, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        twin = np.where(s > 0, DV.e2m1_rne(blocks / np.where(s > 0, s, np.float32(1))), 0)
    assert np.array_equal(np.where(s > 0, code, 0), twin)
    assert np.isin(a, th * s).sum() > len(blocks)


def test_k_derive_scales_by_powers_of_two_as_ldexpf_rounds():
    """harness.cu's scale2 (x * a * b * c, from scale2_of) and e4m3_value (built from its bits), restated: ldexpf's result
    for every shift the derive can meet and past it, on zeros, subnormals, both ends of FP32 and random values."""
    def pow2f(s):
        return np.int32((s + 127) << 23 if s >= -126 else 1 << (s + 149)).view(np.float32)

    def scale2_of(s):
        a = b = np.float32(1)
        if s > 127:
            a, s = pow2f(127), s - 127
            if s > 127:
                b, s = pow2f(127), min(s - 127, 127)
        elif s < -126:
            a, s = np.float32(2.0 ** -102), s + 102
            if s < -126:
                b, s = np.float32(2.0 ** -102), max(s + 102, -126)
        return a, b, pow2f(s)

    rng = np.random.default_rng(3)
    bits = np.concatenate([np.array([0, 1, 2, 0x7FFFFF, 0x800000, 0x7F7FFFFF, 0x3F800000, 0x3FFFFFFF], np.uint32),
                           rng.integers(0, 0x7F800000, 4000, dtype=np.uint32)])
    x = np.concatenate([bits, bits | np.uint32(0x80000000)]).view(np.float32)
    with np.errstate(over="ignore", under="ignore"):
        for s in range(-400, 401):
            a, b, c = scale2_of(s)
            got = (x * a) * b * c
            assert np.array_equal(got.view(np.uint32), np.ldexp(x, s).astype(np.float32).view(np.uint32)), s
    codes = np.arange(256)
    e, mt = (codes >> 3) & 15, codes & 7
    v = np.where(e == 0, mt.astype(np.float32) * np.float32(2.0 ** -9),
                 (((e + 120) << 23) | (mt << 20)).astype(np.int32).view(np.float32))
    v = np.where(codes & 0x80, -v, v)
    finite = np.isfinite(F.E4M3_VALUES)
    assert np.array_equal(v[finite], F.E4M3_VALUES[finite].astype(np.float32))


@pytest.mark.parametrize("seed", [11, 0, 1, 2, 3])
def test_the_derive_preflight_y_is_finite(seed):
    """The device's FMUL canonicalizes a NaN and the twin keeps its bits, so a NaN in y fails the preflight on bytes it
    doesn't mean to test; its y holds none, nor an infinity, at every preflight shape."""
    for m, n, _, _ in H.PREFLIGHT_DERIVE:
        for src in DV.SRC:
            assert np.isfinite(DV.load(src, DV.preflight_y(src, m, n, seed), m, n)).all(), (m, n, src)


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
    `phases`, before and after of their own cost: v0.4's names, or with ``phases="groups"`` step groups (a_tree and forming
    splitting before, cleanup 0.1 after the GEMM, tile_hash and screen splitting after)."""

    def __init__(self, dev, shape, ms=1.0, phases=False, before=0.2, after=0.3):
        m, n, k = shape.m, shape.n, shape.k
        self.dev, self.shape, self.ms = dev, shape, ms
        self.ops = B.Operands(dev.alloc(m * k), dev.alloc(n * k), None, None, dev.alloc(m * n * 2))
        dev.write(self.ops.a.ptr, F.fill(F.E4M3, m * k, id(self) & 0xFFFF))
        self.outputs = {"y": self.ops.d}
        self.launches = 0
        cost = lambda ms: lambda s: self._cost(ms)  # noqa: E731
        if phases == "groups":
            self.phases = {"a_tree": cost(before / 4), "forming": cost(before * 3 / 4), "gemm": self._gemm,
                           "cleanup": cost(0.1), "tile_hash": cost(after / 3), "screen": cost(after * 2 / 3)}
            self.launch = lambda s: [self.phases[g](s) for g in A.GROUPS if g in self.phases]
        elif phases:
            self.phases = {"before": cost(before), "gemm": self._gemm, "after": cost(after)}
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
    b, dev, e = chain_bench(phases="groups")
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
        return A.GateResult(dev.read(call.outputs["y"].ptr, m * n * 2) == want, {"y": "all"})


@pytest.mark.parametrize("phases", ["groups", True])
def test_chain_report_gives_the_divisor_the_split_and_a_dependent_chain_panel_row(monkeypatch, phases):
    monkeypatch.setitem(SH.HEADLINE, "decode", SMALL.label)
    b, dev, e = chain_bench(ms=0.5)
    arm = FakeChainArm(phases)
    cleanup = 0.1 if phases == "groups" else 0.0
    sets = [A.OperandSet(i, i, i, c.ops.a, c.ops.b) for i, c in enumerate(e.calls)]
    arm_es = H.arm_entries(b, dev, arm, SMALL, sets, first=False)
    got, unchained = H.chain_entries(b, dev, SMALL, [e, *arm_es], [arm], {arm.name: sets}, 3, first=False)
    variants = sorted(x.key[3:] for x in got if x.key[2] == arm.name)
    side = [("hash", "side")] if phases == "groups" else []
    assert variants == [("hash", "before-gemm"), ("hash", "full"), ("hash", "gemm"), *side, ("nohash", "full")]
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
    assert split["during_ms"] == pytest.approx(1.0 + cleanup) and split["after_ms"] == pytest.approx(0.3)
    groups = rep["chain"]["arms"][arm.name]["split_groups"]
    if phases == "groups":
        assert groups == {"before_ms": ["a_tree", "forming"], "during_ms": ["gemm", "cleanup"],
                          "after_ms": ["tile_hash", "screen"], "side_stream": ["tile_hash"]}
        assert "side_ms" in split
    else:
        assert groups.startswith("v0.4 phases") and "side_ms" not in split and "side_x" not in split
    panel = rep["arms"][arm.name]["panel"]
    assert panel["decode_method"] == "dependent-chain" and panel["baseline_kernel"] == "k"
    assert panel["slowdown"] == pytest.approx(round((1.6 + cleanup) / 0.6, 4))
    assert panel["hash_free"] == pytest.approx(round((1.5 + cleanup) / 0.6, 4))
    assert panel["chain_length"] == 66 and ("slowdown_side_stream" in panel) == (phases == "groups")


def test_phases_are_step_groups_or_v04s_names_never_both_and_the_side_stream_takes_the_served_deferred_groups():
    """The side stream takes what the served deferred schedule moves (tile hashing) and leaves the screen on the main lane;
    v0.4's one ``after`` can't be split that way, so it gets no side chain."""
    assert A.Arm is A.BenchArm and isinstance(FakeArm(), A.BenchArm) and A.VERSION == "v0.6"
    assert set(A.PARTS["after"]) == A.MOVABLE and sum(A.PARTS.values(), ()) == A.GROUPS and A.DEFERRED <= A.MOVABLE
    try:
        from verity.core.protocols.pouw import serving
    except ImportError:
        serving = None
    if serving is not None:
        assert (A.GROUPS, A.MOVABLE) == (serving.GROUPS, serving.MOVABLE)
        assert A.DEFERRED == serving.DEFERRED.side
    f = {g: (lambda s, g=g: g) for g in A.GROUPS + ("before", "after", "hash")}
    parts, named = A.phase_parts({g: f[g] for g in ("forming", "gemm", "screen")})
    assert named and {p: [fn(0) for fn in fns] for p, fns in parts.items()} == {
        "before": ["forming"], "gemm": ["gemm"], "after": ["screen"]}
    parts, named = A.phase_parts({p: f[p] for p in ("before", "gemm", "after")})
    assert not named and [fn(0) for fn in parts["after"]] == ["after"]
    kept, deferred = A.side_split({g: f[g] for g in A.GROUPS})
    assert ([fn(0) for fn in kept], [fn(0) for fn in deferred]) == (["screen"], ["tile_hash"])
    assert A.side_split({p: f[p] for p in ("before", "gemm", "after")}) is None
    assert A.side_split({g: f[g] for g in ("forming", "gemm", "screen")}) is None and A.side_split(None) is None
    assert A.phase_parts(None) is None and A.phase_parts({"before": f["before"], "gemm": f["gemm"]}) is None
    for bad, why in (({"before": 0, "gemm": 0, "after": 0, "a_tree": 0}, "mix"), ({"gemm": 0, "hash": 0}, "unknown"),
                     ({"forming": 0, "screen": 0}, "gemm among them")):
        with pytest.raises(A.ArmError, match=why):
            A.phase_parts(bad)

    b, dev, e = chain_bench()
    arm = FakeChainArm("groups")
    sets = [A.OperandSet(i, i, i, c.ops.a, c.ops.b) for i, c in enumerate(e.calls)]
    arm_es = H.arm_entries(b, dev, arm, SMALL, sets, first=False)
    links, _ = CH.arm_links(next(x for x in arm_es if x.key[3] == "hash"), arm, sets, SMALL)
    moved = []
    for c in (ln.call for ln in links):
        for g in ("tile_hash", "screen"):
            c.phases[g] = lambda s, g=g, run=c.phases[g]: (moved.append((g, s)), run(s))
    ch = CH.Chain(dev, SMALL, links, "e4m3", "side")
    ch.enqueue(dev.stream, 3)
    assert moved == [("tile_hash", dev.side_stream), ("screen", dev.stream)] * 3
    v04 = chain_bench(phases=True)
    with pytest.raises(A.ArmError, match="deferred group"):
        CH.Chain(v04[1], SMALL, CH.baseline_links(v04[2], B.FAMILIES["fp8-e4m3"]), "e4m3", "side")
    links[1].call.phases["hash"] = f["hash"]
    got, unchained = H.chain_entries(b, dev, SMALL, [e, *arm_es], [arm], {arm.name: sets}, 3, first=False)
    assert "unknown ['hash']" in unchained[f"{arm.name}/hash"]
    assert [x.key[3:] for x in got if x.key[2] == arm.name] == [("nohash", "full")]


def test_an_arm_without_a_y_output_is_not_chained():
    b, dev, e = chain_bench()
    arm = FakeArm()
    arm_es = H.arm_entries(b, dev, arm, SMALL, opsets(dev), first=False)
    got, unchained = H.chain_entries(b, dev, SMALL, arm_es, [arm], {arm.name: opsets(dev)}, 3, first=False)
    assert got == [] and set(unchained) == {"fake-arm/hash", "fake-arm/nohash"}


# ---- poisoned transcripts and the no-write control (bench.poison_dump) ---------------------------------------------------------


class DumpingArm(FakeArm):
    """Dumps its root's bytes (and a setup-written seed's, when its call has one) as the transcript."""

    def dump(self, dev, call, shape, out):
        bufs = H.transcript_buffers(call)
        (out / "manifest.json").write_text(",".join(f"{n}={dev.read(b.ptr, b.nbytes).hex()}" for n, b in sorted(bufs.items())))


class StaleCall(FakeCall):
    """Its setup wrote the root and a seed; its timed launch writes neither (the stale-output case, panel rule)."""

    def __init__(self, dev, ms):
        super().__init__(dev, ms)
        self.seed = dev.alloc(8)
        dev.write(self.outputs["root"].ptr, self.pattern)
        dev.write(self.seed.ptr, b"\x11" * 8)
        self.transcript_buffers = {"root": self.outputs["root"], "seed": self.seed}

    def launch(self, stream):
        self.dev.enqueue(lambda: setattr(self.dev, "t", self.dev.t + self.ms))


def test_the_harness_dumps_a_poisoned_transcript_then_a_no_write_control(tmp_path):
    dev = FakeDevice()
    call = FakeCall(dev, 1.0)
    rec = H.poison_dump(dev, DumpingArm(), SHAPE, call, tmp_path)
    assert rec["by"] == "harness" and rec["poison"] == "0xA5" and rec["written"] == {"root": 32} and "wrote_nothing" not in rec
    honest = (tmp_path / rec["transcript_dir"] / "manifest.json").read_text()
    control = (tmp_path / rec["control_dir"] / "manifest.json").read_text()
    assert honest == "root=" + "5a" * 32 and control == "root=" + "a5" * 32
    assert rec["transcript_dir"] == f"transcripts/fake-arm/{SHAPE.label}" and rec["control_dir"] == f"negative/fake-arm/{SHAPE.label}"


def test_a_stale_output_left_by_setup_is_poison_in_the_dump(tmp_path):
    dev = FakeDevice()
    rec = H.poison_dump(dev, DumpingArm(), SHAPE, StaleCall(dev, 1.0), tmp_path)
    assert rec["wrote_nothing"] and rec["written"] == {"root": 0, "seed": 0} and rec["buffers"] == {"root": 32, "seed": 8}
    assert (tmp_path / rec["transcript_dir"] / "manifest.json").read_text() == "root=" + "a5" * 32 + ",seed=" + "a5" * 8


def test_dump_transcripts_poisons_every_dumping_arm_s_hash_call_and_records_the_rest():
    b, dev = bench()
    arm, own = DumpingArm(), FakeArm()
    own.name = "own-dump"
    entries = [H.Entry(("arm", SHAPE.label, arm.name, "hash"), "arm", [FakeCall(dev, 1.0)])]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        recs = H.dump_transcripts(b, dev, SHAPE, [arm, own], entries, Path(d))
        assert (Path(d) / recs[0]["control_dir"] / "manifest.json").is_file()
    assert [r["by"] for r in recs] == ["harness", "arm"] and "no dump" in recs[1]["why"]
    assert b.poisoning == recs
    lone = DumpingArm()
    lone.name = "no-hash-call"
    assert H.dump_transcripts(b, dev, SHAPE, [lone], [], Path("."))[0]["error"] == "no gated hash call at this shape"


def test_a_transcript_buffer_over_an_operand_is_refused_before_anything_is_poisoned(tmp_path):
    b, dev = bench()
    arm, sets = DumpingArm(), opsets(dev, 1)
    call = FakeCall(dev, 1.0)
    dev.write(sets[0].a.ptr, b"\x07" * 64)
    call.transcript_buffers = {"root": call.outputs["root"], "a": A.Buffer(sets[0].a.ptr + 32, 16)}
    entries = [H.Entry(("arm", SHAPE.label, arm.name, "hash"), "arm", [call])]
    rec = H.dump_transcripts(b, dev, SHAPE, [arm], entries, tmp_path, {arm.name: sets})[0]
    assert rec["by"] == "harness" and "a overlaps operand a" in rec["error"] and not (tmp_path / "transcripts").exists()
    assert dev.read(sets[0].a.ptr, 64) == b"\x07" * 64
    assert H.operand_overlaps({"root": call.outputs["root"]}, sets[0]) == []
