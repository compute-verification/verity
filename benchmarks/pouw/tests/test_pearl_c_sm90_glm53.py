"""GLM-5.3's linears under pearl-c-h100-v1-h3 (benchmarks/pouw/pearl_c_sm90/glm53.py): one TP8 rank's shapes from the
checkpoint's config, the sm_90a kernels' domain verdict on each, γ's formula against a Lean price twin, the H200
session's shapes the table's, glm53_timing's --whole over a stand-in card, and glm53_passes' full check of a retained pass."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1] / "pearl_c_sm90"
spec = importlib.util.spec_from_file_location("pearl_c_sm90_glm53", HERE / "glm53.py")
G = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


def test_linears_are_one_tp8_ranks():
    """The capture map's shapes (glm-moe-dsa-eager-hopper-tp8-v0) at GLM-5.3's config: KVA = QL + KVL + DR, NHL = NH / 8,
    IKW = ID + IH, IL = I / 8, MIL = SIL = MI / 8; the per-rank K (256, 1536, 2048, 6144) are the FP8
    linears' k but kv_b_proj's 512."""
    got = {lin.name: (lin.n, lin.k) for lin in G.linears()}
    assert got == {
        "self_attn.fused_qkv_a_proj": (2624, 6144), "self_attn.q_b_proj": (2048, 2048), "self_attn.kv_b_proj": (3584, 512),
        "self_attn.o_proj": (6144, 2048), "self_attn.indexer.wq_b": (4096, 2048), "self_attn.indexer.wk_weights_proj": (160, 6144),
        "mlp.gate_up_proj": (3072, 6144), "mlp.down_proj": (6144, 1536), "mlp.shared_experts.gate_up_proj": (512, 6144),
        "mlp.shared_experts.down_proj": (6144, 256), "mlp.experts.w13": (512, 6144), "mlp.experts.w2": (6144, 256)}
    fp8 = {lin.k for lin in G.linears() if lin.stock != "cublas-bf16-gemm"}
    assert fp8 == {256, 512, 1536, 2048, 6144}
    assert G.expert_rows(64) == 2 and G.expert_rows(8192) == 256


def test_domain_refuses_the_short_k_and_narrow_n():
    """k = 512 is under the record's k ≥ 1,024 and not its census k = 256 (which takes both down projections); n = 160 is
    not a multiple of 64.  Every other linear is in, with γ at both FADD readings, the data sheet's (cheaper add) a little
    above the record's, and γ falling with n."""
    rows = {r["name"]: r for r in G.table()}
    refused = {name for name, r in rows.items() if r["refused"]}
    assert refused == {"self_attn.kv_b_proj", "self_attn.indexer.wk_weights_proj"}
    assert "1024 <= k" in rows["self_attn.kv_b_proj"]["refused"][0]
    assert "n of 64" in rows["self_attn.indexer.wk_weights_proj"]["refused"][0]
    for name, r in rows.items():
        if name not in refused:
            assert 0 < r["gamma"]["record"] < r["gamma"]["sheet"] < 0.011, name
    assert rows["mlp.experts.w13"]["gamma"]["record"] > rows["self_attn.o_proj"]["gamma"]["record"]


def test_expert_parallel_brings_the_experts_into_the_domain():
    """With the routed experts expert-parallel (EP = TP = 8) a rank holds 32 whole experts, w13 (4096, 6144) and w2
    (6144, 2048), both in the domain; the shared expert stays sharded, its down at the census's k = 256."""
    rows = {r["name"]: r for r in G.table(ep=True)}
    assert {x: (rows[x]["n"], rows[x]["k"]) for x in ("mlp.experts.w13", "mlp.experts.w2")} == {
        "mlp.experts.w13": (4096, 6144), "mlp.experts.w2": (6144, 2048)}
    assert {x for x, r in rows.items() if r["refused"]} == {"self_attn.kv_b_proj", "self_attn.indexer.wk_weights_proj"}
    assert (rows["mlp.shared_experts.down_proj"]["k"], rows["mlp.shared_experts.down_proj"]["refused"]) == (256, [])
    shapes = G.in_domain()
    assert (4096, 6144) in shapes and shapes[(4096, 6144)]["ep"] and not shapes[(6144, 2048)]["ep"]
    assert shapes[(6144, 2048)]["name"] == "self_attn.o_proj"


def test_gamma_is_the_lean_twins():
    """`Pouw.PearlC.pearlCGammaSm120v1Cast32Rev1_8192`: v1 under rev1 (cap 1/400) at the sm_120 record, 8192³, the honest cast
    at 32 (24 over the credited 8), γ = 173561/22302500."""
    from verity.ml.protocols.pouw.schemes import pearl_c as C
    from verity.ml.protocols.pouw.schemes.pearl_c_device import SM120

    s = C.PearlC(forming="v1", device=replace(SM120, cap=Fraction(1, 400)), hashing="h3")
    assert G.gamma(s, 8192, 8192, Fraction(24)) == Fraction(173561, 22302500)
    h = G.scheme()
    assert h.name == G.SCHEME and h.prices.fadd == 32 and G.scheme(Fraction(1979, 67)).prices.fadd == Fraction(1979, 67)


def test_the_profile_gives_each_launch_to_the_step_that_launched_it():
    """glm53_profile's report: a GPU activity is the innermost step's around its launch (the API call's thread and time, not
    when it ran), one launched outside every call is dropped, the GPU span is the calls' first start to their last end over
    the calls, the solo call's is its own, and a reference SHA-512 launch is named as a fallback."""
    spec = importlib.util.spec_from_file_location("pearl_c_sm90_glm53_profile", HERE / "glm53_profile.py")
    P = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    t, label = 7, "6144:2048 m16 hash"
    ranges = [(t, 0, 100, f"call {label} #0"), (t, 10, 30, "step commit_a"), (t, 40, 60, "step gemm_pearlc"),
              (t, 70, 90, "host after"), (t, 75, 85, "host screen"),
              (t, 100, 200, f"call {label} #1"), (t, 110, 130, "step commit_a"), (t, 140, 160, "step hash_leaf"),
              (t, 300, 400, f"solo {label}"), (t, 310, 320, "step commit_a"), (t, 500, 900, f"pass_end {label}")]
    acts = [(t, 15, 1000, 3000, "kernel", "pd_commit_851"), (t, 45, 3000, 5000, "kernel", "gemm_pearlc_d"),
            (t, 80, 5000, 5500, "kernel", "screen"), (t, 72, 5500, 5600, "memcpy", "memcpy"),
            (t, 105, 6000, 6100, "memcpy", "memcpy"), (t, 115, 7000, 9000, "kernel", "pd_commit_851"),
            (t, 145, 9000, 10000, "kernel", "s512_tile_leaves"), (t, 250, 20000, 21000, "kernel", "stray"),
            (t + 1, 15, 21000, 22000, "kernel", "other thread"), (t, 315, 30000, 32000, "kernel", "pd_commit_851")]
    out = P.report(sorted(ranges), [(*a, 7) for a in acts])
    assert set(out) == {label}
    r, steps = out[label], out[label]["steps"]
    assert r["calls"] == 2 and r["host_wall_us"] == 0.1 and r["pass_end_host_us"] == 0.4
    assert {s: round(v["gpu_us"], 9) for s, v in steps.items()} == {
        "step commit_a": 2.0, "step gemm_pearlc": 2.0, "host screen": 0.5, "host after": 0.1, "host other": 0.05, "step hash_leaf": 1.0}
    assert steps["step commit_a"]["host_us"] == 0.02 and steps["step commit_a"]["kernels"]["pd_commit_851"]["count"] == 1
    assert round(steps["host other"]["host_us"], 9) == 0.05 and steps["host screen"]["host_us"] == 0.01
    assert r["gpu_busy_us"] == 3.85 and r["gpu_span_us_per_call"] == 4.5 and r["solo_gpu_span_us"] == 2.0
    assert (r["hash_decode"], r["hash_fast"], r["fallbacks"]) == (["pd_commit_851"], [], ["s512_tile_leaves"])


def test_the_lanes_report_gives_each_graph_node_to_its_launchs_call_and_each_stream_its_lane():
    """glm53_profile's lanes: a graph's nodes go to the call whose range their launch is in, the stream that runs pd_commit
    is the main lane and those that run pd_tiles the side lanes, numbered in their order; the period is a call's first
    start to the next's, and each lane's start, span, busy and idle times are over the steady calls (all but the first and
    the last), its idle time since the call before's last activity on the same lane."""
    spec = importlib.util.spec_from_file_location("pearl_c_sm90_glm53_profile", HERE / "glm53_profile.py")
    P = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    t, label = 7, "4096:6144 m64 hash"
    ranges = [(t, 100 * c, 100 * c + 50, f"call {label} #{c}") for c in range(4)]
    acts = []
    for c in range(4):                      # main on stream 5; side lanes 9 then 11; 10 µs a call apart
        at, side = 10_000 * c, (9, 11)[c % 2]
        acts += [(t, 100 * c + 1, at, at + 4000, "kernel", "pd_commit_979", 5),
                 (t, 100 * c + 1, at + 4000, at + 9000, "kernel", "gemm_pearlc_d", 5),
                 (t, 100 * c + 1, at + 4500, at + 6000, "kernel", "pd_tree_979", side),
                 (t, 100 * c + 1, at + 9000, at + 15000, "kernel", "pd_tiles_979", side)]
    acts.append((t, 450, 0, 1, "kernel", "stray", 5))
    r = P.lanes(sorted(ranges), acts)[label]
    assert r["calls"] == 4 and r["period_us"] == 10.0 and r["periods_us"] == [10.0] * 3
    assert r["lanes"]["main"] == {"start_us": 0.0, "span_us": 9.0, "busy_us": 9.0, "idle_before_us": 1.0}
    assert r["lanes"]["side0"] == {"start_us": 4.5, "span_us": 10.5, "busy_us": 7.5}
    assert r["lanes"]["side1"] == {"start_us": 4.5, "span_us": 10.5, "busy_us": 7.5}
    assert r["kernels"]["side0 pd_tree_979"] == {"count": 0.5, "mean_us": 1.5}
    assert [(a["lane"], a["name"]) for a in r["timeline"]] == [
        ("main", "pd_commit_979"), ("main", "gemm_pearlc_d"), ("side0", "pd_tree_979"), ("side0", "pd_tiles_979")]


def test_the_stamps_give_each_phase_since_its_ctas_start_and_the_tail_since_the_first_start():
    """glm53_profile's stamps: a pd_commit CTA's phases by its own SM's clock (at the clock's rate over its span on the timer),
    the CTAs' starts and ends and the tail's slots on the global timer since the first CTA's start, the tail CTA when
    pd_commit's last CTA ran it, and an unwritten slot left out."""
    import struct
    spec = importlib.util.spec_from_file_location("pearl_c_sm90_glm53_profile", HERE / "glm53_profile.py")
    P = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    slots, count = 16, 2
    words = [0] * (2 * slots * (count + 1))

    def put(row, slot, ns, clock):
        words[2 * (slots * row + slot):2 * (slots * row + slot) + 2] = [ns, clock]
    for r, (t0, c0) in enumerate(((1_000_000, 50_000), (1_004_000, 90_000))):   # CTA 1 starts 4 µs later, on another SM
        for slot, us in ((0, 0), (1, 10), (4, 11), (5, 20), (6, 21), (7, 30), (8, 33)):
            put(r, slot, t0 + 1000 * us, c0 + 2000 * us)                         # a 2 GHz clock
    put(count, 0, 1_037_000, 0)
    put(count, 1, 1_040_000, 0)
    put(count, 9, 1_050_000, 0)
    put(count, 15, 1, 7)
    rows, tail = P.stamp_rows(struct.pack(f"<{len(words)}Q", *words), count, slots)
    s = P.stamp_summary(rows, tail)
    assert s["rows"] == 2 and s["clock_ghz"] == 2.0 and "salt" not in s["since_cta_start_us"]
    assert s["since_cta_start_us"]["segs"] == {"median_us": 10.0, "max_us": 10.0}
    assert s["since_cta_start_us"]["seed"] == {"median_us": 33.0, "max_us": 33.0}
    assert s["cta_start_us"] == {"median": 2.0, "max": 4.0} and s["cta_end_us"] == {"median": 35.0, "max": 37.0}
    assert s["tail_us"] == {"start": 37.0, "level1": 40.0, "seed_a": 50.0} and s["tail_cta"] == [1, 7]


def test_h200_session_runs_the_tables_shapes():
    """h200.sh's run.py SHAPES are every in-domain (n, k) (`in_domain`), at decode's 64 and prefill's 8192; it times decode's
    forward whole (--whole, every in-domain shape) and prefill under the h3 chain beside the default one, and checks the
    captured passes' outputs in full after every lease."""
    sh = (HERE / "h200.sh").read_text()
    listed = re.search(r"for nk in ([\dx ]+); do", sh).group(1).split()
    assert {tuple(map(int, x.split("x"))) for x in listed} == set(G.in_domain())
    assert 'SHAPES="$SHAPES 64x$nk 8192x$nk"' in sh
    assert '"$D/glm53_timing.py" --whole --out "$O/whole"' in sh
    assert 'PEARLC_CHAIN=h3 "$PY" "$D/glm53_timing.py" --out "$O/timing-h3" --m 2048 8192' in sh
    assert sh.index('"$D/glm53_passes.py" full --out "$O/passes" --jobs "$(nproc)"') > sh.index('echo "lease rc $?"')


def _timing():
    spec = importlib.util.spec_from_file_location("pearl_c_sm90_glm53_timing", HERE / "glm53_timing.py")
    T = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    return T


class Card:
    """A stand-in GPU: a clock in ms that the work run on it advances, and the graph torch is capturing into, if any."""

    def __init__(self):
        self.t, self.capturing = 0.0, None

    def launch(self, work):
        if self.capturing is not None:
            self.capturing.work.append(work)
        else:
            work()


def _card(monkeypatch, torch) -> Card:
    """torch.cuda's events, graphs and streams over a `Card`."""
    card = Card()

    class Event:
        def __init__(self, enable_timing=False):
            self.at = None

        def record(self, stream=None):
            self.at = card.t

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return end.at - self.at

    class Graph:
        def __init__(self):
            self.work = []

        def replay(self):
            for f in self.work:
                f()

    @contextlib.contextmanager
    def graph(g, stream=None):
        card.capturing = g
        try:
            yield
        finally:
            card.capturing = None
    stream = SimpleNamespace(wait_stream=lambda other: None)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", graph)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    return card


class Run:
    """`PearlCDeviceRun`'s whole-capture surface over a `Card`: a call is 1 ms of GEMM and, with hashing, 0.5 ms of tile
    hashing, which a deferred capture runs beside the next call, and 0.25 ms more when made alone (its own graph's slot and
    launch).  A pass commits the calls its GPU work ran, in order; `leak` makes the deferred work leave a call's leaves
    unwritten ("deferred"), or the poison in them ("poison")."""

    def __init__(self, card: Card, order: dict):
        self.card, self.order, self.device = card, order, "cuda"
        self.graphs, self.hashing, self.whole_defer, self.leak = True, True, False, None
        self.log, self._whole, self.units, self.ran, self.poisoned = [], None, 0, [], False
        self.repeat_at = None

    def epoch_calls(self):
        return self.units

    def whole_key(self):
        return (self.hashing, self.whole_defer and self.hashing) if self.graphs else None

    def whole_capture_begin(self, key):
        self.log.append(("begin", key, self.hashing, self.whole_defer))
        self._whole = []

    def whole_capture_end(self):
        calls, self._whole = tuple(self._whole), None
        self.log.append(("end", calls))
        return calls

    def gemm(self, w, x, W, bias):
        m, captured = x.shape[0], self.card.capturing is not None
        deferred = captured and self.whole_defer and self.hashing
        if captured:
            self._whole.append((w, m))
        else:
            self.log.append(("call", w, m, self._whole is not None))
            self.units += 1
        cost = 1.0 + 0.5 * (self.hashing and not deferred) + 0.25 * (not captured)
        hashing = self.hashing

        def work():
            self.card.t += cost
            leaked = deferred and (self.leak == "deferred" or (self.leak == "poison" and self.poisoned))
            self.ran.append((w, m, hashing, leaked))
        self.card.launch(work)
        return x

    def whole_replay(self, calls):
        self.log.append(("replay", calls))
        return False

    def whole_replayed(self, calls):
        self.units += len(calls)

    def poison(self):
        self.log.append(("poison",))
        self.poisoned = True

    def end_pass(self):
        out = {"units": self.units, "hashing": self.hashing}
        if self.hashing and self.units:
            self.card.t += 0.125
            out.update(transcript_root=hashlib.sha512(repr(self.ran).encode()).hexdigest(), tiles=4 * len(self.ran),
                       activation_roots=hashlib.sha512(repr([r[:2] for r in self.ran]).encode()).hexdigest(),
                       screened_rows=sum(r[1] for r in self.ran), failing_rows=0, unsure_rows=0)
        self.log.append(("end_pass", self.units, self.hashing))
        self.units, self.ran, self.poisoned = 0, [], False
        return out


def _whole_setup(monkeypatch):
    torch = pytest.importorskip("torch")
    T, card = _timing(), _card(monkeypatch, torch)
    run = Run(card, {7: 0, 5: 1, 6: 2})
    weights = {w: torch.zeros(8 * (w - 4), 4) for w in (5, 6, 7)}
    xs = {w: torch.zeros(16, 4) for w in weights}

    def stock(cost, refuse=False, fail=False):
        def f(x):
            if fail or (refuse and card.capturing is not None):
                raise RuntimeError("operation not permitted when stream is capturing")
            card.launch(lambda: setattr(card, "t", card.t + cost))
        return {w: f for w in weights}
    return T, run, weights, xs, {"cutlass": stock(0.25), "triton": stock(0.5, refuse=True), "deepgemm": stock(0.25, fail=True)}


def test_whole_times_decodes_forward_captured_serial_and_deferred_beside_the_calls_one_at_a_time(monkeypatch):
    """--whole's row: each capture sets its pass's hashing and deferral before `whole_capture_begin`, warms up outside the
    capture and captures every weight once in slot order; each pearlc sample is a fresh pass of `INNER` replays (slots, graph,
    records); a stock forward torch can't capture runs eagerly, one whose warm-up fails is left out; and the hashing arms'
    passes, and one of each from poisoned outputs, commit alike."""
    T, run, weights, xs, stock = _whole_setup(monkeypatch)
    row = T.whole(run, weights, xs, stock, reps=3)
    assert row["ms"] == {"pearlc-calls": 3 * 1.75, "pearlc-whole": 3 * 1.5, "pearlc-deferred": 3.0, "pearlc-whole-nohash": 3.0,
                         "cutlass": 0.75, "triton": 1.5}
    assert row["ms_per_call"]["pearlc-deferred"] == 1.0 and row["calls"] == 3 and row["m"] == 16
    assert row["shapes"] == [[24, 4], [8, 4], [16, 4]]
    assert row["deferred_saving"] == {"vs_calls": 1 - 3 / 5.25, "vs_whole": 1 - 3 / 4.5}
    assert row["fastest_stock"] == "cutlass" and row["slowdown"]["pearlc-deferred"] == 4.0
    assert row["stock_graphed"] == {"cutlass": True, "triton": False, "deepgemm": False}
    assert set(row["stock_errors"]) == {"triton", "deepgemm"} and len(row["runs"]["triton"]) == 3
    assert row["pass_end_ms"] == {"pearlc-calls": 0.125, "pearlc-whole": 0.125, "pearlc-deferred": 0.125, "pearlc-whole-nohash": 0}
    assert row["commitments_equal"] and row["poisoned_equal"] and "commitments" not in row
    assert row["commitment"]["units"] == 3 * T.INNER and row["commitment"]["screened_rows"] == 16 * 3 * T.INNER
    begins = [e for e in run.log if e[0] == "begin"]
    assert begins == [("begin", (True, False), True, False), ("begin", (True, True), True, True),
                      ("begin", (False, False), False, False)]
    at = run.log.index(begins[1])
    assert run.log[at - 1][0] == "end_pass" and run.log[at + 1:at + 4] == [("call", w, 16, True) for w in (7, 5, 6)]
    assert run.log[at + 4] == ("end", ((7, 16), (5, 16), (6, 16))) and run.log[at + 5][0] == "end_pass"
    poisons = [i for i, e in enumerate(run.log) if e == ("poison",)]
    assert len(poisons) == 3 and all(run.log[i - 1][0] == "end_pass" for i in poisons)
    assert run.log[poisons[1] + 1:poisons[1] + 1 + T.INNER] == [("replay", ((7, 16), (5, 16), (6, 16)))] * T.INNER
    assert (run.hashing, run.whole_defer) == (True, False)


@pytest.mark.parametrize("leak", ["deferred", "poison"])
def test_whole_names_a_deferred_capture_that_commits_otherwise(monkeypatch, leak):
    """A deferred capture whose passes commit other bytes than the calls one at a time, always or only from poisoned outputs,
    is flagged, with every arm's commitments in the row."""
    T, run, weights, xs, stock = _whole_setup(monkeypatch)
    run.leak = leak
    row = T.whole(run, weights, xs, stock, reps=2)
    assert row["commitments_equal"] is (leak == "poison") and row["poisoned_equal"] is False
    assert row["commitments"]["pearlc-deferred"][0] != row["commitment"] or leak == "poison"
    assert row["poisoned"]["pearlc-whole"] == row["commitment"] != row["poisoned"]["pearlc-deferred"]


@pytest.mark.parametrize("args", [["--m", "512"], ["--schedule", "deferred"], ["--shapes", "512:6144", "512:6144"]])
def test_whole_refuses_what_it_cant_capture(tmp_path, args):
    """--whole runs the serial schedule at M up to GRAPH_ROWS over distinct shapes (a deferred call joins at once after its
    own shape's), and refuses anything else before it touches a card."""
    pytest.importorskip("torch")
    with pytest.raises(SystemExit) as e:
        _timing().main(["--whole", "--out", str(tmp_path / "out"), *args])
    assert e.value.code == 2 and not (tmp_path / "out").exists()


def _passes_module():
    spec = importlib.util.spec_from_file_location("pearl_c_sm90_glm53_passes", HERE / "glm53_passes.py")
    if spec.name not in sys.modules:
        mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[spec.name]


@pytest.fixture(scope="module")
def retained(tmp_path_factory):
    """A retained pass in glm53_passes' layout, as `prefill` and as `decode` (the same pass): two calls of a test instance of
    pearl-c-h100-v1-h3 (k = 128, 16 x 16 tiles), one with 12 filler rows, proved by the scheme's reference prover and written
    as the device run writes them, and each call's served y from the scheme over the whole call as one tile (U and α), its
    two FP32 products numpy's and its BF16 rounding torch's."""
    import random
    import shutil

    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    GP = _passes_module()
    sys.path.insert(0, str(HERE.parent / "clear"))
    import clear_work as CW
    from clear_audit import Prover
    from verity_vllm.protocols.pouw import pearl_c as PPD

    from verity.core.protocols.pouw.audit import (
        HM96_TRANSCRIPT,
        HM96_WEIGHTS,
        ROW_SCHEMA,
        TILE_SCHEMA,
        Epoch,
        Workload,
        activation_binding,
        hm96_salt,
        rows_domain,
        transcript_domain,
        weight_binding,
        weights_root,
    )
    from verity.core.protocols.pouw.coins import Challenger
    from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, Tile
    from verity.ml.protocols.pouw.schemes import pearl_c_work as PW
    from verity.ml.commitments.merkle import tree_levels

    scheme = "pearl-c-h100-v1-h3"
    s = PPD.scheme_of(scheme, k_min=32, tile=(16, 16))
    rng, k = np.random.default_rng(11), 128
    words = lambda r: (rng.standard_normal((r, k)).astype(np.float32)).view(np.uint32) & 0xFFFF0000
    rows = lambda x: [r.tobytes() for r in x]
    bf16 = lambda x: b"".join((np.frombuffer(r, "<u4") >> 16).astype("<u2").tobytes() for r in rows(x))
    laid = lambda d, leaves: b"".join(b"".join(level) for level in tree_levels(d, leaves))
    wrows, acts = {0: words(16), 1: words(32)}, {0: words(32), 1: words(16)}
    acts[0][20:] = 0
    work = Workload((Matmul(0, Shape(32, k, 16), 0), Matmul(1, Shape(16, k, 32), 1)))
    prover = Prover(s, {w: rows(x) for w, x in wrows.items()}, salt_key=bytes([1]) * 32)
    salted = lambda v, nonce, i: v + hm96_salt(prover.salt_key, nonce, i)
    ch = Challenger(0, random.Random(7).randbytes)
    coins = ch.commitment, ch.open("salt", weights_root(prover.weight_roots))
    epoch = Epoch.start(work, prover.weight_roots, *coins)
    tr = prover.prove(epoch, {i: rows(x) for i, x in acts.items()})
    splits = {w: PW.Split(k) for w in wrows}
    records = CW.declare(s, epoch, {i: rows(x) for i, x in acts.items()}, {w: rows(x) for w, x in wrows.items()}, splits,
                         real_rows={0: 20})
    base = tmp_path_factory.mktemp("retained")
    d = base / "pass"
    for sub in ("calls", "weights", "y"):
        (d / sub).mkdir(parents=True)
    weights = {}
    for w, x in wrows.items():
        dom = rows_domain(weight_binding(w), -2, len(x), s.commitment_hash)
        weights[str(w)] = {"n": len(x), "k": k, "root": prover.weight_roots[w].hex(), "excluded_b": list(records[w].excluded_b),
                           "rows": PPD._write(d / "weights" / f"{w}.rows.bf16", bf16(x), digest=False),
                           "tree": PPD._write(d / "weights" / f"{w}.tree",
                                              laid(dom, [dom.leaf(j, j, ROW_SCHEMA, salted(r, HM96_WEIGHTS + w, j))
                                                         for j, r in enumerate(rows(x))]))}
    matmuls, y_words = [], 0
    for u, rec, root in zip(work.matmuls, records, tr.activation_roots):
        x, m, n = rows(acts[u.index]), u.shape.m, u.shape.n
        dom = rows_domain(activation_binding(epoch.salt, u.index), u.index, m, s.commitment_hash)
        matmuls.append({"index": u.index, "m": m, "k": k, "n": n, "weight": u.weight, "real_rows": rec.real_rows,
                        "k_kept": rec.k_kept, "excluded_a": list(rec.excluded_a), "root_a": root.hex(),
                        "rows": PPD._write(d / "calls" / f"{u.index}.rows.bf16", bf16(acts[u.index]), digest=False),
                        "tree": PPD._write(d / "calls" / f"{u.index}.tree",
                                           laid(dom, [dom.leaf(i, i, ROW_SCHEMA, salted(r, u.index, i)) for i, r in enumerate(x)]))})
        ctx = Context(epoch.salt, u, root, epoch.weight_roots[u.weight])
        noise = s.noise(ctx)
        ops = s.encode(ctx, noise, x, rows(wrows[u.weight]), Tile(tuple(range(m)), tuple(range(n))))
        uu = np.array(s.checked(ctx, ops).u, np.uint32)[:rec.real_rows].view(np.float32)
        inv = lambda alpha: np.float32(1) / np.array(alpha, np.uint32).view(np.float32)
        y = torch.from_numpy((uu * inv(ops.a.alpha)[:rec.real_rows, None]) * inv(ops.b.alpha)[None, :]).to(torch.bfloat16)
        (d / "y" / f"{u.index}.bf16").write_bytes(y.view(torch.int16).numpy().tobytes())
        y_words += y.numel()
    count = tr.commitment.count
    leaves = [prover.digest(t) for t in range(count)]
    dt = transcript_domain(epoch, count, s.commitment_hash)
    head = {"schema": PPD.MANIFEST_SCHEMA, "run": "r20261023-120000-test", "entry": f"pouw:{scheme}", "scheme": scheme,
            "instance": {"k_min": 32, "tile": [16, 16]}, "coins": coins[0].to_json(), "salt_opening": coins[1].to_json(),
            "salt": epoch.salt.hex(), "records_root": PW.records_root(epoch.salt, epoch.weights_root, splits, records).hex(),
            "commitment": {"root": tr.commitment.root.hex(), "count": count}, "leaves": PPD._write(d / "leaves", b"".join(leaves)),
            "tree_t": PPD._write(d / "tree_t", laid(dt, [dt.leaf(i, i, TILE_SCHEMA, salted(v, HM96_TRANSCRIPT, i))
                                                       for i, v in enumerate(leaves)])),
            "hm96_salt_key": prover.salt_key.hex()}
    PPD.write_manifest(d, head, weights, matmuls)
    (base / "coins.json").write_text(json.dumps(coins[0].to_json()) + "\n")
    for name in GP.FULL:
        shutil.copytree(d, base / name)
    shutil.rmtree(d)
    return SimpleNamespace(base=base, count=count, y_words=y_words, GP=GP)


def _copy(retained, tmp_path) -> Path:
    import shutil

    shutil.copytree(retained.base, tmp_path / "passes")
    return tmp_path / "passes"


def test_full_checks_every_tile_and_every_served_y_word(retained, tmp_path):
    """`full` takes every tile of prefill and decode through the verifier's check_tile (openings, then C~ and U against the
    leaf) and every real row's served y against y_unit, on the batch forms with the scalar reference on every 2nd tile too."""
    out = _copy(retained, tmp_path)
    assert retained.GP.main(["full", "--out", str(out), "--jobs", "2", "--cross-check", "2"]) == 0
    res = json.loads((out / "full.json").read_text())
    for name in retained.GP.FULL:
        r = res[name]
        assert r["ok"] and r["tiles"] == retained.count and r["failed"] == 0, r["line"]
        assert r["shapes"] == {"16x128": 2, "32x128": 2} and r["words"] == 32 * 16 + 16 * 32
        assert r["y_words"] == retained.y_words == 20 * 16 + 16 * 32
        assert r["replay"] == "batch" and r["cross_checked"] == (retained.count + 1) // 2 and r["cross_check_failed"] == []
    assert res["ok"]


def test_full_fails_on_one_served_y_word_or_one_activation_row_and_names_it(retained, tmp_path):
    """One served y word off (decode's call 1, row 3, column 17) fails that tile alone, naming the word; one byte of an
    activation row off (prefill's call 0, row 4) fails that row's tiles at their openings."""
    np = pytest.importorskip("numpy")
    out = _copy(retained, tmp_path)
    y = out / "decode" / "y" / "1.bf16"
    words = np.fromfile(y, "<u2").reshape(16, 32)
    words[3, 17] ^= 1
    words.tofile(y)
    a = out / "prefill" / "calls" / "0.rows.bf16"
    raw = bytearray(a.read_bytes())
    raw[2 * 128 * 4] ^= 0x40
    a.write_bytes(bytes(raw))
    assert retained.GP.main(["full", "--out", str(out), "--jobs", "1", "--cross-check", "0"]) == 1
    res = json.loads((out / "full.json").read_text())
    dec, pre = res["decode"], res["prefill"]
    assert dec["failed"] == 1 and "served y differs at 1 of 256 words, first row 3 column 17" in dec["failures"][0]["reason"]
    assert pre["failed"] == 1 and pre["failures"] == [{"tile": 0, "reason": "activation opening"}]
    assert not res["ok"]


def test_full_requires_the_served_y_unless_told_the_pass_has_none(retained, tmp_path):
    """A pass without its served y fails `full`, which checks C~ and U alone only under --without-y."""
    import shutil

    out = _copy(retained, tmp_path)
    for name in retained.GP.FULL:
        shutil.rmtree(out / name / "y")
    assert retained.GP.main(["full", "--out", str(out), "--jobs", "1"]) == 1
    res = json.loads((out / "full.json").read_text())
    assert all("call 0: no served y of 20 x 16 BF16 words" in res[x]["error"] for x in retained.GP.FULL) and not res["ok"]
    assert retained.GP.main(["full", "--out", str(out), "--jobs", "1", "--without-y"]) == 0
    res = json.loads((out / "full.json").read_text())
    assert all(res[x]["y_words"] == 0 and "served y not retained" in res[x]["line"] for x in retained.GP.FULL)
