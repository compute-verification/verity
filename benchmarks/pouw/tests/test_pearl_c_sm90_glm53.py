"""GLM-5.3's linears under pearl-c-h100-v1-h3 (benchmarks/pouw/pearl_c_sm90/glm53.py): one TP8 rank's shapes from the
checkpoint's config, the sm_90a kernels' domain verdict on each, γ's formula against a Lean price twin, the H200
session's shapes the table's, and glm53_timing's --whole over a stand-in card."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
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
    IKW = ID + IH, IL = I / 8, MIL = SIL = MI / 8; compute-accounting's per-rank K (256, 1536, 2048, 6144) are the FP8
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
    """k = 256 and 512 are under the record's k ≥ 1,024; n = 160 is not a multiple of 64.  Every other linear is in, with γ
    at both FADD readings, the data sheet's (cheaper add) a little above the record's, and γ falling with n."""
    rows = {r["name"]: r for r in G.table()}
    refused = {name for name, r in rows.items() if r["refused"]}
    assert refused == {"self_attn.kv_b_proj", "self_attn.indexer.wk_weights_proj", "mlp.shared_experts.down_proj",
                       "mlp.experts.w2"}
    assert all("1024 <= k" in rows[x]["refused"][0] for x in refused - {"self_attn.indexer.wk_weights_proj"})
    assert "n of 64" in rows["self_attn.indexer.wk_weights_proj"]["refused"][0]
    for name, r in rows.items():
        if name not in refused:
            assert 0 < r["gamma"]["record"] < r["gamma"]["sheet"] < 0.011, name
    assert rows["mlp.experts.w13"]["gamma"]["record"] > rows["self_attn.o_proj"]["gamma"]["record"]


def test_expert_parallel_brings_the_experts_into_the_domain():
    """With the routed experts expert-parallel (EP = TP = 8) a rank holds 32 whole experts, w13 (4096, 6144) and w2
    (6144, 2048), both in the domain; the shared expert stays sharded, its down at k = 256 refused."""
    rows = {r["name"]: r for r in G.table(ep=True)}
    assert {x: (rows[x]["n"], rows[x]["k"]) for x in ("mlp.experts.w13", "mlp.experts.w2")} == {
        "mlp.experts.w13": (4096, 6144), "mlp.experts.w2": (6144, 2048)}
    assert {x for x, r in rows.items() if r["refused"]} == {"self_attn.kv_b_proj", "self_attn.indexer.wk_weights_proj",
                                                            "mlp.shared_experts.down_proj"}
    shapes = G.in_domain()
    assert (4096, 6144) in shapes and shapes[(4096, 6144)]["ep"] and not shapes[(6144, 2048)]["ep"]
    assert shapes[(6144, 2048)]["name"] == "self_attn.o_proj"


def test_gamma_is_the_lean_twins():
    """`Pouw.PearlC.pearlCGammaSm120v1Cast32Rev1_8192`: v1 under rev1 (cap 1/400) at the sm_120 record, 8192³, the honest cast
    at 32 (24 over the credited 8), γ = 173561/22302500."""
    from verity.core.protocols.pouw.schemes import pearl_c as C
    from verity.core.protocols.pouw.schemes.pearl_c_device import SM120

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
    out = P.report(sorted(ranges), acts)
    assert set(out) == {label}
    r, steps = out[label], out[label]["steps"]
    assert r["calls"] == 2 and r["host_wall_us"] == 0.1 and r["pass_end_host_us"] == 0.4
    assert {s: round(v["gpu_us"], 9) for s, v in steps.items()} == {
        "step commit_a": 2.0, "step gemm_pearlc": 2.0, "host screen": 0.5, "host after": 0.1, "host other": 0.05, "step hash_leaf": 1.0}
    assert steps["step commit_a"]["host_us"] == 0.02 and steps["step commit_a"]["kernels"]["pd_commit_851"]["count"] == 1
    assert round(steps["host other"]["host_us"], 9) == 0.05 and steps["host screen"]["host_us"] == 0.01
    assert r["gpu_busy_us"] == 3.85 and r["gpu_span_us_per_call"] == 4.5 and r["solo_gpu_span_us"] == 2.0
    assert (r["hash_decode"], r["hash_fast"], r["fallbacks"]) == (["pd_commit_851"], [], ["s512_tile_leaves"])


def test_h200_session_runs_the_tables_shapes():
    """h200.sh's run.py SHAPES are every in-domain (n, k) (`in_domain`), at decode's 64 and prefill's 8192; it times decode's
    forward whole (--whole, every in-domain shape) and prefill under the h3 chain beside the default one."""
    sh = (HERE / "h200.sh").read_text()
    listed = re.search(r"for nk in ([\dx ]+); do", sh).group(1).split()
    assert {tuple(map(int, x.split("x"))) for x in listed} == set(G.in_domain())
    assert 'SHAPES="$SHAPES 64x$nk 8192x$nk"' in sh
    assert '"$D/glm53_timing.py" --whole --out "$O/whole"' in sh
    assert 'PEARLC_CHAIN=h3 "$PY" "$D/glm53_timing.py" --out "$O/timing-h3" --m 2048 8192' in sh


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
