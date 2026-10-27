"""HIGHUTIL's period on the CPU: the pick draw, ε, the operand openings, and whole periods at small shapes through v31.py's
no-fold reference (`period.RefDriver`) with an in-process coin source: an honest period is accepted, and a skipping
prover's, a tampered root's and a swapped operand's are refused."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import period as P
import pipes
import terms


def test_eps_skip_at_delta_one_percent():
    assert math.isclose(P.eps_skip(2048), math.log(100) / (P.ETA * 2048))
    assert P.eps_skip(462) < 0.01 < P.eps_skip(460)


def test_picks_are_the_coin_s_and_weigh_each_mac_equally():
    wl = {"layers": [list(x) for x in P.LINEARS], "m": 16384, "nb": 32, "lp": 512, "L": 16384}
    P.configure(wl)
    P.configure({**wl, "layers": [[c, n, k] for c, n, k, _, _ in P.calls(P.LINEARS)]})
    coin, receipt = b"\x01" * 64, b"\x02" * 64
    a = P.picks(coin, "p", 3, receipt, 300, 400, 32768)
    assert a == P.picks(coin, "p", 3, receipt, 300, 400, 32768)
    assert a != P.picks(coin, "p", 3, b"\x03" * 64, 300, 400, 32768)
    w = np.array([n * k for _, n, k in P.LAYERS], dtype=float)
    counts = np.bincount([p.layer for p in a], minlength=len(w))
    assert np.all(np.abs(counts - 400 * w / w.sum()) < 5 * np.sqrt(400 * w / w.sum()) + 1)
    for p in a:
        _, n, k = P.LAYERS[p.layer]
        sh = P.shape(n, k, 32768)
        assert 0 <= p.step < 300 and 0 <= p.i < 16384 and 0 <= p.j < n and 1 <= p.s <= sh.S


def test_seeded_activations_look_quantized_and_rarely_reach_full_range():
    a = P.seeded_activations(64, 4096, 5).astype(np.int64)
    assert a.min() >= -127 and a.max() <= 127
    assert 0 < (np.abs(a) > 32).mean() < 2 / 1024
    assert 7 < a[np.abs(a) <= 32].std() < 11


def test_the_card_s_share_and_the_terms_of_the_rest_sum_to_one():
    P.configure({"layers": [["a", 256, 512]], "m": 1024, "nb": 4})
    card = {"sku": "test", "clock": {"sm": {"value": 2000, "source": "s"}}, "rate": {"tensor": {"value": 1024, "source": "s"}},
            "sm": {"count": {"value": 100}}}
    assert P.theta(card) == 1024 * 100 * 2000e6 and P.theta(card, 1500) == 1024 * 100 * 1500e6
    steps, macs = 10, 10 * P.step_macs()
    blocks = lambda gpu_ms: [{"gpu_ms": gpu_ms, "sm_mhz": 1500}]  # noqa: E731
    results = [{"arm": "proto", "cheat": "none", "credited_macs": macs, "macs": macs, "wall_s": 0.02, "steps": steps,
                "blocks": blocks(18.0), "gaps_s": [], "session_open_s": 1.0, "q": 64, "eps_skip": 0.07, "accepted": True,
                "verdict": "ACCEPT"},
               {"arm": "plain", "cheat": "none", "macs": macs, "wall_s": 0.012, "steps": steps, "blocks": blocks(11.0)}]
    s = P.summarize(results, P.Plan(1, steps, 64), card=card)
    assert math.isclose(s["certified_share"]["card"], macs / (P.theta(card) * 0.02))
    assert math.isclose(s["certified_share"]["run_clock"], s["certified_share"]["card"] * 2000 / 1500)
    t = terms.terms(s, {"busy_share": 0.9, "gemm_share": 0.7})
    assert math.isclose(t["sum"], 1) and math.isclose(t["terms"]["clock"], 0.25)
    assert math.isclose(t["terms"]["certified"], s["certified_share"]["card"])
    assert math.isclose(t["eps"], 0.07 + 1 - t["terms"]["certified"])


def test_gamma_is_against_the_first_baseline_timed_and_each_baseline_has_its_own():
    P.configure({"layers": [["a", 256, 512]], "m": 1024, "nb": 4})
    steps, macs = 10, 10 * P.step_macs()
    blocks = lambda gpu_ms, mhz=1500: [{"gpu_ms": gpu_ms, "sm_mhz": mhz}]  # noqa: E731
    proto = {"arm": "proto", "cheat": "none", "credited_macs": macs, "macs": macs, "wall_s": 0.03, "steps": steps,
             "blocks": blocks(24.0), "gaps_s": [], "session_open_s": 1.0, "q": 64, "eps_skip": 0.07, "accepted": True,
             "verdict": "ACCEPT"}
    plain = lambda base, wall, gpu: {"arm": "plain", "baseline": base, "cheat": "none", "macs": macs, "wall_s": wall,  # noqa: E731
                                     "steps": steps, "blocks": blocks(gpu, 1200)}
    s = P.summarize([proto, plain("lt", 0.01, 8.0), plain("int_mm", 0.015, 12.0)], P.Plan(1, steps, 64))
    assert s["sm_mhz_median"] == 1500 and s["baselines"]["lt"]["sm_mhz_median"] == 1200   # the run clock is the protocol's
    assert s["baseline"] == "lt" and math.isclose(s["gamma_gpu"], 3.0) and math.isclose(s["gamma_wall"], 3.0)
    assert math.isclose(s["baselines"]["int_mm"]["gamma_gpu"], 2.0) and math.isclose(s["baselines"]["int_mm"]["gamma_wall"], 2.0)
    assert s["baselines"]["lt"]["gamma_gpu"] == s["gamma_gpu"]
    assert P.baselines("lt,int_mm") == ["lt", "int_mm"]
    with pytest.raises(SystemExit):
        P.baselines("scaled_mm")


def test_the_and_credit_per_mac_is_main_s_s8_accumulator_lowering_on_both_cards_imma_steps():
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation.bits import compile as compile_bits
    from verity.ml.circuits.ops.boolean import gemm as BG
    for dot in (BG.GEMM_SM120_S8, BG.GEMM_SM90_S8):
        for k in (64, 128):
            assert compile_bits(bind(BG.GemmAccumulatorS8, K=k, **dot)).counts["and"] * P.ANDS_PER_MAC[1] == k * P.ANDS_PER_MAC[0]


def test_ncu_s_raw_csv_reads_as_each_kernel_s_tensor_pipe_share_and_macs():
    head = ["ID", "Process ID", "Kernel Name", *pipes.METRICS]
    units = ["", "", "", "ns", "", "%", "hz"]
    launches = [["0", "7", "Kernel2", "2,000,000", "4000", "90", "2000000000"],
                ["1", "7", "noise_kernel", "500,000", "0", "0", "2000000000"],
                ["2", "7", "Kernel2", "2,000,000", "4000", "80", "1000000000"]]
    text = "==PROF== Connected to process 7\n" + "\n".join(",".join(f'"{x}"' for x in r) for r in [head, units, *launches])
    card = {"clock": {"sm": {"value": 2000}}, "rate": {"tensor": {"value": 1}}, "sm": {"count": {"value": 1}}}
    r = pipes.pipes(text, card=card)
    gemm, noise = r["kernels"]
    assert (gemm["name"], gemm["launches"], noise["name"]) == ("Kernel2", 2, "noise_kernel")
    assert math.isclose(gemm["s"], 4e-3) and math.isclose(gemm["tensor_active"], 0.85) and math.isclose(gemm["mhz"], 1500)
    assert r["macs"] == 4000 and noise["tensor_active"] == 0 and math.isclose(r["tensor_active"], 0.85 * 4 / 4.5)
    assert math.isclose(gemm["share_of_theta"], 4000 / (4e-3 * 1500e6))


@dataclass
class _Opening:
    label: str
    index: int
    after: bytes
    coin: bytes


class _Client:
    def __init__(self, secret: bytes) -> None:
        self.secret, self.opened = secret, set()

    def open(self, label: str, index: int, after: bytes) -> _Opening:
        assert (label, index) not in self.opened, "a coin opens once"
        self.opened.add((label, index))
        return _Opening(label, index, after, hashlib.sha512(self.secret + label.encode() + P.u64(index) + after).digest())

    def open_many(self, requests):
        return [self.open(*r) for r in requests]

    def close(self) -> None:
        pass


class _Coins:
    """An in-process coin source: each coin opens once, after what it was asked after."""

    class _Session:
        session = "in-process"

        def open(self, root, layout, minutes, scheme):
            return "in-process", root[:64], None

    def session(self):
        return self._Session()

    def connect(self, address, secret):
        return _Client(bytes.fromhex(secret))

    def evidence(self, session, root, openings, dest):
        return {"trusted": True, "bad_openings": [], "session": session.session}


class _Serial:
    def map(self, fn, jobs, chunksize=1):
        return map(fn, jobs)


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    cfg = tmp_path_factory.mktemp("cfg")
    assert P.main(["prep", str(cfg), "--small"]) == 0
    prep = json.loads((cfg / "prep.json").read_text())
    P.configure(prep)
    secret = b"\x07" * 64
    trees = P.operand_trees(cfg, secret, _Serial())
    drv = P.RefDriver(cfg, prep["row"], secret)
    work = tmp_path_factory.mktemp("work")
    return P.Period(drv, _Coins(), trees, cfg, prep["row"], prep["source"], secret, work, _Serial())


def test_every_operand_row_opens_against_its_root_and_no_other(small):
    P.configure(json.loads((small.cfg / "prep.json").read_text()))
    for (kind, li, t), tree in small.trees.items():
        name, n, k = P.LAYERS[li]
        rows = P.M if kind == "A" else n
        X = P.operand(small.cfg, kind, li, t)
        j = (7 * li + 3 * t + 1) % rows
        op = tree.opening(X, j)
        assert P.operand_reason(kind, name, t, rows, k, tree.root, j, op) is None
        other = small.trees[(kind, li, 1 - t)]
        assert P.operand_reason(kind, name, t, rows, k, other.root, j, op)
        bad = (bytes([op[0][0] ^ 1]) + op[0][1:],) + op[1:]
        assert P.operand_reason(kind, name, t, rows, k, tree.root, j, bad)


def test_an_honest_period_is_accepted(small):
    """The whole period at small shapes, over both blocks of the model: coins, seeds, commits, paths, answers and the
    verifier's checks."""
    r = small.proto("t-honest", P.Plan(2, 2, 12))
    assert r["accepted"], r["verdict"]
    assert r["credited_macs"] == r["macs"] == 4 * P.step_macs() and r["q"] == 12
    assert r["verdict"].startswith("ACCEPT")
    assert len(r["registration"]["operand_roots"]["A"][0]) == P.NB == 2


def test_a_skipping_prover_is_refused(small):
    """A prover that leaves noise tile 1 out of every call: a pick catches it when its inner index lands in that tile (1/8
    of layer a's, 1/16 of b's), so 64 picks miss it with probability about 0.2%; the coins here are fixed."""
    r = small.proto("t-skip", P.Plan(2, 2, 64), "skip:1")
    assert not r["accepted"] and r["credited_macs"] == 0 and r["failed"] > 0, r["verdict"]
    assert r["verdict"].startswith("REFUSE")


def test_down_runs_as_k_slices_whose_segments_are_at_most_t_max_noise_periods():
    try:
        P.configure({"layers": [list(x) for x in P.LINEARS], "m": 16384, "nb": 32, "lp": 512, "L": 16384})
        assert P.calls(P.LINEARS) == [("qkv", 6144, 4096, "qkv", 0), ("o", 4096, 4096, "o", 0), ("up", 28672, 4096, "up", 0)] + \
            [(f"down.{s}", 4096, 7168, "down", 7168 * s) for s in range(2)]
        assert [P.lp_of(k) for k in (4096, 7168)] == [512, 1024]
        P.configure({"layers": [list(x) for x in P.LINEARS], "m": 16384, "nb": 32, "lp": 512, "L": 16384, "lp_max": 512})
        assert P.calls(P.LINEARS)[3:] == [(f"down.{s}", 4096, 3584, "down", 3584 * s) for s in range(4)]
        P.configure({"layers": [list(x) for x in P.LINEARS], "m": 16384, "nb": 32, "lp": 512, "L": 4096})
        assert P.calls(P.LINEARS)[3] == ("down", 4096, 14336, "down", 0) and P.lp_of(14336) == 512
    finally:
        P.configure({"layers": [list(x) for x in P.LINEARS], "m": 16384, "nb": 32, "lp": 512, "L": 16384})


@pytest.mark.parametrize("lp_max,layers,lps", [(128, ["a", "b.0", "b.1"], [128, 128, 128]), (1024, ["a", "b"], [128, 256])])
def test_one_segment_a_call_when_L_exceeds_K(small, tmp_path, lp_max, layers, lps):
    """The run's setting, L past every layer's K (S = 1, the output's low 16 bits the only words), with layer b (K = 16 L')
    as two K-slices when L' stops at 128, else whole at L' = 256: an honest period is accepted and a skipping prover
    refused, as at S > 1."""
    cfg = tmp_path / "cfg"
    assert P.main(["prep", str(cfg), "--small", "--L", "2048", "--lp-max", str(lp_max)]) == 0
    prep = json.loads((cfg / "prep.json").read_text())
    try:
        P.configure(prep)
        assert [x[0] for x in P.LAYERS] == layers
        assert [P.shape(n, k, prep["row"]).lp for _, n, k in P.LAYERS] == lps
        assert [P.shape(n, k, prep["row"]).S for _, n, k in P.LAYERS] == [1] * len(layers)
        secret = b"\x07" * 64
        per = P.Period(P.RefDriver(cfg, prep["row"], secret), _Coins(), P.operand_trees(cfg, secret, _Serial()), cfg,
                       prep["row"], prep["source"], secret, tmp_path, _Serial())
        r = per.proto("t-s1", P.Plan(2, 2, 12))
        assert r["accepted"], r["verdict"]
        r = per.proto("t-s1-skip", P.Plan(2, 2, 64), "skip:1")
        assert not r["accepted"] and r["failed"] > 0, r["verdict"]
    finally:
        P.configure(json.loads((small.cfg / "prep.json").read_text()))


def test_a_tampered_root_is_refused(small, monkeypatch):
    """A root the GPU didn't commit (here the first call's, altered after the block) fails every pick of that call."""
    real = small.drv.block

    def block(b, g0, steps, arm, cheat, seeds, roots):
        got = real(b, g0, steps, arm, cheat, seeds, roots)
        raw = bytearray(Path(roots).read_bytes())
        raw[0] ^= 1
        Path(roots).write_bytes(bytes(raw))
        return got
    monkeypatch.setattr(small.drv, "block", block)
    r = small.proto("t-root", P.Plan(1, 1, 24))
    first = [p for p in r["failures"] if "layer=0" in p]
    assert not r["accepted"] and first


class _FirstBlockOnly(P.RefDriver):
    """A prover that runs every step on block 0's operands, never the registered block 1's."""

    def block_of(self, step: int) -> int:
        return 0


def test_computing_on_another_block_s_operands_is_refused(small, monkeypatch):
    """The block-1 step's roots commit values its registered operands don't produce, so its picks fail (all 24 picks
    missing that step: 2^-24)."""
    monkeypatch.setattr(small, "drv", _FirstBlockOnly(small.cfg, small.row, b"\x07" * 64))
    r = small.proto("t-swap", P.Plan(1, 2, 24))
    assert not r["accepted"] and 0 < r["failed"] < 24, r["verdict"]
