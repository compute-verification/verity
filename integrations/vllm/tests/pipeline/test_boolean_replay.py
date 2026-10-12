"""The Boolean re-verification (`pipeline.boolean_replay`): words to bits and back, a Call evaluated by the Boolean version of its
Definition equal to its word evaluation and keyed by the member it is committed as, a Call without a Boolean version or whose
Boolean version changes its boundary not evaluated by name, the Commit's records copied beside a link to its bundle, and the
replay's picks tabulated by family."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import Builder, bind
from verity.ml.circuits.refs import Affine, Coll
from verity.ml.circuits.types import Array, Value
from verity_vllm.pipeline import boolean_replay as BR
from verity_vllm.program.kernels.kernel_registry import Declined
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import boolean_silu as BSL

W16 = Value(16)
SPECIAL = [0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC0, 0x0001, 0x8001, 0x7F7F, 0x4120, 0xC2C8]


def test_words_go_to_bits_least_significant_first_and_back() -> None:
    widths = [16, 16, 32, 1, 1, 8]
    words = np.array([0x3F80, 0x0001, 0xDEADBEEF, 1, 0, 0xA5], dtype=np.uint64)
    bits = BR.to_bits(words, widths)
    assert bits.tolist()[:16] == [(0x3F80 >> j) & 1 for j in range(16)] and bits.size == sum(widths)
    assert BR.to_words(bits, widths).tolist() == words.tolist()
    assert BR.to_bits(np.array([-1, 5], dtype=np.int32), [8, 8]).tolist() == [1] * 8 + [1, 0, 1, 0, 0, 0, 0, 0]
    with pytest.raises(Declined, match="wider than its 16-bit element"):
        BR.to_bits(np.array([1 << 16], dtype=np.uint32), [16])
    with pytest.raises(Declined, match="3 words for 2 elements"):
        BR.to_bits(np.zeros(3, np.uint16), [16, 16])


def test_a_long_run_of_one_width_converts_in_blocks(monkeypatch) -> None:
    words = np.random.default_rng(4).integers(0, 1 << 16, size=11, dtype=np.uint64)
    whole = BR.to_bits(words, [16] * 11)
    monkeypatch.setattr(BR, "_RUN_MAX", 4)
    assert [hi - lo for _, lo, hi, _ in BR._runs([16] * 11)] == [4, 4, 3]
    assert np.array_equal(BR.to_bits(words, [16] * 11), whole) and BR.to_words(whole, [16] * 11).tolist() == words.tolist()


class _Index:
    def __init__(self, ports: dict[str, str]) -> None:
        self.ports = ports

    def member_of_port(self, i: int) -> dict[str, str]:
        return dict(self.ports)


def _row(fn, *types):
    """A one-Call root body: its nodes, and the instance row of the Call (row 0, K = 0)."""
    B = Builder(tuple((f"a{g}", t) for g, t in enumerate(types)), Value(1))
    B.call(fn, *(Coll(t, Affine("p", g, 0, t.leaves)) for g, t in enumerate(types)))
    return B.nodes, {"i": 0, "spec": descriptor_id(B.nodes[0].fn)}


def test_a_call_is_evaluated_on_bits_by_its_boolean_version_and_equals_its_word_evaluation(tmp_path) -> None:
    rng = np.random.default_rng(3)
    word = bind(b1.SiluMul, I=8)
    nodes, r = _row(word, Array(16, W16))
    ev = BR.BooleanRows(nodes, 0, tmp_path / "rows.jsonl")
    for gu in (np.array(SPECIAL + SPECIAL[:4], dtype=np.uint16), rng.integers(0, 1 << 16, 16).astype(np.uint16)):
        got = ev(_Index({"out": "act"}), r, {}, [gu])
        want = np.asarray(evaluate_batch(word, gu[None].astype(np.uint64))).reshape(-1)
        assert list(got) == ["act"] and got["act"].dtype == np.uint16
        assert got["act"].tolist() == want.tolist()
    assert ev.sigma == {r["spec"]: bind(BSL.SiluMul, I=8).id}
    lines = [json.loads(x) for x in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert [x["boolean"] for x in lines] == ["SiluMul_v3{I=8}"] * 2 and {x["i"] for x in lines} == {0}
    assert ev(_Index({}), r, {}, [gu]).keys() == {"out"}                     # an interior row: keyed by its port


def test_a_call_without_a_boolean_version_is_not_evaluated_by_name(tmp_path) -> None:
    nodes, r = _row(bind(b1.Gemm, K=4, N=2), Array(4, W16), Array(2, Array(4, W16)))
    ev = BR.BooleanRows(nodes, 0, tmp_path / "rows.jsonl")
    with pytest.raises(Declined, match="no Boolean version"):
        ev(_Index({}), r, {}, [np.zeros(4, np.uint16), np.zeros(8, np.uint16)])
    assert ev.sigma == {r["spec"]: None} and "Gemm" in ev.why[r["spec"]]
    assert "no Boolean version" in json.loads((tmp_path / "rows.jsonl").read_text())["why"]
    with pytest.raises(Declined, match="is no row of the Program"):
        ev.version("Unknown_v1")
    assert ev.prewarm([{"spec": r["spec"]}, {"spec": "Unknown_v1"}]) == {
        f"boolean:{r['spec']}": f"not compiled: {ev.why[r['spec']]}"[:300],
        "boolean:Unknown_v1": "not compiled: Unknown_v1 is no row of the Program"}


def test_a_picks_stated_peak_bounds_what_its_evaluation_allocates(tmp_path) -> None:
    """A process's first evaluation after the prewarm also imports and traces, once (about 7 MB here), which `PEAK_MARGIN`
    absorbs; which test ran first in the worker decides whether this one sees it, so each evaluation is measured."""
    import tracemalloc

    nodes, r = _row(bind(b1.SiluMul, I=1024), Array(2048, W16))
    ev = BR.BooleanRows(nodes, 0)
    ev.prewarm([r])
    gu = np.random.default_rng(5).integers(0, 1 << 16, 2048).astype(np.uint16)

    def held() -> int:
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            ev(_Index({"out": "act"}), r, {}, [gu])
            return tracemalloc.get_traced_memory()[1] - base
        finally:
            tracemalloc.stop()

    assert held() <= ev.peak_bytes(r)
    assert held() <= ev.peak_bytes(r) / BR.PEAK_MARGIN
    nodes, r = _row(bind(b1.Gemm, K=4, N=2), Array(4, W16), Array(2, Array(4, W16)))
    assert BR.BooleanRows(nodes, 0).peak_bytes(r) == 0                   # declined before it allocates


def test_a_boolean_version_that_changes_the_calls_boundary_is_refused(monkeypatch) -> None:
    import verity_vllm.program.boolean as BB

    monkeypatch.setattr(BB, "boolean_version", lambda fn: bind(BSL.SiluMul, I=4))
    nodes, r = _row(bind(b1.SiluMul, I=8), Array(16, W16))
    with pytest.raises(Declined, match=r"SiluMul_v3\{I=4\} does not keep the Call's boundary"):
        BR.BooleanRows(nodes, 0).version(r["spec"], nodes[0])


def test_the_commit_records_are_copied_beside_a_link_to_the_bundle(tmp_path) -> None:
    commit, out = tmp_path / "commit", tmp_path / "out"
    (commit / "replay_bundle_p0").mkdir(parents=True)
    (commit / "replay_bundle_p0" / "context.json").write_text(json.dumps({"pair": 0}))
    for name in ("runs.jsonl", "verdict.json", "binding_map_p0.json"):
        (commit / name).write_text(name)
    link = BR.scratch(commit / "replay_bundle_p0", out)
    assert link == out / "replay_bundle_p0" and link.is_symlink() and link.resolve() == (commit / "replay_bundle_p0").resolve()
    assert sorted(p.name for p in out.iterdir()) == ["binding_map_p0.json", "replay_bundle_p0", "runs.jsonl", "verdict.json"]
    (out / "runs.jsonl").write_text("rewritten")
    assert (commit / "runs.jsonl").read_text() == "runs.jsonl"
    assert BR.scratch(commit / "replay_bundle_p0", out) == link


def test_the_picks_are_tabulated_by_family() -> None:
    art = {"sampled_replay": {"result": None, "grade": "partial", "why": "1 strata not evaluated",
                              "sample": {"seed": 7, "seed_source": "run_root", "k": 4, "picked": 4, "evaluated": 3, "equal": 2,
                                         "mismatch_n": 1, "not_evaluated_n": 1},
                              "timing": {"wall_s": 1.0, "workers": 2, "per_family_n": {"RoPE_v1": 3, "Gemm_v2": 1}},
                              "strata": {"RoPE_v1{D=64}|r0|0|mid|m|x": {"evaluated": 2, "mismatch": 1},
                                         "RoPE_v1{D=64}|r0|0|last|m|x": {"evaluated": 1},
                                         "Gemm_v2{K=4}|r0|0|mid|m|x": {"evaluated": 0}}}}
    o = BR.outcome(art)
    assert o["by_family"] == {"RoPE_v1": {"picked": 3, "evaluated": 3, "mismatch": 1}, "Gemm_v2": {"picked": 1, "evaluated": 0, "mismatch": 0}}
    assert (o["seed"], o["picked"], o["equal"], o["grade"]) == (7, 4, 2, "partial")


def test_the_uniform_draws_picks_are_tabulated_by_family_not_by_its_one_stratum() -> None:
    sr = {"result": False, "grade": "mismatch", "why": "x",
          "sample": {"strata_by": "uniform", "k": 5, "picked": 5, "evaluated": 4, "equal": 3, "mismatch_n": 1, "not_evaluated_n": 1},
          "timing": {"per_family_n": {"RoPE_v1": 3, "Gemm_v2": 2}},
          "strata": {"all|all|all|all|all|all": {"population": 99, "sampled": 5, "evaluated": 4, "mismatch": 1}},
          "mismatch_by": {"by_family": {"Gemm_v2": 1}}, "not_evaluated": [{"spec": "RoPE_v1{D=64}"}]}
    assert BR.outcome({"sampled_replay": sr})["by_family"] == {"RoPE_v1": {"picked": 3, "evaluated": 2, "mismatch": 0},
                                                              "Gemm_v2": {"picked": 2, "evaluated": 2, "mismatch": 1}}
    sr["sample"]["not_evaluated_n"] = 13
    assert BR.outcome({"sampled_replay": sr})["by_family"]["Gemm_v2"] == {"picked": 2, "evaluated": None, "mismatch": 1}


def test_a_replay_that_did_not_run_is_an_outcome_with_its_reason() -> None:
    why = "sampled replay not run: KeyError('tree node 1 of step 0 level 0 is not in the slim dumped store')"
    o = BR.outcome({"sampled_replay": {"result": None, "grade": "not-evaluated", "why": why, "error": "KeyError(...)"}})
    assert (o["result"], o["grade"], o["why"], o["picked"], o["evaluated"], o["by_family"], o["wall_s"]) == (None, "not-evaluated", why, None, None, {}, None)
