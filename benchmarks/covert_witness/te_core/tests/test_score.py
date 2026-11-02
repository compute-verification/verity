import pytest

from te_core.annotate import Annotation, StepAnnotation, annotate
from te_core.score import (K_BITS, block_c, checkpoints, effective_slowdown, honest, rho_of, score, score_grid,
                           w_star)

#: a toy model: 1e8 updated MACs per token in a chain of 10 matmuls, 1000 tokens a step, |C| 3x the forward's updated
#: MACs (forward plus a backward of twice its size)
PER_TOKEN, NODES, TOKENS = 100_000_000, 10, 1000


def full_step(t: int) -> StepAnnotation:
    per_node = PER_TOKEN // NODES
    profile = tuple((i * per_node, per_node * TOKENS) for i in range(NODES))
    return StepAnnotation(step=t, tokens=TOKENS, macs=3 * PER_TOKEN * TOKENS, updated_macs=PER_TOKEN * TOKENS,
                          updated_per_token=PER_TOKEN, depth=(NODES - 1) * per_node, profile=profile, bits=16,
                          weights={}, shapes={})


def run(n: int) -> Annotation:
    steps = tuple(full_step(t) for t in range(n))
    return Annotation(macs=sum(s.macs for s in steps), steps=steps)


C = block_c(PER_TOKEN)  # 4096 forward passes of the start model
K = K_BITS["10kB"]


def test_rho_formula_and_depth_condition():
    assert rho_of(0.5, 1e9, 1.6, 8e4, 4e9) == pytest.approx(max(1, 0.5 * 4e9 * 1.6 / (8e4 * (8e4 + 1.6))))
    assert rho_of(1.0, 10, 1.6, 8e4, 1e18) == 1.0  # D < k/β: the theorem says nothing
    assert rho_of(0.0, 1e9, 1.6, 8e4, 1e18) == 1.0
    assert rho_of(1.0, 1e9, 0.0, 8e4, 1e18) == 1.0


def test_uncovered_run_scores_option_a_at_p_over_r():
    ann = run(100)
    s = score(ann, {}, K, 10, 16, C)
    assert s["beta"] == 1.6
    assert s["a"]["rho"] == s["b"]["rho"]
    d_min = K / 1.6  # 50,000 MACs: every node but the first (none upstream) is deep
    assert s["phi"] == pytest.approx((NODES - 1) / NODES / 3)
    assert s["d_min"] == pytest.approx(d_min)
    assert s["rho"] == pytest.approx(s["phi"] * C * 1.6 / (K * (K + 1.6)))
    assert s["rho"] > 10


def test_one_cheap_early_step_does_not_collapse_a_long_run():
    """A family covering only step 0 at a tiny β_t pulls option (a)'s β down to it; option (b) leaves step 0 out and
    keeps β = p/r for the other 99, so ρ stays within 1% of the uncovered run's."""
    ann = run(100)
    clean = score(ann, {}, K, 10, 16, C)["rho"]
    s = score(ann, {0: 1e-6}, K, 10, 16, C)
    assert s["a"]["rho"] == 1.0
    assert s["option"] == "b"
    assert s["rho"] == pytest.approx(clean * 0.99)
    assert s["b"]["steps"] == 99


def test_a_family_covering_every_step_lowers_rho():
    ann = run(20)
    clean = score(ann, {}, K, 1, 16, C)
    covered = score(ann, {t: 0.5 for t in range(20)}, K, 1, 16, C)
    assert covered["option"] == "a"
    assert covered["b"]["rho"] == 1.0  # no uncovered step
    assert covered["beta"] == 0.5
    assert 1.0 < covered["rho"] < clean["rho"]


def test_covered_steps_outside_the_prefix_are_ignored():
    ann = run(10)
    assert score(ann, {50: 1e-6}, K, 10, 16, C) == score(ann, {}, K, 10, 16, C)


def test_grid_covers_every_cell():
    grid = score_grid(run(5), {}, 16, C)
    assert set(grid) == {f"{k}/r{r}" for k in K_BITS for r in (1, 10, 100)}
    assert grid["1MB/r100"]["rho"] <= grid["10kB/r100"]["rho"]


def record(n_steps: int) -> dict:
    """A harness record of a toy run: one structure, every step the same."""
    struct = {"updated": [[100, i * 100, 100 * 8, 8, 16] for i in range(4)], "log": []}
    steps = [{"step": t, "tokens": 8, "macs": 3 * 400 * 8, "by": {}, "struct": "s", "weights": {}, "shapes": {}}
             for t in range(n_steps)]
    return {"setup": {"macs": 7, "tokens": 0, "by": {}}, "tail": {"macs": 5, "tokens": 0, "by": {}}, "steps": steps,
            "structures": {"s": struct},
            "checkpoints": [{"step": s, "val_bpb": 1.0 - 0.01 * s} for s in (4, 9)]}


def test_checkpoints_are_prefixes():
    rec = record(10)
    a4 = annotate(rec, through=4)
    assert a4.macs == 7 + 5 * 9600
    assert annotate(rec).macs == 7 + 10 * 9600 + 5
    subs = checkpoints(rec, {}, c=1e9, p=16)
    assert [s["macs"] for s in subs] == [7 + 5 * 9600, 7 + 10 * 9600 + 5]
    assert subs[0]["annotation"]["steps"] == 5
    assert a4.depth() == 300
    assert a4.phi(150) == pytest.approx(2 * 800 * 5 / a4.macs)


def test_effective_slowdown():
    key = "10kB/r10"
    subs = [{"macs": 100, "rho": {key: 3.0}, "val_bpb": 0.9}, {"macs": 50, "rho": {key: 10.0}, "val_bpb": 0.89},
            {"macs": 10, "rho": {key: 1.0}, "val_bpb": 1.2}]
    refs = [{"macs": 80, "val_bpb": 0.9}, {"macs": 120, "val_bpb": 0.85}]
    assert w_star(subs, 0.9, key) == 300
    assert honest(refs, 0.9) == 80
    assert effective_slowdown(subs, refs, 0.9, key) == pytest.approx(300 / 80)
    assert w_star(subs, 0.5, key) is None
