import pytest

from te_core.annotate import Annotation, StepAnnotation, annotate, with_structures
from te_core.score import (K_BITS, block_c, checkpoints, effective_slowdown, honest, rho_of, score, score_grid,
                           step_betas, w_star)

#: a toy model: 1e8 updated MACs per token in a chain of 10 matmuls, 1000 tokens a step, |C| 3x the forward's updated
#: MACs (forward plus a backward of twice its size)
PER_TOKEN, NODES, TOKENS = 100_000_000, 10, 1000


def full_step(t: int, beta: float | None = 1.6) -> StepAnnotation:
    per_node = PER_TOKEN // NODES
    profile = tuple((i * per_node, per_node * TOKENS) for i in range(NODES))
    return StepAnnotation(step=t, tokens=TOKENS, macs=3 * PER_TOKEN * TOKENS, updated_macs=PER_TOKEN * TOKENS,
                          updated_per_token=PER_TOKEN, depth=(NODES - 1) * per_node, profile=profile, bits=16,
                          weights={}, shapes={}, beta=beta)


def run(n: int, beta: float | None = 1.6) -> Annotation:
    steps = tuple(full_step(t, beta) for t in range(n))
    return Annotation(macs=sum(s.macs for s in steps), steps=steps)


C = block_c(PER_TOKEN)  # 4096 forward passes of the start model
K = K_BITS["10kB"]


def test_rho_formula_and_depth_condition():
    assert rho_of(0.5, 1e9, 1.6, 8e4, 4e9) == pytest.approx(max(1, 0.5 * 4e9 * 1.6 / (8e4 * (8e4 + 1.6))))
    assert rho_of(1.0, 10, 1.6, 8e4, 1e18) == 1.0  # D < k/β: the theorem says nothing
    assert rho_of(0.0, 1e9, 1.6, 8e4, 1e18) == 1.0
    assert rho_of(1.0, 1e9, 0.0, 8e4, 1e18) == 1.0


def test_a_traced_run_scores_at_its_cut():
    ann = run(100)
    s = score(ann, {}, K, C)
    assert s["beta"] == s["tau"] == 1.6 and s["steps"] == 100
    d_min = K / 1.6  # 50,000 MACs: every node but the first (none upstream) is deep
    assert s["phi"] == pytest.approx((NODES - 1) / NODES / 3)
    assert s["d_min"] == pytest.approx(d_min)
    assert s["rho"] == pytest.approx(s["phi"] * C * 1.6 / (K * (K + 1.6)))
    assert s["rho"] > 10


def test_a_step_with_no_witness_is_in_no_u():
    """No assumed β: a step the capture didn't trace is in U only at a family's β_t."""
    ann = run(10, beta=None)
    s = score(ann, {}, K, C)
    assert s["rho"] == 1.0 and s["steps"] == 0 and s["option"] == "no U" and s["d_min"] is None
    assert step_betas(ann, {3: 2.0}) == {3: 2.0}
    s = score(ann, {3: 1.6}, K, C)
    assert s["steps"] == 1 and s["rho"] == pytest.approx(score(run(10), {}, K, C)["rho"] / 10)


def test_beta_is_the_least_witness():
    ann = run(20)
    assert step_betas(ann, {0: 0.5, 1: 32.0}) == {0: 0.5, 1: 1.6} | {t: 1.6 for t in range(2, 20)}
    assert score(ann, {t: 32.0 for t in range(20)}, K, C) == score(ann, {}, K, C)  # a family above the cut adds nothing


def test_one_cheap_early_step_does_not_collapse_a_long_run():
    """A family covering only step 0 at a tiny β_t: τ = β_0 gives ρ 1, τ = the cut leaves step 0 out and keeps the other
    99 at the cut, so ρ stays within 1% of the run without it."""
    ann = run(100)
    clean = score(ann, {}, K, C)["rho"]
    s = score(ann, {0: 1e-6}, K, C)
    assert s["tau"] == 1.6 and s["steps"] == 99
    assert s["rho"] == pytest.approx(clean * 0.99)


def test_a_family_covering_every_step_below_the_cut_lowers_rho():
    ann = run(20, beta=16.0)
    clean = score(ann, {}, K, C)
    covered = score(ann, {t: 0.5 for t in range(20)}, K, C)
    assert covered["beta"] == covered["tau"] == 0.5 and covered["steps"] == 20
    assert 1.0 < covered["rho"] < clean["rho"]


def varied(n: int) -> Annotation:
    """Steps of three profiles, depths and cuts, a step that updates nothing and one the capture didn't trace."""
    steps = []
    for t in range(n):
        nodes, per_node = NODES - t % 3, PER_TOKEN // NODES
        profile = tuple((i * per_node, per_node * TOKENS) for i in range(nodes)) if t != 5 else ()
        steps.append(StepAnnotation(step=t, tokens=TOKENS, macs=3 * PER_TOKEN * TOKENS,
                                    updated_macs=nodes * per_node * TOKENS if t != 5 else 0, updated_per_token=PER_TOKEN,
                                    depth=(nodes - 1) * per_node, profile=profile, bits=16, weights={}, shapes={},
                                    beta=None if t == 8 else (16, 1.6, 0.16)[t % 3]))
    return Annotation(macs=sum(s.macs for s in steps), steps=tuple(steps))


@pytest.mark.parametrize("family", ["none", "geometric", "below_the_cut"])
@pytest.mark.parametrize("k", list(K_BITS.values()))
def test_the_grouped_thresholds_are_the_annotations_own(k, family):
    """`score` against the rule computed step by step with Annotation's own φ and D: no family, a family graded from
    0.05 bits, and one covering most steps a little below their cut, where a τ between the least β_t and the largest
    cut wins."""
    ann = varied(30)
    cut = {s.step: s.beta for s in ann.steps}
    if family == "none":
        covered = {}
    elif family == "geometric":
        covered = {t: 0.05 * 1.4 ** t for t in range(0, 30, 2)}
    else:
        covered = {t: (cut[t] or 16) * (0.3 + 0.02 * t) for t in range(30) if t % 7}
    betas = {}
    for s in ann.steps:
        ws = [b for b in (cut[s.step], covered.get(s.step)) if b is not None]
        if s.updated_macs and ws:
            betas[s.step] = min(ws)
    want = (1.0, None)
    for tau in sorted(set(betas.values()), reverse=True):
        keep = [t for t, b in betas.items() if b >= tau]
        o = rho_of(ann.phi(k / tau, keep), ann.depth(keep), tau, k, C * 50)
        if want[1] is None or o > want[0]:
            want = (o, tau)
    s = score(ann, covered, k, C * 50)
    assert s["rho"] == pytest.approx(want[0], rel=1e-12)
    assert s["tau"] == want[1]


def test_covered_steps_outside_the_prefix_are_ignored():
    ann = run(10)
    assert score(ann, {50: 1e-6}, K, C) == score(ann, {}, K, C)


def test_grid_covers_every_k():
    grid = score_grid(run(5), {}, C)
    assert set(grid) == set(K_BITS) == {"10kB", "100kB", "1MB"}
    assert grid["1MB"]["rho"] <= grid["10kB"]["rho"]


def record(n_steps: int) -> dict:
    """A harness record of a toy run: one traced structure, every step the same."""
    struct = {"updated": [[100, i * 100, 100 * 8, 8, 16] for i in range(4)], "log": [], "beta": 2.0, "cut_bits": 800}
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
    subs = checkpoints(rec, {}, c=1e9)
    assert [s["macs"] for s in subs] == [7 + 5 * 9600, 7 + 10 * 9600 + 5]
    assert subs[0]["annotation"]["steps"] == 5
    assert subs[0]["annotation"]["beta"] == [2.0, 2.0] and subs[0]["annotation"]["untraced"] == 0
    assert a4.depth() == 300
    assert a4.phi(150) == pytest.approx(2 * 800 * 5 / a4.macs)


def test_with_structures_scales_the_traced_rows_and_reprices_the_families():
    """The runtime counter's rows (4 nodes of 100 MACs per token, no cut) give way to a traced structure of 2 nodes of 50
    per token, scaled to each step's tokens; step 0 updates nothing and keeps its own; a family's β_t is bits(E_t) over
    the traced updated MACs per token block (4x its old value here), and |C| is the runtime count."""
    rec = record(3)
    rec["structures"] = {"s": {"updated": [[100, i * 100, 800, 8, 16] for i in range(4)], "log": [["x"]]},
                         "z": {"updated": [], "log": []}}
    rec["steps"][0]["struct"] = "z"
    rec["steps"][2]["tokens"] = 4
    rec["families"] = [{"covers": [1, 2], "beta": {"1": 1.0, "2": 2.0}}]
    traced = {"updated": [[50, 0, 50 * 16, 16, 32], [50, 50, 50 * 16, 16, 16]], "beta": 3.0, "cut_bits": 300}
    out = with_structures(rec, {"s": traced, "z": traced})
    ann = annotate(out)
    assert ann.macs == annotate(rec).macs
    assert [s.updated_macs for s in ann.steps] == [0, 100 * 8, 100 * 4]
    assert [s.beta for s in ann.steps] == [None, 3.0, 3.0] and ann.depth() == 50 and ann.bits() == 16
    assert out["families"][0]["beta"] == {"1": 4.0, "2": 8.0}
    assert out["structures"][out["steps"][1]["struct"]]["log"] == [["x"]]
    assert rec["structures"].keys() == {"s", "z"} and rec["families"][0]["beta"] == {"1": 1.0, "2": 2.0}


def test_effective_slowdown():
    key = "10kB"
    subs = [{"macs": 100, "rho": {key: 3.0}, "val_bpb": 0.9}, {"macs": 50, "rho": {key: 10.0}, "val_bpb": 0.89},
            {"macs": 10, "rho": {key: 1.0}, "val_bpb": 1.2}]
    refs = [{"macs": 80, "val_bpb": 0.9}, {"macs": 120, "val_bpb": 0.85}]
    assert w_star(subs, 0.9, key) == 300
    assert honest(refs, 0.9) == 80
    assert effective_slowdown(subs, refs, 0.9, key) == pytest.approx(300 / 80)
    assert w_star(subs, 0.5, key) is None
