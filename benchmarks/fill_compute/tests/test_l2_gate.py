"""l2_gate.py's rule 4: the least of a kernel's repeats is judged only when the run's interleaved controls carry the excess as
judged launches do; otherwise every repeat must pass."""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import l2_gate as G  # noqa: E402

DECLARED = 1000


def probe(judged: list[float], controls: list[float], kernel: str = "fill_persist128_1_3") -> tuple[dict, list[dict]]:
    """One kernel's launches, control first and then judged and control in turn, with these write sectors."""
    assert len(controls) == len(judged) + 1
    launches, rows = [], []
    order = [(True, controls[0])] + [x for j, c in zip(judged, controls[1:]) for x in ((False, j), (True, c))]
    for control, w in order:
        launches.append({"kernel": kernel, "store": "rcpt@128", "control": control,
                         "declared_write_sectors": 0 if control else DECLARED, "declared_atom_sectors": 0})
        rows.append({"kernel": kernel, G.WRITE: w, G.ATOM: 0.0, G.RED: 0.0})
    return {"launches": launches}, rows


def judge(*kernels: tuple[list[float], list[float]]) -> dict:
    launches, rows = [], []
    for i, (j, c) in enumerate(kernels):
        d, r = probe(j, c, f"fill_k{i}")
        launches += d["launches"]
        rows += r
    return G.judge({"launches": launches}, rows)


def test_fisher_matches_the_hypergeometric_tail():
    # 3 of 4 judged hit, 0 of 4 controls: P(X >= 3) with X ~ Hypergeometric(8, 3 hits, 4 drawn) = 4 / 56
    assert math.isclose(G.fisher_greater(3, 1, 0, 4), 4 / 56)
    assert G.fisher_greater(0, 4, 2, 2) == 1.0


def test_clean_run_passes_strictly():
    res = judge(([DECLARED + 14] * 3, [14] * 4))
    assert res["status"] == "passed" and res["rule"] == "strict" and not res["external"]["external"]


def test_excess_on_judged_launches_alone_keeps_the_strict_rule():
    res = judge(*[([DECLARED + 14, DECLARED + 14 + 2000, DECLARED + 14], [14] * 4)] * 6)
    assert res["rule"] == "strict" and res["status"] == "REFUSED"
    assert set(res["by_kernel_min"].values()) == {"passed"} and set(res["by_kernel"].values()) == {"REFUSED"}


def with_dry(judged: list[float], controls: list[float], dry: list[float], kernel: str = "fill_k0") -> tuple[dict, list[dict]]:
    d, rows = probe(judged, controls, kernel)
    for w in dry:
        d["launches"].append({"kernel": kernel, "store": "fold", "control": False, "dry": True,
                              "declared_write_sectors": 0, "declared_atom_sectors": 0})
        rows.append({"kernel": kernel, G.WRITE: w, G.ATOM: 0.0, G.RED: 0.0})
    return d, rows


EXCESS = [DECLARED + 14, DECLARED + 14 + 2000, DECLARED + 14] * 2


def test_zero_tile_controls_alone_never_set_the_min_rule():
    res = judge(*[([DECLARED + 14, DECLARED + 14 + 2000, DECLARED + 14], [14, 14, 14 + 2500, 14])] * 6)
    assert res["external"]["external"] and res["external_dry"] is None and res["rule"] == "strict"
    assert res["status"] == "REFUSED"


def test_dry_controls_alike_allow_the_least_repeat():
    res = G.judge(*with_dry(EXCESS, [14] * 7, [14, 14 + 2500, 14, 14, 14 + 2500, 14]))
    assert res["external_dry"]["external"] and not res["external"]["external"]
    assert res["rule"] == "min" and res["status"] == "passed" and res["by_kernel_strict"]["fill_k0"] == "REFUSED"


def test_min_rule_still_refuses_a_kernel_whose_every_repeat_exceeds():
    d, rows = with_dry(EXCESS, [14] * 7, [14, 14 + 2500, 14, 14, 14 + 2500, 14])
    d2, rows2 = probe([DECLARED + 500] * 3, [14] * 4, "fill_k1")
    res = G.judge({"launches": d["launches"] + d2["launches"]}, rows + rows2)
    assert res["rule"] == "min" and res["by_kernel"]["fill_k1"] == "REFUSED" and res["status"] == "REFUSED"


def test_dry_excess_as_often_but_smaller_keeps_the_strict_rule():
    res = G.judge(*with_dry([DECLARED + 14, DECLARED + 14 + 2000] * 20, [14] * 41, [14, 14 + 100] * 20))
    ext = res["external_dry"]
    assert ext["fisher_p_judged_hit_more"] >= G.ALPHA and ext["mann_whitney_p_judged_larger"] < G.ALPHA
    assert not ext["external"] and res["rule"] == "strict"


def test_a_larger_judged_maximum_alone_does_not_refuse_the_min_rule():
    res = G.judge(*with_dry(EXCESS * 4, [14] * 25, [14, 14 + 300, 14, 14, 14 + 300, 14]))
    assert res["external_dry"]["max_control_excess"] < res["external_dry"]["max_judged_excess"]
    assert res["external_dry"]["external"] and res["rule"] == "min"


def test_dry_controls_hit_less_often_keep_the_strict_rule():
    res = G.judge(*with_dry(EXCESS * 3, [14] * 19, [14] * 17 + [14 + 2500]))
    assert res["external_dry"]["fisher_p_judged_hit_more"] < G.ALPHA and res["rule"] == "strict"


def test_a_variant_is_judged_against_its_own_controls():
    d, rows = probe([DECLARED + 15] * 3, [13] * 4, "fill_chunk_1_3")
    d2, rows2 = probe([DECLARED + 13] * 3, [13] * 4, "fill_chunk_1_3")
    for x in d2["launches"]:
        x["variant"] = "nostop"
    res = G.judge({"launches": d["launches"] + d2["launches"]}, rows + rows2)
    assert res["by_kernel"] == {"fill_chunk_1_3": "REFUSED", "fill_chunk_1_3/nostop": "passed"}


def test_the_floor_is_the_least_dry_launch_and_the_zero_tile_range_is_reported():
    d, rows = with_dry([DECLARED + 15] * 3, [13, 13, 13, 13], [15, 15 + 900, 15], "fill_chunk_1_3")
    res = G.judge(d, rows)
    det = res["detail"]["fill_chunk_1_3"]
    assert res["floor"] == {"source": "dry", "sectors": 15} and det["controls"] == [13, 13] and det["floor"] == 15
    assert det["strict"] == "passed"


def test_without_dry_launches_the_floor_is_the_least_zero_tile_control():
    d, rows = probe([DECLARED + 15] * 3, [13, 15, 14, 13], "fill_chunk_1_3")
    res = G.judge(d, rows)
    assert res["floor"]["source"] == "zero-tile" and res["detail"]["fill_chunk_1_3"]["floor"] == 13
    assert res["status"] == "REFUSED"


def test_mann_whitney_matches_known_cases():
    assert G.mann_whitney_greater([5, 6, 7, 8], [1, 2, 3, 4]) < 0.05
    assert G.mann_whitney_greater([1, 2, 3, 4], [5, 6, 7, 8]) > 0.95
    assert 0.4 < G.mann_whitney_greater([1, 2, 3, 4] * 5, [1, 2, 3, 4]) < 0.6
