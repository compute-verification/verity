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


def test_excess_on_controls_alike_allows_the_least_repeat():
    res = judge(*[([DECLARED + 14, DECLARED + 14 + 2000, DECLARED + 14], [14, 14, 14 + 2500, 14])] * 6)
    assert res["external"]["external"] and res["rule"] == "min" and res["status"] == "passed"
    assert set(res["by_kernel_strict"].values()) == {"REFUSED"}


def test_min_rule_still_refuses_a_kernel_whose_every_repeat_exceeds():
    res = judge(*[([DECLARED + 14, DECLARED + 14 + 2000, DECLARED + 14], [14, 14, 14 + 2500, 14])] * 6,
                ([DECLARED + 500] * 3, [14, 14, 14 + 2500, 14]))
    assert res["rule"] == "min" and res["by_kernel"]["fill_k6"] == "REFUSED" and res["status"] == "REFUSED"


def test_smaller_control_excess_keeps_the_strict_rule():
    res = judge(*[([DECLARED + 14, DECLARED + 14 + 2000, DECLARED + 14], [14, 14, 14 + 300, 14])] * 6)
    assert res["external"]["controls"][1] == 6 and not res["external"]["external"] and res["rule"] == "strict"


def test_dry_launches_are_reported_beside_and_do_not_set_the_rule():
    d, rows = probe([DECLARED + 14, DECLARED + 14 + 2000, DECLARED + 14] * 2, [14] * 7)
    for k in range(6):
        d["launches"].append({"kernel": "fill_persist128_1_3", "store": "rcpt@128", "control": False, "dry": True,
                              "declared_write_sectors": 0, "declared_atom_sectors": 0})
        rows.append({"kernel": "fill_persist128_1_3", G.WRITE: 14 + (2500 if k % 3 == 1 else 0), G.ATOM: 0.0, G.RED: 0.0})
    res = G.judge(d, rows)
    assert res["external_dry"]["external"] and res["external_dry"]["controls"] == [6, 2]
    assert res["external"]["judged"] == [6, 2] and res["rule"] == "strict" and res["status"] == "REFUSED"
