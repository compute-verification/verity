"""`guard.py`'s reading of a records run's log and its verdict, on fixtures: no node, no Lean."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("lean_records_guard", Path(__file__).resolve().parents[1] / "guard.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

ROOT = "/cache/lean-records_build-0"


def _audit(pkg: str, keyed: int, undone: int, spot: float, t: float) -> str:
    return (f"{t:8.1f} sampled: {ROOT}/{pkg}: recompiling 101 of the {undone} modules this build didn't compile\n"
            f"{t + 9:8.1f} AUDIT {ROOT}/{pkg}: PASS  195396 declarations in 3877 modules; 0 guarantees\n"
            f"{t + 9:8.1f}   staged: {undone} of its {keyed} keyed modules from the artifact cache\n"
            f"{t + 9:8.1f}   sampled: 101 of the {undone} modules its build didn't compile, recompiled in {spot} s (8 equal "
            "as object graphs): A.B, C.D\n")


def test_a_log_gives_what_the_first_build_compiled_and_every_spot_check_of_the_package():
    """The first `security/Proofs` audit's keyed modules less those its build didn't compile are what it compiled; every
    one of its audits' spot-checks counts, and another package's don't."""
    log = (_audit("security/Proofs", 5056, 5021, 94.3, 100) + _audit("security", 712, 712, 15.8, 200)
           + _audit("security/Proofs", 5056, 5056, 172.1, 300) + _audit("security/Proofs", 5056, 5056, 99.8, 400))
    assert G.read_log(log) == {"clean": True, "compiled": 35, "spot_s": 366.2, "audits": 3}


def test_an_arm_whose_build_restored_every_module_is_not_clean():
    """A lemma an earlier run on the node built is restored from its artifact cache: its build compiled nothing."""
    assert G.read_log(_audit("security/Proofs", 5056, 5056, 229.2, 100)) == \
        {"clean": False, "compiled": 0, "spot_s": 229.2, "audits": 1}
    assert G.read_log("") == {"clean": False, "compiled": None, "spot_s": 0.0, "audits": 0}


def _arm(kind: str, build: float, cpu: float, peak: float = 11.0, clean: bool = True, gb: str = "93") -> dict:
    return {"arm": kind, "clean": clean, "rc": 0, "seconds": 2000.0, "rest_s": 1000.0, "build_s": build,
            "tool_cpu_s": cpu, "tree_cpu_s": 9000.0, "peak_gib": peak, "max_lean": 23, "largest_gib": 9.0,
            "reserved": {"CHECK_RUN_MEM_GB": gb}}


def test_the_verdict_judges_clean_arms_on_the_build_step_and_the_tools_own_cpu_and_never_whole_seconds():
    """B within 1.1 of A's median on the build step and on the tools' own CPU, over clean arms; a peak past its
    reservation fails; an arm that isn't clean counts for nothing; whole seconds are in the summary and not the verdict."""
    arms = [_arm("A", 110, 60), _arm("A", 1, 1, clean=False), _arm("B", 115, 62), _arm("B", 112, 61),
            _arm("C", 130, 61, gb="86")]
    summary, verdict = G.summarize(arms)
    assert summary["A"]["n"] == 1 and summary["A"]["median_build_s"] == 110 and summary["B"]["median_seconds"] == 2000
    assert all(verdict.values()) and not any("seconds" in k for k in verdict)
    _, slow = G.summarize([*arms, _arm("B", 140, 62), _arm("B", 140, 62)])
    assert not slow["B_build_s_within_1.1_A"] and slow["B_tool_cpu_s_within_1.1_A"]
    _, over = G.summarize([*arms, _arm("C", 130, 61, peak=81.0, gb="86")])
    assert not over["C_peak_within_reservation"], "86 GB is 80.1 GiB"


def test_the_guard_swaps_every_file_of_the_plan_and_reads_reservations_by_its_units():
    """An arm runs one version of the plan whole, its rules and its figures, so `TOOLS` holds both; a reservation is read
    in the unit the plan gives its variable."""
    assert {"check.py", "reserve.py", G.RESERVE.DATA.name} <= set(G.TOOLS)
    assert G.gib_of({"CHECK_SLOT_MEM_GB": "80"}) == 80
    assert G.gib_of({"CHECK_RUN_MEM_GB": "86", "CHECK_SLOT_MEM_GB": "90"}) == G.RESERVE.gib(86)
