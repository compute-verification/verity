"""The census reconciliation's report: `census.reconciliation_markdown` over what `census.reconcile` returns, and the laptop
half of `census.main`, which writes that report beside its JSON and exits 1 on a refusal.

The log, the profile and the device census are `test_census_counts`'s: its toy log of two steps, each declared kernel launched
once per declaring event.  A test changes the device census, or the report where a two-step log cannot show the case, and
reads the lines the change puts in the markdown."""
from __future__ import annotations

import copy
import dataclasses
import json
import re
from pathlib import Path

import pytest

from tests.observe.test_fold_synthetic import Pat
from tests.properties import test_census_counts as C
from verity_vllm.config import UsageError
from verity_vllm.observe.fold import resolve_log as RL
from verity_vllm.observe.fold.resolver import ResolutionError
from verity_vllm.pipeline import cli
from verity_vllm.properties import census as CEN
from verity_vllm.properties import kernel_allowlist as KA

reconcile = C.reconcile
NORM, ALT, INDEX, MM = C.NORM, C.ALT, C.INDEX, "matmul_kernel_persistent"
#: a launch no declared or allowlisted name claims, its Kineto name past every cut the report makes (60, 80, 110)
OTHER = "void other::launch<int>(long)"
STRAY = "void stray::launch<" + ", ".join(["c10::Half"] * 12) + ">(long)"
#: a launch the GEMM's name and the fused norm's both claim, its Kineto name past the ambiguity list's cut (100)
BOTH = "matmul_kernel_persistent<" + "T, " * 30 + "vllm::fused_add_rms_norm_kernel<float>>"

REFUSED, KERNELS, STEPS = "Launch counts refused", "Kernels (all steps)", "Per step"
EXEMPT = "Launch counts exempt (`kernel_allowlist.LAUNCH_COUNT_ALLOWLIST`)"
COUNTS = "Count comparison (declared compute / transparent kernels; step 0 = prefill, step 1 = first decode)"
AMBIGUOUS, MATCH = "Ambiguous compute claims", "Matching table (`census.KERNEL_MATCH`)"
COUNTS_NOTE = ("`census` = kernels of that name in the profiler; `events` = raw root events of the step whose resolution "
               "declares the name; `parts` = resolution parts declaring it (RoPE declares its kernel twice — q and k —, the "
               "KV update twice — K and V —, the attention scope once per distinct T, the two-stage sampler three times for "
               "its Triton kernel).")
KERNELS_HEAD = ["| kernel (Kineto name, truncated) | cat | step 0 | decode steps | total | device µs | claimed by (declared name → pattern) |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
STEPS_HEAD = ["| step | phase | rows | census kernels | distinct | device µs | log events | residue | declared never seen |",
              "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
COUNTS_HEAD = ["| declared | step 0 census/events/parts | step 1 census/events/parts | decode steps 2.. census (set) | patterns |",
               "| --- | --- | --- | --- | --- |"]
COUNTS_ROWS = [f"| `{NORM}` | 2/2/2 | 2/2/2 | [] | `fused_norm` |", f"| `{ALT}` | 0/2/2 | 0/2/2 | [] | `fused_norm` |",
               f"| `{INDEX}` | 1/1/1 | 1/1/1 | [] | `index` |", f"| `{MM}` | 5/5/5 | 5/5/5 | [] | `mm` |"]
STEP_ROWS = [f"| 0 | prefill | 2 | 8 | 3 | 8.0 | 22 | 0 | `{ALT}` |", f"| 1 | decode | 1 | 8 | 3 | 8.0 | 22 | 0 | `{ALT}` |"]


def census() -> dict:
    """`test_census_counts`'s device census as the GPU half writes it: with the run's header, each kernel's category, each
    step's device time, and the engine's trailing step, which launches nothing and has no number."""
    doc = C.device_census()
    for s in doc["steps"]:
        s["device_us"] = 8.0
        for d in s["kernels"].values():
            d["cat"] = "kernel"
    doc["steps"].append({"step": None, "n_kernels": 0, "kernels": {}})
    return {**doc, "utc": "2026-01-02T03:04:05Z", "method": "torch.profiler around each engine step", "engine_steps": 3,
            "tokens_equal_accepted": True,
            "versions": {"torch": "2.13.0", "triton": "3.5.0", "vllm": "0.9.1", "gpu": {"name": "NVIDIA RTX 4090"}}}


def with_launch(doc: dict, step: int, name: str, count: int) -> dict:
    """`doc` with `count` more launches at `step` under the Kineto name `name`, which the step did not hold."""
    s = doc["steps"][step]
    s["kernels"][name] = {"count": count, "us": 1.5, "cat": "kernel"}
    s["n_kernels"] += count
    return doc


def sections(md: str) -> dict[str, list[str]]:
    """The report's lines by `## ` heading, in the report's order; the lines above the first heading are under ""."""
    out: dict[str, list[str]] = {"": []}
    for line in md.split("\n"):
        if line.startswith("## "):
            out[line[3:]] = []
        else:
            out[next(reversed(out))].append(line)
    return out


def test_a_passing_reconciliation_is_this_record(reconcile):
    """Every field of the report but its kernels and steps past the first, which differ from the first in what the markdown
    test reads."""
    rep = reconcile(census(), "traced")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", rep.pop("utc")) and re.fullmatch("[0-9a-f]{64}", rep.pop("digest"))
    kernels, steps = rep.pop("kernels"), rep.pop("steps")
    assert rep == {
        "schema": "verity-capture/census-reconciliation/v2",
        "census": {"utc": "2026-01-02T03:04:05Z", "workload_sha256": None, "engine_steps": 3, "steps_with_device_work": 2,
                   "tokens_equal_accepted": True, "method": "torch.profiler around each engine step",
                   "versions": {"torch": "2.13.0", "triton": "3.5.0", "vllm": "0.9.1", "cuda": None, "gpu": {"name": "NVIDIA RTX 4090"}}},
        "log": "traced", "log_profile_id": "toy-profile", "profile": "toy", "log_steps": 2, "census_steps": 2, "steps_aligned": True,
        "totals": {"census_kernels": 16, "distinct_kernel_names": 3, "residue_kernels": 0, "residue_names": [], "ambiguous": 0,
                   "log_unsupported": 0, "count_mismatches": 0, "count_mismatch_kernels": [], "count_exempt": {}},
        "ambiguous": [], "count_mismatches": [], "count_allowlist": {},
        "match_table": {k: list(v) for k, v in CEN.KERNEL_MATCH.items()}, "compute_kernels": sorted(CEN.COMPUTE_KERNELS),
        "allowlist": sorted(KA.KERNEL_ALLOWLIST), "profile_extra_allowlist": [], "proposed_allowlist": [],
        "residue_zero": True, "counts_equal": True, "ok": True}
    assert [k["kernel"] for k in kernels] == [C.KINETO[NORM], C.KINETO[INDEX], MM] and [s["step"] for s in steps] == [0, 1]
    assert kernels[0] == {"kernel": C.KINETO[NORM], "cat": "kernel", "grid": None, "block": None, "count_step0": 2, "count_decode": [2],
                          "count_total": 4, "us_total": 2.0, "steps_present": [0, 1], "claims": [NORM], "patterns": ["fused_norm"],
                          "residue": False}
    assert steps[0] == {
        "step": 0, "phase": "prefill", "rows": 2, "log_events": 22, "log_roots": 15, "census_kernels": 8, "census_distinct": 3,
        "device_us": 8.0, "declared_by_patterns": [NORM, ALT, INDEX, MM], "residue": [], "declared_never_seen": [ALT],
        "allowlist_never_seen": sorted(set(KA.KERNEL_ALLOWLIST) - {INDEX}),
        "counts": {NORM: {"census_kernels": 2, "declaring_events": 2, "declaring_parts": 2, "patterns": ["fused_norm"]},
                   ALT: {"census_kernels": 0, "declaring_events": 2, "declaring_parts": 2, "patterns": ["fused_norm"]},
                   INDEX: {"census_kernels": 1, "declaring_events": 1, "declaring_parts": 1, "patterns": ["index"], "infrastructure": 1},
                   MM: {"census_kernels": 5, "declaring_events": 5, "declaring_parts": 5, "patterns": ["mm"]}},
        "count_mismatches": []}


def test_a_census_names_its_workload_and_its_kernels_grid_and_block(reconcile):
    doc = census()
    doc["workload"] = {"sha256": "ab" * 32}
    doc["versions"]["cuda"] = "12.8"
    doc["steps"][0]["kernels"][MM].update(grid=[4, 1, 1], block=[128, 1, 1])
    rep = reconcile(doc, "traced")
    assert (rep["census"]["workload_sha256"], rep["census"]["versions"]["cuda"]) == ("ab" * 32, "12.8")
    assert (rep["kernels"][2]["grid"], rep["kernels"][2]["block"]) == ([4, 1, 1], [128, 1, 1])


def test_a_passing_reconciliation_is_reported_whole(reconcile):
    rep = reconcile(census(), "traced")
    assert rep["ok"] and rep["census_steps"] == 2
    assert CEN.reconciliation_markdown(rep).split("\n") == [
        "# Kernel census × resolution: `traced` × profile `toy`",
        "",
        f"- reconciled {rep['utc']}; census taken 2026-01-02T03:04:05Z (torch 2.13.0, triton 3.5.0, vLLM 0.9.1, GPU NVIDIA RTX 4090)",
        "- method: torch.profiler around each engine step",
        "- census: 2 steps with device work (3 engine steps), 16 device kernels, 3 distinct names; tokens == accepted record: True",
        "- log: 2 steps (profile_id `toy-profile`), Unsupported 0; steps aligned: True",
        "- **residue: 0 kernels** (none); ambiguous compute claims: 0  →  **residue_zero = True**",
        "- **launch counts: 0 refused** (none); exempt by the launch-count allowlist: none  →  **counts_equal = True**",
        "",
        f"## {KERNELS}",
        "",
        *KERNELS_HEAD,
        f"| `{C.KINETO[NORM][:107]}...` | kernel | 2 | 2 | 4 | 2.0 | `{NORM}` → `fused_norm` |",
        f"| `{C.KINETO[INDEX]}` | kernel | 1 | 1 | 2 | 2.0 | `{INDEX}` → `index` |",
        f"| `{MM}` | kernel | 5 | 5 | 10 | 2.0 | `{MM}` → `mm` |",
        "",
        f"## {STEPS}",
        "",
        *STEPS_HEAD,
        *STEP_ROWS,
        "",
        f"## {COUNTS}",
        "",
        COUNTS_NOTE,
        "",
        *COUNTS_HEAD,
        *COUNTS_ROWS,
        "",
        f"## {MATCH}",
        "",
        "| declared name | regexes |",
        "| --- | --- |",
        *(f"| `{k}` | " + ", ".join(f"`{p}`" for p in v) + " |" for k, v in CEN.KERNEL_MATCH.items()),
        "",
        "Allowlist (profile `kernel_allowlist`): " + ", ".join(f"`{a}`" for a in sorted(KA.KERNEL_ALLOWLIST)),
        "",
    ]
    assert len(C.KINETO[NORM]) > 110 >= len(C.KINETO[INDEX])
    assert "| `gather_kernel` | `_scatter_gather_elementwise_kernel<`, `(?<![A-Za-z_])gather_kernel` |" in sections(
        CEN.reconciliation_markdown(rep))[MATCH]


def test_the_gpu_is_named_by_a_census_that_holds_its_name_alone(reconcile):
    doc = census()
    doc["versions"]["gpu"] = "RTX 4090"
    md = CEN.reconciliation_markdown(reconcile(doc, "traced"))
    assert md.split("\n")[2].endswith("(torch 2.13.0, triton 3.5.0, vLLM 0.9.1, GPU RTX 4090)")


def test_refused_launch_counts_are_a_section_of_their_own_lines(reconcile):
    """One line a refusal, above the kernels: `count_line`'s, which names four of the declaring events and counts the rest."""
    rep = reconcile(C.launched(C.launched(census(), 0, MM, 1), 1, NORM, -1), "traced")
    s = sections(CEN.reconciliation_markdown(rep))
    assert list(s) == ["", REFUSED, KERNELS, STEPS, COUNTS, MATCH]
    assert s[""][-2:] == [(f"- **launch counts: 2 refused** (`{NORM}`, `{MM}`); exempt by the launch-count allowlist: none"
                           "  →  **counts_equal = False**"), ""]
    gemm, norm = rep["count_mismatches"]
    assert (gemm["patterns"], norm["patterns"]) == (["mm"], ["fused_norm"])
    assert s[REFUSED] == ["", f"- {CEN.count_line(gemm)}", f"- {CEN.count_line(norm)}", ""]
    assert CEN.count_line(norm).startswith(f"step 1: {NORM} launched 1 times on the device against 2 declaring events (buffers [")
    assert CEN.count_line(norm).count("; event ") == 1 and "more" not in CEN.count_line(norm)
    assert CEN.count_line(gemm).startswith(f"step 0: {MM} launched 6 times on the device against 5 declaring events (buffers [")
    assert CEN.count_line(gemm).count("; event ") == 3 and CEN.count_line(gemm).endswith("]; +1 more)")
    assert s[COUNTS][-5:] == [f"| `{NORM}` | 2/2/2 | 1/2/2 | [] | `fused_norm` |", COUNTS_ROWS[1], COUNTS_ROWS[2],
                              f"| `{MM}` | 6/5/5 | 5/5/5 | [] | `mm` |", ""]


def test_a_refusals_line_names_four_events_and_counts_the_rest():
    def event(i, module):
        return {"event": i, "op": "aten.x", "module": module, "writes": [i]}
    events = [event(1, "m.a"), event(2, None), event(3, "m.c"), event(4, "m.d"), event(5, "m.e")]
    m = {"step": 3, "kernel": "k", "device_count": 7, "declaring_events": 5}
    assert CEN.count_line({**m, "buffers": [1, 2, 3, 4, 5], "events": events}) == (
        "step 3: k launched 7 times on the device against 5 declaring events (buffers [1, 2, 3, 4, 5]: "
        "event 1 aten.x m.a writes allocs [1]; event 2 aten.x - writes allocs [2]; "
        "event 3 aten.x m.c writes allocs [3]; event 4 aten.x m.d writes allocs [4]; +1 more)")
    assert CEN.count_line({**m, "buffers": [1, 2, 3, 4], "events": events[:4]}).endswith("; event 4 aten.x m.d writes allocs [4])")
    assert CEN.count_line(m) == "step 3: k launched 7 times on the device against 5 declaring events (buffers None)"


def test_an_exemption_is_reported_with_its_reason(reconcile, monkeypatch):
    why, why_mm = "toy: a second launch per event that writes only its own scratch", "toy: a launch of its own for the last row"
    monkeypatch.setattr(KA, "LAUNCH_COUNT_ALLOWLIST", {NORM: why, MM: why_mm})
    rep = reconcile(C.launched(C.launched(C.launched(census(), 0, NORM, 1), 1, NORM, 1), 0, MM, 1), "traced")
    s = sections(CEN.reconciliation_markdown(rep))
    assert list(s) == ["", EXEMPT, KERNELS, STEPS, COUNTS, MATCH]
    assert s[""][-2:] == [(f"- **launch counts: 0 refused** (none); exempt by the launch-count allowlist: `{NORM}` (2 steps), "
                           f"`{MM}` (1 steps)  →  **counts_equal = True**"), ""]
    assert s[EXEMPT] == ["", f"- `{NORM}`: {why}", f"- `{MM}`: {why_mm}", ""]


def test_residue_is_named_in_the_kernels_table_and_at_its_step(reconcile):
    """A launch no name claims: the summary names it whole, its row of the kernels table says RESIDUE in place of a claim, and
    its step's row counts it."""
    rep = reconcile(with_launch(census(), 0, STRAY, 3), "traced")
    assert not rep["ok"] and not rep["residue_zero"] and rep["counts_equal"]
    assert rep["totals"]["residue_kernels"] == 3 and rep["steps"][0]["residue"] == [{"kernel": STRAY, "count": 3, "us": 1.5}]
    s = sections(CEN.reconciliation_markdown(rep))
    assert list(s) == ["", KERNELS, STEPS, COUNTS, MATCH]
    assert s[""][4:7] == [("- census: 2 steps with device work (3 engine steps), 19 device kernels, 4 distinct names; "
                           "tokens == accepted record: True"),
                          "- log: 2 steps (profile_id `toy-profile`), Unsupported 0; steps aligned: True",
                          f"- **residue: 3 kernels** (`{STRAY}`); ambiguous compute claims: 0  →  **residue_zero = False**"]
    assert len(STRAY) > 110
    assert s[KERNELS][-2:] == [f"| `{STRAY[:107]}...` | kernel | 3 | - | 3 | 1.5 | **RESIDUE** →  |", ""]
    assert s[STEPS] == ["", *STEPS_HEAD, f"| 0 | prefill | 2 | 11 | 4 | 8.0 | 22 | `{STRAY[:60]}`×3 | `{ALT}` |", STEP_ROWS[1], ""]
    assert s[COUNTS] == ["", COUNTS_NOTE, "", *COUNTS_HEAD, *COUNTS_ROWS, ""]


def test_two_residue_names_and_two_names_never_seen_are_each_listed(reconcile):
    """Step 0 holds two launches no name claims, and a launch of the fused norm's other kernel, so no declared name is left
    unseen there; step 1 holds no launch of either fused norm, which is not a refusal: an alternative the step did not take."""
    doc = with_launch(with_launch(with_launch(census(), 0, STRAY, 3), 0, OTHER, 1), 0, f"void vllm::{ALT}<c10::BFloat16, 8>(float*)", 2)
    doc["steps"][1]["n_kernels"] -= doc["steps"][1]["kernels"].pop(C.KINETO[NORM])["count"]
    rep = reconcile(doc, "traced")
    assert not rep["residue_zero"] and rep["counts_equal"] and rep["totals"]["residue_names"] == [OTHER, STRAY]
    s = sections(CEN.reconciliation_markdown(rep))
    assert s[""][6] == f"- **residue: 4 kernels** (`{OTHER}`, `{STRAY}`); ambiguous compute claims: 0  →  **residue_zero = False**"
    assert s[STEPS] == ["", *STEPS_HEAD, f"| 0 | prefill | 2 | 14 | 6 | 8.0 | 22 | `{STRAY[:60]}`×3, `{OTHER}`×1 | - |",
                        f"| 1 | decode | 1 | 6 | 2 | 8.0 | 22 | 0 | `{NORM}`, `{ALT}` |", ""]


def test_a_name_two_patterns_declare_lists_both(reconcile):
    rep = reconcile(census(), "traced")
    rep["steps"][1]["counts"][MM]["patterns"] = ["mm", "addmm"]
    assert sections(CEN.reconciliation_markdown(rep))[COUNTS][-2] == f"| `{MM}` | 5/5/5 | 5/5/5 | [] | `addmm`, `mm` |"


def test_a_what_if_allowlist_is_named_apart_from_the_profiles(reconcile):
    """`proposed_allowlist` claims the launch, so the residue is zero, and the report's last line says the name is not the
    pinned profile's."""
    rep = CEN.reconcile(with_launch(census(), 0, STRAY, 3), "traced", "toy", ("stray::launch",))
    assert rep["ok"] and rep["proposed_allowlist"] == ["stray::launch"] and "stray::launch" not in KA.KERNEL_ALLOWLIST
    md = CEN.reconciliation_markdown(rep)
    assert sections(md)[KERNELS][-2:] == [f"| `{STRAY[:107]}...` | kernel | 3 | - | 3 | 1.5 | `stray::launch` → `allowlist` |", ""]
    assert md.split("\n")[-4:] == [
        "Allowlist (profile `kernel_allowlist`): " + ", ".join(f"`{a}`" for a in sorted({*KA.KERNEL_ALLOWLIST, "stray::launch"})),
        "",
        "Proposed allowlist additions applied in this what-if reconciliation (NOT in the pinned profile): `stray::launch`",
        ""]


def test_a_census_step_the_log_lacks_is_an_error_row(reconcile):
    """The step is refused whatever it launched, and the count comparison is of the steps that resolved."""
    doc = census()
    doc["steps"].append({"step": 2, "n_kernels": 1, "kernels": {C.KINETO[NORM]: {"count": 1, "us": 1.0}}, "device_us": 1.0})
    rep = reconcile(doc, "traced")
    assert not rep["ok"] and not rep["residue_zero"] and not rep["steps_aligned"] and rep["totals"]["residue_kernels"] == 0
    assert rep["steps"][2] == {"step": 2, "error": "log step 2 missing or unresolved: None"}
    s = sections(CEN.reconciliation_markdown(rep))
    assert s[""][4:7] == [("- census: 3 steps with device work (3 engine steps), 17 device kernels, 3 distinct names; "
                           "tokens == accepted record: True"),
                          "- log: 2 steps (profile_id `toy-profile`), Unsupported 0; steps aligned: False",
                          "- **residue: 0 kernels** (none); ambiguous compute claims: 0  →  **residue_zero = False**"]
    assert s[STEPS] == ["", *STEPS_HEAD, *STEP_ROWS, "| 2 | ERROR | | | | | | log step 2 missing or unresolved: None | |", ""]
    assert s[COUNTS] == ["", COUNTS_NOTE, "", *COUNTS_HEAD, *COUNTS_ROWS, ""]
    doc["steps"] = doc["steps"][-1:]
    alone = sections(CEN.reconciliation_markdown(reconcile(doc, "traced")))
    assert alone[STEPS] == ["", *STEPS_HEAD, "| 2 | ERROR | | | | | | log step 2 missing or unresolved: None | |", ""]
    assert alone[COUNTS] == ["", COUNTS_NOTE, ""]


def test_a_step_the_log_does_not_resolve_is_an_error_row_with_why(reconcile, monkeypatch):
    """A profile that refuses the decode step: its row carries the resolver's reason, and the count comparison has the one
    column of the step that resolved."""
    def refuses(node, ctx):
        if ctx.start.step == 1:
            raise ResolutionError("no pattern takes the decode step")
        return None
    toy = C.profile()
    monkeypatch.setattr(RL, "load_profile", lambda name: (dataclasses.replace(toy, patterns=[Pat("refuses", refuses), *toy.patterns]), None))
    rep = reconcile(census(), "traced")
    assert not rep["ok"] and not rep["residue_zero"] and rep["steps_aligned"] and rep["counts_equal"]
    assert rep["steps"][1] == {"step": 1, "error": "log step 1 missing or unresolved: no pattern takes the decode step"}
    s = sections(CEN.reconciliation_markdown(rep))
    assert s[STEPS] == ["", *STEPS_HEAD, STEP_ROWS[0],
                        "| 1 | ERROR | | | | | | log step 1 missing or unresolved: no pattern takes the decode step | |", ""]
    assert s[COUNTS] == ["", COUNTS_NOTE, "",
                         "| declared | step 0 census/events/parts | decode steps 2.. census (set) | patterns |", "| --- | --- | --- | --- |",
                         f"| `{NORM}` | 2/2/2 | [] | `fused_norm` |", f"| `{ALT}` | 0/2/2 | [] | `fused_norm` |",
                         f"| `{INDEX}` | 1/1/1 | [] | `index` |", f"| `{MM}` | 5/5/5 | [] | `mm` |", ""]


def test_a_launch_two_computations_claim_is_listed_as_ambiguous(reconcile):
    """At each step one launch that the GEMM's name and the fused norm's both claim, in place of one of each: the counts are
    equal and the residue zero, so the reconciliation passes, and the report flags the launch by step."""
    doc = census()
    for step in (0, 1):
        with_launch(C.launched(C.launched(doc, step, NORM, -1), step, MM, -1), step, BOTH, 1)
    rep = reconcile(doc, "traced")
    assert rep["ok"] and rep["totals"]["ambiguous"] == 2 and len(BOTH) > 100
    assert rep["ambiguous"] == [{"step": step, "kernel": BOTH, "compute_claims": [NORM, MM]} for step in (0, 1)]
    s = sections(CEN.reconciliation_markdown(rep))
    assert list(s) == ["", KERNELS, STEPS, COUNTS, AMBIGUOUS, MATCH]
    assert s[""][6] == "- **residue: 0 kernels** (none); ambiguous compute claims: 2  →  **residue_zero = True**"
    assert s[AMBIGUOUS] == ["", *(f"- step {step}: `{BOTH[:100]}` ← ['{NORM}', '{MM}']" for step in (0, 1)), ""]
    assert s[KERNELS][-2:] == [f"| `{BOTH[:107]}...` | kernel | 1 | 1 | 2 | 3.0 | `{NORM}`, `{MM}` → `fused_norm`, `mm` |", ""]


def test_decode_steps_past_the_first_are_a_set_of_counts(reconcile):
    """The toy log has one decode step, so the later ones are written into the report: the count comparison gives a column to
    each of the first two steps and the set of the census counts of the rest, with `-` where a step declares no such kernel;
    the kernels table gives the decode steps' distinct counts."""
    rep = reconcile(census(), "traced")
    for step, launches in ((2, 3), (3, 2), (4, 3)):
        later = copy.deepcopy(rep["steps"][1])
        later["step"] = step
        later["counts"][NORM]["census_kernels"] = launches
        del later["counts"][INDEX]
        later["counts"]["late_kernel"] = {"census_kernels": 1, "declaring_events": 1, "declaring_parts": 1, "patterns": ["late"]}
        rep["steps"].append(later)
    rep["kernels"][0]["count_decode"] = [2, 3]
    s = sections(CEN.reconciliation_markdown(rep))
    assert s[COUNTS] == ["", COUNTS_NOTE, "", *COUNTS_HEAD,
                         f"| `{NORM}` | 2/2/2 | 2/2/2 | [2, 3] | `fused_norm` |",
                         f"| `{ALT}` | 0/2/2 | 0/2/2 | [0] | `fused_norm` |",
                         f"| `{INDEX}` | 1/1/1 | 1/1/1 | [] | `index` |",
                         "| `late_kernel` | - | - | [1] | `late` |",
                         f"| `{MM}` | 5/5/5 | 5/5/5 | [5] | `mm` |", ""]
    assert s[KERNELS][3] == f"| `{C.KINETO[NORM][:107]}...` | kernel | 2 | 2/3 | 4 | 2.0 | `{NORM}` → `fused_norm` |"
    assert [row.split(" | ")[0] for row in s[STEPS][3:-1]] == ["| 0", "| 1", "| 2", "| 3", "| 4"]


def run_main(tmp_path, doc: dict, *more: str) -> tuple[int, dict, str]:
    """`verity-vllm census --reconcile` on `doc`: its exit status, the report it wrote and the markdown beside it."""
    (tmp_path / "census.json").write_text(json.dumps(doc))
    base = tmp_path / "reports" / "toy" / "census"
    rc = CEN.main(cli.parse("census", ["--reconcile", str(tmp_path / "census.json"), "--profile", "toy", "--log", "traced",
                                       "--report", str(base), *more], {}))
    return rc, json.loads(Path(f"{base}.json").read_text()), Path(f"{base}.md").read_text()


def test_main_writes_the_report_and_its_markdown(reconcile, tmp_path, capsys):
    """Into a directory it makes, two levels down: the reconciliation as JSON, and beside it the markdown of that same report.
    A second run writes into the directory that is now there."""
    rc, rep, md = run_main(tmp_path, census())
    assert rc == 0 and rep["ok"] and rep["log"] == "traced" and rep["profile"] == "toy" and rep["proposed_allowlist"] == []
    assert md == CEN.reconciliation_markdown(rep) and md.startswith("# Kernel census × resolution: `traced` × profile `toy`\n")
    base = tmp_path / "reports" / "toy" / "census"
    assert Path(f"{base}.json").read_text() == json.dumps(rep, indent=1) + "\n"
    assert capsys.readouterr().out.split("\n") == [
        ("[census] steps 2 (log 2), kernels 16, distinct 3, residue 0 [], ambiguous 0 -> residue_zero=True; "
         "launch counts refused 0 [] -> counts_equal=True"),
        f"wrote {base}.json and {base}.md", ""]
    assert run_main(tmp_path, with_launch(census(), 0, STRAY, 3))[0] == 1
    assert sorted(p.name for p in base.parent.iterdir()) == ["census.json", "census.md"]


def test_main_exits_1_on_a_refusal_and_prints_each(reconcile, tmp_path, capsys):
    """Two stray launches at step 0, one fused norm too few at step 1 and a step the log lacks: one line each, the steps' first."""
    doc = C.launched(with_launch(with_launch(census(), 0, STRAY, 3), 0, OTHER, 1), 1, NORM, -1)
    doc["steps"].append({"step": 2, "n_kernels": 1, "kernels": {C.KINETO[NORM]: {"count": 1, "us": 1.0}}})
    rc, rep, md = run_main(tmp_path, doc)
    assert rc == 1 and not rep["ok"] and md == CEN.reconciliation_markdown(rep)
    (refused,) = rep["count_mismatches"]
    base = tmp_path / "reports" / "toy" / "census"
    assert len(STRAY) > 80
    assert capsys.readouterr().out.split("\n") == [
        (f"[census] steps 3 (log 2), kernels 20, distinct 5, residue 4 ['{OTHER}', '{STRAY}'], ambiguous 0 -> residue_zero=False; "
         f"launch counts refused 1 ['{NORM}'] -> counts_equal=False"),
        f"  step 0: residue {STRAY[:80]} x3; {OTHER} x1",
        "  step 2: log step 2 missing or unresolved: None",
        f"  {CEN.count_line(refused)}",
        f"wrote {base}.json and {base}.md", ""]


def test_main_passes_the_what_if_names_on(reconcile, tmp_path):
    """`--propose-allowlist` is a comma list, its empty items dropped; with the stray launch claimed the status is 0."""
    rc, rep, md = run_main(tmp_path, with_launch(census(), 0, STRAY, 3), "--propose-allowlist", "stray::launch,,late_kernel,")
    assert rc == 0 and rep["ok"] and rep["proposed_allowlist"] == ["late_kernel", "stray::launch"]
    assert md.split("\n")[-2] == ("Proposed allowlist additions applied in this what-if reconciliation (NOT in the pinned profile): "
                                  "`late_kernel`, `stray::launch`")


@pytest.mark.parametrize("argv, why", [
    ([], "either --reconcile CENSUS.json or --workload W --out DIR"),
    (["--out", "census"], "either --reconcile CENSUS.json or --workload W --out DIR"),
    (["--reconcile", "census.json"], "--reconcile needs --profile, the profile the log resolves against")])
def test_main_without_what_it_needs_is_a_usage_error(argv, why):
    with pytest.raises(UsageError) as refused:
        CEN.main(cli.parse("census", argv, {}))
    assert str(refused.value) == why


def test_main_hands_the_gpu_half_its_options(monkeypatch):
    """What `run_census` is called with, the run itself left out: each `--no-...` flag as what stays on, and the
    `--engine-arg` items as the engine's keywords."""
    calls = []
    monkeypatch.setattr(CEN, "run_census", lambda *a, **kw: calls.append((a, kw)))
    assert CEN.main(CEN.Options(workload="w.json", out="census", no_warmup=True, engine_arg=["max_num_batched_tokens=64"])) == 0
    assert CEN.main(CEN.Options(workload="w.json", out="census", checkpoints="c.json", max_tokens=4, max_num_seqs=2, case="B1",
                                no_traces=True)) == 0
    assert calls == [
        (("w.json", "census", "manifests/checkpoints.json"),
         {"max_tokens": None, "warmup": False, "max_num_seqs": 8, "keep_traces": True, "case": None,
          "engine_args": {"max_num_batched_tokens": 64}}),
        (("w.json", "census", "c.json"),
         {"max_tokens": 4, "warmup": True, "max_num_seqs": 2, "keep_traces": False, "case": "B1", "engine_args": {}})]
