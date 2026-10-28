"""curve.py's selection: which arm is which scheme and value retention, which measures count toward G2's curve and why not, and
which counting measure is the best per model, GPU, mode, batch, phase, metric and column. The documents are small stand-ins in
the three formats the store holds (a verity-bench result doc, a tp_decode summary, a hash-loop result's meta)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import curve as C


def harness_doc(*, passed: bool = True, gpu_id: str = "h200", eager: bool = False, seen: str | None = None) -> dict:
    """Qwen3-8B at B32: stock, the full rows-sha512 arm, the per-layer 4 KiB arm and vllm-v1's nativedrain arm."""
    def arm(ttft: float, p50: float, tput: float, ci: tuple[float, float] | None = None) -> dict:
        def r(x: float) -> dict:
            return {"mean": x, "ratio_to_stock": {"mean": x, "n": 3, **({"ci95": list(ci)} if ci else {})}}

        return {"metrics": {"time_to_first_token_s": r(ttft), "inter_token_latency_p50_s": r(p50), "tokens_per_s": r(tput)},
                "extras": {"committed_bytes": {"mean": 5.8e9}}}

    stock = {"metrics": {"time_to_first_token_s": {"mean": 0.0987, "ratio_to_stock": {"mean": 1.0}},
                         "inter_token_latency_p50_s": {"mean": 0.00564, "ratio_to_stock": {"mean": 1.0}},
                         "tokens_per_s": {"mean": 2800.0, "ratio_to_stock": {"mean": 1.0}}}}
    arms = [{"id": "stock"}, {"id": "served-commit-rows", "module_spec": "served_commit.prof_arm:SERVED_COMMIT_ROWS"},
            {"id": "served-commit-rows-layer-4k", "module_spec": "served_commit.prof_arm:SERVED_COMMIT_ROWS_LAYER_4K"},
            {"id": "served-commit-nativedrain", "module_spec": "served_commit.prof_arm:SERVED_COMMIT_NATIVEDRAIN"}]
    return {"schema": "verity-bench/vllm-overhead/v1", "baseline": "stock",
            "identity": {"arms": arms, **({"gpu": seen} if seen else {}),
                         "preset": {"model": "Qwen/Qwen3-8B", "gpu_id": gpu_id,
                                    "engine_settings": {"enforce_eager": eager, "serving_defaults": True}}},
            "gates": [{"kind": "tokens", "arm": "served-commit-rows", "passed": passed}],
            "points": [{"point": {"batch": 32}, "arms": {"stock": stock, "served-commit-rows": arm(2.06, 1.35, 0.6, (2.0, 2.1)),
                                                         "served-commit-rows-layer-4k": arm(1.08, 1.14, 0.88),
                                                         "served-commit-nativedrain": arm(1.90, 1.38, 0.62)}}]}


def tp_summary(*, arm: str = "rows", tokens_equal: bool = True) -> dict:
    return {"arm": arm, "committer": "native_collect_v2_auto_nw+rows-sha512+moe", "row_bytes": 4096 if "4k" in arm else 256,
            "world": 8, "gates_passed": True,
            "batches": {"1": {"control": {"tpot_s": 0.1945, "ttft_mean_s": 0.2095}, "gates_passed": True,
                              "control_tokens_equal": tokens_equal,
                              "tpot_ratio": {"median": 1.398, "min": 1.389, "max": 1.404, "pairs": 3},
                              "ttft_mean_ratio": {"median": 1.441, "min": 1.412, "max": 1.447, "pairs": 3}}}}


def loop_meta(*, status: str = "passed", pin: str = "pinned") -> dict:
    ms = [{"arm": "sha512", "name": "tpot_overhead_pct", "value": 47.603, "ci95": [46.314, 48.893]},
          {"arm": "sha512", "name": "throughput_overhead_pct", "value": 32.25},
          {"arm": "sha512", "name": "prefill_step_ms", "value": 113.4617}, {"arm": "sha512", "name": "decode_step_ms", "value": 8.021},
          {"arm": "sha512-layer", "name": "decode_step_ms", "value": 7.263},
          {"arm": "blake3", "name": "tpot_overhead_pct", "value": 33.839},
          {"arm": "stock", "name": "prefill_step_ms", "value": 49.915}, {"arm": "stock", "name": "decode_step_ms", "value": 6.363},
          {"arm": "stock", "name": "tpot_ms", "value": 0.35518}]
    return {"run_id": "r-loop", "validation": {"status": status}, "measurements": ms,
            "workload_fingerprint": {"pin": {"state": pin}, "retain": {"served-commit-rows": "none"}}}


LOOP_LABELS = {"arm": "qwen3-8b", "row": "11", "hardware": "H200"}


def one(ms: list[C.Measure], **kw) -> C.Measure:
    hits = [m for m in ms if all(getattr(m, k) == v for k, v in kw.items())]
    assert len(hits) == 1, (kw, hits)
    return hits[0]


@pytest.mark.parametrize("arm, committer, want", [
    ("served-commit-rows", "", ("rows-sha512", "full", 256)),
    ("served-commit-rows-4k", "", ("rows-sha512", "full", 4096)),
    ("served-commit-rows-layer-4k", "", ("rows-sha512", "per-layer", 4096)),
    ("served-commit-rows-ring8-cap", "", ("rows-sha512", "full", 256)),
    ("served-commit-nativedrain", "", ("vllm-v1", "full", None)),
    ("served-commit-noretain", "", ("vllm-v1", "full", None)),
    ("served-commit", "", ("vllm-v1", "full", None)),
    ("served-commit-rows-blake3", "", ("rows-blake3", "full", None)),
    ("blake3", "", ("rows-blake3", "full", None)),
    ("sha512-layer", "", ("rows-sha512", "per-layer", 256)),
    ("layer-only", "native_collect_v2+rows-sha512", ("rows-sha512", "per-layer", 256)),
])
def test_classify_names_each_arm_by_its_scheme(arm, committer, want):
    assert C.classify(arm, committer) == want


def test_pin_state_reads_the_served_pin_check():
    assert C.pin_state(None) == "unpinned"
    assert C.pin_state("[served-pin] /x/inductor: winners equal to artifact f23f3e483d51b467's") == "pinned"
    assert C.pin_state("[served-pin] /x/inductor: 2 winners differ from artifact f23f3e483d51b467's") == "pin-failed"


def test_harness_measures_read_each_phase_against_the_runs_own_stock():
    ms = C.harness_measures(harness_doc(), run="r1", source="art:1", pin="pinned")
    ttft = one(ms, arm="served-commit-rows", metric="ttft")
    assert (ttft.phase, ttft.ratio, ttft.lo, ttft.hi, ttft.model, ttft.gpu, ttft.mode) == \
        ("prefill", 2.06, 2.0, 2.1, "Qwen3-8B", "H200", "compiled")
    assert ttft.stock_ms == pytest.approx(98.7)
    assert ttft.retain == "none" and ttft.committed_gb == pytest.approx(5.8)
    assert one(ms, arm="served-commit-rows", metric="itl_p50").phase == "decode"
    tput = one(ms, arm="served-commit-rows", metric="tput")
    assert tput.ratio == pytest.approx(1 / 0.6) and tput.stock_ms is None
    assert all(m.counts() == (True, "") for m in ms if m.scheme == "rows-sha512")


def test_a_failed_gate_takes_every_arm_of_its_doc_off_the_curve():
    ms = C.harness_measures(harness_doc(passed=False), run="r1", source="art:1", pin="pinned")
    assert ms and all(m.counts() == (False, "gates failed") for m in ms)


def test_sha256_is_marked_and_never_counted():
    m = one(C.harness_measures(harness_doc(), run="r1", source="art:1", pin="pinned"), arm="served-commit-nativedrain", metric="ttft")
    assert m.sha256
    assert m.counts() == (False, "SHA-256 (vllm-v1): not the one hash")


def test_a_compiled_measure_counts_only_pinned_and_an_eager_one_needs_no_pin():
    unpinned = C.harness_measures(harness_doc(gpu_id="b200"), run="r1", source="art:1", pin="unpinned")
    assert one(unpinned, arm="served-commit-rows", metric="ttft").counts() == \
        (False, "unpinned: not the Build of record's kernels")
    failed = C.harness_measures(harness_doc(), run="r1", source="art:1", pin="pin-failed")
    assert not one(failed, arm="served-commit-rows", metric="ttft").counts()[0]
    eager = C.harness_measures(harness_doc(eager=True), run="r1", source="art:1", pin="unpinned")
    assert one(eager, arm="served-commit-rows", metric="ttft").counts() == (True, "")


@pytest.mark.parametrize("seen, device, want_gpu", [
    ("NVIDIA B200", None, "B200"),
    (None, "B200", "B200"),
    ("NVIDIA B200", "H200", "B200"),
])
def test_the_gpu_is_the_device_and_a_preset_for_another_gpu_doesnt_count(seen, device, want_gpu):
    ms = C.harness_measures(harness_doc(seen=seen), run="r1", source="art:1", pin="pinned", device=device)
    m = one(ms, arm="served-commit-rows", metric="ttft")
    assert (m.gpu, m.preset_gpu) == (want_gpu, "H200")
    assert m.counts() == (False, "H200's served preset on B200")
    own = one(C.harness_measures(harness_doc(gpu_id="b200", seen=seen), run="r1", source="art:1", pin="pinned", device=device),
              arm="served-commit-rows", metric="ttft")
    assert own.counts() == (True, "")


def test_gpu_names_from_ids_skus_and_families():
    assert C.gpu_name("h200") == "H200"
    assert C.gpu_name("rtxpro6000") == "RTX PRO 6000"
    assert C.gpu_name(None, "NVIDIA RTX PRO 6000 Blackwell Server Edition") == "RTX PRO 6000"
    assert C.gpu_name(None, "NVIDIA B200") == "B200"


def test_a_what_if_arm_never_counts():
    m = one(C.harness_measures(harness_doc(), run="r1", source="art:1", pin="pinned"), arm="served-commit-rows", metric="ttft")
    what_if = C.Measure(**{**m.__dict__, "arm": "served-commit-rows-whatif-q"})
    assert what_if.scheme == "rows-sha512" and what_if.counts() == (False, "a what-if: not a Commit")


def test_tp_measures_gate_on_the_controls_tokens():
    ms = C.tp_measures(tp_summary(), run="r2", source="art:2", gpu="8x H200")
    decode = one(ms, metric="step")
    assert (decode.model, decode.phase, decode.ratio, decode.lo, decode.hi, decode.pin) == \
        ("GLM-5.3", "decode", 1.398, 1.389, 1.404, "eager")
    assert decode.stock_ms == pytest.approx(194.5) and decode.counts() == (True, "")
    assert one(ms, metric="ttft").phase == "prefill"
    bad = C.tp_measures(tp_summary(tokens_equal=False), run="r2", source="art:2", gpu="8x H200")
    assert all(m.counts() == (False, "gates failed") for m in bad)
    layer = one(C.tp_measures(tp_summary(arm="rows-layer-4k"), run="r3", source="art:3", gpu="8x H200"), metric="step")
    assert (layer.scheme, layer.acquisition, layer.row_bytes) == ("rows-sha512", "per-layer", 4096)


def test_loop_measures_read_phase_steps_over_stock_prof_and_skip_what_isnt_g2():
    ms = C.loop_measures(loop_meta(), LOOP_LABELS, source="art:4")
    full = [m for m in ms if m.arm == "served-commit-rows"]
    assert one(full, metric="tpot").ratio == pytest.approx(1.47603)
    assert (one(full, metric="tpot").lo, one(full, metric="tpot").hi) == (pytest.approx(1.46314), pytest.approx(1.48893))
    assert one(full, metric="tput").ratio == pytest.approx(1 / 0.6775)
    assert one(full, phase="prefill").ratio == pytest.approx(113.4617 / 49.915)
    assert one(full, phase="decode").ratio == pytest.approx(8.021 / 6.363)
    assert {m.row_bytes for m in full} == {4096} and {m.retain for m in full} == {"none"}
    layer = one(ms, arm="served-commit-rows-layer")
    assert (layer.acquisition, layer.row_bytes, layer.retain) == ("per-layer", 4096, "none")
    assert one(ms, arm="served-commit-rows-blake3").counts() == (False, "rows-blake3: not the one hash")
    first = C.loop_measures(loop_meta(), {**LOOP_LABELS, "row": "0"}, source="art:4")
    assert {m.row_bytes for m in first if m.arm == "served-commit-rows"} == {256}
    assert C.loop_measures(loop_meta(), {**LOOP_LABELS, "row": "what-if-a"}, source="art:4") == []
    assert C.loop_measures(loop_meta(), {**LOOP_LABELS, "arm": "glm-5.3"}, source="art:4") == []


@pytest.mark.parametrize("labels, meta", [
    ({**LOOP_LABELS, "kernel_outcome": "voided"}, loop_meta()),
    ({**LOOP_LABELS, "superseded_by": "r-new"}, loop_meta()),
    (LOOP_LABELS, loop_meta(status="failed")),
])
def test_a_voided_superseded_or_failed_loop_result_doesnt_count(labels, meta):
    ms = C.loop_measures(meta, labels, source="art:4")
    assert ms and all(m.counts() == (False, "gates failed") for m in ms)


def loop_doc_measures(run: str, row: str | None) -> list[C.Measure]:
    """A loop run's own result doc (the loop's arm names), as run_measures reads it: under its job's ROW when it has one."""
    d = harness_doc()
    d["identity"]["arms"][2]["id"] = "served-commit-rows-layer"
    d["points"][0]["arms"]["served-commit-rows-layer"] = d["points"][0]["arms"].pop("served-commit-rows-layer-4k")
    ms = C.harness_measures(d, run=run, source="art:1", pin="pinned")
    return ms if row is None else [C.as_loop_row(m, row) for m in ms]


def test_a_loop_runs_own_doc_takes_its_rows_iteration_and_row_size():
    eleven, first, rerun = loop_doc_measures("r-11", "11"), loop_doc_measures("r-0", "0"), loop_doc_measures("r-rt", "redteam-7")
    assert {m.row_bytes for m in eleven + rerun if m.scheme == "rows-sha512"} == {4096}
    assert one(first, arm="served-commit-rows", metric="ttft").row_bytes == 256
    assert one(first, arm="served-commit-rows-layer", metric="ttft").row_bytes == 4096
    assert one(rerun, arm="served-commit-rows", metric="itl_p50").iteration == 7
    assert all(m.counts() == (True, "") for m in eleven + first + rerun if m.scheme == "rows-sha512")
    other = one(loop_doc_measures("r-other", None), arm="served-commit-rows-layer", metric="ttft")
    assert (other.row_bytes, other.iteration, other.loop_row) == (256, None, None)


def test_a_loop_what_if_row_and_a_dropped_iteration_are_off_the_line():
    what_if = loop_doc_measures("r-w", "what-if-cap33")
    assert all(m.counts() == (False, "the hash loop's what-if-cap33 row: not its line") for m in what_if
               if m.scheme == "rows-sha512")
    ms = C.with_verdicts(loop_doc_measures("r-13", "13") + loop_doc_measures("r-void", "11") + loop_doc_measures("r-11", "11"),
                         {"r-13": ("dropped", True), "r-void": ("voided", False), "r-11": ("kept", True)})
    rows = [m for m in ms if m.scheme == "rows-sha512"]
    assert all(m.counts() == (False, "the hash loop dropped this iteration's change") for m in rows if m.run == "r-13")
    assert all(m.counts() == (False, "gates failed") for m in rows if m.run == "r-void")
    assert all(m.counts() == (True, "") and m.verdict == "kept" for m in rows if m.run == "r-11")
    dropped = C.loop_measures(loop_meta(), {**LOOP_LABELS, "kernel_outcome": "dropped"}, source="art:4")
    assert dropped and not any(m.counts()[0] for m in dropped)
    assert C.loop_verdict(loop_meta(), {**LOOP_LABELS, "kernel_outcome": "dropped"}) == ("r-loop", "dropped", True)
    assert C.loop_verdict(loop_meta(), {**LOOP_LABELS, "arm": "glm-5.3"}) is None
    assert [C.loop_iteration(r) for r in ("0", "14", "redteam-7", "what-if-a")] == [0, 14, 7, None]


def test_the_vllm_v1_column_takes_only_runs_of_record():
    ms = C.harness_measures(harness_doc(seen="NVIDIA B200"), run="r1", source="art:1", pin="pinned")
    k = ("Qwen3-8B", "B200", "compiled", 32, "prefill", "ttft")
    assert k not in C.best(ms) or "vllm-v1" not in C.best(ms)[k]
    own = C.harness_measures(harness_doc(gpu_id="b200", seen="NVIDIA B200"), run="r2", source="art:2", pin="pinned")
    assert C.best(own)[k]["vllm-v1"].run == "r2"
    unpinned = C.harness_measures(harness_doc(gpu_id="b200", seen="NVIDIA B200"), run="r3", source="art:3", pin="unpinned")
    assert C.best(unpinned).get(k, {}).get("vllm-v1") is None


def test_an_unpinned_loop_result_doesnt_count():
    ms = C.loop_measures(loop_meta(pin="unpinned"), {**LOOP_LABELS, "hardware": "B200"}, source="art:4")
    assert ms and not any(m.counts()[0] for m in ms)


def test_best_is_the_lowest_counting_measure_per_column():
    good = C.harness_measures(harness_doc(), run="r1", source="art:1", pin="pinned")
    lower = [C.Measure(**{**m.__dict__, "ratio": m.ratio - 0.5, "run": "r-low"})
             for m in C.harness_measures(harness_doc(passed=False), run="r-low", source="art:5", pin="pinned")]
    b = C.best(good + lower)
    cols = b[("Qwen3-8B", "H200", "compiled", 32, "prefill", "ttft")]
    assert (cols["full none"].arm, cols["full none"].run) == ("served-commit-rows", "r1")
    assert cols["per-layer none"].arm == "served-commit-rows-layer-4k"
    assert (cols["vllm-v1"].arm, cols["vllm-v1"].run) == ("served-commit-nativedrain", "r1")
    better = C.Measure(**{**cols["full none"].__dict__, "ratio": 1.5, "run": "r9"})
    assert C.best(good + [better])[("Qwen3-8B", "H200", "compiled", 32, "prefill", "ttft")]["full none"].run == "r9"


def glm(arm: str, retain: str | None, decode: float, *, batch: str = "1", run: str = "r-glm") -> list[C.Measure]:
    """A tp_decode arm's measures at one batch, its retention read as run_measures reads it (None: a summary without it)."""
    s = tp_summary(arm=arm)
    s["batches"] = {batch: {**s["batches"]["1"], "tpot_ratio": {"median": decode, "min": decode, "max": decode, "pairs": 3}}}
    if retain:
        s["retain"] = retain
    return C.tp_measures(s, run=run, source="art:g", gpu="8x H200", retain=C.tp_retain(s, {}))


def test_retention_modes_are_columns_of_their_own_and_a_cell_takes_4k_rows_over_256_b():
    old, b8 = glm("rows", None, 1.398, run="r-256"), glm("rows", None, 1.434, batch="8", run="r-256")
    none, host = glm("rows-4k", "none", 1.399, run="r-none"), glm("rows-4k", "host", 1.422, run="r-host")
    assert {m.retain for m in old} == {"none"} and {m.retain for m in host} == {"host"}
    b = C.best(old + b8 + none + host)
    b1 = b[("GLM-5.3", "8x H200", "eager", 1, "decode", "step")]
    assert (b1["full host"].run, b1["full none"].run) == ("r-host", "r-none")
    assert set(b[("GLM-5.3", "8x H200", "eager", 8, "decode", "step")]) == {"full none"}
    assert b[("GLM-5.3", "8x H200", "eager", 8, "decode", "step")]["full none"].run == "r-256"
    head, _, *rows = C.curve_table(old + b8 + none + host).splitlines()
    assert head.index("full Commit, rows-sha512, retain host") < head.index("full Commit, rows-sha512, retain none") \
        < head.index("vllm-v1")
    b8_row = next(r for r in rows if "| 8 | decode | step |" in r)
    assert "| 194.50 | – | 1.43x [1.43, 1.43] rows r-256 | – |" in b8_row


@pytest.mark.parametrize("summary, params, want", [
    ({"retain": "host"}, {"argv": ["env", "RETAIN=none", "bash", "inputs/tp_decode_retain.sh"]}, "host"),
    ({}, {"argv": ["gpu-lease", "8", "--wait", "--mem-gb", "160", "--", "env", "RETAIN=host", "BATCHES=1", "ARM=rows-4k", "bash",
                   "-c", 'bash "$RESEARCH_RUN_DIR/inputs/tp_decode_retain.sh"']}, "host"),
    ({}, {"argv": ["bash", "inputs/tp_decode_retain.sh"], "env": {"RETAIN": "device"}}, "device"),
    ({}, {"argv": ["bash", "inputs/tp_decode_retain.sh"]}, "?"),
    ({}, {"argv": ["gpu-lease", "8", "--wait", "--mem-gb", "400", "--", "env", "ARM=rows", "bash",
                   "benchmarks/served_commit/tp_decode.sh"]}, "none"),
    ({}, {"argv": ["env", "RETAIN=host", "bash", "benchmarks/served_commit/tp_decode.sh"]}, "none"),
    ({}, {"command": ["env", "RETAIN=host", "bash", "inputs/tp_decode_retain.sh"]}, "none"),
    ({}, {}, "none"),
])
def test_a_tp_decode_arms_retention_is_its_summarys_else_t1rs_wrappers_else_none(summary, params, want):
    """`params` is the attempt's, as the store's record of the run has it: the run's command line is its `argv`."""
    assert C.tp_retain(summary, params) == want


def test_a_key_with_only_sha256_or_failed_measures_is_no_curve_row():
    v1 = [m for m in C.harness_measures(harness_doc(), run="r1", source="art:1", pin="pinned") if m.sha256]
    failed = C.harness_measures(harness_doc(passed=False), run="r2", source="art:2", pin="pinned")
    table = C.curve_table(v1 + failed)
    assert len(table.splitlines()) == 2
    assert "| Qwen3-8B | H200 | compiled | 32 | prefill | ttft |" in C.curve_table(
        C.harness_measures(harness_doc(), run="r1", source="art:1", pin="pinned"))


def test_a_curve_rows_stock_is_its_full_columns_run():
    full = C.harness_measures(harness_doc(), run="r1", source="art:1", pin="pinned")
    v1_first = [C.Measure(**{**m.__dict__, "run": "r0", "ratio": 1.01, "stock_ms": 1234.0}) for m in full if m.sha256]
    row = next(ln for ln in C.curve_table(v1_first + full).splitlines() if "| prefill | ttft |" in ln)
    assert "| 98.70 |" in row and "1234" not in row


def test_dedupe_reads_a_run_once():
    ms = C.harness_measures(harness_doc(), run="r1", source="art:1", pin="pinned")
    again = C.harness_measures(harness_doc(), run="r1", source="art:9", pin="pinned")
    assert len(C.dedupe(ms + again)) == len(ms)


def test_the_report_names_why_each_measure_is_off_the_curve():
    ms = C.harness_measures(harness_doc(), run="r1", source="art:1", pin="pinned") \
        + C.tp_measures(tp_summary(tokens_equal=False), run="r2", source="art:2", gpu="8x H200")
    md = C.report(ms)
    assert "no: SHA-256 (vllm-v1): not the one hash" in md and "no: gates failed" in md
    assert "Runs on the curve: r1." in md
