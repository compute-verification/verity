"""`verity_vllm.check.replay.stoch_recompute.recompute_run` and its command on a capture made here: two requests sampled at three
engine steps over a 64-token vocabulary, each step's sampler chain (the top-p mask launch, the Gumbel launch, the SampleEnd) and the
snapshots the value check reads (the bf16 logits rows, the masked fp32 rows, the seeds, positions, temperatures, row mapping and
top-p the kernels consumed).  The served tokens and the masked rows are the Definition's own, so the capture is honest unless a test
changes one thing in it; each test then holds the verdict, the tally and the reason that names the change.  No stored row is read:
the stochastic row of record is the regression suite's `stoch_value` check."""
from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

import numpy as np
import pytest

from verity_vllm.check.replay import stoch_recompute as SR
from verity_vllm.observe.events import (
    RunHeader,
    SampleEnd,
    StepStart,
    TritonLaunch,
    ViewDesc,
)
from verity_vllm.observe.log import LogWriter
from verity_vllm.records.workload import SCHEMA

V, S, PROMPT_LEN, CAP = 64, 4, 5, 3
REQUESTS = (("a", 7), ("b", 9))                      # (request id, declared seed)
TEMPERATURE, TOP_P = 0.8, 0.95
WORDS = {"bfloat16": np.uint16, "float32": np.uint32, "int64": np.uint64, "int32": np.uint32}
OPS = "vllm.v1.sample.ops"


def _f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _view(alloc: int, shape: tuple[int, ...], dtype: str) -> ViewDesc:
    return ViewDesc(alloc, 0, shape, (shape[1], 1) if len(shape) == 2 else (1,), dtype, "cuda:0")


def _launch(event: int, step: int, kernel: str, args: dict, constexprs: dict) -> TritonLaunch:
    return TritonLaunch(id=event, parent=None, step=step, phase="sample", fn=f"{OPS}.{kernel}", src_sha="", args=args, constexprs=constexprs,
                        grid=[1], num_warps=None, num_stages=None, stream=0)


def _capture(root: Path, *, served_xor: int | None = None, without_launch: tuple[str, int] | None = None, without_logits: tuple[int, ...] = (),
             without_arg: str | None = None, masked_xor: int | None = None, device_seed: int | None = None, header_seeds: dict | None = None,
             manifest_seeds: dict | None = None, header_first: bool = True) -> tuple[Path, Path]:
    """Write the capture under `root/capture` and its workload manifest; return (the run directory, the manifest).  The keywords
    each change one thing: the served token of step `served_xor`'s first row, a launch that is absent (`(kernel, step)`), steps whose
    logits rows were not snapshotted, a small buffer of the Gumbel launch that was not, one word of step `masked_xor`'s masked row,
    the seed the kernels consumed for the first request, the seeds the header or the manifest declares, a log without its header."""
    cap = root / "capture"
    (cap / "values").mkdir(parents=True)
    entries = []

    def put(event: int, arg: str, view: ViewDesc, words, step: int) -> None:
        f = cap / "values" / f"{event}_{arg}.bin"
        np.asarray(words).astype(WORDS[view.dtype]).tofile(f)
        entries.append({"event": event, "arg": arg, "view": view.to_json(), "file": str(f.relative_to(cap)), "step": step,
                        "layout": "logical-row-major"})

    n = len(REQUESTS)
    seeds = dict(REQUESTS) | (header_seeds or {})
    with LogWriter(cap / "log.jsonl.gz") as log:
        if header_first:
            log.write(RunHeader(schema="x", created_utc="", profile_id="p", profile={"engine_kwargs": {"async_scheduling": False}}, model={},
                                workload={"requests": [{"request_id": r, "seed": seeds[r]} for r, _ in REQUESTS],
                                          "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": 0}}))
        for step in range(CAP):
            pos = PROMPT_LEN - 1 + step
            rows = [(r, pos) for r, _ in REQUESTS]
            log.write(StepStart(step=step, rows=rows, input_token_ids=[0] * n, positions=[pos] * n, seq_lens={}, num_computed={},
                                block_tables={}, block_size=16, slot_mapping=[], logits_rows=rows, logits_indices=list(range(n)),
                                num_reqs=n, num_tokens=n, num_tokens_padded=n))
            logits = (np.sin(np.arange(n * V) * 0.37 + step).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).reshape(n, V)
            alloc = 100 * (step + 1)
            logits_view, masked_view = _view(alloc, (n, V), "bfloat16"), _view(alloc + 1, (n, V), "float32")
            small = {"P": _view(alloc + 2, (n,), "float32"), "seeds_ptr": _view(alloc + 3, (n,), "int64"),
                     "pos_ptr": _view(alloc + 4, (n,), "int64"), "temp_ptr": _view(alloc + 5, (n,), "float32"),
                     "expanded_idx_mapping_ptr": _view(alloc + 6, (n,), "int32")}
            tokens, masked = [], []
            for row, (_, seed) in enumerate(REQUESTS):
                words = [logits[row].astype(np.uint64), *(np.array([w], dtype=np.uint64) for w in (_f32(TOP_P), seed, pos, _f32(TEMPERATURE)))]
                token, masked_row, _ = SR._eval_job_rows((V, S, words))
                tokens.append(token)
                masked.append(masked_row)
            mask_event, gumbel_event = 2 * step + 1, 2 * step + 2
            if without_launch != ("_topp_sb_mask_kernel", step):
                log.write(_launch(mask_event, step, "topk_topp_triton._topp_sb_mask_kernel", {"P": small["P"]}, {"S": S}))
            if without_launch != ("_gumbel_sample_kernel", step):
                log.write(_launch(gumbel_event, step, "gumbel._gumbel_sample_kernel",
                                  {"logits_ptr": masked_view} | {k: v for k, v in small.items() if k != "P"}, {}))
            if step not in without_logits:
                put(mask_event, "logits", logits_view, logits.reshape(-1), step)
            masked_words = np.concatenate(masked)
            if masked_xor == step:
                masked_words[3] ^= 1
            put(gumbel_event, "logits_ptr", masked_view, masked_words, step)
            consumed = {"P": [_f32(TOP_P)] * n, "seeds_ptr": [device_seed if device_seed is not None and r == REQUESTS[0][0] else s
                                                              for r, s in REQUESTS],
                        "pos_ptr": [pos] * n, "temp_ptr": [_f32(TEMPERATURE)] * n, "expanded_idx_mapping_ptr": list(range(n))}
            for arg, words in consumed.items():
                if arg != without_arg:
                    put(mask_event if arg == "P" else gumbel_event, arg, small[arg], words, step)
            if served_xor == step:
                tokens[0] ^= 1
            if without_launch != ("SampleEnd", step):
                log.write(SampleEnd(step=step, logits_view=logits_view, sampled_token_ids=tokens))
    (cap / "snapshots.json").write_text(json.dumps({"layout": "logical-row-major", "entries": entries}))
    manifest = root / "workload.json"
    declared = dict(REQUESTS) | (manifest_seeds or {})
    manifest.write_text(json.dumps({"schema": SCHEMA, "model": "m", "engine": {}, "sampling": {"variant": "stoch"},
                                    "requests": [{"request_id": r, "prompt_len": PROMPT_LEN, "max_tokens": CAP, "seed": declared[r]}
                                                 for r, _ in REQUESTS],
                                    "variants": {"stoch": {"temperature": TEMPERATURE, "top_p": TOP_P}}}))
    return root, manifest


def _recompute(run: Path, manifest: Path | None, **kw) -> dict:
    kw = {"jobs": 1, "evaluator": "rows", "reference_rows": 0} | kw
    return SR.recompute_run(run / "capture" / "log.jsonl.gz", run / "capture", workload=None if manifest is None else str(manifest), **kw)


def _tally(exact: int = 0, mismatch: int = 0, no_snapshot: int = 0, skipped: int = 0) -> dict:
    return {"exact-match": exact, "mismatch": mismatch, "no-snapshot": no_snapshot, "skipped": skipped}


def test_an_honest_capture_passes_under_either_evaluator_with_and_without_its_manifest(tmp_path):
    run, manifest = _capture(tmp_path)
    facts = {r: {"seed": s, "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": 0} for r, s in REQUESTS}
    for kw in ({"evaluator": "lane"}, {"evaluator": "lane", "jobs": 2}, {"evaluator": "rows", "reference_rows": 2}, {"evaluator": "rows"}):
        rep = _recompute(run, manifest, **kw)
        s = rep["summary"]
        assert (s["verdict"], rep["reasons"], s["rows"]) == ("PASS", [], _tally(exact=6)), kw
        assert (s["engine_steps_in_record"], s["sampled_steps_in_log"], s["sampled_rows_total"], s["rows_checked"], s["complete"]) == (3, 3, 6, 6, True)
        assert s["device_facts_all_match"] and s["device_facts_rows"] == 6 and s["masked_rows_all_equal"] and s["draw_ledger_ok"]
        assert s["program_S_all_agree"] is None and s["requests_at_cap"] == 2 and not s["lag_slot_exercised"]
        assert rep["declared_facts"] == rep["run_header_facts"] == facts and rep["header_vs_manifest"] == []
        assert rep["schema"] == "verity-r15/stoch-recompute/v2" and rep["evaluator"]["name"] == kw["evaluator"]
        ref = rep["evaluator"]["reference"]
        if kw.get("reference_rows"):
            assert (ref["reference_evaluator"], ref["rows_requested"], ref["rows_checked"], ref["disagreements"]) == ("lane", 2, 2, [])
            walked = [r["reference_lane_walk"] for r in rep["rows"] if "reference_lane_walk" in r]
            assert len(walked) == 2 and all(w["token_equal"] and w["masked_row_equal"] for w in walked)
        else:
            assert ref["reference_evaluator"] is None and ref["rows_checked"] == 0
        first = rep["rows"][0]
        assert {k: first[k] for k in ("step", "row", "request_id", "event", "pos", "V", "S", "chain", "status")} == {
            "step": 0, "row": 0, "request_id": "a", "event": 0, "pos": PROMPT_LEN - 1, "V": V, "S": S, "chain": ["_topp_sb_mask_kernel"],
            "status": "exact-match"}
        assert first["token"] == first["served"] and first["masked_row_equal_lanes"] == V
        assert first["declared"] == {"seed": 7, "temperature": TEMPERATURE, "top_p": TOP_P, "pos": PROMPT_LEN - 1}
        assert first["device"] == {"seed": 7, "pos": PROMPT_LEN - 1, "temperature": pytest.approx(TEMPERATURE), "top_p": pytest.approx(TOP_P)}
    # without the manifest the header's records are the declared facts, and the sample ledger has no cap to judge by
    rep = _recompute(run, None)
    assert (rep["summary"]["verdict"], rep["reasons"]) == ("PASS", []) and rep["declared_facts"] == facts and rep["workload"] is None
    assert rep["summary"]["requests_at_cap"] == 0


def test_a_served_token_the_definition_does_not_give_fails_by_name(tmp_path):
    run, manifest = _capture(tmp_path, served_xor=1)
    rep = _recompute(run, manifest)
    bad = rep["rows"][2]
    assert (rep["summary"]["verdict"], rep["summary"]["rows"]) == ("FAIL", _tally(exact=5, mismatch=1))
    assert (bad["step"], bad["row"], bad["status"], bad["served"]) == (1, 0, "mismatch", bad["token"] ^ 1)
    assert rep["reasons"] == [(f"step 1 row 0 (a): served token {bad['served']} != GumbelTopPTokenSelect_v1{{V=64,S=4}} under the declared "
                               f"seed 7 / T=0.8 / p=0.95 / pos=5 -> {bad['token']}")]


def test_a_masked_row_the_kernels_did_not_compute_fails_by_name(tmp_path):
    run, manifest = _capture(tmp_path, masked_xor=2)
    rep = _recompute(run, manifest)
    assert (rep["summary"]["verdict"], rep["summary"]["rows"], rep["summary"]["masked_rows_all_equal"]) == ("FAIL", _tally(exact=6), False)
    assert rep["rows"][4]["masked_row_equal_lanes"] == V - 1 and [d["v"] for d in rep["rows"][4]["masked_row_first_diffs"]] == [3]
    assert rep["reasons"] == [("step 2 row 0 (a): the kernel-consumed masked fp32 row differs from TemperatureScale->TopPMask under the "
                               "declared policy in 1 lanes (first [3])")]


@pytest.mark.parametrize("kw, verdict, tally, reason", [
    ({"without_launch": ("_topp_sb_mask_kernel", 2)}, "INCOMPLETE", _tally(exact=4),
     "step 2: sampler chain incomplete in the record (_topp_sb_mask_kernel launch absent); 2 sampled row(s) not value-checked"),
    ({"without_launch": ("_gumbel_sample_kernel", 0)}, "INCOMPLETE", _tally(exact=4),
     "step 0: sampler chain incomplete in the record (_gumbel_sample_kernel launch absent); 2 sampled row(s) not value-checked"),
    ({"without_launch": ("SampleEnd", 1)}, "FAIL", _tally(exact=4),
     "step 1: sampler chain incomplete in the record (SampleEnd absent); 2 sampled row(s) not value-checked"),
    ({"without_logits": (0,)}, "INCOMPLETE", _tally(exact=4, no_snapshot=1),
     "step 0: 2 sampled row(s) not value-checked (the logits row fed to the sampler was not snapshotted)"),
    ({"without_logits": (0, 1, 2)}, "NOTHING-CHECKED", _tally(no_snapshot=3),
     "step 2: 2 sampled row(s) not value-checked (the logits row fed to the sampler was not snapshotted)"),
])
def test_a_step_whose_chain_or_logits_the_record_lacks_is_counted_and_named_never_passed(tmp_path, kw, verdict, tally, reason):
    run, manifest = _capture(tmp_path, **kw)
    rep = _recompute(run, manifest)
    s = rep["summary"]
    assert (s["verdict"], s["rows"], s["sampled_rows_total"], s["complete"]) == (verdict, tally, 6, False)
    assert reason in rep["reasons"]
    assert f"{s['rows_checked']} of 6 sampled rows value-checked: the sampler chain is unverified at the others" in rep["reasons"][-1]


def test_a_step_without_its_sample_end_leaves_its_requests_served_tokens_unread(tmp_path):
    run, manifest = _capture(tmp_path, without_launch=("SampleEnd", 1))
    rep = _recompute(run, manifest)
    assert rep["summary"]["draw_ledger_ok"] is False
    assert "draw ledger a: a sampled row without its served token: the rule cannot be read for this request" in rep["reasons"]


def test_max_rows_stops_the_check_and_the_verdict_says_how_many_were_checked(tmp_path):
    run, manifest = _capture(tmp_path)
    rep = _recompute(run, manifest, max_rows=3)
    assert (rep["summary"]["verdict"], rep["summary"]["rows"], len(rep["rows"])) == ("INCOMPLETE", _tally(exact=3), 3)
    assert rep["reasons"][-1].startswith("3 of 6 sampled rows value-checked")


def test_a_seed_the_kernels_consumed_that_is_not_the_declared_one_fails_though_the_token_matches(tmp_path):
    run, manifest = _capture(tmp_path, device_seed=8)
    rep = _recompute(run, manifest)
    s = rep["summary"]
    assert (s["verdict"], s["rows"], s["device_facts_all_match"], s["device_facts_rows"]) == ("FAIL", _tally(exact=6), False, 3)
    assert rep["reasons"] == [f"step {k} row 0 (a): the sampler kernels consumed seed=8 but the manifest declares seed=7" for k in range(CAP)]


def test_a_fact_the_kernels_consumed_that_was_not_snapshotted_is_not_tied_to_the_execution(tmp_path):
    run, manifest = _capture(tmp_path, without_arg="pos_ptr")
    rep = _recompute(run, manifest)
    assert (rep["summary"]["verdict"], rep["summary"]["device_facts_rows"]) == ("FAIL", 0)
    assert rep["rows"][0]["device_facts_not_snapshotted"] == ["pos"] and rep["rows"][0]["device_facts_checked"] == ["seed", "temperature", "top_p"]
    assert rep["reasons"][0] == ("step 0 row 0 (a): the kernel-consumed pos not snapshotted -- the declared value cannot be tied to the "
                                 "execution at this step")


def test_a_header_that_disagrees_with_the_manifest_fails_and_the_manifest_is_the_declared_fact(tmp_path):
    run, manifest = _capture(tmp_path, header_seeds={"b": 10})
    rep = _recompute(run, manifest)
    assert rep["summary"]["verdict"] == "FAIL" and rep["summary"]["rows"] == _tally(exact=6)
    assert rep["header_vs_manifest"] == rep["reasons"] == ["b: manifest declares seed=9 but the run's header records seed=10"]
    assert rep["declared_facts"]["b"]["seed"] == 9 and rep["run_header_facts"]["b"]["seed"] == 10
    assert "DISAGREE" in rep["structure_note"]
    # and a manifest that declares another seed than the one served under: the token and the consumed seed both name it
    run, manifest = _capture(tmp_path / "other", manifest_seeds={"a": 8})
    rep = _recompute(run, manifest)
    assert rep["summary"]["verdict"] == "FAIL" and rep["summary"]["rows"]["mismatch"] + rep["summary"]["rows"]["exact-match"] == 6
    assert "a: manifest declares seed=8 but the run's header records seed=7" in rep["reasons"]
    assert "step 0 row 0 (a): the sampler kernels consumed seed=7 but the manifest declares seed=8" in rep["reasons"]


def test_a_request_whose_record_declares_no_seed_is_skipped_and_the_verdict_fails(tmp_path):
    run, _ = _capture(tmp_path, header_seeds={"b": None})
    rep = _recompute(run, None)
    assert (rep["summary"]["verdict"], rep["summary"]["rows"]) == ("FAIL", _tally(exact=3, skipped=3))
    assert rep["rows"][1]["why"] == "request record has no declared seed" and "step 0 row 1: no declared seed for 'b'" in rep["reasons"]


def test_an_evaluator_that_disagrees_with_the_lane_walk_is_named_as_an_evaluator_defect(tmp_path, monkeypatch):
    run, manifest = _capture(tmp_path)
    rows = SR._eval_job_rows

    def off_by_one(job):
        token, masked, seconds = rows(job)
        return token ^ 1, masked, seconds

    monkeypatch.setitem(SR.EVALUATORS, "rows", off_by_one)
    rep = _recompute(run, manifest, reference_rows=6)
    ref = rep["evaluator"]["reference"]
    assert rep["summary"]["verdict"] == "FAIL" and ref["rows_checked"] == 6 and len(ref["disagreements"]) == 6
    first = ref["disagreements"][0]
    assert first == {"step": 0, "row": 0, "request_id": "a", "rows_token": first["lane_token"] ^ 1, "lane_token": first["lane_token"],
                     "masked_row_equal": True}
    assert (f"step 0 row 0 (a): the vectorised evaluator ({first['rows_token']}) and the lane walk ({first['lane_token']}; masked row equal) "
            f"disagree on the same Definition and inputs -- an evaluator defect, not a verdict on the record") in rep["reasons"]


def test_an_evaluator_it_does_not_know_and_a_log_without_its_header_are_refused(tmp_path):
    run, manifest = _capture(tmp_path / "ok")
    with pytest.raises(ValueError, match=r"evaluator 'gpu': one of \['lane', 'rows'\]"):
        _recompute(run, manifest, evaluator="gpu")
    run, manifest = _capture(tmp_path / "headless", header_first=False)
    with pytest.raises(ValueError, match="log does not start with a RunHeader"):
        _recompute(run, manifest)


@pytest.mark.parametrize("kw, verdict, code", [({}, "PASS", 0), ({"served_xor": 0}, "FAIL", 1), ({"without_logits": (1,)}, "INCOMPLETE", 1),
                                               ({"without_logits": (0, 1, 2)}, "NOTHING-CHECKED", 2)])
def test_the_command_writes_the_report_and_its_exit_code_is_the_verdicts(tmp_path, capsys, caplog, kw, verdict, code):
    caplog.set_level(logging.INFO, logger="verity_vllm")
    run, manifest = _capture(tmp_path, **kw)
    assert SR.main(SR.Options(run=str(run), workload=str(manifest), jobs=1, evaluator="rows", reference_rows=0)) == code
    rep = json.loads((run / "stoch_recompute.json").read_text())
    assert rep["summary"]["verdict"] == verdict and rep["workload"] == str(manifest) and rep["summary"]["sampled_rows_total"] == 6
    said = capsys.readouterr().out + caplog.text
    assert f"stoch_recompute {run}: {verdict} {rep['summary']['rows_checked']} of 6 sampled rows value-checked" in said
    assert ("  FAIL: " in said) == bool(rep["reasons"])
    out = tmp_path / "elsewhere.json"
    assert SR.main(SR.Options(run=str(run), out=str(out), jobs=1, evaluator="rows", reference_rows=0, max_rows=6)) == code
    assert json.loads(out.read_text())["summary"]["verdict"] == verdict
