"""What the card reads of a run directory's documents when a block of one is left out (`verity_vllm.pipeline.report`).

A card is assembled from documents a run's stages wrote: the capture's log header and its versions, the census, the resolution, the
fold's summary, the census reconcile, the root policy, tier a's replay, the quarantine, the gates' results and the run's own
manifest.  A block of such a document may be absent, null or empty (a stage that did not run, a writer with nothing to state), and
the card then states `None`, `n/a`, `skipped` or nothing for it.  Each test gives a block absent, null, empty and stated.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from verity_vllm.check import gates as G
from verity_vllm.check.gates import GateResult, RunArtifacts
from verity_vllm.pipeline import report as CARD

GATES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")


def _left_out(key: str) -> list[dict[str, Any]]:
    """A document's block `key` left out three ways: absent, null and empty."""
    return [{}, {key: None}, {key: {}}]


def _each(*docs: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """The documents a test gives one after another."""
    return docs


def _put(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def _run(d: Path, files: dict[str, Any] | None = None, header: dict[str, Any] | None = None) -> RunArtifacts:
    """A run directory with `files` (by their names in it) and, with `header`, a capture log whose first record is that RunHeader."""
    for rel, doc in (files or {}).items():
        _put(d / rel, doc)
    if header is not None:
        log = d / G.ARTIFACTS["log"]
        log.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(log, "wt", encoding="utf-8") as f:
            f.write(json.dumps({"$e": "RunHeader", **header}) + "\n")
    return RunArtifacts(d)


def _results(status: str = "pass", **detail: Any) -> list[GateResult]:
    return [GateResult(id=g, name=g, status=status, detail=detail.get(g, {})) for g in GATES]


def _card(**fields: Any) -> CARD.ReportCard:
    base: dict[str, Any] = {"schema": "s", "utc": "u", "config_id": "c", "model": {}, "C": {}, "engine_args": {}, "workload": {}, "kappa": {},
                            "gates": {}, "verdict": "ACCEPTED"}
    return CARD.ReportCard(**{**base, **fields})


def _rows(card: CARD.ReportCard) -> dict[str, str]:
    """The top section's rows by label."""
    return {ln[:34].rstrip(): ln[34:].rstrip() for ln in CARD.top_section(card).splitlines() if len(ln) > 34}


def _pins(gpu: Any, cc: Any) -> str:
    pins = {"torch": "2.6", "cuda": None, "triton": None, "vllm": None, "gpu": gpu, "cc": cc}
    return hashlib.sha256(json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:10]


def test_kappas_short_form_pins_a_gpu_by_its_name_and_cc_or_by_a_bare_name():
    for gpu in _each({}, {"gpu": None}, {"gpu": {}}, {"gpu": ""}):
        assert CARD.kappa_short({"torch": "2.6", **gpu}) == _pins(None, None), gpu
    assert CARD.kappa_short({"torch": "2.6", "gpu": "L40S"}) == _pins("L40S", None)
    assert CARD.kappa_short({"torch": "2.6", "gpu": {"name": "L40S", "cc": "8.9", "driver": "550"}}) == _pins("L40S", "8.9")
    assert len({_pins(None, None), _pins("L40S", None), _pins("L40S", "8.9")}) == 3


def test_a_workloads_summary_counts_a_request_without_token_ids_as_zero_and_reads_no_sampling_without_variants(tmp_path):
    requests = [{"request_id": "a"}, {"request_id": "b", "prompt_token_ids": None}, {"request_id": "c", "prompt_token_ids": []},
                {"request_id": "d", "prompt_token_ids": [5, 6, 7], "max_tokens": 4, "arrive_step": 2}]
    for variants in _left_out("variants"):
        s = CARD.workload_summary(_put(tmp_path / "w.json", {"requests": requests, **variants}))
        assert [r["prompt_len"] for r in s["requests"]] == [0, 0, 0, 3] and s["prompt_lens"] == [0, 3], variants
        assert (s["sampling"], s["n_requests"], s["max_tokens"], s["arrive_steps"]) == (None, 4, [4], [0, 0, 0, 2]), variants
    greedy = {"temperature": 0.0}
    s = CARD.workload_summary(_put(tmp_path / "w.json", {"requests": requests, "variants": {"greedy": greedy, "sampled": {"temperature": 1.0}}}))
    assert s["sampling"] == greedy
    assert CARD.workload_summary(_put(tmp_path / "w.json", {"requests": requests, "variants": {"sampled": {}}}))["sampling"] is None


PROFILE: dict[str, Any] = {
    "cache": {"max_num_seqs": 8, "enable_prefix_caching": False, "enable_chunked_prefill": True, "kv_cache_dtype": "auto", "block_size": 16},
    "parallelism": {"tensor_parallel_size": 2}, "compilation": {"enforce_eager": True}, "backends": {"attention": "a.b.FlashAttentionImpl"},
    "engine_kwargs": {"seed": 3, "max_model_len": 512, "gpu_memory_utilization": 0.9}, "dtype": "bfloat16", "quantization": "fp8",
    "sampling": {"temperature": 0.0}, "engine_flags": {"F": "1"}, "vllm": {"commit": "abc1234", "version": "0.9.1"},
    "python": {"torch": "2.6", "cuda": "12.4", "triton": "3.2", "version": "3.12"},
    "gpu": {"name": "L40S", "compute_capability": "8.9", "driver": "550"},
    "model": {"weights_sha256": {"model.safetensors": "w" * 64}, "config_sha256": "c" * 64},
}
#: What the header's profile states of the engine: each argument the card records, by where the profile holds it.
FROM_HEADER: dict[str, Any] = {
    "max_num_seqs": 8, "enable_prefix_caching": False, "enable_chunked_prefill": True, "dtype": "bfloat16", "quantization": "fp8",
    "tensor_parallel_size": 2, "enforce_eager": True, "seed": 3, "max_model_len": 512, "gpu_memory_utilization": 0.9,
    "attention_backend": "a.b.FlashAttentionImpl", "kv_cache_dtype": "auto", "block_size": 16, "vllm_commit": "abc1234",
    "sampling": {"temperature": 0.0}, "env_flags": {"F": "1"}, "source": ["capture log header"]}
#: A header that states nothing of the engine: every argument None, but one rank where none is stated.
NOTHING_STATED: dict[str, Any] = {**{k: None for k in FROM_HEADER}, "tensor_parallel_size": 1, "source": ["capture log header"]}


def _without(profile: dict[str, Any], key: str, how: dict[str, Any]) -> dict[str, Any]:
    return {**{k: v for k, v in profile.items() if k != key}, **how}


def test_the_engines_arguments_read_from_a_log_header_with_any_block_of_its_profile_left_out(tmp_path):
    assert CARD.engine_args_of(_run(tmp_path / "stated", header={"profile": PROFILE})) == FROM_HEADER
    for i, profile in enumerate(_left_out("profile")):
        assert CARD.engine_args_of(_run(tmp_path / f"p{i}", header={"model": {"repo": "m"}, **profile})) == NOTHING_STATED, profile
    lost = {"cache": ("max_num_seqs", "enable_prefix_caching", "enable_chunked_prefill", "kv_cache_dtype", "block_size"),
            "parallelism": ("tensor_parallel_size",), "compilation": ("enforce_eager",), "backends": ("attention_backend",),
            "engine_kwargs": ("seed", "max_model_len", "gpu_memory_utilization"), "vllm": ("vllm_commit",)}
    for block, args in lost.items():
        for i, how in enumerate(_left_out(block)):
            got = CARD.engine_args_of(_run(tmp_path / f"{block}{i}", header={"profile": _without(PROFILE, block, how)}))
            want = {**FROM_HEADER, **{a: None for a in args}, **({"tensor_parallel_size": 1} if block == "parallelism" else {})}
            assert got == want, (block, how)


def test_the_engines_sampling_is_the_profiles_then_the_headers_workloads_then_the_censuss(tmp_path):
    no_sampling = _without(PROFILE, "sampling", {})
    for i, workload in enumerate(_left_out("workload")):
        assert CARD.engine_args_of(_run(tmp_path / f"h{i}", header={"profile": no_sampling, **workload}))["sampling"] is None, workload
    of_workload = CARD.engine_args_of(_run(tmp_path / "hw", header={"profile": no_sampling, "workload": {"sampling": {"top_k": 1}}}))
    assert of_workload["sampling"] == {"top_k": 1}
    both = CARD.engine_args_of(_run(tmp_path / "hb", header={"profile": PROFILE, "workload": {"sampling": {"top_k": 1}}}))
    assert both["sampling"] == {"temperature": 0.0}
    for i, workload in enumerate(_left_out("workload")):
        run = _run(tmp_path / f"c{i}", {"census/census.json": {"env_flags": {"E": "2"}, **workload}}, header={"profile": no_sampling})
        got = CARD.engine_args_of(run)
        assert (got["sampling"], got["env_flags"], got["source"]) == (None, {"F": "1"}, ["capture log header", "census.json engine_kwargs"]), workload
    census = {"workload": {"sampling": {"top_p": 0.5}}, "env_flags": {"E": "2"}}
    got = CARD.engine_args_of(_run(tmp_path / "cw", {"census/census.json": census}, header={"profile": _without(no_sampling, "engine_flags", {})}))
    assert (got["sampling"], got["env_flags"]) == ({"top_p": 0.5}, {"E": "2"})


def test_a_censuss_engine_kwargs_override_the_header_and_the_drivers_request_fills_what_neither_states(tmp_path):
    for i, kwargs in enumerate(_left_out("engine_kwargs")):
        run = _run(tmp_path / f"k{i}", {"census/census.json": {"n": 1, **kwargs}}, header={"profile": PROFILE})
        assert CARD.engine_args_of(run) == {**FROM_HEADER, "source": ["capture log header", "census.json engine_kwargs"]}, kwargs
    census = {"engine_kwargs": {"dtype": "float16", "seed": 11, "max_num_seqs": 4, "not_an_argument": 1}}
    got = CARD.engine_args_of(_run(tmp_path / "kw", {"census/census.json": census}, header={"profile": PROFILE}))
    assert got == {**FROM_HEADER, "dtype": "float16", "seed": 11, "max_num_seqs": 4, "source": ["capture log header", "census.json engine_kwargs"]}
    for i, requested in enumerate(_left_out("engine_args_requested")):
        assert CARD.engine_args_of(_run(tmp_path / f"r{i}", {"run.json": {"profile": None, **requested}})) == {
            **{k: None for k in CARD.ENGINE_ARG_KEYS}, "tensor_parallel_size": 1, "source": []}, requested
    asked = {"engine_args_requested": {"dtype": "float32", "seed": 5, "extra": "x"}}
    got = CARD.engine_args_of(_run(tmp_path / "rq", {"run.json": asked}, header={"profile": _without(PROFILE, "engine_kwargs", {})}))
    assert (got["dtype"], got["seed"], got["extra"]) == ("bfloat16", 5, "x") and got["source"] == ["capture log header", "run.json engine_args_requested"]


NO_PINS: dict[str, Any] = {"torch": None, "cuda": None, "triton": None, "vllm": None, "python": None,
                           "gpu": {"name": None, "cc": None, "driver": None}, "host": None, "source": "log header (no versions.json)"}


def test_kappa_from_a_log_header_alone_pins_what_its_profile_states(tmp_path):
    run = RunArtifacts(tmp_path)
    assert CARD.kappa_of(run, None) == {}
    assert CARD.kappa_of(run, {"profile": PROFILE}) == {
        "torch": "2.6", "cuda": "12.4", "triton": "3.2", "vllm": "0.9.1", "python": "3.12",
        "gpu": {"name": "L40S", "cc": "8.9", "driver": "550"}, "host": None, "source": "log header (no versions.json)"}
    for profile in _left_out("profile"):
        assert CARD.kappa_of(run, {"$e": "RunHeader", **profile}) == NO_PINS, profile
    for block, lost in (("python", {"torch": None, "cuda": None, "triton": None, "python": None}), ("vllm", {"vllm": None}),
                        ("gpu", {"gpu": NO_PINS["gpu"]})):
        for how in _left_out(block):
            got = CARD.kappa_of(run, {"profile": _without(PROFILE, block, how)})
            assert got == {**CARD.kappa_of(run, {"profile": PROFILE}), **lost}, (block, how)


def test_a_cards_model_takes_from_the_header_what_the_record_does_not_state(tmp_path):
    C = {"C": {"model": "org/m", "revision": "r1", "H": 8}}
    stated = {"repo": "org/h", "revision": "r2", "config": {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}}
    card = CARD.build_card(_run(tmp_path / "a", {"run.json": C}, header={"model": stated, "profile": PROFILE}).run_dir, _results())
    assert card.model == {"repo": "org/m", "revision": "r1", "family": "Serve_v4", "sha256": "w" * 64, "hf_model_type": "llama",
                          "architectures": ["LlamaForCausalLM"], "config_sha256": "c" * 64}
    card = CARD.build_card(_run(tmp_path / "b", header={"model": stated, "profile": _without(PROFILE, "model", {})}).run_dir, _results())
    assert (card.model["repo"], card.model["revision"], card.model["sha256"], card.model["config_sha256"]) == ("org/h", "r2", None, None)
    nothing = {"repo": None, "revision": None, "family": "Serve_v4", "sha256": None, "hf_model_type": None, "architectures": None,
               "config_sha256": None}
    for i, model in enumerate(_left_out("model")):
        for j, profile in enumerate([*_left_out("profile"), {"profile": {"model": None}}, {"profile": {"model": {}}}]):
            card = CARD.build_card(_run(tmp_path / f"n{i}{j}", header={**model, **profile}).run_dir, _results())
            assert card.model == nothing, (model, profile)
    for i, config in enumerate(_left_out("config")):
        card = CARD.build_card(_run(tmp_path / f"c{i}", header={"model": {"repo": "org/h", **config}}).run_dir, _results())
        assert card.model == {**nothing, "repo": "org/h"}, config
    two = {"a.safetensors": "1" * 64, "b.safetensors": "2" * 64}
    card = CARD.build_card(_run(tmp_path / "two", header={"profile": {"model": {"weights_sha256": two}}}).run_dir, _results())
    assert card.model["sha256"] == two


def test_the_vu_partition_row_prints_n_a_for_what_g8s_detail_leaves_out():
    def text(status: str, detail: Any = None, reason: str | None = None, **g8: Any) -> str:
        return CARD.vu_partition_text({"G8": {"status": status, "reason": reason, **({} if detail is None else {"detail": detail}), **g8}})
    for g8 in _left_out("detail"):
        assert CARD.vu_partition_text({"G8": {"status": "pass", **g8}}) == "None, n/a sets, max w_out n/a/n/a PASS", g8
        assert CARD.vu_partition_text({"G8": {"status": "fail", **g8}}) == "None, n/a sets, max w_out n/a/n/a FAIL", g8
    for how in _each({}, {"partition": None, "width": None}, {"partition": {}, "width": {}}):
        assert text("pass", {"count": 3, **how}) == "None, 3 sets, max w_out n/a/n/a PASS", how
        assert text("fail", {"count": 3, **how}) == "None, 3 sets, max w_out n/a/n/a FAIL", how
    whole = {"count": 3, "partition": {"status": "ok"}, "width": {"max_seen": 16, "w_max": 64}}
    assert text("pass", whole) == "ok, 3 sets, max w_out 16/64 PASS"
    assert text("unsupported", whole, "unsupported (width 'unsupported')") == "unsupported (width 'unsupported'); ok, 3 sets, max w_out 16/64"
    for codes in _each({}, {"codes": None}, {"codes": []}):
        assert text("fail", {**whole, "partition": {"status": "bad", **codes}}) == "bad, 3 sets, max w_out 16/64 FAIL", codes
    failed = {"count": 3, "partition": {"status": "bad", "codes": ["A", "B", "C", "D", "E"]}, "status": "partial",
              "width": {"max_seen": 99, "w_max": 64, "n_violators": 2, "n_unsupported": 1}}
    assert text("fail", failed, "binding: digest; other") == (
        "bad, 3 sets, max w_out 99/64; codes ['A', 'B', 'C', 'D'], 2 violators, 1 unsupported, status partial, binding mismatch FAIL")
    assert text("fail", {**whole, "verdict_status": "ok"}) == "ok, 3 sets, max w_out 16/64 FAIL"
    assert text("fail", {**whole, "verdict_status": "stale"}) == "ok, 3 sets, max w_out 16/64; status stale FAIL"
    assert text("fail", {**whole, "status": "partial", "verdict_status": "stale"}) == "ok, 3 sets, max w_out 16/64; status partial FAIL"
    assert text("skipped") == "skipped (no query-use document)" and text("error", reason="e" * 90) == f"error ({'e' * 80})"
    assert CARD.vu_partition_text({}) == "n/a (record predates G8)"


#: The counts of a run directory that holds nothing, under gates that recorded no detail.
NO_COUNTS: dict[str, Any] = {
    "vu_coverage": "not computed", "vu_zero_mass_specs": None, "vu_partial_tiling_specs": None, "vu_partition": "n/a (record predates G8)",
    "protected_files_modified": None, "existing_primitives_modified": None, "new_root_classes": None, "new_allowlist_entries": None,
    "golden_digest_changes": None, "unsupported": None, "ambiguous": None, "unknown_or_unvalidated_roots": None, "census_residue": None,
    "new_patterns": None, "new_definitions": None, "differential_passed": None, "differential_total": None, "specs_captured": 0,
    "specs_compared": None, "operand_provenance": "skipped", "sampled_vus": "skipped", "served_tokens": "skipped", "chained_slices": "skipped",
    "control_noninterference": "n/a"}
ARTIFACTS = ("resolution.json", "fold_summary.json", "census_reconcile.json", "root_policy.json", "replay_a.json", "quarantine.json")


def test_the_counts_of_a_run_that_recorded_nothing_are_n_a_and_skipped(tmp_path):
    assert CARD.counts_of(_run(tmp_path / "bare"), {}, None) == NO_COUNTS
    for i, gates in enumerate(_each({"G2": None, "G5": None}, {"G2": {}, "G5": {}}, {"G2": {"detail": None}, "G5": {"detail": None}},
                                    {"G2": {"detail": {}}, "G5": {"detail": {}}})):
        assert CARD.counts_of(_run(tmp_path / f"g{i}"), gates, None) == NO_COUNTS, gates
    stating_nothing: list[Any] = [{}, None]      # each document an empty object, then each the JSON null
    for i, doc in enumerate(stating_nothing):
        got = CARD.counts_of(_run(tmp_path / f"e{i}", {name: doc for name in ARTIFACTS}), {}, {})
        assert got == NO_COUNTS, doc
    blocks: dict[str, tuple[str, ...]] = {
        "resolution.json": ("totals", "roots", "by_spec"), "fold_summary.json": ("unknown_roots",), "census_reconcile.json": ("totals",),
        "root_policy.json": ("classes", "roots", "new_root_classes"), "replay_a.json": ("coverage", "uncovered_specs", "vu_population"),
        "quarantine.json": ("differential",)}
    seen = {"unknown_or_unvalidated_roots": 0}     # a run that folded, resolved or has a root policy counts its roots: none here
    for name, keys in blocks.items():
        for i, how in enumerate(_each({k: None for k in keys}, {k: {} for k in keys}, {k: [] for k in keys}, {"other": 1})):
            got = CARD.counts_of(_run(tmp_path / f"{name}{i}", {name: how}), {}, None)
            new_classes = {"new_root_classes": 0} if name == "root_policy.json" else {}
            counted = seen if name in ("resolution.json", "fold_summary.json", "root_policy.json") else {}
            assert got == {**NO_COUNTS, **counted, **new_classes}, (name, how)
    g5 = {"provenance": None, "sample": None, "tokens": None, "chain": None}
    assert CARD.counts_of(_run(tmp_path / "g5"), {"G5": {"detail": g5}}, None) == NO_COUNTS
    assert CARD.counts_of(_run(tmp_path / "g5e"), {"G5": {"detail": {k: {} for k in g5}}}, None) == NO_COUNTS


def test_the_counts_of_a_run_read_each_block_its_documents_state(tmp_path):
    files = {
        "resolution.json": {"totals": {"unsupported": 2}, "roots": {"unknown": ["u1", "u2", "u3"]}, "by_spec": {"s1": 1, "s2": 1}},
        "fold_summary.json": {"n": 1},
        "census_reconcile.json": {"totals": {"ambiguous": 5, "residue_kernels": 4}},
        "root_policy.json": {"classes": {"a": {"failed": 2}, "b": {}, "c": {"failed": None}, "d": "not a class"},
                             "roots": [{"status": "pass"}, {"status": "open"}, {}, "not a root"], "new_root_classes": ["n1", "n2"]},
        "replay_a.json": {"vu_population": {"status": "ok", "count": 9, "instances": 7, "oversize_instances": 1, "instances_without_units": 2,
                                            "zero_mass_specs": ["z"], "partial_tiling_specs": []}},
        "quarantine.json": {"differential": {"passed": 3, "total": 4}}}
    g5 = {"provenance": {"reads": 10, "n_mismatch": 0, "uncovered": 0}, "sample": {"status": "pass", "k": 2, "strata": 4, "seed": 9},
          "tokens": {"evaluated": 6, "equal": 6, "steps": [0, 1], "all_rows_equal": True}, "chain": {"status": "fail", "n_slices": 3, "n_edges_chained": 2}}
    got = CARD.counts_of(_run(tmp_path / "a", files), {"G1": {"status": "fail"}, "G5": {"detail": g5}}, None)
    assert got == {
        **NO_COUNTS, "unsupported": 2, "ambiguous": 5, "unknown_or_unvalidated_roots": 3 + 2 + 2, "census_residue": 4, "new_root_classes": 2,
        "differential_passed": 3, "differential_total": 4, "specs_captured": 2, "specs_compared": None, "control_noninterference": "FAIL",
        "operand_provenance": "10 reads, 0 mismatches PASS", "sampled_vus": "k=2 x 4 strata, seed 9 PASS",
        "served_tokens": "6/6 rows at steps [0, 1] PASS", "chained_slices": "3 slices, 2 chained edges FAIL",
        "vu_coverage": "9 VUs / 7 instances, 1 oversize; 2 zero-mass instances (1 specs), 0 partially tiled specs -- SEE replay_a.vu_population",
        "vu_zero_mass_specs": ["z"], "vu_partial_tiling_specs": []}
    # the fold's unknown roots come before the resolution's; G2's own count before the reconcile's; tier a's coverage before the resolution's specs
    files["fold_summary.json"] = {"unknown_roots": ["f1"]}
    files["replay_a.json"] = {"coverage": {"s1": 1, "s2": 1, "s3": 0}, "uncovered_specs": ["s3"], "vu_population": {"status": "error", "error": "boom"}}
    provenance = {"protected_files_modified": ["p"], "existing_primitives_modified": [], "new_root_classes": ["x", "y", "z"],
                  "new_allowlist_entries": None, "golden_digest_changes": ["g1", "g2"], "new_patterns": ["Qwen2Bias"]}
    got = CARD.counts_of(_run(tmp_path / "b", files), {"G1": {"status": "pass"}, "G2": {"detail": {"ambiguous": 0}},
                                                      "G5": {"detail": {"provenance": {"reads": 1, "n_mismatch": 1}, "sample": {"status": "fail"}}}}, provenance)
    assert (got["unknown_or_unvalidated_roots"], got["ambiguous"], got["specs_captured"], got["specs_compared"]) == (1 + 2 + 2, 0, 3, 2)
    assert (got["vu_coverage"], got["vu_zero_mass_specs"], got["control_noninterference"]) == ("unavailable (boom)", None, "PASS")
    assert (got["operand_provenance"], got["sampled_vus"]) == ("1 reads, 1 mismatches FAIL", "k=None x None strata, seed None FAIL")
    assert [got[k] for k in ("protected_files_modified", "existing_primitives_modified", "new_root_classes", "new_allowlist_entries",
                             "golden_digest_changes", "new_patterns", "new_definitions")] == [1, 0, 3, 0, 2, 1, 0]
    # a sealed provenance that lists none of its seven kinds counts zero of each, the root policy's new classes not read
    got = CARD.counts_of(_run(tmp_path / "s", files), {}, {"sealed_by": "verifier"})
    assert [got[k] for k in ("protected_files_modified", "existing_primitives_modified", "new_root_classes", "new_allowlist_entries",
                             "golden_digest_changes", "new_patterns", "new_definitions")] == [0, 0, 0, 0, 0, 0, 0]
    files["replay_a.json"] = {"coverage": {"s1": 1}, "vu_population": {"status": "ok", "count": 1, "instances": 1, "oversize_instances": 0,
                                                                     "instances_without_units": 0}}
    got = CARD.counts_of(_run(tmp_path / "c", files), {}, None)
    assert (got["specs_captured"], got["specs_compared"]) == (1, 1)
    assert got["vu_coverage"] == "1 VUs / 1 instances, 0 oversize; 0 zero-mass instances (0 specs), 0 partially tiled specs"


def test_a_card_lists_the_kernels_the_missing_patterns_and_the_roots_its_run_states(tmp_path):
    empty = CARD.build_card(_run(tmp_path / "bare").run_dir, _results())
    said = ("census_kernels", "patterns_used", "patterns_missing", "roots_by_class", "per_step_instances", "mechanism_notes", "notes")
    nothing: dict[str, Any] = {"census_kernels": [], "patterns_used": {}, "patterns_missing": [], "roots_by_class": {}, "per_step_instances": [],
                               "mechanism_notes": [], "notes": []}
    assert {k: getattr(empty, k) for k in said} == nothing and empty.spend == {"pod_hours": None, "usd": None}
    assert empty.replay == {"tier_a": {k: None for k in ("status", "k", "specs", "covered", "uncovered_specs", "pass", "fail", "skipped")},
                            "provenance": None, "sample": None, "tokens": None, "chain": None, "full_replay": None, "quarantine": None}
    left_out: list[dict[str, Any]] = [
        {"resolution.json": {"unsupported_examples": None, "primary_report": None, "roots": None}, "census_reconcile.json": {"kernels": None},
         "fold_summary.json": {"per_step": None}, "run.json": {"mechanism_notes": None, "notes": None}, "census/census.json": {"run_histogram": None}},
        {"resolution.json": {"unsupported_examples": [], "primary_report": {}, "roots": {}}, "census_reconcile.json": {"kernels": []},
         "fold_summary.json": {"per_step": []}, "run.json": {"mechanism_notes": [], "notes": []}, "census/census.json": {"run_histogram": {}}},
        {"resolution.json": {"primary_report": {"unsupported_ops": None, "by_pattern": None}, "roots": {"classes": None, "unknown": None}}},
        {"resolution.json": {"primary_report": {"unsupported_ops": {}, "by_pattern": {}}, "roots": {"classes": {}, "unknown": []}},
         "census_reconcile.json": {"kernels": ["not a kernel"]}, "census/census.json": []}]
    for i, files in enumerate(left_out):
        card = CARD.build_card(_run(tmp_path / f"n{i}", files).run_dir, _results())
        assert {k: getattr(card, k) for k in said} == nothing, files
    files = {
        "resolution.json": {"unsupported_examples": [{"op": "o1", "reason": "r", "step": 1}, {"op": "o1", "reason": "r", "step": 2}, {"op": "o2"}],
                            "primary_report": {"unsupported_ops": {"o1": 5, "o3": 1}, "by_pattern": {"P": 2}},
                            "roots": {"classes": {"k": 1}, "unknown": ["u"]}},
        "census_reconcile.json": {"kernels": [{"kernel": "k1"}, "not a kernel", {"kernel": "k2"}]},
        "fold_summary.json": {"per_step": [{"step": 0, "phase": "prefill", "n_instances": 3, "more": 1}]},
        "run.json": {"mechanism_notes": ["m"], "notes": ["n"]}, "spend.json": {"usd": 2, "gpu": "L40S"},
        "census/census.json": {"run_histogram": {"never_read": 1}}}
    card = CARD.build_card(_run(tmp_path / "s", files).run_dir, _results(G4={"k": 3, "quarantine": {"specs": 1}}, G5={"chain": {"status": "pass"}}))
    assert {k: getattr(card, k) for k in said} == {
        "census_kernels": ["k1", "k2"], "patterns_used": {"P": 2},
        "patterns_missing": [{"op": "o1", "reason": "r", "step": 1}, {"op": "o2", "reason": None, "step": None}, {"op": "o3", "count": 1}],
        "roots_by_class": {"k": 1, "unknown_names": ["u"]}, "per_step_instances": [{"step": 0, "phase": "prefill", "n_instances": 3}],
        "mechanism_notes": ["m"], "notes": ["n"]}
    assert card.spend == {"pod_hours": None, "usd": 2, "gpu": "L40S"}
    assert (card.replay["tier_a"]["k"], card.replay["quarantine"], card.replay["chain"]) == (3, {"specs": 1}, {"status": "pass"})
    histogram = CARD.build_card(_run(tmp_path / "h", {"census/census.json": {"run_histogram": {"ka": 2, "kb": 1}}}).run_dir, _results())
    assert histogram.census_kernels == ["ka", "kb"]
    given = CARD.build_card(_run(tmp_path / "g", {"run.json": {"mechanism_notes": ["m"], "notes": ["n"]}}).run_dir, _results(),
                            mechanism_notes=[], notes=["mine"])
    assert (given.mechanism_notes, given.notes) == ([], ["mine"])


def test_the_top_section_claims_an_unknown_sm_without_a_gpus_cc_and_no_holdouts_without_g7s(tmp_path):
    for gpu in _each({}, {"gpu": None}, {"gpu": {}}, {"gpu": "L40S"}, {"gpu": {"name": "L40S"}}, {"gpu": {"cc": None}}):
        assert " / sm_unknown / vLLM 0.9" in CARD.top_section(_card(kappa={"vllm": "0.9", **gpu})), gpu
    assert " / sm_89 / vLLM 0.9" in CARD.top_section(_card(kappa={"vllm": "0.9", "gpu": {"name": "L40S", "cc": "8.9"}}))
    for g7 in _each({}, {"G7": None}, {"G7": {}}, {"G7": {"detail": None}}, {"G7": {"detail": {}}}, {"G7": {"detail": {"holdouts": None}}},
                    {"G7": {"detail": {"holdouts": {}}}}):
        top = CARD.top_section(_card(gates=g7)).splitlines()
        assert top[top.index("HOLDOUTS") + 1:top.index("SUPPORT CLAIM")] == [f"  {'(none)':<32}n/a"], g7
    top = CARD.top_section(_card(gates={"G7": {"detail": {"holdouts": {"h1": {"pass": True}, "h2": {}}}}})).splitlines()
    assert top[top.index("HOLDOUTS") + 1:top.index("SUPPORT CLAIM")] == [f"  {'h1':<32}PASS", f"  {'h2':<32}FAIL"]


def _line(md: str, starts: str) -> str:
    return next(ln for ln in md.splitlines() if ln.startswith(starts))


def test_the_cards_summary_prints_none_for_what_kappa_the_workload_and_the_diagnostics_leave_out():
    no_host = "host clocksource `None` (None µs/clock read)"
    for k in _each({}, {"gpu": None, "host": None}, {"gpu": {}, "host": {}}):
        assert _line(CARD.summarize_md(_card(kappa=k)), "- κ: ").endswith(f"GPU `None` cc None; {no_host}"), k
    assert _line(CARD.summarize_md(_card(kappa={"gpu": "L40S"})), "- κ: ").endswith(f"GPU `L40S` cc ; {no_host}")
    stated = {"gpu": {"name": "L40S", "cc": "8.9"}, "host": {"clocksource": "tsc", "clock_read_us": 0.02}}
    assert _line(CARD.summarize_md(_card(kappa=stated)), "- κ: ").endswith("GPU `L40S` cc 8.9; host clocksource `tsc` (0.02 µs/clock read)")
    assert "- serving regime" not in CARD.summarize_md(_card(workload={"path": "w.json"}))
    for realized in _left_out("realized"):
        regime = _line(CARD.summarize_md(_card(workload={"serving_profile": "burst", **realized})), "- serving regime: ")
        assert regime.endswith("realized prompt lens None, generated None, engine steps capture/control None/None (declared upper bound None), "
                               "peak concurrent requests None"), realized
    realized = {"prompt_lens": [8], "n_generated": 4, "engine_steps_capture": 5, "engine_steps_control": 6, "max_concurrent_requests": 2}
    regime = _line(CARD.summarize_md(_card(workload={"realized": realized, "expected_engine_steps": 7})), "- serving regime: ")
    assert regime.endswith("realized prompt lens [8], generated 4, engine steps capture/control 5/6 (declared upper bound 7), peak concurrent requests 2")
    none = ("- diagnostics: per-decode-step histograms equal mod T: None, instance count constant: None (decode None, prefill None), "
            "linear: None (residual None); family match `None`: None (extra None, unclassified None)")
    for d in _each({}, {"structure": None, "family_match": None, "overhead": None}, {"structure": {}, "family_match": {}, "overhead": {}}):
        md = CARD.summarize_md(_card(diagnostics=d))
        assert _line(md, "- diagnostics: ") == none and "- overhead ladder" not in md, d
    d = {"structure": {"histograms_equal_mod_T": True, "count_constant": False, "decode_instances": 4, "prefill_instances": 9, "linear": True,
                       "linear_residual": 0},
         "family_match": {"family": "Serve_v4", "status": "match", "n_extra": 1, "unclassified": 2}}
    assert _line(CARD.summarize_md(_card(diagnostics=d)), "- diagnostics: ") == (
        "- diagnostics: per-decode-step histograms equal mod T: True, instance count constant: False (decode 4, prefill 9), "
        "linear: True (residual 0); family match `Serve_v4`: match (extra 1, unclassified 2)")
    for rungs in _left_out("rungs"):
        ladder = _line(CARD.summarize_md(_card(diagnostics={"overhead": {"status": "ok", **rungs}})), "- overhead ladder")
        assert ladder.endswith("minus clock): ; host clock read None µs (None)"), rungs
    overhead = {"status": "ok", "host": {"clock_read_us": 0.02, "clocksource": "tsc"},
                "rungs": {"bare": {"step_wall_ms": 10, "x_baseline": 1.0}, "full": {"step_wall_ms": 30, "x_baseline": 3.0, "overhead_per_event_us_minus_clock": 7}}}
    ladder = _line(CARD.summarize_md(_card(diagnostics={"overhead": overhead})), "- overhead ladder")
    assert ladder.endswith("minus clock): bare: 10 (1.0×); full: 30 (3.0×, 7 µs); host clock read 0.02 µs (tsc)")


def test_the_cards_summary_prints_a_gate_it_has_no_result_for_as_missing():
    for gates in _each({}, {g: None for g in GATES}, {g: {} for g in GATES}):
        md = CARD.summarize_md(_card(gates=gates))
        for g in GATES:
            assert f"| {g} | {G.GATE_NAMES[g]} | **missing** |  |" in md, (g, gates)
        assert "| G8 | VU partition (structural) | n/a | record predates G8 |" in md
    md = CARD.summarize_md(_card(gates={"G3": {"name": "mine", "status": "fail", "reason": "a|b"}, "G8": {"status": "pass"}}))
    assert "| G3 | mine | **fail** | a/b |" in md and "| G8 | VU partition (structural) | **pass** |  |" in md
