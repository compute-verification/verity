"""The Commit records the commit rules' tests build on (`_base`, `_both`, `test_absent_commit_fields`)."""
from __future__ import annotations



from verity_vllm.check.cublas_route_pin import stamp as route_pin_stamp
from verity_vllm.definitions.registry import bf16_gemm_check as BC

RUN_ROOT = "71f69377" + "ab" * 28
SEED_ROOT = int.from_bytes(bytes.fromhex(RUN_ROOT)[:8], "big")
ROOT_FORM = "run root (first 8 bytes, after commitment)"
CLASSES = {"input_token_ids": {"how": "bound-as-leaf", "tensors": 4}, "instance_outputs_native": {"how": "bound-as-leaf", "tensors": 400},
           "sampled_token_ids": {"how": "bound-as-leaf", "tensors": 4}, "weights": {"how": "registered-input", "tensors": 100},
           "fa2_hidden_m1_stream": {"how": "bound-as-leaf", "tensors": 64}}


def _checks(faults: list | None = None, **kw) -> dict:
    """`population.query_checks` as the replay's coverage writes it over a reconciled Q(P), with `kw` over it."""
    rule = {"applied": True, "faults": faults or [], **{k: {"n": 0, "first": []} for k in ("evaluated_past_stop", "manifest_cut_before_classification")}}
    return {"ok": True, **{k: {"n": 0, "first": []} for k in ("vus_outside_query", "identities_without_rows", "rows_disagree")},
            "ranks": {"replay": [0], "manifest": [0], "equal": True}, "accounted": True, "unstated_families": [], "executed_prefix_rule": rule} | kw


def _sr(**kw) -> dict:
    return {"result": True, "grade": "complete", "why": None, "strata_unevaluated": [],
            "sample": {"per_stratum": 2, "seed": SEED_ROOT, "seed_source": ROOT_FORM, "picked": 407, "evaluated": 407, "equal": 407, "mismatch_n": 0, "not_evaluated_n": 0},
            "population": {"vus": 57678, "not_evaluable": {}, "not_evaluable_by_family": {},
                           "query_total": {"identities": 57678}, "query_checks": _checks()}, **kw}


def _epr(ok: bool = True) -> dict:
    return {"ok": ok, "lag": 1, "eos_of_record": [2], "served_known": True, "any_eos_stop": False, "problems": [] if ok else ["r0: served 9 tokens over the declared cap 8"],
            "requests": {"r0": {"served_output_len": 8, "executed_prefix_len": 8, "cap": 8, "cap_exhausted": True, "eos_stop": False}}}


def _link(ok: bool = True, required: bool = False, problems: list | None = None) -> dict:
    d = {"ok": ok, "required": required, "lag_attested": ok, "problems": problems or [],
         "global_match": {"path": "/x/global_match.json", "sha256": "9a" * 32, "verdict": "PASS", "g6_status": "PASS", "requests_linked": ["r0"]}}
    if ok:
        d["tokens_json"] = {"path": "/x/match/capture/tokens.json", "sha256": "ef" * 32, "served_by_request": {"r0": 8}, "tokens_equal": {"r0": True}}
    return d


def _base() -> dict:
    # the record's shape: executed_prefix.py's outputs live under binding.coverage (commit_delta) and the facts of record under
    # sampled_replay.executed_prefix_of_record.facts_of_record (sampled_replay executed_prefix=)
    sr = _sr()
    sr["executed_prefix_of_record"] = {"facts_of_record": {"r0": {"executed_prefix_len": 8, "cap": 8, "linked": True, "account_missing": []}}}
    return {"extra": {"expected_tensors": 572, "committed_tensors": 572, "errors_n": 0,
                      "coverage": {"ok": True, "checked": 572, "missing_n": 0, "committed_extras_n": 0, "duplicate_names_n": 0, "manifest_digest": "cd" * 32, "checked_by_family": {"instance_outputs": 400}},
                      "identity_check": {"ok": True, "steps_checked": 4, "violations": [], "violations_n": 0,
                                         "manifest": {"modules": 10, "attention_layers": 2, "fa2_hidden_per_layer": 1, "sampler_ids": 1, "moe_layers": 0, "tp_sites": 0}}},
            "openings": {"n": 64, "verified": 64}, "run_root": RUN_ROOT, "classes": CLASSES,
            "binding": {"program_digest": "ab" * 32, "context": {"ok": True},
                        "coverage": {"ok": True, "expected_n": 572, "bound_n": 572, "missing_n": 0, "ambiguous_n": 0,
                                     "executed_prefix_of_record": _epr(), "match_account_link": _link(),
                                     "rows_past_the_stop": {"ok": True, "entries_n": 0, "requests": {}, "steps_past_by_request": {}, "padding_entries_n": 0, "padded_record": False, "error": None}},
                        "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": 1}},
            "value_correspondence": {"method": "sampled-exact-replay(complete) + boundary-linkage", "result": True},
            "sampled_replay": sr,
            "boundary_linkage": {"result": True, "pairs": 32, "linked": 32, "unlinked_n": 0},
            "weights_pin": {"result": True, "compared": 163, "mismatched": [], "missing": [], "program_digest": "ab" * 32}}


#: the route pin's Definitions a GLM-5.3 TP1 eager Program binds: the MoE router past the cuteDSL tier, a split-K linear
ROUTED = {"CublasSplitKGemmF32Out_v1": 6, "CublasSplitKGemm_v1": 4}
#: what `engine_facts_of_record` measures in a process under the pin's scan's conditions
PIN_FACTS = {**BC.ROUTE_CONDITIONS, "cublaslt_version": 120901, "cublas_library": "/venv/nvidia/cublas/lib/libcublas.so.12"}


def _routed(engine: dict | None, *, stamped: bool = True) -> dict:
    """A TP1 eager row (rank 0) whose Program binds `ROUTED`, its replay's `engine_of_record` `engine` (None: not recorded), stamped as
    `pipeline.commit` stamps it before its verdict."""
    c = _base()
    c["sampled_replay"]["population"]["by_family"] = {"RMSNorm_v1": 8, **ROUTED}
    if engine is not None:
        c["sampled_replay"]["engine_of_record"] = engine
    if stamped:
        route_pin_stamp(c)
    return c


def _decl(classes: dict) -> dict:
    return {"classes": [{"class": k, **v} for k, v in classes.items()]}


# ---------------------------------------------------------------------------------------------------------------- (1) schema


# ---------------------------------------------------------------------------------------------------------------- (2) mapping


# ---------------------------------------------------------------------------------------------------------------- (3) equivalence

def _mutations():
    b = _base
    yield "tokens differ", b(), False
    c = b()
    c["extra"]["committed_tensors"] = 571
    yield "reference integrity", c, True
    c = b()
    c["extra"]["errors_n"] = 2
    yield "collector errors", c, True
    c = b()
    c["extra"]["collector_failure"] = "worker failed on windowed step 0: transient host budget"
    yield "collector failure", c, True
    c = b()
    c["openings"] = {"n": 60, "verified": 60}
    yield "openings < min", c, True
    c = b()
    c["openings"] = {"n": 64, "verified": 63}
    yield "an opening failed", c, True
    c = b()
    c["binding"]["context"] = {"ok": False, "why": "root differs"}
    yield "binding context", c, True
    c = b()
    c["binding"]["coverage"] = {"ok": False, "missing_n": 3, "expected_n": 572, "bound_n": 569, "missing": [{"step": 1, "op_path": "layers.0.mlp", "output_member": "y", "invocation": 0, "request_id": "r0", "rank": 0}]}
    yield "identity coverage", c, True
    c = b()
    c["extra"]["coverage"] = {"ok": False, "checked": 500, "missing_n": 72, "missing_by_family": {"fa2_hidden_m1_stream": 72}, "families_missing_entirely": ["fa2_hidden_m1_stream"],
                                       "missing": [{"step": 0, "op_path": "layers.0.attn", "output_member": "m1", "request_id": "r0"}], "manifest_digest": "cd" * 32}
    yield "manifest coverage", c, True
    c = b()
    c["extra"]["identity_check"] = {"ok": False, "steps_checked": 4, "violations_n": 2, "manifest": {"sampler_ids": 1},
                                             "violations": [{"step": 1, "fa2_layers_bad": {"0": 48}, "sampler_ids": 1}, {"step": 2, "fa2_layers_bad": {"0": 42}, "sampler_ids": 1}]}
    yield "X-03 identity layout", c, True
    c = b()
    c["weights_pin"] = {"result": False, "why": "1 mismatched", "mismatched": ["model.layers.0.q_proj"]}
    yield "checkpoint binding", c, True
    c = b()
    c["boundary_linkage"] = {"result": False, "why": "1 unlinked", "unlinked_n": 1}
    yield "boundary linkage", c, True
    c = b()
    c["sampled_replay"]["result"] = False
    c["sampled_replay"]["why"] = "a VU differs"
    yield "replay mismatch", c, True
    c = b()
    c["sampled_replay"]["sample"]["seed"] = 12345
    c["oracle_compare"] = {"result": True, "complete": False}
    yield "prover-chosen seed, no form (B)", c, True
    c = b()
    c["sampled_replay"]["population"]["query_checks"] = _checks(ok=False, vus_outside_query={"n": 1, "first": [{"request_id": "r0", "step": 1, "op_path": "x", "member": "y"}]})
    yield "population scope", c, True
    c = b()
    del c["weights_pin"]
    yield "weights unpinned", c, True
    c = b()
    c["classes"] = {k: v for k, v in CLASSES.items() if k != "weights"}
    yield "required class undeclared", c, True
    c = b()
    c["classes"] = dict(CLASSES, weights={"how": "registered-input", "tensors": 0})
    yield "declared leaf class bound 0", c, True
    c = b()
    c["openings_after_release"] = {"n": 64, "verified": 60, "source": "host"}
    yield "openings after release", c, True
    c = b()
    c["oracle_compare"] = {"result": False, "mismatch_n": 3}
    yield "oracle mismatch", c, True
    yield "cuBLAS route pin", _routed(None), True


# ---------------------------------------------------------------------------------------------------------------- (8) cuBLAS route pin


# ---------------------------------------------------------------------------------------------------------------- (4) missing input


def _commit_link(ok: bool = True, problems: list | None = None) -> dict:
    return {"ok": ok, "required": True, "lag_attested": ok, "problems": problems or [], "tokens_committed": {"r0": ok},
            "record": {"path": "/x/commit/tokens_p0.json", "sha256": "12" * 32, "schema": "verity-vllm/commit-tokens/v1", "requests_n": 1}}


# ---------------------------------------------------------------------------------------------------------------- (5) X-03


# ---------------------------------------------------------------------------------------------------------------- (6) from_record


# ---------------------------------------------------------------------------------------------------------------- (7) leaks


# ---------------------------------------------------------------------------------------------------------------- (8) correspondence digest


# ---------------------------------------------------------------------------------------------------------------- (9) property citations

