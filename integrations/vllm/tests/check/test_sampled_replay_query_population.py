"""The population of record is Q(P) -- the required manifest: Q_module(P), the module-boundary
population over the Program whose families the canonical VU query selects (the record states the unit and names the
implemented Q) -- never the collector's expected population and never the replay's own enumeration of the Program rows.  `sampled_replay` reconciles its reach against
Q(P) IDENTITY BY IDENTITY and carries the result on the record:

  * `population.query_total`      |Q(P)| and its families (with the manifest / Program / query digests)
  * `population.sampled_strata`   the identities the exact replay REACHES (>= 1 evaluable VU) and EVALUATED, as a share of Q(P)
  * `population.uncovered_by_name` every required identity the exact replay has no evaluable VU for, grouped (family, spec family, class)
                                  with the mechanism of record that covers it -- NOT_YET (state-linked Attention / no-evaluator RoPE),
                                  plane-fed producers, interior-composed VUs, and the query families with no VU row (weights, ids, streams)
  * `population.query_checks`     the fail-closed reconciliation (a VU outside the query, a required identity no row produces, a block
                                  height that disagrees, a rank set that disagrees, a family with no stated mechanism) -> `ok`

and `commit_verdict` states the scope on EVERY PASS and FAILs by name when the reconciliation fails."""
from __future__ import annotations

import copy
import json

import pytest

from verity_vllm.check.c2_rules import _query_population_scope
from verity.ml.definitions.registry import bf16_gemm_check as BC
from verity_vllm.check.commit_pass import commit_verdict
from verity_vllm.check.replay import coverage as COV, driver as SR, index as IX, population as POP
from tests.check import test_sampled_replay as T
from tests.check.test_sampled_replay import moe_world, world  # noqa: F401  -- fixtures


def _programs(w):
    return {rid: w.inst for rid in w.prompts}


def _weights(w):
    def read(field, lo, hi):
        flo, _fhi = w.woff[field]
        return w.weights[flo + lo: flo + hi]
    return read


def _manifest_for(w) -> dict:
    """The world's Q(P): every request shares one Program shape (requests arrive together at engine step 0)."""
    return T.v2_manifest(w.inst, tuple(sorted(w.prompts)))


def test_the_honest_world_reconciles_against_its_manifest_identity_by_identity(world):
    man = _manifest_for(world)
    bmap, committed = world.bmap_and_reader()
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=2, seed=3, workers=0, manifest=man)
    assert rep["result"] is True and rep["grade"] == "complete", rep["why"]
    pop = rep["population"]
    assert pop["vus"] > 0 and pop["authority"].startswith("Q(P)")                        # the legacy fields stand beside the reconciliation
    qt, ss, unc, chk = pop["query_total"], pop["sampled_strata"], pop["uncovered_by_name"], pop["query_checks"]
    # query_total IS the manifest's population, with its digests
    assert qt["identities"] == man["populations"]["identities"] == len(man["identities"])
    assert qt["by_family"] == dict(sorted(man["populations"]["by_family"].items()))
    assert qt["manifest_digest"] == man["manifest_digest"] and qt["program_digest"] == man["program_digest"]
    assert qt["source"].startswith("required manifest: Q_module_body_v1(P)")
    # the query of record BY ID with its definition digest: on the header, on the record, on the scope line
    assert man["query"]["query_id"] == "Q_module_body_v1" and man["query"]["boundary"] == "module-body"
    assert qt["query_id"] == "Q_module_body_v1" and qt["query_id_digest"] == man["query"]["query_id_digest"] and len(qt["query_id_digest"]) == 64
    assert rep["population"]["scope"].startswith("Q_module_body_v1(P) = ")
    # every family of this world has an evaluator: the reach is every instance output + every sampled id, and the sample evaluated some
    n_rows_fams = sum(v for k, v in qt["by_family"].items() if k in COV.QUERY_FAMILIES_WITH_ROWS)
    assert ss["identities_reached"] == n_rows_fams == qt["by_family"]["instance_outputs"] + qt["by_family"]["sampled_token_ids"]
    assert 0 < ss["identities_under_evaluated_module"] <= ss["identities_reached"]
    assert ss["by_family_reached"] == {"instance_outputs": qt["by_family"]["instance_outputs"], "sampled_token_ids": qt["by_family"]["sampled_token_ids"]}
    assert ss["strata"] == len(rep["strata"]) and ss["strata_evaluated"] == len(rep["strata"])
    assert ss["share_of_query"] == pytest.approx(n_rows_fams / qt["identities"], abs=1e-4)
    # the families with no VU row are uncovered BY NAME with their mechanism of record -- nothing is silently "covered"
    assert [(u["family"], u["class"], u["identities"]) for u in unc] == [("input_token_ids", "mechanism", qt["by_family"]["input_token_ids"]),
                                                                        ("weights", "mechanism", qt["by_family"]["weights"])]
    assert all("linkage" in u["mechanism"] or "pin" in u["mechanism"] for u in unc)
    assert pop["uncovered_identities"] == qt["by_family"]["input_token_ids"] + qt["by_family"]["weights"]
    # the reconciliation: reached + uncovered == |Q(P)|, no VU outside the query, no required identity without a row, heights agree
    assert chk["ok"] is True and chk["accounted"] is True, json.dumps(chk, indent=1)
    assert chk["vus_outside_query"]["n"] == 0 and chk["identities_without_rows"]["n"] == 0 and chk["rows_disagree"]["n"] == 0
    assert chk["rows_annotation_inconsistent"]["n"] == 0 and chk["ranks"] == {"replay": [0], "manifest": [0], "equal": True}
    assert pop["scope"].startswith(f"Q_module_body_v1(P) = {qt['identities']} identities") and "uncovered by name: input_token_ids" in pop["scope"]
    # the verdict states the scope on the PASS
    note, fail = _query_population_scope(rep)
    assert fail is None and note.startswith("C2 population scope (query of record Q_module_body_v1): Q_module_body_v1(P) = ") and "exact replay reaches" in note


def test_the_record_carries_the_engine_configuration_the_evaluators_were_conditioned_on_measured_never_declared(world, monkeypatch):
    """VLLM_BATCH_INVARIANT (and the split-K workspace configs it pins) as a named fact beside the evaluator strings: the
    Gemm_v2 Hopper pin holds only under batch-invariant mode.  Measured from the replay's process; a fact the process cannot
    see is 'unavailable: <why>' by name -- never a declared default."""
    man = _manifest_for(world)
    bmap, committed = world.bmap_and_reader()
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.delenv("CUBLASLT_WORKSPACE_SIZE", raising=False)
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=1, seed=3, workers=0, manifest=man)
    eng = rep["engine_of_record"]
    assert eng["VLLM_BATCH_INVARIANT"] == "1" and eng["CUBLAS_WORKSPACE_CONFIG"] == ":16:8" and eng["CUBLASLT_WORKSPACE_SIZE"] is None
    for k in ("vllm_batch_invariant_mode", "torch_allow_bf16_reduced_precision_reduction", "compute_capability"):
        v = eng[k]
        assert isinstance(v, (bool, list)) or (isinstance(v, str) and v.startswith("unavailable: ")), (k, v)    # measured or named-unavailable
    # the preferred BLAS library is measured as the ENUM NAME (`torch.backends.cuda.preferred_blas_library()` -> 'Cublas' /
    # 'Cublaslt' / 'Ck' on a torch pod), a non-empty str; without torch it is 'unavailable: <why>' by name -- the domain is per key
    v = eng["torch_preferred_blas_library"]
    assert isinstance(v, str) and v and (v.startswith("unavailable: ") or v[0].isupper()), ("torch_preferred_blas_library", v)
    assert "VLLM_BATCH_INVARIANT=1" in eng["rule"] and "never declared" in eng["rule"]
    assert set(eng) == {"VLLM_BATCH_INVARIANT", "CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE", "vllm_batch_invariant_mode",
                        "torch_allow_bf16_reduced_precision_reduction", "torch_preferred_blas_library", "compute_capability", "cublas", "rule"}
    # what decides cuBLAS's routes, each fact the route pin's conditions name, measured or named-unavailable, the workspace variables the record's
    cub = eng["cublas"]
    assert set(cub) == set(BC.ROUTE_CONDITIONS) | {"cublaslt_version", "cublas_library"}
    assert cub["env"] == {"CUBLAS_WORKSPACE_CONFIG": ":16:8", "CUBLASLT_WORKSPACE_SIZE": None}
    assert all(isinstance(v, (int, list)) or (isinstance(v, str) and v) for k, v in cub.items() if k != "env"), cub
    # not set -> None by name, never "1" by default
    monkeypatch.delenv("VLLM_BATCH_INVARIANT")
    assert SR.engine_facts_of_record()["VLLM_BATCH_INVARIANT"] is None


def test_the_cublas_version_is_the_mapped_librarys_or_unavailable_by_name(tmp_path):
    """The version is read from the one libcublas the process maps (never the linker's search order): none mapped, two mapped or one
    that does not load is 'unavailable: <why>'."""
    line = "7f0000000000-7f0000001000 r-xp 00000000 08:01 42 {}\n"
    for files, why in (([], "0 libcublas.so* mapped []"),
                       (["/a/libcublas.so.12", "/b/libcublas.so.13", "/a/libcublasLt.so.12"], "2 libcublas.so* mapped ['/a/libcublas.so.12', '/b/libcublas.so.13']")):
        maps = tmp_path / "maps"
        maps.write_text("7f0000002000-7f0000003000 rw-p 00000000 00:00 0\n" + "".join(line.format(f) for f in files))
        got = SR.cublas_versions(str(maps))
        assert got["cublas_version"] == got["cublas_library"] == f"unavailable: LookupError: {why}"
        assert got["cublaslt_version"].startswith("unavailable: ")
    maps.write_text(line.format(tmp_path / "libcublas.so.12"))
    assert SR.cublas_versions(str(maps))["cublas_version"].startswith("unavailable: OSError: ")


def test_a_required_identity_no_row_produces_is_named_and_fails_the_reconciliation_and_the_verdict(world):
    man = _manifest_for(world)
    bad = copy.deepcopy(man)
    ghost = dict(next(r for r in man["identities"] if r["family"] == "instance_outputs"), op_path="model.layers.7.mlp.down_proj", output_member="0")
    bad["identities"].append(ghost)                                              # Q(P) requires something the Program never produces
    bmap, committed = world.bmap_and_reader()
    # a post-commitment seed of record (root form: seed == run_root[:8]; the seed rule grades the label as a fact where it is on the tree)
    run_root = "71f69377" + "ab" * 28
    seed, seed_src = int.from_bytes(bytes.fromhex(run_root)[:8], "big"), "run root (first 8 bytes, after commitment)"
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=1, seed=seed, seed_source=seed_src, workers=0, manifest=bad)
    chk = rep["population"]["query_checks"]
    assert chk["ok"] is False and chk["identities_without_rows"]["n"] == 1
    assert chk["identities_without_rows"]["first"][0]["op_path"] == "model.layers.7.mlp.down_proj"
    assert chk["accounted"] is False                                             # the ghost is neither reached nor uncovered by name
    note, fail = _query_population_scope(rep)
    assert note is None and fail.startswith("C2 population does not reconcile with Q(P): ") and "identities_without_rows x1" in fail
    # through the verdict of record: a row-procedure record (binding block) with this replay FAILs by that name, whatever else holds
    rec = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64}, "run_root": run_root,
           "binding": {"context": {"ok": True}, "coverage": {"ok": True, "expected_n": 1, "bound_n": 1, "missing_n": 0}, "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": 2}},
           "value_correspondence": {"method": "sampled-exact-replay(complete) + boundary-linkage", "result": True},
           "boundary_linkage": {"result": True, "pairs": 6, "linked": 6, "unlinked_n": 0},
           "weights_pin": {"result": True, "compared": 5, "mismatched": [], "missing": []}, "sampled_replay": rep}
    ok, why = commit_verdict(rec, True, min_openings=64)
    assert ok is False and "does not reconcile with Q(P)" in why, why
    # the honest manifest on the same record: PASS carrying the scope
    rep_ok = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=1, seed=seed, seed_source=seed_src, workers=0, manifest=man)
    ok, why = commit_verdict(dict(rec, sampled_replay=rep_ok), True, min_openings=64)
    assert ok is True and "C2 population scope (query of record Q_module_body_v1): Q_module_body_v1(P) = " in why, why


def test_a_vu_whose_identity_the_query_does_not_require_is_outside_the_query_never_added_to_the_reach(world):
    """The replay does not enlarge Q(P): a Program row addressing an identity the manifest does not carry is a named finding (the
    replay's enumeration and the query disagree on what the Program produces), and the identity is not counted as reached."""
    man = _manifest_for(world)
    narrowed = copy.deepcopy(man)
    drop = next(r for r in narrowed["identities"] if r["family"] == "instance_outputs" and r["op_path"].endswith("down_proj") and r["step"] == 1 and r["request_id"] == "r1")
    narrowed["identities"].remove(drop)
    narrowed["populations"]["identities"] -= 1
    bmap, committed = world.bmap_and_reader()
    full = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=1, seed=3, workers=0, manifest=man)["population"]
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=1, seed=3, workers=0, manifest=narrowed)["population"]
    chk = rep["query_checks"]
    assert chk["ok"] is False and chk["vus_outside_query"]["n"] == 1
    assert chk["vus_outside_query"]["first"][0] == {"request_id": "r1", "step": 1, "op_path": drop["op_path"], "member": drop["output_member"], "spec": chk["vus_outside_query"]["first"][0]["spec"], "class": "vu"}
    assert rep["sampled_strata"]["identities_reached"] == full["sampled_strata"]["identities_reached"] - 1
    assert rep["query_total"]["identities"] == full["query_total"]["identities"] - 1 and chk["accounted"] is True


def test_without_a_manifest_the_record_says_the_population_is_not_reconciled(world):
    bmap, committed = world.bmap_and_reader()
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=1, seed=3, workers=0)
    pop = rep["population"]
    assert pop["query_total"]["source"].startswith("NO required manifest supplied") and pop["query_total"]["manifest_digest"] is None
    assert pop["query_checks"]["ok"] is None and pop["query_checks"]["accounted"] is None
    # the Program rows' own addresses (the same members_for rule) are what is counted -- and named as NOT a Q(P) statement
    assert set(pop["query_total"]["by_family"]) == {"instance_outputs", "sampled_token_ids"}
    assert pop["sampled_strata"]["identities_reached"] == pop["query_total"]["identities"]
    note, fail = _query_population_scope(rep)
    assert fail is None and "NOT reconciled against Q(P)" in note


def test_a_replay_record_predating_the_fields_is_graded_as_before_without_a_scope_note():
    assert _query_population_scope({"result": True, "grade": "complete", "population": {"vus": 10, "not_evaluable": {}}}) == (None, None)
    assert _query_population_scope(None) == (None, None)


def test_moe_plane_identities_reconcile_at_their_planes(moe_world):
    """The fused-MoE block rows are addressed at their PLANES (`moe/L<k>` / plane) -- the manifest's moe_block_stream
    identities.  Every plane identity is either reached (an evaluator exists) or uncovered by name (plane-fed producer without one)."""
    man = _manifest_for(moe_world)
    assert man["populations"]["by_family"]["moe_block_stream"] > 0
    bmap, committed = moe_world.bmap_and_reader()
    rep = SR.sampled_replay(_programs(moe_world), bmap, committed, _weights(moe_world), per_stratum=1, seed=3, workers=0, manifest=man)
    pop = rep["population"]
    chk = pop["query_checks"]
    assert chk["ok"] is True and chk["accounted"] is True, json.dumps(chk, indent=1)
    n_planes = man["populations"]["by_family"]["moe_block_stream"]
    reached = pop["sampled_strata"]["by_family_reached"].get("moe_block_stream", 0)
    uncovered = sum(u["identities"] for u in pop["uncovered_by_name"] if u["family"] == "moe_block_stream")
    assert reached + uncovered == n_planes and reached > 0
    for u in pop["uncovered_by_name"]:
        assert u["mechanism"] and u["class"] != "unstated"
        if u["family"] == "moe_block_stream":
            assert u["class"] == "plane-fed" and u["spec"].startswith("Moe")


def test_a_required_family_without_a_stated_mechanism_fails_closed(world):
    man = _manifest_for(world)
    odd = copy.deepcopy(man)
    odd["identities"].append(dict(man["identities"][0], family="future_family", step="*", request_id="*"))
    pi = IX.ProgramIndex(world.inst)
    addresses: dict = {}
    for rid in sorted(world.prompts):
        POP.population(pi, rid, addresses=addresses)
    q = COV.query_population(odd, addresses, set(), ranks=[0])
    assert q["query_checks"]["ok"] is False and q["query_checks"]["unstated_families"] == ["future_family"]
    u = next(u for u in q["uncovered_by_name"] if u["family"] == "future_family")
    assert u["class"] == "unstated" and u["mechanism"].startswith("UNSTATED")
    _note, fail = _query_population_scope({"population": {"query_total": q["query_total"], "query_checks": q["query_checks"], "scope": q["scope"]}})
    assert fail is not None and "no stated mechanism ['future_family']" in fail


def test_f_moe_r17_05_batch_level_planes_are_classified_per_listed_request_and_excluded_only_when_every_request_is_past_its_stop(world):
    """A BATCH-LEVEL identity (request_id '*', `requests` list, one ENGINE step -- the fa2 hidden plane) is not
    executed only when EVERY listed request is past its own stop at that engine step (nothing was committed there); a plane some listed
    request still executes EXISTS (smaller composition) and stays REQUIRED -- its per-request slots past the stop are counted beside it
    (`batch_level[fam].request_slots_not_executed`, `by_request[rid].batch_level_slots_not_executed`) for the Commit's per-request
    reconciliation.  Arrival enters here only: a listed request's request-relative step = engine step - its arrival."""
    man = _manifest_for(world)
    bmap, committed = world.bmap_and_reader()
    cap = IX.ProgramIndex(world.inst).n_steps
    fam = "fa2_hidden_m1_stream"
    planes = [{"family": fam, "program_digest": man["identities"][0]["program_digest"], "request_id": "*", "step": "*", "invocation": 0,
               "op_path": "model.layers.0.self_attn.attn", "output_member": "fa2.m1/L0", "element_range": [0, 64], "rank": 0,
               "requests": ["r0", "r1"], "engine_step": e, "dtype": "u32"} for e in range(cap)]
    man2 = dict(man, identities=list(man["identities"]) + planes)
    ok = dict(linked=True, lag_attested=True, a_r_source="match/capture/tokens.json served_by_request sha256 0000")
    # (i) r0 stops early (n_exec = cap-1), r1 serves its cap: every plane has r1 executing -> none excluded, ONE partially executed (step cap-1)
    ep = {"r0": POP.executed_prefix_of_record("r0", a_r=cap - 2, lag=1, cap=cap, eos_declared=True, served_len_commit=cap - 2, **ok),
          "r1": POP.executed_prefix_of_record("r1", a_r=cap, lag=1, cap=cap, served_len_commit=cap, **ok)}
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=2, seed=3, workers=0, manifest=man2, executed_prefix=ep)
    pop = rep["population"]
    ne = pop["not_executed_by_rule"]
    assert pop["query_total"]["by_family"][fam] == cap and ne["kind_by_family"][fam] == "batch-level"
    assert ne["batch_level"] == {fam: {"identities": cap, "identities_not_executed": 0, "identities_partially_executed": 1,
                                       "request_slots": 2 * cap, "request_slots_not_executed": 1, "cells_not_executed": 1}}
    assert ne["totals"]["by_family"][fam] == {"N_manifest": cap, "N_required": cap, "N_excluded": 0,
                                              "plane_cells_not_executed": {"identities": 0, "cells": 1}} and fam not in ne["by_family"]
    assert "plane_cells_not_executed" not in ne["totals"]["by_family"]["instance_outputs"]
    assert ne["by_request"]["r0"]["batch_level_slots_not_executed"] == 1 and ne["by_request"]["r1"]["batch_level_slots_not_executed"] == 0
    unc = next(u for u in pop["uncovered_by_name"] if u["family"] == fam)
    assert unc["identities"] == cap and unc["class"] == "mechanism"
    chk = pop["query_checks"]
    assert chk["ok"] is True and chk["accounted"] is True
    assert pop["sampled_strata"]["identities_reached"] + pop["uncovered_identities"] + ne["identities"] == pop["query_total"]["identities"]
    # (ii) BOTH stop before the cap (r1 at n_exec = cap-1 too): the plane at engine step cap-1 has no executing request -> excluded by name
    ep2 = dict(ep, r1=POP.executed_prefix_of_record("r1", a_r=cap - 2, lag=1, cap=cap, eos_declared=True, served_len_commit=cap - 2, **ok))
    rep2 = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=2, seed=3, workers=0, manifest=man2, executed_prefix=ep2)
    pop2 = rep2["population"]
    ne2 = pop2["not_executed_by_rule"]
    assert ne2["batch_level"][fam] == {"identities": cap, "identities_not_executed": 1, "identities_partially_executed": 0,
                                       "request_slots": 2 * cap, "request_slots_not_executed": 2, "cells_not_executed": 2}
    assert ne2["totals"]["by_family"][fam] == {"N_manifest": cap, "N_required": cap - 1, "N_excluded": 1,
                                               "plane_cells_not_executed": {"identities": 1, "cells": 2}} and ne2["by_family"][fam] == 1
    assert next(u for u in pop2["uncovered_by_name"] if u["family"] == fam)["identities"] == cap - 1
    assert pop2["query_checks"]["ok"] is True and pop2["query_checks"]["accounted"] is True
    assert pop2["sampled_strata"]["identities_reached"] + pop2["uncovered_identities"] + ne2["identities"] == pop2["query_total"]["identities"]
    # (iii) arrival: r1 arrived at engine step 1 with the same early stop -> at engine step cap-1 its request-relative step is cap-2 < n_exec:
    # it still executes there, the plane is NOT excluded; the plane at engine step cap (none here) would be.  Only the batch-level arithmetic
    # reads the arrival -- the per-request identities carry their own request-relative `step`.
    ep3 = dict(ep, r1=POP.executed_prefix_of_record("r1", a_r=cap - 2, lag=1, cap=cap, arrive=1, eos_declared=True, served_len_commit=cap - 2, **ok))
    rep3 = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=2, seed=3, workers=0, manifest=man2, executed_prefix=ep3)
    ne3 = rep3["population"]["not_executed_by_rule"]
    assert ne3["batch_level"][fam]["identities_not_executed"] == 0 and ne3["batch_level"][fam]["request_slots_not_executed"] == 1
    assert ne3["by_request"]["r1"]["batch_level_slots_not_executed"] == 0 and ne3["by_request"]["r0"]["batch_level_slots_not_executed"] == 1


def test_r17_26_the_executed_prefix_classifies_not_executed_identities_by_name_and_never_shrinks_the_query(world):
    """A request that stopped at a declared EOS executed n_exec = min(a_r + LAG, cap) steps: the manifest of
    record stays the cap-length population (same query_total, same digests); its identities at request-relative steps >= n_exec are the
    NAMED class not-executed(a_r, LAG) -- a third bucket beside reached / uncovered, out of the sampling pool (never a stratum that
    grades 'not evaluated'), on the scope line; reached + uncovered + not-executed == |Q(P)|.  A prefix whose early stop is not a declared
    EOS is NOT classified and fails the checks by name."""
    man = _manifest_for(world)
    bmap, committed = world.bmap_and_reader()
    base = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=2, seed=3, workers=0, manifest=man)
    cap = IX.ProgramIndex(world.inst).n_steps
    assert cap >= 3
    # the (a)(b) facts, asserted by the Commit; a_r_source REQUIRED: where a_r was read, never a default label
    ok = dict(linked=True, lag_attested=True, a_r_source="match/capture/tokens.json served_by_request sha256 0000")
    ep = {"r0": POP.executed_prefix_of_record("r0", a_r=cap - 2, lag=1, cap=cap, eos_declared=True, served_len_commit=cap - 2, **ok),
          "r1": POP.executed_prefix_of_record("r1", a_r=cap, lag=1, cap=cap, served_len_commit=cap, **ok)}   # r1 served its cap: nothing to classify
    assert ep["r0"]["n_exec"] == cap - 1 and ep["r0"]["steps_not_executed"] == [cap - 1, cap] and ep["r0"]["faults"] == []
    assert ep["r1"]["n_exec"] == cap and ep["r1"]["steps_not_executed"] is None and ep["r1"]["faults"] == []
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=2, seed=3, workers=0, manifest=man, executed_prefix=ep)
    pop, bpop = rep["population"], base["population"]
    # the population of record is UNCHANGED: same identities, same digests, same by_family
    assert pop["query_total"] == bpop["query_total"] and pop["vus"] == bpop["vus"]
    ne = pop["not_executed_by_rule"]
    assert ne["applied"] is True and ne["rule"] == POP.EXECUTED_PREFIX_RULE and ne["scope"] == POP.NOT_EXECUTED_SCOPE
    assert "active-reference implementation classification only" in ne["rule"] and "tau(e) = bot" in ne["rule"]   # verbatim, on the rule
    # exactly r0's identities at request-relative step cap-1 left the pool, by name: its ROWS and its named input
    # (`input_token_ids` at that step) -- the rule names identities of a request past its stop whatever their family; `weights` (step '*')
    # and r1 are untouched
    r0_last = [i for i in man["identities"] if i.get("request_id") == "r0" and i.get("step") == cap - 1]
    n_r0_last_rows = sum(1 for i in r0_last if i.get("family") in COV.QUERY_FAMILIES_WITH_ROWS)
    n_r0_last_ids = len(r0_last)
    assert n_r0_last_rows > 0 and n_r0_last_ids > n_r0_last_rows and {i["family"] for i in r0_last} - set(COV.QUERY_FAMILIES_WITH_ROWS) == {"input_token_ids"}
    assert ne["identities"] == n_r0_last_ids and ne["by_request"]["r0"]["identities"] == n_r0_last_ids
    assert ne["by_request"]["r1"]["identities"] == 0 and ne["by_request"]["r0"]["n_exec"] == cap - 1
    assert ne["vus"] > 0 and set(ne["vus_by_request"]) == {"r0"}
    assert {k: v for k, v in ne["totals"].items() if k != "by_family"} == {"N_manifest": pop["query_total"]["identities"], "N_required": pop["query_total"]["identities"] - n_r0_last_ids, "N_excluded": n_r0_last_ids}
    # the flat keys the readers wait for are the same numbers; per-family totals reconcile
    assert pop["not_executed_by_family"] == ne["by_family"] and sum(ne["by_family"].values()) == n_r0_last_ids
    assert ne["by_family"]["input_token_ids"] == n_r0_last_ids - n_r0_last_rows and "weights" not in ne["by_family"]
    assert pop["not_executed_rule"]["n"] == n_r0_last_ids and pop["not_executed_rule"]["requests"]["r0"]["step_range"] == [cap - 1, cap]
    assert pop["not_executed_rule"]["requests"]["r0"]["n"] == n_r0_last_ids and pop["not_executed_rule"]["scope"] == POP.NOT_EXECUTED_SCOPE
    bf = ne["totals"]["by_family"]
    assert set(bf) == set(pop["query_total"]["by_family"]) and all(v["N_manifest"] == v["N_required"] + v["N_excluded"] for v in bf.values())
    assert {k: v["N_excluded"] for k, v in bf.items() if v["N_excluded"]} == ne["by_family"]
    # the reach of the class over every family is stated by kind; the named input's uncovered count shrank by exactly the class
    assert ne["kind_by_family"]["input_token_ids"] == "named-input" and ne["kind_by_family"]["weights"] == "static"
    assert all(ne["kind_by_family"][f] == "row" for f in COV.QUERY_FAMILIES_WITH_ROWS if f in ne["kind_by_family"]) and ne["batch_level"] == {}
    unc_in = next(u for u in pop["uncovered_by_name"] if u["family"] == "input_token_ids")
    unc_in_base = next(u for u in bpop["uncovered_by_name"] if u["family"] == "input_token_ids")
    assert unc_in["identities"] == unc_in_base["identities"] - ne["by_family"]["input_token_ids"]
    r0 = ne["by_request"]["r0"]
    assert r0["executed_prefix"] == cap - 1 and r0["excluded_unexecuted_identities"] == n_r0_last_ids and r0["required_row_identities"] > 0
    assert r0["excluded_row_identities"] == n_r0_last_rows and r0["batch_level_slots_not_executed"] == 0
    assert r0["required_row_identities"] + n_r0_last_rows == sum(1 for i in man["identities"] if i.get("request_id") == "r0" and i.get("family") in COV.QUERY_FAMILIES_WITH_ROWS)
    # the excluded rows are 'not-executed' (never a 'last'): r0's `last` stratum is anchored to its last EXECUTED step
    # n_exec - 1 (the lag step), so r0 HAS a `last` stratum in the pool -- at step cap-2, where the cap-anchored base has a `mid`
    assert ne["strata_excluded"] and all(k.split("|")[1] == "r0" and k.split("|")[3] == "not-executed" and v["label"] == "not evaluable: unexecuted"
                                         for k, v in ne["strata_excluded"].items())
    assert sum(v["population"] for v in ne["strata_excluded"].values()) == ne["vus"]
    r0_last_strata = [k for k in rep["strata"] if k.split("|")[1] == "r0" and k.split("|")[3] == "last"]
    assert r0_last_strata, "r0's last EXECUTED step (the lag step) is its `last` stratum"
    assert any(k.split("|")[1] == "r0" and k.split("|")[3] == "last" for k in base["strata"])
    anchors = rep["strata_keys"]["phase_anchor"]
    assert anchors["r0"]["last_step"] == cap - 2 and anchors["r0"]["anchor"] == "n_exec - 1" and anchors["r0"]["lag_step"] == cap - 2
    assert anchors["r0"]["n_exec"] == cap - 1 and anchors["r0"]["cap_last_step"] == cap - 1 and "anchored there, not at the cap" in anchors["r0"]["why"]
    assert anchors["r1"]["last_step"] == cap - 1 and anchors["r1"]["anchor"] == "cap - 1" and anchors["r1"]["lag_step"] is None
    assert base["strata_keys"]["phase_anchor"]["r0"] == {"last_step": cap - 1, "anchor": "cap - 1", "cap_last_step": cap - 1, "n_exec": None,
                                                         "lag_step": None, "why": "no executed prefix"}
    assert "not-executed" not in rep["strata_keys"]["phases"]                       # the excluded phase is never a sampled stratum
    # the three buckets close over the whole manifest; the grade is over the pool (no 'strata not evaluated' from unexecuted rows)
    ss, chk = pop["sampled_strata"], pop["query_checks"]
    assert chk["ok"] is True and chk["accounted"] is True and chk["executed_prefix_rule"] == {"applied": True, "faults": [], "evaluated_past_stop": {"n": 0, "first": []},
                                                                                              "manifest_cut_before_classification": {"n": 0, "first": []}}
    # the class is an EXCLUSION over the cap-length Q(P): a manifest RESTRICTED to the executed prefix before it reaches the
    # population (moe-style cut handed in as Q(P)) is named and fails -- N_manifest never shrinks silently while the class counts 0
    cut = dict(man, identities=[i for i in man["identities"] if not (i.get("request_id") == "r0" and i.get("step") != "*" and int(i["step"]) >= cap - 1)])
    assert len(cut["identities"]) < len(man["identities"])
    rep_cut = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=2, seed=3, workers=0, manifest=cut, executed_prefix=ep)
    mc = rep_cut["population"]["query_checks"]["executed_prefix_rule"]["manifest_cut_before_classification"]
    assert mc["n"] == 1 and mc["first"][0]["request_id"] == "r0" and mc["first"][0]["identities_at_not_executed_steps"] == 0 and "restricted before" in mc["first"][0]["why"]
    assert rep_cut["population"]["query_checks"]["ok"] is False and rep_cut["population"]["not_executed_by_rule"]["identities"] == 0
    assert ss["identities_reached"] + pop["uncovered_identities"] + ne["identities"] == pop["query_total"]["identities"]
    assert ss["identities_reached"] == bpop["sampled_strata"]["identities_reached"] - n_r0_last_rows           # reach is over rows
    assert rep["result"] is True and rep["grade"] == "complete", rep["why"]
    assert f"not-executed {n_r0_last_ids} identities (r0: a_r {cap - 2} LAG 1 -> steps [{cap - 1}, {cap}))" in pop["scope"]
    # the verdict's scope note carries the class (it reads the scope line)
    note, fail = _query_population_scope(rep)
    assert fail is None and f"not-executed {n_r0_last_ids} identities" in note
    # NEGATIVE: an early stop that is not a declared EOS -> the class is not applicable: nothing classified, the checks fail by name
    bad = {"r0": POP.executed_prefix_of_record("r0", a_r=cap - 2, lag=1, cap=cap, eos_declared=False, served_len_commit=cap - 2, **ok)}
    assert bad["r0"]["faults"] and "not asserted a declared EOS id" in bad["r0"]["faults"][0]
    rep2 = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=2, seed=3, workers=0, manifest=man, executed_prefix=bad)
    p2 = rep2["population"]
    assert p2["not_executed_by_rule"]["identities"] == 0 and p2["not_executed_by_rule"]["vus"] == 0
    assert p2["query_checks"]["ok"] is False and p2["query_checks"]["executed_prefix_rule"]["faults"][0]["request_id"] == "r0"
    assert "executed-prefix FAULTS: r0:" in p2["scope"]
    assert p2["query_total"] == bpop["query_total"] and p2["sampled_strata"]["identities_reached"] == bpop["sampled_strata"]["identities_reached"]
    # the arithmetic is the rule's, not the caller's: LAG outside {0, 1} and a_r > cap are faults by name
    assert POP.executed_prefix_of_record("r0", a_r=2, lag=2, cap=cap, eos_declared=True, served_len_commit=2, **ok)["faults"][0].startswith("LAG 2 outside")
    assert "outside [0, cap" in POP.executed_prefix_of_record("r0", a_r=cap + 1, lag=1, cap=cap, served_len_commit=cap + 1, **ok)["faults"][0]
    # (a)(b)(c) + the served-length rule, each a fault BY NAME; an unasserted fact is a fault, never a default
    f = lambda **kw: POP.executed_prefix_of_record("r0", **{**dict(a_r=cap - 2, lag=1, cap=cap, eos_declared=True, served_len_commit=cap - 2), **ok, **kw})["faults"]
    assert f() == []
    assert any("not linked to the served output" in x for x in f(linked=None)) and any("(condition (a))" in x for x in f(linked=False))
    assert any("LAG rule not attested" in x for x in f(lag_attested=None)) and any("(condition (b))" in x for x in f(lag_attested=False))
    assert any("execution past the stop: 3 committed row(s)" in x for x in f(rows_past_stop=3))
    assert any(f"Commit served length {cap - 3} != record a_r {cap - 2}" in x for x in f(served_len_commit=cap - 3)) and any("!= record a_r" in x for x in f(served_len_commit=None))
    assert POP.executed_prefix_of_record("r1", a_r=cap, lag=0, cap=cap, served_len_commit=cap, **ok)["faults"] == []            # LAG 0 profile, cap stop
    # a_r_source has NO default: omitted -> TypeError (the caller cannot put 'match record' on the record by omission);
    # empty -> a fault by name
    with pytest.raises(TypeError):
        POP.executed_prefix_of_record("r1", a_r=cap, lag=0, cap=cap, served_len_commit=cap, linked=True, lag_attested=True)
    assert any("a_r_source unstated" in x for x in f(a_r_source=""))
    assert ep["r0"]["a_r_source"] == ok["a_r_source"]
    # no executed_prefix -> the class is present, not applied, empty (never silently absent)
    assert bpop["not_executed_by_rule"]["applied"] is False and bpop["not_executed_by_rule"]["identities"] == 0
    assert bpop["query_checks"]["executed_prefix_rule"]["applied"] is False and bpop["query_checks"]["ok"] is True


# ---------------------------------------------------------------------------------------------------------------------------------------
# ONE mechanism: the not-executed class DERIVED FROM r_t (Continuation_v1's recurrence over the served list),
# the arithmetic min(a_r + LAG, cap) demoted to a CHECK; which mechanism produced the class is on the record BY NAME
# ---------------------------------------------------------------------------------------------------------------------------------------

def test_f_r17b_39_the_last_phase_is_anchored_to_the_last_executed_step_and_the_lag_step_is_named(world):
    """On one record the EOS-stopped r0 (a_r 11, LAG 1, n_exec 12, cap 32) had NO `last` stratum: `last` was anchored to
    the cap step (31, not executed), its last executed step 11 -- the lag step, the executed prefix's boundary -- sat in `mid` at catch
    probability n / |mid|.  Now: `last` = n_exec - 1 per request under a fault-free executed prefix, the steps past it are 'not-executed'
    (split out, never sampled), and the anchor is named on the record with the lag step."""
    pi = IX.ProgramIndex(world.inst)
    cap = pi.n_steps
    assert cap >= 3
    # the label function itself (r17_2241's r0: cap 32, n_exec 12 -> last step 11; steps 12..31 not executed)
    assert POP._phase(0, 31) == "prefill" and POP._phase(1, 31) == "first-decode" and POP._phase(11, 31) == "mid" and POP._phase(31, 31) == "last"
    assert POP._phase(11, 11) == "last" and POP._phase(12, 11) == "not-executed" and POP._phase(31, 11) == "not-executed" and POP._phase(5, 11) == "mid"
    assert POP._phase(1, 1) == "last"                                              # n_exec 2: step 1 is the last executed step, not 'first-decode'
    # the population under an anchor: the same rows (the population is Q(P)'s, not the anchor's), the phase labels moved
    ok = dict(linked=True, lag_attested=True, a_r_source="match/capture/tokens.json served_by_request sha256 0000")
    ep = POP.executed_prefix_of_record("r0", a_r=1, lag=1, cap=cap, eos_declared=True, served_len_commit=1, **ok)
    assert ep["n_exec"] == 2 and ep["faults"] == []
    anchor = POP.phase_anchor(cap, ep)
    assert anchor == {"last_step": 1, "anchor": "n_exec - 1", "cap_last_step": cap - 1, "n_exec": 2, "lag_step": 1, "why": anchor["why"]}
    assert "lag step" in anchor["why"] and "anchored there, not at the cap" in anchor["why"]
    base, _, _ = POP.population(pi, "r0")
    moved, _, _ = POP.population(pi, "r0", last_step=anchor["last_step"])
    assert [v["i"] for v in base] == [v["i"] for v in moved]                      # the SAME rows
    by_step_base = {s: {v["phase"] for v in base if v["step"] == s} for s in range(cap)}
    by_step_moved = {s: {v["phase"] for v in moved if v["step"] == s} for s in range(cap)}
    assert by_step_base[0] <= {"prefill"} and by_step_base[1] <= {"first-decode"} and by_step_base[cap - 1] <= {"last"}
    assert by_step_moved[0] <= {"prefill"} and by_step_moved[1] <= {"last"} and all(by_step_moved[s] <= {"not-executed"} for s in range(2, cap))
    assert all(v["stratum"].split("|")[3] == v["phase"] for v in moved)          # the key carries the moved label
    pool, out = POP.split_not_executed(moved, {"r0": ep})
    assert {v["phase"] for v in out} == {"not-executed"} and "not-executed" not in {v["phase"] for v in pool}
    # no prefix / a faulted prefix / a request at its cap: the cap anchors, said so
    assert POP.phase_anchor(cap, None)["anchor"] == "cap - 1" and POP.phase_anchor(cap, None)["why"] == "no executed prefix"
    full = POP.executed_prefix_of_record("r1", a_r=cap, lag=1, cap=cap, served_len_commit=cap, **ok)
    assert POP.phase_anchor(cap, full) == {"last_step": cap - 1, "anchor": "cap - 1", "cap_last_step": cap - 1, "n_exec": cap, "lag_step": None,
                                          "why": "the request ran to its cap"}
    faulted = dict(ep, faults=["a fault"])
    assert POP.phase_anchor(cap, faulted)["anchor"] == "cap - 1" and POP.phase_anchor(cap, faulted)["n_exec"] is None
    assert "faults" in POP.phase_anchor(cap, faulted)["why"]
    # an EOS stop at a_r = cap - 1 under LAG 1: n_exec == cap so the cap anchors, but step cap - 1 IS a lag step -- the
    # flag follows the stop (a_r < cap and LAG > 0), not n_exec < cap
    edge = POP.executed_prefix_of_record("r2", a_r=cap - 1, lag=1, cap=cap, eos_declared=True, served_len_commit=cap - 1, **ok)
    assert not edge["faults"] and edge["n_exec"] == cap
    a = POP.phase_anchor(cap, edge)
    assert a["anchor"] == "cap - 1" and a["last_step"] == cap - 1 and a["n_exec"] == cap and a["lag_step"] == cap - 1 and "lag step" in a["why"]
    # ...and without LAG the same stop has no lag step; at the cap it 'ran to its cap', at a_r = cap - 1 (EOS at the last slot) the why
    # says the STOP, not 'ran to its cap'
    edge0 = POP.executed_prefix_of_record("r3", a_r=cap, lag=0, cap=cap, served_len_commit=cap, **ok)
    assert POP.phase_anchor(cap, edge0)["lag_step"] is None and POP.phase_anchor(cap, edge0)["why"] == "the request ran to its cap"
    eos0 = POP.executed_prefix_of_record("r4", a_r=cap - 1, lag=0, cap=cap, eos_declared=True, served_len_commit=cap - 1, **ok)
    assert not eos0["faults"] and eos0["n_exec"] == cap - 1                     # LAG 0: n_exec = a_r < cap -> anchored at n_exec - 1, no lag step
    a0 = POP.phase_anchor(cap, eos0)
    assert a0["anchor"] == "n_exec - 1" and a0["lag_step"] is None and "EOS step" in a0["why"] and "is the lag step" not in a0["why"]
    assert "is the lag step" in a["why"] and "is the lag step" in anchor["why"]  # ...and the LAG 1 stops say so


def _rule_like_pad(served, *, cap, A, eos):
    """A stand-in with `executed_prefix_by_rule`'s semantics (program/lifting/continuation.py): the FIRST EOS stops, the lag slot(s)
    after it are executed rows without a served token, served past the first EOS / an early stop without EOS / over the cap are problems."""
    eos = set(eos)
    a = len(served)
    problems = []
    first = next((i for i, t in enumerate(served) if t in eos), None)
    past = list(range(first + 1, a)) if first is not None else []
    if a > cap:
        problems.append(f"served {a} tokens over the declared cap {cap}")
    if past:
        problems.append(f"served past an EOS at index {first}: positions {past} (the rule stops at the FIRST EOS)")
    if a < cap and first is None:
        problems.append(f"served {a} of its cap {cap} but no served token is a declared EOS id")
    n = cap if first is None else min(first + 1 + A, cap)
    return {"ok": not problems, "executed_prefix_len": n, "eos_step": first if not problems else None, "lag_steps": list(range(first + 1, n)) if first is not None else [],
            "first_eos_index": first, "served_past_first_eos": past, "cap_exhausted": (a == cap and first is None), "eos_stop": first is not None and not past,
            "problems": problems}


def test_r3_the_not_executed_class_is_derived_from_r_t_when_the_served_list_is_supplied_and_the_arithmetic_is_the_check():
    ok = dict(linked=True, lag_attested=True, a_r_source="match/capture/tokens.json served_by_request sha256 0000")
    EOS = 2
    # early stop: served [5, 7, EOS], cap 8, LAG 1 -> r_t = 1 on rows 0..3 (the lag row 3 executed, its token never served): n_exec 4
    e = POP.executed_prefix_of_record("r0", a_r=3, lag=1, cap=8, eos_declared=True, served_len_commit=3, served=[5, 7, EOS], eos_ids=[EOS], rule=_rule_like_pad, **ok)
    assert e["faults"] == [] and e["n_exec"] == 4 and e["r3_one_mechanism"] is True and e["mechanism"].startswith("recurrence(Continuation_v1)")
    assert e["by_rule"]["executed_prefix_len"] == 4 and e["by_rule"]["eos_step"] == 2 and e["by_rule"]["lag_steps"] == [3] and e["by_rule"]["problems"] == []
    # the cap exhausted: served 8 non-EOS tokens -> every row executed, the class is empty
    e8 = POP.executed_prefix_of_record("r1", a_r=8, lag=1, cap=8, served_len_commit=8, served=[5] * 8, eos_ids=[EOS], rule=_rule_like_pad, **ok)
    assert e8["faults"] == [] and e8["n_exec"] == 8 and e8["steps_not_executed"] is None and e8["r3_one_mechanism"] is True
    # LAG 0: the EOS row is the last executed row
    e0 = POP.executed_prefix_of_record("r2", a_r=3, lag=0, cap=8, eos_declared=True, served_len_commit=3, served=[5, 7, EOS], eos_ids=[EOS], rule=_rule_like_pad, **ok)
    assert e0["faults"] == [] and e0["n_exec"] == 3 and e0["by_rule"]["lag_steps"] == []
    # the served list disagrees with the record's a_r -> a fault by name (the recurrence would run over another prefix)
    bad = POP.executed_prefix_of_record("r0", a_r=4, lag=1, cap=8, eos_declared=True, served_len_commit=4, served=[5, 7, EOS], eos_ids=[EOS], rule=_rule_like_pad, **ok)
    assert any("!= a_r 4" in f and "another prefix" in f for f in bad["faults"])
    # served PAST the first EOS: the rule's problem is the record's fault; a last-token test would have accepted [EOS, x, EOS]
    pe = POP.executed_prefix_of_record("r0", a_r=3, lag=1, cap=8, eos_declared=True, served_len_commit=3, served=[EOS, 5, EOS], eos_ids=[EOS], rule=_rule_like_pad, **ok)
    assert any(f.startswith("Continuation_v1 rule: served past an EOS at index 0") for f in pe["faults"])
    # a rule that disagrees with the arithmetic -> the disagreement is a fault BY NAME and the rule's value stands (the rule is the mechanism)
    dis = POP.executed_prefix_of_record("r0", a_r=3, lag=1, cap=8, eos_declared=True, served_len_commit=3, served=[5, 7, EOS], eos_ids=[EOS], **ok,
                                       rule=lambda served, *, cap, A, eos: dict(_rule_like_pad(served, cap=cap, A=A, eos=eos), executed_prefix_len=5))
    assert dis["n_exec"] == 5 and any("disagrees with the recurrence" in f and "R3" in f for f in dis["faults"])
    # a faulted request produced no class -- its mechanism reads 'not classified', r3_one_mechanism False, never 'recurrence: True' beside a fault
    for faulted in (bad, pe, dis):
        assert faulted["mechanism"].startswith("not classified") and faulted["r3_one_mechanism"] is False and faulted["by_rule"] is not None


def test_r3_without_a_served_list_the_arithmetic_is_the_mechanism_and_is_named_on_the_record():
    ok = dict(linked=True, lag_attested=True, a_r_source="match/capture/tokens.json served_by_request sha256 0000")
    e = POP.executed_prefix_of_record("r0", a_r=3, lag=1, cap=8, eos_declared=True, served_len_commit=3, **ok)
    assert e["faults"] == [] and e["n_exec"] == 4 and e["r3_one_mechanism"] is False and e["by_rule"] is None
    assert e["mechanism"].startswith("arithmetic(a_r, LAG)") and "served list was not supplied" in e["mechanism"]
    # the served list supplied with no rule: Continuation_v1's own
    e2 = POP.executed_prefix_of_record("r0", a_r=3, lag=1, cap=8, eos_declared=True, served_len_commit=3, served=[5, 7, 2], eos_ids=[2], rule=None, **ok)
    assert e2["mechanism"].startswith("recurrence(Continuation_v1)") and e2["r3_one_mechanism"] is True and e2["n_exec"] == 4


def test_r3_a_broken_continuation_unit_fails_the_import_rather_than_falling_back_to_the_arithmetic(monkeypatch):
    """The recurrence is the mechanism whenever a served list is supplied: Continuation_v1 failing to import fails the population's import,
    never a record whose class silently reads 'arithmetic'."""
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "verity_vllm.program.lifting.continuation", None)
    monkeypatch.delitem(sys.modules, "verity_vllm.check.replay.population")
    with pytest.raises(ImportError):
        importlib.import_module("verity_vllm.check.replay.population")


class _NotYetPi:
    """Two NOT_YET rows whose address rule raises on the second: the smallest index `population` reads."""
    n_steps = 1
    rows = [{"i": 0, "spec": "AttentionHead_v1{H=2}"}, {"i": 1, "spec": "AttentionHead_v1{H=2}"}]

    def describe(self, r):
        if r["i"] == 1:
            raise KeyError("op_path")
        return {"op_path": "model.layers.0.attn", "step": 0, "members": [("out",)]}


def test_a_not_yet_row_whose_address_rule_fails_is_a_mapping_fault_by_name_not_a_dropped_identity():
    from verity_vllm.check.replay import replay_codes as RC

    addresses, codes = {}, {}
    vus, ne, nef = POP.population(_NotYetPi(), "r0", addresses=addresses, codes=codes)
    domain = RC.NOT_YET["AttentionHead_v1"]
    assert vus == [] and list(addresses) == [("r0", 0, "model.layers.0.attn", "out")] and ne[domain] == 1
    (why,) = [w for w in ne if w != domain]
    assert why.startswith("AttentionHead_v1 row address unmodelled") and "KeyError" in why and ne[why] == 1
    assert codes[why]["class"] == RC.why_class(why) == "mapping" and "mapping" in RC.WHY_CLASSES_FAULT
    assert nef == {"AttentionHead_v1": 1, "AttentionHead_v1 (address unmodelled)": 1}
    # without an address accumulator the rule is not consulted: both rows are the family's domain label
    assert POP.population(_NotYetPi(), "r0")[1] == {domain: 2}


def test_r3_the_real_continuation_rule_is_the_mechanism_and_agrees_with_the_arithmetic_on_well_formed_records():
    ok = dict(linked=True, lag_attested=True, a_r_source="match/capture/tokens.json served_by_request sha256 0000")
    for a_r, lag, cap in ((12, 1, 32), (9, 1, 32), (12, 0, 32), (32, 1, 32)):
        served = [5] * (a_r - 1) + [2] if a_r < cap else [5] * cap
        e = POP.executed_prefix_of_record("r0", a_r=a_r, lag=lag, cap=cap, eos_declared=(a_r < cap) or None, served_len_commit=a_r, served=served, eos_ids=[2], **ok)
        assert e["faults"] == [] and e["r3_one_mechanism"] is True and e["n_exec"] == (cap if a_r >= cap else min(a_r + lag, cap))


def test_r3_the_population_record_names_the_mechanism_per_request(world):
    man = _manifest_for(world)
    bmap, committed = world.bmap_and_reader()
    cap = IX.ProgramIndex(world.inst).n_steps
    ok = dict(linked=True, lag_attested=True, a_r_source="match/capture/tokens.json served_by_request sha256 0000")
    ep = {"r0": POP.executed_prefix_of_record("r0", a_r=cap - 2, lag=1, cap=cap, eos_declared=True, served_len_commit=cap - 2, served=[5] * (cap - 3) + [2], eos_ids=[2],
                                             rule=_rule_like_pad, **ok),
          "r1": POP.executed_prefix_of_record("r1", a_r=cap, lag=1, cap=cap, served_len_commit=cap, **ok)}
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=2, seed=3, workers=0, manifest=man, executed_prefix=ep)
    nr = rep["population"]["not_executed_rule"]
    assert nr["mechanism"] == {"arithmetic(a_r, LAG)": 1, "recurrence(Continuation_v1)": 1} and nr["r3_one_mechanism"] is False
    assert nr["requests"]["r0"]["mechanism"].startswith("recurrence") and nr["requests"]["r1"]["mechanism"].startswith("arithmetic")
    assert nr["requests"]["r0"]["n_exec"] == cap - 1


# ---------------------------------------------------------------------------------------------------------------------------------------
# The controller families of a LIFTED Program in the POPULATION statement -- 'controller uncovered BY NAME'
# ---------------------------------------------------------------------------------------------------------------------------------------

def _with_guard_rows(inst: dict, specs: list[str]) -> dict:
    """A copy of the world's Program with controller / guard rows appended (the shape the lift emits: LBot / LStepLive / LSelect and, under
    Continuation_v1, LIsEOS / LEmitNext) -- enough for the replay's OWN count of controller rows; the rows carry no refs so the index
    is untouched (the world's evaluators never see them: no evaluator for a guard family, counted not-evaluated by name)."""
    prog = copy.deepcopy(inst)
    rows = prog["rows"]
    for k, spec in enumerate(specs):
        rows.append({"i": len(rows), "spec": spec, "form": "call", "name": f"s={k}/guard", "args": []})
    return prog


GUARD7 = ["LBot32_v1", "LStepLive1_v1", "LSelect32_v1", "LStepLive1_v1", "LSelect32_v1", "LIsEOS1_v1", "LEmitNext[H=1]_v1"]


def test_r4_a_lifted_programs_controller_rows_are_named_in_the_population_statement_with_their_mechanism(world):
    man = _manifest_for(world)
    bmap, committed = world.bmap_and_reader()
    # an un-lifted Program: no controller statement (None), nothing in uncovered_by_name under that name
    rep0 = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=2, seed=3, workers=0, manifest=man)
    assert rep0["population"]["controller"] is None and not any(u["family"] == "controller" for u in rep0["population"]["uncovered_by_name"])
    # the same manifest with a LIFTED query header (the lift: the identities are the wrapped Program's, the guard rows carry none) over the
    # two request Programs (r0, r1) carrying 7 controller rows EACH: the header is the WORKLOAD Program's traversal, so it names the SUM (14)
    #
    man_l = dict(man, query=dict(man["query"], lifted=True, lifted_rows=len(man["identities"]), guard_rows=14,
                                 guard_families={"LBot": 2, "LSelect": 4, "LStepLive": 4, "LIsEOS": 2, "LEmitNext": 2}))
    progs = {rid: _with_guard_rows(world.inst, GUARD7) for rid in world.prompts}
    rep = SR.sampled_replay(progs, bmap, committed, _weights(world), per_stratum=2, seed=3, workers=0, manifest=man_l)
    pop = rep["population"]
    c = pop["controller"]
    assert c is not None and c["family"] == "controller" and c["identities"] == 0 and c["program_rows"] == 14 and c["header_rows"] == 14
    assert c["rows_agree"] is True and c["program_rows_by_request"] == {"r0": 7, "r1": 7} and "SUM over the request Programs" in c["compare_scope"]
    assert c["families"] == {"LBot": 2, "LSelect": 4, "LStepLive": 4, "LIsEOS": 2, "LEmitNext": 2}
    assert "R8 controller replay" in c["mechanism"] and "r_t" in c["mechanism"] and "BY NAME" in c["why"]
    assert [u for u in pop["uncovered_by_name"] if u["family"] == "controller"] == [c]
    # identities 0 by rule: the accounting (reached + uncovered + not-executed == |Q(P)|) and the digests are untouched by the statement
    assert pop["uncovered_identities"] == rep0["population"]["uncovered_identities"] and pop["query_total"]["identities"] == rep0["population"]["query_total"]["identities"]
    assert pop["query_checks"]["ok"] is True and pop["query_checks"]["controller_rows_disagree"]["n"] == 0
    # an older lifted header (guard_rows only) still names the controller, with the families 'unstated' rather than silently absent
    man_o = dict(man, query=dict(man["query"], lifted=True, guard_rows=14))
    rep_o = SR.sampled_replay(progs, bmap, committed, _weights(world), per_stratum=2, seed=3, workers=0, manifest=man_o)
    assert rep_o["population"]["controller"]["program_rows"] == 14 and str(rep_o["population"]["controller"]["families"]).startswith("unstated")


def test_r4_c2_the_controller_statement_is_cross_checked_against_the_replays_own_count_of_guard_rows(world):
    """The header's `guard_rows` is the cached count of the WORKLOAD Program's traversal (one manifest for
    every request: the SUM over the request Programs); the replay counts the guard-prefixed rows of every request Program itself and
    compares AT THE HEADER'S SCOPE (sum vs header), never per request.  A disagreement is a fault BY NAME (`controller_rows_disagree`,
    ok False): a lifted header over an un-lifted Program (foreign / stale header) and a Program with controller rows under a header that
    names none (a header that would drop the statement) both fail; the statement itself is still emitted from the replay's own count."""
    man = _manifest_for(world)
    bmap, committed = world.bmap_and_reader()
    # (i) a lifted header naming 14 controller rows over the UN-lifted Programs (0 + 0): named, ok False, statement carries both numbers
    man_l = dict(man, query=dict(man["query"], lifted=True, guard_rows=14, guard_families={"LBot": 2, "LSelect": 4, "LStepLive": 4, "LIsEOS": 2, "LEmitNext": 2}))
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=2, seed=3, workers=0, manifest=man_l)
    c, ck = rep["population"]["controller"], rep["population"]["query_checks"]
    assert c is not None and c["program_rows"] == 0 and c["header_rows"] == 14 and c["rows_agree"] is False
    assert ck["ok"] is False and ck["controller_rows_disagree"]["n"] == 1
    assert ck["controller_rows_disagree"]["first"][0]["program_rows_sum"] == 0 and ck["controller_rows_disagree"]["first"][0]["header_rows"] == 14
    assert ck["controller_rows_disagree"]["first"][0]["program_rows_by_request"] == {"r0": 0, "r1": 0} and "SUM" in ck["controller_rows_disagree"]["first"][0]["scope"]
    # (ii) two request Programs WITH controller rows under an un-lifted header (guard_rows absent): the statement is emitted from the
    # replay's own count (the collector / a stale header cannot drop it) and the disagreement is named at the header's scope
    progs = {rid: _with_guard_rows(world.inst, GUARD7) for rid in world.prompts}
    rep2 = SR.sampled_replay(progs, bmap, committed, _weights(world), per_stratum=2, seed=3, workers=0, manifest=man)
    c2, ck2 = rep2["population"]["controller"], rep2["population"]["query_checks"]
    assert c2 is not None and c2["program_rows"] == 14 and c2["header_rows"] == 0 and c2["rows_agree"] is False and ck2["ok"] is False
    assert [u["family"] for u in rep2["population"]["uncovered_by_name"] if u["family"] == "controller"] == ["controller"]
    # (iii) MIXED lengths: request Programs of DIFFERENT counts (10 + 4) under a header of 14 AGREE (the header is the sum, not a
    # per-request count); under a header of 7 (a per-request reading, or a stale header) they DISAGREE -- both said with the scope
    progs_mixed = {"r0": _with_guard_rows(world.inst, GUARD7 + ["LStepLive1_v1", "LSelect32_v1", "LIsEOS1_v1"]), "r1": _with_guard_rows(world.inst, GUARD7[:4])}
    man_14 = dict(man, query=dict(man["query"], lifted=True, guard_rows=14))
    rep3 = SR.sampled_replay(progs_mixed, bmap, committed, _weights(world), per_stratum=2, seed=3, workers=0, manifest=man_14)
    c3, ck3 = rep3["population"]["controller"], rep3["population"]["query_checks"]
    assert c3["program_rows_by_request"] == {"r0": 10, "r1": 4} and c3["program_rows"] == 14 and c3["rows_agree"] is True
    assert ck3["controller_rows_disagree"]["n"] == 0 and ck3["ok"] is True
    man_7 = dict(man, query=dict(man["query"], lifted=True, guard_rows=7))
    rep4 = SR.sampled_replay(progs_mixed, bmap, committed, _weights(world), per_stratum=2, seed=3, workers=0, manifest=man_7)
    c4, ck4 = rep4["population"]["controller"], rep4["population"]["query_checks"]
    assert c4["rows_agree"] is False and ck4["ok"] is False and ck4["controller_rows_disagree"]["first"][0] == {
        "program_rows_sum": 14, "header_rows": 7, "program_rows_by_request": {"r0": 10, "r1": 4}, "scope": c4["compare_scope"]}
    # the verdict NAMES the check (never 'see population.query_checks' for a check that has a name)
    _note, fail = _query_population_scope(rep4)
    assert fail is not None and "controller rows: header 7 != request Programs' sum 14" in fail and "sum over request Programs" in fail
    # (iv) the controller families of Continuation_v1 (LIsEOS / LEmitNext) count as controller rows: 2 rows -> 2, never 0
    progs5 = {rid: _with_guard_rows(world.inst, ["LIsEOS1_v1", "LEmitNext[H=1]_v1"]) for rid in world.prompts}
    assert COV.controller_rows_of(progs5) == {str(rid): 2 for rid in world.prompts}
    # (v) no manifest: no reconciliation, but the replay's own count still stands in the statement (rows_agree None: nothing to agree with)
    rep6 = SR.sampled_replay(progs, bmap, committed, _weights(world), per_stratum=2, seed=3, workers=0, manifest=None)
    c6 = rep6["population"]["controller"]
    assert c6 is not None and c6["program_rows"] == 14 and c6["header_rows"] == 0 and c6["rows_agree"] is None

def test_revb_n_2_16_1_the_sampler_families_mechanism_names_what_consumes_them():
    """`sampler_seed` / `sampler_splits` are uncovered BY NAME with a mechanism that says what CONSUMES the
    committed values (the sampler VU's exact replay in form (A); stoch_recompute's attribution), as `input_token_ids`' entry does --
    not only the linkage check on the values themselves."""
    for fam in ("sampler_seed", "sampler_splits"):
        m = COV.QUERY_FAMILY_MECHANISM[fam]
        assert "prescribed_input_linkage" in m and "GumbelTopPTokenSelect_v1" in m and "stoch_recompute" in m, m
    assert "boundary linkage" in COV.QUERY_FAMILY_MECHANISM["input_token_ids"]
