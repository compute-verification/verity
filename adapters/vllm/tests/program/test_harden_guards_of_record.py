"""The export-guard contract of `test_harden_guards` on recorded constructions: (a) the guards preserved in the artifact, (b) one
Program digest under three identities, (c) fail-closed reuse, and G4a/G4b's one source of truth, each read from the recorded artefacts
under `EVIDENCE` (`<case>/{artifact.json, construction_manifest.json}`: real constructions on a GPU host, which no tracked file
holds). Skipped where they are absent.
"""
from __future__ import annotations

import json
import os
import re

import pytest

from tests.program.test_harden_guards import INTEGRATION, _record
from tests.program.test_harden_guards import _dump_results  # noqa: F401  (`HARDEN_OUT`'s dump, autouse)
from verity_vllm.program.frontend.target_profile import ApplicabilityError, ArtifactIdentity, TargetProfile, check_reuse, reuse_mismatches

EVIDENCE = os.path.join(INTEGRATION, "out", "gen", "r10", "cb-fix", "evidence", "fix")
DERIVE_STEP = os.path.join(INTEGRATION, "verity_vllm", "pipeline", "build.py")


def _load(case: str) -> tuple[dict, dict]:
    d = os.path.join(EVIDENCE, case)
    if not os.path.exists(os.path.join(d, "artifact.json")):
        pytest.skip(f"recorded evidence {d} not present in this tree")
    return json.load(open(os.path.join(d, "artifact.json"))), json.load(open(os.path.join(d, "construction_manifest.json")))


GAP = "applicability-contract gap (open item A4): "


# ---------------------------------------------------------------------------------------------------------
# (a) guard preservation
# ---------------------------------------------------------------------------------------------------------


def test_G1a_environment_and_numerics_conditions_are_preserved_in_the_artifact():
    art, man = _load("base")
    app = art["applicability"]
    for k in ("numerics_profile", "attention_impl", "attention_backend", "vllm_flash_attn_version", "fused_add_rms_norm_dispatch", "vllm_version",
              "torch_version", "triton_version", "tuned_gemm_arch_family", "tuned_gemm_table_sha256", "gemm_kernel_src_sha256", "rms_kernel_src_sha256"):
        assert k in app and app[k] not in (None, [], ""), k
    assert art["target_profile"] == man["target"]["declared"]
    assert art["target_profile"]["numerics_profile"] == man["profile"] == app["numerics_profile"]
    _record("G1a environment/numerics conditions in artifact.applicability", "12 observed conditions present, none empty", f"present: {sorted(app)}", "PASS")


def test_G1b_structural_inputs_are_in_the_manifest():
    art, man = _load("base")
    si = man["structural_inputs"]
    assert {"LP", "T", "M", "pos0", "eos", "sampling"} <= set(si), si
    assert man["wrapper"] == "request" and man["model"]["pin"]["dtype"] == "bfloat16" and man["model"]["pin"]["max_model_len"] == 2048
    _record("G1b structural inputs (LP/T/M/pos0/eos/wrapper), dtype, max_model_len in the manifest", "recorded", json.dumps(si), "PASS")


@pytest.mark.xfail(strict=True, reason=GAP + "the export's specialisation facts (input shapes LP/T/M/pos0, wrapper, model pin incl. dtype, "
                   "max_model_len) are in the manifest and in the Program digest but NOT in artifact.applicability / target_profile, so "
                   "`check_reuse` cannot name them")
def test_G1c_structural_inputs_dtype_and_sequence_bound_are_applicability_conditions():
    art, man = _load("base")
    keys = set(art["applicability"]) | set(art["target_profile"])
    missing = [k for k in ("structural_inputs", "wrapper", "model_pin", "dtype", "max_model_len") if k not in keys]
    _record("G1c shapes/dtype/max_model_len/model pin as applicability conditions", "present in artifact.applicability", f"missing: {missing}",
            "GAP (xfail)", "test gap -> contract gap: only in manifest + digest")
    assert not missing, missing


@pytest.mark.xfail(strict=True, reason=GAP + "torch.export's guard facts (range_constraints, graph_signature: lifted params/buffers, "
                   "mutations, user inputs) are not recorded; `construction.export` carries secs / n_nodes / op_histogram / triton kernels only")
def test_G1d_export_guard_facts_are_recorded():
    art, man = _load("base")
    exp = man["construction"]["export"]
    have = sorted(exp)
    _record("G1d export guards (range_constraints, graph_signature facts) recorded", "recorded in manifest.construction.export", f"keys: {have}",
            "GAP (xfail)", "manifest records op histogram only")
    assert "range_constraints" in exp and "graph_signature" in exp, have


# ---------------------------------------------------------------------------------------------------------
# (b) identity distinguishes targets with one digest (recorded evidence: real constructions)
# ---------------------------------------------------------------------------------------------------------


def test_G2_recorded_constructions_share_the_digest_and_differ_in_identity():
    base, _ = _load("base")
    others = {}
    for case in ("cap80_108", "sms132"):
        a, _ = _load(case)
        others[case] = a
        assert a["program_digest"] == base["program_digest"], case
        assert a["identity"] != base["identity"], case
        assert a["target_profile"] != base["target_profile"], case
        assert ArtifactIdentity.from_json(a).recomputed() == a["identity"], case
    idents = {"base": base["identity"][:12], **{k: v["identity"][:12] for k, v in others.items()}}
    assert len(set(idents.values())) == 3
    _record("G2 base vs cap80_108 vs sms132 (recorded)", "one Program digest, three identities", f"digest {base['program_digest'][:12]}; identities {idents}", "PASS")


# ---------------------------------------------------------------------------------------------------------
# (c) fail-closed reuse under other assumptions
# ---------------------------------------------------------------------------------------------------------


def _base_art():
    art, _ = _load("base")
    return ArtifactIdentity.from_json(art), TargetProfile.from_json(art["target_profile"]), dict(art["applicability"])


def test_G3a_device_class_and_kernel_environment_mismatches_are_refused():
    art, prof, app = _base_art()
    with pytest.raises(ApplicabilityError) as e1:
        check_reuse(art, prof.with_overrides(compute_capability=(9, 0)), app)
    assert any("compute_capability" in r for r in e1.value.reasons), e1.value.reasons
    with pytest.raises(ApplicabilityError) as e2:
        check_reuse(art, prof, {**app, "vllm_flash_attn_version": ["3"]})
    with pytest.raises(ApplicabilityError) as e3:
        check_reuse(art, prof, {**app, "triton_version": "3.8.0"})
    with pytest.raises(ApplicabilityError):
        check_reuse(art, prof, None)
    check_reuse(art, prof, app)  # the honest requester is admitted
    _record("G3a reuse under another device class / FA version / Triton / no facts", "refused, field named; honest requester admitted",
            f"{e1.value.reasons[0][:80]} | {e2.value.reasons[0][:60]} | {e3.value.reasons[0][:60]}", "PASS")


@pytest.mark.xfail(strict=True, reason=GAP + "a requester serving another dtype (float16) is admitted: dtype is not an applicability "
                   "condition and extra requester facts are ignored by `reuse_mismatches`")
def test_G3b_reuse_under_another_dtype_is_refused():
    art, prof, app = _base_art()
    bad = reuse_mismatches(art, prof, {**app, "dtype": "float16", "model_dtype": "float16"})
    _record("G3b reuse for a float16 request of a bfloat16 artifact", "refused (dtype named)", f"mismatches: {bad}", "GAP (xfail)",
            "dtype only in the Program digest; requester cannot state it")
    assert bad, "admitted"


@pytest.mark.xfail(strict=True, reason=GAP + "the reuse API has no structural-input argument: an artifact built for request LP=8/T=4 "
                   "(max_model_len 2048, greedy) is admitted for LP=8/T=5 or another sequence bound — nothing in `check_reuse` can see the request shape")
def test_G3c_reuse_under_another_batch_or_sequence_bound_is_refused():
    art, prof, app = _base_art()
    requester = {**app, "structural_inputs": {"LP": 8, "T": 5, "M": 1, "pos0": 8, "eos": 0, "sampling": "greedy"}, "max_model_len": 4096}
    bad = reuse_mismatches(art, prof, requester)
    _record("G3c reuse for LP=8/T=5, max_model_len=4096 of the 8/4/2048 artifact", "refused (shape / bound named)", f"mismatches: {bad}", "GAP (xfail)",
            "structural inputs are not applicability conditions; the identity differs only via the digest the requester cannot compute without building")
    assert bad, "admitted"


# ---------------------------------------------------------------------------------------------------------
# (d) one source of truth
# ---------------------------------------------------------------------------------------------------------


def test_G4a_manifest_and_wrapper_agree_on_target_and_artifact():
    art, man = _load("base")
    assert man["artifact"] == art
    assert man["target"]["declared"] == art["target_profile"]
    assert man["program"]["digest"] == art["program_digest"]
    assert man["target"]["applied"]["compute_capability"] == art["target_profile"]["compute_capability"]
    assert man["target"]["applied_after_export"]["num_sms"] == art["target_profile"]["num_sms"]
    _record("G4a manifest.target.declared / manifest.artifact / program.digest vs artifact.json", "single values", "equal", "PASS")


@pytest.mark.xfail(strict=True, reason="duplicated literal (the stale-manifest bug's shape): `construction_version.mechanism` and "
                   "`construction_manifest.construction.mechanism` are two hand-written strings in derive_step.py that already drifted "
                   "('… on meta + …' vs '… of the vLLM model class on meta + …')")
def test_G4b_mechanism_string_has_one_source():
    art, man = _load("base")
    a, b = art["construction_version"]["mechanism"], man["construction"]["mechanism"]
    src = open(DERIVE_STEP).read()
    sites = re.findall(r'"mechanism":\s*"([^"]+)"', src)
    _record("G4b `mechanism` literal", "one site, one value", f"artifact={a!r}; manifest={b!r}; literal sites in derive_step.py={len(sites)}",
            "GAP (xfail)", "two literals, drifted")
    assert a == b and len(sites) <= 1, (a, b, sites)
