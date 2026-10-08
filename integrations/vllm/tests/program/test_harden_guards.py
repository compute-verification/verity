"""R11 cb-harden (3) — export guard / applicability preservation of the Bet A artifact contract (R10 cb-fix: `TargetProfile`,
`ArtifactIdentity`, `check_reuse`; `verity_vllm.pipeline.build` manifest v3).

Contract under test (brief (3)): (a) the specialisation guards `torch.export` records — input shapes / dtypes, the (empty) range
constraints, the graph-signature facts — are captured into the manifest/artifact; (b) a build for a different declared target has
a different artifact identity even with an identical Program digest; (c) an artifact built under assumptions A is REFUSED
(fail-closed) when applied under B: device class, dtype, batch, sequence bound; (d) manifest fields are not duplicated literals —
one source of truth.

Evidence sources: the R10 recorded artefacts (`integrations/vllm/out/gen/r10/cb-fix/evidence/fix/<case>/{artifact.json,
construction_manifest.json}` — real constructions on cp-cb8; not in this tree, so the checks that read them skip), the pure-Python
contract module `verity_vllm.program.frontend.target_profile`, the builder source `verity_vllm/pipeline/build.py`, and (opt-in, `HARDEN_LIVE=1`, pod venv with vLLM) two live constructions differing only in a
structural input.

Verdict convention: assertions state the CONTRACT. Where the contract is known not to hold today the test is marked
`xfail(strict=True)` with the finding as the reason — fixing the gap turns the xfail into an XPASS failure that forces the marker
to be removed, so the finding stays load-bearing. None of these gaps is a semantic failure of a generated Program (the Program
digest DOES carry shapes/dtypes — G6); they are applicability-contract gaps (the R10 open item A4 "guards not preserved as
artefacts").

Run: laptop `python -m pytest -q tests/program/test_harden_guards.py -k "not live and not G6"` (pure Python); pod
`CUDA_VISIBLE_DEVICES="" PYTHONPATH=. HARDEN_LIVE=1 nice -n 10 python -m pytest -q -p no:cacheprovider
tests/program/test_harden_guards.py`; `HARDEN_OUT=<dir>` writes `guards_results.json`.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys

import pytest

import verity.ml.circuits
from verity_vllm.program.frontend.target_profile import ApplicabilityError, ArtifactIdentity, TargetProfile, check_reuse, reuse_mismatches

INTEGRATION = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # integrations/vllm
CORE_SRC = os.path.dirname(os.path.dirname(os.path.abspath(verity.ml.circuits.__file__)))  # the repository root
EVIDENCE = os.path.join(INTEGRATION, "out", "gen", "r10", "cb-fix", "evidence", "fix")
DERIVE_STEP = os.path.join(INTEGRATION, "verity_vllm", "pipeline", "build.py")

RESULTS: list[dict] = []


def _record(case, expected, observed, verdict, cause="", **extra):
    RESULTS.append({"case": case, "expected": expected, "observed": observed, "verdict": verdict, "cause": cause, **extra})


@pytest.fixture(scope="module", autouse=True)
def _dump_results():
    yield
    out = os.environ.get("HARDEN_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(RESULTS, open(os.path.join(out, "guards_results.json"), "w"), indent=1)


def _load(case: str) -> tuple[dict, dict]:
    d = os.path.join(EVIDENCE, case)
    if not os.path.exists(os.path.join(d, "artifact.json")):
        pytest.skip(f"recorded evidence {d} not present in this tree")
    return json.load(open(os.path.join(d, "artifact.json"))), json.load(open(os.path.join(d, "construction_manifest.json")))


GAP = "applicability-contract gap (R10 open item A4): "


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
# (b) identity distinguishes targets with one digest (recorded evidence: real constructions, cp-cb8)
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


@pytest.mark.xfail(strict=True, reason="duplicated literal (the R10 bug shape): `construction_version.mechanism` and "
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


def test_G4c_no_capability_or_sm_literal_outside_target_profile():
    """The R10 fix's invariant, re-checked on this tree: the declared capability / SM count have one authoritative site."""
    files = ["verity_vllm/program/frontend/export_compat.py", "verity_vllm/program/frontend/vllm_meta.py", "verity_vllm/pipeline/build.py",
             *sorted(os.path.relpath(p, INTEGRATION) for p in glob.glob(os.path.join(INTEGRATION, "verity_vllm/program/frontend/rules/vllm_bindings/*.py")))]
    hits = []
    for rel in files:
        p = os.path.join(INTEGRATION, rel)
        assert os.path.exists(p), f"{rel} moved: point G4c at the file that now holds it"
        for i, line in enumerate(open(p).read().splitlines(), 1):
            # applied / declared VALUES only (the R10 literals): not prose that names the architecture in a docstring
            if re.search(r"DeviceCapability\(major=\d", line) or '"8.9 (sm_89' in line or re.search(r'"NUM_SMS":\s*\d', line) \
                    or re.search(r"=\s*\(8,\s*9\)", line):
                hits.append(f"{rel}:{i}: {line.strip()[:80]}")
    _record("G4c capability / NUM_SMS literals outside target_profile.py", "none", f"{hits}", "PASS" if not hits else "FAIL")
    assert not hits, hits


# ---------------------------------------------------------------------------------------------------------
# G6: the shape / dtype guard lives in the Program digest (derive level; torch needed)
# ---------------------------------------------------------------------------------------------------------


def test_G6_program_digest_carries_input_shape_and_dtype():
    torch = pytest.importorskip("torch")
    from verity_vllm.program.frontend.derive import derive_report
    from verity_vllm.program.frontend import derive_examples as ex
    torch.manual_seed(0)
    m = ex.MLP2(8, 16, 8)
    r3 = derive_report(m, (torch.empty(3, 8),), mode="reference", name="MLP2")
    r4 = derive_report(m, (torch.empty(4, 8),), mode="reference", name="MLP2")
    assert r3.ok and r4.ok, (r3.unsupported, r4.unsupported)
    assert r3.digest != r4.digest
    assert r3.provenance["source"]["example_inputs"] == [("torch.float32", [3, 8])]
    rb = derive_report(m, (torch.empty(3, 8, dtype=torch.bfloat16),), mode="reference", name="MLP2")
    assert (not rb.ok) or rb.digest != r3.digest  # another dtype: refused or another circuit, never the same digest
    _record("G6 digest vs T=3 / T=4 / bf16 input", "distinct digests (or refusal for the dtype)", f"{r3.digest[:10]} {r4.digest[:10]} bf16 ok={rb.ok}",
            "PASS", "the specialisation IS pinned in the digest; the gap is only that the reuse API cannot see the request")


# ---------------------------------------------------------------------------------------------------------
# G5 (live, opt-in): two constructions differing only in a structural input
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("HARDEN_LIVE") != "1", reason="live constructions need the pod venv (vLLM); set HARDEN_LIVE=1")
def test_G5_live_constructions_differing_in_M_are_distinct_but_not_distinguishable_at_reuse():
    out = os.environ.get("HARDEN_OUT", "/tmp/harden_out")
    arts = {}
    for M in (1, 2):
        d = os.path.join(out, f"g5_step_M{M}")
        if not os.path.exists(os.path.join(d, "artifact.json")):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.pathsep.join([INTEGRATION, CORE_SRC]), "HF_HUB_OFFLINE": "1"}
            env.setdefault("HF_HOME", "/workspace/hf" if os.path.isdir("/workspace/hf") else os.path.expanduser("~/.cache/huggingface"))
            r = subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", "build", "--out", d, "--wrapper", "step", "--M", str(M), "--pos0", "8"],
                               cwd=INTEGRATION, env=env, capture_output=True, text=True)
            assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        arts[M] = json.load(open(os.path.join(d, "artifact.json")))
    a1, a2 = arts[1], arts[2]
    assert a1["program_digest"] != a2["program_digest"] and a1["identity"] != a2["identity"]
    assert a1["target_profile"] == a2["target_profile"] and a1["applicability"] == a2["applicability"]
    # the M=2 requester is admitted to the M=1 artifact: nothing in the applicability distinguishes them
    admitted = not reuse_mismatches(ArtifactIdentity.from_json(a1), TargetProfile.from_json(a2["target_profile"]), a2["applicability"])
    _record("G5 live: step M=1 vs M=2", "distinct digest+identity (PASS); reuse of M=1 by an M=2 requester refused (GAP)",
            f"digests {a1['program_digest'][:10]}≠{a2['program_digest'][:10]}; identities differ; applicability equal; M=2 requester admitted to M=1 artifact={admitted}",
            "PASS + GAP", "structural inputs not in applicability")
    assert admitted, "if this fails the gap was closed: update G1c/G3c"
