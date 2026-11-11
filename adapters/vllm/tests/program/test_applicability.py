"""Driver-level applicability tests of Bet A, GPU-free, on the B0 `request 8/4` driver
(`validation/sens/run.sh`): every case is a real construction (`derive_step`) in a subprocess with `CUDA_VISIBLE_DEVICES=""`.

Needs the pinned vLLM (d9105ea80) and the SmolLM2-135M snapshot under `$HF_HOME` (default /workspace/hf); skipped otherwise.
Eight constructions run in parallel (~20 s on a 128-vCPU pod, ~1.9 GB RSS each); one of them (`gpu_host`) leaves the pod's GPU visible.

(i)   the reported inconsistent-capability case: a target whose circuit is genuinely unchanged (sm_80 / 108 SMs; re-pinned 132 SMs)
      builds the SAME Program digest with a DIFFERENT artifact identity, and the manifest reports the capability that was APPLIED;
(i-b) a second capability applied only around the export is refused (declared != applied at export);
(ii)  a consistently changed target (9.0 -> FA3; TRITON_ATTN) is refused fail-closed with the reason verbatim, no artifact written;
(iii) two construction hosts, same profile -> same identity (host block not in any component); the Program digest is byte-identical
      to the recorded builds (`59b35d80…d496`);
(iv)  reuse of an artifact under a mismatched profile / applicability is rejected with every differing field named;
plus: the Program digest of B0 8/4 is unchanged by the patch; the dead env `VLLM_ATTENTION_BACKEND` is not a declared fact."""

import glob
import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from verity_vllm.program.frontend.target_profile import ApplicabilityError, ArtifactIdentity, TargetProfile, load_artifact, reuse_mismatches

B0_DIGEST = "59b35d80faf29438d21a0ea1c96c877ec69e2f21174f11b835ac1ef04c1ed496"  # the recorded request_8_4 and sens/base builds
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORE_SRC = os.path.join(os.path.dirname(os.path.dirname(REPO)), "packages", "verity", "src")
HF_HOME = os.environ.get("HF_HOME", "/workspace/hf")
SNAP = os.path.join(HF_HOME, "hub", "models--HuggingFaceTB--SmolLM2-135M", "snapshots", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2", "config.json")
if not os.path.exists(SNAP):
    pytest.skip(f"pinned SmolLM2 snapshot not under HF_HOME={HF_HOME}", allow_module_level=True)

CASES = {  # name -> --target JSON (None = default profile)
    "base": None,
    "cap80_108": '{"compute_capability": [8, 0], "num_sms": 108}',
    "sms132": '{"num_sms": 132}',
    "cap90": '{"compute_capability": [9, 0]}',
    "cap90_fa2": '{"compute_capability": [9, 0], "flash_attn_version": 2}',
    "cap100": '{"compute_capability": [10, 0]}',
    "triton_attn": '{"attention_backend": "TRITON_ATTN"}',
}

_HOSTILE_EXPORT = textwrap.dedent("""
    # the EXPORT applies a capability other than the declared one: the export is handed a 9.0 profile while the declared
    # target stays 8.9
    import sys
    import verity_vllm.program.frontend.vllm_meta as vm
    from verity_vllm.program.frontend.target_profile import TargetProfile
    _orig = vm.export_module
    def _hostile(*a, **k):
        k["profile"] = TargetProfile(compute_capability=(9, 0))
        return _orig(*a, **k)
    vm.export_module = _hostile
    from verity_vllm.pipeline.build import run
    res = run(sys.argv[1], 1, 8, "b1-eager-v3", "gloo", wrapper="request", LP=8, T=4)
    raise SystemExit(0 if res.get("ok") else 1)
""")


def _env(extra=None, gpu_visible=False):
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_HOME=HF_HOME, PYTHONPATH=os.pathsep.join([REPO, CORE_SRC]), VLLM_BATCH_INVARIANT="1", OMP_NUM_THREADS="8")
    if gpu_visible:  # a second "construction host": whatever GPU the pod has stays visible (the declared target must still win)
        e.pop("CUDA_VISIBLE_DEVICES", None)
    e.update(extra or {})
    return e


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    root = tmp_path_factory.mktemp("cbfix")
    procs = {}
    for name, target in CASES.items():
        out = root / name
        cmd = [sys.executable, "-m", "verity_vllm.pipeline.cli", "build", "--out", str(out), "--wrapper", "request", "--LP", "8", "--T", "4"]
        if target:
            cmd += ["--target", target]
        procs[name] = (out, subprocess.Popen(cmd, cwd=REPO, env=_env(), stdout=open(out.with_suffix(".log"), "w"), stderr=subprocess.STDOUT))
    out = root / "export_divergence"
    procs["export_divergence"] = (out, subprocess.Popen([sys.executable, "-c", _HOSTILE_EXPORT, str(out)], cwd=REPO, env=_env(),
                                                        stdout=open(out.with_suffix(".log"), "w"), stderr=subprocess.STDOUT))
    out = root / "dead_env"  # the default profile with the dead variable set in the environment
    procs["dead_env"] = (out, subprocess.Popen([sys.executable, "-m", "verity_vllm.pipeline.cli", "build", "--out", str(out), "--wrapper", "request",
                                                "--LP", "8", "--T", "4"], cwd=REPO, env=_env({"VLLM_ATTENTION_BACKEND": "TRITON_ATTN"}),
                                               stdout=open(out.with_suffix(".log"), "w"), stderr=subprocess.STDOUT))
    out = root / "gpu_host"  # the default profile on a host whose GPU (if any) is NOT hidden: a second construction host for (iii)
    procs["gpu_host"] = (out, subprocess.Popen([sys.executable, "-m", "verity_vllm.pipeline.cli", "build", "--out", str(out), "--wrapper", "request",
                                                "--LP", "8", "--T", "4"], cwd=REPO, env=_env(gpu_visible=True),
                                               stdout=open(out.with_suffix(".log"), "w"), stderr=subprocess.STDOUT))
    res = {}
    for name, (out, p) in procs.items():
        rc = p.wait()
        r = json.load(open(out / "result.json"))
        assert "traceback_tail" not in r, f"{name} crashed: {r.get('exception')}: {r.get('message')}\n{r.get('traceback_tail')}"
        m = json.load(open(out / "construction_manifest.json"))
        res[name] = {"rc": rc, "out": out, "result": r, "manifest": m,
                     "artifact": json.load(open(out / "artifact.json")) if (out / "artifact.json").exists() else None}
    return res


def _digest(b):
    return (b["result"].get("program") or {}).get("digest")


# -- the Program digest is unchanged by the patch (and is the cross-host identity point) -------------------------------------

def test_program_digest_of_b0_8_4_unchanged_and_cross_host(builds):
    b = builds["base"]
    assert b["rc"] == 0 and b["result"]["ok"] is True
    assert _digest(b) == B0_DIGEST, "the Program digest of B0 request 8/4 must be byte-identical to the recorded builds"
    assert b["result"]["program"]["gates"] == 234235513
    assert b["artifact"]["program_digest"] == B0_DIGEST and b["manifest"]["artifact"]["identity"] == b["artifact"]["identity"]
    assert b["manifest"]["target"]["host"]["cuda_available"] is False  # GPU-free


# -- (i) the reported case: different target, same circuit -> same digest, different identity, manifest says what was applied ----

@pytest.mark.parametrize("name,cap,sms", [("cap80_108", [8, 0], 108), ("sms132", [8, 9], 132)])
def test_case_i_same_digest_different_identity_manifest_reports_applied(builds, name, cap, sms):
    base, b = builds["base"], builds[name]
    assert b["rc"] == 0 and b["result"]["ok"] is True
    assert _digest(b) == _digest(base) == B0_DIGEST  # the circuit is genuinely unchanged for these targets
    assert b["artifact"]["identity"] != base["artifact"]["identity"]
    t = b["manifest"]["target"]
    assert t["declared"]["compute_capability"] == cap and t["declared"]["num_sms"] == sms
    assert t["applied"]["compute_capability"] == cap and t["applied"]["num_sms"] == sms  # read back from vLLM, not a literal
    assert t["applied_after_export"]["compute_capability"] == cap and t["applied_after_export"]["num_sms"] == sms
    decl = {d["field"]: d for d in b["manifest"]["declared"]}
    assert decl["compute capability"]["value"] == f"{cap[0]}.{cap[1]}" and decl["compute capability"]["applied"] == cap
    assert decl["NUM_SMS"]["value"] == sms and decl["NUM_SMS"]["applied"] == sms
    assert "8.9 (sm_89, Ada)" not in json.dumps(b["manifest"]["declared"])  # no hard-coded capability literal
    assert b["artifact"]["target_profile"] == t["declared"]
    # the applicability names the registered GEMM specialization the construction pinned (registry keyed by the target)
    app = b["artifact"]["applicability"]
    assert (app["gemm_target"], app["tuned_gemm_arch_family"]) == (("ampere", None) if cap == [8, 0] else ("ada", "ada")), app
    assert app["gemm_target_dot"] == "AmpereBF16TcDot16" and app["gemm_target_refusal"] is None
    assert (app["tuned_gemm_table_sha256"] is None) == (cap == [8, 0])


def test_case_i_b_capability_applied_only_around_the_export_is_refused(builds):
    """A second capability applied around the export cannot produce an artifact labelled 8.9."""
    b = builds["export_divergence"]
    assert b["rc"] == 1 and b["result"]["ok"] is False and b["artifact"] is None
    u = b["result"]["unsupported"]
    assert u["reason"] == "target_profile_mismatch" and u["phase"] == "export"
    assert "compute_capability declared [8, 9], applied [9, 0]" in u["hint"]
    assert b["manifest"]["target"]["applied"]["compute_capability"] == [8, 9]  # instantiation saw the declared value
    assert b["manifest"]["target"]["applied_after_export"]["compute_capability"] == [9, 0]  # the export did not: recorded, refused
    assert not (b["out"] / "descriptor.json.gz").exists() and not (b["out"] / "artifact.json").exists()


# -- (ii) a consistently changed target: refused fail-closed, verbatim ----------------------------------------------------------

@pytest.mark.parametrize("name,needle,applied", [
    # a capability with NO registered GEMM specialization (Blackwell): the FIRST refusal is the GEMM target registry's
    # (the qkv GEMM precedes attention in the graph).  cc 9.0 is registered (case iv below).
    ("cap100", "numerics_unregistered: no registered GEMM specialization for target cc 10.0 (tuned_gemm_arch_family='blackwell')", {"compute_capability": [10, 0], "tuned_gemm_arch_family": "blackwell"}),
    ("triton_attn", "numerics_unregistered: attention impl='vllm.v1.attention.backends.triton_attn.TritonAttentionImpl' (observed on layer 'model.layers.0.self_attn.attn') differs from the target's 'vllm.v1.attention.backends.flash_attn.FlashAttentionImpl'", {"attention_backend": "TRITON_ATTN"}),
])
def test_case_ii_consistently_changed_target_is_refused_verbatim(builds, name, needle, applied):
    b = builds[name]
    assert b["rc"] == 1 and b["result"]["ok"] is False and b["artifact"] is None
    u = b["result"]["unsupported"]
    assert u["reason"] == "mutation" and needle in u["hint"], u
    t = b["manifest"]["target"]
    for k, v in applied.items():
        assert t["applied"][k] == v, (k, t["applied"])  # the manifest of the refused build still says what was applied
    assert not (b["out"] / "descriptor.json.gz").exists()


# -- (iv) a consistently changed target WITH a registered binding: another Program (digest moves), declared in the manifest ----

@pytest.mark.parametrize("name,fa,attn_kind,label", [
    ("cap90", 3, "Attention_v2{T,NH,KVH,D,BN=fa3_kblock_n(D,seqlen_q,NH/KVH),DOT=HopperBF16WgmmaDot16_v1,INV=Fa3InvSum_v1}", "bi-eager"),
    ("cap90_fa2", 2, "Attention_v2{T,NH,KVH,D,BN=fa2_kblock_n(D),DOT=HopperBF16WgmmaDot16_v1,INV=Fa2InvSum_v1}", "bi-eager-fa2"),
])
def test_case_iv_hopper_target_binds_the_registered_hopper_definitions(builds, name, fa, attn_kind, label):
    """cc 9.0: the `hopper` GEMM target (wgmma k16 step -> `Gemm_v2{K,N,DOT=Hopper…}`) and the attention chain of the EFFECTIVE FA version
    (vLLM's selector default 3, or the declared knob 2 = the labelled `bi-eager-fa2` configuration): a Program other than B0's, with the
    target facts (incl. the FA version read back from the instantiated layers) in the applicability."""
    b = builds[name]
    assert b["rc"] == 0 and b["result"]["ok"] is True, b["result"]
    assert _digest(b) != B0_DIGEST and _digest(b) != _digest(builds["base"])
    app = b["artifact"]["applicability"]
    assert (app["gemm_target"], app["tuned_gemm_arch_family"], app["gemm_target_dot"]) == ("hopper", "hopper", "HopperBF16WgmmaDot16"), app
    assert app["target_bindings"]["gemm_kind"] == "Gemm_v2{K,N,DOT=HopperBF16WgmmaDot16_v1}" and app["target_bindings"]["attention_kind"] == attn_kind, app["target_bindings"]
    assert (app["flash_attn_version_applied"], app["target_bindings"]["flash_attn_version_effective"]) == (fa, fa), app
    assert app["flash_attn_version_declared"] == (2 if name == "cap90_fa2" else None)   # the knob as declared (None = vLLM's selector default)
    t = b["manifest"]["target"]
    assert t["applied"]["compute_capability"] == [9, 0] and t["applied"]["flash_attn_version"] == fa and t["applied"]["tuned_gemm_arch_family"] == "hopper"
    from verity_vllm.program.frontend.target_profile import TargetProfile
    assert TargetProfile.from_json(b["artifact"]["target_profile"]).execution_label == label
    assert _digest(builds["cap90"]) != _digest(builds["cap90_fa2"])   # FA2-forced and native FA3 are two Programs


# -- (iii) two construction hosts, one profile -> one identity ----------------------------------------------------------------

def test_case_iii_identity_is_host_independent(builds):
    a = builds["base"]["artifact"]
    art = ArtifactIdentity.from_json(a)
    assert art.recomputed() == art.identity
    # the host block lives in the manifest only; changing it changes nothing in the identity's components
    for host in ({"hostname": "cp-cba", "cuda_available": True, "host_gpu": "NVIDIA GeForce RTX 4090"}, {"hostname": "cp-cbc", "cuda_available": False},
                 builds["base"]["manifest"]["target"]["host"]):
        assert ArtifactIdentity.compute(art.construction_version, art.target_profile, art.applicability, art.program_digest) == art.identity, host
    assert "host" not in a and "hostname" not in json.dumps(a)
    # the real cross-host point: this pod's Program digest equals the recorded builds'
    assert art.program_digest == B0_DIGEST
    # a second construction in another process on this host with the same profile (dead_env) -> same identity
    assert builds["dead_env"]["artifact"]["identity"] == a["identity"]
    # a second construction HOST configuration: the pod's GPU left visible (e.g. an L40S, 142 SMs).  [X-01] With no declared
    # target the target IS the host device, queried once (`TargetProfile.from_device`): the applied profile shows the device's SM
    # count / capability, the Program digest is the same (NUM_SMS is a scheduling fact), and the identity differs from the GPU-free
    # build exactly when the device differs from the default profile.  Without a visible GPU the build is the default one.
    g = builds["gpu_host"]
    gh, ga = g["manifest"]["target"]["host"], g["manifest"]["target"]["applied"]
    # a Hopper host IS a hopper target: the registered hopper bindings make it case iv's Program (cap90), not B0's
    hopper_host = bool(gh["cuda_available"]) and ga["compute_capability"][0] == 9
    assert g["rc"] == 0 and _digest(g) == (_digest(builds["cap90"]) if hopper_host else B0_DIGEST)
    if gh["cuda_available"]:
        assert isinstance(gh["host_gpu"], str) and gh["host_gpu"] and isinstance(gh["host_sm_count"], int)  # the real GPU is recorded...
        assert ga["num_sms"] == gh["host_sm_count"] and ga["compute_capability"] == gh["host_compute_capability"]  # ...and IS the target
        assert g["artifact"]["target_profile"]["num_sms"] == gh["host_sm_count"] and g["artifact"]["applicability"]["num_sms"] == gh["host_sm_count"]
        assert (g["artifact"]["identity"] == a["identity"]) == (ga["num_sms"] == 128 and ga["compute_capability"] == [8, 9])
        assert gh != builds["base"]["manifest"]["target"]["host"]
    else:
        assert ga["compute_capability"] == [8, 9] and ga["num_sms"] == 128 and g["artifact"]["identity"] == a["identity"]


# -- (iv) reuse under a mismatched profile -> rejected -------------------------------------------------------------------------

def test_case_iv_reuse_under_mismatched_profile_is_rejected(builds):
    base, cap80 = builds["base"], builds["cap80_108"]
    path = str(base["out"] / "artifact.json")
    applic = base["artifact"]["applicability"]
    # its own profile, its own conditions: admitted
    assert load_artifact(path, TargetProfile.default(), applic).identity == base["artifact"]["identity"]
    # another target: rejected, fields named
    with pytest.raises(ApplicabilityError) as ei:
        load_artifact(path, TargetProfile(compute_capability=(8, 0), num_sms=108), applic)
    assert ei.value.reasons == ["target_profile.compute_capability: requested [8, 0], artifact built for [8, 9]",
                                "target_profile.num_sms: requested 108, artifact built for 128"]
    # same target, but the requester's environment (here: cap80's applicability, which differs in the tuned-GEMM family) does not satisfy the conditions
    bad = reuse_mismatches(ArtifactIdentity.from_json(base["artifact"]), TargetProfile.default(), cap80["artifact"]["applicability"])
    # [X-01] the SM count read back through the shim is an applicability condition too (the one TargetProfile fact of the persistent GEMM)
    # and so are the registered GEMM target (ada vs ampere) and the tuned table it consults (the sm_89 table vs none)
    cap80_applic = cap80["artifact"]["applicability"]
    assert bad == ["applicability.gemm_target: artifact requires 'ada', requester has 'ampere'",
                   "applicability.num_sms: artifact requires 128, requester has 108",
                   # the target-keyed bindings (GEMM/attention kinds, dot, tensor-core + tuned family, effective FA version) as one condition
                   f"applicability.target_bindings: artifact requires {applic['target_bindings']!r}, requester has {cap80_applic['target_bindings']!r}",
                   "applicability.tuned_gemm_arch_family: artifact requires 'ada', requester has None",
                   f"applicability.tuned_gemm_table_sha256: artifact requires {applic['tuned_gemm_table_sha256']!r}, requester has None"]
    # the two artifacts share the Program digest and are still not interchangeable
    assert base["artifact"]["program_digest"] == cap80["artifact"]["program_digest"]
    with pytest.raises(ApplicabilityError):
        load_artifact(str(cap80["out"] / "artifact.json"), TargetProfile.default(), applic)


# -- the dead env var is not a declared fact; only consulted variables are ---------------------------------------------------

def test_dead_env_var_is_recorded_as_ignored_not_declared(builds):
    b = builds["dead_env"]
    assert b["rc"] == 0 and _digest(b) == B0_DIGEST and b["artifact"]["identity"] == builds["base"]["artifact"]["identity"]
    c = b["manifest"]["target"]["consulted"]
    assert "VLLM_ATTENTION_BACKEND" not in c["env_consulted"]
    assert c["env_ignored"]["VLLM_ATTENTION_BACKEND"]["value"] == "TRITON_ATTN" and "not defined in this vLLM pin" in c["env_ignored"]["VLLM_ATTENTION_BACKEND"]["reason"]
    assert set(c["env_consulted"]) >= {"VLLM_BATCH_INVARIANT", "VLLM_USE_LAYERNAME"} and c["env_consulted"]["VLLM_BATCH_INVARIANT"] == "1"
    assert "VLLM_ATTENTION_BACKEND" not in json.dumps(b["manifest"]["declared"]) and "declared_profile_facts" not in b["manifest"]
    assert b["manifest"]["target"]["applied"]["attention_backend"] is None  # the knob vLLM actually reads was not set


def test_no_target_literal_remains_in_the_builder_sources():
    bindings = sorted(os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, "verity_vllm/program/frontend/rules/vllm_bindings/*.py")))
    for rel in ("verity_vllm/program/frontend/export_compat.py", *bindings, "verity_vllm/program/frontend/vllm_meta.py"):
        src = open(os.path.join(REPO, rel)).read()
        assert "DeviceCapability(major=8" not in src and "8.9 (sm_89, Ada)" not in src, rel
        assert 'os.environ.get("VLLM_ATTENTION_BACKEND")' not in src, rel
