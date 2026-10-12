"""The three canonical tables (frozen 2026-09-22): the validity predicate, best-of-N cells, the same-SKU invariant, Table 3's
identity and joint display, the SKU normaliser, the JSON round trip and the snapshot -- all over an isolated store root."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from verity.ml.circuits.targets.target import BF16_HOPPER, FIRST, FP8_ADA_MMA, FP8_HOPPER_WGMMA, TARGETS
from verity_numerical.bench import contract
from verity_numerical.bench import tables as T

pytest.importorskip("research.store.local", reason="research.store (tools/research) is not importable")
from research.store.local import LocalStore  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
A100 = "NVIDIA A100-SXM4-80GB"          # as nvidia-smi writes it; the target says "NVIDIA A100 SXM4 80GB"
H100 = "NVIDIA H100 80GB HBM3"
RTX4090 = "NVIDIA GeForce RTX 4090"
LANE, VERIFIER, REDTEAM = "lane-a-gkr", "rust-verifier-audit", "red-team-x"
_RUN = 0


def run_id() -> str:
    global _RUN
    _RUN += 1
    return f"r20260922-{100000 + _RUN:06d}-{_RUN:04x}"


def result(*, candidate: str = "A-GKR", target=FIRST, gpu: str = A100, proof_class: str = "NON_ZK_PROOF", t_total: float = 4.0,
           buckets: dict[str, float] | None = None, joints: dict[tuple[str, ...], float] | None = None, instances: Any = "frozen",
           security: dict | None = None, **fp_over: Any) -> dict[str, Any]:
    """A contract-conforming ``result.json`` of ``candidate`` on ``target``: buckets that sum to ``t_total`` minus ``Other``."""

    backend = {"A-GKR": "verity-gkr", "B-Ligero": "b-ligero chain (archive/direct/ligero)", "SP1": "sp1"}[candidate]
    if instances == "frozen":
        instances = copy.deepcopy(T.FROZEN_INSTANCES[target.name])
    fp = contract.fingerprint(
        profile=target.name, domain="finite-only", K=T.K, B=T.B, proof_class=proof_class, authentication="excluded",
        zk_mode="none" if proof_class not in contract.ZK_CLASSES else "coin-commitment",
        security=security or {"target": -128, "achieved_log2": -128.3, "field": "BabyBear"},
        hardware={"gpu": {"name": gpu, "driver": "570.211.01", "cuda": "12.8"}, "cpu": {"model": "test cpu"}},
        software={"backend": {"name": backend, "commit": "0" * 40}, "toolchain": {"python": "3.12"}},
        preprocessing="none", retention="retained", instances=instances)
    fp.update(fp_over)
    if buckets is None and joints is None:
        buckets = {"witness": 0.25 * t_total, "encoding_commitment": 0.35 * t_total, "arithmetic": 0.2 * t_total, "lookup": 0.1 * t_total,
                   "zk_additional": 0.0, "serialization": 0.05 * t_total}
    ms = [contract.measurement(f"t.{b}", v) for b, v in (buckets or {}).items()]
    ms += [contract.measurement(contract.joint(*parts), v) for parts, v in (joints or {}).items()]
    ms += [contract.measurement(contract.T_TOTAL, t_total), contract.measurement("vu.count", T.B),
           contract.measurement("vu.seconds_per_vu", t_total / T.B), contract.measurement("vu.per_second", T.B / t_total)]
    ms += contract.rate_measurements(t_total, T.B, T.K, target.name)
    return {"schema": contract.RESULT_SCHEMA, "run_id": run_id(), "workload_fingerprint": fp,
            "validation": {"status": "passed", "checker": "synthetic", "detail": "ok", "evidence": []},
            "measurements": ms, "measurement_files": [], "artifacts": []}


def attempt(rid: str, art: str, campaign: str | None) -> dict:
    return {"schema": "research/attempt/v0.1", "id": rid, "tool": {"name": "bench", "version": None}, "derivation": "drv:" + "1" * 64,
            "reuse_key": None, "argv": ["bench"], "params": {}, "key_params": [], "inputs": {}, "conditions": {}, "key_conditions": [],
            "source": {"commit": None, "tree": None, "dirty": None, "dirty_digest": None, "lockfiles": {}}, "closure": {},
            "execution": {"start_utc": None, "end_utc": None, "wall_s": None, "host": None, "project": None, "campaign": campaign, "run_dir": "/x",
                          "stage": None, "exclusive": None, "load_start": None, "load_end": None, "remote": None},
            "cost": None, "status": {"state": "done", "rc": 0, "class": None, "failure": None, "validation": None, "result": "present", "timed_out": False},
            "outputs": {"result": art, "run_files": None}, "telemetry": None}


def put(store: LocalStore, res: dict, *, campaign: str | None = LANE, verified_by: str | None = VERIFIER, labels: list[tuple] = ()) -> str:
    """Store ``res`` as its producing attempt's output, with the lane's own row label and (optionally) a verifier's acceptance."""

    art = store.put_artifact(T.RESULT_KIND, res)
    store.put_attempt(attempt(res["run_id"], art, campaign))
    if campaign:
        store.add_label(art, "candidate", "X", campaign, ts="2026-09-22T10:00:00Z")
    if verified_by:
        store.add_label(art, "independently_verified", True, verified_by, ref=f"{res['run_id']}:verify", ts="2026-09-22T10:01:00Z")
    for key, value, by, ts in labels:
        store.add_label(art, key, value, by, ts=ts)
    return art


def native_peak(store: LocalStore, device: str, dtype: str, flop: float, spec: float) -> str:
    return store.put_artifact(T.NATIVE_PEAK_KIND, {"device": device, "dtype": dtype, "accumulate": "fp32", "method": T.NATIVE_PEAK_METHOD_CUBLAS,
                                                   "flop_per_second": flop, "spec_flop_per_second": spec, "fraction_of_spec": flop / spec,
                                                   "sass": "sm_80", "driver": "570.211.01", "cuda": "12.8"})


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalStore:
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "absent.toml"))
    monkeypatch.setenv("RESEARCH_REPO", str(tmp_path / "norepo"))
    return LocalStore(tmp_path / "store")


def cell(tables: T.Tables, target=FIRST, candidate: str = "A-GKR") -> T.Cell | None:
    return next(r for r in tables.rows if r["target"] == target.name)["cells"][candidate]


def rejected(tables: T.Tables, art: str) -> list[str]:
    return next(r for r in tables.rejected if r["artifact"] == art)["reasons"]


# --- Table 1 -------------------------------------------------------------------------------------------------------------------------


def test_table1_is_definitional_and_frozen() -> None:
    assert T.TABLE1_COLUMNS == ("Candidate", "Description", "ZK", "Verifier model", "Soundness", "Setup", "Interaction", "Security assumptions")
    assert [c.id for c in T.CANDIDATES] == ["A-GKR", "B-Ligero", "SP1"]
    for c in T.CANDIDATES:
        row = c.table1_row()
        assert len(row) == len(T.TABLE1_COLUMNS) and all(isinstance(x, str) and x for x in row)
        assert c.zk == "as configured" or c.zk.startswith(T.ZK_VALUES), c.zk
    a, b, sp1 = T.CANDIDATES
    assert a.proof_classes == ("NON_ZK_PROOF_DIAGNOSTIC", "NON_ZK_PROOF") and a.zk == "No"
    assert b.proof_classes == ("COMPLETE_ZK_BACKEND",) and b.drilldown_classes == ("COMPLETE_HVZK_BACKEND",)
    assert b.zk.startswith("Malicious-verifier ZK (interactive") and "; HVZK (Fiat–Shamir variant" in b.zk
    assert set(sp1.proof_classes) == {"NON_ZK_PROOF", "COMPLETE_HVZK_BACKEND", "COMPLETE_ZK_BACKEND"}


def test_recognisers_match_the_real_backend_names() -> None:
    assert T.candidate_for("verity-gkr").id == "A-GKR"
    assert T.candidate_for("b-ligero chain (archive/direct/ligero)").id == "B-Ligero"
    assert T.candidate_for("b-ligero chain, fp8-ada relation (archive/direct/ligero/fp8)").id == "B-Ligero"
    assert T.candidate_for("sp1").id == T.candidate_for("SP1 (v4.1, cuda prover)").id == "SP1"
    for name in ("verity-direct (plonky3 uni-stark)", "archive/gkr/packed/sumcheck_packed", "b-encode RS encoders (cupy RawKernel NTT)",
                 "archive/gkr/tensor/fused", "sp1x", "verity-gkr2", "b-ligero", "a-gkr", None, 3):
        assert T.candidate_for(name) is None, name


# --- the same-SKU normaliser ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("a", "b"), [
    ("NVIDIA A100-SXM4-80GB", "NVIDIA A100 SXM4 80GB"),
    ("NVIDIA H100 80GB HBM3", "NVIDIA H100 SXM5 80GB"),
    ("NVIDIA H100 SXM5 80GB", "H100-SXM5-80GB"),
    ("NVIDIA GeForce RTX 4090", "RTX 4090"),
    ("nvidia geforce rtx 4090", "NVIDIA GeForce RTX 4090"),
    ("NVIDIA L40S", "L40S"),
])
def test_normalise_sku_equal(a: str, b: str) -> None:
    assert T.normalise_sku(a) == T.normalise_sku(b) and T.same_sku(a, b)


@pytest.mark.parametrize(("a", "b"), [
    ("NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-40GB"),
    ("NVIDIA A100-SXM4-80GB", "NVIDIA A100-PCIE-80GB"),
    ("NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3"),
    ("NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe"),
    ("NVIDIA H100 NVL", "NVIDIA H100 SXM5 80GB"),
    ("NVIDIA A100", "NVIDIA A100 SXM4 80GB"),          # an unqualified A100 never equals a fully named SKU
    ("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090"),
    ("NVIDIA L40S", "NVIDIA L40"),
])
def test_normalise_sku_distinct(a: str, b: str) -> None:
    assert not T.same_sku(a, b)


def test_normalise_sku_values() -> None:
    assert T.normalise_sku("NVIDIA A100-SXM4-80GB") == "a100-sxm4-80gb"
    assert T.normalise_sku("NVIDIA H100 80GB HBM3") == "h100-sxm5-80gb"
    assert T.normalise_sku("NVIDIA GeForce RTX 4090") == "rtx-4090"
    assert T.normalise_sku("NVIDIA A100") == "a100-unknown-unknown"
    assert T.normalise_sku("") is None and T.normalise_sku(None) is None and not T.same_sku(None, None)
    for target in TARGETS.values():
        assert T.normalise_sku(contract.native_peak_of(target).device) is not None and "unknown" not in T.normalise_sku(contract.native_peak_of(target).device)


# --- the frozen instance refs ------------------------------------------------------------------------------------------------------------


def test_frozen_instances_are_pinned() -> None:
    bf16 = T.FROZEN_INSTANCES[FIRST.name]
    assert bf16 == {"dataset": "bench-instances/v1", "tier": "vu-k1536", "range": [0, 4096], "manifest_sha256": T.BF16_INSTANCES_MANIFEST_SHA256}
    manifest = REPO / "fixtures" / "bench-instances" / "v1" / "manifest.json"
    if manifest.exists():
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == T.BF16_INSTANCES_MANIFEST_SHA256, "the BF16 fixture moved under the freeze"
    fp8 = T.FROZEN_INSTANCES[FP8_ADA_MMA.name]
    assert fp8["dataset"] == contract.FP8_ADA_INSTANCES_DATASET == "bench-instances-fp8-ada/v1"
    assert fp8["tier"] == contract.FP8_ADA_INSTANCES_TIER == "vu-k1536-fp8-ada" and fp8["range"] == [0, 4096]
    # chain.py::instances_digest(4096): sha256 of the recipe string, no data (n = B instances of STEPS * K = 1536 operand pairs)
    recipe = f"fp8-ada synthetic|seed={contract.FP8_ADA_INSTANCES_SEED}|n={T.B}|K={T.K}"
    assert contract.FP8_ADA_INSTANCES_SEED == 20260922
    assert fp8["manifest_sha256"] == contract.FP8_ADA_INSTANCES_DIGEST == hashlib.sha256(recipe.encode()).hexdigest()
    # the two Hopper synthetic sets (lane b-hopper's relations, frozen by lane b-merge-h100 with the dumped rows): the same
    # recipe digest, relchain.instances_digest(rel, 4096) = sha256("<relation> synthetic|seed=20260922|n=4096|K=1536")
    for target, rel, prefix in ((BF16_HOPPER, "bf16-hopper", "BF16_HOPPER"), (FP8_HOPPER_WGMMA, "fp8-hopper", "FP8_HOPPER")):
        ref = T.FROZEN_INSTANCES[target.name]
        dataset, tier, seed, digest = (getattr(contract, f"{prefix}_INSTANCES_{k}") for k in ("DATASET", "TIER", "SEED", "DIGEST"))
        assert ref["dataset"] == dataset == f"bench-instances-{rel}/v1" and ref["tier"] == tier == f"vu-k1536-{rel}" and ref["range"] == [0, 4096]
        assert seed == 20260922
        assert ref["manifest_sha256"] == digest == hashlib.sha256(f"{rel} synthetic|seed={seed}|n={T.B}|K={T.K}".encode()).hexdigest()
    assert T.frozen_instances(BF16_HOPPER) is not None and T.frozen_instances(FP8_HOPPER_WGMMA) is not None
    # the fp8-ada / bf16-hopper / fp8-hopper emitter is relchain.py at RELATIONS[...] (relations.py) since lane b-hopper; fp8/chain.py
    # is its shim.  relations.py must emit exactly the pinned refs (dataset + tier from the contract; the digest is the recipe's).
    relations = (REPO / "archive" / "direct" / "ligero" / "relations.py").read_text()
    for prefix in ("FP8_ADA", "BF16_HOPPER", "FP8_HOPPER"):
        assert f"contract.{prefix}_INSTANCES_DATASET" in relations and f"contract.{prefix}_INSTANCES_TIER" in relations, \
            f"relations.py must emit the frozen {prefix} ref"


# --- the predicate and Table 2 ----------------------------------------------------------------------------------------------------------


def test_valid_a100_bf16_agkr_populates_its_cell(store: LocalStore) -> None:
    art = put(store, result(t_total=4.0))
    tab = T.build(store)
    c = cell(tab)
    assert c is not None and c.artifact == art and tab.rejected == [] and tab.not_candidates == {}
    r_proved = 2 * T.K * T.B / 4.0
    assert c.r_proved == pytest.approx(r_proved) and c.overhead == pytest.approx(contract.native_peak_of(FIRST).flop_per_second / r_proved)
    assert c.proof_class == "NON_ZK_PROOF" and c.attempt == store.get_manifest(art).meta["run_id"]
    md = T.render_markdown(tab)
    row = next(l for l in md.splitlines() if l.startswith(f"| {contract.native_peak_of(FIRST).device} |"))
    assert T.fmt_overhead(c.overhead) in row and row.count(T.EMPTY) == 2, row     # B-Ligero and SP1 empty
    assert f"[1] {art}" in md, "each populated cell is a numbered footnote carrying its artifact id"
    assert re.fullmatch(r"\d\.\de\d+×", T.fmt_overhead(c.overhead)) and T.fmt_overhead(2.6e7) == "2.6e7×"
    assert cell(tab, FP8_ADA_MMA) is None and cell(tab, FP8_HOPPER_WGMMA) is None


def test_other_device_never_populates_the_row(store: LocalStore) -> None:
    art = put(store, result(gpu=H100))
    tab = T.build(store)
    assert cell(tab) is None
    reasons = rejected(tab, art)
    assert len(reasons) == 1 and reasons[0].startswith("same-SKU:") and "NVIDIA H100 80GB HBM3" in reasons[0] and contract.native_peak_of(FIRST).device in reasons[0]


def test_diagnostic_class_is_rejected(store: LocalStore) -> None:
    art = put(store, result(proof_class="ARITHMETIC_DIAGNOSTIC"))
    tab = T.build(store)
    assert cell(tab) is None
    assert rejected(tab, art) == ["proof_class 'ARITHMETIC_DIAGNOSTIC' is not A-GKR's declared class NON_ZK_PROOF_DIAGNOSTIC | NON_ZK_PROOF"]


def test_a_gkr_complete_non_zk_proof_is_its_declared_class(store: LocalStore) -> None:
    """The user's label for a complete non-ZK proof is NON_ZK_PROOF_DIAGNOSTIC; every A-GKR emitter writes it."""
    put(store, result(proof_class="NON_ZK_PROOF_DIAGNOSTIC"))
    assert cell(T.build(store)) is not None


def test_redteam_downgrade_is_rejected_and_a_later_reaudit_supersedes(store: LocalStore) -> None:
    art = put(store, result(), labels=[("proof_class", "ARITHMETIC_DIAGNOSTIC", REDTEAM, "2026-09-22T11:00:00Z")])
    tab = T.build(store)
    assert cell(tab) is None
    (reason,) = rejected(tab, art)
    assert reason.startswith("downgraded: red-team-x") and "proof_class=ARITHMETIC_DIAGNOSTIC, below NON_ZK_PROOF" in reason
    # a red team cannot upgrade; a later red-team event supersedes the verdict (latest-event-wins)
    store.add_label(art, "proof_class", "COMPLETE_ZK_BACKEND", "red-team-y", ts="2026-09-22T12:00:00Z")
    assert cell(T.build(store)) is not None
    # the lane asserting proof_class about its own result is not a red team and changes nothing
    art2 = put(store, result(t_total=3.0), labels=[("proof_class", "NON_ZK_PROOF_DIAGNOSTIC", LANE, "2026-09-22T11:00:00Z")])
    assert cell(T.build(store)).artifact == art2


def test_unverified_or_self_verified_is_rejected(store: LocalStore) -> None:
    art = put(store, result(), verified_by=None)
    (reason,) = rejected(T.build(store), art)
    assert reason.startswith("not independently verified") and "independently_verified=True|true or verified=accepted" in reason
    assert f"producers: {LANE}" in reason
    store.add_label(art, "independently_verified", True, LANE)                   # the producing lane vouching for itself
    assert rejected(T.build(store), art)[0].startswith("not independently verified")
    store.add_label(art, "independently_verified", False, VERIFIER)             # a verifier saying no
    assert rejected(T.build(store), art)[0].startswith("not independently verified")
    store.add_label(art, "verified", "accepted", "coordinator")                 # the spec's name for the label, by a third party
    assert cell(T.build(store)) is not None and cell(T.build(store)).artifact == art


def put_registered(store: LocalStore, res: dict, meta: dict | None = None, labels: list[tuple] = ()) -> str:
    """Store ``res`` as ``research data put --kind bench-result/v1 --meta @result.json`` does for a pod run: no attempt, only the
    meta the registering lane wrote (``lane``, ``by``, ...) and the labels it (or anyone) asserts."""

    art = store.put_artifact(T.RESULT_KIND, {**res, **(meta or {})})
    for key, value, by, ts in labels:
        store.add_label(art, key, value, by, ts=ts)
    return art


def test_hand_registered_result_is_attributed_to_the_lane_its_meta_names(store: LocalStore) -> None:
    """A result registered with ``research data put`` has no attempt: its producer is the lane its meta names (``meta.lane``, the
    registrar's ``meta.by``) or a ``lane`` label names.  That lane's own ``verified=accepted`` is a self-assertion (05:00Z render:
    the 4090 / 5090 B-Ligero cells); a non-producer's acceptance makes the cell."""

    kw = dict(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=RTX4090, proof_class="COMPLETE_ZK_BACKEND", t_total=0.17)
    art = put_registered(store, result(**kw), {"lane": "wave-4090-2"},
                         labels=[("verified", "accepted", "wave-4090-2", "2026-09-24T04:12:37Z"),
                                 ("verifier", "ligero-verify@e918ac8", "wave-4090-2", "2026-09-24T04:12:37Z")])
    assert store.index.attempts_by_output(art) == []
    tab = T.build(store)
    assert cell(tab, FP8_ADA_MMA, "B-Ligero") is None
    (reason,) = rejected(tab, art)
    assert reason.startswith("not independently verified") and "producers: wave-4090-2" in reason, reason
    store.add_label(art, "verified", "accepted", "verify-night", ts="2026-09-24T11:00:00Z")      # a non-producer re-verifies
    c = cell(T.build(store), FP8_ADA_MMA, "B-Ligero")
    assert c is not None and c.artifact == art and c.verified_by == "verify-night" and c.attempt is None

    # the registrar (meta.by) and the lane (meta.lane) are both producers; so is the lane of a prose `constructed_by`
    art2 = put_registered(store, result(**{**kw, "t_total": 0.16}), {"lane": "v3-scout", "by": "v3-scout-2"},
                          labels=[("verified", "accepted", "v3-scout-2", "2026-09-24T04:00:00Z")])
    assert "producers: v3-scout, v3-scout-2" in rejected(T.build(store), art2)[0]
    art3 = put_registered(store, result(**{**kw, "t_total": 0.15}), {"constructed_by": "fill-a100 (2026-09-22) from the attempt's records"},
                          labels=[("verified", "accepted", "fill-a100", "2026-09-24T04:00:00Z")])
    assert "producers: fill-a100" in rejected(T.build(store), art3)[0]

    # attribution by a `lane` label (meta carries none): the named lane is a producer, the label's asserter (a back-fill) is not
    art4 = put_registered(store, result(**{**kw, "t_total": 0.14}),
                          labels=[("lane", "wave-4090-3", "coordinator", "2026-09-24T05:00:00Z"),
                                  ("verified", "accepted", "wave-4090-3", "2026-09-24T05:01:00Z")])
    assert "producers: wave-4090-3" in rejected(T.build(store), art4)[0]
    store.add_label(art4, "verified", "accepted", "coordinator", ts="2026-09-24T05:02:00Z")
    c4 = cell(T.build(store), FP8_ADA_MMA, "B-Ligero")
    assert c4.artifact == art4 and c4.verified_by == "coordinator"


def test_hand_registered_result_naming_no_producer_is_not_independently_verified(store: LocalStore) -> None:
    """No attempt and no named producer: nobody can be shown independent of it, whoever accepted it."""

    art = put_registered(store, result(), labels=[("verified", "accepted", "verify-night", "2026-09-24T11:00:00Z")])
    (reason,) = rejected(T.build(store), art)
    assert reason.startswith("not independently verified: no attempt and no named producer"), reason
    store.add_label(art, "lane", "lane-a-gkr", "coordinator", ts="2026-09-24T11:05:00Z")          # attribution back-filled
    assert cell(T.build(store)).artifact == art and cell(T.build(store)).verified_by == "verify-night"


def test_every_frozen_constant_is_checked(store: LocalStore) -> None:
    bad_inst = {**T.FROZEN_INSTANCES[FIRST.name], "manifest_sha256": "a" * 64}
    cases = {
        put(store, result(K=1024)): "K is 1024",
        put(store, result(B=64)): "B is 64",
        put(store, result(B_real=4095)): "B_real 4095 != B 4096",
        put(store, result(security={"target": -100})): "security.target is -100",
        put(store, result(security={"target": -128, "achieved_log2": -127.9})): "security.achieved_log2 is -127.9",
        put(store, result(instances=None)): "instances is null",
        put(store, result(instances=bad_inst)): "instances differ from the frozen set on manifest_sha256",
        put(store, result(instances={**T.FROZEN_INSTANCES[FIRST.name], "range": [4096, 8192]})): "instances differ from the frozen set on range",
        put(store, result(instances={**T.FROZEN_INSTANCES[FIRST.name], "tier": "vu-k1536-neg"})): "instances differ from the frozen set on tier",
        put(store, result(authentication="included")): "authentication 'included'",
    }
    tab = T.build(store)
    assert cell(tab) is None and cell(tab, FP8_ADA_MMA) is None
    for art, fragment in cases.items():
        assert any(fragment in r for r in rejected(tab, art)), (fragment, rejected(tab, art))
    # the row is the fingerprint's profile: an A100 result that claims the 4090 target lands in that row and fails its SKU + instances
    art = put(store, result(profile=FP8_ADA_MMA.name))
    entry = next(r for r in T.build(store).rejected if r["artifact"] == art)
    assert entry["target"] == FP8_ADA_MMA.name and any(r.startswith("same-SKU") for r in entry["reasons"])
    assert any("instances differ from the frozen set on dataset" in r for r in entry["reasons"])
    # a contract-non-conforming result names the contract first
    res = result()
    res["measurements"] = [m for m in res["measurements"] if m["name"] != "t.total"]
    art = put(store, res)
    assert rejected(T.build(store), art)[0].startswith("contract: ")


def test_hopper_rows_need_the_frozen_synthetic_set(store: LocalStore) -> None:
    # a Hopper result that does not name its instances is rejected for that (the sets ARE frozen now: the recipe digests)
    art = put(store, result(target=FP8_HOPPER_WGMMA, gpu="NVIDIA H100 80GB HBM3", instances=None))
    reasons = rejected(T.build(store), art)
    assert any("instances is null" in r for r in reasons) and not any("has no frozen input set" in r for r in reasons)
    # ... one on another seed's set differs from the frozen ref on its digest ...
    other = dict(T.FROZEN_INSTANCES[BF16_HOPPER.name], manifest_sha256="ab" * 32)
    art2 = put(store, result(target=BF16_HOPPER, gpu="NVIDIA H100 80GB HBM3", instances=other))
    assert any("instances differ from the frozen set on manifest_sha256" in r for r in rejected(T.build(store), art2))
    # ... one carrying the pinned ref on the row's SKU without a verifier's acceptance is left with that requirement only ...
    art3 = put(store, result(target=BF16_HOPPER, gpu="NVIDIA H100 80GB HBM3", candidate="B-Ligero", proof_class="COMPLETE_ZK_BACKEND"),
               verified_by=None)
    reasons3 = rejected(T.build(store), art3)
    assert len(reasons3) == 1 and reasons3[0].startswith("not independently verified"), reasons3
    # ... and, accepted by a non-producer, it populates the Hopper BF16 x B-Ligero cell
    art4 = put(store, result(target=BF16_HOPPER, gpu="NVIDIA H100 80GB HBM3", candidate="B-Ligero", proof_class="COMPLETE_ZK_BACKEND", t_total=1.6))
    c = cell(T.build(store), BF16_HOPPER, "B-Ligero")
    assert c is not None and c.artifact == art4 and c.overhead == pytest.approx(contract.native_peak_of(BF16_HOPPER).flop_per_second / (2 * T.K * T.B / 1.6))
    # a target with no frozen set at all still says so (the predicate's own wording, on a target outside FROZEN_INSTANCES)
    unfrozen = dataclasses.replace(FP8_HOPPER_WGMMA, name="no-frozen-set/2026-09-23")
    assert T.frozen_instances(unfrozen) is None
    res = result(target=FP8_HOPPER_WGMMA, gpu="NVIDIA H100 80GB HBM3")
    cand = T.candidate_for(res["workload_fingerprint"]["software"]["backend"]["name"])
    assert any("has no frozen input set yet" in r for r in T.reject_reasons(res, [], None, unfrozen, cand))


def test_best_of_two_is_min_t_total(store: LocalStore) -> None:
    slow = put(store, result(t_total=5.0))
    fast = put(store, result(t_total=3.0))
    tab = T.build(store)
    assert cell(tab).artifact == fast and cell(tab).t_total == 3.0
    assert tab.also_valid == [{"artifact": slow, "candidate": "A-GKR", "target": FIRST.name, "authentication": "excluded", "t_total": 5.0,
                               "best": fast}]
    assert len(tab.cells) == 1 and tab.rejected == []
    assert "Valid but not the best" in T.render_markdown(tab)


def test_included_hash_fills_its_own_column(store: LocalStore) -> None:
    """An ``authentication = included-hash`` result is a second Table 2 column (``<candidate> + in-proof hash``), never the bare
    cell; the column only appears once such a cell exists; Table 3 labels the committed row."""
    bare = put(store, result(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=RTX4090, proof_class="COMPLETE_ZK_BACKEND", t_total=1.0))
    tab = T.build(store)
    assert T.table2_columns(tab) == T.TABLE2_COLUMNS and T.committed_candidates(tab) == []
    doc = result(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=RTX4090, proof_class="COMPLETE_ZK_BACKEND", t_total=2.5)
    doc["workload_fingerprint"].update(authentication="included-hash", hash="poseidon2-babybear-w24", sharing="none")
    com = put(store, doc)
    tab = T.build(store)
    row = next(r for r in tab.rows if r["target"] == FP8_ADA_MMA.name)
    assert row["cells"]["B-Ligero"].artifact == bare and row["committed"]["B-Ligero"].artifact == com
    assert row["committed"]["B-Ligero"].authentication == "included-hash" and row["committed"]["B-Ligero"].hash_name == "poseidon2-babybear-w24"
    assert T.table2_columns(tab) == (*T.TABLE2_COLUMNS, "B-Ligero + in-proof hash") and tab.rejected == []
    md = T.render_markdown(tab)
    assert "B-Ligero + in-proof hash" in md and "authentication included-hash (hash poseidon2-babybear-w24, sharing none)" in md
    js = T.render_json(tab)
    assert js["table2"]["columns"][-1] == "B-Ligero + in-proof hash" and js["table2"]["authentication_modes"] == ["excluded", "included-hash"]
    assert {c.artifact for c in tab.cells} == {bare, com}
    # a committed result never fills the bare cell, even when it is the only result
    store2_bare_missing = [r for r in tab.rows if r["target"] == FP8_ADA_MMA.name][0]["cells"]["B-Ligero"]
    assert store2_bare_missing.artifact == bare


#: The leaf interface block ``relchain`` writes for a hashed run (lane leaf-iface D0), per scheme.
LEAVES = {
    "poseidon2": ("poseidon2-babybear-w24", {"scheme": "poseidon2", "suffix": "poseidon2", "schema": "poseidon2-babybear-w24/row/v2h", "digest_elems": 8}),
    "ajtai-n64": ("ajtai-babybear-n64-b1-r0/v1", {"scheme": "ajtai-n64", "suffix": "ajtai-n64", "schema": "ajtai-babybear-n64-b1-r0/row/v1", "digest_elems": 64}),
    "blake3": ("blake3-keyed", {"scheme": "blake3", "suffix": "blake3", "schema": "blake3-keyed/row/v2", "digest_elems": 8}),
}


def hashed(leaf: str | None = "poseidon2", sharing: str | None = "none", authentication: str = "included-hash", **kw: Any) -> dict[str, Any]:
    """A column-2 candidate result on the 4090 row: ``included-hash`` with ``leaf``'s hash (``None``: the pre-D0 form, hash only)."""

    doc = result(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=RTX4090, proof_class="COMPLETE_ZK_BACKEND", **kw)
    fp = doc["workload_fingerprint"]
    fp.update(authentication=authentication, hash=LEAVES[leaf or "poseidon2"][0])
    if leaf is not None:
        fp["leaf"] = dict(LEAVES[leaf][1])
    if sharing is not None:
        fp["sharing"] = sharing
    return doc


def test_in_proof_hash_column_is_poseidon2_per_row_without_sharing(store: LocalStore) -> None:
    """Column ``B-Ligero + in-proof hash`` is Poseidon2-BabyBear per row, no sharing (user decision keep_frozen, 2026-09-24): an
    Ajtai or BLAKE3 leaf, or a shared tile, is rejected with the reason saying which (05:00Z render: an Ajtai leaf filled the 4090
    cell).  Both fingerprint forms of a Poseidon2 row hash -- ``hash`` only (pre-D0) and ``hash`` + ``leaf`` (D0) -- are accepted."""

    old = put(store, hashed(leaf=None, t_total=0.70))
    new = put(store, hashed(t_total=0.69))
    ajtai = put(store, hashed("ajtai-n64", t_total=0.35))
    blake = put(store, hashed("blake3", t_total=0.40))
    tile = put(store, hashed(sharing="tile64x64", t_total=0.30))
    shared = put(store, hashed(sharing="tile64x64", authentication="included-hash-shared", t_total=0.20))
    unstated = put(store, hashed(sharing=None, t_total=0.50))
    tab = T.build(store)
    row = next(r for r in tab.rows if r["target"] == FP8_ADA_MMA.name)
    assert row["committed"]["B-Ligero"].artifact == new and {a["artifact"] for a in tab.also_valid} == {old}
    (r_ajtai,) = rejected(tab, ajtai)
    assert "is an Ajtai (Ring-SIS) leaf" in r_ajtai and "ajtai-babybear-n64-b1-r0/v1" in r_ajtai and "poseidon2-babybear-w24 per row" in r_ajtai
    (r_blake,) = rejected(tab, blake)
    assert "is a BLAKE3 in-circuit leaf" in r_blake and "'blake3-keyed'" in r_blake
    (r_tile,) = rejected(tab, tile)
    assert r_tile.startswith("sharing 'tile64x64'") and "a shared tile is a drill-down" in r_tile
    (r_shared,) = rejected(tab, shared)
    assert r_shared.startswith("authentication 'included-hash-shared'")
    assert rejected(tab, unstated) == [r for r in rejected(tab, unstated) if r.startswith("sharing None")] != []
    # the letters the drill-down prints: X for another leaf, S for sharing (either spelling), X for an unstated sharing
    cand = T.candidate_for("b-ligero chain")
    codes = {a: [c for c, _ in T.reject_codes(store.get_manifest(a).meta, store.labels(a), None, FP8_ADA_MMA, cand,
                                               authentication=("included-hash" if a != shared else "excluded"))]
             for a in (ajtai, blake, tile, shared, unstated, new)}
    assert codes == {ajtai: ["X"], blake: ["X"], tile: ["S"], shared: ["S"], unstated: ["X"], new: []}, codes
    # a Poseidon2 schema of another version or a leaf block naming another scheme is not the column's either
    fp = hashed()["workload_fingerprint"]
    assert T.in_proof_hash_codes(fp) == []
    assert [c for c, _ in T.in_proof_hash_codes({**fp, "leaf": {**fp["leaf"], "scheme": "ajtai-n64"}})] == ["X"]
    assert [c for c, _ in T.in_proof_hash_codes({**fp, "hash": None, "leaf": None})] == ["X"]
    assert "is no named leaf" in T.in_proof_hash_codes({**fp, "hash": None, "leaf": None})[0][1]


def test_reject_codes_letter_every_reason(store: LocalStore) -> None:
    """Every reason carries the drill-down's letter (D2): U I P S Z K B X; ``reject_reasons`` is ``reject_codes`` without them."""

    over = result(buckets={"witness": 3.0, "encoding_commitment": 2.0}, t_total=4.0)        # 5 s of buckets in a 4 s total
    twice = result(buckets={"witness": 1.0, "arithmetic": 1.0}, joints={("witness", "arithmetic"): 1.0}, t_total=4.0)
    cand = T.CANDIDATES[0]
    small = result()
    small["workload_fingerprint"].update(B=64, instances={**T.FROZEN_INSTANCES[FIRST.name], "range": [0, 64]})
    cases = {"P": over, "B": small, "Z": result(proof_class="ARITHMETIC_DIAGNOSTIC"), "K": result(gpu=H100),
             "I": result(instances=None), "X": result(K=1024)}
    for letter, res in cases.items():
        codes = T.reject_codes(res, [{"key": "verified", "value": "accepted", "by": VERIFIER, "ts": "t"}],
                               attempt(res["run_id"], "art:x", LANE), FIRST, cand)
        assert codes and codes[0][0] == letter, (letter, codes)
    assert T.reject_codes(twice, [], attempt(twice["run_id"], "art:x", LANE), FIRST, cand)[0][0] == "P"
    assert T.PHASE_SUM_PROBLEM.match(contract.validate(over)[0]) and T.PHASE_SUM_PROBLEM.match(contract.validate(twice)[0])
    u = T.reject_codes(result(), [], attempt("r", "art:x", LANE), FIRST, cand)
    assert u == [("U", T.reject_reasons(result(), [], attempt("r", "art:x", LANE), FIRST, cand)[0])]
    assert set(T.REASON_LETTERS) == set("UIPSZKBX")


# --- re-packed instances: instance-equiv/v1 (user decision 2026-09-24; campaign morning-tables BRIEF §5) ----------------------------


def repacked(n: int = 0) -> dict[str, Any]:
    """A re-packed instance ref of the 4090 row (another layout of the frozen VUs), distinct per ``n``."""
    return {"dataset": "bench-instances-fp8-ada-x4/v1", "tier": "vu-k1536-fp8-ada-x4", "range": [0, 4096], "manifest_sha256": f"{n:064x}",
            "seed": 20260922, "recipe": "the frozen fp8-ada VUs re-packed four per column"}


def equiv_doc(candidate: dict[str, Any], target=FP8_ADA_MMA, **over: Any) -> dict[str, Any]:
    arrays = {a: {"frozen_sha256": h * 64, "candidate_sha256": h * 64, "dtype": "e4m3 u8", "shape": [4096, 1536]} for a, h in (("x", "1"), ("W", "2"), ("y", "3"))}
    doc = {"schema": T.INSTANCE_EQUIV_SCHEMA, "target": target.name, "frozen": copy.deepcopy(T.FROZEN_INSTANCES[target.name]),
           "candidate": copy.deepcopy(candidate), "canonical": "per array: E4M3 bit words as u8, index order VU-major then K",
           "arrays": arrays, "equal": True, "tool": "verity_numerical.bench.instance_equiv@0000000"}
    doc.update(over)
    return doc


def put_equiv(store: LocalStore, doc: dict[str, Any], *, lane: str | None = "fused-phases", verdicts: list[tuple] = (("accepted", "verify-night"),),
              payload: Path | None = None) -> str:
    """Register ``doc`` as fused-phases would (``research data put --kind instance-equiv/v1 --meta @doc.json``, or ``--file`` when
    ``payload`` is a path to write it to) and give it ``verdicts`` = ``(verified value, by)`` in order."""

    attribution = {"lane": lane} if lane else {}
    if payload is not None:
        payload.write_text(json.dumps(doc), encoding="utf-8")
        art = store.put_artifact(T.INSTANCE_EQUIV_SCHEMA, attribution, file=payload)
    else:
        art = store.put_artifact(T.INSTANCE_EQUIV_SCHEMA, {**doc, **attribution})
    for i, (value, by) in enumerate(verdicts):
        store.add_label(art, "verified", value, by, ts=f"2026-09-24T12:0{i}:00Z")
    return art


def repacked_result(store: LocalStore, n: int, t_total: float = 0.2) -> str:
    return put(store, result(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=RTX4090, proof_class="COMPLETE_ZK_BACKEND", t_total=t_total,
                             instances=repacked(n)))


def test_repacked_instances_count_only_with_an_accepted_equivalence(store: LocalStore, tmp_path: Path) -> None:
    art = repacked_result(store, 1)
    (reason,) = rejected(T.build(store), art)
    assert reason.startswith("instances differ from the frozen set on dataset") and "instance-equiv" not in reason   # the existing rejection
    eq = put_equiv(store, equiv_doc(repacked(1)))
    tab = T.build(store)
    c = cell(tab, FP8_ADA_MMA, "B-Ligero")
    assert c is not None and c.artifact == art and c.instances_equiv == eq and tab.rejected == []
    assert f"instances re-packed; decoded values byte-identical to the frozen set ({eq})" in T.render_markdown(tab)
    assert T.render_json(tab)["table2"]["rows"][3]["cells"]["B-Ligero"]["instances_equiv"] == eq
    # the document as the artifact's JSON payload (meta carries only the attribution) counts the same
    art2 = repacked_result(store, 2, t_total=0.1)
    eq2 = put_equiv(store, equiv_doc(repacked(2)), payload=tmp_path / "equiv.json")
    assert cell(T.build(store), FP8_ADA_MMA, "B-Ligero").instances_equiv == eq2 and cell(T.build(store), FP8_ADA_MMA, "B-Ligero").artifact == art2
    # a frozen-set result needs none and names none
    frozen = put(store, result(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=RTX4090, proof_class="COMPLETE_ZK_BACKEND", t_total=0.05))
    assert cell(T.build(store), FP8_ADA_MMA, "B-Ligero").artifact == frozen and cell(T.build(store), FP8_ADA_MMA, "B-Ligero").instances_equiv is None


DEFECTS = {
    "unverified": (dict(), dict(verdicts=()), "has no verified=accepted by a non-producer"),
    "self-verified by its producer": (dict(), dict(verdicts=(("accepted", "fused-phases"),)), "has no verified=accepted by a non-producer"),
    "verified by the result's producer": (dict(), dict(verdicts=(("accepted", LANE),)), "has no verified=accepted by a non-producer"),
    "a later rejection": (dict(), dict(verdicts=(("accepted", "verify-night"), ("rejected", "coordinator"))), "latest verdict by a non-producer is verified=rejected"),
    "no named producer": (dict(), dict(lane=None), "names no producer"),
    "digests differ": (dict(arrays={"x": {"frozen_sha256": "1" * 64, "candidate_sha256": "1" * 64}, "W": {"frozen_sha256": "2" * 64, "candidate_sha256": "2" * 64},
                                    "y": {"frozen_sha256": "3" * 64, "candidate_sha256": "4" * 64}}), dict(), "arrays.y: frozen_sha256"),
    "a placeholder digest": (dict(arrays={a: {"frozen_sha256": "…", "candidate_sha256": "…"} for a in "xWy"}), dict(), "arrays.W: frozen_sha256 '…'"),
    "y missing": (dict(arrays={a: {"frozen_sha256": "1" * 64, "candidate_sha256": "1" * 64} for a in "xW"}), dict(), "arrays lack y"),
    "equal false": (dict(equal=False), dict(), "verified, says equal=False"),
    "equal as a word": (dict(equal="true"), dict(), "verified, says equal='true'"),
    "equal false, unverified": (dict(equal=False), dict(verdicts=()), "equal is False"),
    "another frozen ref": (dict(frozen=copy.deepcopy(T.FROZEN_INSTANCES[FIRST.name])), dict(), "frozen is not FROZEN_INSTANCES[target]"),
    "another schema": (dict(schema="instance-equiv/v0"), dict(), "schema is 'instance-equiv/v0'"),
}


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_repacked_instances_stay_rejected_without_a_sound_equivalence(store: LocalStore, defect: str) -> None:
    over, how, fragment = DEFECTS[defect]
    art = repacked_result(store, 7)
    eq = put_equiv(store, equiv_doc(repacked(7), **over), **how)
    tab = T.build(store)
    assert cell(tab, FP8_ADA_MMA, "B-Ligero") is None
    (reason,) = rejected(tab, art)
    assert reason.startswith("instances differ from the frozen set on dataset") and f"(instance-equiv: {eq}" in reason and fragment in reason, reason


def test_an_equivalence_of_another_pair_or_a_contradiction_does_not_count(store: LocalStore) -> None:
    art = repacked_result(store, 3)
    put_equiv(store, equiv_doc(repacked(4)))                                            # another candidate ref
    put_equiv(store, equiv_doc(repacked(3), target=BF16_HOPPER))                        # another target's frozen set
    (reason,) = rejected(T.build(store), art)
    assert "instance-equiv" not in reason, reason                                       # neither is about this pair: the existing rejection
    good = put_equiv(store, equiv_doc(repacked(3)))
    assert cell(T.build(store), FP8_ADA_MMA, "B-Ligero").instances_equiv == good
    bad = put_equiv(store, equiv_doc(repacked(3), equal=False, arrays={a: {"frozen_sha256": "1" * 64, "candidate_sha256": "2" * 64} for a in "xWy"}),
                    lane="verify-night-2", verdicts=(("accepted", "coordinator"),))
    (reason,) = rejected(T.build(store), art)
    assert f"{bad}, verified, says equal=False" in reason, reason


def test_fp8_ada_row_and_hvzk_drilldown(store: LocalStore) -> None:
    zk = put(store, result(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=RTX4090, proof_class="COMPLETE_ZK_BACKEND", t_total=8.0))
    hv = put(store, result(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=RTX4090, proof_class="COMPLETE_HVZK_BACKEND", t_total=6.0))
    put(store, result(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=A100, proof_class="COMPLETE_ZK_BACKEND", t_total=1.0))   # wrong SKU
    tab = T.build(store)
    c = cell(tab, FP8_ADA_MMA, "B-Ligero")
    assert c.artifact == zk and c.overhead == pytest.approx(contract.native_peak_of(FP8_ADA_MMA).flop_per_second / (2 * T.K * T.B / 8.0))
    assert [(d["candidate"], d["proof_class"], d["artifact"]) for d in tab.drilldown] == [("B-Ligero", "COMPLETE_HVZK_BACKEND", hv)]
    assert len(tab.rejected) == 2 and all(r["candidate"] == "B-Ligero" for r in tab.rejected)
    md = T.render_markdown(tab)
    assert "drill-down B-Ligero COMPLETE_HVZK_BACKEND" in md
    row = next(l for l in md.splitlines() if l.startswith(f"| {contract.native_peak_of(FP8_ADA_MMA).device} |"))
    assert "E4M3" in row and FP8_ADA_MMA.instruction in row


def test_sp1_accepts_any_sound_class(store: LocalStore) -> None:
    art = put(store, result(candidate="SP1", proof_class="COMPLETE_HVZK_BACKEND"))
    tab = T.build(store)
    assert cell(tab, FIRST, "SP1").artifact == art
    bad = put(store, result(candidate="SP1", proof_class="ARITHMETIC_DIAGNOSTIC"))
    assert any("is not SP1's declared class NON_ZK_PROOF | COMPLETE_HVZK_BACKEND | COMPLETE_ZK_BACKEND" in r for r in rejected(T.build(store), bad))


def test_component_benchmarks_are_not_candidates(store: LocalStore) -> None:
    res = result()
    res["workload_fingerprint"]["software"]["backend"]["name"] = "verity-direct (plonky3 uni-stark)"
    art = put(store, res)
    tab = T.build(store)
    assert tab.not_candidates == {"verity-direct (plonky3 uni-stark)": [art]} and tab.rejected == []
    assert "Not a Table 1 candidate (1 result" in T.render_markdown(tab)


def test_native_measured_column(store: LocalStore) -> None:
    spec = contract.native_peak_of(FIRST).flop_per_second
    slow = native_peak(store, "NVIDIA A100-SXM4-80GB", "bf16", 0.90 * spec, spec)
    best = native_peak(store, "NVIDIA A100 SXM4 80GB", "bf16", 0.94 * spec, spec)
    native_peak(store, "NVIDIA A100-SXM4-80GB", "e4m3", 0.99 * spec, spec)     # another dtype: not this row
    hspec = contract.native_peak_of(BF16_HOPPER).flop_per_second
    h100 = native_peak(store, "NVIDIA H100 80GB HBM3", "bf16", 0.95 * hspec, hspec)   # another device: the Hopper BF16 row's, not FIRST's
    tab = T.build(store)
    row = next(r for r in tab.rows if r["target"] == FIRST.name)
    assert row["native_measured"]["artifact"] == best and row["native_measured"]["fraction_of_spec"] == pytest.approx(0.94)
    hopper = next(r for r in tab.rows if r["target"] == BF16_HOPPER.name)      # joined by device x dtype (lane b-hopper's target)
    assert hopper["native_measured"]["artifact"] == h100 and hopper["native_measured"]["fraction_of_spec"] == pytest.approx(0.95)
    assert all(r["native_measured"] is None for r in tab.rows if r["target"] not in (FIRST.name, BF16_HOPPER.name))
    md = T.render_markdown(tab)
    line = next(l for l in md.splitlines() if l.startswith(f"| {contract.native_peak_of(FIRST).device} |"))
    assert f"| {0.94 * spec / 1e12:.0f} T (0.94) [1] |" in line and T.UNMEASURED in md
    assert T.cell_artifacts(tab) == sorted([best, h100]) and slow not in T.cell_artifacts(tab)


def test_native_measured_block_scaled_format_names_its_operand_dtype(store: LocalStore) -> None:
    """``benchmarks/native_peak --dtype nvfp4`` measures the sm_120 block-scaled instruction on E2M1 operands: it is the measured
    peak of the ``e2m1`` NVFP4 target (``NATIVE_DTYPE_ALIASES``), and of no other row."""
    from verity.ml.circuits.targets.target import NVFP4_SM120

    spec = contract.native_peak_of(NVFP4_SM120).flop_per_second
    assert T.dtype_of(NVFP4_SM120) == "e2m1" and T.native_dtype({"dtype": "nvfp4"}) == "e2m1" and T.native_dtype({"dtype": "bf16"}) == "bf16"
    art = native_peak(store, "NVIDIA GeForce RTX 5090", "nvfp4", 1.12 * spec, spec)
    native_peak(store, "NVIDIA GeForce RTX 5090", "e4m3", 0.5 * spec, spec)     # another dtype: not this row
    tab = T.build(store)
    row = next(r for r in tab.rows if r["target"] == NVFP4_SM120.name)
    assert row["native_measured"]["artifact"] == art and row["native_measured"]["fraction_of_spec"] == pytest.approx(1.12)
    assert all(r["native_measured"] is None for r in tab.rows if r["target"] != NVFP4_SM120.name)


def test_native_measured_prefers_the_gemm_over_an_instruction_microbenchmark(store: LocalStore) -> None:
    """Lane fp4-proof recorded two RTX 5090 FP4 peaks: the instruction issued back to back (1880 T = 1.12x the sheet, the
    sustained clock above the rated boost) and cuBLASLt's block-scaled GEMM (1283 T).  The column is one methodology across
    rows -- the vendor-library GEMM -- so it takes the GEMM even though the microbenchmark is faster; the microbenchmark is
    the cell only when no GEMM exists, and its footnote says so."""
    from verity.ml.circuits.targets.target import NVFP4_SM120

    spec = contract.native_peak_of(NVFP4_SM120).flop_per_second
    method = ("instruction microbenchmark: mma.sync.aligned.kind::mxf4nvf4 issued back to back from 2720 warps; "
              "no memory traffic -- an upper bound on any GEMM built from it")     # mentions GEMMs without being one
    mma = store.put_artifact(T.NATIVE_PEAK_KIND, {"device": "NVIDIA GeForce RTX 5090", "dtype": "nvfp4", "accumulate": "fp32",
                                                  "flop_per_second": 1.12 * spec, "method": method})
    assert not T.gemm_method({"method": method}) and T.gemm_method({"method": T.NATIVE_PEAK_METHOD_CUBLAS})
    tab = T.build(store)
    row = next(r for r in tab.rows if r["target"] == NVFP4_SM120.name)
    assert row["native_measured"]["artifact"] == mma and not row["native_measured"]["gemm"]
    line = next(l for l in T.render_markdown(tab).splitlines() if l.startswith("[") and mma in l)
    assert "INSTRUCTION-LEVEL MICROBENCHMARK" in line
    # the same tool also ran the library GEMM beside the instruction (fp4-proof's native_peak.json shape): the column takes it
    both = store.put_artifact(T.NATIVE_PEAK_KIND, {"device": "NVIDIA GeForce RTX 5090", "dtype": "nvfp4", "accumulate": "fp32",
                                                   "flop_per_second": 1.12 * spec, "method": method,
                                                   "gemm_attempt": {"attempted": True, "supported": True, "flop_per_second": 0.77 * spec, "n": 8192,
                                                                    "variant": "_scaled_mm blockwise scales", "seconds_all": [1e-3] * 5,
                                                                    "note": "unit scales, random nibbles; throughput only"}})
    tab = T.build(store)
    row = next(r for r in tab.rows if r["target"] == NVFP4_SM120.name)
    assert row["native_measured"]["artifact"] == both and row["native_measured"]["fraction_of_spec"] == pytest.approx(0.77)
    assert row["native_measured"]["gemm"] and "n=8192" in row["native_measured"]["method"]
    md = T.render_markdown(tab)
    assert "INSTRUCTION-LEVEL" not in md and f"| {0.77 * spec / 1e12:.0f} T (0.77) " in md
    # a stand-alone cuBLAS measurement beats the beside-the-instruction one only on FLOP/s, never a microbenchmark on method
    gemm = store.put_artifact(T.NATIVE_PEAK_KIND, {"device": "NVIDIA GeForce RTX 5090", "dtype": "nvfp4", "accumulate": "fp32",
                                                   "flop_per_second": 0.80 * spec, "method": T.NATIVE_PEAK_METHOD_CUBLAS})
    tab = T.build(store)
    assert next(r for r in tab.rows if r["target"] == NVFP4_SM120.name)["native_measured"]["artifact"] == gemm


# --- Table 3 -------------------------------------------------------------------------------------------------------------------------


def test_table3_percentages_and_other(store: LocalStore) -> None:
    buckets = {"witness": 1.0, "encoding_commitment": 1.5, "arithmetic": 0.6, "lookup": 0.4, "zk_additional": 0.0, "serialization": 0.1}
    art = put(store, result(t_total=4.0, buckets=buckets))
    tab = T.build(store)
    ph = cell(tab).phases
    assert ph["total"] == 4.0 and ph["buckets"] == buckets and ph["joints"] == {} and ph["other"] == pytest.approx(0.4)
    assert sum(ph["buckets"].values()) + ph["other"] == pytest.approx(ph["total"]), "T_total = Σ buckets + other"
    md = T.render_markdown(tab)
    row = next(l for l in md.splitlines() if l.startswith(f"| {contract.native_peak_of(FIRST).device} / {FIRST.name} |"))
    assert row == (f"| {contract.native_peak_of(FIRST).device} / {FIRST.name} | A-GKR | 4 s | 1 s (25%) | 1.5 s (38%) | 0.6 s (15%) | 0.4 s (10%) | 0 s (0%) "
                   f"| 0.1 s (2%) | 0.4 s (10%) |"), row
    assert T.TABLE3_COLUMNS == ("Device / target", "Candidate", "Total prover time", "Witness", "Encoding + commitment", "Arithmetic", "Lookup",
                                "ZK additional", "Serialization", "Other")
    assert art == json.loads(json.dumps(T.render_json(tab)))["table3"]["rows"][0]["artifact"]


def test_joint_stage_renders_once(store: LocalStore) -> None:
    put(store, result(t_total=4.0, buckets={"witness": 1.0, "lookup": 0.5, "zk_additional": 0.0, "serialization": 0.25},
                      joints={("encoding_commitment", "arithmetic"): 2.0}))
    tab = T.build(store)
    ph = cell(tab).phases
    assert ph["joints"] == {"t.joint.encoding_commitment+arithmetic": {"covers": ["encoding_commitment", "arithmetic"], "seconds": 2.0}}
    assert ph["buckets"]["encoding_commitment"] is None and ph["buckets"]["arithmetic"] is None and ph["other"] == pytest.approx(0.25)
    md = T.render_markdown(tab)
    row = next(l for l in md.splitlines() if l.startswith(f"| {contract.native_peak_of(FIRST).device} / {FIRST.name} |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[3:] == ["1 s (25%)", "encoding_commitment+arithmetic: 2 s (50%)", T.JOINT_CONTINUATION, "0.5 s (12%)", "0 s (0%)", "0.25 s (6%)",
                         "0.25 s (6%)"], cells
    assert md.count("encoding_commitment+arithmetic: 2 s (50%)") == 1


# --- CLI: json round trip and snapshot ---------------------------------------------------------------------------------------------------


def test_json_round_trips(store: LocalStore, capsys: pytest.CaptureFixture[str]) -> None:
    art = put(store, result(t_total=4.0))
    bad = put(store, result(gpu=H100))
    assert T.main(["--root", str(store.root), "--format", "json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == "verity/canonical-tables/v1" and out["frozen"] == "2026-09-22" and out["caption"]["K"] == 1536 and out["caption"]["B"] == 4096
    assert out["table2"]["columns"] == ["Device", "Datatype / target", "Native dense peak, spec", "Native dense peak, measured", "A-GKR", "B-Ligero", "SP1"]
    row = next(r for r in out["table2"]["rows"] if r["target"] == FIRST.name)
    c = row["cells"]["A-GKR"]
    assert c["artifact"] == art and c["r_proved"] == pytest.approx(2 * T.K * T.B / 4.0) and row["cells"]["B-Ligero"] is None
    assert [r["artifact"] for r in out["rejected"]] == [bad] and out["rejected"][0]["reasons"][0].startswith("same-SKU")
    assert out["frozen_instances"] == T.FROZEN_INSTANCES and out["snapshot"] is None
    # the object is exactly what the library returns (json is lossless for it)
    assert out == json.loads(json.dumps(T.render_json(T.build(store)), sort_keys=True, default=str))


def test_snapshot_freezes_exactly_the_cell_artifacts(store: LocalStore, capsys: pytest.CaptureFixture[str]) -> None:
    a = put(store, result(t_total=4.0))
    put(store, result(t_total=6.0))                                            # valid, not the best: not a member
    put(store, result(gpu=H100))                                               # rejected: not a member
    sp1 = put(store, result(candidate="SP1", proof_class="NON_ZK_PROOF", t_total=40.0))
    spec = contract.native_peak_of(FIRST).flop_per_second
    peak = native_peak(store, "NVIDIA A100-SXM4-80GB", "bf16", 0.94 * spec, spec)
    assert T.main(["--root", str(store.root), "--snapshot"]) == 0
    out, err = capsys.readouterr()
    m = re.search(r"^Snapshot: (art:[0-9a-f]{64})$", out, re.M)
    assert m and m.group(1) in err
    snap = store.get_manifest(m.group(1))
    assert snap.kind == "dataset-snapshot/v1" and snap.meta["name"] == T.snapshot_name() and snap.meta["name"].startswith("tables-2026-")
    assert snap.refs["members"] == sorted([a, sp1, peak])
    kinds = {store.get_manifest(x).kind for x in snap.refs["members"]}
    assert kinds == {T.RESULT_KIND, T.NATIVE_PEAK_KIND}
    # the snapshot id is content: freezing the same tables again yields the same artifact
    assert T.main(["--root", str(store.root), "--snapshot", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["snapshot"] == m.group(1)


def test_snapshot_of_nothing_is_refused(store: LocalStore, capsys: pytest.CaptureFixture[str]) -> None:
    assert T.main(["--root", str(store.root), "--snapshot"]) == 1
    assert "no populated cell" in capsys.readouterr().err
    assert not [r for r in store.index.select_artifacts(kind="dataset-snapshot/v1")]


def test_markdown_has_the_three_tables_and_the_rejected_list(store: LocalStore, capsys: pytest.CaptureFixture[str]) -> None:
    bad = put(store, result(proof_class="ARITHMETIC_DIAGNOSTIC"))
    assert T.main(["--root", str(store.root)]) == 0
    md = capsys.readouterr().out
    assert md.count("\n## Table ") + md.startswith("## Table ") == 3
    for heading in ("## Table 1 — Candidate proof systems (frozen 2026-09-22)", "## Table 2 — Headline proving overhead by target",
                    "## Table 3 — Prover phase decomposition"):
        assert heading in md
    assert "K = 1536; B = 4096; soundness 2^-128" in md and "authentication = excluded" in md
    assert f"- {bad} (A-GKR / {FIRST.name}) → proof_class 'ARITHMETIC_DIAGNOSTIC'" in md
    assert "(no populated Table 2 cell" in md


def test_label_keys_are_the_store_vocabulary() -> None:
    """The predicate reads labels by the keys pinned in ``research.store.vocab`` (and the ``ImportError`` fallback spells them the same);
    the ledger's closed sets are the vocabulary's enums."""
    from research.store import vocab
    from verity_numerical.bench import ledger, ledger_to_store
    from verity_numerical.bench import views as V

    assert T.PRODUCER_LABEL_KEYS == (vocab.CANDIDATE, vocab.LABEL) == ("candidate", "label")
    assert T.PRODUCER_NAME_LABEL_KEYS == (vocab.LANE,) == ("lane",)
    assert T.VERIFIED_LABELS == ((vocab.INDEPENDENTLY_VERIFIED, (True, "true")), (vocab.VERIFIED, (vocab.VERIFIED_ACCEPTED,)))
    assert (vocab.INDEPENDENTLY_VERIFIED, vocab.VERIFIED, vocab.VERIFIED_ACCEPTED) == ("independently_verified", "verified", "accepted")
    assert vocab.key(vocab.INDEPENDENTLY_VERIFIED).status == vocab.DEPRECATED and vocab.key(vocab.VERIFIED_BY).status == vocab.IGNORED
    assert set(vocab.PROOF_CLASSES) == set(ledger.PROOF_CLASSES) == set(contract.PROOF_CLASSES) | {"COMPONENT"}
    assert set(vocab.AUTHENTICATION_VALUES) == set(contract.AUTHENTICATION)
    assert set(vocab.SCOPES) == set(ledger.SCOPES) and set(vocab.TRACKS) == set(ledger.TRACKS)
    assert set(vocab.BACKEND_FAMILIES) == set(V.FAMILY_NAMES.values()) == {V.column_header(f.id) for f in V.FAMILIES}
    assert set(vocab.CANDIDATES) == set(vocab.BACKEND_FAMILIES) | {c for _, c in ledger_to_store.CANDIDATES} | {ledger_to_store.candidate("some-other-lane")}
    assert set(T.RANK) <= set(vocab.PROOF_CLASSES)


# --- verification provenance in the footnotes (D7) -----------------------------------------------------------------------------------


def test_footnote_names_the_verifier_and_flags_interactive_transcripts(store: LocalStore) -> None:
    """An interactive transcript can only be replayed against the runner's step-0 coins (D7: live-verifier-only evidence); the
    footnote says so, names the asserter, and a Fiat-Shamir transcript is called transferable.  The mode comes from
    ``security.bound_model`` when the emitter writes no ``mode``."""
    inter = put(store, result(t_total=4.0, security={"target": -128, "achieved_log2": -128.3, "field": "BabyBear",
                                                     "bound_model": "interactive: statistical, live verifier coins committed at step 0 (HM96)"}))
    tab = T.build(store)
    c = cell(tab)
    assert c is not None and c.artifact == inter and c.mode == "interactive" and c.verified_by == VERIFIER
    md = T.render_markdown(tab)
    assert f"verified by {VERIFIER} against the runner's step-0 coins (interactive transcript: not transferable evidence per D7" in md
    fs = put(store, result(t_total=3.0, security={"target": -128, "achieved_log2": -128.3, "field": "BabyBear",
                                                  "bound_model": "non-interactive: eps_FS <= 2^60 x eps_round (BCS16)"}))
    tab = T.build(store)
    c = cell(tab)
    assert c is not None and c.artifact == fs and c.mode == "fiat-shamir"
    assert f"verified by {VERIFIER} (Fiat–Shamir transcript: transferable)" in T.render_markdown(tab)
    assert T._mode_of({"mode": "interactive"}) == "interactive" and T._mode_of({"mode": "fiat-shamir"}) == "fiat-shamir"
    assert T._mode_of({"security": {"bound_model": 3}}) is None and T._mode_of({}) is None


# --- device variants (same nvidia-smi name, not the datasheet part) ----------------------------------------------------------------


def test_a_rememoried_board_of_the_same_name_is_not_a_cell(store: LocalStore) -> None:
    """2026-09-22: a RunPod "RTX 4090" reported 48 GiB; the datasheet part has 24 GiB.  Same name, so ``same_sku`` passes; the
    memory check rejects it, names both figures, and the reference part (within tolerance) still populates."""
    hw = lambda mem: {"gpu": {"name": A100, "driver": "d", "cuda": "c", "memory_total_bytes": mem}, "cpu": {"model": "AMD EPYC 7763 64-Core Processor", "count": 256}}  # noqa: E731
    bad = put(store, result(t_total=3.0, hardware=hw(2 * 80 * 2**30)))
    good = put(store, result(t_total=4.0, hardware=hw(85_520_809_984)))     # what torch reports for an 80 GiB part
    tab = T.build(store)
    c = cell(tab)
    assert c is not None and c.artifact == good
    reasons = rejected(tab, bad)
    assert len(reasons) == 1 and reasons[0].startswith("device variant: 160.0 GiB of device memory on") and "has 80 GiB" in reasons[0]
    assert c.host == "AMD EPYC 7763 64-Core Processor x256"
    assert f"on {A100} (host AMD EPYC 7763 64-Core Processor x256); attempt" in T.render_markdown(tab)
    assert T.device_variant({"name": A100}, FIRST) is None                  # no memory reported: unchecked
    assert T.device_variant({"name": A100, "memory_total_bytes": 0}, FIRST) is None
    assert contract.native_peak_of(FIRST).memory_bytes == 80 * 2**30 and contract.native_peak_of(FP8_ADA_MMA).memory_bytes == 24 * 2**30
    assert "device_variant" in T.CAPTION


# --- the coordinator's cell marks: the latest `finding` label (lanes/coordinator/evidence/marked-cells.py) ---------------------------


OPEN = "UNDER RE-VERIFICATION: soundness issue R1/R2 (red-team-standard-hash): the verifier lets the prover choose the leaf pairing"


def test_an_open_finding_marks_the_cell_and_cleared_drops_it(store: LocalStore) -> None:
    art = put(store, result(t_total=4.0), labels=[("finding", OPEN, "coordinator", "2026-09-25T07:45:04Z")])
    tab = T.build(store)
    c = cell(tab)
    assert c.artifact == art and c.finding == f"{OPEN} (coordinator, 2026-09-25T07:45:04Z)"
    md = T.render_markdown(tab)
    assert f"{T.fmt_overhead(c.overhead)} (under re-verification) [" in md and f"; {OPEN} (coordinator" in md
    assert re.search(r"^\[\d+\] art:[0-9a-f]{64} — ", md, re.M)          # marked-cells.py still finds the footnote
    store.add_label(art, "finding", "CLEARED R1/R2: fixed verifier re-accepts", "coordinator", ts="2026-09-25T09:00:00Z")
    assert cell(T.build(store)).finding is None and "(under re-verification) [" not in T.render_markdown(T.build(store))


def test_a_pulled_finding_rejects_the_result_like_a_downgrade(store: LocalStore) -> None:
    fast = put(store, result(t_total=2.0), labels=[("finding", OPEN, "coordinator", "2026-09-25T07:45:04Z"),
                                                   ("finding", "PULLED R1/R2: forged pairing accepted", "coordinator", "2026-09-25T09:00:00Z")])
    slow = put(store, result(t_total=3.0))
    tab = T.build(store)
    assert cell(tab).artifact == slow
    (reason,) = rejected(tab, fast)
    assert reason == "pulled: PULLED R1/R2: forged pairing accepted (coordinator, 2026-09-25T09:00:00Z)"
    codes = T.reject_codes(store.get_manifest(fast).meta, store.labels(fast), store.index.attempt(store.get_manifest(fast).meta["run_id"]),
                           FIRST, T.CANDIDATES[0])
    assert [k for k, _ in codes] == ["Z"]


def test_other_findings_change_nothing(store: LocalStore) -> None:
    art = put(store, result(t_total=4.0), labels=[("finding", OPEN, "coordinator", "2026-09-25T07:45:04Z"),
                                                  ("finding", "HOLDS (red-team-lk): equivalent", "red-team-lk", "2026-09-25T08:00:00Z")])
    c = cell(T.build(store))
    assert c.artifact == art and c.finding is None
    assert T.cell_finding([]) == (None, None)


def test_a_4096_equivalence_may_carry_the_runners_seed_and_recipe(store: LocalStore) -> None:
    """One instance-equiv/v1 shape for 4,096 and stream points: the frozen side is the runner's instance ref, compared on
    dataset, tier, range and manifest_sha256 (the extra seed and recipe fields change nothing)."""
    art = repacked_result(store, 3)
    frozen = {**T.FROZEN_INSTANCES[FP8_ADA_MMA.name], "seed": 20260922, "recipe": "fp8-ada synthetic"}
    eq = put_equiv(store, equiv_doc(repacked(3), frozen=frozen))
    assert cell(T.build(store), FP8_ADA_MMA, "B-Ligero").artifact == art and cell(T.build(store), FP8_ADA_MMA, "B-Ligero").instances_equiv == eq
