"""The three canonical tables (frozen 2026-09-22): the validity predicate, best-of-N cells, the same-SKU invariant, Table 3's
identity and joint display, the SKU normaliser, the JSON round trip and the snapshot -- all over an isolated store root."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from verity.verification.target import FIRST, FP8_ADA_MMA, FP8_HOPPER_WGMMA, TARGETS
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

    backend = {"A-GKR": "verity-gkr", "B-Ligero": "b-ligero chain (backends/direct/ligero)", "SP1": "sp1"}[candidate]
    if instances == "frozen":
        instances = copy.deepcopy(T.FROZEN_INSTANCES[target.name])
    fp = contract.fingerprint(
        profile=target.name, domain="finite", K=T.K, B=T.B, proof_class=proof_class, authentication="excluded",
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
    return store.put_artifact(T.NATIVE_PEAK_KIND, {"device": device, "dtype": dtype, "accumulate": "fp32", "method": T.NATIVE_PEAK_METHOD,
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
    assert T.candidate_for("b-ligero chain (backends/direct/ligero)").id == "B-Ligero"
    assert T.candidate_for("b-ligero chain, fp8-ada relation (backends/direct/ligero/fp8)").id == "B-Ligero"
    assert T.candidate_for("sp1").id == T.candidate_for("SP1 (v4.1, cuda prover)").id == "SP1"
    for name in ("verity-direct (plonky3 uni-stark)", "backends/gkr/packed/sumcheck_packed", "b-encode RS encoders (cupy RawKernel NTT)",
                 "backends/gkr/tensor/fused", "sp1x", "verity-gkr2", "b-ligero", "a-gkr", None, 3):
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
        assert T.normalise_sku(target.native_peak.device) is not None and "unknown" not in T.normalise_sku(target.native_peak.device)


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
    assert T.frozen_instances(FP8_HOPPER_WGMMA) is None, "Hopper has no frozen instance set yet"
    chain = (REPO / "backends" / "direct" / "ligero" / "fp8" / "chain.py").read_text()
    assert "contract.FP8_ADA_INSTANCES_DATASET" in chain and "contract.FP8_ADA_INSTANCES_TIER" in chain, "chain.py must emit the frozen ref"


# --- the predicate and Table 2 ----------------------------------------------------------------------------------------------------------


def test_valid_a100_bf16_agkr_populates_its_cell(store: LocalStore) -> None:
    art = put(store, result(t_total=4.0))
    tab = T.build(store)
    c = cell(tab)
    assert c is not None and c.artifact == art and tab.rejected == [] and tab.not_candidates == {}
    r_proved = 2 * T.K * T.B / 4.0
    assert c.r_proved == pytest.approx(r_proved) and c.overhead == pytest.approx(FIRST.native_peak.flop_per_second / r_proved)
    assert c.proof_class == "NON_ZK_PROOF" and c.attempt == store.get_manifest(art).meta["run_id"]
    md = T.render_markdown(tab)
    row = next(l for l in md.splitlines() if l.startswith(f"| {FIRST.native_peak.device} |"))
    assert T.fmt_overhead(c.overhead) in row and row.count(T.EMPTY) == 2, row     # B-Ligero and SP1 empty
    assert f"[1] {art}" in md, "each populated cell is a numbered footnote carrying its artifact id"
    assert re.fullmatch(r"\d\.\de\d+×", T.fmt_overhead(c.overhead)) and T.fmt_overhead(2.6e7) == "2.6e7×"
    assert cell(tab, FP8_ADA_MMA) is None and cell(tab, FP8_HOPPER_WGMMA) is None


def test_other_device_never_populates_the_row(store: LocalStore) -> None:
    art = put(store, result(gpu=H100))
    tab = T.build(store)
    assert cell(tab) is None
    reasons = rejected(tab, art)
    assert len(reasons) == 1 and reasons[0].startswith("same-SKU:") and "NVIDIA H100 80GB HBM3" in reasons[0] and FIRST.native_peak.device in reasons[0]


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


def test_hopper_has_no_frozen_set_yet(store: LocalStore) -> None:
    art = put(store, result(target=FP8_HOPPER_WGMMA, gpu="NVIDIA H100 80GB HBM3", instances=None))
    assert any("has no frozen instance set yet" in r for r in rejected(T.build(store), art))


def test_best_of_two_is_min_t_total(store: LocalStore) -> None:
    slow = put(store, result(t_total=5.0))
    fast = put(store, result(t_total=3.0))
    tab = T.build(store)
    assert cell(tab).artifact == fast and cell(tab).t_total == 3.0
    assert tab.also_valid == [{"artifact": slow, "candidate": "A-GKR", "target": FIRST.name, "t_total": 5.0, "best": fast}]
    assert len(tab.cells) == 1 and tab.rejected == []
    assert "Valid but not the best" in T.render_markdown(tab)


def test_fp8_ada_row_and_hvzk_drilldown(store: LocalStore) -> None:
    zk = put(store, result(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=RTX4090, proof_class="COMPLETE_ZK_BACKEND", t_total=8.0))
    hv = put(store, result(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=RTX4090, proof_class="COMPLETE_HVZK_BACKEND", t_total=6.0))
    put(store, result(candidate="B-Ligero", target=FP8_ADA_MMA, gpu=A100, proof_class="COMPLETE_ZK_BACKEND", t_total=1.0))   # wrong SKU
    tab = T.build(store)
    c = cell(tab, FP8_ADA_MMA, "B-Ligero")
    assert c.artifact == zk and c.overhead == pytest.approx(FP8_ADA_MMA.native_peak.flop_per_second / (2 * T.K * T.B / 8.0))
    assert [(d["candidate"], d["proof_class"], d["artifact"]) for d in tab.drilldown] == [("B-Ligero", "COMPLETE_HVZK_BACKEND", hv)]
    assert len(tab.rejected) == 2 and all(r["candidate"] == "B-Ligero" for r in tab.rejected)
    md = T.render_markdown(tab)
    assert "drill-down B-Ligero COMPLETE_HVZK_BACKEND" in md
    row = next(l for l in md.splitlines() if l.startswith(f"| {FP8_ADA_MMA.native_peak.device} |"))
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
    spec = FIRST.native_peak.flop_per_second
    slow = native_peak(store, "NVIDIA A100-SXM4-80GB", "bf16", 0.90 * spec, spec)
    best = native_peak(store, "NVIDIA A100 SXM4 80GB", "bf16", 0.94 * spec, spec)
    native_peak(store, "NVIDIA A100-SXM4-80GB", "e4m3", 0.99 * spec, spec)     # another dtype: not this row
    native_peak(store, "NVIDIA H100 80GB HBM3", "bf16", 0.95 * spec, spec)     # another device: not this row
    tab = T.build(store)
    row = next(r for r in tab.rows if r["target"] == FIRST.name)
    assert row["native_measured"]["artifact"] == best and row["native_measured"]["fraction_of_spec"] == pytest.approx(0.94)
    assert all(r["native_measured"] is None for r in tab.rows if r["target"] != FIRST.name)
    md = T.render_markdown(tab)
    line = next(l for l in md.splitlines() if l.startswith(f"| {FIRST.native_peak.device} |"))
    assert f"| {0.94 * spec / 1e12:.0f} T (0.94) [1] |" in line and T.UNMEASURED in md
    assert T.cell_artifacts(tab) == [best] and slow not in T.cell_artifacts(tab)


# --- Table 3 -------------------------------------------------------------------------------------------------------------------------


def test_table3_percentages_and_other(store: LocalStore) -> None:
    buckets = {"witness": 1.0, "encoding_commitment": 1.5, "arithmetic": 0.6, "lookup": 0.4, "zk_additional": 0.0, "serialization": 0.1}
    art = put(store, result(t_total=4.0, buckets=buckets))
    tab = T.build(store)
    ph = cell(tab).phases
    assert ph["total"] == 4.0 and ph["buckets"] == buckets and ph["joints"] == {} and ph["other"] == pytest.approx(0.4)
    assert sum(ph["buckets"].values()) + ph["other"] == pytest.approx(ph["total"]), "T_total = Σ buckets + other"
    md = T.render_markdown(tab)
    row = next(l for l in md.splitlines() if l.startswith(f"| {FIRST.native_peak.device} / {FIRST.name} |"))
    assert row == (f"| {FIRST.native_peak.device} / {FIRST.name} | A-GKR | 4 s | 1 s (25%) | 1.5 s (38%) | 0.6 s (15%) | 0.4 s (10%) | 0 s (0%) "
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
    row = next(l for l in md.splitlines() if l.startswith(f"| {FIRST.native_peak.device} / {FIRST.name} |"))
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
    spec = FIRST.native_peak.flop_per_second
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

    assert T.PRODUCER_LABEL_KEYS == (vocab.CANDIDATE, vocab.LABEL) == ("candidate", "label")
    assert T.VERIFIED_LABELS == ((vocab.INDEPENDENTLY_VERIFIED, (True, "true")), (vocab.VERIFIED, (vocab.VERIFIED_ACCEPTED,)))
    assert (vocab.INDEPENDENTLY_VERIFIED, vocab.VERIFIED, vocab.VERIFIED_ACCEPTED) == ("independently_verified", "verified", "accepted")
    assert vocab.key(vocab.INDEPENDENTLY_VERIFIED).status == vocab.DEPRECATED and vocab.key(vocab.VERIFIED_BY).status == vocab.IGNORED
    assert set(vocab.PROOF_CLASSES) == set(ledger.PROOF_CLASSES) == set(contract.PROOF_CLASSES) | {"COMPONENT"}
    assert set(vocab.AUTHENTICATION_VALUES) == set(contract.AUTHENTICATION)
    assert set(vocab.SCOPES) == set(ledger.SCOPES) and set(vocab.TRACKS) == set(ledger.TRACKS)
    assert set(vocab.CANDIDATES) == {c for _, c in ledger_to_store.CANDIDATES} | {ledger_to_store.candidate("some-other-lane")}
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
    assert FIRST.native_peak.memory_bytes == 80 * 2**30 and FP8_ADA_MMA.native_peak.memory_bytes == 24 * 2**30
    assert "device_variant" in T.CAPTION
