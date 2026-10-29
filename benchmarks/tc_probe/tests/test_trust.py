"""``benchmarks/tc_probe/trust.py``: trust is computed from store evidence and repo fixtures, never asserted.

Synthetic evidence in an isolated store root lifts one target (FP8_ADA_MMA over ``sm89.mma.m16n8k32.e4m3``, anchor RTX 4090) from
UNPROBED through HYPOTHESIS and PINNED to TRUSTED and TRUSTED_STABLE as artifacts and labels are added; an open D-entry, a red-team
downgrade label or a failing canary block or re-open it; the dossier round-trips through JSON and publishes as ``trust-dossier/v1``;
the canary tile selection is deterministic.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("research.store.local")
from research.store.local import LocalStore

TC_PROBE = Path(__file__).resolve().parents[1]
if str(TC_PROBE) not in sys.path:
    sys.path.insert(0, str(TC_PROBE))

import trust  # noqa: E402
from trust import (  # noqa: E402
    DOSSIER_KIND, HONEST_MIN, INVALIDATION_TRIGGERS, N_MIN, NEGATIVES_MIN, REGRESSION_TILES, RELITIGATION_PROCEDURE, Level, TrustReport,
    assess, publish, render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET = "fp8-ada-mma-draft/2026-09-22"
INSTRUCTION = "sm89.mma.m16n8k32.e4m3"
MODEL = "ada_e4m3_m16n8k32"
ANCHOR = "rtx4090"
GOLDEN = "verity/ml/circuits/ops/tests/fixtures/golden/ada_e4m3_m16n8k32.json"
FAMILIES = ("randn_zero_acc", "randn_random_acc", "uniform_bits", "adder_width_cancellation", "zero_products_nonzero_acc",
            "acc_only_lsb_walk", "specials")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "absent-store.toml"))
    monkeypatch.setenv("RESEARCH_REPO", str(tmp_path / "repo"))


@pytest.fixture
def repo(tmp_path) -> Path:
    """A synthetic repository: a discrepancy log with one resolved entry naming the target, the reference files the catalogue
    expects (presence only), and one frozen input set with an edge family and a negative tier."""
    root = tmp_path / "repo"
    root.mkdir()
    write_log(root, [dlog("D10", "resolved", evidence_refs=[GOLDEN])])
    for ref in trust.INDEPENDENT_REFERENCES[INSTRUCTION]:
        for p in ref.paths:
            (root / p).parent.mkdir(parents=True, exist_ok=True)
            (root / p).write_text("{}")
    man = root / "fixtures" / "fp8-instances" / "v1" / "manifest.json"
    man.parent.mkdir(parents=True)
    man.write_text(json.dumps({"dataset": "fp8-instances", "target": {"profile": TARGET, "pipeline": MODEL},
                               "tiers": {"vu-k1536": {"composition": {"corner_families": {"cancellation": 64, "near_overflow": 64}}},
                                         "vu-k1536-neg": {}}}))
    return root


def dlog(did: str, status: str, **over) -> dict:
    e = {"id": did, "profile": TARGET, "issue": f"{did} issue", "status": status, "resolution": "hardware decided" if status == "resolved" else None,
         "evidence_refs": [], "owner": "test", "to_resolve": "measure it" if status == "open" else None}
    e.update(over)
    return e


def write_log(root: Path, entries: list[dict]) -> None:
    log = root / "verity" / "ml" / "circuits" / "silicon" / "discrepancy_log.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"schema": "verity/discrepancy-log/v0", "entries": entries}))


@pytest.fixture
def store(tmp_path, repo) -> LocalStore:
    return LocalStore(tmp_path / "store", repo=repo)


def hw(store: LocalStore, *, hardware: str = ANCHOR, elements: int = N_MIN, mismatches: int = 0, families=FAMILIES, replay: str | None = GOLDEN,
       sass: bool = True, host: str = "gpu-0", driver: str = "550.90", nvcc: str = "12.4", agrees_label: bool | None = None, extra_labels=(), canary: bool = False) -> str:
    """One synthetic ``tc-evidence/v1`` in tc-probe-fp8's nested shape, labelled with the contract keys."""
    asserted = [f for f in families if f != "specials"]
    per = -(-elements // max(1, len(asserted)))  # ceil: the asserted families sum to >= elements
    fams = {f: {"elements": per, "asserted": f != "specials", "mismatches": {MODEL: mismatches if f != "specials" else 0}} for f in families}
    counts = {"families": fams, "total_mismatches_by_model": {MODEL: mismatches}, "model_agrees": {MODEL: mismatches == 0}}
    te = {"instruction": INSTRUCTION, "sku": "NVIDIA GeForce RTX 4090" if hardware == ANCHOR else "NVIDIA H100 SXM5 80GB", "arch": "sm_89",
          "driver": driver, "nvcc": nvcc, "seed": 20260902, "models": {MODEL: {"declared": True, "pipeline": MODEL, "product": "e4m3", "lowering": None}}}
    if canary:
        te["canary"] = {**counts, "replays": [], "digest": "x" * 64}
    else:
        te.update(counts)
    if replay:
        te["golden_replay"] = {"file": replay, "records": 360, "mismatches": 0, "pipeline": MODEL}
    if sass:
        te["sass"] = {"ok": True, "single_instruction": True, "tile_kernel_mma_count": 1, "expected_mma_count": 1}
    meta = {"schema": "research/result/v0.1", "kind": "tc-evidence/v1", "run_id": f"r20260922-1200{len(store.artifact_ids()):02d}-{hashlib.sha1(host.encode()).hexdigest()[:4]}",
            "workload_fingerprint": {"instruction": INSTRUCTION, "sku": te["sku"], "seed": 20260902, "hostname": host, **({"family": "canary"} if canary else {})},
            "measurements": [{"name": "elements", "value": sum(f["elements"] for f in fams.values() if f["asserted"]), "unit": "elements"}],
            "validation": {"status": "passed" if mismatches == 0 else "failed", "checker": f"tc_probe vs {MODEL}"}, "tc_evidence": te, "host": host}
    art = store.put_artifact("tc-evidence/v1", meta)
    labels = {"instruction": INSTRUCTION, "model": MODEL, "hardware": hardware, "evidence": "sweep",
              "model_agrees": "true" if (agrees_label if agrees_label is not None else mismatches == 0) else "false"}
    if canary:
        labels["family"] = "canary"
    for k, v in labels.items():
        store.add_label(art, k, v, "test")
    for k, v in extra_labels:
        store.add_label(art, k, v, "test")
    return art


def instances_hw(store: LocalStore, *, hardware: str = ANCHOR, mismatches: int = 0) -> str:
    """The hardware-in-the-loop record under the ``instances-hw`` label contract."""
    meta = {"schema": "research/result/v0.1", "kind": "tc-evidence/v1", "workload_fingerprint": {"instruction": INSTRUCTION},
            "validation": {"status": "passed" if mismatches == 0 else "failed", "checker": "tc_probe --replay-instances"},
            "tc_evidence": {"instruction": INSTRUCTION, "sku": "NVIDIA GeForce RTX 4090"}}
    art = store.put_artifact("tc-evidence/v1", meta)
    for k, v in (("instruction", INSTRUCTION), ("hardware", hardware), ("family", "frozen-instances"), ("instances_dataset", "fp8-instances"),
                 ("instances_tier", "vu-k1536"), ("mismatches", str(mismatches))):
        store.add_label(art, k, v, "test")
    return art


def relation(store: LocalStore, *, honest: int = 4096, negatives: int = 128, verified: bool = False, proof_class: str | None = None, status: str = "passed") -> str:
    meta = {"schema": "research/result/v0.1", "kind": "bench-result/v1", "workload_fingerprint": {"relation": f"fp8-ada-k1536 ({INSTRUCTION})", "checker": "prove+verify"},
            "validation": {"status": status, "checker": "checker.sweep" if proof_class is None else "independent verifier",
                           "detail": f"vu-k1536 {honest}/{honest} ok; vu-k1536-neg {negatives}/{negatives} ok"}}
    art = store.put_artifact("bench-result/v1", meta)
    store.add_label(art, "target", TARGET, "test")
    store.add_label(art, "relation", "fp8-ada-k1536", "test")
    if verified:
        store.add_label(art, "independently_verified", "true", "test")
    if proof_class:
        store.add_label(art, "proof_class", proof_class, "test")
    return art


def crit(rep: TrustReport, cid: str):
    return rep.criterion(cid)


# --- levels --------------------------------------------------------------------------------------------------------------------


def test_empty_store_is_unprobed(store, repo):
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.level == "UNPROBED"
    assert rep.subject == TARGET and rep.instruction == INSTRUCTION and rep.anchor_family == ANCHOR
    assert crit(rep, "H1").passed and not crit(rep, "H2").passed
    assert rep.next_level == "HYPOTHESIS"
    assert any("no hardware record" in m for m in rep.missing_to_next)


def test_agreement_off_anchor_is_a_hypothesis(store, repo):
    art = hw(store, hardware="h100", elements=1000)
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.level == "HYPOTHESIS"
    assert any(e.id == art for e in crit(rep, "H2").evidence)
    assert not crit(rep, "P1").passed
    assert any(ANCHOR in m and "h100" in m for m in rep.missing_to_next)


def test_partial_agreement_on_the_anchor_is_a_hypothesis_not_a_pin(store, repo):
    fams = {f: {"elements": 100_000, "asserted": True, "mismatches": {MODEL: 0 if f == "randn_zero_acc" else 7}} for f in FAMILIES[:3]}
    meta = {"kind": "tc-evidence/v1", "workload_fingerprint": {"instruction": INSTRUCTION}, "validation": {"status": "failed"},
            "tc_evidence": {"instruction": INSTRUCTION, "sku": "NVIDIA GeForce RTX 4090", "families": fams, "total_mismatches_by_model": {MODEL: 14}}}
    art = store.put_artifact("tc-evidence/v1", meta)
    for k, v in (("instruction", INSTRUCTION), ("hardware", ANCHOR), ("model", MODEL), ("model_agrees", "false")):
        store.add_label(art, k, v, "test")
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.level == "HYPOTHESIS"
    assert "randn_zero_acc" in crit(rep, "H2").evidence[0].note
    assert not crit(rep, "P1").passed and any("14 mismatches" in m for m in crit(rep, "P1").missing)


def test_full_anchor_sweep_pins(store, repo):
    art = hw(store)
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.level == "PINNED", rep.missing_to_next
    for cid in ("P1", "P2", "P3", "P4", "P5"):
        assert crit(rep, cid).passed, cid
    assert crit(rep, "P2").details["elements"] >= N_MIN
    assert crit(rep, "P3").details["adder-width"] == art
    # the resolved D10 regression tiles (golden replay + two families) are all replayed by that artifact
    assert set(REGRESSION_TILES["D10"]) <= {t for t in crit(rep, "P5").details.get("D10", [])} or crit(rep, "P5").passed
    assert rep.sass["single_instruction"] is True and rep.sass["source"] == art
    # what TRUSTED still needs, spelled out
    assert rep.next_level == "TRUSTED"
    joined = "\n".join(rep.missing_to_next)
    assert f"need {HONEST_MIN:,}" in joined and f"need {NEGATIVES_MIN}" in joined
    assert "family=frozen-instances" in joined
    assert "independently_verified=true" in joined


def test_a_disagreeing_record_cited_by_a_resolved_d_entry_is_superseded(store, repo):
    """fp4-re's situation: the sweep that *found* the discrepancy disagrees with the model as it stood; once the D-entry citing that run
    is resolved and a fresh sweep reproduces every word, the old record no longer blocks PINNED (it stays in the evidence as SUPERSEDED)."""
    bad = hw(store, mismatches=29_118, elements=N_MIN // 4, host="gpu-old")
    bad_run = store.get_artifact(bad)["meta"]["run_id"] if hasattr(store, "get_artifact") else None
    hw(store, host="gpu-new")                                        # the re-sweep under the corrected model
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.level == "HYPOTHESIS" and not crit(rep, "P1").passed  # the finding is not yet explained
    run_id = bad_run or next(r for r in trust.Evidence.collect(store).hardware if r.id == bad).run_id
    write_log(repo, [dlog("D10", "resolved", evidence_refs=[GOLDEN]), dlog("D99", "resolved", evidence_refs=[run_id])])
    trust.REGRESSION_TILES.setdefault("D99", ("family:randn_random_acc",))
    try:
        rep = assess("FP8_ADA_MMA", store, repo=repo)
    finally:
        trust.REGRESSION_TILES.pop("D99", None)
    assert crit(rep, "P1").passed and crit(rep, "P1").details["superseded"] == {bad: ["D99"]}
    assert rep.level == "PINNED", rep.missing_to_next
    assert any("SUPERSEDED" in e.note for e in crit(rep, "P1").evidence)


def test_fp4_probe_envelope_attributes_the_primary_model_and_ignores_the_control(store, repo, monkeypatch):
    """tc_probe_fp4 scores the registry model ("verity") and autoproof's refuted model5 side by side; the record's mismatches for the
    registry model come from the primary key, model5's count is not attributed, and a model5-only sweep is evidence about nothing here."""
    ins = "sm120.mma.m16n8k64.e2m1.nvf4"
    registry = trust.INSTRUCTIONS[ins].model.name
    fams = {"random_zero_acc": {"elements": 1000, "mismatches": {"model5": 7, "verity": 0}, "record_only": False},
            "specials": {"elements": 10, "mismatches": {"model5": 3, "verity": 3}, "record_only": True}}
    meta = {"schema": "research/result/v0.1", "kind": "tc-evidence/v1", "run_id": "r20260922-000000-fp4a", "instruction": ins, "sku": "NVIDIA GeForce RTX 5090",
            "arch": "sm_120a", "primary_model": "verity", "counts": {"elements": 1010, "mismatches": {"model5": 7, "verity": 0}}, "families": fams,
            "workload_fingerprint": {"instruction": ins}, "validation": {"status": "passed"}}
    art = store.put_artifact("tc-evidence/v1", meta)
    old = {"schema": "research/result/v0.1", "kind": "tc-evidence/v1", "run_id": "r20260922-000000-fp4b", "instruction": ins, "sku": "NVIDIA GeForce RTX 5090",
           "arch": "sm_120a", "counts": {"elements": 500, "mismatches": {"model5": 29118}, "replay_cases": 1601, "replay_matched": 1601},
           "families": {"random_zero_acc": {"elements": 500, "mismatches": {"model5": 29118}}}, "workload_fingerprint": {"instruction": ins, "replays": ["probe3.json"]},
           "validation": {"status": "failed"}}
    art_old = store.put_artifact("tc-evidence/v1", old)
    for a in (art, art_old):
        for k, v in (("instruction", ins), ("hardware", "rtx5090"), ("evidence", "sweep")):
            store.add_label(a, k, v, "test")
    recs = {r.id: r for r in trust.Evidence.collect(store).hardware}
    names = trust.subjects(ins).model_names
    assert recs[art].mismatches_for(names) == 0 and recs[art].agrees(names)
    assert recs[art].families["random_zero_acc"]["mismatches"][trust._norm_model(registry)] == 0
    assert recs[art].families["specials"]["asserted"] is False
    assert recs[art_old].mismatches_for(names) is None
    assert not recs[art_old].agrees(names) and recs[art_old].partially_agrees(names) is None  # the autoproof replay does not vouch for the new model


def test_too_few_elements_or_a_missing_family_blocks_pinned(store, repo):
    hw(store, elements=N_MIN // 10, families=FAMILIES[:-2])
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.level == "HYPOTHESIS"
    assert not crit(rep, "P2").passed and any("more 0-mismatch elements" in m for m in rep.missing_to_next)
    assert not crit(rep, "P3").passed and any("accumulator-lsb-walk" in m for m in rep.missing_to_next)


def test_trusted_needs_relation_instances_reference_and_verifier(store, repo):
    hw(store)
    relation(store)                                                   # Tc: honest 4096 / negatives 128 from the checker sweep detail
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.level == "PINNED"
    assert crit(rep, "Tc").passed and crit(rep, "Ta").passed and crit(rep, "Tb").passed
    assert not crit(rep, "Td").passed and not crit(rep, "Te").passed
    instances_hw(store)                                               # Td: frozen-instances replay on the anchor
    relation(store, verified=True, proof_class="COMPLETE_ZK_BACKEND")  # Te: an independent verifier accepted a full-contract proof
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.level == "TRUSTED", rep.missing_to_next
    assert crit(rep, "Td").passed and crit(rep, "Te").passed
    assert rep.next_level == "TRUSTED_STABLE" and any("second" in m for m in rep.missing_to_next)


def test_second_device_makes_it_stable(store, repo):
    hw(store)
    relation(store)
    instances_hw(store)
    relation(store, verified=True, proof_class="COMPLETE_ZK_BACKEND")
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.level == "TRUSTED" and crit(rep, "S1").details["hosts_without_gpu_uuid"] == ["gpu-0"]   # hostnames are not devices
    hw(store, host="gpu-1", elements=6656, canary=True, extra_labels=(("device", "GPU-1111"),))          # a canary on a 2nd GPU UUID ...
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.level == "TRUSTED" and crit(rep, "S1").details["devices"] == ["GPU-1111"]                 # ... is one identified device
    store.add_label(hw(store, elements=1000), "device", "GPU-0000", "test")                              # the first device identified
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.level == "TRUSTED_STABLE"
    assert crit(rep, "S1").details["devices"] == ["GPU-0000", "GPU-1111"]
    assert rep.next_level is None and rep.missing_to_next == []


def test_second_driver_environment_makes_it_stable(store, repo):
    trusted(store)
    hw(store, elements=6656, canary=True, driver="580.126.16", nvcc="12.6")
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.level == "TRUSTED_STABLE"
    assert len(crit(rep, "S1").details["environments"]) == 2


def test_instances_replay_off_anchor_does_not_count(store, repo):
    hw(store)
    relation(store)
    relation(store, verified=True, proof_class="COMPLETE_ZK_BACKEND")
    instances_hw(store, hardware="h100")
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert not crit(rep, "Td").passed
    assert any("ran on h100" in m for m in crit(rep, "Td").missing)


# --- blockers and re-opening ---------------------------------------------------------------------------------------------------


def trusted(store):
    hw(store)
    relation(store)
    instances_hw(store)
    relation(store, verified=True, proof_class="COMPLETE_ZK_BACKEND")


def test_open_d_entry_blocks_trusted(store, repo):
    trusted(store)
    write_log(repo, [dlog("D10", "resolved", evidence_refs=[GOLDEN]), dlog("D99", "open")])
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.level == "PINNED"
    assert not crit(rep, "Tb").passed
    assert any(m.startswith("D99 is open") for m in rep.missing_to_next)
    assert [e["id"] for e in rep.d_entries] == ["D10", "D99"]


def test_red_team_downgrade_label_blocks_trusted(store, repo):
    trusted(store)
    art = hw(store, elements=1000, extra_labels=(("trust_downgrade", "verifier rejected an honest run"),))
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.level == "PINNED"
    assert not crit(rep, "Tf").passed and any(art in m for m in rep.missing_to_next)
    store.add_label(art, "trust_downgrade", "resolved:D10", "test")
    assert assess("FP8_ADA_MMA", store, repo=repo).level == "TRUSTED"


def test_failing_canary_on_the_anchor_reopens_trust(store, repo):
    trusted(store)
    assert assess("FP8_ADA_MMA", store, repo=repo).level == "TRUSTED"
    bad = hw(store, elements=6656, mismatches=3, canary=True)
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.level == "HYPOTHESIS"                                  # P1 fails: a disagreeing record on the anchor
    assert not crit(rep, "P1").passed
    assert any(n.startswith("INVALIDATION TRIGGER canary-mismatch") and bad in n for n in rep.notes)
    assert [c["id"] for c in rep.canaries] == [bad] and rep.canaries[0]["passed"] is False
    assert any("canary" in m and "D-entry" in m for m in crit(rep, "P1").missing)


def test_passing_canary_is_a_verdict_not_coverage(store, repo):
    good = hw(store, elements=6656, canary=True)                      # a canary alone cannot pin: its tiles are not N_min coverage
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.level == "HYPOTHESIS"
    assert crit(rep, "P1").passed and not crit(rep, "P2").passed and not crit(rep, "P3").passed
    assert crit(rep, "P2").details["elements"] == 0
    assert rep.canaries == [dict(rep.canaries[0])] and rep.canaries[0]["id"] == good and rep.canaries[0]["passed"] is True


def test_registry_ahead_of_evidence_is_noted(store, repo):
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.registry_status == "pinned"
    assert any("ahead of the evidence" in n.lower() and "downgrade" in n for n in rep.notes)
    assert trust.registry_level("pinned") == Level.PINNED > trust.registry_level("hypothesis") > trust.registry_level("unprobed") == trust.registry_level(None)
    with pytest.raises(ValueError):
        trust.registry_level("trusted-by-fiat")


def test_registry_status_values_are_trust_ladder_members_and_never_above_pinned():
    """No store: the registry may assert at most PINNED (TRUSTED and above are computed), every status is a ladder level, and a
    dossier reference, when set, is a store artifact id."""
    from verity.ml.circuits.silicon.instructions import INSTRUCTIONS
    from verity.ml.circuits.targets.target import TARGETS

    for ins in INSTRUCTIONS.values():
        assert ins.status.upper() in Level.__members__, f"{ins.id}: status {ins.status!r} is not a trust-ladder level"
        assert trust.registry_level(ins.status) <= Level.PINNED, f"{ins.id}: the registry cannot assert {ins.status}"
        assert ins.dossier is None or ins.dossier.startswith("art:"), f"{ins.id}: dossier {ins.dossier!r} is not a store artifact id"
        if ins.model is None:
            assert trust.registry_level(ins.status) == Level.UNPROBED
    for t in TARGETS.values():
        assert t.dossier is None or t.dossier.startswith("art:"), f"{t.name}: dossier {t.dossier!r} is not a store artifact id"


def test_binding_class_entries_are_deployment_caveats_not_gates(store, repo):
    trusted(store)
    write_log(repo, [dlog("D10", "resolved", evidence_refs=[GOLDEN]), dlog("D6", "open", **{"class": "binding", "owner": "integration"}),
                     dlog("D98", "open", **{"class": "numerics"})])
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert not crit(rep, "Tb").passed and [e["id"] for e in rep.d_entries] == ["D10", "D98"]           # the numerics entry gates ...
    assert [e["id"] for e in rep.deployment_caveats] == ["D6"] and crit(rep, "Tb").details["binding"] == ["D6"]
    write_log(repo, [dlog("D10", "resolved", evidence_refs=[GOLDEN]), dlog("D6", "open", **{"class": "binding", "owner": "integration"})])
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.level == "TRUSTED" and crit(rep, "Tb").passed                                          # ... the binding one does not
    assert [e["id"] for e in rep.deployment_caveats] == ["D6"] and rep.deployment_caveats[0]["class"] == "binding"
    md = render_markdown(rep)
    assert "## Deployment caveats" in md and "| D6 | open | integration |" in md
    assert TrustReport.from_dict(json.loads(json.dumps(rep.to_dict()))) == rep
    write_log(repo, [dlog("D10", "resolved", evidence_refs=[GOLDEN], **{"class": "folklore"})])
    with pytest.raises(ValueError):
        assess("FP8_ADA_MMA", store, repo=repo)


def test_td_accepts_a_contract_pinned_recipe_digest(store, repo, monkeypatch):
    """The FP8 Ada input set is seeded, not a fixture: its identity is the recipe digest pinned in the bench contract."""
    import functools
    import shutil
    import types

    shutil.rmtree(repo / "fixtures" / "fp8-instances")                                              # no manifest in the repo
    fake = types.SimpleNamespace(FP8_ADA_INSTANCES_DATASET="bench-instances-fp8-ada/v1", FP8_ADA_INSTANCES_TIER="vu-k1536-fp8-ada",
                                 FP8_ADA_INSTANCES_DIGEST="e6" * 32)
    monkeypatch.setattr(trust, "_contract_instance_sets", functools.partial(trust._contract_instance_sets, contract=fake))
    hw(store)
    relation(store)
    relation(store, verified=True, proof_class="COMPLETE_ZK_BACKEND")
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert not crit(rep, "Td").passed
    assert any("bench-instances-fp8-ada/v1/vu-k1536-fp8-ada" in m and "family=frozen-instances" in m for m in crit(rep, "Td").missing)
    assert crit(rep, "Td").details["bench-instances-fp8-ada/v1"]["identified_by"] == "contract:FP8_ADA_INSTANCES_DIGEST"
    assert any("seeded recipe" in m for m in crit(rep, "Tc").missing)                                # no corner families in a recipe
    # a replay of the contract set on the anchor with the digest as its manifest sha256
    meta = {"schema": "research/result/v0.1", "kind": "tc-evidence/v1", "workload_fingerprint": {"instruction": INSTRUCTION},
            "validation": {"status": "passed", "checker": "tc_probe --replay-instances"},
            "tc_evidence": {"instruction": INSTRUCTION, "sku": "NVIDIA GeForce RTX 4090"}}
    art = store.put_artifact("tc-evidence/v1", meta)
    for k, v in (("instruction", INSTRUCTION), ("hardware", ANCHOR), ("family", "frozen-instances"), ("instances_dataset", "bench-instances-fp8-ada/v1"),
                 ("instances_tier", "vu-k1536-fp8-ada"), ("instances_manifest_sha256", "e6" * 32), ("mismatches", "0")):
        store.add_label(art, k, v, "test")
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert crit(rep, "Td").passed and any(e.id == art for e in crit(rep, "Td").evidence)
    # a replay against another digest is not the frozen set
    store.add_label(art, "instances_manifest_sha256", "ab" * 32, "test")
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert not crit(rep, "Td").passed and any("checked digest ab" in m for m in crit(rep, "Td").missing)
    assert any("contract digest e6e6e6e6e6e6" in m for m in crit(rep, "Td").missing)
    # the digest is the identity: instances-hw recorded the set under its own names (`fp8-ada synthetic ...` / `vu-k1536-fp8`) with the
    # digest as tc_evidence.manifest_sha256, and that replay counts
    meta2 = {"schema": "research/result/v0.1", "kind": "tc-evidence/v1", "workload_fingerprint": {"instruction": INSTRUCTION},
             "validation": {"status": "passed", "checker": "instances_hw"},
             "tc_evidence": {"instruction": INSTRUCTION, "sku": "NVIDIA GeForce RTX 4090", "instances_dataset": "fp8-ada synthetic (uniform finite E4M3 bytes)",
                             "instances_tier": "vu-k1536-fp8", "manifest_sha256": "e6" * 32, "instances": 4096, "mismatches": 0,
                             "model": "ada_e4m3_m16n8k32", "model_agrees": True}}
    art2 = store.put_artifact("tc-evidence/v1", meta2)
    for k, v in (("instruction", INSTRUCTION), ("hardware", ANCHOR), ("family", "frozen-instances"), ("mismatches", "0")):
        store.add_label(art2, k, v, "test")
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    td = crit(rep, "Td")
    assert td.passed and any(e.id == art2 for e in td.evidence) and not any(e.id == art for e in td.evidence)
    assert td.details["bench-instances-fp8-ada/v1"]["covered"] == ["vu-k1536-fp8-ada"]
    assert any("recorded as fp8-ada synthetic (uniform finite E4M3 bytes)/vu-k1536-fp8" in e.note for e in td.evidence)
    # the constants must all exist: an absent Hopper set contributes nothing
    assert trust._contract_instance_sets(trust.subjects("FP8_HOPPER_WGMMA"), contract=fake) == []
    real = trust._contract_instance_sets(trust.subjects("FP8_ADA_MMA"))
    assert real == [] or (real[0].contract and real[0].name == "bench-instances-fp8-ada/v1" and len(real[0].sha256) == 64)
    # a recipe may declare its family composition (<PREFIX>_INSTANCES_FAMILIES): the corner families Tc's edge predicate reads,
    # `random` excluded (the NVFP4 set's 14 proof-block families)
    fake4 = types.SimpleNamespace(NVFP4_INSTANCES_DATASET="bench-instances-nvfp4-sm120/v1", NVFP4_INSTANCES_TIER="vu-k1536-nvfp4-sm120",
                                  NVFP4_INSTANCES_DIGEST="d2" * 32, NVFP4_INSTANCES_FAMILIES={"random": 293, "zero-scales": 293, "cancel": 292})
    ds4 = trust._contract_instance_sets(trust.subjects("NVFP4_SM120"), contract=fake4)
    assert len(ds4) == 1 and ds4[0].contract and ds4[0].honest_tiers == ["vu-k1536-nvfp4-sm120"]
    assert ds4[0].corner_families("vu-k1536-nvfp4-sm120") == {"zero-scales": 293, "cancel": 292}
    real4 = trust._contract_instance_sets(trust.subjects("NVFP4_SM120"))
    assert real4 == [] or (real4[0].name == "bench-instances-nvfp4-sm120/v1" and "cancel-acc" in real4[0].corner_families("vu-k1536-nvfp4-sm120"))


# --- constants, dossier, publish -------------------------------------------------------------------------------------------------


def test_triggers_and_procedure_are_frozen_into_the_report(store, repo):
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep.invalidation_triggers == [dict(t) for t in INVALIDATION_TRIGGERS]
    assert rep.relitigation_procedure == list(RELITIGATION_PROCEDURE)
    ids = {t["id"] for t in INVALIDATION_TRIGGERS}
    assert {"canary-mismatch", "toolchain-change", "verifier-rejected-honest", "new-d-entry", "redteam-downgrade"} <= ids
    assert any("bisect" in s.lower() for s in RELITIGATION_PROCEDURE) and any("downgrade" in s.lower() for s in RELITIGATION_PROCEDURE)
    assert any("D-entry" in s for s in RELITIGATION_PROCEDURE) and any("anchor" in s for s in RELITIGATION_PROCEDURE)
    assert rep.constants["N_MIN"] == N_MIN == 10_000_000


def test_dossier_json_round_trip_and_markdown(store, repo):
    trusted(store)
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    doc = json.loads(json.dumps({"schema": trust.DOSSIER_SCHEMA, **rep.to_dict()}))
    back = TrustReport.from_dict(doc)
    assert back == rep
    md = render_markdown(rep)
    assert "# Trust dossier: " + TARGET in md and "**Level: TRUSTED**" in md
    assert "| Ta | TRUSTED |" in md and "## Invalidation triggers" in md and "## Relitigation procedure" in md
    assert "## Canaries on the anchor" in md
    for c in rep.criteria:
        for e in c.evidence[: trust.MD_EVIDENCE_PER_CRITERION]:
            assert e.id in md


def test_publish_writes_a_dossier_artifact_with_refs_and_labels(store, repo):
    trusted(store)
    rep = assess("FP8_ADA_MMA", store, repo=repo)
    res = publish(rep, store, push=False)
    man = store.get_manifest(res["id"])
    assert man.kind == DOSSIER_KIND
    assert man.meta["level"] == "TRUSTED" and man.meta["schema"] == trust.DOSSIER_SCHEMA
    assert set(man.refs["cited"]) == set(rep.cited()) and all(a.startswith("art:") for a in man.refs["cited"])
    labels = {lab.key: lab.value for lab in store.labels(res["id"])}
    assert labels == {"target": TARGET, "instruction": INSTRUCTION, "level": "TRUSTED", "anchor": ANCHOR}
    # a second assessment sees the previous dossier and the second publish links to it
    rep2 = assess("FP8_ADA_MMA", store, repo=repo)
    assert rep2.previous_dossiers == [res["id"]]
    res2 = publish(rep2, store, push=False)
    assert store.get_manifest(res2["id"]).refs["previous"] == [res["id"]]


def test_cli_json_and_md(store, repo, tmp_path, capsys):
    trusted(store)
    out = tmp_path / "dossiers"
    assert trust.main(["FP8_ADA_MMA", "--root", str(store.root), "--repo", str(repo), "--format", "json", "--out", str(out)]) == 0
    files = list(out.glob("*.json"))
    assert len(files) == 1
    doc = json.loads(files[0].read_text())
    assert doc["level"] == "TRUSTED" and doc["schema"] == trust.DOSSIER_SCHEMA
    assert trust.main([INSTRUCTION, "--root", str(store.root), "--repo", str(repo)]) == 0
    assert "**Level: TRUSTED**" in capsys.readouterr().out


def test_subject_resolution_covers_constants_names_and_instructions():
    for name in ("FIRST", "first-campaign-target/2026-09-21", "sm80.mma.m16n8k16.bf16"):
        s = trust.subjects(name)
        assert s.instruction == "sm80.mma.m16n8k16.bf16" and s.anchor_family == "a100"
    with pytest.raises(KeyError):
        trust.subjects("no-such-target")
    assert "sm89.mma.m16n8k16.bf16" in trust.all_subject_names()
    assert Level.UNPROBED < Level.HYPOTHESIS < Level.PINNED < Level.TRUSTED < Level.TRUSTED_STABLE


def test_the_real_discrepancy_log_has_a_regression_tile_decision_for_every_entry():
    log = json.loads((REPO_ROOT / "verity" / "ml" / "circuits" / "silicon" / "discrepancy_log.json").read_text())
    for e in log["entries"]:
        assert e["id"] in REGRESSION_TILES, f"{e['id']}: decide its regression tiles (possibly none) in REGRESSION_TILES"
        if e["status"] == "resolved" and e["id"] not in ("D3",):
            assert REGRESSION_TILES[e["id"]], f"{e['id']} is resolved without a regression tile"


# --- canary --------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def canary():
    import canary as C

    return C


def test_canary_plan_is_deterministic_and_covers_families_and_resolved_d_entries(canary):
    P = canary.P
    plan1 = canary.canary_plan(INSTRUCTION, REPO_ROOT)
    plan2 = canary.canary_plan(INSTRUCTION, REPO_ROOT)
    assert plan1 == plan2
    ins = P.INSTRUCTIONS[INSTRUCTION]
    assert set(plan1.families) == set(canary.family_names(ins.fmt, ins.m, ins.k))          # one batch from every family the format has
    assert "D10" in plan1.regression and plan1.replays == (GOLDEN,)                          # the resolved D-entry's replay tile
    assert plan1.unresolved == ()
    assert 2_000 <= plan1.elements(ins.m) <= 50_000                                           # a few thousand elements
    d1 = canary.canary_digest(plan1, ins.fmt, ins.m, ins.k)
    d2 = canary.canary_digest(plan2, ins.fmt, ins.m, ins.k)
    assert d1 == d2 and len(d1) == 64
    tiles = list(canary.canary_tiles(plan1, ins.fmt, ins.m, ins.k))
    assert [t[0] for t in tiles] == list(plan1.families)
    assert all(t[1].shape[0] == plan1.tiles_per_family for t in tiles)


def test_canary_seed_changes_the_tiles_but_the_plan_shape_stays(canary):
    P = canary.P
    ins = P.INSTRUCTIONS[INSTRUCTION]
    plan = canary.canary_plan(INSTRUCTION, REPO_ROOT)
    other = canary.CanaryPlan(**{**plan.__dict__, "seed": plan.seed + 1})
    assert canary.canary_digest(plan, ins.fmt, ins.m, ins.k) != canary.canary_digest(other, ins.fmt, ins.m, ins.k)


def test_canary_bf16_anchor_prefers_its_own_capture_directory(canary):
    plan = canary.canary_plan("sm80.mma.m16n8k16.bf16", REPO_ROOT)
    assert "D2" in plan.regression
    assert plan.replays == ("verity/ml/circuits/ops/tests/fixtures/tc-total-2026-09-07/a100/mma_special.reduced.json.gz",)
    assert {"adder_width_cancellation", "acc_only_lsb_walk", "zero_products_nonzero_acc", "specials"} <= set(plan.families)  # pin-bf16: BF16 has them too
    assert "fp8_floor_probe" not in plan.families                                             # the D15(b) probe is FP8-only


def test_run_canary_bookkeeping_without_a_device(canary, tmp_path):
    import numpy as np

    P = canary.P
    ins = P.INSTRUCTIONS[INSTRUCTION]
    plan = canary.canary_plan(INSTRUCTION, REPO_ROOT)

    def fake_model(A, Bt, C):
        words = (A.sum(axis=2, dtype=np.uint32)[:, :, None] + Bt.sum(axis=2, dtype=np.uint32)[:, None, :] + C).astype(np.uint32)
        return [(words, np.zeros(words.shape, np.bool_))]

    def replay_ok(path):
        assert path == REPO_ROOT / GOLDEN
        return {"pipeline": MODEL, "records": 360, "mismatches": 0, "other_positions_nonzero": 0, "first_mismatches": []}

    good = canary.run_canary(None, ins, plan, tmp_path, 1, REPO_ROOT, tiles_fn=lambda A, Bt, C: fake_model(A, Bt, C)[0][0], eval_fn=fake_model, replay_fn=replay_ok)
    assert good["failures"] == [] and good["model_agrees"] == {MODEL: True}
    assert good["elements"] == plan.elements(ins.m) and good["digest"] == canary.canary_digest(plan, ins.fmt, ins.m, ins.k)
    assert (tmp_path / "canary_tiles.npz").is_file()
    labels = {lab["key"]: lab["value"] for lab in canary.result_labels(ins, good, {"device": "NVIDIA GeForce RTX 4090", "gpu_uuid": "GPU-abc"})}
    assert labels == {"family": "canary", "instruction": INSTRUCTION, "hardware": ANCHOR, "model": MODEL, "model_agrees": "true", "evidence": "canary", "device": "GPU-abc"}
    sweep_labels = {lab["key"]: lab["value"] for lab in canary.evidence_labels(ins, {"model_agrees": {MODEL: False}}, {"device": "NVIDIA GeForce RTX 4090"}, evidence="sweep")}
    assert sweep_labels == {"instruction": INSTRUCTION, "hardware": ANCHOR, "model": MODEL, "model_agrees": "false", "evidence": "sweep", "device": None}
    assert all("device" not in cmd for cmd in canary.label_commands("art:x", canary.evidence_labels(ins, good, {"device": "x"}, evidence="sweep"), "lane"))  # no UUID: no label

    def flipped(A, Bt, C):
        D = fake_model(A, Bt, C)[0][0].copy()
        D[0, 0, 0] ^= 1
        return D

    bad = canary.run_canary(None, ins, plan, tmp_path, 1, REPO_ROOT, tiles_fn=flipped, eval_fn=fake_model, replay_fn=replay_ok)
    assert bad["total_mismatches_by_model"][MODEL] == len([f for f in plan.families if f not in P.UNASSERTED_FAMILIES])
    assert bad["failures"] and bad["model_agrees"] == {MODEL: False}
    assert canary.result_labels(ins, bad, {"device": "NVIDIA GeForce RTX 4090"})[4] == {"key": "model_agrees", "value": "false"}
    assert all(f["first_mismatches"][0]["tile"] == 0 for f in bad["families"].values() if f["asserted"])


def _tc_probe_tool():
    """tc_probe's tool declaration, by path: a bare `import tool` takes whichever of the repo's many `tool.py` is first on
    sys.path or already in sys.modules, which depends on what the suite ran before this test."""
    import importlib.util

    name = "_tc_probe_tool_declaration"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, TC_PROBE / "tool.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def test_tc_probe_tool_declaration_keys_the_canary_flag():
    T = _tc_probe_tool()

    with_flag = T.parse(["python", "benchmarks/tc_probe/tc_probe.py", "--instruction", INSTRUCTION, "--out", "x", "--canary"])
    without = T.parse(["python", "benchmarks/tc_probe/tc_probe.py", "--instruction", INSTRUCTION, "--out", "x", "--sweep"])
    assert with_flag["canary"] is True and "canary" not in without
    assert "canary" in T.TC_PROBE.key_params


def test_a_capture_replay_counts_the_model_and_the_other_tile_positions() -> None:
    """``tc_probe --replay-capture`` compares the hardware with the recorded words and with the declared model; a replay is clean only
    if all three counts are zero (veritor's H100 wgmma capture, the reference of sm90.wgmma.m64n8k16.bf16)."""
    clean = {"file": "verity/ml/circuits/ops/tests/fixtures/tc-hopper-2026-09-07/special.reduced.json.gz", "records": 44046, "mismatches": 0,
             "model_mismatches": 0, "other_positions_nonzero": 0, "pipeline": "veritor tc-hopper-2026-09-07"}
    assert trust._replay_summary("capture_replay", clean)["mismatches"] == 0
    assert trust._replay_summary("capture_replay", {**clean, "model_mismatches": 3})["mismatches"] == 3
    assert trust._replay_summary("capture_replay", {**clean, "other_positions_nonzero": 5})["mismatches"] == 5
    (ref,) = trust.INDEPENDENT_REFERENCES["sm90.wgmma.m64n8k16.bf16"]
    assert ref.kind == "second-compiled-path" and ref.replay in clean["file"]
    assert all((REPO_ROOT / p).is_file() for p in ref.paths)


def test_the_sm120_fp8_steps_are_anchored_on_the_pro_6000_and_the_arch_default_is_unchanged():
    """The target names its conformance device, and trust matches evidence to it by SKU family, whether asked by target or by
    instruction id.  The e5m2 id has no target: its registry entry names the device, and the sm_120 default anchor (the RTX
    5090) stays for every other sm_120 instruction."""
    from verity.ml.circuits.silicon.models import BLACKWELL_SM120_E4M3_M16N8K32

    for name in ("fp8-sm120-mma-draft/2026-09-30", "sm120.mma.m16n8k32.e4m3"):
        s = trust.subjects(name)
        assert s.model is BLACKWELL_SM120_E4M3_M16N8K32 and s.instruction == "sm120.mma.m16n8k32.e4m3"
        assert s.anchor_sku == "NVIDIA RTX PRO 6000 Blackwell Server Edition" and trust.sku_family(s.anchor_sku) == "rtxpro6000"
    assert trust.subjects("sm120.mma.m16n8k32.e5m2").anchor_family == "rtxpro6000"
    assert trust.subjects("sm120.mma.m16n8k64.e2m1.mxf4").anchor_family == "rtx5090"
    assert trust.ANCHOR_SKU["sm_120"] == "NVIDIA GeForce RTX 5090" and trust.sku_family(trust.ANCHOR_SKU["sm_120"]) == "rtx5090"
