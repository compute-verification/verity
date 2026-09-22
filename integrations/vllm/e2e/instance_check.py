"""The v3 evidence requirement shared by ``certificate.py`` and ``verify_composed.py`` (brief A.4; lane PROTO-A, A4).

A record is certifiable under ``veritor/whole-program-certificate/v3`` only with, in the record directory:

* ``protocol_instance.json`` -- a strict ``veritor/protocol-instance/v1`` document with ``status = pre-challenge``, its sha256 pinned in
  ``manifest.inputs.protocol_instance_sha256`` (bound before READY_PHASE0, like the sampling policy);
* ``compact_validation.json`` -- the ``veritor/compact-validation/v1`` verdict of ``validate_plan`` on the WHOLE pinned program
  (``program_from_verity_ir`` path), digest-pinned to the instance's program and plan (produced on a pod by
  ``scripts/compact_whole_b0.py`` + ``e2e.population from-compact2``, or in-process for TINY); status must be ``valid``;
* ``population.json`` -- the ``veritor/population/v1`` RU membership of the validated plan (``e2e.population``), same digests;
* for a Bernoulli law, ``chronology.json`` -- the ``e2e.security_sample`` chronology whose challenges re-derive from the revealed seed,
  whose every sampled VU lies inside the population and inside J, and which lists every challenge drawn (no rerolls).

Every missing or inconsistent item is a PROBLEM (the certificate is NOT BOUND; the reverify row FAILs).  Nothing here is skipped
with a reason: a record without the evidence is certified only under the explicit ``--historical-v2`` path, which says so in the
certificate (``attests_protocol_instance = false``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from e2e.protocol_instance import (InstanceError, check_against_record, check_against_validation,  # noqa: E402
                                   instance_digest, load_instance, load_validation_artifact)

INSTANCE_FILE = "protocol_instance.json"
VALIDATION_FILE = "compact_validation.json"
POPULATION_FILE = "population.json"
CHRONOLOGY_FILE = "chronology.json"


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_instance_evidence(record: Path, manifest: dict[str, Any], *, instance: Path | None = None, validation: Path | None = None,
                            population: Path | None = None, chronology: Path | None = None) -> tuple[list[str], dict[str, Any]]:
    """Return ``(problems, summary)``.  ``problems`` empty <=> the record carries a consistent pre-challenge instance with its evidence."""

    record = Path(record)
    inp = manifest.get("inputs") or {}
    problems: list[str] = []
    summary: dict[str, Any] = {"instance": None, "validation": None, "population": None, "chronology": None}
    inst_path = Path(instance) if instance else record / INSTANCE_FILE
    val_path = Path(validation) if validation else record / VALIDATION_FILE
    pop_path = Path(population) if population else record / POPULATION_FILE
    chr_path = Path(chronology) if chronology else record / CHRONOLOGY_FILE

    # 1. the instance: present, strict, pre-challenge, pinned
    if not inst_path.exists():
        problems.append(f"{INSTANCE_FILE} missing: no protocol instance (v3 requires the pre-challenge experiment description)")
        return problems, summary
    try:
        inst = load_instance(inst_path)
    except InstanceError as e:
        problems.append(f"protocol instance invalid: {e}")
        return problems, summary
    idig = instance_digest(inst)
    fsha = _sha256(inst_path)
    pin = inp.get("protocol_instance_sha256")
    if pin is None:
        problems.append("manifest inputs.protocol_instance_sha256 absent: the instance was not bound before READY_PHASE0")
    elif pin != fsha:
        problems.append(f"{INSTANCE_FILE} sha {fsha[:12]} != manifest inputs.protocol_instance_sha256 {str(pin)[:12]}")
    if inst.status != "pre-challenge":
        problems.append(f"instance status {inst.status!r}: only a pre-challenge instance certifies (a reconstruction describes, it does not bind)")
    if inp.get("program_digest") and inp["program_digest"] != inst.program.digest:
        problems.append(f"manifest program_digest {str(inp['program_digest'])[:12]} != instance program digest {inst.program.digest[:12]}")
    if inst.security.bound_attached:
        if inst.sampler.law.kind != "bernoulli":
            problems.append(f"instance attaches a Bound under law {inst.sampler.law.kind!r}: the Bound is stated for Bernoulli(q,s) only (appendix A3)")
        if inst.security.bound_value_bits is None:
            problems.append("instance says bound_attached but carries no bound_value_bits")
    summary["instance"] = {"path": str(inst_path), "digest": idig, "sha256": fsha, "status": inst.status, "law": inst.sampler.law.kind,
                          "q": inst.sampler.law.q, "s": inst.sampler.law.s, "N_V": inst.partition.vu_count.value, "N_R": inst.partition.ru_count.value,
                          "plan_digest": inst.partition.plan_descriptor_digest, "partition_digest": inst.partition.partition_digest, "W_out_bits": inst.outputs.W_out_bits.value, "delta": inst.security.delta,
                          "bound_attached": inst.security.bound_attached}

    # 2. the compact whole-program validation artifact, digest-pinned
    if not val_path.exists():
        problems.append(f"{VALIDATION_FILE} missing: the partition's validation verdict is not in evidence (produce it with scripts/compact_whole_b0.py + "
                        "e2e.population from-compact2 on a pod; the entry point refuses without it)")
    else:
        try:
            art = load_validation_artifact(val_path)
            problems += check_against_validation(inst, art)
            summary["validation"] = {"path": str(val_path), "sha256": _sha256(val_path), "status": art["status"], "vus": art["vus"], "cus": art["cus"],
                                     "plan_digest": art["plan_digest"], "program_digest": art["program_digest"], "seconds": art.get("seconds"),
                                     "maxrss_mb": art.get("maxrss_mb"), "verdict_source": art.get("verdict_source")}
        except InstanceError as e:
            problems.append(str(e))

    # 3. the record cross-check (population / policy-law / output domain / pins)
    problems += [p for p in check_against_record(inst, record, val_path if val_path.exists() else None) if not p.startswith("no compact-validation artifact")]

    # 4. the population file (RU membership) and, for Bernoulli, the chronology
    pop = None
    if not pop_path.exists():
        problems.append(f"{POPULATION_FILE} missing: RU membership of the validated plan not in evidence (e2e.population writes it)")
    else:
        pop = json.loads(pop_path.read_text())
        if pop.get("schema") != "veritor/population/v1":
            problems.append(f"{POPULATION_FILE} schema {pop.get('schema')!r} != 'veritor/population/v1'")
        n = inst.partition.instances
        if pop.get("vu_count") != n * inst.partition.vu_count.value or pop.get("ru_count") != n * inst.partition.ru_count.value:
            problems.append(f"{POPULATION_FILE} counts {pop.get('vu_count')}/{pop.get('ru_count')} != instance {n} x {inst.partition.vu_count.value}/{inst.partition.ru_count.value}")
        if pop.get("instances", 1) != n:
            problems.append(f"{POPULATION_FILE} instances {pop.get('instances', 1)} != instance partition.instances {n}")
        if pop.get("plan_digest") != inst.partition.plan_descriptor_digest or pop.get("program_digest") != inst.program.digest:
            problems.append(f"{POPULATION_FILE} plan/program digests != instance")
        if pop.get("partition_digest") != inst.partition.partition_digest:
            problems.append(f"{POPULATION_FILE} partition_digest {str(pop.get('partition_digest'))[:12]} != instance partition_digest "
                            f"{inst.partition.partition_digest[:12]} (the population must be THE partition, not a representation of one)")
        if len(pop.get("ru_of") or []) != pop.get("vu_count"):
            problems.append(f"{POPULATION_FILE} ru_of length != vu_count")
        summary["population"] = {"path": str(pop_path), "sha256": _sha256(pop_path), "vu_count": pop.get("vu_count"), "ru_count": pop.get("ru_count")}
    if inst.sampler.law.kind == "bernoulli":
        if not chr_path.exists():
            problems.append(f"{CHRONOLOGY_FILE} missing: no commitment -> challenge chronology for the Bernoulli draw")
        elif pop is not None and not [x for x in problems if "mismatched population" not in x]:
            from e2e.security_sample import ADAPTER_KIND, verify_adapter_transcript
            from e2e.security_sample import verify as verify_chronology

            doc = json.loads(chr_path.read_text())
            kind = doc.get("chronology_kind") or "security-sample/v1"
            if inst.sampler.chronology.chronology_kind and inst.sampler.chronology.chronology_kind != kind:
                problems.append(f"chronology_kind {kind!r} != instance sampler.chronology.chronology_kind {inst.sampler.chronology.chronology_kind!r}")
            if chr_path.name != CHRONOLOGY_FILE:
                cp = ["chronology must be named chronology.json"]
            elif kind == ADAPTER_KIND:
                # REC: the vLLM adapter's own two-stage Bernoulli transcript (seed committed in T_0, r_q over requests, r_s over the committed
                # relation units of J), re-derived with the core functions; its inner population is NOT the instance's validated VUs -- that
                # difference is reported by check_against_record ("mismatched population") and keeps the Bound off the record
                cp = verify_adapter_transcript(record, chr_path.parent)
                if doc.get("instance_digest") != idig:
                    cp.append("chronology instance digest != instance")
            else:
                cp = verify_chronology(inst_path, pop_path, chr_path.parent)
            problems += [f"chronology: {x}" for x in cp]
            ev = {e["name"]: e for e in doc.get("events", [])}
            summary["chronology"] = {"path": str(chr_path), "sha256": _sha256(chr_path), "kind": kind, "events": [e["name"] for e in doc.get("events", [])],
                                     "challenges_drawn": len(doc.get("challenges_drawn") or []), "rerolls": doc.get("rerolls"),
                                     "realized": doc.get("realized"), "J": (ev.get("r_q") or {}).get("selection"), "Q_size": len((ev.get("r_s") or {}).get("selection") or [])}
        elif pop is not None:
            problems.append("chronology not re-derived: earlier evidence problems must be fixed first")
        # the instance's own chronology section must agree with the file once the record is complete
        if chr_path.exists() and inst.sampler.chronology.events:
            doc = json.loads(chr_path.read_text())
            if [e.name for e in inst.sampler.chronology.events] != [e["name"] for e in doc.get("events", [])]:
                problems.append("instance sampler.chronology.events != chronology.json events")
        # draft 2 (COST-D objection 1): the first boundary the instance declares must have been committed, in the realised chronology,
        # BEFORE the first challenge -- otherwise the RU structure the population assumes was not realised and no Bound attaches
        if chr_path.exists():
            doc = json.loads(chr_path.read_text())
            events = doc.get("events", [])
            fb = inst.partition.first_boundary
            commit = next((e for e in events if e["name"] == fb.committed_in and e["kind"] == "commitment"), None)
            first_challenge = next((e for e in events if e["kind"] == "challenge"), None)
            if commit is None:
                problems.append(f"chronology: first_boundary.committed_in {fb.committed_in!r} is not a commitment event of chronology.json")
            elif first_challenge is not None and commit["order"] > first_challenge["order"]:
                problems.append(f"chronology: first boundary committed in {fb.committed_in!r} (order {commit['order']}) AFTER the first challenge "
                                f"{first_challenge['name']!r} (order {first_challenge['order']}): the partition's RU structure was not realised before r_q")
            if fb.rule != "outgoing boundary runs of every CU" and inst.partition.ru_count.value and inst.partition.ru_count.value > 1:
                problems.append(f"first_boundary.rule {fb.rule!r} over a partition with {inst.partition.ru_count.value} RUs: a two-stage Bound over "
                                "this partition needs every CU's outgoing boundary committed before r_q (paper §5.2)")
    else:
        summary["chronology"] = {"note": f"law {inst.sampler.law.kind}: the fixed-size selector's transcript is verify/challenge_transcript.json; no Bound attaches (appendix A3)"}
    # 5. [REC S5] unconditional weight binding: the auditor's pre-challenge authorized commitment vs what the record committed and what the verifier rebuilt
    if inst.weights is not None:
        problems += check_weights_binding(record, manifest, inst, summary)
    return problems, summary


def check_weights_binding(record: Path, manifest: dict[str, Any], inst: Any, summary: dict[str, Any]) -> list[str]:
    """S5 (GUIDANCE §57.10): (i) the authorized object was fixed pre-challenge (weights_authorized.json digest-pinned before READY_PHASE0, equal to the
    instance's copy); (ii) the prover's committed weights root (statement T_0) == the authorized root, leaf_count and ids_digest; (iii) the verifier's
    rebuild from the pinned checkpoint (bindings.model) matched with no missing / unexpected parameter -- so every weight leaf the program reads is a
    leaf of the authorized tree.  Any failure -> the weight check bound nothing (covers is void)."""
    p: list[str] = []
    inp = manifest.get("inputs") or {}
    ac = inst.weights.authorized_commitment
    wa_path = record / ac.artifact
    out: dict[str, Any] = {"variant": inst.weights.variant, "expected_root": ac.expected_root, "leaf_count": ac.leaf_count}
    if not wa_path.exists():
        p.append(f"weights: {ac.artifact} missing from the record")
    else:
        sha = _sha256(wa_path)
        if sha != ac.artifact_sha256:
            p.append(f"weights: {ac.artifact} sha {sha[:12]} != instance authorized_commitment.artifact_sha256 {ac.artifact_sha256[:12]}")
        if inp.get("weights_authorized_sha256") != sha:
            p.append("weights: weights_authorized.json not digest-pinned in the manifest before READY_PHASE0 (inputs.weights_authorized_sha256)")
        wa = json.loads(wa_path.read_text())
        if wa.get("expected_root") != ac.expected_root or wa.get("ids_digest") != ac.ids_digest or int(wa.get("leaf_count", -1)) != ac.leaf_count:
            p.append("weights: weights_authorized.json root / ids_digest / leaf_count != instance authorized_commitment")
        if (wa.get("covers") or {}).get("leaf_count") != ac.leaf_count:
            p.append("weights: covers.leaf_count != authorized leaf_count")
        out["covers_digest"] = (wa.get("covers") or {}).get("digest")
    st_path = record / "verify" / "statement.json"
    sv_path = record / "verify" / "selective_verify.json"
    if st_path.exists():
        wr = (json.loads(st_path.read_text()).get("weights_root") or {})
        out["committed_root"] = wr.get("root")
        if wr.get("root") != ac.expected_root or int(wr.get("leaf_count", -1)) != ac.leaf_count or wr.get("ids_digest") != ac.ids_digest:
            p.append(f"weights: the prover's committed weights root {str(wr.get('root'))[:12]} (T_0) != the auditor's pre-challenge authorized root "
                     f"{ac.expected_root[:12]}: the committed weights are NOT the authorized object")
        # the checkpoint identity inside the statement must be the authorized one too
        mb = ((json.loads(st_path.read_text()).get("bindings") or {}).get("model") or {})
        if mb.get("revision") != ac.checkpoint_revision or dict(mb.get("file_sha256") or {}) != dict(ac.safetensors_sha256) or mb.get("config_sha256") != ac.config_sha256:
            p.append("weights: statement bindings.model checkpoint identity != the authorized checkpoint (repo/revision/file shas/config sha)")
    else:
        p.append("weights: verify/statement.json missing (no committed weights root to compare)")
    if sv_path.exists():
        bm = ((json.loads(sv_path.read_text()).get("bindings") or {}).get("model") or {})
        det = bm.get("detail") or {}
        out["verifier_rebuild"] = {"ok": bm.get("ok"), "root_matches": det.get("root_matches"), "leaves": f"{det.get('leaves_matched')}/{det.get('leaves_total')}",
                                   "missing_params": det.get("missing_params"), "unexpected_params": det.get("unexpected_params"), "bytes_recomputed": det.get("bytes_recomputed"),
                                   "seconds": bm.get("seconds")}
        if not bm.get("ok") or det.get("root_matches") is not True:
            p.append("weights: the verifier's rebuild of the weights tree from the pinned checkpoint did not match the committed root (bindings.model)")
        if det.get("missing_params") or det.get("unexpected_params"):
            p.append("weights: committed index has missing / unexpected parameters: a weight could enter outside the authorized tree")
        if det.get("leaves_total") is not None and det.get("leaves_matched") != det.get("leaves_total"):
            p.append("weights: not every committed weight leaf matched the checkpoint")
    else:
        p.append("weights: verify/selective_verify.json missing (no verifier rebuild in evidence)")
    out["bound"] = not p
    summary["weights"] = out
    return p
