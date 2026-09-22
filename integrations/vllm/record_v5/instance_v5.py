"""[A9 contract 2026-09-10] Step 0 -- the epoch instance over the v5 population (adapted from D4's ``record_6k/instance_6k.py``; A3's schema).

The instance is A3's ``veritor/protocol-instance/v1`` with the ``epoch`` section (one epoch = the workload's requests; ``B_ex`` an RU count
with provenance; the bound's scope is the epoch) and ``security.B_ex_unit = "replay_units"``.  Its population is ``population_v5``
(``legacy-minus-inputs``), so two of ``validate_instance``'s rules are broken BY CONSTRUCTION and admitted under an ENUMERATED allowlist
(D4's pattern): ``count_basis`` is not ``validated_plan`` (it is the validated plan minus Input kinds) and ``partition_digest_kind`` is
``a9/legacy-minus-inputs/v1``.  For a rules basis (no plan exists at 6500/1599) D4's four labelled violations are admitted as well.  Any
other violation refuses the instance; the list of accepted violations is written beside it (``instance_validation.json``).

    python -m record_v5.instance_v5 --config run.json --population DIR/population.v5.json --weights-authorized W.json --profile P.json
                                    --vocab-digest HEX --out-dir REC
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from record_v5 import deps
from record_v5 import config as cfgmod
from record_v5 import population_v5 as p5
from record_v5 import workload_v5 as wl5

deps.install()

from e2e import protocol_instance as pi  # noqa: E402

CONSTRUCTION_NOTE = f"population construction {p5.CONSTRUCTION} (record_v5/population_v5.py)"

# rule fragment (as validate_instance phrases it) -> why the v5 record admits it
ACCEPTED_ROWS: list[tuple[str, str]] = [
    ("instance.partition.count_basis must be one of", f"count_basis is {p5.COUNT_BASIS['rows']!r}: the validated P1/cu=whole plan's rows minus the Input kinds "
                                                        "(inputs are not VUs, PROTOCOL R2); interim until A1's population"),
    ("instance.partition.partition_digest_kind must be", f"partition_digest is the v5 identity ({p5.PARTITION_DIGEST_KIND}) derived from the validated partition digest + "
                                                           "the excluded kinds + the retained rows; the basis digest is recorded in the population file"),
]
ACCEPTED_RULES: list[tuple[str, str]] = [
    ("instance.partition.count_basis must be one of", f"count_basis is {p5.COUNT_BASIS['rules']!r}: the sampling population must be derived from the validated plan; no plan exists "
                                          "at LP=6500/STEPS=1599 (D4: exact plan OOM at 128 GB), so the kind table is L5's rules minus Input (MODELED)"),
    ("instance.partition.vu_count.label must be MEASURED", "N_V is MODELED: L5's exact structural rules at this shape (holdout 32x15 re-asserted) minus Input kinds"),
    ("instance.partition.ru_count.label must be MEASURED", "N_R = 1 per request is ANALYTICAL (P1/cu=whole: one RU per request)"),
    ("instance.partition.plan_digest_kind must be one of", "plan_digest is D4's rec6k/plan-rule-modeled/v1 identity; no veritor.plans.summary.plan_digest exists at this shape"),
] + ACCEPTED_ROWS[1:]


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def classify_violations(problems: list[str], accepted_rules: list[tuple[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    accepted, refused = [], []
    for msg in problems:
        hit = next((r for r in accepted_rules if msg.startswith(r[0]) or r[0] in msg), None)
        (accepted.append({"rule": msg, "reason": hit[1]}) if hit else refused.append(msg))
    return accepted, refused


def accepted_rules_for(doc: dict[str, Any]) -> list[tuple[str, str]]:
    basis = (doc.get("notes") and next((n for n in doc["notes"] if n.startswith("basis=")), None)) or ""
    return ACCEPTED_RULES if basis == "basis=rules" else ACCEPTED_ROWS


def load_instance_v5(doc: dict[str, Any] | str | Path) -> pi.ProtocolInstance:
    """``load_instance`` under the enumerated allowlist; refuses any other violation, a legacy (no-epoch) instance, and any non-v5 population."""

    if isinstance(doc, (str, Path)):
        p = Path(doc)
        doc = json.loads(p.read_text() if p.exists() else str(doc))
    if not isinstance(doc, dict) or doc.get("schema") != pi.SCHEMA:
        raise pi.InstanceError(f"schema must be {pi.SCHEMA!r}")
    inst = pi._decode(pi.ProtocolInstance, doc, "instance")
    if CONSTRUCTION_NOTE not in " ".join(inst.notes or []):
        raise pi.InstanceError("load_instance_v5 admits only the record_v5 (legacy-minus-inputs) instance; use e2e.protocol_instance.load_instance")
    if not inst.is_contract_instance:  # A3: a property
        raise pi.InstanceError("E_LEGACY_INSTANCE: no epoch / B_ex not an RU count (legacy instance refused by the v5 runner)")
    rules = accepted_rules_for(doc)
    accepted, refused = classify_violations(pi.validate_instance(inst), rules)
    if refused:
        raise pi.InstanceError("; ".join(refused))
    if len(accepted) != len(rules):
        raise pi.InstanceError(f"expected exactly the {len(rules)} enumerated violations, got {len(accepted)}: {[a['rule'][:70] for a in accepted]}")
    return inst


def program_identity(case: str, lp: int, steps: int, attention: dict[str, Any] | None = None) -> dict[str, Any]:
    """The fixed program's identity as A3's verifier rebuilds it (``locator.build_pinned_program``): the registry config ``b1.<case>_CONFIG``
    bound at (LP, STEPS); ``function`` = the bound id (config hash included).  `bind` only -- the program itself is not built here."""
    from verity.ir import bind
    from verity_vllm.registry import b1

    if not hasattr(b1, f"{case}_CONFIG"):
        raise pi.InstanceError(f"registry has no {case}_CONFIG; the identity must name a registry case")
    # [E7 D54/D57 2026-09-13] ``attention`` = the DECLARED kernel geometry {kernel, kBlockM, kBlockN, num_splits} (the shape
    # `b1.fa2_kernel_geometry(profile attention consts, D)` produces; E6 fills it from the pinned profile).  None -> `Serve_v1`, the legacy
    # program (kBlockN 64 implicit; every pre-D54 instance).  A geometry -> `Serve_v4{C, LP, STEPS, BN = kBlockN}` and the identity
    # carries it verbatim under "attention".  The run config's `workload.attention` names it (see `build`).
    fam = b1.DEFAULT_SERVE_FAMILY if attention is None else "Serve_v4"
    sp = b1.serve_program(dict(getattr(b1, f"{case}_CONFIG")), lp, steps, family=fam, attention=attention)
    # ProgramIdentity has {family, config_sha12, case, LP, STEPS, function} (+ "attention" for Serve_v4); the id is "<family>{C={<sha12>},LP=..,STEPS=..[,BN=..]}"
    sha12 = sp.id.split("C={", 1)[1].split("}", 1)[0]
    out = {"family": fam, "config_sha12": sha12, "case": case, "LP": lp, "STEPS": steps, "function": sp.id}
    if attention is not None:
        out["attention"] = dict(attention)
    return out


def build(*, cfg: cfgmod.RunConfig, population: Path, instance_id: str, vocab_digest: str, profile_path: Path, weights_authorized: Path | None,
          s_source: str, workload: dict[str, Any], registry_case: str = "B0", vocabulary_V: int = 49152, seed_custody: str | None = None,
          attention: dict[str, Any] | None = None) -> dict[str, Any]:
    pop = json.loads(Path(population).read_text())
    if pop.get("schema") != p5.SCHEMA:
        raise pi.InstanceError(f"{population}: not a {p5.SCHEMA} document")
    basis_kind = pop["basis_kind"]
    lp, steps, n = cfg.prompt_len, cfg.max_tokens - 1, cfg.requests
    nout = steps + 1
    if basis_kind == "rules" and (int(pop["LP"]), int(pop["STEPS"])) != (lp, steps):
        raise pi.InstanceError(f"rules population is for LP={pop['LP']}/STEPS={pop['STEPS']}, the run is {lp}/{steps}")
    pop_rel = str(population)
    rows = wl5.request_rows(workload)
    if len(rows) != n:
        raise pi.InstanceError(f"workload has {len(rows)} requests, the epoch declares {n}")
    request_ids = [r["request_id"] for r in rows]  # the exact served ids (WR-5 / R16), from the auditor's submission
    n_r = int(pop["ru_count"]) * n
    outputs = [{"name": "token_slot", "multiplicity": nout, "domain_bits_declared": 16, "wire_bits": 32,
                "enforcement": "declared_domain 16 on sampler.select; check_output_domains on every request",
                "source": f"Serve_v1 STEPS+1 = {nout} root output slots per request; OUT16 (GUIDANCE §32)",
                "domain_bits_vocabulary_implied": {"value": math.log2(vocabulary_V), "label": "ANALYTICAL", "source": f"log2(V), V = {vocabulary_V}"}}]
    w_out = sum(o["multiplicity"] * o["domain_bits_declared"] for o in outputs)
    excl = pop["basis"]["excluded"]
    basis_val = pop["basis"].get("validation") or {}
    if basis_kind == "rows":
        status = basis_val.get("status") or "valid (e2e.population_v2 build validated in-process; no compact-validation artifact beside the basis file)"
        validation = {"artifact": basis_val.get("artifact") or pop["basis"]["path"], "program_digest": pop["program_digest"], "plan_digest": pop["plan_digest"],
                      "partition_digest": pop["partition_digest"], "status": status,
                      "source": f"the verdict is the BASIS plan's ({basis_val.get('source') or 'population/v2 build'}; basis content {pop['basis']['content_sha256'][:16]}, "
                                f"partition {pop['basis']['partition_digest'][:16]}); the population is its rows minus {excl['vus']} Input VUs ({excl['kinds']}) "
                                f"(record_v5/population_v5.from_v2)"}
        vu_count = {"value": int(pop["vu_count"]), "label": "MEASURED", "source": f"{pop_rel}#vu_count = retained rows of the validated plan (basis {pop['basis']['vu_count']} - {excl['vus']} Input)"}
        ru_count = {"value": int(pop["ru_count"]), "label": "MEASURED", "source": f"{pop_rel}#ru_count (the validated plan's CUs; unchanged)"}
        gates = {"value": int(pop["gates"]), "label": "MEASURED", "source": f"{pop_rel}#gates (basis circuit size, input gates included)"}
        input_gates = {"value": int(excl.get("gates", 0)), "label": "MEASURED", "source": f"{pop_rel}#basis.excluded.gates = sum(count x unit_size) of the removed Input rows"}
        count_basis = p5.COUNT_BASIS["rows"]
        plan_digest_kind = pop.get("plan_digest_kind") or "plan-json/v1"
    else:
        validation = {"artifact": pop_rel, "program_digest": pop["program_digest"], "plan_digest": pop["plan_digest"], "partition_digest": pop["partition_digest"],
                      "status": "not validated at this shape (rule-generated) minus Input kinds",
                      "source": f"no validation artifact exists at LP={lp}/STEPS={steps}; D4's kind table (L5 rules; {pop['basis'].get('rules')}) minus Input kinds"}
        vu_count = {"value": int(pop["vu_count"]), "label": "MODELED", "source": f"{pop_rel}#vu_count = sum of the kind table minus Input (L5 rules at LP={lp}, STEPS={steps})"}
        ru_count = {"value": int(pop["ru_count"]), "label": "ANALYTICAL", "source": "P1/cu=whole: one RU per request (program instance)"}
        gates = {"value": "UNAVAILABLE", "label": "MODELED", "source": "the program was not built at this shape (D4: OOM at 128 GB); L5's rules count VUs, not gates"}
        input_gates = {"value": int(excl["vus"]), "label": "MODELED", "source": f"{pop_rel}#basis.excluded.vus (the Input kind's rule count; one gate per Input unit)"}
        count_basis = p5.COUNT_BASIS["rules"]
        plan_digest_kind = pop.get("plan_digest_kind") or "rec6k/plan-rule-modeled/v1"
    seed_derivation = ("e2e.partition_sample (A3, contract/a3-sequence): seed = 32 verifier bytes (secrets.token_bytes; production custody: verifier-held, committed as "
                       "H(domain, seed) before r_q); r_q = bernoulli_subset(seed, q/replay-unit, H(PHASE, q, instance_digest, boundary_digest), N_R, q) with N_R = instances x "
                       "ru_count; r_s = bernoulli_subset(seed, s/verification-unit, H(PHASE, s, instance_digest, r_q_digest, step3_digest(interior, exemptions)), |VUs of J \\ F|, s); "
                       "VU rank inside an RU = ascending v5 VU index (population_v5: rows of the validated plan minus Input, renumbered); an empty r_q is kept as an event "
                       "and redrawn under H(PHASE, q/replay-unit/redraw, phase, k) only when condition_nonempty is set (reproducibility runs only; production: a zero-size sample is a valid outcome)")
    doc = {
        "schema": pi.SCHEMA, "instance_id": instance_id, "status": "pre-challenge",
        "program": {"digest": pop["program_digest"],
                    "identity": program_identity(registry_case, lp, steps, attention or (cfg.raw.get("workload") or {}).get("attention")),
                    "input_inclusion": "gates_include_input_gates", "gates": gates, "input_gates": input_gates,
                    "source": f"{pop_rel}#program_digest ({'program_from_verity_ir digest of the built program' if basis_kind == 'rows' else pop.get('program_digest_kind')}); "
                              f"identity = b1.serve_program(b1.{registry_case}_CONFIG, LP, STEPS, family, attention).id (record_v5/instance_v5.program_identity; "
                              f"attention = the run config's workload.attention geometry -> Serve_v4; absent -> Serve_v1, the legacy kBlockN 64)"},
        "vocabulary": {"digest": vocab_digest, "schema": "verity-ir/registry-version/v0", "profile_name": Path(profile_path).stem, "profile_sha256": _sha256_file(profile_path),
                       "source": f"{profile_path} (sha256 of the profile file; DERIVE-CORR s4/binding.json vocabulary digest)"},
        "partition": {"plan_name": cfg.plan, "plan_descriptor_digest": pop["plan_digest"], "plan_digest_kind": plan_digest_kind, "vu_level": "coordinate", "cu_strategy": "whole",
                      "validation": validation, "vu_count": vu_count, "ru_count": ru_count, "count_basis": count_basis, "vu_count_bucketed_upper": None, "address_map": None,
                      "instances": n, "weight_input_check": dict(pi.DEFAULT_WEIGHT_CHECK), "partition_digest": pop["partition_digest"], "partition_digest_kind": p5.PARTITION_DIGEST_KIND,
                      "summary_digest": None, "first_boundary": dict(pi.DEFAULT_FIRST_BOUNDARY), "acquisition_granularity": "ru",
                      "source": f"record_v5/population_v5.py: {CONSTRUCTION_NOTE}; record population = instances x per-instance counts (ReplicatedView)"},
        "outputs": {"designated": outputs, "W_out_bits": {"value": w_out, "label": "ANALYTICAL", "source": "sum over designated outputs of multiplicity x declared bits"},
                    "scope": "per program instance (one request)", "source": "declaration fixed before the first commitment"},
        "sampler": {"law": {"kind": "bernoulli", "q": str(cfg.q), "s": str(cfg.s), "k": None, "N": None, "coverage_size": None,
                            "delta_note": s_source, "source": "declared here, before any commitment (record_v5 run config)"},
                    "population_ref": "partition",
                    "chronology": {"seed_derivation": seed_derivation, "seed_commitment": None, "events": [], "challenges_drawn": [], "rerolls": 0,
                                   "source": "empty before the audit; e2e.partition_sample appends every event to chronology.json and the record copies it here",
                                   "chronology_kind": "partition-sample/v1", **({"seed_custody": seed_custody} if seed_custody else {})},
                    "realized": None,
                    "source": f"docs/PROTOCOL.md two-stage Bernoulli law over the epoch's RUs / the VUs of J \\ F; B_ex = {cfg.epoch.B_ex} RU(s) ({cfg.epoch.label}); randomness mode {cfg.randomness_mode}"},
        "obligations": {"mandatory": [dict(o) for o in pi.DEFAULT_MANDATORY], "diagnostic": [dict(o) for o in pi.DEFAULT_DIAGNOSTIC], "stages": list(pi.DEFAULT_STAGES),
                        "source": "brief A.3; diagnostic checks never count as security samples; stages = COST-D cost-result-format.md §1 protocol-work names (R12)"},
        "security": {"delta": str(cfg.delta), "sigma_bits": {"value": -math.log2(cfg.delta), "label": "ANALYTICAL", "source": "-log2(delta)"},
                     "B_ex": cfg.epoch.B_ex, "B_ex_unit": "replay_units", "B_ex_provenance": f"record_v5 run config epoch.B_ex ({cfg.epoch.label}; declared before r_q)", "kappa": None,
                     "bound_algorithm": "veritor.analysis.bound.bound via veritor.frontier.bound_adapter.FOLD_OPTIONS (frozen fold)", "bound_attached": False, "bound_value_bits": None,
                     "bound_not_attached_reason": "pre-challenge: the Bound is computed after the audit from this instance's law and partition; none attached yet",
                     "fixed_size_analysis_ref": None, "source": "declared here"},
        "discrepancies": [
            {"field_path": "partition.count_basis", "severity": "scope", "finding": f"count_basis = {count_basis}: the population is the legacy population minus Input kinds ({excl['vus']} VUs removed per instance)",
             "resolution": "stated in the instance and the certificate (population label legacy-minus-inputs); replaced by A1's population when it lands", "source": pop_rel},
            {"field_path": "partition.partition_digest_kind", "severity": "scope", "finding": f"{p5.PARTITION_DIGEST_KIND}: a derived identity, not veritor.plans.summary.partition_digest",
             "resolution": f"the basis partition digest {pop['basis']['partition_digest'][:16]} is recorded in the population file and re-derivable", "source": pop_rel},
        ] + ([{"field_path": "partition.validation.status", "severity": "scope",
               "finding": f"validation status {validation['status']!r}: no plan exists at LP={lp}/STEPS={steps}; the kind table is rule-generated (MODELED)",
               "resolution": "cited verbatim; every count derived from the population is MODELED", "source": pop_rel}] if basis_kind == "rules" else
              [{"field_path": "partition.validation.status", "severity": "scope",
                "finding": f"the validation verdict ({validation['status']!r}) is the basis plan's; the sampled population is that plan minus its Input rows",
                "resolution": "cited verbatim as the validation scope; the retained units keep the validated plan's gate intervals and RU membership", "source": pop_rel}]),
        "notes": [f"pre-challenge instance written by record_v5/instance_v5.py; {CONSTRUCTION_NOTE}", f"basis={basis_kind}",
                  "loaded by record_v5/instance_v5.load_instance_v5: validate_instance's rules are all applied; exactly the enumerated accepted violations are admitted (instance_validation.json)",
                  f"F policy: {cfg.f_policy}; randomness mode: {cfg.randomness_mode}; serving: batch-invariant, greedy, prefix caching off, capture = inputs/outputs/seed"],
        **({"weights": pi.weights_section_from_authorized(weights_authorized)} if weights_authorized else {}),
        "epoch": pi.epoch_section(instance_id, request_ids, n_r, B_ex=cfg.epoch.B_ex, B_ex_provenance=f"record_v5 run config epoch.B_ex ({cfg.epoch.label}; declared before r_q)",
                                  delta=str(cfg.delta), family=cfg.family, approved_elf_sha256=list(cfg.approved_elf_sha256),
                                  inputs=pi.declared_inputs_section(rows, source=f"record_v5/workload_v5 ({cfg.workload}): the auditor's request rows, written to REC/workload.json before serving; "
                                                                                  "seed / sampling params not pinned (greedy declared in the run config)", pin_sampling_params=False, LP=lp)),
    }
    return doc


def pinned_problems(doc: dict[str, Any], *, cfg: cfgmod.RunConfig, pop: dict[str, Any], workload: dict[str, Any]) -> list[str]:
    """Disagreements between an owner-pinned instance and this run (config, workload rows, population file). Empty = consistent."""

    p: list[str] = []
    rows = workload["requests"] if isinstance(workload, dict) else workload
    wl_ids = [r["request_id"] for r in rows]
    ep = doc.get("epoch") or {}
    if sorted(ep.get("request_ids") or []) != sorted(wl_ids):
        p.append(f"epoch.request_ids {ep.get('request_ids')} != the run's workload request ids {wl_ids}")
    if len(ep.get("request_ids") or []) != cfg.requests:
        p.append(f"epoch has {len(ep.get('request_ids') or [])} request(s), the run config {cfg.requests}")
    law = (doc.get("sampler") or {}).get("law") or {}
    if Fraction(str(law.get("q"))) != Fraction(str(cfg.q)):
        p.append(f"sampler.law.q {law.get('q')} != config q {cfg.q}")
    if Fraction(str(law.get("s"))) != Fraction(str(cfg.s)):
        p.append(f"sampler.law.s {law.get('s')} != config s {cfg.s}")
    sec = doc.get("security") or {}
    if Fraction(str(sec.get("delta"))) != Fraction(str(cfg.delta)):
        p.append(f"security.delta {sec.get('delta')} != config delta {cfg.delta}")
    if int(ep.get("B_ex", -1)) != int(cfg.epoch.B_ex) or int(sec.get("B_ex", -1)) != int(cfg.epoch.B_ex):
        p.append(f"B_ex epoch {ep.get('B_ex')} / security {sec.get('B_ex')} != config {cfg.epoch.B_ex}")
    prog = doc.get("program") or {}
    if prog.get("digest") != pop.get("program_digest"):
        p.append(f"program.digest {str(prog.get('digest'))[:16]} != population program_digest {str(pop.get('program_digest'))[:16]}")
    ident = prog.get("identity") or {}
    if (ident.get("LP"), ident.get("STEPS")) != (cfg.prompt_len, cfg.max_tokens - 1):
        p.append(f"program.identity LP/STEPS {ident.get('LP')}/{ident.get('STEPS')} != config {cfg.prompt_len}/{cfg.max_tokens - 1}")
    part = doc.get("partition") or {}
    if part.get("partition_digest") != pop.get("partition_digest"):
        p.append(f"partition.partition_digest {str(part.get('partition_digest'))[:16]} != population file's {str(pop.get('partition_digest'))[:16]} "
                 f"({pop.get('partition_digest_kind')})")
    if part.get("partition_digest_kind") != pop.get("partition_digest_kind"):
        p.append(f"partition.partition_digest_kind {part.get('partition_digest_kind')} != population file's {pop.get('partition_digest_kind')}")
    vu = (part.get("vu_count") or {}).get("value")
    if vu is not None and int(vu) != int(pop["vu_count"]):
        p.append(f"partition.vu_count {vu} != population N_V {pop['vu_count']}")
    ru = (part.get("ru_count") or {}).get("value")
    if ru is not None and int(ru) != int(pop["ru_count"]):
        p.append(f"partition.ru_count {ru} != population N_R per instance {pop['ru_count']}")
    if int(part.get("instances", -1)) != cfg.requests:
        p.append(f"partition.instances {part.get('instances')} != config requests {cfg.requests}")
    elfs = set(ep.get("approved_elf_sha256") or [])
    if not elfs or not elfs <= set(cfg.approved_elf_sha256):
        p.append(f"epoch.approved_elf_sha256 {sorted(x[:12] for x in elfs)} not within the run config's {sorted(x[:12] for x in cfg.approved_elf_sha256)}")
    if ep.get("family") and ep.get("family") != cfg.family:
        p.append(f"epoch.family {ep.get('family')} != config {cfg.family}")
    il = (doc.get("commitments") or {}).get("interior_layout") or {}
    if il.get("status") != "pinned" or not il.get("digest"):
        p.append("commitments.interior_layout is not pinned (V3.interior_layout would refuse before r_s)")
    else:
        try:
            from e2e.interior import layout as ilay  # E2's module (read-only)

            bad = ilay.check_pin(il)
            if bad:
                p.append(f"commitments.interior_layout pin fails E2's check_pin: {bad[:2]}")
        except ImportError:
            pass  # the pin is checked by the verifier (E9) at step 4 on a snapshot that has E2's module
    if doc.get("status") != "pre-challenge":
        p.append(f"status {doc.get('status')!r} != 'pre-challenge'")
    return p


def main_pinned(a: argparse.Namespace, cfg: cfgmod.RunConfig, out: Path, workload: dict[str, Any]) -> int:
    """``--pinned``: the owner's instance, cross-checked and copied byte-for-byte (its digest is the owner's, computed before the run)."""

    src = Path(a.pinned)
    raw = src.read_bytes()
    doc = json.loads(raw)
    pop = json.loads(Path(a.population).read_text())
    problems = pinned_problems(doc, cfg=cfg, pop=pop, workload=workload)
    report = {"schema": "veritor/a9/instance-validation/v1", "pinned_by_owner": {"path": str(src), "sha256": hashlib.sha256(raw).hexdigest(), "instance_id": doc.get("instance_id")},
              "cross_checks": {"against": ["run config", "workload rows", "population file"], "problems": problems},
              "validator": "e2e.protocol_instance.validate_instance (A3; all rules applied) via record_v5.instance_v5.load_instance_v5"}
    if problems:
        report["refused"] = problems
        (out / "instance_validation.json").write_text(json.dumps(report, indent=1) + "\n")
        print("[instance_v5] REFUSED pinned instance:\n  " + "\n  ".join(problems))
        return 2
    (out / "protocol_instance.json").write_bytes(raw)  # verbatim: the owner's digest
    try:
        inst = load_instance_v5(out / "protocol_instance.json")
    except Exception as e:  # noqa: BLE001
        report["refused"] = [f"load_instance_v5: {type(e).__name__}: {e}"]
        (out / "instance_validation.json").write_text(json.dumps(report, indent=1) + "\n")
        print(f"[instance_v5] REFUSED pinned instance: load_instance_v5 {type(e).__name__}: {str(e)[:300]}")
        return 2
    rules = accepted_rules_for(doc)
    accepted, refused = classify_violations(pi.validate_instance(inst), rules)
    report.update(accepted=accepted, refused=refused, enumerated=[r[0] for r in rules])
    (out / "instance_validation.json").write_text(json.dumps(report, indent=1) + "\n")
    if a.weights_authorized:
        shutil.copyfile(a.weights_authorized, out / "weights_authorized.json")
    (out / "maps.json").write_text(json.dumps({"schema": "veritor/a9/maps-v5/v1", "program": {"digest": pop["program_digest"], "gates": pop.get("gates", "UNAVAILABLE"),
                                                                                             "input_gates": (doc["program"].get("input_gates") or {}).get("value")},
                                               "plan": {"name": cfg.plan, "vus": pop["vu_count"], "cus": pop["ru_count"], "partition_digest": pop["partition_digest"],
                                                        "partition_digest_kind": pop.get("partition_digest_kind", p5.PARTITION_DIGEST_KIND)},
                                               "population": {"schema": pop["schema"], "construction": pop["construction"], "content_sha256": pop["content_sha256"]}}, indent=1) + "\n")
    digest = pi.instance_digest(inst)
    (out / "instance_digest.txt").write_text(digest + "\n")
    print(f"[instance_v5] PINNED instance {doc.get('instance_id')} from {src} (sha256 {report['pinned_by_owner']['sha256'][:16]}) digest={digest[:16]} "
          f"N_V={inst.partition.vu_count.value:,} x {inst.partition.instances} q={inst.sampler.law.q} s={inst.sampler.law.s} B_ex={inst.epoch.B_ex}; "
          f"cross-checks OK; validate_instance: {len(accepted)} accepted (enumerated), {len(refused)} refused")
    return 2 if refused else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--weights-authorized", default=None)
    ap.add_argument("--profile", default=str(deps.ROOT / "manifests" / "semantic-profiles" / "llama-eager-batch-invariant-v3.json"))
    ap.add_argument("--vocab-digest", required=True)
    ap.add_argument("--s-source", default="s from the run config (record_v5); expected-work basis named in the run report")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workload", default=None, help="REC/workload.json (default: <out-dir>/../workload.json; written here once from the run config if absent)")
    ap.add_argument("--registry-case", default="B0", help="the registry config the fixed program is bound from (b1.<case>_CONFIG)")
    ap.add_argument("--pinned", default=None,
                    help="[E10 EOD 2026-09-11] a protocol_instance.json PINNED by the instance owner (E6) BEFORE the run: used verbatim (bytes copied, so its digest is the "
                         "owner's) after cross-checking it against the run config, the workload and the population (request ids, q/s/delta/B_ex, program + partition digests, "
                         "N_V/N_R, approved ELFs, the interior-layout pin); any disagreement refuses (exit 2, list printed and written)")
    a = ap.parse_args(argv)
    cfg = cfgmod.load(a.config)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wl_path = Path(a.workload) if a.workload else out.parent / "workload.json"
    workload = wl5.write_once(cfg, wl_path)
    if a.pinned:
        return main_pinned(a, cfg, out, workload)
    iid = f"{cfg.run_id}-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # [E1 night 2026-09-12] the vocabulary size and the semantic profile follow the registry case / run config (TINY: B0, 49152, llama-v3 -- unchanged)
    from verity_vllm.registry import b1
    vocabulary_V = int(dict(getattr(b1, f"{a.registry_case}_CONFIG"))["V"]) if hasattr(b1, f"{a.registry_case}_CONFIG") else 49152
    profile = Path(a.profile)
    cfg_prof = (cfg.raw.get("workload") or {}).get("semantic_profile")
    if cfg_prof and a.profile == ap.get_default("profile"):
        profile = deps.ROOT / "manifests" / "semantic-profiles" / cfg_prof
    doc = build(cfg=cfg, population=Path(a.population), instance_id=iid, vocab_digest=a.vocab_digest, profile_path=profile,
                weights_authorized=Path(a.weights_authorized) if a.weights_authorized else None, s_source=a.s_source, workload=workload, registry_case=a.registry_case,
                vocabulary_V=vocabulary_V, seed_custody="verifier_process_memory" if cfg.is_production else "record_dir_file")
    inst = pi._decode(pi.ProtocolInstance, doc, "instance")
    problems = pi.validate_instance(inst)
    rules = accepted_rules_for(doc)
    accepted, refused = classify_violations(problems, rules)
    (out / "instance_validation.json").write_text(json.dumps({
        "schema": "veritor/a9/instance-validation/v1", "validator": "e2e.protocol_instance.validate_instance (A3; all rules applied)",
        "violations_total": len(problems), "accepted": accepted, "refused": refused, "enumerated": [r[0] for r in rules],
        "rule": "record_v5/instance_v5.load_instance_v5 admits exactly the enumerated violations; any other violation refuses the instance"}, indent=1) + "\n")
    if refused:
        print("[instance_v5] REFUSED:\n  " + "\n  ".join(refused))
        return 2
    if len(accepted) != len(rules):
        print(f"[instance_v5] REFUSED: expected exactly {len(rules)} enumerated violations, got {len(accepted)}: {[x['rule'][:70] for x in accepted]}")
        return 2
    (out / "protocol_instance.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    if a.weights_authorized:  # A3's driver reads the auditor's weight commitment from the instance directory and pins it in the record
        shutil.copyfile(a.weights_authorized, out / "weights_authorized.json")
    # maps.json: the header A3's reverify (V1.program) and L1-shaped tools read; from the v5 population, nothing else (the driver copies it into the record)
    pop = json.loads(Path(a.population).read_text())
    (out / "maps.json").write_text(json.dumps({"schema": "veritor/a9/maps-v5/v1", "program": {"digest": pop["program_digest"], "gates": pop.get("gates", "UNAVAILABLE"),
                                                                                             "input_gates": doc["program"]["input_gates"]["value"]},
                                               "plan": {"name": cfg.plan, "vus": pop["vu_count"], "cus": pop["ru_count"], "partition_digest": pop["partition_digest"],
                                                        "partition_digest_kind": p5.PARTITION_DIGEST_KIND}, "population": {"schema": pop["schema"], "construction": pop["construction"],
                                                                                                                             "content_sha256": pop["content_sha256"]}}, indent=1) + "\n")
    inst = load_instance_v5(out / "protocol_instance.json")
    digest = pi.instance_digest(inst)
    (out / "instance_digest.txt").write_text(digest + "\n")
    print(f"[instance_v5] wrote {out / 'protocol_instance.json'} id={iid} digest={digest[:16]} N_V={inst.partition.vu_count.value:,} x {inst.partition.instances} "
          f"q={inst.sampler.law.q} s={inst.sampler.law.s} B_ex={inst.epoch.B_ex} RU ({cfg.epoch.label}); validate_instance: {len(problems)} violations, {len(accepted)} accepted (enumerated), 0 refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
