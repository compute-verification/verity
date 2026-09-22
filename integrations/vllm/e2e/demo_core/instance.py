"""The pinned instance of a CORE run in the engine's OWN formats — ``veritor/population/v2`` (``e2e.population_v2``: rows + CU intervals, the
partition digest re-derived by the shipped ``CompactPopulation.check(digest=True)``) and ``veritor/protocol-instance/v1`` (``e2e.protocol_instance``:
decoded and validated by the shipped ``load_instance`` — every ``validate_instance`` rule).  Both are read by the shipped sampler
``e2e.partition_sample`` (``bind``: counts, program / plan / partition digests must agree) unchanged.

Schema adaptations (the instance schema is Serve_v1-flavoured; each is labelled in the document itself):

* ``program.identity`` is E6's generic registered-composite form (2026-09-11): ``family = DemoCore_v1``, ``params = {N, R}``, ``module`` = the
  registering module, ``LP = STEPS = null``, ``config_sha12 = generic_config_sha12(params)``; ``program.kind = registered`` + ``program.builder``
  (INTEGRATOR D1) so the verifier rebuilds the program itself (``e2e.demo_core.circuit.build(**params)``) and compares the digest;
* ``epoch.inputs.per_request[<rid>].prompt_sha256`` := ``declared_input_digest`` of the 3·N input WORDS in gate order (the declared x by digest —
  the same function the auditor uses for token ids), ``prompt_len`` = 3·N; no ``layout`` pin (its digest rule is the int32 token layout);
* ``vocabulary`` := the output alphabet (a 16-bit word; every pattern admissible), digest of that declaration;
* ``outputs.designated`` := N words of 16 bits (``domain == wire``): W_out = 16·N bits.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from e2e import protocol_instance as pi
from e2e.population_v2 import POPULATION_V2, KIND_SCHEMA, ROW_FIELDS, CompactPopulation, content_sha256

from .circuit import KIND_ADD, KIND_SILU, Partition, identity

__all__ = ["population_doc", "instance_doc", "declared_inputs_doc", "APPROVED_ELF", "FAMILY"]

APPROVED_ELF = "9fd00e60fb6a74762de987cde211aca9de4576de844569da779acac6bf3863db"
FAMILY = "veritor.elem-bf16@1"
DELTA = "1/1000000000000"
BUILDER_MODULE = "e2e.demo_core.circuit"  # the module whose import registers DemoCore_v1 and exposes build(N, R) (E6 pin + D1 builder)
LABEL = "generic protocol demonstration (DemoCore_v1); NOT vLLM inference"


def _sha(doc: Any) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# -- population/v2 ----------------------------------------------------------------------------------------------------------------------


def population_doc(part: Partition) -> dict[str, Any]:
    """Rows ``(first_vu, count, ru, kind, lo, stride, unit_size)``: per RU one row per contiguous same-kind gate run (here: N SiluMul then N Add);
    CU intervals = the stage call's gate interval; the 3·N root Input gates are ``prescribed_outside_partition`` on both sides (they are the declared
    x, checked unconditionally at step 2, never sampled).  Self-checked by the shipped ``CompactPopulation.check(digest=True)``."""
    kinds = [{"id": 0, "fn": KIND_SILU.split("|")[0], "form": "batch", "primitive": True, "out_bits": 16, "size": 1},
             {"id": 1, "fn": KIND_ADD.split("|")[0], "form": "batch", "primitive": True, "out_bits": 16, "size": 1}]
    kid = {KIND_SILU: 0, KIND_ADD: 1}
    rows: list[list[int]] = []
    for ru in part.rus:
        run: list[int] | None = None
        for v in ru.vus:
            k = kid[v.kind]
            if run is not None and run[3] == k and run[4] + run[1] == v.gate and run[0] + run[1] == v.vu_index:
                run[1] += 1
            else:
                if run is not None:
                    rows.append(run)
                run = [v.vu_index, 1, ru.ru_index, k, v.gate, 1, 1]
        rows.append(run)
    cu = [[[ru.lo, ru.hi]] for ru in part.rus]
    in_gates = sorted(g for gs in part.inputs.values() for g in gs)
    ivs: list[list[int]] = []
    for g in in_gates:
        if ivs and ivs[-1][1] == g:
            ivs[-1][1] = g + 1
        else:
            ivs.append([g, g + 1])
    doc: dict[str, Any] = {
        "schema": POPULATION_V2, "plan_name": part.to_json()["kind"], "instances": 1,
        "vu_count": part.n_v, "ru_count": part.n_r, "per_instance": {"vu_count": part.n_v, "ru_count": part.n_r}, "gates": part.gates,
        "program_digest": part.program_digest, "plan_digest": part.digest, "plan_digest_kind": pi.PLAN_DIGEST_KINDS[1],  # plan-rule/v1: the constructor rule (e2e.demo_core.circuit.partition)
        "partition_digest": "", "partition_digest_kind": pi.PARTITION_DIGEST_KIND,
        "kinds": {"schema": KIND_SCHEMA, "key": "(primitive definition name, node form); 'batch' for a member of a verity.ir batch node", "entries": kinds},
        "row_fields": list(ROW_FIELDS), "rows": rows, "cu_intervals": cu,
        "prescribed_outside_partition": {"gate_intervals": ivs, "gates": sum(hi - lo for lo, hi in ivs), "sides": ["verification", "commitment"],
                                         "rule": "every root Input gate of DemoCore_v1 (the typed input x = a, b, c)",
                                         "check": "the committed input leaves are compared with the declared x unconditionally at step 2; never sampled"},
        "label": "MEASURED (rows enumerated from the verity.ir circuit of DemoCore_v1: one VU per computed gate, one CU per stage call); " + LABEL,
    }
    doc["partition_digest"] = CompactPopulation({**doc, "content_sha256": ""}).recompute_partition_digest()
    doc["content_sha256"] = content_sha256(doc)
    CompactPopulation(doc).check(digest=True)  # the shipped self-check: tiling, gate order, CU ownership, digest re-derivation
    return doc


# -- protocol instance ---------------------------------------------------------------------------------------------------------------------


def declared_inputs_doc(part: Partition, inputs: dict[str, list[int]]) -> dict[str, Any]:
    """The auditor's declared x: the 3·N input words in gate order + per-collection digests (kept beside the instance; its digest is in the instance)."""
    words = []
    for name in ("a", "b", "c"):
        words += [int(w) for w in inputs[name]]
    return {"schema": "veritor/demo-core/declared-inputs/v1", "collections": {k: {"gates": list(part.inputs[k]), "words": [int(w) for w in inputs[k]]} for k in ("a", "b", "c")},
            "words_in_gate_order": words, "digest": pi.declared_input_digest(words), "dtype": "bfloat16 (uint16 word)", "label": LABEL}


def instance_doc(part: Partition, pop: dict[str, Any], *, instance_id: str, epoch_id: str, request_id: str, q: str, s: str, B_ex: int,
                 B_ex_provenance: str, declared: dict[str, Any], profile_id: str, run_label: str, interior_layout: dict[str, Any] | None = None) -> dict[str, Any]:
    """``interior_layout``: E2's pin (``record.E2_LAYOUT_PIN``) when the record will spell its interior trees in E2's layout -- pinned here, before the
    first commitment, as ``commitments.interior_layout`` (E6's spelling; E9's ``V3.interior_layout`` refuses interior values without it)."""
    ident = identity(part.N, part.R)
    params = {"N": int(part.N), "R": int(part.R)}
    n_in = part.input_gates
    vocab = {"alphabet": "bf16 word (16 bits); every 16-bit pattern is admissible", "output": f"Array<{part.N}, BF16>", "program": ident["function"]}
    Q = lambda v, label, src, note=None: {"value": v, "label": label, "source": src, **({"note": note} if note else {})}  # noqa: E731
    stages = ["boundary.commit", "challenge.replay_draw", "acquire.commit", "exempt.declare", "challenge.inner_draw", "prove.witness", "prove.checker",
              "prove.argument", "verify.sampling", "verify.coverage", "verify.proofs", "verify.output_checks"]
    doc: dict[str, Any] = {
        "schema": pi.SCHEMA, "instance_id": instance_id, "status": "pre-challenge",
        "program": {"digest": part.program_digest,
                    # [E6 2026-09-11 / INTEGRATOR D1] generic registered composite identity (E6's frozen CORE instance spelling, out/eod/e6/core):
                    # family = registry id, params = static bindings, module = the registering module, LP/STEPS null, config_sha12 = sha256(params)[:12]
                    "identity": {"family": "DemoCore_v1", "params": params, "module": BUILDER_MODULE, "case": f"N{part.N}R{part.R}",
                                 "config_sha12": pi.generic_config_sha12(params), "LP": None, "STEPS": None, "function": ident["function"]},
                    "kind": "registered",
                    "builder": {"module": BUILDER_MODULE, "symbol": "build", "params": params, "program_digest": part.program_digest,
                                "source": "INTEGRATOR D1: the verifier rebuilds the program as e2e.demo_core.circuit.build(**params) in its own process and checks the digest against this pin"},
                    "input_inclusion": "gates_include_input_gates",
                    "gates": Q(part.gates, "MEASURED", "verity.ir.Program(bind(DemoCore, N, R)).gates"),
                    "input_gates": Q(n_in, "MEASURED", "verity.ir.Program.input_gates (3 x N Input<16> gates)"),
                    "source": "e2e.demo_core.circuit.build: program_digest(encode_program(Program(bind(DemoCore, N=, R=)))); " + LABEL},
        "vocabulary": {"digest": _sha(vocab), "schema": "veritor/demo-core/output-alphabet/v1", "profile_name": "demo-core/bf16-words",
                       "profile_sha256": hashlib.sha256(profile_id.encode()).hexdigest(),
                       "source": "the output alphabet of DemoCore_v1 (16-bit words); profile = sha256 of the record profile id (e2e.demo_core.record.profile_of)"},
        "partition": {"plan_name": pop["plan_name"], "plan_descriptor_digest": pop["plan_digest"], "plan_digest_kind": pop["plan_digest_kind"],
                      "vu_level": "gate", "cu_strategy": "stage-call",
                      "validation": {"artifact": "population.compact.json (e2e.population_v2.CompactPopulation.check(digest=True) in-process at instance build)",
                                     "program_digest": part.program_digest, "plan_digest": pop["plan_digest"], "status": "valid",
                                     "source": "rows tile [0, N_V) and, with the prescribed Input gates, [0, gates); every row inside its CU; partition digest re-derived from the rows",
                                     "partition_digest": pop["partition_digest"]},
                      "vu_count": Q(part.n_v, "MEASURED", "population.compact.json#vu_count (one VU per computed gate)"),
                      "ru_count": Q(part.n_r, "MEASURED", "population.compact.json#ru_count (one RU per stage call)"),
                      "count_basis": "validated_plan", "vu_count_bucketed_upper": None, "address_map": None,
                      "source": "e2e.demo_core.instance.population_doc; " + LABEL, "instances": 1,
                      "weight_input_check": {"variant": "none", "obligation": None, "commitment": None, "per_run": None, "reuse_rule": None, "stage": None,
                                             "bound_excludes_weight_family": None,
                                             "source": "DemoCore_v1 has no weight Input class: its 3xN Input gates are the declared x (prescribed outside the partition; checked at step 2)"},
                      "partition_digest": pop["partition_digest"], "partition_digest_kind": pop["partition_digest_kind"], "summary_digest": None,
                      "first_boundary": {"rule": "outgoing boundary runs of every CU",
                                         "rows": Q(part.n_r + 3, "MEASURED", "boundary leaves: one N-word leaf per RU output tuple h_r + the three input vectors a, b, c (RU 0's tree)"),
                                         "bytes": Q(2 * part.N * (part.n_r + 3), "ANALYTICAL", "16-bit words x (N_R + 3) leaves of N words"),
                                         "committed_in": "boundary", "source": "docs/PROTOCOL.md step 1: v_in(C), v_out(C) and every RU's output tuple committed before r_q"},
                      "acquisition_granularity": "ru"},
        "outputs": {"designated": [{"name": "y_word", "multiplicity": part.N, "domain_bits_declared": 16,
                                    "domain_bits_vocabulary_implied": Q(16, "ANALYTICAL", "a bf16 word: 16 bits, every pattern admissible"), "wire_bits": 16,
                                    "enforcement": "the committed output leaf is N x uint16 (dtype bfloat16): the leaf width IS the declared domain; checked at step 2 (well-typed y) and at step 5 (V7.output_domain)",
                                    "source": "DemoCore_v1 returns Array<N, BF16> = h_R (the last stage's output tuple)"}],
                    "W_out_bits": Q(16 * part.N, "ANALYTICAL", "sum over designated outputs of multiplicity x declared bits"),
                    "scope": "per program instance (one DemoCore_v1 evaluation = one request)", "source": "declaration fixed before the first commitment"},
        "sampler": {"law": {"kind": "bernoulli", "q": q, "s": s, "k": None, "N": None, "coverage_size": None,
                            "delta_note": "predeclared in the run config before any commitment (exhaustive q=s=1; sampled audits q,s in (0,1))",
                            "source": run_label},
                    "population_ref": "partition",
                    "chronology": {"seed_derivation": "e2e.partition_sample: seed = 32 verifier bytes held in the verifier process; r_q = bernoulli_subset(seed, 'q/replay-unit', phase_q, N_R, q); r_s over the VUs of J minus F",
                                   "seed_commitment": None, "events": [], "challenges_drawn": [], "rerolls": 0,
                                   "source": "empty before the audit; e2e.partition_sample appends every event to chronology.json", "chronology_kind": "partition-sample/v1",
                                   "seed_custody": "verifier_process_memory"},
                    "realized": None, "source": "docs/PROTOCOL.md two-stage Bernoulli law over the epoch's RUs / the VUs of J minus F"},
        "obligations": {"mandatory": [
            {"name": "output typing / domain on every designated output word (committed leaf N x 16-bit == declared domain)", "role": "unconditional", "counts_as_security_sample": False,
             "source": "step 2 (before r_q) and V7.output_domain at step 5", "stage": "verify.output_checks", "proof_steps": None},
            {"name": "output agreement / linkage: v_in(C) == declared x (leaf words vs epoch.inputs digest) and v_out(C) == published y (RU R-1 output leaf)", "role": "unconditional",
             "counts_as_security_sample": False, "source": "step 2 (before r_q): committed input leaves authenticated against the phase-0 roots", "stage": "verify.output_checks", "proof_steps": None},
            {"name": "sampled VU relation checks: every selected VU of a retained RU proved under the approved guest (coverage_v1 expected statement == proof positions)", "role": "sampled",
             "counts_as_security_sample": True, "source": "coverage_v1.ck_elem builders -> SP1 approved guest -> coverage_v1.coverage.check_coverage", "stage": "prove.argument",
             "proof_steps": ["witness", "checker", "argument"]}],
            "diagnostic": [], "stages": stages,
            "source": "docs/PROTOCOL.md steps 1-5; stages = frontier.result_schema.STAGE_NAMES this record charges"},
        "commitments": ({"interior_layout": {**{k: interior_layout[k] for k in ("framing", "word_order", "tree", "digest", "rule") if k in interior_layout}, "status": "pinned",
                                              "source": "e2e.interior.layout.layout_pin() (E2; the rule named in `rule`), pinned by the auditor before the first commitment (e2e.demo_core.record.E2_LAYOUT_PIN)"},
                         "source": "pins fixed before the first commitment; " + LABEL} if interior_layout else None),
        "security": {"delta": DELTA, "sigma_bits": Q(39.86313713864835, "ANALYTICAL", "-log2(delta)"), "B_ex": int(B_ex), "kappa": None,
                     "bound_algorithm": "none computed by this demonstrator (the two-stage Bound is the fold's; this record publishes the computed U and the verdict only)",
                     "bound_attached": False, "bound_value_bits": None, "bound_not_attached_reason": "pre-challenge; the demonstrator attaches no Bound (U is published; U = W here)",
                     "fixed_size_analysis_ref": None, "source": "declared here", "B_ex_unit": "replay_units", "B_ex_provenance": B_ex_provenance},
        "discrepancies": [], "notes": [LABEL, f"declared inputs: declared_inputs.json sha256(words_in_gate_order) = {declared['digest']}",
                                       "schema adaptations: program.identity in E6's generic registered form (params, module, LP=STEPS=null) + D1 program.builder; epoch.inputs.prompt_sha256 = digest of the input WORDS (see e2e.demo_core.instance)"],
        "weights": None, "proof": {"statement_format": 3, "source": "coverage_v1.ck_elem.build_many (format 3, ranged positions)"},
        "epoch": {"epoch_id": epoch_id, "request_ids": [request_id], "ru_indices": list(range(part.n_r)), "B_ex": int(B_ex), "B_ex_unit": "replay_units",
                  "B_ex_provenance": B_ex_provenance, "delta": DELTA, "bound_scope": "epoch",
                  "source": "e2e.demo_core.instance: one program instance = one request; RUs = its R stage calls; both commitment stages global within the epoch",
                  "family": FAMILY, "approved_elf_sha256": [APPROVED_ELF],
                  "inputs": {"kind": pi.DECLARED_INPUTS_KIND, "per_request": {request_id: {"prompt_sha256": declared["digest"], "prompt_len": n_in, "seed": None, "sampling_params_sha256": None}},
                             "source": "the auditor's declared x: declared_input_digest(3N input words in gate order); layout not pinned (int32-token rule does not apply)", "layout": None}},
    }
    if doc.get("commitments") is None:
        doc.pop("commitments", None)
    inst = pi.load_instance(doc)  # the shipped decoder + every validate_instance rule; raises InstanceError
    assert inst.is_contract_instance
    return doc
