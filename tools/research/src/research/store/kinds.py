"""Known artifact kinds (README §11).  The store validates the envelope only; `put` warns on an unknown kind, never refuses.

Kinds are strings `<name>/v<int>`; kind-specific validation is the producer's.
"""
from __future__ import annotations

import re

KIND_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*/v[0-9]+$")

KINDS: dict[str, str] = {
    "bench-result/v1": "a result.json (measurements, validation, fingerprint) + refs.run_files, refs.proof?",
    "native-peak/v1": "one card's measured dense-GEMM throughput for one dtype (tools/native_peak; the canonical tables' 'native dense peak, "
                      "measured' column): meta {device, dtype, accumulate, method, flop_per_second, spec_flop_per_second, fraction_of_spec, "
                      "sass, driver, cuda, clocks_mhz, power_w, sizes} + the result envelope",
    "tc-evidence/v1": "one tensor-core probe capture: meta {instruction, arch, sku, families, counts, model, model_agrees, first_mismatches, "
                      "seed, n}; payload tree = probe_results.json, capture_sample.json, tiles_*.npz",
    "dataset-snapshot/v1": "meta {name, note, query?}; refs.members = [art:...]; payload null",
    "verification-verdict/v1": "meta {result: PASS|FAIL, verifier, detail}; refs.proof, refs.verifier_attempt",
    "proof/v1": "payload = proof bytes; meta {statement_digest, backend, params}",
    "fixture/v1": "a repo-replicated input (golden records, instances); payload file/tree",
    "run-files/v1": "the files a result.json pointed at, as one tree",
    "labels-backup/v1": "a tarball of the local labels/ directory (every assertion file) -- interim durability until push carries labels; "
                        "meta {targets, files, taken_utc, host}",
    "telemetry/v1": "events.jsonl or resources.jsonl of an attempt",
    "redteam-findings/v1": "meta {findings: [...]}; refs.target",
    "ledger-row/v1": "one frozen overhead-ledger row (backends/numerical/reports/ledger/<lane>.jsonl) whose run is not an attempt here; "
                     "meta = the row as written (schema ledger/v1); payload null; the lane's claims are labels on it",
    "trust-dossier/v1": "the computed trust level of one numerical target (verity.verification.trust): meta = the TrustReport "
                        "(schema verity/trust-dossier/v1: level, criteria with evidence ids, D-entries, references, missing items, invalidation "
                        "triggers, relitigation procedure); refs.cited = every artifact it rests on, refs.previous = earlier dossiers; payload null; "
                        "labels target, instruction, level, anchor",
    # vLLM integration (integrations/vllm/verity_vllm/harness/research_tools.py; one Attempt per sweep stage).
    "vllm-build/v1": "one Build of a row's Program: payload tree = build_summary.json + required-value manifest + descriptor "
                     "files (never the instance set); meta {row, program_digest, correspondence_digest?, stage_status}",
    "vllm-match-record/v1": "one Match account: payload tree = instances.jsonl, control/tokens.json, capture/tokens.json, "
                            "global_match.json, versions.json (never the raw capture planes); meta {row, program_digest, "
                            "match_ok?}; refs.build = the vllm-build/v1 consumed",
    "vllm-verdict/v1": "one Commit verdict (a Verity check/verdict, represented durably): payload tree = commit_verdict.json + "
                       "sampled_replay_p*.json + commit_summary.json (compat); meta {row, verdict, ...}; refs.build, refs.match",
    "execution-evidence/v1": "application evidence of one stage's execution (timeline.jsonl, schedule / arrival facts) — not "
                             "telemetry; meta {row, stage}",
}


def is_kind(s: object) -> bool:
    return isinstance(s, str) and bool(KIND_RE.match(s))


def is_known(kind: str) -> bool:
    return kind in KINDS
