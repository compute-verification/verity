"""Known artifact kinds (README §11).  The store validates the envelope only; `put` warns on an unknown kind, never refuses.

Kinds are strings `<name>/v<int>`, the name one or more `/`-separated segments (`verity/subcircuit-catalog/v1`); kind-specific
validation is the producer's.
"""
from __future__ import annotations

import re

KIND_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*(/[a-z0-9][a-z0-9_.-]*)*/v[0-9]+$")

KINDS: dict[str, str] = {
    "bench-result/v1": "a result.json (measurements, validation, fingerprint) + refs.run_files, refs.proof?",
    "bench-point/v1": "a sweep harness's raw result for its chosen point (a result.json with kind bench-point/v1): the run's attempt keeps "
                      "it and its run files; `bench.cell register` turns it into the cell's one bench-result/v1",
    "native-peak/v1": "one card's measured dense-GEMM throughput for one dtype (benchmarks/native_peak; the canonical tables' 'native dense peak, "
                      "measured' column): meta {device, dtype, accumulate, method, flop_per_second, spec_flop_per_second, fraction_of_spec, "
                      "sass, driver, cuda, clocks_mhz, power_w, sizes} + the result envelope",
    "tc-evidence/v1": "one tensor-core probe capture: meta {instruction, arch, sku, families, counts, model, model_agrees, first_mismatches, "
                      "seed, n}; payload tree = probe_results.json, capture_sample.json, tiles_*.npz",
    "dataset-snapshot/v1": "meta {name, note, query?}; refs.members = [art:...]; payload null",
    "tables-parity/v1": "the switch's parity render (verity_numerical.bench.views --parity --record): meta {schema, rules, ok, differences, "
                        "snapshot, tables_cells, cells: [{line, column, artifact, views_overhead, tables_overhead, printed}]}; refs.snapshot "
                        "(the tables --snapshot compared with), refs.cells; payload null",
    "verification-verdict/v1": "meta {result: PASS|FAIL, verifier, detail}; refs.proof, refs.verifier_attempt",
    "proof/v1": "payload = proof bytes; meta {statement_digest, backend, params}",
    "evidence/v1": "a lane's evidence behind a note, which never goes in the notes (scripts, logs, JSON, plots): payload file or tree; "
                   "meta {lane, what, run?}; the note cites the art: id",
    "fixture/v1": "a repo-replicated input (golden records, instances); payload file/tree",
    "notes-evidence-index/v1": "where the bytes of files removed from the notes tree live: payload index.jsonl, one {path, sha256, "
                               "bytes, art, member, kind} per file (`research data fetch <art> --path <member>` restores it); meta "
                               "{notes_commit, files, bytes, note}",
    "frozen-evidence/v1": "files the frozen r20 campaign's code still reads (the overhead ledger, recorded run results), no longer tracked: "
                          "payload tree at their former repository paths; meta {name, from_commit, note}; pinned by "
                          "verity_numerical.bench.frozen.EVIDENCE",
    "commit-map/v1": "a history rewrite's commit map (research.commitmap): payload file of `<old> <new> <same-tree 0|1>` lines; meta {from, to, "
                     "tool, args}; applies once labelled commit_map=accepted by its approver",
    "input-set/v1": "one subcircuit's input set, synthetic or captured (verity_numerical.bench.input_sets; the vllm-vu-set/v1 "
                    "layout): payload tree = manifest.json + index.json + flat port arrays; meta {set, subcircuit, template, "
                    "parameters, source, n, content_digest, schema, relation, seed?, row?, model?, research_run?}",
    "run-files/v1": "the files a result.json pointed at, as one tree",
    "run-record/v1": "every file of a run dir (stdout.log, stderr.log, launcher.log, job.json, status.json, outputs; markers excluded) as one "
                     "tree, named by its attempt as outputs.run_record: the run's custody on the remote (store.custody); meta {run_id, files, bytes}",
    "node-data/v1": "a path of a node's working set, preserved by `research keep --preserve` (research.retention): payload its file or "
                    "tree; meta {path, node, owner}; its retention record names the id",
    "labels-backup/v1": "a tarball of the local labels/ directory (every assertion file) -- interim durability until push carries labels; "
                        "meta {targets, files, taken_utc, host}",
    "telemetry/v1": "events.jsonl or resources.jsonl of an attempt",
    "check/v1": "a repository check run's result.json (each step's status and seconds); what `research merge` reads is its attempt",
    "check-shard/v1": "one shard's result.json (tools/verity/check/check.py --shard K/N): its steps, each with the plan's parts it ran, and "
                      "meta.shard {k, n, parts, all, commit, config}; no gate takes it alone, only a `check.py --combine` check of "
                      "its set (research.merge.shard_set)",
    "lean-fast/v1": "a Lean fast-path run's result.json (tools/verity/lean/lean_changed.py --records): one package's records audited without "
                    "the kernel replay or the tests; meta {validation, workload_fingerprint {package, update}}; `research merge` refuses "
                    "an attempt with this result whatever its tool (research.merge.REFUSED_KINDS)",
    "lean-review/v1": "an `audit.py --update` run's audit.json (tools/verity/lean): per package its review, statement_changes, and the guarantees "
                      "and reads it compared with (before) beside those it wrote; a `lean_review` label on pr:<n>@<head> may name it, "
                      "as a record nothing waits on",
    "flock-upstream-build/v1": "upstream Flock's verifier builds for the agreement job (verity/ml/flock/verifier/upstream-build.sh): "
                               "a tar.gz of upstream-<PR #83 commit> binaries and BUILD.json; meta {flock, rustc, host, bundle_sha256, "
                               "binaries}, pinned by verity/ml/flock/verifier/upstream.json",
    "flock-agreement-inputs/v1": "the stored inputs the agreement job's replayable sets name (a ci-bundle's store/, tar.gz); meta "
                                 "{arts}, pinned by verity/ml/flock/verifier/upstream.json",
    "redteam-findings/v1": "meta {findings: [...]}; refs.target",
    "circuit-check-report/v1": "a `circuit-check --out` report (tools/verity/circuit_check): payload file circuit-check.json; meta "
                               "{targets, failures}; a subcircuit catalog's counts cite its targets",
    "verity/subcircuit-catalog/v1": "one build of the subcircuit catalog (verity_numerical.bench.subcircuit_catalog): payload tree, one "
                                    "Boolean export per census subcircuit (subcircuits/<id>.boolean.json); meta {schema, census, entries: "
                                    "[{id, display, template, params, descriptor_id, default_statics, program_digests, export, counts, "
                                    "missing?}]}; refs.circuit_check = the circuit-check-report/v1 its counts cite",
    "kernel-attempt/v1": "one PoUW kernel attempt at one shape (benchmarks/pouw/harness/ledger.py, written by verify.py): meta = the "
                         "variant id, parent, tier, harness commit and gate version, metrics and step times, gamma at both FP32 prices, "
                         "the verifier's lines, the hypothesis; refs.parent = the attempt it changes; labels kernel_outcome, kernel_reason",
    "approach/v1": "one approach of the registry (research.approaches): meta {campaign, slug} and nothing else, so the id is a function of "
                   "the name <campaign>/<slug>; payload null; title, hypothesis, owner, status, reason and evidence are labels on it",
    "ledger-row/v1": "one frozen overhead-ledger row (backends/numerical/reports/ledger/<lane>.jsonl) whose run is not an attempt here; "
                     "meta = the row as written (schema ledger/v1); payload null; the lane's claims are labels on it",
    "trust-dossier/v1": "the computed trust level of one numerical target (benchmarks/tc_probe/trust.py): meta = the TrustReport "
                        "(schema verity/trust-dossier/v1: level, criteria with evidence ids, D-entries, references, missing items, invalidation "
                        "triggers, relitigation procedure); refs.cited = every artifact it rests on, refs.previous = earlier dossiers; payload null; "
                        "labels target, instruction, level, anchor",
    # vLLM integration (integrations/vllm/verity_vllm/pipeline/research_tools.py; one Attempt per sweep stage).
    "vllm-build/v1": "one Build of a row's Program: payload tree = build_summary.json + required-value manifest + descriptor "
                     "files (never the instance set); meta {row, program_digest, correspondence_digest?, stage_status}",
    "vllm-commit-plan/v1": "one Commit's plan, derived on CPU before the GPU is held (verity_vllm.pipeline.commit_plan): payload tree = "
                           "plan/commit_plan.json (the key and the producer facts) + the plan stage's logs; meta {row, program_digest, "
                           "plan_key, stage_status}; refs.build = the vllm-build/v1 planned from",
    "vllm-match-record/v1": "one Match account: payload tree = instances.jsonl, control/tokens.json, capture/tokens.json, "
                            "global_match.json, versions.json (never the raw capture planes); meta {row, program_digest, "
                            "match_ok?}; refs.build = the vllm-build/v1 consumed",
    "vllm-verdict/v1": "one Commit verdict (a Verity check/verdict, represented durably): payload tree = commit_verdict.json + "
                       "sampled_replay_p*.json + commit_summary.json (compat); meta {row, verdict, ...}; refs.build, refs.match",
    "vllm-replay-slim/v1": "a decided CPU replay's slim keep (verity_vllm.pipeline.cpu_replay.keep): payload tree = slim.json + the slim "
                           "committed store (only the leaves and Merkle nodes its picks and linkage opened) + weights tree + binding map + "
                           "context; meta {row, stage, run_root, program_digest, picked, bytes, dump_sha256, commit_pass}",
    "execution-evidence/v1": "application evidence of one stage's execution (timeline.jsonl, schedule / arrival facts) — not "
                             "telemetry; meta {row, stage}",
    "vllm-vu-set/v1": "one subcircuit's input set exported from a passing row's sampled VUs (verity_vllm.pipeline.vu_export): payload "
                      "tree = manifest.json + index.json + flat port arrays; meta {set, relation, n, content_digest, row, model, run, "
                      "run_root, program_digest, manifest_digest, source, epoch}; refs.run_record",
    "vllm-vu-store/v1": "one row's hierarchical VU store (verity_vllm.pipeline.vu_store): payload tree = run.json + programs/<digest>/"
                        "{program.json, units.jsonl} + values/<sha256>.<dtype>; the flat vllm-vu-set/v1 sets are its extractions",
    "vllm-vu-export/v1": "one row's VU export: payload = export.json (draw, per-family counts, sets); meta {row, model, run, run_root}; "
                         "refs.sets = the vllm-vu-set/v1 artifacts, refs.run_record",
}


def is_kind(s: object) -> bool:
    return isinstance(s, str) and bool(KIND_RE.match(s))


def is_known(kind: str) -> bool:
    return kind in KINDS
