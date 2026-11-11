"""The schema ids of the records that have no type in this package yet: the string a record carries as ``schema``.

Each id is written once, here; its writers and readers read the constant.  Beside each: the record, and what writes it.
An id leaves this module when its record gets a dataclass in this package, whose ``SCHEMA`` then owns it.
"""
from __future__ import annotations

# --- the observation of one eager run (`verity_vllm.observe`)
#: the raw observation log (its `RunHeader.schema`), which `observe.capture.make_header` writes
CAPTURE_LOG_SCHEMA = "verity-capture/log/v1"
#: a capture's snapshot index, `snapshots.json`, which `observe.capture.Capture._write_snapshot_index` writes
CAPTURE_SNAPSHOTS_SCHEMA = "verity-capture/snapshots/v1"
#: the runtime's module tree, which `observe.runtime_tree.RuntimeTree.to_json` writes and its `from_json` refuses any other
RUNTIME_TREE_SCHEMA = "verity-vllm/runtime-tree/v1"

# --- the query (`verity_vllm.query`)
#: whether a Commit used the manifest of record, which `query.manifest.verify.verify` and `acquire.compiled`'s `main`
#: write
MANIFEST_VERIFY_SCHEMA = "verity-sweep/manifest-verify/v1"

# --- the Match (`verity_vllm.check.match`)
#: the global workload Program's interface as the Match reads it (`check.match.declaration.GP_SCHEMA`)
GLOBAL_PROGRAM_SCHEMA = "cb-a/global-program/v1"
#: the batch decomposition's document, which `check.match.batch_decomp.decompose` writes
BATCH_DECOMP_SCHEMA = "cb-a/batch-decomp/v1"
#: the batch decomposition's phase profile, which `check.match.batch_decomp.decompose` writes
X09_PROFILE_SCHEMA = "cb-a/x09-profile/v1"
#: the cross-rank check's result, which `check.match.rank_match.cross_rank_check` writes
RANK_MATCH_SCHEMA = "tp2/rank_match/v1"
#: the ranks' commit binding, which `check.match.rank_match.commit_binding` writes (not `records.commit_binding`, a
#: Commit record's `binding` block)
TP2_COMMIT_BINDING_SCHEMA = "verity/tp2/commit-binding/v1"
#: the comparison of two Programs, which `check.match.program_compare`'s `main` writes
PROGRAM_COMPARE_SCHEMA = "cb-a/compare/v1"

# --- the sampled replay (`verity_vllm.check.replay`)
#: the sampled replay's record (`check.replay.driver.SCHEMA`), which `check.replay.driver` writes; `pipeline.c2_replay`,
#: `pipeline.replay_bundle` and `pipeline.tp.rank_replay` write it for a replay that did not run
SAMPLED_REPLAY_SCHEMA = "verity-sweep/sampled-replay/v1"
#: the MoE planes' alias check, which `check.replay.population.moe_plane_alias_check` writes
MOE_PLANE_ALIAS_CHECK_SCHEMA = "verity-sweep/moe-plane-alias-check/v1"
#: the PoUW circuit's replay, which `check.replay.pouw_circuit.replay` writes
POUW_NCP_V2_REPLAY_SCHEMA = "verity-vllm/pouw-ncp-v2-replay/v3"

# --- the ahead-of-time compiler (`verity_vllm.program.predict`)
#: a phi (`program.predict.phi.PHI_SCHEMA`), which `Phi.from_json` refuses any other than
PHI_SCHEMA = "verity-predict/phi/v1"
#: the comparison of a predicted Build with the Build of record (`program.predict.aot.SCHEMA`)
AOT_COMPARISON_SCHEMA = "verity-vllm/aot-comparison/v1"
#: `node.json`, which `program.predict.cli.node_main` writes
PREDICT_NODE_SCHEMA = "verity-predict/node/v1"
#: `registration.json`, which `program.predict.node_registration.register` writes
NODE_REGISTRATION_SCHEMA = "verity-vllm/node-registration/v1"

# --- PoUW's H1 (`verity_vllm.protocols.pouw.h1`)
#: a served window's calls record, `pouw-h1-calls.json` (`protocols.pouw.h1.RECORD`), which `H1Run.finish` writes
POUW_H1_CALLS_SCHEMA = "verity-vllm/pouw-h1-calls/v2"
#: a suspended window's hand-off state (`protocols.pouw.h1.handoff.SCHEMA`), which `suspend` writes and `resume` refuses
#: any other than
POUW_H1_HANDOFF_SCHEMA = "verity-vllm/pouw-h1-handoff/v1"
