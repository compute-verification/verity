[2026-11-09T22:14Z] 0a510080f741
lean, LEAN-80 (NET-118's window and session rows in lean's areas): both stages are staged, certified and recorded on cursor/lean-period-moves-fc8d at dcd2aa3e512, on origin/main 7af4d24fca3 (main hasn't moved since). No Move: PR: under TOP-387 its freeze check flags 13 module renames, so I'll restage on main once remote lands and the freeze ends. The full suites on the branch are still running (r20261109-213843-6b7a); I'll send one line when they finish.

The map cuts by area without breaking the build. lean's cut renames only what Security/{Definitions,Proofs}/{Core,Hash,Accounting} and Security/{Properties,Proofs/Properties}/Accounting declare: stage 1 has 135 [words] and 5 [[words_in]] tables, stage 2 has 7 [words] and 4 tables. The draft's other rows rename what other areas declare. They're in art:07838413f0c4a23a31750385b7cb6bd5f473cc7fb66954dbe2e06aef2e23de84 for network-accounting's Tuesday move. Six rows have no owner (Challenger, NetworkPeriod, Spot, Remote, Verity, WeightSecurity) and need one assigned.

Commits on the branch, in order:
- e1bd7b979ee: your decision. RemoteSessionRunConservation becomes RemoteMonitoringPeriodConservation, and its Relay becomes RemoteMonitoringPeriodConservationRelay.
- 3286046261a, 614362e2e00: tools/move/tests/test_maps.py, which is ci's file. Two maps' [[words_in]] tables now meet only where they can rename the same file, and two wildcard globs meet only where their literal prefixes allow it. Without this, the cut maps clash falsely with terms_map.
- 1d927306dea, 93439d44967, 532daa969ae: lean's cut of the map (period_map.toml for stage 1, period_map_2.toml for stage 2). Each [[words_in]] table names its moved files by a pattern that matches them before and after the move.
- 8dfaec72972, c2a37537f03: prep, as ordinary edits that change no record. They qualify `Session` at 16 sites as `Remote.Sketch.Session` or `Remote.Draw.Session`, in RemoteSketch.lean, RemoteSketchP2.lean and RemoteHeld/{Binding,Draw,DrawMeets}.lean. Without that, the move's new short name `Run` would resolve to `Accounting.Run`: Lean takes the innermost enclosing namespace's match before any `open`. This broke an earlier records build, and move_check can't see it.

Stage 1 ran as r20261109-192625-13c7. That one job ran stage.py, move_check, a resolution check and the records.
- Move: d0bee0dba37, "Move by tools/move/period_map.toml", 156 files.
- `verity check move c2a37537f03 d0bee0dba37`: certified c2a37537f036..d0bee0dba374 is a pure move (1 renames, 155 path-only edits). Rerunning the map on the move changes nothing.
- Records: f58939ef51a, from `lean_changed.py --records Security --update --moved moves1.json`. The Security and Security/Properties records were regenerated, Security/Proofs passed, and the audit printed AUDIT PASS. This is the fast tier without kernel replay; the queue's check replays.
- Lock renames: 97 (Accounting 81, Properties 10, Pouw 6). Among them, Guarantees.RemoteSessionRunConservation(Relay) and SecurityProofs.RemoteSessionRunConservation(Relay) become RemoteMonitoringPeriodConservation(Relay). The other renames are RemoteSession* to RemoteRun*, RemoteWindowRun* to RemoteExecutionPhaseRun*, StackedSession* to StackedRun*, Rate.Session to Rate.Run, Remote.{Draw,Sketch}.Session to .Run, *Window* to *ExecutionPhase*, period* to monitoringPeriod*, and Pouw.{Guarantees,SecurityProofs}.PeriodBindingPerProver(_honest,_nontrivial) to MonitoringPeriodBindingPerProver*. The full list is moves1.json in the run's out/.

Stage 2 ran as r20261109-203424-a527.
- Move: 6cc7dc6e854, "Move by tools/move/period_map_2.toml", 37 files.
- `verity check move f58939ef51a 6cc7dc6e854`: certified f58939ef51a5..6cc7dc6e8541 is a pure move (0 renames, 37 path-only edits).
- Records: dcd2aa3e512, with the same outcome as stage 1.
- Lock renames: 2. Accounting.Compose.ExecutionPhasePlay.EveryWindow becomes EveryExecutionPhase, and windowAt becomes executionPhaseAt.

Resolution check: both stages show 0 issues. It is a token-level check that each renamed reference at HEAD resolves to the renamed declaration, with Lean's namespace and open rules. Its scripts are art:9283afc207e5f7f99eaef9debc82ecd7df33c3b848a0b56aeca1e2d3f7011b11. I propose ci or infra fold it into `verity check move` or stage.py.

Per-area counts, in files changed (stage 1 / stage 2):
- lean: Proofs/Accounting 58/18, Definitions/Accounting 41/3, Proofs/Properties/Accounting 9/0, Properties/Accounting 7/1. Four other lean files are touched in stage 1: Security/lean-audit.json, the lean-proofs SKILL.md, tools/verity/lean/tests/test_merge.py and its fixture. Stage 2 touches Security/lean-audit.json.
- compute-accounting, 21/13: Security/{Proofs,Properties}/Pouw Period*, PeriodJ10R* and Served* files, Definitions/Nci/Guarantees/RemotePeriod.lean, Proofs/Nci/FitRate.lean and verity/core/protocols/pouw/README.md. These are references to the renamed names and prose; the Lean ones are needed for the build.
- memory-accounting, 7/0: remote_bandwidth/{audit.py,p2.py,PROTOCOL.md,tests/test_remote_bandwidth.py,lean/RemoteAudit/Window.lean}, bandwidth/openings.py and benchmarks/codec_tensor/README.md.
- network-accounting, 2/0: Security/Proofs/NetworkCertifier/Witnesses.lean and protocols/network/lean/NetworkCheck/Period.lean.
- training, 2/0: benchmarks/rho/rho/envelopes.py and benchmarks/rho/tests/test_envelopes.py.
- soundness, 1: verity/core/protocols/remote/README.md.
- top, 1: tests/lint/allowlists/repo_retired_terms.json.
- infra, 1: tools/research/src/research/store/moves.json.
- Shared, 1/0 and 1/1: Security/Properties/lakefile.toml (an edit, not a rename) and Security/Properties/lean-audit.json.

Held out of the move:
- `period` and `sessions`: bare lowercase words that prose shares.
- `session`, `window` and `windows`: two senses each, in shared files.
- Rate.DeployedSession: its new name, Rate.DeployedRun, is already spelled as the module RateAudit.DeployedRun, which move_check refuses. It needs a reviewed rename or another name.

Infra friction:
- Auto placement ignores the memory ledger. Twice it put a run in a memory-stalled line, after a 30 s probe of another node timed out.
- One node returned at 18:44Z has no build or audit pool in its lean-slots, so records fail there with rc 2 (r20261109-190939-5270).

