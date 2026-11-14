[2026-11-14T21:06Z] 38dd6f2a0c9e
ROWSMERGE to SCANCOUNT2 (lean_names_map over our new Lean, handed to you by the lead), 1:07 PM PST, at the head of both branches, 1d2af9448342.

1. `python3 tools/move/rename.py --map tools/move/lean_names_map.toml --dry-run` refuses outright. Its [[words_in]] entries name 28 files that are no longer on the tree: they moved under the accounting split and the inference-only maps.
   - security/Proofs/InferenceOnly/{FitWitness,FitSlotsWitness,Remote,RemotePeriod,Witnesses}.lean
   - security/Definitions/InferenceOnly/{Guarantees/Remote,Guarantees/RemotePeriod,Guarantees/Witnesses,Witness}.lean
   - security/Proofs/Properties/Verity/WindowFiles.lean
   - security/Proofs/Accounting/{TS/PeriodSound,PerRunPriced,PouwWitness,BoundReceipt/Tile,PerRunNecessity,PerRunPouw,Responder}.lean
   - security/Definitions/Accounting/{CostModel,Guarantees/Bridges,Guarantees/BoundReceipt,Guarantees/PerRun,PerRun,Pouw,Guarantees/Responder,Responder}.lean
   - verity/protocols/accounting/poub/verifier/lean/RemoteAudit/{P2Close,P2Protocol,Difftest}.lean
   I worked around it before with a temporary copy of the map, its words narrowed to our files. That copy loses compound renames, so check them by hand.

2. Old names still in our Lean, all top-level words of the map:
   - WindowsPlaced (now ExecutionPhasesPlaced) and windowsCount (now executionPhasesCount): about 44 sites. Files: security/Definitions/Certifier/Guarantees/MonitoringPeriodFails.lean (8) and security/Proofs/Certifier/MonitoringPeriod.lean (1), MonitoringPeriodChecked.lean (8), MonitoringPeriodCheckedChecks.lean (2), MonitoringPeriodCheckedExclusion.lean (10), MonitoringPeriodFails.lean (5), MonitoringPeriodFailsChecks.lean (10). The declarations were renamed on architecture, so these references break the build.
   - Remote.windows: the Lean is now Core.Remote.executionPhases (security/Proofs/Properties/Remote.lean). The old spelling remains in verity/protocols/compose/verifier/lean/README.md (2), compose/verifier/lean/tests/reference.py (1) and compose/verifier/onsite/lakefile.toml (1); check whether those are prose or a target name.
   - IntegrityHorizon was in onsite_record.py earlier.

3. The certifier's two globs (verity/protocols/accounting/certifier/**, security/**/Certifier/**) are in this map's rewrite scope: it is not one of the excluded maps.

