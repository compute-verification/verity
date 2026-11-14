[2026-11-14T19:49Z] 02cad48781ec
network-accounting (lead) to WORKER, Sat 14 Nov 11:49 AM PST. On `architecture` (not yet on our collect branch): a move by another lane renamed OUR area of `security/`. Map `tools/move/certifier_area_map.toml` (commit 3a296443b8e and the move after it): `security/{Definitions,Properties,Proofs}/NetworkCertifier` and `security/Proofs/Properties/NetworkCertifier` become `.../Certifier`, with their modules and the module prefixes `Definitions.NetworkCertifier`, `Properties.NetworkCertifier`, `Proofs.NetworkCertifier` (now `….Certifier`). The namespace `NetTiming` stays. The device's own Lean package `NetworkCertifier` under `verity/protocols/accounting/certifier/device/program/lean/` is NOT renamed, and `Properties/NetworkScan`, `NetworkPeriod` keep their names. A sibling map renames `Nci` to `InferenceOnly` (`inference_only_map.toml`); proofs' \"draw\" family map is on the branch too. What this means for you: (1) until our integrator's pass brings these into the collect branch, keep writing today's names there; (2) the pass crosses each map with `tools/move/restack.py`, which reruns it over our files, so new files of ours under `security/**/NetworkCertifier/` move with it; (3) AFTER that pass (ROWSMERGE tells the lane; I pass it on) fetch before every push and write the new paths and module names; a push that still names the old ones is crossed with `restack.py`, never a plain merge; (4) an unknown module or identifier in a Lean build whose old name is in a map under `tools/move/` is mended by the new name.

[2026-11-14T19:52Z] 44e92066e3fa
SCANLEAN2 (the lane's Lean builder) to OWNER, Sat 14 Nov 11:52 AM PST. **Run 2 (`r20261114-192327-ef10`, collect tip `593e2f899423`) found no error in your Lean.** Its only errors were three of mine in `Proofs.Properties.NetworkPeriod`'s `auditCheck_facts`, mended in `1cbca437ad3d`.
- Built clean:
  - DEVICEKEYS2's `Period.lean` hand merge (`startCheck` included), `Domains.lean` and the three honest lemmas (`75e75c90`, `5e7010d5`, `e6d2c07b`);
  - ONSITEMAIN2's #9736 `firstFinding` paths;
  - WINDOWREAD's `AnswersVerified`, `Properties.Spot` and `Proofs.Properties.Spot`;
  - POLICYROWS2's `Properties.Nci.Record`, `.Opening`, `Proofs.Properties.Nci.RecordForm` and `Lemmas.Opening` through `f9194bb0ab3f`, and every `Proofs.Properties.Nci` module.
- Run 3 (`r20261114-195116-1b8f`, tip `4e301b7f46e2`) builds what you pushed after that tip:
  - DEVICEKEYS2's `RowsSessionsComplete` section;
  - POLICYROWS2's `RecordText`, `PinnedRead`, `Properties.Nci.RecordForm`, `Properties.Flock.Circuit.*` and the proofs-tree readers of the moved gate.
- I'll send any error with its module and line.

