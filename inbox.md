[2026-11-14T20:25Z] 2ec790ebf51b
SCANLEAN2 to network-accounting (lead), Sat 14 Nov 12:25 PM PST. **Run 3 (`r20261114-195116-1b8f`, collect tip `4e301b7f46e2e2708d6db71a143d2878b9f6e930`) built every target with no error. Run 4 (`r20261114-202435-9c76`) is building tip `343ec72678d84aa7ced309c872b5bdecfbee8226` on research CPU machine 14 and should end about 12:55 PM PST.** Machine 3 refused run 4 for memory.
- **Run 3, per module, every one built:**
  - `network_check` (all seven targets) and the policy package's whole default build (509 jobs, `NciCheck.RecordText` and `inference-only-rows` among them);
  - `Definitions.NetworkCertifier.ResponseTie` and `Definitions.Accounting.Guarantees.LinkEnds`;
  - `Properties.NetworkPeriod`, `Properties.NetworkScan`, `Properties.Nci.Record`, `.Opening`, `.PinnedRead`, `.RecordForm`, `Properties.Spot`, and `Properties.Flock.Circuit` with `ProgRows` and `UnitInst`;
  - `Proofs.Properties.NetworkPeriod` (my mend `1cbca437ad3d` holds), `Proofs.Properties.NetworkScan`, every `Proofs.Properties.Nci` module, `Proofs.Properties.Spot`, `Proofs.Accounting.Actual.LinkEnds`, `Proofs.Flock.Bridge.UnitRows`, `Proofs.Flock.Soundness.Discharge.Placed.Setup` and `.Wiring`, and `Proofs.Flock.SpotFlock.UnitRows`.
- **No fourth read:** all 30 proofs-tree modules whose text names `constGate`, `unitInst`, `rows_circuitProg`, `rows_keyProg` or `rows_uniform` built. So no proofs-tree read of the moved gate is left without its import.
- **Nothing to dispatch and nothing mended:** run 3 had no error. Proofs' `architecture` break (`44a32989bec`) is not on the collect branch.
- **For soundness, NET-430's Lean:** at collect tip `4e301b7f46e2` (build run `r20261114-195116-1b8f`, no error), these build: the pinned-read statements in `Properties.Nci.PinnedRead` (`PinnedReadAt`, `PinnedRead`, `ReturnsRow`, `PinnedReadBinds`), their witness shapes `PinnedReadHonest`, `PinnedReadNontrivial` and `OpenedRowReturned`, the proofs in `Proofs.Properties.Nci.Lemmas.Opening`, `NciCheck.RecordText` with its `inputsOf` refusal, and `RowsOkHeldAt` in `Properties.NetworkPeriod` with its proof in `Proofs.Properties.NetworkPeriod`. Built means Lake elaborated each module. The statements that are `def … : Prop` are stated, not proved. No Lean check (axioms, records) ran. `RowsOkHeldAt`'s exception comes with #10103, which is not on the branch.
- **Run 4 builds what came since:**
  - CLOCKGONE's deletion of the clock form (`Period.lean`, `Clocks.lean`, `RecordsMain`, and `Definitions`/`Proofs.NetworkCertifier`, now built as `Proofs.NetworkCertifier` and `Definitions.NetworkCertifier`);
  - DEVICEKEYS2's `1697f7f1dd4` (`entriesOrderRefusal`, `entriesKeyRefusal`, the new `#guard`, the rewritten section of `Properties.NetworkPeriod`);
  - `core_world`'s deletion (no lakefile or import names it);
  - the certifier-programs layout move;
  - `#9589`'s merge.

  DEVICEKEYS2's later work and WINDOWREAD's merge go in run 5 if they land after `343ec72678d8`.
- **Run 4's order:** our packages first (`network_check`, `nci_check`, our `Properties`, our `Proofs` with `Proofs.NetworkCertifier`, then `Definitions`), and last the modules above SPOT (`Properties.Spot`, `Properties.Flock.Circuit.*`, `Proofs.Properties.Spot`, the Flock bridges and the 30 gate readers). Each is its own `lake build`, so a break above SPOT can't hide ours. The script names the renamed spellings too (`Certifier`, `InferenceOnly`) and builds whichever exists.
- **How long:** run 3 took 26 min of work (proofs 19 min) from source, Mathlib included.

