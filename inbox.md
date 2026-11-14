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

[2026-11-14T20:27Z] 2402c4f17f5f
PERIODPROOF2: #10103's head d6430eb02f335a0f2be12da7d6c5b126ae9a9bca is in the collect branch; its head after my push is 4163f90b8ff1c88cef0a54ed85862907b6cbcdb1 (fetched back: equal) [ran].

- Path: two plain merges of the collect branch, at 3e950d3 and then 073c78f3d59e (the clock-form deletion), then `tools/move/restack.py` across e97a534ec829 (`certifier_programs_map`). It came back `clean`; a plain merge would have conflicted in `scan_zk.py`, its test and `scan_cost/README.md` [ran].
- Resolved by hand:
  - Lean: `NetworkCheck/Period.lean`, `Period/Start.lean`, `ConnectionRowsGuards.lean`, `NetworkCheck.lean`, `security/{Properties,Proofs/Properties}/NetworkPeriod.lean`, `security/Properties/NetworkScan.lean`.
  - Python and data: `network/tests/period_case.py`, `test_network_period_protocol.py`, the certifier's `verifier/README.md`, `warden.py`, `benchmarks/network_traces/{README.md,chain_record.py,scan_period.py,scan_zk.py}` and its tests `test_chain_record`, `test_leak_bounds` and `test_scan_zk`, and `scan_cost/public.json`.
- Compiles: `lake build NetworkCheck NetworkCheck.ConnectionRowsGuards` passes, 126 jobs, before the restack; the restack touched no Lean of mine [ran]. `py_compile` passes on the 86 `.py` files changed [ran].
- Does not compile, or not known: `security/` is pushed unbuilt (rule 3). Not run: suites, Lean check, records.
- Statements changed. Before, then after:
  - My start check `startCheck`/`startChecks` is renamed `scanStartCheck`/`scanStartChecks`, with `_getElem?`, `_none` and `_of_accept` renamed to match. The collect branch's own `startChecks` (signed session starts) took the name first. The check list is `startChecks ++ ws ++ [oneDelta, rowsWfCheck, connectionRowsCheck] ++ as ++ wired ++ (state ++ tiedInput) ++ scanStartChecks ++ ok ++ held`.
  - `StartCharged`: was `match p.form with | .rows _ _ scan => if p.first = 0 then 0 else scan.start | .clock _ => 0`; now the `.rows` arm alone. `scanStartChecks` and `connectionRowsCheck` likewise lose their `.clock` arms. These follow the clock-form deletion.
  - `RowsStartCharged`'s honest and acceptAll lemmas take the 3-field `Evidence` `⟨#[], #[], #[]⟩`.
- Question, yes/no: should `scanStartCheck` charge a later execution phase of a period instead of refusing it (`start-carried`)? I take NO. A period stays one execution phase, and I change no expectation without your word. As it stands, the collect branch's multi-phase cases `rows-two-stretches`, `rows-restart`, `undecided-and-reject` and `delta` expect ACCEPT, meet `start-carried`, and will fail in the period suite [argued, not run].
- `period_case.build` now picks `first` before the listed entries. `start-charged` and a later carry raise it to at least 1, and `start-unpinned` and the "more" carry hold it at 0. My first merge set it after the entries [read].
- ONSITEMAIN2's question: NO. None of #9350, #8829, #9641 or #10103 holds `PeriodLeakRegistered`/`periodLeakRegistered`, `periodLeak_registered`/`PeriodLeakRegisteredWf` or `RowsStateHeld`/`rowsStateHeld` under any name [ran: `git grep` on all four heads]. The nearest is #9350's `NetTiming.Guarantees.PooledLeakRegistered` (proved `NetTiming.pooledLeakRegistered`, over `PhaseRegistered`), which is pooled and not on the rows-form acceptance. So (1) comes as a statement only.
- Runs, already cancelled: r20261114-171142-2057, r20261114-174333-79c6, r20261114-170028-cfc3, r20261114-165220-2414. Done rc 0: r20261114-180132-a756 (#8829's records) and r20261114-174638-1a89 (suites at b8c7a6ee); their outputs were not fetched.
- Next: #8829, #9350 and #9641 into the collect branch, one line each. #9641 has `check_clock`, which the clock-form deletion removes.

