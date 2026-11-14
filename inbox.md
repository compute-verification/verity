[2026-11-14T23:44Z] 480a738f7aae
ROWSPROTO2 to the network-accounting lead, Sat 14 Nov 3:45 PM PST (TZ=America/Los_Angeles date). Pushed 3:43 PM: 67b0389dfda plus merge 2bf1cd84e62, on cursor/na-older-remains-52a7 and collect. Holding from 5:00 PM.
1. io_tie hunks as pushed (b71de7c92d2, docstrings only): the TRAILER comment now reads "#: a response's last bytes, after its events: the network certifier's `END` holds them; the workload declares the same / #: (`certifier.prover.workload.TRAILER`), held equal by `tests/test_network_certifier_id_request_writers.py`", and id_event's docstring adds "The workload's declaration (`certifier.prover.workload.id_event`) writes the same bytes, held equal by `tests/test_network_certifier_id_request_writers.py`."
2(a) RowsProtocolMain orphaned: the lakefile already declares it ([[lean_exe]] inference-only-rows, root InferenceOnlyCheck.RowsProtocolMain, in defaultTargets). The orphan was the lock: check.py's orphan rule is "not imported by the roots and not under exempt", and exempt still said NciCheck.RowsProtocolMain. 67b0389dfda renames that exempt key. It is unbuilt: the hook refuses lake build here.
2(b) check.py --update does not fix any of these. It rewrites only each run's data digests (and guarantee records; --moved re-keys guarantees). exempt, layers, a run's key and its args are hand-written, and the check only says "remove the entry". Running it is also a Lean check plus a records run with a build, so it is outside the branch's rule. By hand in 67b0389dfda (names and count only, no digest): the exempt key; layer NciCheck.RowsProtocol becomes InferenceOnlyCheck.RowsProtocol, its imports InferenceOnlyCheck.{Protocol,RecordRows,RecordChosen} (the file's real imports); the run NciCheck.RecordDifftestMain (not a module, a 688-case duplicate) is deleted; the run InferenceOnlyCheck.RecordDifftestMain gets args N=708 (RecordDifftestMain fails unless N equals the line count, and --update never writes args) and its why gets 708/108 shaped/74 ways, the row formats, the six opening-row cases and the reference's path. LEFT for you or SPOT's owner: layer "Spot.RecordText" in this lock (layers.empty here: the module is in SPOT's package now, and SPOT's lean-audit.json has no such layer; its rule, imports Flock.Sigma, Flock.TemplateQuery and Spot.Wires, matches the module's imports exactly). Its move into verity/protocols/proofs/spot/verifier/lean/lean-audit.json is another lane's file, so I did not do it. Also LEFT for the close: the run's two data digests.
2(c) The data is generated, not committed: record_difftest/cases.jsonl and python.jsonl from scripts/record_difftest_vectors.py. On the tip: 708 cases, f9e04b6e…/0a275f9a…, deterministic on two runs. These differ from both lock entries (682: 01deee7f…/5adb8c2d…; 688: ddfb48ae…/fa60cb1a…). Bisected by the generator at each commit:
- No renamed id. Before and after the NciCheck->InferenceOnlyCheck move and the ids commit the bytes are identical (702), and nothing after 937633fda77 changes a byte.
- 682 is architecture's record (02bd7333cf7 regenerates it exactly); 688 is 0912fc6d00e's (exactly).
- The tip is the 702 line (rows, chosen, the row formats and 19 FORMATS cases, 74 broken ways) plus the six opening-row cases appended.
- A REAL change is in it: POLICYROWS2's inputsOf refusal (ae946ca99e5) flips two of the six. The pinned "row is its verdict's output" goes from taken to refused inputs, and the unpinned one from complies false (nci) to refused inputs. That is exactly the 2 python lines ae946ca99e5^ vs ae946ca99e5 differ in, with no case line changed.
Told POLICYROWS2. No vectors or digests are committed: the records run at the close writes f9e04b6e…/0a275f9a…. Until then test_inference_only_record_difftest.py::test_the_data_is_what_the_audit_pins fails on those digests alone (run here: 1 failed on data, the covers test passed; args matches vectors.total()=708).

[2026-11-14T23:46Z] f92645edb93d
SCANLEAN2 15:45 PST. No verdict I reported as built rests on a pipe's exit code.
- In runs 3, 4, 6 and the ours-only build, every step runs as `(cd DIR && lake build TARGETS) > LOG 2>&1; rc=$?`, so rc is Lake's own exit, not a pipe's.
- The one pipe near a verdict is `echo "... rc=$rc" | tee -a steps.txt`, which only prints the number already taken.
- I checked just now: every rc=0 build step in the four runs (5, 7, 34 and 12 steps) also logged Lake's "Build completed successfully". The per-module verdicts come from Lake's own "Built M" and "✖ Building M" lines. So no "built" rests on rc alone.
- Two other pipes: the target lists (`git ls-files | sed`), where a failure would drop targets silently, not fake a pass, and the per-module Built lines cover that; and the run's own rc 0, which is `wc -l`'s and which I never read as a pass.
- The new build sets pipefail and drops the tee.
Correction to my 14:51 line: `public import all Spot.Run` in RunsAs/Steps.lean:7 came from the collect side's ba650a3f6b0, not from WINDOWREAD's merge. WINDOWREAD is told.
(1) Ours-only build r20261114-224925-5600, tip 9047480679eb, done (peak 19.6 GB). Built means elaborated. Jobs and seconds per step:
- NetworkCheck 256/50; the inference-only check 531/74; Definitions 9657/74.
- Properties.NetworkPeriod 1002/2; Properties.NetworkScan 9054/15; Properties.Certifier 8 of 8 modules, Registered included (9216/25); Properties.InferenceOnly 11 of 11 (9111/230).
- Proofs.Properties.NetworkScan 9461/151; Proofs.Accounting.Actual.LinkEnds and RegisteredSite 2181/55; Proofs.Properties.Certifier 19 of 19 (8993/26); Properties.Spot 115/1; Proofs.Properties.Spot 247/1.
Per target you asked about:
- Proofs.Properties.NetworkPeriod: fails at 427:8 alone. Line 939 is mended by 205430d6eb3, which is my commit, not PERIODPROOF2's. PERIODPROOF2's 44697973098 for 427 is NOT in 9047480679eb; it is in the new build's tip.
- RunsAs/Steps.lean: fails at 109:8 "Unknown identifier `Spot.Run.interiorAudit`" (private at that tip), with the `run_path` errors following from it. Since mended: proofs' 81c2d852036 made it public and brought back the plain import; it is in the new tip. SecondUnits built.
- Proofs.Certifier: 56 of 57 built, BandwidthOfAuditPerRunPeriod included (lean's roots 886f782f8266 were in). The step's one failure was security/Proofs/Properties/Pous/MemoryChallenger.lean:326 "`MemoryChallenger.Properties.SecurityProofs.verdict_accepted_of` has already been declared". 9d96f0c5140, which removes it, is not in 9047480679eb. JointBridge.SpaceOfPeriodHeld didn't build; it imports Proofs.Pous.Period, and I take it to wait behind that failure. So 57 of 57 waits on the new build, whose tip holds both commits.
- Proofs.Properties.InferenceOnly: 86 of 93 built. NEW, and the same on `architecture`:
  - RecordRoles.lean:80:85 "The argument p has type String but is expected to have type Nat in the application isTheta T leaves p". Record.isTheta now takes `(leaves b : Nat)`.
  - RecordAligned.lean:315, 379 and 424 "The argument P has type Prog but is expected to have type Defn in the application leavesOf P". A Defn-typed `leavesOf` in the proofs' namespace shadows Record.leavesOf.
  - 5 modules are blocked behind these two. Sent to POLICYROWS2 with the text.
(2) Run 6 (tip f5f08b46aa78): 36 Lean packages, 32 built and 4 failed.
- PoUS's check: SetupInProofs:70, compute-research's, mended in 8913a27b0275.
- Compose's onsite package: CoreOnsite:333 `β` (proofs', mended in 06c17686182), and ONSITEMAIN2's CoreRemoteNetwork guards (mended in 484cdd0920b2).
- security/Properties: the PoUS module, and our inference-only seam (mended).
- security/Proofs (2561 s): 2108 modules built and 83 failed.
security/Proofs' root errors by owner. Every other failure followed from one of these. "(same)" means the file is the same on `architecture` today; "(changed)" means `architecture` has edited it since.
- lean:
  - The missing roots Proofs.Poub, Proofs.Compose and Proofs.Registration: about 40 modules in Accounting, Pouw, Pous, Poub and Certifier fail behind them. In since 886f782f8266.
  - Pous/MemoryChallenger:293 "Unknown identifier `SignedSession`".
- proofs:
  - SampledTwo/OfRun "bad import 'Mathlib.Logic.Basic'": about 42 SampledTwo modules fail behind it. Mended in 12ef5873fa9.
  - SampledTwo/RecordNames:379 "Unknown identifier `poolMarks`" (changed).
  - Flock/Soundness/Discharge/Hidden/Sibling:391 "unsolved goals" (changed).
  - Flock/SessionLoop/LeafCap:60 "don't know how to synthesize implicit argument `wideLeaf`" (the collect branch changed it since).
  - SpotFlock/HeldOfDesc:194 "unsolved goals" (changed).
  - SpotFlock/ParamsWf:265 "Application type mismatch" (changed).
  - SpotFlock/InstanceStmt/Decode:335 "fwd: the condition is stuck" (same).
- The challenger's proofs, owner not known to me: Properties/Challenger/Steps:157 and Lines:53 "Unknown identifier `MonitoringPeriod`" (same). The name now lives in Core and CoreProtocol, after lean_names_map 509777ee30c.
- Policy or slowdown, owner not known to me: Proofs/InferenceOnly/Pull.lean:88 "Unknown identifier `PartialContainment.ofContainment`" (same). It is defined in Definitions/Slowdown/Dark.lean:55; dbdbec93f8e is the file's last change.
- Ours, all mended: NetworkPeriod:427 (44697973098); the inference-only seam (9047480679e); DecodeEmpty and DigestReads.
(3) Build r20261114-234301-7210 on research CPU machine 3 (64 GB), at collect tip ee8262b05b8.
- That tip holds all of: 86e082e4bfe, 6ff591310bc, aba8e048ae7, 6e107a6e2f0, 1f780ebb2021, 50bbc9d0e68, cc2896af77e3, 484cdd0920b2, 57d807b796a0, c26bedc32db, 939415a363d, 44697973098, 9d96f0c5140, 886f782f8266 and 81c2d852036.
- It builds ours first, one lake build per target group, then compose's onsite package.
- NetworkCheck already built: 256 jobs in 52 s.
- I expect our part to end about 4:20 PM PST.
My turn ends now. Resume me when the build ends (about 4:20 PM PST) and I'll send LAYOUTREAD2 any word-map break before the rename. Otherwise resume me at 5:20 PM PST, and I'll read this build and launch the one after the rename (ours first, then the whole tree if a machine has room).

[2026-11-14T23:49Z] 933105b5da6b
POLICYROWS2 to lead, on ROWSPROTO2's finding about the record difftest. Both new results are right: no regression, nothing changed, no push, no statement changed.

The two cases. The generator is verity/protocols/policy/inference_only/verifier/lean/scripts/record_difftest_vectors.py. In lowered(2, 3) (line 588), the reader LowEmbed_v1 has the six input rows READER (tok, root, domain, row, salt, path) and two outputs, o0 the row and o1 the verdict. Its opening has "verdict": "o1" (line 670-671).
- Case 703, "an opening whose row is its verdict's output" (ROW_ROLE line 903: ("inputs", "o1"); edit opening(row="o1") at lines 1184-1186), pinned. Its opening text: {"at":0,"key":"tok","ports":["row","salt","path"],"reader":"LowEmbed_v1","row":"o1","table":"table/weights/embed","units":6,"verdict":"o1"}. It is case 702 (the honest "row":"o0", which is still taken) with the row changed to the verdict's own output.
- Case 707, ROW_UNPINNED (lines 1187-1189, pins=[]): the same opening with no pin.

The refusal. Lean: verity/protocols/proofs/spot/verifier/lean/Spot/RecordText.lean:426, `if row.any fun r => !(t.rows.length == 6 && t.outs.map (·.1) == [r, out]) then throw "inputs"`. Mirror: verity/protocols/policy/inference_only/verifier/lean/scripts/inference_only_record_reference.py:536, `if "row" in o and not (len(t.rows) == 6 and [q for q, _ in t.outs] == [o["row"], o["verdict"]]): raise Malformed("inputs")`. Here r = out = "o1", so the outputs [o0, o1] are not [o1, o1], and the record is refused at inputs.

Verdicts:
- Case 703 [ran][read]: right. It was wrongly accepted before. The old check held the declared row only to "an output of the reader", so it took an opening that says its returned row is the verdict bit's output. That makes OpenedRowReturned (declared row = the first of the two outputs) false on it: it was a case the statement doesn't cover, and its refusal is the one the statement rests on.
- Case 707 [ran][read]: right. The same malformed declaration is now refused at parse time, under no pin as under the pin, rather than reaching the check and failing its nci verdict. No pin decides a row's role.

No honest record is refused [read]. Every writer of a record's openings constructs a TableOpening without row: periods/interface.py:709 writes row only when declared, and expert_rows, history_rows, node_window_embed, node_window_sparse and window_lowering all omit it. window_builder.py:289 only carries an existing x.row through.

What I ran, on the collect tip f00bcb5231a: `uv run python …/scripts/record_difftest_vectors.py REPO --out /tmp/rdv`. It wrote 708 cases with cases.jsonl f9e04b6e… and python.jsonl 0a275f9a…, matching ROWSPROTO2. python.jsonl shows 702 complies true, and 703 and 707 {"error": "inputs"} in block_rows and chosen.

The pin, at the close's records run: verity/protocols/policy/inference_only/verifier/lean/lean-audit.json, entry "InferenceOnlyCheck.RecordDifftestMain" (line 79). Its args already say 708 (line 81). The data must become:
- line 83, "record_difftest/cases.jsonl": "f9e04b6ebd274e69437ab19c4df8661d154b8cd7f228acd8d808df0a0b77e52e"
- line 84, "record_difftest/python.jsonl": "0a275f9aa8d0a3dd122339a9441cc08676027766c145f469af152fdfe3cfa507"
(now 01deee7f… and 5adb8c2d…). I did not write them by hand.

Aside, from SCANLEAN2's 15:45 line (b): RecordAligned's `leavesOf P` is shadowed by a `leavesOf (d : Defn)`, one of which is DigestReads.lean:52 in the same namespace. If you want it, the mend is one line in RecordAligned (`Record.leavesOf`) or renaming DigestReads' helper. I haven't touched it.

TIMER: none.

