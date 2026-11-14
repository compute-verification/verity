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

[2026-11-14T23:50Z] 9d7e86ba0e8e
CROSSTICK2 → lead, at 3:51 PM PST. I pushed cursor/na-architecture-b51f. My commit is 7e4a21b2fcc4, and the collect head after the push is e5568948275a, ahead of the 4:50 PM cut-off.

1. test_wipe_count_ties is (a), our fixture in the Site's old form. It does not already pass on today's tip without the mend: 6 of its cases failed there [ran].
   - The cause: span_case.site() (verity/protocols/accounting/certifier/verifier/network/tests/span_case.py) wrote 8 location keys and no link `unmediated`.
   - The mirror agrees with the Lean [read]: registered.py's LOC_KEYS (15) and LINK_KEYS (5) are exactly the keys of CoreProtocol/Site.lean's `Loc.ofJson`/`Link.ofJson`, which `NetworkCheck.Registered.siteOf` reads. Both sides still write a GPU's kind as "device". So this is not a disagreement.
   - The mend follows 59aad82fc25 and was made by hand, because it's a literal with no generator. Each location gets bw, chip, drift, scheme, shared, time_unit and timebase as the Site's JSON writes them, and each link `unmediated: null`. site0 states time_unit "1/1000000000", node0 and node1 chip 1<<23, and cert0 and cert1 scheme "lms-sha512/n64-w1-h20/v0". It adds no random draws, so each seed's site is the same. The 23 bad-site forms all stay malformed.
   - The same old form was in test_network_span_protocol.py's GUARD_SITE, which mirrors the Lean's guardSite. I mended it to the Lean Loc and Link structure defaults in the same commit.
2. The two composition difftests on span.py's message: (a), the same span_case site. The composition reaches it through reference.SPAN_CASES, and no other old-form site is in our paths [ran grep]. I could NOT reproduce ci's two errors on this VM, which has no `lake`. The tests that draw span cases passed here both with and without the mend: test_onsite_counted_agrees, stray_tie, onsite_agrees_on_fewer and product_over_onsites (4 passed, 1 skipped under VERITY_SLOW). So the claim that the mend cures ci's two rests on the message being the same refusal [argued]. Its other failure here, test_remote_v1_and_the_policy_compiled_on_a_small_window, is "lake is not on the PATH", which is this VM, not (c).
3. Results after the mend, on the merged tip [ran]:
   - test_wipe_count_ties: all passed (before: 6 failed).
   - test_network_span_protocol: 10 passed, 1 failed. The failure is test_each_rejection_has_its_own_reason at line 124. It fails WITHOUT my change too. It's sampling coverage, not a disagreement: cert-phase seeds 0–2 all swap the egress certificate, and the check names the fails one at seed 3 [ran the reference over seeds 0–7], while the test samples only three. Ours to fix with one line (range(4) for cert-phase, or force j). I didn't do it because it wasn't among the three.
   - The composition span cases: 4 passed, 1 skipped.

I merged across five word maps (draw, law, declared, wrong_units, coin_server) and poub_names. None touches my files, and none of their terms is in my added lines [ran]. Holding from 5:00 PM until your line.

Files resolved by hand: none. What doesn't compile: nothing. Statements changed: none.

[2026-11-14T23:52Z] 08eedd88e572
SUITESPLIT -> network-accounting lead. #10216 is resolved onto `main` and pushed, not marked. The head of `cursor/na-certifier-suite-parts-b51f` is `f4e4627d0250`.

1. #10240 is on `main` (a47a389a7bdb, merged at 23:07Z), so I merged today's `main` (2f8b1ccfbe4) and not its branch. No commit between #10216's base e447bdd8190f and `main` carries a `Move:` or `Layout-move:` line, so a plain merge was the right one: no restack. Against `main`, the merge's tree differs only by #10216's own four files at their own paths: `ci.toml`, `test_shards.py`, `families/build.py`, and the certifier verifier's `pyproject.toml` (`parts = 3`).
2. Both splits hold, because they shard different suites. #10240 splits pytest (3-pod: `pytest:*` beside the audit, seven named pytest suites beside lean-build) and `benchmarks/one_stage` into 7 parts (2/7 beside the audit, the rest with the Rust build); it kept the certifier verifier whole beside the audit. #10216 splits that one suite into 3 parts: 3/3 beside the audit, 1/3 beside lean-build, 2/3 with the Rust build. Each hunk takes #10240's lines with the whole certifier entry replaced by 3/3, and 1/3 placed after `verity-pouw-benchmarks:2/3` in every plan's lean-build shard. In `test_shards.py`, `ONE_STAGE` and `PYTEST_BESIDE_BUILD` stay, and `CERTIFIER` goes back into `BESIDE_AUDIT`, `BESIDE_BUILD`, `WITH_RUST` and the parts test's `split`, in the plans' order (`test_the_plans_are_the_documented_ones` compares in order). `check.suite_parts()` is {vllm 3, one_stage 7, pouw-benchmarks 3, certifier verifier 3, composition 3}. `test_shards.py` on the merged tree: 64 passed.
3. Run `r20261114-231128-4e8b` (`research run --on auto --cwd source`, the same command as before; its workload has no pipe, so the pipefail rule doesn't touch it or my earlier run): rc 0.
   - certifier verifier part 1/3: 447.5 s (7.5 min), 551 passed.
   - part 2/3: 431.4 s (7.2 min), 482 passed and 1 skipped.
   - part 3/3: 466.3 s (7.8 min), 549 passed.
   - verity-check: 1032 passed in 445 s. verity-graph: 246 passed in 765 s. repository: 331 passed in 410 s. Nothing failed.
4. The merge's hunks in the two files (`git show --cc f4e4627d0250 -- tools/verity/check/ci.toml tools/verity/check/tests/test_shards.py`):
```
commit f4e4627d0250ff473b3e86814c751de56238b64d
Merge: 7eee5f2c597 2f8b1ccfbe4
Author: Cursor Agent <cursoragent@cursor.com>
Date:   Sat Nov 14 23:10:46 2026 +0000

    Merge remote-tracking branch 'origin/main' into cursor/na-certifier-suite-parts-b51f
    
    Co-authored-by: danielreuter <danielreuter@users.noreply.github.com>
    
    # Conflicts:
    #       tools/verity/check/ci.toml
    #       tools/verity/check/tests/test_shards.py

diff --cc tools/verity/check/ci.toml
index 03af42fe74a,5890bfc146a..469ea589be3
--- a/tools/verity/check/ci.toml
+++ b/tools/verity/check/ci.toml
@@@ -116,24 -116,31 +116,33 @@@ group = 
  # spare once its build narrows. The rest (`lean-suites:*`) goes with the Rust build. Beside them circuit-check has only what
  # `SIBLINGS_GB` leaves it until they end (`check.cc_mem_gb`, `grant_when_alone`), so the longer they run, the longer a cold
  # circuit-check, and its shard's slot holds them back; the 5-pod plan gives it a shard of its own. The suites in parts by test
 -# id (`[tool.verity.tests] parts`) are split so that no shard runs one whole: verity-core-composition and
 -# verity-pouw-benchmarks put one part in each group, verity-vllm one beside the audit and two with the Rust build, whose
 -# shard holds no other parallel suite, so its lean-suites lane's memory (`check.SHARD_ASKS_GIB`) holds their workers, and
 -# benchmarks/one_stage the part holding its longest test beside the audit and the rest (`lean-suites:*`) with the Rust build.
 +# id (`[tool.verity.tests] parts`) are split so that no shard runs one whole: verity-core-composition, verity-pouw-benchmarks
 +# and the certifier verifier's put one part in each group (two parts of a one-process suite on one shard would run in one
- # process), and verity-vllm one beside the audit and two with the Rust build, whose shard holds no other parallel suite, so its
- # lean-suites lane's memory (`check.SHARD_ASKS_GIB`) holds their workers.
++# process), verity-vllm one beside the audit and two with the Rust build, whose shard holds no other parallel suite, so its
++# lean-suites lane's memory (`check.SHARD_ASKS_GIB`) holds their workers, and benchmarks/one_stage the part holding its
++# longest test beside the audit and the rest (`lean-suites:*`) with the Rust build.
  
- # 3 pods: the security analysis's audit beside pytest and the parallel suites; lean-build, the rest of the Lean and the Flock
- # suite; the Rust build and the circuits beside the rest of lean-suites.
+ # 3 pods: the security analysis's audit beside the parallel suites and most of pytest (`pytest:*`); lean-build, the rest of
+ # the Lean and the Flock suite beside seven of pytest's suites (three parallel ones and four of its longest one-process
+ # ones); the Rust build and the circuits beside the rest of lean-suites. Whole beside the audit, pytest ran its parallel
+ # suites one at a time and its one-process suites nine at a time, the first shard's longest test step. The second shard's
+ # pytest takes no cores from a build (`check.groups`), and its ask (`check.SHARD_ASKS_GIB`'s pytest beside lean-suites)
+ # gives the Flock suite's workers memory they lack beside lean-build alone. verity/definitions goes there: beside the audit
+ # and vLLM's first part it would match the 5-pod plan's first shard's smaller ask.
  [[plan]]
  shards = [
-     ["lean-audit:security/Proofs", "lean-audit:security/Properties", "pytest", "lean-suites:verity-vllm:1/3",
+     ["lean-audit:security/Proofs", "lean-audit:security/Properties", "pytest:*", "lean-suites:verity-vllm:1/3",
       "lean-suites:verity/protocols/proofs/recursion/verifier/vstar_cli", "lean-suites:verity-core-evaluate", "lean-suites:pous-check",
 -     "lean-suites:verity/protocols/accounting/certifier/verifier", "lean-suites:verity-pouw-check-difftest",
 +     "lean-suites:verity/protocols/accounting/certifier/verifier:3/3", "lean-suites:verity-pouw-check-difftest",
-      "lean-suites:verity-core-composition:3/3", "lean-suites:verity-pouw-benchmarks:3/3"],
+      "lean-suites:verity-core-composition:3/3", "lean-suites:verity-pouw-benchmarks:3/3",
+      "lean-suites:benchmarks/one_stage:2/7"],
      ["lean-build", "lean-unit-cut", "lean-audit:*", "lean-suites:verity-flock", "lean-suites:verity-core-composition:1/3",
-      "lean-suites:verity-pouw-benchmarks:2/3", "lean-suites:verity/protocols/accounting/certifier/verifier:1/3", "lean-agreement"],
 -     "lean-suites:verity-pouw-benchmarks:2/3", "lean-agreement", "pytest:research",
 -     "pytest:verity/definitions", "pytest:verity/circuits/ops", "pytest:verity-graph",
++     "lean-suites:verity-pouw-benchmarks:2/3", "lean-suites:verity/protocols/accounting/certifier/verifier:1/3", "lean-agreement",
++     "pytest:research", "pytest:verity/definitions", "pytest:verity/circuits/ops", "pytest:verity-graph",
+      "pytest:benchmarks/training_slowdown", "pytest:repository", "pytest:verity/protocols/accounting/pous/verifier"],
      ["flock-circuit-build", "circuit-check", "flock-rust-tests", "lean-suites:verity-core-composition:2/3",
 -     "lean-suites:verity-pouw-benchmarks:1/3", "lean-suites:verity-vllm:2/3", "lean-suites:verity-vllm:3/3", "lean-suites:*"],
 +     "lean-suites:verity-pouw-benchmarks:1/3", "lean-suites:verity/protocols/accounting/certifier/verifier:2/3",
 +     "lean-suites:verity-vllm:2/3", "lean-suites:verity-vllm:3/3", "lean-suites:*"],
  ]
  
  # 4 pods: the 3-pod plan with pytest on a pod of its own, and circuit-check's four shards one to each pod, so a cold
@@@ -142,15 -149,14 +151,16 @@@
  shards = [
      ["lean-audit:security/Proofs", "lean-audit:security/Properties", "circuit-check:1/4", "lean-suites:verity-vllm:1/3",
       "lean-suites:verity/protocols/proofs/recursion/verifier/vstar_cli", "lean-suites:verity-core-evaluate", "lean-suites:pous-check",
 -     "lean-suites:verity/protocols/accounting/certifier/verifier", "lean-suites:verity-pouw-check-difftest",
 +     "lean-suites:verity/protocols/accounting/certifier/verifier:3/3", "lean-suites:verity-pouw-check-difftest",
-      "lean-suites:verity-core-composition:3/3", "lean-suites:verity-pouw-benchmarks:3/3"],
+      "lean-suites:verity-core-composition:3/3", "lean-suites:verity-pouw-benchmarks:3/3",
+      "lean-suites:benchmarks/one_stage:2/7"],
      ["lean-build", "lean-unit-cut", "lean-audit:*", "lean-suites:verity-flock", "lean-suites:verity-core-composition:1/3",
 -     "lean-suites:verity-pouw-benchmarks:2/3", "lean-agreement", "circuit-check:2/4"],
 +     "lean-suites:verity-pouw-benchmarks:2/3", "lean-suites:verity/protocols/accounting/certifier/verifier:1/3", "lean-agreement",
 +     "circuit-check:2/4"],
      ["pytest", "circuit-check:3/4"],
      ["flock-circuit-build", "flock-rust-tests", "circuit-check:*", "lean-suites:verity-core-composition:2/3",
 -     "lean-suites:verity-pouw-benchmarks:1/3", "lean-suites:verity-vllm:2/3", "lean-suites:verity-vllm:3/3", "lean-suites:*"],
 +     "lean-suites:verity-pouw-benchmarks:1/3", "lean-suites:verity/protocols/accounting/certifier/verifier:2/3",
 +     "lean-suites:verity-vllm:2/3", "lean-suites:verity-vllm:3/3", "lean-suites:*"],
  ]
  
  # 5 pods: pytest apart, and circuit-check alone in the last shard, so no step on the gate waits behind a cold circuit-check, in
@@@ -163,13 -169,14 +173,14 @@@
  shards = [
      ["lean-audit:security/Proofs", "lean-audit:security/Properties", "pytest:verity/definitions", "lean-suites:verity-vllm:1/3",
       "lean-suites:verity/protocols/proofs/recursion/verifier/vstar_cli", "lean-suites:verity-core-evaluate", "lean-suites:pous-check",
 -     "lean-suites:verity/protocols/accounting/certifier/verifier", "lean-suites:verity-pouw-check-difftest",
 +     "lean-suites:verity/protocols/accounting/certifier/verifier:3/3", "lean-suites:verity-pouw-check-difftest",
-      "lean-suites:verity-core-composition:3/3", "lean-suites:verity-pouw-benchmarks:3/3"],
+      "lean-suites:verity-core-composition:3/3", "lean-suites:verity-pouw-benchmarks:3/3",
+      "lean-suites:benchmarks/one_stage:2/7"],
      ["flock-circuit-build", "flock-rust-tests", "lean-audit:*", "lean-suites:verity-core-composition:2/3",
 -     "lean-suites:verity-pouw-benchmarks:1/3", "lean-suites:verity-vllm:2/3", "lean-suites:verity-vllm:3/3", "lean-suites:*",
 -     "ratchet-gate"],
 +     "lean-suites:verity-pouw-benchmarks:1/3", "lean-suites:verity/protocols/accounting/certifier/verifier:2/3",
 +     "lean-suites:verity-vllm:2/3", "lean-suites:verity-vllm:3/3", "lean-suites:*", "ratchet-gate"],
      ["lean-build", "lean-unit-cut", "lean-suites:verity-flock", "lean-suites:verity-core-composition:1/3",
 -     "lean-suites:verity-pouw-benchmarks:2/3", "lean-agreement"],
 +     "lean-suites:verity-pouw-benchmarks:2/3", "lean-suites:verity/protocols/accounting/certifier/verifier:1/3", "lean-agreement"],
      ["pytest:*"],
      ["circuit-check"],
  ]
diff --cc tools/verity/check/tests/test_shards.py
index 14a3566a2c8,a8d2bb202fe..d72ac68e4bd
--- a/tools/verity/check/tests/test_shards.py
+++ b/tools/verity/check/tests/test_shards.py
@@@ -31,19 -31,25 +31,24 @@@ PACKAGES = LA.packages(CHECK.tracked_fi
  SECURITY = tuple(f"lean-audit:{p}" for p in ("security", "security/Proofs", "security/Properties"))
  #: those the hand-written plans name: the spec goes with its provers (`declaration.resolve`), so no plan spells its path
  NAMED = SECURITY[1:]
 -#: lean-suites' parts: its suites, the composition and PoUW benchmark suites and vLLM's each in the three parts by test id its
 -#: pyproject declares, the one-stage benchmark's in seven
 +#: lean-suites' parts: its suites, the composition and PoUW benchmark suites, the certifier verifier's and vLLM's each in the
- #: three parts by test id its pyproject declares
++#: three parts by test id its pyproject declares, the one-stage benchmark's in seven
  SUITES = CHECK._parts_of()["lean-suites"]
  COMPOSITION = tuple(f"lean-suites:verity-core-composition:{k}/3" for k in (1, 2, 3))
  POUW = tuple(f"lean-suites:verity-pouw-benchmarks:{k}/3" for k in (1, 2, 3))
 +CERTIFIER = tuple(f"lean-suites:verity/protocols/accounting/certifier/verifier:{k}/3" for k in (1, 2, 3))
  VLLM = tuple(f"lean-suites:verity-vllm:{k}/3" for k in (1, 2, 3))
+ ONE_STAGE = tuple(f"lean-suites:benchmarks/one_stage:{k}/7" for k in range(1, 8))
  #: the lean-suites every plan runs beside the security analysis's audit, and beside lean-build; the rest go with the Rust build
  BESIDE_AUDIT = (VLLM[0], *(f"lean-suites:{s}" for s in ("verity/protocols/proofs/recursion/verifier/vstar_cli",
 -                                                        "verity-core-evaluate", "pous-check",
 -                                                        "verity/protocols/accounting/certifier/verifier",
 -                                                        "verity-pouw-check-difftest")),
 -                COMPOSITION[2], POUW[2], ONE_STAGE[1])
 -BESIDE_BUILD = ("lean-suites:verity-flock", COMPOSITION[0], POUW[1])
 -WITH_RUST = (COMPOSITION[1], POUW[0], *VLLM[1:], "lean-suites:*")
 +                                                        "verity-core-evaluate", "pous-check")),
-                 CERTIFIER[2], "lean-suites:verity-pouw-check-difftest", COMPOSITION[2], POUW[2])
++                CERTIFIER[2], "lean-suites:verity-pouw-check-difftest", COMPOSITION[2], POUW[2], ONE_STAGE[1])
 +BESIDE_BUILD = ("lean-suites:verity-flock", COMPOSITION[0], POUW[1], CERTIFIER[0])
 +WITH_RUST = (COMPOSITION[1], POUW[0], CERTIFIER[1], *VLLM[1:], "lean-suites:*")
+ #: pytest's suites the 3-pod plan runs beside lean-build; the rest (`pytest:*`) go beside the security analysis's audit
+ PYTEST_BESIDE_BUILD = tuple(f"pytest:{s}" for s in (
+     "research", "verity/definitions", "verity/circuits/ops", "verity-graph",
+     "benchmarks/training_slowdown", "repository", "verity/protocols/accounting/pous/verifier"))
  
  
  @pytest.fixture(autouse=True)
@@@ -110,15 -116,33 +115,35 @@@ def test_the_5_pod_plan_runs_pytest_in_
          D.resolve((("pytest:nowhere", "pytest:*"),), CHECK.JOBS, _parts(n=5))
  
  
+ def test_the_3_pod_plan_runs_pytest_in_two_shards_by_suite_neither_beside_a_build(tmp_path):
+     """Whole beside the security analysis's audit, pytest was the 3-pod plan's first shard's longest test step: its parallel
+     suites one at a time beside verity/commitments, its one-process suites nine at a time.  The plan names seven of its
+     suites beside lean-build, whose shard holds no build to share cores with (`check.groups`), and the rest (`pytest:*`)
+     beside the audit; the two cover every suite of pytest's once, and the second shard books pytest beside lean-suites' ask
+     (`check.SHARD_ASKS_GIB`), not lean-build's."""
+     names = [s.name for s in CHECK.SUITES.discover() if s.name not in CHECK.AFTER_LEAN]
+     shards = [CHECK.shard(f"{k}/3")["parts"] for k in range(1, 4)]
+     held = [[p.partition(":")[2] for p in s if p.partition(":")[0] == "pytest"] for s in shards]
+     beside_build = [p.partition(":")[2] for p in PYTEST_BESIDE_BUILD]
+     assert sorted(held[1]) == sorted(beside_build) and not held[2]
+     assert sorted(held[0]) == sorted(n for n in names if n not in beside_build)
+     for k, jobs in ((1, "16"), (2, "16")):
+         [argv] = [argv for g in CHECK.groups(tmp_path, 16, only=shards[k - 1]) for name, argv, _ in g if name == "pytest"]
+         assert argv[2:2 + len(held[k - 1])] == held[k - 1] and argv[argv.index("--jobs") + 1] == jobs, argv
+     assert CHECK.shard_ask_gib(shards[0], []) == 330 and CHECK.shard_ask_gib([p for p in shards[0] if "lean-audit" not in p], []) == 240
+     assert CHECK.shard_ask_gib(shards[1], []) == 240 and CHECK.shard_ask_gib(shards[1], ["lean-agreement"]) == 460
+ 
+ 
  @pytest.mark.parametrize("n", sorted(D.plans()))
  def test_every_plan_runs_each_suite_in_parts_on_three_shards(n):
 -    """verity-core-composition, verity-pouw-benchmarks and verity-vllm declare three parts by test id: lean-suites' parts are
 -    its suites with each of those as its three, and every plan's shards, gated and after landing, hold each part once and no
 -    shard all three of a suite, so no shard runs one whole: the composition and PoUW suites one part on each of three shards,
 -    vLLM's one beside the security analysis's audit and two beside the Rust build."""
 -    split = {"verity-core-composition": COMPOSITION, "verity-pouw-benchmarks": POUW, "verity-vllm": VLLM,
 +    """verity-core-composition, verity-pouw-benchmarks, the certifier verifier's suite and verity-vllm declare three parts by
-     test id: lean-suites' parts are its suites with each of those as its three, and every plan's shards, gated and after
-     landing, hold each part once and no shard all three of a suite, so no shard runs one whole: the composition, PoUW and
-     certifier suites one part on each of three shards (two parts of one suite on a shard run in one process), vLLM's one beside
-     the security analysis's audit and two beside the Rust build."""
++    test id, and benchmarks/one_stage seven: lean-suites' parts are its suites with each of those as its parts, and every
++    plan's shards, gated and after landing, hold each part once and no shard all of a suite, so no shard runs one whole: the
++    composition, PoUW and certifier suites one part on each of three shards (two parts of one suite on a shard run in one
++    process), vLLM's one beside the security analysis's audit and two beside the Rust build."""
 +    split = {"verity-core-composition": COMPOSITION, "verity-pouw-benchmarks": POUW,
-              "verity/protocols/accounting/certifier/verifier": CERTIFIER, "verity-vllm": VLLM}
++             "verity/protocols/accounting/certifier/verifier": CERTIFIER, "verity-vllm": VLLM,
+              "benchmarks/one_stage": ONE_STAGE}
      assert CHECK.suite_parts() == {s: len(parts) for s, parts in split.items()}
      assert SUITES == [x for s in CHECK.AFTER_LEAN
                        for x in ([p.removeprefix("lean-suites:") for p in split[s]] if s in split else [s])]
```

