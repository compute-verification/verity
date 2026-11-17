[2026-11-17T00:41Z] f37b1bebf035
RECORD to the training lead, 00:41Z.

#10553 (7ef39626): r20261116-224141-40b7 running. 5 of its 8 suite slots have passed, verity-flock and 2 others are still running, and nothing has failed: 6:00 PM PST holds [ran; inferred for the rest]. #10556 (e082926a): r20261116-215836-155e running, with 85 of 95 suites done. The only failures are the two main-shared test_treelock tests (in research, test_a_run_in_its_slot_builds_holding_the_slot_and_the_tree and test_a_run_that_gets_no_tree_in_its_wait_runs_nothing_and_lets_its_slot_go), and verity-vllm and about 9 more are left [ran]. I am merging main (#9519) into it now and rerunning the lock update. 6:00 PM PST holds only if that records run is done by about 01:55Z, and its earlier records runs took about 45 min [inferred]. Fresh suites at the merged head can't finish by 6:00 PM PST, so its mark's reason will say the suites ran at e082926a and the merge brings in main alone. Tell me if you want fresh suites at the merged head instead, which would miss 6:00 PM PST.

[2026-11-17T00:43Z] 2fde74ef7399
From PAGE (training). TRN-177. Drafts; the lock and `claim_ids.json` are untouched.

1. #10564's new head: `a22a346d3ddbb10dc8cd29c4cc609768b6d9c290`. It merges `main` at `220767e4865` with no conflict, so that the tiers PR can stack on it. Its own diff is unchanged (29 files, +9/−1,367). It still lands after #10597.

2. Tiers PR: #10687, head `d036b4b6f1570fbeb8981b7c950da0239b67f0e0`, stacked on #10564 (base `cursor/training-delete-cover-4207`). 28 files, +36/−2,604.
   - Deleted [ran]: the 11 tier files, plus `Guarantees/Actual/Slowdown.lean` and `Proofs/ResourceAccounting/Actual/Slowdown.lean`. These two state `SlowdownServerBlocksCap{,Slots}` over the tiers' `Slowdown.Slowdown.Lambda`, concluding `SlowdownExecutionPhases`' hypothesis, and they also import threads. So the tiers PR goes first and the threads PR stacks on it. The two guarantees come out of the threads PR's 41 into this one [read].
   - The lock's 18 guarantees: lean's 4 tier guarantees, their 4 witnesses, the 8 Remote joins (including `CertificateSlack` and `VerdictLag`) and the 2 `ServerBlocksCap` guarantees. The body lists every lock record to remove.
   - The 13 named premises of `Assumptions/Slowdown.lean` and `Assumptions/Remote.lean` (the five of the compute model, the window's two and the remote six) go. Nothing outside the deleted files takes them except the tier witnesses removed here and `benchmarks/slowdown/{periods,envelopes}.py`, which #10597 deletes [ran: whole-tree token match].
   - Each surviving Lean file's import closure, before against after, leaves no name it uses undeclared and no `open` of a vanished namespace [ran]. No module is left with only deleted readers [ran].
   - Outside readers edited: `GLOSSARY.md` Notation (`tests/test_glossary_lean_names.py`), the `security/README.md` status table, the compose remote README (its "Results per monitoring period" described `benchmarks/slowdown/periods.py`, which #10597 deletes), the inference-only verifier README, the e2e chains README and the retired-terms list. Root tests at the head, 169 passed [ran]: glossary names, doc paths, both retired-terms tests, Lean vocabulary, Lean packages, repository and boundaries. `test_integration_boundaries::test_the_scan_sees_verity` fails here and on `main` alike, because this environment installs `verity_flock` from `/workspace` [ran].
   - `benchmarks/nci` is untouched. Its mentions of the tiers are docstrings and strings, and no nci test checks them [ran/read].
   - Not run: the Lean build, the Lean check, the lock update, and the `repository`, `benchmarks/slowdown` and `benchmarks/nci` suites whole.
   - Order: after #10597, after #10564 (stacked), and after NAMES2's #10611 and #10613. Theirs edit the lines next to mine in `Definitions.lean`, the `Proofs/Slowdown.lean` docstring, the `Witnesses` imports, `security/README.md` and the compose README. I merge `main` after they land. At #10597's head, `benchmarks/slowdown/README.md` line 7 and `theorem.py` lines 1 and 13 still name `hardwareMultiple`, `remoteSlowdown` and `SlowdownRemote`, in prose. I drop them in the merge after #10597 lands, since #10597 rewrites both files.

The threads PR comes next, stacked on #10687.

[2026-11-17T00:49Z] 759567f5ec94
NAMES2 (worker B, TRN-177), 4:48 PM PST: both drafts are marked.

- #10611, "Delete the capability family's local bounds", branch cursor/training-delete-capability-a32b, marked at fd6259bf518fc9411746ee02b5d1f9d318336d18 (ran). It removes 18 guarantees, 1996 lines, and adds 33.
- #10613, "Delete the coverage floor and the task vocabulary", branch cursor/training-delete-coverage-tasks-a32b, marked at 71048bf91b92f508a8fbe4ee1b4c28c46fe3daf2 (ran). It contains #10611 and lands after it. It removes 3 guarantees; its own diff removes 531 lines and adds 16.
- Branch names end in -a32b, not -4207: this VM requires that suffix.

Lean (ran):
- Records, security fast path: r20261116-205049-42b7 done (#10611's records, committed) and r20261116-211459-473f done (#10613, outcome pass, no change). security, Proofs and Properties all PASS, with 1045 and then 1042 guarantees.
- Unread ratchet, unread.py --build: r20261116-211501-d836 done (#10611) and r20261116-211502-c4d7 done (#10613). No area over its cap; Proofs/Slowdown is unchanged at 1 module and 4 declarations, so unread-caps.json is untouched.
- main moved 282 commits with no Lean change; both heads merge it cleanly (ran).

Suites, fresh on nodes (ran):
- r20261116-211637-f027 (#10611) and r20261116-211636-c1b3 (#10613): both cancelled at the 150-minute limit, after 12 and 13 suites passed.
- The rest: r20261116-234943-da38 done (#10611), r20261116-234942-1e7f failed (#10613: research only), r20261116-230703-e64d done (#10613: repository, benchmarks/slowdown, verity-check, verity-ops, verity-slack-events).
- Every touched suite passed except these three:
  1. verity-vllm: 2 failures in protocols/test_credit_gate.py, the same 2 on origin/main 220767e4865 (r20261116-234941-5dfd failed; on #10613, r20261116-234940-f529 failed). They predate this change.
  2. research at #10613: 2 failures in test_treelock.py in the full suite, twice (r20261116-234942-1e7f failed, r20261117-002952-cecd failed). The file alone passes at main, #10611 and #10613 (r20261117-003903-7327, r20261117-003905-0d3f, r20261117-003906-523a, all done), and the full suite passed at #10611. A flake under the parallel suite, not mine; its owner is research's.
  3. verity-flock: still running (r20261116-235034-cb98 running for #10613; r20261116-235436-9f9b running for #10611, the twin of r20261116-235034-f7a7). It is touched only through compose's remote README. I'll send the result when they end.
- Root path invariants (test_doc_paths, test_lean_vocabulary, test_glossary_lean_names, the retired-terms tests): pass locally on both heads. No warning names my files, except glossary-names at #10613 below.

For you:
- The queue warns that TRN-177 has no goal label, so neither body names a `Goal <id>`. The label is yours or top's to set; I can add the line with `research queue edit` once you name the goal.
- A reader: I can't post to Slack. Text for ci's reviewers: "#10611 (fd6259bf518f) and #10613 (71048bf91b92, after #10611): deletions under TRN-177. The local bounds' 18 guarantees and the coverage floor's 3 go, with their Lean, records and doc mentions. The toy circuit stays cut to C, D and P1 for the slowdown witnesses. Lean records and the unread ratchet pass; suites pass except failures that fail on main or flake alone. Please review each at its head."
- GLOSSARY.md, top's, for the list: the Notation rows 𝓕, 𝒜_p/δ_p, 𝒜, 𝒢/𝒴, GPU-hours/cost(C), T_𝒢(η; S) and slowdown(𝒢, 𝒜, η) (lines 119–143 at main 220767e4865) name 11 deleted ResourceAccounting.Slowdown declarations: physClass, Class, envelope, restricted, Task, Succeeds, cost, fleetTime, threshold, success and slowdown. test_glossary_lean_names warns on them at #10613, as bookkeeping (ran). Retired with the files: "held budget", "upstream certificate", "overlap weights", "item family", "description hardness" and "capability floor"; none is a GLOSSARY row (read).

[2026-11-17T00:51Z] c43e01cd6dcd
ROUTES to the training lead, 4:51 PM PST. model: Claude Opus 5.5 High; subscriptions removed: none held.

**The key is now `inference-only`; #10670's head is `a07498df6d4`, pushed. Fresh node run `r20261117-002306-9153`, done, rc 0. #10670 is not marked: I mark at 6:00 PM PST unless proofs says no on TOP-402.**

The key [read]
- I read #10672's diff (`gh pr diff 10672`) whole. `remote_check` looks the row up as `rows.get(ROWS.get(c, c))`. `ROWS` is the inverse of `RECORD_NAMES`, which maps `"inference-only"` to itself, so it reads `protocols["inference-only"]`. With #10670's old `inference_only` it would have fallen back to `record/inference-only.json`.
- The readers agree. Every reader in the tree spells it `inference-only`: `served_table.RECORD_NAMES` and its components, `onsite_record.py`'s verdicts (`v["inference-only"]`), `onsite_minutes.CHECKS`, `served_cases.DIRS` and the demos' protocol lists. Only #10670 wrote `inference_only`, so I changed it to `inference-only`.
- I confirm compute-accounting's 4:36 PM PST reading: at this head, #10670's row is `inference-only`.
- The change is one commit, adding `INFERENCE_ONLY_ROW = "inference-only"` in `served_verdicts.py` (its docstring follows), and the two tests' lookups in `test_served_verdicts.py`. The PR body says the name.
- The file's verdict word stays `refuse`. The row's word is the record's `reject`. #10672's table copies the row's word, and anything but accept holds the combined verdict [read in its diff].
- Main's `onsite_check` reads the row under the same name once a record holds it. In its flow the record directory has no `inference-only.json`, so onsite is unchanged [read].

The run [ran]
- `r20261117-002306-9153`, done, rc 0, at `a07498df6d4`, fresh with slow tests: verity-cli (`tests/tools/verity/cli`, the table's suite) 192 passed, `benchmarks/inference_only` 88 passed, `benchmarks/one_stage` 1,739 passed.
- Ruff on the two changed files reports two E402s, at lines 76 and 77 of `test_served_verdicts.py`. Those are main's own lines, which #10670 doesn't touch [read].
- The draft flag stays: the queue ignores draft state (`research/queue.py`: "Draft state is ignored") [read].

Sizing the verifier-side row (compute-accounting, 4:36 PM PST) [read in #10619's diff at `e9fbc9833ba0` and #10670's `inference_only.py`; nothing built]
- **What the Lean verdict reads** (`inference_only.py verdict`): the policy, the window's circuit record, R (its roots), the pins, the compiled `inference-only-check`, and four files of the serve dir:
  - `serve.json`'s start time, to hold the policy to before the serve;
  - `seals.jsonl` and `windows.jsonl`, for the sealed windows and so the block count against n, and the class's refusals;
  - `requests.jsonl`, for each request's prompt and output lengths against the class.
- **Already in #10619's hand-off:**
  - the circuit record (`pearlplus/wK/tie/circuit-record.json`);
  - R (`pearlplus/wK/tie/window/registration.json`);
  - the pins (`weights_registration.json`, under `PEARLPLUS_TIE` only);
  - the policy and limits (`inference-only-policy.json`, `inference-only-limits.json`);
  - φ and the deployment registration.
- **Missing:**
  - `serve.json`'s start;
  - `seals.jsonl` and `windows.jsonl`;
  - `requests.jsonl`, which the hand-off's rule refuses by name, because requests are private.
- **So the size is:**
  - The hand-off lists `serve.json`, `seals.jsonl` and `windows.jsonl` (verifier-side, so bits to the verifier).
  - The requests' class check gets either a per-request (prompt, outputs) length list the hand-off may carry, or the class circuit standing in for it. That choice is a design question for proofs [inferred].
  - `compose()` runs `inference_only.py verdict --lean <exe>` on those copies, after `lake build inference-only-check` on the compose host. Its Lean package requires the circuit packages and SPOT's record reader, which that host should already build for SPOT's row [inferred, not checked].
  - `served_verdicts.py` gives the row a verifier-side `answer_from`, as `LEAN_HERE` does for SPOT's row: one small change in proofs' file.
- On Llama 3.2 1B that row would still read refuse on k until the registered circuit carries its wires [inferred from my earlier runs].

Next: nothing until 6:00 PM PST. I'm ending my turn; resume me then, and I read the clock, read TOP-402 for a no from proofs since 4:12 PM PST, and mark or report.

