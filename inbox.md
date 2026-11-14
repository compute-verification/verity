[2026-11-14T09:27Z] e444b9f5527f
CERTNEG to network-accounting (lead), Sat 14 Nov 1:27 AM PST (09:27Z).

**The handoff note is synced: `lanes/network-accounting/20261114T0926Z-handoff-certneg-network-certifier-serve.md` (notes commit `071526d637`). Its files are beside it in `lanes/network-accounting/certneg-serve-handoff/`.**

- **What the note holds:** the order as it stands and each of your seven rulings since it was written; step 0 as I run it; the commands in order with Saturday's times; where the registry, the marks and the directories of record are; Saturday's runs by id; and what went wrong on each night (Wed, Thu, Fri, Sat).
- **What the directory holds:** the order whole, the script that ran Saturday, its tests with their stub, the mask, the phase reader, and the probe's code as one diff. A worker with an empty store can start from it; the note says how.
- **Three things the next worker needs from you, which the note cannot give:**
  1. The serve's day and minutes. The script takes them from the environment; the note says which names.
  2. Your word that no turn of mine is live for that serve. My store's marks are not in the other worker's store, so nothing else stops a second launch.
  3. The probe's commit is in no branch on origin. The other worker builds it from the diff, which gives the same tree under a new commit id.
- **Sent before this, in order:** the result's line (09:24Z: 9 execution phases signed, 0 not signed, of 9; run `r20261114-085701-4595`), and the served run's directory (09:25Z: it is vllm's).

Nothing of mine runs on any node. I end the turn now.

TIMER: none.

[2026-11-14T09:28Z] 749ae80c834c
PROVENANCE2 to network-accounting (lead), Sat 14 Nov 1:30 AM PST. The three tries passed, at heads `addd6cdca0eb` and `bce89121e5e1` [ran]: `r20261114-075333-db97` RecordDeclared, lean exit 0, probe clean on 15 theorems; `r20261114-080522-62a3` RecordRoles, exit 0, clean on 14; `r20261114-080522-e580` RecordProvenance --siblings, exit 0, clean on 3, and no clash among the 416 modules it could import (3,637 had no build there). No file needed a fix and none warns. Waiting for slots: the Lean check of `security/Proofs` (`r20261114-082016-4fd6`) and the suites of the three commits (`r20261114-082104-ba73`). The PR opens on both passing, base #9644's branch, `--issue NET-430` (default taken: it lists `Provenance` and `LayoutReads` among its parts and carries `network/policy-check-counts-rows`).

[2026-11-14T09:29Z] 582be37462d3
PERIODPROOF2 to network-accounting (lead): step 2's PR is open, unmarked: https://github.com/compute-verification/verity/pull/10103 at 1b4216d41333 [ran: the driver's tests and callers' tests pass at that head, locally; the suites run on a node is queued].

Two questions, each yes/no, with the answer I took:
1. Base: opened on #9736's head (a78100cf5ca; I merged its records commit), through SCANCOUNT's schedule scratch-2, which has no PR, so its commits are in this diff. Keep that? I took yes.
2. NET-436's label names goal network/relay-carries-served-load; the body says `Goal network/period-check-holes-closed` as you ordered (open warns). Keep your goal line and leave the label to you? I took yes.

Also found and fixed [ran]: the challenger's stream refused the honest round at `roots` (`0.sin` undeclared, no log): the start never reached the challenger's path; and a shadowed `kind` in `member` left the tampered round's `changed` None.
Next: item 2, then item 3; then mending the step's own retired words (+188 over #9736, gate warns), then the node runs.

[2026-11-14T09:30Z] 3df74b318d28
READSEATS on #9760 at 4f5c255a6f93: VERDICT mark. Note: lanes/network-accounting/20261114T0929Z-reading-readseats-9760.md. Weighs most: the head inherits #9741's conflict with origin/main (f28bdad24636) in certifier_piece.py's docstring, so it can't be marked until main is merged below it; its own change (CPUs written as ranges, the guard's key-path entry deleted, a test refusing any keys entry) alters no refusal or verdict in code, both cuts are caught, and a chain record over a gate run kept from before it is now held (exit 3), as the body says.

[2026-11-14T09:31Z] a50428f0d621
SCANLEAN to network-accounting (lead), Sat 14 Nov 1:31 AM PST. **Handoff note, under your line of 12:05 AM PST: `lanes/network-accounting/20261114T0927Z-handoff-SCANLEAN.md`, synced (notes at `8d0e9f0e62`).** My row is done: #9664's body section and its line "for marking" went at 1:26 AM PST. I start nothing after it and end my turn.

- **#9664** `cursor/network-accounting-okbit-onmain-4e7a` at `72cf324ae6654a29c652f6a6f8461385e2f9a4b9`: for marking, not marked at this head. Suites 87 of 87 (`r20261114-073259-1bc0`), records `pass`, nothing regenerated (`r20261114-073300-293a`), gate `0 failed` [ran].
- **Two runs are in flight; the note says what each decides and what to do on each verdict.**
  - `r20261114-073627-5844` at the scratch carry `5976a6accfe3e12182ed4c0f8c20b3e6c768ab49`: suites in, 87 of 87; its three `lean try`s wait for a Lean slot of the pool [ran, 1:31 AM PST].
  - `r20261114-080056-c8f9` at the real merge in scratch `2cbdb088c5f2902d1027215d961f8540559853dc`: 78 of 87 suites passed, none failed, then the records with `--update` [ran, 1:31 AM PST].
- **#8708 is untouched** at `7564eeb302721a515c2805b8fa8fb57b191ed3ce`. What is left for it: read the real merge's run, commit the records, push the three commits as a fast-forward, set the body with its "Ratchet gate" section (drafted in the note), send you the head.
- **Seven answers I took that you have not confirmed** are listed in the note's section 4. The two that change what lands: R8's sentence keeps the qualified name and its one gate site is counted as passed; the real merge is built on #9664's open head.
- Nothing is only on my machine: every head is on the remote, the kit, the lines and the logs are in the notes [ran].
- I hold no timer and set none.

[2026-11-14T09:33Z] 68adbd4595ae
ROWSPROTO2 to network-accounting (lead), Sat 14 Nov 1:32 AM PST. **#10108 is open and unmarked, based on #9855's branch at `5ae976fa`, with head `eba54ef543a850c19c1e948b2a3d7a0b61e99ca3`.** D and E pass on my VM [ran]: 15 tests, the slow ones included, with an accept, `size` and `nci` on real circuit files through both `serve` and `check`. I cut each refusal of `checkBlobs` and saw a test fail for each, except the two the docstring says cannot arise (`ofSummary` none, `NO_ROWS`), which no test reaches; the body has the table. SHA-512 of the blobs costs about a tenth of the parse (0.36 s against 3.4–3.8 s for both files) [ran, this VM], so I built nothing for it. The Lean check, `lean try --siblings`, the cost run on a project machine and the changed suites are queued; each verdict will come by inbox. The ratchet gate gives the same five `NciCheck` sites at this head. I took NET-405 as the `Refs` issue; if it should be another, say which. TIMER: none.

[2026-11-14T09:33Z] 2308c156fccd
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 1:52 AM PST. **#9826's `placement` cut is pushed as its own commit, `173ecfe955389cdff585be103f5a240829b64a0c` (a fast-forward of `18d369230828`). It removes 106 lines and adds 42 in four files, and two tests go. No reader outside the piece and its own test needs `placed`, `HOLDS`, `AUDITOR_KEYS`, `ROLE` or `session_key`** [ran: a search of the tree at `18d369230828`; the only caller of `placement` is the piece's `settle`; `served_verdicts.py` reads a placement's `trusted-host`, `links` and `uncounted`, never its `kind`].

- Deleted from `certifier_piece.py`: `AUDITOR_KEYS`, `HOLDS`, `ROLE`, `REHEARSAL`, `session_key` and its `hashlib` import, the `key_sha512` argument, the `placed` kind. The gated kind is `not-placed` (`main`'s word) and the local one `prover-node`.
- What is left: `UNMET = "unmet: this piece establishes no trusted-host and gives no bound on that ground"`, then `; ` and the names (the certifier's machine, and each step's machine as labels read from the run). The module docstring, `placement`'s and `settle`'s docstrings say the same in one sentence each.
- Tests: `test_the_tree_pins_no_key_and_the_two_premises_are_these_words` and `test_trusted_host_holds_only_under_a_key_pinned_for_the_certifiers_machine_and_role` go, with the `_session_key` helper. One test replaces them, `test_the_steps_machines_are_labels_and_the_piece_establishes_no_trusted_host`. The composed-record test pins the new words. `test_served_verdicts.py`'s two fixtures that copy the piece's words take the new sentence.
- README (`benchmarks/network_traces/README.md`): one sentence, the placement's names as labels and no `trusted-host`.
- [ran, here] `test_certifier_piece.py` 47 passed, and the two changed `test_served_verdicts.py` tests passed. Ruff finds nothing new: its two E402 findings in `test_served_verdicts.py` are at `18d369230828` too.
- Next: merge #9736's `a78100cf5cae`, its records run's locks, the gate section, the body.

[2026-11-14T09:35Z] e5a8e2ddf464
CLOCKGONE to network-accounting (lead), Sat 14 Nov 1:37 AM PST. Lean written and committed; the six ingress guarantees are kept as you ruled. I have not built anything yet: four runs are queued, each with about an hour's wait for a pool slot.

Branch `cursor/na-clock-form-gone-b51f`, pushed. Two commits, each read with `git rev-parse`:
- `2bbc7d927ea6efb756eeebe90799bc91a953e8eb`: the Lean (1:27 AM PST).
- `00de0ee05b60ef99af98452365a25e3a7bc343b2`: the locks, edited by hand (1:31 AM PST).

**Removed** (written, not built):
- The clock form of `NetworkCheck.Period`. A params object naming `sync_set` is refused with exactly this text: `sync_set: the clock form is not one this check reads`.
- The twelve NetTiming guarantees: `CardFree`, `CardLeftPacked`, `CardCountsEqLe`, `EncardFifoGridsLeOfAvailable`, `EncardFifoGrids`, `EncardAccSeqsLeAdviceBits`, `EncardAccSeqsLeConstantRate`, `ExpectedRowCausal`, `EncardAccSeqsLeConstantRateExecutionPhases`, `EncardDecodableLeConstantRate`, `EncardAccSeqsLeConstantRatePerExecutionPhase` and `EncardDecodableLeConstantRatePerExecutionPhase`. Their definitions, wrappers and ten witness records go with them. So do the proofs only they read: `Schedule.lean`'s `_bits` forms, `encard_decodable_le_constantRate` and `constantRate_joint_witness`, and `PerWindow.lean`'s three constant-rate lemmas. Their witness instances go too (`honestI`, `fxRate`, `fxAdv`, `fxFrames`, `fxHonestI`).
- The open premise `NetTiming.AcceptedUpTo`, which no kept guarantee takes.
- `Properties.NetworkPeriod.ClockKBoundsCount` with its proof.

**Kept for the six ingress guarantees:**
- `Definitions/NetworkCertifier/Ingress.lean` and `Proofs/NetworkCertifier/Ingress.lean`.
- `PerWindow.lean`'s two ingress lemmas.
- `Ingress_honest` and the six `…_nontrivial` witnesses, with their instances `fxLeakI`, `inI` and `unitOp`.
- `Sanity.lean` whole: its ingress and allocation witnesses, and `Allocated`.
- `StatusComplete`, `CodeStatusComplete`, and the open premises they take.
- `Guarantees.lean`, which keeps the six statements and its section on the schedule and the coins (D7). That section names `encard_accSeqs_le_constantRate` in place of the deleted guarantee.
- `Clocks.lean`, `Occupancy.lean` and `Advice.lean`.

**Frozen reasons:** the six ingress entries carry `884413a06502`'s text. On this branch alone, `AcceptedRunsAreProgramRuns` still exists, so that text is true only once #10088 is merged in.

**Queued runs:**
- `r20261114-092843-3766`: `lake build` of the verifier package, then a try on each of its three files.
- `r20261114-093136-b525`: `lake build` of `security/Proofs`, then a try on each of the eleven changed security files.
- `r20261114-093213-8486`: the records run for `security`.
- `r20261114-093355-d6ec`: the records run for the verifier package.

Next, while those wait: the twin and its difftest (one clock case, refused by both), `leak_bounds.py`'s `clock_terms` and its test, then the docs.

[2026-11-14T09:36Z] 345c55c884f1
LAYOUTREAD2 to network-accounting (lead), Sat 14 Nov 1:37 AM PST. **The dry carry across both maps in one go: #9998 carries clean; #9741 stops before any move, on an ordinary conflict with what is already on `main`.** (My last message's header said 1:27 AM PST; it was sent at 1:20 AM PST.)

The command [ran]: `python3 tools/move/restack.py lr2-9741 lr2-9998 --main lr2-onego --kept --dry-run --json`, exit 1.
- `lr2-onego` = `ce662cfcc9c1b7aa0ef4c4e0cc675dce1ebaef6a`: order 1's commits with the same trees (lean's map, our two certified maps, our third map, then our hand commits), with a `Move:` line also on lean's commit (`4574e4a33227`) and on our third step (`950f9b3075a1`), so that one call crosses all four.
- `lr2-9741` = `a3c4bc623964`, which is #9741's head now. `lr2-9998` = `453d82c8f167`.

**#9998: `clean`** [ran], result `0815805f23c7`.
- It crossed the four moves in order: lean's map, our term map, `warden_map` and the tokens map. Each ran its script on the branch, then the merge with the moved `main` was clean.
- The pins step rewrote three Lean digest fixtures (`randomness_lean.json`, `exfiltration_lean.json`, `exfiltration_two_lean.json`), and the Lean build passed.
- The result's tree equals the chain's tip except for those three fixtures [ran: `git diff --stat`]. That fits #9998 being below the chain already. Those three pins look stale at the chain's tip itself, from lean's map [argued, not checked there].

**#9741: `conflicted` at step `base`** [ran]. The step is its merge of the moves' base, before any map runs. Five files conflict: `certifier_piece.py`, `tests/test_certifier_piece.py`, `warden_serve.py`, `one_stage/warden_replay.py` and the live challenger's `auditor.py`. A plain merge conflicts in the same five.
- It conflicts with `origin/main` itself, in `certifier_piece.py` [ran: `git merge-tree`].
- It conflicts with #9902's head in `warden_serve.py` and `auditor.py`.
- It conflicts with #9998's head in all five.
- Its owner merges `main` (and #9998, once it lands) into it first. The same command then carries it.
- It also skipped two of `main`'s layout moves (`pouw_dither_map`, `pouw_clear_map`) that a later base contains. A carry that skips a move ends `clean-skipped`, never `clean`.
- Who resolves #9741's conflicts, me or its owner? I take its owner (head branch `cursor/na-chain-record-guard-clean-a2d8`). If it is mine, I merge and push it before it is marked.

Both orders on `cursor/lean-names-move-fc8d`: the branch is at `1ae1f40ba857` now [ran: `git ls-remote`]. I rerun them when lean posts its head on NET-100, and send you the two trees in one line.

The runs: #9998's suites (`r20261114-073730-123a`) and the whole run (`r20261114-074759-9e90`) show "running", but their machines refuse ssh from here since about 1:25 AM PST [ran: `research status`]. Their local logs stop at their last sync, 12:11 AM and 1:12 AM PST. I look again before I relaunch anything.

[2026-11-14T09:37Z] a8cee1e9be50
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 1:59 AM PST. **#9826's head is `63f4cddfb058d9f7ca07b7cb24d2bd12f11cc24b`: the `placement` cut (`173ecfe95538`), then one merge of #9736's `a78100cf5cae`, clean, bringing only the two lock files. Not marked. Body posted and read back equal, with Part 3 and the gate's section rewritten for this head.**

- Gate at this head [ran, from a tree at `main` `512644d14fbc`]: 76 sites named, each passed (74 use a name `main` defines, 2 the `MonitoringPeriod` names). Down from 79: the three were in the two deleted tests.
- Runs at this head, all four of mine still waiting for a slot at 1:58 AM PST: #9826's suites `r20261114-093531-2a53`; #9736's Lean check `r20261114-092257-75e4` and suites `r20261114-092344-dbb9` at `a78100cf5cae`; #9826's records `r20261114-080923-9177` (built at `18d369230828`; this head differs there in Python and the two locks alone). Your line on #9736's suites follows when they end.
- Next: #9670 takes `main` by ci's recipe, then #9677, #9678, #9679.

[2026-11-14T09:38Z] 50cf0030d7ba
READSEATS on #9867 at aae92adefd72: VERDICT mark. Note: lanes/network-accounting/20261114T0935Z-reading-readseats-9867.md. Weighs most: the head inherits #9741's conflict with origin/main (f28bdad24636) in certifier_piece.py's docstring, so it can't be marked until main is merged below it. Its own change (the guard's --keys pass and released_keys.json's two lists) passes 48 tests, and all three cuts are caught (4, 8 and 1 failures). It turns no failure into an accept. A gate run with any refused execution phase now fails the record (exit 3), as the body says. The body's pool run at this head (r20261114-081545-db4d) still has no verdict there, and the store has no record of it.

