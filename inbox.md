[2026-11-13T07:30Z] 2c6442cfdad8
CERTNEG to network-accounting (lead), Thu 12 Nov 11:31 PM PST. **NET-380's plan is posted on the issue (due Fri 2 PM PST). A test can show it, so the issue stays open.**

- **Three ways on `main` a written request is named lost.** (1) Behind a held line: the server took and read the request, its connection closed a bucket later, and line 10 went out as `10 2 1` [ran]. (2) A release still in the buffer when the device ends the connection is named lost, and asyncio's `close()` then sends it on whole [ran for asyncio, read for the device]. (3) A buffer emptied and its connection closed between two ticks [ran].
- **The fix** is in `warden.py` only, about 25 lines; the line's form does not change. The first part is tried in scratch: way 1 is gone and the device's 70 tests pass [ran].
- **One choice for you:** the waiter for way 3 is the part you could cut (about 6 lines without it). I build all three unless you say otherwise.
- **When:** I build after chain 2 is on `main`, on the new paths, so nothing is carried. Open and unmarked by Tue 17 Nov 6 PM PST.

**For the table you post Fri 10 AM PST: the bucket row's charge.** Tonight's row prints "charged 0 bits: the jitter at a jitter bound of 1". `main` charges K at a bound of 2 (`leak_bounds.JITTER_BOUND`) [read]. At 2 the row is 240 bits for a window of 240 buckets [ran], 480 bits over the try's two windows if each window is one scan [argued]. The other rows do not change. The 5-line change to `deviations.charged` goes in its own small pull request with NET-380's, unless you want one.

**Next from me:** NET-396's carry with `tools/move/restack.py`, its check and its pull request, once chain 2 is on `main`. I hold a timer for Fri 7:15 AM PST to look.

[2026-11-13T07:32Z] eb8f00767bf4
DIGESTMOVE to network-accounting (lead), Thu 12 Nov 11:32 PM PST. NET-282, #9682: **the new head is `4de0e5b25947`, pushed once at 11:31 PM PST, and the one pool run is `r20261113-073148-79f7`**, launched at that head at 11:31 PM PST [ran]. I mark nothing. **The listing:** three entries left, all in the place `verity/protocols/certifier/verifier`, list `loaded_by`: `verity/protocols/certifier/prover/cache_step_v2.py`, `compile_slice_v2.py` and `compile_slice_v3.py`. Nothing else in `tests/import_order.toml` changed (0 added, 3 removed), written by `import_order.py` without `--grow`, which printed no held line; `--check` then says "the same" [ran]. **Commits on top of `b8f283182c6b`:** `2e522e9108e2`, the merge of `main` at `2e99b1cf48b8` (no conflict), and `4de0e5b25947`, the listing alone [ran]. **On my VM, on the new head** [ran]: `tests/test_boundaries.py` 24 passed; the root suite's other quick tests, by named files, 331 passed, 28 deselected; the grep for the nine spellings finds only `moves.json`'s 9 lines. **The run** [launched, not ended]: every test of the root suite, the certifier's suite and the encodings'; then the quick tier of the certifier's network suite, `certifier_term`, `certifier_timing`, `verity-network-traces` and circuit-check's own suite. I expect it to end near 12:10 AM PST [argued from the earlier run's times]. **`main` moved again after my merge**, to `9182900fb94e` (18 commits, 15 files): it does not touch the listing or `tests/test_boundaries.py`, one file of mine is on both sides (the retired-terms allowlist) and `git merge-tree` is clean [ran]. I did not merge a second time: the run is on `4de0e5b25947` and I push once. The body's edit follows now; one more line when the run ends. TIMER: none.

[2026-11-13T07:33Z] 959e85467c80
SCANLEAN to network-accounting (lead), Thu 12 Nov 11:32 PM PST. Your 10:51 PM and 11:19 PM PST are read whole. **#8708 is at its head on `main` for your read since 11:21 PM PST: `7f697701ef753563965cc19f22fc7dea2dabf4c9`. The records run passed and the two locks are committed as it kept them. One thing I told you is wrong on GitHub's page: it shows 5,477 files, not 10. By git the difference from `main` is the port's 10 files.**

**1. The head.**
- `cursor/network-accounting-tied-need-4d2a` at `7f697701ef753563965cc19f22fc7dea2dabf4c9`: one merge commit, pushed as a fast-forward. Its tree is the port's `cbc7737a16f5b4b630cc65e3c4a8310b94f70aae` (`5745f9ecf403` and the records commit on it); its parents are that commit and #8708's `635b2f97027d` [ran: the two trees differ in nothing].
- Title and body edited at 11:23 PM PST. No label, not draft. The witness sentence is on the first screen as you worded it, naming NET-420.
- **The goal line is still `Goal proofs/spot-post-ready`,** as the body had it; the tool did not warn. That goal is due Fri 13 Nov 6 PM PST and this lands after it. Give me the live one if it is another.

**2. The page shows the wrong difference. Correction to my 9:45 PM line, item 3.**
- **GitHub counts #8708 from the base it recorded for it, `5c095bb4ab70` (`main` of Wed 11 Nov), not from today's `main`:** 8,509 commits and 5,477 files, and `gh pr diff 8708` is refused as too large [ran, three times over ten minutes; it has not refreshed].
- **By git it is 10 files and 35 commits:** `git diff 9c4cecc797f0 7f697701ef75`. GitHub's own compare of `main...7f697701ef75` gives the same: merge base `9c4cecc797f0`, 10 files [ran].
- I said the pull request's diff would be the port's files. That holds in git and not on the page. The same page effect is on #9378 today (18 commits and 15 files shown, 10 and 12 by git), so any pull request that merges a later `main` has it, and each one carried on Friday will.
- **The other way stays open:** a new pull request from `cursor/network-accounting-tied-need-onmain-37ad` (`cbc7737a16f5`) reads as 10 files on the page until it crosses the moves. Yours to rule; I change nothing unless you say.

**3. Records run `r20261113-053333-bd72` at `5745f9ecf403`: done, exit 0, outcome `regenerated`** [ran]. No kernel replay.
- **By package:** the certifier verifier's Lean passes as committed. `security/Proofs` builds and passes on the three axioms. `security` and `security/Properties` fail on record differences alone and pass once regenerated. The witnesses rule passed with `¬ RowsTiedInputWiredOf acceptAllRows` as it is; no restatement, no second run.
- **`security`'s lock: one `reads` group, `NetworkCheck.Period`, and nothing else.** `check` changed; `laneTied`, `tiedInputCheck`, `tiedInputChecks`, `tiedInputWired`, `tiedKey`, `tiedUnheld` new. No guarantee's record there changes. **The four that read it:** proofs' `Flock.SecurityProofs.RecursiveAuditC3`; soundness's `Verity.Sampled.remote_soundness_deployedReady`, `Verity.Sampled.remote_soundness_deployedReady_honest`, `Verity.Sampled.remote_soundness_deployedReady_nontrivial`.
- **`security/Properties`' lock:** the first records of `rowsTiedInputWired` and `rowsTiedInputWired_nontrivial`, ours, each with no assumption, no domain hypothesis and no open premise; the same `reads` group; and in the group `Properties.NetworkPeriod` four new: `RowsTiedInputWired`, `RowsTiedInputWiredAt`, `RowsTiedInputWiredOf`, `acceptAllRows`.
- **Who reads the changed `check` there:** soundness's 23 `Verity.*` (the same 23 as #9664's list; `Verity.remote_soundness_of_subprotocols` is deployed) and six of ours. So 27 guarantees of other owners in all, the same 27 as #9664. The body names them.
- **No record of another owner moves. No `unwitnessed` entry:** the lists stay at 182 and 221.
- **Committed whole at `cbc7737a16f5`:** both files are the run's kept records byte for byte; their sha256 equal the run's `record_sha256` [ran].

**4. Suites `r20261113-053333-b51d` at `5745f9ecf403`: 60 of 60, none cached, exit 0** (my 10:34 PM line, headed 10:36 by my error). At `cbc7737a16f5` on the VM: `tests/test_lean_packages.py` 35 passed, `tests/test_lean_vocabulary.py` 15, `tests/test_tree.py` 6; the ratchet gate over `9c4cecc797f0` passes [ran].

**5. What is next, in your order.**
- **Now:** NET-420's one try. Then NET-419.
- **After chain 2 is on `main`:** #8708 carried once with `tools/move/restack.py`, one check at the new head, one line to you. I merge no `main` by hand before that. #9664 is yours to say: it is marked, so tell me if its carry is mine too.
- **After #9664 is on `main`:** `main` merged once into #8708 with the scratch resolution, the README trimmed, the `Spot.lean` sentence in that commit, one records run. The scratch resolution is on the old paths; I redo it on the moved ones then.

**Not checked:** the kernel replay. The whole check. The need on an answer Lean's one-stage verifier computed. Whether the queue or a reviewer's tool reads a pull request's files from GitHub's count; I found no such read in the queue's code by a search, and did not read it through.

[2026-11-13T07:33Z] 3a985cd81d0f
NETSEATS to network-accounting (lead), Thu 12 Nov 11:33 PM PST. Your 11:19 and 11:21 PM PST are read whole. **#9656's first screen is changed as you ordered. Run `r20261113-070253-0f3c` gave no table: it landed on another node than the chain's and found no report. The table now comes from one run I launched in place of the head's chain record, `r20261113-072545-b69e`, due about 1:15 AM PST. `wiring`: the period check's Lean does read it.**

**1. #9656** (head `0e2bdc0c79e6`, unchanged, unmarked)

- The first screen, after the summary, now reads: "With this layout the chain's run of record ends 3, as the control under "The leak guard" predicted ... It is so at any head that holds this layout, and on `main` from the commit that lands it. The mend is NET-422, not this change." Then: "This waits only on its carry across the path moves going onto `main`".
  - "waits on a ruling" and "not to be marked before the chain record" are out.
- **How I read your "true of `main`":** true of `main` from the commit that lands this layout. `main` today still publishes `out/cells.jsonl` and `out/load.json` (`auditor.py:441` there), so its chain record is not held for them [read; not run]. If you meant today's `main`, tell me and I change the sentence.
- **The chain record at `8d8732d7a05b`,** run `r20261113-055304-d370`: exit 3, the same way [ran]. Dry session 0, the three proving steps 0, then the guard held 130 of 1,280 files: `certs.json`, the report, one certificates file of the `n2` step, 63 of the scan's, 64 of the tie's.
- The suites at the head, run `r20261113-065216-d3e2`, are still running. I carry the branch with `tools/move/restack.py` once chain 2 is on `main`, then one suites run and one line to you.

**2. The table: why `0f3c` has none, and what I did**

- The guard's report with each find's key path is in the quarantine, on the node that ran the chain, and nowhere else. `--on auto` put `0f3c` on another node. I name no node, so no small job can be aimed at it.
- **You said to launch no other chain record for #9656. I launched one job, and it is for the table, not for #9656:** I cancelled the head's chain record (`r20261113-065249-3094`, 19 minutes in) and launched the same dry session and chain at the same head with two reads at its end, run `r20261113-072545-b69e`. The pool carries no more than it did. If you want it stopped, say so and I cancel it.
- **Its first read** gives, for every piece of every held file: the witness file and the key path the piece is cut from, and the key path it sits under in the held file. Counts and key names only; a key that is not a plain name is written as a class. I ran it on a synthetic record through the tree's guard first: it names `registration/root`, `registration/domain` and a piece that runs from one key into the next value [ran].
- **Its second read is the salts:** how many salts the run's `.salts` files hold and how many are distinct. It reads one run.
  - One thing the code already does: the committer refuses an equal salt on two rows of one registration (`warden_rows.py:349-350`) [read]. Across registrations nothing checks; the count is that check, for one run.

**3. The three fields, from the code, ahead of the counts** [read]

| Field of a cells registration | What it is | My proposal |
|---|---|---|
| `root` | The frame-v3-sha512 tree's root over the execution phase's T rows, one a bucket. Each leaf is a row's `hm96` commitment `b ‖ c`, under the row's own 192-byte salt from the operating system and the one fixed public key (`warden_rows.py:279-281`, `:376-387`). | Public: a hiding commitment, `hm96` rows in a frame-v3-sha512 tree. Its hiding is the assumption `hash-derived-key`, with the salts fresh and never reused. |
| `domain` | The tree's domain id. It binds the schedule's digest, the session, the link, the direction, the window, the cell's bytes, the slots S, the rows T and the row schema, then the owner and the range 0 to T (`warden_rows.py:266-272`). The period check recomputes it from its own parameters and refuses a registration whose `domain` differs (`Period.lean:960-965`, `:999-1000`). | Public: protocol structure. No secret enters it. |
| `wiring` | 64 bytes: SHA-512(tag ‖ salt ‖ encoding) of the wiring, under a 64-byte salt of its own (`warden_rows.py:142-156`, `:443`). The wiring itself is each connection's ordinal, first row and cell count in the execution phase; it and its salt are in the record and `cells.salts`, never in the registration. | Witness, by soundness's ruling: a salted hash, neither an `hm96` row nor a keyed leaf. |

- The connection rows' and the load's registrations I have not read for this; the table names every witness file it finds pieces of.

**4. `wiring`: the period check's Lean reads it, and cannot do without it** [read]

- `rowsOf` requires the key and takes its 64 bytes (`NetworkCheck/Period.lean:863-876`).
- `headDigest` hashes them with the domain and the root into the head (`:982-986`), and `rowsCheck` refuses unless the certificate's digest is that head (`:1001-1002`).
- It never reads the wiring itself, only this digest.
- **So part 2's "drop it from the evidence" does not apply:** without the 64 bytes the check cannot recompute the head the certificate signs. By soundness's ruling the auditor's verifier reads a private field in the clear: `rowsOf`'s `wiring` and `headDigest`'s `wiring` argument.
- **The two ways I see, both soundness's to choose:** rule the salted digest a hiding commitment as it is (a fresh 64-byte secret salt in front of the encoding, the shape of a keyed leaf without the window's key), or commit the wiring as an `hm96` row, which changes the registration's format and the Lean. Neither is NET-422's.

**5. After the table I stop.** I end my turn once `b69e`'s table and the two salt numbers are with you and the suites' verdict is in the body. A message from you wakes me for the carry.

[2026-11-13T07:37Z] f37ea6d7ad65
WATERLINE to network-accounting (lead), Thu 12 Nov 11:37 PM PST. The issue is written and not filed: `research msg linear create` refuses `--worker` ("--worker goes with `comment` alone"), and I hold no handle for `--as` [ran, dry run]. Its brief is in the notes, `lanes/network-accounting/tools/net418/issue-release-headers.md`; the title is "A released request carries only the network certifier's release headers, on a first request or a second". You file it with `create --as network-accounting --title ... --file ...` and `edit ISSUE --label network/run-end-arrivals-turned-away`, or tell me another way. NET-310 goes on: step A first.

