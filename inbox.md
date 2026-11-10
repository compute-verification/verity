[2026-11-10T19:41Z] 86a085bf0e1e
CROSS, NET-311, part 3, Tue 10 Nov 11:41 AM PST. **Four of the five can be marked; #8477 is red at its own head.** `origin/main` is 5482988e34 (11:39 AM PST). Every head below merges it with exit 0 and no file [run: `git merge-tree --write-tree --name-only --no-messages origin/main HEAD`].

1. **#8405: markable as it stands** at ca7d59c951. NET-281 holds `r20261110-032722-e64f` at that head: `58 suites: 58 passed (0 cached), 0 failed` (`--changed 7af4d24fca`, slow tests included) [read: the run's stdout and job.json].
2. **#8499: markable as it stands** at 858eb03d0a, which holds #8405's head [run]. `r20261110-032722-2bd4` at that head: `58 suites: 58 passed (0 cached), 0 failed` [read, the same way].
   - Both verdicts are on the head's own tree, 1,074 commits of `main` behind. `main` and each PR both changed `Security/Definitions.lean`, `Security/lean-audit.json`, `Security/Proofs/NetworkCertifier.lean`, the certifier's PROTOCOL.md and `test_warden_serve.py` (#8499 also `Security/Properties/lean-audit.json`). Git merges them with no conflict; no run is on the merged tree [run for the list; argued: your rule makes them markable, and the mark's tier is the first run of the merge].
3. **#8557: `cursor/net311-main-8557-a9af` at 8c0e58114cb047f0046716e1b43f8cf24d49cea7**: eff460bb25, then `main` e7b03e4a6c, no conflict. Run `r20261110-180904-73c9`: `55 suites: 55 passed (0 cached), 0 failed; 129.2 min of test time`. NET-283 held no suites verdict at eff460bb25: `r20261110-055328-07df` was cancelled with 44 of 54 judged [read].
4. **#8477: red, not markable.** `cursor/net311-main-8477-a9af` at 2986a0cdb3d896f2406cc338dc542f02d538b673: 96de5d04f2, then `main`, no conflict. Run `r20261110-181025-1596`: `3 suites: 2 passed (0 cached), 1 failed`. The failure is `tools/move`, `tests/test_maps.py::test_two_word_maps_that_rename_one_word_reach_one_name_in_either_order`: `warden_tokens_map.toml` gives seven words a second name.
   - Its row `"WARDEN" = "NETWORK_CERTIFIER"` (line 98) meets `warden_map.toml`'s `"warden"` and `"Warden"` rows (50, 63). Its prose rows (35, 36, 43) meet `warden_term_map.toml`'s `"the relay"` rows (81 to 83).
   - **It is the PR's own, not the merge's.** The test's function finds the same seven at 96de5d04f2 without `main`, and none at d0f81d7947, where `r20261110-003003-9e88` passed before the third map was added [run: `two_names` called on the three trees]. NET-100 holds no suites run at 96de5d04f2 [read].
   - I changed no map: which name wins is the map owner's.
5. **#7109: `cursor/net311-main-7109-a9af` at 33a93601e32db3e35625887becd24fa42f17f901**: 53f2a6bfd0, then my mend e6d157c1a9, then `main` e7b03e4a6c, the second merge with no conflict. Run `r20261110-181109-c595`: `59 suites: 59 passed (0 cached), 0 failed; 138.7 min of test time`.
   - The entry for ci and lean, `tools/verity/check/public_boundary.toml` lines 293 to 296 at that head, after main's two entries from `certifier`:

         [[needed]]
         why = "the ZK server's record as its Lean code runs it, stated and proved, which the recursive session's stop count is proved from"
         paths = ["Security/{Properties,Proofs/Properties}/Firewall.lean"]
         from = ["flock"]

   - What it does [run: `public_boundary.py --public` at the head and on `main` e7b03e4a6c]: 8,115 public files against 8,107. The entry makes two public, `Security/Properties/Firewall.lean` and `Security/Proofs/Properties/Firewall.lean`. The other six are files the PR adds (four StopCount Lean files, two tests). `public_boundary.py` prints `0 edges, 0 new`.

- **The four runs:** `research run --on auto`, `suites.py --changed e7b03e4a6c89 --quick`, one a head, source the head named [read: each job.json]. Each new head's tree equals `git merge-tree --write-tree` of its two parents, and each descends from its PR's head [run].
- **Not covered:** `main` is 59 commits past e7b03e4a6c, and no head was merged or run again. Quick tier only. No Lean and no circuit-check ran.
- **Least sure of: the Security locks at #8557's and #7109's new heads.** `Security/lean-audit.json` and `Security/Properties/lean-audit.json` there are the lock merge driver's output from both sides. No records run is at either merged head, so a record `main` changed since 7af4d24fca may be stale in them.
- **The queue, 11:39 AM PST** [run]: #8040 is at 802daf1f77, `waiting`, after [7978]. #8040's branch is at 802daf1f77 on the remote. #8415 now reads after [7978], no longer [8040]: its head holds 069df3e007 and not 802daf1f77 [run: `git merge-base --is-ancestor`].
- I pushed to no PR's branch and marked nothing. Note: `note:network-accounting/20261110T1940Z-report-net311-cross-main`, synced after this message.

[2026-11-10T19:43Z] 2753cbe6a7c0
LENGTHSUNIT, NET-300, Tue 10 Nov 11:46 AM PST. Step 2's PR is open, not marked.

1. **#8702** (https://github.com/compute-verification/verity/pull/8702): `cursor/net300-lengths-unit-7c3e` at 38817494b1 onto `main`, open, not marked, not a draft, no label, MERGEABLE, linked to NET-300 [run: `gh pr view 8702`]. It holds step 1's commits (cut from e42861fca5), so it lands after #8690; through #8690 it holds #8405's seven commits too.

2. **Ruling 1's commit passes on a node, at 38817494b1.**
   - r20261110-184612-c325, `suites.py --changed 544cd87567 --quick`: `56 suites: 56 passed (0 cached), 0 failed; 104.3 min of test time` [run]. Certifier 1201 passed, 1 skipped; circuit-check's suite 329 passed, 2 skipped (its read guard takes the two inputs I declared).
   - r20261110-184632-a2a2: `circuit-check: 5 target(s), 0 failure(s), 0 unchecked`, every pin `ok` at the committed counts, so the gates did not move: 349 ANDs at K = 1, 5,614 at K = 16, 351 a slot against 600 by hand [run]. Then the certifier's suite without `--quick`: `1221 passed, 1 skipped`, `1 suites: 1 passed (0 cached), 0 failed` [run].
   - Not run at this head: the slow tier of the other 55 touched suites. I read your "`suites.py --changed`" as the quick tier; say if you want the rest.

3. **Step 1's full tier, r20261110-171751-7d26: still running.** 49 of 59 suites passed, none failed, at 11:41 AM PST [run: `research fetch`, the run's `stdout.log`]. Its timeout is 1:38 PM PST. I report its verdict when it ends.

4. Both branches merge clean with `origin/main` 5482988e34 [run: `git merge-tree --write-tree`, rc 0]. I merged nothing.

5. #8690's body says the full tier's verdict "goes on NET-300 and here". I will put it on NET-300; I edit #8690's body (`research queue edit`) only on your word.

[2026-11-10T19:43Z] ea609eeac874
1. LAIDCOUNT, NET-297 part F, Tue 10 Nov 11:43 AM PST. A read of `main` 5482988e34, no run. Marks [read], [argued]. File: `verity/core/protocols/certifier/warden.py` ("w."). Two of job C's three [argued] are now [read]; each has a finding. Note section 13.
2. F1 [read]: yes. `_tick(t)` (w. 458-499) holds no `await`. It cuts bucket t's ingress from `c.ingress` (w. 467), which only `_client` fills (w. 550, 560), a task of the same loop, so bucket t's ingress closes when the tick starts (w. 438). Bucket t's egress leaves after: written in the tick (w. 475), or handed to the pacer (w. 479), which never writes before `origin + t·τ + paced` (`pacer.py` 275). Rows go to the publisher at w. 488-491; records close at the window's last tick (w. 492-499).
3. F2 [read]: the due time is the host's clock alone, `origin + t·τ` on `time.monotonic` (w. 349, 431-432). Nothing carried moves it, a bucket's index or its length.
4. F2 FINDING [read]: the instant a tick runs is the event loop's, and the same loop reads every client and server (w. 543-599), so what a link carries can delay a tick by up to `slack` unseen; past it the window faults (w. 433-437). The module says so (w. 70-72). On one link ticks still run in index order (w. 430).
5. Across links: two buckets due less than `slack` apart can close in either order, and content can decide. UNPACED (the default, w. 355) a bucket's egress leaves at its tick, so which of the two can read the other turns on the record: job C does not cover it. PACED, a bucket leaves at due + `paced` > due + `slack` (w. 357) and an ingress closes by due + `slack` or its window faults, so a bucket holds only rows whose due time is strictly earlier: τ is the order of due times, one for every record [argued from the read]. It asks every certifier paced and each `paced` ≥ every certifier's `slack`; nothing on `main` holds that across devices.
6. What would cover unpaced [argued, not written]: the count pays Dj a bucket for a tick's offset, and two records with one code have one jitter at every bucket, so τ read from due times and jitters is one order for both; `fleet_record_inj` would take τ from the code. Under J (Dj = 1) the fleet must be paced.
7. F3 [read]: the device fails the connection closed and releases nothing. At bucket `s + ingress_buckets − 1` with no `END` released it sets `ingress` (w. 622-623) and sends one `FAIL` cell in that bucket (w. 648-649, 674-676); the lane is freed (w. 481-484). The server gets a request only at its `END`'s release (w. 625-633; `PROTOCOL.md` 2104-2105), and no release index is spent (w. 626).
8. F3 FINDING [read]: the bound is `ingress_buckets`, not `adm c + D`; nothing ties them (w. 156-166) and the defaults are 50 and 10 (w. 146, 151). With `ingress_buckets > D` a request can be released after its connection's first egress bucket, if the server writes cells before it has the request. The scratch's `rel c ≤ adm c + D` and the count's induction exclude that. The design's rule needs `ingress_buckets ≤ D` held in `WardenParams`.
9. The scratch's laying, to be the device's: `rel` an option; `before`, `rank`, `ord`, `span`, `standsAt`, `Fits` over released connections only; one conjunct on the counted records (an unreleased connection's lane holds no data). 80 to 150 lines [my guess], not a few: not done. The count does not move [argued]: such a connection stands at no span and its lane is fixed by its fail cell, which the code holds.
10. Also [read]: the Lean under `verity/core/devices/certifier/lean/` is `active.py`'s proxy (`CertifierDevice.lean` 17), not `warden.py`: lanes, admission, D and the release order are Python only on `main`. Its driver also takes pending reads before a due tick (`Driver.lean` 70-94). Next: E1 by 3 PM PST.

