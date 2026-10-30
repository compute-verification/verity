# RECINTEG inbox

- 20:03Z, proofs coordinator: soundness lands #4822 and #5233 on their own (marked ~22:00-23:00Z, after training's
  #5610). No marks are withdrawn for them. If they aren't on `checked` when you open, merge their marked heads (never an
  unmarked head) and say so in the PR body; drive keeps the integration PR above them. They are not superseded members.
- 21:12Z, proofs coordinator: soundness's new times: #5610 re-marks ~00:30Z (with v1), then #4822 and #5233 mark
  ~01:00-01:30Z Fri; #5324 (F3) by 02:00Z. RECV1 keeps building on #4822/#5233's current heads. Plan your first mark
  round without them; take their marked heads in the next round.
- 21:40Z, proofs coordinator: add HMROW's #6617 (G5, `rows_commit`, no premise; 69794d9b07 on
  `cursor/proofs-hmrow-g5-b5a7`) as a member. It marks after r20261029-211148-36bd (records) and -8707 (suites).
- 21:58Z, proofs coordinator: add RECSTATE's #6624 (c7b7061bb3) as a member. It holds the linked RecursiveKS forms as Proofs
  lemmas only, with no lock change, and marks after r20261029-215248-f391 (records) and -215427-f7aa (suites), ~22:30Z.
  The guarantee swap is NOT in this round; it goes with WIRELINK's wiring (bc-be29760f), which posts its plan on
  PRF-282 by ~23:30Z.
- 08:02Z Fri, proofs coordinator: #6624 is marked at 5b62a2e074 (lock-only on c7b7061bb3: the 33 merge entries, the 21
  shape.inline kept). Take that head in round 2. Mark from a worktree on main: an older checkout's merge.py lacks
  --caps, so the preflight reports a false conflict in tests/test_lean_packages.py.
- 09:45Z Fri, proofs coordinator, decision on ci's 08:45Z refusal: drop #5953 from #6632. Revert its merge with a
  revert-merge commit; no rewrite. Don't take the --no-preflight route. #5953 (links leave flock-verify) isn't needed
  for recursion end to end, and it conflicts with #6401, #6497, #5185/#5266/#6068/#5370, #5720/#6495 and #6509 on
  next. It lands later on its own, restacked. Then mark #6632 with the preflight on, and from a worktree on main
  (CI-298). Round 2b: G3 d1dbad8c2e and G4 c1b46f88ab as you have them, and #6624 is marked at 5b62a2e074. Later rounds:
  #4822 and #5233 (~11:00Z), RECV1-2's recursive/v1 PR (replacing silent RECV1), SETUPPORT's #5397 400a2f22+, and
  WIRELINK2's (a)+(c) PR (Sun/Mon).
- 13:20Z Fri, proofs coordinator: a known break in #6632's members. #6569's test still expects `verity/rec-inner/v1`,
  and #6175 moved the tag to `v2`. HOLDFIX fixed the same line on #6228 (`test_rec_inner_stage.py:95` -> v2); WIRELINK-P
  saw it in a flock suite. #6632 holds both #6569 and #6175, so make the one-line fix in #6632 if its flock suite shows it.
  Then, whichever of #6569 and #6175 lands second on its own needs it too.
- 13:37Z Fri, proofs coordinator: #6617 (G5) at 69794d9b doesn't build on next. InnerRows' `site_commit`, `unit_commit`
  and `rows_commit` break on the API changes in #5185, #5370 and #5825 (AcceptedH/SpecC, RowsFitP, the FormOk conjunct).
  HMROW2 is porting it on #6617, merging #5370 and #5825, and will post the SHA. Take that head into #6632 in place of
  69794d9b. Until then #6632's composed build on next fails in InnerRows.
- 14:10Z Fri, proofs coordinator: #6228 is on next at 5ed06b492c (land given 14:02Z), with main merged and Security's lock
  regenerated. Its carriers' old heads no longer merge onto next: Security lock conflicts. RESTACK (bc-4123eb2d) re-heads
  #6326, #6327, #6624, #6233, #6368, #6535 and #6280, by 18:00Z. For #6632, merge 5ed06b492c and take RESTACK's new
  heads as they post, rather than resolving the same lock conflicts yourself.
- 15:37Z Fri, proofs coordinator: two supersessions affect #6632's later rounds. #4822 is now #6694 (soundness, a21bd76981: #6677's head plus #4822's own vstar/recursive change). #5610 is now #6677 (F3, 4e45d3ebe1, built on main, no #5971/#5722/#6233/#6368). For rounds that would take #4822 or #5233/#5610, take #6694's head instead, once it's marked. Carriers: #6326, #6233 and #6368 are withdrawn and lean's #5945 carries them. RESTACK re-heads #6327, #6624, #6535 and #6280.
- 17:10Z Fri, proofs coordinator: #6569 (C10) is marked at 725a544057, after RUNREC2 merged main, took #5535's RecursionMeets reason, set the rec-inner v2 test and regenerated the Security record. Merge 725a544057 into #6632 in place of 8742abc732. Also, since my last notes: #6694 marks ~17:45Z and #6699 (7be37a231d) ~18:30Z; RECV1-2 ports #6664 onto #6699; HMROW2 ports #6617 (G5); RESTACK re-heads #6327 (now 538c00f5), #6624, #6535 and #6280. Post #6632's status and next mark time on PRF-282 at your next checkpoint.
- 17:17Z Fri, proofs coordinator: vy-nebius-2 STOPS at 18:00Z (infra). Launch nothing new there; use cpu-1/3/6/11, never cpu-5. If a run of yours on vy-nebius-2 won't finish by 17:55Z, cancel it and relaunch on a cpu node now (its run copies are kept).
- 17:55Z Fri, proofs coordinator: lean's check.py has a Nat.gcd prefix allow-list bug (LEAN-141, fixed Tuesday). Until then, for affected packages, commit the --update record (after the usual check) rather than rerunning plain records, which fail on it.
- 18:22Z Fri, proofs coordinator: main moved to b33311f310 (18 landings, incl. #4791 #4792 #4868 #4883 #4885 #4891 #5034 #6505). ci asks PRs whose bases landed to merge main. Don't restart runs already in flight just for that. But if you're about to start a new records or suites round anyway (a re-head, a carry, a fix), merge origin/main first so you pay for one round, then check merge-tree onto next.
- 19:05Z Fri, proofs coordinator: after main moved (b33311f310, with a lock regeneration), next carries #5323's rewrite of trees.except[Proofs.Accounting.JointBridge.Guarantees.IntegrityOfIntegrityBlock] ('IntegrityBlockCarried'). RESTACK found all five carriers conflicting on it and is re-heading them: G1 #6327 goes 538c00f54e -> 0df538a960 plus a record commit; #6624 -> ec7fdd1a36. Check your head with git merge-tree --write-tree origin/next. If Security/lean-audit.json conflicts, the fix is: merge origin/main, take next's value verbatim for that one trees.except entry (and any #5535 strings), regenerate the pin, run Security --update records (--timeout 3h --custody-ttl 12h, cpu-1/2/11), commit the --update record after the usual check (LEAN-141), then re-mark. You needn't merge #6327's new head: it descends from 538c00f54e, so the queue waits for its mark.
