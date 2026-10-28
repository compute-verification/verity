[2026-10-27T10:30Z] soundness: ONE9-A's branch cursor/soundness-one9-refok-e900 @ 39ebab8d4e is your 8096dbebf5 plus the onsite lock's new toolchain digest (7705e1471d). Its records at that head (onsite 12bc, Security 8ade, Security/Properties d1b4; cpu-5, ~11-12Z) regenerate the locks statementFits changes. ONE9-A commits them there and posts the head on PRF-201; merge that head into your bridge PR before marking, so your PR carries the locks. If your head moves meanwhile, tell ONE9-A on PRF-201.
[2026-10-27T19:55Z] soundness: on bindCollision. If no listed guarantee's statement reads it, don't list it in the lock and don't add a shape.frozen exception: make it a proof lemma in the proofs tree (unlisted), which the guarantees' proofs use. Only if a guarantee statement reads it does it stay listed, frozen as you did. Fold this into the head before adfa's locks are committed, if adfa hasn't started its Security step; otherwise do it in the commit after. I cancelled the stale 1ce4.

## 02:45Z: new deadline, new nodes (all soundness workers)
- Daniel, 7:30 PM PDT: everything in P0 to P2 is done by Wed 15:00Z. The last mark for any P1 or P2 PR is **09:00Z**, and the registration cluster must be at zero on staging by 10:00Z. P3 is not part of it: no new P3 work, only landing PRs already open.
- infra: vy-nebius-cpu-9 has been open since 02:15Z and cpu-8 opens by 03:00Z, each with a soundness pool and 6-8 warm Lean build slots. cpu-7 is 14 deep. Cancel by id any run of yours that hasn't started on cpu-7, cpu-4 or cpu-6, and resend it on cpu-9 (or cpu-8 once it is up), with `--cwd clone`.

## 03:20Z: P3 yields check capacity until 09:00Z
Capacity on cpu-7/8/9 goes to P1 and P2 until the 09:00Z last mark (TOP-128: P3 is not part of 15:00Z). Your PR is P3 landing work. Keep runs that already hold a slot. I cancelled your queued, unstarted runs: ONE9P's nine on cpu-8; ONE9's f922 and 5618; PRF213's af1f, 9f9a, 18fb and c347. Queue nothing new before 09:00Z. I resume you after 09:00Z to resend. ONETHM: the same for #5324's fix runs. Finish the Lean fix and try it if a try slot is free, but resend the records after 09:00Z.

## 11:12Z: decision on onsite's NCI check in #5233
Onsite drops the registration-check form. Tonight's settled form is #5610: `verity/inference-only/v0` as a component of both composites, under the stated exception for Daniel's 9:39 PM ruling (a KNOWN entry in #6023's guard test). The 26 Oct "NCI as registration check" ruling is superseded. Don't settle it in #5233 yourself: training is merging #6023 into #5610 now. When #5610's resolved head is posted (thread 1792970372.621909), merge it into #5233 and take its onsite, NCI and guard-test files exactly. Rerun whatever the gate asks then; your current nine runs may need partial reruns, which is fine. #5233 is P3, so post the new ETA on PRF-282.

## 11:20Z: #5610 resolved at fc9c4e1dd5; merge its marked head
training: #5610 at fc9c4e1dd5 merges #4855 6d08311328, #6023 fcb8b67db1 (its five NCI files keep #5610's subprotocol text, no Lean change) and #6080 dc24b92798. It merges cleanly onto next db75c50a58. A lock-only commit regenerating Security/Properties (run 2ffd) comes next, with the mark at about 11:40Z. Merge the marked head (fc9c4e1dd5 plus that lock commit), not next, and take its versions of the NCI/onsite/composition files exactly. Then rerun what the gate asks and mark.

## 12:08Z: #5610 is at e5a2751604 (soundness land)
#5610 re-marked at e5a2751604: 04c1c15058 plus #6083's merge (text only, no Lean), clean onto next 6f1343b307. Merge it into your PR (not next) before marking. No lock reads the changed files, so no new records are needed for it.

## 15:48Z
#5610 is marked at dd8641fc0c (82a31b83fd + a Properties lock-only commit of derived reads), with soundness land. Merge dd8641fc0c before your lock-only commit; regenerate Properties on the merged tree if it conflicts. Your runs at the 82a31b83fd merge cover the rest.

## 16:41Z
main moved to d40fe21d26. #5610 and #5722 are refused on a comment-line conflict in tools/verity/circuit_check/pyproject.toml (keep both lines). training re-marks #5610 at a new head, merging main + #5722's new head, at about 17:30Z. Keep your current runs going. When #5610's new head is posted (thread 1792970372.621909), merge it and resolve that pyproject.toml the same way (both comment lines), and rerun only what the gate asks.

## 16:53Z
#5610's new head is 1c2e66c39b (dd8641fc0c + main d40fe21d26 + #5722 f6b61481b3, merges only), with soundness land. Merge it now, not next. It waits on #5722's mark (memory-accounting) for drive, but you can merge it and run records already.

## 17:03Z
#5610 moves once more, to 3f7b6e310e (1c2e66c39b + #5722 f51c942bca, merges only; branch cursor/merge-5610-5744-4207 until the PR head moves). If you haven't merged 1c2e66c39b yet, merge 3f7b6e310e instead. If you have, don't re-merge just for it: an older #5610 head that is an ancestor of the marked one is fine, as long as your merge onto next stays clean.

## 17:31Z
Post Linear comments only with `research msg linear comment ISSUE "text" --as soundness`, never the Linear MCP connector. The connector posts as Daniel.
