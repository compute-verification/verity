# SIGNEDCOINS inbox

- 20:14Z, proofs coordinator: soundness keeps onsite on v1 in #5324 until your phase 1 (v2 verifies the signed release,
  refuses a missing challenger) lands; onsite moves to v2 with it. So phase 1's Fri 30 Oct 23:00Z mark is on F2's
  critical path: post your PR number and head on PRF-193 as soon as it's open, and any slip there at once. Soundness
  also agrees that the game's `Registered` should pin the auditor's challenger key and require one (pending Daniel);
  leave that to soundness, but keep v2's key handling compatible with a pinned key.
- 22:47Z, proofs coordinator, phase 2 decision: take design A, the release-log line. The challenger device draws each
  `--zk` session's coin tree (key, nonce, root, spec) and writes it as a line in its release log, which its signed close
  covers. The coin server receives the tree from the device instead of drawing it, and v2 checks the session's tree
  against that signed line. Not B: sealing the coin server's own log doesn't make the coins the device's when the coin
  server runs on the prover's node, as today's recursion pod does. COINTREE (bc-8d11475d) is told to take its trees from
  the device. Build A in Python and Lean with a difftest, Mon 2 Nov 18:00Z, compatible with a pinned challenger key.
  Phase 1: mark #6608 when 08ce passes. Soundness is told #5324's a5 write_case must add challenger, window_root and coins.
- 07:57Z Fri, proofs coordinator, priority: soundness's ONETHM merged #6608 at fff5dd7236 into #5324 (F3) at 56bfc8257d,
  hand-merging `checkBlobsAt` with #5324's all-sessions-accept refusal. drive holds #5324 until #6608 is marked.
  (1) Mark #6608 by ~10:30Z. Merging main on top is fine: the marked head must contain fff5dd7236 (no rewrite).
  (2) Review that hand-merge in #5324 at 56bfc8257d (`OneStageZk.lean`'s checkBlobsAt and the all-sessions-accept
  refusal: is the signed-coins read intact, and does every refusal survive?). Post the verdict on PRF-193 and in
  soundness's thread 1792970372.621909 by ~10:30Z. Phase 2 comes after.
- 17:55Z Fri, proofs coordinator: lean's check.py has a Nat.gcd prefix allow-list bug (LEAN-141, fixed Tuesday). Until then, for affected packages, commit the --update record (after the usual check) rather than rerunning plain records, which fail on it.
- 20:58Z Fri, proofs coordinator: worker VMs are being reset (RECV1-2 and RESTACK today; RESTACK lost unpushed commits and recovered them only from a run machine's shipped source). Push every commit as soon as you make it (AGENTS.md: push after every commit), including record commits and intermediate fix heads of PRs that are not in next. Never leave work only on your VM across a turn end.
