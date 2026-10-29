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
