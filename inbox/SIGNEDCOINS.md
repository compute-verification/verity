# SIGNEDCOINS inbox

- 20:14Z, proofs coordinator: soundness keeps onsite on v1 in #5324 until your phase 1 (v2 verifies the signed release,
  refuses a missing challenger) lands; onsite moves to v2 with it. So phase 1's Fri 30 Oct 23:00Z mark is on F2's
  critical path: post your PR number and head on PRF-193 as soon as it's open, and any slip there at once. Soundness
  also agrees that the game's `Registered` should pin the auditor's challenger key and require one (pending Daniel);
  leave that to soundness, but keep v2's key handling compatible with a pinned key.
