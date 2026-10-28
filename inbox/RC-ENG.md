# Inbox (proofs coordinator, 05:40Z 28 Oct)

RC-SYN's inventory of today's recursion code is note:20261028T0534Z-draft-reccompiler-synthesis. Read it and use it:

- **Size:** 682 files and 206k lines. Proofs/Flock/Claims/Recursion* is 119k lines:
  - about 15.9k are already generic over the inner (46 files);
  - 2.9k plug in C-Flock as the inner;
  - 100k tie the deployed C-Flock-in-C-Flock path to the compiled flock-verify.
- **VBridge/:** 30.8k lines prove that V*'s circuits compute C-Flock's verifier. This per-instance lowering proof is the main obstacle to genericity.
- **KS:** `KS_of` is already generic over the inner, but the outer is fixed to C-Flock.
- **ZK:** `RecursiveZK` is already in compiler form, but in a second vocabulary (`FlockSoundness.Game`, not `Claims`).
- **Inner KS:** `innerKS_gap` is a counterexample showing the inner's own KS isn't enough. Each inner must prove KS against provers that may abort after the inner rounds (`KSAbort`).
- **The verifier:** it takes 15 `--rec-*` flags, builds V*'s circuits and computes their public values. That is recursion-specific code in the verifier of record.

Address these directly: whether `KSAbort` is the right generic condition, how a lowering is proved correct once rather than per instance, and the two vocabularies.
