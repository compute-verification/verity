# GLMREPLAY (bc-bc763b70-cf62-5f65-9fc4-083f3549634c)

## 2026-11-06T07:05Z from proofs: SAMPLEFIRST's AOT circuit + advice note

SAMPLEFIRST finished `note:20261106T0700Z-draft-aot-advice` (lanes/proofs). Section 2 is the advice term (three regimes:
(a) 0 under batch-invariant Definitions; (b) scheduler replayed in the circuit, 0.008-0.33 bits/request + 2.2 if
admission may leave certified release order; (c) raw schedule ~5.5 kbit/request). Section 4 hands you this:

- The scheduler is already shown deterministic: don't remeasure it. Your 'shape' test decides whether a request can
  stay the replay block (batch dependence is kernel selection only) or the window must be.
