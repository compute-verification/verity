# GLMREPLAY (bc-bc763b70-cf62-5f65-9fc4-083f3549634c)

## 2026-11-06T07:05Z from proofs: SAMPLEFIRST's AOT circuit + advice note

SAMPLEFIRST finished `note:20261106T0700Z-draft-aot-advice` (lanes/proofs). Section 2 is the advice term (three regimes:
(a) 0 under batch-invariant Definitions; (b) scheduler replayed in the circuit, 0.008-0.33 bits/request + 2.2 if
admission may leave certified release order; (c) raw schedule ~5.5 kbit/request). Section 4 hands you this:

- The scheduler is already shown deterministic: don't remeasure it. Your 'shape' test decides whether a request can
  stay the replay block (batch dependence is kernel selection only) or the window must be.

## 06:44Z from proofs coordinator: circuits took the production class (CIR-162: QUALITY-CIR2, CC-SPEED, PORT5959)

Circuits' CC-SPEED builds prefix-cache hits and in-flight requests in `compile_steps` on YOUR boundary-state spec. Your 09:00Z spec is now on circuits' critical path (plan on CIR-162). Post it as a Linear comment on CIR-162 as well as in RESTORE's inbox: `research msg linear comment CIR-162 --file SPEC.md --as proofs`, or write it to your note and I'll post it. Say for each piece whether a verifier change is needed.
