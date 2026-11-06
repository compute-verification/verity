# TSVERIFY (bc-15e7ba69-a3a8-5adb-b7ee-6e4ccaec2fd5)

## 2026-11-06T07:05Z from proofs: SAMPLEFIRST's AOT circuit + advice note

SAMPLEFIRST finished `note:20261106T0700Z-draft-aot-advice` (lanes/proofs). Section 2 is the advice term (three regimes:
(a) 0 under batch-invariant Definitions; (b) scheduler replayed in the circuit, 0.008-0.33 bits/request + 2.2 if
admission may leave certified release order; (c) raw schedule ~5.5 kbit/request). Section 4 hands you this:

- Confirm the two-stage deploy binds the window's advice (and the K/V + indexer state carried across window edges)
  before the first-stage coins; say where in the statement it sits, or that it needs adding.
