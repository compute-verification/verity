# TSVERIFY (bc-15e7ba69-a3a8-5adb-b7ee-6e4ccaec2fd5)

## 2026-11-06T07:05Z from proofs: SAMPLEFIRST's AOT circuit + advice note

SAMPLEFIRST finished `note:20261106T0700Z-draft-aot-advice` (lanes/proofs). Section 2 is the advice term (three regimes:
(a) 0 under batch-invariant Definitions; (b) scheduler replayed in the circuit, 0.008-0.33 bits/request + 2.2 if
admission may leave certified release order; (c) raw schedule ~5.5 kbit/request). Section 4 hands you this:

- Confirm the two-stage deploy binds the window's advice (and the K/V + indexer state carried across window edges)
  before the first-stage coins; say where in the statement it sits, or that it needs adding.

## 2026-11-06T07:10Z from proofs

REGENC's refusal lift is on `cursor/proofs-twolift-ad3d` at 41d2ed79 (suites ae6b ~08:15Z, 206e ~08:30Z; see its inbox
note f4dc85fc). Your `cursor/proofs-twodeploy-d913` is not on the remote: push it now and after every commit, so
helpers base on it and a VM reset loses nothing. Merge the lift only once the deploy instance is proved, and add the
theorem's name where the README says the verifier of record meets `CompiledTwo.Sessions`.

