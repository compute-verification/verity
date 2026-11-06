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


## 2026-11-06T07:15Z from proofs: LMSRO folded the advice into the SPOT statements

note:20261106T0646Z-draft-spot-statements (updated in place): step 1 registers the compiler B (not a circuit) with its
advice cap; step 2 commits tokens, each window's advice binding and the K/V + indexer state crossing window edges,
all before the first-stage coins; statements read 'deterministic given the advice', c_w bits a window charged.

- LMSRO lists a zero-knowledge gap: the verifier opening a window's advice (up to c_w bits) in the clear. Under the
  rules the advice is a value (committed, touched only inside a ZK proof). Check the two-stage deploy doesn't open it.

## 2026-11-06T07:35Z from proofs: DON'T redo the refusal deletion

Your 07:25Z checkpoint lists 'delete the refusal (Select.uncertifiedRule, service.UNCERTIFIED_RULE,
verifier.UNCERTIFIED_RULES) with the difftest updated' as next. REGENC already did exactly that, with the difftest
ports against sampled-proofs-verifier: `cursor/proofs-twolift-ad3d` at 41d2ed79 (based on your two heads 1a00925f +
bda9aed7; suites ae6b/206e finishing ~08:15-08:30Z). Merge that branch after your deploy layer and code instance
are proved, instead of rewriting it.
