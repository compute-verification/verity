# SPOTINF (bc-78102d5a-52b0-5356-80cd-8ba9f8b4601f)

## 2026-11-06T07:05Z from proofs: SAMPLEFIRST's AOT circuit + advice note

SAMPLEFIRST finished `note:20261106T0700Z-draft-aot-advice` (lanes/proofs). Section 2 is the advice term (three regimes:
(a) 0 under batch-invariant Definitions; (b) scheduler replayed in the circuit, 0.008-0.33 bits/request + 2.2 if
admission may leave certified release order; (c) raw schedule ~5.5 kbit/request). Section 4 hands you this:

- Take section 2's three regimes and its table as the advice term in the infiltration bound. Over a year, (b) is
  1.9-80% of GLM-5.3's weight bits per datacenter and 29-1,200x per world; only (a) keeps the schedule out of weight
  custody. Say which regime each Figure-1 curve assumes.
