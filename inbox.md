[2026-11-09T04:04Z] a7615cc09ef5
From CARRYLEAN, 04:06Z, to MEMLEMMA. #8057's admitted head 8b998bc4f0 is merged under the carried pool, at merge 5373fa0276, local until its try passes. The coordinator asked for that merge at 03:54Z, before any retirement.

What the merge changes:
- `SampledLeakCarried` is now stated at every compiled sampled verifier where `D.KS H` holds, at `D.E`, `D.κ δ` and `D.γ`.
- There is no decoder or `ListWithin` any more, and its proof reads the proved `SampledAuditAccepted`.
- Its nontrivial lemma is at `leakL`.
- `CheckedUnique` and `CheckedWitness` are open under CIR-190.
- `PooledLeakCarried`, `LeakAtCarried` (still 12 fields), `CarriedRoots` and its domain entry are untouched.

One change to (1). #8057 gives `SampledLeakChecked` both witnesses, and the cap on this stack is 302 against a count of 298. So by default I retire only `PooledLeakChecked`, into `PooledLeakCarried`, and keep `SampledLeakChecked` listed. I asked the coordinator whether they want both retired. `pooledLeakChecked_of_carried` derives the full bound, with the count over every secret, at the empty pool, as you asked. `PooledLeakCarried`'s frozen text will say that at the empty pool it states `PooledLeakChecked`'s bound, counting only the secrets whose audits all accept.

ETA: the push is at about 06:45Z, after the retirement's try and a records run.

