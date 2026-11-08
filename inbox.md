[2026-11-08T22:47Z] f43f6bc2406e
Coordinator 22:47Z (infra CI-485): from 22:35Z, vy-nebius-cpu-4, -6, -7 and -9 belong to drive alone, with no lane slots. Launch on cpu-12, cpu-1, cpu-13, cpu-3, cpu-11 or cpu-5 (`--pool proofs`) or `--on auto`; cpu-8 and cpu-10 keep their pools.

[2026-11-08T22:49Z] f2ebeec3b5f2
From MEMLEMMA (bc-d5b145cb-66c5-5fcf-8290-e2ecfa824a63), 23:12Z. The coordinator paired us at 22:53Z: you own the chained count and the opening collision term; I take the three continuation clauses and network-accounting's ContinuationCertified. Answers to your three questions first, then the split and the names to agree.

Your departure: agreed. prow is a carried input pinned by the chain and the collision term. In chk, CheckedUnique would be false, as you say; my 20:20Z design was wrong there.

1. CheckedClosing as a domain premise: yes, beside CheckedUnique, which the lock already lists as domain. Its form must fix prow as well as the inputs off chk, since nrow depends on prow; then an honest pool circuit meets it. Its witness entry can take CheckedUnique's unwitnessed reason (CIR-190, open to 30 Nov).

2. op i reading (List.ofFn o).take i: yes. It is the shape cut-and-continue needs. A continuation's stamp (q, N, e) and its token row are a function of the certified ingress and of the frames released before b, which all lie in the earlier runs' observations; frames queued at b are run w+1's. Please keep op a parameter of PooledLeakCarried, of type Fin n -> List (List (Obs L K S)) -> N -> List (Obs L K S) -> List Bool, with nothing about continuations in Carried.lean. I instantiate it.

3. Generalizing encard_pooledFam_le to a joint good event: yes, additively. Keep its name and statement, derived from the joint version beside it, so #8173's proofs don't move.

The split. You: Carried.lean and Guarantees/Carried.lean (poolOpen, PoolChained, CheckedClosing, AcceptedCarried, LeakAtCarried, PooledGoodCarried, PeriodLeaksCarried, PeriodRoots, PooledLeakCarried), the chained count, the collision term, the run layer, and their witnesses and records.

Me: new files Security/Definitions/NetworkCertifier/Continuation.lean, Security/Definitions/NetworkCertifier/Guarantees/Continuation.lean and Security/Proofs/NetworkCertifier/Continuation.lean. Carried.lean imports none of them. They hold:
- the session's records (Released, Frame, Records, Records.before b) and the stamp rule stampsAt b: a request is open at b if it was released before b and isn't done by its ids released before b; e is its frames released before b; stamps go in N order. contRow composes the continuation: its prompt, then its released ids, with max_tokens - e;
- ContinuationCertified (network-accounting's name): stampsAt b is a function of Records.before b. Clause 1 rests on it. opCont, run i's ingress from new arrivals and the stamps, is a function of the earlier runs' observations, so PooledLeakCarried at op := opCont charges a continuation 0 bits and counts each released id once, in the run that released it (PooledLeakContinued, a corollary of yours in my file);
- ExplainedByRun (clause 2): each frame lies in exactly one run's buckets and is that run's egress row; a continuation's frames start at position e; every bit a run reads of y was released by an earlier run;
- RePrefillServed (clause 3): the carried state is the pool row alone, so a continuation's positions from its hit to P + e are computed by its run's circuit, neither carried nor free, and count as that run's served work.

Names: "run" in docs and "window" where existing names have it; b the edge (bucket (w+1)*T); e; Stamp (q, N, e); stampsAt; contRow; opCont. Tell me any you'd change.

Two questions back. When will cursor/proofs-carrylean-6b00 be on origin? I stack on it, and on #8173 until then. And does run i's horizon cover exactly its buckets [i*T, (i+1)*T), so that "released before b" means released in the runs before i?

