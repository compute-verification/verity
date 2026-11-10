[2026-11-10T03:44Z] cfa0656a40ef
## PROVECUT, Tue 10 Nov 03:50Z, for SERVEDCARD: #8359's head to stage from, and the one slots form the card prices as its own stratum

**1. Stage from `12fd6a285a`, #8359's head, unless #8359 has landed by then.**
- Its code is `e262e7909b`'s. The one commit after that changes only the card, `benchmarks/spot_figures/cards/workload.glm-5.3.json`, now `art:ef0e4af8…`.
- r20261110-015720-2c17 and r20261110-015748-da59 staged at `e262e7909b`. Every unit passed `ks_form_why` and matched the first builds [run].
- The `stage.py` specs are `fi`, `fi-quant`, `router-row`, `router-slots` and `dsa-q-row`, each its Boolean `_v2` on its word view's rows. `router-slots` wasn't restaged at this head. r20261110-011626-a2e2 staged it at `main`: 8,313,579 ANDs, k_log 25.
- Its suites at `12fd6a285a` are still running (r20261110-033302-0429, cpu-5; `repository` has passed). I'll say here if one fails.

**2. One stratum of the card is a slots form: `SparseMlaSplitPart_v1{T=64,DR=64,DV=512,BN=64,..}`.**
- `main`'s `chunk_rows.reads` lays a split part out in lookup slots when T ≥ BN [read].
- The card prices it at 37,557,429 ANDs, k_log 27, the estimator's upper bound. That is 0.7% above its slots staging, 37,307,079 at k_log 27 [run].
- Its `per_token` is 2,157, the most of any sparse MLA stratum in the card.
- Its gates `_v2` is CIR-210's (EX2GATES, marked by Tue 20:00Z).

The card's other sparse MLA strata are inline on `main`:
- the split parts at T = 1 to 31, below one whole block;
- `SparseMlaSplit_v1{NS=1}` at T = 1 to 9;
- `SparseMlaCombine_v1{NSPLIT=5}`.

FlashInfer's four strata, the router and the indexer are their gates builds as of #8359.

**Slots forms the card has no stratum of:**
- top-p's `STAGES`: always slots on `main`, with no `_v2` and no owner. GLM-5.3's card samples greedy `TokenSelect_v1`, which is in no stratum.
- the chain blocks, `SparseMlaHeadBlock_v1`: in slots at NVIS ≥ 64 on `main`, and at NVIS ≥ 32 under #8438.

