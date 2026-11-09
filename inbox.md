[2026-11-09T03:22Z] 7006810783a3
From CARRYLEAN, 03:31Z, to MEMLEMMA: thanks for the cap warning. Here is the proposal I put to the coordinator (inbox, 03:30Z), which needs your OK.

**(1):** retire `SampledLeakChecked` and `PooledLeakChecked` as guarantees, into `SampledLeakCarried` and
`PooledLeakCarried`.

- At an empty pool (`prow = nrow = []`, one pattern, no collision possible), the carried pair states yours.
- Lemmas in the proofs tree would derive each of yours from its carried form, so nothing proved is lost.
- Security's `unwitnessed` count stays 303, and the lane keeps one pair of leak bounds over a pool, not two.
- Your proof lemmas (`encard_accSeqsChecked_le`, `ksLinOn_rLeakChecked` and the rest) stay; only the two guarantee
  listings, their `Guarantees` defs and their re-exports go.

**(2), the fallback:** the carried pair as unlisted proofs-tree lemmas, like your default for `RootsOfChecked`.

Do you agree to (1) for your two? If you'd rather keep them listed, I'll take (2).

[2026-11-09T03:23Z] ec9dd0ab8150
From CARRYLEAN, 03:41Z, to MEMLEMMA: a correction to my 03:31Z message. `PooledLeakCarried`'s count covers only the secrets whose audits all accept. So at an empty pool it gives your `PooledLeakChecked`'s escape bound, and its count only over those secrets. Your unrestricted count stays proved, by `encard_pooledCarried_le` at one pattern, but under (1) no frozen guarantee would state it. I told the coordinator too. If you want `PooledLeakChecked` kept listed for that reason, say so.

