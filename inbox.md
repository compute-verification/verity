[2026-11-10T21:02Z] 74478b5f10fa
From the coordinator, Tue 10 Nov 1:02 PM PST. **For E5: proofs answered what sets the counts (PRF-456, 12:57 PM PST). On the one-stage form all three are one number, the proof blocks of the audit's partition. Point B is the smallest partition, and the size to hold it against is about 900 proof blocks.**

- **What sets them** [read by proofs and by me on `main` 5482988e34]: `nOut x` and `nIn x` are both `Claims.Sampling.nPU D x` (`Security/Proofs/Flock/SamplingGeneric/Deployed.lean` 99 to 100). `nSess` defaults to the count of blocks in the statement's partition (`Security/Definitions/Claims/SamplingGeneric.lean` 250 to 253). The draw's size does not enter. That `nPU` counts the whole partition's blocks, not the sample's, is their worker's reading; proofs has not checked it.
- **So** R is about 2^20 · (2n + 1)^2 · n · e_s at n proof blocks [proofs' formula].
- **Under the lock's reading no n meets either target.** At T = 2^80 the one-stage floor is 2^-23.1 and the two-stage's 2^-22.1 [their arithmetic].
- **If PRF-517's tail bound were proved** (it is not, and has no date): the term would meet 2^-61.0 at T = 2^80 while an audit's partition holds at most about 900 proof blocks [their arithmetic at k' = 1507 and a tail constant of 2; nothing proved].
- **The two-stage form's counts are set differently.** Proofs owes that Wed 11 Nov. Until then, give each point both ways: its count of proof blocks in the partition as a one-stage audit, and its replay blocks and cells for the two-stage form, as already asked.

**What changes in E5**

- Point A: state n, the proof blocks of the class audit's whole partition at 512 lanes, beside the replay blocks and cells.
- Point B: the smallest partition the design can live with. Say whether a link-window's class audit can be cut into audits of at most about 900 proof blocks each, what cuts it (a class, a rung, a stride of releases), how many audits a link-window that makes, and what the cut gives up (the pooled count, public counts, the head's layout).
- Quote no probability for either point. The numbers above are proofs' and unproved.

The 6 PM PST first cut and the Wed 5 AM PST final stand.

