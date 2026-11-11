**Title:** The joint period guarantee's error is a sum with no instance, and the communication bridge counts the sampling $\delta$ once a window

**Team:** lean (LEAN). Not filed: `research msg linear create` refuses a worker. Body for the coordinator to file.

**Why.** Daniel ordered an audit of every concrete-security number we quote (PRF-523); the bar is $2^{-100}$ at a prover of $T = 2^{80}$ SHA-512 compressions. A worker of proofs (PRF-537 AUDIT-RESOURCE) read the joint guarantee and its bridges at `2b869ed35db5`. Each row is a difference between what is stated and what it gives at served sizes. No second reader has checked them. Where your own number differs, yours is the one to state, with why.

**Done when**, for each row: the quote is corrected, or the bound is improved, or a reply says our reading is wrong, with why. A first reply is asked for by Wed 11 Nov 6 PM PST.

**Open decisions these rows depend on (Daniel's, none decided):** whether the collision premise and $T$ count SHA-512 calls or compressions (row 2); the route for SPOT's binding and whether `KSReg` is restated (PRF-523), whose number is not quoted here.

**1. The communication bridge.**
- Quoted: `r.Sound c p .network (n * (δ + εks + εbind + εtap))` (`Security/Definitions/Accounting/Guarantees/JointBridge/CommOfPeriodLeakAccepted.lean:55-75`).
- Evaluated: the sampling $\delta$ is multiplied by the number of windows. $n\delta \ge 1$ from $n = 100$ at $\delta = 1\%$; at $\delta = 2^{-40}$ it is $2^{-32.6}$, $2^{-30.5}$, $2^{-26.9}$ for hourly windows over a week, a month and a year ($n = 168$, $730$, $8{,}766$; the cadence is my guess). The certifier's pooled form (`PooledLeakAccepted`) keeps $\delta$ once.
- From: `Accounting.SecurityProofs.Bridge.CommOfPeriodLeakAccepted`.
- How known: read in Lean; computed by me. network-accounting's 00:15Z comment on PRF-523 reads the bridge the same way.
- What would change it: the bridge reading the pooled form.

**2. The joint guarantee.**
- Quoted: $\sum_a err_{a,s}$; $\sum_a (\delta_{a,s} + loss_a(q_{a,s}/2^{256}))$; $\sum_j \sum_a err_{j,a,s}$ (`Security/Definitions/Accounting/Guarantees/JointLive.lean:57-95`).
- Evaluated: no file gives these sums a value for any demo, so the joint guarantee has no number at a week, a month or a year. The argument of $loss$ at $q = 2^{80}$ is $2^{-176}$; what $loss$ makes of it differs by audit (a square root gives $2^{-88}$ before any count outside it).
- From: `JointPeriodLive`, `JointPeriodLiveECR`, `Bridge.WorkOfH1CheckedCreditPeriod`, `Bridge.SpaceOfPeriodHeld`.
- How known: read in Lean and the lock; $2^{-176}$ and $2^{-88}$ computed by me.
- What would change it: one instance at the remote demo's parameters, each audit's term beside its owner's number.

**3. Docstring figures in files `research msg owners` names lean for.**
- Quoted: "`2^-51.7` to `2^-41.7` at `T = 2^80`" and "only if `δ_ks ≤ 2^-56.3` a window" (`Security/Definitions/Accounting/Guarantees/ComposeMonitoringPeriod.lean:193-196`); "The budget asks `δ_ZK ≤ 2^-41/(48,000·n_π)` a proof" (`Guarantees/Headline.lean:139-141`); `PeriodHeldDeployed` at 105 rounds and $2^{19}$ blocks (`Security/Properties/Pous/PeriodDeployed.lean:53-84`).
- Evaluated: I recompute the first two and agree; they hold at $\kappa = 0$ and under a budget that is a hypothesis. The third is at a size and round count the onsite demo does not run (320 rounds, $2^{26.02}$ blocks). The rows are with their guarantees' owners (CMP-395, MEM-248); they are here because the files are yours.
- How known: read in Lean; computed by me.

**Pointers:** PRF-537 (final 00:15Z), `note:proofs/20261111T0014Z-finding-resource-audits-concrete-security`.
