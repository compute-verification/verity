**Title:** The certifier's period bound has no pooled number at a week, a month or a year in the tree

**Team:** network-accounting (NET). Not filed: `research msg linear create` refuses a worker. Body for the coordinator to file.

**Why.** Daniel ordered an audit of every concrete-security number we quote (PRF-523); the bar is $2^{-100}$ at a prover of $T = 2^{80}$ SHA-512 compressions. A worker of proofs (PRF-537 AUDIT-RESOURCE) read the certifier's guarantees at `2b869ed35db5`. Each row is a difference between what is quoted and what the statement gives at served sizes. No second reader has checked them. Where your own number differs, yours is the one to state, with why. Your 00:15Z comment on PRF-523 already says the first row in your own words.

**Done when**, for each row: the quote is corrected, or the bound is improved, or a reply says our reading is wrong, with why. A first reply is asked for by Wed 11 Nov 6 PM PST.

**Open decisions these rows depend on (Daniel's, none decided):** whether `KSReg` is restated, and the route for SPOT's binding (PRF-523); whether the collision premise and $T$ count SHA-512 calls or compressions. SPOT's own binding number is on PRF-523 and is not quoted here.

**1. No pooled figure for a monitoring period.**
- Quoted: "The union of the windows' failures over the monitoring period's live coins is not stated here." (`Security/Definitions/NetworkCertifier/Guarantees/Period.lean:63`); the pooled form is "conditional on two per-execution-phase hypotheses not yet discharged" (`verity/core/protocols/certifier/README.md:29, 36-38`).
- Evaluated: one window's bound is $\kappa_\delta + \sqrt{\gamma\cdot\Pr[B]} + \Pr[B_{ext}] + \Pr[B_{tap}]$ (`Period.lean:96-124`). Its terms other than the sample's add over $n$ windows, so a period costs $\log_2 n$ bits against one window: $7.4$, $9.5$ and $13.1$ bits for hourly windows over a week, a month and a year ($n = 168$, $730$, $8{,}766$; the cadence is my guess). $2^{-100}$ for a year then needs $2^{-113.1}$ a window from those terms. No statement fixes $\delta$, $n$ or a size.
- From: `NetTiming.SecurityProofs.PeriodLeakAccepted`; `PooledLeak` ($\delta + \sum_i e_i + \sum_i e'_i$).
- How known: read in Lean; the bit counts computed by me. The only evaluation at served sizes is your own table on PRF-456 (relayed from your 00:15Z comment on PRF-523), which is not in the tree.
- What would change it: a pooled statement with the two hypotheses discharged, evaluated at a week, a month and a year for a datacenter and the world.

**2. The window's knowledge error is an open premise.**
- Quoted: `trust_gap` of `NetTiming.SecurityProofs.PeriodLeakAccepted` in `Security/lean-audit.json`: `Claims.SamplingGeneric.Compiled.KS`, `NetTiming.CertifierRegisters`, `NetTiming.ScopeAcc`.
- Evaluated: the guarantee reads `Compiled.KS` (the extracted form), not `Compiled.KSReg`. In that form the work term grows with $n_{out}\cdot n_{in}\cdot n_{sess}$, and `main`'s instances set all three to $n_{PU}$ (`Security/Proofs/Flock/SamplingGeneric/Deployed.lean:98-100, 124`), so each costs $\log_2 n_{PU}$ bits at the certifier's sizes. The number itself is SPOT's, on PRF-523.
- How known: read in Lean and the lock. The sizes are yours (relayed through PRF-523's context); I did not recount them.
- What would change it: Daniel's route on PRF-523, and how many sessions an audit of the certifier's has.

**3. The sampling $\delta$ counted once a window.**
- Quoted: `r.Sound c p .network (n * (δ + εks + εbind + εtap))` (`Security/Definitions/Accounting/Guarantees/JointBridge/CommOfPeriodLeakAccepted.lean:75`; the lock names @lean for this record, and it reads your period statement).
- Evaluated: $n\delta \ge 1$ from $n = 100$ at $\delta = 1\%$; at $\delta = 2^{-40}$ it is $2^{-32.6}$, $2^{-30.5}$, $2^{-26.9}$ for hourly windows over a week, a month, a year. `PooledLeakAccepted` keeps $\delta$ once.
- How known: read in Lean; computed by me.
- What would change it: the bridge reading the pooled form.

**Pointers:** PRF-537 (final 00:15Z), `note:proofs/20261111T0014Z-finding-resource-audits-concrete-security`; PRF-456.
