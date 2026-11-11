**Title:** The inference-only remote period takes PoUW's failure as $p = 3\cdot 2^{-40}$, and two of its three terms have no number

**Team:** training (TRN). Not filed: `research msg linear create` refuses a worker. Body for the coordinator to file.

**Why.** Daniel ordered an audit of every concrete-security number we quote (PRF-523); the bar is $2^{-100}$ at a prover of $T = 2^{80}$ SHA-512 compressions. A worker of proofs (PRF-537 AUDIT-RESOURCE) read the remote period's guarantee at `2b869ed35db5`. Each row is a difference between a quoted number and what the statements it rests on give. No second reader has checked them. Where your own number differs, yours is the one to state, with why.

**Done when**, for each row: the quote is corrected, or the bound is improved, or a reply says our reading is wrong, with why. A first reply is asked for by Wed 11 Nov 6 PM PST.

**Open decisions these rows depend on (Daniel's, none decided):** the route for SPOT's binding and whether `KSReg` is restated (PRF-523), which set $\delta_{ks}$; whether the collision premise and $T$ count SHA-512 calls or compressions. SPOT's own binding number is on PRF-523 and is not quoted here.

**1. PoUW's leg, $p$.**
- Quoted: "`p = δ_s + δ_bind + δ_ks = 3 · 2⁻⁴⁰`" and "that bridge is not stated" (`Security/Definitions/Nci/Guarantees/RemotePeriod.lean:17-19`); "`3 · 2⁻⁴⁰` at `2⁻⁴⁰` each" (`Security/Definitions/Nci/Assumptions/Remote.lean:29`).
- Against: "δ_ks rests on the undischarged `ServiceSound`, and δ_bind has no deployed bound, so neither has a number yet" (`verity/core/protocols/remote/README.md:300-304`).
- Evaluated: the bound sets $\epsilon' = 2p/\delta$, so $p$ is paid in $\epsilon$. At $\delta = 10^{-6}$: $5.5\times10^{-6}$ at the quoted $p$; $1.9\times10^{-3}$ at $p = 2^{-30}$; $6.0\%$ at $p = 2^{-25}$. Of the two unnumbered terms, $\delta_{bind} = |L|\cdot 2^{-88}$ at $q = 2^{80}$ with $\kappa = 0$, that is $2^{-68}$ to $2^{-58}$ for $|L| = 2^{20}$ to $2^{30}$, inside $2^{-40}$ if $\kappa$ is; $\delta_{ks}$ is proofs' and has no value here.
- From: `Nci.SecurityProofs.DampenerRemotePeriod` (the lock names @training; `research msg owners` names @compute-accounting for the file). compute-accounting's own row 7 (its 00:02Z comment on PRF-537) says the same of "3 · 2^-40".
- How known: read in Lean; the three values of $\epsilon'$ computed by me.
- What would change it: $p$ taken from PoUW's owner once $\delta_{bind}$ and $\delta_{ks}$ have numbers.

**2. The sum over windows.**
- Quoted: failure $\delta = p/\epsilon' + J(1-q)^{K+1} + \sum_t ks_t$, at most $10^{-6}$ at the honest witness (`RemotePeriod.lean:102, 146-152`).
- Evaluated: over a year of hourly windows ($n = 8{,}766$; the cadence is my guess) $\sum_t ks_t \le 10^{-6}$ needs $ks_t \le 2^{-33.0}$ a window, and $2^{-100}$ needs $2^{-113.1}$ a window.
- How known: read in Lean; computed by me.
- What would change it: the value of $ks_t$ at the remote demo's sizes, which is proofs' (PRF-523).

**3. The headline slowdown (relayed, not checked by me).**
- vllm's lead relays your round comment on TOP-398 (PRF-523, 00:15Z): alone the inference-only protocol proves a slowdown of 1, and the number quoted for it is the composite's. I did not read this in the Lean and name no file for it.
- What would change it: where the quoted number is stated, and which statement gives it.

**Pointers:** PRF-537 (final 00:15Z), `note:proofs/20261111T0014Z-finding-resource-audits-concrete-security`.
