**Title:** The composite's published error functions have no instance at the remote demo's parameters

**Team:** soundness (PRF, label `lane:soundness`). Not filed: `research msg linear create` refuses a worker. Body for the coordinator to file. Row 2 is also a comment on PRF-427, the existing issue on the same finding.

**Why.** Daniel ordered an audit of every concrete-security number we quote (PRF-523); the bar is $2^{-100}$ at a prover of $T = 2^{80}$ SHA-512 compressions. A worker of proofs (PRF-537 AUDIT-RESOURCE) read the composite at `2b869ed35db5`. Each row is a difference between what is quoted and what is evaluated. No second reader has checked them. Where your own number differs, yours is the one to state, with why.

**Done when**, for each row: the quote is corrected, or the bound is improved, or a reply says our reading is wrong, with why. A first reply is asked for by Wed 11 Nov 6 PM PST.

**Open decisions these rows depend on (Daniel's, none decided):** the random-oracle ruling (row 2); whether the collision premise and $T$ count SHA-512 calls or compressions (rows 1 and 3); the route for SPOT's binding (PRF-523), whose number is not quoted here.

**1. No total for any demo.**
- Quoted: "`foreignBits`, `signatures`, `delta`, `epsForge` and `loss` (what the composite publishes, functions of the parameters" (`Security/Properties/Verity/Game.lean:727`; `crBound`, `epsForge` at `:74, 85, 109`).
- Evaluated: no file gives these functions a value at the remote demo's parameters, so the composite's error there has no number. `trust_gap` of `Verity.remote_soundness_of_subprotocols` (`Security/Properties/lean-audit.json`) lists `InferenceOnlySound`, `PouwSound`, `SpotSound`.
- How known: read in Lean and the lock.
- What would change it: each subprotocol's `delta`, `epsForge` and `loss` at the remote demo's parameters, and their sum.

**2. LMS's forging bound.**
- Quoted: `LmsUnforgeable … εForge : ℕ → ℕ → ℝ` (`Security/Definitions/Hash/Assumptions.lean:55-58`); "**The published `ε_forge(t, Q)`**" (`Game.lean:725-727`).
- Evaluated: $\epsilon_{forge}$ is a free function whose one instance is `epsForge _ _ _ := 0` (`Security/Properties/Verity/Idle.lean:138`; read by AUDIT-CRYPTO and by me). The deployed set is `Params(64, 1, 20)` (`verity/ml/crypto/lms.py:112`). AUDIT-CRYPTO (PRF-536, relayed) carries the 6 Nov note's formula to it "by analogy, not proved": $\epsilon_{forge} \approx (4t + 28)/2^{512} = 2^{-430}$ at $t = 2^{80}$, $2^{-410}$ pooled over $2^{20}$ device keys. My own figure is from memory of the literature and not fetched, so AUDIT-CRYPTO's is the one to check.
- Depends on the random-oracle ruling: the bound is one in the random oracle model.
- What would change it: the claim restated with its formula and its counting mode (PRF-427).

**3. The remote README's target for PoUW's two unnumbered terms.**
- Quoted: "δ_bind + δ_ks must stay below about 1.1·10⁻⁶ a phase" at $n = 168$, $730$, $8{,}766$ (`verity/core/protocols/remote/README.md:300-304`).
- Evaluated: $1.1\times10^{-6} = 2^{-19.7}$ a phase holds the pooled failure to 1% a year. $2^{-100}$ a year needs $2^{-113.1}$ a phase; $\delta_{bind}$'s form reads $|L|\cdot 2^{-88}$ at $q = 2^{80}$ with $\kappa = 0$ (compute-accounting's row; CMP-395).
- How known: read; computed by me.
- What would change it: the text saying which target the threshold is for.

**Not in it:** hm96's hiding at the pinned key (`remote/README.md:175` quotes it) is proofs' own row, on PRF-536 and LEAN-186.

**Pointers:** PRF-537 (final 00:15Z), `note:proofs/20261111T0014Z-finding-resource-audits-concrete-security`; PRF-536, `note:proofs/20261111T0013Z-finding-audit-crypto-primitives-at-2-80`; PRF-427.
