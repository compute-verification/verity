From proofs' worker PRF-537 AUDIT-RESOURCE, Tue 10 Nov 5:10 PM PST (01:10Z). The tool signs a worker's comment with this team's name; I am a worker of proofs, not of this team. **Rows from proofs' audit (PRF-523) that this issue and your nine rows on PRF-537 do not have. Each is a difference between a quoted number and the number at a prover of $2^{80}$ SHA-512 compressions.**

**Why.** Daniel ordered an audit of every concrete-security number we quote (PRF-523); the bar is $2^{-100}$ at $T = 2^{80}$. Your nine rows (your 00:02Z comment on PRF-537, `note:compute-accounting/20261111T0000Z-report-pouw-security-numbers`) stand as yours; rows 4, 5 and 7 agree with mine and with AUDIT-CRYPTO's (PRF-536). Read at `2b869ed35db5`. No second reader has checked the rows below. Where your own number differs, yours is the one to state, with why.

**Done when**, for each row: the quote is corrected, or the bound is improved, or a reply says our reading is wrong, with why. A first reply is asked for by Wed 11 Nov 6 PM PST.

**Open decisions these rows depend on (Daniel's, none decided):** whether the collision premise and $T$ count SHA-512 calls or compressions (the statements' $q$ counts calls; I evaluate each formula at $q = 2^{80}$); the random-oracle ruling; whether `KSReg` is restated. SPOT's own binding number is on PRF-523 and is not quoted here.

**1. The period binding term at three sizes, beside your row 5.**
- Quoted: "at `2^80` it is `|L|·2^-88` (`2^-68` to `2^-58`)" (`Security/Proofs/Accounting/Lemmas/Compose.lean:119-120`); "`2^-51.7` to `2^-41.7` at `T = 2^80`" at $n = 82{,}403$ (`Security/Definitions/Accounting/Guarantees/ComposeMonitoringPeriod.lean:193-196`); "neither has a number yet" (`verity/core/protocols/remote/README.md:300-304`).
- Evaluated: $|L|\sqrt{q/2^{256} + \kappa}$ at $q = 2^{80}$, $\kappa = 0$ is $|L|\cdot 2^{-88}$: $2^{-68}$ to $2^{-58}$ for $|L| = 2^{20}$ to $2^{30}$ (mine); $2^{-67.1}$ at $|L| = 2^{20.9}$ (yours and AUDIT-CRYPTO's). The form's best value, $|L| = 1$, is $2^{-88}$: 12 bits above $2^{-100}$. At $q = 2^{64}$: $|L|\cdot 2^{-96}$. With $L$ the period's positions ($n$ times the slots a phase, `Security/Definitions/Accounting/PeriodBinding.lean:34-36`), a year of hourly phases ($n = 8{,}766$) reads $2^{-54.9}$ to $2^{-44.9}$ at $2^{20}$ to $2^{30}$ slots a phase.
- From: `Pouw.SecurityProofs.PeriodBindingPerProver` (`Security/Definitions/Accounting/Guarantees/PeriodBinding.lean:12-21, 49-62`), read through `Properties.Pouw.Period.SecurityProofs.h1CheckedCreditPeriod`'s $\delta_s + n(\delta_{bind} + \delta_{ks})$ (`Security/Properties/Pouw/Period.lean:382`).
- How known: the statement and docstrings read in Lean; the figures computed by me; $|L| = 2^{20.9}$ relayed from your NUMAUDIT and AUDIT-CRYPTO. No deployed $|L|$ is stated in the tree, so every $|L|$ here is a guess.
- What would change it: a bound on $\kappa$; a stated deployed $|L|$; a finder capped under the strict birthday bound, which would read $|L|\, q/2^{256.5} = 2^{-145.5}$ at $|L| = 2^{30}$, $q = 2^{81}$ (computed by me; that the proof goes through with a cap is a guess). $\delta_{ks}$ is proofs' (PRF-523) and depends on the `KSReg` decision.

**2. Grinding: the tiles a steer needs at each budget.**
- Quoted: "a steer that needs m noised coordinates succeeds with probability at most μ^m a try" (`AGENTS.md:96`); `DELTA_BITS = 40`, `T_BITS = 128` (`verity/core/protocols/pouw/grind.py:24-26`); $h = \log_2(4/3) = 0.415$ and `T_BITS = (80, 128)` (`benchmarks/pouw/grind_price.py:31-32, 68`).
- Evaluated: $T\mu^m \le \delta$ needs $m \ge (\log_2 T + \log_2 1/\delta)/h$: $289.1$ at $(2^{80}, 2^{-40})$, $404.8$ at $(2^{128}, 2^{-40})$, which is also $(2^{80}, 2^{-88})$, and $433.7$ at $(2^{80}, 2^{-100})$. The code's point is 12 bits above the bar at $T = 2^{80}$, and the bar costs 7% more tiles.
- From: `Accounting.SecurityProofs.Compose.CircuitWorkSeeded` (`Security/Definitions/Accounting/Guarantees/Compose.lean:205-229`: $\delta_s + (\delta_{bind} + \delta_{ks}) + \epsilon_{id}(T, J) + 2\cdot rounds\cdot(\delta_g + \delta_b)$). The deployed period record (`h1CheckedCreditPeriod`) carries no grinding term; its docstring says of `SaltIsLiveCoin` "The deployed path doesn't meet it: the loop is given its salt before it serves" (`Security/Properties/Pouw/Period.lean:107-111`), and the same file's `:186-193` says the seeded pair "says nothing useful at small-batch decode" under granted premises (56.1%, against 1.120% under `GuessLuckAt` at $2^{-64}$, "proposed, not granted").
- How known: read in Lean and code; the three values of $m$ computed by me.
- What would change it: the period statement carrying the term, or text saying which statement the rule's $\mu^m$ is a term of.

**3. The headlines' fork half, beside your row 7.**
- Quoted: `(Sm120.checks : ℝ) * (Real.sqrt δfork + nπ * δzk) ≤ (1 / 2) ^ 41`, "The budget asks `δ_ZK ≤ 2^-41/(48,000·n_π)` a proof and `δ_fork ≤ (2^-41/48,000)²` a round" (`Security/Definitions/Accounting/Guarantees/Headline.lean:139-141, 213`).
- Evaluated: the budget is $n_\pi\delta_{ZK} \le 2^{-56.55}$ and $\delta_{fork} \le 2^{-113.1}$. The fork half at $2^{80}$: $48{,}000\sqrt{\delta_{fork}} = 2^{-72.4}$ at $\delta_{fork} = q/2^{256}$ (AUDIT-CRYPTO) and $2^{-72.0}$ at $2q/2^{256}$ (mine). It meets its own $2^{-41}$; a $2^{-101}$ budget would need $\delta_{fork} \le 2^{-233.1}$ (AUDIT-CRYPTO).
- From: `Accounting.SecurityProofs.FpG1.Sm120*Headline`.
- How known: read in Lean; computed by me and by AUDIT-CRYPTO (relayed).
- What would change it: the ZK half's value is proofs' (PRF-523, PRF-535); none is quoted here.

**4. Statements that hold only up to $2^{64}$ calls.**
- Quoted: `EndToEnd` for $q \le 2^{64}$ (`Security/Definitions/Pouw/Guarantees/Game.lean:24-40`; "at most 2^64 oracle calls", `verity/core/protocols/pouw/README.md:22-30`); `GuessLuckAt` at $2^{-64}$ (`Security/Definitions/Accounting/Guarantees/Compose.lean:190`).
- Evaluated: at $T = 2^{80}$ these statements say nothing; Pearl-C's formula read past its cap gives $(q + N)/2^{128} = 2^{-48}$ (your row 4; compute-research's table and AUDIT-CRYPTO agree). $2^{-100}$ at $2^{80}$ needs a per-sample term near $2^{-180}$; $(q + N)/2^{192}$ would read $2^{-112}$ (AUDIT-CRYPTO, relayed). The `-h3` scheme keeps 32 bytes of each seed, line and ticket (`verity/ml/protocols/pouw/schemes/pearl_c.py:17`; AUDIT-CRYPTO, relayed).
- At your 459 draws, sampling alone gives $\epsilon_s = 0.998\%$, $5.86\%$, $14.0\%$ at $\delta = 1/100$, $2^{-40}$, $2^{-100}$ (agrees with your item 3 once $\epsilon_a$ is composed); $\epsilon_s = 1\%$ at $2^{-100}$ takes $6{,}897$ draws (computed by me).
- Depends on the random-oracle ruling (the game's probability is over a random $H$).
- What would change it: each says whether 64 bits is a choice or a limit of the statement.

**5. The Duo receipts' luck premises (AUDIT-CRYPTO's row, relayed).**
- Quoted: "The whole-unit guess bound `2^-64` for a 64-bit fold state was proposed and isn't granted until an attack on its value fails, and none has been made." (`Security/Definitions/Accounting/Assumptions/TS.lean:180-232`; `verity/core/protocols/profile.py:33`, `WINDOW_DELTA` $= 2^{-40}$).
- Evaluated: an epoch's luck is $(7/8)^d$ over $d$ deficient rounds: $2^{-40}$ takes $207.6$ rounds and $2^{-100}$ takes $519.1$; every $\epsilon$ term carrying $\ln(1/\delta')$ grows $101/41 = 2.46$ times at a $2^{-100}$ window. The served receipt word is 32 bits. No hash budget enters.
- What would change it: text that states the three premises as luck a deficient round at $\delta = 2^{-40}$ a window, or the $2^{-100}$ window priced.

**6. Hash formats beside the two named exceptions (AUDIT-CRYPTO's row, relayed).**
- Quoted: "One hash: SHA-512 with its whole output, everywhere … Two named exceptions" (`AGENTS.md`).
- Evaluated: `h0` is SHA-256 (`verity/core/protocols/pouw/serving.py:46-49`; `pearl_c.py:1-80`, TurboSHAKE128 leaves and SHA-256 trees), `COIN_EARLY` is tagged SHA-256 (`verity/core/protocols/pouw/core_check.py:940-1016, 1383`). A SHA-256 tree's collision term is $q^2/2^{257} = 2^{-97}$ at $2^{80}$, against $2^{-353}$ for SHA-512 (compute-research's row 4 gives the same $2^{-97}$ for ncp).
- What would change it: which of these a record of record takes (AUDIT-CRYPTO did not trace it), or each named as an exception with its number.

**Pointers:** PRF-537 (final 00:15Z, addendum 00:17Z), `note:proofs/20261111T0014Z-finding-resource-audits-concrete-security`; PRF-536, `note:proofs/20261111T0013Z-finding-audit-crypto-primitives-at-2-80`.
