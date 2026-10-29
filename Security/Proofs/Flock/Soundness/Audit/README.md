# The audit layer (`Audit/`): what is proved

The sampled audit's guarantee, the integrity profile of `docs/audit-protocols.md` §2, over any Boolean circuit, any
partition, any sampling design, and any per-unit proof system. The derivations and the deviations from that plan are in
`../DESIGN.md` §12. Everything here is proved; the per-unit proof system enters only through named hypotheses.

**The model.** The prover registers `R` (its roots); the verifier samples units from its own randomness after registration;
one batched session proves the sampled proof blocks (`audit`). The **committed trace** `X` is an analysis function of the
prover's state at registration (its registration and its session strategy for every sample), so the sample is independent of
its faulty proof blocks; nobody computes it. A unit is **faulty** in `X` (the Lean's `wrong`) when its committed outputs differ
from its gates applied to its committed inputs (`Partition.Correct`). Every bound holds for every prover strategy
(`Game.prob … σ`), the form `table_sound_exec` takes. The Lean's names predate the Glossary's: a `Law` is a sampling
design, its `escape` (here $e_L(B)$) is the acceptance probability $P_a(B)$, and $\delta_{link}$ (`LinkSound`) is the
Glossary's $\varepsilon_{bind}$.

**Named hypotheses** (`Prop`s taken as hypotheses; no `axiom`): each is stated in `../ASSUMPTIONS.md`.

**What the reviewer checks here.** The theorems hold for every choice of committed trace `X` and extractor `ext`,
so their meaning rests on `X` being the committed values. At the oracle layer `X` is the registration itself
(`Partition.oracle`). At the compiled layer it is the plurality of what the extractor recovers behind each commit
string, which the instantiation defines and `LinkSound` ties to the extractor. A consumer who opens values under the
roots gets `X`'s values up to that opening's own binding term, which the audit does not count.

**Theorems, one-stage** (`Audit/OneStage.lean`), with $e_L(B)$ the probability that the sample misses $B$:

- `audit_le`: $\Pr[\text{accept} \wedge E(X, S)] \le \Pr_S[E(X, S) \wedge S \cap \mathrm{wrong}(X) = \emptyset] +
  \varepsilon_{ks} + \delta_{link}$ for every event $E$ on the committed trace and the sample;
- `audit_profile`: $\Pr[\text{accept} \wedge \mathrm{wrong}(X) \in \mathcal B] \le \sup_{B \in \mathcal B} e_L(B) +
  \varepsilon_{ks} + \delta_{link}$;
- `audit_count`: $\delta(K) = \mathrm{miss}_L(K) + \varepsilon_{ks} + \delta_{link}$;
- `audit_drawn`: sampled proof blocks are correct up to $\varepsilon_{ks} + \delta_{link}$, even for a consumer who looks after
  the sample;
- `audits_seq`: audits in sequence fail with at most the sum of their bounds, each holding from every history. Nested,
  this is the lifetime union bound, with every term against a prover that took part in all the audits;
- `Partition.audit_cone`: for committed wires $T$ fixed before the sample, $\Pr[\text{accept} \wedge X_T \ne
  \mathrm{eval}_C(a)_T] \le \max_{u \in \mathrm{cone}(T)} (1 - \pi_u) + \varepsilon_{ks} + \delta_{link} + \delta_{in}$;
- `Partition.audit_cone_count`: at least $j$ faulty proof blocks in a cone;
- `Partition.audit_covered`: a cone inside the sample, for wires chosen even after the sample: $\varepsilon_{ks} +
  \delta_{link} + \delta_{in}$;
- `Partition.oracle_count`, `oracle_profile`, `oracle_drawn`, `oracle_cone`: the same at the oracle layer, from
  `SessionSound ε` and `InputsSound δ_in`, with $\delta_{link} = 0$.

**Theorems, sampling designs** (`Audit/Law.lean`): `bernoulli_escape`, a Bernoulli($p$) sample's acceptance probability
$(1-p)^{|B|}$; `subset_escape`, the uniform $k$-subset's acceptance probability $\binom{n-|B|}{k}/\binom{n}{k}$;
`subset_miss`, its count curve $\binom{n-K}{k}/\binom{n}{k}$; `subset_escape_le_prod`, the product bound
$e(B) \le \prod_{u \in B}(1 - \pi_u) = (1 - k/n)^{|B|}$, so inclusion probabilities alone give a consumer a valid bound;
`subset_minimax`, **uniform is optimal**: no sampling design that proves $k$ units on average has a smaller count curve,
at any $K$; `demo_count_590`, the demo's point (1,024 heads, 16 sampled): 590 or more faulty heads have acceptance
probability at most $2^{-20}$, by exact arithmetic. **A block's share** (`Audit/Share.lean`): for a uniform $kd$-subset
$S$ of $N$ units and a block $B$, conditioned on $|S \cap B| = j$ the share $S \cap B$ is a uniform $j$-subset of $B$
(`share_uniform`, `share_cond`), the share size is hypergeometric (`share_card`), and $W \subseteq B$ escapes at share
size $j$ with that size's probability times the member's own $\binom{|B|-|W|}{j}/\binom{|B|}{j}$ (`share_escape`,
`share_escape_total`). **The pooled escape** (`Audit/Pooled.lean`): over $T$ periods, each a fresh uniform
$kd$-subset of the same $N$ units, an adaptive adversary that fixes period $t$'s wrong set $W_t$ from the earlier draws
escapes every draw with $\sum_t |W_t| \ge M$ with probability at most $(1 - kd/N)^M$ (`horizon_escape`), at most
$\prod_t \binom{N-w_t}{kd}/\binom{N}{kd}$ when $|W_t| \ge w_t$ (`horizon_escape_prod`); at $\delta$, the escaped total
reaches $\ln(1/\delta)/q$, $q = kd/N$, with probability at most $\delta$ (`pooled_escape_log`, in `Claims.Pr`, which is outside the audit layer:
`Claims/Sampling/PooledPr.lean`).
These are instances of **the monitoring-period step** (`Audit/Periods.lean`, `FlockSoundness.Audit.Periods`): $T$
periods of independent uniform coins, any coin space per period, and an event of period $t$ that may read the periods
up to $t$ with probability at most $p_t$ given any history (`avg_prod_le`, the one induction). It holds in at least
$M$ periods with probability at most $\sum_{|S|=M}\prod_{t\in S} p_t$ (`prCoin_count_le`), $\binom{T}{M}p^M$ at one $p$
(`prCoin_count_le_choose`), $(1+(z-1)p)^T/z^M$ for $z \ge 1$ (`prCoin_count_mul_pow_le`, and at $\delta$
`prCoin_count_le_of_log`). When $p$ bounds "accepted and short" and shortness is fixed before any coins are flipped, every
period is accepted with at least $M$ short with probability at most $p^M$ (`prCoin_forall_count_le_pow`); shortness the
period's own coins decide gets only the count bounds.
`subset_exec_escape_le` (`ExecDraw.lean`, beside the executable's
other checks, since it imports `Flock.Draw`), **the running sampler**: over a uniform byte stream, the verifier's
executable `Flock.Draw.subset k n` returns a sample missing $B$ with probability at most `subset_escape`'s
$\binom{n-|B|}{k}/\binom{n}{k}$, through its rejection sampler, its `HashMap` Fisher–Yates and its `Array.qsort`.
`stratified_exec_escape_le` (`ExecStratified.lean`): the executable `Flock.Draw.stratified`, on strata that describe
$\sigma$'s (`ExecStrata`), misses $B$ with probability at most `Law.stratified`'s acceptance probability, the product
over strata; `flock_e2e_drawn_exec` and `flock_e2e_count_exec` are the e2e theorems at the executable's sampling design
(`execStratified`), the count curve bounded by `Law.stratified`'s `miss`. `execStrata_of_ofJson`
(`ExecStrataOf.lean`): the strata of a sample file `UnitDraw.ofJson` accepts describe some assignment's (`ExecStrata`), each
stratum's `k` at most its size, so those two hypotheses follow from the verifier accepting. The `FailClosed` claims take
neither: their law is the verifier's own (`liveLaw`) and their bound the infimum over such assignments
(`liveStrata_of_ofJson`: non-empty at an accepted sample file's law). `drawOS_subset_escape_le`,
`drawOS_stratified_escape_le` and `drawOS_work_escape_le` (`ExecDrawOS.lean`), **the live sample**: `Flock.Draw.drawOS`,
which is `drawWith` on `IO.getRandomBytes` (`drawOS_def`), extending a stream that runs out and rerunning, misses $B$
with probability at most the same acceptance probabilities, on a byte source that A3 makes uniform
(`Assumptions.UniformRandomBytes`); `execOS_escape_le` and `execOS_subset_escape_le` state it for the live sample as a
sampling design (`execOS`). `subsetOn_escape_le`, `drawOn_escape_le` and `keyedWindow_escape_le` (`KeyedDraw.lean`),
**the PoUW circuit's keyed samples**. `Key.subset`, which samples each swap from its own `verity.ml.randomness` stream, has
no higher acceptance probability than the uniform $k$-subset. The window's per-call samples under one key form one
`Law.stratified` over its (call, template) strata. So the live window sample (`keyedWindow`) misses $B$ with probability
at most `Law.stratified`'s acceptance probability plus the $\eta$ of A4 (`Assumptions.KeyedStreamsUniform`), on a
uniform window secret (A5). `keyedWindow_audit_of_record` and `keyedWindow_extraction_audit_of_record` put it through
the window pins (`audit_window_split_of_record_of_le_slack`, and at the compiled layer
`extraction_audit_window_split_of_record_of_le_slack`): the 0.1% event has probability at most
$2^{-40} + \eta + \varepsilon_{ks} + \delta_{link}$. The key's context is the receipt digest, so the real game's sample
depends on the registration: `auditReg` (`Audit/RegDraw.lean`), and `prob_auditReg` moves each of its strategies to the
game whose sampling design is fixed at its registration, which `keyedWindowReg_extraction_audit_of_record` reads.

**Theorems, the stratified sampling design** (`Audit/Stratified.lean`, the default sampling design;
`docs/sampling-strategies.md` §4): `Law.stratified σ k`, an independent uniform $k_s$-subset inside each stratum of
$\sigma : \mathrm{Fin}\,n \to \mathrm{Fin}\,m$; `stratified_escape`, the product of hypergeometrics
$\prod_s \binom{n_s - m_s}{k_s}/\binom{n_s}{k_s}$ with $m_s = |B \cap U_s|$, which is what
`verity.core.protocols.profile.Stratified` computes; `stratified_escape_floor`, **a floor catches its whole stratum**:
with $k_s \ge 1$, a set containing $U_s$ has acceptance probability 0; `audit_whole_stratum` and
`extraction_audit_whole_stratum`, the one-stage profile instantiated, so an audit accepts while a whole floored stratum
is faulty with probability at most the per-unit terms. `stratified_miss_eq_greedy` (`Audit/StratifiedMiss.lean`), **the
stratified count curve**: a count vector that some $\tau \in (0, 1]$ separates, every acceptance factor
$r_s(c) = (n_s - c - k_s)/(n_s - c)$ it takes at least $\tau$ and every one it leaves at most $\tau$, attains
$\mathrm{miss}(\sum_s a_s) = \prod_s \binom{n_s - a_s}{k_s}/\binom{n_s}{k_s}$. A greedy fill has such a $\tau$, so this
is `Stratified.miss` and discharges `audit_exfiltration`'s `hk` for a stratified sampling design.

**Theorems, the work sampling design** (`Audit/Work.lean`, the default sampling design's second version;
`../../PROTOCOL.md` §7.3): `Law.work σ w f K` is `Law.stratified` with stratum $s$ sampling
$k_s = \min(n_s, \max(f_s, \lceil K w_s n_s / W \rceil))$, where $w_s$ is the PoUW work of each of its units,
$W = \sum_s w_s n_s$ and $f_s \ge 1$ its floor, set per stratum in the verifier's work table; every stratified and
one-stage theorem holds of it as stated, and the theorems below hold for every such $f$. `floor_le_workK`, **the
floor**: a stratum samples at least $\min(n_s, f_s)$ units, so every non-empty stratum samples at least one and
`audit_work_whole_stratum` and `extraction_audit_work_whole_stratum` catch a wholly faulty one. `work_escape_floor_le`,
**the per-stratum floor bound**: a set with $b_s$ units in a non-empty stratum has acceptance probability at most
$((n_s - b_s)/n_s)^{f_s}$, audited at both layers (`audit_work_floor`, `extraction_audit_work_floor`). `workK_ge`: a
stratum samples at least $K w_s n_s / W$ units, or all of them. `sum_workK_le`: at most $K + \sum_s f_s$ units are sampled.
`workK_rest`: a stratum that does no work samples $\min(n_s, f_s)$. `work_escape_le_count`: when those strata have floors
of at least the stratified sampling design's sizes at $C$ (`countK`, which `countK_eq` shows is `stratumK C n_s N`), a
set's acceptance probability under the work sampling design is at most its units' in them under the stratified one at
$C$ (`choose_div_anti`: the hypergeometric acceptance probability falls as the sample grows), and `audit_work_rest` and
`extraction_audit_work_rest` carry any such integrity profile to the audit. `work_escape_le`, **the work bound**: a set
$B$ of units doing $W_B$ of the work has acceptance probability at most $((W - W_B)/W)^K$, from the per-stratum
hypergeometric and weighted AM-GM over the work shares. `audit_work` and `extraction_audit_work` bound
$\Pr[\text{accept} \wedge W_{\mathrm{wrong}(X)} \ge T]$ by $((W - T)/W)^K$ plus the per-unit terms. `record_sizing` and
`audit_work_of_record`: at $K = 27{,}713$, faulty proof blocks doing at least 0.1% of the work have acceptance probability at
most $2^{-40}$; `record_sizing_one_fewer`: $K = 27{,}712$ does not reach it under this bound. `Law.widen` and
`escape_widen_le`, the seam for closure samples: proving more units than are sampled, on the same coins, never raises an
acceptance probability. `FlockWork.lean`'s `workRule_eq_draw` and `countRule_eq_draw`: the executable's
`Flock.Draw.workK` and `stratumK` are `Law.workRule` and `countRule`.

**Theorems, harm bounds** (`Audit/HarmBound.lean`, `IsHarmBound` proved for the stratified sampling design):
`stratified_escape_le_exp`, **the closed form**: a set with $c_s$ units in stratum $s$ has acceptance probability at most
$\exp(-\sum_s k_s c_s / n_s)$ (`work_escape_le_exp` for the work sampling design), so one with acceptance probability at
least $\delta > 0$ has $c_s \le n_s - k_s$ and $\sum_s k_s c_s / n_s \le \ln(1/\delta)$ (`stratified_risk_le`).
`stratified_isHarmBound_dual`: with harm $g_s$ per unit of stratum $s$, every $\tau \ge 0$ gives the harm bound
$\tau \ln(1/\delta) + \sum_s (n_s - k_s) \max(0, g_s - \tau k_s / n_s)$, weak duality for that fractional knapsack.
`stratified_isHarmBound_fill`: a fill that some $\tau$ separates attains it, so the greedy fill's value (the sampling
optimizer's `density_bound`) is a harm bound. `work_isHarmBound_fill` is the work sampling design's, with harm the
credited work; `stratified_isHarmBound_bits` and `Partition.audit_exfiltration_stratified` are the exfiltration form,
the floor of the fill's value over output bits as `audit_exfiltration`'s $H$. `stratified_isHarmBound_exact` is the same
duality over the exact risk $\ln(\binom{n_s}{k_s}/\binom{n_s - c}{k_s})$ (`stratumRisk`, at least $k_s c / n_s$ by
`le_stratumRisk`), the knapsack whose value core's `Stratified.harm_bound` computes.

**Theorems, the cumulative harm budget** (`Audit/HarmBudget.lean`, `Audit/HarmBudgetLife.lean`), over a run of adaptive
rounds (`Adaptive.Play`): each round's faulty proof blocks $B_t$ and sampling design are fixed by the history before that
round's own coins, and the prover may stop after any round. A sampling design meets the **design condition**
(`Law.Design L π`) when every $B$ fixed before the coins has acceptance probability at most $\exp(-\sum_{u \in B}
\pi_u)$: `bernoulli_design` ($\pi_u = q$), `subset_design` ($k/n$), `stratified_design` ($k_s/n_s$, the closed form of
`stratified_escape_le_exp`) and `work_design` ($K w_u / W$). `harm_budget`: for harms $h \ge 0$ and any $\lambda \ge 0$
with $\lambda h_u \le \pi_u$, $\Pr[\text{accept} \wedge \sum_t h(B_t) \ge H] \le e^{-\lambda H} + \sum_t
\mathrm{err}_t$, where $\mathrm{err}_t$ bounds the probability that round $t$'s sample hits $B_t$ and the round still
passes (`Play.Drawn`, indexed by the rounds left after it); `harm_budget_rate` takes $\lambda = \min_{h_u > 0} \pi_u /
h_u$. `harm_count`: under Bernoulli($q$) samples, an accepted run with $K$ or more faulty proof blocks summed over its rounds has
probability at most $(1-q)^K + \sum_t \mathrm{err}_t$, tight for one round (`harm_count_tight`); `harm_count_kstar`
gives $1 - p + \sum_t \mathrm{err}_t$ at $k^*(p) = \ln(1/(1-p)) / (-\ln(1-q))$. `life_harm_budget` is the instance over
a lifetime of $N$ one-stage audits (`life`): after each accepted audit the prover chooses whether to go on, and a
rejection ends the lifetime with no resample, so $\mathrm{err}_m = \varepsilon_{ks,m} + \delta_{link,m}$ by `audit_drawn`.
It fixes each audit's sampling design in advance by its index, over one $n$; `harm_budget` itself lets the history
choose the design. For cumulative harm this replaces the union bound of `audits_seq`: $e^{-\lambda H}$ is paid once for
the whole run.

**Theorems, closure samples** (`Audit/Closure.lean`; `../../PROTOCOL.md` §7.3): `Law.closure L cl` is `L` widened on its coins
by the verifier's closure map, each sampled proof block proved with its closure. `closure_escape`: its acceptance probability is
the base sampling design's acceptance probability of $B \cup \mathrm{unsound}(B)$, $\mathrm{unsound}(B)$ being the units
whose closure meets $B$, so a faulty proof block that is never sampled on its own is charged to the units above it. `unsoundWork`
is the work of $B \cup \mathrm{unsound}(B)$: a closure map need not list a unit in its own closure, so the faulty proof blocks'
own work is counted explicitly (`workOf_le_unsoundWork`). `harm_le_unsoundWork`: a
faulty proof block's harm (its own work and that of the units whose closure holds it) is in the unsound work, sampled or not;
`unsoundWork_le_harm` (coverage) and `unsoundWork_singleton`. `audit_work_closure` and `extraction_audit_work_closure`:
under the work sampling design's closure sample, at any floors $f_s \ge 1$,
$\Pr[\text{accept} \wedge \mathrm{work}(B \cup \mathrm{unsound}(B)) \ge T] \le ((W - T)/W)^K$ plus the per-unit terms, for
$B = \mathrm{wrong}(X)$; `audit_work_closure_harm`: likewise for a faulty proof block's harm; `audit_work_closure_of_record`:
$2^{-40}$ at $K = 27{,}713$ for unsound work of at least 0.1%.

**Theorems, windows of calls** (`Audit/Window.lean`): a window is one stratified sampling design
over its (call, template) strata, the calls sampled independently. `Covers σ k w K`: every stratum samples at least its
share $K w_s n_s / W$, or all its units. `stratified_escape_le_of_covers`: then
$\mathrm{escape}(B) \le ((W - W_B)/W)^K$, for any such sampling design. `covers_work`: the work sampling design covers
its own work. `covers_window`: the per-call split, call $c$ at its share
$K_c = \min(N_c, \max(1, \lceil K W_c / W \rceil))$, covers the window's work when each call's work is in one stratum
(`hone`); with two strata doing work in a call, a binding cap breaks the bound itself (the module docstring's example).
`covers_of_floor`: y's floors $f_s$ with $K_y v_s n_s \le f_s N$ cover the y weights $v$ at $K_y$. `audit_window` and
`extraction_audit_window`:
$\Pr[\text{accept} \wedge (\mathrm{unsoundWork}_w \ge T \vee \mathrm{unsoundWork}_v \ge T_y)]$ is at most the larger of
the two bounds plus the per-unit terms, not their sum, since both bound the one committed faulty set.
`audit_window_of_le`: the same for any sampling design whose acceptance probabilities are at most the product design's,
such as an executable sample; `audit_window_of_le_slack` and, at the compiled layer,
`extraction_audit_window_of_le_slack`: for a sampling design within $\eta$ of it, with $\eta$ added to the bound, as a
keyed sample needs, and `audit_window_split_of_record_of_le_slack`, its record form, $2^{-40} + \eta$ plus the per-unit
terms, with `extraction_audit_window_split_of_record_of_le_slack` at the compiled layer. `audit_window_of_record`:
$2^{-40}$ at $K = K_y = 27{,}713$ for a 0.1% share of either; `audit_window_split_of_record`: the same for the per-call
split with y's floors, the sampling design the PoUW window runs (`PlanDraw.lean`). The product design assumes each call's sample is under a distinct
context.

**Theorems, two-stage** (`Audit/TwoStage.lean`):

- `twoStage_profile`, variant (b), nested sampling: the profile over the coarse partition with the effective acceptance
  probability
  $e_{\text{eff}}(B) = \mathbb E_{S_1}\big[\prod_{u \in B \cap S_1} \max_{v \subseteq u}(1 - \pi_{L_2 u}(v))\big]$, with
  $\varepsilon_{ks}$ and $\delta_{link}$ at the fine level;
- `effEscape_full`, `twoStage_full_profile`: **(b1) is one-stage over the coarse partition** ($e_{\text{eff}} = e_{L_1}$);
- `effEscape_bernoulli_prod`: with a Bernoulli first stage, $e_{\text{eff}}(B) = \prod_{u \in B}\big(1 - p + p\,m_u\big)$ for
  any second stage, $m_u$ the largest second-stage miss probability inside $u$;
- `Partition.two_stage_a_coarse`: **(a) is one-stage over the fine partition**, and its curve bounds coarse counts;
- `Partition.Refines.exists_wrong_fine`: a faulty coarse unit has a faulty fine unit under every interior, the lemma
  behind (b).
- `Refines.oracle₂_profile`, `oracle₂_full_profile`: the same at the oracle layer, from a sound fine-level session
  (`SessionSound₂ ε`), with the link term zero. These are the plan's `two_stage_b_profile` and
  `two_stage_full_eq_one_stage` as stated there.

**Composition** (`Audit/Circuit.lean`): `compose` (every unit correct and the inputs anchored give the circuit's
evaluation on every committed wire) and `compose_cone` (the same for a consumer's wires, from its cone alone).

**Influence through a separator** (`Audit/Influence.lean`). $D$ is the set of delivered wires (the served tokens); the
circuit's other outputs, such as commitments and leaves, are not delivered. $K$ **separates** the units $B$ from $D$
(`Separates K B D`, a downstream cut) when every gate path from a unit of $B$ to a wire of $D$ meets $K$.

- `outputs_eq_of_separates`, **the influence cap**: two traces with the anchors' inputs, whose faulty proof blocks lie in
  $B$, and that agree on a committed separator $K$ of $B$, deliver the same values on $D$;
- `card_delivered_le`: so they deliver at most $2^{|K|}$ values on $D$;
- `separates_exits`: the committed wires of the units of $B$ separate $B$ (a replay block's exports); `Separates.mono`,
  `Separates.union`;
- `card_influenceSet_le`: the influence set at $\delta$, what is delivered with the faulty proof blocks inside some set of
  acceptance probability at least $\delta$, fixed before the prover, has at most $\sum_{B : e(B) \ge \delta} 2^{|K_B|}$
  members;
- `audit_influence`: $\Pr[\text{accept} \wedge \mathrm{out}_D(X) \notin \mathrm{influenceSet}] \le \delta +
  \varepsilon_{ks} + \delta_{link} + \delta_{in}$, so the prover's influence over what is delivered is
  $\log_2 |\mathrm{influenceSet}|$ bits; `Refines.twoStage_influence` is the same over replay blocks, with $e_{\text{eff}}$
  and the two-stage input hypothesis `AnchorsSound₂` (`Audit/TwoStage.lean`).
- `card_influenceSet_le_harm` and `card_admissible_le`: that influence is at most $H + \log_2 \sum_{j \le K}
  \binom{n}{j}$ bits, for a harm bound $H$ (`IsHarmBound`, `Audit/Harm.lean`: the specification of core's `harm_bound`) with
  each unit's harm the width it adds to a separator, such as its exits (`card_exits_le`), and $\mathrm{miss}(K+1) <
  \delta$. $H$ counts the value bits and the second term the choice of which units are faulty.
- `audit_exfiltration`, **exfiltration with the location term**: a prover that picks its strategy from a uniform message
  $m \in M$, and a receiver that sees only what is delivered on $D$, recover $m$ from an accepted audit with probability
  at most $2^H \sum_{j \le K} \binom{n}{j} / |M| + \delta + \varepsilon_{ks} + \delta_{link} + \delta_{in}$; with
  $|M| = 2^\ell$, the first term is $2^{H + \log_2 \sum_{j \le K} \binom{n}{j} - \ell}$. `card_guess_le` is its counting
  step and `audit_exfiltration_influenceSet` its form over $|\mathrm{influenceSet}|$. Core's reading is
  `verity.core.protocols.sampled_proofs.consumers.exfiltration_bound`: `harm_bound` over output bits plus `location_bits`.
- **Not vacuous** (`Audit/InfluenceWitness.lean`, the statement reviewer's witness): on a four-gate circuit,
  `W.lemma47` applies the influence cap to two different traces that both have a faulty proof block, with a separator
  narrower than the unit's exits (`W.three_mem_exits`), and `W.count` bounds what they deliver by 2; `A.influence`
  meets `audit_influence`'s hypotheses at 0 on an audit that accepts with probability 1 (`A.accepts`); `Gen.harm_witness`
  meets `card_influenceSet_le_harm`'s on any partition and sampling design, with the exits as separators and each unit's
  exported width as its harm.
  `Two.influence` meets `Refines.twoStage_influence`'s, with `AnchorsSound₂`, on a two-stage audit that accepts with
  probability 1 (`Two.accepts`), and `Ex.exfil` meets `audit_exfiltration`'s eight hypotheses together, for a one-bit
  message.

**Tightness and insecurity** (`Audit/Examples.lean`, on `K` chains of `nv` NOT gates, one coarse unit per chain and one
fine unit per gate, with a uniform `k₂`-subset of each sampled chain's gates in the second stage and an ideal session):

- `two_stage_b_tight`: for every first-stage sampling design, a prover whose chains are all faulty is accepted with
  probability exactly `twoStage_count`'s bound, $\mathbb E_{S_1}[q^{|S_1|}]$ with
  $q = \binom{n_v - 1}{k_2}/\binom{n_v}{k_2} = 1 - k_2/n_v$;
- `naive_two_level_unsound`: if the second stage leaves a gate undrawn and the first samples anything, that exceeds
  $\mathrm{miss}_{L_1}(K)$, the bound that one-stage over the coarse partition (and the `m`-dependent reading of
  the retired Python `TwoStageLaw.profile`, at $m = n_v$) would give. Nested sampling dilutes;
- `late_interior_insecure`: with the interior committed after the fine sample (`twoStageLate`), a faulty chain is accepted
  with probability 1.

- `effEscape_bernoulli`: **(b2) is one-stage over the coarse partition, thinned.** With a Bernoulli($p$) first stage,
  the effective acceptance probability of any set of chains is its acceptance probability under the one-stage
  Bernoulli($p\,k_2/n_v$) sample, $(1 - p\,k_2/n_v)^{|B|}$ (`Law.bernoulli_escape`, `Law.bernoulli_avg_prod`).

**The Flock instantiation, oracle layer** (`Audit/Flock.lean`, the only audit file that imports Flock's theorems):

- `value_batch_le`: in a batched session, whose tables run concurrently on a schedule the prover chooses, every table
  accepting is at most as likely as any one table accepting (`Game.value_interleave_le`, table by table);
- `FlockTable.value_le`: one table, with the executable's arithmetic, at most `2^-205` for `22 ≤ m ≤ 35`;
- `flock_session_sound`: Flock's batched session is `SessionSound` at `2^-205`, given `LoweringSound`;
- `flock_inputs_sound`: the input unit, likewise, given `InputsLoweringSound`;
- `flock_oracle_count`, `flock_oracle_count_subset`: the audit's count curve, $\delta(K) = \mathrm{miss}_L(K) + 2^{-205}$,
  and $\binom{n-K}{k}/\binom{n}{k} + 2^{-205}$ for the uniform $k$-subset;
- `flock_oracle_drawn`: sampled proof blocks correct up to $2^{-205}$;
- `flock_oracle_cone`: a consumer's wires $T$, $\max_{u \in \mathrm{cone}(T)}(1 - \pi_u) + 2 \cdot 2^{-205}$.

- `flock_two_stage_sound`, `flock_two_stage_full_profile`: **two-stage (b1) on Flock** is one-stage over the coarse
  partition, $\sup_{B \in \mathcal B} e_{L_1}(B) + 2^{-205}$, given the fine partition's lowering.

The hypotheses are the lowerings.

**The compiled layer** (`Audit/Extraction.lean`, `Audit/FlockCompiled.lean`, `Audit/FlockBatched.lean`; `../DESIGN.md`
§12):

- `extraction_audit_le`, `extraction_audit_count`, `extraction_audit_drawn`: the one-stage audit with a **randomized
  extractor**. Per sampled proof block, `Pr[accept ∧ extraction fails]` and `Pr[accept ∧ extraction disagrees]` are joint
  probabilities over the fresh run and the extractor's reruns (`ExtractionAnalysis`), and the named bounds may depend
  on the prover's state. A deterministic extractor is the special case (`Analysis.toExtraction`, with
  `KnowledgeSound.toExtraction` and `LinkSound.toExtraction`);
- `flock_batched_knowledgeSound`: **ε_ks is discharged** for the compiled Flock audit in the protocol's batched session
  of several tables (`batchedSession`, `sessionB`), with the leaf layer registered. Each sampled proof block is read from one
  table of the plan (`tab`). The knowledge term is, after each sample, the largest over the plan's tables of the table's
  $\varepsilon_c^-(\sigma) + K\,\mathrm{Adv}_0(\sigma) + N_0/(eK)$ computed in the session (`joint_le_B'`, from
  `session_knowledge_sound`), averaged over the sample (`ksAvgB`);
- `flock_compiled_knowledgeSound`: the same for one table on its own (`sessionC`), from
  `table_knowledge_sound_joint_tight`: $\mathbb E_\omega[\varepsilon_c^-(\sigma_\omega) + K\,\mathrm{Adv}_0(\sigma_\omega)
  + N_0/(eK)]$, which is $\varepsilon_c^- + t\cdot2^{-244.7}$ at the best $K$ under the generic collision bound. Neither
  has a named assumption;
- `flock_batched_count`, `flock_batched_drawn` (and `flock_compiled_count`, `flock_compiled_drawn`):
  $\delta(K) = \mathrm{miss}_L(K) + \varepsilon_{ks}(\sigma) + \delta_{link}(\sigma)$, and sampled proof blocks up to
  $\varepsilon_{ks}(\sigma) + \delta_{link}(\sigma)$.

**The audit's circuit `C`** is abstract in these theorems. For a program it is the rows the verifier parses
(`Prog.circuit`, `Lowering.lean`, `../README.md` §1.5): each unit an instance of its template's `Rows`, one `Op.row` gate per
computed row. No hypothesis relates those rows to another circuit (the circuit is chosen on the untrusted side,
and the lowering does no optimizing); what a correct unit computes is its certificate's (`IsRowsUnit.computes_of_cert`,
`../Certificate.lean`; `../ASSUMPTIONS.md`). The decoder is **per sampled proof block**: two sampled proof blocks that share a wire hold it in their own columns, and only
the link makes the readings agree.

The compiled layer's named hypotheses are stated there too.

**Not yet:** the serving-tree term when the leaf layer is not registered; `ValueBinding` and `Placement` for the
executable's statements.
