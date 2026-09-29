# The audit layer (`Audit/`): what is proved

The sampled audit's guarantee, the integrity profile of `docs/audit-protocols.md` §2, over any Boolean circuit, any
partition, any draw law, and any per-unit proof system. The derivations and the deviations from that plan are in
`../../DESIGN.md` §12. Everything here is proved; the per-unit proof system enters only through named hypotheses.

**The model.** The prover registers `R` (its roots); the verifier draws units from its own randomness after registration;
one batched session proves the drawn units (`audit`). The **committed transcript** `X` is an analysis function of the
prover's state at registration (its registration and its session strategy for every draw), so the draw is independent of
its wrong units; nobody computes it. A unit is **wrong** in `X` when its committed outputs differ from its gates applied
to its committed inputs (`Partition.Correct`). Every bound holds for every prover strategy (`Game.prob … σ`), the form
`table_sound_exec` takes.

**Named hypotheses** (`Prop`s taken as hypotheses; no `axiom`): each is stated in `../../ASSUMPTIONS.md`.

**What the reviewer checks here.** The theorems hold for every choice of committed transcript `X` and extractor `ext`,
so their meaning rests on `X` being the committed values. At the oracle layer `X` is the registration itself
(`Partition.oracle`). At the compiled layer it is the plurality of what the extractor recovers behind each commit
string, which the instantiation defines and `LinkSound` ties to the extractor. A consumer who opens values under the
roots gets `X`'s values up to that opening's own binding term, which the audit does not count.

**Theorems, one-stage** (`Audit/OneStage.lean`), with $e_L(B)$ the probability that the draw misses $B$:

- `audit_le`: $\Pr[\text{accept} \wedge E(X, S)] \le \Pr_S[E(X, S) \wedge S \cap \mathrm{wrong}(X) = \emptyset] +
  \varepsilon_{ks} + \delta_{link}$ for every event $E$ on the committed transcript and the draw;
- `audit_profile`: $\Pr[\text{accept} \wedge \mathrm{wrong}(X) \in \mathcal B] \le \sup_{B \in \mathcal B} e_L(B) +
  \varepsilon_{ks} + \delta_{link}$;
- `audit_count`: $\delta(K) = \mathrm{miss}_L(K) + \varepsilon_{ks} + \delta_{link}$;
- `audit_drawn`: drawn units are correct up to $\varepsilon_{ks} + \delta_{link}$, even for a consumer who looks after
  the draw;
- `audits_seq`: audits in sequence fail with at most the sum of their bounds, each holding from every history. Nested,
  this is the lifetime union bound, with every term against a prover that took part in all the audits;
- `Partition.audit_cone`: for committed wires $T$ fixed before the draw, $\Pr[\text{accept} \wedge X_T \ne
  \mathrm{eval}_C(a)_T] \le \max_{u \in \mathrm{cone}(T)} (1 - \pi_u) + \varepsilon_{ks} + \delta_{link} + \delta_{in}$;
- `Partition.audit_cone_count`: at least $j$ wrong units in a cone;
- `Partition.audit_covered`: a cone inside the draw, for wires chosen even after the draw: $\varepsilon_{ks} +
  \delta_{link} + \delta_{in}$;
- `Partition.oracle_count`, `oracle_profile`, `oracle_drawn`, `oracle_cone`: the same at the oracle layer, from
  `SessionSound ε` and `InputsSound δ_in`, with $\delta_{link} = 0$.

**Theorems, laws** (`Audit/Law.lean`): `bernoulli_escape`, a Bernoulli($p$) draw's escape $(1-p)^{|B|}$; `subset_escape`, the uniform $k$-subset's escape
$\binom{n-|B|}{k}/\binom{n}{k}$; `subset_miss`, its count curve $\binom{n-K}{k}/\binom{n}{k}$; `subset_escape_le_prod`, the product bound $e(B) \le \prod_{u \in B}(1 - \pi_u) = (1 -
k/n)^{|B|}$, so inclusion probabilities alone give a consumer a valid bound; `subset_minimax`, **uniform is
optimal**: no law that proves $k$ units on average has a smaller count curve, at any $K$; `demo_count_590`, the demo's
point (1,024 heads, 16 drawn): 590 or more wrong heads escape with probability at most $2^{-20}$, by exact arithmetic.

**Theorems, the stratified law** (`Audit/Stratified.lean`, the default draw law; `docs/sampling-strategies.md` §4):
`Law.stratified σ k`, an independent uniform $k_s$-subset inside each stratum of $\sigma : \mathrm{Fin}\,n \to \mathrm{Fin}\,m$;
`stratified_escape`, the product of hypergeometrics $\prod_s \binom{n_s - m_s}{k_s}/\binom{n_s}{k_s}$ with $m_s = |B \cap U_s|$,
which is what `verity.proofs.profile.Stratified` computes; `stratified_escape_floor`, **a floor catches its whole
stratum**: with $k_s \ge 1$, a set containing $U_s$ never escapes; `audit_whole_stratum` and
`extraction_audit_whole_stratum`, the one-stage profile instantiated, so an audit accepts while a whole floored stratum is
wrong with probability at most the per-unit terms.

**Theorems, the work law** (`Audit/Work.lean`, the draw law's second version; `../../PROTOCOL.md` §7.3): `Law.work σ w f K`
is `Law.stratified` with stratum $s$ drawing $k_s = \min(n_s, \max(f_s, \lceil K w_s n_s / W \rceil))$, where $w_s$ is the
PoUW work of each of its units, $W = \sum_s w_s n_s$ and $f_s \ge 1$ its floor, set per stratum in the verifier's work
table; every stratified and one-stage theorem holds of it as stated, and the theorems below hold for every such $f$.
`floor_le_workK`, **the floor**: a stratum draws at least $\min(n_s, f_s)$ units, so every non-empty stratum draws at least
one and `audit_work_whole_stratum` and `extraction_audit_work_whole_stratum` catch a wholly wrong one.
`work_escape_floor_le`, **the per-stratum floor escape**: a set with $b_s$ units in a non-empty stratum escapes with at most
$((n_s - b_s)/n_s)^{f_s}$, audited at both layers (`audit_work_floor`, `extraction_audit_work_floor`). `workK_ge`: a
stratum draws at least $K w_s n_s / W$ units, or all of them. `sum_workK_le`: at most $K + \sum_s f_s$ units are drawn.
`workK_rest`: a stratum that does no work draws $\min(n_s, f_s)$. `work_escape_le_count`: when those strata have floors of
at least the stratified law's sizes at $C$ (`countK`, which `countK_eq` shows is `stratumK C n_s N`), a set escapes the
work law at most as often as its units in them escape the stratified law at $C$ (`choose_div_anti`: the hypergeometric
escape falls as the draw grows), and `audit_work_rest` and `extraction_audit_work_rest` carry any such integrity profile to
the audit. `work_escape_le`,
**the work bound**: a set $B$ of units doing $W_B$ of the work escapes with probability at most $((W - W_B)/W)^K$, from
the per-stratum hypergeometric and weighted AM-GM over the work shares. `audit_work` and `extraction_audit_work` bound
$\Pr[\text{accept} \wedge W_{\mathrm{wrong}(X)} \ge T]$ by $((W - T)/W)^K$ plus the per-unit terms. `record_sizing` and
`audit_work_of_record`: at $K = 27{,}713$, wrong units doing at least 0.1% of the work escape with at most $2^{-40}$;
`record_sizing_one_fewer`: $K = 27{,}712$ does not reach it under this bound. `Law.widen` and `escape_widen_le`, the seam for
closure draws: proving more units than are drawn, on the same coins, never raises an escape. `FlockWork.lean`'s
`workRule_eq_draw` and `countRule_eq_draw`: the executable's `Flock.Draw.workK` and `stratumK` are `Law.workRule` and
`countRule`.

**Theorems, closure draws** (`Audit/Closure.lean`; `../../PROTOCOL.md` §7.3): `Law.closure L cl` is `L` widened on its coins
by the verifier's closure map, each drawn unit proved with its closure. `closure_escape`: its escape is the base law's
escape of $B \cup \mathrm{unsound}(B)$, $\mathrm{unsound}(B)$ being the units whose closure meets $B$, so a wrong unit that is
never drawn on its own is charged to the units above it. `unsoundWork` is the work of $B \cup \mathrm{unsound}(B)$: a closure
map need not list a unit in its own closure, so the wrong units' own work is counted explicitly (`workOf_le_unsoundWork`).
`harm_le_unsoundWork`: a
wrong unit's harm (its own work and that of the units whose closure holds it) is in the unsound work, drawn or not;
`unsoundWork_le_harm` (coverage) and `unsoundWork_singleton`. `audit_work_closure` and `extraction_audit_work_closure`:
under the work law's closure draw, at any floors $f_s \ge 1$,
$\Pr[\text{accept} \wedge \mathrm{work}(B \cup \mathrm{unsound}(B)) \ge T] \le ((W - T)/W)^K$ plus the per-unit terms, for
$B = \mathrm{wrong}(X)$; `audit_work_closure_harm`: likewise for a wrong unit's harm; `audit_work_closure_of_record`:
$2^{-40}$ at $K = 27{,}713$ for unsound work of at least 0.1%.

**Theorems, two-stage** (`Audit/TwoStage.lean`):

- `twoStage_profile`, variant (b), nested sampling: the profile over the coarse partition with the effective escape
  $e_{\text{eff}}(B) = \mathbb E_{S_1}\big[\prod_{u \in B \cap S_1} \max_{v \subseteq u}(1 - \pi_{L_2 u}(v))\big]$, with
  $\varepsilon_{ks}$ and $\delta_{link}$ at the fine level;
- `effEscape_full`, `twoStage_full_profile`: **(b1) is one-stage over the coarse partition** ($e_{\text{eff}} = e_{L_1}$);
- `effEscape_bernoulli_prod`: with a Bernoulli first stage, $e_{\text{eff}}(B) = \prod_{u \in B}\big(1 - p + p\,m_u\big)$ for
  any second stage, $m_u$ the largest second-stage miss probability inside $u$;
- `Partition.two_stage_a_coarse`: **(a) is one-stage over the fine partition**, and its curve bounds coarse counts;
- `Partition.Refines.exists_wrong_fine`: a wrong coarse unit has a wrong fine unit under every interior, the lemma
  behind (b).
- `Refines.oracle₂_profile`, `oracle₂_full_profile`: the same at the oracle layer, from a sound fine-level session
  (`SessionSound₂ ε`), with the link term zero. These are the plan's `two_stage_b_profile` and
  `two_stage_full_eq_one_stage` as stated there.

**Composition** (`Audit/Circuit.lean`): `compose` (every unit correct and the inputs anchored give the circuit's
evaluation on every committed wire) and `compose_cone` (the same for a consumer's wires, from its cone alone).

**Influence through a separator** (`Audit/Influence.lean`). $D$ is the set of delivered wires (the served tokens); the
circuit's other outputs, such as commitments and leaves, are not delivered. $K$ **separates** the units $B$ from $D$
(`Separates K B D`, a downstream cut) when every gate path from a unit of $B$ to a wire of $D$ meets $K$.

- `outputs_eq_of_separates`, **the influence cap**: two transcripts with the anchors' inputs, whose wrong units lie in
  $B$, and that agree on a committed separator $K$ of $B$, deliver the same values on $D$;
- `card_delivered_le`: so they deliver at most $2^{|K|}$ values on $D$;
- `separates_exits`: the committed wires of the units of $B$ separate $B$ (a replay unit's exports); `Separates.mono`,
  `Separates.union`;
- `card_influenceSet_le`: the influence set at $\delta$, what is delivered with the wrong units of some set escaping with
  probability at least $\delta$, fixed before the prover, has at most $\sum_{B : e(B) \ge \delta} 2^{|K_B|}$ members;
- `audit_influence`: $\Pr[\text{accept} \wedge \mathrm{out}_D(X) \notin \mathrm{influenceSet}] \le \delta +
  \varepsilon_{ks} + \delta_{link} + \delta_{in}$, so the prover's influence over what is delivered is
  $\log_2 |\mathrm{influenceSet}|$ bits; `Refines.twoStage_influence` is the same over replay units, with $e_{\text{eff}}$
  and the two-stage input hypothesis `AnchorsSound₂` (`Audit/TwoStage.lean`).
- `card_influenceSet_le_harm` and `card_admissible_le`: that influence is at most $H + \log_2 \sum_{j \le K}
  \binom{n}{j}$ bits, for a harm bound $H$ (`IsHarmBound`, `Audit/Harm.lean`: the specification of core's `harm_bound`) with
  each unit's harm the width it adds to a separator, such as its exits (`card_exits_le`), and $\mathrm{miss}(K+1) <
  \delta$. $H$ counts the value bits and the second term the choice of which units are wrong.
- `audit_exfiltration`, **exfiltration with the location term**: a prover that picks its strategy from a uniform message
  $m \in M$, and a receiver that sees only what is delivered on $D$, recover $m$ from an accepted audit with probability
  at most $2^H \sum_{j \le K} \binom{n}{j} / |M| + \delta + \varepsilon_{ks} + \delta_{link} + \delta_{in}$; with
  $|M| = 2^\ell$, the first term is $2^{H + \log_2 \sum_{j \le K} \binom{n}{j} - \ell}$. `card_guess_le` is its counting
  step and `audit_exfiltration_influenceSet` its form over $|\mathrm{influenceSet}|$. Core's reading is
  `verity_one_stage.consumers.exfiltration_bound`: `harm_bound` over output bits plus `location_bits`.
- **Not vacuous** (`Audit/InfluenceWitness.lean`, the statement reviewer's witness): on a four-gate circuit,
  `W.lemma47` applies the influence cap to two different transcripts that both have a wrong unit, with a separator
  narrower than the unit's exits (`W.three_mem_exits`), and `W.count` bounds what they deliver by 2; `A.influence`
  meets `audit_influence`'s hypotheses at 0 on an audit that accepts with probability 1 (`A.accepts`); `Gen.harm_witness`
  meets `card_influenceSet_le_harm`'s on any partition and law, with the exits as separators and each unit's exported
  width as its harm.

**Tightness and insecurity** (`Audit/Examples.lean`, on `K` chains of `nv` NOT gates, one coarse unit per chain and one
fine unit per gate, with a uniform `k₂`-subset of each drawn chain's gates in the second stage and an ideal session):

- `two_stage_b_tight`: for every first-stage law, a prover whose chains are all wrong is accepted with probability
  exactly `twoStage_count`'s bound, $\mathbb E_{S_1}[q^{|S_1|}]$ with $q = \binom{n_v - 1}{k_2}/\binom{n_v}{k_2} = 1 -
  k_2/n_v$;
- `naive_two_level_unsound`: if the second stage leaves a gate undrawn and the first draws anything, that exceeds
  $\mathrm{miss}_{L_1}(K)$, the bound that one-stage over the coarse partition (and the `m`-dependent reading of
  `TwoStageLaw.profile` at $m = n_v$) would give. Nested sampling dilutes;
- `late_interior_insecure`: with the interior committed after the fine draw (`twoStageLate`), a wrong chain is accepted
  with probability 1.

- `effEscape_bernoulli`: **(b2) is one-stage over the coarse partition, thinned.** With a Bernoulli($p$) first stage,
  the effective escape of any set of chains is its escape under the one-stage Bernoulli($p\,k_2/n_v$) draw,
  $(1 - p\,k_2/n_v)^{|B|}$ (`Law.bernoulli_escape`, `Law.bernoulli_avg_prod`).

**The Flock instantiation, oracle layer** (`Audit/Flock.lean`, the only audit file that imports Flock's theorems):

- `value_batch_le`: in a batched session, whose tables run concurrently on a schedule the prover chooses, every table
  accepting is at most as likely as any one table accepting (`Game.value_interleave_le`, table by table);
- `FlockTable.value_le`: one table, with the executable's arithmetic, at most `2^-205` for `22 ≤ m ≤ 35`;
- `flock_session_sound`: Flock's batched session is `SessionSound` at `2^-205`, given `LoweringSound`;
- `flock_inputs_sound`: the input unit, likewise, given `InputsLoweringSound`;
- `flock_oracle_count`, `flock_oracle_count_subset`: the audit's count curve, $\delta(K) = \mathrm{miss}_L(K) + 2^{-205}$,
  and $\binom{n-K}{k}/\binom{n}{k} + 2^{-205}$ for the uniform $k$-subset;
- `flock_oracle_drawn`: drawn units correct up to $2^{-205}$;
- `flock_oracle_cone`: a consumer's wires $T$, $\max_{u \in \mathrm{cone}(T)}(1 - \pi_u) + 2 \cdot 2^{-205}$.

- `flock_two_stage_sound`, `flock_two_stage_full_profile`: **two-stage (b1) on Flock** is one-stage over the coarse
  partition, $\sup_{B \in \mathcal B} e_{L_1}(B) + 2^{-205}$, given the fine partition's lowering.

The hypotheses are the lowerings.

**The compiled layer** (`Audit/Extraction.lean`, `Audit/FlockCompiled.lean`, `Audit/FlockBatched.lean`; `../../DESIGN.md`
§12):

- `extraction_audit_le`, `extraction_audit_count`, `extraction_audit_drawn`: the one-stage audit with a **randomized
  extractor**. Per drawn unit, `Pr[accept ∧ extraction fails]` and `Pr[accept ∧ extraction disagrees]` are joint
  probabilities over the fresh run and the extractor's reruns (`ExtractionAnalysis`), and the named bounds may depend
  on the prover's state. A deterministic extractor is the special case (`Analysis.toExtraction`, with
  `KnowledgeSound.toExtraction` and `LinkSound.toExtraction`);
- `flock_batched_knowledgeSound`: **ε_ks is discharged** for the compiled Flock audit in the protocol's batched session
  of several tables (`batchedSession`, `sessionB`), with the leaf layer registered. Each drawn unit is read from one
  table of the plan (`tab`). The knowledge term is, after each draw, the largest over the plan's tables of the table's
  $\varepsilon_c^-(\sigma) + K\,\mathrm{Adv}_0(\sigma) + N_0/(eK)$ computed in the session (`joint_le_B'`, from
  `session_knowledge_sound`), averaged over the draw (`ksAvgB`);
- `flock_compiled_knowledgeSound`: the same for one table on its own (`sessionC`), from
  `table_knowledge_sound_joint_tight`: $\mathbb E_\omega[\varepsilon_c^-(\sigma_\omega) + K\,\mathrm{Adv}_0(\sigma_\omega)
  + N_0/(eK)]$, which is $\varepsilon_c^- + t\cdot2^{-244.7}$ at the best $K$ under the generic collision bound. Neither
  has a named assumption;
- `flock_batched_count`, `flock_batched_drawn` (and `flock_compiled_count`, `flock_compiled_drawn`):
  $\delta(K) = \mathrm{miss}_L(K) + \varepsilon_{ks}(\sigma) + \delta_{link}(\sigma)$, and drawn units up to
  $\varepsilon_{ks}(\sigma) + \delta_{link}(\sigma)$.

**The audit's circuit `C`** is abstract in these theorems. For a program it is the rows the verifier parses
(`Prog.circuit`, `Lowering.lean`, `../../README.md` §1.5): each unit an instance of its template's `Rows`, one `Op.row` gate per
computed row. "Each template's rows compute its gates" in the program's Boolean circuit is L1 (`../../ASSUMPTIONS.md`), named and
tested. The decoder is **per drawn unit**: two drawn units that share a wire hold it in their own columns, and only
the link makes the readings agree.

The compiled layer's named hypotheses are stated there too.

**Not yet:** the serving-tree term when the leaf layer is not registered; `ValueBinding` and `Placement` for the
executable's statements.
