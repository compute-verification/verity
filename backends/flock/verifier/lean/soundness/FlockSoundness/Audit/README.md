# The audit layer (`Audit/`): what is proved

The sampled audit's guarantee, the integrity profile of `docs/audit-protocols.md` §2, over any Boolean circuit, any
partition, any draw law, and any per-unit proof system. The derivations and the deviations from that plan are in
`../../DESIGN.md` §12. Everything here is proved; the per-unit proof system enters only through named hypotheses,
which `../../ASSUMPTIONS.md` indexes with the rest.

**The model.** The prover registers `R` (its roots); the verifier draws units from its own randomness after registration;
one batched session proves the drawn units (`audit`). The **committed transcript** `X` is an analysis function of the
prover's state at registration (its registration and its session strategy for every draw), so the draw is independent of
its wrong units; nobody computes it. A unit is **wrong** in `X` when its committed outputs differ from its gates applied
to its committed inputs (`Partition.Correct`). Every bound holds for every prover strategy (`Game.prob … σ`), the form
`table_sound_exec` takes.

**Named hypotheses** (`Prop`s taken as hypotheses; no `axiom`):

| Hypothesis | Term | Meaning | To be discharged by |
|---|---|---|---|
| `Analysis.KnowledgeSound` | `ε_ks` | the session accepts while extraction of a drawn unit (picked before the session's first coin) fails | `table_knowledge_sound_joint` (`../../README.md` §1.3), after Jensen over the prover's states |
| `Analysis.LinkSound` | `δ_link` | extraction succeeds but differs from `X` on the unit's committed inputs and outputs | compiled layer: `flock_batched_linkSoundE` (A2, `ValueBinding`) |
| `Partition.AnchorsSound` | `δ_in` | the session accepts while the committed inputs differ from the anchors' values | `registered_weights` (`../../README.md` §1.3); at the oracle layer, `InputsSound` (`oracle_anchorsSound`) |
| `Partition.InputsSound` | `δ_in` | oracle layer: the session accepts inputs other than the anchors' with probability at most `δ_in` | **proved for Flock** (`flock_inputs_sound`, `2^-205`), given `InputsLoweringSound` |
| `Partition.InputsLoweringSound` | — | inputs other than the anchors' leave some session table (the always-proved input statement) unsatisfied | the input unit's lowering (level 3) |
| `Partition.SessionSound` | `ε` | oracle layer only: a session with a drawn wrong unit accepts with probability at most `ε` | **proved for Flock** (`flock_session_sound`, `2^-205`), from `table_value_sound`, `execArith_correct` and the lowering |
| `Partition.LoweringSound` | — | a drawn wrong unit lies in a session table whose statement no committed table satisfies | the lowering theorem (level 3, another lane) |

Both per-unit hypotheses are averages over the whole audit, for every prover and every rule that picks a drawn unit
before the session's first coin, not bounds state by state.

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
computed row. "Each template's rows compute its gates" in the program's Boolean circuit is L1 (`../../assumptions/l1-template-rows.md`), named and
tested. The decoder is **per drawn unit**: two drawn units that share a wire hold it in their own columns, and only
the link makes the readings agree.

| Named at the compiled layer | Meaning | Discharged by |
|---|---|---|
| `ExtractionAnalysis.LinkSound` (`analysisB`, `analysisC`, `analysisBE`) | `δ_link`: extraction succeeds, but every close satisfying candidate decodes, on the unit's committed wires, to values other than the committed transcript's | for `analysisBE`: `flock_batched_linkSoundE`, under A2 for each commit string's finder and `ValueBinding` (named; the `hm96` layout discharges it); composed in `flock_batched_count_linked`, `_drawn_linked` (`_placed`: from `UnitPlace`) |
| `Partition.LoweringSoundB`, `Partition.LoweringSoundC` | a message packing a witness that satisfies the statement of a drawn unit's table decodes, for that unit, to values on which it is correct | `loweringSoundB_of_place`, `loweringSoundC_of_place` from each drawn unit's `UnitPlace` (`lowering_sound`, `../../README.md` §1.5), given the statement builder's `Placement` (level 3) and L1 |

**Not yet:** the serving-tree term when the leaf layer is not registered; `ValueBinding` and `Placement` for the
executable's statements.
