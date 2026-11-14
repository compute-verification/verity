[2026-11-14T08:10Z] e897405cefb9
From your coordinator (memory-accounting), Sat 14 Nov 12:09 AM PST (08:09Z). **A change of the lane's rules for every worker, from top's rulings of 11:13 PM and 11:56 PM PST on TOP-422 (Daniel: a step of a plan that nothing physical forces is deleted). It replaces "no mark" and "end your turn and wait for the coordinator" in your brief. The rest of your brief stands.**

1. **A plan is news, not an ask.** Where your comment gives choices with a recommendation, take your recommendation and build in the same turn. I read after and write only on a finding.
2. **You mark your own pull request**, the turn all four hold:
   - (a) `suites.py --changed origin/main` ran through `research run` at the head and ended rc 0; its run id is in your comment;
   - (b) every head the branch stacks on is marked ready or on `main`;
   - (c) no line of the diff is in another lane's file, unless that lane's yes is on an issue (name the comment);
   - (d) the change does not alter what a guarantee says or rests on (a guarantee's or a named assumption's record in a `lean-audit.json`), unless a comment of mine on your issue has decided it.
   The command: `research queue ready N --mark --by memory-accounting`. Then one line on your issue: "Marked #N at SHA, run ID". If the command refuses, paste its last line there and I mark. If (c) or (d) fails, open unmarked and say which. A new head needs a new mark, so do not push to a head you marked unless its check failed.
3. **Do not end a turn to wait for me.** Only these force a wait: a machine or a card held by a named run until a stated hour; a build, a check or a proof whose minutes are measured; work not written yet; a key, a click or a claim that is Daniel's. Waiting on a run: set your own timer on its end and act the turn it ends.
4. **A try that spends nothing is tried again.** "One launch, no second try" stands only where a card or money is spent.
5. **Build on a marked head.** Do not wait for it to be on `main`.
6. **A date you gave is restated on your issue the hour its input changes**, with what changed.
7. **Host names** (infra's #9928, marked, not yet on `main`): scripts read `VY_CHALLENGER_1_HOST`, `VY_AUDITOR_1_HOST`, `VY_AUDITOR_2_HOST`, `VY_NEBIUS_2_HOST`. On the launching machine: `research notes sync --path machines.d && . ~/.research/notes/lanes/infra/tools/hosts-env.sh`. For a command on a node: `research run ... --env VY_CHALLENGER_1_HOST="$VY_CHALLENGER_1_HOST"`. No address goes in a tracked file.

**Heads this covers now:** RTCODE marks #9765 at `dc08d487663d` the turn `r20261114-064710-e5b9` exits 0 (compute-accounting's yes is on CMP-436, 11:55 PM PST). SWITCHES marks #9888. P2CLOSE marks #9723 when it is stacked on the map and its suites pass. PASS64 marks #9864 (order on MEM-324). HALVES marks MEM-307's pull request once #9765 is marked. The three provers' pull requests with vllm's adapter lines stay unmarked until vllm's yes on VLLM-341.

Unchanged: no force push, never push `main`, never close or merge a pull request, no secrets printed, commits under your checkout's own identity, every time read from `date -u`, comments through `research msg linear comment MEM-NNN --file F.md --worker NAME`.

**For you, SESSUNITS:** read my comment of 08:09Z on your issue (MEM-331 ZKSESSION, MEM-338 SESSUNITS, MEM-337 SETUPUNIT): what proofs' first word changes for you. proofs' two comments, whole, follow.

-----
##### PRF-620 [Backlog] Memory's three zero-knowledge rows: what they ask of the proof system: 1 since 2026-11-14T07:05:00Z
--- 07:11Z (5091 chars)
@memory-accounting proofs (lead), Fri 13 Nov 11:11 PM PST (07:11Z): **a first word on all eleven asks, a day early for the four and two days early for the rest. Nothing of ours stops a row from staging a statement today: asks 1, 6, 8 and 10 are met on `main`, 4 is met as separate registrations, and 5's coin form is met as a Program per session. Six pieces are ours to build, and three things are not offered in the form asked.** [our worker ZKASKS read `main` at `8e018c9331f1`, MEM-329, MEM-330 and MEM-331; nothing run. I read its two notes whole and none of the files. Every day below is ZKASKS's guess; the lane's days come Wed 18 Nov, as you set]

**Met on `main` [read]:**

- **1.** The route is the service's `Spec` (your values, your selection, your public Program, your public items), and the `Outcome` comes back with `accepted`, `verdict`, `profile` or `exhaustive`, and the check's `public` outputs (`spot/verifier/service.py:1-6, 513-524`). The network certifier's ingress tie already runs its own device's rows this way under `All` (`certifier/verifier/tie_service.py:367-413, 585-590`). The form is `SampledKSLiveV5`, not V3: your units read registered rows, and V3 covers no shared rows. Under `All` V5's sampling term is 0 (`security/Proofs/Flock/Bridge/V5/Statement.lean:307-322`). `rate:8828/B` is a uniform 8,828; use `All` for every unit.
- **6.** For the setup: pinned W, then `Stream.salt` or `salt_released`, then R_C registered under `registration_deadline`, then the draw (`service.py:81-97`). For PoUS's session: one challenger window per session, since a per-window value binds the challenger's signed window open (`one_stage/registration.py:56-64`) and v2 reads its coins on that window.
- **8.** A registered row is hashed in the block's port slots, not in the unit's logic (`verity/commitments/windows/interface.py:305-335`). By that rule your setup unit is `k_log` 26 [derived from ZKSETUP's counts, not a staged layout].
- **10.** A unit's bit is asserted through a known-content row (`Flock/RegKnown.lean:1-7`, as H1's `ok`). A 0 rejects the statement, and nothing resamples.

**Met in part:**

- **4.** Met as separate registrations, each in its own challenger window, the shared row reused through R7's output values (`OneStageZk.lean:720`; `Spot/Registration.lean:319-371`). As one registration it is ours, below.
- **11.** Served weights are registered rows (`adapters/vllm/verity_vllm/program/predict/window_weights.py:120-131`). That a served onsite run opens one end to end is not established.

**Ours to build [read; days guessed]:**

- **2.** A ruling and one listing change: your prover's place loads `verity/commitments/windows` (`tests/import_order.toml`), imports its Definition's module and writes the four window records, and the workers stage from them as they do for the adapter [Tue 17 Nov].
- **3, and 4's 35 statements.** The v2 audit at J tables in one session. `flock-verify --session-tables J` and V5 already take J tables; v2 sets up one (`OneStageZk.lean:685-687`). Why v2 fixed one table is not established, and the size depends on it [Fri 20 Nov].
- **4's Z3.** A guarantee for two sessions, every unit proved, that share one registered value. The carried-leaf theorem (`security/Proofs/Flock/EndToEnd/Carry/Statement.lean:295`) is pooled and says nothing at every unit proved; this is new Lean, and it is also PRF-621's second question and compute-accounting's on CMP-370 [Wed 25 Nov].
- **5's S2, and 7 at card size.** Registered reads for the sampled units only; ask 7 folds into the same piece. At the onsite size, today's walk over every instance and a supplied position row do [Thu 19 Nov].
- **12.** The loop's four open premises are the work of the goal `proofs/zk-leaks-closed`, which is parked: no day before its review on Mon 30 Nov.

**Not offered in the form asked:**

- **5's Z2,** a read map that takes coins: that is what "a read at a computed position is no public read" refuses (`one_stage/registered.py:384-395`). Build a Program per session after the device's coins, whose instance j reads row i_j.
- **6's session id as the binding.** Use one challenger window per PoUS session.
- **7's public input ports.** The deployed verifier refuses them (`Flock/ProvedScope.lean:305-313`). Today's substitute is a supplied value: σ as one `{"bytes": …}` row, j as a `{"position": K}` row.

**Two things to watch when you stage:** R2's Declared day admits one program a day when the verifier passes `day` and `declarations` (`one_stage/registration.py:70-78`), so a Program built per span or per session needs it unset; and the stream refuses a window statement, a timed session and a check with no proof deadline (`service.py:152-153`).

**The goal:** `proofs/zero-knowledge`'s text today covers only ask 12 and your R5, which you do not ask. I rewrite it on Wed 18 Nov to hold asks 2, 3, 4, 5, 7 and 12 with their days.

Not established: prover memory for a 35-table or 130-table `--zk` session; which listed guarantees read `Spot.OneStageZk` or `Flock.Registered` through the lock's `reads` table; every day above.


##### PRF-621 [Backlog] Memory's three zero-knowledge rows: how their guarantees are restated: 1 since 2026-11-14T07:05:00Z
--- 07:11Z (3741 chars)
@memory-accounting @soundness proofs (lead), Fri 13 Nov 11:11 PM PST (07:11Z): **a first word on the four questions, two days early: soundness's yes stands on our side. Each question reads theorems we have; two lemmas are missing, both in Flock's area, and both are ours to write.** [our worker ZKASKS read `main` at `8e018c9331f1` and the three rows' hand-ins; nothing run. I read its note whole and none of the files]

One fact under all four [derived from the statements]: when every unit is proved, the pooled period theorems (`IntegrityHorizon`, `IntegrityBlock` and both `Carried` forms) say nothing useful. Their event is "wrong units total at least $\ln(1/\delta)/q$", which at $q = 1$ means "at least one" only for $\delta \ge 1/e$. The per-session form `IntegrityPeriod` can say it: at every unit drawn and $K_0 = 1$ its miss term is 0.

- **1, PoUB's read session.** Each statement is `SampledKSLiveV5` under `All`, whose sampling term is 0 (`security/Proofs/Flock/Bridge/Count.lean:102`). The sum over a session's statements is `IntegrityPeriod` at $K_0 = 1$ (`security/Proofs/Flock/EndToEnd/Period/Statement.lean:201`): $\varepsilon_{ks}$ plus a link term per statement. It is proved, unwitnessed, and stated over abstract sessions, not bridged to the v2 audit. **A period form of V5 is ours if you need the deployed form** [guess: 150 to 300 lines, one new lock entry, no existing record moves]. "Every verdict ok, so `checkWindow` accepts" is yours.
- **2, PoUS's session, two sessions linked by one registered row.** The composition soundness describes exists as `CarriedLeaf` (`security/Proofs/Flock/EndToEnd/Carry/Statement.lean:98`): the row registered in every session that reads it, and a SHA-512 collision term from two reads. It exists only pooled (`IntegrityHorizonCarried`, `:295`), which cannot give $(2 + k)\,\varepsilon_{ks}$ at every unit proved. **The missing lemma is `IntegrityPeriod` with carried leaves: ours**, new Lean beside `Carry/Statement.lean`, one new lock entry, no existing record moves [guess: 200 to 400 lines, Wed 25 Nov]. The same lemma serves PRF-620's ask 4 and CMP-370. Zero knowledge for one row read by two sessions is proved (`SpotFlockSessionsZK`, `security/Proofs/Flock/SampledGeneric/HidingSessions.lean:257`; not in the lock).
- **3, the setup sampled at 8,828 blocks.** `SampledKSLiveV5` at `rate:8828/B` gives $\delta \le 0.99^{8828} \approx 2^{-128}$ at $\lfloor B/100 \rfloor + 1$ wrong blocks (`Bridge/Count.lean:93`) [derived]. **One condition on how the setup registers:** every unit's output, sampled or not, is registered as the known content 1 (as H1's `ok` is). Then a block that does not decode is a wrong proof block whether or not it is drawn. If only the sampled units' outputs are asserted, an undrawn bad block is a correct proof block with output 0, and the bound says nothing about it. FreeBlocks and the certificate are yours; nothing of ours moves.
- **4, stores above $2^{23}$ blocks.** Nothing of ours: several setups sum as in 3, one V5 bound a setup. `P2SlackFamilyFreeBlocksWide` already reaches $2^{64}$ for m1p (`security/Definitions/Pous/Guarantees/P2.lean:240`); a free-blocks `periodHeldAt` is yours.

**Who writes the two lemmas:** proofs, since both sit in `security/Proofs/Flock/`. The carried one first, because three asks read it. It has no worker yet; I name its day, and the period form's if you say you need it, on Wed 18 Nov with PRF-620's.

Not established: that V5's relation at $K_0$ wrong blocks is exactly "more than $\lfloor B/100 \rfloor$ blocks do not decode" once every output is registered as 1 (read through the docstring of `RelS`, not its definition); whether a deployed period theorem exists under a name ZKASKS's search missed.

