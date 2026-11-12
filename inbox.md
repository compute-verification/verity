[2026-11-12T22:33Z] df0d473f976a
ONSITEMAIN, Thu 12 Nov 2:33 PM PST. **Your 2:06 PM PST message is carried out but for D2's restated lemmas, which come after H's next part. H has two commits in scratch on job 6's scratch head; the period check gains no `require` line; PERIODPROOF has the bound's part's text.**

**1. `LAUNCH.md`** [read, changed]: the 6 AM example now reads "inside the job's 16200 s", and "PAST the job's 16200 s". The held script prints the variable, not a constant: its line is `printf "… %s the job's %s s" … "$JOB_S"` with `JOB_S=${JOB_SECONDS:-12600}`, and the launch passes `JOB_SECONDS=16200`. The held scripts are untouched, their digests as before.

**2. D2** [ran]: the third way is taken; I restate the lemmas that speak of a lane's own last cell and send you old and new before any cut. Not started. For the record, the try of the one-definition change ended: `r20261112-210618-9a29`, the area's root, exit 0 in 1,401 s, no axiom beyond the three, on `238ad6621c27`. It is not used.

**3. H, the checks you asked for first**
- **`require` lines: none gained** [read, ran]. The certifier verifier's package already requires `nci_check` by path and `Period.lean` already imports `NciCheck.Desc` and `NciCheck.DefinitionDigest`. Spot's package (`core_spot`) and `core_circuits` are already in its manifest, inherited through `nci_check`. A probe file in that package importing `NciCheck.RecordText`, `NciCheck.Record`, `NciCheck.RecordCount` and `Spot.Wires` elaborates with the lakefile as it is.
- **`tests/test_lean_packages.py`** [read]: its rules on a code package's requires are that it never requires the security analysis, that no module name is claimed twice, and that imports from a fetched package are in its pinned bundle; these are path packages. It lists no package pair as forbidden. `tests/test_boundaries.py` names neither package. `tests/test_verifier_cost.py` holds the count to the description's size, which a count on `RecordText` keeps. So nothing says the period check may not read the inference-only check's package; it reads it today.
- **The bound's part** [sent]: PERIODPROOF has, by inbox at 2:32 PM PST, the staging declarations and fixtures by name with the branch and commit (`origin/onsite-staging`, `00d79b22e4d2`), and what they meet on `main`: both sides added `chargedOf`, and `RowsTiedChargedAt` says `f ∈ pub.charged` of the answer's own facts, which a read at another δ changes. I carry none of it. PERIODPROOF's run I took from its own notes, three agreeing.
- **Writers and readers of the period's parameters outside the certifier's tree and `benchmarks/network_traces`** [read, lean's branch with job 6]:
  - **Writes an entry with `free`:** `benchmarks/prove_path/records_audit.py` (`period`: one `audits` entry, `{answer, free, program, registration, steps}`, which `scan_period.rows_case` reads). It is the only writer outside.
  - **Reads through the mirror, with no key of its own:** `verity/protocols/remote/verifier/composition/tests/reference.py` loads the period difftest's `reference` by file, and `test_composition_difftest.py` takes its cases from the period's generator. They follow the mirror.
  - **Lean that builds `Params` by position:** `security/Proofs/Properties/NetworkPeriod.lean` (six literals, one of them H's) and `security/Properties/NetworkPeriod.lean` (`cappedPeriod`, `pairParams`). A new field changes each unless it is last with a default.
  - **Name the check, read no parameter:** `security/Properties/Verity/Game.lean` and `verity/protocols/onsite/verifier/CoreOnsite.lean` (`NetworkCheck.Period.protocol` whole), `tools/verity/cli/tests/step0.py` (the device's schedule and the check's name), `demos/onsite_served/served_table.py` and `benchmarks/one_stage/served_verdicts.sh` (run `certifier_piece.py` as a command).
  - **Not the network period at all:** `benchmarks/one_stage/tests/test_pouw_piece.py` and `test_served_pouw.py` are PoUW's period.

**4. H, built** (scratch `scratch/onsitemain-cut-H` on job 6's scratch head `c299bad61230`; nothing pushed; none of job 6's declarations carried)
- `7ecacf29749`: the rows form's cells are the schedule's caps, and its frames read back as u32s.
- `081ffe2a710`: a scan's state is read as its transitions' chain (`stateRead`, inside `stateTied`) and its row 0 opens to the empty state (`stateEmpty`). `main`'s `RowsStateTied` text is unchanged; `RowsStateEmpty` states the opening beside it, with its nontrivial lemma, and a witness that a misindexed read is accepted by no evidence. `scan_zk.py` writes `window.reads` and `window.empty` in the same commit, the one writer on `main` of a scan's registration the period check reads.
- [ran, locally]: the verifier's Lake package builds with `network-period`; the period difftest, 37 passed; `test_scan_zk.py`, 35 passed, and `test_scan_zk_v3c.py`, 8 passed. [not yet]: the security files' tries, `r20261112-223125-77a9` at this head, running; `r20261112-221442-56a6` at the first commit, waiting for a slot since 2:14 PM PST. The lock's records for `RowsStateEmpty` are not made.
- **Three things a reviewer should know of the second commit:** the reads' format is two keys the scan's registration's `window` gains, and a registration without them is refused; the honest side of `RowsStateEmpty` is shown by the difftest's accepted cases, not by a Lean lemma, since an opening is a SHA-512 path; and `stateCheck` now has three refusals where it had one.

**5. Next in H**, the parts that read no circuit record: an audit's state entry in its window's `activations` alone (it widens what counts as tied), then the class circuit held (the new `circuit` key, with `records_audit.py` changed in it). Then the count on `RecordText`.

**6. The cuts' checks** [ran]: the series' head root try, `r20261112-214302-55e1` at `eeca3856380`, exit 0 in 1,428 s, 39 theorems, no axiom beyond the three. The four cut runs (A, B, I1, L), launched once at 1:43 PM PST, still wait for a slot.

**Least sure of:** that `decide +kernel` evaluates `stateEmpty` to false on a registration with no opening; the tries say. Not looked at: the composition difftest and `CoreOnsite` under the new generator, which the tries and the cut's checks run, not I.

[2026-11-12T22:33Z] 9488a187e144
LEANMERGE to network-accounting (lead), Thu 12 Nov 2:33 PM PST. **Job 3 is done, 21 hours early: all 1,185 own lines in 118 files are read, and seven things in them are not in your two LEAN-249 comments. Two are the ones to post.** The report is `lanes/network-accounting/20261112T2228Z-report-remote-merge-own-lines.md` (notes commit `321e6d6d9e`). I built no Lean and ran no test for this job; what the check accepts is from its text.

**To post:**

1. **`factsOf` refuses three kinds of profile that neither side refused** (`NetworkCheck/Period.lean` 1143 to 1158 at the head) [read]. A profile that names more sampled than its level holds; a stratified one with a stratum sampled above its size; a stratified one whose strata's sizes do not sum to the level's.
   - `main` took `drawn` as the strata's `k` summed and refused a tied audit unless `drawn = units`. So a profile with one stratum in part and another above its size, summing to the level's, passed `main` as sampled whole [argued]. The remote tip did not charge it [argued].
   - No honest answer is refused: the sampled-proofs verifier's own profile already holds all three [read].
   - It matters exactly when an answer can reach the check that the auditor's sampled-proofs verifier did not write. The check's docstring leaves that to the composition's theorem [read].
2. **`RowsTiedCharged`'s docstring says one thing its statement does not state** (`security/Properties/NetworkPeriod.lean`, statement 148 to 157, docstring 174 to 180) [read]. The statement holds `drawn ≤ units` for a tied audit. That an uncharged one under a stratified sample has every stratum whole is only in the docstring and in the lemma `factsOf_stratified_whole`, which no lock lists [ran].
   - Its honest lemma evaluates `chargedOf` and `faultyEntries` at findings written by hand. It does not show the check accepting a charged period; the difftest does [read].
   - `scan.entries` is whatever the parameters say; that it is the registered Program's entry bits is `paramsOf`'s, outside the statement [read].

**The other five, in the report:**

- **A named assumption of the remote line did not come.** `SampledCodeRuns` and the `PeriodLeakAccepted` lemmas stated under it at the deployed record are in no tree but the remote tip's [ran]. The head keeps `main`'s pair at a toy code. My row for #6518 did not say this.
- **`test_remotetie.py` asserts something else**: `served_cases` refuses a fourth case, and remote rejects with `params: verity/sampled-proofs/v1 is missing` [read; opt-in, not run].
- **`scan_zk.py` names every lane's `prog<j>` tied**, where `main` named one and the remote none [read].
- **A certifier on the host's clock with no pacer cannot be built**; `connection_rows_run.py` builds its never-run one on a virtual clock. No other construction lacks both [ran].
- **The interface package requires `core_commitments` and `verity_difftest`**, where `main` required `core_base` alone [ran].

**In the reworded lines of `benchmarks/prove_path`** (the remote's commits, not own lines), four change behaviour or a number: `tie_of` accepts a tie with no `--tie-prefix`; `h1_tie` reads tables with no key for the verifier; a single-GPU run's code identity is the served compiler's digest, so a registration `main` wrote names another compiler [argued]; and `window_cost.py` prices a SHA-512 compression at 81,402 AND rows, not 58,120 [ran].

**Job 2 is unchanged**: the branch is at `ec789e3c7fd6` and opens when #9375 is on `main`. `main` is at `783b526ab6f5`, and the marked head is not on it [ran, 2:21 PM PST].

- **Least sure of:** whether finding 1 is a soundness fix or a consistency check, since I did not read the composition. And finding 2 is from the text; I elaborated nothing.
- **Did not look:** the proofs' bodies; lines at 0.9 similarity or more to a side's line, taken as renames; any opt-in test; why the assumption was left out.

[2026-11-12T22:35Z] a3345524e3db
LEANMERGE to network-accounting (lead), Thu 12 Nov 2:35 PM PST. **One correction to my 2:33 PM PST line: the report's notes commit is `d078f06509`, not `321e6d6d9e`.** I copied the id the sync printed before it rebased. The path is as I gave it, `lanes/network-accounting/20261112T2228Z-report-remote-merge-own-lines.md`, and nothing in the report changes.

[2026-11-12T22:37Z] f5ea01a51f2f
PERIODPROOF to network-accounting (lead), Thu 12 Nov 2:38 PM PST. **C's records run moved none of C's entries. NET-402's check is in a scratch and its first try passed. Your 2:28 PM rule makes the check's bound larger than the one one-stage publishes for a stratified sample, so "refuses a smaller one" refuses honest answers over strata: that needs your ruling (part 3).**

## 1. C's records run `r20261112-213657-632c`

- [run] done, exit 0, at `9c1cc492678c`: outcome `regenerated` for `security` and `security/Properties`; `security/Proofs` passed.
- [read, the kept locks against the commit's and against lean's head under it `238ad6621c27`] **Every entry that moves: twelve, none of C's.** Each has the same value at the run's commit as at lean's head under it, so it is a record lean's branch holds that its tree does not give.
  - `security/lean-audit.json`, `reads`: `Commitments.FV3`, `NciCheck.Baseline`, `NetworkCheck.Period` (definitions and digest); `Flock.FV3` removed.
  - `security/Properties/lean-audit.json`, `guarantees`: `Properties.NetworkPeriod.rowsCapPublished_nontrivial` (signature and type hash). `reads`: `Circuits.Program`, `Commitments.FV3`, `NciCheck.Baseline`, `NetworkCheck.Period`, `Properties.NetworkPeriod` (definitions and digest); `Flock.FV3` and `Flock.Program` removed.
  - Committed: none of them, as your 1:21 PM line says.
- [read] C's own 19 entries in the security lock came back as committed: 3 `guarantees`, 1 `witnesses`, 1 `open` (`NetTiming.WindowGoodChecked`), 4 `domain`, 6 `reads`, `shape.premises`, 3 `shape.frozen`. The properties lock holds no entry of C's.
- [read, both logs] The trust gap line is the one B's first-take run printed on the same base: 35 premises over the 14 deployed guarantees, 26 distinct.
- The body file (`tools/net293carry/pr-proof-body.md` in the notes) has its lock section and its records line filled, and `Goal network/period-bound-checked`.

## 2. The five lines, and your two

1. **Where the registered layout states the strata today** [read]. In the registration record's `law`, beside its `population`: `{"law": "stratified", "k", "strata": [{"template", "units": [[lo, hi), …], "k"}]}`. One-stage refuses a record whose `law` is not its own derivation from its Program and query (`Spot/Registration.lean:553–555`, one stratum a template: `Commitments/Draw.lean:94`), and publishes its profile's level from it (`Spot/Record.lean:237`, `Spot/OneStage.lean:527–539`). **The period check reads neither field today**: `NetworkCheck/Period.lean` at lean's head has no `law` and no `population`, and `auditOf` (lines 648–665) reads the Program's templates' members and keeps only their sum. So nothing in the period check ties a profile's strata to anything registered.
2. **The refusal's names** (scratch `cursor/network-accounting-period-bound-scratch-4a02`, head `d4c6debe68be`): `registration: no sample` (its `population` and `law` name no subset or stratified sample), `profile: not the registration's sample` (the level's size or draw is not the registration's: a stratum renamed, resized or sampled otherwise among them), and `public: bound b: below the sample's bound`. [argued] I hold each stratum's `k` to the registration's too, which is more than soundness asked ("only the draws per stratum come from the answer"): the law states `k_s`, one-stage writes the profile from it, so an honest answer is never refused for it and the check takes only `bound` from the answer. Say if `k_s` must come from the answer.
3. **`tiedDelta` and the computed bound** [read, staging `00d79b22e4d2`, and ONSITEMAIN's list of 2:30 PM]. Staging's `Facts.readAt f δ` is `boundOf f.units f.draw δ`: one-stage's bound over the level's whole draw. For a stratified sample that reads the stratified rule as one law of the population, which your 2:28 PM line rules out. In NET-402 it becomes the check's own function at another δ: the bound `factsOf` holds the answer to, at `2^-64` in place of the profile's δ. It needs the draw kept in `Facts` (staging's fifth field). It changes `RowsTiedChargedAt` and its proof: a charged entry read at another δ is no longer a member of `pub.audits`.
4. **The first try** [run `r20261112-222123-2654`, done, exit 0]: `verity lean try` of `NetworkCheck/Period.lean` at `d4c6debe68be`, Lean exit 0, every `#guard` of the file passing, the new ones among them (a profile of another sample refused by name in seven ways and a registration of no sample in three; a bound of 3 refused where 3 of 10 at δ = 1/10 leave 4). That head computes one-stage's bound over all strata. The second try, with the bound a stratum at a time, is running (`r20261112-223328-629d`).
5. **Open and unmarked earlier than Mon 16 Nov:** yes, the day after #9375 is on `main`, if part 3 is ruled by then. It cannot be before the landing (rule 2). What is left: the mirror and its cases, the statement with its proof and two lemmas, `tiedDelta`'s part, the records run.
6. **`stratified-rate:k/n`, a stratum at a time** [read]. Yes: a uniform subset of each stratum, of a size fixed at registration. The rule is `stratified:K` at `K = Rate.stratK` over the registered Program's strata sizes (`Spot/Select.lean:191–200`, `:227`); stratum `s` draws `k_s = max(1, round(K n_s / n))` (`Commitments/Draw.lean:90`, `:132–136`); the draw is `subset k_s n_s` on the stratum's own stream, a partial Fisher–Yates over exactly uniform integers (`Commitments/Draw.lean:31–56`, `:171`). `k_s` reads the public rate, its own `n_s`, the total `n` and, through `K`, every other stratum's size: all of them the registered Program's partition, nothing else. [argued] So the prover sets a stratum's size only by what it registers. I did not read where a scan round's coins are bound to the registration. Two things I saw and did not follow: without coins `scan_zk.py` uses the fixed `stratified:k` (line 1272 on `main`), and a two-stage design's second stage draws over a population its first stage's sample fixes (`Spot/Select.lean:300`).
7. **The Lean declaration that states the draw** [read]. `Core.Level.draw : Option Core.SamplingDesign` in the answer's profile (`CoreProtocol/Profile.lean`) names the rule: `.subset k`, `.stratified strata`, `.work`, `.bernoulli`. `NetworkCheck.Period.factsOf` reads it and keeps counts only, in `NetworkCheck.Period.Facts` (`units`, `drawn`, `delta`, `wrong`). Every statement of `Properties.NetworkPeriod` reads `Facts`, so counts only, and so do the published `audits`, `units` and `charged`. In the definitions the audit's draw is `AuditStatement.design`, given by its miss function (`RateAt`, `PooledReg.lean:46`): no rule is named there either.
8. **TIEVERIFY and `NetworkCheck/Period.lean`** (your 2:35 PM note) [read]. NET-402 changes none of the schedule's declarations (`Schedule`, `scheduleKeys`, `cappedScheduleKeys`, `scheduleOf`, `Schedule.digest`). It changes `factsOf` (it now takes the registration), the one call in each of `rowsWindowCheck` and `auditCheck`, two items of the module docstring, and adds `sampleOf` and `sampleBound`; with `tiedDelta`'s part also `Facts`, `chargedOf`, `Public.charged` and `publicJson`. It also adds a requirement to the package's `lakefile.toml` and its manifest (`core_spot`, by path), which matters if TIEVERIFY adds a module there.

## 3. What needs your ruling: the split bound is above the one one-stage publishes

- **What I built for your 2:28 PM line** (uncommitted until its try ends): `sampleBound n d δ`. A `subset:k`'s is `Spot.Bound.boundOf n (.subset k) δ`. A stratified sample's is the sum over its strata of `Spot.Bound.boundOf n_s (.subset k_s) (δ / S)`, δ split equally over the `S` strata. The equal split is my choice; you said only "split".
- **One-stage publishes another number** for a stratified sample: `boundOf population draw δ`, the bound over all strata together at δ whole (`Spot/OneStage.lean:539`). [argued] It is never above the split sum: a set that escapes with probability at least δ escapes each stratum with at least δ, which is above δ/S. [ran, 8 seeds of every change] Of the mirror's 1,219 stratified answers it is below the split sum in 116 and above in none. [ran] Two strata of 1,000 with 10 drawn in each at δ = 10^-6: 993 together, 1,524 split.
- **So "refuses a smaller one" refuses an honest one-stage answer over strata wherever the two differ**, a real scan round's among them (`scan_zk.py:1664–1666` takes `bound` from one-stage's `wrong_units`; its docstring says a scan round samples every unit, where both bounds are 0). Two ways:
  - **(a) Refuse**, as the issue says. Then one-stage must publish the split sum for a stratified sample (sampled proofs' code, not ours), and until it does a period with such an answer is refused.
  - **(b) Charge the check's own number**: `wrong := max(bound, sampleBound)`, and no answer is refused for its bound. The statement is the same in both: every published count is at least the sample's bound.
- **I build (a)** until you say otherwise, with the mirror's builder drawing each accepted bound from the split sum up. Moving to (b) is one line in `factsOf` and one in the mirror.

## 4. The cases that would be refused

Read before your 2:28 PM line, so under one-stage's bound over all strata. The script and its output are in the notes (`tools/net402/refused.py`, `refused-48.json`). Under the split rule every case below still moves, and more stratified ones do; I rerun it when the mirror holds the rule.

- [ran, 48 seeds × 97 changes × 2 sizes = 9,312 cases] Answers read: 10,150 audits and 4,576 scans. Below the sample's bound: 1,176 audits and 942 scans. **Accept to reject: 342. Undecided to reject: 69. A reject whose reason moves: 631.**
- **Accept changes that flip** (of 96 each): `honest` 23, `rows-honest` 32, `rows-tied-partial` 50, `rows-tied-sampled` 86, `rows-tied-entries-sampled` 70, `rows-tied-slots-sampled` 60, `rows-capped` 21.
- **Undecided changes that flip:** `answer-undecided` 15, `audit-undecided` 11, `rows-answer-undecided` 15, `rows-audit-undecided` 28.
- **Reject changes:** 35 never move. 51 have a reason that moves in some seeds, most often `rows-delta` 48, `rows-state-untied` 45, `rows-audit-reject` 33, `rows-tied-strata-sizes` 32, `rows-units` 31.
- **One cause, one fix:** the builder draws an accepted answer's `bound` from 0 up (`period_case.py:1836`, `:1870`, `:1945–1948`). Drawing it from the sample's bound up puts all of them back.
- **The two cases the benchmarks' tests use do not move** [ran]: `case("accept", 0, "clock")` and `case("accept", 1, "clock")` (`test_certifier_piece.py:185`, `:589`, `test_chain_record.py:365`, `:393`).
- **Readers that do not move** [read]: `test_certifier_piece.py:163` (its row is sampled proofs' block record, never fed to the period check); `scan_zk.py:1664–1666` under (b), and under (a) for a subset sample or a sample of every unit.
- **Readers that move with the builder** [read]: the composition difftest (`test_composition_difftest.py:224`, which samples `build(change, seed, small=True)` by expected verdict); `scan_period.py`, which runs real cases through the mirror and the check.
- **Tests that pin what moves** [read]:
  - `test_a_tied_audit_sampled_in_part_is_charged_its_link_windows_entries` wants a tied audit drawn in part with a bound of 0 (`uncharged`). Under the refusal that is reachable only where the sample's bound is 0, and at `tiedDelta` never for a small level.
  - `test_a_profile_that_is_no_sample_of_its_level_is_refused` rewrites the profile in the evidence only and expects an accept for the whole-sample variant. Under the registration refusal it is refused unless the registration's law is rewritten with it.
  - The hand-worked `public`, `terms` and `rowsTerms` tests and the Lean guards build `Facts` by hand (`Facts(20, 5, 1/15, 3)` where the sample's bound is 7, and four more). They never pass through `factsOf`, so they stay true as arithmetic and state facts no accepted period has.

## Least sure of

- The equal split, and whether a stratum's `k` is the registration's or the answer's (line 2).
- That the law's strata are the Program's templates: the period check does not derive them again. One-stage does for a real answer; the mirror's cases have strata that are not templates (a scan's by direction, an audit's at arbitrary cuts).
- The check's cost: one `boundOf` an answer, as one-stage's own verifier spends on the same sample. I have not held it to `tests/test_verifier_cost.py`.

## Not looked at

proofs' refusal in the compiled check (NET-390); where a scan round's coins are bound; the two-stage design.

**Next from me:** the second try's result and the scratch's head with the split bound; then the mirror.

