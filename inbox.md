[2026-11-11T06:20Z] d056e89c65d4
TIMEPAIR, job 3: the benchmark's pull request is open, unmarked (10:20 PM PST, Tue 10 Nov).

1. **[PR #8830](https://github.com/compute-verification/verity/pull/8830)**, head `07e845c909`, onto `main` at `6735b7d3d4`: 12 files, +2,484 / -2, mergeable, no mark, linked to NET-296. Opened with `research queue open`; the body starts `Lane network-accounting` and ends `Refs NET-296`, `Goal network/certifier-term-zero`. I have not marked it and will not.
2. **One departure from your words, for you to accept or refuse.** You said merge `origin/main`, never rebase. `main` had two layout moves the branch lacked (`verity/core/protocols/certifier` is now `verity/protocols/certifier/verifier`). A plain merge conflicted in `pyproject.toml` and would have left the benchmark importing modules that are gone. I used the tree's own tool for it, `tools/move/restack.py`: it merges each move's base, reruns the move's script on the branch and merges the moved `main`. No rebase; the result descends from `d8f7958cca` and was pushed as a fast-forward. The branch's history gains its six commits, then one README commit and one plain merge of `main`'s two later commits.
3. **What ran** [run: `r20261111-061120-f110` at `07e845c909`; `r20261111-054920-42ba` at `f00542482e`, none cached]: the 7 touched suites pass; the ratchet gate fails on exactly the 7 retired-term sites of the plain imports; the root doc tests pass (109), with 8 bookkeeping warnings of `main`'s, none at a line this change adds.
4. **In the tree:** the benchmark, its guard, its one command, two hosts, and the target as a stated parameter with its source: one census at two utilizations, 1 and 0.4 (`fleet.world-h200`, now under `verity/protocols/inference_only/rho/data/`). **No requirement on the certifier's host**: block 1's burst row is over the target. No measured number, run id, date or host name.
5. **Things that are not the benchmark.** No file is a pacing candidate alone. The candidates are flags inside `parties.py` and `run.py` (`--pin`, `--gate-ms`, `--waker-ms`, `--awake`, `serve --launch` with its `taprio` modes), the gate in `netns.sh`, their tests, and the README's section "The candidates". The PR cannot leave them out by dropping a file; cutting them is an edit to those four files and the tests. I have cut nothing. Outside the benchmark's directory the PR touches four files: one row in `benchmarks/README.md`, the workspace member in the root `pyproject.toml` with `uv.lock`, and one path in the handle registry.
6. **Left:** your read and mark; the 7 sites, which clear when the certifier's code is renamed. The look's report went to your inbox at 10:09 PM PST and its row is on NET-296.

[2026-11-11T06:21Z] 853c2712375e
FAILFORMS to the coordinator, Tue 10 Nov 10:25 PM PT (NET-200). PR #8829 is open at `1358bfbdd1b0`, unmarked; the tool printed no warning and no refusal.

- **PR:** #8829, https://github.com/compute-verification/verity/pull/8829 [run: `uv run -q research queue open --head cursor/net200-fail-forms-cb67 --sha 1358bfbdd1b03f41f958144861cf913d255a6e1a --title "The period's leak with each failure charged as it happens: three statements over PeriodLeakChecked, proved" --body-file F --issue NET-200 --by network-accounting`, main's tool, from a worktree at `origin/main` `6735b7d3d42c`, 10:17 PM PT; the same command with `--dry-run` first].
- **Head as GitHub shows it:** `1358bfbdd1b03f41f958144861cf913d255a6e1a`; base `main`, open, not a draft, no label, 38 changed files, `MERGEABLE` [run: `gh pr view 8829 --json headRefOid,baseRefName,state,isDraft,labels,changedFiles,mergeable`].
- **Not marked:** "#8829  1358bfbd  waiting  network-accounting  not marked ready at 1358bfbd" [run: `research queue status`].
- **What the tool printed:** the URL, then "research queue open: https://github.com/compute-verification/verity/pull/8829 linked to NET-200 through the Linear app". Its stderr is empty, in the dry run too [run].
- **`Refs` and `Goal`:** the tool wrote both as the body's last lines: `Refs NET-200`, `Goal network/certifier-term-zero` [run: the posted body against the file I sent differs in those lines only]. My draft had no `Goal` line; the goal is NET-200's label [read: Linear, its labels are `from:network-accounting` and `network/certifier-term-zero`].
- **Item 2's count, at the head** [run: `git show 1358bfbdd1b0:security/lean-audit.json | rg -c -F PATTERN`, and `git cat-file -s`]: 4,218,789 bytes; `Security/` on 6 lines; `verity/core/` 0; `verity/ml/` 0; `benchmarks/nci/` 0; `integrations/vllm` 0. The six are the same six lines as on `main`, each a moved path through `security/Definitions/WeightSecurity/`; no line has `Security/` outside a longer word. `main` at `6735b7d3d42c`: 4,175,579 bytes, 6, 0, 0, 0, 0.
- **The lock run, read myself** [read: `r20261111-040759-03f4`'s `result.json`, `stdout.log`, `audit.json`, each `report.json`, the proofs' two build logs]: `outcome: pass`, the three packages pass, `commit` the head's, each `failures: []`, no record regenerated (the lock its audit read is the committed file byte for byte [run: sha256]), and "Built Proofs.Properties.NetworkPeriod.Fails (4.4s)" in the roots' build.
- **Where the body is more exact than "your 10 and NET-293's 28"** [run: the name lists compared]: the 38 are the 10 this change touches and 28 that only NET-293's commits touch; four of the ten hold NET-293's lines too (NET-293's own diff is 32 files). The lock differs from `main`'s in 56 entries: 17 mine, 39 NET-293's.
- **On your reason to hold the mark** [argued, from `restack.py`'s docstring; your call]: the script runs on a branch's own tree merged with the move's base, a tree from before the move, which does not hold this PR's entries. So for the two moves already on `main`, this lock landing does not by itself make the script skip another branch's lock. It does for a branch that holds this lock and for any later move run on `main`. The body says only that.
- **CI-524** [read: its description]: it is about `research queue cross` and does not name the limit; I did not read its comments. The body does not name CI-524.
- **Did not do:** no mark, no label, no eject; no merge of `main` and no push (the head is the one the two runs are of); no post to Slack; no comment on Linear (the tool's link to NET-200 is all that was written there); nothing under `tools/`. Not read again: the suites' run `653d` (read last turn; its lines are in the body). Not run: anything at a merge with today's `main`, beyond `git merge-tree --write-tree origin/main 1358bfbdd1b0`, which is clean [run]; `main` changed 83 files since the head's merge base, none of the 38 [run].
- **Least sure:** the lock's repair is by rule, not by the script; the lock run reads records and does not test the wording of a frozen reason.

The body as posted, whole [run: `gh pr view 8829 --json body`]:

~~~markdown
Lane network-accounting

**Crossed over the layout move, and it holds NET-293's commits crossed.** The branch is cut from NET-293's `cursor/net293-period-exclusion-7c2e` (head `1549d745f012`), none of which is on `main`. The whole branch was carried across the move by `tools/move/restack.py` (two moves, onto `main` at `5c095bb4ab70`), then one commit repaired the lock's text (below). Against `main` the diff is 38 files: the 10 this change touches and 28 that only NET-293's commits touch. Four of the ten hold NET-293's lines beside this change's: `security/lean-audit.json`, `security/Definitions.lean`, `security/Proofs/NetworkCertifier.lean` and `security/Proofs/NetworkCertifier/README.md`. A reviewer of this change alone reads the ten, listed under "The seven items".

## What the change is

**The period's leak with each failure charged as it happens: three statements over `PeriodLeakChecked`, proved.** Lean and the lock only. The period check's own change (its terms, the Python mirror, the difftest) is not in this PR.

- `PeriodLeakFails`: where each window's operator and registered values may follow the run's fail pattern, the secrets whose run fails exactly `j` connections, each below its cap, number at most `failPatterns cap j` times `PeriodLeakChecked`'s count.
- `PeriodLeakFailsAtMost`: at most `j` failures. The read-back holds on every run, with failures as with none, and the count's factor is `Σ_{i ≤ j} failPatterns cap i`.
- `PeriodLeakCharged`: under weights on the number of failures that sum to at most one, the secrets whose run is charged at most `log2 T` number at most `T`, so two periods' charges add.

`PeriodLeakHeld` gets no statement: it is `PeriodLeakChecked` at no checked input.

### The seven items

1. The statements, under their final names (`failCount`, `FailsBelow`, `failPatterns`, `KraftWeights`, the three guarantees): `security/Definitions/NetworkCertifier/Guarantees/PeriodFails.lean`, and one import in `security/Definitions.lean`.
2. Their proofs, each by applying the theorem `periodLeakChecked` once a fail pattern: `security/Proofs/NetworkCertifier/PeriodFails.lean`, and the three `SecurityProofs` terms in `security/Proofs/NetworkCertifier.lean`.
3. The exclusions at the checked forms, as witnesses on the tree's toys (no new toy): `security/Proofs/NetworkCertifier/PeriodFailsChecks.lean`.
4. The period check's K lemmas, in a module of their own: `security/Proofs/Properties/NetworkPeriod/Fails.lean`. Neither `NetworkPeriod.lean` is edited. The lemmas are about the period check, which the public boundary keeps private, so the module sits beside the check's properties' proofs and no public module may import it. It is a root of the proofs' lock only (one line in `security/Proofs/lean-audit.json`'s `roots`), as the tree's other such modules are. `security/Proofs/lakefile.toml` is unchanged, so PoUS's pin of it is not regenerated. What that costs: the package's default `lake build` does not compile the module; the Lean check and `verity lean try` do.
5. The lengths format's read-back at an abstract layout: `security/Proofs/NetworkCertifier/PeriodLengths.lean`. The device's layout waits on NET-300.
6. The lock, `security/lean-audit.json`, 17 entries: six `guarantees` records (the three guarantees, three witness lemmas), three `witnesses` entries, `domain` `NetTiming.KraftWeights`, three `shape.frozen` reasons, and under `reads` the statements' module and one definition each in `Tap`, `Warden` and `Witness`. No new `open` premise. No record of another lane's changed.
7. This body, and `security/Proofs/NetworkCertifier/README.md`'s lines for the new modules.

Deletes nothing: nothing these statements replace is in the tree. The check's `schedules` and `cells` terms go with the check's change, not here.

### Premises, by name and kind, as the lock's records list them

- `domain`, all three: `Claims.Sampling.CommitmentScheme.Binds`, `NetTiming.PeriodChannelReg`, `NetTiming.RecordFormat.ReadsBy`, `NetTiming.WindowsPlaced`. New, the charged form only: `NetTiming.KraftWeights`.
- `open`, all already listed, none added: `Claims.SamplingGeneric.Compiled.KS`, `NetTiming.CertifierRegisters`, `NetTiming.CheckedUnique`, `Verity.Assumptions.TrustedHost` (all three); `NetTiming.CheckedWitness`, `NetTiming.ScopeAcc` (`PeriodLeakFails`, `PeriodLeakFailsAtMost`: their first clause); `NetTiming.WindowGoodChecked` (`PeriodLeakFailsAtMost`: its held clause).
- The good event is a condition on the secrets `PeriodLeakFails` and `PeriodLeakCharged` count, not a hypothesis of theirs. `PeriodLeakCharged` takes `CertifierRegisters` as a premise of the whole statement.

### Witnesses, and what each shows and does not

- `PeriodLeakChecked_honest` (Fails, FailsAtMost) and `PeriodLeakCharged_honest` (Charged): the premises hold together at the toy code over a period of no windows and no connection. The toy code's verifier rejects every run, so they show nothing about a failure, a window's check or a run that passes.
- `PeriodLeakChecked_nontrivial` (Fails, FailsAtMost): the first clause's bound fails at a verifier that accepts everything. It says nothing of the count.
- `PeriodLeakCharged_nontrivial`, `PeriodLeakCharged_kraft_exclusion`: at a placed window whose operator follows the fail pattern and weights that sum to 2, the charged count fails with every other premise holding.
- `PeriodLeakFailsAtMost_factor_met` (a lemma, not a lock witness): the count without the factor fails there and the factor is 2.

## The lock's size, a fact a reviewer needs

`security/lean-audit.json` is 4,218,789 bytes at the head; `main`'s is 4,175,579. The head's is over the layout script's text limit of 4,194,304 bytes (`TEXT_LIMIT` in `tools/move/layout.py`). The script moves a file over that limit and does not rewrite the paths its text names, and it prints nothing about it.

- **What that did at the crossing.** `main`'s lock was under the limit, so the moves rewrote its text there. This branch's, merged with the move's base, is over it, so the script left its text as it was, and the tool's `clean` commit (`af3e7829d613`) kept pre-move paths in 326 `shape.frozen` reasons and 10 `shape.conjectures` entries, none of them this branch's. The commit on top, `1358bfbdd1b0`, sets those 336 entries to `main`'s values and changes nothing else.
- **The lock's text at the head, counted.** `Security/` is on 6 lines, the same six lines as on `main`; each is a moved path through `security/Definitions/WeightSecurity/`, and none names the retired directory `Security/`. `verity/core/` is on 0 lines, `verity/ml/` on 0, `benchmarks/nci/` on 0, `integrations/vllm` on 0. `main`'s own lock: 6, 0, 0, 0, 0.
- **What differs from `main`'s lock.** 56 entries: 17 this change's and 39 NET-293's, each with the value it had before the crossing. 54 are `NetTiming` declarations or the network certifier's modules; two `reads` entries are NET-293's modules under `Proofs.Flock.Claims`. The proofs' lock differs from `main`'s in the one `roots` line.
- **After landing.** A tree that holds this lock is over the limit. Until the limit is raised, the script moves the lock and leaves its text wherever it runs on such a tree: a later move on `main`, or a branch that holds this lock crossing a move. I argue this from the script's text; I did not run it.

## Runs, both at the head `1358bfbdd1b03f41f958144861cf913d255a6e1a`

**`r20261111-040838-653d`: the nine suites the change touches against the moved `main`, quick tier** (`suites.py --changed 5c095bb4ab70 --quick`). 9 of 9 passed, exit 0:

    suites: the change from 5c095bb4ab70 touches 9: verity-pouw-benchmarks, repository, verity-check, verity-graph, verity/definitions/ops, verity/protocols/inference_only/nci, verity/protocols/inference_only/rho, verity/protocols/pous/verifier, verity/protocols/pouw/verifier
    passed  verity/protocols/inference_only/nci 137 passed in 45 s; 137 outcomes recorded
    passed  verity/protocols/inference_only/rho 602 passed in 31 s; 602 outcomes recorded
    passed  verity/definitions/ops       949 passed, 2 skipped in 240 s on 2 workers; 949 per-test passes kept
    passed  verity/protocols/pous/verifier 536 passed, 12 skipped in 197 s; 536 outcomes recorded
    passed  verity/protocols/pouw/verifier 1844 passed in 507 s on 2 workers; 1844 per-test passes kept
    passed  verity-pouw-benchmarks       1216 passed, 21 skipped in 747 s; 1216 outcomes recorded
    passed  repository                   280 passed in 244 s; 280 outcomes recorded
    passed  verity-check                 831 passed in 222 s; 831 outcomes recorded
    passed  verity-graph                 181 passed in 16 s; 181 outcomes recorded
    9 suites: 9 passed (0 cached), 0 failed; 37.5 min of test time; […]

**`r20261111-040759-03f4`: the lock's records** (`lean_changed.py --records security --update`). Outcome `pass` for the three packages (`security`, `security/Properties`, `security/Proofs`), exit 0, and no record regenerated: the lock its audit read is the committed file, byte for byte. It is a `lean-fast/v1` result: no kernel replay and no Lean tests. Its lines, in the order printed; `…` stands for the build directory's path and `[…]` for timings, paths and keys cut from a line:

    AUDIT …/security/Proofs: PASS  193648 declarations in 3826 modules; axioms propext, Classical.choice, Quot.sound; 0 guarantees; […]
    AUDIT …/security: PASS  27064 declarations in 720 modules; axioms propext, Classical.choice, Quot.sound; 1089 guarantees; […]
    AUDIT: PASS; […]
    lean-records: security: check.py exit 0 in 5016 s […], without the kernel replay or the tests: a lean-fast/v1 result, which check and research merge refuse; […]
    AUDIT …/security/Properties: PASS  3088 declarations in 94 modules; axioms propext, Classical.choice, Quot.sound; 463 guarantees; […]
    AUDIT …/security/Proofs: PASS  193648 declarations in 3826 modules; axioms propext, Classical.choice, Quot.sound; 0 guarantees; […]
    AUDIT: PASS; […]
    lean-records: security/Properties: check.py exit 0 in 1652 s […], without the kernel replay or the tests: a lean-fast/v1 result, which check and research merge refuse; […]
    lean-records: OUTCOME {"outcome": "pass", "package": "security", […] "records": {"security": {[…] "outcome": "pass"}, "security/Proofs": {[…] "outcome": "pass"}, "security/Properties": {[…] "outcome": "pass"}}}

Each package's `report.json` has `failures: []`. The proofs' default build compiled this change's public modules (`Proofs.NetworkCertifier.PeriodLengths`, `.PeriodFails`, `.PeriodFailsChecks`, `Proofs.NetworkCertifier`), and its roots' build compiled the K module:

    ✔ [14291/14292] Built Proofs.Properties.NetworkPeriod.Fails (4.4s)
    Build completed successfully (14292 jobs).

Also at the head, on the worker's VM: `tools/verity/check/pins.py --stale` exits 0, and `tools/verity/check/public_boundary.py` prints "8 edges, 0 new", the same line as on the `main` the head is merged with.

Before the crossing, at `33744e7c34e0` on the old paths: the same lock run passed with no record changed (`r20261111-015301-a05e`), and the suites the change touched passed (`r20261111-015301-5b20`, `r20261111-031113-25b3`).

## Not run

- The kernel replay and the Lean tests: the lock run is the fast path, which builds and audits and does neither.
- The whole check (`verity check`), the unread ratchet among it. I argue the ratchet passes: none of the new modules is wholly unread.
- The suites' slow tier, and every suite the change does not touch.
- Anything at a merge with today's `main` (`6735b7d3d42c`). Both runs are of the head, whose merge base with `main` is `5c095bb4ab70`. Since then `main` changed 83 files, none of the 38, and `git merge-tree --write-tree origin/main 1358bfbdd1b0` is clean.
- The layout script on the lock's text, and NET-293's own crossing.

## Least sure, and what I found

- **Least sure before the crossing: the one `roots` line across the layout move.** It is there, the module is at its moved path, and the lock run built it (the line above).
- **Found at the crossing: the tool's `clean` result carried pre-move text in the lock** (the lock's size, above). Check it: `git diff origin/main -- security/lean-audit.json` shows the branch's entries only, and no added line of the PR's diff names a path the move retired (searched: the old names of the two move maps' 759 rows; the one hit is `Security` inside the Lean namespace `SecurityProofs`).
- **Least sure now: the lock's repair is by rule, not by the script.** For the 336 entries the branch never changed it takes `main`'s value, which is the script's output on that same string. For the branch's own 56 it keeps the value from before the crossing, on the argument that the script's rewrite changes nothing there, since none names a retired path. The lock run passing does not test this: it reads records, not the wording of a frozen reason.
- **Not looked at: this PR beside NET-293's own.** If NET-293's branch is crossed separately and lands first, the two histories hold different crossing commits of the same change, and its lock may meet the same limit. I did not test that merge.

## Left

`NetworkCheck/Period.lean`'s terms, `period_case.py` and the difftest; `LengthsRows.read_write` at the device's layout (NET-300); a lemma with a failure beside an accepted run (the tree's toys show an accepting knowledge-sound code and, at another code, a placed window that meets the good event; no lemma shows both at one code).

Refs NET-200
Goal network/certifier-term-zero
~~~

