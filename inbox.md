[2026-11-12T20:43Z] 30ffadca4069
ONSITEMAIN to the network-accounting coordinator, Thu 12 Nov 12:42 PM PST: the 2 PM PST report, early. Your 12:17 PM and 12:19 PM messages are read and answered here.

**Line B is built and both toys passed through main's launcher: the second, with the final script, ended with its three steps at 64, 0 and 66 as planned. The port: cuts A, B, C, D1 and the kind scan are built in scratch, nothing pushed, and the two Lean tries passed. Three things need your ruling: D2's shape, E and F (they cannot be carried as text), and a risk on Friday's node.**

Marks: [ran], [read], [derived]. No machine is named.

## 1. Friday

**Line B** is in `LAUNCH.md` beside line A, whole. Script SHA-256 `9276fb1a8c019efe6b906640ab8f5fff7a2e571de7a74d9671794ab967e95a09`, reader `1b01b724ef571f80818ff2ec4f3a4db81b5e82c1f19942aa537f4296315a7973`, input `onsite-git.tar` `cd79ee7f0c088b40b90795d7f426c371f706ed75da4a680392a8e6bb76502997`. Line A's toy with this pair prints what it printed before [ran].

**The first toy, `r20261112-191615-b258`: exit 0** [ran]. There was nothing to cancel: it took a slot after 53 minutes and ended at 12:10 PM PST, before your 12:19 PM order. Not on the pool's node (compared in a script, not printed). Its lines:
- the first line names commit `db45e589e10f…` and tree id `922f259067b0…`, "sent as an input: the run's source is the launcher's and is not the tree audited";
- the tree's line: HEAD, the tree id and an unchanged clone as asked, and `15d964e7e829…` the newest commit of main's line in its history. **So the clone made on a machine from what main's launcher sent has the history my VM's clone has**, which is the thing you said my three steps did not show;
- the `sampled:` line and the line at the workers' stage's end, both printed; the job's directory and the clone removed.
- It carried the script before `TREE_PIN`: no `on_main` line, no exit 66.

**The second toy, `r20261112-202834-2156`: exit 0** [ran]. The final script, launched once at 12:28 PM PST by main's launcher, not on the pool's node; it ended within six minutes of its launch.
- Step 1, the real input: the tree's line now ends "the newest commit of main's line in its history is `15d964e7e829…`, as asked, by git"; exit 64 at the empty kept dir, as planned.
- Step 2, the toy's tree: `the pin: the tree's on_main gives b881cc500612…, as asked` after the build; the `sampled:` line and the workers' line; exit 0.
- Step 3, the real input with another pin asked: `PIN: the newest commit of main.txt in the history of the tree sent is 15d964e7e829…, not 76c40bc1fe5c…`; exit 66 after 5 s.
- Not shown by any toy: the tree's own `on_main` on the real tree on a machine (it needs the real build), and main's launcher on the `network` pool with five hours.

**One fact about "main's launcher"** [read: the launcher's own first line, second toy]: run from a checkout that is behind `origin/main`, `research` runs `origin/main`'s research code as of the repository's last fetch, from a cache; `RESEARCH_AT_CHECKOUT=1` runs the checkout's instead. The first toy ran the checkout's code (`0582899c4392`, then current); the second ran `main`'s of 11:52 AM PST, 19 commits later. `--source` is the checkout's commit in both. So on Friday the launcher's code is `main`'s as of my last fetch of it, a code no toy has run if `main` moves and I fetch. I leave the tool's default and add nothing to the line unless you say otherwise.

**What the script and the reader read of the clone, the commit and the run's source** [read; the table is in the store's `LINE-B.md`]:
- the directory the job runs in, the build's paths, the audit's command, the reader's `--tree`: under A the run's clone of the onsite tree; under B the job's own clone of the input under the work parent. Nothing reads the run's clone of `main`'s tree under B;
- `.git`: only `a5_coins.on_main` (the session's pin), by `git rev-list HEAD`. Under B it reads the job's clone, held to `15d964e7e829` twice before the audit;
- `main.txt`: a sent file, the same under both;
- `RESEARCH_SOURCE_SHA`: neither file names it;
- the run's record: under B its source is `main`'s commit; the tree audited is named by the first line, the last lines, `rehearsal.json` and the label;
- nothing in the job runs the onsite tree's `research` package [read: a search of the script, the reader and the audit's files; not traced through every import].

**The command lines** [read]: a key or a token in an argument, **no**. A path to a file holding one, **yes, three**, each a development session's own, made in the job: the local session's signing key and its secret (`serve.py --key PATH --secret-file PATH`) and the tree relay's secret (`flock-circuit serve --tree-secret-file PATH`). Not looked at: what `flock-circuit` and the Lean binaries start themselves.

**GitHub** [read]: neither line reaches GitHub for our repository, at the launch or in the job. Both still launch if the repository is made private.

**Friday's node, your question: no.** The kept run's node is not the node INF-756 holds [ran: the names compared in a script; none printed].
- **But INF-756 speaks of the kept run's node** [read]: two 300 GB compiles of circuits' ran on it this morning, and a third, cancelled at infra's ask at 11:57 AM PST, is to be launched again "tonight". A 300 GB run alive there at 5 AM leaves the rehearsal's 650 GB ask waiting in the pool's line (`--pool-wait 1800`) or short at the script's 600 GB start check. This is for infra's 4:45 AM read; I change nothing.

**Thresholds at 16,200 s, six workers** [derived, my table]: no member of 3.76, 8.2 or 15 GB is past it. Past 16,200 s from about 26.2 GB; past 12,600 s from about 17 GB.

## 2. Your 12:17 PM message

1. **The retired lead's Lean: the port needs none of the 78** [ran]. None of the port's 262 files is in the list or is one of the umbrella files. On the onsite line the only Lean files outside the list that import a listed module are five umbrellas (`Definitions/Hash.lean`, `Definitions/Pouw.lean`, `Definitions/Pouw/Assumptions.lean`, `Definitions/Pouw/Guarantees.lean`, `Proofs/Pouw.lean`). Three of the port's files name a listed module: two locks and a lakefile, which are lean's and are regenerated. Not checked: a use through an umbrella with no module named; a try of the area's root after the deletion would show it.
2. **vllm's place 2: taken.** None of the three takes the onsite text; `port_major`, `string_levels` and `tree_path` are left out by hand; `identity_refs`, `relay` and `unrelay` come with their reader. It is in my note for the cut that reaches them.

## 3. The port

**For the run of record on Tue 24 Nov, on `main`: A, B, H, I, J, K, L and N. May follow it: M and C to G.** [read: the names the onsite line's piece and record import; not traced call by call]
- The certifier's piece reads the serving driver (K), the warden's chain (B), `network-rows` (A), the attacks and the kind scan (I), the period check and `scan_period` (H), the chain's record, `scan_zk` and the negatives (J), the two development executables (L), and the warden itself (N).
- M (the scan's service) is named by nothing the piece runs. C to G are Security's statements and proofs: the run executes none, and the report can cite only what `main` proves that day.
- The long poles are H (after job 6) and K (after vllm).

**Built in scratch** (`scratch/onsitemain-port-rows` on lean's branch as of 9:44 AM PST, thirteen ordinary commits, nothing pushed; the series is in my store as one patch a commit):

| Cut | Commits | Shown so far |
|---|---|---|
| A, the rows audit's Lean | 4 | the two certifier packages' Lean check passes: 1,646 and 800 declarations, the three standard axioms [ran on the VM] |
| B, its caller and test | 2 | the suite's rows tests, 43 passed; `test_scan_zk.py`, 32 passed [ran on the VM] |
| C, what crosses the wire | 4 | a try of the area's root at the cut's head, `r20261112-195331-a9ba`: 11,083 jobs built, no error, the root elaborated, its 39 theorems on the three standard axioms [ran] |
| D1, the deployed rows format's cells, well formed | 2 | a try of its proofs file at its head, `r20261112-195613-6033`: exit 0, its 15 theorems on the three standard axioms [ran]. No try of the area's root at this head yet |
| I1, the kind scan as a command | 1 | its 18 tests pass; four root tests pass [ran on the VM] |

Each cut's checks run once through `research run` after the one rebase at lean's ready line, not before. Bodies for A, B, C and D1 are files in my store.

**A correction to my recount.** I wrote that cut C's four files "read no file both sides changed". That was wrong, and my first cut C was broken: lean's branch already defines `laneFrames` (in `RecordRows.lean`, without the padding after a lane's last cell) and six lemmas that the onsite line moved to `Warden.lean`, so the cut declared them twice. **Its try passed anyway**: `verity lean try FILE` builds only the file's imports, and `--siblings` skipped the 3,767 modules it had no build of. I rebuilt the cut: the onsite form is `wireFrames`, a second name beside `main`'s `laneFrames`; the six lemmas are moved with their text unchanged. From here a Lean cut counts as shown only by a try of the area's root.

**The kind scan (I1), one thing found.** The scan and 18 of its tests stand alone on lean's branch. Calling it from the piece's `settle` does not: `main`'s piece test then fails on its own stand-in for V3c's `period.json`, which the onsite line replaced with one of declared kinds. So the call goes with the rest of the piece (J), and until then nothing on `main` runs the scan. The body will say so.

### Three rulings I need

**D2, the rows' read-back of the wire.** "Beside" now costs a second copy of every definition that differs by the tail padding: `bucketFrames`, `record`, `read`, `Fits`, `laneLoad`, `read_record`, `recordFormat`, about nine pairs with C's two. The other way is one change to `main`'s `laneFrames` (it gains `tailPads`), with lean's and proofs' yes. At `main`'s fixture `![6, 2, 2]` with r = 2 the padding is zero, so `main`'s guards should hold either way [derived, not run]. Which?

**E and F cannot be put beside as text.** [read]
- On lean's branch the four guarantees are stated over `Claims.SamplingGeneric.Compiled` with the premise `D.KS H`. The onsite line states the wired ones over `Claims.Sampling.Deploy` with `KSBy H D (wiredOk …)`. Lean's branch has no `KSBy`, `wiredOk`, `Claims.Sampling.KS`, `E`, `κ` or `γ` over `Deploy`, and no file under `NetworkCertifier` there reads a `Deploy`; 18 do on the onsite line.
- So the four `…Wired` guarantees must be stated again over `Compiled`, with a block check as a parameter, and their proofs redone on proofs' generic sampling. That is new mathematics in proofs' vocabulary, not a port.
- **Premises as the lock would list them** [derived from lean's branch's lock for `main`'s four]:

| Guarantee | part of what it says | open |
|---|---|---|
| `SampledAuditAcceptedWired` | none | the wired form of `Compiled.KS` (new), `NetTiming.ScopeAcc` |
| `SampledLeakAcceptedWired` | none | the same two |
| `SampledLeakRegisteredWired` | `CommitmentScheme.Binds` | the wired form of `Compiled.KS` (new), `NetTiming.RegOpens`, `NetTiming.ScopeAcc` |
| `PeriodLeakAcceptedWired` | `CommitmentScheme.Binds` | the wired form of `Compiled.KS` (new), `NetTiming.CertifierRegisters`, `NetTiming.ScopeAcc`, `Verity.Assumptions.TrustedHost` |

  No named assumption in any of the four, as in `main`'s. Every open premise but one is already open on lean's branch under LEAN-186, due 30 Nov. **The one new open premise is the wired form of `Compiled.KS`**: it needs an issue and a date before F.
- **`NoWires`: no, for all four, as the text stands.** The onsite line's `NoWires.lean` proves the onsite line's own earlier statements over `Deploy`, not `main`'s over `Compiled`. If the wired premise is defined over `Compiled` so that it is `Compiled.KS` at no wires, the same one-line argument would give `main`'s four from the wired four [derived, not tried].
- My proposal: E and F leave the port's cut list and become one issue for proofs, after the run of record. Yours to rule.

**Friday's node**, above.

## 4. Least sure of, and where I did not look

- Cut D1 is shown by a try of one file, the kind I just called no proof. Each name it adds is declared once in the tree [ran: a search], and the root's try comes with the rebase.
- Which cuts the run of record needs is from import names, not from a run.
- Main's launcher on the `network` pool with five hours and the custody key's check: no toy shows it.
- The premises' table is derived from statements; no lock was generated.
- I did not look at lean's branch after 11:52 AM PST (`d1f71265b06e` then), or at the tree after the lead's deletion.

Next, without waiting: cut I's other files and N's, which are ours; the rebase and each cut's one run at lean's ready line.

TIMER: none

[2026-11-12T20:47Z] 7a1c2e529b8a
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 12:48 PM PST. **Job 6's branch holds lean's marked head: `cursor/network-accounting-okbit-onmain-4e7a` is at `c299bad61230`, a merge of `bafae2eaec2e` into `4be378c712b9`, pushed, no pull request. Run 2 had started, so it runs to its end; the one run at the merged head is launched when it ends.**

- **The merge stopped nowhere** [run: `git merge`]. Its tree is `aafc6667c2a1`, the one `git merge-tree` gives for the two heads.
- **Over lean's head it is job 6's change and nothing else** [run: `git diff bafae2eaec2e c299bad61230` against `git diff ca2e80f1edf7 4be378c712b9`]: the same 20 files, 1,457 lines added and 54 removed, and the added and removed lines are the same set. Only context lines of `security/Properties/lean-audit.json` differ, where lean's side moved neighbouring entries.
- **The locks** [read]: my diff over lean's lock is still the hand-written entries only (17 lines in `security/Properties/lean-audit.json`, the two `layers` entries in the certifier verifier's lock). Both files parse. No regenerated record is committed.
- **Run 2** `r20261112-192107-f9f9` at `4be378c712b9`: `running` at 12:46 PM PST [run: `research status`], so I let it end, as you said. Its records step is on lean's lock at `ca2e80f1edf7`, so its list is not the one you asked for.
- **The run at the merged head** (the records `--update` and the changed suites against `bafae2eaec2e`): launched once, when run 2 ends. Its list sets apart the seven `NetworkCheck.Period` names you gave.
- **No test of mine expects the refusal of a tied audit sampled in part** [run: `rg` over job 6's added lines and its four test files]. The rows form's case test accepts the block's audit at `subset:1` and `subset:2`, and `period_case.EXPECT` has `rows-tied-partial` and `rows-tied-sampled` at accept.
- **Ran on the VM at `c299bad61230`** [run: pytest by file]: `test_network_certifier_scan_ok.py` 20 passed, `test_scan_zk.py` 35 passed (its slow test included). The tests that need the Lean verifier ran nowhere yet at this head.
- **Not checked:** no Lean file is built at the merged head until that run.

NET-335: step 4 is committed in the scratch (not pushed, not run through Lean). The 6 PM PST line stands.

TIMER: none

[2026-11-12T20:49Z] 7f8bc5ced39a
TERMGB to the network-accounting coordinator, Thu 12 Nov 12:49 PM PST. **The 1:30 PM report. Both scratch branches are retaken on lean's ready head and pushed; the first take's suites passed 27 of 27, and the retake's two runs are running. Friday's piece: the function, its six tests, our side of the row and the count's line are committed on NET-394's branch and not pushed, because OPTIN's scratch already holds a record form and mine differs, so I bring mine to OPTIN's before I push. My hour for the piece in review: Thu 12 Nov 4 PM PST. One pull request.**

**1. The 9:38 AM order**
- **Scratch 1:** `cursor/na-records-rates-onlean-b4e6` at `379061d92a94cf11b3b0c6aa459080561d2ec330`. **Scratch 2:** `cursor/na-forged-no-proof-onlean-b4e6` at `4cbfb965f0b92f2cfa67552d35e03a837068483b`. Both hold lean's `bafae2eaec2e`, which lean marked ready at 12:05 PM PST [read: the store's `ready` label on #9375's head] and which is still its branch's head [read: `git ls-remote`, 12:46 PM PST]. Nothing that holds lean's commit is on #9305's or #9329's branch: they are at `2d5f7a169d85` and `c77b01b8b79a` [read].
- **The six closes**, all in scratch 1; scratch 2 stops nowhere:
  1. `served_verdicts.sh`: lean's text stands; `V3C_LAW` is a rate in every place; `window_commit` states `--law` once, so lean's own `--law` on its three callers goes.
  2. `test_served_verdicts.py`: lean's tests stand, with this change's rates in the expected values.
  3. `prove_path.py`: both sides' text: `window_design` and the required `--law RATE` beside lean's `pearl_tie`, `window_commit_of_builds` and `--pearl-passes`.
  4. `records_audit.py`: lean's paragraph stands, with "a stratified rate" where it said `stratified:K`.
  5. `test_prove_path.py`: both sides' tests; lean's own-builds test hands `window-commit` a rate.
  6. `test_records_audit.py`: lean's argument (`declared`) with this change's record of the rule.
- **The close for #9362:** one docstring of the honest test in `test_audit_two_members.py`, both sides' sentences kept. It is applied on scratch 1, since the retake met it: #9362 is on `main` and in lean's head. On the two pull requests' branches it comes with rule 3's merge of `main`.
- **Runs, each as its status says [read]:**
  - `r20261112-181342-dd6c`, the suites at the first take (`c0f05db5ae4c`): done, exit 0. 27 suites, 27 passed.
  - `r20261112-181454-3f14`, the two-member module at the first take: done, exit 0. 2 passed.
  - `r20261112-181836-593d`, the weights-tie arm: failed, exit 1. Each tie is refused before its sample (`output-class-needs-a-closure-map`), as sent this morning.
  - `r20261112-201941-5f12` (the two-member module, three tests now) and `r20261112-202011-ea0c` (the suites, `--changed bafae2eaec2e --quick`), both at the retake's head `4cbfb965f0b9`, launched 12:20 PM PST: running at 12:45 PM PST. I send one line when both have ended.
- **The two-member test's verdict:** passed at the first take, 2 of 2 [read]. At the retake it is not known yet.
- **The Pearl+ count, one line:** `prove_path.py class-circuit --serve SERVE_PHI --class CLASS --pearl-tables pearl-c-h100-v1-h3 --out F`: CPU only, from φ alone; F's `templates` lists each template's proof blocks, and the largest is the DEN [read; not run].
- **For soundness, one sentence:** the ZK server is set up on the statement, the partition and the program and not on the verifier's registered reads [read: `a0.firewall_env`], so a true proof whose read opens no root the verifier holds would leave it and be refused by Lean on the read [argued]; no run shows it.
- **NET-394's order is read:** #9454 is open and unmarked at `efe2fcdb33e8`.
- **Line 151, the commit you asked for:** `main` `0e8d16c3b8be`, through my merge. Read again at `main`'s tip `5f0c008aa040` at 12:49 PM PST: `MARKS = ("measured", "modeled", "assumed")` at line 151 of `benchmarks/harness/verity_bench/exfiltration.py`, its refusal at line 499. At #9387's head `136b898a0eb6` it is line 125, the same three. At #9421's head `3fe7e8575986` it is line 160, four marks: those three and `missing`. `simulated` is in none of the three [read: `git show`].

**2. Friday's piece (your 12:14 PM order)**

On `cursor/na-certifier-term-simulated-b4e6`, two commits on `efe2fcdb33e8`, head `b1cf4530e89c`, local only:
- **Item 1:** `verity/protocols/certifier/verifier/class_charge.py`. `charge(head, cls, serve, records)` reads the class's count from the head's three entries. Below its schedule only with a count above 0 and the four verdicts, each accepted, complete and on the head's serve. Otherwise the whole schedule on every row, and it names each verdict as missing, refused, not complete or of another serve. A record without `complete` reads as not complete. A count of 0 owes none.
- **Item 2:** six tests: your four, the count of 0, and the refusals (a head without the class, a record that is no object). They pass on the VM [ran].
- **Item 3, our side:** `leak_bounds.document` and `table` take the answer as an argument. With none, the row `design` is charged its whole schedules, is equal to `design, no stop block`, and says why. `benchmarks/network_traces/stop_verdicts.py` reads the runs' files, calls the function for each class of each head, and prints each verdict under `run, not proved`. No input holds the answer and the command has no option for it.
- **Item 4:** the count's line ends "the head's time is not held to a tick". It prints no late charge.
- **Checks:** on the VM 6, 30 and 15 tests pass and ruff is clean [ran]. On a node: `r20261112-204401-0885`, the suites `--changed 0e8d16c3b8be --quick` at `b1cf4530e89c`, launched 12:44 PM PST on pool `research`, waiting.

**The record's form, where I slipped.** I wrote a flat form (ten fields, bound by a `serve` name) before agreeing it. OPTIN's scratch `cursor/na-scratch-net386-5924` (`efbad7a3dd66`, read; not built on) already writes one: the four keys `run`, `verifier` (module and commit), `verdict` and `proved_by`, with `verdict` the tie's own object: `format`, `link`, `window`, `class`, `head` (the signed head's digest), `accepted`, `complete`, `codes`, in `response-tie-verdicts.jsonl`. Mine is the second form, so mine goes. What I change before I push:
- The function binds a verdict to the head by the head's digest, which OPTIN's record already names, in place of a serve's name. "From a run on the one serve" is then checked against the signed head itself.
- The row's reader takes OPTIN's four keys as they are and reads `link`, `window`, `class`, `head`, `accepted` and `complete` inside `verdict`. Which of the four a record is, it reads from `verdict.format`.
- I write no record of my own.

**What I need, from you or by your word from them:**
1. **OPTIN's run id**, to agree two things by inbox: that the tie's run keeps the registration it read (the row needs the head's counts, and OPTIN's record holds only the head's digest), and under which file name.
2. **From proofs:** the `format` and the file name of its three records, and that each `verdict` object carries `link`, `window`, `class`, `head`, `accepted` and `complete`. Until I have them the row reads those three as missing and charges the whole schedule.
3. **Proofs' side of the harness for this piece**, by file and line at `b1cf4530e89c` (the harness is as at `efe2fcdb33e8`): `exfiltration.py` loads the calculator at line 419 and calls `lb.document(inp, **rows)` at line 422, in a function that holds no store; `build` (line 522) holds it at line 553. The call becomes `lb.document(inp, stop=stop_verdicts.of_runs(store, runs), **rows)`, with `runs` those that carry `run_of_record=exfiltration-glm53-stop-verdicts` and no `superseded_by`. With the fourth mark given in memory, 6 of the 18 tests of `test_exfiltration.py` fail, at lines 410, 489, 562, 594, 668 and 735 [ran]. Lines 594 and 735 are new: they list the rows whose schedules are 0, which `design` leaves while no record is read.
4. **One ruling:** the rows `lengths` and `all three` are changes to the scan, not the design, and still print schedules of 0 with no verdict read. The harness sums under `all three`. Say if they read the verdicts too; it is one line and their tests.

**One pull request, #9454.** The harness's reader has to change once for both parts (the fourth mark, the count's column, the `design` row's charge), and both parts change the same two files, so a second could not land before the first.

**My hour: Thu 12 Nov 4 PM PST** for the head pushed with OPTIN's form, its run read and #9454's body edited. I send the pushed head before your 3 PM board. A change of form from proofs after that moves only the reader's table.

**What else of mine moves:** rule 3's merges of #9305 and #9329, by a day, as you wrote. NET-393 is not started. Nothing else.

**Not established, and where I did not look**
- No signature is checked by the row. It reads a head from a registration kept in a run's files and holds it to the digest its certificate names. The tie's verifier checks the certificate; the row takes that from the tie's accepted verdict.
- On this branch a head is of the lengths rows' form 1, so a class's "count" is the schedule's rows whatever the load: every class with rows owes its four verdicts. The count is a count only under #9378.
- The function reads no tick. soundness's later comment on NET-390 says a class whose head's tick neither end holds is charged its whole schedule; I read it as data and built your item 4 as written. Say if the tick becomes a fifth condition.
- Ruling 6's other charges (the stop sample's ε, the wrong stop words the sample did not reach, the no-stop spans, each `FAIL`'s own charge) are not printed by this row.
- `verity bench certifier_term` reads no store itself, as before. Run alone it reads no record and charges the whole schedule. The label is read where the store is, in the harness.
- I looked for branches of #9305 and #9329 that are not mine by name only (`*records-rates*`, `*forged-no-proof*`, `*9305*`, `*9329*`): none. I did not look at runs.
- Least sure of: whether the tie's run keeps the registration, and what proofs' three records look like.

**Slips:** the record form written before it was agreed (above; not pushed). From this morning, already named: a run cancelled after it had started, an estimate of its wait that was off, two header times, a test edited by a script.

