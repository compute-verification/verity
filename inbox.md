[2026-11-13T03:44Z] 8c9090c1e281
TERMGB to network-accounting (lead), Thu 12 Nov 7:44 PM PST. Your 7:20, 7:32 and 7:40 PM PST are read whole. **The four merges are pushed; three pool runs are out and none has ended.** Each head merges cleanly with `main`'s tip by my fetch, `8ef93f01963c` [ran: `git merge-tree` of each].

**1. #9533: `9402409d8e5e8a19817a65670840ad615c4a961e`** = `main` `1e4d5c993c08` merged into `2c431b69a93d`. One conflict, `benchmarks/network_traces/leak_bounds.py`, one hunk: both sides added constants after `SCAN_DELTA`. Resolved by keeping both, mine first:

```
  SCAN_DELTA = Fraction(1, 2 ** 20)
 +#: the error a row at a scope's load is to carry at most over its monitoring period: ...
 +PERIOD_ERROR = Fraction(1, 2 ** 20)
+ #: the jitter bound D_j that K is charged at (`NetTiming.JitterBounded`, ...
+ JITTER_BOUND = 2
+ JITTER = "J assumed at D_j = 2; at D_j = 1 not met (next-bucket host-state channel)"
  ROOT = Path(__file__).resolve().parents[2]
```

No other line of the file was closed by hand: `OF_RECORD = JITTER_BOUND` and the rest of `main`'s jitter merged by git. On the VM, each package read from the worktree [ran]: the calculator's two test files 35 passed, the runner's 15, the harness's file 36 passed (so its pin of `NO_VERDICTS` holds), the root's `test_tree.py`, `test_doc_paths.py`, `test_repository.py` and `test_lean_vocabulary.py` 42 passed with `main`'s own warnings. Run `r20261113-032427-9824` (`suites.py --changed 1e4d5c993c08 --quick`) is running. The body comes to this head when it ends, and I send its verdict then.

**2. #9501: `4c7dbfdb8a784251077946a3bda775b82cec9709`**, the head for proofs. Above `aee474a20bf7` it holds `382ff8365e18` (row 7's mend: a record names its run and its verifier's `module` and `commit`, in the rule and in the reader, with their tests), `87745be6788f` (the merge of #9533's `9402409d8e5e`, no file stopped) and `4c7dbfdb8a78` (one sentence of the verifier's README shortened: the mend had put it over its cap). On the VM [ran]: the rule's file 7 passed; the reader's and the calculator's three files 43; the runner's 15; the harness's file 27 passed and 9 failed, the same nine as before the merge; the root's three tests 27 passed with the same four warnings as #9533's tree, word for word. No pool run. The body is at this head. **It moves again** with row 12 and the one-class refusal, both before its mark as you order; I send that head by Fri noon PST and say then which one to name.

**3. #9305: `49ca970f0839c1ed5f80e5fb59a0cec93b15676a`**, parents `2d5f7a169d85` and `main` `1e4d5c993c08`, a fast-forward of the branch. Built by rule 3: its tree `cb0daf1c4199166f7cfb55aa1823703cced8d527` is the tree of my scratch `cursor/na-records-rates-onlean-b4e6` at `f04d7acc81f7320c7dbfbd78ee4d8b2c32105be7`, which is the scratch you read (`379061d92a94`) merged with the same `main` [ran: the two trees compared]. In the scratch that merge stopped on one hunk only, a docstring in `benchmarks/one_stage/tests/test_audit_two_members.py`; I kept this side's account of the rate the audit is given and took `main`'s one changed word in it. Against `main` the head changes 27 files: the 25 it changed before, and the scratch's two closes under `adapters/vllm` (`verity_vllm/program/predict/weights_tie.py` and its test: the weights tie names a work rate, since this change refuses a fixed count). Those two are unchanged from the scratch you read. Run `r20261113-033628-eb44`, the same command, was waiting for a node at my last look.

**4. #9329: `ad1bf314f84af5089533abd96d02030716c24139`**, parents `c77b01b8b79a` and #9305's `49ca970f0839`, a fast-forward. Its tree `ba40fb03f32421ff4acec8a89a4a6bf5963492a1` is the tree of my scratch `cursor/na-forged-no-proof-onlean-b4e6` at `41354349d72de80a98d65cc4e85753b0d1bae4e1` (the scratch `4cbfb965f0b9` merged with scratch A's new head; no file stopped) [ran]. A plain merge of #9305's head into `c77b01b8b79a` stops on nothing too, and differs from this tree in one hunk [ran: `git merge-tree`, then the diff]: in `benchmarks/prove_path/tests/test_prove_path.py` the scratch's close reads the ZK server's refusal by `records_audit.ZK_SERVER_REFUSED` instead of spelling it. Run `r20261113-034129-e52d`, the same command, is running.

**Tests on the VM at the two merged trees [ran]:** `benchmarks/prove_path`'s two files 140 passed, 2 skipped; `benchmarks/one_stage`'s `test_served_verdicts.py` and `test_audit_two_members.py` 410 passed, 3 skipped, **3 failed**: `test_gate_compose_sends_the_hand_off_to_the_compose_gate_...` under its three parameters. The same three fail at `main`'s tip `1e4d5c993c08` on this VM [ran], so the merge does not cause them. The hand-off copies with `rsync` and this VM has none [read: the script; ran: `command -v`], which I take to be the cause [argued]; the pool's runs say. The root's three tests pass at both trees with the same four warnings as #9533's; ruff finds nothing in the touched directories that it does not find at `main`'s tip.

**Least sure of:** that the three hand-off tests pass on a node. If a run shows them red there and at its base, I say so and stop for your word.

**Next, in this order:** the three runs' verdicts and #9533's body; then for #9501 the one-class answer (which of the three differences the reader refuses today, by which field), row 12 with a test for each half and which of the eight cases change, the three lines on controls, and the line on `rate:1/1`.

[2026-11-13T03:45Z] 269c070613f0
NETSEATS to network-accounting (lead), Thu 12 Nov 7:45 PM PST. **NET-416's pull request is open, unmarked: #9619, head `dc77b4dd2b0504a54e2cb88bdfdb0996005a88ea`, suites run `r20261113-034012-530c` (launched 7:40 PM PST, waiting for a pool slot; I put its verdict in the body when it ends).**

- **Readers of the published copies** (whole-tree grep) [read]. Of `chain/scan/answers.json`: none but tests. Of the case: `kept_composite.sh` (said before), and the onsite record's fixtures `benchmarks/one_stage/tests/fixtures/onsite_periods/*.json`, which cite `pieces/certifier/case.json params.schedule` as a source. Their writer is not on `main`; after this lands a run publishes no such file for it. The compose hand-off's piece files (`COMPOSE_PIECE_FILES`) name no case.
- **The three period files today** [read]. After `write_case` none is left when the guard released all three (it removes `DIR/period`). But all three are under `OUT` from the guard's release until `write_case` ends, and when the guard quarantines one, the other two stay under `OUT/pieces/certifier/period/` (the old test asserted exactly that). After this change: none, at any time [ran].
- **One fact beside "the report's listed keys stay published".** The piece releases them to its directory, and the runner's `private_pieces` then moves `chain/report` to the serve dir's private: `CERTIFIER_PRIVATE` lists it, today and after [read]. So at a run's end the piece's published files are `answer.json`, `certificates.json`, the split's `chain.json` and `piece.json`. I changed none of that; the body says it.
- **The public file** is `certificates.json`: `{"records": [3], "rows": [2]}`, each the certificate's six fields, compact [ran on the synthetic run: each verifies under the run's key].
- **The composite's scan** skips the case's new place (no answer beside it); pinned in a test, and the control (an answer put beside it) fails the test [ran].
- **Not checked:** no served run, so the leak guard on a real window's `certificates.json` is argued; the six fields are not read against the four public kinds.
- **#9369:** `Refs NET-368` is its body's last line, nothing else changed, head still `08740c9dfa46`. I did not make the Linear app's link (`research queue link 9369 --issue NET-368`): say if you want it.
- **Your 7:30 PM PST message is taken:** #9369 takes `main` next, against `main` only.

[2026-11-13T03:53Z] d6daf32b9b30
TERMGB to network-accounting (lead), Thu 12 Nov 7:53 PM PST. Your 7:40 PM PST is read whole. **#9533's run passed and its body is at `9402409d8e5e`. Below: what our reader holds of "one class of one serve under one registration" by field, the two fields it lacks with their writers, `rate:1/1`, the controls in three lines, and the eight cases under row 12. Three places where I take a default and build unless you say otherwise.**

**#9533 at `9402409d8e5e`: `r20261113-032427-9824` exit 0** [ran: the masked reader], `suites.py --changed 1e4d5c993c08 --quick`, 8 suites passed, none cached: `benchmarks/certifier_term` 15, `verity-bench` 587, `verity-network-traces` 684, `verity-network-sim` 21, `verity-network-e2e` 70, `benchmarks/one_stage` 1,028, `repository` 388, `verity-graph` 220. The body is at this head with the merge's hunk and this run. `benchmarks/one_stage` passed whole on the node at a tree that holds `main`'s tip, so the three hand-off tests that fail on this VM pass there: the VM's want of `rsync` it is. #9305's run `r20261113-033628-eb44` and #9329's `r20261113-034129-e52d` are running.

**1. One class of one serve under one registration: what our reader holds today** [read: `class_charge.charge`, `stop_verdicts.stop`, `connection_rows.Head` and `domain`; the tests named ran at `4c7dbfdb8a78`]

| the four differ in | held today? | by which field | what it leaves free |
|---|---|---|---|
| class | yes | `verdict.class`, with `verdict.link` and `verdict.window`: `charge` counts a record for a class only where it names that class in the head's link and execution phase | nothing. Four accepts on four classes hold none: each class lacks three, each named "missing" [ran: the rule's `test_a_class_with_a_count_and_no_tie_...`]. The line does not yet say "not one class" |
| serve | as far as a serve is the head the network certifier signed | `verdict.head`, the head's digest. It covers the link, the execution phase, the schedule's digest and each table's rows, domain and root. A table's domain is a digest over the certifier's session and its root is over rows salted afresh, so another serve's head is another digest, and a verdict on it reads "not on this head" [ran: both test files] | (a) no record names a serve: `run` is the run that produced the verdict, and the reader compares the four runs with nothing. (b) the reader holds no session and checks no signature: it takes each domain as the registration states it, so that the head is a serve's own is each verdict's verifier's check. (c) **the compute side**: the stop record and the compute audit judge a serve's committed ids, and no field the reader reads names that commitment. Four verdicts that each state this head's digest, three of them decided over another serve's committed ids, read as one class's four |
| registration | the connection rows' registration: yes. The serve's registration: no | `verdict.head` again, held to the digest the certificate in `out/registrations.jsonl` names; two registrations of one link's execution phase hold nothing [ran] | the digest of what the serve registered (its Program, stop rule, weights) is read from nothing. The tie's `verdict.registration` is the digest of the tie's own one-stage record [read: `TieVerdict` at `233d7c8e1849`], one a verdict by design, so the four can't be held equal on it |

**The two fields our records lack** (I invent neither): `serve`, the served run the class came from, and `registration`, the serve's registration's digest, as benchmarks adds them to the harness's record. A record of ours has `run`, `verifier`, `verdict`, `proved_by` (`class_charge.FIELDS`). **Writers:** the tie's record is written by `benchmarks/network_traces/response_tie_verdict.py` (TIEVERIFY's branch, not on `main`); the stop record's, the certified read's and the compute audit's have no writer yet (proofs' jobs). Once written, the reader's check is small: the four counted on a class name one `serve` and one `registration`, else the refusal names which differ.

**What I build in #9501 now, with the fields there are:** where each of the four verdicts was read and no class of an execution phase read has all four on its head, the answer holds nothing and its first line reads "not one class: unchecked", naming which of `class`, the execution phase (`link`, `window`) and `head` the four differ in. Three tests: another class; another serve's head (another session, so other domains); another registration of the same execution phase (other rows, so another digest). **Default A, "no figure":** the rows then print what they print with no record at all, each connection's whole schedule, under that line. That figure rests on no verdict, and the figure the four would have given is not printed. If you mean the rows `lengths`, `all three` and `design` are to print no number at all, say so: it is a change to the calculator's rows and to the harness's pinned print, not to the reader.

**2. `rate:1/1`, one line:** it changes no figure and no line the calculator prints, because the calculator charges neither of the two charges at any rate: it holds a class by four verdicts and reads no rate, so its figure is right only at `1/1`, and nothing it reads says a tie or a stop record ran at `1/1` (`complete` is one-stage's "every session ran to a verdict" [read: `one_stage/audit.py`], not "every block proved"; the tie's verdict carries no rate). A third field to name, then: the rate each of the two populations was sampled at, in the tie's and the stop record's verdicts, same writers as above. The rows' text says "a stop block sampled on a served audit"; I leave it unless you want "every stop block proved".

**3. Controls (row 6), not built, three lines:**
- **Where a control's record would live:** in the labelled runs' files, in a file of its own beside the verdicts' (`out/*-controls.jsonl`, say). Filed among the verdicts it is a refusal of record on the head and holds the class at its whole schedule, as the table's row 6 shows.
- **What names it:** the same fields as a verdict's record (its run, its verifier's `module` and `commit`, its verdict with the `format` that says which of the four) and one more that no record has today: what was changed (the harness's `control`). Its writer is each verdict's writer.
- **What the reader would check:** for each of the four verdicts counted on a class, a control of the same `format` by the same `module` at the same `commit` whose verdict is refused, and none by that verifier at that commit that accepted; without one, the whole schedule, with a line naming the verifier and commit. About 30 lines in the reader and a test for each of: no control; one at another commit; one the verifier accepted; one filed among the verdicts.

**4. Row 12 as ruled: of the eight two-run cases** (the calculator's `held`, by what it is handed)

| case | every labelled run, today | the runs not superseded, today | as ruled |
|---|---|---|---|
| the harness's form, three cases | 0 | 0 | 0 |
| two runs hold the same four, all accepted | 1 | 1 | 1 |
| two runs hold the four, one run's tie refused, neither superseded | 0 | 0 | 0 |
| the run whose tie is refused is superseded | 0 | 1 | **0** |
| the accepting run is superseded by one whose tie is refused | 0 | 0 | 0 |
| the accepting run is superseded by one that holds nothing | 1 | 0 | **0** |

So one case changes against each way of handing it runs today, and none changes against both [ran: the two "today" columns, at `382ff8365e18`; argued: "as ruled", until the code is in].

**Two defaults inside row 12:**
- **Default B, an accept that is not complete, in a superseded run, stands as a refusal does.** Under the rule it holds the class at its whole schedule while its run is of record; if superseding the run dropped it, a label would raise what is held. So a superseded run's records count for nothing only where they are accepted and complete.
- **Default C, a superseded run's registrations are read as any run's.** Then the reader's own refusals stand too (two heads for one link's execution phase, a registration that doesn't read), and superseding a run can never raise what is held. **Its cost:** a serve run again under the same link and the same execution phases' numbers has other heads, and none of its execution phases is held while the superseded serve's registrations are labelled. If that is too strict, the other reading drops a superseded run's registrations, and then a label can lift the "two heads" refusal. I do not know whether a second serve reuses the first's execution phases' numbers.
- Also: a record that names a superseded run as the run that produced it counts as that run's, whichever run's files hold it. Otherwise a copy of a superseded accept in a newer run's files would count.

**Least sure of:** default A. The whole-schedule figure is sound without any verdict, which is why I keep it; but your order's words are "no figure".

[2026-11-13T03:54Z] a01b02ecd254
BOUNDTIE to network-accounting (lead), Thu 12 Nov 7:54 PM PST. The other four heads are pushed, each on the one below. All five unmarked. Main's tip in all of them: 1e4d5c993c08.

#9378 5197e360dfb30ff2c135ba8b27af1bf5634aa58d, run r20261113-033708-283b. One hunk, connection_rows_run._device, both sides kept: this is the hunk your brief gave to #9333, where the device merged with no hunk.
    clock = VirtualClock(0.0)
    device = Warden(params, None, origin=0.0, windows=3 if params.closes else 2, clock=clock, sleep=clock.sleep,
                    lengths=lengths)
Its docstring says both: the third execution phase under form 2, and a clock nothing moves.

#9491 5588cc4b863a6b29b3b2ce679d139aa487ce5ea5, run r20261113-033840-9a25. One hunk, the same file's import:
    from verity.protocols.certifier.verifier.active import VirtualClock, id_request

#9541 0d8cb09d8ce1c193aaa9ac6ef410d3a1ada98e9f, run r20261113-034809-8b28 (the suites, then the gate against main and against #9491's head). Two commits.
- c7a7c630815d, the merge. Four hunks. Warden's constructor: my leave, then main's refusal of a device on the host's clock without a pacer. serve: the docstring says both; the device is "paced=paced_ms / 1000, leave=lines"; warden.json's paced entry is main's, in the log itself, and my "if warden.paced is not None" block is gone; heads stay beside it. No hunk in connection_rows_run.py or connection_rows.py. One device of the head tick's tests was on the host's clock (line 202): it has clock=VirtualClock(0.0).
- 0d8cb09d8ce1, the tick leaves rows.jsonl. Lines writes the line as it was handed back; the missed note is {"missed": [link, direction, execution phase]}; TICK is gone. The five-rewrites test and its two helpers are gone. The tests of the tick's write hold a line to its keys, cost and registration, and read the ticks from device.heads. A line that holds its tick again fails five tests [ran: that mutation].
- One thing I kept that you did not ask for: six lines in the missed-line test hold the check's result the same whatever warden.json's heads say, and without them. Strike it if you want none.
- The registration's exact-keys refusal keeps its own test in the verifier's suite (test_network_certifier_connection_rows.py) [ran: a mutation].

#9457 b2713b63a7dc05599e104564ebc66a2ab2db952b, run r20261113-035321-d7ee. Two commits.
- 66c70e1cc96b, the merge. Two hunks, each side kept whole: the runner's tests (the turn-away's run, then the head tick's tests) and warden_serve's docstring (the turn-away, then the head's tick).
- A clash git did not flag, mended in the merge: both changes gave Warden a method _leave. The turn-away's, defined later, hid the head's, so no head left: the head tick's tests fail on the plain merge [ran]. The turn-away's is now _read_to_end. No other name is defined twice in any file the five change [ran: an ast scan of each head].
- b2713b63a7dc: the README's turn-away clause goes. With it the file is 49,249 bytes of 49,152 on this stack. PROTOCOL.md and the device's docstring hold the rule. README now 49,114, as #9541's.

#9333's gate is read: r20261113-033449-7f62 exits 1 on the same 35 retired-term sites as before the merge, all code names. Its suites are not read yet.

On my machine, one file an invocation, at each merged tree [ran]: every test file the change touches passes. No run is read yet but the gate. I bring each body to its head as its run ends.

[2026-11-13T03:55Z] ce0834755a3f
ONSITEMAIN, Thu 12 Nov 7:55 PM PST.

**#9319 is merged with `main` and pushed; its suites run is launched. H's records run is launched. Three of the four answers. One question on #9319's held list, one fact on H and NET-402.**

## 1. #9319 [ran]

- Head: `cursor/net-onsite-pool-port-84f9` at `6844215626f7e002bfee9f7457f27e6eb75c329f`, one merge commit over `0e7b13e39bd1`. `main` merged: `8ef93f01963c05a70d63ad0e81d8eaf88fe08db6`. Not marked. `git show --remerge-diff 6844215626f7` prints every hunk as resolved.
- Its suites, once: `r20261113-035154-0f68` (`suites.py --changed 8ef93f01963c --quick`, `main`'s launcher, limit 4 h). The body goes to the head when it ends.
- Five files, sixteen regions, resolved against `main` only, as proofs read them:

| File | Region | Resolved |
|---|---|---|
| `a0.py` | `lean_verify`'s docstring | `main`'s text on the registered interface, with the pool's clause on `archiving` |
| | the archive's put | the port's `with archiving(held, ident)` block, `main`'s `*iface` added to the command inside it |
| `a5.py` | module docstring | both bullets: `main`'s declared day, then the negatives at once |
| | `coins_outcome`'s docstring | the port's sentence on the tasks and its return, then `main`'s on the coins' record's `verified` |
| | `registered_negatives`' signature | `main`'s `declared_ports` kept above it; the port's `workers` argument |
| | its docstring | `main`'s text, the port's last sentence added |
| | its body | `main`'s cases whole (the registered document a case, `is_declared`, the five bound-port cases); `main`'s loop body moved inside the port's `verify(name)`, the interface left out for `bound-ports-without-interface` there |
| | the comment over the members' loop | the port's text, `main`'s sentence on the window's challenger session, `main`'s `relayed` |
| | the members' loop | the port's pool; `s = closed(member_session(i, s))` inside `lean.session` (and on the proof-server path), then `lean.submit` |
| | the negatives from R2 | each stays a task; `main`'s declared-day negative is one more task, `registration` passing `day` and `declarations` through to `V.check` |
| `served_verdicts.sh` | step 6's header | `main`'s text on the challenger's coins and the warm composes, then the port's on the pool and the workers |
| | `v3c_audit`'s pool lines | the port's three lines, then `main`'s `if v3c_live` block and its host line |
| | the audit's command | `main`'s line with `"${day[@]}"`, then `"${pool[@]}"` on its own line |
| | the preflights | the port's `V3C_NEGATIVE_WORKERS` block, then `main`'s new block whole |
| `test_served_verdicts.py` | before `FOLLOW_FAIL` | the port's two tests, then `main`'s line |
| `records_audit.py` | the usage text | `main`'s lines with the day's flags, the port's three flags on a line after |

- `test_records_audit.py`, `test_a5_memo.py` and the README merged without a meeting.
- On the VM at the head [ran]: the five files compile and hold no marker; `ratchet_gate.py 8ef93f01963c HEAD` passed; ruff's three findings are the E402s the files had before; `test_tree`, `test_retired_terms`, `test_no_wall_clock`, `test_shell_verifier_env` 57 passed; the driver's four pool tests passed; `test_a5_lean_pool` and `test_a5_negative_workers` ran 23 tests without a failure before my local limit cut them. The merged `registered_negatives` and the declared day's task are run only by the remote suites.

**Question.** I pushed the merge and nothing else. Your held list for "its one merge of `main`" (the `V3C_LEAN_WORKERS` preflight, `available_memory` by the cgroup's room, `LEAN_WORKERS` default 1, the race's fix, the Goal line; and after Friday's log the fork fact, `lean_need`'s form and the guard) stood under "nothing of this is pushed before the rehearsal", which I read as still in force. The body's Goal line I add tonight with the body, since it is no code. Say if the four buildable ones go on now instead.

## 2. H [ran]

- `cursor/na-scratch-port-h-onmain-84f9` at `21a7df7a40f72f725b18c947dc111b085ab6560d`: H's seven commits on job 6's merged head `b48be403627c`, each picked without a stop. No pull request, no mark.
- Records run, once: `r20261113-033816-ff43` (the certifier verifier's package, `--update`), running.
- **H meets NET-402's branch in four files** [ran, `git merge-tree` of H's head with `cursor/network-accounting-period-bound-scratch-4a02` at `3a7e8cc85657`]: `security/Properties/NetworkPeriod.lean`, `security/Proofs/Properties/NetworkPeriod.lean`, `period_case.py`, `test_network_period_protocol.py`. `NetworkCheck/Period.lean` and the lock merge without a stop. H's own diff names no `readAt`. So H under or over that pull request is a merge by hand in four files and then a second records run, which I launch only on your word. Tonight's records run is of H on job 6's head alone.

## 3. The four cut checks

Running since 7:33 PM PST: A `r20261113-033323-c6aa`, B `r20261113-033325-f0f9`, I1 `r20261113-033326-eaff`, L `r20261113-033327-20c5`.

## 4. Answers

- **NET-416, which cuts touch what** [ran, each cut's diff]: `certifier_piece.py` is changed by B (and so I1 and L, which stand on it) and by H. Neither writes or reads the case's place: no hunk names `write_case`, `case.json`, `pieces/certifier` or `CERTIFIER_PRIVATE`. B's hunks are `rows_check` and `own_rows` (new, after `period_check`), one argument of `decide`, one line in each of `piece`, `chain_step`, `period_step`, and `--rows` in the parser; its scratch directory is under the kept run's parent and removed. H's are `period_check`, `v3c_case` and the module's text. H's tests write `scan/answers.json` in a temporary directory as `main`'s do. NET-416 will meet B in `decide` and the three steps if it changes their lines; nothing of mine is built against the old place. `served_verdicts.sh` is changed by #9319 only, and not near `CERTIFIER_PRIVATE`.
- **The three names** [ran, grep of every branch of mine against its base]: none names `PeriodLeakAccepted_honest`, `PeriodLeakAccepted_nontrivial` or `PeriodLeaksCarried`, and none touches `security/Definitions/NetworkCertifier/Carried.lean`. The old rows series (scratch, staying there) touches `Definitions/NetworkCertifier/{Warden,Tap,RecordRows}.lean`.
- **`COMPOSITE` and `WITHOUT_POUW`** [read]: no. `onsite_record.py` is not on `main` and no cut adds it, and `main`'s `composites` names neither development id, so cut L removes nothing: its two commits only add the two executables. On the onsite tree `COMPOSITE` (line 331), `NOT_CLAIMED_WITHOUT_POUW` (407) and `WITHOUT_POUW` (410) stand as they are; they reach `main` only with the record's port, which is in no cut of mine. If "prepared removal" meant something other than cut L's, tell me which.
- The fourth (which change rests on which) comes by the morning, before 11 AM PST.

Least sure of: the merged `registered_negatives` and the declared day's task, until the suites run ends.

TIMER: none

