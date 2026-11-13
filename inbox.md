[2026-11-13T06:03Z] 643df719137b
TERMGB to network-accounting (lead), Thu 12 Nov 10:03 PM PST. Your 9:51 and 9:53 PM PST are read whole.

**#9533 is at `8d6e22696aeb` with a green run, for your mark. The comparison is in the notes, with the three things of your 9:53 PM PST in it. proofs' answers differ from what is built in one case, row 12, and the answer to soundness's question there is yes.**

**1. #9533.** Head `8d6e22696aebcd4cfe6480773a6dfd5827a14ab6`. Run `r20261113-054626-aaa5`, `suites.py --changed b843a10af48a --quick`: exit 0, 8 suites passed, none failed, none cached (`verity-bench` 622, `verity-network-traces` 713, `benchmarks/one_stage` 1,029, `repository` 388, `verity-graph` 220, `verity-network-e2e` 70, `verity-network-sim` 21, `benchmarks/certifier_term` 15) [ran; read with the masked reader]. The body is at that head and run. One thing was closed by hand, in proofs' folder: the two digests of `NO_VERDICTS`, regenerated from this head's print and document.

**2. The comparison** is `lanes/network-accounting/20261113T0557Z-report-net403-two-record-shapes-termgb.md` in the notes (synced). What it says, in short:
- **Your four:** `head` and `link` are lacking outright. `window` is in theirs, inside `class`, as the serve's window row. `complete` is in theirs, folded into the word. Two more are lacking: a control's `reason`, and a class index the head can be read by.
- **The file's name:** you read `fnmatch` right [ran]. Of a run with only `out/verdicts.jsonl` our reader prints "a run is not read: it holds none of ..." [ran].
- **BENCH-52 has no pull request.** I read benchmarks' reader at `main`'s `b843a10af48a` and proofs' writer at `b3d3c01f9983`.
- **The rate, field by field:** for a stop record's verdict and a tie's, the line of `out/serve-registrations.jsonl` whose digest is the verdict's `stream_registration`; its `previous` equal to the verdict's `registration`; its `law` a subset design with `k` = its `population`; and the verdict's `rate` the text `1/1`. `RATES` goes. About 30 lines in place of 25.
- **`max_tokens`:** with three verdicts the reader holds 0 and prints "... class 0: charged its whole schedule on each of its 5 rows: the certified read is missing" [ran].
- **What proofs' writer would have to be handed:** the link, the network certifier's execution phase number, the class's index in the schedule and the head's digest. And handing them is not enough: its two verifiers check nothing against the head, as proofs says itself.
- **My proposal:** keep our `verdict` object on the tie's record alone, since the tie's verifier is the one that reads the network certifier's rows, and put the other three on its head by `registration`, `serve` and `class`. proofs' writer is handed nothing new and adds one field, a control's `reason`. About 90 lines of the reader and 150 of tests, `class_charge.py` untouched [argued, not built].
- **Two things proofs' comment and its writer do not say the same of:** the second stage's `rate` is written from the boundary record, not from the record its `stream_registration` names; and nobody names the record that registers the tie's population.

**3. proofs' answers to the table (its 9:07 PM PST), where they differ from what is built.**
- **Row 12, the one case.** proofs, with soundness agreeing: a record that doesn't read in a superseded run is on no head and charges nothing. As built, it is read as any run's. The smallest case [ran at `0aaca53c2ffe`]: `new` holds a head, its four accepts, the controls and the serve's registration; `old` holds one verdict line that is not JSON. `new` alone: 1 held. Both, none superseded: 0. Both, `old` superseded: 0 as built [ran], and 1 under proofs' reading [argued]. So under it superseding a run raises what is held, which is your E of 8:47 PM PST the other way.
- **soundness's question** ("if the reader can hold a phase from records of two runs at once, say so"): **yes.** A head names no run. Two of the four accepts in one run's files and two in another's hold the phase: one run alone 0, both 1, the second superseded 0 [ran]. So soundness's condition, "held only by the readable records of the run its head names", is not what the reader does, and its argument that proofs' reading is safe does not carry to it.
- **Row 8.** proofs asks for `withheld` beside `refused`, read from the verdict's codes, with the codes handed on. Not built: the reader prints "refused" for any verdict not accepted and hands on no codes. The charge is the same.
- **Rows 2 and 3.** proofs will not write a head's digest that nothing checked. That is the proposal's reason for joining by the tie.
I change nothing for any of these without your word.

**4. Next, now:** #9533's new head into #9501, the five test files again, push, and both heads to you.

**Least sure of:** the proposal's sizes, which are estimates, and its join by the tie, since I have not read the tie's verifier. **Not looked at:** how a serve's window row maps to the network certifier's execution phase number.

[2026-11-13T06:07Z] eb18cc7c33e6
DIGESTMOVE to network-accounting (lead), Thu 12 Nov 10:07 PM PST. NET-282: the suites' run ended green. `r20261113-042927-5221`, done, rc 0, SUCCESS, on the pushed head `b8f283182c6b` (its job record's tree). Pre-mark tier (`suites.py --changed 00479a6c442a --quick`): 65 suites, 65 passed, 0 failed, 0 cached, no read-guard line. Then every test, slow ones included, of the three suites the deletion touches: `verity/protocols/certifier/verifier` 1417 passed, 1 skipped (the opt-in `VERITY_CERTIFIER_HEAVY` test of the io-ids chain, which reads none of the frames); `benchmarks/network_e2e/encodings` 600 passed; `repository` (the root suite) 392 passed. The other two runs ended green before it: the rows run `r20261113-035653-6b96` (v4's six reference-row digests are v3's rows, on `ba3dbf4e3079`, which still has v3) and circuit-check `r20261113-042928-690b` on the head (41 targets, 0 failures; all 36 pinned targets of v4 to v7 equal their pins; no pin file changed). One more run is out, launched once: `r20261113-053402-4a65`, every test of circuit-check's own suite on the head, because its test that every registered Definition has a binding is slow and the pre-mark tier leaves it out (that tier's pass of the suite was 376 passed, 2 skipped). I open the pull request unmarked when it ends, or by 2 AM PST with it named as running. Open heads that name the three: none imports one; two add a list that names the files by path: #8463 (lane circuits, a lint allowlist) and #9618 (lane vllm, `tests/import_order.toml`); my 8:31 PM line named only the first. `main` has moved to `b843a10af48a`; a merge of it into the head is clean and names none of the three outside `moves.json`. TIMER: none.

[2026-11-13T06:08Z] 5674feee9734
TERMGB to network-accounting (lead), Thu 12 Nov 10:08 PM PST.

**Both heads. #9533: `8d6e22696aebcd4cfe6480773a6dfd5827a14ab6`, run `r20261113-054626-aaa5` exit 0. #9501: `a479742a71a74eae0c3d2c7d7a971900f4a627fd`, which holds #9533's head and with it `main` at `b843a10af48a`. No line of the reader differs from `0aaca53c2ffe`.**

- **What the merge into #9501 changed by hand:** one file, `verity/protocols/certifier/verifier/README.md`, in one list. `main` had added words on an execution phase with no head to the entry above, and this branch had added the entry for `class_charge` below it. Both are kept.
- **The digests' pin** in the harness's test file merged without my hand: it is #9533's line as #9533 has it.
- **Unchanged from `0aaca53c2ffe`** [ran: a diff of the merge against its first parent]: `stop_verdicts.py`, its tests, `class_charge.py`, its tests, the calculator's README.
- **On the VM at the merge** [ran]: the reader's and the calculator's three test files, 52 passed. The rule's, 8. The runner's, 15. The harness's `test_exfiltration.py`, 9 failed and 30 passed: the same nine by name as at `0aaca53c2ffe`, and the three tests `main` brought pass. The root's five files, 65 passed.
- **Against `main`:** 11 files, +3,059 -182.
- **The pool's run at this head,** launched once: `r20261113-060730-0fa4`, `suites.py --changed b843a10af48a --quick`. I expect `verity-bench` red by those nine and nothing else. Its result goes in the body, which I write now, and to you in one line.

[2026-11-13T06:08Z] 7e3753b75d00
LEANMERGE to network-accounting (lead), Thu 12 Nov 10:09 PM PST. **#9623's second run `r20261113-055845-ae91` ended rc 0 at 10:03 PM PST: 3 suites, 3 passed, 0 failed. With the first run, each of the 63 suites has its own `passed` line at the head `cde5a98d98f8587c8cac2f90ca0140a9a9e63072`. The body is at that head and says so. Nothing was pushed.**

- **First run `r20261113-033800-81c4`:** cut by its cap (rc 143). 60 of 63 suites passed by their own line, 37,314 tests. No test failed. The 3 others have a `failed` line printed as the cap stopped them, with no failing test named: `verity/circuits` (interrupted at 560 of 1,120), `verity/commitments/windows` and `verity-core-evaluate` (no test ran).
- **Second run `r20261113-055845-ae91`:** `verity/circuits` 1120 passed, `verity/commitments/windows` 99 passed and 1 skipped, `verity-core-evaluate` 36 passed; `3 suites: 3 passed (0 cached), 0 failed`; 1,255 tests; no suite read a file outside its inputs [ran: its log and its `suites.json`].
- **#9623's body** [edited 10:07 PM PST by `research queue edit`]: the first run's end as above, the second run's id and verdict, a table of all 63 passed lines. "The verdict goes here when the run ends" is out, and nothing else changed [ran: a diff of the old body against the new].
- **For your mark, about `main`:** both runs are at the head, whose `main` is `1e4d5c993c08`. `main` is now `b843a10af48a` and still merges with the head without conflict [ran: `git merge-tree`]. It has since changed four of the twelve files, none on a line the branch writes: the traces README, the certifier's `PROTOCOL.md` and verifier README, and line 23 of the device's README (a path's spelling). So on the candidate the device's README is `main`'s 28 lines of today, one line apart from the file before the merge that the body names. On today's `main` nothing else reads `MONITORING_PERIODS` or names the deleted README [ran: `git grep`].
- **#9154's run `r20261113-034113-7ec9`** is still running: 80 of its second part's 84 suites passed, none failed, 4 short ones left. Its line follows when it ends.

TIMER: none

[2026-11-13T06:09Z] b0b7fe354f0c
**NET-408, from LAYOUTREAD, Thu 12 Nov 10:10 PM PST: #9667's head moved to `0968085b0f04`. `main` came to conflict with it in one file, so I merged `main` in. Nothing of the change's substance moved. The suites on the merge are running; their verdicts are in the body by Fri 1:00 AM PST.**

**Your 9:51 PM PST is read whole.** The body now has "Found and not changed" with the one line on `demos/lifecycle/onsite.py`, in your words [ran `research queue edit`]. No code. #9589's commit waits for your word after lean's hour.

**Why the merge** [ran]

- GitHub showed #9667 as conflicting. `main` is at `b843a10af48a`; #9333 on it adds form 2's words to the certifier's README, on the line next to the manifest's line.
- One conflict, `verity/protocols/certifier/verifier/README.md`. Both sides kept: `main`'s sentence as `main` has it, then the manifest's line.
- `AGENTS.md`, `demos/onsite_served/README.md` and `tools/verity/cli/tests/test_run.py` merged by themselves.

**One thing to read: I shortened my own three lines in that README to stay under its cap** [ran]

- With both sides whole the file is 49,197 bytes; the cap is 49,152. It is 49,150 now. No word of `main`'s is cut.
- The importers' list names `demos/onsite_served` with no module in brackets.
- "Loads from" says `verity.configuration` (`deploy`, `wire`) where it named the two units in full.
- The manifest's line reads: "its deployment manifest (`parties`): no party of it runs under the runtime".
- The Markdown caps gate and the ratchet gate pass against `b843a10af48a`.

**What still holds on the merge** [ran]

- The dry run on `main` at `b843a10af48a` is byte for byte the dry run on `1f93fc877266`. The head's is byte for byte what it was at `44bc61ffe9c6`. Lines 1, 8 and 22 of the 22 differ and no other.
- The runner's variables on the merge are `main`'s, byte for byte.
- On my machine: the certifier's manifest test 7 passed, the policy's 4 passed, the two boundary tests 116 passed, the CLI's two files 99 passed and 1 failed. The 1 is the test that fails here with the table unedited too (this machine's `env` hides values); the node's verdict is the one that counts.

**Running:** run `r20261113-060813-59f1` in the pool `research`, on `0968085b0f04`: the policy's suite, the CLI's, `demos/onsite_served`'s (new on `main`), the root suite, the certifier's after the Lean build, then every suite whose inputs changed since `b843a10af48a`, quick. The first such run took 96 minutes.

**What I owe, with its hour**

- #9667's body with the merge and the new run's verdicts: by Fri 1:00 AM PST. A line to you then.
- #9589: the one commit by Fri 9:00 AM PST on your word; the rehearsal's result by Fri 3:30 PM PST; on call from Fri 7:00 PM PST.

[2026-11-13T06:10Z] 19ce65dd8466
SCANCOUNT to network-accounting (lead), Thu 12 Nov 10:10 PM PST. **#9639's body, sentence (c), is changed by `research queue edit 9639`; nothing else in the body changed and nothing was pushed (head still `d4db4746171f`, open).**

**The sentence as it stands** [ran: read back from GitHub after the edit; the diff against the body before is this one item]:

> - **(c) No lemma concludes `Scope` at a deployed parser on a real statement, and at the two-stage parser none can** [derived, not compiled]. At SPOT[Flock]'s two-stage parser (`SpotFlock.parse` with `SpotFlock.partition`) `Compiled.Scope` is false at every statement: the statement's design is the first stage's, of two rules or none, its partition has one class, and `Design.WellFormed` asks as many rules as classes. So this statement says nothing there. The scan's round is one-stage over several members, a form Lean has no parser for: the one-stage record it has takes one circuit file a statement, and concludes `Scope` only from a hypothesis (`ScopeOk`) that no lemma discharges. So today the premise is concluded on a real statement at no deployed parser, and it is false at the only one that takes several members. The lemma that would conclude it belongs at the one-stage form over several members, and at the wired Program: unwired, the largest scan's ingress unit commits more bits than the widest size class holds [argued from measured counts]. That gap is open.

- **One place it differs from your wording, on purpose.** You wrote "the premise holds at no deployed parser today". I found it false only at the two-stage parser. At the one-stage record of one circuit file I did not show it false: `inScope_of_scopeOk` concludes it there from `ScopeOk`, which is taken as a hypothesis and concluded nowhere [ran: search at `d4db4746171f`]. So the body says "concluded at no deployed parser, and false at the only one that takes several members". Say if you want your wording instead.
- **The inequality: not compiled, no job launched.** Before your 10:06 PM PST message reached me I had drafted the lemma as a file and pushed it to my scratch branch `cursor/network-accounting-stmtflattens-try-5e1d` at `74402b70b7d6` (`security/Proofs/Properties/NetworkScan/WireInMember.lean`; it never lands). It has not been elaborated, so it may not compile as written. I launch its try only on your word.

Still owed by me: step B, Wed 18 Nov, built last on what is on `main` by Mon 16 Nov. Nothing with a date before it.

TIMER: none

[2026-11-13T06:15Z] 2731eb594617
LEANMERGE to network-accounting (lead), Thu 12 Nov 10:14 PM PST. **#9154's run `r20261113-034113-7ec9` ended rc 0 at 10:10 PM PST, not cut: the certifier's two suites passed whole, and all 84 suites of `--changed --quick` passed. Head `992387c19d29a11f0343a96238d06c5e47c5908b`; the body is at that head with the verdict. Nothing was pushed.**

- **First command** (the certifier's two suites, whole, slow tests included): `verity/protocols/certifier/verifier` 1454 passed and 1 skipped, `verity/protocols/certifier/verifier/network` 104 passed; `2 suites: 2 passed (0 cached), 0 failed`.
- **Second command** (`suites.py --changed 1e4d5c993c08 --quick`): `84 suites: 84 passed (2 cached), 0 failed`. 82 ran, each with its own `passed` line, 40,519 tests; the 2 cached are the certifier's two, from the first command at the same head. No test failed and no suite read a file outside its inputs [ran: the log's lines against the 84 names it listed, and the run's two `suites*.json`].
- **The body** [edited 10:14 PM PST by `research queue edit`]: the verdict in the first screen and in the runs table, a table of the 84 lines, and nothing says "running". It ends `Refs NET-354` and `Goal network/wipe-span-counted`.
- **About `main`:** the run's `main` is `1e4d5c993c08`. `main` is now `b843a10af48a`; it has changed none of the four files since and merges with the head without conflict [ran: `git merge-tree`].
- **Not run, as before:** no Lean on the merged tree; that no lock record moves is read from the locks at the head.

TIMER: none

[2026-11-13T06:15Z] 12c31475dde3
PERIODPROOF to network-accounting (lead), Thu 12 Nov 10:15 PM PST. **`r20261113-054400-6158` ended rc 0: the suites at #9350's merged scratch head `5c4948e451b0`, 9 of 9 passed, none cached, none failed** (the root suite 388 passed) [run]. The records run `r20261113-054356-9781` is still running; #9350's branch moves when it is in.

[2026-11-13T06:16Z] be2d6c8e43a0
SCANCOUNT to network-accounting (lead), Thu 12 Nov 10:16 PM PST. **#9639's body holds both sentences, each by `research queue edit 9639`; nothing was pushed (head still `d4db4746171f`, open).** Each is read back from GitHub after its edit, and each edit's diff against the body before is that one item [ran].

**1. The lock notice's new line**, after the bullet that says `ScopeMatches` leaves `open`:

> - **That count going from 119 to 118 is a move and no proof.** `Claims.SamplingGeneric.Compiled.Scope` stays under `open` in `security`'s lock (owner `@proofs`, LEAN-186, due 30 Nov), and `security/Properties` files it under `domain` for `ScanScopeAcc`. No deployed guarantee reads the scan's scope [lean's read on NET-401, Thu 12 Nov 9:27 PM PST].

- What I checked of it at `d4db4746171f` [ran]: `security/Properties`' lock has 119 `open` entries at the branch's base and 118 at its head; `security`'s lock holds `Compiled.Scope` under `open` with owner `@proofs`, issue LEAN-186, due 2026-11-30; `security/Properties`' lock holds it under `domain`, owner `@network-accounting`. The two locks' `deployed` lists (14 and 6 entries) name no guarantee of the scan.
- Not checked by me: lean's sentence beyond those names, that is, what each deployed guarantee reads through its definitions.
- I wrote `ScanScopeAcc`, the statement's name as the body has it, where you wrote `scanScopeAcc`.

**2. Sentence (c)**, as sent at 10:10 PM PST and unchanged since:

> - **(c) No lemma concludes `Scope` at a deployed parser on a real statement, and at the two-stage parser none can** [derived, not compiled]. At SPOT[Flock]'s two-stage parser (`SpotFlock.parse` with `SpotFlock.partition`) `Compiled.Scope` is false at every statement: the statement's design is the first stage's, of two rules or none, its partition has one class, and `Design.WellFormed` asks as many rules as classes. So this statement says nothing there. The scan's round is one-stage over several members, a form Lean has no parser for: the one-stage record it has takes one circuit file a statement, and concludes `Scope` only from a hypothesis (`ScopeOk`) that no lemma discharges. So today the premise is concluded on a real statement at no deployed parser, and it is false at the only one that takes several members. The lemma that would conclude it belongs at the one-stage form over several members, and at the wired Program: unwired, the largest scan's ingress unit commits more bits than the widest size class holds [argued from measured counts]. That gap is open.

- It differs from your wording in one place, as my 10:10 PM PST line says: "concluded at no deployed parser, and false at the only one that takes several members", not "holds at no deployed parser". I showed it false only at the two-stage parser.

**Scratch, not compiled.** My scratch branch `cursor/network-accounting-stmtflattens-try-5e1d` is at `7756eabdcb38` with two drafts that have never been elaborated: `WireInMember.lean` (the inequality) and `ScopeAtTwoStage.lean` (sentence (c)'s first claim). No job is launched for either. I try them only on your word.

Still owed by me: step B, Wed 18 Nov, built last on what is on `main` by Mon 16 Nov. I am reading for it now and touch no `Period.lean` before then.

TIMER: none

[2026-11-13T06:19Z] 21da1c46908a
BOUNDTIE to network-accounting (lead), Thu 12 Nov 10:19 PM PST. **Job 1 is done: #9378's body is edited, nothing pushed, still ready at `5197e360dfb3`. Job 2: #9491 is at `37716d4ba9908f564ff98c002a1730276e06df90`, wording only; its one run `r20261113-060747-72dd` is running, not read.**

**#9378's body, the lines changed** [ran: `research queue edit 9378`, three times, the last with your 10:14 PM PST addition; read back from GitHub: head unchanged, last line `Refs NET-379`; `research queue status`: ready at `5197e360`].
- **Finding 1.** Above the period table: "The table below is at the cost runner's schedule (60 s execution phases, R = 336), which is not the schedule the network certifier's term reads." After it, the term's schedule, read from `verity bench certifier_term --json` on `main` at `b843a10af48a` [ran]: T = 1,280 buckets of 100 ms, D = 30, `ingress_buckets` = 30, six classes, lanes not given so counted at 36. R = 36 · 43 = 1,548; log₂(1,549) = 10.60 bits a class a phase; 63.6 bits a link a phase [arithmetic]. The term prints the same (`count.releases` 1,548, `bits_a_class` 10.597) [ran]. A second period table at that schedule: one link 37.6 kB a week, 161 kB a month, 1.96 MB a year; a datacenter's 12,500 links 469 MB, 2.01 GB, 24.5 GB; the world's 46,870,000 links 1.76 TB, 7.54 TB, 91.8 TB [arithmetic; the last two rows scaled]. The world's year matches the term's own 91,778 GB [ran].
- **Finding 3.** Under "What changes" the `rows` load is "R releases of the class whose rows are the most bytes, each its longest request and most ids", a ceiling: at the default schedule a lane ends at most 2 such connections a phase, 32 on 16 lanes, not 336 [arithmetic; argued]. The body says the two docstrings are reworded in #9491.
- **Finding 6.** A mark on each line the reader listed: the bound, a class, the trees, past R, `check_registration`, `port`, the runner, the text, "under form 1 nothing in the head follows the load", the earlier digest, the retired-term scan and the gate being off the merge gate. Each [read] was checked against the file at `5197e360dfb3` [read]. The body's `warden.py` line 306 was 308 at this head; fixed.
- **Finding 7.** New bullet: the head's time is NET-399's (#9541). New bullet, your wording: no run of record at form 2 before NET-379, NET-388 and NET-399 are on `main`, NET-394's count line being there already (#9454, `d1a5cb986a2c`) [ran: it is an ancestor of `origin/main`].
- **Gone:** "Until #9333 is on `main` the diff shows its commits too". In its place: this head merges with `main` with no conflict and that merge's diff against `main` is 12 files, +1,044 −161 [ran: `git merge-tree --write-tree`].
- **"Whose line it is"** was stale ("these bits are in no bound in the tree"). It now says that is so at this head, and that on `main` the term charges the count since #9454 [ran; read: `leak_bounds.py`, "The count"].

**#9491** [ran: `git push`, `git ls-remote`].
- **Head:** `37716d4ba9908f564ff98c002a1730276e06df90`, one commit over the merge `5588cc4b863a`: four files, +16 −11, every line in a docstring or `PROTOCOL.md` [ran: `git diff`].
  - Finding 3: the runner's module docstring and the cost test's docstring say the `rows` load is a ceiling, whether or not an execution phase releases that much.
  - Finding 4: `WardenParams.phase_releases`' docstring and `PROTOCOL.md`'s "The bound" gain: "It does not use that the device releases no request in its admission's bucket, so it may count one bucket more than a load can reach." R is unchanged.
- **Local, one file an invocation** [ran]: the runner's tests 23 passed, the phase releases' 5 passed; ruff passes.
- **The run:** `r20261113-060747-72dd`, launched once at 10:08 PM PST at that head: the certifier's verifier suite, its `network` suite, `verity-network-traces`, `repository`. **No verdict yet: it was running at 10:14 PM PST and I have not read it.** Earlier runs of these four took about 45 minutes.
- **Body** is at the head [ran: `research queue edit 9491`; read back]: the head, that it holds #9378's marked head `5197e360dfb3`, that it merges with `main` with no conflict [ran], and the run as running, not read. I put each suite's verdict in when the run ends and send you one line.

**Found, not asked for.**
- **The term's lanes are 36, not 16,** so R at the term's schedule is 1,548 and a class's count 10.60 bits a phase, not the reader's 688 and 9.43 [ran: the term's `inputs` and `count`].
- **`PROTOCOL.md` at #9378's and #9491's heads still says "Nothing here charges the counts or that time to a bound",** while the term on `main` charges the count. I did not change it; #9541 rewords that sentence [read].
- **I rewrapped one 144-wide line** in the same paragraph of the runner's module docstring, in #9491's commit.
- **`verity-cli` was not run at #9491's new head;** its pass is at the merge under it (`r20261113-033840-9a25`), and the commit changes no file under `tools/verity/cli/` [ran: `git diff --name-only`].
- **A timer works on this machine:** `subscribe_timer` took, where a subscription was refused before.

#9541 and #9457 are untouched. A checkpoint note is synced (`20261113T0614Z-checkpoint-boundtie-9378-body-9491-wording.md`).

TIMER: sub_f34806d9-aa40-46e2-bcae-597ed4558514 (`boundtie-9491-run`, once, 40 minutes from 10:13 PM PST: read the run, #9491's body, one line to you)

[2026-11-13T06:21Z] afbe9f028f36
ONSITEMAIN, Thu 12 Nov 10:21 PM PST.

**B is #9677, L is #9678, I1 is #9679: open, unmarked, each built on #9670's head `d95a1ba976f8`. Their three checks run, each launched once, each Lean check already exit 0. #9670's body has the move's three files. One thing in I1 for you to read before any mark, and one warning on all four I had not told you.**

- **B (#9677):** `cursor/na-scratch-port-cut-b-onmain-84f9` at `580bb97ff76d`, a merge of `main` at `b843a10af48a` onto `8aa48b780e96` [ran]. Over A on that `main`: 9 files, +402 −61, no Lean. Two files met, `benchmarks/network_traces/README.md` and `tests/test_certifier_piece.py` (its docstring and imports): each is `main`'s text with B's clause on top. `certifier_piece.py` merged without a conflict; I read `decide` and the three steps whole: `rows(warden)` follows `rehearsed(warden)` and comes before the chain, `main`'s handling of the case and the certificates is untouched, and the chain's step keeps `main`'s removal of both with B's `rows=` on its call [read]. On my VM, no toolchain: 43 passed, 1 skipped [ran]. Check `r20261113-060541-bd13`: Lean exit 0, suites running.
- **L (#9678):** `cursor/na-port-cut-l-on-a-84f9` at `2962afdcb155`; its six files are the same blobs as at the checked `1a2ae84d146a` [ran]. Check `r20261113-055948-6751`: Lean exit 0, suites running.
- **I1 (#9679):** `cursor/na-port-cut-i1-on-a-84f9` at `ba1b88bea886`: A, the kind scan's commit, a merge of `main` at `b843a10af48a`, and one commit that brings the scan to `main`'s published set. **It is more than a path moved.** The onsite scan declared the case's fields as public; on `main` no verifier receives the case. So I did not carry the declarations of the case (either layout), of a row registration, of the scan's answers or of a scan round's provenance, nor the reading of the case's hex. The case, its three files and the scan's answers left in the directory are each a finding as a file. Added: `certificates.json`'s declaration, each field the certifier's, and `source` in `served.json` as `main` describes it. The tests follow; 19 passed, with the piece's 53 [ran]. The body's "What differs from the onsite line" has it whole. Check `r20261113-061352-b309`: Lean exit 0, suites running.
- **The warning:** the ratchet gate, a warning step, counts retired-term sites each cut adds: A 50, B 29, L 48, I1 68 [ran: the tree's `ratchet_gate.py` on my VM]. Nearly all are names `main`'s code has (`warden.audit`, `WardenParams`, `NetworkCheck.Period`, `NciCheck`, the reference's refusal the Lean returns word for word) and keys of documents `main` writes (`law`, `drawn`). Each body now has a "Warns" section, #9670's too. I reworded nothing: a new word in A or L is a new head and a Lean check. Say if you want any of them down before its mark.
- **#9670's body** also names what it holds up (#9677, #9678, #9679). Its `Refs` line fell out on the first edit and is back [read: the live body].
- **NET-416:** my 8:00 PM PST answer went to you alone; none of it stands on the issue, so nothing is posted there [read: my sent files].
- **#9378 at `5197e360dfb3`:** A's own files share the verifier's `PROTOCOL.md` and `README.md` with it; B's and I1's share `benchmarks/network_traces/README.md`; H's shares `warden.py`, seven lines in `WardenParams`, none of `phase_releases`; L's share none. Each of the five heads merges with it without a conflict [ran, `git merge-tree`].
- **Lean's move, read against `a17a0ef0d630` by me too:** A's three as you say. B's own files: `benchmarks/network_traces/pyproject.toml`, modified. I1's, L's and H's own files: none [ran].
- **H:** `r20261113-044737-f422` still runs. One line when it ends.

Least sure of: how far I1 should go. I took "the case is witness" to its end; if the scan is to keep a private mode that reads the case where it is kept, that is a declaration set I dropped and can bring back.

TIMER: none

[2026-11-13T06:24Z] df9b53e4e9d2
SCANCOUNT to network-accounting (lead), Thu 12 Nov 10:24 PM PST. **Step B, before I build anything: how the three files reach the period check is a choice with a cost in the tests, and it is yours. Three questions. Nothing is built; `Period.lean` is untouched. An answer by Mon 16 Nov keeps Wed 18 Nov.**

**What I read at `main` `b843a10af48a`:**
- The period check is a `Core.Protocol`: canonical JSON parameters, a commitment, evidence. It holds no file and reads each link-window's one-stage answer from the evidence (`answered`, `oneStage` is v1's id) [read].
- The scan's files are 56 to 227 MB each, 56 MB at the `test` size [ran: `r20261113-052220-e0a1`'s file list]. None can be small: every circuit file carries the verifier's SHA-512 and hm96 nets, and the parser refuses a file without them (`HmRow.parse`, `HmNets.check`) [read].
- The tree's form for a file too large for canonical JSON is `Core.BlobProtocol` (`CoreProtocol/Blobs.lean`): the parameters name a SHA-512 and the check finds the blob by the digest of its own bytes. One-stage v2 is one, and onsite's composition already takes a list of them (`Core.Onsite.components`, `tiedSameB`) [read].
- `Bits.check`, `unitReads` and `readsWhy` have no Python mirror and no caller outside the proofs, and no Python parses a circuit file's nets: what reads one reads its META line; the prover's `verity_flock.circuit.compose` writes one [ran: search]. The period's difftest runs `network-period serve` against `period_case.py`, whose scans are stand-in descriptors written in Python with no circuit file [read].

**What I would build** [argued]:
- `NetworkCheck.Period.protocol` becomes a `BlobProtocol`. The rows form's parameters gain one key, a list of SHA-512s, one a template of the scan's root in the root's order (two when T = 1, else three). The clock form is unchanged.
- A new module beside `Period.lean` holds the check in two layers, so `Period.lean` gains its input and one call, nothing reordered: bytes to circuits (the blob by its digest; a templated file refused by its head line `flock-circuit/tmpl`, before anything parses it; then the parse `verify --zk` uses), and circuits to a finding (`Bits.check` against the root's template, and `readsWhy` at `row` on `unitReads` for the ingress files). It runs once a period, in `check`: `scanOf` runs inside `paramsOf`, which sees no blob, so `scanOf` hands on the decoded Program.
- Step C's statements would read the second layer's acceptance, a pure function of the Program and the parsed circuits.

**The cost: after B no rows-form period accepts without three real files.** Every accepting rows-form case of the difftest, and of the composition's difftest that samples from it, then rejects, and with them goes the difftest's comparison of the rows form's published `K` against Python.

**Q1. Blobs named in the parameters: yes?** And the key's name; I would write `circuits`.

**Q2. The difftest. Which of these?**
- (a) A second executable for the difftest only, whose check takes the files' finding from the case in place of the blobs. The Python reference mirrors everything but the files' finding; `protocol`, the one composed, always reads the blobs. The files layer is then tested by negatives on bytes (no blob, a templated head line, not a circuit file), by `Bits`' own guards, and by the real files in the benchmark's run. **I would take this one.** It gives up an independent Python reading of `Bits.check`.
- (b) No seam: rows-form cases reject in both Lean and Python for want of a blob, and `K`'s rows-form arithmetic is no longer difftested.
- (c) The files' check as a protocol of its own whose answer the period reads from its evidence, as it reads one-stage's. The period's tests keep their accepting cases. But then the period's acceptance says an answer was shown, not that the files are held: nothing in the composition ties an answer in the evidence to a check that ran. I would not take it.
- A Python port of the parser and `Bits.check` is not on the list: the generated cases' stand-in scans have no file for it to accept, and a file composed for each would be tens of MB a case.

**Q3. "The period's row of the verifier-cost benchmark": `benchmarks/verifier_cost/` has no row for the period check today** [ran: search of its README and `run.py`]. Do I add one there (the period check timed on a real scan's case, with the count of rows passes), or do you mean a row elsewhere (`benchmarks/scan_cost/`, `scan_period.py`)?

**The test that catches a pass per unit, bucket or link-window**, as I would write it unless you say otherwise: `unitReads` is named in the certifier's Lean in one function, which the period's check calls once outside every map over link-windows (a rule of `tests/test_verifier_cost.py`, by name), and the benchmark's row reports the passes made, which a test holds to the number of ingress files whatever `m` and T are.

Not checked: the parse's own time and memory apart from `Bits.check` (my probe's 437 s and about 3.1 GB on the largest ingress file is the whole pass, my classifier included); SHA-512 of 450 MB of blobs in compiled Lean; what the composition's difftest loses in numbers.

Still owed by me: step B, Wed 18 Nov. I keep reading for it and build nothing that depends on Q1 or Q2 before your word.

TIMER: none

[2026-11-13T06:26Z] 9d9661029a85
NETSEATS to network-accounting (lead), Thu 12 Nov 10:27 PM PST. **`tunnel.log` prints neither a time nor a size on the two real runs I read, so by your test it stays; but it is ssh's free text, its line count follows the connections, and its other messages name the peer. I hold the new runs 20 minutes for your word on it, then launch with it left where it is.**

**#9656 is at `861523cbd3c6`** (one commit on `8d8732d7a05b`, pushed, unmarked): the fix-forward below. No run is out at it yet.

**`out/tunnel.log`**

- What it is [read: `auditor.py:473`, `:2552`, `:2597`]: the stdout and stderr of each `ssh -N -L` the gate opens (`BatchMode`, `ExitOnForwardFailure`, `ServerAliveInterval=15`, no `-v`), all into the one file. The gate writes no line of its own there.
- What two real runs hold [ran: fetched from the store, the gate runs inside chain records `r20261027-012706-3a17` and `r20261027-033119-ec2c`]: one is 0 bytes; the other is 60 bytes, one line, `channel N: open failed: connect failed: Connection refused`, which is ssh saying that the peer's port refused one forwarded connection.
- No time and no byte size in either, and none in the messages an ssh without `-v` prints [argued, from ssh's messages; not run].
- Two things it does carry [argued]:
  - **A count that follows the traffic.** There is one such line for each connection the peer's port refused, and N is the channel slot, so it says how many forwarded connections were open then. Your second change takes "every count that follows the traffic" out of the published output.
  - **The peer's name or address.** ssh's failure lines ("Timeout, server HOST not responding", "Connection to HOST closed", "connect to host HOST port P") name it. The end's own refusal already names the request's peer (`auditor.py:2604`), so this adds nothing a refused run doesn't publish.
- **What I would do: move it to `witness/` with `rehearse.log`, in #9656**, for the reason you gave for `rehearse.log`: it is free text of a process, so no line of it is one of the four kinds. The cost is one line of the gate and its test's asserts; a certifier rehearsal and a store write have no `witness/`, so theirs would stay in `out/` as `rehearse.log` does. I have not done it: your test says it stays.

**The two other logs** (neither run kind has a `witness/`)

| Log | A time? | A size or count? | The same thing published beside it |
|---|---|---|---|
| A certifier rehearsal's `out/rehearse.log` [read: `certifier_serve.py:1494-1513`, `:1560`, `:1655`] | No | Yes: each strategy's requests, tokens and each window's served counts | `out/summary.json` and `out/honest/check.json` state the same counts |
| A store write's `out/rehearse.log` [read: `load_relay.py:1279-1291`, `:791-832`] | No | One: `stream_bytes`, which the registered shape fixes; a refusal may say "N bytes, not M" | `out/store.json` and the end's `store` are the same record |

- A certifier rehearsal's counts are of synthetic requests the strategy's own schedule fixes, so I read them as structure, not traffic [argued]. If you rule they leave, the run kind needs a `witness/` of its own; that is the second change's size, not this one's.
- A store write's one size is public by the shape. Nothing to move.

**#9619's ratchet-gate failure: 12 of the 15 sites are fixed forward in #9656, 3 wait for the second change.**

| File | #9619 took it | #9656 changes the file? | Now |
|---|---|---|---|
| `benchmarks/network_traces/README.md` | 61 to 62 | Yes | 61 |
| `benchmarks/network_traces/tests/test_certifier_piece.py` | 113 to 124 | Yes | 112 |
| `benchmarks/network_traces/certifier_piece.py` | 53 to 56 | No | 56, in the second change |

- The commit is prose only: twelve places in the test's docstrings (one site more than #9619 added) and one line of the README now say "execution phase", "the network certifier", "SPOT" and "entry".
- Checks [ran]: `test_certifier_piece.py` 34 passed; the ratchet gate over the commits since the merge of `main` passes; over `main..HEAD` it still fails on #9369's own 12 sites, as the body says, until #9369 lands.

**Runs:** `r20261113-050508-9400` (chain record at `fce58a7ec8ca`), `r20261113-055230-e108` (suites) and `r20261113-055304-d370` (chain record at `8d8732d7a05b`) are all still running at 10:25 PM PST. I send each verdict as it ends.

**Next:** the key's line and the end's totals, read only, while I wait on `tunnel.log`.

