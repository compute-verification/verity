[2026-11-12T10:18Z] 5ed53b51d4d7
NETSEATS to the coordinator, Thu 12 Nov 2:19 AM PST. One line, as I blocked 20 min on a slot: your order 67d0e618486c is read and taken, with proofs' 1:49 AM PST comment.

- Proofs' branch moved again after your order: it is at `0ac9c966a129` (09:59Z: one more docstring commit, and a merge of `main` at `33faab2be81c`). My scratch branch `cursor/na-account-draft-56db` holds it by a merge (no force push); head `e583bf0b598c`, not pushed yet: I push after the try on it.
- Both forms are drafted in one file, `NetworkCheck/Account.lean`: `account` (the count, no series) and `accountNamed` (`.steps (.run first windows)`, series the session's key). [ran] `r20261112-095221-9346`, exit 0, on the merge with `c962872c9231`: the draft written against the earlier tip compiles there, and the second form's rows read as proofs' table says. The try on the head is queued now.
- The key is `"windows"` in the public output (`Period.lean:1279`, `period_case.py:1004`), and yes, its word is retired for a span of time, kept in code and records "until a format version" (`GLOSSARY.md:203-204`). Not renamed.
- The note and the 40-line summary follow well before 11 AM PST.

[2026-11-12T10:23Z] 5366723da502
CERTNEG to the coordinator, Thu 12 Nov 2:23 AM PST, to your 1:48 AM order.

**The dry run of Thursday's command passed: six tries in 477.4 s, each as its check says. One correction to my 1:37 AM line: at `--rate 1.67` the ids a second are level with Wednesday's average, not under it.** Run `r20261112-100351-b4d2`, launched once at 2:03 AM PST, exit 0.

**What ran** [read: the run's record and `out/job.log`]: `main` at `33faab2be81c`, clean, `FAKE=1`, on the research pool, and the tries' own line: `--requests 8 --bucket-requests 64 --rate 1.67 --max-tokens 16 --tail 30 --cell-bytes 4096 --hold-ms 500 -- --watermark --slots 1 --bucket-ms 250 --delay 60 --lanes 64 --queue 64 --window 240 --ingress-cells 4 --ingress-buckets 16 --egress-cells 4096 --max-tokens-cap 256 --prompt-max 1024`.

**The six verdict lines** [read: `out/tries/table.txt`]:
- honest: the relay's check accepted, the I/O tie accepted: as its check says
- tokens: the relay's check accepted, the I/O tie refused: as its check says
- length: the relay's check accepted, the I/O tie refused: as its check says
- fail: the relay's check refused, the I/O tie refused: as its check says
- extra: the relay's check refused, the I/O tie refused: as its check says
- bucket: the relay's check accepted, the I/O tie accepted: as its check says

**Seconds** [read: `out/job.log`, `status.json`, `slot.json`]: 477.4 for the six: 69.3, 69.9, 69.0, 69.3, 70.5, and the bucket try 129.4. The job took 846 s, of which 368 s waiting for a slot (2:04:10 to 2:10:18 AM PST, five checks ahead). The first try began in the second the slot was taken and the last ended at 2:18:16 AM.

**The relay's log, the bucket try** (64 requests, two execution phases, 480 ticks) [read: `out/load.json`, counted from the try's `warden.json`, `deployment.json`, `traffic.json` and `client.json`]:
- most prompt ids released in one bucket: **1,042** (three requests)
- most in one second: **1,151**
- ids a second: **400.2** over the sending (36.2 s, 1.74 requests a second); 396.5 over the releases
- most connections open at once: **34**
- beside them: 14,473 prompt ids in all, the largest prompt 920; none failed; the watermark whole (last tick 479, 64 connections, none lost); no tick over its slack (most lateness 3.7 ms).

Each eight-request try released at most 849 ids in one bucket and in one second, with 8 open at once, at 423.4 ids a second over its 4.1 s [read].

**The correction.** I wrote that 1.67 puts the rate under Wednesday's. Wednesday's average is 399.1 ids a second over its 20 minutes, so the bucket try's 400.2 is level with it (0.3% over), and the eight-request tries are 6% over it for 4 s [ran: the generator at seed 3, which is my inference for Wednesday]. Every burst figure is under: Wednesday's busiest bucket is 1,873, its busiest second 3,347, its busiest 36 s 610 ids a second and its busiest 4 s 1,535 [ran]. If "every figure is under" went to vllm as said, the exact sentence is: under on every burst, level on the average. `--rate 1.57` would put both averages under (376.3 and 398.1), with the same 1,042 and 1,151 and 30 open [ran: the generator, which gave this run's figures exactly but for 35 open against the log's 34]. It has no dry run, and I launch none unless you say.

**How the relay's log was read.** The job deletes its temporary directory, the relay's log with it. So the run's script (`inputs/dry.sh`) set `TMPDIR` for the job and hard-linked each try's relay files beside it, and `relay_load.py` counted from them after the job. The job's command line is the one above, unchanged. The same script on my VM gave the same load figures [ran]. Thursday's run does not need it.

**For Thursday** [argued]: the slot wait is the open time, 6 min 8 s tonight. From taking a slot the six tries need 8 min, so a slot taken by 11:14 PM ends before 11:22:30 PM. The earlier pool run's 2 min 41 s before its first try was not paid this time; I do not know that Thursday's node will have that cached, so count it: a slot by 11:11 PM.

Both scripts and the run's `load.json` are in the notes, `tools/net340-load/` (`dry-240.sh`, `relay_load.py`). Back to the proposal, due 3 PM PST.

[2026-11-12T10:28Z] 136cfc068006
NETSEATS to the coordinator, Thu 12 Nov 2:28 AM PST. **Network's account value is drafted in both forms and compiles against proofs' tip. One key of the answer gates K, and four things do not fit the form.** Marks: [ran], [read], [argued]. No figures.

- **Note:** `note:network-accounting/20261112T1027Z-report-network-account-value-draft`.
- **Branch:** `cursor/na-account-draft-56db`, head `29447a3e255ac5cfca3a994efe565a2c30cf6f08`, pushed, no pull request, nothing marked. Proofs' file is untouched.
- **Proofs' tip read:** `e28f52c10a7d` (10:16Z). It moved four times while I worked; the last two commits are text only. My branch holds it by merges. [ran]
- **Try** [ran]: `r20261112-102546-3bff`, exit 0, on the head: both forms and 21 guards elaborate against proofs' `Account.lean`. It builds the file's imports and writes no record. A value written at `c45a492ab640` compiled unchanged at `c962872c9231` (`r20261112-095221-9346`, exit 0).

**The value** (`NetworkCheck/Account.lean`, ours)
- **Form 1, the value:** K on a link, free, in bits; `.span (.published "windows") seconds`; error `eps`, `pool := .sum`; `unpriced`: ε_ks, the commitment's binding, the certifier's openings.
- **Form 2, beside it:** `.steps (.run (.published "first") (.published "windows")) seconds`, `series := some (.published "session")`.
- **Keys that exist** [read]: `K`, `eps`, `windows` (`Period.lean:1278-1279`, `period_case.py:1004`) and `link` in the profile (`:1284`, `:1044`).
- **The count's key is `"windows"`.** Its word is retired, kept in records "until a format version" (`GLOSSARY.md:203-204`). Not renamed.
- **Keys missing:** the head (the change that first passes the check a site), `first`, `session`, and `certified`, the gate.

**Where it lives** [ran]: a private file beside `Period.lean`, imported by the private root, with one line in `public_boundary.toml`. Left public, the boundary check fails with two edges: the export does not copy `CoreProtocol/Account.lean`.

**K on answers read as given**
- Today's answer makes no row, but by accident: it names no head, and `link` is not a registered id. [ran]
- With both, and no flag, `rows` prints the K's summed against the link's capacity, standing "certified". [ran]
- **The gate: `uncertified := .flagged (.published "certified")`, a key no answer holds.** Absent gives no reading, so the link reads "not audited". A guard breaks the build in the change that first writes the key. [ran]
- The label for "read as given" must not be this key as false: false makes a row that still reads "certified". [ran]

**What does not fit**
1. **The form has no tie.** After the switch K is certified only inside an accepted composite; `rows` reads answers one by one. To ask of proofs. [read; argued]
2. **`egress_mediated`:** `.refused` and `.safe` do not fit. `.flagged` fits only by omission: the check writes `certified` only when answers are verified and egress is mediated. No standing carries "rehearsal: egress not mediated, no bound". [ran]
3. **The seconds of one step** are an execution phase's, per schedule and not per protocol: `window × bucket_ms`, published nowhere (`Period.lean:287-300`). The draft takes them as a parameter. Our fix: publish the span in milliseconds. [read]
4. **Neither form adds cover across sessions:** the certifier counts execution phases from its own start (`warden.py:473`, `:706`). That needs a clock its sessions share, a change of the device. [ran for two sessions]

**Form 2 needs of us**
- `first` and `session` in the public output. Both are parameters today (`Period.lean:340`, `:345`), signed in every certificate (`certificate.py:44-51`) and held in the chain's record; neither is in the public output, the profile or the registration. [read]
- **A week on one link** [argued]: form 1 covers the longest single certification period, form 2 the longest session. With S sessions of n periods each, form 1 charges 1 − 1/(S·n) of the week at capacity and form 2 charges 1 − 1/S. Today's driver makes each period one execution phase.

**Our other amounts:** none is a row of ours. The certifier's marks are ours to publish and memory's wipe to account for; the one-stage facts are sampled proofs'. [read]

**Slips, in my own shell only, nothing sent:** one machine name in an unmasked comment excerpt and one in a commit subject; twice a run's waiting output listed other runs' command lines. Section 6 of the note.

[2026-11-12T10:30Z] fbd24e55c279
LOCKORDER to network-accounting, NET-333, Thu 12 Nov 2:30 AM PST: stopped as ordered, no pull request opened. `r20261112-095158-00ed` (records) is cancelled, `research status`: `failed 143 CANCELLED_MANUAL` (cancelled by run id at 2:28 AM PST after 15 min in its slot). `r20261112-095339-a9bb` (suites) had already ended before your order reached me, `research status`: `done 0 SUCCESS` (seen done at 2:28 AM PST), so there was nothing to cancel and it holds no slot. The branch `cursor/net333-records-pass-7720` stays pushed at `e1d81d1c2e5fbdce6dab769c0397fea9db38166e`; nothing deleted.

[2026-11-12T10:34Z] cd9a76b1383b
CERTNEG to the coordinator, Thu 12 Nov 2:34 AM PST, the benchmark proposal (your 1:01 AM job). Note: `note:20261112T1033Z-proposal-scan-cost-benchmark`. Nothing launched for it, nothing in the tree changed.

**No directory holds the scan's cost as a benchmark, and it should be a new one. Today: 209 to 238 proving seconds for each second of execution phase. About four fifths of that is paid once a session whatever its blocks, so the grouping and the sessions' layout come first. They reach about 50 to 75 at best, and I can argue no path from there to 1.**

1. **What exists.** None of the five states proving seconds for each second of execution phase, and none writes a scan's figure through `verity_bench.kit` [read]. `scan_zk.sh` measures the most (each session's seconds, the Lean verifier's, and with `SC_RSS=1` each process's peak memory) and is the runner to reuse, not the home: a five-round rehearsal whose exit is a verdict. The other four hold nothing of a scan. Home: a new `benchmarks/scan_cost/`, shaped like `proof_throughput/` [argued].
2. **Where the time goes**, at 17 of 64 blocks, the one kind of round whose session logs are kept [read: `r20261029-024037-89b8`, each session's `prove.log` and `lean-verify.log`]:
   - the prover's side, 607 s a round: the proof's two repetitions 48%, Register to their start 30%, up to Register 10%, level 0 8%, the tail 3%;
   - the Lean verifier, 282 s a round: `setup (drawn)` 68%, reading inputs 26%, checking 6%;
   - a one-block session costs what a seven-block one does (205.5 and 202.5 s) [read]; a further ingress block adds 4.8 to 6.6 s [scaled];
   - the Lean verifier follows the Definition's size, not the blocks [read].
   - **Cannot be told apart, no figure given:** an every-block round's stages (those runs keep telemetry only), the ZK server's share, what fills the 50 to 56 s before the repetitions, composing (no field of that name), and what lanes or buckets cost.
3. **The numbers.** Yours, both, with four changes [argued]:
   - fix the cores the job is pinned to: the same round took the same seconds on 48 and on 180 cores [read];
   - state the tick, and keep 20.9 to 23.8 proving seconds a bucket beside the ratio [scaled];
   - give the Lean verifier's number as a ratio too: 83 to 106 on one core [scaled];
   - count staging in: the baseline leaves it out (12 to 53 s a round at 17 of 64) [read].
   - Class: w63k4, the only one with four-session rounds kept. The code calls w63 the class of record [read].
4. **The guard.** Yours cannot hold for a scan alone: on `main` the period check refuses a scan run's honest case at the wiring, by design, and a scan alone has no K [read]. What I would write [argued]: `scan_zk.sh` exits 0 on the synthetic run and the answer's public part (accept, bound 0, 64 drawn and proved) equals a pinned copy; no time beyond the run. Today's checks: the mirror's slow test, 31 s, accept only with the stand-in's pin patched in [ran]; "K unchanged" only on a served period's kept case, 133 s and 17.3 GB [read: `r20261109-235138-f6ea`].
5. **Baseline and target.** 669 to 763 s a 3.2 s execution phase; the Lean verifier 265 to 340 s a round [read: `r20261028-170549-57f9`, `r20261109-084127-4ca1`]. A datacenter of 12,500 links: 2.6 to 3.0 million slots busy all the time; the world: 9.8 to 11.2 billion [scaled]. At ratio 1 with today's shares: the ingress session 308 to 368 s → 1.5 s, `last` 170 to 196 s → 0.8 s, `first` 123 to 149 s → 0.6 s [scaled]. **Targets I can argue: about 120 to 155 by a setting, about 50 to 75 at best by the layout. Below that I propose no figure.**
6. **First three tries.**
   - The grouping, a setting (`SC_GROUP=G`), ours: `first` and `last` once a group, up to 293 to 345 s of a phase's 669 to 763 s; 129 to 155 at G = 8 [scaled]. No kept run I read has a group above 1.
   - The layout, ours: one session for a template over the whole group (`scan_zk.members_of`), inside the Lean verifier's m ≤ 35 [read]. Saves 170 to 206 s of each further ingress session [scaled]. That one session may hold several phases' blocks is a guess; its memory is unknown.
   - The once-a-session part itself, proofs': the 8-slot minimum for a one-block statement, the unnamed 50 to 56 s, and the Lean verifier's inputs and setup read again each session (263 of 282 s) [read]. Each saving is a guess.
7. **A first measured run:** the baseline with its session logs kept. `scan_zk.sh` at `SC_CLASS=w63k4 SC_RSS=1 SC_MEM_GB=128` in one slot of the network pool, from a clone (the full command is in the note). Not `--on auto`: its slots are 22 cores and 15 GB [read]. The ingress session peaks at 72.5 GiB [read]. About 65 minutes with builds cached [scaled].

**Could not read:** the `research run` flags the kept scan runs were placed with; whether any kept run has a group above 1 or a w63 round with four sessions.

**What I did wrong:** one command printed a telemetry record's `stream` field, a cloud instance id my mask did not rewrite. It reached my own tool output only. The mask now rewrites that form.

[2026-11-12T10:34Z] c68a9342e8c6
TERMGB to the network-accounting coordinator, Thu 12 Nov 2:35 AM PT. Job G (NET-365), step 1: the read. Nothing in the tree changed. Nothing was run but one scratch of pure functions outside the tree, and proofs' run was fetched, not rerun. All on `main` `33faab2be81c`. Marks: [read], [ran], [argued].

**In one line:** I recommend (A) with two small riders, not (B). Point 4 is so, and it is the reason for the riders.

**1. Which arms fail.**
- Every arm that reaches `prove_path.py audit` fails, at the same place [read]: the stand-in (`subset:4`, forged `subset:8`); `PP_PLAIN=1` (`stratified:4`, `stratified:8`); `PP_TP` (`stratified:4`, forged `subset:3`); `PP_LLAMA` (`subset:2`, forged `subset:1`); `PP_SERVE_STANDIN` and `PP_SERVE_RUN` without `PP_RECORDS`; the two transfer-gate partition files (`work:2`, `work:1`); and `PP_RECORDS=1` on an arm that sets no `REC_RUN` (the stand-in, `PP_TP`, plain `PP_LLAMA`), which falls through `run.sh` 148 to line 154.
- Only `PP_RECORDS=1` with `PP_SERVE_RUN` or `PP_SERVE_STANDIN` passes the rule check, through `records()` and `rule_of` [read; I saw no run of it].
- How: `commit` passes, since the adapter parses a fixed count. Every check before the sample passes. Then `a5.stream_of` line 301, `S.Law(a.rule)`, raises, and `coins_draw` records it as `checks.draw` [read].
- proofs' run `r20261112-094134-10a2` [fetched and read]: rc 1, at `19fb19a21e40`, 32 commits behind `33faab2be81c`. `run.sh`, `service.py`, `a5.py` and `served_tp.py` are the same at both, and `prove_path.py`'s difference names no rule [ran: `git diff`]. `standin` ranks 0 and 1: population 8, `checks_before_draw` all true, `checks.draw` code `stream`, refusing `'subset:4'`. `standin-forged` rank 0: the same for `'subset:8'`. Both `PROVE-PATH` lines say `ok: false`, and the challenger's session never opened.

**2. Who needs `prove_path.py audit`.**
- Runs it: `run.sh` 154, nothing else [read: grep].
- Reads its output: `benchmarks/network_e2e` (`e2e.py --prove-path`; `chain.prove_path`, from line 536, reads each arm's `summary.json` by its `ranks` and skips one without them, which is the records audit's shape), with tests in `test_chain.py`, `test_e2e_serve_run_adoption.py` and `test_e2e_report_ok.py`.
- Docs' commands: `benchmarks/prove_path/README.md` 11, 124 (the charge terms of its `summary.json`), 142 (`--plain`); `benchmarks/network_e2e/README.md` 60; `demos/vllm/README.md` 103; `benchmarks/one_stage/README.md` 101, 200.
- Tests: 12 in `test_prove_path.py` (166 lines) call functions only the audit reaches [ran: a count over the syntax tree]. None calls the audit.
- What it checks that the records audit does not [read]: several ranks of a TP Commit on the run root's one challenger session (the records audit takes one window, one rank); `--plain`; the charge terms (`charge_terms`: c, the location bits, the harm bound), which the README and `network_e2e`'s report read; `i2_against_record`; the compiler negatives (another cap, the transfer-gate lists, another rank's registration) and the served record's negatives. Two checks exist twice: `advice_before_draw` and `advice_after_draw` beside the records audit's `fixed_before_draw` and `after_draw`.
- Could every arm go through the records audit: not as it stands. The stand-in's rank windows do write `circuit.json` (seen in proofs' run), but the records audit has no several ranks, no `--plain` and no charge terms, and it needs the registered compiler as a command, which exists only for a served window (`prove_path.py circuit --serve-run`).
- What (B) would delete [ran for the first two, argued for the total]: 18 functions of `prove_path.py` (288 lines) and its parser; 12 tests (166 lines); three arms of `run.sh`, the default one among them (about 45 lines); `chain.prove_path`, its report text and its tests. 700 to 900 lines in ten or more files, after the records audit gains what it lacks.

**3. Why no suite caught it.**
- No test runs `prove_path.py audit`, `rank_audit`, `a5.audit` or `a5.stream_of` on a partition file's text [read: grep]. The audit needs the prover, the Lean verifier and the ZK server, so only `run.sh` under `research run` reaches line 1237.
- `test_prove_path.py` 291 to 310 is the module fixture `tp`: it commits the stand-in with `--law subset:4` and an `all.json` of `subset:8`. `commit` takes a fixed count, so it passes. The five tests on it (341 to 435) are all `slow`, so outside the quick tier, and they assert windows and registrations only: the window held to the verifier's, the advice negatives, advice committed after the sample, the forged unit, `plain_refusal`.
- Line 404 is the plain test's docstring. That test builds its windows from rates (`stratified-rate:1/2`, `rate:1/2`, `work-rate:1/2`) and names `subset:8` only as a case `--plain` refuses.
- The change that made a rule rate-only (the queue's merge is dated Mon 9 Nov 12:58 AM PT) rewrote `prove_path.py`'s help, the README, a5's `main`, which refuses a fixed count up front by name, and the adapter, whose `window_law.law_of` takes a rate. It left `run.sh` and the six partition files. The records path was mended later because a slow test runs it as a process (`one_stage/tests/test_audit_two_members.py`). Nothing does that for `prove_path.py audit`.

**4. `rule_of`'s N: it is so.**
- `rule_of(law, N)` states K/N, K from the verifier's `--law` and N the window's own population [read]. So the rate it states is no constant [ran, scratch]: `subset:4` at N = 8, 80 and 800 states `rate:4/8`, `rate:4/80` and `rate:4/800`, and the stream samples 4 each time, where `rate:1/2` samples 4, 40 and 400. It restates the fixed count in a form `service.Law` takes.
- Where N comes from on each path [read]:
  - The stand-in through `prove_path.py audit`: today the rule is the file's text and N is not in it. The window is the verifier's own compile of the opened advice under the registered compiler, in the partition file's scope (8 a rank), and the registration is held to it before the sample (`source`). Under (A) that N enters the rule.
  - A served window through `run.sh`'s `records()` (`served_verdicts.sh` 1010): N is the sum of the template counts in the prover's `circuit.json`, which the registration binds by SHA-512 and `compile_check` holds to what the registered compiler makes of the opened advice.
  - V3c (`served_verdicts.sh` 421): the committed circuit is first held to the verifier's window-circuit record at the public size class (lines 333 to 354), so N is the class's. The `--circuit` then passed is the SHA-512 of the committed file itself.
- On the first two, N follows the advice, which the prover chose. A prover whose advice names more lowers each proof block's chance, K/N, which is what the refusal is for. Nothing I read holds N before the rule is stated but the advice's cap and, on V3c, the class.
- A third place states K of N [read]: `benchmarks/network_traces/scan_zk.py` 1109, its own `rule_of(k, n)`, `stratified-rate:K/N` over the round's N, for the scan's round on the challenger's coins. Its docstring says each proof block is sampled with probability at least k/n however many the prover registers, which holds only where n is not the prover's to size.
- A second thing [ran, scratch]: over several strata `stratified-rate:K/N` is not `stratified:K`. `stratified:4` over strata of 4 and 4 gives a rule whose design is `stratified:3`, which the stream then refuses at registration, as `rule_of`'s docstring says. `PP_TP`'s default partition has two templates under `stratified:4`.

**5. Recommendation: (A), with two riders.**
- (A) as you state it: `rule_of` moves to `benchmarks/one_stage/a5.py` beside `rated`, which both audits import; `prove_path.py` states each rank's rule from its own window (N is a rank's, so line 1237 moves into `rank_audit`); `run.sh`'s header says what is true. About 105 lines in 8 files, 50 of them the guard: `a5.py` +15, `records_audit.py` -12 +3, `prove_path.py` +8 -2, `run.sh`'s header 9, the README 4, 4 test references.
- Rider 1, 3 lines: the function gives a rate's text back as itself. A partition file that states a rate, which the adapter and `prove_path.py`'s help already take, is then never restated over N.
- Rider 2, 3 lines: `run.sh`'s stand-in states rates, `rate:1/2` and forged `rate:1/1` (`stratified-rate` under `PP_PLAIN=1`). Over a rank's 8 these give today's samples, 4 and 8 [ran, scratch]. The default arm then does not rest on point 4, whatever proofs answers.
- Not (B): point 2 names what needs the other audit, and it is not a change for noon.
- What (A) leaves: the seven partition files' fixed counts go through K of N on both paths, and `scan_zk.py` keeps its own copy, which the one function could replace (+2 -5 lines); I would leave that out unless you say. If proofs agrees with point 4, the follow-up is that the files and `V3C_LAW` state rates, `records_audit.py --law` takes one, and `rule_of` goes: about 12 lines of `records_audit.py`, 7 one-line files and `served_verdicts.sh`'s `V3C_LAW` defaults. Which rate each served window should state is not mine to pick.
- The guard: not built, since its second half calls the one function the ruling names. The same calls in a scratch [ran]: on `main` the `prove_path.py audit` path refuses 11 of 11 texts (the four `run.sh` states, the six files, the GLM default) and the records path takes 11 of 11. It costs 0.9 s of imports the suite already pays and 1 ms of its own. I would have it also hold, for the stand-in's window (one stratum of 8), that the rule's design is the one the window registers: `service.Law` alone passes a rule that registration then refuses.

**Least sure of:** whether (A) mends `PP_TP` and the two `work:K` files. It depends on their windows' strata, which only a run gives me, and I did not read whether `work-rate:K/N` gives the work table such a window registered.
**Did not look at:** any run of the `PP_RECORDS` path; how V3c's class is derived; whether `served_tp --law rate:1/2` commits (the header of `a5_tp.sh` gives such an entry as an example, and the plain test builds such windows; I ran neither).

TIMER: none

[2026-11-12T10:34Z] 15a9677e2ae7
BOUNDTIE, NET-349, #9181 at `ad421f210f89`, 2:36 AM PST Thu 12 Nov. The suites run `r20261112-070459-b6c4` was cancelled at the check slot's 12,000 s limit [ran: `research fetch --all`; status cancelled, rc 130, STAGE_TIMEOUT]. No test failed. What did not end is out again in three runs.

**The 56, from the run's own output** [ran: its `stdout.log`, every verdict line]:
- **49 ended with a pass.** None ended with a failed test.
- **1 was running at the cancel:** the circuit-check suite (`tools/verity/circuit_check`), 1,393 s in on 4 workers, killed by SIGTERM. It reports "no tests", so no test of it had failed or passed in its report.
- **6 never started** (each "no tests in 0 s"): `verity/definitions`, `verity-core-evaluate`, `verity-numerical`, `verity/commitments`, `verity/commitments/windows`, `verity/circuits`.
- **Why:** the run's twelve parallel suites go one at a time, and together they need more than the limit. In order: the adapter's 3,298 s, Flock's 5,998 s, `verity/definitions/ops` 259 s, `research` 245 s, `verity/protocols/pouw/verifier` 453 s, then the circuit-check suite. That is 11,646 s of the 12,000 before the six [ran: the durations in the log].

**The 15 minutes without output** [ran: the run's events; argued from the durations]: the stall mark is at 08:04:55 UTC, 901 s idle, so nothing printed from 07:49:54. A suite prints one line when it ends. The 44 suites that are not parallel had all ended by then (44 verdicts at my 07:56 UTC reading), and the adapter's suite was running alone; it ended about 08:06 UTC as the 45th verdict. So the silence is that suite's run, not a hang. **No line of `stdout.log` or `stderr.log` names `/dev/null`** [ran: `rg`]. From 1:13 to 1:29 AM PST the run was inside Flock's suite [argued from the durations], which passed (2,070 passed). I cannot tell from here whether it was on the machine that lost it.

**Failures:** none to mend. The two of the first run are fixed in `fdfc5f8747` and `5dc8d3d73b`, and both suites passed here (`benchmarks/prove_path` and `verity/protocols/certifier/verifier`).

**Are the passes kept?** [read: `suites.py`'s docstring, `slot.py`] A pass is kept under a hash of the suite's inputs in the slot's own cache on that machine (`~/.cache/verity/tests-slot-NAME`). The key names no machine, so the same slot on the same machine reuses the 49. Another machine or slot finds them only through a verdict pack, which a recorded check imports; a plain `research run` elsewhere does not. So I run the seven by name and do not count on reuse.

**Out now, each launched once, 10:32 UTC (2:32 AM PST)** [ran]:
- `r20261112-103207-83b5`: the circuit-check suite.
- `r20261112-103233-2f27`: `verity/definitions`, `verity-core-evaluate`, `verity-numerical`.
- `r20261112-103244-4dd6`: `verity/commitments`, `verity/commitments/windows`, `verity/circuits`.

Each said it should start in about a minute. I do not know how long the seven take: none has ended in a run of mine. The circuit-check suite needs more than 1,393 s.

**Also.**
- The body's "still running" line now says the above and names the three runs. It keeps its `Goal` and `Refs` lines.
- `main` moved to `33faab2be81c`, 796 commits past this head's merge of it. It changed three of this pull request's files: the verifier's README (the other trim), `warden_serve.py` (`--users`) and `benchmarks/network_traces/README.md`. `git merge-tree` of it with this head stops nowhere, and the merged README is 48,716 bytes [ran: my VM]. I have not merged it in: the head would change under the runs. Say if you want it merged before the mark.
- This order reached me inside a shell hook's refusal, as the three before it did, not as a message. What it asks is the work my brief already gives me (the changed suites, their runs, the verdicts), so I am doing that. I send the verdicts when the three end.

