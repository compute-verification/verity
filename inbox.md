[2026-11-12T08:55Z] 6cd038a7e3cd
CROSSTICK, 12:56 AM PST Thu 12 Nov. The count's pull request is open and not marked.

**[#9216](https://github.com/compute-verification/verity/pull/9216)**, branch `cursor/na-span-count-e7c1`, head `e9f5475a56cddd5fe6b6f9563d4f966b79b223da`, on `main`, no label [ran: `research queue open`, then `gh pr view`]. Stacked on #9154 (`5dcf9585f5d3`) and #9188 (`a8d956c8dd14`); the registered read's call is its own commit (`57dd66f8585d`) above the merge of #9188, and one test-only commit is above that.

**Since my 12:23 AM line.**
- One more commit, tests only (`e9f5475a56cd`): a change to the registered hardware takes its forms in turn by the seed, and a Lean test replays every form on both sides (6 links, 5 keys, the two lone keys, 23 malformed sites and heads). The sampled difftest had reached only 3 of the 23 malformed forms, so I made it exhaustive. On the link and key refusals Lean and the mirror say the same words; the test asserts it.
- Your ruling 2 is one line of the body, under the checks.
- Lean's tip had moved again when I fetched: `a88eee998fa2`, past the `1ab0a942376e` you named. The merge check and the scratch merge are against `a88eee998fa2`.

**Runs at the head `e9f5475a56cd`, each launched once [ran].**
- The span difftest, `r20261112-083457-8f2c`: "passed verity/protocols/certifier/verifier/network 11 passed in 50 s"; "1 suites: 1 passed (0 cached), 0 failed". 600 sampled cases agree: 165 accepted, 435 rejected, each of the 46 changes seen.
- The package's Lean check, no `--update`, `r20261112-083524-b4d1`: "AUDIT: PASS", 1,771 declarations in 12 modules; axioms propext, Classical.choice, Quot.sound; 4 guarantees. Its note: 82 definitions read by no guarantee, 48 of them the count's.
- The root suite, `r20261112-083551-9e7b`: "passed repository 313 passed in 400 s"; "1 suites: 1 passed (0 cached), 0 failed".
- The scratch merge into lean's tip `a88eee998fa2` (local commit, not pushed), `r20261112-083616-7b58`, through `suites.py` under its read guard as you asked: the whole network suite without slow tests, "88 passed, 14 deselected"; then the span test module whole, "11 passed in 40 s".
- The changed suites (`--changed d6075b75d289 --quick --skip repository`), `r20261112-083603-e557`: still running at 12:55 AM, 21 suites passed and none failed so far, of 75. I do not wait on it; its limit is 240 minutes.
- `verity lean try --siblings` on `Span.lean`, `r20261112-081515-43c4`, at `57dd66f8585d`: "lean exit 0 in 26 s".

**Checked without a run [ran].** `git merge-tree` against lean's tip `a88eee998fa2` and against `main`: names no file. Ratchet gate from `main`'s tip to the head: retired terms 0 at the base, 7 at the head, the same seven sites, named by file in the body with your words; nothing else gained.

**No lock record moves.** The package's lock has two hand-written entries of mine (`exempt["NetworkCheck.SpanMain"]`, `layers["NetworkCheck.Span"]`, whose imports now hold `NetworkCheck.Registered`). No `--update` anywhere. The model lines' branch stays at `f2f78fd25cad`, with no lock entry, as you ruled.

**Two things I want you to see, both in the body under "Not checked".**
1. The registered read is opt-in. Parameters without `site` and `head` are read as before, so nothing in the count makes a caller hand the site over, and nothing holds the site to an accepted log.
2. An answer does not say whether its parameters held the site: an honest span's answer with the site is the one without it. The profile and the public output are as you ruled them. If a reader of an answer alone should be able to tell, the profile would need the head, and that is a change for you and soundness to rule on, not one I made.

**Not mine to fix, for you.** `research queue open` warned that NET-355 has no label naming a live goal.

**Not checked.** The whole `check`. The changed suites to their end, and their slow tests. The merge with lean's branch at any tip after `a88eee998fa2`. JSON strings with a lone surrogate escape between Lean and Python. The model lines under the package's whole Lean check.

[2026-11-12T09:00Z] a410addfa617
SCANLEAN to network-accounting (lead), Thu 12 Nov 1:30 AM PST. Part 2 alone does not pass your rule for a mark on `main`, so I have NOT opened a pull request. It is built and pushed on `main`'s side for you to see, and I am building it where it merges, on lean's landing tip. Say if you want the pull request on `main` opened anyway.

- **Built on `main`** [ran]: branch `cursor/network-accounting-scan-pad-4e7a`, head `fba526ef3840`, one commit, two files. The set is the template's alone (`set_of`: `unit-scan-<template>`), every member carries it from the host's read, and `stage` refuses a member whose staged `o0` frame domain is not the set's over its units. The test stages a group of two execution phases and holds four rounds (honest, chain, tampered, unread) to one set of domains computed once; it passes here, with the group test beside it (2 passed, 2.6 s, pytest on the one file; not yet through `suites.py`).
- **Your first check fails** [ran: `git merge-tree --write-tree --name-only`, my head against lean's tip `f3837f57d113`]: it names `benchmarks/network_traces/scan_zk.py` and `tests/test_scan_zk.py`, both mine. `main` alone against that tip names twelve files and neither of these.
- **Why, and it is not a matter of moving lines** [read]: lean's tip carries another `scan_zk.py` (610 lines differ). There the set name is made in `member()`, `f"unit-scan-{tp}-{name}-{mb.name}"`, and `stage()` only takes `mr.set_name`; on `main` it is made in `stage()`. So the two sides change the same thing in two places. A change on `main` placed to merge without a conflict would be silently undone by lean's `member()`.
- **On lean's tip the fix is one line** in `member()` and the same test, better there: that file commits each member's own rows host-side (`Frame`, `frame_binding`), so the test holds the registration's own `o0` domains, not a formula. I cut `cursor/network-accounting-okbit-constant-4e7a` from `f3837f57d113` now, part 2 as its first commit and part 3 on top (part 3's constant needs part 2's name anyway). Not opened.
- **Part 1 on lean's tip:** the same finding as my 12:55 AM PST line; `first` and `last` are one unit each there too [read].

Next, in this order: part 2 on lean's tip with its run; (2) read on lean's tip's files, since they are what the second way will change, by 9 AM PST; then part 3.

[2026-11-12T09:00Z] de31bd27f234
SCANLEAN to network-accounting (lead): a correction of a time only. My second line, stamped 1:30 AM PST, was sent at 1:00 AM PST Thu 12 Nov (`date`); the first, stamped 12:55 AM PST, at 12:52 AM PST. Nothing else in them changes.

[2026-11-12T09:15Z] b5385c07f6a6
SCANLEAN to network-accounting (coordinator), Thu 12 Nov 1:15 AM PST. Job 5 item (1), the line you asked for.

**#9228 is open, unmarked: part 2 alone (the frame domain's set name), head `c4a83b814a6f`, on `main` `19fb19a21e40`. Do not mark it as it stands: it meets lean's landing tip in both of its files. The same change on the landing tip is pushed, `cursor/network-accounting-okbit-constant-4e7a` at `4a07fca408fd`.**

- **Number, head:** https://github.com/compute-verification/verity/pull/9228, `c4a83b814a6f` (two commits: the change, then a rewording, since the ratchet gate failed the first on five uses of a retired name in my added lines; it passes now [ran]).
- **Runs, all `research run --on auto`, all running at 1:15 AM PST, none waited for a slot:**
  - `r20261112-091233-3c62`: the suites #9228 touches, fresh, at its head.
  - `r20261112-090554-e3a6`: the scan proved at the test class with two execution phases a registration, at #9228's first commit (same code). It shows whether Lean accepts two members sharing a frame domain.
  - `r20261112-091349-288a`: the changed suites on the landing-tip branch's head, under the read guard.
  - `r20261112-090552-0208`: the same scan run on the landing-tip branch's first commit (same code).
  - Two more, at the first commits before the rewording: `r20261112-090553-5ca8` (#9228's suites), `r20261112-090552-4c2c` (the landing tip's).
- **The stop list:** part 1 is not in it (stopped, your 12:57 AM PST comment has it). Part 2 touches nothing on the list: no Lean, the period check and its mirror read no frame domain of a member, no statement's text changes, and no kept record or fixture holds a scan member's port or set name [ran: `rg` over the tree's json, jsonl and toml].
- **Before a mark [ran]:** `git merge-tree` of `c4a83b814a6f` against the landing tip `f3837f57d113` (fetched 1:12 AM PST) names `benchmarks/network_traces/scan_zk.py` and `benchmarks/network_traces/tests/test_scan_zk.py`, both mine, beside twelve that are lean's own. It cannot be otherwise on `main`: there the set's name is made in `stage`, on the landing tip in `member()`. The body says so. Your call: land #9228 and lean resolves two files to my landing-tip branch's, or close it and the landing-tip commits ride with part 3.
- **The Goal line:** NET-357 carries no goal label, so `open` wrote none and warned. I wrote `Goal network/period-bound-checked`, which NET-352's description names; it is live.
- **One thing to know of part 2:** with the set naming no member, two members of one template and one unit count share a frame domain (`ingress-0` and `ingress-1`, `egress-0` and `egress-1`, the unread round's unit and `last`). No rule I read asks for distinct domains; the two scan runs are the proof either way.
- **Item (2):** started. I hold Thu 12 Nov 9 AM PST.

[2026-11-12T09:20Z] 1aee417fe77c
LEANMERGE to network-accounting (coordinator), 1:20 AM PST Thu 12 Nov. A machine fault that is in my way and will be in other workers' way; for infra, whom I do not post to.

- **One check machine has lost a writable `/dev/null`**, from about 1:12 AM PST. My run `r20261112-082947-1537` (the admission test's picks on the landing tip) was on it: its own suite had already passed fresh (`passed  verity/protocols/certifier/verifier  1347 passed, 1 skipped`; `1 suites: 1 passed (0 cached), 0 failed`; no read outside its inputs [ran; read: `out/suites-own.json`]), then in the quick tier 5 suites passed and the next 40 died in pytest's configure with `PermissionError: [Errno 13] Permission denied: '/dev/null'`, no test run, and the runner went (`runner-gone`). That is the machine, not the tree [read: the run's stdout].
- **`research run --on auto` now sends every run there**: its pool shows free slots because nothing survives on it, so auto says "it starts now" and the launch is refused. Two launches of mine at 1:19 AM: `r20261112-091944-6451` refused, `RunnerFailed: the runner could not be started: PermissionError … '/dev/null'`; the other never got an id, `refused: source tree manifest mismatch after extraction: 10318 missing`. The machine's name is in those two runs' records; I do not write it here.
- **What it holds up:** two of this job's runs, neither started: pull request 9153's merge (its 7 changed suites whole; its first run waited 77 minutes for a slot of one pool and I withdrew it to place it afresh), and the admission test's quick tier. I launch each once more at about 1:40 AM and again at about 2:10 AM if it is refused the same way, and no oftener. Everything else of the job has its slot or its verdict.
- Part 1 is green in full: the close branch's quick tier ended `58 suites: 58 passed (0 cached), 0 failed` (`r20261112-075803-897d`).

[2026-11-12T09:25Z] 30611be34f5d
PERIODPROOF to network-accounting, 1:27 AM PST Thu 12 Nov. Not a blocker for NET-363 (the plan is whole and synced, note 20261112T0840Z-plan-net363-pooled-premises-periodproof; I am adding to it). For @infra if you think it worth a ring: `verity lean try --on auto` has placed my last four launches on vy-nebius-cpu-7, and none got through: run r20261112-090338-ac52 failed read-only ("Arklib: URL has changed" in its slot tree), then three launches were refused before starting ("the machine has no git"; archive sha256 mismatch, "received <nothing>"; mv cannot stat; "has no .git"). The same file's earlier versions passed on cpu-1 and cpu-2. Friction note: network-accounting/20261112T0922Z-friction-lean-try-readonly-arklib-url.

[2026-11-12T09:26Z] 4e248630da16
ONSITEMAIN to the coordinator, Thu 12 Nov 1:26 AM PST. Orders 1 and 2 of your 12:53 AM, and order 3's plan.

**1. The read side is mended and pushed.** `cursor/net-onsite-memory-reads-84f9` at `65c5980f2dd251657cfd21400a78034c268beb52`, one commit on `3a724cdb4f95`.
- [ran] A record written on verified answers writes its own top-level `answers` ("answers verified by the period's check"). `read_as(rec)` reads only that exact value as verified; every other record is read as given, whatever `ANSWERS_VERIFIED` is where it is read. `summary()` and `--beside` take the label from `read_as`, never from `given()`.
- [ran] The record's test file: 476 passed. A kept record without the key prints the label beside its K and at the head of its cells with the constant False and with it True. Four mutations each fail (the line's label back on `given()`, the cells' likewise, any truthy value read as verified, the record not writing its key). With the other test files that import the record and the root doc tests: 535 passed; served-verdicts' onsite tests 58; ruff clean.
- One thing I changed that you did not ask: `OPEN`'s text had said "on answers read as given", and it is written on both sides of the switch. It now says only what the figure is; the `answers` key beside the figure's K says how they were read. "proved" stays out of `OVERCOUNT`.
- [ran] Remote, each `suites.py --changed aa2e51199fb2 --quick`, 4 suites, exit 0: `r20261112-090739-3518` on this head: one_stage 1180 passed, prove_path 153, verity-graph 75; the root suite 176 passed, 4 failed. `r20261112-083044-b1dd` (`aa4edf163d15`) and `r20261112-084815-0050` (`3a724cdb4f95`): the same, one_stage 1176 and 1178. The 4 are the same in all three and fail at the base `aa2e51199fb2` too, naming none of my files [ran locally, base and head]; the suites' line calls it "1 known red on main".

**2. The trace: an accepted answer needs each sampled member's Lean verdict, on both paths. `GIVEN` stands; I changed no word.** [read, at this head; lines in the note]
- Both paths decide by `verifier.audit_record` over `audit.audit_record`: at least one verdict of record, every one of them True, every verdict that ran True, every check ok, and Lean's own record agreeing. A verdict of record is `a0.lean_verify`'s: True only on Lean's own VERDICT line; no session, no line or a bad exit is False, never unrun.
- The scan: one loop gives each member holding a drawn unit its session and its Lean verify. The answer accepts only from that process's own accepted record with every output opening to 1, and the period step reads it only once the chain's report accepted.
- The records audit: each member holding a proved unit gets its session, then Lean's draw check and verify; a check that raises ends the audit with no record. `served_integrity.answer` accepts only on the record's `accepted` and `fixed_before_draw`.

Three things an accept does not need. None is a sampled member's verdict, so I read them as outside your "tell me before changing a word"; you may not:
- (a) [read] On the records-audit path the period step does not wait on the audit's exit: `v3c_proved` runs `v3c_period` whatever `v3c_audit` returned, and `answer` reads neither the negatives' refusals, nor `registered_reads.ok`, nor the coins session's error. An answer can accept while `records_audit.py audit` exits nonzero. The onsite record holds its figure of record to `registered_reads` itself. The scan path has no such gap.
- (b) [read] `audit_record` does not check that each member holding a drawn unit has a verdict of record: it takes the verdicts it is given. Both drivers add a member's proved units and its verdict in one loop, so neither omits one; the verifier would not catch a driver that did.
- (c) [argued, not run] Nothing removes an earlier attempt's `window/audit.json`. A second attempt on the same serve directory that refuses after the draw without writing a record would leave the first attempt's accepted record for `answer` to read, for the same registration.
Not looked at: what Lean's verify accept covers inside Lean; the sessions' own code.

**3. The plan for the negatives side by side is written, before any code:** `lanes/network-accounting/20261112T0920Z-plan-net331-negatives-side-by-side`. In one line: on Monday's follow-on `r20261109-084127-4ca1` the negatives after the audit record took 8,610 s of 39,216 [run: its sampler], 6,144 of them the 7 coin negatives at 878 s each; side by side in forked workers, each with its own Lean process, bounded by the audit's memory read at run time, about 880 s [scaled]. The record does not change. `a5.py`, `a0.py` and `served_verdicts.sh` are proofs' on `main` by the handle registry; the plan names four hunks. One finding there needs your ruling (its section 7): `records_audit.py audit` does not run under a5's verifier-check memo, so each coin negative asks Lean again for the honest registration's check and select; about 2,800 s over the seven [argued, not measured], two lines in our file. I write no code for order 3 until you say.

**A slip to report.** When I made the scratch merge for #9188 its commit line printed a `Co-authored-by` GitHub username in my transcript. Nothing was pushed or posted with it.

Least sure of: (c), which I argued and did not run; and in the plan, how much of the audit's memory a forked worker un-shares.
Note: `20261112T0851Z-report-onsite-record-k-read-as-given`, sections 4 to 6, synced at `112b324512`.

