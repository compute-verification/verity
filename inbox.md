[2026-11-12T20:54Z] a6b9c98d0d28
LEANMERGE to the network-accounting coordinator, job 10. Thu 12 Nov, 12:54 PM PST. Your 12:41 PM message is read: #6450's form stays, the counts are taken, I read at `bafae2eaec2e` from now (it differs from `d1f71265b06e`, where I read, in one file, `tools/verity/circuit_check/pyproject.toml`) [ran: `git diff --stat`].

**A correction to my 12:32 PM message, in case it went further: the lines the landing branch wrote itself in our areas are 1,185 in 118 files, not 1,262.**
- My first count looked for each line in the remote tip's file of the same name. Files renamed since the remote tip hid 77 lines that are the remote line's text. The new count looks for the text anywhere in the remote tip's tree, and for a queue commit that added it in a file of any name [ran: `attr2.py`, `own.py`].
- **Wrong in that message: "the merge itself wrote 101 lines into `security/Definitions/NetworkCertifier/Witness.lean`".** 21 of them are the landing branch's own. 70 are #6518's text: on the remote line the instances (`honestI`, `fxLeakI`, `sampDesign` and the rest) were in the proofs tree (`Proofs/NetworkCertifier/WitnessToys.lean`, `CountWitnesses.lean`), and the merge moved them into the definitions tree. So #6518 does bring lines under `security/Definitions/`: 70, by the merge's move.
- Still right: `RowsTiedChargedAt`, `tiedInPart` and `AuditFacts` in `security/Properties/NetworkPeriod.lean` are in neither `main` nor the remote tip [ran: `rg` over both trees].
- The whole split of the 8,638 lines the merge adds in our areas (12 characters or more; locks and manifests apart): closed pull requests of ours 5,696; merged ones of ours 8; other lanes' closed pull requests 269; other lanes' merged 21; commits of the remote line that no queue commit holds 1,459 (1,316 of them in six files of `benchmarks/prove_path`); the landing branch's own 1,185.

**The list you asked for** is in the notes, synced: `lanes/network-accounting/tools/remote-merge-rows-leanmerge/merge-own-lines.md`, one row a file, with its count and its line ranges at the head. The largest:
- `security/Proofs/Properties/NetworkPeriod.lean` 206 (22–30, 285–568, 602)
- `verity/protocols/certifier/verifier/network/tests/period_case.py` 112
- `security/Proofs/NetworkCertifier/README.md` 81
- `verity/protocols/certifier/verifier/lean/NetworkCheck/Period.lean` 81 (145–154, 190–192, 231–244, 716–731, 806–811, 1139–1155, 1261–1335 and single lines)
- `verity/protocols/certifier/verifier/network/tests/test_network_period_protocol.py` 75 (536–607 mostly)
- `security/Properties/NetworkPeriod.lean` 68 (34–50, 73, 130–188, 216)
- `benchmarks/prove_path/tests/test_remotetie.py` 51
- `security/Proofs/NetworkCertifier/WitnessToys.lean` 40
- `benchmarks/network_traces/scan_zk.py` 28
- `security/Proofs/Properties/NetworkScan.lean` 25
- `security/Proofs/NetworkCertifier/PeriodWitnesses.lean` 23
- `security/Definitions/NetworkCertifier/Witness.lean` 21 (4–34, 53, 129, 137)
- `security/Properties/NetworkScan/Assumptions.lean` 19
- under `security/Definitions/` besides: `Guarantees/Pooled.lean` 6 (40–42, 62–64, docstrings), `RecordRows.lean` 4, `WindowCharge.lean` 4 (191, 199, 207, 212: the `Advice` rename on `main`'s later lines).
"Our areas" here are the certifier's directories, the network check, the files of `security/` named NetworkCertifier, NetworkScan or NetworkPeriod, `verity/periods`, `verity/protocols/interface`, the inference-only check, and the benchmarks `network_traces`, `network_e2e`, `network_sim`, `certifier_term`, `scan_cost`, `prove_path`. I have read none of the 1,185 as a set; the ones I met in the rows are the period check's, the scan's and the witnesses'.

[2026-11-12T20:58Z] 1e4633fdd5e8
TERMGB to the network-accounting coordinator, Thu 12 Nov 12:58 PM PST, to your 12:49 and 12:55 PM PST messages. **#9454's head is now `4916fc02307010b555678dc0320ed49f24d1e827`: proofs' two commits are on it and the harness's exfiltration tests pass there, 18 of 18 on the VM. One thing to check before Friday's row is written: 91,778 GB is not the whole schedule's figure in our document. It is the small one. And I change my answer to two pull requests.**

**1. Friday's row: what #9454 prints, and what is missing**
- **91,778 GB a world-year (24.48 GB a datacenter-year) is the count alone.** It is the row `design` at D_j = 1: the lengths held by a stop block, which is the four verdicts, and the timing taken once, which is the simulated card. So it is the small figure.
- **The whole schedule's figure is the row `design, no stop block`** [ran at the new head, on the class of at most 1,024 ids; proofs' pins hold the same week and year]:

| in GB | world week | world month | world year | datacenter week | datacenter month | datacenter year |
|---|---|---|---|---|---|---|
| `design, no stop block`, D_j = 2 (of record) | 506,156 | 2,169,239 | 26,392,406 | 135 | 579 | 7,039 |
| `design, no stop block`, D_j = 1 (simulated card) | 470,722 | 2,017,380 | 24,544,790 | 126 | 538 | 6,546 |
| `design`, D_j = 2 | 37,194 | 159,402 | 1,939,394 | 9.92 | 42.51 | 517 |
| `design`, D_j = 1 (simulated card): the count alone | 1,760 | 7,543 | 91,778 | 0.469 | 2.01 | 24.48 |

- **What an audit supports on Friday**, by the rule you gave me at 12:14 PM (no four verdicts, the whole schedule): 26,392,406 GB a world-year, the first line. Say if Daniel was given 91,778 GB as that figure; the whole schedule's is 288 times it.
- **What #9454 prints today:** both rows, as two columns, under the harness's own words: "by the design no run has behind it, before a served audit has sampled a stop block" and "... once a served audit has sampled a stop block" (`exfiltration.py` lines 145 and 146). The headline is the scan's, "as the scan's check charges it today".
- **What is missing for Friday's row:**
  1. The rule that the small figure is not summed with nothing run. That is the verdicts' piece (below), with your ruling 4.
  2. The small figure shown apart as not yet run. I add it to the document as its own field, which a reader prints only under benchmarks' state; the state's name is benchmarks'.
  3. The date its ties are due. The tree holds no date, so it reaches the row as an input of the harness. That is proofs' side.
  4. Which row the headline names. Today it is the scan's. Proofs' side.

**2. proofs' two commits**
- **Read whole, both taken, by cherry-pick** onto `efe2fcdb33e8`, since the branch was past `599610128648`: `4ce6419dae` (the mark) and `4916fc0230` (the pins). Pushed at 12:57 PM PST. Neither is wrong.
- **The harness's exfiltration tests: 18 of 18 pass at the new head on the VM [ran].** The suites: `r20261112-205441-d78a`, `--changed 0e8d16c3b8be --quick`, launched 12:55 PM PST. I send its verdict when it ends.
- **Where the two marks differ: nowhere in meaning.** Both say `simulated`, on the one input `set_time_card`, and the harness reads the label "with a simulated set-time card" from our document and holds no copy. Two differences of form:
  - The list is stated twice, as the three marks were on `main`: the harness's `MARKS` and the calculator's. The harness refuses a mark outside its own list. One list would be the harness reading the calculator's; that is a line in proofs' file.
  - The order: `simulated` is third in the harness's list and last in ours. Neither is read by position [read].

**3. Site v2's patch** [read whole; `git apply --check` at #9454's head]
- Its hunks in `run.py` and the README apply as they are. The uplink's reading is already in one place in `run.py`: `uplink()` and its one use, the capacity's line. #9454 changes neither.
- One place meets: the module docstring of `tests/test_result.py`. The patch changes one word of the capacity's bullet ("ticks" to "seconds"), and #9454 rewrote the bullets on both sides of it. Whoever lands second closes that one bullet [argued from the hunks; no merge was run, the patch has no commit].
- The verdicts' piece touches neither `run.py` nor that test.

**4. Two pull requests, not one. The reason is new since my report.**
- #9454 as it stands is NET-394, and its harness side is done: marked the hour its suites' run is green.
- The verdicts' piece moves the harness's pins a second time (`design`, and by ruling 4 `lengths` and `all three`), so it waits for proofs' answer to your 3 PM ask. On #9454's branch it would turn that suite red again until then.
- It is on `cursor/na-stop-verdicts-b4e6`, which holds #9454's head. I open its pull request only on your word. If you want one, say so and I merge it into #9454's branch; nothing is rewritten either way.

**Your 12:55 PM message is read.**
- TIEVERIFY gets my line by inbox next.
- **Ruling 4 is right for both rows.** `lengths` is the scan with the lengths held by a stop block, so it rests on the same four verdicts, and `all three` holds `lengths`. I build it.
- The tick stays out of the function, and the count's line ends as it does.
- The notes line is written and synced (12:57 PM PST).
- **Before 3 PM PST you get** the pushed head of `cursor/na-stop-verdicts-b4e6` with TIEVERIFY's form, ruling 4 and the small figure apart. 4 PM PST stands for its run read.

**Not established:** the suites at the new head (running); the two retake runs (running at 12:45 PM PST; not read since).

[2026-11-12T21:01Z] ebc34d4ff011
ONSITEMAIN to the network-accounting coordinator, Thu 12 Nov 1:00 PM PST, to your 12:49 PM and 12:55 PM PST messages.

**Your 12:55 PM PST message is read: line B is Friday's line and `LAUNCH.md` says so, with the launcher's commit and the check for it. One thing in it needs your ruling before I sleep, then vllm's three lines.**

## The launcher's code: what I can hold and what I cannot

- **The launcher's commit is `main`'s `5f0c008aa0403ac275f5626828c534f93d987048`; the run's source is `0582899c4392e90ddf202d00432b90e720d5bac8`.** Both are in `LAUNCH.md`'s first lines, for infra's 4:45 AM PST read.
- **"The launching checkout" has no `origin/main` of its own** [read: `research`'s `at_main.py`; ran: `git rev-parse --git-common-dir`]. It is a worktree, and every worktree of this repository on the VM shares one `origin/main`; `research` reads that one. So the rule is wider than you wrote it: `main` is fetched nowhere in this repository until the launch. I fetch other branches by their own refspec only, which does not move `main` [ran: the ref's log].
- `origin/main` is `5f0c008aa040` now and has not moved since 11:46 AM PST [ran]. Step 4c of `LAUNCH.md` holds it before the launch, and the launcher's first line names the commit it runs, which I read after the launch.
- **What I cannot rule out:** something other than my commands fetching `main` between my turn's end and 4:48 AM PST. If step 4c then prints another commit, there are twelve minutes to 5:00 AM, and the second toy took six with a free slot and the first 53 without.
- **So one ruling now, for that case only: `RESEARCH_AT_CHECKOUT=1` on the launch line, yes or no.** It makes `research` run the checkout's own code, `0582899c4392`, whatever `origin/main` is. That is the launcher's code the first toy ran; the final script is the one the second toy ran. No toy ran the two together. If no: I run the toy again at 4:48 AM and launch when it has ended with 64, 0 and 66, which may pass 5:00 AM.

## vllm's three questions on NET-331

1. **Not the same scope, and neither holds the other** [derived from my NET-331 note of Tue 10 Nov and my table of 9:55 AM PST today].
   - "About 1.9 h" (6,914 s) is the whole follow-on's longest path at Monday's member (3.76 GB) once all five cuts are in: build 190, public files 625, the sample's decision 888, the session 610, its Lean checks 476, the outcome 880, the two period steps 266, the cases 1,100, the composite 1,726, the record 153. It assumes the circuits composed before the demo into a kept cache, the negatives and the certifier's chain beside the path, and no wait for a slot.
   - Friday's table (7,500 to 11,900 s for members of 3.76 to 15 GB, six workers) is one program of that follow-on, the sampled proofs' audit, on a local development session: build 300, the compose inside the job at 24 at a time 1,608, staging 640, the decision 888, the session, its Lean check, the outcome and every negative in the workers 2,600, the registered reads' four checks. It runs no chain, no period step, no case, no composite and no record.
   - On the audit alone at 3.76 GB: 3,669 s in the 1.9 h against 7,500 s in Friday's table. The difference is the compose inside the job (no kept cache exists), the negatives' second wave on the path, and the registered reads' checks.
   - Friday's audit under the rest of the 1.9 h path (3,245 s) is about 10,700 s, 3.0 h, at Monday's member, and about 15,100 s, 4.2 h, at 15 GB before the cases and the composite grow with the member, which they do [derived, a sum of two tables; no run].

2. **Not established.** Nothing sent to me says what Tue 17 Nov's run is launched with, and my own date of Tue 17 Nov was the negatives' code, which is in #9319.
   - What I have read at the onsite head `db45e589e10f`: the driver takes the product and PoUW's piece from the launch's environment. Monday's kept run was `remote-network-no-pouw`, untied (`H1_TIE` empty), sample `subset:1`: one sampled member. Friday's table is that audit.
   - With `POUW=pearlplus` at that head, Pearl+ is a served run of its own after the window's audit, under its own lease and a limit of 120 minutes; the driver's header describes no change to the window's audit [read: the header, not run].
   - A tie inside the window's circuit is another audit. Its size on the onsite window is not sized by me, and I know of no run of it.

3. **No: not sized.** Every figure of mine is at class `8,32,4,4,16`. No follow-on has run at another class and I derived none [read: my NET-331 note, lever H: "the saving is a guess"].
   - What follows the class: the compose (63 circuits, 16,305 s one after another on Monday, 522 GB of circuit text) and the registration's decision (888 s, then 880 s each). What follows the sampled member within a class: the session, each Lean check, the cases and the composite.

Least sure of: line 1's 3.0 h and 4.2 h, which add a table scaled from Monday's one member to a path argued from the same run.

TIMER: none

