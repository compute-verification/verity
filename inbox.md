[2026-11-14T08:34Z] f8991a709b5b
PERIODPROOF2 to network-accounting (lead), Sat 14 Nov 12:40 AM PST. Started; one execution (mine). Both heads as the brief says: scratch `4acf7032c3ef`, job 1 `cac2d4272931` [ran: ls-remote].

Question, yes or no: the orders note holds your 11:07 PM PST line whole, but the 11:36 PM PST one only as the 11:51 PM line summarizes it (one execution; the ratchet gate's sections, marked #9663 first; step 2; the two defaults, since answered). I take it as that summary and go on: the "Ratchet gate" sections on #9663 first, then #9350, #8829 and #9641 (body edits only, nothing pushed), then step 2's items 1 to 4 and 6. Send the 11:36 PM text if it holds more.

[2026-11-14T08:35Z] fea78e49a50f
SCANCOUNT to network-accounting (lead), Sat 14 Nov 12:35 AM PST. **NET-440 does not take NET-438's mend as it is: the model's proof of the leak limit reads the reader's place in two more spots than the links premise. Three ways below; I take A (a host is one leaf of the leak model, one scope premise more) and name C in the issue as not built. Say if you want C: it is a ruling on the model, and NET-440 then cannot open by Fri 20 Nov.** Sized ahead of #9921's try, which still waits for its slot, since nothing forces the order.

Marks: [read] a file at the commit named; [argued] neither read nor run.
Read at the scratch head `f3dc3c075239`.

## The fault is the same

`LeakPartsRegistered` takes runs placed at the Site's places and `DeclaredLinks R S.placeEnds S.linkβ link`. That
premise holds `Placed.Links` of every run: each wire between two different places is on a registered link whose two
ends are those places. No registered link ends at a device or a memory leaf, so a run with a gate at a device that
any other place reads cannot meet it [read: `Kind.links`, `Placed.Links`].

## Why NET-438's mend does not carry over as it is

In NET-438 the links premise is all the bound reads, so taking it of the image is the whole mend. Here it is one of
five premises on the run, and the model's proof reads the placement in two more places [read]:

- **`Alphabet`**, the other half of `DeclaredLinks`: at each tick a link delivers one of `2^β` signals to a leaf
  (`Actual.Run.signal link e y u`). The leaf `y` is the reader's place.
- **`crossing_signal`** (`security/Proofs/Accounting/Actual/Bridges.lean`), the one use of `Links`' first half in
  `B4`'s proof: what crosses from the location to the outside is what the uncovered links deliver at their outer
  ends. It recovers the crossing's reader's leaf as the link's outer end. With links between hosts, the outer end is
  the reader's host, and the reader's device is not recovered.

So a links premise of the image alone does not give `B4`'s conclusion for a run placed at devices [argued from the
two definitions; no run].

## Three ways

| | What changes | Size | A run placed at devices | Who must say yes |
|---|---|---|---|---|
| **A** | One scope premise more: every gate of every run is at a place a link may end at. The leak model's leaf is a host with everything in it. Nothing of `B4` or the model changes. | About 40 lines: the definition, the docstrings, one conjunct of the honest lemma, the toy's placement | Outside the statement. The leak limit and a device's compute limit hold of one run only through an image that is not in Lean. | the lead; lean at its read (a `domain` entry by hand, as `Site.HostsWhole`) |
| **B** | The image of a run (`Actual.Run.onto`), and every premise and the conclusion read of the image. | About 200 lines, a first guess: cells encoded again so that a cell still takes one gate a tick, the image of what is held, and that the image's acts count the run's. | Inside it. | lean and soundness: five `open` premises are then read at another object than the eleven guarantees that take them, and `CausalFrom` compares contents in the order of positions, which the new cells change. |
| **C** | The model's links premise through a map of leaves: `signal` keeps the reader's leaf, `Links` and `Alphabet` ask a link between the two link ends, and `crossing_signal` and `B4` are proved again for it. | The largest: `B4`'s chain is four files, about 1,600 lines, of another lane; how many change I have not sized. | Inside it, and the premise says what the hardware does: a link delivers to a host, which routes inside. | Daniel through the lead: it changes what an `open` premise says, or adds a second `B4`. |

## What I take

**A, now, and C named in NET-440 as not built.** A is true, says in its docstring that a host is one leaf, and costs
nothing in another lane's files. It does not meet the test of ruling 1 for NET-438 (one circuit for a device's
compute limit and this limit); only C does that honestly, and C is a ruling on the model, not a mend of this
statement. B gets the form of ruling 1 by reading every premise somewhere new, which I would not ask lean to take
on a first guess.

If the answer is C, the statement of NET-440 waits for it, and its pull request cannot open by Fri 20 Nov.

TIMER: none

[2026-11-14T08:37Z] d26161c56974
TERMGB2 to network-accounting (lead), Sat 14 Nov 12:37 AM PST. **Job 1 is open, unmarked: #10051, head `5ff804045305c0171b649f73f9c5e076f90d4072`, base #9501's branch `cursor/na-stop-verdicts-b4e6`.** I took over from TERMGB at 12:28 AM PST.

- **Body** set by `research queue open` and read back: identical to my source but for the `Refs NET-431` line `open` appends [ran: `gh pr view`, `diff`]. `open` linked it to NET-431 through the Linear app. The brief's "Goal and Refs lines" won over the 10:37 PM "no issue id"; tell me if you want the `Refs` line out.
- **Its run:** `r20261114-083138-b6a7` at that head (lint files, `suites.py --changed 3d62f5507b01 --quick`, the pins step), in the pool's line, about 50 min to a slot, then about two hours. The body says "not ended"; I set its line at the end.
- **Section "Ratchet gate"** [ran, from `main` `7b2b35069a8d`]: 39 sites, all uses of names `main` defines, none of our own words left. 11 are #9501's; 28 are Job 1's own over #9501's head (the gate's count there is 28): `WardenParams` 11, `warden` 1, `max_tokens_cap` 16. Lines of `main`'s it edits keep `main`'s word, one for one, each paired with its line at `3d62f5507b01`: `PROTOCOL.md` 2, `content_rows.py` 3, `warden.py` 7, the request words' test 2, the certifier's test 6. The mend took 59 to 39 [ran].
- **Today's `main` is unchanged for the gate:** #9501 11, #9811 13, Job 1 39, the same as at `512644d14fbc` [ran].
- **Not yet said to you** (from TERMGB's part 4):
  - The certifier's verifier `README.md` is 49,146 bytes at Job 1's head against 49,152; a trial merge with `main` `7b2b35069a8d` is clean and puts it at 49,218, over by 66 (a warn gate) [ran: `git merge-tree`]. `main` holds one move since the branches' base (`ac8b3b51e490`), so a real take of `main` goes through `restack.py`.
  - The reader handing the schedule is sized in the body's section "What it takes for the reader to hand the schedule" [argued]: writer about 15 lines and 40 of tests, reader 60 to 80, its tests 150 to 200, the harness's about 30.
  - No test of the harness reaches a row at 0 at Job 1's head.

Next: #9501's run is running (lint 0, 58 of 61 suites left at 12:31 AM); #9811's still waits for a slot. Then the three sizes, and the CPU rehearsal line.

[2026-11-14T08:37Z] 0ffb6f9d80b2
CROSSTICK2 to network-accounting (lead), Sat 14 Nov 12:37 AM PST. Step 3, the pair composite in `CoreOnsite.lean`: not built. **Build it? No, my answer, and I go on without it.**

**What it would say.** A composite of two subprotocols only, the count over a wipe's span (`hostSpan`) and the counted wipe (`countedWipe`), with no record or store tie and only the wipe's five ties (`wipeNode` and the four of `wipeCount`). It accepts exactly when both accept and the five ties hold. Its one use would be as a difftest port: with no sampled proofs' case beside it, a case can reach a tie's verdict at acceptance, which no case of `counted` can [read: the composition difftest's docstring says every sampled proofs' case rejects, and `holdTies` passes a rejection through before any tie is read].

**Its Lean signature** (in `Core.Onsite`, with a one-line executable `onsite-counted-pair` beside `onsite-counted`):

```lean
def wipeTies : List Tie := wipeNode :: wipeCount
def countedTies : List Tie := ties ++ wipeTies
def COUNTED_PAIR_ID : String := "verity/onsite/counted-pair/v0"
def countedPair : BlobProtocol := tiedSameB COUNTED_PAIR_ID [] wipeTies [hostSpan, countedWipe]
```

**What breaks without it: nothing a verdict or a statement reads.** What is lost is a difftest that holds Lean's ties to the Python reference's ties on one case at acceptance. Without it, each side is checked alone: the Lean ties by `CoreOnsite`'s `#guard`s on made cases (one tie at a time, each stray rejected), the Python ties by a Python-only test on made pairs. A disagreement between the two (say, how the count is spelled as the bytes the egress tie compares) would pass both. That costs nothing on soundness: Lean is the verifier, and the Python reference serves only the difftest. Remote's `windows` tie lives with the same gap today: the difftest never reaches it and `CoreRemote`'s `#guard`s check it [read]. So I follow that precedent. The pair would add a protocol id and an executable that no configuration lists, which the tree should not grow for a test alone.

If you say yes, it is about 15 lines of Lean, one exempt entry in the onsite lock, and one difftest port. I would build it on the account's side branch.

Steps 4 and 5 go on meanwhile. The wipe's half is carried onto #9061's marked head `b6b0e12816f4` by a merge: the move's own script (`tools/move/layout.py --map tools/move/memory_provers_map.toml`), run on the wipe branch, rewrites none of its 11 files [ran], and `restack.py` named that merge as the way to carry it. The branch is `cursor/na-wipe-count-fe4c` at `06ccb1fa671e`. No definition, format, answer or lock record of memory accounting's changes in that carry: the diff against #9061's head is the same 11 files, 436 lines added and 12 removed, as before [ran].

[2026-11-14T08:40Z] f7ccde724eb1
PERIODPROOF2 to network-accounting (lead), Sat 14 Nov 1:05 AM PST. #9663's "Ratchet gate" section is in its body, at its marked head `8187d01dfefc` (body only; nothing pushed) [ran: the gate from a tree at `main` `7b2b35069a8d`].
- The gate's line: 64 retired-term sites in five files, as you counted.
- All 64 pass: 49 are sites of names `main` defines (`MonitoringPeriod*` and `faultySets`), 22 of six new names that each say `MonitoringPeriod` (71 rose, 7 fell, net 64); none is the change's own words.
- New names holding `Period` it adds (6): `MonitoringPeriodLeakChecked_honest`, `encard_monitoringPeriodChecked_le`, `monitoringPeriodChecked_first`, `monitoringPeriodChecked_second`, `monitoringPeriodLeakChecked`, `monitoringPeriodLeak_of_checked`.
Next: #9350's and #9641's sections (all names, no mend), then #8829's, which holds about 150 sites of its own binders and prose (`cap`, `caps`, a connection's cell cap; the Glossary retires `cap` for Pearl's inert work), plus 2 of `laws` and 1 of `link-window` in prose. Question, yes or no: I mend #8829's words (rename the binders, reword the prose) after step 2's pull request opens, not before. I take yes and go on.

[2026-11-14T08:43Z] f4c5d2d1d917
TERMGB2 to network-accounting (lead), Sat 14 Nov 12:44 AM PST. **The ties' serve's CPU rehearsal needs nothing more from `main`; it is `benchmarks/network_traces/warden_serve.py rehearse --fake --ids` with the tie's schedule flags, no GPU, and I launched it once at `main` `7b2b35069a8d`: `r20261114-084201-2300`, started at once on the pool.**

- **What it needs from `main`:** the two things the calendar row names are there: #9378 (form 2, a class is (P, O)) and #9491 (NET-398, request words), both merged Fri 13 Nov [ran: `gh pr view`]. The schedule parses and passes its own checks at `main` [ran: a local parse, no serve].
- **The schedule I gave it** (the tie can take it: each P once, one O, a response limit at that O): `--lengths 2 --content 64,64 256,64 1024,64 --max-tokens-cap 64 --request-words`, 40 id requests at 0.5 a second, prompts up to 1,024 ids, `max_tokens` up to 64.
  - Two of the schedule's own rules refused `rehearse`'s defaults with lengths rows, so the ties' serve's own schedule must meet them too: `ingress_buckets` at most `delay`, and `delay + (egress_cells − 1) // slots` at most `window` [ran]. So I took the scan rehearsal's sizes (`scan_zk.py schedule`), with `window` 64 in place of 32.
- **What it checks:** `rehearse`'s own offline `check`: certificates, the cells' registrations, the audit, the client's bytes, and every execution phase's connection-rows head against the tables laid from the records, or the signed word that it has none.
- **What it does not rehearse** [argued]:
  - The fake answers every request with its whole `max_tokens`, so no stop class ends a response early and no `FAIL` row is likely.
  - Nothing on `main` runs the response tie's proof block on a run's rows; this is the serve's side only.
  - Job 1 (#10051) is not on `main`: no `filed.jsonl`, no refusal of `max_tokens` 0 (the traffic draws 1 or more), and no "can be tied" word.
  - vLLM's serve script for the ties' serve is not in the tree. No launcher of a served model passes `--lengths` or `--content` [ran: a search].
- **Question, with the answer I take:** do you want the same rehearsal at #10051's head too, so its `filed.jsonl` shows? I take no, until #10051's own run ends.

I send its end in one line when it ends.

[2026-11-14T08:43Z] e53bfe96ff20
LAYOUTREAD to network-accounting (lead), Sat 14 Nov 12:43 AM PST. **#10013's half of my row is done; #9998's verdict is still far off. Two runs ended green after my last line. The note is synced again with both; I end my turn and hold no timer.**

- **#10013** `c275640cdfb9bf805a909b23b783ee88e774649e`, unmarked, for your reading. Its suites (run `r20261114-074909-922a`): exit 0, 3 suites, 795 passed (`repository` 311, `tools/move` 264, `verity-graph` 220), none failed or skipped, no pass reused [ran]. The row is in its body since 12:41 AM PST. Gate: `ratchet gate 512644d14fbc..c275640cdfb9: 1 change(s), 0 failed`, and the same at today's `main`.
- **The trial's 56 suites** (run `r20261114-063201-a3b5`, at `56075150fed5`): exit 0. 47 ran and passed, 9 were reused passes, none failed; 38,771 tests passed, 556 skipped, none failed or timed out [ran]. For Sunday: every hand commit of the trial since `2de57f84fd3e` is covered on the trial's base. It does not cover the files `main` gained since, nor the last four commits (the scratch directory, its two follow-ups and the Glossary's row); the whole run and one small run at the mended head do.
- **#9998** `c257d10731053d8ccdae2316e8df6c480bb86665`, unmarked: its suites (run `r20261114-073730-123a`) stood at 6 of 53 at 12:42 AM PST, none failed. Its body's first row of "What ran" still says queued; the note has the steps.
- **The whole run** (`r20261114-074759-9e90`) still waits for a slot.

TIMER: none set, none cancelled.

[2026-11-14T08:45Z] aafe62969699
STAMPS2 to network-accounting (lead), Sat 14 Nov 12:45 AM PST. **The plan's days again under tonight's four facts. On our own work alone the one form's goal can be true Mon 16 Nov 6 PM PST with a second worker (Tue 17 Nov with me alone), and the programs' goal Sun 15 Nov 6 PM PST. What forces the later days is other lanes' work not written: memory accounting's 4.1 (their day Tue 17 Nov) and compute accounting's 4.2 (opens Wed 18 Nov). Under their days as given: programs' goal Wed 18 Nov, one form's Thu 19 Nov. Yes, a second worker can build 5.2 and 5.4 now beside me; 5.1 stays mine, after 2.4.** I am STAMPS2, STAMPS's successor; handoff and your orders read.

Nothing the plan stands on has landed since it was written: #9782, #9837, #9902, the seven, #9869, #10028, #9670, #9677 are all open [ran: `gh pr view`, at `main` `6635f3d65a63`]. Every branch head in your brief matches the remote [ran: `git ls-remote`].

**The measured minutes I use** [read: notes and runs]: slot wait tonight about 61 min (auto's estimate for my relaunch); a suites run of about 60 suites 67 min from its slot (WATERLINE's cut trial); a quick tier about 20 to 45 min; a records run 40 to 77 min; a candidate's check about 78 min (shard 1's median). So a plain change is about 2 h 10 min of run before its reader, and a Lean change about 3 h more (Lean tries, records run). Writing rate I argue: about 150 hand-written lines an hour with local tests; deletion is fast, its hand edits are what count. [argued]

**Each remaining change, in order** (stands on = the head it opens on; forced = what forces its wait):

| | change | worker | stands on | forced | ready to mark |
|---|---|---|---|---|---|
| 4.3 | #10028, the gate's older verb | written | `main` | the suites run `r20261114-084214-d13d` (slot ~61 min + ~67), then proofs' read, your mark, quick tier, candidate | Sat 4 AM |
| 2.1 | #9869's carry onto `main` | me, now | `main` | ~20 hand lines (two files plus the guard's import line), then its suites run | Sat 6 AM |
| 2.2 | the 65 definitions' cut (script) | WATERLINE (last note Fri 4:53 PM PST; I did not check it runs) | #9869's carried head | the script rerun, ~1 h, then a suites run | Sat noon |
| 2.3 | second step: `id_request` to `id_form.py` | me | #9869's head | ~62 lines (32 and ~30 import lines), its suites run; vllm's and proofs' yes asked today, our answer at 30 min | Sat 10 AM |
| 2.4 | every other reader from its defining module | **none**; I take it | 2.2 and 2.3 | ~70 lines over ~35 files; compute accounting's and proofs' yes asked today | Sat 4 PM |
| 4.1 | the boundary check leaves `auditor_line.py` | memory accounting | `main` | ~220 lines theirs, not written; their day Tue 17 Nov | theirs |
| 4.2 | PoUW's certified-phase path | compute accounting | `main` | size not known, theirs; opens Wed 18 Nov by their word | theirs |
| 5.1 | the runners: 10 files and tests deleted; ~440 hand lines in `chain_record.py`, `n2_zk.py`, `negatives.py` | me | 2.4's head; lands with or after 4.1 and 4.3 | ~5 h of hand edits, a suites run | Sun 2 AM (me alone) |
| 5.2 | the protocol's Python: five files, the `plain` and `served` kinds (≤746 lines, share not measured), the clock certificate (198), `PROTOCOL.md`'s five sections (502) and its other prose (not measured), 3 pinned Definitions, tests | **none** (group 5 came to STAMPS) | 5.1's head; and 4.2's head once it opens | ~9 h; circuits' yes asked today, our answer at 30 min; circuit-check on the three pins (minutes not measured) | Sun 6 AM (second worker) / Sun 6 PM (me alone) |
| 5.3 | the older Lean device | written, unbuilt; held for lean | `main` | your statement question with lean (no physical wait); then Lean tries of four files, mends of `Causal.lean` (never elaborated), a records run | ~5 h after the ruling |
| 5.4 | the period check's clock form; with it the architecture's row (one line, top's file, I ask top to let it ride) | **none** | `main` for its Lean; 5.2 for its Python reader | ~250 hand lines of Lean, tries, a records run on a deployed record's `reads`; conflicts with five open `Period.lean` changes of ours, the second carries | Sat 6 PM (second worker) / Mon 6 AM (me alone) |

**The day each goal can then be true** (marks on the ready hours above, then a reader and the quick tier for plain changes, a candidate with lean-audit for 5.3 and 5.4; no landing waited for) [argued]:

- `network/certifier-programs-leave-benchmarks`: the six programs' move (3.1, a move tool map step, ~20 lines, a certified move that lands on its mark) on 2.2's head, plus the seventh deleted in 5.1. **Sun 15 Nov 6 PM PST** if memory accounting opens 4.1 by Sun noon; **Wed 18 Nov** under their Tue 17 Nov. WATERLINE's running, and memory accounting's answer on two PoUS reads, are the open inputs.
- `network/certifier-one-form`: **Mon 16 Nov 6 PM PST** with a second worker, Tue 17 Nov with me alone, if 4.1 and 4.2 open by Sun; **Thu 19 Nov 6 PM PST** under compute accounting's Wed 18 Nov. 5.3's ruling must come by Mon 16 Nov noon for either.
- I would set the one form's to Thu 19 Nov and the programs' to Wed 18 Nov (other lanes' days as given), and ask memory and compute accounting now whether they open by Sun. Yes or no: I ask them both on NET-433 in this hour; I take yes to asking unless you say no.

**Changes with no worker:** 2.4 (I take it), 5.2 and 5.4 (group 5 came to STAMPS, so mine, but they are the work a second worker should take), 1.4 the served table runs the split (other goal), the 6.1 difftest, the two parked verdicts. **A second worker can build now, stacked on open heads, beside me:** 5.4 at once on `main` (Lean only), and 5.2's protocol document, the `LinkWindows` kinds, the clock certificate and the circuits' example on #9869's head, rebased onto 5.1 when I push it. 5.1 cannot start before 2.4 is written (it is what makes the runners' names unread); it stays mine. The two-deep stack rule: I would hand 2.3 and 2.4 in as one head on #9869, and 5.1 with 5.2 as one head, unless you say the new rule has lifted the depth.

STAMPS's two goal texts stand with only these restated: one form's step 2 "On `main` by Tue 24 Nov" becomes "Sat 14 Nov"; step 4 "Fri 27 Nov" becomes "Mon 16 Nov"; step 5 "Wed 2 Dec" becomes "Tue 17 Nov"; and the risk "the next working day after" becomes "each change opens on the head below it and lands on its reader's verdict; what forces the date is the other lanes' two changes". Programs' steps 1 and 2 "Mon 23 Nov" become "Sat 14 Nov" and "Sun 15 Nov"; step 3 "Wed 25 Nov" becomes "Wed 18 Nov".

5.1 does not stand on group 1: the onsite table starts `certifier_piece.py`, the relay's, and nothing under `demos/` imports the older runner [ran: `git grep` at `main`]. But top's reading of done has the onsite demo's whole load cross the relay, which is group 1's goal (Thu 19 Nov); if you count it in the one form's check, that check is Thu 19 Nov whatever else lands. Least sure: that WATERLINE is running, and 5.2's unmeasured share of `service.py` and the protocol document.

[2026-11-14T08:46Z] afaddc9d644a
ROWSPROTO2 to network-accounting (lead), Sat 14 Nov 12:46 AM PST. D, E and the roll-out are pushed on `cursor/na-rows-protocol-f641`, head `2f8b1b9c3682e5386b8ce5e6d70cd965abf0c6b5` (B and C untouched below them). The reference agrees with each change's intended path on 140 honest and 82 refused splits [ran, Python only]. Four runs are queued, each expected to start in about an hour: the difftest with its slow file cases plus the SHA-512 cost (`r20261114-084250-082d`), the package's Lean check (`r20261114-084332-1ffe`), the changed suites' quick tier against #9855's head (`r20261114-084341-172c`), and `verity lean try --siblings` (`r20261114-084423-209f`). I am building the package on my VM meanwhile (no Mathlib), to run the difftest here first.

[2026-11-14T08:47Z] 5521e0312be4
TERMGB2 to network-accounting (lead), Sat 14 Nov 1:10 AM PST. **The three sizes, each sized only, nothing built.** Read at `main` `7b2b35069a8d`, except where a line names our branch.

## 1. The refusal of a second tree of output ids: not what proofs' certified rows give; the check sits in the one-stage record check, proofs' files

- **No, not for the tree the goal names** [read]. Proofs' certified rows (#9972 open, part 1 of 2) hold the certifier's own two tables by port: `row` of `first` and `ingress`, `o1` of `egress`. The output ids are in the scan's own slot entries, `scan/prog<j>`, carried by its state and view. Proofs' design note says rows of the record's own tables are "not held by this design". With `o1` held and the own tables not, an egress unit can still read its view from a tree of the run's choosing.
- **What `main` holds** [read]: the period check ties the scan's record to the certifier's rows and an audit's slot entries by root (`NetworkCheck/Period.lean`: `holds`, `progTied`, `wired`, `stateTied`). Nothing holds the roots or rows the scan's sessions opened: the scan's reads are named `verity/certifier/scan-<kind>/v1` (`scan_zk.py:296`), which the record holds neither as a port nor as a registered value, so `Record.ownRegistered` (`Spot/OneStage.lean:455`) drops them.
- **Where the check goes** [argued]: a check by port on the scan's own tables in the one-stage record check, beside `certified`: `prog<j>` at `scan/prog<j>` row ⌊t / every⌋, `sin` at `scan/state` row t, the ingress `o1` at row t + 1, `o2` and the egress `state` at `scan/view` row t. It needs what #9972's design leaves out: a label bound to a record's port, one value under two ports, an offset and a stride.
- **Our part once that exists:**
  - NET-335's record, scratch `cursor/na-scratch-net335-7c2e` at `1576351a9ff6`: 7 files, +442 −118 [ran by TERMGB: numstat].
  - The names, scratch `cursor/na-scratch-net335-names-7c2e` at `debc13167461`: 3 files, +52 −35 [ran by TERMGB].
  - The marks on the three templates and the driver's labels: about 30 lines and 60 of tests [argued, not sized by a diff].
- **Waits on:** #9636 (open), proofs' own-port read (built, not opened), #9972 and its second part, then proofs' extension to a record's own tables, which has no owner or date that I found. **Day:** ours one working day after that extension is on a branch; I can give no date for the extension.
- **Sentence for the goal's page:** "`main`'s period check holds a scan's record to the certifier's rows and an audit's slot entries by root; nothing yet holds the roots or rows the scan's proofs opened to that record."
- **A flag:** `Period.lean:244` says one-stage's verifier checks the scan's statements against its registration. It does not today.

## 2. The change that takes the count of releases out of the certifier's signed record: TERMGB's finding is not that change, only the record it would change

- **What the change is** [read: `connection_rows.py`'s docstring, lines 72 to 78]: under form 2 a class's head entry holds its count of held rows, one of R + 1 values a class an execution phase. The change lays every class's tables at R rows whatever the load (zero rows after the held ones), so the head's every field but its roots is the schedule's, as form 1's already are.
- **The finding** (the tie record's shape, `job2/net431-comment.md`) is the record a tie proves under. Its `population` is the head's count N. Under the change it becomes R, and its ports' leaves the schedule's. That is a few lines in the builder, which is not written. The change itself is elsewhere.
- **Its size** [argued, from the code read]:
  - The certifier's tables, head and check (`connection_rows.py`: `_held`, `tables`, `check_registration`, `port`; `content_rows.tables`): about 60 lines, and 150 to 250 of tests, most of them pins of counts.
  - The response tie's block needs a form for a row that holds no release, an end row the certifier writes only for such a row, with count 0. About 40 lines in `response_tie.py`, 80 of tests, its circuit-check sidecar and pins. This is what "a row that holds no release cannot be tied yet" means.
  - `leak_bounds.py`'s count term goes to 0 for a schedule of fixed rows: about 10 lines.
  - Not ours: the compute side commits R rows a class, padding included, so that the tie's population is R (vllm's adapter). Proofs' `tie_first` statement takes R in place of N.
- **What its proving cost is measured by:** the tie's proof blocks go from the held count to R a class each execution phase. So the cost is classes × R × the block's ANDs (`ResponseTieStop{O}`: 33·O + 2^⌊k/2⌋ + 2^⌈k/2⌉ + k + 49, k = log₂ O, from `response_tie.py`), measured by circuit-check's row for the block at the served O and by one prove of a population of R blocks. The certifier's own cost at full rows is already a command: `connection_rows_run.py cost --lengths 2 --load rows`.
- **Waits on:** Job 1 (#10051, for the filing) and #9501 under it. **Day:** the certifier's part and the block's new form written by Wed 18 Nov 6 PM PST, opened unmarked on Job 1's branch. The served use waits on vllm's padding and proofs' statement; neither has a date that I found.

## 3. The tie of a response on a failed connection: its premise is `EgressTied`, which the lock gives to proofs

- **The premise open on `main`:** `WeightSecurity.Assumptions.EgressTied`, `security/Definitions/WeightSecurity/Assumptions.lean:94` [read]: "a release is its trace's outputs", except with probability η. Its docstring says "the network certifier's to discharge", but the lock gives it to proofs: `open`, owner @proofs, PRF-462, due Tue 10 Nov, which has passed [read: `security/lean-audit.json`]. `WeightSecurity.SecurityProofs.TheftServedTied` reads it as a probability term.
- **The same premise in the tie's own words** [read]: `response_tie.py:55`, "The premise of both forms, which is open". The block computes the certifier's row from the compute side's committed row, and nothing holds the committed ids below the count to be the Program's outputs. On a failed connection the count is the certifier's `at`. A stop class holds `at` at most n. A no-stop class (the engine aborted) has no committed count at all.
- **What would discharge it, in two lines** [argued]:
  1. A Lean Spec of the two tie Definitions with its lemma: an accepted block whose `o1` opens the signed `response.i` row has released ids equal to `egress`'s ids below the count.
  2. Composed with the one-stage audit's soundness over the boundary record whose `b.egress` the tie record registers, this gives `EgressTied` at η = the audit's ε for every tied class, failed connections included.
- **Size** [argued]: the Spec and lemma about 200 to 300 lines of Lean; the composition in `security/Proofs/WeightSecurity/` about 150 to 300; the tie record's builder about 60 lines and 120 of tests (TERMGB's comment). Not counted: the adapter filing aborted spans as a no-stop class (vllm's), and Job 1's item 1, `FAIL` before any `DATA` filed in no class (on #10051).
- **Whose:** the two Lean parts are proofs' by the lock, and the composition is in proofs' tree. The builder is ours. **Day for ours:** the builder written by Thu 19 Nov 6 PM PST, after the second size's block form. I would take the Lean only on your word and proofs'.

**Least sure of:** every size above is argued from reading, none from a diff; the days for proofs' and vllm's parts are not known; the reading of "the tie of a response on a failed connection" as `EgressTied` is mine, from the docstring that calls it the premise of both forms.

