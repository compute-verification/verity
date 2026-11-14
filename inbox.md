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

