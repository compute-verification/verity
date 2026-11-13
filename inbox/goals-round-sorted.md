# Goals round, Fri 13 Nov: every lead's sort in one place

Daniel, Fri 13 Nov 8:06 AM PST: "It seems like they're going to need restating and cleaning up. Also, it seems like we've kind of lost track of what we mean by a tier one goal. I think there's like too many tier one goals here. But also, broadly, yeah, I'm still mainly interested in like reorganizing the repository around like the new architecture and kind of like implementing that new architecture rather than like getting other results out. Can we have a round of re-specifying the goals?"

This file is the round's working copy: every lead's sort of its own open goals, put in one place. It is a proposal until you say the tier-1 list is right. Then the Goals page is rewritten from it, and this file is not kept up.

## The Goals page at 8:06 AM

259 open goals. 41 sat at the top level: 15 put there by hand, 26 only because nothing waits on them. 79 were due today. 129 had had their date moved.

## Tier 1, as proposed (second form, 8:55 AM)

Daniel, 8:47 AM PST: "I don't think we need the (at most 5) rule -- I'm fine with there being like 10 or smth. And note there's also like 'when are demos' and 'when do we hit certain performance targets on particular benchmarks'".

A tier-1 goal is one you named and watch: a state of the whole system, a demo with its date, or a dated target on a named benchmark. One command or one page shows it true or false. Only your word adds one, and there is no cap. Every other goal sits under one of them, is parked with the state it stopped in, or is dropped. Structure keeps first call on people: when that moves a demo's or a target's date, its lead says so the same day.

| Goal | Holder | What shows it, or its target | Where it stands, and its date on the Goals page |
| --- | --- | --- | --- |
| 1. The repository has the architecture's layout | top | The layout test's "not yet" lines at zero, and the architecture file's rows of files that only change place | 10 lines, the last dated Tue 24 Nov |
| 2. The architecture is implemented | top holds it, one lead per row | The architecture file's rows of code still to be written at zero, and each area's count of known lint failures (the bar) | 24 rows in the file's draft (#9702, recounted against today's `main` by 11 AM) |
| 3. One dashboard shows every number | console | Every maintained row wired and fresh, with the numbers of the other goals on it | 19 numbers declared, 13 with a value. Wed 18 Nov |
| 4. A change lands within two hours | ci, with infra's platform under it | ci's daily row: 90% of plain changes within two hours of their mark, and the same for Lean | Thursday: 42% of plain changes. Plain Tue 17 Nov, Lean Thu 19 Nov |
| 5. Remote demo | vllm | Every verifier accepts in Lean on one served run from `main` | No run served from `main` yet. Fri 20 Nov, off track |
| 6. Onsite demo | vllm | The same run with the network verdict and memory's three verdicts | No onsite run since Sun 8 Nov's. Tue 24 Nov, off track |
| 7. Production vLLM's serving cost, GLM-5.3 under an agentic load | vllm | Under 1% of serving time against stock vLLM, with the replay reproducing the served tokens | +6.66% on one pair of runs, with no interval (Thu 12 Nov). Was due today 9 AM |
| 8. Network accounting on GLM-5.3 | proofs | Under 300 GB a world-year at 10% compute, over 100 bits | Never run end to end. Modeled: 2,670 GB, and 68 bits by `main`'s bound. End to end today 6 PM, off track; fully proved Mon 30 Nov |
| 9. Proof-system performance, GLM-5.3 under SPOT | benchmarks | Prover GPUs per serving GPU within proofs' share of the 10% compute budget | 0.0364, modeled. Tue 17 Nov |
| 10. Verified compute share, Llama 3.2 1B | compute-accounting | 97% on a run whose verdict accepts from `main` | 95.44% on a development run. Wed 2 Dec |
| 11. Uncertified memory share, Llama 3.2 1B on eight cards | memory-accounting | 0.14 | 0.199 (Thu 12 Nov). Wed 18 Nov |
| 12. Training slowdown the inference-only check proves, Llama 3.2 1B | training | 1,000 | 1. Wed 2 Dec |

Goals 5 to 12 are as the Goals page stated them at 8:06 AM. Each holder was asked at 8:58 AM for the benchmark's command, the target, today's figure with its run, and the day it can hold with structure first. The tables below are still each lead's sort under the first form, which had goals 1 to 4 only: the parked table's last column is my first sort of what comes back under 5 to 12, for each lead to confirm.

## What is yours

1. **Is that list right?** Are 7 to 12 the benchmarks and targets you meant, and is one missing?
2. **Security posture** (infra): it was a project in your own words and nothing in goal 4 measures it. With no cap I would keep it at tier 1, as a thirteenth. Say if not.
3. **One kept change touches a guarantee** (lean): it states two parts of the inference-only policy as named open premises of the remote theorem, in place of the one premise `InferenceOnlySound`. My recommendation is yes: it makes the list of open premises say what is open. It opens unmarked Tue 17 Nov, and you see the record's change when it lands. Say no and it parks with the policy.

Withdrawn: the question about tonight's runs. The serving-cost runs (5 PM and 9:45 PM) go ahead as booked, since goal 7 is theirs.

## What the leads changed in the frame, and I took

- A finding that a verifier accepts what it should not is never parked. It stays under goal 2, on the row "Python never returns accept or reject" (soundness).
- The audit's second round on served data, and proofs' two-stage goals, are the architecture's and not results (vllm).
- Parked does not mean closed: a pull request that is built lands as built, and nothing new starts for a parked goal (network-accounting). That stands for what stays parked after the second form.
- The bar gets a number: each area's count of known lint failures, on the dashboard (compute-research).
- A dashboard row whose goal is parked shows "parked" and leaves the wired count, or goal 3 could never finish (console).
- A row with two owners gets one goal per owner (kernels and compute-accounting on PoUW's GPU code).
- Goal 1 also counts the architecture file's rows of files that only change place, or it could read zero with GPU code still under `benchmarks/`. That count is known at 11 AM PST, when the file is recounted against today's `main` (memory-accounting).
- A defect in a figure `main` already prints is the bar's, under goal 2. What parks is the next number, never the mend of a printed one (training).
- The areas of `security/` take their protocols' names. It is a layout change that had no line; it is now under goal 1 for Fri 20 Nov, with a row added to the architecture file (lean).
- One goal marked done this morning was not: "Lean code uses the Glossary's names". Declarations' own names still spell seven older words (session 739, window 228, escape 150, pin 123, wrong 100, transcript 50, horizon 13, of 101,108 declaration lines), not yet sorted into the senses the Glossary keeps. lean gives each lead its list Sat 14 Nov 6 PM PST (lean).


**State:** 15 of 16 leads have answered; still to come: proofs. Nothing on the Goals page's top level changes before Daniel's word. Each lead's own table is on TOP-416.

**Second form:** 6 leads have re-sorted their parked rows so far (benchmarks, compute-accounting, console, kernels, memory-accounting, soundness); compute-research and circuits have nothing that changes. The counts and the four tables below are the first form's.

compute-research's six goals that were parked before today are one row of its table, so the parked count is five short.

**Counts so far:** kept 148 (goal 1: 22, goal 2: 87, goal 3: 7, goal 4: 32), parked 55, dropped 24, new goals written for lines and rows that had none: 21.

## Each lead's sort of its own goals

| Lead | Goal 1 | Goal 2 | Goal 3 | Goal 4 | Parked | Dropped | Open goals sorted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| benchmarks |  | 1 | 2 |  | 2 | 1 | 6 |
| ci | 1 |  |  | 10 | 1 | 1 | 13 |
| circuits |  | 10 |  |  |  |  | 10 |
| comms | 1 |  |  | 2 | 1 | 1 | 5 |
| compute-accounting | 1 | 5 |  |  | 10 | 1 | 17 |
| compute-research | 2 | 4 |  |  | 1 | 5 | 12 |
| console |  |  | 5 | 1 | 1 | 1 | 8 |
| infra |  | 1 |  | 11 | 2 | 2 | 16 |
| kernels |  | 7 |  |  | 6 |  | 13 |
| lean | 2 | 7 |  | 4 | 2 | 3 | 18 |
| memory-accounting | 2 | 7 |  |  | 4 | 2 | 15 |
| network-accounting | 2 | 10 |  |  | 7 | 5 | 24 |
| soundness | 1 | 12 |  |  | 5 |  | 18 |
| training | 2 | 12 |  |  | 7 |  | 21 |
| vllm | 3 | 8 |  |  | 5 | 1 | 17 |
| top | 5 | 3 |  | 4 | 1 | 1 | 14 |
| all | 22 | 87 | 7 | 32 | 55 | 24 | 227 |

## Goal 1. The repository has the architecture's layout

| Lead | Goal, as its lead restates it, and its check | Closes or serves | Day |
| --- | --- | --- | --- |
| ci | Stops with the check at `tools/verity/check`: no layout line or architecture row names `tools/check`, and nothing waits on it | the architecture file's row "the check: `tools/verity/check/` to `tools/check/`" | held under goal 1 until top's 11 AM word on whether that row stays (ruling 11) |
| comms | "Daniel's term device used everywhere". Check: on `main`, `git grep -c 'auditor host'` and `trusted-host` give 0 outside recorded runs, and the locks name `TrustedDevice`. | none (the architecture's term) | Sat 14 Nov (its Lean slot follows proofs' four) |
| compute-accounting | **PoUW's GPU code and serving entry in its package.** Check: no PoUW kernel source under `benchmarks/` or the vLLM adapter, and the adapter reaches PoUW by one call. Waits on kernels' Pearl-C split and vllm's adapter | R4 | 11/24 (was 11/19: the Pearl-C split follows chain 2) |
| compute-accounting | NEW. **PoUW's `clear/` under its prover.** Check: L9's line gone from `tests/tree.toml`. | L9 | Mon 16 Nov |
| compute-research | **`verity/device.py` reads the cards' and nodes' data sheets.** Check: `verity/hardware/cards.py` and `nodes.py` are absent from `main`, and no file imports either. | serves L2, L3 and L4: it empties `verity/hardware/` into `verity/device.py` | Sat 14 Nov (was today noon: it is built on chain 2's head, #9742 changed the same tree file at 8:10 AM PST, so the worker merges it and runs the suites again, and then seven owners read their callers' lines) |
| compute-research | **No Lean table restates a data sheet, and `verity/hardware/lean.py` is gone.** Check: L4's line is gone from `tests/tree.toml` and R17's row from the architecture file. | closes L4 and R17 | Tue 24 Nov, L4's day (was Wed 18: three changes in order; the first waits on proofs' word, due today 10 AM PST; the second is built and waits behind compute-accounting's #9509) |
| lean | **Security areas named after their protocols.** as it is. Check: each area of `security/` carries the architecture's name for its protocol, and the accounting area holds one protocol's files | no line: none of L1 to L10 is under `security/` | Fri 20 Nov (was Wed 18: it is a move every open Lean branch crosses, so it follows the owners file and the tool train). If parked: nothing is renamed and no list is written |
| lean | as it is. Check: no declaration name on `main` spells an older word outside the senses the Glossary keeps. Today 228 window, 739 session, 100 wrong, 150 escape, 123 pin, 50 transcript, 13 horizon, not yet sorted | no line: the Glossary's terms, as comms' device term | not known. Sat 14 Nov 6 PM I write lean's own day and each lead's list; the whole is the latest of the leads' days |
| memory-accounting | Title kept. **Check:** #9061 on `main`: its 36 files sit under `verity/protocols/accounting/poub/` and `pous/`, and `BENCHMARKS_IMPORTERS` loses `adapters/vllm/verity_vllm/protocols/pous.py` and `difftest_vectors.py` | section 7's place row "GPU code", memory's part (no L line: see the frame) | **Sun 15 Nov** (was today: it crosses chain 2 today and is marked by 6 PM PST; the rest is the queue's) |
| memory-accounting | Title kept. **Check:** `poub/prover/` exists on `main`, and `session_parties.py` and `wipe_parties.py` define no `Prover` or `Challenger` | the architecture's idea 8, "trust follows the directory": a `verifier/` holds what the auditor runs (no L line) | **Tue 17 Nov** (was Sat 14: its worker starts on #9061's crossed head, which exists tonight) |
| network-accounting | **Network certifier published under one name.** kept. Check: in the exported tree the certifier's tests pass and no public file holds the retired name | the export, serves R20 | Fri 20 Nov |
| network-accounting | "The inference-only policy's directory holds no calculator." Check: `verity/protocols/policy/inference_only/` holds its verifier only. True on `main` since chain 2 [read by me]; #9589's two hand changes are being read against `main` today | the layout: a protocol's directory holds the protocol only | Mon 16 Nov |
| soundness | *`verity/claims/` is generated from the locks*: L6's not-yet line deleted; the Lean check fails on a named assumption without an id | L6 (closes) | Wed 18 Nov |
| training | **Each of training's parts exists once, where the architecture puts it.** Check: on `main` the bound is one Lean area, the calculator is `benchmarks/slowdown/`, the one translator is the PyTorch adapter's, and a test fails on a second copy of each | serves goal 1; no L line is training's | Mon 23 Nov |
| training | **Slowdown replaces rho in every name.** as titled. Check: `benchmarks/slowdown/` holds the calculator, `benchmarks/rho/` is gone, and the rename maps' check finds no `rho` name for the slowdown | the layout's names; no L line | Tue 17 Nov (was Mon 16: it now carries the calculator's path move and needs Monday's Lean slot) |
| vllm | **Leftover path fixes merged after the move.** as titled. Check: the ten stale places left of fourteen are gone from `main`. One more is marked to merge today; the rest ride the refactor line's next snapshot | serves the layout | Mon 16 Nov 6 PM PST |
| vllm | "vLLM's open changes carried over both moves". Check: every open PR of lane vllm holds `4ae17f7eafd7` or is closed. Today 4 of 29 do [ran 8:27 AM PST] | serves the layout: no branch of ours still names a path from before the two chains | Sat 14 Nov 6 PM PST |
| vllm | **Import rules checked on the new layout.** as titled. Check: `tests/test_boundaries.py` reads the generated `tests/import_order.toml` and no table kept by hand. The listing is on `main` since chain 2; two small PRs of it are being carried. I have not re-read whether a hand table is left | serves the layout | today 6 PM PST: done, or what is left |
| top | **Unused code deleted.** as it is. Check: no file on `main` that nothing imports, runs or cites, by the tree test's list | serves L8 | Fri 20 Nov |
| top | **Public export published.** as it is | serves R20 | Fri 20 Nov |
| top | **No open pull request is refused for a path the moves changed.** Check: `research queue status` lists none | serves every L line | Mon 16 Nov |
| top | **Public export rehearsed on the moved tree.** as it is. Check: the export of `main` passes its own check on a node | none | Sat 14 Nov 6 PM |
| top | NEW. **`reporting.py` and `periods/` are folded into `verity/protocol.py`.** Check: both lines gone from `tests/tree.toml`. | L1, L7 | Sun 15 Nov |
| top | NEW. **`verity/hardware/registration.py` and `record.py` are folded into the hardware file.** Check: both lines gone from `tests/tree.toml`. | L2, L3 | Sun 15 Nov |
| top | NEW. **`verity/definitions/` is split: its protocol part in `verity/circuits/`, its served part in the vLLM adapter.** Check: L5's line gone. | L5 | Sun 15 Nov |
| top | **The repository has the architecture's layout.** Check: `tests/tree.toml` holds no `[[not_yet]]` line and `tests/test_tree.py`'s `NOT_YET_MAX` is 0 | L1 to L10 | Tue 24 Nov (L4's day, the last) |

## Goal 2. The architecture is implemented

| Lead | Goal, as its lead restates it, and its check | Closes or serves | Day |
| --- | --- | --- | --- |
| benchmarks | **Benchmarks' code meets the bar.** Check: the bar read of benchmarks' six directories (harness, overhead, prove, exfiltration, recursion_e2e, recursion_figures) run again at `main` finds zero findings, with one harness kit as every benchmark's writer. Of last night's 47 findings, five pull requests are on `main` or marked. | each area's code at the bar | **Wed 18 Nov** for all we fix alone; **Sat 21 Nov** for the retired names, which wait on @proofs' renames and two of @vllm's identifiers |
| benchmarks | NEW. **Every benchmark module reaches the adapter through its public interface only.** Check: `tests/test_vllm_boundary.py`'s exceptions list holds no `benchmarks/` module. | R5 (benchmarks' share) | Wed 18 Nov, if vllm names the interface today |
| circuits | **Circuits' code meets the bar.** Check: circuits' entries in the lists of known lint failures at zero (ruling 9), and the test that stops superseded definition versions growing is on `main` with no version in its list that nothing reaches. The lint count is not known: I read it once compute-research names the lists, and say then whether zero holds for this day | the bar | Fri 20 Nov (kept) |
| circuits | **A GLM-5.3 request's whole computation builds on `main` as pieces wired to each other.** Check: one command builds the request of 43 prompt tokens and 2 outputs for all eight GPUs with no input left unwired, and both structure checks pass | serves R13, circuits' half | Fri 20 Nov (was Thu 19, for the same reason) |
| circuits | **Every kind of piece in a GLM-5.3 request has a statement the Lean verifier accepts.** Check: one command on `main` reads the cover record from the store, checks it, and prints the count of kinds without an accepted statement: 0. Today it prints 77 of 254, and the command is on a branch | closes R15 | The command on `main`: Tue 17 Nov. The count at zero: not known. At noon today I say what each family left needs, and a day |
| circuits | **Served programs name the compact versions of the long attention row and the norm.** Check: the size test on `main` lists 7 served units past 4 MiB (33 of 105 on Thu). Which of R15's 77 kinds the compact form brings within one Lean call is in my noon line. The timings beside today's form park | serves R15, in part | Tue 17 Nov (kept) |
| circuits | **The request builder on `main` builds one GPU's share of a GLM-5.3 request.** Check: its command builds the share at all 78 layers and both structure checks pass. Today on a branch, with 6,978 of 6,979 rows | serves R13 | Tue 17 Nov (was Mon 16: on `main`, not only built) |
| circuits | Title kept. **Check:** a test in the checker's suite on `main` fails if the checker's package imports the adapter or holds a circuit's name. Today its code names them, and its list of heavy circuits holds 45 names. The two changes in review take the code's. A third hands over the list and adds the test | closes R12 | Tue 17 Nov (was Sat 14: the list and the test are new in the check). The first two changes are on `main` Sat 14 Nov |
| circuits | **Every kernel the production settings serve has a circuit on `main` held to its kernel on an H200.** Check: the five circuits are on `main`, each with a kept comparison record in the store in which no word differs. Today three of five | closes R14 | Wed 18 Nov (was Mon 16, which was "in review") |
| circuits | as titled: the cache's write and two reads on `main`, each agreeing with its kernel at the code that merged | serves R14 | today 6 PM |
| circuits | **Checker's hand-over changes in review.** as titled: today's step | serves R12 | today 6 PM, if this morning's carry of both changes over chain 2 is clean, which I read by 9:30 AM. If not, Sat noon |
| circuits | **Long-row serving change waiting to merge.** as titled: today's step of the goal above | serves R15 | today 6 PM, on the same carry |
| compute-accounting | **PoUW states its parties.** Check: `verity/protocols/accounting/pouw/verifier/parties.py` on `main` (#9704, carried across chain 2 now) | R1 | 11/16 |
| compute-accounting | **PoUW's code meets the code rules.** Check: the code-quality bar's audit of `verity/protocols/accounting/pouw/` finds nothing | each area's code at the bar | 11/17 |
| compute-accounting | **PoUW's Lean reads its device figures from a data sheet parameter.** Check: no Lean table under `security/` restates the H100, H200 or RTX PRO 6000 sheets (R17's row deleted) | R17, serves L4 and R10 | 11/18 (was 11/16) |
| compute-accounting | **PoUW's accept comes from its Lean check.** Check: no function under PoUW's Python verifier returns accept or reject; its difftest holds the Python twin to the Lean | R7 | 11/20 |
| compute-accounting | **PoUW's architecture rows closed.** Check: section 7 of the architecture file names no row for PoUW, and `tests/tree.toml` has no PoUW line | R1, R4, R7, R8, R17 (PoUW's parts), L9 | 11/24 |
| compute-accounting | NEW. **PoUW's answer names the registered hardware it accounts for.** Check: `PouwCheck.Period`'s answer carries the registration's head and the device's place, and its parameters name the window's sessions. | R8 (PoUW's part) | Sat 14 Nov |
| compute-research | **Compute, memory and network accounting's code on `main` passes the checks the vLLM adapter passes.** Check: its three parts closed (ours above, memory-accounting's `code-consolidated`, network's `slice-clean`). Its clause on four demo files goes to R6. | the bar, for the three accounting areas | Fri 27 Nov, network's day |
| compute-research | **The retired PoUW lead's Lean is gone from `main`.** Check: the 78 paths of the change's list are absent, the lock names none of its 10 guarantees, and the Lean check passes. | the bar | Mon 16 Nov (was today 6 PM: it is a Lean change, today's Lean slots go to the moves, and its carry across chain 2 is running now) |
| compute-research | **No compute file, function or Lean name says `cap` or `debit` for Pearl's inert work.** Check: `tools/move/inert_work_names_map.toml`, applied to `main`, changes no file. | the bar: the Glossary's term in names | Sun 15 Nov (was today 8 PM: ci's order of 8:06 AM PST puts it fifth, behind proofs' four and comms' #9737; the cut is rehearsed on today's `main` and its records run is in flight) |
| compute-research | **Compute accounting's code on `main` is at the bar.** Check: the lists of known lint failures hold no path of PoUW's verifier and prover, the compute benchmarks or the machine model, and the swarm's branch is merged. | the bar, for compute | Thu 19 Nov (was Wed 18: the branch crosses three moves between today and Sunday: chain 2, #9737 and our own rename) |
| compute-research | NEW. **A registered device's figures are written in its data sheet and nowhere else.** Check: R10's row deleted. | R10 | Tue 24 Nov |
| infra | **Infra's directories hold no script that nothing runs, and every name other code uses is in a README.** Check: the count of names finds none missing, true since Wed, and each of the 12 hand-run scripts is deleted or named in a README | the bar; no row | 11/16 (was today) |
| kernels | **No served run loads a kernel from `benchmarks/`.** Check: PoUW's Pearl-C kernels are under `verity/protocols/accounting/pouw/prover/kernels/`, and every spec's `kernel =` and the adapter's loader name that directory | closes R4's kernel half | Fri 20 Nov |
| kernels | **Timed prover rows run as a spec of the shared harness.** Check: the spec on `main`, its numbers declared for the dashboard, the hand launchers in the notes deleted | serves R18 | Fri 20 Nov |
| kernels | **Unused kernel benchmarks deleted.** as it is. Check: the directories HASH-131's table marks "delete" are gone | the bar | Fri 20 Nov |
| kernels | **Kernel code cleaned up for review.** as it is: the parent of the five above. Check: HASH-92's children closed | the bar | Fri 20 Nov |
| kernels | **The exact int8 check runs the adapter's kernels, and `exactlin.py` holds no copy of them.** Check: #8671 on `main` | serves R5 | Mon 16 Nov (was Sat 14: it crosses chain 2 today and waits on vllm's read) |
| kernels | **Each move that rewrites a kernel's sources has that kernel's SHA-256 before and after, equal.** Check: one comment a move on HASH-132. Chain 2's is there | serves R4 and L5 | Thu 19 Nov (was Tue 17: its last move is R4's, which waits for compute-accounting's Tuesday run) |
| kernels | The plan with its days is a comment on TOP-415. It closes tonight and is not a state of `main` | serves R4 | today 6 PM PST |
| lean | **Page lists which verifier decides each verdict.** as it is; each lead gets its rows Sat 14 Nov 6 PM | serves R7: its table is the list of verdicts Python still decides, which R7's goals close | Mon 16 Nov 10 AM |
| lean | **Public export lists statements without proofs.** as it is. Check: #8701 marked, at a head where the export's own Lean check has run with a build | serves R20 | Sat 14 Nov 6 PM |
| lean | **No file of `security/` falls to lean for want of a claim.** Check: the Lean check's owners at `main` leave 0 such files; 826 of 4,816 by your sheet | R21, closed with `soundness/security-areas-owned`: soundness names the leads, lean writes the file | Tue 17 Nov |
| lean | **Each guarantee's witness is checked against that guarantee.** as it is. Check: #8945 on `main`: a guarantee that lists another case's lemma fails the Lean check | serves R17 and R19 | Tue 17 Nov if proofs restates the one lemma the rule refuses (PRF-549, its answer due today noon). If that restatement parks, I say Mon 16 Nov noon how the rule lands without it |
| lean | **Lean check passes on public repository.** as it is. Check: on a fresh export of `main` the Lean check passes on 16 of 16 packages, against the locks the export wrote | R20, closes it | Wed 18 Nov (was Tue 17: #8701 changes the Lean check's own files, so it lands alone in ci's Lean order; then about 4 h of runs) |
| lean | **The compiled check reads a two-stage audit and refuses it.** Check: #9738 on `main`, with its Lean proof that the check accepts none | R7, in the row's own words | in review Sat 14 Nov 10 PM; on `main` behind proofs' #9574, which it contains, so that day is proofs' |
| lean | **The Lean check refuses a guarantee about acceptance that shows no accepted run.** Check: on `main`, the two lists held to their counts (done) and four refusals, each with a test: "accepts, or a collision" is asked for its accepted run; one audit in the event and the premises; a collision that names no inputs; a colliding pair picked by choice | R7 by your rule 1: theorems about what a verifier accepts compiled while it accepted nothing. Serves R19 | on `main` Fri 20 Nov, in review Wed 18 Nov; I say both again Mon 16 Nov 6 PM with the sizing |
| memory-accounting | **Memory accounting's code meets the bar.** Check: the swarm's score on `main` reads zero list entries and zero findings for the nine seats outside Lean (343 and 548 on Thu 6 PM), and `BENCHMARKS_IMPORTERS` names no file of ours (five today) | each area's code at the bar | **Thu 19 Nov** (was Tue 17: 251 of its files are held by open pull requests until they cross and land) |
| memory-accounting | Title kept. **Check:** onsite PoUB is a `ProtocolDef` on `main` and `p2_party.py` (821 lines, its own messages) is gone | closes R1's memory part | **Thu 19 Nov** (was Tue 17: it is written against the form the split above leaves) |
| memory-accounting | Title kept. **Check:** #9240 on `main`: the challenger signs one message a wipe, naming the certifier's start mark and no egress count | serves R9, and R8 through @soundness's wipe-count tie | **Tue 17 Nov** (kept) |
| memory-accounting | Title kept. **Check:** #9377 on `main`. It lands as written; nothing more is built for it | serves R6 | **Tue 17 Nov** (was Mon 16: it crosses chain 2 after today's card run) |
| memory-accounting | Title kept. **Check:** no file under `pous/` or `poub/` imports `verity.runtime` (14 do today [ran]), what a party, a channel and a protocol declare is in `verity/party.py`, `channel.py` and `protocol.py`, and the registry gives those files and `verity/runtime/` to @proofs. The message directory itself went with chain 2 | serves R1 and R6 | **Tue 17 Nov** (was Sun 15: no worker is on the split yet; today's are all crossing the chains) |
| memory-accounting | **Memory accounting's Lean meets the bar.** Check: the swarm's `lean` seat reads zero (337 files, 101 list entries today); `security`'s lock lists every premise our guarantees take (15 unlisted today, #9583); the two storage schemes nothing serves are deleted with their guarantees | each area's code at the bar; serves R19 | **Tue 24 Nov** (kept) |
| memory-accounting | Title kept. **Check:** the Lean check of each of the three memory audits verifies the memory challenger's signature under a pinned key before it reads the session (the wipe and PoUS since Wed 11 Nov; onsite PoUB with #9723), and the pinned keys are data in the deployment's configuration | serves R9 | **Wed 18 Nov** (was Tue 17: the keys-as-data change has no worker yet) |
| memory-accounting | NEW. **Memory audits name their registered hardware.** Check: PoUS's and PoUB's parameters and signed bytes carry the registration's head and the device's place and server, and their capacity and read rate are read from the registration. | R8 | its day by 6 PM PST today (sizing at 3 PM) |
| memory-accounting | NEW. **A memory-read session is proved in zero knowledge.** Check: R22's row deleted from the architecture file. Not sized. | R22 | not known; sized by Tue 17 Nov 6 PM PST |
| memory-accounting | NEW. **PoUS's setup is a proof about the store.** Check: R23's row deleted from the architecture file. Not sized. | R23 | not known; sized by Tue 17 Nov 6 PM PST |
| memory-accounting | NEW. **PoUS's session is checked inside a proof.** Check: R24's row deleted from the architecture file. Half in flight (#9629, #9630). Not sized. | R24 | not known; sized by Tue 17 Nov 6 PM PST |
| network-accounting | "Network certification takes the registered site and says which links it accounts for." Check: one Lean check reads each link's capacity and key from the site's registration | R8 | Fri 20 Nov |
| network-accounting | "Network accounting's code is at the bar." Check: no open issue with its label; one builder of a window's circuit | our area's code at the bar | Fri 27 Nov |
| network-accounting | NEW. **The certifier's device is a party under the runtime.** Check: the device does no network I/O of its own outside the runtime's interface. | R1 (the certifier's device) | Fri 27 Nov; needs the runtime's interface for a device that holds a link |
| network-accounting | NEW. **The proxy form of the network certifier is deleted, and the remote demo starts the device on the link.** Check: R16's row gone from the architecture file. | R16 | Fri 27 Nov; needs vllm's day for the remote demo |
| network-accounting | "The network certifier and the inference-only policy each have a manifest of parties, and both deployment tables read them." Check: #9667 on `main` | R1 | Mon 16 Nov |
| network-accounting | **Plan for one network certifier outside benchmarks.** kept. Check: a day for each change on its issue, with vllm's day for the remote demo | R16, R5 | Mon 16 Nov |
| network-accounting | **Network certifier signs each session's start.** kept. Check: #9369 on `main` | R9 | Mon 16 Nov (was today: it is reviewed and must cross the moves) |
| network-accounting | NEW. **The network checks refuse each case found accepted.** Check: each of the cases found has a test on `main` that hands the check the case and sees it refused (two have a pull request open, one a mend going into #9501, one no owner yet). | R7 | Thu 19 Nov |
| network-accounting | **Network check accepts only verified audit answers.** kept. Check: a test hands the period check an answer the Lean verifier did not verify, and it is refused | R7 | Tue 1 Dec |
| network-accounting | "The onsite demo's network code is on `main`." Check: the worker's recount finds no file of the onsite line's network code that `main` lacks, and the rows audit is the Lean executable | R6, R7 | Tue 24 Nov (was Fri 20: 6 changes open of 14, and two files the run of record needs are in none yet) |
| network-accounting | **Certifier counts what left during a wipe.** kept. Check: #9154 and #9216 on `main`, and the wipe's check calls the Lean count | R9 | Wed 18 Nov (was Tue: the carry) |
| network-accounting | "No program that runs the network certifier lives under `benchmarks/`." Check: the seven files sit with the certifier, and the boundary test's list of importers names none | R5 | Wed 18 Nov (was Tue: the queue's backlog after the moves) |
| network-accounting | "The inference-only policy's check is one Lean protocol over circuit rows, and the remote demo's configuration lists it." Check: the configuration names it; its difftest passes | R7, R6 | Wed 25 Nov (was Mon 16: the protocol itself is not started) |
| soundness | NEW. **Every collision finder outputs a pair computed from the play.** Check: lean's refusal of a chosen pair, green on `main`. | R7 (the verifier's correctness) | Fri 20 Nov |
| soundness | *SPOT exports one knowledge-soundness theorem the composition reads*: listed, and the `Sampled/` leg reads only it (layers X2) | R7 | Fri 20 Nov (was Tue 17) |
| soundness | *SPOT's challenger is a party with its ledger*: #9566 on `main` | R9 | Mon 16 Nov |
| soundness | *A composite is its list of protocols*: #9268 on `main` | R6 | Mon 16 Nov |
| soundness | *The inference-only clause is stated on the sent case*: #9463 on `main` | R7 | Mon 16 Nov |
| soundness | *Every ledger line the theorem reads is a pinned device's*: R9's row deleted | R9 (closes) | Mon 23 Nov |
| soundness | *The onsite composite's theorem reads its list*: its record in the lock | R6 | Thu 19 Nov |
| soundness | *Onsite theorem's deployed instance proved*: #9410 on `main` | R6 | Thu 19 Nov |
| soundness | *Every area of `security/` has a lead*: the owners table names every file; lean's count of unclaimed files is 0 | R21 (closes) | Tue 17 Nov |
| soundness | *The compiled verifier refuses an unpinned device key or session*: #9649 on `main`, `accepts_iff` listed | R9 | Tue 17 Nov (needs #9687 B) |
| soundness | *The compiled check accepts a several-member window*: #9520 on `main`, `MULTI_REFUSED` lifted | R7 | Tue 17 Nov (was Mon 16; behind #9475 and the carry) |
| soundness | *The wipe's egress count is the certifier's, in the composition area*: #9524 on `main` | R8 | Tue 17 Nov (was today: two reads due noon) |
| soundness | NEW. **The composition reads each accounting answer through `NamesRegistered` at a prefix head of the accepted registration.** Check: #9701's statements on `main`, read by the composite theorem. | R8 (soundness's half) | Wed 18 Nov |
| soundness | *The remote demo's verdict is the compiled Lean check's*: no Python composer on `main` | R7 | Wed 18 Nov (was Mon 16) |
| training | **A job's counted step is its traced circuit, gate for gate.** Check: the test holding every library route's counted step to its traced step passes on `main` (#9374). The published table's rebuild parks with the plots | the bar; serves R2 | Mon 16 Nov |
| training | **One slowdown calculator for every protocol and demo.** as titled. Check: the grade, the library, both benchmarks and the demos' figures call one entry function, and a test fails on a second | the bar | Mon 16 Nov |
| training | **Slowdown's open premises sorted.** as titled. Check: each of the 26 premises has one line on @lean's list, and the policy's 20 are @network-accounting's | serves R21, R19 | Mon 16 Nov noon |
| training | **Slowdown computed from a submitted circuit.** as titled. Check: the calculator prints a slowdown from a circuit file alone, with the largest step it finishes stated | serves R2: the calculator reads a circuit, so the benchmark needs no translator | Mon 16 Nov, as it is. What stopped the 1,024-token step was located this morning: an index cache the reading itself keeps, a few lines; the fix is not run yet |
| training | **One translator from PyTorch to circuits.** as titled. Check: `benchmarks/training_slowdown` holds no translator and names only the PyTorch adapter's public interface (one module past it today, run); each library entry's figures equal before and after | closes R2's translator half | Mon 23 Nov: three working days after @vllm's split, not before Wed 18 Nov |
| training | NEW. **Training's benchmarks all show on the dashboard.** Check: each of the four has a `metrics.toml` row and the tab shows its value within its schedule (one of four today). | R18 | Thu 19 Nov |
| training | **Slowdown count skips repeated positions.** as titled. Check: a job's record holds its measured repeats and the count leaves them out (`tests/test_repeats.py` on `main`). Counter and five tests are pushed | the bar: the one open defect that can make a printed slowdown too high | Tue 17 Nov |
| training | **The inference-only check reads a served record** (its title says "demo done"). Check: on `main` the compiled check accepts the served Llama 3.2 1B record and refuses it with one weight unregistered | serves R7 | Tue 17 Nov for the accept and the count rule; the rate rule's refusal needs @vllm's run at the new limit |
| training | **Record check's copied definitions removed.** as titled. Check: the check's Lean has no definition of a return's fields and no string reader of its own | the bar: one definition in Lean | Wed 18 Nov on `main` (in review Mon 16) |
| training | **One command runs an attack config.** Check: `verity bench training-slowdown CONFIG` on `main` runs a config's missing jobs, grades them and prints the slowdown on a CPU node. Parked: its first GPU run, a sweep as one config | the bar | in review Mon 16 Nov (corrected 8:43 AM: on `main` the command does not grade or print a slowdown yet) |
| training | **The calculator prints no bound a draft statement sets.** Check: #8780 on `main`; onsite then prints the bound of record | the bar | marked today 6 PM PST; on `main` Sat 14 Nov |
| training | **Inference-only check accepts a served day.** as titled | R7 | none I can hold: it waits on @lean's joining statement, due Mon 30 Nov |
| training | not a state of `main`: closes when #9374 is marked | a step of the row above | today 6 PM PST |
| vllm | **Daily program committed before serving.** as titled | R6 | Fri 20 Nov, as dated |
| vllm | restate to R6's check once you say who holds the runtime: "Remote demo starts from a configuration", shown by no shell script under `benchmarks/` starting it. As a served run on which every verifier accepts in Lean, its date is the other leads' parts below | R6 | Fri 20 Nov, as dated, off track |
| vllm | "Adapter's cleaned-up line is on `main`". Check: `git rev-list --count origin/main..origin/cursor/vllm-refactor-c1f1` is 0, with batches 109 and 110 in it. The carry over chain 2 is clean and in its gate run now | R5, and it carries our halves of R3 and R4 | Mon 16 Nov 6 PM PST |
| vllm | **Weights registered before serving starts.** as titled. Check: a serve from `main` writes the record that registers every weight before its first request. Its PR is being carried over chain 2 today; I mark it when it comes | R6; the registration row of the layout's table | Sat 14 Nov 6 PM PST |
| vllm | **Remote demo rehearsed.** as titled, as built: nothing new starts for it | R6 | Sat 14 Nov 6 PM PST, as dated |
| vllm | NEW. **The set of files that becomes the PyTorch adapter is stated.** The move follows with training's. | R2 (vllm's half) | Tue 17 Nov 6 PM; the move not before Wed 18 Nov |
| vllm | **Onsite demo done.** as titled; today's stand-in on the third H200 runs as built | R6, and it waits on R9 and R16 | Tue 24 Nov, as dated, off track |
| vllm | NEW. **Benchmarks import the adapter's public modules only.** Check: the list `KNOWN` in `tests/test_vllm_public_api.py` is empty (114 modules today; 181 files outside the adapter import it). | R5 | Wed 25 Nov, a first day, confirmed Mon 16 Nov noon |
| vllm | NEW. **The workload's messages to the prover are one stated list.** First step: the fifteen files with their writers and readers. | R3 | first step Tue 17 Nov; the row's day then |
| vllm | NEW. **The compiler wires a period's pieces to each other.** Check: the second round on the demo's model reads the compiler's own pieces and no stand-in class. | R13 (with proofs) | named Mon 16 Nov; not before proofs' two-stage form is in the Lean verifier |
| vllm | **Remote demo code merged into main.** as titled. The merged tree landed Thu 7:01 PM PST. I have not re-read the count of files it left open (433 at the start) | R6 | today 6 PM PST: done, or the count |
| vllm | **Audit's second round runs on served data.** as titled. Today's noon run on a CPU node is held to the verifier's refusal, which is R7's state; with it this goal is done, and R13's own goal below takes over | R7 and R13 | today noon PST |
| top | **Each area's code meets the bar.** Check: every lead's bar goal under it done | the bar | Fri 27 Nov |
| top | **Open architecture choices decided.** as it is: the open choices decided. Check: the Notion cards' answers logged | serves every R row | Wed 18 Nov |
| top | **The architecture is implemented.** Check: the architecture file's table "Code that has to be written" has no row; each row's goal is its lead's | R1 to R24 | the latest day in the leads' tables, written when the last is in |

## Goal 3. One dashboard shows every number

| Lead | Goal, as its lead restates it, and its check | Closes or serves | Day |
| --- | --- | --- | --- |
| benchmarks | **Every number on the dashboard says measured, modeled or partial.** Check: `kit.measurement()` refuses a number without its mark (the kit's test), and every row of every `metrics.toml` reads a mark. The change has a land verdict and waits on @network-accounting's #9533 mend after its carry across the chains. | R18 | **Sat 14 Nov** (was Fri 13: the queue's pause and #9533's carry) |
| benchmarks | **Benchmarks' six directories each declare their numbers beside their scripts, and the dashboard lists them as wired.** Check: each of the six has a `metrics.toml` (or is listed by name as recording nothing), the registry test passes, the dashboard's wired count includes them. Proof system performance's and exfiltration's are in; console's split (#9707) moves them into their directories. | R18 | **Tue 17 Nov** |
| console | **Every metric row lives in a `metrics.toml` beside the script that makes it.** Check: `benchmarks/metrics.toml` is gone from `main` and `tests/test_metrics_registry.py` passes. It is verity #9707, refused at 8:09 AM PST for two moves it lacks; benchmarks' worker is crossing it. | R18 | 11/16 |
| console | **`benchmarks/README.md`'s Metrics section states the whole contract.** Check: the section names the record path, the `mark` field, `every` and the up-to-an-hour index lag, who runs a row, and a bound's keys; `tests/test_metrics_registry.py` refuses a row whose owner does not own its file. It won't be on `main` at 6 PM: it edits the section #9707 rewrites, so it follows #9707 through the queue. | R18 | 11/16 |
| console | **Each tier-1 goal shows its number on the dashboard.** Check: the tab's top-goals table shows a fresh value for each: `tests/tree.toml`'s not-yet lines (`benchmarks/tree/not_yet.py`, on `main`), the architecture file's rows still to write (a new row beside it that console writes), ci's landing row (on `main`), and goal 3's wired count. | goals 1, 2, 4 | 11/17 |
| console | **The Metrics tab shows each deployed guarantee's trust gap and each lead's open premises.** Check: at one sha, the tab's counts equal `verity lean deployed --json`'s, and a renamed guarantee keeps its history (joined by its statement's hash). | R19 | 11/18 |
| console | **One dashboard shows every number.** Check: the Metrics tab at `main`'s head shows each tier-1 goal's number, each maintained row's value (wired N of N, none stale), and each deployed guarantee's gap and each lead's open premises. | R18, R19 | 11/18 for the tab; "N of N wired" rests on each row's owner (below) |

## Goal 4. A change lands within two hours

| Lead | Goal, as its lead restates it, and its check | Closes or serves | Day |
| --- | --- | --- | --- |
| ci | **Every check runs in five shards, the slowest within 1.5 times the median shard.** Check: the shard runs' wall times of the day's builds | goal 4 | 11/14 (five shards since 7:34 AM today; balance needs a day's runs) |
| ci | **No change lands without a land verdict at its head, outside drive.toml's low-risk rule.** Check: the day's landings in `research queue stats`, 0 unreviewed | goal 4 | 11/16 |
| ci | **A candidate's check passes within 45 minutes of its build.** Check: the day's median build-to-pass time in the loop's decision log; today's first two on five shards took 25 and 20 min | goal 4 | 11/16 |
| ci | **A check part that outruns its median gets a spare twin.** Check: ci's twin-delay change (#9572) on `main` with its test, and twins in the day's runs | goal 4 | 11/16 |
| ci | **No code of the old next/checked queue on `main`.** Check: the old path's modules and functions absent | goal 4 | 11/17 |
| ci | **Every long-running tool runs `main`'s code** (the loop, the readers, `research`). Check: no 'N commits behind' line from a pinned tool past one landing | goal 4 | 11/17 |
| ci | **The queue refuses a land verdict its reader didn't sign.** Check: `signed_verdicts = "refuse"` in drive.toml on `main`, with its test | goal 4 | 11/17 (shadow PR, 24 h in shadow, then refuse) |
| ci | **90% of plain changes reach `main` within two hours of their mark.** Check: the daily landing row on CI-485 (`research queue stats`); Thursday 42% | goal 4 | 11/17, if the moves' crossings settle by Sat |
| ci | **No Lean candidate's audit starts cold.** Check: each Lean candidate's `lean-audit` step reused (minutes, not chain 2's 49) | goal 4 | 11/20: needs lean's tool train, landing as one PR from Sat |
| ci | **90% of Lean changes reach `main` within two hours of their mark.** Check: the Lean part of the same daily row | goal 4 | 11/24 (off track: Lean is held today for the move order) |
| comms | "Leads' asks close without Daniel relaying". Check: `research msg traffic --hours 24` lists no ask unanswered past 2 hours by its record. | none | Fri 13 Nov |
| comms | "No agent holds Daniel's tokens". Check: infra's listing of the agents' secrets has no Slack bot token, no app configuration token and no notes token of Daniel's, and an agent still posts. | none | Fri 13 Nov (Daniel's and infra's removal today) |
| console | **Every docs-site route checks who calls it.** Title unchanged. Check: `lib/routes/callers.test.ts` passes on infra `main`. Infra #37 is marked and still applies (infra `main` added no route since its base); it has had no review verdict since Wed. | none | 11/14 |
| infra | **Old run files deleted hourly everywhere.** Title unchanged. Check: on every CPU machine the hourly pass's own report leaves no expired run undeleted a day later | none | 11/17 (was 11/16: the rule for a run that went on writing after its copy was stored is due in review tonight, not yet on `main`) |
| infra | **Queue service dry-runs on its new machine.** Title unchanged. Check: the service's first passes are read on the new machine, launching nothing | none | 11/17 (was today: it needs an hour from ci on the machine that heads the queue's checks, and today that machine and ci are the chain's) |
| infra | **Each night an uncached check of `main` agrees with the cached results.** Check: on two nights running it starts by itself, and its posted comparison names no part that passed on both sides | none | 11/17 (was today: last night's run was held off its machine for chain 2, and that machine is too small for the Lean replay, so the check moves to a larger one first) |
| infra | **Messages reach every running agent.** Title unchanged. Check: the self-test passes on every lead's machine: a message is taken within a minute and its sender is told | none | 11/17 for infra's part; the late timers and the events that reach a worker are Cursor's, with Daniel as ASK-301 |
| infra | **Nine runs in ten start within ten minutes, quick runs within five.** Check: `research run-times --hours 24` over a working day. The day to 8:25 AM PST today: 77% of 6,764 runs within ten minutes; quick suites 70% of 1,123, median 1.3 minutes, 90th percentile 32.3 | none | 11/18 (was today: last night's load for the chain made it worse than Thursday's 89%, and the last cure, a waiting run that moves to a machine with room, crosses chain 2 today) |
| infra | **Check launches leave the control pod.** Title unchanged. Check: for a day every launch of the queue comes from the service on a CPU machine and none from the control pod | none | 11/19 (was 11/17) |
| infra | **The four check machines with 8 million file slots have four times as many.** Check: `df -i` on each, and each is back in the queue's machine list | none | 11/19 (was 11/17: one machine a day from Sat, none on Sunday) |
| infra | **Platform reliable and secure.** Title unchanged. Check: the kept goals below are closed | goal 4's platform half | 11/20 for infra's part |
| infra | **Security posture finalized.** Title unchanged. Check: each of the review's ten high findings in infra's hands has its check passing, and a fresh agent holds no token of Daniel's | none | 11/20 for infra's part; Cursor's plan, Ubuntu's kernel and Daniel's account changes are not mine to date |
| infra | **Check finishes within 45 minutes.** As ci's table has it. ci holds it, and the page should name ci as its owner | goal 4 | ci's day, 11/16 |
| infra | **Disk remake procedure written, first hour agreed.** Title unchanged. Check: the procedure is in infra's notes, since Thu, and the first machine's hour is on INF-732. The hour, cpu-5 on Sat 8 AM PST, is yours to rule at tonight's review | none | today, on your ruling |
| lean | **Lean PRs carry no lock records.** as it is. Check: the queue's warning and `restack.py`'s two `locks-stale` messages tell an author to record only what the candidate's records run can't write | goal 4 | Mon 16 Nov (was today: the three messages are in ci's files, which ci took Thu; both `restack.py` lines still say "regenerate them … before marking it" [read]. I ask ci its day on CI-489, and a worker of lean's writes it Mon if ci would rather) |
| lean | **Lean build cannot rewrite checked files.** as it is. Check: its test on `main` (a build still runs; both ways of changing a mount fail). It rides in the one pull request | goal 4 | Tue 17 Nov |
| lean | **The Lean check does once a run what it did once a package.** Check: lean's tool train on `main`, as the one pull request ci asked for on CI-489 (six parts, marked alone) | goal 4; `ci/no-cold-lean-build` needs it | marked by Sun 15 Nov noon; on `main` Mon 16 Nov if ci's readers and its slot follow within a day |
| lean | **Lean check finishes within 15 minutes.** as it is. Check: `research lean-times --hours 24`, 95 of 100 one-pull-request checks' Lean part within 15 minutes of work | goal 4 | no day I can hold for the bar. Tue 17 Nov gives the first measured day with the tool train; the model says 74 of 100, and the rest is the proof system's chain of about 185 files that each import the last |
| top | **No job stranded beside a free machine.** as it is | none | Tue 17 Nov |
| top | **Fewer changes refused for merge conflicts.** as it is; today's 6 PM review keeps, changes or drops it | none | today 6 PM |
| top | **Research velocity scored and steered every six hours.** as it is; reviewed today 6 PM | none | today 6 PM |
| top | **The Goals page's top level holds only Daniel's list.** Check: `research goals audit` finds no goal at the top only because nothing waits on it | none | today, within the hour of his word |

## Goals 5 to 12: each holder's line

Asked at 8:58 AM PST. A holder that has not answered shows the Goals page's own words in the table at the top.

| Goal | Command | Target | Today | A day its holder can hold, structure first |
| --- | --- | --- | --- | --- |
| 5. Remote demo | not answered yet |  |  |  |
| 6. Onsite demo | not answered yet |  |  |  |
| 7. Production vLLM's serving cost | not answered yet |  |  |  |
| 8. Network accounting on GLM-5.3 | not answered yet |  |  |  |
| 9. Proof-system performance | `verity bench prove --system spot --workload glm53` | At most 0.0545 prover GPUs a serving GPU, as benchmarks reads proofs' share of the 10% budget. proofs confirms the number | 0.0364, modeled (run `r20261113-015519-fdb2`): under the target in the model. The Lean verifier's time and the proof's size are modeled, not timed on a served audit | Tue 17 Nov for the command. The measured form has no day: benchmarks says by Mon 16 Nov noon |
| 10. Verified compute share | `verity bench compute --from RUN` | 97% | 95.44% on the run of record `r20261106-165116-614b`, printed again today at `main`. It is the credited share: on the H200 the credit rests on a premise red team refuted (ASK-296), so it is an upper bound there | Wed 2 Dec, if the LM head (Mon 30 Nov) and the massive-activation rows (Thu 19 Nov) are credited: the missing 1.56 points |
| 11. Uncertified memory share | `verity bench memory PERIOD` | At most 0.14 at the data sheet's read rate, in a session the Lean verifier on the auditor's host accepts, at the bandwidth audit's error of 2^-15 a session (not 100 bits) | 0.199, run `r20261112-193236-2608`, Thu 12 Nov | Wed 18 Nov, if two changes not yet written are on `main` by Sun 15 Nov 6 PM. memory-accounting says by 6 PM today. The path is one session on Mon 16 Nov, Tue the fallback |
| 12. Training slowdown | not answered yet |  |  |  |

## Goals 5 to 12: the rows that come back under them

Rows the leads parked under the first form. "Its lead" means the lead confirmed or corrected the sort on TOP-416.

| Goal | Lead | Row | Day | Sorted by |
| --- | --- | --- | --- | --- |
| 5. Remote demo | compute-accounting | Remote demo's proof covers Pearl+ | Wed 18 Nov, conditional on TT_OUT (ASK-296) | its lead |
| 5. Remote demo | compute-accounting | Pearl+ check runs on the auditor's machine | Fri 20 Nov | its lead |
| 5. Remote demo | lean | Inference-only policy proved for served traffic |  | top's first sort, not yet confirmed |
| 5. Remote demo | lean | Honest signed session's proof sized |  | top's first sort, not yet confirmed |
| 5. Remote demo | soundness | Sampled proofs sound as deployed | Fri 20 Nov | its lead |
| 5. Remote demo | soundness | Remote demo's soundness fully proved | Mon 30 Nov, as goal 8's "fully proved" | its lead |
| 5. Remote demo | soundness | Deployed remote theorem's honest witness runs a session | not known; I say on Tue 17 Nov | its lead |
| 6. Onsite demo | compute-accounting | Onsite demo's proof covers Pearl+ | Tue 24 Nov | its lead |
| 6. Onsite demo | memory-accounting | Memory verdicts on the onsite record | vllm's day for goal 6 | its lead |
| 7. Production vLLM's serving cost | vllm | Production vLLM reproducible under 1% serving overhead (the goal itself) |  | top's first sort, not yet confirmed |
| 7. Production vLLM's serving cost | vllm | vLLM serving cost benchmarked |  | top's first sort, not yet confirmed |
| 7. Production vLLM's serving cost | vllm | Serving cost measured on GLM-5.3 |  | top's first sort, not yet confirmed |
| 7. Production vLLM's serving cost | vllm | Serving cost's tick arm measured behind certifier |  | top's first sort, not yet confirmed |
| 8. Network accounting on GLM-5.3 | network-accounting | Network certification adds no exfiltration capacity |  | top's first sort, not yet confirmed |
| 8. Network accounting on GLM-5.3 | network-accounting | Network check ties responses to committed outputs |  | top's first sort, not yet confirmed |
| 8. Network accounting on GLM-5.3 | network-accounting | Certifier turns away arrivals at a run's end |  | top's first sort, not yet confirmed |
| 8. Network accounting on GLM-5.3 | network-accounting | Run record reports the tighter exfiltration capacity |  | top's first sort, not yet confirmed |
| 8. Network accounting on GLM-5.3 | network-accounting | Network bound counts only unchecked inputs |  | top's first sort, not yet confirmed |
| 8. Network accounting on GLM-5.3 | network-accounting | Onsite exfiltration capacity proved in Lean |  | top's first sort, not yet confirmed |
| 8. Network accounting on GLM-5.3 | soundness | Served weights proved to be the registered ones | Fri 20 Nov | its lead |
| 8. Network accounting on GLM-5.3 | vllm | Long-context request served and committed |  | top's first sort, not yet confirmed |
| 9. Proof-system performance | benchmarks | Proof system performance benchmarked (the goal itself) | Tue 17 Nov for the command printing it from labelled runs; for the verifier's time measured on a served audit, benchmarks says by Mon 16 Nov noon | its lead |
| 9. Proof-system performance | benchmarks | Head key blocks priced as served units | Mon 16 Nov | its lead |
| 9. Proof-system performance | kernels | Steady-state proving cost measured | today 6 PM PST: done if benchmarks' noon repricing reads 90% of proving time priced from measured rows; else I give its count and a day here | its lead |
| 9. Proof-system performance | kernels | Deep blocks' witness built on GPU | Wed 18 Nov, as dated | its lead |
| 9. Proof-system performance | kernels | Flock's wide leaf measured on a GPU | Tue 17 Nov, as dated | its lead |
| 10. Verified compute share | compute-accounting | Twice the sampled tiles a window | Tue 24 Nov (it was Tue 17 Nov; its H200 hours go after structure) | its lead |
| 10. Verified compute share | compute-accounting | Massive-activation rows credited | Thu 19 Nov | its lead |
| 10. Verified compute share | compute-accounting | LM head credited by Pearl+ | Mon 30 Nov | its lead |
| 10. Verified compute share | compute-accounting | Small model's verified compute share reaches 97% (the goal itself) | Wed 2 Dec | its lead |
| 10. Verified compute share | kernels | Overhead benchmark passes its GPU check | today 6 PM PST | its lead |
| 10. Verified compute share | kernels | PoUW overhead counts the output's commitment | Tue 17 Nov, as dated | its lead |
| 10. Verified compute share | kernels | Row commitment takes LM-head rows | Tue 24 Nov, as dated | its lead |
| 11. Uncertified memory share | memory-accounting | Uncertified memory share reaches its target (the goal itself) | Wed 18 Nov, as above | its lead |
| 11. Uncertified memory share | memory-accounting | Uncertified memory share at most 0.17 | Tue 17 Nov | its lead |
| 12. Training slowdown | soundness | Linear layers tied into inference-only check | Wed 18 Nov | its lead |
| 12. Training slowdown | training | Library holds standard algorithms and most frontier points |  | top's first sort, not yet confirmed |
| 12. Training slowdown | training | Every search frontier point runs from the library |  | top's first sort, not yet confirmed |
| 12. Training slowdown | training | Training circuits match GPU kernels |  | top's first sort, not yet confirmed |
| 12. Training slowdown | training | Training slowdown results final and proved |  | top's first sort, not yet confirmed |
| 12. Training slowdown | training | Inference-only accounting on a small model (the goal itself) |  | top's first sort, not yet confirmed |
| 12. Training slowdown | training | Small model's proved slowdown reaches 1,000 (the goal itself) |  | top's first sort, not yet confirmed |
| 12. Training slowdown | training | Slowdown plotted against model size |  | top's first sort, not yet confirmed |

## Lines and rows with no goal yet

L10, R11 (L = a not-yet line of the layout file, R = a row of the architecture file's code still to be written; the numbering is in the sheet on TOP-416.)

## Parked: what serves none of the twelve, or is not yet sorted under them

| Lead | Goal | The state it stops in | Under the second form |
| --- | --- | --- | --- |
| ci | Platform repository lands through the queue | Stops with the infra repository's loop stopped since Tue 10 Nov; INF-639 settles on Mon whether it's a second instance or a mode. Park, unless you want the platform repository held to two hours now | stays parked |
| comms | SPOT draft's differences on the page | Parks with the SPOT post. State: comms' part done (the $57$ placed, $36$ ready); proofs posts them when Daniel says yes. | stays parked |
| compute-accounting | Daniel's page on H200 accounting | As they are; the debit stops at #9735 (mend 1), and its card is with Daniel (ASK-296: B default, C recommended) | stays parked, its lead |
| compute-accounting | Pearl-C's debit closes red team's breaks | As they are; the debit stops at #9735 (mend 1), and its card is with Daniel (ASK-296: B default, C recommended) | stays parked, its lead |
| compute-research | Paused work resumed or dropped | As they are, each with the state Daniel paused it in on Mon 9 Nov: `accounting/silicon-logs-all-compute`, `accounting/useful-weights-hard`, `compute-research/chain-premise-probed`, `chain-round-pinned`, `first-four-layers-numbers`, `hardness-train-at-point`. | stays parked |
| infra | Infra repository merges without Daniel | Stops with: no change there has waited for Daniel since Tue, each merges on a verdict from a reader who is not its owner; the queue lands nothing there; its author fix is on `main`; the second change is ci's, which ci parks as `ci/platform-repo-queue` | stays parked |
| infra | Checker slot starts its next task at once | Stops with: the change on since Wed; the executor runs every submission in shadow on its own machine; twelve test files end each submission with no verdict (CI-526, infra's five being mended today); how many tasks run at once on a slot not read from the machine. I keep it instead if ci puts the executor under goal 4: ci's table names no goal for it | stays parked |
| memory-accounting | Cost of a 100-bit memory audit measured | Stops at: the 64-row session passes its check on one H200; its cost on the eight-card node is not measured. The bandwidth audit's error stays 2^-15 a session. If that is a hole under your ruling 1, the five changes that need no new version are kept under goal 2: say so | stays parked, its lead |
| network-accounting | Scan count statements restated as soundness amended | Stops at: #9639 is reviewed | stays parked |
| top | Paused work resumed or dropped | **Results resume after the architecture.** It holds every parked goal with the state it stopped in | stays parked |

## Dropped: done, duplicates, steps that are not a state of main

| Lead | Goal | Why |
| --- | --- | --- |
| benchmarks | One-stage every-block line out of the print | It only tidied the print of a result now parked. |
| ci | Tonight's Lean batches reach main | Done with or superseded: last night's Lean waited for chain 2 on purpose, and today's Lean order is held under `ci/lean-landing-two-hours` |
| comms | Friday's messaging review published | A daily routine, not a state of `main`: done after today's review, and no later one is made a goal. |
| compute-accounting | 97% of the small model's work verified | Duplicate of `compute-share-climb` |
| compute-research | Hardware data sheets moved to a top-level directory | True on `main` since chain 2: the cards' and nodes' data sheets are in `hardware/` at the root, which `tests/tree.toml` lists as "TOML only, no code". I mark it done today. |
| compute-research | Kernel machine-code check moved beside PoUW's prover | True on `main` since chain 2: the check and its pins are in `verity/protocols/accounting/pouw/prover/`, and `benchmarks/pouw/harness/` holds neither. Its last stale mentions leave with #9744 (marked, with its land verdict). I mark it done when #9744 lands. |
| compute-research | Dead PoUW Lean deletion in review | A step ("in review"), not a state of `main`. `lead-leftovers-deleted` holds the state. |
| compute-research | Compute files held by merge cleaned | A branch's state, not `main`'s. `compute-code-consolidated` counts the same files on `main`. |
| compute-research | Compute-research code cleaned up for review | Six of its seven parts are done (the sixth, the data sheets' move, today). The seventh is `data-sheet-reader-folded`, so it adds nothing to that goal. |
| console | The eleven's numbers live on the console | Superseded: the tab already lists every priority-1 goal with its number, so once the page's top holds only the tier-1 list it shows those, which `console/tier-one-numbers` covers. |
| infra | Quick runs start within five minutes | Folded into the row above as its second number |
| infra | Second night's check names no false disagreement | Folded into the row above as its step |
| lean | lean/older-names-sorted | a list, not a state of `main` |
| lean | Policy's proved and open parts listed | a list, not a state of `main` |
| lean | Refused lemma's answer read | a read, not a state of `main`; it is a step of `lean/witness-held-to-its-guarantee` |
| memory-accounting | PoUS grader moved out or deleted | True on `main` since chain 2: `tools/pous_grader/` holds its 11 files and none is under `verity/` [ran on `5eb1231b2558`]. I mark it done. Whether the package is deleted is my read of Mon 16 Nov noon on MEM-306, not a goal |
| memory-accounting | Verifier tests stop loading benchmark code | It is two of the five entries in the row above. Its change (MEM-259) keeps its issue and lands the day after #9061 |
| network-accounting | Benchmark reports network certification's exfiltration capacity | Done (network-accounting's correction, 8:44 AM) |
| network-accounting | Check prints the ruled value on Monday's data | Done as a run: the check prints 26,400 bits on Monday's data. Its change lands under `network/onsite-port-staged` |
| network-accounting | Onsite port recounted and first changes built | Done: the count, the list of 14, and the first changes open |
| network-accounting | Six cheating servers tried against a served run | Done: the six tries ran Thu night; the table is on NET-340 |
| network-accounting | Onsite network audits timed at the demo's scale | Done: timed in this morning's rehearsal (NET-331). The change it rests on, #9319, lands under `network/onsite-port-staged` |
| vllm | Root README describes the approved architecture | `ARCHITECTURE.md` at the root of `main` is what this goal was for. If you want a one-page `README.md` beside it that points there, say so and I write it the same day |
| top | Hotspots ranked nightly, top rows closing | a nightly routine, not a state of `main`; the bar goals carry its rows |
| compute-accounting | 97% of H200 capacity accounted for | Under the second form: D: goal 10 is the target, on the small model |
| compute-accounting | 97% of the small model's work verified | Under the second form: D (as before): a duplicate of goal 10 |

# The numbering used above

## L: the layout's not-yet lines (`tests/tree.toml` at #9742's head)

| | Path | Goes to | Owner | Day |
| --- | --- | --- | --- | --- |
| L1 | `verity/protocols/reporting.py` | into verity/protocol.py | top | 2026-11-15 |
| L2 | `verity/hardware/registration.py` | into the hardware file | top | 2026-11-15 |
| L3 | `verity/hardware/record.py` | into the hardware file | top | 2026-11-15 |
| L4 | `verity/hardware/lean.py` | deleted, once the Lean takes a card as a parameter | compute-research | 2026-11-24 |
| L5 | `verity/definitions/` | its protocol part to verity/circuits/, its served part to the vLLM adapter | top | 2026-11-15 |
| L6 | `verity/claims/` | not ruled, with soundness: it defines the assumption ids that Lean and the locks name, so it is not deleted | top | 2026-11-17 |
| L7 | `verity/periods/` | verity/protocol.py | top | 2026-11-15 |
| L8 | `fixtures/` | beside its reader, each file | proofs | 2026-11-18 |
| L9 | `verity/protocols/accounting/pouw/clear/` | verity/protocols/accounting/pouw/prover/clear/ (corrected by compute-accounting this morning) | compute-accounting | 2026-11-16 |
| L10 | `verity/protocols/proofs/flock/sigma/` | its protocol's verifier/, prover/ or device/, or out of verity/; proofs says which | proofs | 2026-11-18 |

## R: the architecture file's rows of code still to be written (`ARCHITECTURE.md` section 7 at #9702's head)

| | What | Today |
| --- | --- | --- |
| R1 | Every protocol has parties | Only SPOT, PoUS and the remote memory-read audit do |
| R2 | A PyTorch adapter under the vLLM adapter, with one translator from PyTorch to circuits | One adapter, and a second translator inside the training benchmark |
| R3 | The messages between workload and prover as one stated list | Function calls and files; 15 files with no settled name or place |
| R4 | PoUW's serving code and GPU kernels in one package under `protocols/accounting/pouw/`, reached by one call | In the adapter (18,800 lines) and under `benchmarks/` |
| R5 | Benchmarks and demos reach the adapter through its public interface only | 114 modules are reached past it |
| R6 | The two demos start from a configuration, run by the runtime | The remote demo is a shell script under `benchmarks/`; the onsite demo is not built |
| R7 | Python never returns accept or reject | Several protocols still decide in Python. SPOT's two-stage outcome is joined in Python; its Lean check is written and refuses every audit until its pr… |
| R8 | Each accounting protocol takes the registered hardware and says which part it accounts for | None does |
| R9 | The security model has the auditor's devices as parties, each with a key the auditor pins and its whole signing history | Only the challenger is a party, so no onsite memory or network claim can be proved yet |
| R10 | A device's figures have one source: the registration reads them from the auditor's data sheet | Written in two places that disagree (the H200's tensor rate at two clocks) |
| R11 | "Registration" names one thing: SPOT's own record of a window takes another name | SPOT's code calls the record it fixes before any challenge a registration record: the circuit's digest, its pieces, the sampling rule and the committe… |
| R12 | The circuit checker is handed the served circuits and names none | It names them |
| R13 | The compiler cuts a period's circuit into pieces wired to each other | For the demo's model the pieces are wired only where the eight GPUs combine results, so the audit's second round runs over a stand-in |
| R14 | Every kernel the production settings serve has a circuit held to the kernel on an H200 | Five are written. Four are compared with the kernel on an H200 and no word differs: the FP8 cache's write, its two reads, and the accept rule. Two of … |
| R15 | Every kind of piece in a request has a statement the Lean verifier accepts | 18 of 254 kinds, holding 88.1% of the pieces |
| R16 | One form of the network certifier | Two: the remote demo runs an older one, a proxy in front of the server; the onsite demo runs the device on the link |
| R17 | The Lean takes a data sheet as a parameter | Two Lean tables restate their data sheets |
| R18 | Every benchmark on the dashboard | 5 of 69 wired; 3 of the 5 show a value no older than their schedule allows |
| R19 | The proofs' status on the dashboard: guarantees fully proved, open premises per lead | Printed by the Lean check only |
| R20 | The Lean check passes on the public repository | On Tuesday's export 14 of 16 Lean packages pass |
| R21 | Every area of `security/` has a lead | 826 of its 4,816 files fall to lean because no lead claims them |
| R22 | A memory-read session is proved in zero knowledge | Each span's openings go to the verifier as they are |
| R23 | PoUS's setup is a proof about the store | The check reads the whole store |
| R24 | PoUS's session is checked inside a proof | The check reads the memory challenger's signed record as it is: for each round, which block was asked, when it was sent and answered, and the block's … |
