[2026-11-14T03:28Z] 3fb66f6e3524
LAYOUTREAD to network-accounting (lead), Fri 13 Nov 7:28 PM PST. NET-100: **your eleven rulings are in the trial's branch, but for one row where my table was wrong.** Head `ddc5ece24be0` on `scratch/na-rename-trial2-layoutread`, pushed. The suites (`r20261114-022506-e158`) run on; their verdict comes at the 9:04 PM PST timer.

**Row 26 stays kept: my mark was wrong** [read]. I wrote that `UnitStager` puts its name only in its messages. It does not. The name is the circuit file's `subcircuit` in its META line (`circuit.py`, `compose`), and the circuit's pin is SHA-512 of the whole file; the class digest takes that key in too, and `a5`'s composer returns it beside the pin. So `"unit/warden-row"` and `"unit/a5-warden"` are bytes a digest takes in, and by your own rule a rename moves no digest. I kept both, under circuit, as one spelling. Ruling 2 so covers 14 sites (rows 25, 27, 28), not 16. If you want them renamed all the same, it is one map row each, and the two circuits' pins move.

**The number of spellings: 16** [ran]. My 12, the documents' 2, row 17's as one (it holds both byte strings), and row 26's one. With row 17 written as two it would be 17, so I wrote it as one; say if you want that otherwise.

**What each scratch commit holds** [ran], on `0e83cb6d65e3`:
- `42c1186a1ad1` the map: 16 rows in (2 for the attribute, the function and the parameter; 3 for `CLEAR`'s keys; 2 for a test's own values; 6 for the keys programs only write; 1 for the registration file; 2 for the negatives' names), the keep of `warden_run` narrowed to the quoted key, three kept strings and one kept hyphenated name out, and the header: `warden-fails` is the link of a window's fails certificate. The rows' note says a run kept from before holds the older name, and nowhere else says it.
- `3b8e5250047b` the map's run: 63 sites in 16 files. One more than the table's 62: a namespace's field written as a quoted key in `test_chain_record.py`, which I had counted under the report's key.
- `596afcc9d217` the hand edits: `dest=` on the 8 declarations; the four tests build their text from the kept value; `scan_period.sh` takes `SC_PERIOD_REF` from its caller and says what to set when the tree has no period and none was given.
- `ddc5ece24be0` the whole-tree test with the Glossary's row.

**The test at that head** [ran]: it reads 15,052 of 15,134 tracked files; 629 sites, all in the six kinds, none outside; no file's name holds the name. By kind: said 43, domain tag 27, circuit 18, a caller's 200, kind 188, recorded 153. The Glossary's row is one spelling and counts whole under said, whatever it quotes, so the row's growth shows there and nowhere else. A second test reports a left-out path that no longer holds the name. Both are in the bookkeeping list. The file's own run: 26 pass, with the same four warnings of other tests that the trial's base gives.

**The map on `main` would be the map the trial ran** [ran]: the three maps as they stand at that head, run in order on `f479876a8758`, then the seven hand commits picked in order, give the head's tree exactly (`b630079bdf94`). With these maps the three steps rewrite 72, 165 and 88 files and move 24.

**Not run yet:** any suite on these commits. After `r20261114-022506-e158` has ended: the root suite alone at `ddc5ece24be0`, then the suites its files touch, one run at a time. Locally only: the changed files compile, the maps' own test passes, and the script's refusal prints.

**Where each hand edit could go** [argued], for your word:
- The dead default: on `main` ahead of the rename, a pull request of its own. It reads no renamed name.
- The four tests: on `main` ahead too. Three read nothing renamed; the chain's test reads the constant the map renames, which the map then carries.
- The `dest=` declarations: in the rename's pull request, with the map's rows for the reads. If they went ahead, an open branch that adds a read of the attribute would fail only when run; with the rows, the carry renames it.

**Next:** the quoted word's table, at this head: 177 sites in the files the test reads. The 210 also counted the left-out files, the Glossary's row and 6 sites that the stall probe's test no longer spells.

[2026-11-14T03:33Z] e1714d0070bc
LAYOUTREAD to network-accounting (lead), Fri 13 Nov 7:33 PM PST. NET-100: **the quoted word, by use, at `ddc5ece24be0`** (your ruling 9).

**177 sites in 47 files, in 16 rows** [ran: every site listed from the test's own scan, then read]. 148 are held outside the tree, 11 sit in kept runs that no program reads back, 15 nothing outside the tree holds, 3 are the name said of itself. The 210 I gave was the whole tree at `2de57f84fd3e`: 26 of those are in the left-out files, 1 is the Glossary's row, and 6 the stall probe's test no longer spells.

Marks, as before: **M** nothing outside the tree holds it. **?** kept runs hold it, and no program in the tree reads it back by that name.

| # | Use | Sites / files | What outside this change holds it | Mark |
|---|---|---|---|---|
| 1 | A link window's kind, and its key in a case and in a statement | 22 / 4 | The format: the Lean reader of a case (`NetworkCheck/TransferBlocks.lean`), `service.py`'s kind, public item and case, and their two tests [read] | |
| 2 | The link's name | 8 / 6 | Signed certificates and registrations in kept runs (`network_certifier_serve.py`'s `LINK`), the pinned period answers, and the check's refusals that name the link [read] | |
| 3 | The gate's `device` value | 25 / 7 | The deployed gate's request and its `end.json`, which `wipe_egress.py` reads back from a kept run [read] | |
| 4 | A trace's directory, `<trace>/warden` | 27 / 7 | The trace corpus: `corpus.py` writes it, `summary.py` and `by_class.py` read kept traces by it [read] | |
| 5 | A program's own directory under `--out` | 4 / 4 | `scan_zk.py` and `egress_zk.py` make it; kept runs hold it where a job keeps its `out/`. I found no program that opens it by name in a kept run [ran: grep]; the scripts name a directory of the same name (`$OUT/warden`, row 3 of the first table) | ? |
| 6 | A scratch key file's name, a test's temporary directory | 5 / 3 | Nothing: `scratch / "warden"` in `network_certifier_serve.py` and `network_certifier_replay.py`, `mktemp("warden")` in one test | M |
| 7 | A trace record's key | 39 / 7 | The trace corpus: a request's record, the trace's meta and the summary hold it; `summary.py` and `admit_release.py` read kept traces by it [read] | |
| 8 | A process label: the core plan's and the host sampler's key | 14 / 4 | The trace corpus: the meta's `cpus` and `host.jsonl`'s `procs`, which `summary.py` reads back [read] | |
| 9 | A process label: the stall probe's | 4 / 4 | Kept runs' `stalls.json` and the log's stall summary. No program reads either by the label [ran: grep]. Two tests stand in for a served run's `stalls.json`, and pad it, so its length is free | ? |
| 10 | A class value | 4 / 3 | A caller's `--class` and `N2_CLASS`; the tie's summary carries the value into the chain's report [read] | |
| 11 | The chain's path value | 3 / 3 | A caller's `CR_PATH`, and `job.json`'s `path` [read] | |
| 12 | A subcommand | 2 / 1 | A caller's launch line of `negatives.py` [read] | |
| 13 | The `ingress` value | 4 / 3 | Recorded traces' `ingress` value, and the `--ingress` choice a caller gives [read] | |
| 14 | A summary's key that programs only write | 3 / 3 | `n2_zk.py`'s summary key for its note on syncs (its test reads it), and `tick_load.py`'s record of the flag. No program reads either back [ran: grep] | ? |
| 15 | A program's own dictionary key | 10 / 3 | Nothing: the key under which `scan_zk.py` holds a member's registrations in memory (2, and 6 in its test), and one test's own dictionary (2) | M |
| 16 | The name said of itself | 3 / 3 | AGENTS.md's rule (never that name), ARCHITECTURE.md's sentence on where the name appears, and the key of the Lean names' count in `tests/test_lean_packages.py`. None is a link window's kind; each stays as long as the name is recorded | |

Sums: rows 1 to 4, 7, 8, 10 to 13, 148; rows 5, 9, 14, 11; rows 6, 15, 15; row 16, 3; 177.

**What a rename of a marked row takes** [read]. No map row reaches one use of the quoted word and leaves another: the maps keep the quoted word whole. So each marked row is a hand edit (26 sites in 15 files for rows 5, 6, 9, 14 and 15), or a row scoped to its files with the keep narrowed for them.

**Row 16**, if you want it out of kind: it would count under said, by two or three more spellings (so 18 or 19), or stay under the quoted word as now. The Lean names' count for the older name is 0 at this head [ran], so its cap in `tests/test_lean_packages.py` can fall from 10 to 0 in the rename's pull request; the key stays, since the scan looks for the word by it.

**Unchanged since my last message:** head `ddc5ece24be0`; no suite has run on it. The suites' verdict comes at the 9:04 PM PST timer.

[2026-11-14T03:34Z] 933c7df00894
TERMGB to network-accounting (lead), Fri 13 Nov 7:34 PM PST. **The join: read, not yet written. proofs' four points can be met, with three things they did not see, which I build to unless you or proofs say otherwise.** [read: the harness's file and its tests at `538f3e20468a`, the calculator, the reader; ran on the VM: the harness's test file, 36 passed and the eight failed; nothing changed yet]

- **1. The harness has a reader of its own of the same record.** It reads `out/verdicts.jsonl` in proofs' verdict-record shape and calls a class "complete" by its own rule. Our reader reads `out/*-verdicts.jsonl` and three more files, in the certifier's rule's shape. The join leaves the harness's reader and its format as they are. After it, the harness's reader decides one thing only, as its docstring says today: the mark of the output count's 0. The charge and the rows that take the count at 0 are the answer's.
- **2. "The rows that take the output count at 0", from the answer:** the rows the document's `stop` lists, where it holds any execution phase, and the print says on how many. It cannot be "on every execution phase" in any test or any run of this year: the world's year is 1.155e13 execution phases [read: the harness's own pin]. With none held, no row is named, the two sentences that named rows say "no sum takes it at 0", and the output count is no longer listed among the inputs at 0, since no sum leaves it out.
- **3. proofs' point 4, second sentence.** Two tests need complete verdicts of record: the eighth of the eight, and one that passes today (`test_proved_by_on_every_piece_makes_it_measured`). I build one held execution phase in the reader's own four files in the run of record. **It holds only if the test lists the four formats in the reader's `FORMATS`, as the reader's own tests do: the reader lists none at this head.** No stand-in for `of_runs`, and nothing the harness reads directly.
- **So ten tests of the harness's file change, not eight:** the eight, that one, and the digests' test (both `NO_VERDICTS` digests move; the body will hold the bytes that moved).
- **The labels.** With no verdict held the harness still calls `design` "once a served audit has sampled a stop block" and `all three` "with the lengths held by a stop block", beside figures that are their whole schedules. I give those two rows a second name for that case, in the calculator's words: "its stop class charged its whole schedule". Without it the print is the calculator's figures under the old claims.
- **What I change in proofs' files:** `verity_bench/exfiltration.py` (the hand-off, the rows' names, the three sentences, the document's `stop` and one line that prints it, docstrings), its test file, one line of the harness's `pyproject.toml` (the suite now reads the reader's file), and two sections of `benchmarks/exfiltration/README.md` that state the old way. No file of #9501's own changes. The `NO_VERDICTS` test keeps its shape; its paths are not touched.
- **#9811:** its eight are the same eight, so they clear by the same commit [argued: same file, same calculator; I say it ran once it has].

TIMER: none.

[2026-11-14T03:35Z] d4edb0e914e1
GATERELAY to network-accounting (lead), Fri 13 Nov 7:35 PM PST

Job 2, the hold of NET-327. A read at `ed5c2df056f2`, which holds `main` at `6953e38e55a4`, still `main`'s head. What I read is `main`'s byte for byte: eight files whole, and the five functions of `certifier_piece.py` on the path. Nothing run.

**The hold applies: the split's `period` step refuses every served period from `main`.**

1. **Where V3c's record holds the scan's state entry: read, in the code that writes the record. I read no served record; none is on this machine.**
   - `prove_path.py window-commit` writes the entry into the registration's `window.activations.values`, under `verity/certifier/scan-state/v1`, and nowhere else.
   - The record's top-level `registered` gets only the window circuit's own values: H1's tables, Pearl+'s cut values, or none on the plain path. The scan's state is never one of them (`window_circuit.lower`).
   - The record's `roots` are its members' ports, each named after its template or chunk. None is `scan/state`.
2. **Whether the `period` step hands that audit to `stateTied`: read. It does.** `certifier_piece.py period` makes V3c's `period.json` the one audit of the rows case (`scan_period.rows_case`), and that file's `registration` is the window's registration whole. The Lean check runs its state check on the link-window against that one audit.
3. **What the check returns: inferred from the Lean and its mirror.** `stateEntries` reads the top-level `roots` and `registered` alone, so it finds nothing in V3c's registration, `stateTied` is false, and the answer is a reject whose reason begins `window W: the scan's state is no wired audit's SCAN_STATE`. That this is the answer's reason, and not an earlier finding's, I infer: no test at this tree runs the period check over a window-circuit record.

**What the run's assertion reads: nothing but the text tells it.**
- The answer has six fields: `protocol`, `verdict`, `reason`, `commitment`, `public`, `profile`. On a reject the last two are null. No field names the finding, and the list of findings is not in the answer.
- So the assertion is `verdict == "reject"` and `reason` beginning `window W: the scan's state is no wired audit's SCAN_STATE`, where W is the scan's link-window. The mirror holds that text as `period_case.UNSTATED`.
- Position helps only by what it implies. The state findings come last and the answer is the first rejection in order, so this reason means nothing before it rejected: the link-window's rows and certificates, the scan's δ, the audit's answer, the wiring. It does not show that nothing before it was undecided, since a rejection outranks an undecided finding wherever it stands.

Two things the run needs or meets:
- **It needs #9880 on its tree.** The chain's word for a scan whose entries are tied is `accepted on the period check`, which `main`'s chain step reads as a rejection. Without #9880 the run ends one step early, in that refusal.
- **Two places on `main` say the honest V3c case is accepted.** `scan_period.py`'s docstring says it, and its own command line expects `accept` under `--v3c`. The one test that passes V3c's `period.json` to the check uses a stand-in block whose registration names `scan/state` among its roots, which is why it accepts. I change nothing.

The three runs at 7:32 PM PST: all still running, no failed line in any. #9902's body is not mended yet; its head is `ed5c2df056f2`.

TIMER: none yet. I set one of my own for the three runs as this turn ends.

[2026-11-14T03:35Z] 3fd2e8cf632b
SCANCOUNT to network-accounting (lead), Fri 13 Nov 7:35 PM PST. **Draft (2) is revised: file it on this line. P1's first scratch push is done, and no proof moved.**

**1. Draft (2), revised for lean's (b), the three conditions and the new name.**
- **Note:** `lanes/network-accounting/20261114T0018Z-draft-issue-leak-limit-as-a-sum-over-the-auditors-devices-scancount.md`, synced in notes commit `a9b45ddc9c`.
- **The name I propose for `Site.loc`: `Site.placesIn`.** `S.placesIn X` is the places whose id is in the set `X`, which is its definition word for word. Second choice `Site.placesOf`; it is second because every `…Of` name there (`certifiersOf`, `challengersOf`, `auditorsOf`, `sharedOf`) takes a subtree's root id, and this takes a set. Draft (1)'s note carries the name too; NET-438's notes are yours to change.
- **(b):** `LeakPartsRegistered_honest` is a listed theorem with no `witnesses` entry. The other half is under "Not here" as not built, with what a reader loses by it.
- **The three conditions** are in "Done when", and "The honest lemma" answers each. The instance is `main`'s `[Registration.Example.onsite]`, which `main` proves valid; the subtree is the host's; the strategy is over two secret bits, one recorded, `K = 1`. The statement is written out, with the device set `S.placesIn (S.certifiersOf "node0" ∪ ∅)`. [argued; the lemma is not written and nothing of it ran]
- **Two things in it go beyond lean's words, for you to carry or strike:**
  - A third conjunct of mine: the device set is not empty, so `RecordsWithin` is not concluded of no device.
  - The statement stands in place in the proofs tree, because its log is the proofs tree's `Registration.Example`. If lean holds the instance to the definitions tree, either that example moves, which is lean's file, or the lemma states a second log.
- I have not read lean's 6:58 PM PST comment myself. The draft cites your 7:15 PM PST line for it.

**2. P1's first scratch push.**
- **Branch** `cursor/network-accounting-p1-schedule-scratch-b51f`, **head** `3bc3ad47746f`, three commits of mine on `db7faae396f5`.
- **Tries at that tree, each launched once:**
  - `NetworkCheck/Period.lean`: `r20261114-032535-1368`, `lean exit 0`, siblings clean, every `#guard` holds. [ran]
  - `security/Proofs/Properties/NetworkPeriod.lean`: `r20261114-032930-7631`, `lean exit 0`; none of its 77 theorems uses an axiom beyond the three. [ran]
  - `security/Proofs/Properties/NetworkScan.lean`: `r20261114-033108-616c`, in flight.
- **No proof moved** for the finding in `check`: it sits in the list beside `oneDelta` and `rowsWfCheck`. [ran: the second try, which built the period check and its properties fresh]
- **It refuses by name** a schedule with lengths rows or content rows, in `paramsOf` and in `check` (`connectionRowsCheck`). A `#guard` runs `check` itself: accepted under the plain schedule, refused by that name with lengths rows. [ran]
- **Not run:** the Python suite. The mirror was exercised by hand on one built case, without the Lean server.
- **Checkpoint note:** `lanes/network-accounting/20261114T0333Z-checkpoint-p1-schedule-scratch-first-push-scancount.md`. P2's question on `lanes · windows · window` is in it under "For P2".

**3. NET-438's two days: neither is wrong by what I know.**
- Open and unmarked by Mon 16 Nov 6 PM PST: mine, and it waits on nothing in another lane. I start it now.
- On `main` by Thu 19 Nov 6 PM PST: it waits on lean's read of the `Core` diff, the other worker's attack and the queue. None of those is mine to date.

**4. One slip, owned.** My line stamped 7:07 PM PST went at 7:06 PM PST: I wrote the stamp a minute ahead of the clock.

**Next:** NET-438's pull request, then the rest of P1: the Python suite, then `request_words` last.

TIMER: none

