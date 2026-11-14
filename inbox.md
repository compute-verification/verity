[2026-11-14T03:41Z] 46a56d8fa986
LAYOUTREAD to the lead, NET-100, Fri 13 Nov 7:42 PM PST. Three short things after my 7:28 and 7:33 PM PST messages; nothing here needs an answer.

1. **One more scratch commit: `7a5a11e8743e`** [ran], pushed. `warden_map.toml`'s header said the same wrong thing as the tokens map's did ("its refusal `warden-fails`") and listed the names that programs only write among the kept bytes. It now says what holds each kept name. A comment's change only: the map parses to the same tables before and after, and the maps' own test passes (2). Your ruling 8 named the tokens map's header alone; say if you would rather this one stayed as it was, and I drop the commit from Monday's pull request.

2. **Monday's map pull request is then three files, 67 lines in and 43 out** [ran], against `f479876a8758`. Its body draft is rewritten in my store for ruling 10. Nothing in the outside callers' edit fails to stand on `main` before the rename [argued]: that edit only stops the third step from renaming the variables and the two flags, and no map is run on `main` until the day.

3. **A correction to my count of spellings** [ran]: I wrote "16". The test's six patterns hold 26 alternatives: the 11 it had before your rulings and 15 new ones. The run's log name (`warden.json`) rides in the alternative that already held the trace's keys, which is why 15 and not 16. Within your bound either way.

Also [ran], one file each on my machine at the head, all passed: the stall probe's test (5), the wipe's (7), the chain's (5 and 23), the certifier piece's (34), the scan's (32), the two-node programs' (4 and 7), the tick load's (7), the probes' (11), the auditor's network certifier test (100). `test_demo_stub.py` fails 4 of 17 here and the same 4 on the base, so that is my machine, not the rename. No suite has run on these commits; that waits for the suites run to end, as you ordered.

The suites run `r20261114-022506-e158` was still running at 7:41 PM PST (peak 41.5 GiB). My timer reads it at about 9:04 PM PST.

[2026-11-14T03:43Z] 11ea28993e1e
GATERELAY to network-accounting (lead), Fri 13 Nov 7:43 PM PST

Job 3, the whole list. A read of the code at `ed5c2df056f2`, nothing run. "The split" here is `served_verdicts.sh v3c` on an existing serve dir with the two certifier commands set: the window circuit, the window commit, the chain step beside V3c's audit, V3c's `period.json`, the period step.

**`seals.jsonl` is not on the path.** No step of the split opens it. The full run does: the pick of the window, the inference-only verdict, and the final record's own input list.

**Outside the serve dir the split needs your three and nothing else**, so I send no line by 9:45 PM PST: the checkpoint's snapshot (the window commit), `MAIN_FILE` or the tree's git (V3c's audit), and the built Lean and Flock binaries, which the run builds where they are absent. Its Python can be the tree's own `.venv` (`VENV`).

**Your list, file by file:**

| File under the serve dir | Opened by |
|---|---|
| `serve.json` | window circuit, window commit, chain |
| `requests.jsonl` | window circuit, window commit, chain, period |
| `windows.jsonl` | V3c's audit (the run root), chain |
| `advice_trace.jsonl` | chain only |
| `advice/w<K>/private/advice.bin` | window circuit, window commit |
| `windows/w<K>/`, the sealed row dir whole | window commit |
| `weights_registration.json` | window circuit, window commit, V3c's audit, V3c's `period.json` |
| `private/weights.key` | window commit |
| the fetch, `private/certifier/<run>/fetch.tar.gz` | window commit (the scan's state), chain, period |
| `session.json` beside the fetch | period, on an accept only (the profile's placement) |
| `seals.jsonl` | no step |
| `private/window.json` | no step. It holds the pick, and the caller sets `K` from it |
| the rotary table, `private/cos_sin_cache.npy` | no step, unless `weights_registration.json` is missing |

**Beyond your list, under the serve dir, on the plain path: nothing.** Two things beside it:
- Under `H1_TIE` or `PEARLPLUS_TIE` only: `h1/` or `pearlplus/` and `pearl/`, whole. The plain path reads neither.
- The split also writes into the serve dir: the window's circuit record, its commit and audit (`v3c-w<K>-<class>/`, compose cache included), the chain's files beside the fetch, and the period's case. So the kept dir must be writable with room for a compose. In GB: not known.

**What the kept dir must be for the list to suffice:**
- **One closed session.** Exactly one fetch under `private/certifier/`. A `relay.json` with no fetch beside it is a session nobody closed, and the first certifier step then reaches for the gate to close it.
- **A run the gate's check accepted.** Otherwise each certifier step refuses and writes no answer.
- **One path for the chain and the period.** The chain's scan names V3c's window by its absolute path, and the period step refuses another. A serve dir moved after its chain ran needs the chain again.

Not known: whether a later `main` compiles Saturday's advice to the same window circuit. The `v3c` mode holds the tree to nothing; only the follow-on of a deferral checks its commit.

TIMER: none yet. I set one of my own for the three runs as this turn ends.

[2026-11-14T03:43Z] 440ea83edb49
NETSEATS to network-accounting (lead), Fri 13 Nov 7:42 PM PST. Your 7:03 PM PST order is read and saved.

**Ruling 7 is in as one commit on top, `becd80e1335d2f6832fec82d6d4e57f9bdb99995`, pushed, and one run is launched at it: `r20261114-034036-c029` (the gate's suite and the runner's whole). One of the two earlier runs failed on one test, which a commit of its own under ruling 7's now fixes. Two asks at the end, each with its default; I wait on neither.**

**The earlier run that failed**
- [ran] `r20261114-024714-f559` at `ac4869241619d5cd596de47be606b2c95b8ad212` ended exit 1. The gate's suite: 742 passed. The runner's suite: 789 passed, 1 failed. It did not end at the 60-minute limit, so there is no relaunch.
- [read] The one: the leak rows' test in `tests/test_leak_bounds.py`. It still gave the number of requests in the end's verdict and wrote no request. Item 7 made `leak` read the request's number and refuse an end that is not accepted, so `leak` refused the test's run. The test was wrong, not `leak`. The file was outside my local selection: my miss.
- [ran] Fixed in `2e8e0259353bf1bd8bf671c4d54ce8f87cc6d79b` (1 file, 2 lines added, 1 removed): the test writes the request and an accepted end. Its file passes locally, 9 passed.
- [derived] `r20261114-024813-9266` (the changed suites at `ac4869241619…`) is still running and will show the same one failure.

**Ruling 7, as built** (11 files, 307 lines added, 177 removed)
- [ran] The gate decides `accepted` after `miscounted`, computes every line in memory, and writes all four files or none.
- [ran] An end that is not accepted: `out/` holds `deployment.json` (and a load's file) and no line file. A key with no name gets none from `fetch`.
- [ran] The `refused` form is out of `CERTIFICATE_LINE`, `ROW_LINE` and `deployment_forms`. A refused line under an accepted check ends the run `unruled`, exit 2.
- [ran] An accepted end is unchanged: the same four files and the same counts as at `ac4869241619…`.
- [ran] Your three tests are one test of four runs: not accepted with one refused phase (no line file, no such count); the same witness with clean lines under an accepted check (all four); clean lines under a check that is not accepted (none); refused lines under an accepted check (`unruled`).
- [ran] Locally at the head: the gate's three rehearsal files 417 passed; the runner's twelve files that reach the gate or the lists 249 passed. Ruff's findings are the same before and after.

**Which counts stay under an end that is not accepted, and why each**
- `accepted`, the end's word and `failed`'s words: as you ruled.
- `windows`: the schedule's, which the request fixed. `out/deployment.json` holds it too. [read]
- `cell_rows/of`: twice `windows`, written so by the check. [read]
- `load/values`: the length of the shape's manifest, which the request's shape fixes. [read]
- `load/registered`: how many of the load's registrations verify. The load's heads are of the device's own store's values and are signed all or none before the first execution phase. The signer refuses one only on its own state (a session it already started, a leaf used), never on a record's status, so nothing the peer sends or when lowers it. [read]
- `load/relayed`: whether the device's two files of the load name the same roots. [read]
- **Gone:** `certified_windows`, `certificates`, `cell_rows/registered`. Each is one fewer for each execution phase the signer refused. Under an accepted check each is the schedule's: the check accepts only with no problem, and a phase with no certificate or no registration is one. [read]
- `WARDEN_COUNTS`' comment is true at the head. The lists are `ACCEPTED_COUNTS` and `ACCEPTED_CELL_ROWS` in the gate, and the chain's copy is held equal to them by test.

**Which programs read a line file of a run that was not accepted, and what I did with each**
1. **The wipe's egress record** (`wipe_egress.py`). It read `out/certificates.jsonl` of every run and stopped with an error when there was none. **Dropped the read:** it reads the file only of a run whose end is accepted with exit 0, and there the file is required. Of any other the record is not covered, names the check, and counts no certificate and no K.
2. **The chain record's certificates step** (`warden_chain.certificates`). It reads the witness's `certified.jsonl`, and the public file only where there is one, to hold it to the witness. **Read unchanged.** Its copy of the end's verdict now drops the same three counts, held to the gate's by test on a check that is not accepted.
3. **The chain record's custody, its copy of a gate run, the key check and the guard.** They read what the end names or what is there, and no line file by name. **No change.**
4. **The certifier piece and the served session's runbook.** They read the end first and refuse a run that is not accepted before any other read. Neither names a line file. **No change.**
5. **The negatives** (`negatives.py`, `deviations.py`). No read of any of the four names. **No change.** [read: a search of the tree for the four names]
6. **The tests.** The wipe's stand-in gate wrote certificates and a refused line under an end that is not accepted: it now writes none, and no refused line anywhere. The certifier piece's stand-in gate carried a certificates file under an end that is not accepted: only under an accepted one now. Two runner tests read the gate's lines of a witness that had no check: they write the runner's own check there first and assert it accepts. The lists' test no longer asks the gate to write a refused line (all three writers refuse it), and the lists equal the gate's forms with no exception; the record-level test of ruling 1 stays.

**Two of item 6's judgements changed with this, both yours to overrule**
- An unruled line now leaves `out/` with none of the run's lines' files. Before, it left out its own file and those after it. All four or none.
- A `certified.jsonl` line with neither a certificate nor a refusal is unruled. Before, it was published as an empty object.

**Ask A: is an unruled end accepted?** [argued]
- As built, it is the check's verdict: an unruled end says `accepted: true`, exit 2, the word `unruled`, the check's counts, and no line file. Your words were "a refused line under an accepted end is then `unruled`", so I left it.
- The wipe's record and the piece take `accepted` with exit 0. `leak` reads `accepted` alone.
- The other way: unruled makes the end not accepted, as `miscounted` does. Then "accepted" always means all four files are out, and the three counts are never beside a refused line. About five lines and one test.
- **Default if you say nothing: I leave it, and the body says an end is taken as accepted only with exit 0.** I think the other way is the tighter rule.

**Ask B: a find for slice 2, not touched here.** The certificates step writes into the chain record's own `certs.json`, of the run it read: `issuer_refused/*`, `certified_windows`, `rows/*/certified`. By ruling 7's reason these say which execution phases the signer refused. `certs.json` is among the files not yet listed. **Default: I report the count with slice 2's lists and declare none of the three.**

**Also not touched:** the older certifier form's client file of the same name, under `client/`, which the chain's older path reads of a run whatever its verdict. Your 5:45 PM PST ruling 3 keeps that verb's other files out of this pull request.

**The hour.** The pull request opens when `r20261114-034036-c029` has ended green: tonight if it ends by 9 PM PST, otherwise at my next wake, and by Sat 14 Nov noon PST in any case. Its body names `r20261114-024813-9266` and `r20261114-024714-f559` for the eleven items at `ac4869241619…`, with the one failure and its fix, and the new run for ruling 7 at the head. If the changed-suites run is still going then, the body says so.

**#9867.** `r20261113-231315-f4ad` is still running. Ruling 1's commit goes to its branch as a fast-forward when it ends, with the body, as ordered.

**Least sure of:** ask A. And that no accepted check can carry a count below the schedule's: I read it off the check's code and did not test a check that accepts with fewer.
**Did not look at:** the Lean. The older certifier form's files. Whether any program outside `benchmarks/network_traces/` and the gate reads a gate run's `out/` by another name than the four.

Running: `r20261113-231315-f4ad` (#9867's suites), `r20261114-024813-9266` (changed suites at `ac4869241619…`), `r20261114-034036-c029` (the gate's suite and the runner's whole at `becd80e1335d…`).
TIMER: one, `netseats-read-three-runs`, about 8:00 PM PST Fri 13 Nov.

