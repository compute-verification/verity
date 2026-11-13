[2026-11-13T02:30Z] 001bad913fb4
From LAYOUTREAD, Thu 12 Nov 6:31 PM PST, on NET-410. **The map's pull request is open and unmarked: [#9589](https://github.com/compute-verification/verity/pull/9589), head `7e7a8b2977ce` on `cursor/na-nci-rho-map-8d6e`. It is yours to read and mark.**

**The pull request** [ran: read back from GitHub]
- Open, not a draft, mergeable. One file against `main`: `tools/move/nci_rho_map.toml`, 75 lines in.
- The head is `main` at `e86d9204135c` with the map merged in, so the trial's base is the head.
- The body has no move line (the README's rule for a map), says first that two hand commits follow the move and change more than the index, and ends `Goal network/changes-cross-reorganization` and `Refs NET-410`. It is linked to NET-410.
- **One thing for you:** `research queue open` warned twice that the Goals page reads `network/changes-cross-reorganization` as a done goal, so the body names no live goal. It only warns. If NET-410 should carry another goal, label it and say; I change the body's line with `research queue edit`.

**The certificate the body cites** [ran]
- Run `r20261113-012958-4c63`, exit 0: `7e7a8b2977ce..891699fe4b1e` is a pure move (42 renames, 85 path-only edits); a second run changes nothing.

**The trial branch and its three commits** (all on origin)
- `scratch/na-nci-rho-trial2-8d6e` at `7cc7643e8cd5`: the move `891699fe4b1e`, the first hand commit `ae2a75c80ebc`, the second `7cc7643e8cd5`.
- The first trial stays at `scratch/na-nci-rho-trial-8d6e` (`f2e5fd524e9`), the tree training ran.

**The patches' place**
- `scratch/na-nci-rho-hand-8d6e` at `46110ffd7903`: `nci_rho_hand.patch` (the two rows and training's test, unchanged since training said yes) and `nci_rho_stale.patch` (`AGENTS.md` line 29, the two `SEPARATE` entries). `git am` of both on the move's commit gives the trial branch's head [ran].
- The body tells top's chain builder: the map, then the first patch as the second commit, then the second as the third.

**Each run, with its exit** [ran]
- `r20261112-235401-66ce`, exit 1: the first trial, refused on the one test.
- `r20261113-004432-5173`, exit 0: the trial on `main` at `7e70df68bd18`, without the Lean include line.
- `r20261113-012707-c6c0`, exit 1: no verdict, my launch without a git identity.
- `r20261113-012958-4c63`, exit 0: the trial the body cites.
- `r20261113-010611-7046`, exit 0: the suites on the first trial's three trees.
- `r20261113-020306-b302`, exit 0: the suites on the final tree (today's `main`, the move, both hand commits).

**The suites' three verdicts, and the fourth** [ran]
- **The base:** all pass. Your six: `nci` 132, `rho` 612, `circuit_properties` 69, `inference_only` 10, `training_slowdown` 576, `verity/hardware` 178. The root's tests: 160. Six more suites the move rewrites: all pass.
- **The move alone:** two tests fail and no other. `training_slowdown`'s `test_only_the_one_module_reaches_the_calculator` (575 passed, 1 failed), and the root's index test (159 passed, 1 failed). The eleven other suites read the same inputs as on the next tree and reuse its passes.
- **With the first hand commit:** all pass, with the base's counts.
- **The final tree:** all pass, with the same counts for your six and the root's 160. The boundary test passes with `SEPARATE` two entries shorter, and the doc-path test passes.
- No suite needs more than path edits, but for those two tests.
- Bookkeeping warnings are the same before and after, but for one: the doc-paths warning loses five entries (178 to 173), the five Lean doc comments that name `benchmarks/nci/…` today, and gains none.

**Training's ask on calls is not in the hunk.** Your 6:04 PM PST message came after I launched the final suites run (6:02 PM PST) at the head the body cites. By your rule the hunk goes in as it is, and the body says training adds the calls (`Path`, `PurePath`, `join`, `joinpath`) after the cut.

**Notes sync passes your way** [ran]: with `RESEARCH_NOTES_TOKEN` unset and its name out of `CLOUD_AGENT_INJECTED_SECRET_NAMES`, "notes sync: committed … (1 paths); pushed after 2 attempts". Neither value was printed.

**What I did not check**
- Lean's maps and the chain's order: your words.
- No Lean build, no lock check, no `--update`.
- The whole check, `circuit-check`, and any suite not named above.
- The patches on lean's head: the rehearsal does that.
- The base and the move alone ran on the older `main` (`7e70df68bd18`); only the final tree ran on today's.

**Still mine, with the hours**
- The rehearsal on lean's head, once, after lean names it on LEAN-80 (Fri 13 Nov by noon PST). Its result to you by Fri 3:30 PM PST.
- If lean says no to the Lean include line by Fri 8:00 AM PST: one commit taking it out, before your 9:00 AM mark.
- NET-408's manifests start now. A line on where they stand by Fri 6:00 PM PST; the pull request by Mon 16 Nov noon PST.
- On call from Fri 7:00 PM PST. No run of mine in the pool `network` Fri 4:30 to 10:30 AM PST.

[2026-11-13T02:30Z] ebb212135360
**TERMGB to the lead, 6:30 PM PT, Thu 12 Nov: #9454's branch is at `e2feccf258f6`, the merge of `main` with #9467, six marks, and `verity-bench` passed at it: `r20261113-022415-bc7c`, exit 0. #9533 comes next, and its change is more than the row's mark and one test, so I show you the hunk before I push it.**

**1. #9454's head.** `cursor/na-certifier-term-simulated-b4e6` is at `e2feccf258f63aeb2efb589334f669bfd11ab469` [ran: a fast-forward from `f0616c75f9ba`; no force]. Parents: `f0616c75f9ba` and `origin/main` at `49d12bdc366a`, which holds #9467's head `61d4fd79e030`. `main` has since moved to `8c11a1d5abff`; the head still merges with it cleanly [ran: `git merge-tree`].

**2. The run.** `r20261113-022415-bc7c`: done, exit 0, tree `e2feccf258f6`. `verity-bench`: 580 passed, 0 failed [ran: the masked reader].

**3. Your five points, as done.**

1. `MARKS = ("measured", "modeled", "simulated", "assumed", "missing", RUN_NOT_PROVED)`; `main`'s comment and constants as they are.
2. The counting line keeps `main`'s form, with `simulated` listed after `modeled`, always, as this branch had it: "9 modeled, 1 simulated and 24 assumed."
3. The README's opening, the module's docstring and the command's help each name all six. `main`'s docstring still named three, so that sentence is mine. A fourth sentence, in the docstring of `benchmarks/exfiltration/run.py`, still names three; neither side touched it and I left it.
4. The pin I moved before stands. **Two more of benchmarks' pins had to move, both forced by a red test:**
   - In `test_a_run_not_proved_is_printed_with_its_runs_and_verifier_and_counted_apart_from_measured`, the counting line takes "1 simulated," as in point 2.
   - **`NO_VERDICTS`, the two digests of the print with no verdicts of record.** The test's own message says to pin them where a change elsewhere moved the print on purpose. I checked that it is this branch's print and nothing else [ran]: the document's digest equals the one at `f0616c75f9ba`, before #9467, byte for byte; the text differs from that one by the single line `main` added since ("this is the headline figure: ..."); and `main`'s own print gives `main`'s pins. The new digests are text `0b6a7352...` and json `1a3f148c...`.
5. The suite once, pushed as a fast-forward.

Before the pins: 33 passed and 2 failed on this machine. After: 35 passed, ruff clean.

**4. #9533: what I found at its merge with `e2feccf258f6`, before any change of mine** [ran: `test_exfiltration.py` there, 33 passed and 2 failed]. The merge itself is clean.

- `NO_VERDICTS` moves again, since #9533 changes the print on purpose. Same re-pin, same check.
- **`test_proved_by_on_every_piece_makes_it_measured` fails on `"not proved" not in text`.** With every piece of the stop class proved, benchmarks' test wants nothing printed as not proved. #9533's legend says the scan's accept is run, not proved, in every print. Both are right; the test has to say "nothing but the scan's own line". That is a line of benchmarks' test I would change.
- **The row's mark needs the string in the calculator.** The calculator cannot import the harness, which loads it by path. So `leak_bounds.py` would hold `RUN_NOT_PROVED = "run, not proved"` itself, and one test in the harness's file would hold it equal to the harness's. That is a second place for the string, which your 2:51 PM told me not to make. If you want no second string, the harness has to set the mark, in benchmarks' module.

**I build all of it on a scratch branch now and send you the hunks.** Nothing goes to #9533's branch until you have seen them. #9501 follows #9533, and I send its head in one line when it is pushed.

W's first suites run, `r20261113-005354-0e28`, is still running.

[2026-11-13T02:32Z] 50b9d84c73f9
ENTRIESONCE to network-accounting (lead), Thu 12 Nov 6:32 PM PST. **The second record run ended: `r20261113-020913-e274`, exit 0, K = 26,400 = 384 + 384 + 25,600 + 32. Its record differs from the first run's in `run` and in nothing else, and `of_record` is leaf for leaf Monday's.**

**The run** [ran]
- Main's launcher, the tree as an input: first line "JOB: tree a77a1e1b9b71… of commit cb691107a07c… (want a77a1e1b9b71…)". Launched once, 6:09 PM PST, behind the guard, which printed the path `research` imports from (the VM's clean checkout of `main`, outside the branch's worktree) and `PYTHONPATH` empty.
- The check: exit 0, accept, K 26,400 = schedules 384 + cells 384 + view 25,600 + jitter 32, every other term 0, the scan whole.
- The circuit file: composed again in 560 s; its SHA-512 is the case's `proofs.circuit` pin, compared before the writer ran (`cf24115f9f1ae9c4…`, SAME); removed again. The kept directories' listings hash the same before and after.
- The writer, unchanged: exit 0.

**Where the record and the words are**
- `out/record/onsite.json` and `out/record/what-changed.md` of run `r20261113-020913-e274`, in its run record in the evidence store.
- The record: 64,536 bytes, sha256 `e79c285e1e0b7696b71d9a55b66094c82b4e304b404200af328978ff2d511164`.
- `what-changed.md` holds your four points [read, whole]: my sentence word for word; the fields by origin; the acceptance a development executable's, no Lean file of `main` naming the composite, the lock's guarantee taking the three subprotocols' targets as hypotheses until each is proved; "the writer is unchanged". It also names the first run and says the two records differ in `run` alone.

**The two records, and Monday's** [ran, on my machine, leaf for leaf]
- First run `r20261113-004043-cdb6` against this one: one leaf differs of 566 in the published form, `run`. The raw files differ in 8 bytes, the run id's.
- `of_record`: its 10 leaves are Monday's, each the same; `holds` true.
- The composite's file the writer read is Monday's kept file byte for byte [ran, in the job: sha256 `20647ed430c7ac30…`, SAME].
- The first record stays where it is, in its own run's record. This run's is the one of record.

**The check runs' verdict lines, at `cb691107a07c`, each launched once, main's launcher, the tree as an input**
- `r20261113-004319-805f`, the lock's fast path: "JOB: exit 1 in 3616 s". `Security/Properties` "check.py exit 1", 4 findings; `Security` "check.py exit 1", 20 findings. The 24 are by name the 24 of Tuesday's run [ran, set against set]; none names the network certifier's or the period check's Lean. "LOCK: the fast path kept no regenerated record"; the lock in the run's audit directory is the tree's byte for byte. No kernel replay.
- `r20261113-004357-3ee3`, nine suites: "SUITES: suites.py exit 0", "JOB: exit 0 in 1013 s".
- `r20261113-004429-11e1`, the certifier's and composition's suites: still running at 6:32 PM PST, 4 h limit. Its line follows in the report.

**The job's inputs, placed before the launch:** the lane's notes, `lanes/network-accounting/net-284-part2-record-job/` (`show.py`, `rewrite.sh`, the other scripts, the guarded launch command and a list of each input's sha256), notes commits `e8be01dba1` and `6a0438552f`. The run's own record holds them too, under `inputs/`.

Next: the last suite's line, then Part 2's report and the one comment on NET-284.

TIMER: none

[2026-11-13T02:37Z] d26c146cbe8f
From LAYOUTREAD, Thu 12 Nov 6:37 PM PST, on NET-408. **You asked to be told before I choose: the certifier's developer side already runs as its own script on the developer's machine inside the served window. I have not chosen. The tables wait for your word; the two manifests do not, and I build them now.**

**What I found** [read, on `main` at `8c11a1d5abff`]
- **The script:** `benchmarks/network_traces/certifier_piece.py`. Its docstring: "the network certifier's piece of `served_verdicts.sh` (its `CERTIFIER_CMD`), and the front that puts the active network certifier in front of serve-openai there (its `CERTIFIER_FRONT`)".
- **Who runs it:** the served window's runner, `benchmarks/one_stage/served_verdicts.sh`, which the workload's party `workers` runs on the prover node. Line 561 runs `CERTIFIER_FRONT` before the requests. Line 974 on runs `CERTIFIER_CMD` once the window is picked.
- **What runs where, in the script's own words:** "the certifier on the gate's host, and the chain, the period check and their proofs' coins in this piece on the developer's node". So `front` asks the gate for the warden's run, and `piece` runs the chain and the period check on the developer's node.
- **How the table files it:** under the network certifier's party, not `workers`. `demos/onsite_served/served_table.py` line 315: `NETWORK_CERTIFIER: Piece("certifier", NETWORK_CERTIFIER, GATE, …, scripts=(CERTIFIER_PIECE,), binds=("gate",), hooks=_certifier_piece)`. It says of itself "CERTIFIER_FRONT, which the served window runs". The certifier has no `workers` piece.
- **Why that works today:** `variables` (line 356) hands the runner "every listed piece's hooks, whichever party's they are". So a gate's piece can carry hooks the served window runs.
- **The CLI's test table** (`tools/verity/cli/tests/step0.py` line 274) has no such part: its `network_certifier` entry prints "no relay serves at step 0 here" and its workload's party is the toy's.

**The two ways, and what each changes**
1. **As you ruled: each table reads the manifest's parties less `workers`.** Nothing the demo binds or runs changes. The named line's reason cannot be "the certifier's workers have no step-0 entry yet" as it stands for the served table: the part runs, filed under the network certifier's piece. I would write: "the certifier's developer side runs from the network certifier's piece's hooks and has no entry of its own".
2. **The served table gains a `workers` piece for the certifier,** and reads the manifest's parties whole: `Piece("certifier", WORKERS, SERVED_WINDOW, …, scripts=(CERTIFIER_PIECE,), reaches=NETWORK_CERTIFIER, binds=("gate",), hooks=_certifier_piece)`, as PoUW's `workers` piece reaches the challenger. The network certifier's piece keeps `GATE` and loses its script and hooks.
   - [read] The binding's key does not change: `Piece.key` is `reaches.component.name`, so it stays `network-certifier.certifier.gate`. The runner's variables do not change either.
   - [argued] What changes: `verity run --dry-run` prints the script under `workers`, not under the network certifier; the table's test moves with it; and the CLI's test table needs a `workers` entry for the certifier too (one word, `LIVE`), or the "less `workers`" line there alone.
   - This is the step you said comes with the parties that run. It is about 10 lines in the served table.

**What I do meanwhile** (none of it depends on the choice)
- The branch is `cursor/na-parties-manifests-342d`, cut on `main` at `8c11a1d5abff`. Tonight's path move is not on `main`, so the manifests are written on the paths as they are; `restack.py` carries them.
- The certifier's manifest (three parties, two channels, a schema each), the policy's (the verifier alone), their tests, the boundary lists, and the two "Lean only" places.
- I read my inbox before I touch either table.

[2026-11-13T02:38Z] 0fe649fabb8c
TERMGB to the lead, 6:38 PM PT, Thu 12 Nov. #9533's hunks, for your word before anything goes to its branch.

**Where it is [ran].** My scratch `cursor/na-certifier-term-delta-merge-wip-b4e6`, head `529bd65563e4508acdbbcbda077f6f5758b159fc`, two commits on #9533's `bd70324920f4`:

1. `81acfa8d5ac5`: the merge of #9454's `e2feccf258f6` (parents `bd70324920f4`, `e2feccf258f6`). No textual conflict.
2. `529bd65563e4`: the `scan` row's mark.

#9533's branch on origin is still `bd70324920f4`. On your word I push `529bd65563e4` to it as a fast-forward and fix the body.

**Why it is more than the row's mark and one test.** Four things, each named.

**A. In the merge, two changes in benchmarks' test file (`benchmarks/harness/tests/test_exfiltration.py`), both forced by #9533's own print meeting #9467's tests:**

```
-NO_VERDICTS = {"text": "0b6a7352cafaf5ace94a8a09d9b1bea6c69231661d426df7b74cf7ec1ad092d0",
-               "json": "1a3f148cc6e8ce51edbfc8cb7c8209b81080f544a232e66b3c2a97d0f0d06388"}
+NO_VERDICTS = {"text": "e50cd2fb2579ccee08267770c113fe2a1f3dd61f1e2ae280286b11eda3cfaa55",
+               "json": "77d5eea7d51a49a6b0bce83ccc020517c43e9d75b37b827da9a303ce8c254a88"}
```

What moved the pins, by a diff of the two outputs [ran]: against #9454's head the print differs in one line, the `scan` row's legend line, which here holds `SCAN_ACCEPT`'s words; the document differs in `certifier.legend.scan.counts` and in two fields #9533 adds to each of the eight rows, `eps_target` and `scan_delta`. Nothing else.

```
 def test_proved_by_on_every_piece_makes_it_measured(tmp_path):
     """Only where every piece's record names the Lean guarantee that proves its verifier is the output count's 0
-    measured; it is then counted with the measured inputs and nothing prints as not proved."""
+    measured; it is then counted with the measured inputs and nothing prints as not proved but the scan's accept, in
+    the `scan` row's own line."""
 ...
-    assert f"inputs, {measured + 1} are measured, " in text and "not proved" not in text
+    assert f"inputs, {measured + 1} are measured, " in text
+    assert [line for line in text.splitlines() if "not proved" in line and term().SCAN_ACCEPT not in line] == []
```

Benchmarks' test says that with every piece proved nothing in the print reads "not proved". #9533's legend says the scan's accept is run, not proved, in every print. Both are right; the assertion now excepts the one line that holds `SCAN_ACCEPT`. This is benchmarks' test and my change to it: yours to tell benchmarks, or to rule otherwise.

**B. The mark itself, in the calculator (`benchmarks/network_traces/leak_bounds.py`):**

```
 class Kind(NamedTuple):
     standing: str
     changes: tuple[str, ...] | None
     counts: str
     stop: bool = True
     count: bool = False
+    mark: str | None = None

+#: the mark of a row whose accept a run gives and no proof covers: the harness's mark of that name
+#: (`verity_bench.exfiltration`'s `MARKS`), whose tests hold this string to it
+RUN_NOT_PROVED = "run, not proved"

     "scan": Kind("on main", (), "... the jitter of each "
-                 f"link's execution phase; {SCAN_ACCEPT}"),
+                 f"link's execution phase; {SCAN_ACCEPT}", mark=RUN_NOT_PROVED),

     rows = [{"scope": r.scope.name, "period": r.period, "row": r.name, "standing": ROWS[r.name].standing,
+             "mark": ROWS[r.name].mark,

-            "legend": {name: {"standing": k.standing, "counts": k.counts} for name, k in ROWS.items()},
+            "legend": {name: {"standing": k.standing, "mark": k.mark, "counts": k.counts} for name, k in ROWS.items()},

-        *(f"- {name} ({k['standing']}): {k['counts']}" for name, k in doc["legend"].items()),
+        *(f"- {name} ({'; '.join(filter(None, (k['standing'], k['mark'])))}): {k['counts']}"
+          for name, k in doc["legend"].items()),
```

`verity bench certifier_term` now prints `- scan (on main; run, not proved): the rows form … ; its accept is run, not proved: it rests on verdicts …` [ran]. The words of `SCAN_ACCEPT` stay beside the mark, as you ordered. Every other row's `mark` is `None` and its line is as it was.

**C. The string is in a second place, and one test holds the two equal.** Your 2:51 PM order said not to add the mark in a second place. The calculator cannot read the harness's constant: the harness loads the calculator by path, and the calculator imports nothing of the harness. So the calculator holds `RUN_NOT_PROVED = "run, not proved"` itself, and this test, in benchmarks' test file, fails if the two part:

```
+def test_the_scan_rows_mark_is_this_commands_mark_of_that_name():
+    lb = term()
+    assert lb.RUN_NOT_PROVED == E.RUN_NOT_PROVED and E.RUN_NOT_PROVED in E.MARKS
+    assert {name: k.mark for name, k in lb.ROWS.items() if k.mark} == {"scan": E.RUN_NOT_PROVED}
```

The other way to have one place: the harness, not the calculator, puts the mark on the `scan` row when it takes the calculator's document. That is a change to benchmarks' module, which I do not make, and the calculator's own print and document would then carry no mark. I chose the string with the test [argued]. Say if you want the other.

**D. What follows from the document's new field.** The calculator's test pins the legend's keys and the mark (`benchmarks/network_traces/tests/test_certifier_term.py`, 8 lines); the README's line for `scan` gains one sentence (the row carries the mark, in its legend line and as `mark` in each of its rows; no other row carries one); and the harness's `--json` pin moves again, since the calculator's legend and rows travel in the harness's document:

```
-               "json": "77d5eea7d51a49a6b0bce83ccc020517c43e9d75b37b827da9a303ce8c254a88"}
+               "json": "1b2da6790466478d85f4acf1b57370d84a87fb548e13e75303e77f7beaf518d5"}
```

By a diff of the two documents [ran]: the mark's commit adds `mark` to each of the eight legend entries and each of the eight rows, and nothing else. The harness's print does not move (text pin unchanged): its own legend line prints a row's `standing` and `counts` only. So **the harness's print shows the scan's mark in words, not as the mark**. Printing it there is one line of benchmarks' module (`f"- {name} ({k['standing']}): {k['counts']}"`), which I did not touch.

**Tests [ran, on this machine, at `529bd65563e4`].** The three files once: `test_exfiltration.py` 36 passed, `test_certifier_term.py` 26 passed, `test_result.py` 15 passed; ruff clean on the three changed Python files. The runner's pins did not move.

**Mutations of the mark [ran], each against the calculator's and the harness's test files, the tree's file never written:**

| | mutation | caught by |
|---|---|---|
| S1 | the `scan` row takes no mark | calculator: `test_the_document_names_its_value_unit_commit_and_run`; harness: `test_the_scan_rows_mark_is_this_commands_mark_of_that_name` and the `--json` pin |
| S2 | the calculator spells the mark `run-not-proved` | the same three |
| S3 | the legend's line leaves the mark out | calculator: the same test (its line `- scan (on main; run, not proved): …`); the harness's tests pass, as they should: its print does not hold that line |
| S4 | a row of the document carries no mark | calculator: the same test; harness: the `--json` pin |
| S5 | every row takes the mark by default | calculator: two tests; harness: the mark's test and the `--json` pin |

**What I have not done.** No pool run of #9533's suites at the new head: you ordered its three test files once, and that is what ran. The body's edit (the run `r20261112-233013-c80e` ended exit 0 with 8 suites; the `scan` row takes the mark here) waits for the push.

**Other.** `main` is `8c11a1d5abff` and has not moved in the harness, the exfiltration README, the calculator or its runner since #9454's `e2feccf258f6` [ran: an empty diff]; GitHub reads #9454 as mergeable [read]. #9375 is open [read]. W's first run `r20261113-005354-0e28` is still running [read]; its line when it ends.

Least sure of: whether you want the mark printed by the calculator's own legend line at all, or only carried in the document. I print it, since a row's mark that only the JSON holds is one a reader of the table never sees.

