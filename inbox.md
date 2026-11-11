[2026-11-11T18:06Z] 1e0e3cb60f89
LEANMERGE to the network-accounting coordinator, Wed 11 Nov, 10:05 AM PST. The recut by lean's rule: the closes alone, then the lines beyond the hunks.

**Branch `cursor/na-cross-closes-period-9b2e`, cut from `c094f1ef8a0f`, two commits.**

- **A, the closes only: `0eb2a07a2cd0c2f42007cbbbd6916756e9e4e320`.** e3b0f6c6e8af's tree with the seven changes outside the hunks put back as `c094f1ef8a0f` has them. Changes outside hunks: 0, in all five files.
- **B, on A: `eb361a9b364c310ae42ffeb6bfabcf03e02fecb1`.** Exactly those seven changes again. Its tree is e3b0f6c6e8af's (`git diff e3b0f6c6e8af B` prints nothing; both trees are `c06830594dfd`).

`cursor/na-cross-closes-period-4d1a` and `-7c3e` stay as they are, superseded. No amend of a pushed commit, no force push. Marks: [vm] a command on my VM, [run] a remote run's record, [read], [argued].

## One closed line that only makes sense with a fix (kept as it is)

`benchmarks/network_traces/scan_zk.py`, the hunk at `main` 948 (marker 1145), `program`'s docstring, A's lines 1145-1148: "its ``prog<j>``, one a lane, the values it names tied (``tied``) … so R8 lets them through and the registration verdict publishes them". At A the code under it (A's line 1180, `c094f1ef8a0f`'s 1187) still says `tied=("prog",)`. So at A this sentence says what B's code does, not A's. I did not bend it: `main`'s wording ("its ``prog`` the one value it names tied") names an input the merged scan root does not have, so no wording of this close is true of A's code; the cure is the code line, which is B's. No other closed line names `lane_kinds`, `_strata_resize`, the seed or "lane 0's slot entries'". [vm: `rg` over the lines A adds]

## The checks, each with its output [vm]

1. **Markers at A**, `rg -n '^(<<<<<<<|\|\|\|\|\|\|\||=======|>>>>>>>)'` on the five files: no match (rg exit 1).
2. **`python3 -m py_compile` at A**: ok for `scan_zk.py`, `period_case.py`, `test_network_period_protocol.py`.
3. **`ruff check --isolated --select E9,F63,F7,F82` at A** (ruff 0.17.2): "All checks passed!", exit 0.
4. **`git diff --stat A e3b0f6c6e8af -- '*.lean' '*.md'`**: empty (0 lines). `Period.lean` at A has sha256 `8a6fd6583bcc…`, the one run `r20261111-172327-a08c` printed before its build; A against `b30915af6b62`, the commit that run built, differs in no `.lean` or `.md` file. So a08c's build ("Build completed successfully (67 jobs)") stands for A.
5. **Changes outside hunks at A: 0.** How: my own script (`outside.py`, in the notes), two readings of each file that must agree. First, every hunk of `git diff -U0 c094f1ef8a0f A` whose old lines are not inside a conflict region of `c094f1ef8a0f` (from `<<<<<<<` to `>>>>>>>`). Second, with no diff: `c094f1ef8a0f`'s lines outside its regions, run by run, must stand whole and in order in A's file, the first at its head and the last at its tail. Output at A:
   - `scan_zk.py`: 1 region, changes outside hunks 0, context runs whole and in order: yes
   - `README.md`: 1 region, 0, yes
   - `Period.lean`: 2 regions, 0, yes
   - `period_case.py`: 7 regions, 0, yes
   - `test_network_period_protocol.py`: 2 regions, 0, yes
   - "changes outside hunks, all files: 0"
   Control: the same script on B's tree finds your list and nothing else (`scan_zk.py` 89-93 and 1187; `period_case.py` 1660, 1662-1665, 1978, 2180-2181; the test file 464-466; 7 in all), and the second reading says NO for those three files.
6. **`git diff --stat c094f1ef8a0f A`**: 5 files, 51 insertions, 192 deletions: `scan_zk.py` 19, `README.md` 16, `Period.lean` 66, `period_case.py` 107, `test_network_period_protocol.py` 35.
7. **`git diff --stat c094f1ef8a0f B`**: the same 5 files, 69 insertions, 209 deletions: 31, 16, 66, 124, 41.
8. **`git diff e3b0f6c6e8af B | wc -c`**: 0.
9. Parents: A's is `c094f1ef8a0f2f39f030f797afffead992b4db48`, B's is A. `git diff --stat A B`: 3 files, 18 insertions, 17 deletions (`scan_zk.py` 12, `period_case.py` 17, the test file 6).

The whole output is in `na-cross-closes-period-recut-checks.log` (my artifacts).

## Known wrong at A, cured at B

- **The coverage assertion of `test_a_profile_that_is_no_sample_of_its_level_is_refused` fails at A and passes at B.** Is run 180f's tree A's `period_case.py` and test file? **No, not byte for byte; yes as code.** 180f's three closed files were `b30915af6b62`'s. Its `Period.lean` is A's file. Its `period_case.py` and test file differ from A's in two docstring sentences only (the "lane 0's" words: 2 lines and 3 lines); with docstrings removed the parsed trees are equal. [vm: `samecode.py`] So 180f's verdict (28 passed, this one failed) is A's code's [run, argued for A from the equal code]. B's three files are byte for byte the ones run `r20261111-173610-3878` ran (they were e3b0f6c6e8af's, checked when I launched it): 32 passed. [run] Both runs were on the branch's side with the three closed files, not the merged tree.
- **`tied=("prog",)` in `scan_zk.py`** names an input the merged scan root does not have (its inputs are the lanes' `prog<j>`); B names `lane_kinds(cls)`. [read, not run: a scan round needs one-stage's `registration.py`, which holds markers]
- Words only: at A two docstrings say the `entries` negative puts the egress state on "the slot entries' input node"; with a node a lane it is lane 0's, which B says. [read]

## Not done

No new run, as you said. Nothing on Slack, Linear or a PR.

[2026-11-11T18:13Z] 1d3838a77a01
CROSSTICK to network-accounting, Wed 11 Nov 10:14 AM PST: the renames commit is pushed.

Branch `cursor/na-cross-renames-tick-a1a4`, ONE commit `ed6ed895a527d8514d80c8f998b5274d0d84fba0`, parent `fdb4b47c69d014dbc574a756538d701db44eda63`. 7 lines in 3 files, each a pure rename; none of the three holds a marker at c094f1ef8a0f. [ran]

LINES CHANGED (file:line, old -> new)
- verity/protocols/certifier/verifier/continuation.py:60   Record -> BlockGrid (the import from grid)
- verity/protocols/certifier/verifier/continuation.py:242  Record -> BlockGrid
- verity/protocols/certifier/verifier/continuation.py:261  Record -> BlockGrid, twice
- verity/protocols/certifier/verifier/continuation.py:300  Record -> BlockGrid, twice
- benchmarks/network_traces/tests/test_compile_slice_zk.py:80  Window -> Advice (the import from window_charge)
- benchmarks/network_traces/tests/test_compile_slice_zk.py:90  Window -> Advice
- benchmarks/network_e2e/encodings/tests/test_served_window.py:146  CV.Window(arrived=d) -> CV.Advice(arrived=d)
The merged Advice takes `arrived` (window_charge.py:154, default ()); lines 108 and 109 of the same test already say CV.Advice(arrived=d). [read]

STEP 2: nothing more to add to the commit. [ran]
How I searched, at c094f1ef8a0f, under verity/protocols/certifier, benchmarks/network_traces, benchmarks/network_e2e, benchmarks/datacenter:
(a) git grep of each old name: Window, Record, firewall, _draw_salt, advice_before_draw, fixed_before_draw, advice_after_draw (the pairs read off the branch's commits d29cc15517, 13d7eba072, dec4744e6a), and the names the branch deleted from those directories' modules.
(b) the name check over the 326 Python files there that parse: every `from M import N` and every attribute of a module alias (modules loaded by file too) against what the merged module declares, with main's own tree as the control. Before the renames it reports exactly the three sites above; with them, none.

FOUND AND LEFT
- continuation.py:44, a docstring, still says `Record`s. Prose, not an import or a call, and not on your list, so I left it. The same rename fits the line (117 characters, no reflow): say the word and it is one more line.
- Prose that names the old class, in no Python: verity/protocols/certifier/device/README.md:26 (`Record`), verity/protocols/certifier/verifier/PROTOCOL.md:1713 (`grid.Record`s), verity/protocols/certifier/verifier/README.md:281 (`Window`). All three are on the branch's own side too (c08d034460b0). [read]
- A Lean doc comment, verity/protocols/certifier/device/lean/CertifierDevice/Device.lean:138, says `_draw_salt`; Python declares `_fresh_salt`. Lean, not touched. [read]
- No other use is wrong: every `firewall` left in an unmarked file is warden_zk's own (still declared, warden_zk.py:321); no `_draw_salt` in Python; the branch's tip holds no `advice_before_sample`, `fixed_before_sample` or `advice_after_sample` anywhere, so the `_draw` names agree on both sides. [ran]
- 211 lookups cannot be decided yet: they read modules that still hold markers (scan_zk.py 57; proofs' registered.py 118, registration.py 34, profile.py 1; period_case.py 1). The check wants a rerun once those close. [ran]
- Not touched, as told: test_scan_zk_challenger.py, the adapter's advice_code/codec.py.
Limit of (b): it sees imports and module attributes, not methods, fields or keyword arguments; for those I have only the greps of (a). [argued]

CHECKS [ran]
- git diff --stat fdb4b47c69 HEAD: test_served_window.py 2, test_compile_slice_zk.py 4, continuation.py 8; "3 files changed, 7 insertions(+), 7 deletions(-)".
- Every changed line: old line with the name replaced equals the new line, and back (script): "every changed line differs only in the renamed name".
- py_compile: ok on each of the three.
- ruff 0.17.2 check --isolated --select E9,F63,F7,F82: "All checks passed!" (so does --select F).
- Name check on the three files: before, 76 names checked, 3 not declared (the three sites); with the renames, 0 not declared.

TESTS
- On the committed tree none can run. [ran] test_network_certifier_continuation.py, test_continuation_trace.py and test_served_window.py fail at collection, and test_compile_slice_zk.py's six tests fail at import: SyntaxError at the markers of proofs' spot/verifier/one_stage/registration.py and registered.py.
- In a second copy that is NOT the crossed tree (the 15 marked Python files filled with the branch's side; no module loaded from another checkout): [ran]
  - test_network_certifier_continuation.py 25 passed, test_continuation_trace.py 4 passed. With the three files as before the renames, both fail at collection: ImportError, cannot import name 'Record' from grid.
  - test_compile_slice_zk.py: 6 failed, all on a replicated fixture absent on my VM (arith_encode_vectors.json), the same six on main's own tree. Before the renames one of the six failed on the `Window` import instead.
  - test_served_window.py: 3 failed, 1 passed, 11 errors, all on a replicated fixture absent on my VM, the same on main's own tree. So no test reaches line 146 here. I evaluated its expression against the merged code_v1 instead: CV.Advice(arrived=d) builds and CV.encode gives lengths; CV.Window raises AttributeError.
- Not run: the whole check, a suite runner, any Lean.

Nothing on Slack, Linear or a PR. Pushed: this one new branch, and this message.

[2026-11-11T18:19Z] cd6ade7b760f
TERMGB to the network-accounting coordinator. NET-326, #8777. Wed 11 Nov, 10:20 AM PST.

**New head `6dadd19ca33288b8be80fb8838b212f8b2635ef1`, pushed as a fast-forward of `7d82236c6b50`. Text only: every number of every row is unchanged. `45a4` ended rc 0 and its lines are in the body. The text commit changes five suites' inputs, so I launched one run at the new head, `r20261111-180807-5e63`: rc 0, in the body too. No mark.**

## 1. The two rows' text

One commit, `6dadd19ca332`: three files, 27 lines added, 18 removed.

| row | before | after |
| --- | --- | --- |
| `lengths` | a response's length held to its Program's output count: the schedules are 0, and … | a response's length held to its Program's output count, **and that count held to its ids by a stop block, a proof block that checks where a response's ids stop, sampled on a served audit**: the schedules are 0, and … (the rest unchanged) |
| `all three` | the three changes together | the three changes together, **with the stop block the lengths' change rests on** |

**Nothing but those two texts differs.** [run]

- The `--json` documents at `7d82236c6b50` and at `6dadd19ca332`, at the four sets of inputs: the default; `--world-requests 1.778e16 --response-ids 1024`; each with `--failure-rate 1e-6`.
- In each of the four: 72 rows before and after, the same keys in the same order, all 72 equal field for field. Schedules, cells, jitter, term and entries are equal in all 72.
- The only fields that differ are the document's `commit` and the legend's `counts` text for `lengths` and `all three`. No standing differs.
- In the printed output the only lines that differ are the commit's line and those two legend lines.
- `CHANGES` is still three, with the same keys; no new row; the row order is the same. `lengths`' text is `CHANGES`' entry for it, so that entry's text is what changed.

**Tests.** [run]

- No test pinned either text, so no pin changed.
- I added one assertion to `test_only_the_lengths_change_reads_the_stop_block`: the rows whose text names the stop block are exactly the four that hold the lengths (`lengths`, `all three`, `design, no stop block`, `design`). A row that prints this 0 without naming the stop block now fails.
- Local at the head: `test_certifier_term.py` 17 passed, `test_leak_bounds.py` 9, `test_chain_record.py -k leak` 2, ruff clean, the nine root files 112 passed.

**Also changed, outside the documents.** The module's docstring and `benchmarks/certifier_term/README.md` say the same. In each the definition is given once, where it first uses the words, which is now the changes' bullet; so it left the design's bullet there.

**`timing` and `entries` do not presume it.** [run, by the test above] Neither holds the lengths, and `terms` reads the stop block only where they are held: with it or without it their terms are equal. Their schedules are the scan's, charged in full. I changed neither.

**Three things for you to decide. I changed none of them.**

1. **The legend now defines a stop block twice**: in `lengths`, where a reader first meets the words, and in `design, no stop block`. You said only the two texts differ, so I left the second. Dropping the clause there is a one-line change to a third row's text.
2. **One more line presumes the stop block without naming it.** The line above the tables: "With the lengths held and the timing taken once, the world's year meets the target at a failure rate of at most 4.316e-06" (`failure_rate_at_target`; the README has the same sentence). Without a stop block the schedules alone are 28.9 million GB, so no failure rate meets 300 GB.
3. **Benchmarks' own wording in #8779 has the same gap.** [read] `TOGETHER["all three"]` in `benchmarks/harness/verity_bench/exfiltration.py` (lines 82 to 84) reads "with the lengths held, the timing taken once and the entries charged through the compute audit". That file is benchmarks'.

## 2. The runs

**`r20261111-173407-45a4` at `7d82236c6b50`: done, rc 0, 909 s.** [run] Its commit and tree are the head's as it then was, not dirty. It ran six suites by name, then the nine root files. Its verdict lines as printed:

```
passed  verity-network-traces        537 passed in 203 s; 537 outcomes recorded
passed  benchmarks/one_stage         507 passed in 81 s; 507 outcomes recorded
passed  verity-network-e2e           70 passed in 11 s; 70 outcomes recorded
passed  verity-graph                 207 passed in 19 s; 207 outcomes recorded
passed  verity-network-sim           21 passed in 2 s; 21 outcomes recorded
passed  research                     4236 passed, 3 skipped in 786 s on 2 workers; 4236 per-test passes kept
6 suites: 6 passed (0 cached), 0 failed; 18.4 min of test time
112 passed, 9 warnings in 105.23s (0:01:45)
```

- [read] All 17 tests of `test_certifier_term.py` are among the 537, the row's two new ones included. No suite read a file outside its declared inputs.
- [read] The nine warnings are bookkeeping and none names a file of this change. One is new since `4bf6`, `test_no_new_synonym_flags`, on a file under `tools/research/` that is `main`'s.

**`r20261111-180807-5e63` at `6dadd19ca332`: done, rc 0, 430 s.** [run] One run, `--on auto`, no node named. Its commit and tree are the new head's, not dirty.

- Why a run: `suites.py --changed 7d82236c6b50 --list` names six suites whose inputs the text commit changes: `verity-network-traces`, `verity-network-e2e`, `verity-network-sim`, `benchmarks/one_stage`, `verity-graph` and the root's. `research` is not among them.
- It ran those five by name, then the same nine root files for the root's. Its verdict lines as printed:

```
passed  verity-graph                 207 passed in 25 s; 207 outcomes recorded
passed  verity-network-e2e           70 passed in 57 s; 70 outcomes recorded
passed  verity-network-sim           21 passed in 2 s; 21 outcomes recorded
passed  benchmarks/one_stage         507 passed in 142 s; 507 outcomes recorded
passed  verity-network-traces        537 passed in 203 s; 537 outcomes recorded
5 suites: 5 passed (0 cached), 0 failed; 7.1 min of test time
112 passed, 9 warnings in 129.43s (0:02:09)
```

- [read] All 17 tests of `test_certifier_term.py` are among the 537. No suite read a file outside its declared inputs. The nine warnings are the same nine tests' as in `45a4`.

## 3. The PR

- [run] Unmarked, no label, head `6dadd19ca332`. Title unchanged: it names the two changes; the text is in the body.
- [run] Body replaced, 66 lines added and 13 removed. The changed lines:
  - The opening: "Two changes …, and one change of text", a third item for the two rows, and the diff's size (five files, 326 added, 89 removed).
  - "What changes": one bullet, every row whose schedules' 0 rests on the stop block says so.
  - "What the row does not show": the bullet that said `lengths`' text does not name the stop block is replaced by the failure-rate line of item 2 above.
  - A new section, "The rows `lengths` and `all three` name the stop block": both texts before and after, the comparison, the test, the definition given twice, `timing` and `entries`.
  - "What it waits on": the stat against `main` at the new head, and one line for the text commit.
  - "Checks": a block for the head `6dadd19ca332` with `5e63`'s verdict lines, and `45a4`'s verdict lines under `7d82236c6b50` in place of "under way".
  - The scratch merge with #8779, redone at the new head.

## 4. Against #8779, at the new head

- [run] Scratch, nothing pushed, worktrees removed. #8779's `9b5d80511bdf` into `main`, and that into `6dadd19ca332`: both clean. `test_exfiltration.py` 10 passed, 1 failed, the same assertion at line 369.
- [read] No file of #8779's pins either row's text. Its report prints the legend, so it prints the two new texts with no change of benchmarks'.

## 5. One line for benchmarks

> `design, no stop block`: the design with one check missing. Nothing proves where a response's ids stop, so the server picks its length, and each connection is charged for that.

That is 30 words, the row's name included.

## Least sure of, and not looked at

- Whether the definition belongs twice in the legend (item 1 above).
- `entries` hands its term to the compute audit's own bound. Whether that bound presumes a stop block is outside this calculator. I did not look.
- The row's charge is still the scan's rule, from your brief; I did not argue it is tight.

WAITS ON: #8779 on `main` (the assertion at line 369), and your review. No run is under way.

