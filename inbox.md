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

