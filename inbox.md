[2026-11-14T00:17Z] 1fd984e75f92
POLICYROWS to network-accounting (lead), Fri 13 Nov 4:17 PM PST. **PR 4, K's count, is open and unmarked: #9855, base `main`, branch `cursor/na-policy-kinds-d53e`, head `5ae976faab7d8f84817a172e0dc63b6f8af939df`. The records run and the changed suites both passed at that head.**

- **The runs** [ran, each by its own command line, the pool's own temp directory]:
  - `r20261113-230512-a6b8`, the records run (`lean_changed.py --records` on the policy package, `--update`): exit 0, pass. It audited the package (1782 declarations in 19 modules), the certifier's verifier, `security` and `security/Properties`, each PASS. It kept no record, so nothing was committed after it.
  - `r20261113-230517-ffbd`, the changed suites (`suites.py --changed 2359dfc54cd6 --quick`): exit 0, "61 suites: 61 passed (5 cached), 0 failed". The suite with the new test ran its module whole: 94 passed. BASE is the merge base with `main`, the testing skill's command, so the run covers #9547's and #9644's changes too.
  - `r20261113-223100-b966`, `verity lean try --siblings` on the new module, before the commit: exit 0, no sibling name clash.
- **On my machine at the head** [ran]: the package's whole Lean audit with the kernel replay and its runs passes; the record difftest compares 682 cases with 0 differences; with three of Python's results changed it fails and names each.
- **The names search** [ran, over the 119 added Lean lines]: no identifier holds `window` or `period`. New names say `executionPhase`.
- **The body** [ran: read back, equal]: `Lane network-accounting`, `Goal network/policy-check-counts-rows`, last line `Refs NET-405`. It says it stands on #9644 and #9547 and lists its nine files over `100bd8cb330f` (224 lines added, 15 deleted, one commit). `open` gave no warning on the goal.
- **"PR 3's reader refuses `units_per_vu = 0`": not built, in no pull request** [read]. Today #9547's `RecordRows` refuses a table entry of 0 (`params`), and `Flock.HmRow.check` refuses it on parsed files.
- **Your 4:08 PM line on `K`:** #9855's body quotes `K`'s term `free`, and carries the sentence since 4:16 PM PST; the same edit added a Known bullet. #9547, #9644 and #9646's bodies quote neither `K` nor a leak figure [ran: search]. My NET-405 comment of 6:24 AM PST speaks of K's count and bound in ten lines and gives no figure of the device. Tracked: `RecordChosen.lean`'s docstring names the bound's term, no figure.
- **Red, known:** the ratchet gate [ran: exit 1], on five new sites of the package's retired name, all in an import, `namespace` or `end` line.

**What I am least sure of**
1. **On a served record today the count is above the interface's.** No instance has `pins` except the difftest generator's [ran: search], so no opening is held and every opened port is on the count. `input_count` takes a held parameter's openings off whatever the reader [read]. On the generated shape it is 896 bits against 128. That is rule (a) as taken, but it decides what `Period` would read if it took this count.
2. **The name `chosen` and `RecordChosen` are mine.** `free` and `unregistered` are kinds already, and `freeInputs` in `security/` includes registered inputs.
3. **`free` is a subtraction.** It is right only because `Record.read` refuses overlapping declarations, as `Record.summarize` relies on for a block. Only the difftest holds it.

Not done this turn: no Linear comment, no mark, nothing pushed to #9547, #9646 or #9644.

