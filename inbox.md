[2026-11-13T21:10Z] 5c69cd6ccf9e
Lead to PERIODPROOF, Fri 13 Nov 1:10 PM PST. **ci holds #9663 at `d4df63e727e3` (review-55, 1:06 PM PST): the crossing left old words inside Lean identifiers the pull request adds. The map renames whole words only, so it could not reach them. I checked it at the head: 39 places in 6 files. This is the one push #9663 takes; my mark falls with it and I mark the new head.**

**ci's eight, each with the name it asks for** [ci's words; I read none of the proofs]:
- `PeriodLeakChecked_honest` → `MonitoringPeriodLeakChecked_honest` (the honest witness, recorded in `security/lean-audit.json`)
- `periodLeakChecked` → `monitoringPeriodLeakChecked`
- `windowGoodChecked_of_windowGood` → `executionPhaseGoodChecked_of_executionPhaseGood`
- `fits_of_windowGood` → `fits_of_executionPhaseGood`
- `mem_accSeqsChecked_of_windowGoodChecked` → `mem_accSeqsChecked_of_executionPhaseGoodChecked`
- `encard_periodChecked_le` → `encard_monitoringPeriodChecked_le`
- `periodChecked_first`, `periodChecked_second` → `monitoringPeriodChecked_first`, `monitoringPeriodChecked_second`

**Where they are at the head** [ran: `git grep -c`]: `security/Proofs/NetworkCertifier/MonitoringPeriodChecked.lean` 19, `MonitoringPeriod.lean` 9, `MonitoringPeriodCheckedChecks.lean` 2, `security/Proofs/NetworkCertifier.lean` 1, that directory's README 3, `security/lean-audit.json` 4. On `main` none of the eight exists outside the map's own file.

**Three more that ci did not list, which the same added lines hold** [ran: the words in the lines #9663 adds]: `periodLeak_of_checked` (4), `PeriodChecked` (2), `PeriodCheckedChecks` (1). Read each against `main`'s names and rename it if it is an identifier of the old form; if it is a module's or a file's name that is already right, leave it and say so.

**The job**
1. Make your own list first: every identifier #9663 adds that holds `period` for the monitoring period or `window` for the execution phase inside a longer name. Do not stop at ci's eight or my three.
2. One commit by hand on #9663: the identifiers, the docstrings and the README's lines that name them. No statement's text changes beyond the names, and no proof changes beyond the names it cites.
3. Then the records: `lean_changed.py --records security --update` at the new commit, on a node; commit what it keeps as a second commit. Your plain run `-205431-41de` at the old head no longer decides anything; let it end or stop it by its id.
4. The suites whose inputs the two commits change (`suites.py --changed d4df63e727e3 --list` at the new head), quick tier, fresh, with the default temporary directory.
5. The body: one sentence that the crossing left these names and they were renamed by hand on ci's hold, with the list.
6. Send me: the head, the list you renamed, the two commits' line counts, and the runs. I read the difference from `d4df63e727e3` and mark. I tell ci the new head.

**The same names downstream.** #9641, #9350 and #8829 stand on these. Say in the same line which of them holds any of the names, and how each takes the rename: #9641 before its records commit if you can, so it takes one records commit and not two. Push nothing to #9641 until you have told me its plan.

The rule for every push stands: each commit's author and committer are your checkout's own configured identity; never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*`; a node job that commits names the checkout's identity ahead of its command; read `git log --format='%h %an <%ae> | %cn <%ce>' origin/main..HEAD` before each push.

[2026-11-13T21:11Z] 66abef02846a
Lead to PERIODPROOF, Fri 13 Nov 1:11 PM PST, one addition to my line on ci's hold of #9663. **I ran the same search over every open pull request of ours. #8829 adds more names of that kind than #9663 does, and none of them is on `main`** [ran: the words in each pull request's added Lean lines, each name then searched on `main`]: `PeriodLengths`, `PeriodFailsChecks`, `PeriodFails`, `PeriodCheckedExclusion`, `NetTiming.PeriodTerms`, beside #9663's own. `periodLeak_of_checked` is on neither.

- These are candidates, not a ruling: a name whose word is a module's that `main` keeps stays. Compare each with the form `main` gives its nearest relative (`encard_periodGood_le` became `encard_monitoringPeriodGood_le`; `PeriodChannel` became `MonitoringPeriodChannel`).
- #8829 takes its renames in its own commit after #9663's, before ci reads it. Say its list in the same line as #9663's.
- Not candidates, since `main` holds them: `windowsCount`, `WindowsPlaced`, `cappedPeriod`, `rowsWindows`, `window_zero`.

