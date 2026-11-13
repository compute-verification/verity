[2026-11-13T03:35Z] 89d3abb84b5e
**NET-408, from LAYOUTREAD, Thu 12 Nov 7:36 PM PST: both manifests and both tables are pushed; the suites' run is launched. Condition 2 holds less tightly than you asked: three lines of the dry run differ, not one thing.**

**The branch** [ran]: `cursor/na-parties-manifests-342d` at `2e0a85a60ce2`, pushed. 15 files. No pull request yet. Cut at `8c11a1d5abff`, before the path move; `main` merged in at `4a7291116e11` (#9375) and at `1f93fc877266`.

**Your three conditions**

1. [ran] Met. `variables` (21) and the bindings' keys (13) are literals in the test, taken from `main`'s output at `4a7291116e11`. The new table gives the same.
2. [ran] **Not as you worded it.** Against `main`, three of the dry run's 23 lines differ, bound and unbound alike:
   - the `workers` plan gains a `certifier` entry: `served-window`, reaching `network-certifier`, with the script and the two hooks;
   - the `network-certifier` entry loses the script, the hooks and its unbound key, and its `what` text changes;
   - the count of entries goes from 20 to 21.
   Parties, channels, the `unbound` list, the configuration's digest and the other 20 lines are the same. A piece of its own is a new entry, so I see no way 2 without the first and third. Say if the `what` texts should stay as `main` has them.
3. [ran `rg`] Met. Outside its test, only `verity_cli.run` reads the served table, generically.

**The CLI's test table** [ran]: your line, in your words, there alone.

**Running** [ran]: `r20261113-033305-8a93`, launched once at 7:32 PM PST in the pool `research`, 340 min from taking a slot: the policy's suite, the CLI's, the root suite, the certifier's, then every suite whose inputs changed, quick.

**[argued], yours to overturn**

- The body's goal line: `network/slice-clean`. Brief 2 named `network/changes-cross-reorganization`, which you said is done.
- The policy's verifier needs `component`.
- Both `params` name the core command line (`network-period check PARAMS`, `inference-only check PARAMS`), not a Python "module:attr": no Python in `verity/` reads either protocol's parameters, and the true names spell retired terms.

**I still owe:** the run's verdicts and the pull request, unmarked, by Mon 16 Nov noon PST; a line by Fri 6 PM PST; for #9589, the one commit by Fri 9 AM PST if lean says no, and the rehearsal's result by Fri 3:30 PM PST.

