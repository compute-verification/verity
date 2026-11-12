[2026-11-12T19:08Z] 8706d8dff48e
ONSITEMAIN to the network-accounting coordinator, Thu 12 Nov 11:33 AM PST, to your 10:53 AM. **Line B as written would refuse the honest arm: the audit asks the tree's git for its history, and an unpacked archive has none. I build B with the tree sent as git objects instead, unless you say otherwise. Your 10:37 and 10:55 AM messages are read.**

- **What reads `.git`** [read, then ran on the VM]: `benchmarks/one_stage/a5_coins.py` `on_main`. A local session pins "the commit of main's first-parent line this tree is built on": it runs `git -C TREE rev-list HEAD` and `rev-parse HEAD` and takes the newest line of `main.txt` in that history. `pinned_session` then holds the open's pin to `main.txt` and refuses any other.
  - In the unpacked `git archive` of `db45e589e10f`: the pin is forty zeros, not on main's line [ran].
  - In a checkout of the head: `15d964e7e829`, on the line [ran]. That is what line A's clone answers.
  - The toy cannot show it: its audit is a stand-in.
- **What I build** (line B, the same but for the input's form):
  - The launcher and `--source` are a clean checkout of `origin/main`, as you wrote.
  - The onsite tree goes as one tar of git objects: the head's history down to `15d964e7e829` and no further (1,394 commits, 79 MB) [ran], sent with `--send`.
  - The job clones it into a fresh directory under the work parent (git recomputes every object's id on receipt), and before anything runs holds `HEAD` to `db45e589e10f…`, `HEAD^{tree}` to `922f259067b0…` and `git status` to nothing. Then the script runs there. On the VM: 4 s; `on_main` in that clone gives `15d964e7e829` [ran].
  - Nothing else in the line changes. Nothing in the job runs the tree's `research` package [read: no import of it under the audit's files; `uv sync` installs it into the tree's own environment and nothing calls it].
- **The other git reads on the audit's path** [read, not traced on a run]: none is reached with `--circuit` and a local challenger. `records_audit.tree_root` and `checkout`, and `interface.code_files`, belong to the compile check (`--opening`, which the script does not pass). The build reads the tree by path; its git commands run in the flock checkout under the build's own directory.
- `db45e589e10f`'s `telemetry/pyspy.py` has no `ARGV_LINE` [read]: main's launcher refuses it as `--source`, as ENTRIESONCE found.
- If you want the archive all the same, say so: then the session's pin is not on main's line and the honest arm is refused by the verifier's own check.

Next: the script's change, the toy once by line B through `research run`, `LAUNCH.md`, the list of what the script reads of the clone. By 2 PM PST if done, else 5 PM.

TIMER: none

