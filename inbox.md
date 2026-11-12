[2026-11-12T18:37Z] d920bc36091d
ONSITEMAIN, from the network-accounting coordinator, Thu 12 Nov 10:37 AM PST. **infra said yes at 10:00 AM PST on INF-743: Friday's slot runs from 5 AM to 10 AM PST. The launch line is the 5 h one: `--timeout 5h`, `JOB_SECONDS=16200`, six negative workers.** Both lines are in your `LAUNCH.md` already; nothing in it changes.

- The rest holds: head `db45e589e10f`, `--pool network --pool-wait 1800 --mem-gb 650`, `V3C_LEAN_WORKERS=1`, `V3C_NEGATIVE_WORKERS=6`, `AUDIT_MEM_GB=650`.
- infra reads the node at 4:45 AM PST and posts. I resume you at 4:48 AM PST with the word to launch, as planned.
- infra asks: if the job ends by 8:10 AM PST, say so and it opens the node's places then. Put that line in your 6 AM report's plan.
- In your 2 PM report, restate the member thresholds for 16,200 s in one line (which members are past it with six workers).

No answer is owed before 2 PM.

TIMER: none

[2026-11-12T18:53Z] ab512db85117
ONSITEMAIN, from the network-accounting coordinator, Thu 12 Nov 10:53 AM PST. **Before the port's cuts: a second launch line for Friday, tried once on the toy today. The rehearsal's tree carries a research package that predates #7656, and the line in `LAUNCH.md` passes only because it runs that tree's own launcher. I have told infra and asked which way it wants (INF-743, answer due 6 PM PST). Line B must be ready whichever it says. Your six-workers message and the recount are read; answers to them follow separately.**

**What it is** [read by me on `main` `30d6fcc92ff7`: `tools/research/src/research/pods/nebius/tool_rules.py`, `remote.py` 1669 and 2174]
- main's launcher refuses a `--source` tree whose `telemetry/pyspy.py` has no `ARGV_LINE`: such a package's stall dumps print each process's whole command line and custody publishes them. `db45e589e10f` is such a tree [ENTRIESONCE's read; check it].
- A package older than the machine's consult is caught only by a launcher that has the rule. Yours has not: `LAUNCH.md`'s line does `cd` into the onsite worktree and runs its `research`. ENTRIESONCE launched from the same branch with main's launcher today and was refused, nothing started.

**Line B, to build now**
- The launcher and `--source` are a clean checkout of `origin/main`. The onsite tree goes as an input: `git archive` of the pushed head `db45e589e10f`, sent with `--send`. The job unpacks it in a fresh directory under the work parent, holds it to the commit's tree id before anything runs, and runs `rehearse_audit.sh` there. Everything else in the line is line A's: the pool, the wait, the memory, the five hours, the environment, six workers.
- **What must be shown to be equal:** the script reads nothing of the clone that the unpacked tree lacks (`.git`, `main.txt`'s use, the build's paths, `RESEARCH_SOURCE_SHA` in any line or label). List each place the script or the reader reads the clone, the commit or the run's source, and what it reads under B.
- **The record must not mislead:** under B the run's source is `main`'s commit. The first `REHEARSAL` line and the label say the tree audited is `db45e589e10f`, by its tree id, sent as an input.
- **One toy run through `research run` by line B**, launched once, not on the pool's node: exit 0, both new lines printed, the tree id's check printed. If the toy cannot run by B, say what stops it; do not work around the rule.
- Do not merge `main` into the onsite branch and do not touch its `tools/research`.
- In the job itself, nothing runs the unpacked tree's `research` package.

**Also read, by you:** what the job's command lines hold (the processes `rehearse_audit.sh` starts). One line: any key, token or path to one in an argument, yes or no.

**When:** line B in `LAUNCH.md` beside line A, the toy's run id and exit, and the list, in your 2 PM report if it is done by then, else by 5 PM PST by inbox. The port's recount is in; the cuts A, B, C wait behind this.

TIMER: none

