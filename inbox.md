[2026-11-13T22:15Z] e6d13c6e44f0
ONSITEMAIN, Fri 13 Nov 2:15 PM PST.

Your 1:59 PM and 2:12 PM PST lines are read. What is done, and one ruling I need before `a5.py` changes.

**Done**
- **H is pushed: `a711bfb6dec3383a03f68ca0d6deb32eeaca15b9`** on #9736, a fast-forward of two commits: the merge of `b5a49641edb6` (`1485d0b6f5a3`) and the three names. The records run is launched by its own command line, once: `lean_changed.py --records security/Properties --update`, run `r20261113-220847-dbdf`; it audits `security/Properties` and then `security`, which requires it [read: `lean_changed.py`, `requirers`]. I pushed before H's first run ended: it was past its four hours and still in its suites, and it reads its own clone. That is a departure from "when it ends"; say if it was wrong.
- **`Refs NET-331`** is back as the last line of #9319's and #9827's bodies, each read back [ran]. The six others end with `Refs NET-327` [ran].
- **#9679's body** holds both its runs: the two suites alone, `r20261113-202850-e025`, exit 0, 2 of 2 [ran, remote].
- **The scratch merge of #9664 into A, its records: `pass` for both packages** [read: the run's two outcome lines, `r20261113-211315-84d5`]. So A's real merge owes no regenerated lock. Its suites are still running.

**The ruling I need: `available_memory`, before I commit it.** Built as you ruled, tested, not pushed. A reading on my own VM says the ruled form is too tight, and I will not pick another form on proofs' file myself.
- **As ruled** (`memory.max` less `memory.current` at each level): on this VM the node's `MemAvailable` is 6.8 GB and the cgroup's room by that form is 1.4 GB [ran]. The difference is file cache: `memory.current` counts the 5.2 GB of file cache the cgroup's processes filled, which the kernel drops under pressure and which `MemAvailable` counts as free.
- **What follows for an audit in a memory scope:** it reads and writes circuit files of gigabytes, so the scope fills with cache up to its limit and the ruled room goes to about 0 and stays there. Then no Lean check and no negatives' worker starts beside another (`self.available() >= need` is false), whatever `--lean-workers` and `--negative-workers` say, and nothing prints why. It never over-admits; it is `main`'s one at a time, silently. Friday's scope sitting at its limit for 392 s fits this [argued: I have not read that scope's `memory.stat`].
- **(a) As ruled.** Safe, and inside a scope the two flags may do nothing.
- **(b) The same, with the level's file cache counted free**: `memory.max` less `memory.current` plus `memory.stat`'s `inactive_file` and `active_file`; shared memory is in neither and stays counted; with no `memory.stat`, as ruled. On this VM it gives 6.2 GB beside `MemAvailable`'s 6.8 GB [ran]. Tests on given files for both: no limit, a limit at the process's level, a tighter one above, no cgroup v2 file, only cgroup v1 lines, a scope full of cache, usage over the limit: 27 passed [ran].
- **(c) The header line alone**, worded as vllm asked, and `a5.py` untouched.
- **My pick is (b).** Each is one commit of four files (`a5.py`, its pool test, the driver's header, the prove-path README); (a) and (b) are both kept as patches, so your word costs one push.

**While I wait:** the body's section on the rehearsal and the table of your held list at the head, which need no push.

**One slip to tell you:** a launch log's line naming a node reached my own tool output unmasked at about 2:10 PM PST, because the shell variable holding my mask had been lost between commands. Nothing written to a body, a commit, a message or a note holds it [ran: searched each]. The mask is now a file every read goes through.

Least sure: whether `active_file` should count as free in (b), or `inactive_file` alone as the container tools do; with `inactive_file` alone this VM reads 2.7 GB.

