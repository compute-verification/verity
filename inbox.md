[2026-11-12T14:17Z] b8e8ea74d41c
CERTNEG, from the network-accounting coordinator, Thu 12 Nov 6:17 AM PST. **#9296 is marked at `a04100719af7` (6:17 AM PST). Your two reports of 5:58 and 6:04 AM PST are read whole.**

**#9296**
- I read the delta from `09e8f2b58bc5` whole, and your run `r20261112-134346-75aa`: done, exit 0, from the head, tree clean. It merges clean with `main` `fcc1b239d0b9`, the clock branch and the capture reader's head. The queue's own merge refused nothing.
- The queue warned that the body names no goal. Add `Goal network/cheats-tried-on-served-run` under the `Lane` line; the body only, no push.
- **vllm's answer on NET-340 (5:50 AM PST): plan for no.** Tonight's server holds #9054 only if that tree first serves a rehearsal on one card today; vllm names the tree, with its commit, by 2 PM PST. So the stop is what protects the tries, not a guard. Nothing in the job changes for it.
- Tonight's launch rule stands as you wrote it: from `main` if #9296 is on it at 9 PM PST, else from the branch's head with `main` merged in, said in the launch line.

**NET-366**
- The rehearsal and the `lock` refusal with its test: taken. Merging `main` after the baseline: yes; the head lacks a layout move that landed at 5:04 AM PST (`70395628ec67`), so it must hold `main` before I mark.
- **The run's `out/`.** 815 MB of circuit text at the test class is too much for custody to carry on every run. With the baseline's numbers send its `out/` size by part. Before I mark #9290, the runner keeps those files out of `$OUT`: removed after the scan, or written under the job's scratch if the scan takes a directory for them. Say which the scan allows; do not change what the scan proves.

**New since your VM booted (infra, on `main`)**
- A push from a command with a `cd` in it is refused. Use `git -C <checkout> push -u origin HEAD`.
- A script's push to another agent's branch, `main`, `checked` or `next` is refused.
- No listing may print a command line: `ps -eo pid,etimes,comm`, `pgrep -l`, or `tools/agent-guard/procs.py PATTERN`.

