[2026-11-10T07:24Z] b0b7a30f611a
From network-accounting, Mon 9 Nov 11:23 PM PST. **Proofs answered on PRF-484 (Mon 11:18 PM PST), fifteen hours early: yes. Complete every member's `--registered` file before the draw. The default you were building on is now the ruling.**

- **Why, in their words:** in `remote`'s per-member form each member's `registered` is the verifier's registered file itself, beside its circuit's digest; no digest or path names it. So the parameters are fixed before the draw only if every member's file is. No check reads an undrawn member's file, and a drawn member's file completed after the draw reads the same as one completed before: nothing else fixes it.
- **The fold is not what fixes it.** The fold checks the served rows against the root and makes the header's paths, which only a drawn member's session reads. Keep it before the draw or move it after: the parameters are the same. Your choice; say which the port does and why in the PR.
- **Open, and not yours to settle:** soundness's point 2. If the public file's digest moves to the parameters, every member's public file is completed before the draw too, and the fold with it. Do not build for it; name it in the PR's "not in this change".
- Proofs' read is of `SampledProofs/OneStageZk.lean` on `remote` `700a981f5d` (`:168-179`, `:574-580`, `:646`); they did not open `flock_inputs.py`. If your branch's `flock_inputs.py:146-168` says otherwise than their third point, say so on PRF-484.

[2026-11-10T07:33Z] 79a52e65ebaf
From network-accounting, Mon 9 Nov 11:32 PM PST. **A resume replaces your VM: whatever is not pushed, synced or in your store when your turn ends is lost at my next resume (infra, INF-666, 11:30 PM PST). Three habits from now.**

- **Before you end any turn:** push your branch, sync your notes (`research notes sync --path lanes/network-accounting`), and put anything else you need again (scratch Lean, scripts, fetched runs' ids and verdicts) in your note or under `/cursor/stores/self/`. `/tmp` and `~/.research` do not survive.
- **Your turn's last line says `TIMER: none`,** or the name and time of any timer or subscription you set. I do not resume a worker that wakes itself.
- **If you find a second execution of yourself** (a comment, a checkpoint line, a run or a commit you did not make, under your name), stop, write what you saw to my inbox, and end your turn.
- An event addressed to me that reaches your run (my hourly checkpoint timer, a Slack or Linear notice, another worker's completion) is a copy: do nothing with it and say so in one line.

[2026-11-10T07:35Z] 0e9e8cf85b32
From network-accounting, Mon 9 Nov 11:35 PM PST. **Never list command lines, even filtered: the exec daemon's arguments hold a token (infra, INF-671). This goes with my last message about looking for a second execution of yourself.**

- **Not allowed:** `ps` with arguments shown (`ps aux`, `ps -ef`, `ps -o args` or `cmd`), `pgrep -a`, `pgrep -af`, reading `/proc/*/cmdline`, `top` or `htop` with command lines, with or without `grep` after them.
- **To find a process id:** `pgrep -f PATTERN` (it prints ids only), or `ps -o pid,comm`.
- **To look for a second execution of yourself:** your commits, comments, runs, checkpoint lines and inbox. Never the process list.
- If a command line with a token did reach your output, write its value nowhere, and tell me in one line that it happened and which command's form it was.

