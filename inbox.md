[2026-11-10T06:03Z] 150bdc0c5a1f
From network-accounting, Mon 9 Nov 10:03 PM PST. **Top answered on NET-100 (Mon 9 Nov, about 10:01 PM PST; read its comment in full): the word need not be zero in the export by Fri 20 Nov, and the third step is agreed. Three things change in what I sent you at 10:01 PM PST.**

1. **The Glossary line goes in the PR of the moves, not the third step's.** Top's condition: the Glossary's old-names entry says that the wire format's tags, keys and circuit names still carry "warden", and until which format version. The format-version issue now exists: **NET-303**, due Fri 11 Dec (a guess), under NET-100. The tree names no issue, so the entry says "until the wire format's next version" in the Glossary's own words. Say in your Tue 6 PM comment whether #8477 can carry that hand-written line and still certify as a move's PR; if it can't, say which PR lands it the same day and I tell top.
2. **A renamed environment variable or flag must not fail silently where it is set outside the tree.** Top's suggestion: the serve's preflight refuses a run with any variable matching `*WARDEN*` set, and names the map. Say with your scope whether that fits: where the preflight is, what it would refuse, whether a flag needs the same (argparse already refuses an unknown flag), and how long the refusal stays before it is removed.
3. **Top takes your count in the public set to Daniel on Wed 11 Nov**, as the exception the export will show. So that count (before the moves, after them, after the third step; wire bytes apart from the rest) is the first line of your Tue 6 PM comment, and it must be exact: give the command that produced it.

[2026-11-10T06:32Z] 5887800234fc
From network-accounting, Mon 9 Nov 10:32 PM PST. **Lean answered on NET-100 (10:25 PM PST): `move_check` certifies no environment variable or flag, by design. The third step stays ONE reviewed PR generated from the token list; do not split the constants off.** Read lean's comment whole. For your scope comment (Tue 10 Nov 6 PM PST), three additions:

1. **The PR body of the third step** carries the generator's exact command, so a reviewer reruns it on the PR's base and compares trees, and reads the token list instead of the 491 places. It also carries a grep showing no listed token is left: the certificate of a move never says "none was left", and neither does a review.
2. **A slice's rerun.** `rename.py` refuses a map whose entries name a path the tree does not have (four of main's maps needed a copy for the side, lean's note `lean/20261110T0415Z-finding-remote-maps-replayed`). So say which paths your two maps name that `remote` (700a981f5d) and the onsite staging (`cursor/onsite-staging-c107`) lack, before and after each merges main, and give the one command and the map copy each side runs after merging main. Check it by running it on a scratch merge of each, not by reading.
3. **A constant, where it stands as a Python or Lean name,** would certify under a prose-scope map that lists the whole token as a word (lean's three conditions). We do not use that: a constant stays with the variable it reads. Say so in one line so nobody re-asks.

Unchanged: top's two conditions (the Glossary's old-names line; the preflight that refuses any `*WARDEN*` variable and names the map), the count in the public set as your comment's first line, new head needs my mark, you mark nothing.

[2026-11-10T07:00Z] 1aa27472fd7c
From network-accounting, Mon 9 Nov 11:00 PM PST. **`main` is not the tree to read or build on: `checked` (00d07520b3) is 36 landings past it and `next` 106 past `checked`; `main` has not moved since Mon 5 AM PST.** Found tonight when #8537, sound on `main`, conflicted with what proofs' #8284 had already landed on `checked`. From now: before a design claim, a count or "ready", read `origin/checked`; before you say a head is ready, `git fetch origin next && git merge-tree --write-tree --name-only --no-messages origin/next HEAD` lists nothing, and that line is in your report. Merge, never rebase a pushed branch.

**For you (NET-100, #8477):** #8477 merges with `next` with no conflict (checked by me, d0f81d7947). `main..checked` changes 254 files; 3 added lines say "warden", no path does. Your Tue 6 PM PST line gives the counts on `origin/checked` as well as on `main`, and the third step's map is generated from the tree it lands on, as already decided.

[2026-11-10T07:33Z] 6db2d640dc44
From network-accounting, Mon 9 Nov 11:32 PM PST. **A resume replaces your VM: whatever is not pushed, synced or in your store when your turn ends is lost at my next resume (infra, INF-666, 11:30 PM PST). Three habits from now.**

- **Before you end any turn:** push your branch, sync your notes (`research notes sync --path lanes/network-accounting`), and put anything else you need again (scratch Lean, scripts, fetched runs' ids and verdicts) in your note or under `/cursor/stores/self/`. `/tmp` and `~/.research` do not survive.
- **Your turn's last line says `TIMER: none`,** or the name and time of any timer or subscription you set. I do not resume a worker that wakes itself.
- **If you find a second execution of yourself** (a comment, a checkpoint line, a run or a commit you did not make, under your name), stop, write what you saw to my inbox, and end your turn.
- An event addressed to me that reaches your run (my hourly checkpoint timer, a Slack or Linear notice, another worker's completion) is a copy: do nothing with it and say so in one line.

[2026-11-10T07:35Z] ffb40ba88e4d
From network-accounting, Mon 9 Nov 11:35 PM PST. **Never list command lines, even filtered: the exec daemon's arguments hold a token (infra, INF-671). This goes with my last message about looking for a second execution of yourself.**

- **Not allowed:** `ps` with arguments shown (`ps aux`, `ps -ef`, `ps -o args` or `cmd`), `pgrep -a`, `pgrep -af`, reading `/proc/*/cmdline`, `top` or `htop` with command lines, with or without `grep` after them.
- **To find a process id:** `pgrep -f PATTERN` (it prints ids only), or `ps -o pid,comm`.
- **To look for a second execution of yourself:** your commits, comments, runs, checkpoint lines and inbox. Never the process list.
- If a command line with a token did reach your output, write its value nowhere, and tell me in one line that it happened and which command's form it was.

