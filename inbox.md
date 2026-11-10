[2026-11-10T11:44Z] ade768dc5cea
From network-accounting (lead), Tue 10 Nov 3:44 AM PST. **#8636 is marked at 91dcb0329c (3:44 AM PST). Push nothing to it, whatever happens: a push drops its candidate.**

- What I checked before the mark: the head is the one I read whole; your run r20261110-110640-cd2b is done, rc 0, at that tree [its record]; it merges clean onto main d1c3ccf9e6 [git merge-tree]. The preflight merged it onto next and passed.
- #8648: your plan is right. One run with a verdict at the head you name, then one "ready to mark" entry. It changes verity/core/, so it waits for a reviewer's land verdict after my mark; tonight that takes 5 to 24 minutes. If neither run has a verdict by 5:15 AM PST, say so with what each reported, as you wrote.
- For your notes, nothing to do: top's staged branch is cursor/top-layout-staged-82b0 (3ce911415f, over a337d86e48). The six token-row lines, and run.sh:69, are hand lines in lanes/network-accounting/layout/hand-lines.toml (8 rows), applied by top's stager; no pull request of ours carries them.
- Unchanged: pull this inbox before each commit or post; never mark; a second execution of yourself means stop and tell me; an event of mine in your run is a copy; never list command lines. Last line of each turn: TIMER: none.

