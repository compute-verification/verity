[2026-11-12T09:26Z] d50c85318aad
bounds to training, TRN-67, Thu 12 Nov 1:28 AM PST. A machine blocks step 6, not #8930: `vy-nebius-cpu-7` refuses every source ship, and `--on auto` sends every launch there because its slots look free.

- RAN twice (1:22 and 1:23 AM PST), the Lean records fast path at `2558ba3cdb7d`: "research run --on vy-nebius-cpu-7: refused: source archive sha256 mismatch: the machine received <nothing> … nothing was extracted, nothing is READY", exit 3, after "git ship … not used (the machine has no git)".
- Cause, RAN (`research pods ssh vy-nebius-cpu-7`): `/dev/null` there is `crw-r--r-- root root`, changed 09:13:36Z (1:13 AM PST). The `research` user cannot write it ("bash: line 1: /dev/null: Permission denied"), so the launcher's `command -v git >/dev/null` exits 69, read as "no git", and the ship script's `sha256sum … 2>/dev/null` prints no digest.
- The fix is infra's, one line as root on that node: `chmod 666 /dev/null`. I post nothing on Slack: please ring @infra.
- `research doctor` (1:07 AM PST, before that change) also names ten hosts no ssh reaches: vy-dc-h200, -2, -3, -4, vy-dc-l40s, vy-glm53-cpu-2, -eu, -eu2, -h8, -s1. None is a node I use.

Meanwhile I name no node and force no transport. I read `/dev/null`'s mode there every few minutes and relaunch the same commands once it is writable, or once `auto` places elsewhere.

Steps 1 to 5 are done (RAN): head `2558ba3cdb7d1689ffb3ab70aef50bc16df9c255` pushed; the merge of `main` `19fb19a21e40` had no conflict; the cherry-pick of `206df2144886` was clean and nothing on today's `main` reads the three lemmas; the ratchet gate says 0 failed; the re-scan found no new stale sentence. #8930 stays a draft with yesterday's body until the runs exist.

