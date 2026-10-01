# Overnight goal numbers: the cloud console's relay

What fed the console's Goals view (`/console/overnight`) from a cloud VM on 1 Oct, beside node 1's publisher
(`verity_console.py --group node1`, which turns `/workspace/usage/overnight/*.json` into `verity/overnight-<owner>` panels).

- `loop.sh` runs every 5 minutes until `STOP_UTC`:
  - writes `console.json` (open PRs on danielreuter/verity, through `gh`);
  - copies the PR captain's Project-store file to `pr-captain.json` when its sha changes;
  - writes `morning-set.json` from `docs/goals.md`'s Morning set (`morning_set.py`).

  All writes go to node 1 over `research pods ssh vy-nebius-1` (temp file, `chmod 664`, rename).
- `goals_notes.py` turns the Overnight set into `notes/goals.json` (`verity/overnight-goal-notes/v0`). The publisher uses a goal's
  status note unless the owner's file is at least as new. It maps each goal line to the view's key by a phrase (`KEYS`); a new
  goal needs a line there.
- `morning_set.py` writes the Morning set as an owner file (`morning-set`): per goal, keyed `<section>-<n>`, the goal text, its
  latest status note and its checked state.
- `check.sh` prints one status block. It first ends the loop's sleep, because a suspended VM stalls it.
- `morning_table.py` renders one row per goal from node 1's published panels.

Paths assume the Project store at `/cursor/stores/bc-7f347b4b-6175-4b6e-84c6-731add2f8589` and a working directory of
`~/console-overnight`.
