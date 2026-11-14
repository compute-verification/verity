[2026-11-14T07:39Z] 3342740a92fe
## Coordinator, Sat 14 Nov 07:39Z: top's ruling: rename bare Period* to MonitoringPeriod* on the heads now

From your lead (proofs), Sat 14 Nov ~07:42Z. top answered on TOP-422 at 07:32Z (11:32 PM PST). Act on it now.

THE RULING: an identifier that holds `MonitoringPeriod`, `monitoringPeriod` or `monitoring_period` (any case for the snake form) is NOT a retired-term site. A bare `Period*` name still counts, and so does a name that is another row's old name (`ComposePeriod`, `Dampener*`). The lint change is top's #9961, marked at `edc3c298ffed`, not on main yet. Until it lands, ci's reader passes a site whose identifier says `MonitoringPeriod`.

THE ORDER (top's words: "rename the stack's bare `Period*` names to `MonitoringPeriod*` now, on the heads, without waiting for #9961 to land; do not withdraw #9595's and #9581's marks for this"):
1. Finish the table you are building only as far as you need it to do the rename; the note `gate-period-stack` is still due 08:15Z and may be short: per head, the count by term, and which sites the rename cures, which are prose you reword, which are Flock's Merkle cap (top said yes to the cap tonight; PLACER opens the `OWN_CAPS` lines; do not touch them), and which are something else.
2. Rename, bottom of the stack first: #9595, then #9581, #9625, #9665 (and #9935 only if it needs the carry). Use `tools/move/period_map.toml`'s names where it has one; where it has none, `MonitoringPeriod…` for the type or module and `monitoringPeriod…` / `monitoring_period…` for the rest. Files move with their names (`Spot/Period.lean` and the like), with every import, lakefile glob, lock record (`check.py --update --moved MOVES.json` for a guarantee that only moved), difftest name and doc path. Reword the prose sites. No statement changes: a rename only.
3. To see what the gate will say once #9961 is in: run the gate from a worktree of `origin/main` with #9961's head merged in (`git fetch origin pull/9961/head`), as well as from plain `origin/main`. Report both counts per head.
4. Each head: the Lean check of the packages touched and the changed suites' quick tier, on a node (`research run --on auto --cwd source`), never on your VM. Then `ready-<PR>-at-<head12>` with the gate's two counts in it. I mark each as its note comes; a head whose only remaining sites are identifiers holding `MonitoringPeriod` is marked on top's ruling.
5. #9595 and #9581 are admitted. A new head needs my new mark, so push #9595's renamed head only when its checks are green, and write its ready note the same minute.

First word: a checkpoint note within 30 minutes saying how many names and files the rename touches per head, and your hour for #9595's ready note.

