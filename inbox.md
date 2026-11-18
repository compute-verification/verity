[2026-11-18T07:49Z] 65d8c9aa5189
From top, about tests/interfaces_ratchet.json. Three things, none stops your sweep.

1. The ratchet branch `cursor/top-interfaces-ratchet-5c2e` has a new head, 1eef469457d7: one line, `"calibration/sampling_plan/inputs.py": 3` back in `card_names_in_code`. It contains the branch's tip e87f1e0b6c34, so it merges with no conflict. Take it at your next sweep; the card-names red on the tip clears with it. `lean_old_names` (Claims.Msg 605 over its ceiling 603) stays red until lean's rename; do not raise the ceiling.

2. How the entry was lost, so it does not repeat: move 8889cc2764a re-keyed the entry to the file's new path, a later merge of the list brought the old key back whole, and `--prune` then dropped the old key as stale. When `tests/interfaces_ratchet.json` conflicts or two sides both changed it, never take one side whole: keep every key either side has, then run `python3 tests/test_interfaces_ratchet.py --prune`, then run the test.

3. If after a sweep the test is red on a file that a commit in the sweep moved, and the old path was listed, put the entry under the new path at the old count (never higher) with the file's own `dump`, and name the file in your sweep report. Any other red is its lane's: report it with the commit and the lane, change nothing.

