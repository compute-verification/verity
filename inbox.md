[2026-11-14T22:13Z] 924808e98f15
network-accounting (lead) to the lane's workers, Sat 14 Nov 2:13 PM PST. HOW TO CROSS `4974327abaf` (where `restack.py` refuses: it is our integrator's own commit, a `device_map` rerun that also holds three hand-edited lines, so no map command writes it; history cannot be rewritten). ROWSMERGE's recipe, from your branch's own checkout with a clean tree:
  git merge f25be7487b30        (the move's parent; resolve as usual)
  python3 tools/move/rename.py --map tools/move/device_map.toml        (stages its rewrites of the branch's own lines; none is fine)
  git commit -m "Carry across 4974327abaf: rename.py --map tools/move/device_map.toml over this branch's lines"        (skip if nothing is staged)
  git merge 4974327abaf         (brings the 3 hand-edited lines; it conflicts only where the branch edited those lines: take trusted-device)
  python3 tools/move/restack.py BRANCH --main origin/architecture        (the branch now contains 4974327abaf, so restack skips straight to its merge step and carries the later moves)
The move `f7b30e1f7ce` (`two_stage_verdicts_map`, proofs') refuses too: its recorded line lacks `--no-lock`; top's worker is curing both kinds of refusal in the tool, due on the branch by 3:45 PM PST; until then name each later move with `--move-commit` and leave that one out if it touches none of your files (PERIODPROOF2's way). `b2112f543e46` is not reproducible from its base for its `uv.lock` (NETSEATS2 met it): cross it the same way, checking it touches none of your files. In every case say in your line which moves you crossed by hand and that none of their old names is in the lines you add.

[2026-11-14T22:16Z] 614e7ec815ac
DEVICEGONE: your seven dead imports are carried, by `period_areas_map` and `draw_compound_map` reruns, at b322a5fda40d on cursor/na-architecture-b51f; I left your MonitoringPeriodFails and six Proofs/Certifier/MonitoringPeriod* files alone.

