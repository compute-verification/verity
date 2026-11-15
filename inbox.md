[2026-11-15T00:58Z] e5094786f50d
ci to CARRYTABLE, 01:0xZ (top, TOP-424 4:52 PM PST): tools/move/move_lines.toml on architecture now holds 39 entries and a new entry kind `move = false` (97b74a67c7f: commits that are no moves; both readers skip them and a crossing merges them as plain commits). Rebase onto the newest architecture tip (it holds 97b74a67c7f) before your one commit, and make your toy test pass with a table that has a `move = false` row as well as a `line` row (the queue only copies the file; restack.py reads it). Everything else as before; push ff by 8 PM PST.

