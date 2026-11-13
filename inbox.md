[2026-11-13T08:47Z] 184e36ffa45d
From your lead, 08:47Z: chain 1 of the path moves landed on main at 08:44Z (Queue: #9697, e4da505fe48). Fetch again and merge THAT main, not b3902993caea. It changes one line in six adapter files (verity config/settings paths) and tests/import_order.toml. If a path of yours moved, carry it with tools/move/restack.py, never by hand. Run the quick tier on the merged head.

