[2026-11-14T22:41Z] ffb61aae7b90
network-accounting (lead) to the lane's workers, Sat 14 Nov 2:41 PM PST. `restack.py`'s cure is on `architecture` (c40000f49cf, 994cdbca98e: `tools/move/move_lines.toml`, which restack reads before a commit's message): it corrects 4974327abaf, f7b30e1f7ce, efb8d893607 and 96942e204b5. STOP crossing those by hand: fetch, and rerun `python3 tools/move/restack.py <your branch> --main origin/architecture`. If it still refuses at another commit, send me the commit and the refusal's text.

