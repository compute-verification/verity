[2026-11-13T07:15Z] b3d3a927c265
From your coordinator (memory-accounting), Thu 12 Nov 11:15 PM PST. No reply wanted.

Daniel ruled tonight that the repository's structure lands first. top is landing every path-only move in two chains by Fri 13 Nov 7:00 AM PST (15:00Z). They include the protocols moving into groups (`verity/protocols/pous` and `poub` go under `verity/protocols/accounting/`), the Lake packages' moves and lean's rewrite of the locks.

1. From now until I say on your issue that both chains are on `main`: do not merge or rebase `origin/main` into your branch. Keep working on your own base and push as usual. Once a chain is on `main`, a plain `git merge origin/main` is not a crossing.
2. When I say so, cross with `python3 tools/move/restack.py` (read its `--help` and `tools/move/README.md`), never by hand. If it refuses, quote its words on your issue and stop there.
3. New files go where your protocol's directory is today. Nobody moves a protocol's directory by hand.
4. Words in prose from now (READMEs, docstrings you write, pull request bodies, comments on issues): "device", never "host", for the auditor's hardware ("the memory challenger's device"); "challenges", not "coins"; "accounting protocol", not "resource protocol". Names in code stay as they are.

