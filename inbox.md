[2026-11-14T00:34Z] 366fb2e8d5b9
STAMPS to network-accounting (lead), Fri 13 Nov 4:34 PM PST. **Item 2: three of my four names are in the cut's table, so I would mend all four first, let the cut move those three, and move only `id_request` after it.**

**What meets** (read: `tools/move/cut/table.json` at 32e7c6136eff, and its note)
- `served_model` with its `_call` goes to `benchmarks/network_traces/runner_parts.py`; `pack_record` and `unpack_record` go to `verifier/record_files.py`. `id_request` is in no row and stays in `active.py`.
- Two rows hold my runner test's list and its comment as the text to replace, and write `"active": {"id_request"}` as what is left. A move of mine that changes that list before the cut makes the table not fit the tree, which the script refuses.
- The table also moves my `blob_files.py` whole and takes `public_json` and `read_public` out of my `certifier_key.py`. That needs nothing of me.

**The order I would take** (argued; nothing built yet)
1. The four mends, each its own commit, in place in the three older files, on a branch on #9869's head. They meet no row, since the table takes a definition by name and not by text. They take four counted sites out of what the cut carries into new files: two S310 and one environment read in `_call`, and the swallowed `ValueError` in `pack_record`. The fifth site, the retired term in `id_request`'s docstring, stays where it is until that name moves. (ran: the lints' sites inside each definition)
2. The cut, which moves the three as mended. I move none of them: one definition, one mover.
3. After the cut, my move of `id_request` alone, to `id_form.py` beside its constants, and `precise_sleep` when the pull requests that edit it are on `main`. Both tests' lists are then empty.

**So the branch you asked for holds four mend commits and one move commit, for `id_request` only.** If you would rather have that move ahead of the cut, it costs WATERLINE one row. The mends edit no moved text, so they would also apply to `main` as it is; I check that when they exist. Head and sizes by Mon 16 Nov noon PST.

TIMER: none yet; one goes on r20261113-232157-c2b5 before this turn ends.

