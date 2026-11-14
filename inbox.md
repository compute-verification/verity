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

[2026-11-14T00:42Z] c2efc7db226e
LAYOUTREAD to network-accounting (lead), Fri 13 Nov 4:42 PM PST. The trial's Lean check with replay: PASS. I read exit 0 as you did, and no failure in its output. Nothing is pushed yet.

**Run `r20261113-223549-faaa`** [ran]: the check's lean-audit job at `2de57f84fd3e`, as a lane's deferred recorded check with `--no-regenerate`. The check passed in 2,177 s: the three preflight steps, then lean-audit in 1,940 s. 30 of 36 packages were reused, each because every file its audit reads is unchanged. Six ran again, and each is PASS with axioms `propext`, `Classical.choice` and `Quot.sound` only:

| Package | Declarations, modules | Guarantees | Modules the replay checked (its own; of what it imports) | Replay |
|---|---|---|---|---|
| the certifier's verifier | 1,582 in 11 | 4 | 3 of 11; 14 of 876 | 11.4 s |
| the certifier's device | 1,080 in 11 | 0 | 0 of 11; 1 of 738 | 8.6 s |
| the onsite composite | 34 in 2 | 0 | 2 of 2; 39 of 1,211 | 13.3 s |
| `security` | 27,776 in 730 | 1,137 | 30 of 730; 0 of 10,744 | 30.5 s |
| `security/Properties` | 3,485 in 107 | 540 | 10 of 107; 8 of 11,354 | 24.2 s |
| `security/Proofs` | 204,314 in 4,051 | 0 | 180 of 4,051; 2 of 12,830 | 904.3 s |

- **The records:** under `--no-regenerate` a lock record that differs from the build fails the step. It passed, so the two locks I committed from the records run are the ones the full audit holds.
- **What it warned of, none of it a failure and none in a file the rename touches:**
  - The unread ratchet passed (336 wholly unread modules, no area above its cap) and says three areas of the proofs tree came down below their caps in `tools/verity/lean/unread-caps.json`: Accounting 5 to 1 modules, Core 4 to 2, Flock 273 to 259. The rename changes no cap and makes nothing read that was not, so I take this as `main`'s own slack [argued, not checked against a check of `main`]. A loose cap only warns.
  - The standing notes, the same as in the records run: the premises outside their trust over the deployed guarantees (35 in `security`, 30 in `security/Properties`), and the definitions of each spec that no guarantee reads.
- **Cost:** no wait for a slot; 38 min in one research slot of 39 cores (2:36 to 3:14 PM PST).
- **One thing you should know about that cost:** the run's memory peaked at 114.5 GiB, at 3:04 PM PST, in the replay of `security/Proofs` (its own line says 115.7 GiB). A deferred check in the research pool names no memory, so the slot reserved only its own figure and my run went about 80 GB over it for some minutes. I saw no failure from it, but I cannot see other lanes' runs. I do not launch this check in the research pool again without your word; on the rename's day it runs in the merge check's own slot.

**Next:** I read the pool, then launch the suites alone (64 GB asked), and the export only after they end.

