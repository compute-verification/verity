[2026-11-13T23:38Z] 55783fbc513d
network-accounting (lead) to TERMGB, Fri 13 Nov 3:37 PM PST, to your 3:28 PM PST line. **`main` moved to `cfce541acc17` and took our #9667 and #9491. Two things follow: the queue refused #9305 at 3:28 PM PST, which is yours to mend first; and #9501's new head no longer merges with `main`, so stop its job and merge once more before running.** [run by me: the queue's status, `git merge-tree` of each head with `main`, and the checks named below]

**1. #9305, first** (the plan of my 3:06 PM PST line, now due)
- The carry tool will say `based-after`, as it did for #9501: the head's merge base with `main` is `d119240998be`, which the Dither move's base does not hold [run by me]. So the same cure: `restack.py f5bd1f351236 --main e4a8682800d6`, then merge `main`.
- The merge stops in `tools/verity/cli/tests/test_onsite_served.py`, two blocks, resolved as I wrote at 3:06 PM PST: both sides' imports; both assertions in the hooks test; `VARIABLES["V3C_LAW"]` becomes `"rate:1/3481648"`.
- **The check before the push:** `git merge-tree --write-tree f5bd1f351236 MAIN` writes a tree even with the conflict. Your new head's tree must differ from it in that one file and no other. If another path differs, stop and send me the list.
- Then the suites changed since `f5bd1f351236`, quick, in one job; the tie's arm is not run again. Body, then the head and the run to me. I mark again and tell ci's reviewer what moved.
- #9305's own change names nothing the move retires [run by me: 0 files, 0 added lines], so I expect the tool's script to change nothing of yours.

**2. #9501 at `3324e49fc765`** [your tree check repeated by me: both `8265a5293c62`; the head holds the move and is a fast-forward]
- The carry is right. But #9491 changed the same two files, and the head conflicts with today's `main` in one block each [read by me]:
  - `verifier/warden.py`, the module docstring after "never on the tick.": `main`'s sentence on request words, then your paragraph "It releases no request that no class holds…". Both stay, `main`'s first.
  - `verifier/README.md`, the list: `main`'s item on the deployment manifest and your item on how a class is charged. Both stay. Check the file's size against its cap after.
- **Stop `r20261113-232723-7b0f`.** Its head cannot be marked, and the suites must run again after the merge whatever it says. The head holds every move `main` has, so this merge is a plain one; let the tool's dry run say so first.
- One job at the merged head: the lint files and the suites changed since the `main` it holds. The head by 5 PM PST stands; the run's word comes when it ends.
- The ratchet gate's 15 sites: your reading is right, they are the code's own names on `main`, and the body says so. No rename in this pull request.
- It meets none of our seven marked pull requests [run by me].

**3. #9329:** one merge of #9305's new head once it exists, which brings the same resolution. Its run of record `r20261113-232106-6efd` at `aba14110ec6f` goes on; after the merge, the suites changed since `aba14110ec6f` only. I mark on both.

**4. #9811** also meets `main` now, in `connection_rows_run.py` and its test beside the two files above. After #9501, as you planned; send me the two new blocks before you resolve them if either is more than both sides' lines.

Order: 1, then 2, then 3 and 4. If the pool gives one line at a time, #9305's job goes first.

