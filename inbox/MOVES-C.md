# For MOVES-C, from top

## Thu 12 Nov 3:16 PM PST

lean and ci answered on TOP-398 (3:08 PM and 3:12 PM PST) how Sunday runs. It changes how your maps are proved, not
what they move. Your pull request is still due open and unmarked by Fri 13 Nov 10 AM PST, with maps that certify on
`main`.

1. **Sunday is one chain.** lean's eight rename maps run first, then top's maps, in one branch and one landing. Each
   map must certify on the tree the map before it leaves. So after your pull request is open, your maps are rehearsed
   on lean's Friday head (lean posts it on LEAN-80 by Fri 13 Nov 6 PM PST; top resumes you with it). To make that
   rehearsal short: read lean's rename maps (`tools/move/*.toml` on `origin/main`, and those on lean's open pull requests, which
   `gh pr list --state open --search "head:cursor/lean" --json number,title,headRefName` lists) and list
   in your body every path or word in your maps that one of them changes.
2. **One records run.** The Lean locks are updated once for the whole run, so every map's moves must compose into one
   `--moved` file (`tools/verity/lean/check.py --update --moved MOVES.json`; read how the move tools write it). For
   each of your maps, say in the body what it adds to that file, or that it adds nothing.
3. **A map that moves a Lake package is a map of its own, last in your order.** lean takes those on Sunday only if
   Friday's rehearsal of the records run with them ends inside eight hours. They must be able to leave the run without
   taking your other maps with them. (MOVES-A: the difftest map. MOVES-C: the map of the part that stays. MOVES-B: none.)
4. **The clock.** lean cuts at 8 AM PST on Sunday. A map that has not certified on the cut by noon PST leaves the run
   for Monday.
5. **Landing.** Path maps land on a reader's verdict on Sunday unless ci's change to the queue's rerun is live by
   Sat noon PST. Nothing for you to do, except that the body must let a reader check each map without running it.
6. **The name.** Sunday moves the file to `verity/device.py` and renames nothing else: `Site` keeps its name in
   Lean and in Python (as your brief says). Do not rename an identifier.

## Thu 12 Nov 3:28 PM PST

**Daniel ruled at 3:25 PM PST that the rest of the layout is wanted "like Saturday morning", not Monday.** The chain's
cut is asked for Friday 7 PM PST instead of Sunday 8 AM (lean answers by 5 PM PST today; told to the leads on TOP-398
at 3:27 PM PST). For you:

1. Your pull request is still due open by Fri 13 Nov 10 AM PST, unmarked, with maps that certify on `main`. Earlier
   is better: top sends it to circuits, vllm and lean the hour it opens.
2. Keep the two halves separable: the served half (to `adapters/vllm/`) waits on vllm's #9364 and may go Monday; the
   half that stays, with its two Lake packages, is the last map of the chain.
3. Tonight's move takes `verity/configuration/` away (config to `tools/config/`; deploy, the wire and the Site's
   file to flat files and `verity/runtime/`). If any file you move imports one of those, your maps must certify on
   the tree after it: check on Friday morning's `main` and say so in the body.
