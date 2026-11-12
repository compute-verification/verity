# For MOVES-A, from top

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
7. **For you alone.** network-accounting moves `verity/protocols/inference_only/nci/` and `rho/` back to
   `benchmarks/nci/` and `benchmarks/rho/` in a map of its own, in Sunday's run after your config map. It is not
   yours to write. If one of your maps touches a file under those two directories, say so in your body.

## Thu 12 Nov 3:19 PM PST

**Daniel asked at 3:17 PM PST whether the layout can be on `main` tonight or tomorrow morning.** Top is asking ci,
proofs and vllm, and tells Daniel by 5:30 PM PST. Whichever night is chosen (tonight, Friday night after the 6 PM
run, or Sunday), the pieces with no Lake package go first. So split your work in two pull requests:

1. **First, as early as you can today, by 7 PM PST if the node lines allow:** the config map
   (`verity/configuration/config/` to `tools/config/`) with `verity/settings.py`, and the bench instances. Open it
   even if a node run is still in line; say at the top of the body what is proved and what is still running.
2. **Second, by Fri 13 Nov 10 AM PST as before:** the challenger's service to `compose/`, `artifacts.json`, and the
   difftest map (the one with the Lake package, last).
