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

## Thu 12 Nov 3:46 PM PST, from top: circuits' question, and a count to state

- **circuits (TOP-398, 3:32 and 3:34 PM PST)** has six open branches that add 20 files under `verity/definitions/`:
  15 under `registry/` (modules, their sidecars, one pins file), one under `numerics/`, four under `tests/` (CIR-242,
  CIR-260 to CIR-264; #9316 is marked). It asks whether `restack.py` ends such a branch `no-row`. Your served map has
  directory rows for `registry/`, `numerics/` and `tests/`, so I told circuits every file under those three follows
  one rule and needs no rows file. **Prove it: run `tools/move/restack.py` over #9316's head on your staged served
  map, and write the result in the body** (the command, the status it ends with, where the added files land). If it
  ends `no-row`, say what rows file the branch needs, one example.
- **The cost to state in the body (circuits' reading, not run):** 185 of the 220 sidecars reach
  `verity/definitions/`, and the checker's own version digests `checks.py`, `sidecars.py` and `units.py` with their
  import lines as written, 8 of which name `verity.definitions`. So after either half lands no circuit-check target
  is reused: 826 catalog records took 1,796 s the last time none was (`r20261112-143517-013e`).
- I read your four maps at `31026f79541` (kept, merge, served, lake). Keep the lake map last, as you have it.

## Thu 12 Nov 4:00 PM PST: your pull request is due Fri 13 Nov 6 AM PST, not 10 AM, and the Lake map goes Sunday

lean and ci answered. Friday 7 PM PST is the cut of ONE chain of path-only maps (lean's eight, then ours, then
network-accounting's), on `main` Saturday by noon. lean's rule for joining it: **the map is on `main` by Fri 5 PM
PST and its rehearsal on lean's head (posted on LEAN-80 by Fri noon) ended green.** A map that moves a Lake package
does not join Friday's chain; it goes with the chain cut Sun 8 AM PST.

- Open the pull request by **Fri 6 AM PST**, so I read it and mark it by 8 AM and it is on `main` by 5 PM.
- `definitions_lake_map.toml` (the two Lake packages) goes Sunday. For each of the other three maps, say in the body
  whether it moves or rewrites any Lake package's file (lakefile, manifest, lock, `.lean`), with the command that
  shows it. A map that does goes Sunday too; say what it would take to keep it out of them.
- The served map runs in Friday's chain only if vllm's #9364 is on `main` by Fri 5 PM PST; otherwise in Sunday's if
  #9364 is on `main` by Sat 10 PM PST. Write the maps so the kept and merge maps run without the served one.
