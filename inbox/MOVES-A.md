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

## Thu 12 Nov 3:28 PM PST

**Daniel ruled at 3:25 PM PST: "let's get as much of the cosmetic stuff done as possible ASAP, like by tonight or by
tomorrow morning. And then the rest can be like Saturday morning". Told to the leads on TOP-398 at 3:27 PM PST.**

1. **Tonight: your `config` and `settings` maps only.** They are the first two links of tonight's chain; MOVES-B's
   flat files come after them. Your first pull request, open by 7 PM PST, holds exactly those two maps, their hand
   edits, and the staged moves on today's `main`, certified. Bench instances join it only if they need no hand edit
   and certify; otherwise they go with the rest. Write in the body, as commands someone else can run on a fresh
   clone, how to restage it on a newer `main` (which commits are kept, which are generated and regenerated).
2. **The rest is asked for Saturday morning, not Sunday:** the chain's cut is Friday 7 PM PST if lean can hold it
   (lean answers by 5 PM PST today). Your second pull request (the challenger's service, `artifacts.json`, difftest
   last) is still due open by Fri 13 Nov 10 AM PST, certified on `main`.
3. **End your turn when each pull request is open**, naming its number; everything pushed.

## Thu 12 Nov 3:31 PM PST

Two lines from lean (TOP-398, 3:25 PM PST) for tonight's two maps:

- **Change no Lean input.** Of the files naming `verity_config` or `settings`, none is in the queue's `lean_inputs`
  by lean's count, but check your staged moves' changed files against it yourself (find `lean_inputs` and
  `lean_audit.own_files` in the check tools) and write "changes no Lean input" in the body with the command. Never
  let a map rewrite a `lakefile.toml`, a `lean-audit.json` or a `.lean` file tonight.
- lean's #9402 changes `tools/verity/lean/tests/test_merge.py`, a fixture that names `verity/settings.py`. Say in the
  body whether your settings map rewrites that file, so lean knows to restack.
