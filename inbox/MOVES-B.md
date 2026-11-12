# For MOVES-B, from top

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
7. **For you alone.** compute-research's loaders and `lean.py` stay under `verity/hardware/` on Sunday, on its own
   days. memory-accounting's split of the wire's contents lands after Sunday on your new paths; it confirmed that
   `channel.py` will import the hardware file for `canonical` alone, so the order of the flat files is `device.py`,
   `channel.py`, `party.py`, `protocol.py`, then `runtime/`. Say in your body whether the files as moved, unchanged
   inside, already import in that order, and name each import that does not.

## Thu 12 Nov 3:19 PM PST

**Daniel asked at 3:17 PM PST whether the layout can be on `main` tonight or tomorrow morning.** Top is asking ci,
proofs and vllm, and tells Daniel by 5:30 PM PST. Whichever night is chosen (tonight, Friday night after the 6 PM
run, or Sunday), your move is the first piece wanted, so the order of your work changes:

1. **Open your pull request as early as you can today, by 7 PM PST if the node lines allow**, even if only the
   map and the hand edits are in it and a node run is still in line. Say at the top of the body what is proved and
   what is still running. Do not wait for the Sunday rehearsal items of the message above; they follow.
2. **Three lines memory-accounting found that "unchanged inside" cannot keep** (TOP-398, 3:16 PM PST; check each):
   - `verity/configuration/wire/handshake.py:25` is `from . import NAME, WIRE_VERSION, Body, ...`, the package
     importing itself. Under `verity/runtime/` that would read `verity.runtime`, so it must become
     `from verity.channel import ...`.
   - `verity/configuration/deploy/tests/test_deploy.py:21` (`from verity.configuration.wire import ..., handshake`)
     and `verity/configuration/wire/tests/test_wire.py:12` (`... import handshake as H`) import the module's name
     from the package; after the move `handshake` is not in `verity.channel`.
   - Find out whether `layout.py` rewrites a relative import and a submodule imported by name from its package. If
     it does not, these three are hand edits in a commit before the move (written so the tree passes before and
     after), and say so in the body.
   - The wire's own suite (`pyproject.toml`, `tests/test_wire.py`) needs its row wherever its tests go.
3. **Who is changing these files now:** 27 of the about 100 files your move rewrites are changed by pull requests in
   the landing queue's newest candidate (`origin/queue/be7fa274adc6` at 3:18 PM PST), nine of them under
   `verity/configuration/deploy/`; memory-accounting's #9426 edits nine files there and #9425 adds imports of the
   wire. For each open pull request that touches a file you move, say whether a plain merge onto the moved tree is
   clean (git follows the rename) and whether a line it adds would still name an old module.
