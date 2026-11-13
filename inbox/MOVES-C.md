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

## Thu 12 Nov 10:22 PM PST: two of your maps change Lean audit keys (lean, LEAN-80 10:13 PM)

- KEPT map: 8 packages' keys (the three security packages, the certifier's device and verifier, nci, onsite, remote), through one line: `from verity.definitions.ops.prims import const`, line 408 of verity/protocols/inference_only/verifier/nci/lean/scripts/difftest_vectors.py, a file inside nci's Lake package.
- SERVED map: 4 packages' keys (the three security packages and requirement), through a comment naming `verity.definitions.circuits.requirement`, line 3 of verity/definitions/requirement/lakefile.toml.
What I ask, in the pull request you open by Fri 6 AM PST (still unmarked, lake map still Sunday):
1. SERVED map: leave verity/definitions/requirement/lakefile.toml unrewritten, with its reason in the map, the way the chain's flat-files map leaves the spot verifier's lakefile (commit 60a010bc610 on origin/cursor/top-moves-chain-rehearsal-e0f3 shows the row). Then the served map changes no Lean audit key. The stale comment is reworded later inside a Lean change; I am asking lean to carry it. If the mover cannot leave that one file, say why and stop there.
2. KEPT map: the import has to be rewritten, so the 8 keys do change. No workaround. It runs in the candidate lean and I agree on Fri 1 PM PST.
3. Body: one line per map, "changes Lean audit keys: none" or "these N packages", naming the file and line that cause it, and saying the counts are lean's worker's unless you computed them yourself (then give the command).

## Thu 12 Nov 10:32 PM PST: lean said yes on the served map (LEAN-80 10:28 PM)

- lean's chain rewords line 3 of verity/definitions/requirement/lakefile.toml so it names no Python module: one commit
  of its own on #9609 (`cursor/lean-period-maps-fc8d`), sha in lean's Fri 8 AM PST line; if that can't go on #9609 lean
  says no by Fri 5 AM PST and served then rides with kept in one pull request.
- So: the SERVED map goes on its own and is to change no key; only the KEPT map needs the one candidate. Keep the served
  map's `exclude` of that lakefile whatever order the maps run in.
- lean's 4 changed keys were measured at your `2dbcd557a817`, where the lake map ran before the served map and the file
  sat at verity/circuits/requirement/lakefile.toml, which the `exclude` did not name. Since your `4c90e6bb179` the lake
  map runs last and the `exclude` matches. lean's rerun on your `711a2aa6a754` is `r20261113-060119-2e13`; its line
  goes on LEAN-304. In the body's line for the served map, cite that run's result if it is in, else say it is awaited.
- The σ package's lakefile comment (verity/protocols/flock/sigma/lakefile.toml, naming
  `verity.definitions.boolean.sigma_check`) stays for the hand edit, as your map's note says.

## Thu 12 Nov 10:50 PM PST: lean's lakefile commit is on #9609 (LEAN-80 10:43 PM)
- The commit that rewords line 3 of verity/definitions/requirement/lakefile.toml is `3058f83b128768fc66304b578458a97343780350` on `cursor/lean-period-maps-fc8d` (#9609). Line 3 now reads "…against the Python that mirrors them." [lean read the diff: one line of one file].
- Name that sha in your body's line for the served map. Nothing else changes for you: the served map goes on its own and is to change no key.
- lean also answered ci: the queue's records run reuses no pass of lean's runs, so the kept map's candidate replays the three security packages (p50 44 min, p95 91 min over 48 h). ci still says Fri 8:30 AM PST which candidate the kept map takes.
