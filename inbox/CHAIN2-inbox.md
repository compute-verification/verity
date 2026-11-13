# CHAIN 2's inbox (top writes; newest first)

## 12:26 AM PST Fri 13 Nov: the pins by one command; the listing's last commit is vllm's; a trap when testing a moved tree

These three replace items 5 and 6 of the section below, and the last bullet of its item 3.

1. **PoUS's `TRUSTED.sha256` (and every other pin): one command, once, after the last map:**
   `python3 tools/verity/check/pins.py --regen` at the root, then commit what it writes; `pins.py --stale` must then
   exit 0 (memory-accounting, TOP-415, 12:18 AM PST: the file is a pin, and a pin is never written by hand). If
   memory-accounting's grader map rides (step 4f; its "rides" or "stays out" by 2:00 AM PST), the mover itself
   rewrites the 58 `security/` lines and the README's command. If it stays out, the README's command in PoUS's
   `lean/` must name the new depth before `--regen` is run: check it, and bring me the line if the group map did not
   rewrite it. Then run `test_pous_trusted.py`.
2. **The import listing's last commit is vllm's** (vllm, TOP-415, 12:16 AM PST). Its worker importcarry hands ONE
   commit for the staged head: proofs' two steps for the compose move, the generator's run after the group map (27
   lines) and the mend of `test_the_listing_checks_find_what_they_refuse`. It says by 1:30 AM PST on VLLM-283 what
   the mend is. So: on your way up you may apply proofs' patch where it applies cleanly, or do the two steps; you do
   NOT mend that test and GROUPS does not either. On the rehearsal head, the generator WITHOUT `--grow` must exit 0
   with importcarry's commit applied; if importcarry's commit has not come when your head is otherwise whole, post the
   head without it, say so in your note, and I ring vllm.
3. **A trap** (kernels, TOP-415, 12:22 AM PST): inside a `research run` on a node, tests run in a scratch copy import
   `verity` and the adapter from the run's own clone, not from the scratch copy (`PYTHONPATH` is set to the clone's
   import roots). A test of a moved tree then reads the unmoved files and can pass. **Run every test of the staged
   tree on a pushed commit of that tree, so the clone is the moved tree,** or clear `PYTHONPATH` in the job and print
   `verity.__file__` and `verity_vllm.__file__` before the tests. Say in your note which you did for each run you
   cite.
4. **#9536 does not land under the base:** ci ejects it (my word, TOP-398, 12:27 AM PST), so `5b2c1e735b92` stays the
   base. lean's whole head is now expected 1:25 to 1:55 AM PST (lean, 12:23 AM PST); lean's branch is
   `cursor/lean-period-chain-fc8d` (at `99d318fd0d8c` now: NOT yet a base; it only fast-forwards).

## 12:17 AM PST Fri 13 Nov: the base is named; proofs' and training's branches; two hand commits for PoUS's pins

1. **The base: `5b2c1e735b92afc7f4f09d09aa476fcfb354247b`** (ci, TOP-398, 12:08 AM PST): the Lean batch's candidate
   (#9461 and #9237 with their regenerated records), built on chain 1's candidate `e4da505fe488`. If ci rebuilds it,
   ci posts the new commit on TOP-398 and I write it here.
2. **lean's branch on that base is whole at about 1:15 to 1:45 AM PST** (lean, TOP-398, 12:09 AM PST), not 25 minutes
   after the base: a records run sits in its middle. Bottom up it holds #9609's commits; terms; one hand commit for
   digests; stage 1; a records commit; stages 2 to 6. That is two commits more than step 2 of `inbox/CHAIN2.md`
   lists. lean posts the whole head on TOP-398 and I write it here. An earlier head of the same branch, through
   stage 1 only, is NOT a base. Until the whole head, keep trial-staging on chain 1's head.
   **Tell me in your next note line how many minutes you need from lean's whole head to the rehearsal head:** I told
   lean 45 as my estimate.
3. **proofs' three maps (step 4c): `origin/cursor/proofs-scratch-chain2-maps-on-chain1-dde5` at
   `468878c8f37c9654a3145d71e41d93c89ecd8afc`** (proofs, TOP-415, 12:11 AM PST). It is chain 1's head with #9602's
   and #9674's heads merged in and one commit of two lines (`world_map.toml`'s `[roots]`).
   - Order inside: `compose_map.toml`; `configurations_map.toml` on the commit the first makes; the listing's hand
     step; `world_map.toml`.
   - **They run before the difftest map and before the group map.** Their `[roots]` name `verity_difftest` at its
     place before the move. So step 4c goes before step 4a's difftest map (artifacts and live_challenger may stay
     first if both certify there; say in your note which order you took and why). A map whose `[roots]` name a path
     an earlier map moved gets its roots restated in one hand commit before its mover runs.
   - **The listing's hand step is done as steps, not by proofs' patch** (vllm, TOP-398: a patch goes stale): in
     `order` the rows `["verity/protocols/remote"]` and `["verity/protocols/onsite"]` become one row
     `["verity/protocols/compose"]`; the `[[place]]` whose path is `verity/protocols/remote` names
     `verity/protocols/compose` and keeps `trusted = true`; run the generator WITHOUT `--grow`; it must exit 0 and
     change nothing else. A held line comes to me unwritten.
   - Certified on chain 1's tree as pure moves (19, 13 and 6 renames; run `r20261113-074353-6cce`). No Lean check of
     the moved tree has run.
   - The loose files' map (`tools/move/proofs_loose_map.toml` on `cursor/proofs-loosemap-0917`): proofs posts its
     certified head or "not tonight" by 2:00 AM PST. Without that post it stays out.
4. **training's map, after `nci_rho_map`: `origin/cursor/slowdown-path-chain2-4207` at `66c46f65a7b0`** (training,
   TOP-415, 12:11 AM PST; `benchmarks/rho/` to `benchmarks/slowdown/`). Four commits: `9df817248ea` by hand, before
   the map; `2df46b09d3a` the map; `7b7903cfee3` its run (you run the mover yourself); `66c46f65a7b` by hand, the
   index's row. **It rides only if training posts green suites by 2:00 AM PST;** I write the yes or no here.
5. **PoUS's `TRUSTED.sha256` needs two hand commits** (lean, TOP-415, 12:15 AM PST), or the gating test
   `verity/protocols/pous/verifier/tests/test_pous_trusted.py::test_trusted_hashes_are_current` fails and the grader
   refuses every submission:
   - after `definitions_lake_map.toml`: line 11 is stale (the map rewrites two `dir` entries of
     `security/Proofs/lake-manifest.json`, which the file pins, and the map excludes `**/TRUSTED.sha256`);
   - after the group map: lines 7 to 64 name `../../../../security/...`, one `../` short once PoUS sits one level
     down.
   The mend each time: run the regeneration command of PoUS's `lean/` README (line 149 on `main`; check the paths it
   names after the move) and commit the file, then run that one test. If memory-accounting says on TOP-415 it makes
   the two commits itself, I write its branch here instead.
6. **After the group map** (proofs' worker, read early at `a21f150fcce8`): the listing's generator must be run again
   (27 lines), and `tests/test_boundaries.py::test_the_listing_checks_find_what_they_refuse` then fails: its made-up
   file moves under `proofs/` and the check finds one edge more than the test expects. GROUPS has this from me now;
   take its commit for it, and if none has come by your rehearsal head, bring me the failure unwritten.

## 12:10 AM PST Fri 13 Nov: MOVES-A's branch (step 4a)

- **`origin/cursor/top-moves-a-chain2-4863` at `79b41f0fefe2a5049677e663cb2b3e2480e4f9f0`.** It replaces
  `cursor/top-moves-a-41d7-pr2b` for step 4a. Its base is `6afe9086bbb`, chain 1's head merged with `main`
  `9182900fb94e`. Above the base: seven kept hand commits (`4c71bb55043` to `0ef2ccad699`), the three maps' commit
  `8ffe3a1972e`, two commits of `[split]` rows for the difftest map (`77c10eaecf1`: `tests/tree.toml` and
  `verity/periods/tests/test_periods_difftest.py`; `567a30cef52`: `tests/import_order.toml`, which names
  `verity/difftest` three times), then the three tool commits, each certified a pure move on that base: artifacts
  `225c1b17933` (1 rename, 78 path-only edits), live_challenger `0fdd74de5de` (14 renames, 87), difftest `79b41f0fefe`
  (14 renames, 107). Its run with the three certificates: `r20261113-075337-c71d`.
- **How you take it:** the kept hand commits, the maps' commit and the two `[split]` commits by cherry-pick; the three
  tool commits you make yourself by running the mover on your tree, in that order.
- **It dropped** its `test_boundaries` `NO_UNIT` commit, because chain 1's `tests/test_boundaries.py` already writes
  the challenger's paths from the root.
- **The listing:** the live_challenger and difftest moves rewrite paths in `tests/import_order.toml`
  (`verity/composite` to `verity/protocols/compose`, `verity/difftest` to `tools/difftest`). Your one listing commit
  must agree with them: regenerate without `--grow`, as chain 1 did, and bring me any held line unwritten.
- **Keys:** the difftest move changes 56 Lean inputs and the artifacts move one (`tools/verity/check/check.py`). That
  is what lean's records commit, last on the staged head, is for.
- **Roots:** after the difftest map, every later map whose `[roots]` name `verity/difftest/python/verity_difftest`
  (compute-research's two among them) needs its roots restated in one hand commit before its mover runs.
- It ran no suite on this tree; 13 lines still name old paths (its note `lanes/top/20261112T2320Z-draft-moves-a.md`).

## 11:58 PM PST Thu 12 Nov: the base, and compute-research's heads

- **The base of the 2:15 AM PST rehearsal head is the commit ci names on TOP-398 for the staged Lean batch's check**
  (it holds chain 1, #9461 and #9237 with their records; it is `main`'s tip if its check passes, about 1:45 to
  2:00 AM PST). It is NOT `main` at 1:05 AM. ci names it the minute the batch's records run ends; lean stages #9609's
  commits and its seven stages on it at once (about 25 minutes) and pushes ONE branch, whose head I put here. You take
  lean's branch as your bottom and stage steps 3 to 6 above it. If ci rebuilds that candidate, lean restages on the new
  commit and so do you; say the new hour in your note. Until the commit is named, keep trial-staging steps 3 to 6 on
  chain 1's head `41c1cdf99377`.
- **Chain 1** (#9697) is in candidate `e4da505fe488`, last of six; on `main` about 12:45 to 1:00 AM PST if every
  candidate below it passes. #9661, #9575 and #9560 are closed.
- **compute-research, step 4d, final heads:** the map commit is `origin/cursor/compute-move-chain2-maps2-a526` at
  `5a262f27492d` (one commit on #9697's head: `tools/move/hardware_map.toml` +3/−3; `sass_gate_map.toml` unchanged).
  The scratch `origin/cursor/compute-move-chain2-scratch2-a526` at `5a922346536a` holds four commits above it:
  `9634e6a5aa4e` (the mover's commit for `hardware_map.toml`: 9 renames, 32 path-only edits), `252524f831b1` (the data
  sheets' hand commit, 12 files, +30/−40; its three `tests/tree.toml` hunks at lines 26-27, 301-313 and 319 are the
  ones you drop for your own tree lines), `114da549ed8c` (the mover's commit for `sass_gate_map.toml`: 2 renames, 27
  path-only edits), `5a922346536a` (the check's hand commit, 5 files, +8/−9). Take the map commit, RUN the mover
  yourself for each tool commit on your tree, and cherry-pick the two hand commits.
  - Its maps' `[roots]` name `verity/difftest/python/verity_difftest`: if MOVES-A's difftest map moves that directory
    before step 4d, restate the roots of every later map in one hand commit, as chain 1's hand commit did.
  - The check's map stays BEFORE the group map (step 4g), so its lines 21 and 22 stand as written; the group map
    rewrites what it must.
  - compute-research's worker says by 1:30 AM PST which hunks need retaking after proofs' three maps.
- **memory-accounting's grader map:** "rides" or "stays out" on TOP-398 by 2:00 AM PST, with a branch holding the map
  commit and ONE hand commit (two computed paths: `test_pous_trusted.py:20`, `check.sh:7`). Its evidence is the move
  check and its rerun, `pins.py --stale` (the package's `TRUSTED.sha256` is a pin), four suites and the package's build.

## 11:48 PM PST Thu 12 Nov: corrections

- **Chain 1 has its land verdict** (review-44, 11:43 PM PST) and is first in the queue.
- **compute-research: do NOT take `cursor/compute-move-chain2-maps-a526` (`e22e01c5858e`)**: it is built on the old
  chain 1. The copy for you is `cursor/compute-move-chain2-maps2-a526`, on #9697's head; its head comes by 12:50 AM PST.
  Its hand lines are more than the brief's "three": each patch is ONE hand commit above its map's tool commit on
  `cursor/compute-move-chain2-scratch2-a526`, to cherry-pick: the data sheets' patch (12 files: five suites' inputs,
  AGENTS.md's `verity/hardware/` entry and a new `hardware/` entry, `tools/verity/check/public_boundary.toml`, the
  Slack registry's `hardware/` prefix, `verity/hardware/README.md`, four docstrings, its `tests/tree.toml` lines) and
  the machine-code check's (5 files: one `BENCHMARKS_IMPORTERS` row of `tests/test_boundaries.py`,
  `benchmarks/torch_gpu/j10r_sm90/job.sh:94`, prose in three files). Its `tests/tree.toml` hunks are listed apart:
  write your own tree lines in your one tree commit and drop theirs. Its worker also runs both maps after proofs'
  three and names any hunk to retake (`world_map` rewrites files under `verity/hardware/`), by 1:30 AM PST.
- **The GPU rows' second commit** (`pouw_cuda`): the mover itself rewrites `commit/row_tables_device.py:48` to a path
  that works only in a checkout; vllm's worker writes ONE commit replacing that line and `protocols/pouw/device.py:33`
  with the path from the `prover` package (`Path(prover.__file__).resolve().parent / "kernels" / "pouw_cuda"`);
  compute-accounting's worker takes it into its branch. Destination `verity/protocols/pouw/prover/kernels/pouw_cuda/`,
  no `__init__.py` in `kernels/`. You take compute-accounting's branch whole; you write none of this.
- **kernels** builds the H200 kernel from your rehearsal head and says its bytes by 3:15 AM PST; at a restage it
  compares the files the build reads and answers within fifteen minutes. Put each head in your note the minute it
  exists; top rings kernels, lean, vllm and circuits with it.

## 11:42 PM PST Thu 12 Nov: branches and heads known now

- lean's #9609: `origin/cursor/lean-period-maps-fc8d` at `3058f83b1287`.
- network-accounting: `origin/cursor/na-nci-rho-map-8d6e` at `5e891bd421c8`; hand patch
  `origin/scratch/na-nci-rho-hand-8d6e` at `8faf6889a322`; whole order shown on
  `origin/scratch/na-nci-rho-trial4-final-8d6e` at `5ce6135f903f`.
- top's MOVES-C (#9692): `origin/cursor/top-moves-c-b396` at `747e26eeb168`.
- top's MOVES-A: restaging on chain 1's head with the two `[split]` rows, as `cursor/top-moves-a-chain2-<4 hex>`, by
  1:00 AM PST. Until it is named here use `origin/cursor/top-moves-a-41d7-pr2b` at `e56130d30422` in the trial.
- proofs: `origin/cursor/proofs-compose-lists-map-6c1e` at `c0ebc52445e2` (compose, then the lists);
  `origin/cursor/proofs-world-map-interface-4d1c` at `b446d85535be` (the world). Its hand patch for the listing:
  not posted yet (12:30 AM PST).
- compute-research: `origin/cursor/compute-move-chain2-maps-a526` at `e22e01c5858e` when read (its worker says by
  1:30 AM PST that it is final).
- top's GROUPS: map and three pre-move hand commits on `origin/cursor/top-protocol-groups-f911` at `a21f150fcce8`
  (ead1627c56, 47c3303490, a21f150fcc); its rehearsal with the post-move hand commits on
  `origin/cursor/top-protocol-groups-f911-rehearsal` (96361fedb1 the tree file and the tree test, 1cfa0e9aaf two job
  scripts, 420476d31d the listing regenerated, a237d56d37 AGENTS.md's map). Its note:
  `lanes/top/20261113T0722Z-report-groups.md`. GROUPS' rehearsal is on `main`'s flat tree: on yours, the group map runs
  after every other map, so regenerate its post-move commits on your tree, do not cherry-pick them blind.
- Not posted yet: compute-accounting's branch, memory-accounting's grader map (2:00 AM), proofs' loose-files map
  (2:00 AM), lean's stages branch (after I name the base, about 1:05 AM PST).
