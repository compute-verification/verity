# CHAIN 2: every other path-only move, staged as one pull request

From top, Thu 12 Nov, late evening PST. Daniel made the structure tonight's first priority (11:03 PM PST): the
repository in the page's tree when he wakes, Fri 13 Nov 7:00 AM PST. Chain 1 is #9697 (`cursor/top-moves-chain-6a54`
at `41c1cdf99377`), marked 11:30 PM PST. Chain 2 is everything else that only changes paths, in ONE pull request that
ci lands alone, in one candidate, in a queue it pauses for it. You stage it.

## The order, bottom to top (lean's and ci's, TOP-415, 11:13 to 11:30 PM PST)

0. **The base.** Top names it on TOP-398 and in your inbox at about 1:05 AM PST, once the Lean batch now in the queue
   (#9461, #9237) has landed: `main` if chain 1 is on it by then, else `main` with chain 1's head merged.
   Until then, work on a TRIAL base: `origin/main` merged with `origin/cursor/top-moves-chain-6a54`.
1. **#9609's commits** (`origin/cursor/lean-period-maps-fc8d`, at `3058f83b1287` when this was written): nine word
   maps, their test, eleven Lean files, one lakefile comment. Hand commits; they rename nothing.
2. **lean's seven stages.** lean's worker stages them on the base top names (about 25 minutes) and pushes ONE branch
   for you to build on. Its name comes on TOP-398; top passes it to your inbox. On the trial base, skip this step.
3. **network-accounting's nci/rho map**, after lean's stages: `tools/move/nci_rho_map.toml` from
   `origin/cursor/na-nci-rho-map-8d6e` (`5e891bd421c8`, #9589), then by hand:
   a. `nci_rho_hand.patch` from `origin/scratch/na-nci-rho-hand-8d6e` (`8faf6889a322`), `git am`;
   b. as steps, not as a patch: AGENTS.md's list of code a move brought from `benchmarks/` drops the two directories;
      `tests/import_order.toml` loses the two `[[distribution]]` entries `rho` and `nci` by hand; the generator runs;
      `tests/test_boundaries.py`'s `NO_DATE_OR_REASON` loses `nci`'s entry;
   c. at the very END of the chain (step 6), the harness's pin.
   `origin/scratch/na-nci-rho-trial4-final-8d6e` (`5ce6135f903f`) shows the whole order on `main` `2e99b1cf48b8`.
4. **The path maps, in this order.** Every map but the last writes FLAT destinations
   (`verity/protocols/pouw/...`), and the group map then carries them:
   a. top's three of `origin/cursor/top-moves-a-41d7-pr2b` (`e56130d30422`): `artifacts_map`, `live_challenger_map`,
      `difftest_map`. The difftest map lacks two `[split]` rows, written out in `inbox/MOVES-A.md` on this branch.
      MOVES-A is being resumed to restage them on chain 1's head; if its branch is not named in your inbox by
      1:15 AM PST, add the two rows yourself as a hand commit.
   b. top's four of `origin/cursor/top-moves-c-b396` (#9692, `747e26eeb168`): `definitions_kept_map`,
      `definitions_merge_map`, `definitions_served_map`, `definitions_lake_map`, in the order its body gives.
   c. `compose_fix_map.toml` from `origin/cursor/top-tree-inventory-485d` when INVENTORY has pushed it (it replaces
      `main`'s `compose_map.toml` and `configurations_map.toml`; proofs rules by 1:30 AM PST, silence is yes).
   d. compute-research's two, on one branch `cursor/compute-move-chain2-maps-...` by 1:30 AM PST: `hardware_map.toml`
      re-keyed for chain 1's tree with its patch file of three hand lines, and `sass_gate_map.toml` with its hand lines.
   e. compute-accounting's map, on a pushed branch by 1:30 AM PST: PoUW's `clear/` and `receipts/` into `verifier/`,
      its served GPU directories into `prover/kernels/`, and those rows for Dither and the FP4 variant that are
      path-only. A row that needs a code edit is not in it.
   f. memory-accounting's map of the PoUS grader (`verity/protocols/pous/lean/` to `tools/pous_grader/`), by 2:00 AM.
   g. LAST: `protocol_groups_map.toml` from `origin/cursor/top-protocol-groups-f911` (GROUPS), with its hand commits.
   Top passes each branch and head to your inbox as it is posted. `world_map.toml` and the maps for Flock's and SPOT's
   loose files stay OUT unless top says proofs ruled.
5. **Hand commits after the maps:** the import listing (delete the entries of moved directories, run the generator
   WITHOUT `--grow`; vllm's worker importcarry also runs it on the head top posts and answers on TOP-398);
   `tests/tree.toml` (the lines the chain lands are deleted, as the test asks); what each map's owner listed as its
   hand lines; nothing else.
6. **The harness's pin:** run `benchmarks/harness`'s
   `test_exfiltration.py::test_with_no_verdicts_of_record_the_print_is_as_it_was_byte_for_byte` on the staged tree (a
   node) and put the two digests it prints into `NO_VERDICTS`. Once, after every map.
7. **lean's records commit**, made by lean's worker on your staged head, pushed onto your branch (top hands lean the
   branch). The pull request is not marked before it is there.

## Rules

- Inside a Lake package, a lakefile, a manifest or a `.lean` file, a map changes paths and nothing else: no comment or
  string that names a module (lean). A map that breaks this goes back to its owner, by a line in your note.
- No commit of the chain changes a script under `tools/move/` (`*.py`): the queue builds no move beside or on top of
  such a change (ci). Maps (`*.toml`) are fine. If a mover gap blocks a map, the map stays out.
- A map that does not certify (`stage.py` / `move_check.py`), or after which Lake cannot resolve every package, STAYS
  OUT and the rest goes on. Keep a table in your note: map, branch, head, in or out, why.
- One generated commit per map, by `stage.py`; each hand commit on its own with a message that says why.
- Never `--grow`. A held line of the listing comes to top unwritten.
- Push only to your own branches. Heavy work on nodes (`research run --on auto --cwd clone`); no call blocks over 20
  minutes; when you would only wait, end your turn with the run id and its ETA.
- You post nothing on Linear or Slack. Top posts each head. Put every head in your note the minute it exists and
  `research notes sync --path lanes/top`.

## Hours (PST, Fri 13 Nov)

- **Now to 1:05 AM:** the trial. Stage steps 1, 3, 4a, 4b and 4g on the trial base, to find what breaks between
  maps before the real base exists. Push it as `cursor/top-moves-chain2-trial-<4 hex>` and write what broke.
- **2:15 AM:** the rehearsal head on the named base with every map that was on a pushed branch by 2:00 AM, pushed as
  `cursor/top-moves-chain2-<4 hex>`. lean makes its records on it; kernels builds the H200 kernel from it; importcarry
  runs the listing on it. Run on it: the tree test, the boundaries, compileall, the count of changed Lean inputs.
- **3:30 AM:** ci has paused the queue since 2:30 AM and `main` stands still. If `main` moved since the rehearsal's
  base, restage on it and say in your note whether the 36 Lake packages' input keys are equal on both heads (lean's
  records carry over only if they are).
- **Then:** open the pull request, unmarked (`research queue open ... --by top`; body `Lane top`, `Refs TOP-398`,
  `Goal top/repo-cleaned-up`, no `Move:` line; a table of every map in order, what certified, what stayed out, and
  every hand commit with its reason). Top reads it and marks at about 4:00 AM PST, after lean's go.

## Your report

Branch, head, the table of maps in and out, each check with [ran] and its run id and exit code, every hand commit and
why, every open pull request of top's that this chain supersedes (#9692, MOVES-A's second, #9407), and what you were
unsure of.
