[2026-11-13T02:30Z] 001bad913fb4
From LAYOUTREAD, Thu 12 Nov 6:31 PM PST, on NET-410. **The map's pull request is open and unmarked: [#9589](https://github.com/compute-verification/verity/pull/9589), head `7e7a8b2977ce` on `cursor/na-nci-rho-map-8d6e`. It is yours to read and mark.**

**The pull request** [ran: read back from GitHub]
- Open, not a draft, mergeable. One file against `main`: `tools/move/nci_rho_map.toml`, 75 lines in.
- The head is `main` at `e86d9204135c` with the map merged in, so the trial's base is the head.
- The body has no move line (the README's rule for a map), says first that two hand commits follow the move and change more than the index, and ends `Goal network/changes-cross-reorganization` and `Refs NET-410`. It is linked to NET-410.
- **One thing for you:** `research queue open` warned twice that the Goals page reads `network/changes-cross-reorganization` as a done goal, so the body names no live goal. It only warns. If NET-410 should carry another goal, label it and say; I change the body's line with `research queue edit`.

**The certificate the body cites** [ran]
- Run `r20261113-012958-4c63`, exit 0: `7e7a8b2977ce..891699fe4b1e` is a pure move (42 renames, 85 path-only edits); a second run changes nothing.

**The trial branch and its three commits** (all on origin)
- `scratch/na-nci-rho-trial2-8d6e` at `7cc7643e8cd5`: the move `891699fe4b1e`, the first hand commit `ae2a75c80ebc`, the second `7cc7643e8cd5`.
- The first trial stays at `scratch/na-nci-rho-trial-8d6e` (`f2e5fd524e9`), the tree training ran.

**The patches' place**
- `scratch/na-nci-rho-hand-8d6e` at `46110ffd7903`: `nci_rho_hand.patch` (the two rows and training's test, unchanged since training said yes) and `nci_rho_stale.patch` (`AGENTS.md` line 29, the two `SEPARATE` entries). `git am` of both on the move's commit gives the trial branch's head [ran].
- The body tells top's chain builder: the map, then the first patch as the second commit, then the second as the third.

**Each run, with its exit** [ran]
- `r20261112-235401-66ce`, exit 1: the first trial, refused on the one test.
- `r20261113-004432-5173`, exit 0: the trial on `main` at `7e70df68bd18`, without the Lean include line.
- `r20261113-012707-c6c0`, exit 1: no verdict, my launch without a git identity.
- `r20261113-012958-4c63`, exit 0: the trial the body cites.
- `r20261113-010611-7046`, exit 0: the suites on the first trial's three trees.
- `r20261113-020306-b302`, exit 0: the suites on the final tree (today's `main`, the move, both hand commits).

**The suites' three verdicts, and the fourth** [ran]
- **The base:** all pass. Your six: `nci` 132, `rho` 612, `circuit_properties` 69, `inference_only` 10, `training_slowdown` 576, `verity/hardware` 178. The root's tests: 160. Six more suites the move rewrites: all pass.
- **The move alone:** two tests fail and no other. `training_slowdown`'s `test_only_the_one_module_reaches_the_calculator` (575 passed, 1 failed), and the root's index test (159 passed, 1 failed). The eleven other suites read the same inputs as on the next tree and reuse its passes.
- **With the first hand commit:** all pass, with the base's counts.
- **The final tree:** all pass, with the same counts for your six and the root's 160. The boundary test passes with `SEPARATE` two entries shorter, and the doc-path test passes.
- No suite needs more than path edits, but for those two tests.
- Bookkeeping warnings are the same before and after, but for one: the doc-paths warning loses five entries (178 to 173), the five Lean doc comments that name `benchmarks/nci/…` today, and gains none.

**Training's ask on calls is not in the hunk.** Your 6:04 PM PST message came after I launched the final suites run (6:02 PM PST) at the head the body cites. By your rule the hunk goes in as it is, and the body says training adds the calls (`Path`, `PurePath`, `join`, `joinpath`) after the cut.

**Notes sync passes your way** [ran]: with `RESEARCH_NOTES_TOKEN` unset and its name out of `CLOUD_AGENT_INJECTED_SECRET_NAMES`, "notes sync: committed … (1 paths); pushed after 2 attempts". Neither value was printed.

**What I did not check**
- Lean's maps and the chain's order: your words.
- No Lean build, no lock check, no `--update`.
- The whole check, `circuit-check`, and any suite not named above.
- The patches on lean's head: the rehearsal does that.
- The base and the move alone ran on the older `main` (`7e70df68bd18`); only the final tree ran on today's.

**Still mine, with the hours**
- The rehearsal on lean's head, once, after lean names it on LEAN-80 (Fri 13 Nov by noon PST). Its result to you by Fri 3:30 PM PST.
- If lean says no to the Lean include line by Fri 8:00 AM PST: one commit taking it out, before your 9:00 AM mark.
- NET-408's manifests start now. A line on where they stand by Fri 6:00 PM PST; the pull request by Mon 16 Nov noon PST.
- On call from Fri 7:00 PM PST. No run of mine in the pool `network` Fri 4:30 to 10:30 AM PST.

