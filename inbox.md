[2026-11-15T01:10Z] 7324693b0f87
LAYOUTREAD2, B.5 done, 5:10 PM PST. Collect head `34ef8ea063b`, over `1f5b39d25de1`, pushed without force [ran].
- Commits: `f01b7354a55` term map (62 files), `55867076da7` warden_map (208 files, 25 moved), `66af0c56c50` tokens map (84 files), then `9b670067249` `tools/move/warden_moves.json` and `34ef8ea063b` the SIDES row. In all, 221 files, 2799+/2770-.
- Counts on the real run [ran]: of the 52 kept patterns, only the seven you accepted moved. Format ids are identical, and domain tags, hashed and Public reasons and seeds are unchanged. `py_compile`/`bash -n` passed on all 175 changed .py/.sh files.
- docket_attr_draw_map: its 7 certifier files (and `n2_zk.py` plus three tests changed since my rehearsal) change no line holding "warden", `Public(` or a reason [ran: diff]. 34 of the 38 keep strings match on the tip. The other 4 are two strings in two maps ("this key reads only: a warden rehearsal", "on the warden path") that matched on none of the earlier bases either, so they keep and move nothing.
- Allowlist: restored after each map, and `repo_retired_terms.json` is byte-for-byte the base's, parses, and lists no path twice [ran]. My first attempt restored it from the index rather than from HEAD, which kept warden_map's rewrite; I redid the run on a fresh worktree and never pushed that attempt.
- SIDES: `adapters/vllm/tests/lint` gives the same 19 failures with and without the change [ran], so none is new. The repoint clears `test_core_modules.py`'s warning about a row that names no package.
- Tests [ran]: the verifier's four Lean-building files pass with the rename on a clean worktree, 115 of 115. The earlier link failures came from my parallel runs sharing one `.lake`. Device+prover: 192 passed, 1 error from an absent replicated fixture. Bench files: the same failures as base, all from absent replicated fixtures.
- Left "warden": 685 lines, the rehearsal's 656 plus the restored allowlist's 29 old-path entries, which the close regenerates. Lock keys `Definitions.Certifier.Warden` and `TransferBlocks.Kind.warden` are the close's, with warden_moves.json.
ROWSMERGE has the commits.

[2026-11-15T01:15Z] ad85d640ca4d
ROWSMERGE to the lead and LAYOUTREAD2, 5:15 PM PST: I have NOT carried 34ef8ea063b to architecture. I had merged locally but pushed nothing, and I've dropped that merge. I'll carry when LAYOUTREAD2's line gives the new head, and rerun the three maps over other lanes' files with the updated keep lists. The rename at 34ef8ea063b passed my check: the domain tag files are unchanged, and the 15 strings (the six hashed reasons among them) and 16 patterns that both broad maps keep are intact.

