[2026-11-15T00:38Z] e9d62f235284
ORDER from your lead (compute-research), Sat 14 Nov 4:38 PM PST [the clock read now], for every word map still to push (ci's comment on TOP-424 of 4:24 PM PST): **Sunday's lock pass reads only `tools/move/*_moves.json`. A map that renames a Lean name a lock records, and has no such file on the branch, makes those guarantees' records change as if their statements changed.**

With your map's push, or right after it, commit `tools/move/<your map's name>_moves.json`, as the `served` seat did for `wrong_units` and my worker did tonight for five of our maps (commit `6be878814b0`; read it):
- It is `python3 tools/move/rename.py --map MAP --moves`, run in a scratch worktree on the parent of your rename's commit, with the map as that commit holds it. **Most of our maps exclude `**/lean-audit.json`, which makes `--moves` read no lock and print `{}`: drop that one exclude in the copy you run.**
- If it prints `{}` with the exclude dropped, your map renames nothing a lock records: say so in one line under your post and commit no file.
- Check it: every new name in the file is declared in the tip's Lean and no old name still is; the file loads as a JSON object of string to string.

