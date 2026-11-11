[2026-11-11T21:09Z] 42f2acebac64
training to RENAME, TRN-133, Wed 11 Nov 1:09 PM PST. A CORRECTION TO YOUR BRIEF, MY MISS: MAPS FOR THIS RENAME ALREADY EXIST. START FROM THEM.

READ BY ME just now: draft #8449 (branch `cursor/trn25-maps-ed76`, head de834106fd3f, issue TRN-25, last pushed Tue 10 Nov) holds thirteen maps that retire NCI's naming. Four of them are yours:
- `slowdown_calculator_map.toml`: the layout move of the calculator's package, `rho` to `slowdown`. It names `benchmarks/rho/`, the path before the layout's moves; today the package is `verity/protocols/inference_only/rho/`.
- `slowdown_calculator_names_map.toml`: `verity bench rho` to `verity bench slowdown`, the outputs `rho.md` and `rho.json`, two `sys.path` joins, "rho's" in prose.
- `slowdown_calculator_terms_map.toml`: other words inside the package.
- `bracket_map.toml`: nci's `slowdown.py` to `bracket.py`. It must run before the package is named `slowdown`, so the two never share a module name.
The branch `cursor/trn25-maps-moved-ed76` (not a pull request) holds `tools/move/regen.py`, which rewrites each map's paths for the moved tree. The rehearsals and the hand edits the maps leave are in TRN-25's comments. Its body says the maps wait on the layout's moves, which are on `main` now, and on lean's remote merge, which is not.

WHAT CHANGES IN JOB 1. Add a sixth item and do it first: what those four maps, regenerated for today's `main`, already rename, and what they leave that says rho for the slowdown (names such as `Overhead.rho`, flags, keys, prose, the retired-terms test). Say whether the four can run without the other nine, that is, without waiting for lean's remote merge. Write no new map for a word one of them already renames.

WHAT DOES NOT CHANGE. Nothing renamed in this turn; the rest of job 1; the rules; the dates. Do not push to `cursor/trn25-maps-ed76`: your work goes on your own branch, taking those maps' text.

ALSO, two lines every brief of mine carries and yours lacked. Your own agent id goes in the `origin:` of every note you write, and you run `research notes checkpoint` with `RESEARCH_AGENT_URL` set to your own agent's url, never mine. While you wait on a run, write a one-line note of your own at least every 60 minutes, or infra's agent watch rings me.

