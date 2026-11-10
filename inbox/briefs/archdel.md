# ARCHDEL: a pull request that deletes `archive/`, prepared now, marked only on Daniel's word.

**Why.** The architecture Daniel signed off (Linear TOP-398, sections 7 and 9) has no `archive/` in the tree: "Daniel recalls ruling that the archive is deleted", and no record of that ruling was found. top asks Daniel tonight. If he says yes, the pull request must already exist. If he says no, nothing is lost but your time.

**Base.** The moved tree: staging A, `origin/cursor/top-layout-staged-e15c` at `50f940c4f28531bc33f23bbb73a2970d5820caf1` (built on `main` `2b869ed35db5`). top tells you in your inbox when the move is on `main`; rebase onto it then.

**What to do.**
1. **Measure first, and write it in your note before you delete anything:** files and lines under `archive/`; every file outside `archive/` that imports, loads by path, runs or names something under it (code, tests, `pyproject.toml` members and sources, `uv.lock`, `tools/verity/check/*.toml`, suites' `inputs`, `fixtures/` that only the archive reads, `benchmarks/numerical/` and any table that still quotes A-GKR, B-Ligero, D-SP1 or the VOLE prover, `tests/test_boundaries.py`, `public_boundary.toml`, docs). Group them by owner (`python3 -m research msg owners PATH` if it exists, else by directory).
2. **Sort each reader into:** (a) goes with the archive (it exists only for it); (b) keeps working without it after a small edit, which you make; (c) a live result or table that would lose its source: **stop on these and list them; do not edit them.** Those are what Daniel's yes would cost, and top shows him the list.
3. **The change:** `git rm -r archive/`, the (a) and (b) edits, the workspace and lock files regenerated on a node (`uv lock` is a node's work), `AGENTS.md` and `archive/AGENTS.md`'s mentions, and `tests/` invariants that name the directory. The history keeps everything: say in the body the last commit that holds `archive/`.
4. **Check on a node:** `uv sync` and `uv lock --check`; `suites.py --changed <base> --quick`; `tests/` whole.

**Deliver.** Branch `cursor/top-archive-out-<hex>`; a pull request opened **unmarked** after the move is on `main` (body: `Lane top`, `Goal top/repo-cleaned-up`, the (c) list, the counts, the run ids). Before that: the branch pushed and, **by 01:30Z Wed (5:30 PM PST)**, your final message with the counts, the (c) list and what is still red. Never mark it.
