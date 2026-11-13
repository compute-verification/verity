# FOLD: `verity/periods/` and `verity/protocols/reporting.py` into `verity/protocol.py`

You are a worker of `top` (bc-af075d78-d5b9-590b-acbb-eb47d6f88f39) in compute-verification/verity. Reread
`git show origin/main:AGENTS.md` first: your snapshot can be older than `main`. Daniel made the repository's structure
tonight's first priority (Thu 12 Nov 11:03 PM PST), and this is one of top's own code-change rows of `tests/tree.toml`.

Rules that bind you:
- Create your own branch (named below) and push only to it: never the launcher's branch, a PR head that is not yours,
  or `main`; never `--force`. Push after every commit.
- Your VM does editing, git and one named test file under a minute. Every suite, build or generator run over the tree
  goes to a node: `research run --on auto --cwd source [--wait] -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && ...'`.
  No tool call blocks more than 20 minutes; when you would wait on a run, end your turn naming the run id and its ETA.
- Write a checkpoint note under `~/.research/notes/lanes/top/<YYYYMMDDTHHMMZ>-draft-<slug>.md` before long work and each
  hour, and `research notes sync --path lanes/top`.
- Read `origin/cursor/verity-top-4db4:inbox/<your agent id>.md` at each hourly checkpoint if it exists
  (`git fetch -q origin cursor/verity-top-4db4`): top's changes to this brief arrive there.
- Mark every claim in your report and PR body: [ran], [read] or [not checked]. Do not mark the PR ready; top reads it
  and marks. Body: `Lane top`, `Refs TOP-398`, `Goal top/repo-cleaned-up`, no `Move:` line.
- An open PR of another lane that crosses yours is not yours to change; list it in your report.

## The task

Branch `cursor/top-fold-periods-<4 hex>` from chain 1's head, `origin/cursor/top-moves-chain-6a54` (PR #9697, at
`41c1cdf99377` when this was written; it makes `verity/protocol.py`). If #9697 is on `main` when you start, branch from
`origin/main` instead. Rows 3 and 21 of `tests/tree.toml`.

What top's inventory worker read on `main` `2e99b1cf48` (check each line yourself; it ran nothing):

**Brief A: fold `verity/periods/` and `verity/protocols/reporting.py` into `verity/protocol.py`.** Do it after chain 1,
which makes `verity/protocol.py`, and after `nci_rho_map`, since `rho/` imports `reporting`.

- What moves: `verity/periods/__init__.py` (173 lines, stdlib: `TickSpan`, `Cut`, `MonitoringPeriod`,
  `MONITORING_PERIODS`, `block_count`, `PeriodError`, the JSON) is appended to `verity/protocol.py` under a "monitoring
  periods" section. `reporting.py`'s `PERIODS` is the same tuple as `MONITORING_PERIODS`: keep one name (AGENTS.md's
  one vocabulary) and move `DELTAS` beside it.
- Importers to rewrite:
  - in verity: `pous/verifier/wipe_monitoring.py`, `poub/verifier/monitoring.py` and their tests;
  - in benchmarks: `sampling_plan/{transformer,cpu_budget,served_window}.py`, `spot_figures/{figure1,binding}.py`,
    `recursion_figures/figures.py`, `network_traces/{leak_bounds,block_certifier}.py`, `prove_path/window_cost.py`,
    `one_stage/interfaces/verity_e2e_demo/block_record.py` with its test, and `benchmarks/rho` after its move;
  - `demos/lifecycle/lifecycle.py`.
- Tests: `verity/periods/tests/{test_periods,test_periods_difftest}.py` go to `verity/tests/tests/`. Add `verity-difftest`
  and `hypothesis` to `verity/tests/pyproject.toml`, and its inputs get the interface package, the difftest Lean and
  Flock's three files from `verity/periods/pyproject.toml`. Delete `verity/periods/pyproject.toml` and drop it from the
  root workspace `members` and `uv.lock` (distribution `verity-core-periods`).
- Lean: the strings that name the Python mirror's path need updating: `PeriodDifftest.lean` and `Period.lean` (doc
  comments), `verity/protocols/interface/lean-audit.json` (a `why` text) and the interface README. That changes Lean
  inputs, so lean's records run.
- Listing: in `tests/import_order.toml`, the places `verity/periods` and `verity/protocols/reporting.py` become
  `verity/protocol.py`'s. Rerun `uv run tools/verity/check/import_order.py`. `pous/verifier` and `poub/verifier` then
  load `verity/protocol.py`, which pulls in `verity/channel.py`, so check that no trusted place reaches past its
  frozen list; if one does, the fold needs `--grow` and a reason. The `inputs` of `benchmarks/pous`,
  `pous/verifier` and `poub/verifier` must name `verity/protocol.py`.
- Suites that hold it: `verity/tests`, `pous/verifier`, `poub/verifier`, `benchmarks/{pous,sampling_plan,spot_figures,
  one_stage}`, root `tests/test_boundaries.py`, `test_tree.py` (delete lines 3 and 21), and `test_doc_paths`.

## Two conditions from lean, both taken by top (they override the paragraph "Lean" above)

1. The difftest moves with the fold, the port `core-periods-difftest` keeps its name, and its Python half holds every
   function it holds now.
2. The fold edits NO `.lean` file and NO lock. The three places that name the old path
   (`CoreProtocol/Period.lean` line 5, `CoreProtocol/PeriodDifftest.lean` line 10 and
   `verity/protocols/interface/lean-audit.json` line 7) are rewritten by lean's own chain. Touch Python, the pyprojects,
   `uv.lock`, the listing and the READMEs only, and list in your report every place you left for lean.

## Order with the chains

- `benchmarks/rho` moves in chain 2 (`nci_rho_map`); `rho/` imports `reporting`. Rewrite the importer where your base
  holds it. If chain 2 lands before your PR, carry your branch across with `tools/move/restack.py` (its README says how).
- Chain 2 also moves the eight protocols into `verity/protocols/proofs/`, `accounting/` and `policy/`. Same answer:
  `restack.py`, no rebase by hand.
- The import listing (`tests/import_order.toml`) is vllm's: run its generator WITHOUT `--grow`
  (`uv run tools/verity/check/import_order.py`, on a node if it is over a minute). If it holds a line, stop and write
  the held line to top in your report; do not pass `--grow`.

## Done when

- A PR is open, unmarked, from your branch onto `main` (after #9697 is on `main`; until then the branch is pushed and
  your report names its head), with: the fold, `tests/tree.toml` without lines 3 and 21 (and `ARRIVING_MAX` lowered if
  `verity/periods/` was the last `arriving` line), the suites the brief names green on a node [ran], the generator's
  exit and last line quoted, and `python3 tests/test_tree.py HEAD` on a scratch merge with `origin/main`.
- Hours: branch pushed with the fold by Fri 13 Nov 2:30 AM PST; node runs green and the PR open (or the head named) by
  4:30 AM PST. If an hour will not hold, say so in your checkpoint note the moment you know.
