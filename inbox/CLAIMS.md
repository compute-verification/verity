# CLAIMS: delete `verity/claims/`, its readers keep what they need

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

Branch `cursor/top-claims-delete-<4 hex>` from `origin/main`. Row 20 of `tests/tree.toml`.

What top's inventory worker read on `main` `2e99b1cf48` (check each line yourself; it ran nothing):

**Brief C: delete `verity/claims/`** (623 lines, an id registry).

- Python readers: `pous/verifier/protocol.py` (`Entry`, `get`), `spot/device/challenger.py` (`claims`), and the tests
  `commitments/tests/test_hm96.py`, `pous/verifier/tests/test_pous_protocol.py`,
  `spot/verifier/tests/test_coin_claims.py` and `poub/verifier/tests/test_ledger.py`.
- Each reader keeps the few entries it cites as a local constant table, and its test checks that table instead. Then
  delete `verity/claims/` with its suite.
- Strip `verity/claims` from about 15 pyprojects' `inputs`, from `tests/import_order.toml` (rerun the generator), from
  `tools/verity/check/drive.toml` and from the docs (ARCHITECTURE.md, `verity/README.md`, `hardware/README.md`, PoUS's
  and the certifier's PROTOCOL.md, `security/Proofs/Flock/Soundness/ASSUMPTIONS.md`).
- Check the docs site's assumptions registry: the docstring says it shares the id format. If the site reads this module,
  the deletion needs its owner.

## What to settle first, before any deletion

- Find EVERY reader, not the list above alone: `git grep -n "verity.claims\|verity/claims\|from verity import claims"`
  over the whole tree, Lean, TOML, JSON and Markdown included, and `git grep -n claims` under `tools/` and the docs site.
  A reader outside Python (the docs site's assumptions registry, a Lean file that names an id, a lock) decides whether
  this is a deletion or a move: if anything other than the Python readers and the listed docs reads the registry's
  CONTENT, stop, push what you have, and report the reader to top. Do not edit a `.lean` file or a `lean-audit.json`.
- An id that a guarantee's statement or a lock names is a claim: you change no id, no statement and no lock. The
  deletion removes the Python registry only; each reader keeps the entries it cites, word for word.

## Order with the chains

- Chain 1 (#9697) does not move `verity/claims/` or its readers. Chain 2 moves the readers' protocols into
  `verity/protocols/proofs/`, `accounting/` and `policy/`: if it lands before your PR, carry your branch across with
  `tools/move/restack.py` (its README says how), no rebase by hand.
- The import listing (`tests/import_order.toml`) is vllm's: after the deletion run its generator WITHOUT `--grow`
  (`uv run tools/verity/check/import_order.py`). If it holds a line, stop and report the line; do not pass `--grow`.

## Done when

- A PR is open, unmarked, from your branch onto `main`: `verity/claims/` gone with its suite, every reader's table and
  test in place, `tests/tree.toml` without line 20, the pyprojects' `inputs`, `uv.lock`, `drive.toml` and the docs
  cleaned, the suites of every touched package green on a node [ran]
  (`uv run tools/verity/check/suites.py --changed <main's sha> --quick`), `tests/test_tree.py`, `tests/test_doc_paths.py`
  and `tests/test_boundaries.py` green, the generator's exit and last line quoted.
- Hours: the reader list in your checkpoint note by Fri 13 Nov 12:45 AM PST; PR open by 3:30 AM PST. If an hour will
  not hold, say so in your checkpoint note the moment you know.
